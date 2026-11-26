//! End-to-end search pipeline: freshen → embed query → score → results.
//!
//! Orchestrates all building blocks:
//! - [`crate::storage`] — manifest, vectors, chunk texts on disk
//! - [`crate::sparse`] — BM25 indexing and scoring
//! - [`crate::fusion`] — cosine similarity and score combination
//! - [`crate::embeddings`] — query embedding via fastembed
//! - [`crate::index`] — reindex pipeline for stale indexes

use std::collections::BTreeSet;
use std::path::Path;

use notectl_core::config::{Config, SearchConfig};
use notectl_core::progress::ProgressSink;

use crate::fusion::{combine_scores, cosine_top_k};
use crate::index::document_title;
use crate::loaded::{DenseVectors, IndexCache};
use crate::rerank::Reranker;
use crate::sparse::SparseIndexer;
use crate::storage::{SearchManifest, StalenessDiff};
use crate::{RankedChunk, SearchError, SearchResult};

/// Default empty string for fallback when chunk text is missing.
const EMPTY_STR: &str = "";

use crate::embeddings::{Embedder, EmbeddingConfig, TaskType};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Which scoring paths to run during search.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
    clap::ValueEnum,
)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum SearchMode {
    /// Weighted mean of dense (cosine) and sparse (BM25) scores.
    #[default]
    Hybrid,
    /// Dense (cosine similarity) only.
    Dense,
    /// Sparse (BM25) only.
    Sparse,
}

impl SearchMode {
    /// Does this mode require dense vectors?
    pub fn needs_dense(&self) -> bool {
        matches!(self, SearchMode::Hybrid | SearchMode::Dense)
    }

    /// Does this mode require BM25?
    pub fn needs_sparse(&self) -> bool {
        matches!(self, SearchMode::Hybrid | SearchMode::Sparse)
    }
}

/// Data required to score results in a given search mode.
///
/// Each variant owns exactly the inputs its scoring path needs — no Option
/// plumbing or runtime assertions.  Built once after all auto-degradation
/// decisions so the type system guarantees correctness.
enum ScoreInputs<'a> {
    Dense {
        query_vec: Vec<f32>,
        dense: &'a DenseVectors,
    },
    Sparse {
        indexer: &'a SparseIndexer,
    },
    Hybrid {
        query_vec: Vec<f32>,
        dense: &'a DenseVectors,
        indexer: &'a SparseIndexer,
    },
}

/// Options controlling search behavior.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct SearchOptions {
    /// Which scoring paths to use.
    pub mode: SearchMode,
    /// Maximum number of results to return.
    pub max_results: usize,
    /// Weight of the BM25 score in hybrid mode (default 1.0).
    pub rrf_bm25_weight: f64,
    /// Weight of the cosine similarity in hybrid mode (default 1.0).
    pub rrf_cosine_weight: f64,
    /// How strongly to favor recent documents (default 0.5). Set to 0 to disable.
    pub rrf_recency_weight: f64,
    /// Skip staleness check and reindexing; use existing index as-is.
    pub no_reindex: bool,
    /// Filter results to only chunks whose file has ALL specified tags (AND logic).
    #[serde(default)]
    pub tags: Vec<String>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        let sc = SearchConfig::default();
        Self {
            mode: SearchMode::default(),
            max_results: sc.max_results,
            rrf_bm25_weight: sc.rrf_bm25_weight,
            rrf_cosine_weight: sc.rrf_cosine_weight,
            rrf_recency_weight: sc.rrf_recency_weight,
            no_reindex: false,
            tags: Vec::new(),
        }
    }
}

impl SearchOptions {
    /// Build from a [`SearchConfig`].
    pub fn from_config(config: &SearchConfig) -> Self {
        Self {
            mode: SearchMode::default(),
            max_results: config.max_results,
            rrf_bm25_weight: config.rrf_bm25_weight,
            rrf_cosine_weight: config.rrf_cosine_weight,
            rrf_recency_weight: config.rrf_recency_weight,
            no_reindex: false,
            tags: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Search pipeline
// ---------------------------------------------------------------------------

/// Outcome of a search operation: ranked results plus the effective search mode.
///
/// The `mode_used` field reflects the actual mode the search ran in, which may
/// differ from the requested mode due to auto-degradation (e.g., Hybrid → Sparse
/// when vectors are missing).
#[derive(Debug, Clone)]
pub struct SearchOutcome {
    /// Ranked search results.
    pub results: Vec<RankedChunk>,
    /// Effective search mode used (may differ from requested due to auto-degradation).
    pub mode_used: SearchMode,
    /// Whether the rerank model decided the results. False when none is
    /// configured, it failed, or the index is empty.
    pub reranked: bool,
}

/// Run the full search pipeline.
///
/// Steps:
/// 1. **Freshen** — staleness check + conditional reindex (unless `no_reindex`).
/// 2. **Load** — manifest chunks, chunk texts, dense vectors, rebuild BM25.
/// 3. **Embed query** — only for Dense/Hybrid mode.
/// 4. **Score & rank** — by the rerank model when configured; otherwise by
///    cosine similarity, scaled BM25, or their weighted mean.
/// 5. **Map** — chunk indices back to [`RankedChunk`] with file path, heading, preview.
pub async fn search(
    base_path: &Path,
    config: &Config,
    query: &str,
    options: SearchOptions,
    progress: ProgressSink,
    cache: &IndexCache,
) -> SearchResult<SearchOutcome> {
    let mut loaded = cache.load(base_path, config)?;

    // ---- Step 1: Freshen (staleness check + conditional reindex) ----
    if !options.no_reindex {
        let diff = crate::storage::compute_staleness_diff(base_path, config, loaded.manifest())?;
        if !matches!(diff, StalenessDiff::UpToDate) {
            // Rebuild the index to pick up changes.
            let summary = crate::index::build_index(base_path, config, false, progress).await?;
            tracing::info!(
                "Reindex complete: {} files, {} chunks, embeddings={}",
                summary.files_indexed,
                summary.chunks_produced,
                summary.has_embeddings
            );
            // Let go of the old version so the cache can free it before
            // reading the new one.
            drop(loaded);
            loaded = cache.load(base_path, config)?;
        }
    }

    let manifest = loaded.manifest();

    // Empty corpus → empty results.
    if manifest.chunks.is_empty() {
        tracing::debug!("Empty corpus: no chunks indexed.");
        return Ok(SearchOutcome {
            results: Vec::new(),
            mode_used: options.mode,
            reranked: false,
        });
    }

    // ---- Step 2: Load index artifacts ----

    let chunk_texts = loaded.chunk_texts();

    // Read dense vectors only if the requested mode could use them.
    // For Sparse mode, skip entirely — has_vectors cannot affect the outcome.
    let dense = options.mode.needs_dense().then(|| loaded.dense());
    let has_vectors = dense.is_some_and(|d| !d.vectors.is_empty());

    // -----------------------------------------------------------------------
    // Steps 2-4: Load artifacts, embed query, score & rank
    //
    // All degradation decisions are collapsed into a single ScoreInputs
    // construction point so the type system guarantees the right data is
    // present for each scoring path — no .expect() / .unwrap() needed.
    // -----------------------------------------------------------------------

    // Determine effective mode (auto-degrade if vectors missing on disk).
    let effective_mode = match (options.mode, has_vectors) {
        (SearchMode::Dense, false) => {
            tracing::warn!(
                "Dense mode requested but no vectors available. Auto-degrading to sparse."
            );
            SearchMode::Sparse
        }
        (SearchMode::Hybrid, false) => {
            tracing::warn!(
                "Hybrid mode requested but no vectors available. Auto-degrading to sparse."
            );
            SearchMode::Sparse
        }
        (mode, _) => mode,
    };

    // Embed query + read vectors when the mode needs dense scoring.
    let dense_data: Option<(Vec<f32>, &DenseVectors)> = if effective_mode.needs_dense()
        && let Some(dense) = dense
        && let Some(emb_config) = EmbeddingConfig::from_search_config(&config.search)
    {
        let mut embedder = Embedder::new(emb_config);

        match embedder
            .embed_single(query, None, TaskType::RetrievalQuery)
            .await
        {
            Ok(qvec) => Some((qvec, dense)),
            Err(e) => {
                tracing::warn!("Query embedding failed: {e}. Degrading to sparse.");
                None
            }
        }
    } else {
        None
    };

    // Final mode after embedding availability is known.
    let final_mode = match (effective_mode, dense_data.as_ref()) {
        (SearchMode::Dense, None) => {
            tracing::warn!("Dense mode unavailable, falling back to sparse.");
            SearchMode::Sparse
        }
        (SearchMode::Hybrid, None) => {
            tracing::warn!("Dense component unavailable, running sparse-only for hybrid query.");
            SearchMode::Sparse
        }
        (mode, _) => mode,
    };

    // Build BM25 indexer based on final_mode (not effective_mode), so degradation
    // from Dense → Sparse at query time still gets a working sparse indexer.
    let sparse_indexer: Option<&SparseIndexer> = final_mode.needs_sparse().then(|| loaded.sparse());

    // Capture debug info before moving values into the match.
    let has_dense = dense_data.is_some();
    let has_sparse = sparse_indexer.is_some();

    // Construct typed scoring inputs — the type system now guarantees correctness.
    let inputs = match (final_mode, dense_data, sparse_indexer) {
        // Dense: needs query_vec + vectors
        (SearchMode::Dense, Some((qvec, dense)), _) => ScoreInputs::Dense {
            query_vec: qvec,
            dense,
        },
        // Sparse: needs indexer
        (SearchMode::Sparse, _, Some(indexer)) => ScoreInputs::Sparse { indexer },
        // Hybrid: needs all three
        (SearchMode::Hybrid, Some((qvec, dense)), Some(indexer)) => ScoreInputs::Hybrid {
            query_vec: qvec,
            dense,
            indexer,
        },
        // Any other combination means an internal invariant was violated.
        // Return an error instead of panicking.
        _ => {
            return Err(SearchError::Other(format!(
                "Inconsistent search state: mode={:?}, has_dense={}, has_sparse={}",
                final_mode, has_dense, has_sparse,
            )));
        }
    };

    // Score each path the mode uses, with the weight each carries in the
    // combined score.
    let (mut dense_scores, mut sparse_scores, w_dense, w_sparse) = match inputs {
        ScoreInputs::Dense { query_vec, dense } => {
            (score_dense(dense, &query_vec), Vec::new(), 1.0, 0.0)
        }
        ScoreInputs::Sparse { indexer } => (Vec::new(), indexer.score_query(query), 0.0, 1.0),
        ScoreInputs::Hybrid {
            query_vec,
            dense,
            indexer,
        } => (
            score_dense(dense, &query_vec),
            indexer.score_query(query),
            options.rrf_cosine_weight,
            options.rrf_bm25_weight,
        ),
    };

    // Drop chunks without the requested tags (AND logic, case-insensitive)
    // before anything picks the best of them.
    let filter_tags: Vec<String> = options.tags.iter().map(|t| t.to_lowercase()).collect();
    let tagged = |idx: usize| {
        filter_tags.is_empty()
            || manifest.chunks.get(idx).is_some_and(|entry| {
                let tags: Vec<String> = entry.tags.iter().map(|t| t.to_lowercase()).collect();
                filter_tags.iter().all(|wanted| tags.contains(wanted))
            })
    };
    dense_scores.retain(|&(idx, _)| tagged(idx));
    sparse_scores.retain(|&(idx, _)| tagged(idx));

    let reranked = match Reranker::from_search_config(&config.search) {
        Some(reranker) => {
            // Both lists are sorted best first.
            let pool = config.search.rerank_candidates.max(1);
            let dense = dense_scores
                .iter()
                .take(pool)
                .filter(|&&(_, score)| f64::from(score) >= config.search.rerank_min_cosine);
            let sparse = sparse_scores
                .iter()
                .take(pool)
                .filter(|&&(_, score)| score >= config.search.rerank_min_bm25);
            let candidates: BTreeSet<usize> = dense
                .map(|&(idx, _)| idx)
                .chain(sparse.map(|&(idx, _)| idx))
                .collect();
            let max_chars = config.search.rerank_max_chars.max(1);
            let documents = rerank_documents(candidates, manifest, chunk_texts, max_chars);
            rerank(&reranker, query, documents).await
        }
        None => None,
    };
    let was_reranked = reranked.is_some();
    let scored = reranked
        .unwrap_or_else(|| combine_scores(&dense_scores, &sparse_scores, w_dense, w_sparse));

    // Apply recency weighting if configured.
    let scored = if options.rrf_recency_weight > 0.0 {
        apply_recency_boost(&scored, manifest, options.rrf_recency_weight)
    } else {
        scored
    };

    // Truncate to max_results.
    let top_results: Vec<(usize, f64)> = scored.into_iter().take(options.max_results).collect();

    // ---- Step 5: Map chunk indices to RankedChunk ----
    let results: Vec<RankedChunk> = top_results
        .into_iter()
        .filter_map(|(chunk_idx, score)| {
            let entry = manifest.chunks.get(chunk_idx)?;

            let text = chunk_texts
                .get(chunk_idx)
                .map(|s| s.as_str())
                .unwrap_or(EMPTY_STR);
            let preview = extract_preview(text, 200);
            let heading = if entry.heading_path.is_empty() {
                entry.heading.clone()
            } else {
                Some(entry.heading_path.join(" > "))
            };

            Some(RankedChunk {
                id: entry.id.clone(),
                source_file: entry.source_file.clone(),
                score,
                heading,
                tags: entry.tags.clone(),
                date: entry.date,
                preview,
            })
        })
        .collect();

    Ok(SearchOutcome {
        results,
        mode_used: final_mode,
        reranked: was_reranked,
    })
}

/// Cosine similarity of every stored vector, highest first, reported as chunk
/// indices.
fn score_dense(dense: &DenseVectors, query_vec: &[f32]) -> Vec<(usize, f32)> {
    cosine_top_k(&dense.vectors, query_vec, dense.vectors.len())
        .into_iter()
        .map(|(i, score)| (dense.chunks[i], score))
        .collect()
}

/// What the rerank model reads for each candidate chunk: its title and the
/// start of its text, paired with the chunk's index.
fn rerank_documents(
    candidates: BTreeSet<usize>,
    manifest: &SearchManifest,
    chunk_texts: &[String],
    max_chars: usize,
) -> Vec<(usize, String)> {
    candidates
        .into_iter()
        .filter_map(|idx| {
            let entry = manifest.chunks.get(idx)?;
            let text = chunk_texts.get(idx).map_or(EMPTY_STR, String::as_str);
            let text = truncate_chars(text, max_chars);
            let title = document_title(
                &entry.source_file,
                &entry.heading_path,
                entry.heading.as_deref(),
            );
            let document = match title {
                Some(title) => format!("{title}\n{text}"),
                None => text.to_string(),
            };
            Some((idx, document))
        })
        .collect()
}

/// Score the candidate documents with the rerank model, best first. No
/// candidates means no results. `None` when the model gives no usable answer.
async fn rerank(
    reranker: &Reranker<'_>,
    query: &str,
    candidates: Vec<(usize, String)>,
) -> Option<Vec<(usize, f64)>> {
    if candidates.is_empty() {
        return Some(Vec::new());
    }
    let (indices, documents): (Vec<usize>, Vec<String>) = candidates.into_iter().unzip();

    match reranker.score(query, &documents).await {
        Ok(scores) => {
            let mut scored: Vec<(usize, f64)> = indices.into_iter().zip(scores).collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));
            Some(scored)
        }
        Err(e) => {
            tracing::warn!("Reranking failed: {e}. Using combined scores.");
            None
        }
    }
}

/// The first `max` characters of `text`.
fn truncate_chars(text: &str, max: usize) -> &str {
    text.char_indices()
        .nth(max)
        .map_or(text, |(end, _)| &text[..end])
}

/// Apply recency weighting to scored search results.
///
/// For each chunk, looks up its document date (from frontmatter) or falls back
/// to the file's mtime. Normalizes dates to [0, 1] relative to the corpus,
/// then scales scores: `adjusted = score * (1 + weight * factor) / (1 + weight)`.
/// The newest keeps its score and older ones lose some, so scores stay in
/// `0..=1` and recency alone never lifts a poor match.
fn apply_recency_boost(
    scored: &[(usize, f64)],
    manifest: &crate::storage::SearchManifest,
    weight: f64,
) -> Vec<(usize, f64)> {
    if scored.is_empty() {
        return scored.to_vec();
    }

    // Build a map from chunk_idx to epoch seconds.
    // Fall back to file mtime when no frontmatter date is available.
    let mut file_mtime_cache: std::collections::HashMap<String, u64> =
        std::collections::HashMap::new();
    let dates: Vec<u64> = scored
        .iter()
        .map(|&(chunk_idx, _)| {
            let entry = manifest.chunks.get(chunk_idx);
            entry
                .map(|e| {
                    // Prefer frontmatter date; fall back to file mtime
                    e.date.unwrap_or_else(|| {
                        *file_mtime_cache
                            .entry(e.source_file.clone())
                            .or_insert_with(|| {
                                manifest
                                    .files
                                    .iter()
                                    .find(|f| f.path == e.source_file)
                                    .map(|f| f.mtime)
                                    .unwrap_or(0)
                            })
                    })
                })
                .unwrap_or(0)
        })
        .collect();

    // Find min/max for normalization.
    let min_date = dates.iter().min().copied().unwrap_or(0);
    let max_date = dates.iter().max().copied().unwrap_or(0);
    let range = max_date as f64 - min_date as f64;

    // If all dates are the same, use factor 0.5 (neutral).
    let factors: Vec<f64> = if range < 1.0 {
        vec![0.5; dates.len()]
    } else {
        dates
            .iter()
            .map(|&d| (d as f64 - min_date as f64) / range)
            .collect()
    };

    // Apply boost and re-sort.
    let mut boosted: Vec<(usize, f64)> = scored
        .iter()
        .zip(factors)
        .map(|((idx, score), factor)| (*idx, *score * (1.0 + weight * factor) / (1.0 + weight)))
        .collect();

    boosted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    boosted
}

/// Extract a preview of ~`max_len` characters from the beginning of the text.
fn extract_preview(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.trim().to_string()
    } else {
        // Find a good break point (whitespace near max_len).
        let truncated: String = text.chars().take(max_len).collect();
        truncated.trim().to_string()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::tests::FakeEndpoint;
    use crate::storage::ChunkConfigSnapshot;
    use crate::storage::SearchIndex;
    use notectl_core::config::SearchConfig;
    use tempfile::TempDir;

    fn test_config() -> Config {
        Config {
            exclude_paths: Vec::new(),
            daily_note_patterns: vec!["YYYY-MM-DD.md".to_string()],
            exclude_headings: Vec::new(),
            search: SearchConfig::default(),
        }
    }

    /// Build the index without a real embedder.
    ///
    /// These tests only exercise sparse search and dense-degradation behavior —
    /// none of them need real dense vectors — so skip the real embedding
    /// backend entirely (no network, no multi-GB model load, deterministic
    /// regardless of what's cached locally).
    async fn build_index_no_embed(base: &Path, config: &Config) -> crate::index::BuildSummary {
        let chunk_config = ChunkConfigSnapshot {
            max_tokens: config.search.max_seq_tokens,
            overlap_tokens: config.search.chunk_overlap_tokens,
            min_chunk_size: config.search.min_chunk_tokens,
            merge_threshold: config.search.merge_threshold,
            exclude_headings: Vec::new(),
        };
        let mut index = SearchIndex::open_or_create(
            &config.search.resolve_index_dir(base),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            chunk_config,
        )
        .unwrap();
        let chunker =
            crate::chunker::Chunker::new(crate::chunker::ChunkerConfig::from_config(config));
        let mut builder = crate::index::IndexBuilder::new(&mut index, &chunker, None);
        builder.build(base, config).await.unwrap()
    }

    // ---- SearchMode tests ----

    #[test]
    fn test_search_mode_needs_dense() {
        assert!(SearchMode::Hybrid.needs_dense());
        assert!(SearchMode::Dense.needs_dense());
        assert!(!SearchMode::Sparse.needs_dense());
    }

    #[test]
    fn test_search_mode_needs_sparse() {
        assert!(SearchMode::Hybrid.needs_sparse());
        assert!(!SearchMode::Dense.needs_sparse());
        assert!(SearchMode::Sparse.needs_sparse());
    }

    #[test]
    fn test_search_mode_default() {
        assert_eq!(SearchMode::default(), SearchMode::Hybrid);
    }

    /// SearchMode JSON (de)serialization must use lowercase variant names.
    /// Regression test for TASK-12: HTTP/MCP clients send {"mode":"hybrid"}
    /// per the documented schema, so serde must accept lowercase.
    #[test]
    fn test_search_mode_json_uses_lowercase() {
        // Serialize produces lowercase
        assert_eq!(
            serde_json::to_string(&SearchMode::Hybrid).unwrap(),
            "\"hybrid\""
        );
        assert_eq!(
            serde_json::to_string(&SearchMode::Dense).unwrap(),
            "\"dense\""
        );
        assert_eq!(
            serde_json::to_string(&SearchMode::Sparse).unwrap(),
            "\"sparse\""
        );

        // Deserialize accepts lowercase
        assert_eq!(
            serde_json::from_str::<SearchMode>("\"hybrid\"").unwrap(),
            SearchMode::Hybrid
        );
        assert_eq!(
            serde_json::from_str::<SearchMode>("\"dense\"").unwrap(),
            SearchMode::Dense
        );
        assert_eq!(
            serde_json::from_str::<SearchMode>("\"sparse\"").unwrap(),
            SearchMode::Sparse
        );
    }

    // ---- SearchOptions tests ----

    #[test]
    fn test_search_options_default() {
        let opts = SearchOptions::default();
        assert_eq!(opts.mode, SearchMode::Hybrid);
        assert_eq!(opts.max_results, 50);
        assert!((opts.rrf_bm25_weight - 1.0).abs() < f64::EPSILON);
        assert!((opts.rrf_cosine_weight - 1.0).abs() < f64::EPSILON);
        assert!(!opts.no_reindex);
    }

    #[test]
    fn test_search_options_from_config() {
        let sc = SearchConfig {
            max_results: 10,
            rrf_bm25_weight: 2.0,
            rrf_cosine_weight: 0.5,
            ..Default::default()
        };
        let opts = SearchOptions::from_config(&sc);
        assert_eq!(opts.max_results, 10);
        assert!((opts.rrf_bm25_weight - 2.0).abs() < f64::EPSILON);
        assert!((opts.rrf_cosine_weight - 0.5).abs() < f64::EPSILON);
    }

    // ---- extract_preview tests ----

    #[test]
    fn test_extract_preview_short_text() {
        let text = "Short text";
        assert_eq!(extract_preview(text, 200), "Short text");
    }

    #[test]
    fn test_extract_preview_long_text() {
        let text = "a".repeat(500);
        let preview = extract_preview(&text, 200);
        assert_eq!(preview.len(), 200);
        assert!(preview.chars().all(|c| c == 'a'));
    }

    #[test]
    fn test_extract_preview_exact_length() {
        let text = "exact";
        assert_eq!(extract_preview(text, 5), "exact");
    }

    // ---- Integration: end-to-end search with synthetic data (no model needed) ----

    /// Build a small test index with known content, then search it.
    /// Without the `embeddings` feature this exercises the sparse-only path.
    #[tokio::test]
    async fn test_search_sparse_only() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        std::fs::create_dir_all(&base).unwrap();

        // Create test files with sufficiently long content to produce chunks.
        std::fs::write(
            base.join("rust.md"),
            "# Rust Programming\n\nRust is a systems programming language focused on safety and performance. It provides fine-grained control over memory management while guaranteeing thread safety through its ownership and borrowing system. Rust has excellent tooling including cargo for package management and rustfmt for code formatting. The compiler provides helpful error messages that guide developers toward correct solutions.",
        )
        .unwrap();
        std::fs::write(
            base.join("python.md"),
            "# Python Guide\n\nPython is great for data science and machine learning. It has a rich ecosystem of libraries including numpy, pandas, and scikit-learn for statistical analysis. Python's dynamic typing and interpreted nature make it ideal for rapid prototyping and scripting tasks. The standard library provides extensive functionality for common programming patterns.",
        )
        .unwrap();
        std::fs::write(
            base.join("graphql.md"),
            "# GraphQL API Design\n\nGraphQL allows clients to request exactly the data they need. This reduces over-fetching and under-fetching problems common with REST APIs. Schema design follows best practices with clear type definitions and resolver functions. Federation enables composing multiple subgraphs into a unified supergraph schema for large-scale distributed systems.",
        )
        .unwrap();

        let config = test_config();

        // Build the index first.
        let summary = build_index_no_embed(&base, &config).await;
        assert!(
            summary.chunks_produced >= 3,
            "Expected at least 3 chunks, got {}",
            summary.chunks_produced
        );

        // Search with sparse mode (works without embeddings feature).
        let options = SearchOptions {
            mode: SearchMode::Sparse,
            max_results: 10,
            ..Default::default()
        };

        let outcome = search(
            &base,
            &config,
            "rust programming",
            options,
            notectl_core::progress::ignore(),
            &IndexCache::default(),
        )
        .await
        .unwrap();
        let results = outcome.results;

        // Should get results.
        assert!(!results.is_empty(), "Should have search results");

        // rust.md should rank highest for "rust programming" query.
        assert!(
            results[0].source_file.contains("rust"),
            "Top result should be rust.md, got: {}",
            results[0].source_file
        );
        assert!(results[0].score > 0.0, "Score should be positive");
    }

    /// Test search with an empty vault returns empty results.
    #[tokio::test]
    async fn test_search_empty_vault() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        std::fs::create_dir_all(&base).unwrap();

        let config = test_config();
        let options = SearchOptions {
            mode: SearchMode::Sparse,
            ..Default::default()
        };

        let outcome = search(
            &base,
            &config,
            "anything",
            options,
            notectl_core::progress::ignore(),
            &IndexCache::default(),
        )
        .await
        .unwrap();
        assert!(outcome.results.is_empty());
    }

    /// Test search respects max_results limit.
    #[tokio::test]
    async fn test_search_max_results_limit() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        std::fs::create_dir_all(&base).unwrap();

        // Create multiple files with similar content.
        for i in 0..5 {
            std::fs::write(
                base.join(format!("note{}.md", i)),
                format!(
                    "# Note {}\n\nThis note discusses programming languages and software development.",
                    i
                ),
            )
            .unwrap();
        }

        let config = test_config();
        build_index_no_embed(&base, &config).await;

        // Limit results to 2.
        let options = SearchOptions {
            mode: SearchMode::Sparse,
            max_results: 2,
            ..Default::default()
        };

        let outcome = search(
            &base,
            &config,
            "programming",
            options,
            notectl_core::progress::ignore(),
            &IndexCache::default(),
        )
        .await
        .unwrap();
        let results = outcome.results;
        assert!(
            results.len() <= 2,
            "Should have at most 2 results, got {}",
            results.len()
        );
    }

    /// Test no_reindex flag uses existing index without rebuilding.
    #[tokio::test]
    async fn test_search_no_reindex_uses_existing() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        std::fs::create_dir_all(&base).unwrap();

        std::fs::write(
            base.join("original.md"),
            "# Original Content\n\nThis is the original note about databases.",
        )
        .unwrap();

        let config = test_config();
        build_index_no_embed(&base, &config).await;

        // Add a new file after indexing.
        std::fs::write(
            base.join("new-file.md"),
            "# New File\n\nThis new file talks about cloud computing.",
        )
        .unwrap();

        // Search with no_reindex=true should NOT find the new file.
        let options = SearchOptions {
            mode: SearchMode::Sparse,
            no_reindex: true,
            ..Default::default()
        };

        let outcome = search(
            &base,
            &config,
            "cloud computing",
            options,
            notectl_core::progress::ignore(),
            &IndexCache::default(),
        )
        .await
        .unwrap();
        let results = outcome.results;

        // The new file shouldn't appear since we didn't reindex.
        assert!(
            !results.iter().any(|r| r.source_file.contains("new-file")),
            "New file should not appear with no_reindex=true"
        );
    }

    /// Test result ranking — results should be sorted by descending score.
    #[tokio::test]
    async fn test_search_results_sorted_by_score() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        std::fs::create_dir_all(&base).unwrap();

        std::fs::write(
            base.join("match.md"),
            "# Exact Match\n\nRust memory safety and zero-cost abstractions are key features.",
        )
        .unwrap();
        std::fs::write(
            base.join("partial.md"),
            "# Partial Match\n\nSome unrelated content here with a few keywords.",
        )
        .unwrap();
        std::fs::write(
            base.join("nomatch.md"),
            "# No Match\n\nCompletely different topic about cooking recipes.",
        )
        .unwrap();

        let config = test_config();
        build_index_no_embed(&base, &config).await;

        let options = SearchOptions {
            mode: SearchMode::Sparse,
            max_results: 10,
            ..Default::default()
        };

        let outcome = search(
            &base,
            &config,
            "rust memory safety",
            options,
            notectl_core::progress::ignore(),
            &IndexCache::default(),
        )
        .await
        .unwrap();
        let results = outcome.results;

        // Results should be sorted descending by score.
        for window in results.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "Results must be sorted by descending score: {:.4} >= {:.4}",
                window[0].score,
                window[1].score
            );
        }

        // match.md should rank highest.
        if !results.is_empty() {
            assert!(
                results[0].source_file.contains("match"),
                "Top result should be match.md, got: {}",
                results[0].source_file
            );
        }
    }

    /// Test RankedChunk fields are populated correctly.
    #[tokio::test]
    async fn test_ranked_chunk_fields() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        std::fs::create_dir_all(&base).unwrap();

        std::fs::write(
            base.join("test.md"),
            "# Test Document\n\n## Section One\n\nContent about testing frameworks and unit tests.",
        )
        .unwrap();

        let config = test_config();
        build_index_no_embed(&base, &config).await;

        let options = SearchOptions {
            mode: SearchMode::Sparse,
            max_results: 10,
            ..Default::default()
        };

        let outcome = search(
            &base,
            &config,
            "testing frameworks",
            options,
            notectl_core::progress::ignore(),
            &IndexCache::default(),
        )
        .await
        .unwrap();
        let results = outcome.results;

        assert!(!results.is_empty(), "Should have results");

        // Verify each result has required fields populated.
        for result in &results {
            assert!(!result.id.is_empty(), "id should not be empty");
            assert!(
                !result.source_file.is_empty(),
                "source_file should not be empty"
            );
            assert!(result.score > 0.0, "score should be positive");
            // Preview should be reasonable length.
            assert!(
                result.preview.len() <= 200,
                "preview should be <= 200 chars, got {}",
                result.preview.len()
            );
        }
    }

    /// Test that Dense mode degrades to Sparse when embedding is unavailable at query time.
    ///
    /// Reproduces TASK-10: vectors.bin exists on disk (simulating a previous indexing run),
    /// but the model cache is missing so Embedder::is_ready() returns false.
    /// search() with SearchMode::Dense should degrade to sparse-only and return Ok,
    /// NOT Err(SearchError::Other("Inconsistent search state...")).
    #[tokio::test]
    async fn test_dense_mode_degrades_to_sparse_when_embedding_unavailable() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        std::fs::create_dir_all(&base).unwrap();

        // Create files with enough content to produce chunks.
        std::fs::write(
            base.join("rust.md"),
            "# Rust Programming\n\nRust is a systems programming language focused on safety and performance. It provides fine-grained control over memory management while guaranteeing thread safety through its ownership and borrowing system. Rust has excellent tooling including cargo for package management and rustfmt for code formatting.",
        )
        .unwrap();
        std::fs::write(
            base.join("python.md"),
            "# Python Guide\n\nPython is great for data science and machine learning. It has a rich ecosystem of libraries including numpy, pandas, and scikit-learn for statistical analysis. Python's dynamic typing and interpreted nature make it ideal for rapid prototyping and scripting tasks.",
        )
        .unwrap();

        let config = test_config();

        // Build the index first (without a real model, no embeddings produced).
        build_index_no_embed(&base, &config).await;

        // Re-open the index to get a fresh handle after build_index wrote artifacts.
        let index_dir = config.search.resolve_index_dir(&base);
        let chunk_config = crate::storage::ChunkConfigSnapshot {
            max_tokens: config.search.max_seq_tokens,
            overlap_tokens: config.search.chunk_overlap_tokens,
            min_chunk_size: config.search.min_chunk_tokens,
            merge_threshold: config.search.merge_threshold,
            exclude_headings: Vec::new(),
        };
        let index = crate::storage::SearchIndex::open_or_create(
            &index_dir,
            config.search.model_id.clone(),
            config.search.embedding_dim,
            chunk_config.clone(),
        )
        .unwrap();

        // Write fake vectors to simulate "vectors exist from a previous indexing run".
        let chunk_count = index.manifest().chunks.len();
        assert!(chunk_count > 0, "Expected chunks in manifest");
        let dim = config.search.embedding_dim as usize;
        let mut writer = index.begin_vector_write(dim).unwrap();
        for entry in &index.manifest().chunks {
            let key = blake3::Hash::from_hex(&entry.embedding_key).unwrap();
            writer.write(key, &vec![0.1f32; dim]).unwrap();
        }
        writer.finish().unwrap();

        // Now search with Dense mode. The model cache dir doesn't have a downloaded model,
        // so Embedder::is_ready() == false, but has_vectors == true (fake vectors.bin).
        // This should degrade to sparse and return Ok, NOT hard-error.
        let options = SearchOptions {
            mode: SearchMode::Dense,
            max_results: 10,
            ..Default::default()
        };

        let result = search(
            &base,
            &config,
            "rust programming",
            options,
            notectl_core::progress::ignore(),
            &IndexCache::default(),
        )
        .await;

        // Assert: should be Ok with sparse-scored results, NOT Err.
        match result {
            Ok(outcome) => {
                assert!(
                    !outcome.results.is_empty(),
                    "Should have sparse-scored results after degradation"
                );
                assert!(
                    outcome.results[0].score > 0.0,
                    "Top result should have positive score"
                );
                // mode_used should reflect the actual degraded mode (Sparse), not Dense.
                assert!(
                    matches!(outcome.mode_used, SearchMode::Sparse),
                    "mode_used should be Sparse after degradation, got {:?}",
                    outcome.mode_used
                );
            }
            Err(e) => {
                panic!("search() should degrade to sparse, not error. Got: {:?}", e);
            }
        }
    }

    /// Test that SearchOutcome.mode_used reflects actual degradation.
    ///
    /// Regression test for TASK-13: when Dense or Hybrid is requested but no
    /// vectors are available, the returned mode_used must be Sparse (the actual
    /// mode used), NOT the originally requested mode.
    #[tokio::test]
    async fn test_search_mode_used_reflects_degradation() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        std::fs::create_dir_all(&base).unwrap();

        // Create files with enough content to produce chunks.
        std::fs::write(
            base.join("rust.md"),
            "# Rust Programming\n\nRust is a systems programming language focused on safety and performance. It provides fine-grained control over memory management while guaranteeing thread safety through its ownership and borrowing system. Rust has excellent tooling including cargo for package management and rustfmt for code formatting.",
        )
        .unwrap();
        std::fs::write(
            base.join("python.md"),
            "# Python Guide\n\nPython is great for data science and machine learning. It has a rich ecosystem of libraries including numpy, pandas, and scikit-learn for statistical analysis. Python's dynamic typing and interpreted nature make it ideal for rapid prototyping and scripting tasks.",
        )
        .unwrap();

        let config = test_config();
        build_index_no_embed(&base, &config).await;

        // Request Hybrid mode (which needs dense vectors).
        // Without the embeddings feature (or with no vectors on disk),
        // this should auto-degrade to Sparse.
        let options = SearchOptions {
            mode: SearchMode::Hybrid,
            max_results: 10,
            ..Default::default()
        };

        let outcome = search(
            &base,
            &config,
            "rust programming",
            options,
            notectl_core::progress::ignore(),
            &IndexCache::default(),
        )
        .await
        .unwrap();

        // mode_used should be Sparse (degraded), NOT Hybrid (requested).
        assert!(
            matches!(outcome.mode_used, SearchMode::Sparse),
            "mode_used should be Sparse after degradation, got {:?}",
            outcome.mode_used
        );

        // Results should still be populated (from sparse scoring).
        assert!(
            !outcome.results.is_empty(),
            "Should have results despite degradation"
        );
        assert!(
            outcome.results[0].score > 0.0,
            "Top result should have positive score"
        );

        // Also test Dense → Sparse degradation.
        let options_dense = SearchOptions {
            mode: SearchMode::Dense,
            max_results: 10,
            ..Default::default()
        };

        let outcome_dense = search(
            &base,
            &config,
            "rust programming",
            options_dense,
            notectl_core::progress::ignore(),
            &IndexCache::default(),
        )
        .await
        .unwrap();

        assert!(
            matches!(outcome_dense.mode_used, SearchMode::Sparse),
            "Dense mode should also degrade to Sparse, got {:?}",
            outcome_dense.mode_used
        );
    }

    // ---- Score tests ----

    /// Two notes the fake endpoint embeds at right angles to each other,
    /// searched with a query embedded in the same direction as `near.md`.
    async fn search_embedded(mode: SearchMode, query: &str) -> SearchOutcome {
        search_pair(mode, query, None).await
    }

    async fn search_pair(
        mode: SearchMode,
        query: &str,
        rerank_model: Option<&str>,
    ) -> SearchOutcome {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        let (_fake, mut config) = FakeEndpoint::start().await;
        config.search.rerank_model = rerank_model.map(String::from);
        for (name, word) in [("near.md", "rust "), ("far.md", "orthogonal ")] {
            let note = format!("## Note\n\n{}", word.repeat(80));
            std::fs::write(base.join(name), note).unwrap();
        }
        crate::index::build_index(base, &config, false, notectl_core::progress::ignore())
            .await
            .unwrap();

        let options = SearchOptions {
            mode,
            rrf_recency_weight: 0.0,
            ..Default::default()
        };
        let outcome = search(
            base,
            &config,
            query,
            options,
            notectl_core::progress::ignore(),
            &IndexCache::default(),
        )
        .await
        .unwrap();
        assert_eq!(outcome.mode_used, mode);
        outcome
    }

    fn score_of(outcome: &SearchOutcome, file: &str) -> f64 {
        let result = outcome.results.iter().find(|r| r.source_file == file);
        result.unwrap().score
    }

    #[tokio::test]
    async fn dense_score_is_cosine_similarity() {
        let outcome = search_embedded(SearchMode::Dense, "rust").await;

        assert_eq!(score_of(&outcome, "near.md"), 1.0);
        assert_eq!(score_of(&outcome, "far.md"), 0.0);
    }

    #[tokio::test]
    async fn hybrid_score_averages_cosine_and_bm25() {
        let sparse = search_embedded(SearchMode::Sparse, "rust").await;
        let hybrid = search_embedded(SearchMode::Hybrid, "rust").await;

        let expected = (1.0 + score_of(&sparse, "near.md")) / 2.0;
        assert!((score_of(&hybrid, "near.md") - expected).abs() < 1e-9);
        assert_eq!(score_of(&hybrid, "far.md"), 0.0);
    }

    #[tokio::test]
    async fn reranker_sets_scores_and_order() {
        for mode in [SearchMode::Hybrid, SearchMode::Dense] {
            let outcome = search_pair(mode, "rust", Some("rerank")).await;

            assert!(outcome.reranked);
            assert_eq!(outcome.results[0].source_file, "far.md");
            assert_eq!(score_of(&outcome, "far.md"), 0.9);
            assert_eq!(score_of(&outcome, "near.md"), 0.1);
        }
    }

    #[tokio::test]
    async fn failed_rerank_falls_back_to_combined_scores() {
        let outcome = search_pair(SearchMode::Dense, "rust", Some("unavailable")).await;

        assert!(!outcome.reranked);
        assert_eq!(outcome.results[0].source_file, "near.md");
        assert_eq!(score_of(&outcome, "near.md"), 1.0);
    }

    /// Search a fresh index of `notes` (name, body) through the fake endpoint.
    async fn search_notes(
        notes: &[(&str, String)],
        configure: impl FnOnce(&mut Config),
        options: SearchOptions,
    ) -> SearchOutcome {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        let (_fake, mut config) = FakeEndpoint::start().await;
        config.search.rerank_model = Some("rerank".to_string());
        configure(&mut config);
        for (name, body) in notes {
            std::fs::write(base.join(name), body).unwrap();
        }
        crate::index::build_index(base, &config, false, notectl_core::progress::ignore())
            .await
            .unwrap();
        search(
            base,
            &config,
            "rust",
            options,
            notectl_core::progress::ignore(),
            &IndexCache::default(),
        )
        .await
        .unwrap()
    }

    fn note(body: &str) -> String {
        format!("## Note\n\n{body}")
    }

    fn files(outcome: &SearchOutcome) -> Vec<&str> {
        outcome
            .results
            .iter()
            .map(|r| r.source_file.as_str())
            .collect()
    }

    #[tokio::test]
    async fn reranker_only_reads_chunks_above_a_cutoff() {
        let notes = [
            ("near.md", note(&"rust ".repeat(80))),
            ("far.md", note(&"orthogonal ".repeat(80))),
        ];
        let outcome = search_notes(
            &notes,
            |config| config.search.rerank_min_cosine = 0.5,
            SearchOptions::default(),
        )
        .await;

        assert_eq!(files(&outcome), ["near.md"]);
    }

    #[tokio::test]
    async fn nothing_above_the_cutoffs_means_no_results() {
        let notes = [("near.md", note(&"rust ".repeat(80)))];
        let outcome = search_notes(
            &notes,
            |config| {
                config.search.rerank_min_cosine = 1.1;
                config.search.rerank_min_bm25 = 1.1;
            },
            SearchOptions::default(),
        )
        .await;

        assert!(outcome.reranked);
        assert!(outcome.results.is_empty());
    }

    #[tokio::test]
    async fn reranker_reads_only_the_start_of_a_long_chunk() {
        // One chunk, with the word the fake reranker likes past the cap.
        let body = format!("{}orthogonal", "abcdefghij ".repeat(250));
        let notes = [("long.md", note(&body))];
        let options = SearchOptions {
            rrf_recency_weight: 0.0,
            ..Default::default()
        };
        let outcome = search_notes(&notes, |_| {}, options).await;

        assert_eq!(outcome.results.len(), 1);
        assert_eq!(outcome.results[0].score, 0.1);
    }

    #[tokio::test]
    async fn rerank_limits_come_from_config() {
        let notes = [
            ("a.md", note(&format!("{}orthogonal", "rust ".repeat(80)))),
            ("b.md", note(&"rust ".repeat(80))),
        ];
        let options = SearchOptions {
            mode: SearchMode::Sparse,
            rrf_recency_weight: 0.0,
            ..Default::default()
        };
        let outcome = search_notes(
            &notes,
            |config| {
                config.search.rerank_candidates = 1;
                config.search.rerank_max_chars = 20;
            },
            options,
        )
        .await;

        // One candidate, cut before the word the fake reranker likes.
        assert_eq!(outcome.results.len(), 1);
        assert_eq!(outcome.results[0].score, 0.1);
    }

    #[test]
    fn truncation_counts_characters() {
        assert_eq!(truncate_chars("héllo", 2), "hé");
        assert_eq!(truncate_chars("hé", 5), "hé");
    }

    #[tokio::test]
    async fn tagged_chunks_are_found_behind_many_better_untagged_ones() {
        let mut names: Vec<String> = (0..SearchConfig::default().rerank_candidates + 1)
            .map(|i| format!("{i:02}.md"))
            .collect();
        names.push("zz.md".to_string());
        let notes: Vec<(&str, String)> = names
            .iter()
            .map(|name| {
                let body = note(&"rust ".repeat(80));
                let body = if name == "zz.md" {
                    format!("---\ntags: [keep]\n---\n\n{body}")
                } else {
                    body
                };
                (name.as_str(), body)
            })
            .collect();
        let options = SearchOptions {
            tags: vec!["Keep".to_string()],
            ..Default::default()
        };
        let outcome = search_notes(&notes, |_| {}, options).await;

        assert_eq!(files(&outcome), ["zz.md"]);
    }

    // ---- Recency scoring tests ----

    #[test]
    fn test_recency_boost_weight_zero_returns_identical() {
        let manifest = crate::storage::SearchManifest {
            version: crate::storage::INDEX_FORMAT_VERSION,
            model_id: String::new(),
            embedding_dim: 1024,
            chunk_config: crate::storage::ChunkConfigSnapshot {
                max_tokens: 512,
                overlap_tokens: 64,
                min_chunk_size: 32,
                merge_threshold: 30,
                exclude_headings: Vec::new(),
            },
            files: vec![
                crate::storage::FileInfo {
                    path: "old.md".to_string(),
                    content_hash: "abc".to_string(),
                    mtime: 1_700_000_000,
                    chunk_ids: vec!["old:0".to_string()],
                },
                crate::storage::FileInfo {
                    path: "new.md".to_string(),
                    content_hash: "def".to_string(),
                    mtime: 1_800_000_000,
                    chunk_ids: vec!["new:0".to_string()],
                },
            ],
            chunks: vec![
                crate::storage::ChunkEntry {
                    id: "old:0".to_string(),
                    source_file: "old.md".to_string(),
                    line_start: 0,
                    line_end: 10,
                    heading: None,
                    heading_path: Vec::new(),
                    tags: vec![],
                    date: Some(1_700_000_000),
                    embedding_key: String::new(),
                },
                crate::storage::ChunkEntry {
                    id: "new:0".to_string(),
                    source_file: "new.md".to_string(),
                    line_start: 0,
                    line_end: 10,
                    heading: None,
                    heading_path: Vec::new(),
                    tags: vec![],
                    date: Some(1_800_000_000),
                    embedding_key: String::new(),
                },
            ],
            content_hash: String::new(),
            last_indexed: None,
            has_embeddings: false,
        };

        // Equal scores, weight=0 should preserve original order
        let fused = vec![(0, 1.0), (1, 1.0)];
        let result = apply_recency_boost(&fused, &manifest, 0.0);
        // With zero weight, scores are unchanged but still sorted by score (equal)
        assert_eq!(result.len(), 2);
        // Both should have the same adjusted score (1.0 * (1 + 0 * factor) = 1.0)
        assert!((result[0].1 - 1.0).abs() < f64::EPSILON);
        assert!((result[1].1 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_recency_boost_newest_ranks_higher() {
        let manifest = crate::storage::SearchManifest {
            version: crate::storage::INDEX_FORMAT_VERSION,
            model_id: String::new(),
            embedding_dim: 1024,
            chunk_config: crate::storage::ChunkConfigSnapshot {
                max_tokens: 512,
                overlap_tokens: 64,
                min_chunk_size: 32,
                merge_threshold: 30,
                exclude_headings: Vec::new(),
            },
            files: vec![],
            chunks: vec![
                crate::storage::ChunkEntry {
                    embedding_key: String::new(),
                    id: "old:0".to_string(),
                    source_file: "old.md".to_string(),
                    line_start: 0,
                    line_end: 10,
                    heading: None,
                    heading_path: Vec::new(),
                    tags: vec![],
                    date: Some(1_700_000_000), // older
                },
                crate::storage::ChunkEntry {
                    embedding_key: String::new(),
                    id: "new:0".to_string(),
                    source_file: "new.md".to_string(),
                    line_start: 0,
                    line_end: 10,
                    heading: None,
                    heading_path: Vec::new(),
                    tags: vec![],
                    date: Some(1_800_000_000), // newer
                },
            ],
            content_hash: String::new(),
            last_indexed: None,
            has_embeddings: false,
        };

        // Equal base scores, positive weight should boost newer doc
        let fused = vec![(0, 1.0), (1, 1.0)];
        let result = apply_recency_boost(&fused, &manifest, 0.5);

        // Newest should rank first
        assert_eq!(result[0].0, 1, "newest doc should rank first");
        assert_eq!(result[1].0, 0, "oldest doc should rank second");
        // Newest keeps its score: 1.0 * (1 + 0.5 * 1.0) / 1.5 = 1.0
        assert!((result[0].1 - 1.0).abs() < f64::EPSILON);
        // Oldest is scaled down: 1.0 * (1 + 0.5 * 0.0) / 1.5
        assert!((result[1].1 - 1.0 / 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_recency_boost_all_same_date() {
        let manifest = crate::storage::SearchManifest {
            version: crate::storage::INDEX_FORMAT_VERSION,
            model_id: String::new(),
            embedding_dim: 1024,
            chunk_config: crate::storage::ChunkConfigSnapshot {
                max_tokens: 512,
                overlap_tokens: 64,
                min_chunk_size: 32,
                merge_threshold: 30,
                exclude_headings: Vec::new(),
            },
            files: vec![],
            chunks: vec![
                crate::storage::ChunkEntry {
                    id: "a:0".to_string(),
                    source_file: "a.md".to_string(),
                    line_start: 0,
                    line_end: 10,
                    heading: None,
                    heading_path: Vec::new(),
                    tags: vec![],
                    date: Some(1_750_000_000),
                    embedding_key: String::new(),
                },
                crate::storage::ChunkEntry {
                    id: "b:0".to_string(),
                    source_file: "b.md".to_string(),
                    line_start: 0,
                    line_end: 10,
                    heading: None,
                    heading_path: Vec::new(),
                    tags: vec![],
                    date: Some(1_750_000_000),
                    embedding_key: String::new(),
                },
            ],
            content_hash: String::new(),
            last_indexed: None,
            has_embeddings: false,
        };

        let fused = vec![(0, 1.0), (1, 2.0)];
        let result = apply_recency_boost(&fused, &manifest, 0.5);

        // All same date -> factor 0.5, so both get same multiplier
        let multiplier = (1.0 + 0.5 * 0.5) / 1.5;
        // Higher original score should still rank first
        assert_eq!(result[0].0, 1, "higher original score should rank first");
        assert!((result[0].1 - 2.0 * multiplier).abs() < f64::EPSILON);
        assert!((result[1].1 - 1.0 * multiplier).abs() < f64::EPSILON);
    }

    #[test]
    fn test_recency_boost_falls_back_to_mtime() {
        let manifest = crate::storage::SearchManifest {
            version: crate::storage::INDEX_FORMAT_VERSION,
            model_id: String::new(),
            embedding_dim: 1024,
            chunk_config: crate::storage::ChunkConfigSnapshot {
                max_tokens: 512,
                overlap_tokens: 64,
                min_chunk_size: 32,
                merge_threshold: 30,
                exclude_headings: Vec::new(),
            },
            files: vec![crate::storage::FileInfo {
                path: "no-date.md".to_string(),
                content_hash: "abc".to_string(),
                mtime: 1_700_000_000,
                chunk_ids: vec!["no-date:0".to_string()],
            }],
            chunks: vec![crate::storage::ChunkEntry {
                embedding_key: String::new(),
                id: "no-date:0".to_string(),
                source_file: "no-date.md".to_string(),
                line_start: 0,
                line_end: 10,
                heading: None,
                heading_path: Vec::new(),
                tags: vec![],
                date: None, // no frontmatter date
            }],
            content_hash: String::new(),
            last_indexed: None,
            has_embeddings: false,
        };

        let fused = vec![(0, 1.0)];
        let result = apply_recency_boost(&fused, &manifest, 0.5);

        // Should not panic and should produce results using mtime fallback
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn dense_scores_name_chunks_not_vector_positions() {
        // Chunks 0 and 2 have no vector.
        let dense = DenseVectors {
            chunks: vec![1, 3],
            vectors: vec![vec![0.0, 1.0], vec![1.0, 0.0]],
        };
        let scores = score_dense(&dense, &[1.0, 0.0]);
        assert_eq!(scores, vec![(3, 1.0), (1, 0.0)]);
    }
}
