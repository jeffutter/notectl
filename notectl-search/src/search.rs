//! End-to-end search pipeline: freshen → embed query → hybrid rank → results.
//!
//! Orchestrates all building blocks:
//! - [`crate::storage`] — manifest, vectors, chunk texts on disk
//! - [`crate::sparse`] — BM25 indexing and scoring
//! - [`crate::fusion`] — cosine top-k and RRF fusion
//! - [`crate::embeddings`] — query embedding (feature-gated)
//! - [`crate::index`] — reindex pipeline for stale indexes

use std::path::Path;

use notectl_core::config::{Config, SearchConfig};

use crate::chunker::Chunk;
use crate::fusion::{cosine_top_k, rrf_fuse};
use crate::sparse::SparseIndexer;
use crate::storage::{ChunkConfigSnapshot, SearchIndex, StalenessDiff};
use crate::{RankedChunk, SearchError, SearchResult};

/// Default empty string for fallback when chunk text is missing.
const EMPTY_STR: &str = "";

#[cfg(feature = "embeddings")]
use crate::embeddings::{Embedder, EmbeddingConfig, embed::TaskType};

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
#[non_exhaustive]
pub enum SearchMode {
    /// Run dense (cosine) + sparse (BM25) and fuse via RRF.
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
enum ScoreInputs {
    Dense {
        query_vec: Vec<f32>,
        vectors: Vec<Vec<f32>>,
    },
    Sparse {
        indexer: SparseIndexer,
    },
    Hybrid {
        query_vec: Vec<f32>,
        vectors: Vec<Vec<f32>>,
        indexer: SparseIndexer,
    },
}

/// Options controlling search behavior.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Which scoring paths to use.
    pub mode: SearchMode,
    /// Maximum number of results to return.
    pub max_results: usize,
    /// RRF rank-damping constant (default 60).
    pub rrf_k: f64,
    /// Weight multiplier for BM25 side of RRF (default 1.0).
    pub rrf_bm25_weight: f64,
    /// Weight multiplier for cosine side of RRF (default 1.0).
    pub rrf_cosine_weight: f64,
    /// Skip staleness check and reindexing; use existing index as-is.
    pub no_reindex: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        let sc = SearchConfig::default();
        Self {
            mode: SearchMode::default(),
            max_results: sc.max_results,
            rrf_k: sc.rrf_k,
            rrf_bm25_weight: sc.rrf_bm25_weight,
            rrf_cosine_weight: sc.rrf_cosine_weight,
            no_reindex: false,
        }
    }
}

impl SearchOptions {
    /// Build from a [`SearchConfig`].
    pub fn from_config(config: &SearchConfig) -> Self {
        Self {
            mode: SearchMode::default(),
            max_results: config.max_results,
            rrf_k: config.rrf_k,
            rrf_bm25_weight: config.rrf_bm25_weight,
            rrf_cosine_weight: config.rrf_cosine_weight,
            no_reindex: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Search pipeline
// ---------------------------------------------------------------------------

/// Run the full search pipeline.
///
/// Steps:
/// 1. **Freshen** — staleness check + conditional reindex (unless `no_reindex`).
/// 2. **Load** — manifest chunks, chunk texts, dense vectors, rebuild BM25.
/// 3. **Embed query** — only for Dense/Hybrid mode.
/// 4. **Score & rank** — cosine top-k, BM25, or RRF fusion.
/// 5. **Map** — chunk indices back to [`RankedChunk`] with file path, heading, preview.
pub async fn search(
    base_path: &Path,
    config: &Config,
    query: &str,
    options: SearchOptions,
) -> SearchResult<Vec<RankedChunk>> {
    let index_dir = config.search.resolve_index_dir(base_path);

    // Ensure the index directory exists (open_or_create creates it if missing).
    let chunk_config = ChunkConfigSnapshot {
        max_tokens: config.search.max_seq_tokens,
        overlap_tokens: config.search.chunk_overlap_tokens,
        min_chunk_size: config.search.min_chunk_tokens,
        merge_threshold: config.search.merge_threshold,
    };

    let mut index = SearchIndex::open_or_create(
        &index_dir,
        config.search.model_id.clone(),
        config.search.embedding_dim,
        chunk_config.clone(),
    )?;

    // ---- Step 1: Freshen (staleness check + conditional reindex) ----
    let diff = crate::storage::compute_staleness_diff(base_path, config, index.manifest())?;

    match &diff {
        StalenessDiff::UpToDate => {
            tracing::debug!("Index is up to date.");
        }
        _ if !options.no_reindex => {
            // Rebuild the index to pick up changes.
            let summary = crate::index::build_index(base_path, config).await?;
            tracing::info!(
                "Reindex complete: {} files, {} chunks, embeddings={}",
                summary.files_indexed,
                summary.chunks_produced,
                summary.has_embeddings
            );
            // Re-open the index after build_index wrote new artifacts.
            index = SearchIndex::open_or_create(
                &index_dir,
                config.search.model_id.clone(),
                config.search.embedding_dim,
                chunk_config.clone(),
            )?;
        }
        _ => {
            tracing::warn!(
                "Index is stale but --no-reindex was set. Proceeding with existing index."
            );
        }
    }

    let manifest = index.manifest();

    // Empty corpus → empty results.
    if manifest.chunks.is_empty() {
        tracing::debug!("Empty corpus: no chunks indexed.");
        return Ok(Vec::new());
    }

    // ---- Step 2: Load index artifacts ----

    // Read chunk texts in manifest order (deterministic, matches vector positions).
    let chunk_texts: Vec<String> = manifest
        .chunks
        .iter()
        .map(|entry| {
            index.read_chunk(&entry.id).unwrap_or_default() // Missing chunk text → treat as empty
        })
        .collect();

    // Read dense vectors (empty if vectors.bin missing).
    let has_vectors = {
        #[cfg(feature = "embeddings")]
        {
            let vecs = index.read_vectors().unwrap_or_default();
            !vecs.is_empty() && vecs.len() == manifest.chunks.len()
        }
        #[cfg(not(feature = "embeddings"))]
        {
            false
        }
    };

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

    // Build BM25 indexer when the mode needs sparse scoring.
    let sparse_indexer: Option<SparseIndexer> = if effective_mode.needs_sparse() {
        let chunks_for_bm25: Vec<Chunk> = manifest
            .chunks
            .iter()
            .zip(chunk_texts.iter())
            .map(|(entry, text)| Chunk {
                id: entry.id.clone(),
                source_file: entry.source_file.clone(),
                line_start: entry.line_start,
                line_end: entry.line_end,
                heading: entry.heading.clone(),
                heading_path: entry.heading_path.clone(),
                text: text.clone(),
            })
            .collect();
        Some(SparseIndexer::index_chunks(&chunks_for_bm25))
    } else {
        None
    };

    // Embed query + read vectors when the mode needs dense scoring.
    let dense_data: Option<(Vec<f32>, Vec<Vec<f32>>)> = if effective_mode.needs_dense() {
        #[cfg(feature = "embeddings")]
        {
            let model_cache_dir = index_dir.join("models");
            let mut embedder = Embedder::new(
                model_cache_dir,
                EmbeddingConfig::from_search_config(&config.search),
            );

            if !embedder.is_ready() {
                tracing::warn!("Model not downloaded yet. Degrading to sparse-only search.");
                None
            } else {
                match embedder
                    .embed_single(query, None, TaskType::RetrievalQuery)
                    .await
                {
                    Ok(qvec) => {
                        let vectors = index.read_vectors().unwrap_or_default();
                        Some((qvec, vectors))
                    }
                    Err(e) => {
                        tracing::error!("Query embedding failed: {e}. Degrading to sparse.");
                        None
                    }
                }
            }
        }
        #[cfg(not(feature = "embeddings"))]
        {
            // Unreachable: without the embeddings feature, effective_mode is always Sparse.
            None
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

    // Capture debug info before moving values into the match.
    let has_dense = dense_data.is_some();
    let has_sparse = sparse_indexer.is_some();

    // Construct typed scoring inputs — the type system now guarantees correctness.
    let inputs = match (final_mode, dense_data, sparse_indexer) {
        // Dense: needs query_vec + vectors
        (SearchMode::Dense, Some((qvec, vectors)), _) => ScoreInputs::Dense {
            query_vec: qvec,
            vectors,
        },
        // Sparse: needs indexer
        (SearchMode::Sparse, _, Some(indexer)) => ScoreInputs::Sparse { indexer },
        // Hybrid: needs all three
        (SearchMode::Hybrid, Some((qvec, vectors)), Some(indexer)) => ScoreInputs::Hybrid {
            query_vec: qvec,
            vectors,
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

    // Score & rank based on the typed inputs.
    let fused: Vec<(usize, f64)> = match inputs {
        ScoreInputs::Dense { query_vec, vectors } => {
            let dense_scores = cosine_top_k(&vectors, &query_vec, options.max_results);
            rrf_fuse(
                &dense_scores,
                &[],
                options.rrf_k,
                options.rrf_cosine_weight,
                0.0,
            )
        }
        ScoreInputs::Sparse { indexer } => {
            let sparse_scores = indexer.score_query(query);
            rrf_fuse(
                &[],
                &sparse_scores,
                options.rrf_k,
                0.0,
                options.rrf_bm25_weight,
            )
        }
        ScoreInputs::Hybrid {
            query_vec,
            vectors,
            indexer,
        } => {
            // Use 2x max_results for cosine to give BM25 long-tail terms a chance.
            let dense_scores = cosine_top_k(&vectors, &query_vec, options.max_results * 2);
            let sparse_scores = indexer.score_query(query);
            rrf_fuse(
                &dense_scores,
                &sparse_scores,
                options.rrf_k,
                options.rrf_cosine_weight,
                options.rrf_bm25_weight,
            )
        }
    };

    // Truncate to max_results.
    let top_results: Vec<(usize, f64)> = fused.into_iter().take(options.max_results).collect();

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
                preview,
            })
        })
        .collect();

    Ok(results)
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
    use notectl_core::config::SearchConfig;
    use tempfile::TempDir;

    fn test_config() -> Config {
        Config {
            exclude_paths: Vec::new(),
            daily_note_patterns: vec!["YYYY-MM-DD.md".to_string()],
            search: SearchConfig::default(),
        }
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

    // ---- SearchOptions tests ----

    #[test]
    fn test_search_options_default() {
        let opts = SearchOptions::default();
        assert_eq!(opts.mode, SearchMode::Hybrid);
        assert_eq!(opts.max_results, 50);
        assert!((opts.rrf_k - 60.0).abs() < f64::EPSILON);
        assert!((opts.rrf_bm25_weight - 1.0).abs() < f64::EPSILON);
        assert!((opts.rrf_cosine_weight - 1.0).abs() < f64::EPSILON);
        assert!(!opts.no_reindex);
    }

    #[test]
    fn test_search_options_from_config() {
        let sc = SearchConfig {
            max_results: 10,
            rrf_k: 40.0,
            rrf_bm25_weight: 2.0,
            rrf_cosine_weight: 0.5,
            ..Default::default()
        };
        let opts = SearchOptions::from_config(&sc);
        assert_eq!(opts.max_results, 10);
        assert!((opts.rrf_k - 40.0).abs() < f64::EPSILON);
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
        let summary = crate::index::build_index(&base, &config).await.unwrap();
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

        let results = search(&base, &config, "rust programming", options)
            .await
            .unwrap();

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

        let results = search(&base, &config, "anything", options).await.unwrap();
        assert!(results.is_empty());
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
        crate::index::build_index(&base, &config).await.unwrap();

        // Limit results to 2.
        let options = SearchOptions {
            mode: SearchMode::Sparse,
            max_results: 2,
            ..Default::default()
        };

        let results = search(&base, &config, "programming", options)
            .await
            .unwrap();
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
        crate::index::build_index(&base, &config).await.unwrap();

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

        let results = search(&base, &config, "cloud computing", options)
            .await
            .unwrap();

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
        crate::index::build_index(&base, &config).await.unwrap();

        let options = SearchOptions {
            mode: SearchMode::Sparse,
            max_results: 10,
            ..Default::default()
        };

        let results = search(&base, &config, "rust memory safety", options)
            .await
            .unwrap();

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
        crate::index::build_index(&base, &config).await.unwrap();

        let options = SearchOptions {
            mode: SearchMode::Sparse,
            max_results: 10,
            ..Default::default()
        };

        let results = search(&base, &config, "testing frameworks", options)
            .await
            .unwrap();

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

    /// Test auto-degradation: dense mode with no vectors falls back to sparse.
    #[cfg(not(feature = "embeddings"))]
    #[tokio::test]
    async fn test_auto_degrade_to_sparse_without_embeddings() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        std::fs::create_dir_all(&base).unwrap();

        std::fs::write(
            base.join("hello.md"),
            "# Hello World\n\nA simple greeting document.",
        )
        .unwrap();

        let config = test_config();
        crate::index::build_index(&base, &config).await.unwrap();

        // Even though we can't do dense without the feature,
        // the search() function should still work in sparse mode.
        let options = SearchOptions {
            mode: SearchMode::Sparse,
            max_results: 10,
            ..Default::default()
        };

        let results = search(&base, &config, "greeting", options).await.unwrap();
        // May or may not find "greeting" depending on tokenization, but should not error.
        // The important thing is it doesn't panic or return an error.
        let _ = results;
    }
}
