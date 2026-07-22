//! Index build pipeline: walk -> diff -> chunk -> embed -> persist.
//!
//! `IndexBuilder` is the top-level orchestrator that wires together
//! [`SearchIndex`] (storage), [`Chunker`], and an optional [`Embedder`] into
//! a single build pipeline. It handles incremental updates, full rebuilds,
//! and atomic persistence of all index artifacts.

use std::collections::BTreeMap;
use std::fs;
use std::io::IsTerminal;
use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};
use notectl_core::config::Config;

use crate::SearchError;
use crate::chunker::{Chunk, Chunker};
use crate::storage::{
    ChunkConfigSnapshot, ChunkEntry, FileInfo, SearchIndex, StalenessDiff, blake3_hash_str,
    chrono_now_rfc3339, compute_overall_content_hash,
};

use crate::embeddings::{Embedder, EmbeddingConfig, TaskType};

/// Build summary returned by [`IndexBuilder::build`].
#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct BuildSummary {
    /// Number of files indexed
    pub files_indexed: usize,
    /// Number of chunks produced
    pub chunks_produced: usize,
    /// Whether embeddings were computed
    pub has_embeddings: bool,
    /// Overall content hash
    pub content_hash: String,
}

/// Top-level orchestrator for building/updating the search index.
///
/// Holds references to the storage layer ([`SearchIndex`]), the text
/// splitter ([`Chunker`]), and optionally a dense embedding model
/// ([`Embedder`]). The [`IndexBuilder::build`] method implements the full
/// walk -> diff -> chunk -> embed -> persist pipeline.
pub struct IndexBuilder<'a> {
    /// Storage layer — owns manifest, chunk files, vectors on disk.
    index: &'a mut SearchIndex,
    /// Text splitter configured from search config.
    chunker: &'a Chunker,
    /// Optional dense embedder (None to skip embeddings).
    embedder: Option<&'a mut Embedder>,
}

impl<'a> IndexBuilder<'a> {
    /// Create a new builder from an open [`SearchIndex`], a [`Chunker`],
    /// and an optional [`Embedder`].
    ///
    /// This constructor does **not** create or open the index on disk —
    /// callers should do that first via [`SearchIndex::open_or_create`].
    pub fn new(
        index: &'a mut SearchIndex,
        chunker: &'a Chunker,
        embedder: Option<&'a mut Embedder>,
    ) -> Self {
        Self {
            index,
            chunker,
            embedder,
        }
    }

    /// Run the full build pipeline: walk -> diff -> chunk -> embed -> persist.
    ///
    /// # Steps
    /// 1. Compute staleness diff against the current manifest.
    /// 2. Handle diff result:
    ///    - `UpToDate` → return early with no changes.
    ///    - `FullRebuild` → clear old chunks/vectors, process everything.
    ///    - `Incremental` → drop chunks for removed files, re-process changed files.
    /// 3. Walk all markdown files (honoring exclusion patterns), then sort by
    ///    relative path so processing order — and therefore chunk/vector
    ///    order — is deterministic regardless of filesystem walk order.
    /// 4. Stream: for each file in that order, read/hash/chunk it, and feed
    ///    its chunks into a bounded batch buffer. Whenever the buffer fills
    ///    (or at the end, for the final partial batch), flush it: persist
    ///    that batch's chunk text, and — if an embedder is available and
    ///    loads successfully — embed that batch and stream its vectors to
    ///    disk. Peak memory is bounded to one batch's worth of text/vectors,
    ///    not the whole vault (see [`Self::flush_batch`]).
    /// 5. Update and save the manifest atomically.
    ///
    /// Returns [`BuildSummary`] with stats about the build.
    pub async fn build(
        &mut self,
        base_path: &Path,
        config: &Config,
    ) -> Result<BuildSummary, SearchError> {
        // Step 1: Compute staleness diff.
        let diff =
            crate::storage::compute_staleness_diff(base_path, config, self.index.manifest())?;

        match &diff {
            StalenessDiff::UpToDate => {
                tracing::debug!("Index is up to date.");
                return Ok(BuildSummary {
                    files_indexed: self.index.manifest().document_count(),
                    chunks_produced: self.index.manifest().chunk_count(),
                    has_embeddings: self.index.manifest().has_embeddings,
                    content_hash: self.index.manifest().content_hash.clone(),
                });
            }
            StalenessDiff::FullRebuild(reason) => {
                tracing::info!("Full rebuild required: {:?}", reason);
                self.index.clear_chunks()?;
                let vectors_path = self.index.base_dir().join("vectors.bin");
                if vectors_path.exists() {
                    fs::remove_file(&vectors_path).map_err(|e| {
                        SearchError::Storage(format!("Failed to remove vectors: {e}"))
                    })?;
                }
            }
            StalenessDiff::Incremental { removed, .. } => {
                tracing::info!("Incremental update: {} removed", removed.len());
                if !removed.is_empty() {
                    self.collect_and_remove_chunk_ids_for_removed_files(removed)?;
                }
            }
        }

        // Step 2: Walk current files, then sort by relative path AS A STRING
        // (not PathBuf's component-wise Ord, which can disagree — e.g.
        // "a-b.md" vs "a/b.md" sort differently as strings vs. path
        // components). This reproduces the same deterministic chunk order
        // the old "collect everything then sort by source_file" approach
        // gave us, without needing to buffer every chunk to sort them:
        // chunk_file has no cross-file state, so processing files in this
        // order and appending their chunks as produced yields an identical
        // final order.
        let current_files = notectl_core::file_walker::collect_markdown_files(base_path, config)
            .map_err(|e| SearchError::Storage(format!("Failed to collect markdown files: {e}")))?;

        let mut current_files: Vec<(String, std::path::PathBuf)> = current_files
            .into_iter()
            .map(|abs_path| {
                let rel_path = abs_path
                    .strip_prefix(base_path)
                    .unwrap_or(abs_path.as_path())
                    .to_string_lossy()
                    .to_string();
                (rel_path, abs_path)
            })
            .collect();
        current_files.sort_by(|a, b| a.0.cmp(&b.0));

        // Progress bar tracks files, not chunks — chunk count isn't known
        // until the vault is walked, since chunking happens incrementally
        // alongside embedding rather than all up front.
        let pb = if std::io::stdout().is_terminal() {
            let bar = ProgressBar::new(current_files.len() as u64);
            bar.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "{spinner:.green} [{elapsed}] {bar:40} {percent}% ({pos}/{len} files)",
                    )
                    .unwrap()
                    .progress_chars("##-"),
            );
            bar.set_message("Indexing");
            Some(bar)
        } else {
            None
        };

        // Steps 3-4: walk -> chunk -> embed -> persist, streamed in bounded
        // batches instead of materializing the whole vault's text/vectors.
        //
        // Sized by chunk count, not a token budget — worst case (every chunk
        // at the default 512-token max, plus title-prefix overhead) that's
        // ~18k tokens, comfortably under common embedding-endpoint context
        // windows (e.g. 32768). A larger batch size risks a hard
        // context-window-exceeded error on real content once chunk sizes
        // vary, even if a same-size synthetic benchmark happened to fit.
        const BATCH_SIZE: usize = 32;

        let mut all_entries: Vec<ChunkEntry> = Vec::new();
        let mut file_hashes: BTreeMap<String, String> = BTreeMap::new();
        let mut file_info_map: BTreeMap<String, FileInfo> = BTreeMap::new();
        let mut pending: Vec<Chunk> = Vec::with_capacity(BATCH_SIZE);
        let mut vector_writer: Option<crate::storage::VectorWriter> = None;
        let mut embeddings_unavailable = false;
        let mut total_chunks: usize = 0;

        for (rel_path, abs_path) in &current_files {
            let content = match fs::read_to_string(abs_path) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("Failed to read {}: {e}. Skipping.", rel_path);
                    continue;
                }
            };

            let content_hash = blake3_hash_str(&content);
            file_hashes.insert(rel_path.clone(), content_hash.clone());

            let mtime_secs = metadata_mtime_secs(abs_path).unwrap_or_else(|e| {
                tracing::warn!("Failed to stat {}: {e}. Using 0.", rel_path);
                0
            });

            let chunks = self.chunker.chunk_file(Path::new(rel_path), &content);
            let chunk_ids: Vec<String> = chunks.iter().map(|c| c.id.clone()).collect();
            total_chunks += chunks.len();

            for chunk in chunks {
                all_entries.push(ChunkEntry {
                    id: chunk.id.clone(),
                    source_file: chunk.source_file.clone(),
                    line_start: chunk.line_start,
                    line_end: chunk.line_end,
                    heading: chunk.heading.clone(),
                    heading_path: chunk.heading_path.clone(),
                    tags: chunk.tags.clone(),
                });
                pending.push(chunk);

                if pending.len() >= BATCH_SIZE {
                    self.flush_batch(
                        &mut pending,
                        &mut vector_writer,
                        &mut embeddings_unavailable,
                    )
                    .await?;
                }
            }

            file_info_map.insert(
                rel_path.clone(),
                FileInfo {
                    path: rel_path.clone(),
                    content_hash,
                    mtime: mtime_secs,
                    chunk_ids,
                },
            );

            if let Some(ref bar) = pb {
                bar.inc(1);
            }
        }

        // Flush the final partial batch, if any.
        self.flush_batch(
            &mut pending,
            &mut vector_writer,
            &mut embeddings_unavailable,
        )
        .await?;

        // has_embeddings mirrors the old embed_chunks contract: true only if
        // an embedder was configured, the model loaded successfully, and
        // there was at least one chunk to embed (vector_writer is created
        // lazily on the first successful batch — see flush_batch).
        let has_embeddings = if let Some(writer) = vector_writer {
            writer.finish()?;
            true
        } else {
            false
        };

        if let Some(bar) = pb {
            bar.finish_with_message("Indexing complete");
        }

        // Step 5: Build file info list (sorted by path).
        let files: Vec<FileInfo> = file_info_map.values().cloned().collect();
        let file_count = files.len();

        // Compute overall content hash.
        let overall_hash = compute_overall_content_hash(&file_hashes);

        // Update manifest.
        let manifest = self.index.manifest_mut();
        manifest.model_id = config.search.model_id.clone();
        manifest.embedding_dim = config.search.embedding_dim;
        manifest.chunk_config = ChunkConfigSnapshot {
            max_tokens: config.search.max_seq_tokens,
            overlap_tokens: config.search.chunk_overlap_tokens,
            min_chunk_size: config.search.min_chunk_tokens,
            merge_threshold: config.search.merge_threshold,
        };
        manifest.files = files;
        manifest.chunks = all_entries;
        manifest.content_hash = overall_hash.clone();
        manifest.last_indexed = Some(chrono_now_rfc3339());
        manifest.has_embeddings = has_embeddings;

        // Save manifest atomically.
        self.index.save_manifest()?;

        tracing::info!(
            "Index built: {} files, {} chunks, embeddings={}",
            file_count,
            total_chunks,
            has_embeddings
        );

        Ok(BuildSummary {
            files_indexed: file_count,
            chunks_produced: total_chunks,
            has_embeddings,
            content_hash: overall_hash,
        })
    }

    /// Collect chunk IDs from the manifest's FileInfo entries for removed files,
    /// then remove those chunk files from disk.
    fn collect_and_remove_chunk_ids_for_removed_files(
        &self,
        removed_paths: &[String],
    ) -> Result<(), SearchError> {
        let manifest = self.index.manifest();
        let mut chunk_ids_to_remove: Vec<String> = Vec::new();

        for fi in &manifest.files {
            if removed_paths.contains(&fi.path) {
                chunk_ids_to_remove.extend_from_slice(&fi.chunk_ids);
            }
        }

        if !chunk_ids_to_remove.is_empty() {
            self.index.remove_chunks(&chunk_ids_to_remove)?;
        }

        Ok(())
    }

    /// Flush one batch of chunks: persist their text, and — if embeddings
    /// are available — embed them and stream the resulting vectors to disk.
    ///
    /// Bounds peak memory to one batch's worth of text/vectors rather than
    /// the whole vault. Failure handling: if this is the *first* batch this
    /// build has attempted to embed (`vector_writer` still `None`) and the
    /// embedding call fails (endpoint unreachable, unconfigured, etc.), log
    /// a warning once and gracefully degrade the rest of the build to
    /// no-embeddings via `embeddings_unavailable` (BM25 keyword search still
    /// works). If a *later* batch fails after earlier ones already
    /// succeeded, that's a hard error that aborts the whole build — we'd
    /// rather fail loudly than silently ship a partially-embedded index.
    ///
    /// Derives document titles from `heading_path.join(" > ")` for each
    /// chunk, falling back to the filename stem if `heading_path` is empty.
    async fn flush_batch(
        &mut self,
        pending: &mut Vec<Chunk>,
        vector_writer: &mut Option<crate::storage::VectorWriter>,
        embeddings_unavailable: &mut bool,
    ) -> Result<(), SearchError> {
        if pending.is_empty() {
            return Ok(());
        }

        // Always persist chunk text for this batch, regardless of embedding
        // availability — write_chunks already writes one file per chunk, so
        // calling it per-batch instead of once for the whole vault produces
        // identical on-disk output, just with bounded peak memory.
        self.index.write_chunks(pending)?;

        if !*embeddings_unavailable && let Some(embedder) = self.embedder.as_deref_mut() {
            let texts: Vec<String> = pending.iter().map(|c| c.text.clone()).collect();
            let titles: Vec<Option<String>> = pending
                .iter()
                .map(|c| {
                    if c.heading_path.is_empty() {
                        c.source_file
                            .rsplit('/')
                            .next()
                            .and_then(|f| f.rsplit('.').next())
                            .map(String::from)
                    } else {
                        Some(c.heading_path.join(" > "))
                    }
                })
                .collect();

            match embedder
                .embed_batch_in_batches(&texts, &titles, TaskType::RetrievalDocument, texts.len())
                .await
            {
                Ok(vectors) => {
                    if vector_writer.is_none() {
                        // Derive dim from the actual output, not the
                        // configured embedding_dim: truncate() only
                        // shrinks embeddings, never pads them, so a
                        // configured dim larger than a model's native
                        // output would otherwise write a header that
                        // disagrees with the real per-vector byte width
                        // and corrupt the file.
                        let dim = vectors.first().map(|v| v.len() as u32).ok_or_else(|| {
                            SearchError::Other(
                                "embedding endpoint returned no vectors for a non-empty batch"
                                    .to_string(),
                            )
                        })?;
                        *vector_writer = Some(self.index.begin_vector_write(dim)?);
                    }
                    vector_writer.as_mut().unwrap().write_batch(&vectors)?;
                }
                Err(e) if vector_writer.is_none() => {
                    tracing::warn!(
                        "Embedding endpoint unavailable for model '{}': {e}. \
                         Indexing without dense embeddings (BM25 keyword search still works).",
                        embedder.model_id(),
                    );
                    *embeddings_unavailable = true;
                }
                Err(e) => {
                    return Err(SearchError::Other(format!("Embedding failed: {e}")));
                }
            }
        }

        pending.clear();
        Ok(())
    }
}

/// Get modification time in seconds since epoch for a file.
fn metadata_mtime_secs(path: &Path) -> std::io::Result<u64> {
    let mtime = fs::metadata(path)?.modified()?;
    Ok(mtime
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs())
}

/// Convenience: create an IndexBuilder from base_path and config.
///
/// Opens or creates the SearchIndex, builds the Chunker, and optionally
/// creates an Embedder. Callers can then invoke `.build()` directly.
pub async fn build_index(base_path: &Path, config: &Config) -> Result<BuildSummary, SearchError> {
    let chunk_config = ChunkConfigSnapshot {
        max_tokens: config.search.max_seq_tokens,
        overlap_tokens: config.search.chunk_overlap_tokens,
        min_chunk_size: config.search.min_chunk_tokens,
        merge_threshold: config.search.merge_threshold,
    };

    let mut index = SearchIndex::open_or_create(
        &config.search.resolve_index_dir(base_path),
        config.search.model_id.clone(),
        config.search.embedding_dim,
        chunk_config.clone(),
    )?;

    let chunker = Chunker::new(crate::chunker::ChunkerConfig::from_search_config(
        &config.search,
    ));

    let mut embedder = EmbeddingConfig::from_search_config(&config.search).map(Embedder::new);
    let mut builder = IndexBuilder::new(&mut index, &chunker, embedder.as_mut());
    builder.build(base_path, config).await
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use notectl_core::config::{Config, SearchConfig};
    use tempfile::TempDir;

    fn test_config() -> Config {
        Config {
            exclude_paths: Vec::new(),
            daily_note_patterns: vec!["YYYY-MM-DD.md".to_string()],
            search: SearchConfig::default(),
        }
    }

    fn test_chunker() -> Chunker {
        let sc = SearchConfig {
            max_seq_tokens: 128,
            chunk_overlap_tokens: 16,
            min_chunk_tokens: 8,
            merge_threshold: 5,
            ..Default::default()
        };
        Chunker::new(crate::chunker::ChunkerConfig::from_search_config(&sc))
    }

    /// Helper: run build_index in a test environment.
    ///
    /// Passes `embedder: None` so these tests never touch the real dense
    /// embedding backend (no network, no multi-GB model load) — they only
    /// exercise chunking/manifest/staleness behavior, none of which depends
    /// on `has_embeddings` being true.
    async fn run_build(_tmp: &TempDir, base: &Path, config: &Config) -> BuildSummary {
        let index_dir = config.search.resolve_index_dir(base);
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
            chunk_config,
        )
        .unwrap();

        let chunker = test_chunker();
        let mut builder = IndexBuilder::new(&mut index, &chunker, None);

        builder.build(base, config).await.unwrap()
    }

    #[tokio::test]
    async fn test_build_initial_index() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        fs::write(base.join("note.md"), "# Hello\n\nThis is a longer note with enough content to produce chunks from the chunker pipeline.").unwrap();

        let config = test_config();
        let summary = run_build(&tmp, &base, &config).await;

        assert_eq!(summary.files_indexed, 1);
        assert!(summary.chunks_produced >= 1);
        assert!(!summary.content_hash.is_empty());
    }

    #[tokio::test]
    async fn test_build_up_to_date() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        fs::write(base.join("note.md"), "# Hello\n\nThis is a longer note with enough content to produce chunks from the chunker pipeline.").unwrap();

        let config = test_config();

        // First build.
        let summary1 = run_build(&tmp, &base, &config).await;
        let hash1 = summary1.content_hash.clone();

        // Second build should detect no changes.
        let summary2 = run_build(&tmp, &base, &config).await;

        assert_eq!(summary2.content_hash, hash1);
        assert_eq!(summary2.files_indexed, summary1.files_indexed);
        assert_eq!(summary2.chunks_produced, summary1.chunks_produced);
    }

    #[tokio::test]
    async fn test_build_incremental_added_file() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        fs::write(base.join("note1.md"), "# Note 1\nContent here.").unwrap();

        let config = test_config();
        let summary1 = run_build(&tmp, &base, &config).await;

        // Add a new file.
        fs::write(base.join("note2.md"), "# Note 2\nMore content here.").unwrap();

        let summary2 = run_build(&tmp, &base, &config).await;

        assert_eq!(summary2.files_indexed, 2);
        assert_ne!(summary2.content_hash, summary1.content_hash);
    }

    #[tokio::test]
    async fn test_build_incremental_modified_file() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        fs::write(base.join("note.md"), "# Hello\n\nThis is a longer note with enough content to produce chunks from the chunker pipeline.").unwrap();

        let config = test_config();
        let summary1 = run_build(&tmp, &base, &config).await;

        // Modify the file.
        fs::write(base.join("note.md"), "# Hello\nWorld\nNew line added.").unwrap();

        let summary2 = run_build(&tmp, &base, &config).await;

        assert_ne!(summary2.content_hash, summary1.content_hash);
        assert_eq!(summary2.files_indexed, 1);
    }

    #[tokio::test]
    async fn test_build_full_rebuild_model_changed() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        fs::write(base.join("note.md"), "# Hello\n\nThis is a longer note with enough content to produce chunks from the chunker pipeline.").unwrap();

        let config = test_config();
        let _summary1 = run_build(&tmp, &base, &config).await;

        // Change model_id — triggers full rebuild.
        let mut modified_config = config.clone();
        modified_config.search.model_id = "different/model".to_string();

        let summary2 = run_build(&tmp, &base, &modified_config).await;

        assert_eq!(summary2.files_indexed, 1);
        assert!(summary2.chunks_produced >= 1);
    }

    #[tokio::test]
    async fn test_build_exclusion_patterns() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        fs::write(base.join("included.md"), "# Included\nContent here.").unwrap();
        fs::create_dir_all(base.join("Template")).unwrap();
        fs::write(base.join("Template/tmpl.md"), "# Template\nContent.").unwrap();

        let config = Config {
            exclude_paths: vec!["Template".to_string()],
            daily_note_patterns: vec!["YYYY-MM-DD.md".to_string()],
            search: SearchConfig::default(),
        };

        let summary = run_build(&tmp, &base, &config).await;

        assert_eq!(summary.files_indexed, 1);
    }

    #[tokio::test]
    async fn test_build_empty_vault() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        let config = test_config();
        let summary = run_build(&tmp, &base, &config).await;

        assert_eq!(summary.files_indexed, 0);
        assert_eq!(summary.chunks_produced, 0);
    }

    #[test]
    fn test_metadata_mtime_secs() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.txt");
        fs::write(&path, "hello").unwrap();

        let secs = metadata_mtime_secs(&path).unwrap();
        assert!(secs > 0);
    }

    // ---- Integration tests ported from storage.rs (were exercising dead SearchIndex::build_index) ----

    /// Verify content hash changes when file content is modified.
    #[tokio::test]
    async fn test_content_hash_changes_on_modification() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        fs::write(base.join("note.md"), "# Hello\nWorld").unwrap();

        let config = test_config();
        let summary1 = run_build(&tmp, &base, &config).await;
        let hash1 = summary1.content_hash.clone();

        // Modify the file.
        fs::write(base.join("note.md"), "# Hello\nWorld\nNew line").unwrap();

        let summary2 = run_build(&tmp, &base, &config).await;
        let hash2 = summary2.content_hash.clone();
        assert_ne!(
            hash1, hash2,
            "Content hash should change after modification"
        );
    }

    /// Touch without content change should not reindex (blake3 catches touch-only changes).
    #[tokio::test]
    async fn test_touch_without_content_change_no_reindex() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        fs::write(base.join("note.md"), "# Hello\nWorld").unwrap();

        let config = test_config();
        let summary1 = run_build(&tmp, &base, &config).await;

        // Touch the file (change mtime without changing content).
        std::process::Command::new("touch")
            .arg(base.join("note.md"))
            .output()
            .unwrap();

        // Should detect no changes because content hash didn't change.
        let summary2 = run_build(&tmp, &base, &config).await;
        assert_eq!(
            summary2.files_indexed, summary1.files_indexed,
            "files_indexed should not change after touch"
        );
        assert_eq!(
            summary2.chunks_produced, summary1.chunks_produced,
            "chunks_produced should not change after touch"
        );
    }

    /// Verify manifest persists to disk and can be re-opened.
    #[tokio::test]
    async fn test_manifest_persists_after_build() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        fs::write(base.join("note.md"), "# Hello\nWorld").unwrap();

        let config = test_config();
        let summary = run_build(&tmp, &base, &config).await;

        // Re-open the index and verify data persisted.
        let index_dir = config.search.resolve_index_dir(&base);
        let chunk_config = ChunkConfigSnapshot {
            max_tokens: config.search.max_seq_tokens,
            overlap_tokens: config.search.chunk_overlap_tokens,
            min_chunk_size: config.search.min_chunk_tokens,
            merge_threshold: config.search.merge_threshold,
        };
        let index2 = SearchIndex::open_or_create(
            &index_dir,
            config.search.model_id.clone(),
            config.search.embedding_dim,
            chunk_config,
        )
        .unwrap();

        assert_eq!(index2.manifest().document_count(), 1);
        assert_eq!(index2.manifest().chunk_count(), summary.chunks_produced);
    }

    /// Full rebuild clears old chunks and writes new ones.
    #[tokio::test]
    async fn test_full_rebuild_clears_chunks() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        fs::write(
            base.join("note.md"),
            "# Hello\n\nThis is a longer note with enough content to produce multiple chunks when chunked. It has several sentences of text that should be sufficient for the chunker to generate output.\n\n## Section One\n\nMore content here to ensure we exceed the minimum chunk size threshold and get actual chunks produced by the chunker logic.\n",
        )
        .unwrap();

        let config = test_config();
        let _summary1 = run_build(&tmp, &base, &config).await;

        // Verify chunks exist.
        let index_dir = config.search.resolve_index_dir(&base);
        let chunks_dir = index_dir.join("chunks");
        assert!(
            chunks_dir.exists(),
            "Chunks dir should exist after first build"
        );

        // Trigger a full rebuild by changing model_id.
        let mut modified_config = config.clone();
        modified_config.search.model_id = "different/model".to_string();

        let _summary2 = run_build(&tmp, &base, &modified_config).await;

        // Chunks should have been rebuilt (full rebuild clears then rebuilds).
        assert!(chunks_dir.exists(), "Chunks dir should exist after rebuild");

        // Verify manifest was updated with new model_id.
        let chunk_config = ChunkConfigSnapshot {
            max_tokens: modified_config.search.max_seq_tokens,
            overlap_tokens: modified_config.search.chunk_overlap_tokens,
            min_chunk_size: modified_config.search.min_chunk_tokens,
            merge_threshold: modified_config.search.merge_threshold,
        };
        let index = SearchIndex::open_or_create(
            &index_dir,
            modified_config.search.model_id.clone(),
            modified_config.search.embedding_dim,
            chunk_config,
        )
        .unwrap();
        assert_eq!(index.manifest().model_id, "different/model");
    }

    /// Regression test: chunk source_file must be vault-relative, not absolute.
    /// Ensures IndexBuilder::build passes rel_path (not abs_path) to the chunker,
    /// so Chunk.source_file and Chunk.id are portable across machines/mount points.
    #[tokio::test]
    async fn test_chunk_source_file_is_relative_not_absolute() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(base.join("sub")).unwrap();

        // File with enough content to produce at least one chunk.
        fs::write(
            base.join("sub").join("note.md"),
            "# Hello\n\nThis is a longer note with enough content to produce chunks from the chunker pipeline.",
        )
        .unwrap();

        let config = test_config();
        let summary = run_build(&tmp, &base, &config).await;
        assert!(summary.chunks_produced >= 1, "Expected at least one chunk");

        // Re-open the index and inspect manifest chunks.
        let index_dir = config.search.resolve_index_dir(&base);
        let chunk_config = ChunkConfigSnapshot {
            max_tokens: config.search.max_seq_tokens,
            overlap_tokens: config.search.chunk_overlap_tokens,
            min_chunk_size: config.search.min_chunk_tokens,
            merge_threshold: config.search.merge_threshold,
        };
        let index = SearchIndex::open_or_create(
            &index_dir,
            config.search.model_id.clone(),
            config.search.embedding_dim,
            chunk_config,
        )
        .unwrap();

        let manifest = index.manifest();
        assert!(!manifest.chunks.is_empty(), "Manifest should have chunks");

        let abs_prefix = tmp.path().to_string_lossy().to_string();
        for entry in &manifest.chunks {
            // Must NOT contain the temp dir's absolute path.
            assert!(
                !entry.source_file.starts_with(&abs_prefix)
                    && !entry
                        .source_file
                        .contains(std::env::temp_dir().to_string_lossy().as_ref()),
                "Chunk source_file '{}' must not contain absolute temp dir path",
                entry.source_file,
            );
            // Must equal the expected relative path.
            assert_eq!(
                entry.source_file, "sub/note.md",
                "Expected relative path 'sub/note.md', got '{}'",
                entry.source_file,
            );
        }
    }

    /// Files are processed (and their chunks/vectors positioned) in
    /// relative-path-STRING sorted order, not filesystem walk order.
    ///
    /// Regression test for the streaming rewrite of `IndexBuilder::build`,
    /// which now sorts files up front instead of buffering every chunk to
    /// sort them afterward. Uses names chosen so string order and
    /// `PathBuf`'s component-wise order would disagree ("a-note.md" vs
    /// "a/note.md") to guard against a sort-key regression (see TASK notes
    /// in the streaming refactor plan).
    #[tokio::test]
    async fn test_build_processes_files_in_sorted_relative_path_order() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(base.join("a")).unwrap();
        fs::create_dir_all(base.join("b")).unwrap();

        let long_enough = |label: &str| {
            format!(
                "# {label}\n\nThis note has enough content to produce at least one chunk from the chunker pipeline for the streaming order regression test."
            )
        };

        fs::write(base.join("a-note.md"), long_enough("a-note")).unwrap();
        fs::write(base.join("a").join("note.md"), long_enough("a/note")).unwrap();
        fs::write(base.join("b").join("note.md"), long_enough("b/note")).unwrap();
        fs::write(base.join("z.md"), long_enough("z")).unwrap();

        let config = test_config();
        let summary = run_build(&tmp, &base, &config).await;
        assert!(
            summary.chunks_produced >= 4,
            "Expected at least one chunk per file"
        );

        let index_dir = config.search.resolve_index_dir(&base);
        let chunk_config = ChunkConfigSnapshot {
            max_tokens: config.search.max_seq_tokens,
            overlap_tokens: config.search.chunk_overlap_tokens,
            min_chunk_size: config.search.min_chunk_tokens,
            merge_threshold: config.search.merge_threshold,
        };
        let index = SearchIndex::open_or_create(
            &index_dir,
            config.search.model_id.clone(),
            config.search.embedding_dim,
            chunk_config,
        )
        .unwrap();

        let manifest = index.manifest();
        let source_files: Vec<&str> = manifest
            .chunks
            .iter()
            .map(|c| c.source_file.as_str())
            .collect();

        let mut sorted = source_files.clone();
        sorted.sort();
        assert_eq!(
            source_files, sorted,
            "manifest.chunks must be in ascending source_file string order"
        );

        // Sanity check: string order must put "a-note.md" before
        // "a/note.md" ('-' = 0x2D < '/' = 0x2F). Confirms this test
        // actually exercises the String-vs-PathBuf ordering distinction
        // rather than trivially passing on already-sorted input.
        let a_note_pos = source_files.iter().position(|s| *s == "a-note.md").unwrap();
        let a_slash_note_pos = source_files.iter().position(|s| *s == "a/note.md").unwrap();
        assert!(
            a_note_pos < a_slash_note_pos,
            "'a-note.md' must sort before 'a/note.md' under string order"
        );
    }
}
