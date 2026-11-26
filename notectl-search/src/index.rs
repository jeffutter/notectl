//! Index build pipeline: walk -> diff -> chunk -> embed -> persist.
//!
//! `IndexBuilder` is the top-level orchestrator that wires together
//! [`SearchIndex`] (storage), [`Chunker`], and an optional [`Embedder`] into
//! a single build pipeline. It handles incremental updates, full rebuilds,
//! and atomic persistence of all index artifacts.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

use notectl_core::config::Config;
use notectl_core::progress::{Progress, ProgressSink};

use crate::SearchError;
use crate::chunker::{Chunk, Chunker};
use crate::storage::{
    ChunkConfigSnapshot, ChunkEntry, FileInfo, SearchIndex, SearchManifest, StalenessDiff,
    VectorWriter, blake3_hash_str, chrono_now_rfc3339, compute_overall_content_hash,
};

use crate::embeddings::{Embedder, EmbeddingConfig, TaskType, document_key};

/// Build summary returned by [`IndexBuilder::build`].
#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct BuildSummary {
    /// Number of files indexed
    pub files_indexed: usize,
    /// Number of chunks produced
    pub chunks_produced: usize,
    /// Number of chunks sent to the embedding endpoint; the rest reused
    /// stored vectors
    pub chunks_embedded: usize,
    /// Whether every chunk has an embedding
    pub has_embeddings: bool,
    /// Overall content hash
    pub content_hash: String,
}

/// Chunks per embedding request.
///
/// Sized by chunk count, not a token budget — worst case (every chunk at the
/// default 512-token max, plus title-prefix overhead) that's ~18k tokens,
/// comfortably under common embedding-endpoint context windows (e.g. 32768).
/// A larger batch size risks a hard context-window-exceeded error on real
/// content once chunk sizes vary.
const BATCH_SIZE: usize = 32;

/// A chunk waiting to be embedded.
struct PendingChunk {
    key: blake3::Hash,
    text: String,
    title: Option<String>,
}

/// Collects the vectors of one build: those carried over from the previous
/// build and those newly embedded.
#[derive(Default)]
struct VectorBuild {
    previous: HashMap<blake3::Hash, Vec<f32>>,
    /// Created on the first vector, once the dimension is known.
    writer: Option<VectorWriter>,
    pending: Vec<PendingChunk>,
    embedded: usize,
    /// Set after a failed request; no further requests are made.
    endpoint_failed: bool,
}

impl VectorBuild {
    fn has(&self, key: &blake3::Hash) -> bool {
        self.writer.as_ref().is_some_and(|w| w.contains(key))
    }

    fn write(
        &mut self,
        index: &SearchIndex,
        key: blake3::Hash,
        vector: &[f32],
    ) -> Result<(), SearchError> {
        let writer = match &mut self.writer {
            Some(writer) => writer,
            // The dimension comes from the vector itself: truncation never
            // pads, so the configured ceiling may exceed the model's output.
            None => self.writer.insert(index.begin_vector_write(vector.len())?),
        };
        writer.write(key, vector)
    }
}

/// Top-level orchestrator for building/updating the search index.
///
/// Holds references to the storage layer ([`SearchIndex`]), the text
/// splitter ([`Chunker`]), and optionally a dense embedding model
/// ([`Embedder`]). The [`IndexBuilder::build`] method implements the full
/// walk -> diff -> chunk -> embed -> persist pipeline.
///
/// Every build re-reads and re-chunks the whole vault, but only embeds chunks
/// that have no stored vector.
pub struct IndexBuilder<'a> {
    /// Storage layer — owns manifest, chunk files, vectors on disk.
    index: &'a mut SearchIndex,
    /// Text splitter configured from search config.
    chunker: &'a Chunker,
    /// Optional dense embedder (None to skip embeddings).
    embedder: Option<&'a mut Embedder>,
    /// Receives a report as the build advances.
    progress: ProgressSink,
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
            progress: notectl_core::progress::ignore(),
        }
    }

    /// Report build progress to `progress`.
    pub fn with_progress(mut self, progress: ProgressSink) -> Self {
        self.progress = progress;
        self
    }

    /// Run the full build pipeline: walk -> diff -> chunk -> embed -> persist.
    ///
    /// # Steps
    /// 1. Compare the vault against the manifest; return early if up to date.
    /// 2. Walk all markdown files (honoring exclusion patterns), then sort by
    ///    relative path so chunk order is deterministic regardless of
    ///    filesystem walk order.
    /// 3. Stream: read/hash/chunk each file and persist its chunk text. A
    ///    chunk with a stored vector under the same key reuses it; the rest
    ///    are embedded in batches of `BATCH_SIZE`.
    /// 4. Persist the vectors, then the manifest, each atomically.
    /// 5. Delete chunk files the manifest no longer refers to.
    ///
    /// If the embedding endpoint fails, the build still completes with the
    /// vectors it has (BM25 keyword search covers every chunk), and the next
    /// build tries the missing ones again.
    ///
    /// Callers must hold the index lock (see [`SearchIndex::lock`]).
    ///
    /// Returns [`BuildSummary`] with stats about the build.
    pub async fn build(
        &mut self,
        base_path: &Path,
        config: &Config,
    ) -> Result<BuildSummary, SearchError> {
        (self.progress)(Progress {
            message: "Checking for changes".to_string(),
            current: 0,
            total: None,
        });

        // Step 1: Compute staleness diff.
        let diff =
            crate::storage::compute_staleness_diff(base_path, config, self.index.manifest())?;

        match &diff {
            StalenessDiff::UpToDate => {
                tracing::debug!("Index is up to date.");
                return Ok(BuildSummary {
                    files_indexed: self.index.manifest().document_count(),
                    chunks_produced: self.index.manifest().chunk_count(),
                    chunks_embedded: 0,
                    has_embeddings: self.index.manifest().has_embeddings,
                    content_hash: self.index.manifest().content_hash.clone(),
                });
            }
            StalenessDiff::FullRebuild(reason) => {
                tracing::info!("Full rebuild required: {:?}", reason);
            }
            StalenessDiff::Incremental {
                added,
                modified,
                removed,
            } => {
                tracing::info!(
                    "Incremental update: {} added, {} modified, {} removed",
                    added.len(),
                    modified.len(),
                    removed.len()
                );
            }
            StalenessDiff::MissingEmbeddings => {
                tracing::info!("Retrying chunks that have no embedding.");
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

        // Step 3: walk -> chunk -> embed -> persist, streamed in bounded
        // batches instead of materializing the whole vault's text.
        let mut all_entries: Vec<ChunkEntry> = Vec::new();
        let mut file_hashes: BTreeMap<String, String> = BTreeMap::new();
        let mut file_info_map: BTreeMap<String, FileInfo> = BTreeMap::new();
        let mut vectors = VectorBuild {
            previous: self.index.read_vectors().unwrap_or_else(|e| {
                tracing::warn!("{e}. Embedding every chunk again.");
                HashMap::new()
            }),
            ..VectorBuild::default()
        };

        // Progress counts files, not chunks: the chunk total isn't known
        // until the whole vault has been walked.
        let files_total = current_files.len() as u64;

        for (files_done, (rel_path, abs_path)) in (0u64..).zip(&current_files) {
            self.report_files(
                "Indexing",
                files_done,
                files_total,
                all_entries.len(),
                &vectors,
            );
            // Reading and chunking never await on their own. Yield so that
            // progress listeners run, and so the build can be cancelled.
            tokio::task::yield_now().await;

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
            self.index.write_chunks(&chunks)?;

            for chunk in chunks {
                let title = document_title(&chunk);
                let key = document_key(
                    &config.search.model_id,
                    config.search.embedding_dim,
                    &chunk.text,
                    title.as_deref(),
                );
                all_entries.push(ChunkEntry {
                    id: chunk.id,
                    source_file: chunk.source_file,
                    line_start: chunk.line_start,
                    line_end: chunk.line_end,
                    heading: chunk.heading,
                    heading_path: chunk.heading_path,
                    tags: chunk.tags,
                    date: chunk.date,
                    embedding_key: key.to_hex().to_string(),
                });

                if let Some(vector) = vectors.previous.get(&key).cloned() {
                    vectors.write(self.index, key, &vector)?;
                } else if self.embedder.is_some() && !vectors.endpoint_failed && !vectors.has(&key)
                {
                    vectors.pending.push(PendingChunk {
                        key,
                        text: chunk.text,
                        title,
                    });
                }

                if vectors.pending.len() >= BATCH_SIZE {
                    // Embedding is the slow step, so say when it is running.
                    self.report_files(
                        "Embedding",
                        files_done,
                        files_total,
                        all_entries.len(),
                        &vectors,
                    );
                    self.flush_batch(&mut vectors).await?;
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
        }

        // Flush the final partial batch, if any.
        self.report_files(
            "Finishing",
            files_total,
            files_total,
            all_entries.len(),
            &vectors,
        );
        self.flush_batch(&mut vectors).await?;

        let total_chunks = all_entries.len();
        let chunks_embedded = vectors.embedded;
        let has_embeddings = total_chunks > 0
            && all_entries.iter().all(|e| {
                blake3::Hash::from_hex(&e.embedding_key).is_ok_and(|key| vectors.has(&key))
            });

        // Step 4: persist vectors, then the manifest that refers to them.
        match vectors.writer {
            Some(writer) => writer.finish()?,
            None => self.index.remove_vectors()?,
        }

        // Build file info list (sorted by path).
        let files: Vec<FileInfo> = file_info_map.values().cloned().collect();
        let file_count = files.len();

        // Compute overall content hash.
        let overall_hash = compute_overall_content_hash(&file_hashes);

        // Update manifest.
        let manifest = self.index.manifest_mut();
        manifest.model_id = config.search.model_id.clone();
        manifest.embedding_dim = config.search.embedding_dim;
        manifest.chunk_config = ChunkConfigSnapshot::from_config(config);
        manifest.files = files;
        manifest.chunks = all_entries;
        manifest.content_hash = overall_hash.clone();
        manifest.last_indexed = Some(chrono_now_rfc3339());
        manifest.has_embeddings = has_embeddings;

        // Save manifest atomically.
        self.index.save_manifest()?;

        // Step 5: only now is it safe to drop the old chunk files.
        self.index.prune_chunks()?;

        tracing::info!(
            "Index built: {} files, {} chunks ({} embedded), embeddings={}",
            file_count,
            total_chunks,
            chunks_embedded,
            has_embeddings
        );

        Ok(BuildSummary {
            files_indexed: file_count,
            chunks_produced: total_chunks,
            chunks_embedded,
            has_embeddings,
            content_hash: overall_hash,
        })
    }

    fn report_files(
        &self,
        activity: &str,
        done: u64,
        total: u64,
        chunks: usize,
        vectors: &VectorBuild,
    ) {
        (self.progress)(Progress {
            message: format!(
                "{activity} files ({chunks} chunks, {} embedded)",
                vectors.embedded
            ),
            current: done,
            total: Some(total),
        });
    }

    /// Embed the pending chunks and write their vectors.
    ///
    /// A failed request is not an error: embedding stops for the rest of this
    /// build, which keeps every vector obtained so far.
    async fn flush_batch(&mut self, vectors: &mut VectorBuild) -> Result<(), SearchError> {
        let pending = std::mem::take(&mut vectors.pending);
        let Some(embedder) = self.embedder.as_deref_mut() else {
            return Ok(());
        };
        if pending.is_empty() || vectors.endpoint_failed {
            return Ok(());
        }

        let (keys, (texts, titles)): (Vec<_>, (Vec<_>, Vec<_>)) = pending
            .into_iter()
            .map(|c| (c.key, (c.text, c.title)))
            .unzip();

        match embedder
            .embed_batch_in_batches(&texts, &titles, TaskType::RetrievalDocument, texts.len())
            .await
        {
            Ok(embeddings) if embeddings.len() == keys.len() => {
                for (key, embedding) in keys.into_iter().zip(embeddings) {
                    vectors.write(self.index, key, &embedding)?;
                    vectors.embedded += 1;
                }
            }
            Ok(embeddings) => {
                return Err(SearchError::Other(format!(
                    "embedding endpoint returned {} vectors for {} chunks",
                    embeddings.len(),
                    keys.len()
                )));
            }
            Err(e) => {
                tracing::warn!(
                    "Embedding endpoint unavailable for model '{}': {e}. \
                     Finishing without the remaining embeddings \
                     (BM25 keyword search still works).",
                    embedder.model_id(),
                );
                vectors.endpoint_failed = true;
            }
        }

        Ok(())
    }
}

/// The title embedded with a chunk: the headings leading to it, ending with
/// its own. The file's name stands in when there is no parent heading.
fn document_title(chunk: &Chunk) -> Option<String> {
    let mut parts = chunk.heading_path.clone();
    if parts.is_empty() {
        parts.extend(
            Path::new(&chunk.source_file)
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned()),
        );
    }
    parts.extend(chunk.heading.clone().filter(|h| !h.is_empty()));
    (!parts.is_empty()).then(|| parts.join(" > "))
}

/// Get modification time in seconds since epoch for a file.
fn metadata_mtime_secs(path: &Path) -> std::io::Result<u64> {
    let mtime = fs::metadata(path)?.modified()?;
    Ok(mtime
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs())
}

/// Build or update the index for the vault at `base_path`.
///
/// Locks the index, opens or creates it, then runs an [`IndexBuilder`],
/// reporting to `progress`. With `reindex`, the existing artifacts are deleted
/// first, so every chunk is embedded again.
pub async fn build_index(
    base_path: &Path,
    config: &Config,
    reindex: bool,
    progress: ProgressSink,
) -> Result<BuildSummary, SearchError> {
    let index_dir = config.search.resolve_index_dir(base_path);
    let _lock = SearchIndex::lock(&index_dir, || {
        progress(Progress {
            message: "Waiting for another index build".to_string(),
            current: 0,
            total: None,
        });
    })
    .await?;

    let mut index = SearchIndex::open_or_create(
        &index_dir,
        config.search.model_id.clone(),
        config.search.embedding_dim,
        ChunkConfigSnapshot::from_config(config),
    )?;

    if reindex {
        index.remove_manifest()?;
        index.clear_chunks()?;
        index.remove_vectors()?;
        *index.manifest_mut() = SearchManifest::new_empty(
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot::from_config(config),
        );
    }

    let chunker = Chunker::new(crate::chunker::ChunkerConfig::from_config(config));

    let mut embedder = EmbeddingConfig::from_search_config(&config.search).map(Embedder::new);
    let mut builder =
        IndexBuilder::new(&mut index, &chunker, embedder.as_mut()).with_progress(progress);
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
            exclude_headings: Vec::new(),
            search: SearchConfig::default(),
        }
    }

    fn test_chunker() -> Chunker {
        let config = Config {
            exclude_paths: Vec::new(),
            daily_note_patterns: Vec::new(),
            exclude_headings: Vec::new(),
            search: SearchConfig {
                max_seq_tokens: 128,
                chunk_overlap_tokens: 16,
                min_chunk_tokens: 8,
                merge_threshold: 5,
                ..Default::default()
            },
        };
        Chunker::new(crate::chunker::ChunkerConfig::from_config(&config))
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
            exclude_headings: Vec::new(),
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
            exclude_headings: Vec::new(),
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
            exclude_headings: Vec::new(),
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
            exclude_headings: Vec::new(),
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
            exclude_headings: Vec::new(),
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
            exclude_headings: Vec::new(),
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

    // ---- Incremental embedding ----

    use axum::{Json, Router, http::StatusCode, routing::post};
    use std::sync::{Arc, Mutex};

    /// An in-process embeddings endpoint that records what it is asked for.
    #[derive(Clone, Default)]
    struct FakeEndpoint {
        inputs: Arc<Mutex<Vec<String>>>,
        /// Requests still to be served before failing; `None` never fails.
        remaining: Arc<Mutex<Option<usize>>>,
    }

    impl FakeEndpoint {
        async fn start() -> (Self, Config) {
            let fake = Self::default();
            let handler = fake.clone();
            let app = Router::new().route(
                "/v1/embeddings",
                post(move |Json(body): Json<serde_json::Value>| {
                    let fake = handler.clone();
                    async move {
                        if let Some(remaining) = fake.remaining.lock().unwrap().as_mut() {
                            if *remaining == 0 {
                                return Err(StatusCode::SERVICE_UNAVAILABLE);
                            }
                            *remaining -= 1;
                        }
                        let inputs: Vec<String> =
                            serde_json::from_value(body["input"].clone()).unwrap();
                        let data: Vec<_> = (0..inputs.len())
                            .map(|i| serde_json::json!({"index": i, "embedding": [1.0, 0.0]}))
                            .collect();
                        fake.inputs.lock().unwrap().extend(inputs);
                        Ok(Json(serde_json::json!({ "data": data })))
                    }
                }),
            );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}/v1", listener.local_addr().unwrap());
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

            let mut config = test_config();
            config.search.embedding_api_base = Some(url);
            (fake, config)
        }

        fn set_remaining(&self, remaining: Option<usize>) {
            *self.remaining.lock().unwrap() = remaining;
        }

        fn take_inputs(&self) -> Vec<String> {
            std::mem::take(&mut self.inputs.lock().unwrap())
        }
    }

    async fn rebuild(base: &Path, config: &Config) -> BuildSummary {
        build_index(base, config, false, notectl_core::progress::ignore())
            .await
            .unwrap()
    }

    fn open_index(base: &Path, config: &Config) -> SearchIndex {
        SearchIndex::open_or_create(
            &config.search.resolve_index_dir(base),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot::from_config(config),
        )
        .unwrap()
    }

    fn diff(base: &Path, config: &Config) -> StalenessDiff {
        crate::storage::compute_staleness_diff(base, config, open_index(base, config).manifest())
            .unwrap()
    }

    /// A heading section long enough to be a chunk of its own.
    fn section(title: &str, word: &str) -> String {
        format!("## {title}\n\n{}\n\n", format!("{word} ").repeat(80))
    }

    #[tokio::test]
    async fn editing_one_section_embeds_only_that_chunk() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        let (fake, config) = FakeEndpoint::start().await;
        let write = |second: &str| {
            let note = section("One", "alpha") + &section("Two", second);
            fs::write(base.join("note.md"), note).unwrap();
        };
        write("beta");
        fs::write(base.join("other.md"), section("Other", "gamma")).unwrap();
        let first = rebuild(base, &config).await;
        assert_eq!(first.chunks_embedded, first.chunks_produced);
        fake.take_inputs();

        write("delta");
        let second = rebuild(base, &config).await;

        let inputs = fake.take_inputs();
        assert_eq!(second.chunks_embedded, 1, "{inputs:?}");
        assert!(inputs[0].contains("delta"));
        assert!(second.has_embeddings);
        let index = open_index(base, &config);
        assert_eq!(
            index.read_vectors().unwrap().len(),
            second.chunks_produced,
            "the replaced chunk's vector is dropped"
        );
    }

    #[tokio::test]
    async fn moving_a_note_reuses_its_vectors_and_prunes_old_chunk_files() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        let (_fake, config) = FakeEndpoint::start().await;
        fs::write(base.join("note.md"), section("One", "alpha")).unwrap();
        rebuild(base, &config).await;

        // A top-level chunk's title includes the file name, so keep that.
        fs::create_dir(base.join("archive")).unwrap();
        fs::rename(base.join("note.md"), base.join("archive/note.md")).unwrap();
        let summary = rebuild(base, &config).await;

        assert_eq!(summary.chunks_embedded, 0);
        assert!(summary.has_embeddings);
        let chunks_dir = config.search.resolve_index_dir(base).join("chunks");
        assert_eq!(
            fs::read_dir(chunks_dir).unwrap().count(),
            summary.chunks_produced
        );
    }

    #[tokio::test]
    async fn edited_notes_leave_no_orphaned_chunk_files() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        let config = test_config();
        fs::write(base.join("note.md"), section("One", "alpha")).unwrap();
        rebuild(base, &config).await;

        // Moving the heading down changes the chunk's ID.
        let moved = format!("intro\n\n\n{}", section("One", "alpha"));
        fs::write(base.join("note.md"), moved).unwrap();
        let summary = rebuild(base, &config).await;

        let chunks_dir = config.search.resolve_index_dir(base).join("chunks");
        assert_eq!(
            fs::read_dir(chunks_dir).unwrap().count(),
            summary.chunks_produced
        );
    }

    #[tokio::test]
    async fn missing_embeddings_are_retried_without_a_file_change() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        let (fake, config) = FakeEndpoint::start().await;
        fs::write(base.join("note.md"), section("One", "alpha")).unwrap();

        fake.set_remaining(Some(0));
        let down = rebuild(base, &config).await;
        assert!(!down.has_embeddings);
        assert!(matches!(
            diff(base, &config),
            StalenessDiff::MissingEmbeddings
        ));

        fake.set_remaining(None);
        let up = rebuild(base, &config).await;
        assert!(up.has_embeddings);
        assert_eq!(up.chunks_embedded, up.chunks_produced);
        assert!(matches!(diff(base, &config), StalenessDiff::UpToDate));
    }

    #[tokio::test]
    async fn outage_during_an_update_keeps_existing_vectors() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        let (fake, config) = FakeEndpoint::start().await;
        fs::write(base.join("a.md"), section("A", "alpha")).unwrap();
        rebuild(base, &config).await;

        fake.set_remaining(Some(0));
        fs::write(base.join("b.md"), section("B", "beta")).unwrap();
        let summary = rebuild(base, &config).await;

        assert!(!summary.has_embeddings);
        assert_eq!(open_index(base, &config).read_vectors().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn later_batch_failure_keeps_earlier_batches() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        let (fake, config) = FakeEndpoint::start().await;
        for i in 0..BATCH_SIZE + 3 {
            let note = section("Note", &format!("word{i}"));
            fs::write(base.join(format!("{i}.md")), note).unwrap();
        }

        fake.set_remaining(Some(1));
        let partial = rebuild(base, &config).await;
        assert_eq!(partial.chunks_embedded, BATCH_SIZE);
        assert!(!partial.has_embeddings);

        fake.set_remaining(None);
        let complete = rebuild(base, &config).await;
        assert_eq!(complete.chunks_embedded, 3);
        assert!(complete.has_embeddings);
    }

    #[tokio::test]
    async fn changing_the_model_or_reindexing_embeds_everything_again() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        let (_fake, mut config) = FakeEndpoint::start().await;
        fs::write(base.join("note.md"), section("One", "alpha")).unwrap();
        rebuild(base, &config).await;

        config.search.model_id = "other-model".to_string();
        let changed = rebuild(base, &config).await;
        assert_eq!(changed.chunks_embedded, changed.chunks_produced);

        let forced = build_index(base, &config, true, notectl_core::progress::ignore())
            .await
            .unwrap();
        assert_eq!(forced.chunks_embedded, forced.chunks_produced);
    }

    #[tokio::test]
    async fn changing_excluded_headings_rebuilds_the_index() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        let mut config = test_config();
        let note = section("Keep", "alpha") + &section("Drop", "beta");
        fs::write(base.join("note.md"), note).unwrap();
        let before = rebuild(base, &config).await;

        config.exclude_headings = vec!["Drop".to_string()];
        assert!(diff(base, &config).requires_full_rebuild());
        let after = rebuild(base, &config).await;
        assert!(after.chunks_produced < before.chunks_produced);
    }

    #[tokio::test]
    async fn chunk_title_includes_its_own_heading() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        let (fake, config) = FakeEndpoint::start().await;
        let nested = format!(
            "{}#{}",
            section("Parent", "alpha"),
            section("Child", "beta")
        );
        fs::write(base.join("Plan.md"), nested).unwrap();
        rebuild(base, &config).await;

        let inputs = fake.take_inputs();
        let titles: Vec<&str> = inputs.iter().filter_map(|i| i.lines().next()).collect();
        assert!(titles.contains(&"passage: Plan > Parent"), "{titles:?}");
        assert!(titles.contains(&"passage: Parent > Child"), "{titles:?}");
    }

    #[tokio::test]
    async fn chunk_outside_a_heading_is_titled_with_the_file_name() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        let (fake, config) = FakeEndpoint::start().await;
        fs::write(base.join("Shopping.md"), "plain text ".repeat(50)).unwrap();
        rebuild(base, &config).await;

        let inputs = fake.take_inputs();
        assert!(inputs[0].starts_with("passage: Shopping\n"), "{inputs:?}");
    }
}
