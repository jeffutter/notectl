//! Index build pipeline: walk -> diff -> chunk -> embed -> persist.
//!
//! `IndexBuilder` is the top-level orchestrator that wires together
//! [`SearchIndex`] (storage), [`Chunker`], and an optional [`Embedder`] into
//! a single build pipeline. It handles incremental updates, full rebuilds,
//! and atomic persistence of all index artifacts.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use notectl_core::config::Config;

use crate::SearchError;
use crate::chunker::{Chunk, Chunker};
use crate::storage::{
    ChunkConfigSnapshot, ChunkEntry, FileInfo, SearchIndex, StalenessDiff, blake3_hash_str,
    chrono_now_rfc3339, compute_overall_content_hash,
};

#[cfg(feature = "embeddings")]
use crate::embeddings::{Embedder, EmbeddingConfig, embed::TaskType};

/// Build summary returned by [`IndexBuilder::build`].
#[derive(Debug)]
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
    /// Optional dense embedder (None when `embeddings` feature is disabled).
    #[allow(dead_code)]
    #[cfg(feature = "embeddings")]
    embedder: Option<&'a mut Embedder>,
}

impl<'a> IndexBuilder<'a> {
    /// Create a new builder from an open [`SearchIndex`], a [`Chunker`],
    /// and an optional [`Embedder`].
    ///
    /// This constructor does **not** create or open the index on disk —
    /// callers should do that first via [`SearchIndex::open_or_create`].
    #[cfg(feature = "embeddings")]
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

    /// Create without an embedder (no embeddings feature).
    #[cfg(not(feature = "embeddings"))]
    pub fn new(index: &'a mut SearchIndex, chunker: &'a Chunker) -> Self {
        Self { index, chunker }
    }

    /// Run the full build pipeline: walk -> diff -> chunk -> embed -> persist.
    ///
    /// # Steps
    /// 1. Compute staleness diff against the current manifest.
    /// 2. Handle diff result:
    ///    - `UpToDate` → return early with no changes.
    ///    - `FullRebuild` → clear old chunks/vectors, process everything.
    ///    - `Incremental` → drop chunks for removed files, re-process changed files.
    /// 3. Walk all markdown files (honoring exclusion patterns).
    /// 4. For each file: read content, compute blake3 hash, chunk via [`Chunker`].
    /// 5. Collect all chunks sorted by source_file path for deterministic ordering.
    /// 6. If embedder is available: derive titles from heading paths, batch-embed,
    ///    write vectors atomically.
    /// 7. Write chunk texts atomically.
    /// 8. Update and save manifest atomically.
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

        // Step 2: Walk current files (honors exclusion patterns).
        let current_files = notectl_core::file_walker::collect_markdown_files(base_path, config)
            .map_err(|e| SearchError::Storage(format!("Failed to collect markdown files: {e}")))?;

        // Step 3: Process each file — read, hash, chunk.
        let mut all_chunks: Vec<Chunk> = Vec::new();
        let mut file_hashes: BTreeMap<String, String> = BTreeMap::new();
        let mut file_info_map: BTreeMap<String, FileInfo> = BTreeMap::new();

        for abs_path in &current_files {
            let rel_path = abs_path
                .strip_prefix(base_path)
                .unwrap_or(abs_path.as_path())
                .to_string_lossy()
                .to_string();

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

            let chunks = self.chunker.chunk_file(abs_path, &content);
            let chunk_ids: Vec<String> = chunks.iter().map(|c| c.id.clone()).collect();

            all_chunks.extend(chunks);

            file_info_map.insert(
                rel_path.clone(),
                FileInfo {
                    path: rel_path,
                    content_hash,
                    mtime: mtime_secs,
                    chunk_ids,
                },
            );
        }

        // Sort chunks by source_file for deterministic ordering.
        all_chunks.sort_by(|a, b| a.source_file.cmp(&b.source_file));

        // Step 4: Build chunk entries for the manifest.
        let chunk_entries: Vec<ChunkEntry> = all_chunks
            .iter()
            .map(|c| ChunkEntry {
                id: c.id.clone(),
                source_file: c.source_file.clone(),
                line_start: c.line_start,
                line_end: c.line_end,
                heading: c.heading.clone(),
                heading_path: c.heading_path.clone(),
            })
            .collect();

        // Step 5: Build file info list (sorted by path).
        let files: Vec<FileInfo> = file_info_map.values().cloned().collect();
        let file_count = files.len();

        // Step 6: Compute overall content hash.
        let overall_hash = compute_overall_content_hash(&file_hashes);

        // Step 7: Embedding step (feature-gated).
        let has_embeddings = self.embed_chunks(&all_chunks).await?;

        // Step 8: Write chunk texts atomically.
        if !all_chunks.is_empty() {
            self.index.write_chunks(&all_chunks)?;
        }

        // Step 9: Update manifest.
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
        manifest.chunks = chunk_entries;
        manifest.content_hash = overall_hash.clone();
        manifest.last_indexed = Some(chrono_now_rfc3339());
        manifest.has_embeddings = has_embeddings;

        // Step 10: Save manifest atomically.
        self.index.save_manifest()?;

        tracing::info!(
            "Index built: {} files, {} chunks, embeddings={}",
            file_count,
            all_chunks.len(),
            has_embeddings
        );

        Ok(BuildSummary {
            files_indexed: file_count,
            chunks_produced: all_chunks.len(),
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

    /// Embed all chunks using the embedder (if available).
    ///
    /// Derives document titles from `heading_path.join(" > ")` for each chunk.
    /// Falls back to filename stem if heading_path is empty.
    ///
    /// On any incremental update where chunks change, rebuilds the entire
    /// vector array because chunk IDs may shift and the binary format is positional.
    ///
    /// Returns `true` if embeddings were computed, `false` otherwise.
    #[cfg(feature = "embeddings")]
    async fn embed_chunks(&mut self, chunks: &[Chunk]) -> Result<bool, SearchError> {
        let embedder = match &mut self.embedder {
            Some(e) => e,
            None => return Ok(false),
        };

        if chunks.is_empty() {
            return Ok(false);
        }

        // If model isn't downloaded yet, skip embeddings gracefully.
        if !embedder.is_ready() {
            tracing::info!(
                "Model not downloaded yet. Indexing without embeddings. \
                 Run index after downloading the model for dense search."
            );
            return Ok(false);
        }

        // Derive titles from heading paths.
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let titles: Vec<Option<String>> = chunks
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

        let vectors = embedder
            .embed_batch(&texts, &titles, TaskType::RetrievalDocument)
            .await
            .map_err(|e| SearchError::Other(format!("Embedding failed: {e}")))?;

        self.index.write_vectors(&vectors)?;

        Ok(true)
    }

    #[cfg(not(feature = "embeddings"))]
    async fn embed_chunks(&mut self, _chunks: &[Chunk]) -> Result<bool, SearchError> {
        Ok(false)
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
    let index_dir = config.search.resolve_index_dir(base_path);
    #[cfg(feature = "embeddings")]
    let model_cache_dir = index_dir.join("models");

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

    let chunker = Chunker::new(crate::chunker::ChunkerConfig::from_search_config(
        &config.search,
    ));

    #[cfg(feature = "embeddings")]
    {
        let mut embedder = Embedder::new(
            model_cache_dir,
            EmbeddingConfig::from_search_config(&config.search),
        );
        let mut builder = IndexBuilder::new(&mut index, &chunker, Some(&mut embedder));
        builder.build(base_path, config).await
    }

    #[cfg(not(feature = "embeddings"))]
    {
        let mut builder = IndexBuilder::new(&mut index, &chunker);
        builder.build(base_path, config).await
    }
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

    /// Helper: run build_index in a test environment (with embeddings).
    #[cfg(feature = "embeddings")]
    async fn run_build(_tmp: &TempDir, base: &Path, config: &Config) -> BuildSummary {
        let index_dir = config.search.resolve_index_dir(base);
        let model_cache_dir = index_dir.join("models");
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
        let mut embedder = Some(Embedder::new(
            model_cache_dir,
            EmbeddingConfig::from_search_config(&config.search),
        ));

        let mut builder = IndexBuilder::new(&mut index, &chunker, embedder.as_mut());

        builder.build(base, config).await.unwrap()
    }

    /// Helper: run build_index in a test environment (without embeddings).
    #[cfg(not(feature = "embeddings"))]
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
        let mut builder = IndexBuilder::new(&mut index, &chunker);

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
}
