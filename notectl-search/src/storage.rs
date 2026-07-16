use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use blake3::Hasher as Blake3Hasher;
use notectl_core::config::Config;
use tempfile::NamedTempFile;

use crate::SearchError;
use crate::chunker::Chunk;

/// Current manifest format version. Bumped from 1 → 2 for the richer schema.
pub const INDEX_FORMAT_VERSION: u32 = 2;

/// Old v1 manifest version — handled gracefully on open (treated as empty).
#[allow(dead_code)]
const OLD_MANIFEST_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Manifest types (v2 schema)
// ---------------------------------------------------------------------------

/// Snapshot of chunking parameters captured at index time, so we can detect
/// when the user changes config and needs a full rebuild.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ChunkConfigSnapshot {
    pub max_tokens: usize,
    pub overlap_tokens: usize,
    pub min_chunk_size: usize,
    pub merge_threshold: usize,
}

/// Per-file metadata stored in the manifest.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FileInfo {
    /// Relative path from base_path (e.g. "notes/my-note.md")
    pub path: String,
    /// blake3 hex digest of file content at index time
    pub content_hash: String,
    /// Modification time in seconds since epoch
    pub mtime: u64,
    /// Chunk IDs belonging to this file
    pub chunk_ids: Vec<String>,
}

/// One chunk entry stored in the manifest (metadata only; text lives on disk).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChunkEntry {
    pub id: String,
    pub source_file: String,
    pub line_start: usize,
    pub line_end: usize,
    pub heading: Option<String>,
    pub heading_path: Vec<String>,
}

/// Richer manifest stored as `.notectl/search/manifest.json`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchManifest {
    /// Format version (bump to 2)
    pub version: u32,
    /// e.g. "google/embedding-gemma-300m"
    pub model_id: String,
    /// Embedding dimension (for matryoshka truncation)
    pub embedding_dim: u32,
    /// Chunking params captured at index time
    pub chunk_config: ChunkConfigSnapshot,
    /// Per-file metadata
    pub files: Vec<FileInfo>,
    /// Chunk list with heading_path and line spans
    pub chunks: Vec<ChunkEntry>,
    /// blake3 hex of combined per-file hashes (sorted by path)
    pub content_hash: String,
    /// RFC 3339 timestamp of last successful index
    pub last_indexed: Option<String>,
    /// Whether dense embeddings are available
    pub has_embeddings: bool,
}

impl SearchManifest {
    /// Create a fresh v2 manifest with sensible defaults.
    pub fn new_empty(
        model_id: String,
        embedding_dim: u32,
        chunk_config: ChunkConfigSnapshot,
    ) -> Self {
        Self {
            version: INDEX_FORMAT_VERSION,
            model_id,
            embedding_dim,
            chunk_config,
            files: Vec::new(),
            chunks: Vec::new(),
            content_hash: String::new(),
            last_indexed: None,
            has_embeddings: false,
        }
    }

    /// Total number of chunks (derived from chunks.len()).
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Number of documents indexed (derived from files.len()).
    pub fn document_count(&self) -> usize {
        self.files.len()
    }
}

// ---------------------------------------------------------------------------
// Staleness detection
// ---------------------------------------------------------------------------

/// Reasons that force a full rebuild.
#[derive(Debug, Clone)]
pub enum RebuildReason {
    /// Format version mismatch (old v1 manifest present).
    VersionMismatch,
    /// Model ID changed since last index.
    ModelChanged(String),
    /// Embedding dimension changed.
    DimensionChanged(u32),
    /// Chunking parameters changed.
    ChunkConfigChanged,
}

/// Result of comparing the current filesystem state against the manifest.
#[derive(Debug, Clone)]
pub enum StalenessDiff {
    /// Nothing has changed — index is up to date.
    UpToDate,
    /// Some files were added, modified, or removed; incremental update is possible.
    Incremental {
        added: Vec<PathBuf>,
        modified: Vec<PathBuf>,
        removed: Vec<String>,
    },
    /// Full rebuild required (param mismatch, version mismatch, etc.).
    FullRebuild(RebuildReason),
}

impl StalenessDiff {
    /// Convenience: is a full rebuild needed?
    pub fn requires_full_rebuild(&self) -> bool {
        matches!(self, StalenessDiff::FullRebuild(_))
    }
}

/// Compute the staleness diff between current filesystem state and the manifest.
///
/// Algorithm:
/// 1. Param check first — model_id, embedding_dim, chunk_config mismatch → FullRebuild.
/// 2. Walk files using `collect_markdown_files` (honors exclusion patterns).
/// 3. mtime pre-check — if all mtimes match manifest, return UpToDate.
/// 4. blake3 content hash for changed-mtime files — only count as modified if hash differs.
/// 5. Detect removed files: in manifest but not on disk.
/// 6. Compute overall content_hash (blake3 of sorted per-file hashes).
pub fn compute_staleness_diff(
    base_path: &Path,
    config: &Config,
    manifest: &SearchManifest,
) -> Result<StalenessDiff, SearchError> {
    // Step 1: Param check first.
    if manifest.model_id != config.search.model_id {
        return Ok(StalenessDiff::FullRebuild(RebuildReason::ModelChanged(
            manifest.model_id.clone(),
        )));
    }
    if manifest.embedding_dim != config.search.embedding_dim {
        return Ok(StalenessDiff::FullRebuild(RebuildReason::DimensionChanged(
            manifest.embedding_dim,
        )));
    }
    let current_chunk_config = ChunkConfigSnapshot {
        max_tokens: config.search.max_seq_tokens,
        overlap_tokens: config.search.chunk_overlap_tokens,
        min_chunk_size: config.search.min_chunk_tokens,
        merge_threshold: config.search.merge_threshold,
    };
    if manifest.chunk_config != current_chunk_config {
        return Ok(StalenessDiff::FullRebuild(
            RebuildReason::ChunkConfigChanged,
        ));
    }

    // Step 2: Walk files using the exclusion-aware walker from core.
    let current_files = notectl_core::file_walker::collect_markdown_files(base_path, config)
        .map_err(|e| SearchError::Storage(format!("Failed to collect markdown files: {e}")))?;

    // Build a map of relative path → absolute path for quick lookup.
    let current_rel_to_abs: BTreeMap<String, PathBuf> = current_files
        .iter()
        .map(|abs| {
            let rel = abs
                .strip_prefix(base_path)
                .unwrap_or(abs.as_path())
                .to_string_lossy()
                .to_string();
            (rel, abs.clone())
        })
        .collect();

    // Build a map of relative path → FileInfo from the manifest.
    let mut manifest_by_path: BTreeMap<String, &FileInfo> = BTreeMap::new();
    for fi in &manifest.files {
        manifest_by_path.insert(fi.path.clone(), fi);
    }

    // Step 3 + 4: Check each current file's mtime and content hash.
    let mut added = Vec::new();
    let mut modified = Vec::new();
    let mut all_hashes: BTreeMap<String, String> = BTreeMap::new();

    for (rel_path, abs_path) in &current_rel_to_abs {
        let metadata = fs::metadata(abs_path)
            .map_err(|e| SearchError::Storage(format!("Failed to stat {rel_path}: {e}")))?;
        let mtime = metadata.modified().map_err(|e| {
            SearchError::Storage(format!("Failed to get mtime for {rel_path}: {e}"))
        })?;
        let _mtime_secs = mtime
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if let Some(old_fi) = manifest_by_path.get(rel_path.as_str()) {
            // File exists in manifest — always hash to detect content changes reliably
            // (mtime alone is unreliable on fast/test filesystems).
            let content = fs::read_to_string(abs_path)
                .map_err(|e| SearchError::Storage(format!("Failed to read {rel_path}: {e}")))?;
            let hash = blake3_hash_str(&content);
            if hash != old_fi.content_hash {
                modified.push(abs_path.clone());
            }
            all_hashes.insert(rel_path.clone(), hash);
        } else {
            // New file — not in manifest.
            added.push(abs_path.clone());
            let content = fs::read_to_string(abs_path)
                .map_err(|e| SearchError::Storage(format!("Failed to read {rel_path}: {e}")))?;
            all_hashes.insert(rel_path.clone(), blake3_hash_str(&content));
        }
    }

    // Step 5: Detect removed files.
    let mut removed = Vec::new();
    for rel_path in manifest_by_path.keys() {
        if !current_rel_to_abs.contains_key(rel_path) {
            removed.push(rel_path.clone());
        }
    }

    // If nothing changed at all, return UpToDate.
    if added.is_empty() && modified.is_empty() && removed.is_empty() {
        return Ok(StalenessDiff::UpToDate);
    }

    Ok(StalenessDiff::Incremental {
        added,
        modified,
        removed,
    })
}

/// Compute the overall content hash (blake3 of all per-file hashes sorted by path).
pub fn compute_overall_content_hash(file_hashes: &BTreeMap<String, String>) -> String {
    let mut hasher = Blake3Hasher::new();
    for hash in file_hashes.values() {
        // Append each per-file hash as hex bytes.
        hasher.update(hash.as_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

/// Compute blake3 hash of a string, returning hex.
pub(crate) fn blake3_hash_str(s: &str) -> String {
    let mut hasher = Blake3Hasher::new();
    hasher.update(s.as_bytes());
    hasher.finalize().to_hex().to_string()
}

// ---------------------------------------------------------------------------
// Atomic write helper
// ---------------------------------------------------------------------------

/// Write a string to disk atomically using a temp file + rename.
pub(crate) fn atomic_write_json(path: &Path, data: &str) -> Result<(), SearchError> {
    let dir = path
        .parent()
        .ok_or_else(|| SearchError::Storage(format!("Cannot determine parent dir for {path:?}")))?;
    fs::create_dir_all(dir)
        .map_err(|e| SearchError::Storage(format!("Failed to create dir {dir:?}: {e}")))?;

    let mut tmp = NamedTempFile::new_in(dir)
        .map_err(|e| SearchError::Storage(format!("Failed to create temp file in {dir:?}: {e}")))?;
    tmp.write_all(data.as_bytes())
        .map_err(|e| SearchError::Storage(format!("Failed to write temp file: {e}")))?;
    tmp.flush()
        .map_err(|e| SearchError::Storage(format!("Failed to flush temp file: {e}")))?;
    tmp.persist(path).map_err(|e| {
        SearchError::Storage(format!("Failed to persist temp file to {path:?}: {e}"))
    })?;
    Ok(())
}

/// Write binary data atomically using a temp file + rename.
#[cfg(feature = "embeddings")]
pub(crate) fn atomic_write_binary(path: &Path, data: &[u8]) -> Result<(), SearchError> {
    let dir = path
        .parent()
        .ok_or_else(|| SearchError::Storage(format!("Cannot determine parent dir for {path:?}")))?;
    fs::create_dir_all(dir)
        .map_err(|e| SearchError::Storage(format!("Failed to create dir {dir:?}: {e}")))?;

    let mut tmp = NamedTempFile::new_in(dir)
        .map_err(|e| SearchError::Storage(format!("Failed to create temp file in {dir:?}: {e}")))?;
    tmp.write_all(data)
        .map_err(|e| SearchError::Storage(format!("Failed to write temp file: {e}")))?;
    tmp.flush()
        .map_err(|e| SearchError::Storage(format!("Failed to flush temp file: {e}")))?;
    tmp.persist(path).map_err(|e| {
        SearchError::Storage(format!("Failed to persist temp file to {path:?}: {e}"))
    })?;
    Ok(())
}

// ---------------------------------------------------------------------------
// SearchIndex
// ---------------------------------------------------------------------------

/// Reference to an embedder (type-erased to avoid circular deps in tests).
/// When the `embeddings` feature is enabled this wraps `&crate::Embedder`;
/// otherwise it is always `None`.
pub enum EmbedderRef {
    #[cfg(feature = "embeddings")]
    Dense(Box<crate::Embedder>),
    None,
}

impl EmbedderRef {
    pub fn is_some(&self) -> bool {
        match self {
            #[cfg(feature = "embeddings")]
            EmbedderRef::Dense(_) => true,
            EmbedderRef::None => false,
        }
    }
}

/// A complete search index on disk.
pub struct SearchIndex {
    /// Base directory for the index (e.g., `.notectl/search/`).
    base_dir: PathBuf,
    manifest: SearchManifest,
}

impl SearchIndex {
    /// Open an existing index at the given directory, or create a new one if it doesn't exist.
    ///
    /// On version mismatch (old v1 → new v2), treats the manifest as empty and logs a warning.
    pub fn open_or_create(
        base_dir: &Path,
        model_id: String,
        embedding_dim: u32,
        chunk_config: ChunkConfigSnapshot,
    ) -> Result<Self, SearchError> {
        fs::create_dir_all(base_dir)
            .map_err(|e| SearchError::Storage(format!("Failed to create index directory: {e}")))?;

        let manifest_path = base_dir.join("manifest.json");
        let manifest = if manifest_path.exists() {
            let data = fs::read_to_string(&manifest_path)
                .map_err(|e| SearchError::Storage(format!("Failed to read manifest: {e}")))?;

            // Try to parse as current schema first.
            match serde_json::from_str::<SearchManifest>(&data) {
                Ok(parsed) if parsed.version == INDEX_FORMAT_VERSION => parsed,
                Ok(parsed) => {
                    // Version mismatch — treat as empty manifest.
                    tracing::warn!(
                        "Manifest version {} does not match current {}. Starting fresh.",
                        parsed.version,
                        INDEX_FORMAT_VERSION
                    );
                    SearchManifest::new_empty(model_id, embedding_dim, chunk_config)
                }
                Err(e) => {
                    // Parse error (e.g. old v1 schema missing new fields) — treat as empty.
                    tracing::warn!("Failed to parse manifest ({}). Starting fresh.", e);
                    SearchManifest::new_empty(model_id, embedding_dim, chunk_config)
                }
            }
        } else {
            SearchManifest::new_empty(model_id, embedding_dim, chunk_config)
        };

        Ok(Self {
            base_dir: base_dir.to_path_buf(),
            manifest,
        })
    }

    /// Get the base directory path.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Get a reference to the manifest.
    pub fn manifest(&self) -> &SearchManifest {
        &self.manifest
    }

    /// Get a mutable reference to the manifest.
    pub fn manifest_mut(&mut self) -> &mut SearchManifest {
        &mut self.manifest
    }

    /// Update the manifest and persist it atomically.
    pub fn save_manifest(&mut self) -> Result<(), SearchError> {
        let manifest_path = self.base_dir.join("manifest.json");
        let data = serde_json::to_string_pretty(&self.manifest)
            .map_err(|e| SearchError::Storage(format!("Failed to serialize manifest: {e}")))?;
        atomic_write_json(&manifest_path, &data)
    }

    /// Write chunk texts to disk (one file per chunk in a flat directory).
    pub fn write_chunks(&self, chunks: &[Chunk]) -> Result<(), SearchError> {
        let chunks_dir = self.base_dir.join("chunks");
        fs::create_dir_all(&chunks_dir)
            .map_err(|e| SearchError::Storage(format!("Failed to create chunks directory: {e}")))?;

        for chunk in chunks {
            let safe_id = chunk.id.replace(['/', '\\', ':'], "_");
            let chunk_path = chunks_dir.join(format!("{safe_id}.txt"));
            atomic_write_json(&chunk_path, &chunk.text).map_err(|e| {
                SearchError::Storage(format!("Failed to write chunk {}: {e}", chunk.id))
            })?;
        }

        Ok(())
    }

    /// Read a chunk's text from disk by its ID.
    pub fn read_chunk(&self, chunk_id: &str) -> Result<String, SearchError> {
        let safe_id = chunk_id.replace(['/', '\\', ':'], "_");
        let chunk_path = self.base_dir.join("chunks").join(format!("{safe_id}.txt"));
        fs::read_to_string(&chunk_path)
            .map_err(|e| SearchError::Storage(format!("Failed to read chunk {}: {e}", chunk_id)))
    }

    /// Write dense embedding vectors to disk (flat f32 binary format).
    ///
    /// Wire format: `[count: u64 LE][dim: u32 LE][vec0 as dim * 4 bytes]...[vecN as dim * 4 bytes]`
    #[cfg(feature = "embeddings")]
    pub fn write_vectors(&self, vectors: &[Vec<f32>]) -> Result<(), SearchError> {
        let vectors_path = self.base_dir.join("vectors.bin");

        // Build the binary buffer in memory first.
        let dim = vectors
            .iter()
            .find_map(|v| {
                if v.is_empty() {
                    None
                } else {
                    Some(v.len() as u32)
                }
            })
            .unwrap_or(0);

        let count = vectors.len() as u64;
        let dim_usize = dim as usize;
        let mut buf: Vec<u8> = Vec::with_capacity(12 + vectors.len() * dim_usize * 4);
        buf.extend_from_slice(&count.to_le_bytes());
        buf.extend_from_slice(&dim.to_le_bytes());
        for vec in vectors {
            for v in vec {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        atomic_write_binary(&vectors_path, &buf)
    }

    /// Read dense embedding vectors from disk.
    ///
    /// Wire format: `[count: u64 LE][dim: u32 LE][vec0 as dim * 4 bytes]...[vecN as dim * 4 bytes]`
    #[cfg(feature = "embeddings")]
    pub fn read_vectors(&self) -> Result<Vec<Vec<f32>>, SearchError> {
        let vectors_path = self.base_dir.join("vectors.bin");
        if !vectors_path.exists() {
            return Ok(Vec::new());
        }

        let data = fs::read(&vectors_path)
            .map_err(|e| SearchError::Storage(format!("Failed to open vectors file: {e}")))?;

        if data.len() < 12 {
            // Not enough bytes for header.
            return Ok(Vec::new());
        }

        let count = u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]) as usize;
        let dim = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;

        if count == 0 || dim == 0 {
            return Ok(Vec::new());
        }

        let expected_len = 12 + count * dim * 4;
        if data.len() < expected_len {
            return Err(SearchError::Storage(format!(
                "Vectors file truncated: expected {expected_len} bytes, got {}",
                data.len()
            )));
        }

        let mut vectors = Vec::with_capacity(count);
        for i in 0..count {
            let offset = 12 + i * dim * 4;
            let mut vec = Vec::with_capacity(dim);
            for j in 0..dim {
                let byte_offset = offset + j * 4;
                let bytes = [
                    data[byte_offset],
                    data[byte_offset + 1],
                    data[byte_offset + 2],
                    data[byte_offset + 3],
                ];
                vec.push(f32::from_le_bytes(bytes));
            }
            vectors.push(vec);
        }

        Ok(vectors)
    }

    /// Clear all chunks from disk without removing the manifest.
    pub fn clear_chunks(&self) -> Result<(), SearchError> {
        let chunks_dir = self.base_dir.join("chunks");
        if chunks_dir.exists() {
            fs::remove_dir_all(&chunks_dir)
                .map_err(|e| SearchError::Storage(format!("Failed to clear chunks: {e}")))?;
        }
        Ok(())
    }

    /// Remove specific chunk files from disk by their IDs.
    pub fn remove_chunks(&self, chunk_ids: &[String]) -> Result<(), SearchError> {
        let chunks_dir = self.base_dir.join("chunks");
        if !chunks_dir.exists() {
            return Ok(());
        }

        for id in chunk_ids {
            let safe_id = id.replace(['/', '\\', ':'], "_");
            let chunk_path = chunks_dir.join(format!("{safe_id}.txt"));
            if chunk_path.exists() {
                fs::remove_file(&chunk_path).map_err(|e| {
                    SearchError::Storage(format!("Failed to remove chunk {id}: {e}"))
                })?;
            }
        }

        Ok(())
    }

    /// Remove the manifest file from disk.
    pub fn remove_manifest(&self) -> Result<(), SearchError> {
        let manifest_path = self.base_dir.join("manifest.json");
        if manifest_path.exists() {
            fs::remove_file(&manifest_path)
                .map_err(|e| SearchError::Storage(format!("Failed to remove manifest: {e}")))?;
        }
        Ok(())
    }

    /// Remove the vectors binary file from disk.
    pub fn remove_vectors(&self) -> Result<(), SearchError> {
        let vectors_path = self.base_dir.join("vectors.bin");
        if vectors_path.exists() {
            fs::remove_file(&vectors_path)
                .map_err(|e| SearchError::Storage(format!("Failed to remove vectors: {e}")))?;
        }
        Ok(())
    }

    /// Reset the index (delete all data files).
    pub fn reset(&self) -> Result<(), SearchError> {
        fs::remove_dir_all(&self.base_dir)
            .map_err(|e| SearchError::Storage(format!("Failed to reset index: {e}")))?;
        Ok(())
    }

    /// Build or update the search index for all markdown files in the base path.
    ///
    /// **NOTE**: For new code, prefer [`crate::index::IndexBuilder`] which provides
    /// an async API with embedding support. This synchronous method is kept for
    /// backward compatibility and testing.
    ///
    /// SearchIndex owns persistence: open_or_create, save_manifest, write_chunks,
    /// write_vectors, read_vectors, read_chunk, clear_chunks, remove_chunks,
    /// remove_manifest, remove_vectors, reset.
    pub fn build_index(
        &mut self,
        base_path: &Path,
        config: &Config,
        chunker: &crate::Chunker,
        _embedder: Option<EmbedderRef>,
    ) -> Result<StalenessDiff, SearchError> {
        // Compute staleness diff.
        let diff = compute_staleness_diff(base_path, config, &self.manifest)?;

        match diff {
            StalenessDiff::UpToDate => {
                tracing::debug!("Index is up to date.");
                return Ok(StalenessDiff::UpToDate);
            }
            StalenessDiff::FullRebuild(ref reason) => {
                tracing::info!("Full rebuild required: {:?}", reason);
                self.clear_chunks()?;
                self.remove_vectors()?;
            }
            StalenessDiff::Incremental { ref removed, .. } => {
                // Drop chunks for removed files.
                if !removed.is_empty() {
                    self.remove_chunks(removed)?;
                }
            }
        }

        // Walk current files and chunk everything.
        let current_files = notectl_core::file_walker::collect_markdown_files(base_path, config)
            .map_err(|e| SearchError::Storage(format!("Failed to collect markdown files: {e}")))?;

        let mut all_chunks: Vec<Chunk> = Vec::new();
        let mut file_hashes: BTreeMap<String, String> = BTreeMap::new();
        let mut file_info_map: BTreeMap<String, FileInfo> = BTreeMap::new();

        for abs_path in &current_files {
            let rel_path = abs_path
                .strip_prefix(base_path)
                .unwrap_or(abs_path.as_path())
                .to_string_lossy()
                .to_string();

            let content = fs::read_to_string(abs_path)
                .map_err(|e| SearchError::Storage(format!("Failed to read {rel_path}: {e}")))?;

            let content_hash = blake3_hash_str(&content);
            file_hashes.insert(rel_path.clone(), content_hash.clone());

            let metadata = fs::metadata(abs_path)
                .map_err(|e| SearchError::Storage(format!("Failed to stat {rel_path}: {e}")))?;
            let mtime = metadata.modified().map_err(|e| {
                SearchError::Storage(format!("Failed to get mtime for {rel_path}: {e}"))
            })?;
            let mtime_secs = mtime
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            let chunks = chunker.chunk_file(abs_path, &content);
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

        let files: Vec<FileInfo> = file_info_map.values().cloned().collect();
        let overall_hash = compute_overall_content_hash(&file_hashes);

        // Update manifest.
        self.manifest.model_id = config.search.model_id.clone();
        self.manifest.embedding_dim = config.search.embedding_dim;
        self.manifest.chunk_config = ChunkConfigSnapshot {
            max_tokens: config.search.max_seq_tokens,
            overlap_tokens: config.search.chunk_overlap_tokens,
            min_chunk_size: config.search.min_chunk_tokens,
            merge_threshold: config.search.merge_threshold,
        };
        self.manifest.files = files;
        self.manifest.chunks = chunk_entries;
        self.manifest.content_hash = overall_hash;
        self.manifest.last_indexed = Some(chrono_now_rfc3339());

        // Write chunks to disk.
        if !all_chunks.is_empty() {
            self.write_chunks(&all_chunks)?;
        }

        // Save manifest atomically.
        self.save_manifest()?;

        Ok(diff)
    }
}

/// Return the current time as an RFC 3339 string (no external chrono dependency needed).
pub(crate) fn chrono_now_rfc3339() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    // Simple RFC 3339 formatting without chrono dependency.
    // This gives UTC time in "YYYY-MM-DDTHH:MM:SSZ" format.
    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Calculate year, month, day from days since epoch (1970-01-01).
    let (year, month, day) = days_to_ymd(days_since_epoch);

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097; // day of the 400-year cycle
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // year of the 4-year cycle
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of the year (0-based)
    let mp = (5 * doy + 2) / 153; // month (0 = March, 11 = February)
    let d = doy - (153 * mp + 2) / 5 + 1; // day (1-based)
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // month (1-based)
    let y = if m <= 2 { y + 1 } else { y }; // adjust year for Jan/Feb

    (y, m, d)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use notectl_core::config::{Config, SearchConfig};
    use tempfile::TempDir;

    /// Helper: create a default Config for testing.
    fn test_config() -> Config {
        Config {
            exclude_paths: Vec::new(),
            daily_note_patterns: vec!["YYYY-MM-DD.md".to_string()],
            search: SearchConfig::default(),
        }
    }

    /// Helper: create a chunker with small token limits for faster tests.
    fn test_chunker() -> crate::Chunker {
        use notectl_core::config::SearchConfig;
        let sc = SearchConfig {
            max_seq_tokens: 128,
            chunk_overlap_tokens: 16,
            min_chunk_tokens: 8,
            merge_threshold: 5,
            ..Default::default()
        };
        crate::Chunker::new(crate::chunker::ChunkerConfig::from_search_config(&sc))
    }

    // ---- Manifest tests ----

    #[test]
    fn test_manifest_new_empty() {
        let config = test_config();
        let manifest = SearchManifest::new_empty(
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        );

        assert_eq!(manifest.version, INDEX_FORMAT_VERSION);
        assert_eq!(manifest.chunk_count(), 0);
        assert_eq!(manifest.document_count(), 0);
        assert!(!manifest.has_embeddings);
        assert!(manifest.content_hash.is_empty());
    }

    #[test]
    fn test_manifest_serialization_round_trip() {
        let config = test_config();
        let mut manifest = SearchManifest::new_empty(
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        );

        // Add a file and chunk entry.
        manifest.files.push(FileInfo {
            path: "test.md".to_string(),
            content_hash: "abc123".to_string(),
            mtime: 1700000000,
            chunk_ids: vec!["test.md:0:intro".to_string()],
        });
        manifest.chunks.push(ChunkEntry {
            id: "test.md:0:intro".to_string(),
            source_file: "test.md".to_string(),
            line_start: 0,
            line_end: 5,
            heading: Some("Intro".to_string()),
            heading_path: vec!["Intro".to_string()],
        });
        manifest.content_hash = "deadbeef".to_string();
        manifest.has_embeddings = true;

        // Serialize and deserialize.
        let json = serde_json::to_string_pretty(&manifest).unwrap();
        let parsed: SearchManifest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.version, manifest.version);
        assert_eq!(parsed.files.len(), 1);
        assert_eq!(parsed.files[0].path, "test.md");
        assert_eq!(parsed.chunks.len(), 1);
        assert_eq!(parsed.content_hash, "deadbeef");
        assert!(parsed.has_embeddings);
    }

    // ---- Open or create tests ----

    #[test]
    fn test_open_or_create_new() {
        let tmp = TempDir::new().unwrap();
        let config = test_config();

        let index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        assert_eq!(index.manifest().version, INDEX_FORMAT_VERSION);
        assert_eq!(index.manifest().chunk_count(), 0);
    }

    #[test]
    fn test_open_or_create_existing() {
        let tmp = TempDir::new().unwrap();
        let config = test_config();

        // Create and save an index.
        let mut index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        index.manifest.has_embeddings = true;
        index.save_manifest().unwrap();

        // Re-open and verify.
        let index2 = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        assert!(index2.manifest().has_embeddings);
    }

    #[test]
    fn test_open_or_create_version_mismatch() {
        let tmp = TempDir::new().unwrap();
        let config = test_config();

        // Write an old v1 manifest manually.
        let old_manifest = serde_json::json!({
            "version": 1,
            "chunk_count": 5,
            "document_count": 2,
            "content_hash": "oldhash",
            "last_indexed": null,
            "has_embeddings": false
        });
        let manifest_path = tmp.path().join("manifest.json");
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&old_manifest).unwrap(),
        )
        .unwrap();

        // Open with current version — should treat as empty.
        let index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        assert_eq!(index.manifest().version, INDEX_FORMAT_VERSION);
        assert_eq!(index.manifest().chunk_count(), 0);
    }

    // ---- Atomic write tests ----

    #[test]
    fn test_atomic_write_json() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.json");

        atomic_write_json(&path, r#"{"key": "value"}"#).unwrap();

        assert!(path.exists());
        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, r#"{"key": "value"}"#);
    }

    #[test]
    fn test_atomic_write_no_temp_leak() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.json");

        atomic_write_json(&path, "{}").unwrap();

        // Verify no leftover temp files in the directory.
        let entries: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].file_name().to_string_lossy(), "test.json");
    }

    // ---- Chunk read/write tests ----

    #[test]
    fn test_write_and_read_chunks() {
        let tmp = TempDir::new().unwrap();
        let config = test_config();
        let index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        let chunks = vec![
            Chunk {
                id: "test.md:0:intro".to_string(),
                source_file: "test.md".to_string(),
                line_start: 0,
                line_end: 5,
                heading: Some("Intro".to_string()),
                heading_path: vec!["Intro".to_string()],
                text: "Hello world. This is a test chunk.".to_string(),
            },
            Chunk {
                id: "test.md:5:section1".to_string(),
                source_file: "test.md".to_string(),
                line_start: 5,
                line_end: 10,
                heading: Some("Section 1".to_string()),
                heading_path: vec!["Section 1".to_string()],
                text: "More content here for testing.".to_string(),
            },
        ];

        index.write_chunks(&chunks).unwrap();

        let text = index.read_chunk("test.md:0:intro").unwrap();
        assert_eq!(text, "Hello world. This is a test chunk.");

        let text2 = index.read_chunk("test.md:5:section1").unwrap();
        assert_eq!(text2, "More content here for testing.");
    }

    #[test]
    fn test_remove_chunks() {
        let tmp = TempDir::new().unwrap();
        let config = test_config();
        let index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        let chunks = vec![Chunk {
            id: "file1.md:0:a".to_string(),
            source_file: "file1.md".to_string(),
            line_start: 0,
            line_end: 3,
            heading: None,
            heading_path: Vec::new(),
            text: "chunk one".to_string(),
        }];

        index.write_chunks(&chunks).unwrap();
        assert!(index.read_chunk("file1.md:0:a").is_ok());

        // Remove the chunk.
        index.remove_chunks(&["file1.md:0:a".to_string()]).unwrap();
        assert!(index.read_chunk("file1.md:0:a").is_err());
    }

    // ---- Staleness diff tests ----

    #[test]
    fn test_staleness_diff_up_to_date() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        // Create a markdown file.
        fs::write(base.join("note.md"), "# Hello\nWorld").unwrap();

        let config = test_config();
        let chunker = test_chunker();

        // Build initial index.
        let mut index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();

        // Second build should return UpToDate.
        let diff = index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();
        match diff {
            StalenessDiff::UpToDate => {} // expected
            other => panic!("Expected UpToDate, got: {:?}", other),
        }
    }

    #[test]
    fn test_staleness_diff_modified_file() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        fs::write(base.join("note.md"), "# Hello\nWorld").unwrap();

        let config = test_config();
        let chunker = test_chunker();

        // Build initial index.
        let mut index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();

        // Modify the file.
        fs::write(base.join("note.md"), "# Hello\nWorld\nNew line").unwrap();

        let diff = index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();
        match diff {
            StalenessDiff::Incremental {
                added,
                modified,
                removed,
            } => {
                assert!(added.is_empty());
                assert_eq!(modified.len(), 1);
                assert!(removed.is_empty());
            }
            other => panic!("Expected Incremental, got: {:?}", other),
        }
    }

    #[test]
    fn test_staleness_diff_removed_file() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        fs::write(base.join("note1.md"), "# Note 1").unwrap();
        fs::write(base.join("note2.md"), "# Note 2").unwrap();

        let config = test_config();
        let chunker = test_chunker();

        // Build initial index.
        let mut index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();

        // Remove a file.
        fs::remove_file(base.join("note2.md")).unwrap();

        let diff = index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();
        match diff {
            StalenessDiff::Incremental {
                added,
                modified,
                removed,
            } => {
                assert!(added.is_empty());
                assert!(modified.is_empty());
                assert_eq!(removed.len(), 1);
            }
            other => panic!("Expected Incremental with removed, got: {:?}", other),
        }
    }

    #[test]
    fn test_staleness_diff_added_file() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        fs::write(base.join("note1.md"), "# Note 1").unwrap();

        let config = test_config();
        let chunker = test_chunker();

        // Build initial index.
        let mut index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();

        // Add a new file.
        fs::write(base.join("note2.md"), "# Note 2").unwrap();

        let diff = index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();
        match diff {
            StalenessDiff::Incremental {
                added,
                modified,
                removed,
            } => {
                assert_eq!(added.len(), 1);
                assert!(modified.is_empty());
                assert!(removed.is_empty());
            }
            other => panic!("Expected Incremental with added, got: {:?}", other),
        }
    }

    #[test]
    fn test_staleness_diff_full_rebuild_model_changed() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        fs::write(base.join("note.md"), "# Hello\nWorld").unwrap();

        let config = test_config();
        let chunker = test_chunker();

        // Build initial index.
        let mut index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();

        // Change the model_id in config.
        let mut modified_config = config.clone();
        modified_config.search.model_id = "different/model".to_string();

        let diff = index
            .build_index(&base, &modified_config, &chunker, Some(EmbedderRef::None))
            .unwrap();
        match diff {
            StalenessDiff::FullRebuild(RebuildReason::ModelChanged(_)) => {} // expected
            other => panic!("Expected FullRebuild with ModelChanged, got: {:?}", other),
        }
    }

    #[test]
    fn test_staleness_diff_full_rebuild_dimension_changed() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        fs::write(base.join("note.md"), "# Hello\nWorld").unwrap();

        let config = test_config();
        let chunker = test_chunker();

        // Build initial index.
        let mut index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();

        // Change embedding_dim in config.
        let mut modified_config = config.clone();
        modified_config.search.embedding_dim = 512;

        let diff = index
            .build_index(&base, &modified_config, &chunker, Some(EmbedderRef::None))
            .unwrap();
        match diff {
            StalenessDiff::FullRebuild(RebuildReason::DimensionChanged(_)) => {} // expected
            other => panic!(
                "Expected FullRebuild with DimensionChanged, got: {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_staleness_diff_full_rebuild_chunk_config_changed() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        fs::write(base.join("note.md"), "# Hello\nWorld").unwrap();

        let config = test_config();
        let chunker = test_chunker();

        // Build initial index.
        let mut index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();

        // Change chunking param in config.
        let mut modified_config = config.clone();
        modified_config.search.max_seq_tokens = 1024;

        let diff = index
            .build_index(&base, &modified_config, &chunker, Some(EmbedderRef::None))
            .unwrap();
        match diff {
            StalenessDiff::FullRebuild(RebuildReason::ChunkConfigChanged) => {} // expected
            other => panic!(
                "Expected FullRebuild with ChunkConfigChanged, got: {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_staleness_diff_exclusion_filtering() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        // Create files in included and excluded paths.
        fs::write(base.join("included.md"), "# Included").unwrap();
        fs::create_dir_all(base.join("Template")).unwrap();
        fs::write(base.join("Template/tmpl.md"), "# Template").unwrap();

        let config = Config {
            exclude_paths: vec!["Template".to_string()],
            daily_note_patterns: vec!["YYYY-MM-DD.md".to_string()],
            search: SearchConfig::default(),
        };
        let chunker = test_chunker();

        // Build initial index.
        let mut index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();

        // Verify only the included file is in the manifest.
        let file_paths: Vec<&str> = index
            .manifest()
            .files
            .iter()
            .map(|f| f.path.as_str())
            .collect();
        assert!(file_paths.contains(&"included.md"));
        assert!(!file_paths.contains(&"Template/tmpl.md"));

        // Modify the excluded file — should not trigger change detection.
        fs::write(base.join("Template/tmpl.md"), "# Template Modified").unwrap();

        let diff = index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();
        match diff {
            StalenessDiff::UpToDate => {} // expected — excluded file change is ignored
            other => panic!(
                "Expected UpToDate (excluded file changed), got: {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_staleness_diff_empty_index() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        // Empty vault — build index.
        let config = test_config();
        let chunker = test_chunker();

        let mut index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();

        // Second build on empty vault should return UpToDate.
        let diff = index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();
        match diff {
            StalenessDiff::UpToDate => {} // expected
            other => panic!("Expected UpToDate for empty index, got: {:?}", other),
        }
    }

    #[test]
    fn test_content_hash_changes_on_modification() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        fs::write(base.join("note.md"), "# Hello\nWorld").unwrap();

        let config = test_config();
        let chunker = test_chunker();

        // Build initial index.
        let mut index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();

        let hash1 = index.manifest().content_hash.clone();

        // Modify the file.
        fs::write(base.join("note.md"), "# Hello\nWorld\nNew line").unwrap();

        index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();

        let hash2 = index.manifest().content_hash.clone();
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_touch_without_content_change_no_reindex() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        fs::write(base.join("note.md"), "# Hello\nWorld").unwrap();

        let config = test_config();
        let chunker = test_chunker();

        // Build initial index.
        let mut index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();

        // Touch the file (change mtime without changing content).
        // Use touch command to update mtime without changing content.
        std::process::Command::new("touch")
            .arg(base.join("note.md"))
            .output()
            .unwrap();

        // Should still be UpToDate because content hash didn't change.
        let diff = index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();
        match diff {
            StalenessDiff::UpToDate => {} // expected — blake3 catches touch-only changes
            other => panic!("Expected UpToDate after touch, got: {:?}", other),
        }
    }

    #[test]
    fn test_manifest_persists_after_build() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        fs::write(base.join("note.md"), "# Hello\nWorld").unwrap();

        let config = test_config();
        let chunker = test_chunker();

        let mut index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();

        // Verify manifest file exists on disk.
        let manifest_path = tmp.path().join("manifest.json");
        assert!(manifest_path.exists());

        // Re-open and verify data persisted.
        let index2 = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        assert_eq!(index2.manifest().document_count(), 1);
        assert_eq!(
            index2.manifest().chunk_count(),
            index.manifest().chunk_count()
        );
    }

    #[test]
    fn test_full_rebuild_clears_chunks() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        fs::write(
            base.join("note.md"),
            "# Hello\n\nThis is a longer note with enough content to produce multiple chunks when chunked. It has several sentences of text that should be sufficient for the chunker to generate output.\n\n## Section One\n\nMore content here to ensure we exceed the minimum chunk size threshold and get actual chunks produced by the chunker logic.\n",
        )
        .unwrap();

        let config = test_config();
        let chunker = test_chunker();

        // Build initial index.
        let mut index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        index
            .build_index(&base, &config, &chunker, Some(EmbedderRef::None))
            .unwrap();

        // Verify chunks exist.
        let chunks_dir = tmp.path().join("chunks");
        assert!(chunks_dir.exists());

        // Trigger a full rebuild by changing model_id.
        let mut modified_config = config.clone();
        modified_config.search.model_id = "different/model".to_string();

        index
            .build_index(&base, &modified_config, &chunker, Some(EmbedderRef::None))
            .unwrap();

        // Chunks should have been rebuilt (full rebuild clears then rebuilds).
        assert!(chunks_dir.exists());
        // Verify manifest was updated with new model_id.
        assert_eq!(index.manifest().model_id, "different/model");
    }

    #[test]
    fn test_rfc3339_formatting() {
        // Verify the date algorithm produces reasonable output.
        let (year, month, day) = days_to_ymd(0);
        assert_eq!(year, 1970);
        assert_eq!(month, 1);
        assert_eq!(day, 1);

        // 2024-01-01 is day 19723.
        let (year, month, day) = days_to_ymd(19723);
        assert_eq!(year, 2024);
        assert_eq!(month, 1);
        assert_eq!(day, 1);
    }

    #[test]
    fn test_compute_overall_content_hash_deterministic() {
        let mut hashes1: BTreeMap<String, String> = BTreeMap::new();
        hashes1.insert("a.md".to_string(), "hash_a".to_string());
        hashes1.insert("b.md".to_string(), "hash_b".to_string());

        let mut hashes2: BTreeMap<String, String> = BTreeMap::new();
        // Same hashes but different order — should produce same result since BTreeMap is sorted.
        hashes2.insert("b.md".to_string(), "hash_b".to_string());
        hashes2.insert("a.md".to_string(), "hash_a".to_string());

        assert_eq!(
            compute_overall_content_hash(&hashes1),
            compute_overall_content_hash(&hashes2)
        );
    }

    // ---- remove_manifest / remove_vectors tests ----

    #[test]
    fn test_remove_manifest_removes_existing_file() {
        let tmp = TempDir::new().unwrap();
        let config = test_config();

        // Create an index and save manifest.
        let mut index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();
        index.save_manifest().unwrap();

        let manifest_path = tmp.path().join("manifest.json");
        assert!(manifest_path.exists());

        // Remove the manifest.
        index.remove_manifest().unwrap();
        assert!(!manifest_path.exists());
    }

    #[test]
    fn test_remove_manifest_noop_when_absent() {
        let tmp = TempDir::new().unwrap();
        let config = test_config();

        let index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        // No manifest written yet — should be a no-op.
        index.remove_manifest().unwrap();
    }

    #[test]
    fn test_remove_vectors_removes_existing_file() {
        let tmp = TempDir::new().unwrap();

        // Create a vectors.bin file manually.
        let vectors_path = tmp.path().join("vectors.bin");
        fs::write(&vectors_path, [0u8; 32]).unwrap();
        assert!(vectors_path.exists());

        let index = SearchIndex {
            base_dir: tmp.path().to_path_buf(),
            manifest: SearchManifest::new_empty(
                "test".to_string(),
                384,
                ChunkConfigSnapshot {
                    max_tokens: 512,
                    overlap_tokens: 64,
                    min_chunk_size: 32,
                    merge_threshold: 5,
                },
            ),
        };

        index.remove_vectors().unwrap();
        assert!(!vectors_path.exists());
    }

    #[test]
    fn test_remove_vectors_noop_when_absent() {
        let tmp = TempDir::new().unwrap();
        let config = test_config();

        let index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        // No vectors.bin exists — should be a no-op.
        index.remove_vectors().unwrap();
    }

    // ---- Reindex cleanup preserves models/ ----

    #[test]
    fn test_reindex_cleanup_preserves_models_dir() {
        let tmp = TempDir::new().unwrap();
        let config = test_config();

        // Create an index with all artifacts present.
        let mut index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
            },
        )
        .unwrap();

        // Write a manifest.
        index.save_manifest().unwrap();

        // Write chunks.
        let chunks = vec![Chunk {
            id: "test.md:0:intro".to_string(),
            source_file: "test.md".to_string(),
            line_start: 0,
            line_end: 5,
            heading: None,
            heading_path: Vec::new(),
            text: "Hello world.".to_string(),
        }];
        index.write_chunks(&chunks).unwrap();

        // Write a vectors.bin file.
        let vectors_path = tmp.path().join("vectors.bin");
        fs::write(&vectors_path, [0u8; 32]).unwrap();

        // Create a models/ directory with a placeholder file.
        let models_dir = tmp.path().join("models");
        fs::create_dir_all(&models_dir).unwrap();
        fs::write(models_dir.join("model.bin"), b"model data").unwrap();

        // Verify everything exists before cleanup.
        assert!(tmp.path().join("manifest.json").exists());
        assert!(tmp.path().join("chunks").is_dir());
        assert!(vectors_path.exists());
        assert!(models_dir.is_dir());
        assert!(models_dir.join("model.bin").exists());

        // Run the cleanup sequence (same as --reindex path).
        index.remove_manifest().unwrap();
        index.clear_chunks().unwrap();
        index.remove_vectors().unwrap();

        // Verify manifest, chunks, and vectors are removed.
        assert!(!tmp.path().join("manifest.json").exists());
        assert!(!tmp.path().join("chunks").exists());
        assert!(!vectors_path.exists());

        // Verify models/ directory is preserved.
        assert!(models_dir.is_dir());
        assert!(models_dir.join("model.bin").exists());
    }
}
