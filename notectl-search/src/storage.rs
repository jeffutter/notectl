use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, TryLockError};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use blake3::Hasher as Blake3Hasher;
use notectl_core::config::Config;
use tempfile::NamedTempFile;

use crate::SearchError;
use crate::chunker::Chunk;

/// Current index format version. Bumped from 3 → 4 because vectors.bin became
/// a store keyed by [`ChunkEntry::embedding_key`] instead of an array whose
/// positions had to match the manifest's chunk order. Older indexes are
/// discarded and rebuilt.
pub const INDEX_FORMAT_VERSION: u32 = 4;

/// First bytes of vectors.bin.
const VECTORS_MAGIC: &[u8; 4] = b"NCVK";

/// Length of the vectors.bin header: magic, then dimension as u32 LE.
const VECTORS_HEADER_LEN: usize = 8;

/// How often a build waiting for the index lock tries again.
const LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(200);

/// Held while modifying an index; released on drop.
pub struct IndexLock {
    _file: File,
}

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
    /// Heading sections left out of the index
    pub exclude_headings: Vec<String>,
}

impl ChunkConfigSnapshot {
    /// The settings in `config` that determine how files are chunked.
    pub fn from_config(config: &Config) -> Self {
        Self {
            max_tokens: config.search.max_seq_tokens,
            overlap_tokens: config.search.chunk_overlap_tokens,
            min_chunk_size: config.search.min_chunk_tokens,
            merge_threshold: config.search.merge_threshold,
            exclude_headings: config.exclude_headings.clone(),
        }
    }
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
    /// YAML frontmatter tags from the source file
    #[serde(default)]
    pub tags: Vec<String>,
    /// Document date from frontmatter (epoch seconds), with priority:
    /// updated > created > date. None if no date found.
    #[serde(default)]
    pub date: Option<u64>,
    /// Hex key of this chunk's vector in vectors.bin. It identifies the
    /// model, dimension and exact text embedded, so a stored vector can be
    /// reused whenever the key matches.
    pub embedding_key: String,
}

/// Richer manifest stored as `.notectl/search/manifest.json`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchManifest {
    /// Format version (bump to 2)
    pub version: u32,
    /// e.g. "Qwen/Qwen3-Embedding-0.6B"
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
    /// Whether every chunk has a dense embedding
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
    /// Files are unchanged, but an embedding endpoint is configured and some
    /// chunks have no vector yet (it failed or was unset during a build).
    MissingEmbeddings,
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
    let current_chunk_config = ChunkConfigSnapshot::from_config(config);
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
        let missing_embeddings = config.search.embedding_api_base.is_some()
            && !manifest.chunks.is_empty()
            && !manifest.has_embeddings;
        return Ok(if missing_embeddings {
            StalenessDiff::MissingEmbeddings
        } else {
            StalenessDiff::UpToDate
        });
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

// ---------------------------------------------------------------------------
// SearchIndex
// ---------------------------------------------------------------------------

/// A complete search index on disk.
pub struct SearchIndex {
    /// Base directory for the index (e.g., `.notectl/search/`).
    base_dir: PathBuf,
    manifest: SearchManifest,
}

impl SearchIndex {
    /// Remove stale on-disk artifacts (chunks/ directory and vectors.bin)
    /// without touching the models/ directory or the base_dir itself.
    ///
    /// Called from [`Self::open_or_create`] when a version mismatch or parse
    /// error is detected, so that orphaned chunk files with absolute-path
    /// IDs (pre-TASK-6) and old vector data are cleaned up before the fresh
    /// index is built.
    fn clean_stale_artifacts(base_dir: &Path) {
        let chunks_dir = base_dir.join("chunks");
        if chunks_dir.exists()
            && let Err(e) = fs::remove_dir_all(&chunks_dir)
        {
            tracing::warn!("Failed to remove stale chunks directory: {e}");
        }
        let vectors_path = base_dir.join("vectors.bin");
        if vectors_path.exists()
            && let Err(e) = fs::remove_file(&vectors_path)
        {
            tracing::warn!("Failed to remove stale vectors file: {e}");
        }
    }

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
                    // Clean up stale on-disk artifacts from the old index format.
                    Self::clean_stale_artifacts(base_dir);
                    SearchManifest::new_empty(model_id, embedding_dim, chunk_config)
                }
                Err(e) => {
                    // Parse error (e.g. old v1 schema missing new fields) — treat as empty.
                    tracing::warn!("Failed to parse manifest ({}). Starting fresh.", e);
                    // Clean up stale on-disk artifacts from the corrupted index.
                    Self::clean_stale_artifacts(base_dir);
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

    /// Take the exclusive right to modify the index in `base_dir`, waiting for
    /// any other build, in this process or another, to finish. `waiting` is
    /// called each time the lock is found to be held.
    pub async fn lock(base_dir: &Path, waiting: impl Fn()) -> Result<IndexLock, SearchError> {
        let failed = |e: std::io::Error| SearchError::Storage(format!("Failed to lock index: {e}"));
        fs::create_dir_all(base_dir).map_err(failed)?;
        let file = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(base_dir.join("lock"))
            .map_err(failed)?;

        loop {
            match file.try_lock() {
                Ok(()) => return Ok(IndexLock { _file: file }),
                Err(TryLockError::WouldBlock) => waiting(),
                Err(TryLockError::Error(e)) => return Err(failed(e)),
            }
            // Polling keeps the wait cancellable and off the blocking pool.
            tokio::time::sleep(LOCK_RETRY_INTERVAL).await;
        }
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

    /// Produce a filesystem-safe filename from an arbitrary chunk ID.
    ///
    /// Chunk IDs can contain any characters (EJS templates, unicode, etc.), so we
    /// hash the full ID with blake3 and attach a short sanitized suffix for
    /// human readability. The resulting name stays well under the 255-byte limit
    /// on ext4/APFS.
    fn chunk_filename(id: &str) -> String {
        // Hash prefix: first 12 hex chars of blake3 digest (6 bytes)
        let hash = blake3::hash(id.as_bytes());
        let prefix: String = hash
            .as_bytes()
            .iter()
            .take(6)
            .map(|b| format!("{:02x}", b))
            .collect();

        // Sanitized suffix: keep only alphanumeric, dash, underscore; truncate to ~180 chars
        let suffix: String = id
            .chars()
            .filter(|c| c.is_alphanumeric() || matches!(c, '-' | '_'))
            .take(180)
            .collect();

        // If the suffix is empty (all special chars), just use the hash
        if suffix.is_empty() {
            prefix.to_string()
        } else {
            format!("{}_{}", prefix, suffix)
        }
    }

    /// Write chunk texts to disk (one file per chunk in a flat directory).
    pub fn write_chunks(&self, chunks: &[Chunk]) -> Result<(), SearchError> {
        let chunks_dir = self.base_dir.join("chunks");
        fs::create_dir_all(&chunks_dir)
            .map_err(|e| SearchError::Storage(format!("Failed to create chunks directory: {e}")))?;

        for chunk in chunks {
            let fname = Self::chunk_filename(&chunk.id);
            let chunk_path = chunks_dir.join(format!("{fname}.txt"));
            atomic_write_json(&chunk_path, &chunk.text).map_err(|e| {
                SearchError::Storage(format!("Failed to write chunk {}: {e}", chunk.id))
            })?;
        }

        Ok(())
    }

    /// Read a chunk's text from disk by its ID.
    pub fn read_chunk(&self, chunk_id: &str) -> Result<String, SearchError> {
        let fname = Self::chunk_filename(chunk_id);
        let chunk_path = self.base_dir.join("chunks").join(format!("{fname}.txt"));
        fs::read_to_string(&chunk_path)
            .map_err(|e| SearchError::Storage(format!("Failed to read chunk {}: {e}", chunk_id)))
    }

    /// Read the stored dense vectors, by embedding key.
    ///
    /// Wire format: `[magic: 4 bytes][dim: u32 LE]` followed by one record per
    /// vector, `[key: 32 bytes][dim f32 LE values]`.
    pub fn read_vectors(&self) -> Result<HashMap<blake3::Hash, Vec<f32>>, SearchError> {
        let vectors_path = self.base_dir.join("vectors.bin");
        if !vectors_path.exists() {
            return Ok(HashMap::new());
        }

        let data = fs::read(&vectors_path)
            .map_err(|e| SearchError::Storage(format!("Failed to open vectors file: {e}")))?;
        let corrupt = || SearchError::Storage("Vectors file is corrupt".to_string());

        let (header, records) = data
            .split_at_checked(VECTORS_HEADER_LEN)
            .ok_or_else(corrupt)?;
        if &header[..4] != VECTORS_MAGIC {
            return Err(corrupt());
        }
        let dim = u32::from_le_bytes(header[4..].try_into().map_err(|_| corrupt())?) as usize;
        let record_len = blake3::OUT_LEN + dim * 4;
        if dim == 0 || !records.len().is_multiple_of(record_len) {
            return Err(corrupt());
        }

        records
            .chunks_exact(record_len)
            .map(|record| {
                let (key, values) = record.split_at(blake3::OUT_LEN);
                let key = blake3::Hash::from_slice(key).map_err(|_| corrupt())?;
                let vector = values
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                Ok((key, vector))
            })
            .collect()
    }

    /// Begin a streaming write of `vectors.bin`.
    ///
    /// Callers append vectors one at a time, so peak memory stays bounded
    /// while the file is still replaced atomically: nothing is renamed into
    /// place until [`VectorWriter::finish`] is called.
    ///
    /// Every vector written must have `dim` values.
    pub fn begin_vector_write(&self, dim: usize) -> Result<VectorWriter, SearchError> {
        let dest = self.base_dir.join("vectors.bin");
        fs::create_dir_all(&self.base_dir)
            .map_err(|e| SearchError::Storage(format!("Failed to create index directory: {e}")))?;

        let mut tmp = NamedTempFile::new_in(&self.base_dir).map_err(|e| {
            SearchError::Storage(format!("Failed to create temp file for vectors: {e}"))
        })?;

        let dim_header = u32::try_from(dim)
            .map_err(|_| SearchError::Storage(format!("Vector dimension {dim} is too large")))?;
        tmp.write_all(VECTORS_MAGIC)
            .and_then(|_| tmp.write_all(&dim_header.to_le_bytes()))
            .map_err(|e| SearchError::Storage(format!("Failed to write vectors header: {e}")))?;

        Ok(VectorWriter {
            tmp: BufWriter::new(tmp),
            dest,
            dim,
            written: HashSet::new(),
        })
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

    /// Delete chunk files that no chunk in the manifest refers to, left behind
    /// by edited and removed notes.
    pub fn prune_chunks(&self) -> Result<(), SearchError> {
        let chunks_dir = self.base_dir.join("chunks");
        if !chunks_dir.exists() {
            return Ok(());
        }

        let live: HashSet<String> = self
            .manifest
            .chunks
            .iter()
            .map(|c| format!("{}.txt", Self::chunk_filename(&c.id)))
            .collect();
        let failed =
            |e: std::io::Error| SearchError::Storage(format!("Failed to prune chunks: {e}"));

        for entry in fs::read_dir(&chunks_dir).map_err(failed)? {
            let entry = entry.map_err(failed)?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.ends_with(".txt") && !live.contains(&name) {
                fs::remove_file(entry.path()).map_err(failed)?;
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
}

/// Streaming writer for `vectors.bin`, obtained via [`SearchIndex::begin_vector_write`].
///
/// A build that fails before calling [`Self::finish`] leaves the existing file
/// untouched.
pub struct VectorWriter {
    tmp: BufWriter<NamedTempFile>,
    dest: PathBuf,
    dim: usize,
    written: HashSet<blake3::Hash>,
}

impl VectorWriter {
    /// Whether a vector has already been written under `key`.
    pub fn contains(&self, key: &blake3::Hash) -> bool {
        self.written.contains(key)
    }

    /// Append a vector. A key that is already present is ignored.
    pub fn write(&mut self, key: blake3::Hash, vector: &[f32]) -> Result<(), SearchError> {
        if vector.len() != self.dim {
            // Same model name, different output: the server's model changed.
            return Err(SearchError::Storage(format!(
                "Embedding dimension changed from {} to {}; reindex with --reindex true",
                self.dim,
                vector.len()
            )));
        }
        if !self.written.insert(key) {
            return Ok(());
        }
        let values: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.tmp
            .write_all(key.as_bytes())
            .and_then(|_| self.tmp.write_all(&values))
            .map_err(|e| SearchError::Storage(format!("Failed to write vector: {e}")))
    }

    /// Atomically persist the file.
    pub fn finish(self) -> Result<(), SearchError> {
        let tmp = self
            .tmp
            .into_inner()
            .map_err(|e| SearchError::Storage(format!("Failed to flush vectors: {e}")))?;
        tmp.persist(&self.dest).map_err(|e| {
            SearchError::Storage(format!("Failed to persist vectors to {:?}: {e}", self.dest))
        })?;
        Ok(())
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
    let (year, month, day) = crate::civil_date::civil_from_days(days_since_epoch as i64);

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
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
            exclude_headings: Vec::new(),
            search: SearchConfig::default(),
        }
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
                exclude_headings: Vec::new(),
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
                exclude_headings: Vec::new(),
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
            embedding_key: String::new(),
            id: "test.md:0:intro".to_string(),
            source_file: "test.md".to_string(),
            line_start: 0,
            line_end: 5,
            heading: Some("Intro".to_string()),
            heading_path: vec!["Intro".to_string()],
            tags: vec!["test".to_string()],
            date: None,
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
                exclude_headings: Vec::new(),
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
                exclude_headings: Vec::new(),
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
                exclude_headings: Vec::new(),
            },
        )
        .unwrap();

        assert!(index2.manifest().has_embeddings);
    }

    /// Regression test: v2 manifests with absolute-path chunk IDs should be
    /// treated as stale after the TASK-6 relative-path fix (version bumped to 3).
    #[test]
    fn test_open_or_create_v2_manifest_with_absolute_paths_is_rebuilt() {
        let tmp = TempDir::new().unwrap();
        let config = test_config();

        // Write a v2-format manifest with absolute-path chunk entries
        // (the kind left behind by pre-TASK-6 binaries).
        let old_manifest = serde_json::json!({
            "version": 2,
            "model_id": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_dim": 1024,
            "chunk_config": {
                "max_tokens": 512,
                "overlap_tokens": 64,
                "min_chunk_size": 32,
                "merge_threshold": 30
            },
            "files": [
                {
                    "path": "note.md",
                    "content_hash": "abc123",
                    "mtime": 1700000000,
                    "chunk_ids": ["/home/alice/vault/note.md:0:intro"]
                }
            ],
            "chunks": [
                {
                    "id": "/home/alice/vault/note.md:0:intro",
                    "source_file": "/home/alice/vault/note.md",
                    "line_start": 0,
                    "line_end": 5,
                    "heading": "Intro",
                    "heading_path": ["Intro"]
                }
            ],
            "content_hash": "deadbeef",
            "last_indexed": null,
            "has_embeddings": false
        });
        fs::write(
            tmp.path().join("manifest.json"),
            serde_json::to_string_pretty(&old_manifest).unwrap(),
        )
        .unwrap();

        // Open with current version — should treat v2 as stale and start fresh.
        let index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
                exclude_headings: Vec::new(),
            },
        )
        .unwrap();

        assert_eq!(index.manifest().version, INDEX_FORMAT_VERSION);
        assert!(index.manifest().chunks.is_empty());
        assert!(index.manifest().files.is_empty());
    }

    /// Regression test (TASK-24): when a version-mismatch manifest is detected,
    /// orphaned chunk files with absolute-path-derived filenames and vectors.bin
    /// are cleaned up during open_or_create. The models/ directory is preserved.
    #[test]
    fn test_open_or_create_v2_manifest_cleans_up_orphaned_chunk_files() {
        let tmp = TempDir::new().unwrap();
        let config = test_config();

        // Write a v2-format manifest (stale version).
        let old_manifest = serde_json::json!({
            "version": 2,
            "model_id": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_dim": 1024,
            "chunk_config": {
                "max_tokens": 512,
                "overlap_tokens": 64,
                "min_chunk_size": 32,
                "merge_threshold": 30
            },
            "files": [],
            "chunks": [],
            "content_hash": "deadbeef",
            "last_indexed": null,
            "has_embeddings": false
        });
        fs::write(
            tmp.path().join("manifest.json"),
            serde_json::to_string_pretty(&old_manifest).unwrap(),
        )
        .unwrap();

        // Create chunks/ directory with an old absolute-path-derived chunk file.
        let chunks_dir = tmp.path().join("chunks");
        fs::create_dir_all(&chunks_dir).unwrap();
        let old_chunk_file = chunks_dir.join("_home_alice_vault_note.md_0_intro.txt");
        fs::write(&old_chunk_file, "old chunk content").unwrap();

        // Create a vectors.bin file.
        let vectors_path = tmp.path().join("vectors.bin");
        fs::write(&vectors_path, [0u8; 32]).unwrap();

        // Create a models/ directory with a placeholder (must be preserved).
        let models_dir = tmp.path().join("models");
        fs::create_dir_all(&models_dir).unwrap();
        fs::write(models_dir.join("model.bin"), b"model data").unwrap();

        // Verify preconditions.
        assert!(old_chunk_file.exists());
        assert!(vectors_path.exists());
        assert!(models_dir.is_dir());
        assert!(models_dir.join("model.bin").exists());

        // Open with current version — should clean up stale artifacts.
        let index = SearchIndex::open_or_create(
            tmp.path(),
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot {
                max_tokens: config.search.max_seq_tokens,
                overlap_tokens: config.search.chunk_overlap_tokens,
                min_chunk_size: config.search.min_chunk_tokens,
                merge_threshold: config.search.merge_threshold,
                exclude_headings: Vec::new(),
            },
        )
        .unwrap();

        // Manifest should be fresh.
        assert_eq!(index.manifest().version, INDEX_FORMAT_VERSION);

        // Old chunk file must be gone.
        assert!(
            !old_chunk_file.exists(),
            "Orphaned chunk file with absolute-path ID must be removed"
        );
        assert!(
            !chunks_dir.exists(),
            "Entire chunks/ directory must be removed"
        );

        // vectors.bin must be gone.
        assert!(!vectors_path.exists(), "Stale vectors.bin must be removed");

        // models/ must still exist.
        assert!(
            models_dir.is_dir(),
            "models/ directory must be preserved across version upgrade"
        );
        assert!(
            models_dir.join("model.bin").exists(),
            "models/model.bin must be preserved across version upgrade"
        );
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
                exclude_headings: Vec::new(),
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
                exclude_headings: Vec::new(),
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
                tags: vec![],
                date: None,
                text: "Hello world. This is a test chunk.".to_string(),
            },
            Chunk {
                id: "test.md:5:section1".to_string(),
                source_file: "test.md".to_string(),
                line_start: 5,
                line_end: 10,
                heading: Some("Section 1".to_string()),
                heading_path: vec!["Section 1".to_string()],
                tags: vec![],
                date: None,
                text: "More content here for testing.".to_string(),
            },
        ];

        index.write_chunks(&chunks).unwrap();

        let text = index.read_chunk("test.md:0:intro").unwrap();
        assert_eq!(text, "Hello world. This is a test chunk.");

        let text2 = index.read_chunk("test.md:5:section1").unwrap();
        assert_eq!(text2, "More content here for testing.");
    }

    // ---- Staleness diff tests (direct calls to compute_staleness_diff) ----

    /// Helper: create a FileInfo entry for a given path and content.
    fn make_file_info(path: &str, content: &str) -> FileInfo {
        FileInfo {
            path: path.to_string(),
            content_hash: blake3_hash_str(content),
            mtime: 1700000000,
            chunk_ids: vec![format!("{path}:0:intro")],
        }
    }

    /// Helper: create an empty manifest matching the default test config.
    fn empty_manifest() -> SearchManifest {
        SearchManifest::new_empty(
            String::new(), // matches default_model_id()
            4096,          // matches default_embedding_dim()
            ChunkConfigSnapshot {
                max_tokens: 512,
                overlap_tokens: 64,
                min_chunk_size: 32,
                merge_threshold: 30,
                exclude_headings: Vec::new(),
            },
        )
    }

    #[test]
    fn test_staleness_diff_up_to_date() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        // Create a markdown file.
        let content = "# Hello\nWorld";
        fs::write(base.join("note.md"), content).unwrap();

        let config = test_config();

        // Build manifest matching current file state.
        let mut manifest = empty_manifest();
        manifest.files.push(make_file_info("note.md", content));

        let diff = compute_staleness_diff(&base, &config, &manifest).unwrap();
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

        // Write the file with different content than the manifest.
        fs::write(base.join("note.md"), "# Hello\nWorld\nNew line").unwrap();

        let config = test_config();

        // Manifest has old content hash.
        let mut manifest = empty_manifest();
        manifest
            .files
            .push(make_file_info("note.md", "# Hello\nWorld"));

        let diff = compute_staleness_diff(&base, &config, &manifest).unwrap();
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

        // Only one file on disk.
        fs::write(base.join("note1.md"), "# Note 1").unwrap();

        let config = test_config();

        // Manifest references two files (note2.md was removed).
        let mut manifest = empty_manifest();
        manifest.files.push(make_file_info("note1.md", "# Note 1"));
        manifest.files.push(make_file_info("note2.md", "# Note 2"));

        let diff = compute_staleness_diff(&base, &config, &manifest).unwrap();
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

        // Two files on disk.
        fs::write(base.join("note1.md"), "# Note 1").unwrap();
        fs::write(base.join("note2.md"), "# Note 2").unwrap();

        let config = test_config();

        // Manifest references only note1.md (note2.md is new).
        let mut manifest = empty_manifest();
        manifest.files.push(make_file_info("note1.md", "# Note 1"));

        let diff = compute_staleness_diff(&base, &config, &manifest).unwrap();
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

        let config = test_config();

        // Manifest has different model_id than config.
        let mut manifest = empty_manifest();
        manifest.model_id = "different/model".to_string();
        manifest.files.push(make_file_info("note.md", "content"));

        let diff = compute_staleness_diff(&base, &config, &manifest).unwrap();
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

        let config = test_config();

        // Manifest has different embedding_dim than config.
        let mut manifest = empty_manifest();
        manifest.embedding_dim = 512;
        manifest.files.push(make_file_info("note.md", "content"));

        let diff = compute_staleness_diff(&base, &config, &manifest).unwrap();
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

        let config = test_config();

        // Manifest has different chunk_config than config.
        let mut manifest = empty_manifest();
        manifest.chunk_config = ChunkConfigSnapshot {
            max_tokens: 1024, // differs from default 512
            overlap_tokens: 64,
            min_chunk_size: 32,
            merge_threshold: 5,
            exclude_headings: Vec::new(),
        };
        manifest.files.push(make_file_info("note.md", "content"));

        let diff = compute_staleness_diff(&base, &config, &manifest).unwrap();
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
            exclude_headings: Vec::new(),
            search: SearchConfig::default(),
        };

        // Manifest matches only the included file (excluded path is not tracked).
        let mut manifest = empty_manifest();
        manifest
            .files
            .push(make_file_info("included.md", "# Included"));

        // Diff should be UpToDate — excluded file is not counted.
        let diff = compute_staleness_diff(&base, &config, &manifest).unwrap();
        match diff {
            StalenessDiff::UpToDate => {} // expected — excluded file is ignored
            other => panic!(
                "Expected UpToDate (excluded file should not affect diff), got: {:?}",
                other
            ),
        }

        // Now modify the excluded file — should still be UpToDate.
        fs::write(base.join("Template/tmpl.md"), "# Template Modified").unwrap();
        let diff2 = compute_staleness_diff(&base, &config, &manifest).unwrap();
        match diff2 {
            StalenessDiff::UpToDate => {} // expected
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

        // Empty vault.
        let config = test_config();
        let manifest = empty_manifest();

        let diff = compute_staleness_diff(&base, &config, &manifest).unwrap();
        match diff {
            StalenessDiff::UpToDate => {} // expected
            other => panic!("Expected UpToDate for empty index, got: {:?}", other),
        }
    }

    #[test]
    fn test_rfc3339_formatting() {
        // Verify the date algorithm produces reasonable output.
        let (year, month, day) = crate::civil_date::civil_from_days(0);
        assert_eq!(year, 1970);
        assert_eq!(month, 1);
        assert_eq!(day, 1);

        // 2024-01-01 is day 19723.
        let (year, month, day) = crate::civil_date::civil_from_days(19723);
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
                exclude_headings: Vec::new(),
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
                exclude_headings: Vec::new(),
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
                    exclude_headings: Vec::new(),
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
                exclude_headings: Vec::new(),
            },
        )
        .unwrap();

        // No vectors.bin exists — should be a no-op.
        index.remove_vectors().unwrap();
    }

    fn open(dir: &Path) -> SearchIndex {
        let config = test_config();
        SearchIndex::open_or_create(
            dir,
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot::from_config(&config),
        )
        .unwrap()
    }

    #[test]
    fn test_vectors_round_trip_by_key() {
        let tmp = TempDir::new().unwrap();
        let index = open(tmp.path());
        let (a, b) = (blake3::hash(b"a"), blake3::hash(b"b"));

        let mut writer = index.begin_vector_write(2).unwrap();
        writer.write(a, &[1.0, 2.0]).unwrap();
        writer.write(b, &[3.0, 4.0]).unwrap();
        writer.write(a, &[9.0, 9.0]).unwrap();
        writer.finish().unwrap();

        let vectors = index.read_vectors().unwrap();
        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[&a], vec![1.0, 2.0]);
        assert_eq!(vectors[&b], vec![3.0, 4.0]);
    }

    #[test]
    fn test_vector_writer_rejects_a_different_dimension() {
        let tmp = TempDir::new().unwrap();
        let mut writer = open(tmp.path()).begin_vector_write(2).unwrap();
        assert!(writer.write(blake3::hash(b"a"), &[1.0]).is_err());
    }

    #[test]
    fn test_read_vectors_rejects_a_truncated_file() {
        let tmp = TempDir::new().unwrap();
        let index = open(tmp.path());
        let mut writer = index.begin_vector_write(2).unwrap();
        writer.write(blake3::hash(b"a"), &[1.0, 2.0]).unwrap();
        writer.finish().unwrap();

        let path = tmp.path().join("vectors.bin");
        let data = fs::read(&path).unwrap();
        fs::write(&path, &data[..data.len() - 1]).unwrap();
        assert!(index.read_vectors().is_err());
    }

    #[tokio::test]
    async fn test_lock_waits_for_the_current_holder() {
        let tmp = TempDir::new().unwrap();
        let held = SearchIndex::lock(tmp.path(), || {}).await.unwrap();

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let second = SearchIndex::lock(tmp.path(), move || {
            let _ = tx.send(());
        });
        let release = async {
            rx.recv().await.unwrap();
            drop(held);
        };
        let (second, ()) = tokio::join!(second, release);
        assert!(second.is_ok());
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
                exclude_headings: Vec::new(),
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
            tags: vec![],
            date: None,
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
