use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use blake3::Hasher as Blake3Hasher;
use notectl_core::config::Config;
use tempfile::NamedTempFile;

use crate::SearchError;
use crate::chunker::Chunk;

/// Current manifest format version. Bumped from 2 → 3 because
/// IndexBuilder::build now writes vault-relative Chunk.source_file/id
/// values instead of absolute paths (TASK-6); old v2 manifests may
/// contain absolute-path chunk entries for files that haven't changed
/// since upgrade, so they must be treated as stale and rebuilt.
pub const INDEX_FORMAT_VERSION: u32 = 3;

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
            "model_id": "google/embedding-gemma-300m",
            "embedding_dim": 256,
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
            },
        )
        .unwrap();

        assert_eq!(index.manifest().version, INDEX_FORMAT_VERSION);
        assert!(index.manifest().chunks.is_empty());
        assert!(index.manifest().files.is_empty());
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
            "google/embedding-gemma-300m".to_string(),
            256, // matches default_embedding_dim()
            ChunkConfigSnapshot {
                max_tokens: 512,
                overlap_tokens: 64,
                min_chunk_size: 32,
                merge_threshold: 30,
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
