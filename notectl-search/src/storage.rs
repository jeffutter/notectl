use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self};
use std::path::{Path, PathBuf};

use crate::SearchError;
use crate::chunker::Chunk;

/// Version of the search index format
const INDEX_FORMAT_VERSION: u32 = 1;

/// Manifest stored as .notectl/search/manifest.json
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchManifest {
    /// Format version (for future compatibility)
    pub version: u32,
    /// Total number of chunks in the index
    pub chunk_count: usize,
    /// Number of documents indexed
    pub document_count: usize,
    /// Content hash of the base path at last indexing time
    pub content_hash: String,
    /// Timestamp of last successful index
    pub last_indexed: Option<String>,
    /// Whether dense embeddings are available
    pub has_embeddings: bool,
}

/// A complete search index on disk
pub struct SearchIndex {
    /// Base directory for the index (e.g., .notectl/search/)
    base_dir: PathBuf,
    manifest: SearchManifest,
}

impl SearchIndex {
    /// Open an existing index at the given directory, or create a new one if it doesn't exist.
    pub fn open_or_create(base_dir: &Path) -> Result<Self, SearchError> {
        fs::create_dir_all(base_dir)
            .map_err(|e| SearchError::Storage(format!("Failed to create index directory: {e}")))?;

        let manifest_path = base_dir.join("manifest.json");
        let manifest = if manifest_path.exists() {
            let data = fs::read_to_string(&manifest_path)
                .map_err(|e| SearchError::Storage(format!("Failed to read manifest: {e}")))?;
            serde_json::from_str(&data)
                .map_err(|e| SearchError::Storage(format!("Failed to parse manifest: {e}")))?
        } else {
            SearchManifest {
                version: INDEX_FORMAT_VERSION,
                chunk_count: 0,
                document_count: 0,
                content_hash: String::new(),
                last_indexed: None,
                has_embeddings: false,
            }
        };

        Ok(Self {
            base_dir: base_dir.to_path_buf(),
            manifest,
        })
    }

    /// Get the base directory path
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Get a reference to the manifest
    pub fn manifest(&self) -> &SearchManifest {
        &self.manifest
    }

    /// Update the manifest and persist it
    pub fn save_manifest(&mut self) -> Result<(), SearchError> {
        let manifest_path = self.base_dir.join("manifest.json");
        let data = serde_json::to_string_pretty(&self.manifest)
            .map_err(|e| SearchError::Storage(format!("Failed to serialize manifest: {e}")))?;
        fs::write(&manifest_path, data)
            .map_err(|e| SearchError::Storage(format!("Failed to write manifest: {e}")))?;
        Ok(())
    }

    /// Write chunk texts to disk (one file per chunk in a flat directory)
    pub fn write_chunks(&self, chunks: &[Chunk]) -> Result<(), SearchError> {
        let chunks_dir = self.base_dir.join("chunks");
        fs::create_dir_all(&chunks_dir)
            .map_err(|e| SearchError::Storage(format!("Failed to create chunks directory: {e}")))?;

        for chunk in chunks {
            // Sanitize chunk ID for use as filename
            let safe_id = chunk.id.replace(['/', '\\', ':'], "_");
            let chunk_path = chunks_dir.join(format!("{safe_id}.txt"));
            fs::write(&chunk_path, &chunk.text).map_err(|e| {
                SearchError::Storage(format!("Failed to write chunk {}: {e}", chunk.id))
            })?;
        }

        Ok(())
    }

    /// Read a chunk's text from disk by its ID
    pub fn read_chunk(&self, chunk_id: &str) -> Result<String, SearchError> {
        let safe_id = chunk_id.replace(['/', '\\', ':'], "_");
        let chunk_path = self.base_dir.join("chunks").join(format!("{safe_id}.txt"));
        fs::read_to_string(&chunk_path)
            .map_err(|e| SearchError::Storage(format!("Failed to read chunk {}: {e}", chunk_id)))
    }

    /// Write dense embedding vectors to disk (flat f32 binary format)
    #[cfg(feature = "embeddings")]
    pub fn write_vectors(&self, vectors: &[Vec<f32>]) -> Result<(), SearchError> {
        let vectors_path = self.base_dir.join("vectors.bin");
        let mut file = fs::File::create(&vectors_path)
            .map_err(|e| SearchError::Storage(format!("Failed to create vectors file: {e}")))?;

        // Write count first (as u64)
        let count = vectors.len() as u64;
        std::io::Write::write_all(&mut file, &count.to_le_bytes())
            .map_err(|e| SearchError::Storage(format!("Failed to write vector count: {e}")))?;

        // Write each vector as flat f32 array
        for vec in vectors {
            let bytes: Vec<u8> = vec.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::io::Write::write_all(&mut file, &bytes)
                .map_err(|e| SearchError::Storage(format!("Failed to write vector data: {e}")))?;
        }

        Ok(())
    }

    /// Read dense embedding vectors from disk
    #[cfg(feature = "embeddings")]
    pub fn read_vectors(&self) -> Result<Vec<Vec<f32>>, SearchError> {
        let vectors_path = self.base_dir.join("vectors.bin");
        if !vectors_path.exists() {
            return Ok(Vec::new());
        }

        let mut file = fs::File::open(&vectors_path)
            .map_err(|e| SearchError::Storage(format!("Failed to open vectors file: {e}")))?;

        // Read count
        let mut count_bytes = [0u8; 8];
        std::io::Read::read_exact(&mut file, &mut count_bytes)
            .map_err(|e| SearchError::Storage(format!("Failed to read vector count: {e}")))?;
        let count = u64::from_le_bytes(count_bytes) as usize;

        // Determine embedding dimension by reading the first vector
        let mut first_vec_bytes = [0u8; 4];
        std::io::Read::read_exact(&mut file, &mut first_vec_bytes)
            .map_err(|e| SearchError::Storage(format!("Failed to read first vector: {e}")))?;
        let dim = u32::from_le_bytes(first_vec_bytes) as usize;

        // Read all vectors
        let mut vectors = Vec::with_capacity(count);
        for _ in 0..count {
            let mut buf = vec![0u8; dim * 4];
            std::io::Read::read_exact(&mut file, &mut buf)
                .map_err(|e| SearchError::Storage(format!("Failed to read vector data: {e}")))?;
            let floats: Vec<f32> = buf
                .chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            vectors.push(floats);
        }

        Ok(vectors)
    }

    /// Compute a simple content hash for the base path by hashing all file mtimes + sizes.
    /// Used for incremental reindexing - if the hash hasn't changed, no reindex needed.
    pub fn compute_content_hash(base_path: &Path) -> Result<String, SearchError> {
        let mut hasher = DefaultHasher::new();

        // Walk the directory and hash each file's path + mtime + size
        collect_file_info(base_path, &mut hasher)
            .map_err(|e| SearchError::Storage(format!("Failed to compute content hash: {e}")))?;

        let hash = hasher.finish();
        Ok(format!("{hash:x}"))
    }

    /// Check if the index needs reindexing by comparing content hashes
    pub fn needs_reindex(&self, base_path: &Path) -> Result<bool, SearchError> {
        let current_hash = Self::compute_content_hash(base_path)?;
        Ok(current_hash != self.manifest.content_hash)
    }

    /// Reset the index (delete all data files)
    pub fn reset(&self) -> Result<(), SearchError> {
        fs::remove_dir_all(&self.base_dir)
            .map_err(|e| SearchError::Storage(format!("Failed to reset index: {e}")))?;
        Ok(())
    }

    /// Clear all chunks from disk without removing the manifest
    pub fn clear_chunks(&self) -> Result<(), SearchError> {
        let chunks_dir = self.base_dir.join("chunks");
        if chunks_dir.exists() {
            fs::remove_dir_all(&chunks_dir)
                .map_err(|e| SearchError::Storage(format!("Failed to clear chunks: {e}")))?;
        }
        Ok(())
    }
}

/// Recursively collect file info for hashing
fn collect_file_info(path: &Path, hasher: &mut impl Hasher) -> io::Result<()> {
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            collect_file_info(&entry.path(), hasher)?;
        }
    } else if path.extension().is_some_and(|ext| ext == "md") {
        let metadata = fs::metadata(path)?;
        path.hash(hasher);
        metadata.modified()?.hash(hasher);
        metadata.len().hash(hasher);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_and_save_manifest() {
        let tmp = TempDir::new().unwrap();
        let index = SearchIndex::open_or_create(tmp.path()).unwrap();

        assert_eq!(index.manifest().version, INDEX_FORMAT_VERSION);
        assert_eq!(index.manifest().chunk_count, 0);

        // Modify and save
        let mut index = index;
        index.manifest.chunk_count = 10;
        index.manifest.document_count = 3;
        index.save_manifest().unwrap();

        // Re-open and verify
        let index2 = SearchIndex::open_or_create(tmp.path()).unwrap();
        assert_eq!(index2.manifest().chunk_count, 10);
        assert_eq!(index2.manifest().document_count, 3);
    }

    #[test]
    fn test_write_and_read_chunks() {
        let tmp = TempDir::new().unwrap();
        let index = SearchIndex::open_or_create(tmp.path()).unwrap();

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
    }

    #[test]
    fn test_content_hash_changes_on_file_modification() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();

        // Create a markdown file
        fs::write(base.join("note.md"), "# Hello\nWorld").unwrap();

        let hash1 = SearchIndex::compute_content_hash(&base).unwrap();

        // Modify the file
        fs::write(base.join("note.md"), "# Hello\nWorld\nNew line").unwrap();

        let hash2 = SearchIndex::compute_content_hash(&base).unwrap();
        assert_ne!(hash1, hash2);
    }
}
