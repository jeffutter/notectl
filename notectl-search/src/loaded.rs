//! The in-memory form of the on-disk index that queries run against, and a
//! cache that keeps it between queries.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::SystemTime;

use notectl_core::config::Config;

use crate::SearchError;
use crate::chunker::Chunk;
use crate::sparse::SparseIndexer;
use crate::storage::{ChunkConfigSnapshot, SearchIndex, SearchManifest};

/// Dense vectors, positioned for scoring.
pub(crate) struct DenseVectors {
    /// `chunks[i]` is the manifest chunk that `vectors[i]` belongs to. Chunks
    /// without a vector are absent.
    pub chunks: Vec<usize>,
    pub vectors: Vec<Vec<f32>>,
}

/// One version of the index, read from disk.
///
/// The manifest and chunk texts are read up front. Vectors and the keyword
/// index are prepared on first use, because a query may need only one of them.
pub(crate) struct LoadedIndex {
    index: SearchIndex,
    chunk_texts: Vec<String>,
    dense: OnceLock<DenseVectors>,
    sparse: OnceLock<SparseIndexer>,
}

impl LoadedIndex {
    fn read(index_dir: &Path, config: &Config) -> Result<Self, SearchError> {
        // Creates the index directory if it is missing.
        let index = SearchIndex::open_or_create(
            index_dir,
            config.search.model_id.clone(),
            config.search.embedding_dim,
            ChunkConfigSnapshot::from_config(config),
        )?;
        // Manifest order. A missing chunk file reads as empty text.
        let chunk_texts = index
            .manifest()
            .chunks
            .iter()
            .map(|entry| index.read_chunk(&entry.id).unwrap_or_default())
            .collect();

        Ok(Self {
            index,
            chunk_texts,
            dense: OnceLock::new(),
            sparse: OnceLock::new(),
        })
    }

    pub fn manifest(&self) -> &SearchManifest {
        self.index.manifest()
    }

    /// Chunk texts, in manifest order.
    pub fn chunk_texts(&self) -> &[String] {
        &self.chunk_texts
    }

    pub fn dense(&self) -> &DenseVectors {
        self.dense.get_or_init(|| {
            let stored = self.index.read_vectors().unwrap_or_else(|e| {
                tracing::warn!("{e}. Searching without dense vectors.");
                Default::default()
            });
            let (chunks, vectors) = self
                .manifest()
                .chunks
                .iter()
                .enumerate()
                .filter_map(|(chunk_idx, entry)| {
                    let key = blake3::Hash::from_hex(&entry.embedding_key).ok()?;
                    Some((chunk_idx, stored.get(&key)?.clone()))
                })
                .unzip();
            DenseVectors { chunks, vectors }
        })
    }

    /// The BM25 index over chunk text and tags.
    pub fn sparse(&self) -> &SparseIndexer {
        self.sparse.get_or_init(|| {
            let documents: Vec<Chunk> = self
                .manifest()
                .chunks
                .iter()
                .zip(&self.chunk_texts)
                .map(|(entry, text)| {
                    // Inject tags into the indexed text so they're searchable by BM25.
                    let indexed_text = if entry.tags.is_empty() {
                        text.clone()
                    } else {
                        format!("tags: {}\n{text}", entry.tags.join(" "))
                    };
                    Chunk {
                        id: entry.id.clone(),
                        source_file: entry.source_file.clone(),
                        line_start: entry.line_start,
                        line_end: entry.line_end,
                        heading: entry.heading.clone(),
                        heading_path: entry.heading_path.clone(),
                        tags: entry.tags.clone(),
                        date: entry.date,
                        text: indexed_text,
                    }
                })
                .collect();
            SparseIndexer::index_chunks(&documents)
        })
    }
}

/// Identifies a saved manifest. Every build saves the manifest last, by
/// renaming a new file into place, so a changed stamp means a changed index,
/// whichever process built it.
#[derive(Debug, Clone, PartialEq)]
struct Stamp {
    index_dir: PathBuf,
    modified: SystemTime,
    len: u64,
    /// Tells apart saves within the filesystem's timestamp resolution.
    #[cfg(unix)]
    inode: u64,
}

impl Stamp {
    /// `None` when there is no manifest to identify.
    fn read(index_dir: &Path) -> Option<Self> {
        let metadata = fs::metadata(index_dir.join("manifest.json")).ok()?;
        Some(Self {
            index_dir: index_dir.to_path_buf(),
            modified: metadata.modified().ok()?,
            len: metadata.len(),
            #[cfg(unix)]
            inode: std::os::unix::fs::MetadataExt::ino(&metadata),
        })
    }
}

/// Keeps the most recently loaded index in memory until it changes on disk.
#[derive(Default)]
pub struct IndexCache {
    entry: Mutex<Option<(Stamp, Arc<LoadedIndex>)>>,
}

impl IndexCache {
    /// The current index in `config`'s index directory under `base_path`.
    pub(crate) fn load(
        &self,
        base_path: &Path,
        config: &Config,
    ) -> Result<Arc<LoadedIndex>, SearchError> {
        let index_dir = config.search.resolve_index_dir(base_path);
        // Held while reading, so concurrent queries load the index once.
        // A panic elsewhere cannot leave the entry half-written.
        let mut entry = self.entry.lock().unwrap_or_else(PoisonError::into_inner);

        // Taken before reading: if a build saves during the read, the next
        // call sees a different stamp and reads again.
        let stamp = Stamp::read(&index_dir);
        if let Some((cached, loaded)) = entry.as_ref()
            && Some(cached) == stamp.as_ref()
        {
            return Ok(Arc::clone(loaded));
        }

        // Release the old version first, so two are not held at once.
        *entry = None;
        let loaded = Arc::new(LoadedIndex::read(&index_dir, config)?);
        *entry = stamp.map(|stamp| (stamp, Arc::clone(&loaded)));
        Ok(loaded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn build(base: &Path, config: &Config) {
        crate::index::build_index(base, config, false, notectl_core::progress::ignore())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn index_is_reused_until_a_build_changes_it() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path();
        let config = Config::default();
        let note = |word: &str| format!("## One\n\n{}", format!("{word} ").repeat(80));
        fs::write(base.join("note.md"), note("alpha")).unwrap();
        build(base, &config).await;

        let cache = IndexCache::default();
        let first = cache.load(base, &config).unwrap();
        assert!(Arc::ptr_eq(&first, &cache.load(base, &config).unwrap()));

        // A build that finds nothing to do leaves the cached index valid.
        build(base, &config).await;
        assert!(Arc::ptr_eq(&first, &cache.load(base, &config).unwrap()));

        fs::write(base.join("note.md"), note("bravo")).unwrap();
        build(base, &config).await;
        let second = cache.load(base, &config).unwrap();
        assert!(second.chunk_texts()[0].contains("bravo"));
    }

    #[test]
    fn a_missing_index_is_not_cached() {
        let tmp = TempDir::new().unwrap();
        let config = Config::default();
        let cache = IndexCache::default();
        let first = cache.load(tmp.path(), &config).unwrap();
        assert!(!Arc::ptr_eq(
            &first,
            &cache.load(tmp.path(), &config).unwrap()
        ));
    }
}
