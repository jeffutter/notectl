pub mod bm25;
pub mod capability;
pub mod chunker;
pub mod fusion;
pub mod index;
pub mod search;
pub mod sparse;
pub mod storage;
pub mod tokenize;

#[cfg(feature = "embeddings")]
pub mod embeddings;

pub use capability::*;
pub use chunker::Chunker;
pub use search::{SearchMode, SearchOptions, search};
pub use storage::{SearchIndex, SearchManifest};

#[cfg(feature = "embeddings")]
pub use embeddings::{Embedder, EmbeddingConfig, embed::TaskType};

use std::fmt;
use std::path::PathBuf;

/// Errors returned by search operations
#[derive(Debug)]
pub enum SearchError {
    EmbeddingsNotEnabled,
    IndexNotFound(PathBuf),
    Storage(String),
    Chunking(String),
    Bm25(String),
    Other(String),
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchError::EmbeddingsNotEnabled => write!(
                f,
                "embeddings feature is not enabled; rebuild with --features embeddings to enable dense search"
            ),
            SearchError::IndexNotFound(path) => write!(f, "index not found at: {}", path.display()),
            SearchError::Storage(msg) => write!(f, "storage error: {msg}"),
            SearchError::Chunking(msg) => write!(f, "chunking error: {msg}"),
            SearchError::Bm25(msg) => write!(f, "BM25 error: {msg}"),
            SearchError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for SearchError {}

impl From<SearchError> for rmcp::model::ErrorData {
    fn from(err: SearchError) -> Self {
        match err {
            SearchError::EmbeddingsNotEnabled => notectl_core::invalid_params(
                "Dense search requires the 'embeddings' feature. \
                 Rebuild with: cargo build --features embeddings",
            ),
            SearchError::IndexNotFound(path) => notectl_core::invalid_params(format!(
                "Search index not found at: {}",
                path.display()
            )),
            SearchError::Storage(msg) => {
                notectl_core::internal_error(format!("Storage error: {msg}"))
            }
            SearchError::Chunking(msg) => {
                notectl_core::internal_error(format!("Chunking error: {msg}"))
            }
            SearchError::Bm25(msg) => notectl_core::internal_error(format!("BM25 error: {msg}")),
            SearchError::Other(msg) => notectl_core::internal_error(msg),
        }
    }
}

/// Result type for search operations
pub type SearchResult<T> = Result<T, SearchError>;

/// A ranked search result with relevance score
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct RankedChunk {
    /// The chunk ID
    pub id: String,
    /// Source file path
    pub source_file: String,
    /// Relevance score (higher = more relevant)
    pub score: f64,
    /// Optional heading context
    pub heading: Option<String>,
    /// Preview of the matching text
    pub preview: String,
}

/// Re-export the authoritative SearchConfig from notectl-core.
pub use notectl_core::config::SearchConfig;

/// Engine struct that holds state for search operations
pub struct SearchEngine {
    pub config: SearchConfig,
    pub base_path: PathBuf,
}

impl SearchEngine {
    pub fn new(base_path: PathBuf, config: SearchConfig) -> Self {
        Self { config, base_path }
    }

    /// Execute a search query. Returns dense results if the embeddings feature is enabled,
    /// otherwise falls back to BM25-only sparse search.
    pub async fn search(&self, query: &str) -> SearchResult<Vec<RankedChunk>> {
        use notectl_core::config::Config;

        // Build a minimal Config from our SearchConfig for the search pipeline.
        let config = Config {
            exclude_paths: Vec::new(),
            daily_note_patterns: vec!["YYYY-MM-DD.md".to_string()],
            search: self.config.clone(),
        };

        let options = search::SearchOptions::from_config(&self.config);

        search::search(&self.base_path, &config, query, options).await
    }

    /// Build or update the search index for all markdown files in the base path.
    pub async fn index(&self) -> SearchResult<()> {
        use notectl_core::config::Config;

        // Build a minimal Config from our SearchConfig for the index pipeline.
        let config = Config {
            exclude_paths: Vec::new(),
            daily_note_patterns: vec!["YYYY-MM-DD.md".to_string()],
            search: self.config.clone(),
        };

        let summary = crate::index::build_index(&self.base_path, &config).await?;

        tracing::info!(
            "Index complete: {} files, {} chunks, embeddings={}",
            summary.files_indexed,
            summary.chunks_produced,
            summary.has_embeddings
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = SearchConfig::default();
        assert_eq!(config.max_results, 50);
        assert!((config.rrf_bm25_weight - 1.0).abs() < f64::EPSILON);
        assert!((config.rrf_cosine_weight - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_resolve_absolute() {
        let config = SearchConfig {
            cache_dir: "/tmp/search".to_string(),
            ..Default::default()
        };
        let resolved = config.resolve_index_dir(std::path::Path::new("/base"));
        assert_eq!(resolved, PathBuf::from("/tmp/search"));
    }

    #[test]
    fn test_config_resolve_relative() {
        let config = SearchConfig {
            cache_dir: ".notectl/search".to_string(),
            ..Default::default()
        };
        let resolved = config.resolve_index_dir(std::path::Path::new("/base"));
        assert_eq!(resolved, PathBuf::from("/base/.notectl/search"));
    }

    #[cfg(not(feature = "embeddings"))]
    #[tokio::test]
    async fn test_search_without_embeddings_runs_sparse_only() {
        use std::fs;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();
        fs::write(base.join("hello.md"), "# Hello\n\nThis is a test document.").unwrap();

        // Build index first.
        use notectl_core::config::Config;
        crate::index::build_index(
            &base,
            &Config {
                exclude_paths: Vec::new(),
                daily_note_patterns: vec!["YYYY-MM-DD.md".to_string()],
                search: SearchConfig::default(),
            },
        )
        .await
        .unwrap();

        let cap = SearchEngine::new(base, SearchConfig::default());
        let result = cap.search("test document").await;
        // Without embeddings feature, search runs sparse-only and should succeed.
        assert!(
            result.is_ok(),
            "Search should work in sparse-only mode: {:?}",
            result
        );
    }
}
