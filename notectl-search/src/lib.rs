pub mod bm25;
pub mod capability;
pub mod chunker;
pub mod embeddings;
pub mod fusion;
pub mod index;
pub mod search;
pub mod sparse;
pub mod storage;
pub mod tokenize;

pub use capability::*;
pub use chunker::Chunker;
pub use embeddings::{Embedder, EmbeddingConfig, TaskType};
pub use search::{SearchMode, SearchOptions, SearchOutcome, search};
pub use storage::{SearchIndex, SearchManifest};

use std::fmt;
use std::path::PathBuf;

/// Errors returned by search operations
#[derive(Debug)]
pub enum SearchError {
    IndexNotFound(PathBuf),
    Storage(String),
    Chunking(String),
    Bm25(String),
    Other(String),
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
}
