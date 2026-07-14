use glob::Pattern;
use serde::Deserialize;
use std::fs;
use std::path::Path;

pub fn default_daily_note_patterns() -> Vec<String> {
    vec![
        "YYYY-MM-DD.md".to_string(),
        "0. PeriodicNotes/YYYY/Daily/MM/YYYY-MM-DD.md".to_string(),
    ]
}

pub fn default_cache_dir() -> String {
    ".notectl/search".to_string()
}

pub fn default_embedding_dim() -> u32 {
    256
}

pub fn default_max_seq_tokens() -> usize {
    512
}

pub fn default_chunk_overlap_tokens() -> usize {
    64
}

pub fn default_min_chunk_tokens() -> usize {
    32
}

pub fn default_rrf_k() -> f64 {
    60.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchConfig {
    /// Model ID for dense embeddings (e.g., "google/embedding-gemma-300m")
    #[serde(default = "default_model_id")]
    pub model_id: String,

    /// Model revision/tag
    #[serde(default)]
    pub model_revision: Option<String>,

    /// Embedding dimension (for matryoshka truncation)
    #[serde(default = "default_embedding_dim")]
    pub embedding_dim: u32,

    /// Maximum sequence tokens for chunking
    #[serde(default = "default_max_seq_tokens")]
    pub max_seq_tokens: usize,

    /// Token overlap between adjacent chunks
    #[serde(default = "default_chunk_overlap_tokens")]
    pub chunk_overlap_tokens: usize,

    /// Minimum tokens per chunk before merging forward
    #[serde(default = "default_min_chunk_tokens")]
    pub min_chunk_tokens: usize,

    /// RRF k parameter for reciprocal rank fusion
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f64,

    /// Optional dense model weights path (overrides hf-hub download)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dense_weights: Option<String>,

    /// Optional sparse model weights path (overrides BM25 defaults)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sparse_weights: Option<String>,

    /// Cache directory for search index and model weights
    #[serde(default = "default_cache_dir")]
    pub cache_dir: String,
}

fn default_model_id() -> String {
    "google/embedding-gemma-300m".to_string()
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            model_id: default_model_id(),
            model_revision: None,
            embedding_dim: default_embedding_dim(),
            max_seq_tokens: default_max_seq_tokens(),
            chunk_overlap_tokens: default_chunk_overlap_tokens(),
            min_chunk_tokens: default_min_chunk_tokens(),
            rrf_k: default_rrf_k(),
            dense_weights: None,
            sparse_weights: None,
            cache_dir: default_cache_dir(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub exclude_paths: Vec<String>,

    #[serde(default = "default_daily_note_patterns")]
    pub daily_note_patterns: Vec<String>,

    #[serde(default)]
    pub search: SearchConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            exclude_paths: Vec::new(),
            daily_note_patterns: default_daily_note_patterns(),
            search: SearchConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration from a file at the specified path
    /// Falls back to default config if file doesn't exist or can't be read
    pub fn load_from_file(config_path: &Path) -> Self {
        if !config_path.exists() {
            return Config::default();
        }

        match fs::read_to_string(config_path) {
            Ok(content) => toml::from_str(&content).unwrap_or_default(),
            Err(_) => Config::default(),
        }
    }

    /// Load configuration from the base path (looks for .notectl.toml)
    /// Also merges configuration from environment variables
    pub fn load_from_base_path(base_path: &Path) -> Self {
        let config_path = base_path.join(".notectl.toml");
        let mut config = Self::load_from_file(&config_path);

        // Merge in environment variable configuration
        config.merge_from_env();

        config
    }

    /// Merge configuration from environment variables
    /// NOTECTL_EXCLUDE_PATHS: comma-separated list of exclusion patterns
    /// NOTECTL_DAILY_NOTE_PATTERNS: comma-separated list of daily note patterns
    /// NOTECTL_SEARCH_CACHE_DIR: search cache directory
    /// NOTECTL_SEARCH_EMBEDDING_DIM: embedding dimension
    /// NOTECTL_SEARCH_MAX_SEQ_TOKENS: maximum sequence tokens
    fn merge_from_env(&mut self) {
        self.merge_from_env_var("NOTECTL_EXCLUDE_PATHS");

        // Merge daily note patterns from environment variable
        if let Ok(env_patterns) = std::env::var("NOTECTL_DAILY_NOTE_PATTERNS") {
            let env_daily_patterns: Vec<String> = env_patterns
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            // Extend existing patterns with env var patterns
            self.daily_note_patterns.extend(env_daily_patterns);
        }

        // Merge search configuration from environment variables
        self.merge_search_from_env();
    }

    /// Merge search-specific configuration from environment variables
    fn merge_search_from_env(&mut self) {
        if let Ok(cache_dir) = std::env::var("NOTECTL_SEARCH_CACHE_DIR") {
            self.search.cache_dir = cache_dir;
        }

        if let Ok(dim) = std::env::var("NOTECTL_SEARCH_EMBEDDING_DIM")
            && let Ok(dim_val) = dim.parse::<u32>()
        {
            self.search.embedding_dim = dim_val;
        }

        if let Ok(tokens) = std::env::var("NOTECTL_SEARCH_MAX_SEQ_TOKENS")
            && let Ok(tokens_val) = tokens.parse::<usize>()
        {
            self.search.max_seq_tokens = tokens_val;
        }
    }

    /// Merge configuration from a specific environment variable
    fn merge_from_env_var(&mut self, var_name: &str) {
        if let Ok(env_excludes) = std::env::var(var_name) {
            let env_patterns: Vec<String> = env_excludes
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            // Extend existing patterns with env var patterns
            self.exclude_paths.extend(env_patterns);
        }
    }

    /// Check if a given path should be excluded based on configured patterns
    pub fn should_exclude(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();

        for pattern_str in &self.exclude_paths {
            // Try to compile the pattern
            if let Ok(pattern) = Pattern::new(pattern_str)
                && pattern.matches(&path_str)
            {
                return true;
            }

            // Also check if the path contains the pattern as a substring
            // This handles simple cases like "Template" or "Recipes"
            if path_str.contains(pattern_str) {
                return true;
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_should_exclude_substring() {
        let config = Config {
            exclude_paths: vec!["Template".to_string(), "Recipes".to_string()],
            daily_note_patterns: default_daily_note_patterns(),
            search: SearchConfig::default(),
        };

        assert!(config.should_exclude(&PathBuf::from("/vault/Templates/note.md")));
        assert!(config.should_exclude(&PathBuf::from("/vault/Recipes/recipe.md")));
        assert!(!config.should_exclude(&PathBuf::from("/vault/Notes/note.md")));
    }

    #[test]
    fn test_should_exclude_glob_pattern() {
        let config = Config {
            exclude_paths: vec!["**/Template/**".to_string(), "**/Recipes/**".to_string()],
            daily_note_patterns: default_daily_note_patterns(),
            search: SearchConfig::default(),
        };

        assert!(config.should_exclude(&PathBuf::from("/vault/Template/note.md")));
        assert!(config.should_exclude(&PathBuf::from("/vault/Recipes/recipe.md")));
        assert!(config.should_exclude(&PathBuf::from("/vault/sub/Template/note.md")));
        assert!(!config.should_exclude(&PathBuf::from("/vault/Notes/note.md")));
    }

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.exclude_paths.is_empty());
        assert!(!config.should_exclude(&PathBuf::from("/vault/anything.md")));
    }

    #[test]
    fn test_merge_from_env() {
        // Use a unique env var name for this test to avoid parallel test conflicts
        const TEST_VAR: &str = "NOTECTL_TEST_MERGE_FROM_ENV";

        // Set env var
        unsafe {
            std::env::set_var(TEST_VAR, "Archive, Backup, **/tmp/**");
        }

        let mut config = Config {
            exclude_paths: vec!["Template".to_string()],
            daily_note_patterns: default_daily_note_patterns(),
            search: SearchConfig::default(),
        };

        config.merge_from_env_var(TEST_VAR);

        // Should have both original and env var patterns
        assert_eq!(config.exclude_paths.len(), 4);
        assert!(config.exclude_paths.contains(&"Template".to_string()));
        assert!(config.exclude_paths.contains(&"Archive".to_string()));
        assert!(config.exclude_paths.contains(&"Backup".to_string()));
        assert!(config.exclude_paths.contains(&"**/tmp/**".to_string()));

        // Clean up
        unsafe {
            std::env::remove_var(TEST_VAR);
        }
    }

    #[test]
    fn test_env_with_empty_patterns() {
        // Use a unique env var name for this test to avoid parallel test conflicts
        const TEST_VAR: &str = "NOTECTL_TEST_EMPTY_PATTERNS";

        // Test that empty strings are filtered out
        unsafe {
            std::env::set_var(TEST_VAR, "Archive, , Backup,  ,");
        }

        let mut config = Config::default();
        config.merge_from_env_var(TEST_VAR);

        assert_eq!(config.exclude_paths.len(), 2);
        assert!(config.exclude_paths.contains(&"Archive".to_string()));
        assert!(config.exclude_paths.contains(&"Backup".to_string()));

        // Clean up
        unsafe {
            std::env::remove_var(TEST_VAR);
        }
    }

    #[test]
    fn test_search_config_default() {
        let config = Config::default();
        assert_eq!(config.search.model_id, "google/embedding-gemma-300m");
        assert_eq!(config.search.embedding_dim, 256);
        assert_eq!(config.search.max_seq_tokens, 512);
        assert_eq!(config.search.chunk_overlap_tokens, 64);
        assert_eq!(config.search.min_chunk_tokens, 32);
        assert!((config.search.rrf_k - 60.0).abs() < f64::EPSILON);
        assert_eq!(config.search.cache_dir, ".notectl/search");
        assert!(config.search.model_revision.is_none());
        assert!(config.search.dense_weights.is_none());
        assert!(config.search.sparse_weights.is_none());
    }

    #[test]
    fn test_search_config_from_toml() {
        let toml_str = r#"
[search]
model_id = "custom/model"
embedding_dim = 128
max_seq_tokens = 256
chunk_overlap_tokens = 32
min_chunk_tokens = 16
rrf_k = 30.0
cache_dir = "/tmp/search-cache"
"#;

        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.search.model_id, "custom/model");
        assert_eq!(config.search.embedding_dim, 128);
        assert_eq!(config.search.max_seq_tokens, 256);
        assert_eq!(config.search.chunk_overlap_tokens, 32);
        assert_eq!(config.search.min_chunk_tokens, 16);
        assert!((config.search.rrf_k - 30.0).abs() < f64::EPSILON);
        assert_eq!(config.search.cache_dir, "/tmp/search-cache");
    }

    #[test]
    fn test_search_config_merge_from_env() {
        // Set the test env vars (we'll read them via the actual config env var names)
        unsafe {
            std::env::set_var("NOTECTL_SEARCH_CACHE_DIR", "/custom/cache");
            std::env::set_var("NOTECTL_SEARCH_EMBEDDING_DIM", "512");
            std::env::set_var("NOTECTL_SEARCH_MAX_SEQ_TOKENS", "1024");
        }

        let mut config = Config::default();
        config.merge_search_from_env();

        assert_eq!(config.search.cache_dir, "/custom/cache");
        assert_eq!(config.search.embedding_dim, 512);
        assert_eq!(config.search.max_seq_tokens, 1024);

        // Clean up
        unsafe {
            std::env::remove_var("NOTECTL_SEARCH_CACHE_DIR");
            std::env::remove_var("NOTECTL_SEARCH_EMBEDDING_DIM");
            std::env::remove_var("NOTECTL_SEARCH_MAX_SEQ_TOKENS");
        }
    }
}
