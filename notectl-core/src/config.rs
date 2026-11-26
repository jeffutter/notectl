use glob::Pattern;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

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
    4096
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

pub fn default_max_results() -> usize {
    50
}

pub fn default_rrf_bm25_weight() -> f64 {
    1.0
}

pub fn default_rrf_cosine_weight() -> f64 {
    1.0
}

pub fn default_merge_threshold() -> usize {
    30
}

pub fn default_rrf_recency_weight() -> f64 {
    0.5
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchConfig {
    /// Model ID sent as `"model"` to the embedding API (e.g. "qwen3-embedding:0.6b")
    #[serde(default = "default_model_id")]
    pub model_id: String,

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

    /// Weight of the BM25 score in hybrid mode (not the BM25 k1 saturation constant)
    #[serde(default = "default_rrf_bm25_weight")]
    pub rrf_bm25_weight: f64,

    /// Weight of the cosine similarity in hybrid mode
    #[serde(default = "default_rrf_cosine_weight")]
    pub rrf_cosine_weight: f64,

    /// Base URL for an OpenAI-compatible embeddings endpoint (e.g.
    /// "https://host/v1"). `/embeddings` is appended to build the request
    /// URL. `None` disables dense embeddings entirely — there is no
    /// bundled/local model; BM25 keyword search still works either way.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_api_base: Option<String>,

    /// Optional bearer token for the embedding API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_api_key: Option<String>,

    /// Model ID sent to the rerank endpoint at
    /// `{embedding_api_base}/rerank`, with the same API key. When set, search
    /// results are scored and ordered by this model. `None` disables it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank_model: Option<String>,

    /// Lowest cosine similarity a chunk found by the dense search may have and
    /// still be sent to the rerank model (default 0, no cutoff). The useful
    /// value depends on the embedding model.
    #[serde(default)]
    pub rerank_min_cosine: f64,

    /// Lowest BM25 score (as a share of the most the query could score, 0 to 1)
    /// a chunk found by the keyword search may have and still be sent to the
    /// rerank model (default 0, no cutoff).
    #[serde(default)]
    pub rerank_min_bm25: f64,

    /// The most chunks each search path (dense, keyword) sends to the rerank
    /// model, whatever the requested result count (default 20, at least 1).
    /// Higher looks further for good matches, takes longer, and raises the
    /// most results a reranked search can return.
    #[serde(default = "default_rerank_candidates")]
    pub rerank_candidates: usize,

    /// How many characters of each chunk the rerank model reads (default 2000,
    /// at least 1). Its time grows with the text it is sent.
    #[serde(default = "default_rerank_max_chars")]
    pub rerank_max_chars: usize,

    /// Optional sparse model weights path (overrides BM25 defaults)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sparse_weights: Option<String>,

    /// Cache directory for search index and model weights
    #[serde(default = "default_cache_dir")]
    pub cache_dir: String,

    /// Maximum number of results to return from a search query
    #[serde(default = "default_max_results")]
    pub max_results: usize,

    /// Merge tiny sections into the next one if below this threshold (token count)
    #[serde(default = "default_merge_threshold")]
    pub merge_threshold: usize,

    /// Default recency weight applied to scores when a search request
    /// opts in to recency boosting (default 0.5). Recency boosting itself is
    /// off by default; callers enable it per-request.
    #[serde(default = "default_rrf_recency_weight")]
    pub rrf_recency_weight: f64,
}

fn default_rerank_candidates() -> usize {
    20
}

fn default_rerank_max_chars() -> usize {
    2000
}

fn default_model_id() -> String {
    String::new()
}

impl SearchConfig {
    /// Returns the resolved index/cache directory path relative to a base path.
    /// If `cache_dir` is absolute, returns it as-is; otherwise joins it to `base_path`.
    pub fn resolve_index_dir(&self, base_path: &std::path::Path) -> std::path::PathBuf {
        if std::path::Path::new(&self.cache_dir).is_absolute() {
            std::path::PathBuf::from(&self.cache_dir)
        } else {
            base_path.join(&self.cache_dir)
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            model_id: default_model_id(),
            embedding_dim: default_embedding_dim(),
            max_seq_tokens: default_max_seq_tokens(),
            chunk_overlap_tokens: default_chunk_overlap_tokens(),
            min_chunk_tokens: default_min_chunk_tokens(),
            rrf_bm25_weight: default_rrf_bm25_weight(),
            rrf_cosine_weight: default_rrf_cosine_weight(),
            embedding_api_base: None,
            embedding_api_key: None,
            rerank_model: None,
            rerank_min_cosine: 0.0,
            rerank_min_bm25: 0.0,
            rerank_candidates: default_rerank_candidates(),
            rerank_max_chars: default_rerank_max_chars(),
            sparse_weights: None,
            cache_dir: default_cache_dir(),
            max_results: default_max_results(),
            merge_threshold: default_merge_threshold(),
            rrf_recency_weight: default_rrf_recency_weight(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub exclude_paths: Vec<String>,

    #[serde(default = "default_daily_note_patterns")]
    pub daily_note_patterns: Vec<String>,

    /// Heading patterns to exclude from search indexing and from raw
    /// file/daily-note reads (case-insensitive substring match). Useful for
    /// skipping Dataview queries, template blocks, etc.
    #[serde(default)]
    pub exclude_headings: Vec<String>,

    #[serde(default)]
    pub search: SearchConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            exclude_paths: Vec::new(),
            daily_note_patterns: default_daily_note_patterns(),
            exclude_headings: Vec::new(),
            search: SearchConfig::default(),
        }
    }
}

/// Partial config structs — every field is Option so we can distinguish
/// "not set" from "set to default value" when merging layers.
#[derive(Debug, Clone, Deserialize)]
struct PartialSearchConfig {
    model_id: Option<String>,
    embedding_dim: Option<u32>,
    max_seq_tokens: Option<usize>,
    chunk_overlap_tokens: Option<usize>,
    min_chunk_tokens: Option<usize>,
    rrf_bm25_weight: Option<f64>,
    rrf_cosine_weight: Option<f64>,
    embedding_api_base: Option<String>,
    embedding_api_key: Option<String>,
    rerank_model: Option<String>,
    rerank_min_cosine: Option<f64>,
    rerank_min_bm25: Option<f64>,
    rerank_candidates: Option<usize>,
    rerank_max_chars: Option<usize>,
    sparse_weights: Option<String>,
    cache_dir: Option<String>,
    max_results: Option<usize>,
    merge_threshold: Option<usize>,
    rrf_recency_weight: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct PartialConfig {
    exclude_paths: Option<Vec<String>>,
    daily_note_patterns: Option<Vec<String>>,
    exclude_headings: Option<Vec<String>>,
    search: Option<PartialSearchConfig>,
}

impl Config {
    /// Resolve the global config path: $XDG_CONFIG_HOME/notectl/config.toml
    /// or ~/.config/notectl/config.toml as a fallback.
    fn global_config_path() -> Option<PathBuf> {
        if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
            return Some(Path::new(&xdg).join("notectl/config.toml"));
        }

        dirs::home_dir().map(|h| h.join(".config/notectl/config.toml"))
    }

    /// Load configuration from a file at the specified path.
    /// Returns None if the file doesn't exist or can't be parsed.
    fn load_partial(path: &Path) -> Option<PartialConfig> {
        if !path.exists() {
            return None;
        }

        match fs::read_to_string(path) {
            Ok(content) => toml::from_str(&content).ok(),
            Err(_) => None,
        }
    }

    /// Merge a partial config onto this one, only overriding fields that were set.
    fn apply_partial(&mut self, partial: &PartialConfig) {
        if let Some(ref paths) = partial.exclude_paths {
            self.exclude_paths.extend(paths.clone());
        }

        if let Some(ref patterns) = partial.daily_note_patterns {
            self.daily_note_patterns.extend(patterns.clone());
        }

        if let Some(ref headings) = partial.exclude_headings {
            self.exclude_headings.extend(headings.clone());
        }

        if let Some(ref s) = partial.search {
            if let Some(ref v) = s.model_id {
                self.search.model_id = v.clone();
            }
            if let Some(v) = s.embedding_dim {
                self.search.embedding_dim = v;
            }
            if let Some(v) = s.max_seq_tokens {
                self.search.max_seq_tokens = v;
            }
            if let Some(v) = s.chunk_overlap_tokens {
                self.search.chunk_overlap_tokens = v;
            }
            if let Some(v) = s.min_chunk_tokens {
                self.search.min_chunk_tokens = v;
            }
            if let Some(v) = s.rrf_bm25_weight {
                self.search.rrf_bm25_weight = v;
            }
            if let Some(v) = s.rrf_cosine_weight {
                self.search.rrf_cosine_weight = v;
            }
            if let Some(ref v) = s.embedding_api_base {
                self.search.embedding_api_base = Some(v.clone());
            }
            if let Some(ref v) = s.embedding_api_key {
                self.search.embedding_api_key = Some(v.clone());
            }
            if let Some(ref v) = s.rerank_model {
                self.search.rerank_model = Some(v.clone());
            }
            if let Some(v) = s.rerank_min_cosine {
                self.search.rerank_min_cosine = v;
            }
            if let Some(v) = s.rerank_min_bm25 {
                self.search.rerank_min_bm25 = v;
            }
            if let Some(v) = s.rerank_candidates {
                self.search.rerank_candidates = v;
            }
            if let Some(v) = s.rerank_max_chars {
                self.search.rerank_max_chars = v;
            }
            if let Some(ref v) = s.sparse_weights {
                self.search.sparse_weights = Some(v.clone());
            }
            if let Some(ref v) = s.cache_dir {
                self.search.cache_dir = v.clone();
            }
            if let Some(v) = s.max_results {
                self.search.max_results = v;
            }
            if let Some(v) = s.merge_threshold {
                self.search.merge_threshold = v;
            }
            if let Some(v) = s.rrf_recency_weight {
                self.search.rrf_recency_weight = v;
            }
        }
    }

    /// Load configuration from a file at the specified path.
    /// Falls back to default config if file doesn't exist or can't be read.
    #[deprecated(since = "0.10.1", note = "Use load_from_base_path instead")]
    pub fn load_from_file(config_path: &Path) -> Self {
        if let Some(partial) = Self::load_partial(config_path) {
            let mut config = Config::default();
            config.apply_partial(&partial);
            config
        } else {
            Config::default()
        }
    }

    /// Load configuration with three-layer precedence:
    /// 1. Hardcoded defaults
    /// 2. Global config ($XDG_CONFIG_HOME/notectl/config.toml or ~/.config/notectl/config.toml)
    /// 3. Vault config (<base_path>/.notectl.toml)
    /// 4. Environment variables (applied after return via merge_from_env)
    pub fn load_from_base_path(base_path: &Path) -> Self {
        // Start with defaults
        let mut config = Config::default();

        // Layer 2: global user config
        if let Some(global_path) = Self::global_config_path()
            && let Some(partial) = Self::load_partial(&global_path)
        {
            tracing::debug!(path = %global_path.display(), "loaded global config");
            config.apply_partial(&partial);
        }

        // Layer 3: vault-level config
        let vault_config_path = base_path.join(".notectl.toml");
        if let Some(partial) = Self::load_partial(&vault_config_path) {
            tracing::debug!(path = %vault_config_path.display(), "loaded vault config");
            config.apply_partial(&partial);
        }

        // Layer 4: environment variables
        config.merge_from_env();

        config
    }

    /// Merge configuration from environment variables
    /// NOTECTL_EXCLUDE_PATHS: comma-separated list of exclusion patterns
    /// NOTECTL_DAILY_NOTE_PATTERNS: comma-separated list of daily note patterns
    /// NOTECTL_EXCLUDE_HEADINGS: comma-separated list of heading exclusion patterns
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

        // Merge excluded headings from environment variable (comma-separated)
        if let Ok(v) = std::env::var("NOTECTL_EXCLUDE_HEADINGS") {
            let patterns: Vec<String> = v
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            self.exclude_headings.extend(patterns);
        }

        // Merge search configuration from environment variables
        self.merge_search_from_env();
    }

    /// Merge search-specific configuration from environment variables.
    ///
    /// Covers every field on `SearchConfig` so that no user-set value is inert:
    /// - Model: NOTECTL_SEARCH_MODEL_ID
    /// - Embedding endpoint: NOTECTL_SEARCH_EMBEDDING_API_BASE, NOTECTL_SEARCH_EMBEDDING_API_KEY,
    ///   NOTECTL_SEARCH_EMBEDDING_DIM
    /// - Chunking: NOTECTL_SEARCH_MAX_SEQ_TOKENS, NOTECTL_SEARCH_CHUNK_OVERLAP_TOKENS,
    ///   NOTECTL_SEARCH_MIN_CHUNK_TOKENS, NOTECTL_SEARCH_MERGE_THRESHOLD
    /// - Scoring: NOTECTL_SEARCH_RRF_BM25_WEIGHT,
    ///   NOTECTL_SEARCH_RRF_COSINE_WEIGHT
    /// - Reranking: NOTECTL_SEARCH_RERANK_MODEL,
    ///   NOTECTL_SEARCH_RERANK_MIN_COSINE, NOTECTL_SEARCH_RERANK_MIN_BM25,
    ///   NOTECTL_SEARCH_RERANK_CANDIDATES, NOTECTL_SEARCH_RERANK_MAX_CHARS
    /// - Weights paths: NOTECTL_SEARCH_SPARSE_WEIGHTS
    /// - Output: NOTECTL_SEARCH_CACHE_DIR, NOTECTL_SEARCH_MAX_RESULTS
    fn merge_search_from_env(&mut self) {
        if let Ok(v) = std::env::var("NOTECTL_SEARCH_MODEL_ID") {
            self.search.model_id = v;
        }

        if let Ok(v) = std::env::var("NOTECTL_SEARCH_EMBEDDING_API_BASE") {
            self.search.embedding_api_base = Some(v);
        }

        if let Ok(v) = std::env::var("NOTECTL_SEARCH_RERANK_MODEL") {
            self.search.rerank_model = Some(v);
        }
        if let Ok(v) = std::env::var("NOTECTL_SEARCH_RERANK_MIN_COSINE")
            && let Ok(val) = v.parse::<f64>()
        {
            self.search.rerank_min_cosine = val;
        }
        if let Ok(v) = std::env::var("NOTECTL_SEARCH_RERANK_MIN_BM25")
            && let Ok(val) = v.parse::<f64>()
        {
            self.search.rerank_min_bm25 = val;
        }
        if let Ok(v) = std::env::var("NOTECTL_SEARCH_RERANK_CANDIDATES")
            && let Ok(val) = v.parse::<usize>()
        {
            self.search.rerank_candidates = val;
        }
        if let Ok(v) = std::env::var("NOTECTL_SEARCH_RERANK_MAX_CHARS")
            && let Ok(val) = v.parse::<usize>()
        {
            self.search.rerank_max_chars = val;
        }
        if let Ok(v) = std::env::var("NOTECTL_SEARCH_EMBEDDING_API_KEY") {
            self.search.embedding_api_key = Some(v);
        }

        if let Ok(dim) = std::env::var("NOTECTL_SEARCH_EMBEDDING_DIM")
            && let Ok(val) = dim.parse::<u32>()
        {
            self.search.embedding_dim = val;
        }

        if let Ok(v) = std::env::var("NOTECTL_SEARCH_MAX_SEQ_TOKENS")
            && let Ok(val) = v.parse::<usize>()
        {
            self.search.max_seq_tokens = val;
        }

        if let Ok(v) = std::env::var("NOTECTL_SEARCH_CHUNK_OVERLAP_TOKENS")
            && let Ok(val) = v.parse::<usize>()
        {
            self.search.chunk_overlap_tokens = val;
        }

        if let Ok(v) = std::env::var("NOTECTL_SEARCH_MIN_CHUNK_TOKENS")
            && let Ok(val) = v.parse::<usize>()
        {
            self.search.min_chunk_tokens = val;
        }

        if let Ok(v) = std::env::var("NOTECTL_SEARCH_MERGE_THRESHOLD")
            && let Ok(val) = v.parse::<usize>()
        {
            self.search.merge_threshold = val;
        }

        if let Ok(v) = std::env::var("NOTECTL_SEARCH_RRF_BM25_WEIGHT")
            && let Ok(val) = v.parse::<f64>()
        {
            self.search.rrf_bm25_weight = val;
        }

        if let Ok(v) = std::env::var("NOTECTL_SEARCH_RRF_COSINE_WEIGHT")
            && let Ok(val) = v.parse::<f64>()
        {
            self.search.rrf_cosine_weight = val;
        }

        if let Ok(v) = std::env::var("NOTECTL_SEARCH_SPARSE_WEIGHTS") {
            self.search.sparse_weights = Some(v);
        }

        if let Ok(v) = std::env::var("NOTECTL_SEARCH_CACHE_DIR") {
            self.search.cache_dir = v;
        }

        if let Ok(v) = std::env::var("NOTECTL_SEARCH_MAX_RESULTS")
            && let Ok(val) = v.parse::<usize>()
        {
            self.search.max_results = val;
        }

        if let Ok(v) = std::env::var("NOTECTL_SEARCH_RRF_RECENCY_WEIGHT")
            && let Ok(val) = v.parse::<f64>()
        {
            self.search.rrf_recency_weight = val;
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

    /// Check if a heading title matches any exclusion pattern (case-insensitive substring).
    pub fn should_exclude_heading(&self, title: &str) -> bool {
        let lower = title.to_lowercase();
        self.exclude_headings.iter().any(|pattern| {
            let lp = pattern.to_lowercase();
            lower.contains(&lp)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// `std::env::set_var`/`remove_var` mutate process-global state, but
    /// Rust's default test runner executes tests in parallel across threads
    /// in the same process — so any two tests that touch env vars can race
    /// and leak values into each other, regardless of each test's own
    /// cleanup. Every test that reads/writes `NOTECTL_SEARCH_*` (or any
    /// other env var this module consults) must hold this lock for its
    /// entire body, serializing them against each other without affecting
    /// unrelated tests.
    static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_should_exclude_substring() {
        let config = Config {
            exclude_paths: vec!["Template".to_string(), "Recipes".to_string()],
            daily_note_patterns: default_daily_note_patterns(),
            exclude_headings: Vec::new(),
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
            exclude_headings: Vec::new(),
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
        let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Use a unique env var name for this test to avoid parallel test conflicts
        const TEST_VAR: &str = "NOTECTL_TEST_MERGE_FROM_ENV";

        // Set env var
        unsafe {
            std::env::set_var(TEST_VAR, "Archive, Backup, **/tmp/**");
        }

        let mut config = Config {
            exclude_paths: vec!["Template".to_string()],
            daily_note_patterns: default_daily_note_patterns(),
            exclude_headings: Vec::new(),
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
        let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        assert_eq!(config.search.model_id, "");
        assert_eq!(config.search.embedding_dim, 4096);
        assert_eq!(config.search.max_seq_tokens, 512);
        assert_eq!(config.search.chunk_overlap_tokens, 64);
        assert_eq!(config.search.min_chunk_tokens, 32);
        assert!((config.search.rrf_bm25_weight - 1.0).abs() < f64::EPSILON);
        assert!((config.search.rrf_cosine_weight - 1.0).abs() < f64::EPSILON);
        assert_eq!(config.search.max_results, 50);
        assert_eq!(config.search.merge_threshold, 30);
        assert_eq!(config.search.cache_dir, ".notectl/search");
        assert!(config.search.embedding_api_base.is_none());
        assert!(config.search.embedding_api_key.is_none());
        assert!(config.search.rerank_model.is_none());
        assert_eq!(config.search.rerank_candidates, 20);
        assert_eq!(config.search.rerank_max_chars, 2000);
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
# A removed key, still present in older config files, is ignored.
rrf_k = 30.0
cache_dir = "/tmp/search-cache"
"#;

        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.search.model_id, "custom/model");
        assert_eq!(config.search.embedding_dim, 128);
        assert_eq!(config.search.max_seq_tokens, 256);
        assert_eq!(config.search.chunk_overlap_tokens, 32);
        assert_eq!(config.search.min_chunk_tokens, 16);
        assert_eq!(config.search.cache_dir, "/tmp/search-cache");
    }

    #[test]
    fn test_search_config_toml_new_fields() {
        let toml_str = r#"
[search]
model_id = "custom/model"
embedding_dim = 128
max_seq_tokens = 256
chunk_overlap_tokens = 32
min_chunk_tokens = 16
rrf_bm25_weight = 2.0
rrf_cosine_weight = 0.5
max_results = 25
merge_threshold = 40
rerank_candidates = 12
rerank_max_chars = 900
cache_dir = "/tmp/search-cache"
"#;

        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.search.model_id, "custom/model");
        assert_eq!(config.search.embedding_dim, 128);
        assert!((config.search.rrf_bm25_weight - 2.0).abs() < f64::EPSILON);
        assert!((config.search.rrf_cosine_weight - 0.5).abs() < f64::EPSILON);
        assert_eq!(config.search.max_results, 25);
        assert_eq!(config.search.merge_threshold, 40);
        assert_eq!(config.search.rerank_candidates, 12);
        assert_eq!(config.search.rerank_max_chars, 900);
        assert_eq!(config.search.cache_dir, "/tmp/search-cache");
    }

    #[test]
    fn test_search_config_all_env_vars() {
        let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::set_var("NOTECTL_SEARCH_MODEL_ID", "custom/model");
            std::env::set_var(
                "NOTECTL_SEARCH_EMBEDDING_API_BASE",
                "https://example.com/v1",
            );
            std::env::set_var("NOTECTL_SEARCH_EMBEDDING_API_KEY", "secret-key");
            std::env::set_var("NOTECTL_SEARCH_RERANK_MODEL", "rerank");
            std::env::set_var("NOTECTL_SEARCH_RERANK_MIN_COSINE", "0.5");
            std::env::set_var("NOTECTL_SEARCH_RERANK_MIN_BM25", "0.2");
            std::env::set_var("NOTECTL_SEARCH_RERANK_CANDIDATES", "15");
            std::env::set_var("NOTECTL_SEARCH_RERANK_MAX_CHARS", "1500");
            std::env::set_var("NOTECTL_SEARCH_EMBEDDING_DIM", "512");
            std::env::set_var("NOTECTL_SEARCH_MAX_SEQ_TOKENS", "1024");
            std::env::set_var("NOTECTL_SEARCH_CHUNK_OVERLAP_TOKENS", "128");
            std::env::set_var("NOTECTL_SEARCH_MIN_CHUNK_TOKENS", "16");
            std::env::set_var("NOTECTL_SEARCH_MERGE_THRESHOLD", "50");
            std::env::set_var("NOTECTL_SEARCH_RRF_BM25_WEIGHT", "2.5");
            std::env::set_var("NOTECTL_SEARCH_RRF_COSINE_WEIGHT", "0.5");
            std::env::set_var("NOTECTL_SEARCH_RRF_RECENCY_WEIGHT", "0.75");
            std::env::set_var("NOTECTL_SEARCH_SPARSE_WEIGHTS", "/path/to/sparse.bin");
            std::env::set_var("NOTECTL_SEARCH_CACHE_DIR", "/custom/cache");
            std::env::set_var("NOTECTL_SEARCH_MAX_RESULTS", "100");
        }

        let mut config = Config::default();
        config.merge_search_from_env();

        assert_eq!(config.search.model_id, "custom/model");
        assert_eq!(
            config.search.embedding_api_base.as_deref(),
            Some("https://example.com/v1")
        );
        assert_eq!(
            config.search.embedding_api_key.as_deref(),
            Some("secret-key")
        );
        assert_eq!(config.search.rerank_model.as_deref(), Some("rerank"));
        assert!((config.search.rerank_min_cosine - 0.5).abs() < f64::EPSILON);
        assert!((config.search.rerank_min_bm25 - 0.2).abs() < f64::EPSILON);
        assert_eq!(config.search.rerank_candidates, 15);
        assert_eq!(config.search.rerank_max_chars, 1500);
        assert_eq!(config.search.embedding_dim, 512);
        assert_eq!(config.search.max_seq_tokens, 1024);
        assert_eq!(config.search.chunk_overlap_tokens, 128);
        assert_eq!(config.search.min_chunk_tokens, 16);
        assert_eq!(config.search.merge_threshold, 50);
        assert!((config.search.rrf_bm25_weight - 2.5).abs() < f64::EPSILON);
        assert!((config.search.rrf_cosine_weight - 0.5).abs() < f64::EPSILON);
        assert!((config.search.rrf_recency_weight - 0.75).abs() < f64::EPSILON);
        assert_eq!(
            config.search.sparse_weights.as_deref(),
            Some("/path/to/sparse.bin")
        );
        assert_eq!(config.search.cache_dir, "/custom/cache");
        assert_eq!(config.search.max_results, 100);

        // Clean up all env vars
        for var in [
            "NOTECTL_SEARCH_MODEL_ID",
            "NOTECTL_SEARCH_EMBEDDING_API_BASE",
            "NOTECTL_SEARCH_EMBEDDING_API_KEY",
            "NOTECTL_SEARCH_RERANK_MODEL",
            "NOTECTL_SEARCH_RERANK_MIN_COSINE",
            "NOTECTL_SEARCH_RERANK_MIN_BM25",
            "NOTECTL_SEARCH_RERANK_CANDIDATES",
            "NOTECTL_SEARCH_RERANK_MAX_CHARS",
            "NOTECTL_SEARCH_EMBEDDING_DIM",
            "NOTECTL_SEARCH_MAX_SEQ_TOKENS",
            "NOTECTL_SEARCH_CHUNK_OVERLAP_TOKENS",
            "NOTECTL_SEARCH_MIN_CHUNK_TOKENS",
            "NOTECTL_SEARCH_MERGE_THRESHOLD",
            "NOTECTL_SEARCH_RRF_BM25_WEIGHT",
            "NOTECTL_SEARCH_RRF_COSINE_WEIGHT",
            "NOTECTL_SEARCH_RRF_RECENCY_WEIGHT",
            "NOTECTL_SEARCH_SPARSE_WEIGHTS",
            "NOTECTL_SEARCH_CACHE_DIR",
            "NOTECTL_SEARCH_MAX_RESULTS",
        ] {
            unsafe {
                std::env::remove_var(var);
            }
        }
    }

    #[test]
    fn test_should_exclude_heading() {
        let config = Config {
            exclude_headings: vec!["Query".to_string(), "daily tasks".to_string()],
            ..Default::default()
        };

        // Exact match
        assert!(config.should_exclude_heading("Query"));
        // Case insensitive
        assert!(config.should_exclude_heading("QUERY"));
        assert!(config.should_exclude_heading("query"));
        // Substring match
        assert!(config.should_exclude_heading("My Dataview Query Block"));
        assert!(config.should_exclude_heading("Daily Tasks"));
        // No match
        assert!(!config.should_exclude_heading("Notes"));
        assert!(!config.should_exclude_heading("Introduction"));
    }

    #[test]
    fn test_exclude_headings_env_var() {
        let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::set_var("NOTECTL_EXCLUDE_HEADINGS", "Query, Daily Tasks, Template");
        }

        let mut config = Config::default();
        config.merge_from_env();

        assert!(config.exclude_headings.contains(&"Query".to_string()));
        assert!(config.exclude_headings.contains(&"Daily Tasks".to_string()));
        assert!(config.exclude_headings.contains(&"Template".to_string()));

        unsafe {
            std::env::remove_var("NOTECTL_EXCLUDE_HEADINGS");
        }
    }

    #[test]
    fn test_global_config_path_xdg_set() {
        let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::set_var("XDG_CONFIG_HOME", "/custom/xdg/config");
        }

        let path = Config::global_config_path();
        assert_eq!(
            path,
            Some(PathBuf::from("/custom/xdg/config/notectl/config.toml"))
        );

        unsafe {
            std::env::remove_var("XDG_CONFIG_HOME");
        }
    }

    #[test]
    fn test_global_config_path_fallback_to_home() {
        let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Clear XDG_CONFIG_HOME to test fallback
        unsafe {
            std::env::remove_var("XDG_CONFIG_HOME");
        }

        let path = Config::global_config_path();
        // Should fall back to ~/.config/notectl/config.toml
        assert!(path.is_some());
        let p = path.unwrap();
        assert!(p.to_string_lossy().ends_with(".config/notectl/config.toml"));
    }

    #[test]
    fn test_load_partial_none_for_missing_file() {
        let result = Config::load_partial(&PathBuf::from("/nonexistent/.notectl.toml"));
        assert!(result.is_none());
    }

    #[test]
    fn test_apply_partial_only_sets_provided_fields() {
        let mut config = Config {
            exclude_paths: vec!["Template".to_string()],
            daily_note_patterns: default_daily_note_patterns(),
            exclude_headings: Vec::new(),
            search: SearchConfig {
                model_id: "default/model".to_string(),
                embedding_dim: 1024,
                ..Default::default()
            },
        };

        // Parse a partial config that only sets model_id
        let toml_str = r#"
[search]
model_id = "custom/model"
"#;
        let partial: PartialConfig = toml::from_str(toml_str).unwrap();
        config.apply_partial(&partial);

        // model_id should be overridden
        assert_eq!(config.search.model_id, "custom/model");
        // Other fields should remain unchanged
        assert_eq!(config.search.embedding_dim, 1024);
        assert_eq!(config.exclude_paths, vec!["Template"]);
    }

    #[test]
    fn test_load_from_base_path_layering() {
        let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Save and restore env vars to avoid interference from parallel tests
        let saved_xdg = std::env::var("XDG_CONFIG_HOME").ok();
        let saved_model = std::env::var("NOTECTL_SEARCH_MODEL_ID").ok();
        unsafe {
            std::env::remove_var("NOTECTL_SEARCH_MODEL_ID");
        }

        let temp_dir = tempfile::tempdir().unwrap();
        let vault_dir = temp_dir.path();

        // Create a vault-level config with only model_id
        let vault_config = vault_dir.join(".notectl.toml");
        fs::write(
            &vault_config,
            r#"
[search]
model_id = "vault/model"
"#,
        )
        .unwrap();

        // Create a global config directory and file
        let global_dir = temp_dir.path().join("global_config");
        fs::create_dir_all(&global_dir).unwrap();
        let global_file = global_dir.join("notectl/config.toml");
        fs::create_dir_all(global_file.parent().unwrap()).unwrap();
        fs::write(
            &global_file,
            r#"
[search]
embedding_dim = 512
max_seq_tokens = 256
"#,
        )
        .unwrap();

        // Set XDG_CONFIG_HOME to point to our test global config
        unsafe {
            std::env::set_var("XDG_CONFIG_HOME", global_dir.to_str().unwrap());
        }

        let config = Config::load_from_base_path(vault_dir);

        // Vault-level model_id should override global defaults
        assert_eq!(config.search.model_id, "vault/model");
        // Global-level settings should be inherited
        assert_eq!(config.search.embedding_dim, 512);
        assert_eq!(config.search.max_seq_tokens, 256);
        // Default values for unset fields
        assert_eq!(config.search.cache_dir, ".notectl/search");

        // Restore env vars
        unsafe {
            match saved_xdg {
                Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
            if let Some(v) = saved_model {
                std::env::set_var("NOTECTL_SEARCH_MODEL_ID", v);
            } else {
                std::env::remove_var("NOTECTL_SEARCH_MODEL_ID");
            }
        }
    }

    #[test]
    fn test_env_overrides_vault_and_global() {
        let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Save and restore env vars to avoid interference from parallel tests
        let saved_xdg = std::env::var("XDG_CONFIG_HOME").ok();
        let saved_model = std::env::var("NOTECTL_SEARCH_MODEL_ID").ok();

        let temp_dir = tempfile::tempdir().unwrap();
        let vault_dir = temp_dir.path();

        // Vault config sets model_id
        fs::write(
            vault_dir.join(".notectl.toml"),
            r#"
[search]
model_id = "vault/model"
embedding_dim = 512
"#,
        )
        .unwrap();

        // Env var should override both
        unsafe {
            std::env::set_var("NOTECTL_SEARCH_MODEL_ID", "env/model");
        }

        let config = Config::load_from_base_path(vault_dir);

        assert_eq!(config.search.model_id, "env/model");
        assert_eq!(config.search.embedding_dim, 512); // From vault, not overridden by env

        // Restore env vars
        unsafe {
            match saved_xdg {
                Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
            if let Some(v) = saved_model {
                std::env::set_var("NOTECTL_SEARCH_MODEL_ID", v);
            } else {
                std::env::remove_var("NOTECTL_SEARCH_MODEL_ID");
            }
        }
    }
}
