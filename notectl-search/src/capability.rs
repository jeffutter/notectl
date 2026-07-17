use crate::{RankedChunk, SearchMode};
use clap::{CommandFactory, FromArgMatches};
use notectl_core::CapabilityResult;
use notectl_core::config::Config;
use notectl_core::error::internal_error;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Operation metadata
// ---------------------------------------------------------------------------

pub mod index {
    pub const DESCRIPTION: &str = "Build or update the search index for all markdown files in the vault. Computes chunks, optional embeddings, and persists index artifacts.";
    pub const CLI_NAME: &str = "index";
    pub const HTTP_PATH: &str = "/api/search/index";
}

pub mod search {
    pub const DESCRIPTION: &str = "Search across all indexed notes using hybrid (dense + sparse), dense-only, or sparse-only scoring. Auto-degrades when vectors are unavailable.";
    pub const CLI_NAME: &str = "search";
    pub const HTTP_PATH: &str = "/api/search";
}

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

/// Parameters for the index operation
#[derive(Debug, Deserialize, Serialize, JsonSchema, clap::Parser)]
#[command(name = "index", about = "Build or update the search index")]
pub struct IndexRequest {
    /// Path to vault (CLI only - not used in HTTP/MCP)
    #[arg(index = 1, required = true, help = "Path to vault root")]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub vault_path: Option<PathBuf>,

    /// Force a full reindex even if the index appears up-to-date
    #[arg(long, help = "Force full reindex")]
    #[schemars(description = "If true, delete existing index artifacts and rebuild from scratch")]
    pub reindex: Option<bool>,

    /// Override the embedding model ID (e.g., "google/embedding-gemma-300m")
    #[arg(long, help = "Override embedding model ID")]
    #[schemars(description = "Override the embedding model ID from config")]
    pub model: Option<String>,

    /// Override the embedding dimension
    #[arg(long, help = "Override embedding dimension")]
    #[schemars(description = "Override the embedding dimension from config")]
    pub dim: Option<u32>,
}

/// Response from the index operation
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct IndexResponse {
    /// Number of files indexed
    pub files_indexed: usize,
    /// Number of chunks produced
    pub chunks_produced: usize,
    /// Whether embeddings were computed
    pub has_embeddings: bool,
    /// Overall content hash
    pub content_hash: String,
    /// Duration of the index build in milliseconds
    pub duration_ms: u128,
}

/// Parameters for the search operation
#[derive(Debug, Deserialize, Serialize, JsonSchema, clap::Parser)]
#[command(name = "search", about = "Search across indexed notes")]
pub struct SearchRequest {
    /// Path to vault (CLI only - not used in HTTP/MCP)
    #[arg(index = 1, required = true, help = "Path to vault root")]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub vault_path: Option<PathBuf>,

    /// Search query string
    #[arg(index = 2, required = true, help = "Search query")]
    #[schemars(description = "The text to search for")]
    pub query: String,

    /// Maximum number of results (default 50)
    #[arg(long, default_value = "50", help = "Maximum number of results")]
    #[schemars(description = "Maximum number of results to return (default 50)")]
    pub limit: Option<usize>,

    /// Search mode: hybrid, dense, or sparse (default hybrid)
    #[arg(long, value_enum, default_value = "hybrid", help = "Search mode")]
    #[schemars(
        description = "Scoring mode: hybrid (dense+sparse fused), dense only, or sparse only (default hybrid)"
    )]
    pub mode: Option<SearchMode>,

    /// Skip staleness check and reindexing
    #[arg(long, help = "Skip reindexing, use existing index as-is")]
    #[schemars(
        description = "If true, skip the staleness check and use the existing index without rebuilding"
    )]
    pub no_reindex: Option<bool>,
}

/// Response from the search operation
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SearchResponse {
    /// Ranked search results
    pub results: Vec<RankedChunk>,
    /// Total number of results returned
    pub total_count: usize,
    /// Effective search mode used (may differ from requested due to auto-degradation)
    pub mode_used: String,
}

// ---------------------------------------------------------------------------
// Capability
// ---------------------------------------------------------------------------

/// Capability for search operations (index, search)
pub struct SearchCapability {
    base_path: PathBuf,
    config: Arc<Config>,
}

impl SearchCapability {
    /// Create a new SearchCapability
    pub fn new(base_path: PathBuf, config: Arc<Config>) -> Self {
        Self { base_path, config }
    }

    /// Build or update the search index.
    ///
    /// If `reindex` is true, existing index artifacts (manifest.json, chunks/,
    /// vectors.bin) are deleted before rebuilding. The models/ directory is
    /// preserved to avoid unnecessary redownloads.
    pub async fn build_index(
        &self,
        reindex: bool,
        model_override: Option<String>,
        dim_override: Option<u32>,
    ) -> CapabilityResult<IndexResponse> {
        let start = std::time::Instant::now();

        // Apply overrides by cloning the config.
        let mut config = (*self.config).clone();
        if let Some(model) = model_override {
            config.search.model_id = model;
        }
        if let Some(dim) = dim_override {
            config.search.embedding_dim = dim;
        }

        // If --reindex, wipe index artifacts (preserve models/).
        if reindex {
            let index_dir = config.search.resolve_index_dir(&self.base_path);
            if index_dir.exists() {
                let index = crate::storage::SearchIndex::open_or_create(
                    &index_dir,
                    config.search.model_id.clone(),
                    config.search.embedding_dim,
                    crate::storage::ChunkConfigSnapshot {
                        max_tokens: config.search.max_seq_tokens,
                        overlap_tokens: config.search.chunk_overlap_tokens,
                        min_chunk_size: config.search.min_chunk_tokens,
                        merge_threshold: config.search.merge_threshold,
                    },
                )
                .map_err(|e| internal_error(format!("Failed to open index for cleanup: {e}")))?;

                index
                    .remove_manifest()
                    .map_err(|e| internal_error(format!("Failed to remove manifest: {e}")))?;
                index
                    .clear_chunks()
                    .map_err(|e| internal_error(format!("Failed to clear chunks: {e}")))?;
                index
                    .remove_vectors()
                    .map_err(|e| internal_error(format!("Failed to remove vectors: {e}")))?;
            }
        }

        let summary = crate::index::build_index(&self.base_path, &config)
            .await
            .map_err(|e| internal_error(format!("Index build failed: {e}")))?;

        let duration_ms = start.elapsed().as_millis();

        Ok(IndexResponse {
            files_indexed: summary.files_indexed,
            chunks_produced: summary.chunks_produced,
            has_embeddings: summary.has_embeddings,
            content_hash: summary.content_hash,
            duration_ms,
        })
    }

    /// Execute a search query with the given options.
    pub async fn do_search(
        &self,
        query: &str,
        limit: usize,
        mode: SearchMode,
        no_reindex: bool,
    ) -> CapabilityResult<SearchResponse> {
        let options = crate::search::SearchOptions {
            mode,
            max_results: limit,
            rrf_k: self.config.search.rrf_k,
            rrf_bm25_weight: self.config.search.rrf_bm25_weight,
            rrf_cosine_weight: self.config.search.rrf_cosine_weight,
            no_reindex,
        };

        let outcome = crate::search::search(&self.base_path, &self.config, query, options)
            .await
            .map_err(|e| internal_error(format!("Search failed: {e}")))?;

        // Use the actual effective mode from the search pipeline, not the requested mode.
        let mode_used = match outcome.mode_used {
            SearchMode::Hybrid => "hybrid".to_string(),
            SearchMode::Dense => "dense".to_string(),
            SearchMode::Sparse => "sparse".to_string(),
        };

        let total_count = outcome.results.len();

        Ok(SearchResponse {
            results: outcome.results,
            total_count,
            mode_used,
        })
    }
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

/// Operation struct for index (HTTP, CLI, and MCP)
pub struct IndexOperation {
    capability: Arc<SearchCapability>,
}

impl IndexOperation {
    pub fn new(capability: Arc<SearchCapability>) -> Self {
        Self { capability }
    }
}

/// Operation struct for search (HTTP, CLI, and MCP)
pub struct SearchOperation {
    capability: Arc<SearchCapability>,
}

impl SearchOperation {
    pub fn new(capability: Arc<SearchCapability>) -> Self {
        Self { capability }
    }
}

#[async_trait::async_trait]
impl notectl_core::operation::Operation for IndexOperation {
    fn name(&self) -> &'static str {
        index::CLI_NAME
    }

    fn path(&self) -> &'static str {
        index::HTTP_PATH
    }

    fn description(&self) -> &'static str {
        index::DESCRIPTION
    }

    fn get_command(&self) -> clap::Command {
        IndexRequest::command()
    }

    // NOTE: This same panic-on-missing-arg risk applies to every other capability file
    // (notectl-outline, notectl-tags, notectl-files, notectl-daily-notes, notectl-tasks)
    // since they all follow the identical pattern of omitting vault_path from
    // get_remote_command while args_to_json routes through Request::from_arg_matches.
    // See TASK-14 for details. A follow-up ticket should fix those systematically.
    fn get_remote_command(&self) -> clap::Command {
        // Rebuild without the vault_path positional.
        clap::Command::new("index")
            .about("Build or update the search index")
            .arg(
                clap::Arg::new("reindex")
                    .long("reindex")
                    .value_parser(clap::value_parser!(bool))
                    .help("Force full reindex"),
            )
            .arg(
                clap::Arg::new("model")
                    .long("model")
                    .value_parser(clap::value_parser!(String))
                    .help("Override embedding model ID"),
            )
            .arg(
                clap::Arg::new("dim")
                    .long("dim")
                    .value_parser(clap::value_parser!(u32))
                    .help("Override embedding dimension"),
            )
    }

    async fn execute_json(
        &self,
        json: serde_json::Value,
    ) -> Result<serde_json::Value, rmcp::model::ErrorData> {
        let request: IndexRequest = serde_json::from_value(json)
            .map_err(|e| notectl_core::error::invalid_params(e.to_string()))?;
        let response = self
            .capability
            .build_index(request.reindex.unwrap_or(false), request.model, request.dim)
            .await?;
        Ok(serde_json::to_value(response).unwrap())
    }

    async fn execute_from_args(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let request = IndexRequest::from_arg_matches(matches)?;

        let response = if let Some(ref vault_path) = request.vault_path {
            let config = Arc::new(Config::load_from_base_path(vault_path.as_path()));
            let capability = SearchCapability::new(vault_path.clone(), config);
            capability
                .build_index(
                    request.reindex.unwrap_or(false),
                    request.model.clone(),
                    request.dim,
                )
                .await?
        } else {
            self.capability
                .build_index(request.reindex.unwrap_or(false), request.model, request.dim)
                .await?
        };

        Ok(serde_json::to_string_pretty(&response)?)
    }

    fn input_schema(&self) -> serde_json::Value {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(IndexRequest)).unwrap()
    }

    // Build JSON field-by-field instead of routing through IndexRequest::from_arg_matches,
    // which would panic on a missing vault_path arg id when called from get_remote_command.
    fn args_to_json(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut obj = serde_json::Map::new();
        if let Some(v) = matches.get_one::<bool>("reindex") {
            obj.insert("reindex".into(), serde_json::Value::Bool(*v));
        }
        if let Some(v) = matches.get_one::<String>("model") {
            obj.insert("model".into(), serde_json::Value::String(v.clone()));
        }
        if let Some(v) = matches.get_one::<u32>("dim") {
            obj.insert("dim".into(), serde_json::json!(v));
        }
        Ok(serde_json::Value::Object(obj))
    }
}

#[async_trait::async_trait]
impl notectl_core::operation::Operation for SearchOperation {
    fn name(&self) -> &'static str {
        search::CLI_NAME
    }

    fn path(&self) -> &'static str {
        search::HTTP_PATH
    }

    fn description(&self) -> &'static str {
        search::DESCRIPTION
    }

    fn get_command(&self) -> clap::Command {
        SearchRequest::command()
    }

    // NOTE: This same panic-on-missing-arg risk applies to every other capability file
    // (notectl-outline, notectl-tags, notectl-files, notectl-daily-notes, notectl-tasks)
    // since they all follow the identical pattern of omitting vault_path from
    // get_remote_command while args_to_json routes through Request::from_arg_matches.
    // See TASK-14 for details. A follow-up ticket should fix those systematically.
    fn get_remote_command(&self) -> clap::Command {
        // Rebuild without the vault_path positional; shift query to index 1.
        clap::Command::new("search")
            .about("Search across indexed notes")
            .arg(
                clap::Arg::new("query")
                    .index(1)
                    .required(true)
                    .help("Search query"),
            )
            .arg(
                clap::Arg::new("limit")
                    .long("limit")
                    .value_parser(clap::value_parser!(usize))
                    .default_value("50")
                    .help("Maximum number of results"),
            )
            .arg(
                clap::Arg::new("mode")
                    .long("mode")
                    .value_parser(["hybrid", "dense", "sparse"])
                    .default_value("hybrid")
                    .help("Search mode"),
            )
            .arg(
                clap::Arg::new("no_reindex")
                    .long("no-reindex")
                    .value_parser(clap::value_parser!(bool))
                    .help("Skip reindexing"),
            )
    }

    async fn execute_json(
        &self,
        json: serde_json::Value,
    ) -> Result<serde_json::Value, rmcp::model::ErrorData> {
        let request: SearchRequest = serde_json::from_value(json)
            .map_err(|e| notectl_core::error::invalid_params(e.to_string()))?;
        let response = self
            .capability
            .do_search(
                &request.query,
                request.limit.unwrap_or(50),
                request.mode.unwrap_or_default(),
                request.no_reindex.unwrap_or(false),
            )
            .await?;
        Ok(serde_json::to_value(response).unwrap())
    }

    async fn execute_from_args(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let request = SearchRequest::from_arg_matches(matches)?;

        let response = if let Some(ref vault_path) = request.vault_path {
            let config = Arc::new(Config::load_from_base_path(vault_path.as_path()));
            let capability = SearchCapability::new(vault_path.clone(), config);
            capability
                .do_search(
                    &request.query,
                    request.limit.unwrap_or(50),
                    request.mode.unwrap_or_default(),
                    request.no_reindex.unwrap_or(false),
                )
                .await?
        } else {
            self.capability
                .do_search(
                    &request.query,
                    request.limit.unwrap_or(50),
                    request.mode.unwrap_or_default(),
                    request.no_reindex.unwrap_or(false),
                )
                .await?
        };

        Ok(serde_json::to_string_pretty(&response)?)
    }

    fn input_schema(&self) -> serde_json::Value {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(SearchRequest)).unwrap()
    }

    // Build JSON field-by-field instead of routing through SearchRequest::from_arg_matches,
    // which would panic on a missing vault_path arg id when called from get_remote_command.
    fn args_to_json(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut obj = serde_json::Map::new();
        if let Some(v) = matches.get_one::<String>("query") {
            obj.insert("query".into(), serde_json::Value::String(v.clone()));
        }
        if let Some(v) = matches.get_one::<usize>("limit") {
            obj.insert("limit".into(), serde_json::json!(v));
        }
        if let Some(v) = matches.get_one::<String>("mode") {
            obj.insert("mode".into(), serde_json::Value::String(v.clone()));
        }
        if let Some(v) = matches.get_one::<bool>("no_reindex") {
            obj.insert("no_reindex".into(), serde_json::Value::Bool(*v));
        }
        Ok(serde_json::Value::Object(obj))
    }
}

// ---------------------------------------------------------------------------
// Tests for get_remote_command() grammar consistency (TASK-14)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod remote_command_tests {
    use super::*;
    use notectl_core::operation::Operation;

    /// Create a dummy SearchCapability for testing (base_path doesn't matter for these tests).
    fn dummy_capability() -> Arc<SearchCapability> {
        Arc::new(SearchCapability::new(
            PathBuf::from("/tmp"),
            Arc::new(Config::default()),
        ))
    }

    // -- IndexOperation tests --

    #[test]
    fn index_remote_command_reindex_accepts_bool_value() {
        let op = IndexOperation::new(dummy_capability());
        let cmd = op.get_remote_command();

        // --reindex true should succeed (value_parser bool expects a value)
        let matches = cmd
            .clone()
            .try_get_matches_from(["index", "--reindex", "true"])
            .unwrap();
        assert_eq!(matches.get_one::<bool>("reindex").copied(), Some(true));

        // --reindex false should also succeed
        let matches = cmd
            .try_get_matches_from(["index", "--reindex", "false"])
            .unwrap();
        assert_eq!(matches.get_one::<bool>("reindex").copied(), Some(false));
    }

    #[test]
    fn index_remote_command_reindex_bare_flag_fails() {
        let op = IndexOperation::new(dummy_capability());
        let cmd = op.get_remote_command();

        // Bare --reindex without a value should fail (it's not SetTrue)
        let result = cmd.try_get_matches_from(["index", "--reindex"]);
        assert!(result.is_err());
    }

    #[test]
    fn index_remote_command_args_to_json_no_vault_path_panic() {
        let op = IndexOperation::new(dummy_capability());
        let cmd = op.get_remote_command();

        // Parse without vault_path — this must NOT panic
        let matches = cmd
            .try_get_matches_from(["index", "--reindex", "true"])
            .unwrap();
        let json = op
            .args_to_json(&matches)
            .expect("args_to_json must not panic");

        // Verify the JSON contains expected fields
        assert!(json.get("reindex").is_some());
        assert_eq!(json["reindex"], true);
    }

    // -- SearchOperation tests --

    #[test]
    fn search_remote_command_no_reindex_accepts_bool_value() {
        let op = SearchOperation::new(dummy_capability());
        let cmd = op.get_remote_command();

        // --no-reindex true should succeed
        let matches = cmd
            .clone()
            .try_get_matches_from(["search", "hello", "--no-reindex", "true"])
            .unwrap();
        assert_eq!(matches.get_one::<bool>("no_reindex").copied(), Some(true));

        // --no-reindex false should also succeed
        let matches = cmd
            .try_get_matches_from(["search", "hello", "--no-reindex", "false"])
            .unwrap();
        assert_eq!(matches.get_one::<bool>("no_reindex").copied(), Some(false));
    }

    #[test]
    fn search_remote_command_no_reindex_bare_flag_fails() {
        let op = SearchOperation::new(dummy_capability());
        let cmd = op.get_remote_command();

        // Bare --no-reindex without a value should fail
        let result = cmd.try_get_matches_from(["search", "hello", "--no-reindex"]);
        assert!(result.is_err());
    }

    #[test]
    fn search_remote_command_args_to_json_no_vault_path_panic() {
        let op = SearchOperation::new(dummy_capability());
        let cmd = op.get_remote_command();

        // Parse without vault_path — this must NOT panic
        let matches = cmd.try_get_matches_from(["search", "my query"]).unwrap();
        let json = op
            .args_to_json(&matches)
            .expect("args_to_json must not panic");

        // Verify the JSON contains expected fields
        assert!(json.get("query").is_some());
        assert_eq!(json["query"], "my query");
    }

    #[test]
    fn search_remote_command_with_all_options() {
        let op = SearchOperation::new(dummy_capability());
        let cmd = op.get_remote_command();

        let matches = cmd
            .try_get_matches_from([
                "search",
                "test query",
                "--limit",
                "10",
                "--mode",
                "dense",
                "--no-reindex",
                "true",
            ])
            .unwrap();
        let json = op
            .args_to_json(&matches)
            .expect("args_to_json must not panic");

        assert_eq!(json["query"], "test query");
        assert_eq!(json["limit"], 10);
        assert_eq!(json["mode"], "dense");
        assert_eq!(json["no_reindex"], true);
    }
}

// ---------------------------------------------------------------------------
// End-to-end tests for SearchCapability::build_index --reindex path (TASK-20)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod build_index_tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Build index with --reindex removes artifacts, rebuilds them, and preserves models/.
    #[tokio::test]
    async fn build_index_reindex_removes_and_rebuilds_artifacts_preserves_models() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();
        // Write content long enough to produce chunks (min_chunk_tokens default is 32).
        fs::write(
            base.join("hello.md"),
            "# Hello\n\nThis is a longer note with enough content to produce chunks from the chunker pipeline. It has several sentences of text that should be sufficient for the chunker to generate output. Additional paragraphs ensure we exceed the minimum token threshold required by the default configuration settings.",
        )
        .unwrap();

        let cap = SearchCapability::new(base.clone(), Arc::new(Config::default()));
        let config = Config::default();
        let index_dir = config.search.resolve_index_dir(&base);

        // Initial build (non-reindex) to create real index artifacts.
        let resp = cap.build_index(false, None, None).await.unwrap();
        assert!(!resp.content_hash.is_empty());
        assert!(resp.files_indexed >= 1);
        assert!(
            resp.chunks_produced >= 1,
            "initial build must produce chunks"
        );

        // Verify artifacts exist before reindex.
        assert!(index_dir.join("manifest.json").exists());
        assert!(index_dir.join("chunks").is_dir());

        // Create a models/ directory with a placeholder file under the index dir.
        let models_dir = index_dir.join("models");
        fs::create_dir_all(&models_dir).unwrap();
        fs::write(models_dir.join("model.bin"), b"model data").unwrap();

        // Rebuild with --reindex.
        let result = cap.build_index(true, None, None).await;
        assert!(
            result.is_ok(),
            "build_index(reindex=true) should succeed: {:?}",
            result
        );

        // Assert models/ still exists after reindex.
        assert!(
            models_dir.join("model.bin").exists(),
            "models/model.bin must survive --reindex"
        );

        // Assert index was rebuilt (artifacts recreated).
        assert!(
            index_dir.join("manifest.json").exists(),
            "manifest.json must be rebuilt after --reindex"
        );
        assert!(
            index_dir.join("chunks").is_dir(),
            "chunks/ must be rebuilt after --reindex"
        );

        // Assert response shows valid index state.
        let resp = result.unwrap();
        assert!(
            !resp.content_hash.is_empty(),
            "content_hash must not be empty"
        );
        assert!(resp.files_indexed >= 1, "files_indexed must be >= 1");
    }

    /// --reindex on a vault with no existing index succeeds (covers the exists() false branch).
    #[tokio::test]
    async fn build_index_reindex_when_no_existing_index_succeeds() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join("vault");
        fs::create_dir_all(&base).unwrap();
        // Write content long enough to produce chunks (min_chunk_tokens default is 32).
        fs::write(
            base.join("hello.md"),
            "# Hello\n\nThis is a longer note with enough content to produce chunks from the chunker pipeline. It has several sentences of text that should be sufficient for the chunker to generate output. Additional paragraphs ensure we exceed the minimum token threshold required by the default configuration settings.",
        )
        .unwrap();

        let cap = SearchCapability::new(base.clone(), Arc::new(Config::default()));

        // Call build_index with reindex=true on a fresh vault (no prior index).
        let result = cap.build_index(true, None, None).await;
        assert!(
            result.is_ok(),
            "build_index(reindex=true) on fresh vault should succeed: {:?}",
            result
        );

        // The index should have been built successfully.
        let resp = result.unwrap();
        assert!(!resp.content_hash.is_empty());
        assert!(resp.files_indexed >= 1);
    }
}
