use crate::capabilities::CapabilityRegistry;
use notectl_core::config::Config;
use notectl_daily_notes::{
    GetDailyNoteRequest, GetDailyNoteResponse, SearchDailyNotesRequest, SearchDailyNotesResponse,
};
use notectl_files::{
    ListFilesRequest, ListFilesResponse, ReadFilesRequest, ReadFilesResponse, RecentFilesRequest,
    RecentFilesResponse,
};
use notectl_tags::{
    ExtractTagsRequest, ExtractTagsResponse, ListTagsRequest, ListTagsResponse,
    SearchByTagsRequest, SearchByTagsResponse,
};
use notectl_tasks::{SearchTasksRequest, TaskSearchResponse};
#[cfg(feature = "search")]
use rmcp::handler::server::router::tool::{AsyncTool, ToolBase};
use rmcp::{
    ServerHandler,
    handler::server::{
        router::tool::ToolRouter,
        wrapper::{Json, Parameters},
    },
    model::*,
    tool, tool_handler, tool_router,
};
#[cfg(feature = "search")]
use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::Arc;

/// MCP Service for task searching and tag extraction
#[derive(Clone)]
#[allow(dead_code)]
pub struct TaskSearchService {
    tool_router: ToolRouter<TaskSearchService>,
    capability_registry: Arc<CapabilityRegistry>,
}

#[tool_router]
impl TaskSearchService {
    pub fn new(base_path: PathBuf) -> Self {
        // Load configuration from base path
        let config = Arc::new(Config::load_from_base_path(&base_path));

        // Create capability registry
        let capability_registry = Arc::new(CapabilityRegistry::new(
            base_path.clone(),
            Arc::clone(&config),
        ));

        // Build tool router starting with macro-generated routes
        #[cfg(not(feature = "search"))]
        let tool_router = Self::tool_router();

        #[cfg(feature = "search")]
        let tool_router = Self::tool_router()
            .with_async_tool::<search_tools::SearchTool>()
            .with_async_tool::<search_tools::IndexTool>();

        Self {
            tool_router,
            capability_registry,
        }
    }

    #[tool(
        description = "Search for tasks in Markdown files with optional filtering by status, dates, and tags"
    )]
    async fn search_tasks(
        &self,
        Parameters(request): Parameters<SearchTasksRequest>,
    ) -> Result<Json<TaskSearchResponse>, ErrorData> {
        // Delegate to TaskCapability
        let response = self
            .capability_registry
            .tasks()
            .search_tasks(request)
            .await?;

        Ok(Json(response))
    }

    #[tool(description = "Extract all unique tags from YAML frontmatter in Markdown files")]
    async fn extract_tags(
        &self,
        Parameters(request): Parameters<ExtractTagsRequest>,
    ) -> Result<Json<ExtractTagsResponse>, ErrorData> {
        // Delegate to TagCapability
        let response = self
            .capability_registry
            .tags()
            .extract_tags(request)
            .await?;

        Ok(Json(response))
    }

    #[tool(
        description = "List all tags in the vault with document counts. Returns tags sorted by frequency (most common first). Useful for understanding the tag taxonomy, finding popular topics, and discovering content organization patterns."
    )]
    async fn list_tags(
        &self,
        Parameters(request): Parameters<ListTagsRequest>,
    ) -> Result<Json<ListTagsResponse>, ErrorData> {
        // Delegate to TagCapability
        let response = self.capability_registry.tags().list_tags(request).await?;

        Ok(Json(response))
    }

    #[tool(
        description = "Search for files by YAML frontmatter tags with AND/OR matching. Returns files that match the specified tags."
    )]
    async fn search_by_tags(
        &self,
        Parameters(request): Parameters<SearchByTagsRequest>,
    ) -> Result<Json<SearchByTagsResponse>, ErrorData> {
        // Delegate to TagCapability
        let response = self
            .capability_registry
            .tags()
            .search_by_tags(request)
            .await?;

        Ok(Json(response))
    }

    #[tool(
        description = "List the directory tree of the vault. Returns a hierarchical view of all files and folders. Useful for understanding vault structure and finding files."
    )]
    async fn list_files(
        &self,
        Parameters(request): Parameters<ListFilesRequest>,
    ) -> Result<Json<ListFilesResponse>, ErrorData> {
        // Delegate to FileCapability
        let response = self.capability_registry.files().list_files(request).await?;

        Ok(Json(response))
    }

    #[tool(description = "Read one or more markdown files from the vault")]
    async fn read_files(
        &self,
        Parameters(request): Parameters<ReadFilesRequest>,
    ) -> Result<Json<ReadFilesResponse>, ErrorData> {
        // Delegate to FileCapability
        let response = self.capability_registry.files().read_files(request).await?;

        Ok(Json(response))
    }

    #[tool(
        description = "List recently modified markdown files, sorted by modification time descending. Checks the frontmatter `updated` field first; falls back to filesystem mtime."
    )]
    async fn recent_files(
        &self,
        Parameters(request): Parameters<RecentFilesRequest>,
    ) -> Result<Json<RecentFilesResponse>, ErrorData> {
        let response = self
            .capability_registry
            .files()
            .recent_files(request)
            .await?;
        Ok(Json(response))
    }

    #[tool(
        description = "Get the content of a daily note for a specific date. Returns the note content, file path, and whether the note was found. Missing notes return found: false (not an error)."
    )]
    async fn get_daily_note(
        &self,
        Parameters(request): Parameters<GetDailyNoteRequest>,
    ) -> Result<Json<GetDailyNoteResponse>, ErrorData> {
        // Delegate to DailyNoteCapability
        let response = self
            .capability_registry
            .daily_notes()
            .get_daily_note(request)
            .await?;

        Ok(Json(response))
    }

    #[tool(
        description = "Search for daily notes within a date range. Returns metadata for all matching notes. Use get_daily_note to retrieve full content for specific notes."
    )]
    async fn search_daily_notes(
        &self,
        Parameters(request): Parameters<SearchDailyNotesRequest>,
    ) -> Result<Json<SearchDailyNotesResponse>, ErrorData> {
        // Delegate to DailyNoteCapability
        let response = self
            .capability_registry
            .daily_notes()
            .search_daily_notes(request)
            .await?;

        Ok(Json(response))
    }
}

// ---------------------------------------------------------------------------
// Trait-based search tools (conditionally compiled, added manually to router)
// ---------------------------------------------------------------------------

#[cfg(feature = "search")]
mod search_tools {
    use super::*;
    use notectl_search::{IndexResponse, SearchMode, SearchResponse};
    use schemars::JsonSchema;
    use serde::Deserialize;

    // --- search tool ---

    #[derive(Debug, Deserialize, JsonSchema, Default)]
    pub struct McpSearchParams {
        /// The text to search for
        pub query: String,
        /// Maximum number of results to return (default 50)
        pub limit: Option<usize>,
        /// Scoring mode: hybrid (dense+sparse fused), dense only, or sparse only (default hybrid)
        pub mode: Option<SearchMode>,
        /// If true, skip the staleness check and use the existing index without rebuilding
        pub no_reindex: Option<bool>,
    }

    pub struct SearchTool;

    impl ToolBase for SearchTool {
        type Parameter = McpSearchParams;
        type Output = SearchResponse;
        type Error = ErrorData;

        fn name() -> Cow<'static, str> {
            "search".into()
        }

        fn description() -> Option<Cow<'static, str>> {
            Some(
                "Search across all indexed notes using hybrid (dense + sparse), dense-only, or sparse-only scoring. Auto-degrades when vectors are unavailable.".into(),
            )
        }
    }

    impl AsyncTool<TaskSearchService> for SearchTool {
        async fn invoke(
            service: &TaskSearchService,
            params: Self::Parameter,
        ) -> Result<Self::Output, Self::Error> {
            let response = service
                .capability_registry
                .search()
                .do_search(
                    &params.query,
                    params.limit.unwrap_or(50),
                    params.mode.unwrap_or_default(),
                    params.no_reindex.unwrap_or(false),
                )
                .await?;
            Ok(response)
        }
    }

    // --- build_search_index tool ---

    #[derive(Debug, Deserialize, JsonSchema, Default)]
    pub struct McpIndexParams {
        /// If true, delete existing index artifacts and rebuild from scratch
        pub reindex: Option<bool>,
        /// Override the embedding model ID from config
        pub model: Option<String>,
        /// Override the embedding dimension from config
        pub dim: Option<u32>,
    }

    pub struct IndexTool;

    impl ToolBase for IndexTool {
        type Parameter = McpIndexParams;
        type Output = IndexResponse;
        type Error = ErrorData;

        fn name() -> Cow<'static, str> {
            "build_search_index".into()
        }

        fn description() -> Option<Cow<'static, str>> {
            Some(
                "Build or update the search index for all markdown files in the vault. Computes chunks, optional embeddings, and persists index artifacts.".into(),
            )
        }
    }

    impl AsyncTool<TaskSearchService> for IndexTool {
        async fn invoke(
            service: &TaskSearchService,
            params: Self::Parameter,
        ) -> Result<Self::Output, Self::Error> {
            let response = service
                .capability_registry
                .search()
                .build_index(params.reindex.unwrap_or(false), params.model, params.dim)
                .await?;
            Ok(response)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// McpSearchParams with an empty JSON object (query omitted) must fail
        /// deserialization, matching SearchRequest's required-query behavior on
        /// the HTTP/CLI path.
        #[test]
        fn mcp_search_params_rejects_missing_query() {
            let result = serde_json::from_value::<McpSearchParams>(serde_json::json!({}));
            assert!(
                result.is_err(),
                "deserializing McpSearchParams without 'query' must fail"
            );
            let err = result.unwrap_err();
            assert!(
                err.to_string().contains("query"),
                "error must mention the missing 'query' field: {}",
                err
            );
        }

        /// McpSearchParams with query present succeeds.
        #[test]
        fn mcp_search_params_accepts_query() {
            let result = serde_json::from_value::<McpSearchParams>(serde_json::json!({
                "query": "hello"
            }));
            assert!(result.is_ok());
            assert_eq!(result.unwrap().query, "hello");
        }
    }
}

#[tool_handler]
impl ServerHandler for TaskSearchService {
    fn get_info(&self) -> ServerInfo {
        // Build instructions from capability metadata
        #[cfg(not(feature = "search"))]
        let instructions: String = [
            "A Markdown task extraction service. Available operations:",
            &format!("- {}", notectl_tasks::capability::search_tasks::DESCRIPTION),
            &format!("- {}", notectl_tags::extract_tags::DESCRIPTION),
            &format!("- {}", notectl_tags::list_tags::DESCRIPTION),
            &format!("- {}", notectl_tags::search_by_tags::DESCRIPTION),
            &format!("- {}", notectl_files::list_files::DESCRIPTION),
            &format!("- {}", notectl_files::read_files::DESCRIPTION),
            &format!("- {}", notectl_daily_notes::get_daily_note::DESCRIPTION),
            &format!("- {}", notectl_daily_notes::search_daily_notes::DESCRIPTION),
        ]
        .join("\n");

        #[cfg(feature = "search")]
        let instructions: String = [
            "A Markdown task extraction service. Available operations:",
            &format!("- {}", notectl_tasks::capability::search_tasks::DESCRIPTION),
            &format!("- {}", notectl_tags::extract_tags::DESCRIPTION),
            &format!("- {}", notectl_tags::list_tags::DESCRIPTION),
            &format!("- {}", notectl_tags::search_by_tags::DESCRIPTION),
            &format!("- {}", notectl_files::list_files::DESCRIPTION),
            &format!("- {}", notectl_files::read_files::DESCRIPTION),
            &format!("- {}", notectl_daily_notes::get_daily_note::DESCRIPTION),
            &format!("- {}", notectl_daily_notes::search_daily_notes::DESCRIPTION),
            &format!("- {}", notectl_search::capability::index::DESCRIPTION),
            &format!("- {}", notectl_search::capability::search::DESCRIPTION),
        ]
        .join("\n");

        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(instructions)
    }
}
