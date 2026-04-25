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
use rmcp::{
    ServerHandler,
    handler::server::{
        router::tool::ToolRouter,
        wrapper::{Json, Parameters},
    },
    model::*,
    tool, tool_handler, tool_router,
};
use std::path::PathBuf;
use std::sync::Arc;

/// MCP Service for task searching and tag extraction
#[derive(Clone)]
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

        Self {
            tool_router: Self::tool_router(),
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

#[tool_handler]
impl ServerHandler for TaskSearchService {
    fn get_info(&self) -> ServerInfo {
        // Build instructions from capability metadata
        let instructions = format!(
            "A Markdown task extraction service. Available operations:\n\
             - {}\n\
             - {}\n\
             - {}\n\
             - {}\n\
             - {}\n\
             - {}\n\
             - {}\n\
             - {}",
            notectl_tasks::capability::search_tasks::DESCRIPTION,
            notectl_tags::extract_tags::DESCRIPTION,
            notectl_tags::list_tags::DESCRIPTION,
            notectl_tags::search_by_tags::DESCRIPTION,
            notectl_files::list_files::DESCRIPTION,
            notectl_files::read_files::DESCRIPTION,
            notectl_daily_notes::get_daily_note::DESCRIPTION,
            notectl_daily_notes::search_daily_notes::DESCRIPTION
        );

        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation::from_build_env(),
            instructions: Some(instructions),
        }
    }
}
