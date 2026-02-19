use crate::tag_extractor::{TagCount, TagExtractor, TaggedFile};
use clap::{CommandFactory, FromArgMatches};
use markdown_todo_extractor_core::CapabilityResult;
use markdown_todo_extractor_core::config::Config;
use markdown_todo_extractor_core::error::internal_error;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

/// Operation metadata for extract_tags
pub mod extract_tags {
    pub const DESCRIPTION: &str = "Extract all unique tags from YAML frontmatter in Markdown files";
    #[allow(dead_code)]
    pub const CLI_NAME: &str = "tags";
    pub const HTTP_PATH: &str = "/api/tags";
}

/// Parameters for the extract_tags operation
#[derive(Debug, Deserialize, Serialize, JsonSchema, clap::Parser)]
#[command(name = "tags", about = "Extract all unique tags from YAML frontmatter")]
pub struct ExtractTagsRequest {
    /// Path to scan (CLI only - not used in HTTP/MCP)
    #[arg(index = 1, required = true, help = "Path to file or folder to scan")]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub path: Option<PathBuf>,

    #[arg(long, help = "Subpath within the directory to search")]
    #[schemars(
        description = "Subpath within the base directory to search (optional, defaults to base path)"
    )]
    pub subpath: Option<String>,
}

/// Response from the extract_tags operation
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ExtractTagsResponse {
    pub tags: Vec<String>,
}

/// Operation metadata for list_tags
pub mod list_tags {
    pub const DESCRIPTION: &str = "List all tags in the vault with document counts. Returns tags sorted by frequency (most common first). Useful for understanding the tag taxonomy, finding popular topics, and discovering content organization patterns.";
    #[allow(dead_code)]
    pub const CLI_NAME: &str = "list-tags";
    pub const HTTP_PATH: &str = "/api/tags/list";
}

/// Parameters for the list_tags operation
#[derive(Debug, Deserialize, Serialize, JsonSchema, clap::Parser)]
#[command(name = "list-tags", about = "List all tags with document counts")]
pub struct ListTagsRequest {
    /// Path to scan (CLI only - not used in HTTP/MCP)
    #[arg(index = 1, required = true, help = "Path to file or folder to scan")]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub path: Option<PathBuf>,

    #[arg(long, help = "Subpath within the vault to search")]
    #[schemars(
        description = "Subpath within the vault to search (optional, defaults to entire vault)"
    )]
    pub subpath: Option<String>,

    #[arg(long, help = "Minimum document count to include a tag")]
    #[schemars(description = "Minimum document count to include a tag (optional, defaults to 1)")]
    pub min_count: Option<usize>,

    #[arg(long, help = "Maximum number of tags to return")]
    #[schemars(description = "Maximum number of tags to return (optional, defaults to all)")]
    pub limit: Option<usize>,
}

/// Response from the list_tags operation
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ListTagsResponse {
    /// List of tags with their document counts
    pub tags: Vec<TagCount>,
    /// Total number of unique tags found (before filtering/limiting)
    pub total_unique_tags: usize,
    /// Whether the results were truncated due to limit parameter
    pub truncated: bool,
}

/// Operation metadata for search_by_tags
pub mod search_by_tags {
    pub const DESCRIPTION: &str = "Search for files by YAML frontmatter tags with AND/OR matching. Returns files that match the specified tags.";
    #[allow(dead_code)]
    pub const CLI_NAME: &str = "search-tags";
    pub const HTTP_PATH: &str = "/api/tags/search";
}

/// Parameters for the search_by_tags operation
#[derive(Debug, Deserialize, Serialize, JsonSchema, clap::Parser)]
#[command(
    name = "search-tags",
    about = "Search for files by YAML frontmatter tags"
)]
pub struct SearchByTagsRequest {
    /// Path to scan (CLI only - not used in HTTP/MCP)
    #[arg(index = 1, required = true, help = "Path to file or folder to scan")]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub path: Option<PathBuf>,

    #[arg(long, value_delimiter = ',', help = "Tags to search for")]
    #[schemars(description = "Tags to search for")]
    pub tags: Vec<String>,

    #[arg(
        long,
        help = "File must have ALL tags (AND logic). Default: false (OR logic)"
    )]
    #[schemars(
        description = "If true, file must have ALL tags (AND logic). If false, file must have ANY tag (OR logic). Default: false"
    )]
    pub match_all: Option<bool>,

    #[arg(long, help = "Subpath within the directory to search")]
    #[schemars(description = "Subpath within the base directory to search (optional)")]
    pub subpath: Option<String>,

    #[arg(long, help = "Limit the number of files returned")]
    #[schemars(description = "Limit the number of files returned")]
    pub limit: Option<usize>,
}

/// Response from the search_by_tags operation
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SearchByTagsResponse {
    pub files: Vec<TaggedFile>,
    pub total_count: usize,
}

/// Capability for tag operations (extract, list, search)
pub struct TagCapability {
    base_path: PathBuf,
    tag_extractor: Arc<TagExtractor>,
}

impl TagCapability {
    /// Create a new TagCapability
    pub fn new(base_path: PathBuf, config: Arc<Config>) -> Self {
        Self {
            base_path,
            tag_extractor: Arc::new(TagExtractor::new(config)),
        }
    }

    /// Extract all unique tags from YAML frontmatter
    pub async fn extract_tags(
        &self,
        request: ExtractTagsRequest,
    ) -> CapabilityResult<ExtractTagsResponse> {
        // Determine the search path (base path + optional subpath)
        let search_path = if let Some(subpath) = request.subpath {
            self.base_path.join(subpath)
        } else {
            self.base_path.clone()
        };

        // Extract tags from the search path
        let tags = self
            .tag_extractor
            .extract_tags(&search_path)
            .map_err(|e| internal_error(format!("Failed to extract tags: {}", e)))?;

        Ok(ExtractTagsResponse { tags })
    }

    /// List all tags with document counts
    pub async fn list_tags(&self, request: ListTagsRequest) -> CapabilityResult<ListTagsResponse> {
        // Resolve search path
        let search_path = if let Some(ref subpath) = request.subpath {
            self.base_path.join(subpath)
        } else {
            self.base_path.clone()
        };

        // Extract tags with counts
        let mut tags = self
            .tag_extractor
            .extract_tags_with_counts(&search_path)
            .map_err(|e| internal_error(format!("Failed to extract tags: {}", e)))?;

        // Track total before filtering
        let total_unique_tags = tags.len();

        // Filter by min_count if specified
        if let Some(min_count) = request.min_count {
            tags.retain(|t| t.document_count >= min_count);
        }

        // Apply limit if specified
        let truncated = if let Some(limit) = request.limit {
            if tags.len() > limit {
                tags.truncate(limit);
                true
            } else {
                false
            }
        } else {
            false
        };

        Ok(ListTagsResponse {
            tags,
            total_unique_tags,
            truncated,
        })
    }

    /// Search for files by YAML frontmatter tags
    pub async fn search_by_tags(
        &self,
        request: SearchByTagsRequest,
    ) -> CapabilityResult<SearchByTagsResponse> {
        // Determine the search path (base path + optional subpath)
        let search_path = if let Some(ref subpath) = request.subpath {
            self.base_path.join(subpath)
        } else {
            self.base_path.clone()
        };

        let match_all = request.match_all.unwrap_or(false);

        // Search for files by tags
        let mut files = self
            .tag_extractor
            .search_by_tags(&search_path, &request.tags, match_all)
            .map_err(|e| internal_error(format!("Failed to search by tags: {}", e)))?;

        let total_count = files.len();

        // Apply limit if specified
        if let Some(limit) = request.limit {
            files.truncate(limit);
        }

        Ok(SearchByTagsResponse { files, total_count })
    }
}

/// Operation struct for extract_tags (HTTP, CLI, and MCP)
pub struct ExtractTagsOperation {
    capability: Arc<TagCapability>,
}

impl ExtractTagsOperation {
    pub fn new(capability: Arc<TagCapability>) -> Self {
        Self { capability }
    }
}

/// Operation struct for list_tags (HTTP, CLI, and MCP)
pub struct ListTagsOperation {
    capability: Arc<TagCapability>,
}

impl ListTagsOperation {
    pub fn new(capability: Arc<TagCapability>) -> Self {
        Self { capability }
    }
}

/// Operation struct for search_by_tags (HTTP, CLI, and MCP)
pub struct SearchByTagsOperation {
    capability: Arc<TagCapability>,
}

impl SearchByTagsOperation {
    pub fn new(capability: Arc<TagCapability>) -> Self {
        Self { capability }
    }
}

#[async_trait::async_trait]
impl markdown_todo_extractor_core::operation::Operation for ExtractTagsOperation {
    fn name(&self) -> &'static str {
        extract_tags::CLI_NAME
    }

    fn path(&self) -> &'static str {
        extract_tags::HTTP_PATH
    }

    fn description(&self) -> &'static str {
        extract_tags::DESCRIPTION
    }

    fn get_command(&self) -> clap::Command {
        // Get command from request struct's Parser derive
        ExtractTagsRequest::command()
    }

    async fn execute_json(
        &self,
        json: serde_json::Value,
    ) -> Result<serde_json::Value, rmcp::model::ErrorData> {
        let request: ExtractTagsRequest = serde_json::from_value(json)
            .map_err(|e| markdown_todo_extractor_core::error::invalid_params(e.to_string()))?;
        let response = self.capability.extract_tags(request).await?;
        Ok(serde_json::to_value(response).unwrap())
    }

    async fn execute_from_args(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Parse request from ArgMatches
        let request = ExtractTagsRequest::from_arg_matches(matches)?;

        // Handle CLI-specific path if present
        let response = if let Some(ref path) = request.path {
            let config = Arc::new(Config::load_from_base_path(path.as_path()));
            let capability = TagCapability::new(path.clone(), config);
            let mut req_without_path = request;
            req_without_path.path = None;
            capability.extract_tags(req_without_path).await?
        } else {
            self.capability.extract_tags(request).await?
        };

        // Serialize to JSON
        Ok(serde_json::to_string_pretty(&response)?)
    }

    fn input_schema(&self) -> serde_json::Value {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(ExtractTagsRequest)).unwrap()
    }
}

#[async_trait::async_trait]
impl markdown_todo_extractor_core::operation::Operation for ListTagsOperation {
    fn name(&self) -> &'static str {
        list_tags::CLI_NAME
    }

    fn path(&self) -> &'static str {
        list_tags::HTTP_PATH
    }

    fn description(&self) -> &'static str {
        list_tags::DESCRIPTION
    }

    fn get_command(&self) -> clap::Command {
        // Get command from request struct's Parser derive
        ListTagsRequest::command()
    }

    async fn execute_json(
        &self,
        json: serde_json::Value,
    ) -> Result<serde_json::Value, rmcp::model::ErrorData> {
        let request: ListTagsRequest = serde_json::from_value(json)
            .map_err(|e| markdown_todo_extractor_core::error::invalid_params(e.to_string()))?;
        let response = self.capability.list_tags(request).await?;
        Ok(serde_json::to_value(response).unwrap())
    }

    async fn execute_from_args(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Parse request from ArgMatches
        let request = ListTagsRequest::from_arg_matches(matches)?;

        // Handle CLI-specific path if present
        let response = if let Some(ref path) = request.path {
            let config = Arc::new(Config::load_from_base_path(path.as_path()));
            let capability = TagCapability::new(path.clone(), config);
            let mut req_without_path = request;
            req_without_path.path = None;
            capability.list_tags(req_without_path).await?
        } else {
            self.capability.list_tags(request).await?
        };

        // Serialize to JSON
        Ok(serde_json::to_string_pretty(&response)?)
    }

    fn input_schema(&self) -> serde_json::Value {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(ListTagsRequest)).unwrap()
    }
}

#[async_trait::async_trait]
impl markdown_todo_extractor_core::operation::Operation for SearchByTagsOperation {
    fn name(&self) -> &'static str {
        search_by_tags::CLI_NAME
    }

    fn path(&self) -> &'static str {
        search_by_tags::HTTP_PATH
    }

    fn description(&self) -> &'static str {
        search_by_tags::DESCRIPTION
    }

    fn get_command(&self) -> clap::Command {
        // Get command from request struct's Parser derive
        SearchByTagsRequest::command()
    }

    async fn execute_json(
        &self,
        json: serde_json::Value,
    ) -> Result<serde_json::Value, rmcp::model::ErrorData> {
        let request: SearchByTagsRequest = serde_json::from_value(json)
            .map_err(|e| markdown_todo_extractor_core::error::invalid_params(e.to_string()))?;
        let response = self.capability.search_by_tags(request).await?;
        Ok(serde_json::to_value(response).unwrap())
    }

    async fn execute_from_args(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Parse request from ArgMatches
        let request = SearchByTagsRequest::from_arg_matches(matches)?;

        // Handle CLI-specific path if present
        let response = if let Some(ref path) = request.path {
            let config = Arc::new(Config::load_from_base_path(path.as_path()));
            let capability = TagCapability::new(path.clone(), config);
            let mut req_without_path = request;
            req_without_path.path = None;
            capability.search_by_tags(req_without_path).await?
        } else {
            self.capability.search_by_tags(request).await?
        };

        // Serialize to JSON
        Ok(serde_json::to_string_pretty(&response)?)
    }

    fn input_schema(&self) -> serde_json::Value {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(SearchByTagsRequest)).unwrap()
    }
}
