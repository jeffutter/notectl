use crate::outline_extractor::{Heading, HeadingMatch, OutlineExtractor, Section};
use clap::{CommandFactory, FromArgMatches};
use notectl_core::CapabilityResult;
use notectl_core::config::Config;
use notectl_core::error::{internal_error, invalid_params};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

/// Operation metadata for get_outline
pub mod get_outline {
    pub const DESCRIPTION: &str = "Extract heading hierarchy from a markdown file. Returns a list of headings with their levels and line numbers. Can return flat list or hierarchical tree structure.";
    pub const CLI_NAME: &str = "outline";
    pub const HTTP_PATH: &str = "/api/outline";
}

/// Parameters for the get_outline operation
#[derive(Debug, Deserialize, Serialize, JsonSchema, clap::Parser)]
#[command(
    name = "outline",
    about = "Extract heading hierarchy from a markdown file"
)]
pub struct GetOutlineRequest {
    /// Path to vault (CLI only - not used in HTTP/MCP)
    #[arg(index = 1, required = true, help = "Path to vault root")]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub vault_path: Option<PathBuf>,

    /// File path relative to vault root
    #[arg(index = 2, required = true, help = "File path relative to vault root")]
    #[schemars(description = "File path relative to vault root")]
    pub file_path: String,

    /// Return hierarchical structure instead of flat list
    #[arg(long, help = "Return hierarchical tree structure")]
    #[schemars(
        description = "If true, return hierarchical tree structure with nested children. If false, return flat list (default)"
    )]
    pub hierarchical: Option<bool>,
}

/// Response from the get_outline operation
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetOutlineResponse {
    /// File path relative to vault root
    pub file_path: String,
    /// File name
    pub file_name: String,
    /// List of headings (flat or hierarchical)
    pub headings: Vec<Heading>,
    /// Total number of headings found
    pub total_count: usize,
}

/// Operation metadata for get_section
pub mod get_section {
    pub const DESCRIPTION: &str = "Extract content under a specific heading in a markdown file. Returns the section content with start/end line numbers.";
    pub const CLI_NAME: &str = "section";
    pub const HTTP_PATH: &str = "/api/outline/section";
}

/// Parameters for the get_section operation
#[derive(Debug, Deserialize, Serialize, JsonSchema, clap::Parser)]
#[command(name = "section", about = "Extract content under a specific heading")]
pub struct GetSectionRequest {
    /// Path to vault (CLI only - not used in HTTP/MCP)
    #[arg(index = 1, required = true, help = "Path to vault root")]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub vault_path: Option<PathBuf>,

    /// File path relative to vault root
    #[arg(index = 2, required = true, help = "File path relative to vault root")]
    #[schemars(description = "File path relative to vault root")]
    pub file_path: String,

    /// Heading title to find
    #[arg(index = 3, required = true, help = "Heading title to search for")]
    #[schemars(description = "The heading title to find (case-insensitive match)")]
    pub heading: String,

    /// Include subsections in the extracted content
    #[arg(long, help = "Include subsection content")]
    #[schemars(
        description = "If true, include content from subsections. If false, stop at subsection headings (default)"
    )]
    pub include_subsections: Option<bool>,
}

/// Response from the get_section operation
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetSectionResponse {
    /// File path relative to vault root
    pub file_path: String,
    /// File name
    pub file_name: String,
    /// Sections found (can be multiple if headings have same title)
    pub sections: Vec<Section>,
    /// Number of sections found
    pub section_count: usize,
}

/// Operation metadata for search_headings
pub mod search_headings {
    pub const DESCRIPTION: &str = "Search for headings matching a pattern across all markdown files in the vault. Returns matching headings with file paths. Case-insensitive substring matching.";
    pub const CLI_NAME: &str = "search-headings";
    pub const HTTP_PATH: &str = "/api/outline/search";
}

/// Parameters for the search_headings operation
#[derive(Debug, Deserialize, Serialize, JsonSchema, clap::Parser)]
#[command(
    name = "search-headings",
    about = "Search for headings across all files"
)]
pub struct SearchHeadingsRequest {
    /// Path to vault (CLI only - not used in HTTP/MCP)
    #[arg(index = 1, required = true, help = "Path to vault root")]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub vault_path: Option<PathBuf>,

    /// Search pattern (case-insensitive substring)
    #[arg(index = 2, required = true, help = "Pattern to search for in headings")]
    #[schemars(
        description = "Pattern to search for in headings (case-insensitive substring match)"
    )]
    pub pattern: String,

    /// Minimum heading level (1-6)
    #[arg(long, help = "Minimum heading level to include")]
    #[schemars(description = "Minimum heading level to include (1-6, optional)")]
    pub min_level: Option<u8>,

    /// Maximum heading level (1-6)
    #[arg(long, help = "Maximum heading level to include")]
    #[schemars(description = "Maximum heading level to include (1-6, optional)")]
    pub max_level: Option<u8>,

    /// Limit number of results
    #[arg(long, help = "Maximum number of results")]
    #[schemars(description = "Maximum number of results to return")]
    pub limit: Option<usize>,
}

/// Response from the search_headings operation
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SearchHeadingsResponse {
    /// Matching headings found
    pub matches: Vec<HeadingMatch>,
    /// Total number of matches
    pub total_count: usize,
}

/// Capability for outline operations (get_outline, get_section, search_headings)
pub struct OutlineCapability {
    base_path: PathBuf,
    config: Arc<Config>,
    outline_extractor: OutlineExtractor,
}

impl OutlineCapability {
    /// Create a new OutlineCapability
    pub fn new(base_path: PathBuf, config: Arc<Config>) -> Self {
        Self {
            base_path,
            config,
            outline_extractor: OutlineExtractor::new(),
        }
    }

    /// Validate and resolve a file path within the vault
    fn resolve_file_path(&self, file_path: &str) -> CapabilityResult<PathBuf> {
        // Construct full path
        let requested_path = PathBuf::from(file_path);
        let full_path = self.base_path.join(&requested_path);

        // Canonicalize paths for security check
        let canonical_base = self
            .base_path
            .canonicalize()
            .map_err(|e| internal_error(format!("Failed to resolve base path: {}", e)))?;

        let canonical_full = full_path
            .canonicalize()
            .map_err(|_| invalid_params(format!("File not found: {}", file_path)))?;

        // Security: Ensure path is within base directory
        if !canonical_full.starts_with(&canonical_base) {
            return Err(invalid_params(format!(
                "Invalid path '{}': must be within vault",
                file_path
            )));
        }

        // Validate it's a markdown file
        if canonical_full.extension().and_then(|s| s.to_str()) != Some("md") {
            return Err(invalid_params(format!(
                "Invalid file type '{}': only .md files allowed",
                file_path
            )));
        }

        Ok(canonical_full)
    }

    /// Get outline from a file
    pub async fn get_outline(
        &self,
        request: GetOutlineRequest,
    ) -> CapabilityResult<GetOutlineResponse> {
        let file_path = self.resolve_file_path(&request.file_path)?;

        let hierarchical = request.hierarchical.unwrap_or(false);

        let headings = self
            .outline_extractor
            .get_outline(&file_path, hierarchical)
            .map_err(|e| internal_error(format!("Failed to extract outline: {}", e)))?;

        let total_count = headings.len();
        let file_name = file_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        Ok(GetOutlineResponse {
            file_path: request.file_path,
            file_name,
            headings,
            total_count,
        })
    }

    /// Get section content under a specific heading
    pub async fn get_section(
        &self,
        request: GetSectionRequest,
    ) -> CapabilityResult<GetSectionResponse> {
        let file_path = self.resolve_file_path(&request.file_path)?;

        let include_subsections = request.include_subsections.unwrap_or(false);

        let sections = self
            .outline_extractor
            .get_section(&file_path, &request.heading, include_subsections)
            .map_err(|e| internal_error(format!("Failed to extract section: {}", e)))?;

        let section_count = sections.len();
        let file_name = file_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        Ok(GetSectionResponse {
            file_path: request.file_path,
            file_name,
            sections,
            section_count,
        })
    }

    /// Search for headings across files
    pub async fn search_headings(
        &self,
        request: SearchHeadingsRequest,
    ) -> CapabilityResult<SearchHeadingsResponse> {
        // Validate level parameters
        if let Some(min) = request.min_level
            && (!(1..=6).contains(&min))
        {
            return Err(invalid_params("min_level must be between 1 and 6"));
        }
        if let Some(max) = request.max_level
            && (!(1..=6).contains(&max))
        {
            return Err(invalid_params("max_level must be between 1 and 6"));
        }

        let matches = self
            .outline_extractor
            .search_headings(
                &self.base_path,
                &request.pattern,
                request.min_level,
                request.max_level,
                request.limit,
                &self.config,
            )
            .map_err(|e| internal_error(format!("Failed to search headings: {}", e)))?;

        let total_count = matches.len();

        Ok(SearchHeadingsResponse {
            matches,
            total_count,
        })
    }
}

/// Operation struct for get_outline (HTTP, CLI, and MCP)
pub struct GetOutlineOperation {
    capability: Arc<OutlineCapability>,
}

impl GetOutlineOperation {
    pub fn new(capability: Arc<OutlineCapability>) -> Self {
        Self { capability }
    }
}

/// Operation struct for get_section (HTTP, CLI, and MCP)
pub struct GetSectionOperation {
    capability: Arc<OutlineCapability>,
}

impl GetSectionOperation {
    pub fn new(capability: Arc<OutlineCapability>) -> Self {
        Self { capability }
    }
}

/// Operation struct for search_headings (HTTP, CLI, and MCP)
pub struct SearchHeadingsOperation {
    capability: Arc<OutlineCapability>,
}

impl SearchHeadingsOperation {
    pub fn new(capability: Arc<OutlineCapability>) -> Self {
        Self { capability }
    }
}

#[async_trait::async_trait]
impl notectl_core::operation::Operation for GetOutlineOperation {
    fn name(&self) -> &'static str {
        get_outline::CLI_NAME
    }

    fn path(&self) -> &'static str {
        get_outline::HTTP_PATH
    }

    fn description(&self) -> &'static str {
        get_outline::DESCRIPTION
    }

    fn get_command(&self) -> clap::Command {
        GetOutlineRequest::command()
    }

    fn get_remote_command(&self) -> clap::Command {
        // Rebuild without the vault_path positional; shift file_path to index 1
        clap::Command::new("outline")
            .about("Extract heading hierarchy from a markdown file")
            .arg(
                clap::Arg::new("file_path")
                    .index(1)
                    .required(true)
                    .help("File path relative to vault root"),
            )
            .arg(
                clap::Arg::new("hierarchical")
                    .long("hierarchical")
                    .value_parser(clap::value_parser!(bool))
                    .help("Return hierarchical tree structure"),
            )
    }

    async fn execute_json(
        &self,
        json: serde_json::Value,
    ) -> Result<serde_json::Value, rmcp::model::ErrorData> {
        let request: GetOutlineRequest = serde_json::from_value(json)
            .map_err(|e| notectl_core::error::invalid_params(e.to_string()))?;
        let response = self.capability.get_outline(request).await?;
        Ok(serde_json::to_value(response).unwrap())
    }

    async fn execute_from_args(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let request = GetOutlineRequest::from_arg_matches(matches)?;

        // Handle CLI-specific vault path if present
        let response = if let Some(ref vault_path) = request.vault_path {
            let config = Arc::new(Config::load_from_base_path(vault_path.as_path()));
            let capability = OutlineCapability::new(vault_path.clone(), config);
            let mut req_without_path = request;
            req_without_path.vault_path = None;
            capability.get_outline(req_without_path).await?
        } else {
            self.capability.get_outline(request).await?
        };

        Ok(serde_json::to_string_pretty(&response)?)
    }

    fn input_schema(&self) -> serde_json::Value {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(GetOutlineRequest)).unwrap()
    }

    fn args_to_json(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut request = GetOutlineRequest::from_arg_matches(matches)?;
        request.vault_path = None;
        Ok(serde_json::to_value(request)?)
    }
}

#[async_trait::async_trait]
impl notectl_core::operation::Operation for GetSectionOperation {
    fn name(&self) -> &'static str {
        get_section::CLI_NAME
    }

    fn path(&self) -> &'static str {
        get_section::HTTP_PATH
    }

    fn description(&self) -> &'static str {
        get_section::DESCRIPTION
    }

    fn get_command(&self) -> clap::Command {
        GetSectionRequest::command()
    }

    fn get_remote_command(&self) -> clap::Command {
        // Rebuild without the vault_path positional; shift file_path/heading down by one
        clap::Command::new("section")
            .about("Extract content under a specific heading")
            .arg(
                clap::Arg::new("file_path")
                    .index(1)
                    .required(true)
                    .help("File path relative to vault root"),
            )
            .arg(
                clap::Arg::new("heading")
                    .index(2)
                    .required(true)
                    .help("Heading title to search for"),
            )
            .arg(
                clap::Arg::new("include_subsections")
                    .long("include-subsections")
                    .value_parser(clap::value_parser!(bool))
                    .help("Include subsection content"),
            )
    }

    async fn execute_json(
        &self,
        json: serde_json::Value,
    ) -> Result<serde_json::Value, rmcp::model::ErrorData> {
        let request: GetSectionRequest = serde_json::from_value(json)
            .map_err(|e| notectl_core::error::invalid_params(e.to_string()))?;
        let response = self.capability.get_section(request).await?;
        Ok(serde_json::to_value(response).unwrap())
    }

    async fn execute_from_args(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let request = GetSectionRequest::from_arg_matches(matches)?;

        // Handle CLI-specific vault path if present
        let response = if let Some(ref vault_path) = request.vault_path {
            let config = Arc::new(Config::load_from_base_path(vault_path.as_path()));
            let capability = OutlineCapability::new(vault_path.clone(), config);
            let mut req_without_path = request;
            req_without_path.vault_path = None;
            capability.get_section(req_without_path).await?
        } else {
            self.capability.get_section(request).await?
        };

        Ok(serde_json::to_string_pretty(&response)?)
    }

    fn input_schema(&self) -> serde_json::Value {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(GetSectionRequest)).unwrap()
    }

    fn args_to_json(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut request = GetSectionRequest::from_arg_matches(matches)?;
        request.vault_path = None;
        Ok(serde_json::to_value(request)?)
    }
}

#[async_trait::async_trait]
impl notectl_core::operation::Operation for SearchHeadingsOperation {
    fn name(&self) -> &'static str {
        search_headings::CLI_NAME
    }

    fn path(&self) -> &'static str {
        search_headings::HTTP_PATH
    }

    fn description(&self) -> &'static str {
        search_headings::DESCRIPTION
    }

    fn get_command(&self) -> clap::Command {
        SearchHeadingsRequest::command()
    }

    fn get_remote_command(&self) -> clap::Command {
        // Rebuild without the vault_path positional; shift pattern to index 1
        clap::Command::new("search-headings")
            .about("Search for headings across all files")
            .arg(
                clap::Arg::new("pattern")
                    .index(1)
                    .required(true)
                    .help("Pattern to search for in headings (case-insensitive substring match)"),
            )
            .arg(
                clap::Arg::new("min_level")
                    .long("min-level")
                    .value_parser(clap::value_parser!(u8))
                    .help("Minimum heading level to include"),
            )
            .arg(
                clap::Arg::new("max_level")
                    .long("max-level")
                    .value_parser(clap::value_parser!(u8))
                    .help("Maximum heading level to include"),
            )
            .arg(
                clap::Arg::new("limit")
                    .long("limit")
                    .value_parser(clap::value_parser!(usize))
                    .help("Maximum number of results"),
            )
    }

    async fn execute_json(
        &self,
        json: serde_json::Value,
    ) -> Result<serde_json::Value, rmcp::model::ErrorData> {
        let request: SearchHeadingsRequest = serde_json::from_value(json)
            .map_err(|e| notectl_core::error::invalid_params(e.to_string()))?;
        let response = self.capability.search_headings(request).await?;
        Ok(serde_json::to_value(response).unwrap())
    }

    async fn execute_from_args(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let request = SearchHeadingsRequest::from_arg_matches(matches)?;

        // Handle CLI-specific vault path if present
        let response = if let Some(ref vault_path) = request.vault_path {
            let config = Arc::new(Config::load_from_base_path(vault_path.as_path()));
            let capability = OutlineCapability::new(vault_path.clone(), config);
            let mut req_without_path = request;
            req_without_path.vault_path = None;
            capability.search_headings(req_without_path).await?
        } else {
            self.capability.search_headings(request).await?
        };

        Ok(serde_json::to_string_pretty(&response)?)
    }

    fn input_schema(&self) -> serde_json::Value {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(SearchHeadingsRequest)).unwrap()
    }

    fn args_to_json(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut request = SearchHeadingsRequest::from_arg_matches(matches)?;
        request.vault_path = None;
        Ok(serde_json::to_value(request)?)
    }
}
