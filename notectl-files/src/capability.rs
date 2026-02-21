use clap::{CommandFactory, FromArgMatches};
use notectl_core::CapabilityResult;
use notectl_core::config::Config;
use notectl_core::error::{internal_error, invalid_params};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Operation metadata for list_files
pub mod list_files {
    pub const DESCRIPTION: &str = "List the directory tree of the vault. Returns a hierarchical view of all files and folders. Useful for understanding vault structure and finding files.";
    #[allow(dead_code)]
    pub const CLI_NAME: &str = "list-files";
    pub const HTTP_PATH: &str = "/api/files";
}

/// Parameters for the list_files operation
#[derive(Debug, Deserialize, JsonSchema, clap::Parser)]
#[command(name = "list-files", about = "List the directory tree of the vault")]
pub struct ListFilesRequest {
    /// Path to scan (CLI only - not used in HTTP/MCP)
    #[arg(index = 1, required = true, help = "Path to vault to scan")]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub path: Option<PathBuf>,

    #[arg(long, help = "Subpath within the vault to list")]
    #[schemars(
        description = "Subpath within the vault to list (optional, defaults to vault root)"
    )]
    pub subpath: Option<String>,

    #[arg(long, help = "Maximum depth to traverse")]
    #[schemars(description = "Maximum depth to traverse (optional, defaults to unlimited)")]
    pub max_depth: Option<usize>,

    #[arg(long, help = "Include file sizes in output")]
    #[schemars(description = "Include file sizes in output (optional, defaults to false)")]
    pub include_sizes: Option<bool>,
}

/// A node in the file tree
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct FileTreeNode {
    pub name: String,
    pub path: String,
    pub is_directory: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub children: Vec<FileTreeNode>,
}

/// Response from the list_files operation
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ListFilesResponse {
    /// Visual tree representation with indented structure
    pub visual_tree: String,
    pub total_files: usize,
    pub total_directories: usize,
}

/// Operation metadata for read_files
pub mod read_files {
    pub const DESCRIPTION: &str = "Read one or more markdown files from the vault. Returns content for all requested files with per-file success/error status.";
    #[allow(dead_code)]
    pub const CLI_NAME: &str = "read-files";
    pub const HTTP_PATH: &str = "/api/files/read";
}

/// Result for a single file read operation
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ReadFileResult {
    /// File path relative to vault root
    pub file_path: String,
    /// File name only
    pub file_name: String,
    /// Whether this file was successfully read
    pub success: bool,
    /// File content (only present if success=true)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Error message (only present if success=false)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Response from the read_files operation
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ReadFilesResponse {
    /// Successfully read files
    pub files: Vec<ReadFileResult>,
    /// Total number of files requested
    pub total_requested: usize,
    /// Number of files successfully read
    pub success_count: usize,
    /// Number of files that failed
    pub failure_count: usize,
}

/// Parameters for the read_files operation
#[derive(Debug, Deserialize, JsonSchema, clap::Parser)]
#[command(name = "read-files", about = "Read one or more markdown files")]
pub struct ReadFilesRequest {
    /// Vault path (CLI only - not used in HTTP/MCP)
    #[arg(index = 1, required = true, help = "Path to vault")]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub vault_path: Option<PathBuf>,

    /// File paths relative to vault root (comma-separated for CLI)
    #[arg(
        index = 2,
        required = true,
        value_delimiter = ',',
        help = "Comma-separated file paths relative to vault root"
    )]
    #[schemars(description = "File paths relative to vault root (one or more)")]
    pub file_paths: Vec<String>,

    /// Continue on error (return partial results)
    #[arg(long, help = "Continue reading files even if some fail")]
    #[schemars(description = "If true, continue on errors and return partial results")]
    pub continue_on_error: Option<bool>,
}

/// Capability for file operations (list, read)
pub struct FileCapability {
    base_path: PathBuf,
    config: Arc<Config>,
}

impl FileCapability {
    /// Create a new FileCapability
    pub fn new(base_path: PathBuf, config: Arc<Config>) -> Self {
        Self { base_path, config }
    }

    /// List the directory tree of the vault
    pub async fn list_files(
        &self,
        request: ListFilesRequest,
    ) -> CapabilityResult<ListFilesResponse> {
        // Resolve the search path
        let search_path = if let Some(ref subpath) = request.subpath {
            let requested_path = PathBuf::from(subpath);
            self.base_path.join(&requested_path)
        } else {
            self.base_path.clone()
        };

        // Canonicalize paths for security check
        let canonical_base = self
            .base_path
            .canonicalize()
            .map_err(|e| internal_error(format!("Failed to resolve base path: {}", e)))?;

        let canonical_search = search_path
            .canonicalize()
            .map_err(|_e| invalid_params(format!("Path not found: {:?}", request.subpath)))?;

        // Security: Ensure path is within base directory
        if !canonical_search.starts_with(&canonical_base) {
            return Err(invalid_params(
                "Invalid path: path must be within the vault",
            ));
        }

        // Build the file tree
        let include_sizes = request.include_sizes.unwrap_or(false);

        let (root, total_files, total_directories) = build_file_tree(
            &canonical_search,
            &canonical_base,
            &self.config,
            0,
            request.max_depth,
            include_sizes,
        )
        .map_err(|e| internal_error(format!("Failed to build file tree: {}", e)))?;

        // Generate visual tree representation
        let visual_tree = format_tree_visual(&root, 0);

        Ok(ListFilesResponse {
            visual_tree,
            total_files,
            total_directories,
        })
    }

    /// Read one or more markdown files
    pub async fn read_files(
        &self,
        request: ReadFilesRequest,
    ) -> CapabilityResult<ReadFilesResponse> {
        let continue_on_error = request.continue_on_error.unwrap_or(false);

        // Validation phase (if fail-fast mode)
        if !continue_on_error {
            self.validate_all_paths(&request.file_paths)?;
        }

        // Reading phase
        let mut results = Vec::new();
        let mut success_count = 0;
        let mut failure_count = 0;

        for file_path in &request.file_paths {
            match self.read_single_file(file_path) {
                Ok(content) => {
                    let file_name = extract_file_name(file_path);
                    results.push(ReadFileResult {
                        file_path: file_path.clone(),
                        file_name,
                        success: true,
                        content: Some(content),
                        error: None,
                    });
                    success_count += 1;
                }
                Err(e) => {
                    if continue_on_error {
                        let file_name = extract_file_name(file_path);
                        results.push(ReadFileResult {
                            file_path: file_path.clone(),
                            file_name,
                            success: false,
                            content: None,
                            error: Some(e.to_string()),
                        });
                        failure_count += 1;
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Ok(ReadFilesResponse {
            files: results,
            total_requested: request.file_paths.len(),
            success_count,
            failure_count,
        })
    }

    /// Validate all paths before reading (fail-fast mode)
    fn validate_all_paths(&self, file_paths: &[String]) -> CapabilityResult<()> {
        // Check non-empty
        if file_paths.is_empty() {
            return Err(invalid_params("file_paths cannot be empty"));
        }

        // Canonicalize base path once
        let canonical_base = self
            .base_path
            .canonicalize()
            .map_err(|e| internal_error(format!("Failed to resolve base path: {}", e)))?;

        // Validate each path
        for file_path in file_paths {
            let requested_path = PathBuf::from(file_path);
            let full_path = self.base_path.join(&requested_path);

            // Check existence
            let canonical_full = full_path
                .canonicalize()
                .map_err(|_| invalid_params(format!("File not found: {}", file_path)))?;

            // Security check
            if !canonical_full.starts_with(&canonical_base) {
                return Err(invalid_params(format!(
                    "Invalid path '{}': must be within vault",
                    file_path
                )));
            }

            // File type check
            if canonical_full.extension().and_then(|s| s.to_str()) != Some("md") {
                return Err(invalid_params(format!(
                    "Invalid file type '{}': only .md files allowed",
                    file_path
                )));
            }
        }

        Ok(())
    }

    /// Read a single file (internal helper)
    fn read_single_file(&self, file_path: &str) -> CapabilityResult<String> {
        // 1. Construct the full path
        let requested_path = PathBuf::from(file_path);
        let full_path = self.base_path.join(&requested_path);

        // 2. Canonicalize paths for security check
        let canonical_base = self
            .base_path
            .canonicalize()
            .map_err(|e| internal_error(format!("Failed to resolve base path: {}", e)))?;

        let canonical_full = full_path
            .canonicalize()
            .map_err(|_| invalid_params(format!("File not found: {}", file_path)))?;

        // 3. Security: Ensure path is within base directory
        if !canonical_full.starts_with(&canonical_base) {
            return Err(invalid_params(format!(
                "Invalid path '{}': must be within vault",
                file_path
            )));
        }

        // 4. Validate it's a markdown file
        if canonical_full.extension().and_then(|s| s.to_str()) != Some("md") {
            return Err(invalid_params(format!(
                "Invalid file type '{}': only .md files allowed",
                file_path
            )));
        }

        // 5. Read the file content
        let content = std::fs::read_to_string(&canonical_full)
            .map_err(|e| internal_error(format!("Failed to read file: {}", e)))?;

        Ok(content)
    }
}

/// Operation struct for list_files (HTTP, CLI, and MCP)
pub struct ListFilesOperation {
    capability: Arc<FileCapability>,
}

impl ListFilesOperation {
    pub fn new(capability: Arc<FileCapability>) -> Self {
        Self { capability }
    }
}

/// Operation struct for read_files (HTTP, CLI, and MCP)
pub struct ReadFilesOperation {
    capability: Arc<FileCapability>,
}

impl ReadFilesOperation {
    pub fn new(capability: Arc<FileCapability>) -> Self {
        Self { capability }
    }
}

/// Extract file name from path
fn extract_file_name(file_path: &str) -> String {
    Path::new(file_path)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string()
}

#[async_trait::async_trait]
impl notectl_core::operation::Operation for ListFilesOperation {
    fn name(&self) -> &'static str {
        list_files::CLI_NAME
    }

    fn path(&self) -> &'static str {
        list_files::HTTP_PATH
    }

    fn description(&self) -> &'static str {
        list_files::DESCRIPTION
    }

    fn get_command(&self) -> clap::Command {
        // Get command from request struct's Parser derive
        ListFilesRequest::command()
    }

    async fn execute_json(
        &self,
        json: serde_json::Value,
    ) -> Result<serde_json::Value, rmcp::model::ErrorData> {
        let request: ListFilesRequest = serde_json::from_value(json)
            .map_err(|e| notectl_core::error::invalid_params(e.to_string()))?;
        let response = self.capability.list_files(request).await?;
        Ok(serde_json::to_value(response).unwrap())
    }

    async fn execute_from_args(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Parse request from ArgMatches
        let request = ListFilesRequest::from_arg_matches(matches)?;

        // Handle CLI-specific path if present
        let response = if let Some(ref path) = request.path {
            let config = Arc::new(Config::load_from_base_path(path.as_path()));
            let capability = FileCapability::new(path.clone(), config);
            let mut req_without_path = request;
            req_without_path.path = None;
            capability.list_files(req_without_path).await?
        } else {
            self.capability.list_files(request).await?
        };

        // Return the visual tree directly
        Ok(response.visual_tree)
    }

    fn input_schema(&self) -> serde_json::Value {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(ListFilesRequest)).unwrap()
    }
}

#[async_trait::async_trait]
impl notectl_core::operation::Operation for ReadFilesOperation {
    fn name(&self) -> &'static str {
        read_files::CLI_NAME
    }

    fn path(&self) -> &'static str {
        read_files::HTTP_PATH
    }

    fn description(&self) -> &'static str {
        read_files::DESCRIPTION
    }

    fn get_command(&self) -> clap::Command {
        // Get command from request struct's Parser derive
        ReadFilesRequest::command()
    }

    async fn execute_json(
        &self,
        json: serde_json::Value,
    ) -> Result<serde_json::Value, rmcp::model::ErrorData> {
        let request: ReadFilesRequest = serde_json::from_value(json)
            .map_err(|e| notectl_core::error::invalid_params(e.to_string()))?;
        let response = self.capability.read_files(request).await?;
        Ok(serde_json::to_value(response).unwrap())
    }

    async fn execute_from_args(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Parse request from ArgMatches
        let request = ReadFilesRequest::from_arg_matches(matches)?;

        // Handle CLI-specific vault path if present
        let response = if let Some(ref vault_path) = request.vault_path {
            let config = Arc::new(Config::load_from_base_path(vault_path.as_path()));
            let capability = FileCapability::new(vault_path.clone(), config);
            let mut req_without_path = request;
            req_without_path.vault_path = None;
            capability.read_files(req_without_path).await?
        } else {
            self.capability.read_files(request).await?
        };

        // Serialize to JSON
        Ok(serde_json::to_string_pretty(&response)?)
    }

    fn input_schema(&self) -> serde_json::Value {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(ReadFilesRequest)).unwrap()
    }
}

/// Format a byte count as a human-readable size string
fn format_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;
    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }
    if unit_idx == 0 {
        format!("{} B", bytes)
    } else {
        format!("{:.1} {}", size, UNITS[unit_idx])
    }
}

/// Helper function to format a file tree as visual indented text
fn format_tree_visual(node: &FileTreeNode, indent_level: usize) -> String {
    let mut output = String::new();
    let indent = "  ".repeat(indent_level);

    // Add current node
    if node.is_directory {
        output.push_str(&format!("{}{}/\n", indent, node.name));
    } else if let Some(size) = node.size_bytes {
        output.push_str(&format!(
            "{}{} ({})\n",
            indent,
            node.name,
            format_size(size)
        ));
    } else {
        output.push_str(&format!("{}{}\n", indent, node.name));
    }

    // Recursively add children
    for child in &node.children {
        output.push_str(&format_tree_visual(child, indent_level + 1));
    }

    output
}

/// Helper function to recursively build file tree
fn build_file_tree(
    path: &Path,
    base_path: &Path,
    config: &Config,
    current_depth: usize,
    max_depth: Option<usize>,
    include_sizes: bool,
) -> Result<(FileTreeNode, usize, usize), Box<dyn std::error::Error>> {
    // Check depth limit
    if let Some(max) = max_depth
        && current_depth >= max
    {
        // Still need to check if it's a file or directory
        let metadata = std::fs::metadata(path)?;
        let is_dir = metadata.is_dir();
        let size = if !is_dir && include_sizes {
            Some(metadata.len())
        } else {
            None
        };

        return Ok((
            FileTreeNode {
                name: path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
                path: path
                    .strip_prefix(base_path)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .to_string(),
                is_directory: is_dir,
                size_bytes: size,
                children: vec![],
            },
            if is_dir { 0 } else { 1 }, // Count as file if it's a file
            0,
        ));
    }

    // Check if path should be excluded
    if config.should_exclude(path) {
        return Err("Path excluded by configuration".into());
    }

    let metadata = std::fs::metadata(path)?;

    if !metadata.is_dir() {
        // It's a file
        let size = if include_sizes {
            Some(metadata.len())
        } else {
            None
        };

        return Ok((
            FileTreeNode {
                name: path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
                path: path
                    .strip_prefix(base_path)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .to_string(),
                is_directory: false,
                size_bytes: size,
                children: vec![],
            },
            1, // 1 file
            0, // 0 directories
        ));
    }

    // It's a directory - recurse
    let mut children = Vec::new();
    let mut total_files = 0;
    let mut total_directories = 1; // Count this directory

    let entries = std::fs::read_dir(path)?;
    for entry in entries {
        let entry = entry?;
        let entry_path = entry.path();

        // Skip hidden files/directories (starting with .)
        if let Some(name) = entry_path.file_name()
            && name.to_string_lossy().starts_with('.')
        {
            continue;
        }

        // Try to build subtree, skip if excluded
        match build_file_tree(
            &entry_path,
            base_path,
            config,
            current_depth + 1,
            max_depth,
            include_sizes,
        ) {
            Ok((child_node, child_files, child_dirs)) => {
                children.push(child_node);
                total_files += child_files;
                total_directories += child_dirs;
            }
            Err(_) => {
                // Skip excluded paths
                continue;
            }
        }
    }

    // Sort children: directories first, then files, alphabetically
    children.sort_by(|a, b| match (a.is_directory, b.is_directory) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => a.name.cmp(&b.name),
    });

    Ok((
        FileTreeNode {
            name: path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            path: path
                .strip_prefix(base_path)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string(),
            is_directory: true,
            size_bytes: None,
            children,
        },
        total_files,
        total_directories,
    ))
}
