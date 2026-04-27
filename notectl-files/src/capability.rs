use clap::{CommandFactory, FromArgMatches};
use notectl_core::CapabilityResult;
use notectl_core::config::Config;
use notectl_core::error::{internal_error, invalid_params};
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing;

/// Operation metadata for list_files
pub mod list_files {
    pub const DESCRIPTION: &str = "List the directory tree of the vault. Returns a hierarchical view of all files and folders. Useful for understanding vault structure and finding files.";
    #[allow(dead_code)]
    pub const CLI_NAME: &str = "list-files";
    pub const HTTP_PATH: &str = "/api/files";
}

/// Parameters for the list_files operation
#[derive(Debug, Serialize, Deserialize, JsonSchema, clap::Parser)]
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
#[derive(Debug, Serialize, Deserialize, JsonSchema, clap::Parser)]
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

    pub async fn list_files(
        &self,
        request: ListFilesRequest,
    ) -> CapabilityResult<ListFilesResponse> {
        let base_path = self.base_path.clone();
        let config = Arc::clone(&self.config);
        tracing::debug!(path = %base_path.display(), "Listing files");
        let (visual_tree, total_files, total_directories) =
            tokio::task::spawn_blocking(move || list_files_blocking(base_path, config, request))
                .await
                .map_err(|e| internal_error(format!("File listing panicked: {}", e)))??;
        tracing::debug!(total_files, total_directories, "File listing complete");
        Ok(ListFilesResponse {
            visual_tree,
            total_files,
            total_directories,
        })
    }

    pub async fn read_files(
        &self,
        request: ReadFilesRequest,
    ) -> CapabilityResult<ReadFilesResponse> {
        let base_path = self.base_path.clone();
        tracing::debug!(file_count = request.file_paths.len(), "Reading files");
        tokio::task::spawn_blocking(move || read_files_blocking(base_path, request))
            .await
            .map_err(|e| internal_error(format!("File read panicked: {}", e)))?
    }
}

fn list_files_blocking(
    base_path: PathBuf,
    config: Arc<Config>,
    request: ListFilesRequest,
) -> CapabilityResult<(String, usize, usize)> {
    let search_path = if let Some(ref subpath) = request.subpath {
        base_path.join(PathBuf::from(subpath))
    } else {
        base_path.clone()
    };

    let canonical_base = base_path
        .canonicalize()
        .map_err(|e| internal_error(format!("Failed to resolve base path: {}", e)))?;

    let canonical_search = search_path
        .canonicalize()
        .map_err(|_e| invalid_params(format!("Path not found: {:?}", request.subpath)))?;

    if !canonical_search.starts_with(&canonical_base) {
        return Err(invalid_params(
            "Invalid path: path must be within the vault",
        ));
    }

    let include_sizes = request.include_sizes.unwrap_or(false);

    let (root, total_files, total_directories) = build_file_tree(
        &canonical_search,
        &canonical_base,
        &config,
        0,
        request.max_depth,
        include_sizes,
    )
    .map_err(|e| internal_error(format!("Failed to build file tree: {}", e)))?;

    Ok((format_tree_visual(&root, 0), total_files, total_directories))
}

fn read_files_blocking(
    base_path: PathBuf,
    request: ReadFilesRequest,
) -> CapabilityResult<ReadFilesResponse> {
    let continue_on_error = request.continue_on_error.unwrap_or(false);

    let canonical_base = base_path
        .canonicalize()
        .map_err(|e| internal_error(format!("Failed to resolve base path: {}", e)))?;

    if !continue_on_error {
        if request.file_paths.is_empty() {
            return Err(invalid_params("file_paths cannot be empty"));
        }
        for file_path in &request.file_paths {
            let canonical_full = base_path
                .join(PathBuf::from(file_path))
                .canonicalize()
                .map_err(|_| invalid_params(format!("File not found: {}", file_path)))?;
            if !canonical_full.starts_with(&canonical_base) {
                return Err(invalid_params(format!(
                    "Invalid path '{}': must be within vault",
                    file_path
                )));
            }
            if canonical_full.extension().and_then(|s| s.to_str()) != Some("md") {
                return Err(invalid_params(format!(
                    "Invalid file type '{}': only .md files allowed",
                    file_path
                )));
            }
        }
    }

    let mut results = Vec::new();
    let mut success_count = 0;
    let mut failure_count = 0;

    for file_path in &request.file_paths {
        let result = read_single_file_blocking(&base_path, &canonical_base, file_path);
        match result {
            Ok(content) => {
                results.push(ReadFileResult {
                    file_path: file_path.clone(),
                    file_name: extract_file_name(file_path),
                    success: true,
                    content: Some(content),
                    error: None,
                });
                success_count += 1;
            }
            Err(e) => {
                if continue_on_error {
                    results.push(ReadFileResult {
                        file_path: file_path.clone(),
                        file_name: extract_file_name(file_path),
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

fn read_single_file_blocking(
    base_path: &Path,
    canonical_base: &Path,
    file_path: &str,
) -> CapabilityResult<String> {
    let canonical_full = base_path
        .join(PathBuf::from(file_path))
        .canonicalize()
        .map_err(|_| invalid_params(format!("File not found: {}", file_path)))?;
    if !canonical_full.starts_with(canonical_base) {
        return Err(invalid_params(format!(
            "Invalid path '{}': must be within vault",
            file_path
        )));
    }
    if canonical_full.extension().and_then(|s| s.to_str()) != Some("md") {
        return Err(invalid_params(format!(
            "Invalid file type '{}': only .md files allowed",
            file_path
        )));
    }
    std::fs::read_to_string(&canonical_full)
        .map_err(|e| internal_error(format!("Failed to read file: {}", e)))
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

    fn get_remote_command(&self) -> clap::Command {
        self.get_command()
            .mut_arg("path", |a| a.required(false).hide(true))
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

    fn args_to_json(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut request = ListFilesRequest::from_arg_matches(matches)?;
        request.path = None;
        Ok(serde_json::to_value(request)?)
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

    fn get_remote_command(&self) -> clap::Command {
        // Rebuild without the vault_path positional; shift file_paths to index 1
        clap::Command::new("read-files")
            .about("Read one or more markdown files")
            .arg(
                clap::Arg::new("file_paths")
                    .index(1)
                    .required(true)
                    .value_delimiter(',')
                    .help("Comma-separated file paths relative to vault root"),
            )
            .arg(
                clap::Arg::new("continue_on_error")
                    .long("continue-on-error")
                    .value_parser(clap::value_parser!(bool))
                    .help("Continue reading files even if some fail"),
            )
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

    fn args_to_json(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut request = ReadFilesRequest::from_arg_matches(matches)?;
        request.vault_path = None;
        Ok(serde_json::to_value(request)?)
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

// ── recent-files ─────────────────────────────────────────────────────────────

/// Operation metadata for recent_files
pub mod recent_files_op {
    pub const DESCRIPTION: &str = "List recently modified markdown files, sorted by modification time descending. Checks the frontmatter `updated` field first; falls back to filesystem mtime.";
    #[allow(dead_code)]
    pub const CLI_NAME: &str = "recent-files";
    pub const HTTP_PATH: &str = "/api/files/recent";
}

/// Parameters for the recent_files operation
#[derive(Debug, Serialize, Deserialize, JsonSchema, clap::Parser)]
#[command(name = "recent-files", about = "List recently modified vault files")]
pub struct RecentFilesRequest {
    /// Path to vault (CLI only)
    #[arg(index = 1, required = true, help = "Path to vault to scan")]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub path: Option<PathBuf>,

    /// Only include files modified on or after this date (YYYY-MM-DD)
    #[arg(long, help = "Only files modified on or after YYYY-MM-DD")]
    #[schemars(description = "ISO date lower bound: YYYY-MM-DD (inclusive)")]
    pub since: Option<String>,

    /// Maximum number of results (default 20)
    #[arg(long, help = "Maximum number of results (default 20)")]
    #[schemars(description = "Maximum results to return (default 20)")]
    pub limit: Option<usize>,
}

/// A single result entry for recent_files
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct RecentFileEntry {
    /// File path relative to vault root
    pub file_path: String,
    /// File name only
    pub file_name: String,
    /// The timestamp used for sorting, as an ISO 8601 string
    pub updated_at: String,
    /// Whether the date came from frontmatter or filesystem mtime
    pub date_source: String,
}

/// Response from the recent_files operation
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct RecentFilesResponse {
    pub files: Vec<RecentFileEntry>,
    /// Total matching files before the limit was applied
    pub total_found: usize,
}

impl FileCapability {
    pub async fn recent_files(
        &self,
        request: RecentFilesRequest,
    ) -> CapabilityResult<RecentFilesResponse> {
        use crate::recent_files::parse_iso8601_to_unix;

        let since_ts: Option<i64> = request
            .since
            .as_deref()
            .map(|s| {
                let expanded = format!("{}T00:00:00Z", s);
                parse_iso8601_to_unix(&expanded)
                    .ok_or_else(|| invalid_params(format!("Invalid since date: {}", s)))
            })
            .transpose()?;

        let limit = request.limit.unwrap_or(20);
        let base_path = self.base_path.clone();
        let config = Arc::clone(&self.config);

        tracing::debug!(path = %base_path.display(), "Listing recent files");

        let (files, total_found) = tokio::task::spawn_blocking(move || {
            recent_files_blocking(base_path, config, since_ts, limit)
        })
        .await
        .map_err(|e| internal_error(format!("Recent files scan panicked: {}", e)))??;

        tracing::debug!(total_found, "Recent files scan complete");

        Ok(RecentFilesResponse { files, total_found })
    }
}

fn recent_files_blocking(
    base_path: PathBuf,
    config: Arc<Config>,
    since_ts: Option<i64>,
    limit: usize,
) -> CapabilityResult<(Vec<RecentFileEntry>, usize)> {
    use crate::recent_files::file_timestamp;

    let markdown_files = notectl_core::file_walker::collect_markdown_files(&base_path, &config)
        .map_err(|e| internal_error(format!("Failed to walk vault: {}", e)))?;

    let mut entries: Vec<(i64, RecentFileEntry)> = markdown_files
        .par_iter()
        .filter_map(|path| {
            let (ts, source, display) = file_timestamp(path)?;
            if let Some(threshold) = since_ts
                && ts < threshold
            {
                return None;
            }
            let rel_path = path
                .strip_prefix(&base_path)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();
            let file_name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            Some((
                ts,
                RecentFileEntry {
                    file_path: rel_path,
                    file_name,
                    updated_at: display,
                    date_source: source.to_string(),
                },
            ))
        })
        .collect();

    entries.sort_by(|a, b| b.0.cmp(&a.0));
    let total_found = entries.len();
    let files = entries.into_iter().take(limit).map(|(_, e)| e).collect();
    Ok((files, total_found))
}

/// Operation struct for recent_files (HTTP, CLI, and MCP)
pub struct RecentFilesOperation {
    capability: Arc<FileCapability>,
}

impl RecentFilesOperation {
    pub fn new(capability: Arc<FileCapability>) -> Self {
        Self { capability }
    }
}

#[async_trait::async_trait]
impl notectl_core::operation::Operation for RecentFilesOperation {
    fn name(&self) -> &'static str {
        recent_files_op::CLI_NAME
    }

    fn path(&self) -> &'static str {
        recent_files_op::HTTP_PATH
    }

    fn description(&self) -> &'static str {
        recent_files_op::DESCRIPTION
    }

    fn get_command(&self) -> clap::Command {
        RecentFilesRequest::command()
    }

    fn get_remote_command(&self) -> clap::Command {
        self.get_command()
            .mut_arg("path", |a| a.required(false).hide(true))
    }

    async fn execute_json(
        &self,
        json: serde_json::Value,
    ) -> Result<serde_json::Value, rmcp::model::ErrorData> {
        let request: RecentFilesRequest = serde_json::from_value(json)
            .map_err(|e| notectl_core::error::invalid_params(e.to_string()))?;
        let response = self.capability.recent_files(request).await?;
        Ok(serde_json::to_value(response).unwrap())
    }

    async fn execute_from_args(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let request = RecentFilesRequest::from_arg_matches(matches)?;
        let response = if let Some(ref path) = request.path {
            let config = Arc::new(Config::load_from_base_path(path.as_path()));
            let capability = FileCapability::new(path.clone(), config);
            let mut req = request;
            req.path = None;
            capability.recent_files(req).await?
        } else {
            self.capability.recent_files(request).await?
        };
        Ok(serde_json::to_string_pretty(&response)?)
    }

    fn input_schema(&self) -> serde_json::Value {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(RecentFilesRequest)).unwrap()
    }

    fn args_to_json(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut request = RecentFilesRequest::from_arg_matches(matches)?;
        request.path = None;
        Ok(serde_json::to_value(request)?)
    }
}
