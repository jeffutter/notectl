//! Daily Notes capability
//!
//! Provides operations for querying Obsidian daily notes by date or date range.
//! Supports configurable date patterns and leverages multi-file reading for efficiency.

use clap::{CommandFactory, FromArgMatches};
use notectl_core::CapabilityResult;
use notectl_core::config::Config;
use notectl_core::error::{internal_error, invalid_params};
use notectl_files::{FileCapability, ReadFilesRequest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tracing;

// Re-export for internal use
use crate::date_utils::{date_range, today, validate_date};
use crate::pattern::get_daily_note_relative_path;

/// Operation metadata for get_daily_note
pub mod get_daily_note {
    pub const DESCRIPTION: &str = "Get the content of a daily note for a specific date. Returns the note content, file path, and whether the note was found. Missing notes return found: false (not an error).";
    #[allow(dead_code)]
    pub const CLI_NAME: &str = "get-daily-note";
    pub const HTTP_PATH: &str = "/api/daily-notes";
}

/// Operation metadata for search_daily_notes
pub mod search_daily_notes {
    pub const DESCRIPTION: &str = "Search for daily notes within a date range. Returns metadata for all matching notes. Use get_daily_note to retrieve full content for specific notes.";
    #[allow(dead_code)]
    pub const CLI_NAME: &str = "search-daily-notes";
    pub const HTTP_PATH: &str = "/api/daily-notes/search";
}

/// Parameters for the get_daily_note operation
#[derive(Debug, Deserialize, Serialize, JsonSchema, clap::Parser)]
#[command(name = "get-daily-note", about = "Get daily note for a specific date")]
pub struct GetDailyNoteRequest {
    /// Path to vault (CLI only - not used in HTTP/MCP)
    #[arg(index = 1, required = true, help = "Path to vault")]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub vault_path: Option<PathBuf>,

    /// Date in YYYY-MM-DD format
    #[arg(long, help = "Date in YYYY-MM-DD format")]
    #[schemars(description = "Date in YYYY-MM-DD format (e.g., 2025-01-20)")]
    pub date: String,
}

/// Response from the get_daily_note operation
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetDailyNoteResponse {
    /// Whether the daily note was found
    pub found: bool,
    /// Date in YYYY-MM-DD format
    pub date: String,
    /// File path relative to vault root (only present if found=true)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
    /// File name (only present if found=true)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_name: Option<String>,
    /// Note content (only present if found=true)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Parameters for the search_daily_notes operation
#[derive(Debug, Deserialize, Serialize, JsonSchema, clap::Parser)]
#[command(
    name = "search-daily-notes",
    about = "Search daily notes by date range"
)]
pub struct SearchDailyNotesRequest {
    /// Path to vault (CLI only - not used in HTTP/MCP)
    #[arg(index = 1, required = true, help = "Path to vault")]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub vault_path: Option<PathBuf>,

    /// Start date in YYYY-MM-DD format (inclusive)
    #[arg(long, help = "Start date in YYYY-MM-DD format")]
    #[schemars(
        description = "Start date in YYYY-MM-DD format (inclusive). Defaults to 30 days before end_date if not specified."
    )]
    pub start_date: Option<String>,

    /// End date in YYYY-MM-DD format (inclusive)
    #[arg(long, help = "End date in YYYY-MM-DD format")]
    #[schemars(
        description = "End date in YYYY-MM-DD format (inclusive). Defaults to today if not specified."
    )]
    pub end_date: Option<String>,

    /// Maximum number of notes to return
    #[arg(long, help = "Maximum number of notes to return")]
    #[schemars(description = "Maximum number of notes to return (optional, defaults to 100)")]
    pub limit: Option<usize>,

    /// Sort order: asc (oldest first) or desc (newest first)
    #[arg(long, help = "Sort order: asc or desc", default_value = "desc")]
    #[schemars(
        description = "Sort order: 'asc' (oldest first) or 'desc' (newest first). Default: desc"
    )]
    pub sort: Option<String>,

    /// Whether to include note content in results
    #[arg(long, help = "Include full note content in results")]
    #[schemars(
        description = "If true, include full note content for all matching notes. If false, return only metadata. Default: false"
    )]
    pub include_content: Option<bool>,
}

/// Response from the search_daily_notes operation
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SearchDailyNotesResponse {
    /// Daily notes metadata (or full notes if include_content=true)
    pub notes: Vec<DailyNoteResult>,
    /// Total number of notes found
    pub total_count: usize,
    /// Total number of dates in the requested range
    pub dates_searched: usize,
}

/// A daily note result (metadata with optional content)
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct DailyNoteResult {
    /// Date in YYYY-MM-DD format
    pub date: String,
    /// File path relative to vault root
    pub file_path: String,
    /// File name
    pub file_name: String,
    /// Note content (only present if include_content=true)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Error message if reading failed (only present if include_content=true and read failed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Capability for daily note operations
pub struct DailyNoteCapability {
    base_path: PathBuf,
    config: Arc<Config>,
    file_capability: Arc<FileCapability>,
}

impl DailyNoteCapability {
    /// Create a new DailyNoteCapability
    pub fn new(
        base_path: PathBuf,
        config: Arc<Config>,
        file_capability: Arc<FileCapability>,
    ) -> Self {
        Self {
            base_path,
            config,
            file_capability,
        }
    }

    /// Get daily note for a specific date
    pub async fn get_daily_note(
        &self,
        request: GetDailyNoteRequest,
    ) -> CapabilityResult<GetDailyNoteResponse> {
        if !validate_date(&request.date) {
            return Err(invalid_params("Date must be in YYYY-MM-DD format"));
        }

        let base_path = self.base_path.clone();
        let config = Arc::clone(&self.config);
        let date = request.date.clone();

        tracing::debug!(date = %date, "Looking up daily note");

        let relative_path = tokio::task::spawn_blocking(move || {
            find_note_path_blocking(&base_path, &date, &config)
        })
        .await
        .map_err(|e| internal_error(format!("Daily note lookup panicked: {}", e)))?;

        match relative_path {
            Some(path) => {
                // File exists, read its content using FileCapability
                let read_request = ReadFilesRequest {
                    vault_path: None,
                    file_paths: vec![path.clone()],
                    continue_on_error: Some(false),
                };

                let read_response = self
                    .file_capability
                    .read_files(read_request)
                    .await
                    .map_err(|e| internal_error(format!("Failed to read daily note: {}", e)))?;

                if let Some(file_result) = read_response.files.first() {
                    if file_result.success {
                        let file_name = PathBuf::from(&path)
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| path.clone());

                        Ok(GetDailyNoteResponse {
                            found: true,
                            date: request.date,
                            file_path: Some(path),
                            file_name: Some(file_name),
                            content: file_result.content.clone(),
                        })
                    } else {
                        // Reading failed but file exists
                        Ok(GetDailyNoteResponse {
                            found: false,
                            date: request.date,
                            file_path: None,
                            file_name: None,
                            content: None,
                        })
                    }
                } else {
                    // No results (shouldn't happen with fail-fast mode)
                    Ok(GetDailyNoteResponse {
                        found: false,
                        date: request.date,
                        file_path: None,
                        file_name: None,
                        content: None,
                    })
                }
            }
            None => {
                // Note not found - soft error, not exception
                Ok(GetDailyNoteResponse {
                    found: false,
                    date: request.date,
                    file_path: None,
                    file_name: None,
                    content: None,
                })
            }
        }
    }

    /// Search for daily notes in a date range
    pub async fn search_daily_notes(
        &self,
        request: SearchDailyNotesRequest,
    ) -> CapabilityResult<SearchDailyNotesResponse> {
        // Determine date range
        let end_date = request.end_date.unwrap_or_else(today);
        let start_date = request.start_date.unwrap_or_else(|| {
            // Default to 30 days before end_date
            let dates = date_range(&end_date, &end_date);
            if dates.is_empty() {
                end_date.clone()
            } else {
                // Try to go back 30 days
                let all_dates = crate::date_utils::date_range("2000-01-01", &end_date);
                let start_idx = all_dates.len().saturating_sub(30);
                all_dates
                    .get(start_idx)
                    .cloned()
                    .unwrap_or(end_date.clone())
            }
        });

        // Validate dates
        if !validate_date(&start_date) {
            return Err(invalid_params("start_date must be in YYYY-MM-DD format"));
        }
        if !validate_date(&end_date) {
            return Err(invalid_params("end_date must be in YYYY-MM-DD format"));
        }

        // Check date range limit (365 days max)
        let dates = date_range(&start_date, &end_date);
        if dates.is_empty() {
            return Err(invalid_params(
                "Invalid date range: start_date must be <= end_date",
            ));
        }
        if dates.len() > 365 {
            return Err(invalid_params("Date range limited to 365 days"));
        }

        // Determine sort order
        let sort_desc = request.sort.as_deref() != Some("asc");
        let limit = request.limit.unwrap_or(100);
        let include_content = request.include_content.unwrap_or(false);

        let base_path = self.base_path.clone();
        let config = Arc::clone(&self.config);
        let dates_clone = dates.clone();

        tracing::debug!(date_range_days = dates.len(), "Searching daily notes");

        let (mut notes, total_count) = tokio::task::spawn_blocking(move || {
            scan_date_range_blocking(&base_path, &config, &dates_clone, sort_desc, limit)
        })
        .await
        .map_err(|e| internal_error(format!("Daily note search panicked: {}", e)))?;

        tracing::debug!(found = total_count, "Daily note search complete");

        let dates_searched = dates.len();

        // If include_content is true, read all found notes in batch
        if include_content {
            let file_paths: Vec<String> = notes.iter().map(|n| n.file_path.clone()).collect();

            if !file_paths.is_empty() {
                let read_request = ReadFilesRequest {
                    vault_path: None,
                    file_paths: file_paths.clone(),
                    continue_on_error: Some(true),
                };

                match self.file_capability.read_files(read_request).await {
                    Ok(read_response) => {
                        // Map results back to notes
                        let content_map: std::collections::HashMap<
                            String,
                            (bool, Option<String>, Option<String>),
                        > = read_response
                            .files
                            .into_iter()
                            .map(|f| (f.file_path, (f.success, f.content, f.error)))
                            .collect();

                        for note in &mut notes {
                            if let Some((success, content, error)) =
                                content_map.get(&note.file_path)
                            {
                                if *success {
                                    note.content = content.clone();
                                } else {
                                    note.error = error.clone();
                                }
                            }
                        }
                    }
                    Err(_) => {
                        // If batch read fails, notes remain without content
                    }
                }
            }
        }

        Ok(SearchDailyNotesResponse {
            notes,
            total_count,
            dates_searched,
        })
    }
}

fn find_note_path_blocking(
    base_path: &std::path::Path,
    date: &str,
    config: &notectl_core::config::Config,
) -> Option<String> {
    get_daily_note_relative_path(base_path, date, &config.daily_note_patterns, config)
}

fn scan_date_range_blocking(
    base_path: &std::path::Path,
    config: &notectl_core::config::Config,
    dates: &[String],
    sort_desc: bool,
    limit: usize,
) -> (Vec<DailyNoteResult>, usize) {
    let mut found_notes: Vec<DailyNoteResult> = Vec::new();

    for date in dates {
        if let Some(file_path) =
            get_daily_note_relative_path(base_path, date, &config.daily_note_patterns, config)
        {
            let file_name = PathBuf::from(&file_path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| file_path.clone());
            found_notes.push(DailyNoteResult {
                date: date.clone(),
                file_path,
                file_name,
                content: None,
                error: None,
            });
        }
    }

    if sort_desc {
        found_notes.sort_by(|a, b| b.date.cmp(&a.date));
    } else {
        found_notes.sort_by(|a, b| a.date.cmp(&b.date));
    }

    let total_count = found_notes.len();
    if found_notes.len() > limit {
        found_notes.truncate(limit);
    }
    (found_notes, total_count)
}

/// Operation struct for get_daily_note (HTTP, CLI, and MCP)
pub struct GetDailyNoteOperation {
    capability: Arc<DailyNoteCapability>,
}

impl GetDailyNoteOperation {
    pub fn new(capability: Arc<DailyNoteCapability>) -> Self {
        Self { capability }
    }
}

/// Operation struct for search_daily_notes (HTTP, CLI, and MCP)
pub struct SearchDailyNotesOperation {
    capability: Arc<DailyNoteCapability>,
}

impl SearchDailyNotesOperation {
    pub fn new(capability: Arc<DailyNoteCapability>) -> Self {
        Self { capability }
    }
}

#[async_trait::async_trait]
impl notectl_core::operation::Operation for GetDailyNoteOperation {
    fn name(&self) -> &'static str {
        get_daily_note::CLI_NAME
    }

    fn path(&self) -> &'static str {
        get_daily_note::HTTP_PATH
    }

    fn description(&self) -> &'static str {
        get_daily_note::DESCRIPTION
    }

    fn get_command(&self) -> clap::Command {
        GetDailyNoteRequest::command()
    }

    fn get_remote_command(&self) -> clap::Command {
        self.get_command()
            .mut_arg("vault_path", |a| a.required(false).hide(true))
    }

    async fn execute_json(
        &self,
        json: serde_json::Value,
    ) -> Result<serde_json::Value, rmcp::model::ErrorData> {
        let request: GetDailyNoteRequest = serde_json::from_value(json)
            .map_err(|e| notectl_core::error::invalid_params(e.to_string()))?;
        let response = self.capability.get_daily_note(request).await?;
        Ok(serde_json::to_value(response).unwrap())
    }

    async fn execute_from_args(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let request = GetDailyNoteRequest::from_arg_matches(matches)?;

        let response = if let Some(ref vault_path) = request.vault_path {
            let config = Arc::new(Config::load_from_base_path(vault_path.as_path()));
            let file_cap = Arc::new(FileCapability::new(vault_path.clone(), Arc::clone(&config)));
            let capability =
                DailyNoteCapability::new(vault_path.clone(), Arc::clone(&config), file_cap);
            let mut req_without_path = request;
            req_without_path.vault_path = None;
            capability.get_daily_note(req_without_path).await?
        } else {
            self.capability.get_daily_note(request).await?
        };

        Ok(serde_json::to_string_pretty(&response)?)
    }

    fn input_schema(&self) -> serde_json::Value {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(GetDailyNoteRequest)).unwrap()
    }

    fn args_to_json(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut request = GetDailyNoteRequest::from_arg_matches(matches)?;
        request.vault_path = None;
        Ok(serde_json::to_value(request)?)
    }
}

#[async_trait::async_trait]
impl notectl_core::operation::Operation for SearchDailyNotesOperation {
    fn name(&self) -> &'static str {
        search_daily_notes::CLI_NAME
    }

    fn path(&self) -> &'static str {
        search_daily_notes::HTTP_PATH
    }

    fn description(&self) -> &'static str {
        search_daily_notes::DESCRIPTION
    }

    fn get_command(&self) -> clap::Command {
        SearchDailyNotesRequest::command()
    }

    fn get_remote_command(&self) -> clap::Command {
        self.get_command()
            .mut_arg("vault_path", |a| a.required(false).hide(true))
    }

    async fn execute_json(
        &self,
        json: serde_json::Value,
    ) -> Result<serde_json::Value, rmcp::model::ErrorData> {
        let request: SearchDailyNotesRequest = serde_json::from_value(json)
            .map_err(|e| notectl_core::error::invalid_params(e.to_string()))?;
        let response = self.capability.search_daily_notes(request).await?;
        Ok(serde_json::to_value(response).unwrap())
    }

    async fn execute_from_args(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let request = SearchDailyNotesRequest::from_arg_matches(matches)?;

        let response = if let Some(ref vault_path) = request.vault_path {
            let config = Arc::new(Config::load_from_base_path(vault_path.as_path()));
            let file_cap = Arc::new(FileCapability::new(vault_path.clone(), Arc::clone(&config)));
            let capability =
                DailyNoteCapability::new(vault_path.clone(), Arc::clone(&config), file_cap);
            let mut req_without_path = request;
            req_without_path.vault_path = None;
            capability.search_daily_notes(req_without_path).await?
        } else {
            self.capability.search_daily_notes(request).await?
        };

        Ok(serde_json::to_string_pretty(&response)?)
    }

    fn input_schema(&self) -> serde_json::Value {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(SearchDailyNotesRequest)).unwrap()
    }

    fn args_to_json(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut request = SearchDailyNotesRequest::from_arg_matches(matches)?;
        request.vault_path = None;
        Ok(serde_json::to_value(request)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_validate_date() {
        assert!(validate_date("2025-01-20"));
        assert!(validate_date("2024-02-29")); // Leap year
        assert!(!validate_date("2025-02-29")); // Not leap year
        assert!(!validate_date("2025-13-20")); // Invalid month
        assert!(!validate_date("2025-01-32")); // Invalid day
    }

    #[test]
    fn test_get_daily_note_request_validation() {
        // This is just a compile-time check that the struct is valid
        let request = GetDailyNoteRequest {
            vault_path: None,
            date: "2025-01-20".to_string(),
        };
        assert_eq!(request.date, "2025-01-20");
    }

    #[test]
    fn test_search_daily_notes_request_validation() {
        let request = SearchDailyNotesRequest {
            vault_path: None,
            start_date: Some("2025-01-20".to_string()),
            end_date: Some("2025-01-22".to_string()),
            limit: Some(10),
            sort: Some("desc".to_string()),
            include_content: Some(false),
        };
        assert_eq!(request.start_date, Some("2025-01-20".to_string()));
        assert_eq!(request.limit, Some(10));
    }

    #[tokio::test]
    async fn test_get_daily_note_found() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Create test daily note
        fs::write(base_path.join("2025-01-20.md"), "# January 20, 2025").unwrap();

        let config = Arc::new(Config {
            exclude_paths: vec![],
            daily_note_patterns: notectl_core::config::default_daily_note_patterns(),
        });
        let file_cap = Arc::new(FileCapability::new(
            base_path.to_path_buf(),
            Arc::clone(&config),
        ));
        let capability = DailyNoteCapability::new(base_path.to_path_buf(), config, file_cap);

        let request = GetDailyNoteRequest {
            vault_path: None,
            date: "2025-01-20".to_string(),
        };

        let response = capability.get_daily_note(request).await.unwrap();
        assert!(response.found);
        assert_eq!(response.date, "2025-01-20");
        assert_eq!(response.file_path, Some("2025-01-20.md".to_string()));
        assert!(response.content.as_ref().unwrap().contains("January 20"));
    }

    #[tokio::test]
    async fn test_get_daily_note_not_found() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        let config = Arc::new(Config::default());
        let file_cap = Arc::new(FileCapability::new(
            base_path.to_path_buf(),
            Arc::clone(&config),
        ));
        let capability = DailyNoteCapability::new(base_path.to_path_buf(), config, file_cap);

        let request = GetDailyNoteRequest {
            vault_path: None,
            date: "2025-01-20".to_string(),
        };

        let response = capability.get_daily_note(request).await.unwrap();
        assert!(!response.found);
        assert_eq!(response.date, "2025-01-20");
        assert!(response.file_path.is_none());
    }

    #[tokio::test]
    async fn test_search_daily_notes() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Create test daily notes
        fs::write(base_path.join("2025-01-20.md"), "# Jan 20").unwrap();
        fs::write(base_path.join("2025-01-22.md"), "# Jan 22").unwrap();

        let config = Arc::new(Config {
            exclude_paths: vec![],
            daily_note_patterns: notectl_core::config::default_daily_note_patterns(),
        });
        let file_cap = Arc::new(FileCapability::new(
            base_path.to_path_buf(),
            Arc::clone(&config),
        ));
        let capability = DailyNoteCapability::new(base_path.to_path_buf(), config, file_cap);

        let request = SearchDailyNotesRequest {
            vault_path: None,
            start_date: Some("2025-01-20".to_string()),
            end_date: Some("2025-01-22".to_string()),
            limit: Some(100),
            sort: Some("asc".to_string()),
            include_content: Some(false),
        };

        let response = capability.search_daily_notes(request).await.unwrap();
        assert_eq!(response.notes.len(), 2); // Only 2 notes found (Jan 21 doesn't exist)
        assert_eq!(response.total_count, 2); // 2 notes found
        assert_eq!(response.dates_searched, 3); // Searched all 3 days

        // Check sorting (asc) - only found notes returned
        assert_eq!(response.notes[0].date, "2025-01-20");
        assert_eq!(response.notes[1].date, "2025-01-22");
    }

    #[tokio::test]
    async fn test_search_daily_notes_with_content() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        fs::write(base_path.join("2025-01-20.md"), "# Meeting Notes").unwrap();

        let config = Arc::new(Config {
            exclude_paths: vec![],
            daily_note_patterns: notectl_core::config::default_daily_note_patterns(),
        });
        let file_cap = Arc::new(FileCapability::new(
            base_path.to_path_buf(),
            Arc::clone(&config),
        ));
        let capability = DailyNoteCapability::new(base_path.to_path_buf(), config, file_cap);

        let request = SearchDailyNotesRequest {
            vault_path: None,
            start_date: Some("2025-01-20".to_string()),
            end_date: Some("2025-01-20".to_string()),
            limit: Some(100),
            sort: Some("desc".to_string()),
            include_content: Some(true),
        };

        let response = capability.search_daily_notes(request).await.unwrap();
        assert_eq!(response.total_count, 1);
        assert!(
            response.notes[0]
                .content
                .as_ref()
                .unwrap()
                .contains("Meeting Notes")
        );
    }

    #[tokio::test]
    async fn test_search_daily_notes_descending_sort() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        fs::write(base_path.join("2025-01-20.md"), "# Jan 20").unwrap();
        fs::write(base_path.join("2025-01-22.md"), "# Jan 22").unwrap();

        let config = Arc::new(Config {
            exclude_paths: vec![],
            daily_note_patterns: notectl_core::config::default_daily_note_patterns(),
        });
        let file_cap = Arc::new(FileCapability::new(
            base_path.to_path_buf(),
            Arc::clone(&config),
        ));
        let capability = DailyNoteCapability::new(base_path.to_path_buf(), config, file_cap);

        let request = SearchDailyNotesRequest {
            vault_path: None,
            start_date: Some("2025-01-20".to_string()),
            end_date: Some("2025-01-22".to_string()),
            limit: Some(100),
            sort: Some("desc".to_string()),
            include_content: Some(false),
        };

        let response = capability.search_daily_notes(request).await.unwrap();
        assert_eq!(response.notes.len(), 2); // Only 2 found notes
        assert_eq!(response.notes[0].date, "2025-01-22");
        assert_eq!(response.notes[1].date, "2025-01-20");
    }

    #[tokio::test]
    async fn test_search_daily_notes_limit() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Create notes for 5 days
        for day in 20..=24u32 {
            fs::write(
                base_path.join(format!("2025-01-{:02}.md", day)),
                format!("# Jan {}", day),
            )
            .unwrap();
        }

        let config = Arc::new(Config {
            exclude_paths: vec![],
            daily_note_patterns: notectl_core::config::default_daily_note_patterns(),
        });
        let file_cap = Arc::new(FileCapability::new(
            base_path.to_path_buf(),
            Arc::clone(&config),
        ));
        let capability = DailyNoteCapability::new(base_path.to_path_buf(), config, file_cap);

        let request = SearchDailyNotesRequest {
            vault_path: None,
            start_date: Some("2025-01-20".to_string()),
            end_date: Some("2025-01-24".to_string()),
            limit: Some(3),
            sort: Some("desc".to_string()),
            include_content: Some(false),
        };

        let response = capability.search_daily_notes(request).await.unwrap();
        assert_eq!(response.notes.len(), 3); // Limited to 3
        assert_eq!(response.dates_searched, 5); // But searched all 5 days
        assert_eq!(response.total_count, 5); // Found all 5 notes
    }

    #[tokio::test]
    async fn test_search_daily_notes_invalid_date_range() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        let config = Arc::new(Config {
            exclude_paths: vec![],
            daily_note_patterns: notectl_core::config::default_daily_note_patterns(),
        });
        let file_cap = Arc::new(FileCapability::new(
            base_path.to_path_buf(),
            Arc::clone(&config),
        ));
        let capability = DailyNoteCapability::new(base_path.to_path_buf(), config, file_cap);

        let request = SearchDailyNotesRequest {
            vault_path: None,
            start_date: Some("2025-01-22".to_string()),
            end_date: Some("2025-01-20".to_string()),
            limit: Some(100),
            sort: Some("desc".to_string()),
            include_content: Some(false),
        };

        let result = capability.search_daily_notes(request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_search_daily_notes_date_range_too_large() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        let config = Arc::new(Config {
            exclude_paths: vec![],
            daily_note_patterns: notectl_core::config::default_daily_note_patterns(),
        });
        let file_cap = Arc::new(FileCapability::new(
            base_path.to_path_buf(),
            Arc::clone(&config),
        ));
        let capability = DailyNoteCapability::new(base_path.to_path_buf(), config, file_cap);

        // Try to search 400 days (exceeds 365 limit)
        let request = SearchDailyNotesRequest {
            vault_path: None,
            start_date: Some("2024-01-01".to_string()),
            end_date: Some("2025-02-05".to_string()),
            limit: Some(100),
            sort: Some("desc".to_string()),
            include_content: Some(false),
        };

        let result = capability.search_daily_notes(request).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("365 days"));
    }
}
