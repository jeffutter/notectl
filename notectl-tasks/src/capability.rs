use crate::extractor::{Task, TaskExtractor};
use crate::filter::{FilterOptions, filter_tasks};
use clap::{CommandFactory, FromArgMatches, Parser};
use notectl_core::CapabilityResult;
use notectl_core::config::Config;
use notectl_core::error::internal_error;
use rmcp::model::ErrorData;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

/// Operation metadata for search_tasks
pub mod search_tasks {
    pub const DESCRIPTION: &str =
        "Search for tasks in Markdown files with optional filtering by status, dates, and tags";
    #[allow(dead_code)]
    pub const CLI_NAME: &str = "tasks";
    pub const HTTP_PATH: &str = "/api/tasks";
}

/// Parameters for the search_tasks operation
#[derive(Debug, Deserialize, Serialize, JsonSchema, Parser)]
#[command(
    name = "tasks",
    about = "Search for tasks in Markdown files with optional filtering"
)]
pub struct SearchTasksRequest {
    /// Path to scan (CLI only - not used in HTTP/MCP)
    #[arg(index = 1, required = true, help = "Path to file or folder to scan")]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub path: Option<PathBuf>,

    #[arg(long)]
    #[schemars(description = "Filter by task status (incomplete, completed, cancelled)")]
    pub status: Option<String>,

    #[arg(long, help = "Filter by exact due date (YYYY-MM-DD)")]
    #[schemars(description = "Filter by exact due date (YYYY-MM-DD)")]
    pub due_on: Option<String>,

    #[arg(long, help = "Filter tasks due before date (YYYY-MM-DD)")]
    #[schemars(description = "Filter tasks due before date (YYYY-MM-DD)")]
    pub due_before: Option<String>,

    #[arg(long, help = "Filter tasks due after date (YYYY-MM-DD)")]
    #[schemars(description = "Filter tasks due after date (YYYY-MM-DD)")]
    pub due_after: Option<String>,

    #[arg(long, help = "Filter tasks completed on a specific date (YYYY-MM-DD)")]
    #[schemars(description = "Filter tasks completed on a specific date (YYYY-MM-DD)")]
    pub completed_on: Option<String>,

    #[arg(
        long,
        help = "Filter tasks completed before a specific date (YYYY-MM-DD)"
    )]
    #[schemars(description = "Filter tasks completed before a specific date (YYYY-MM-DD)")]
    pub completed_before: Option<String>,

    #[arg(
        long,
        help = "Filter tasks completed after a specific date (YYYY-MM-DD)"
    )]
    #[schemars(description = "Filter tasks completed after a specific date (YYYY-MM-DD)")]
    pub completed_after: Option<String>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Filter by tags (must have all specified tags)"
    )]
    #[schemars(description = "Filter by tags (must have all specified tags)")]
    pub tags: Option<Vec<String>>,

    #[arg(long, value_delimiter = ',', help = "Exclude tasks with these tags")]
    #[schemars(description = "Exclude tasks with these tags (must not have any)")]
    pub exclude_tags: Option<Vec<String>>,

    #[arg(long, help = "Limit the number of tasks returned")]
    #[schemars(description = "Limit the number of tasks returned")]
    pub limit: Option<usize>,
}

/// Response from the search_tasks operation
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct TaskSearchResponse {
    pub tasks: Vec<Task>,
}

/// Capability for task operations (search, filter, extract)
pub struct TaskCapability {
    base_path: PathBuf,
    task_extractor: Arc<TaskExtractor>,
}

impl TaskCapability {
    /// Create a new TaskCapability
    pub fn new(base_path: PathBuf, config: Arc<Config>) -> Self {
        Self {
            base_path,
            task_extractor: Arc::new(TaskExtractor::new(config)),
        }
    }

    /// Search for tasks with optional filtering
    pub async fn search_tasks(
        &self,
        request: SearchTasksRequest,
    ) -> CapabilityResult<TaskSearchResponse> {
        // Extract tasks from the base path using the pre-compiled extractor
        let tasks = self
            .task_extractor
            .extract_tasks(&self.base_path)
            .map_err(|e| internal_error(format!("Failed to extract tasks: {}", e)))?;

        // Apply filters
        let filter_options = FilterOptions {
            status: request.status,
            due_on: request.due_on,
            due_before: request.due_before,
            due_after: request.due_after,
            completed_on: request.completed_on,
            completed_before: request.completed_before,
            completed_after: request.completed_after,
            tags: request.tags,
            exclude_tags: request.exclude_tags,
        };
        let mut filtered_tasks = filter_tasks(tasks, &filter_options);

        // Apply limit (use provided limit, or default from env/50)
        let limit = request.limit.unwrap_or_else(get_default_limit);
        filtered_tasks.truncate(limit);

        Ok(TaskSearchResponse {
            tasks: filtered_tasks,
        })
    }
}

/// Get the default limit for task results
/// Reads from NOTECTL_DEFAULT_LIMIT env var, defaults to 50
fn get_default_limit() -> usize {
    std::env::var("NOTECTL_DEFAULT_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50)
}

/// Operation struct for search_tasks (HTTP, CLI, and MCP)
pub struct SearchTasksOperation {
    capability: Arc<TaskCapability>,
}

impl SearchTasksOperation {
    pub fn new(capability: Arc<TaskCapability>) -> Self {
        Self { capability }
    }
}

#[async_trait::async_trait]
impl notectl_core::operation::Operation for SearchTasksOperation {
    fn name(&self) -> &'static str {
        search_tasks::CLI_NAME
    }

    fn path(&self) -> &'static str {
        search_tasks::HTTP_PATH
    }

    fn description(&self) -> &'static str {
        search_tasks::DESCRIPTION
    }

    fn get_command(&self) -> clap::Command {
        SearchTasksRequest::command()
    }

    async fn execute_json(&self, json: serde_json::Value) -> Result<serde_json::Value, ErrorData> {
        let request: SearchTasksRequest = serde_json::from_value(json)
            .map_err(|e| notectl_core::error::invalid_params(e.to_string()))?;
        let response = self.capability.search_tasks(request).await?;
        Ok(serde_json::to_value(response).unwrap())
    }

    async fn execute_from_args(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let request = SearchTasksRequest::from_arg_matches(matches)?;

        let response = if let Some(ref path) = request.path {
            let config = Arc::new(Config::load_from_base_path(path.as_path()));
            let capability = TaskCapability::new(path.clone(), config);
            let mut req_without_path = request;
            req_without_path.path = None;
            capability.search_tasks(req_without_path).await?
        } else {
            self.capability.search_tasks(request).await?
        };

        Ok(serde_json::to_string_pretty(&response.tasks)?)
    }

    fn input_schema(&self) -> serde_json::Value {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(SearchTasksRequest)).unwrap()
    }

    fn args_to_json(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut request = SearchTasksRequest::from_arg_matches(matches)?;
        request.path = None;
        Ok(serde_json::to_value(request)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use notectl_core::operation::Operation;
    use std::sync::Arc;

    fn make_operation() -> SearchTasksOperation {
        // Capability is not invoked in args_to_json, so a dummy path is fine
        let config = Arc::new(notectl_core::config::Config::default());
        let capability = Arc::new(TaskCapability::new(
            std::path::PathBuf::from("/tmp"),
            config,
        ));
        SearchTasksOperation::new(capability)
    }

    #[test]
    fn args_to_json_strips_path_and_preserves_filters() {
        let op = make_operation();
        // Simulate: notectl tasks /some/vault --status incomplete --limit 10
        let cmd = SearchTasksRequest::command();
        let matches = cmd
            .try_get_matches_from([
                "tasks",
                "/some/vault",
                "--status",
                "incomplete",
                "--limit",
                "10",
            ])
            .expect("parse failed");
        let json = op.args_to_json(&matches).expect("args_to_json failed");

        // path should not appear in output (it's CLI-only)
        assert!(json.get("path").is_none(), "path should be stripped");
        // filters should be present
        assert_eq!(json["status"], "incomplete");
        assert_eq!(json["limit"], 10);
    }

    #[test]
    fn args_to_json_minimal_args() {
        let op = make_operation();
        let cmd = SearchTasksRequest::command();
        let matches = cmd
            .try_get_matches_from(["tasks", "/vault"])
            .expect("parse failed");
        let json = op.args_to_json(&matches).expect("args_to_json failed");

        assert!(json.get("path").is_none());
        // Optional filters absent → serialized as null or absent
        assert!(json.get("status").is_none() || json["status"].is_null());
    }
}
