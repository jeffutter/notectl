use async_trait::async_trait;
use rmcp::model::ErrorData;
use std::error::Error;

/// Unified trait for operations that can be exposed via HTTP, CLI, or MCP
///
/// This trait combines the functionality of HttpOperation and CliOperation,
/// providing a single interface for all operation types. Operations implement
/// this trait once and can be automatically registered for all interfaces.
#[async_trait]
pub trait Operation: Send + Sync + 'static {
    /// Unique identifier for the CLI command (e.g., "tasks", "list-tags")
    fn name(&self) -> &'static str;

    /// HTTP path for this operation (e.g., "/api/tasks")
    fn path(&self) -> &'static str;

    /// Human-readable description of the operation
    fn description(&self) -> &'static str;

    /// Get the clap Command definition for CLI parsing
    ///
    /// This is typically derived from the request struct's `Parser` implementation.
    fn get_command(&self) -> clap::Command;

    /// Get the clap Command definition for remote CLI parsing
    ///
    /// Unlike `get_command`, this omits the vault/path positional argument since
    /// the remote server already has its vault path configured. Defaults to
    /// `get_command()` — operations with a vault path positional arg should override.
    fn get_remote_command(&self) -> clap::Command {
        self.get_command()
    }

    /// Execute the operation with JSON input (for HTTP/MCP)
    ///
    /// This method performs type erasure by accepting and returning JSON values,
    /// allowing dynamic dispatch across different operation types.
    async fn execute_json(&self, json: serde_json::Value) -> Result<serde_json::Value, ErrorData>;

    /// Execute the operation from parsed CLI arguments
    ///
    /// This method receives:
    /// - matches: The clap ArgMatches for this subcommand
    ///
    /// Returns output string (typically JSON)
    async fn execute_from_args(&self, matches: &clap::ArgMatches)
    -> Result<String, Box<dyn Error>>;

    /// Get the JSON Schema for this operation's input
    ///
    /// Returns the schema as a serde_json::Value for easy serialization.
    /// Implementations should use schemars::schema_for! on their request type.
    fn input_schema(&self) -> serde_json::Value;

    /// Parse CLI arguments into a JSON value suitable for HTTP/MCP execution
    ///
    /// Parses the request struct from ArgMatches, clears any CLI-only fields
    /// (e.g., path/vault_path), and serializes to JSON. This lets a remote
    /// client reuse CLI argument parsing without local execution.
    fn args_to_json(&self, matches: &clap::ArgMatches)
    -> Result<serde_json::Value, Box<dyn Error>>;
}
