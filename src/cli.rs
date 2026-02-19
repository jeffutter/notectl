use clap::{CommandFactory, FromArgMatches, Parser, Subcommand as ClapSubcommand};
use rmcp::model::ErrorData;
use std::path::PathBuf;

/// Server mode options for MCP
#[derive(Debug, Clone, ClapSubcommand)]
pub enum ServerMode {
    /// Start MCP server on stdin/stdout
    Stdio {
        /// Path to file or folder to scan (base path for server)
        #[arg(index = 1, required = true)]
        path: PathBuf,
    },
    /// Start MCP server on HTTP
    Http {
        /// Path to file or folder to scan (base path for server)
        #[arg(index = 1, required = true)]
        path: PathBuf,

        /// Port for HTTP MCP server
        #[arg(long, default_value = "8000")]
        port: u16,
    },
}

impl ServerMode {
    pub fn path(&self) -> &PathBuf {
        match self {
            ServerMode::Stdio { path } => path,
            ServerMode::Http { path, .. } => path,
        }
    }
}

/// Start MCP or HTTP server
#[derive(Parser, Debug)]
#[command(name = "serve", about = "Start MCP server (stdio or HTTP)")]
pub struct ServeCommand {
    /// Server mode (stdio or http)
    #[command(subcommand)]
    pub mode: ServerMode,
}

/// CliOperation implementation for serve command
pub struct ServeOperation;

impl ServeOperation {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl notectl_core::operation::Operation for ServeOperation {
    fn name(&self) -> &'static str {
        "serve"
    }

    fn path(&self) -> &'static str {
        // ServeOperation is CLI-only and doesn't have an HTTP endpoint
        ""
    }

    fn description(&self) -> &'static str {
        "Start MCP server (stdio or HTTP)"
    }

    fn get_command(&self) -> clap::Command {
        ServeCommand::command()
    }

    async fn execute_json(&self, _json: serde_json::Value) -> Result<serde_json::Value, ErrorData> {
        // ServeOperation is CLI-only and doesn't support JSON execution
        Err(ErrorData {
            code: rmcp::model::ErrorCode(-32601),
            message: std::borrow::Cow::from("serve command is only available via CLI"),
            data: None,
        })
    }

    async fn execute_from_args(
        &self,
        matches: &clap::ArgMatches,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let _cmd = ServeCommand::from_arg_matches(matches)?;

        // This will be handled specially in main.rs since it needs to start a server
        // For now, return an error indicating this should be handled specially
        Err("serve command must be handled by main.rs".into())
    }

    fn input_schema(&self) -> serde_json::Value {
        // ServeOperation is CLI-only and doesn't have a meaningful JSON schema
        // Return a minimal empty object schema
        serde_json::json!({
            "type": "object",
            "properties": {}
        })
    }
}
