mod capabilities;
mod cli;
mod cli_router;
mod http_router;
mod mcp;

use clap::FromArgMatches;
use cli::{ServeCommand, ServerMode};
use mcp::TaskSearchService;
use rmcp::{
    ServiceExt,
    transport::{stdio, streamable_http_server::session::local::LocalSessionManager},
};
use std::sync::Arc;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

async fn tools_handler(
    axum::extract::State(registry): axum::extract::State<Arc<capabilities::CapabilityRegistry>>,
) -> impl axum::response::IntoResponse {
    use axum::Json;
    use serde_json::json;

    // Get all operations from the registry
    let operations = registry.create_operations();

    // Build the tools array dynamically from operations
    let tools: Vec<_> = operations
        .into_iter()
        .map(|op| {
            json!({
                "name": op.name(),
                "description": op.description(),
                "input_schema": op.input_schema()
            })
        })
        .collect();

    Json(json!({ "tools": tools }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use capabilities::CapabilityRegistry;
    use notectl_core::config::Config;
    use std::path::PathBuf;

    // Create a minimal registry (base path will come from the parsed request or command)
    let config = Arc::new(Config::default());
    let registry = CapabilityRegistry::new(PathBuf::from("."), config);

    // Get all operations including serve
    let mut operations = registry.create_operations();
    operations.push(Arc::new(cli::ServeOperation::new()));

    // Build CLI from operations
    let cli = cli_router::build_cli(&operations);

    // Parse command line arguments
    let matches = cli.get_matches();

    // Check if this is the serve command
    if let Some(("serve", serve_matches)) = matches.subcommand() {
        // Parse the serve command
        let serve_cmd = ServeCommand::from_arg_matches(serve_matches)?;
        let base_path = serve_cmd.mode.path().clone();

        match serve_cmd.mode {
            ServerMode::Stdio { .. } => {
                // Start stdio MCP server
                let service = TaskSearchService::new(base_path).serve(stdio()).await?;

                // Wait for either service completion or Ctrl-C
                tokio::select! {
                    result = service.waiting() => {
                        result?;
                    }
                    _ = tokio::signal::ctrl_c() => {
                        eprintln!("Received Ctrl-C, shutting down...");
                    }
                }

                return Ok(());
            }
            ServerMode::Http { port, .. } => {
                // Start HTTP MCP server
                use rmcp::transport::streamable_http_server::{
                    StreamableHttpServerConfig, StreamableHttpService,
                };
                use tokio_util::sync::CancellationToken;

                let ct = CancellationToken::new();
                let base_path_clone = base_path.clone();
                let service = StreamableHttpService::new(
                    move || Ok(TaskSearchService::new(base_path_clone.clone())),
                    Arc::new(LocalSessionManager::default()),
                    StreamableHttpServerConfig {
                        cancellation_token: ct.clone(),
                        ..Default::default()
                    },
                );

                // Load configuration from base path
                let config = Arc::new(Config::load_from_base_path(&base_path));

                // Create capability registry
                let capability_registry = Arc::new(capabilities::CapabilityRegistry::new(
                    base_path.clone(),
                    config.clone(),
                ));

                // Create router with base routes and state
                let mut router = axum::Router::new()
                    .nest_service("/mcp", service)
                    .route("/tools", axum::routing::get(tools_handler))
                    .with_state(capability_registry.clone());

                // Automatically register all HTTP operations
                for operation in capability_registry.create_operations() {
                    router = http_router::register_operation(router, operation);
                }

                let addr = format!("0.0.0.0:{}", port);
                let listener = tokio::net::TcpListener::bind(&addr).await?;

                eprintln!("HTTP MCP server listening on http://{}/mcp", addr);
                eprintln!("Tools documentation available at http://{}/tools", addr);
                eprintln!("REST API available at:");

                // Dynamically print all registered operations
                for operation in capability_registry.create_operations() {
                    eprintln!(
                        "  - GET/POST http://{}{} ({})",
                        addr,
                        operation.path(),
                        operation.description()
                    );
                }

                axum::serve(listener, router.into_make_service())
                    .with_graceful_shutdown(async move {
                        tokio::signal::ctrl_c().await.ok();
                        eprintln!("Received Ctrl-C, shutting down...");
                        ct.cancel();
                    })
                    .await?;

                return Ok(());
            }
        }
    }

    // For all other commands, use the cli_router
    cli_router::execute_cli(&operations, matches).await
}
