use std::sync::Arc;

/// Build a clap Command dynamically from all registered operations
pub fn build_cli(operations: &[Arc<dyn notectl_core::operation::Operation>]) -> clap::Command {
    let mut cmd = clap::Command::new("notectl")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Extract todo items from Markdown files")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .arg(
            clap::Arg::new("path")
                .long("path")
                .short('p')
                .help("Base path to search (defaults to current directory)")
                .global(true),
        );

    // Add each operation's command definition
    for operation in operations {
        cmd = cmd.subcommand(operation.get_command());
    }

    cmd = cmd.subcommand(crate::prime::command());

    cmd
}

/// Execute CLI command by routing to the appropriate operation
pub async fn execute_cli(
    operations: &[Arc<dyn notectl_core::operation::Operation>],
    matches: clap::ArgMatches,
) -> Result<(), Box<dyn std::error::Error>> {
    // Find the matching operation
    if let Some((subcommand_name, sub_matches)) = matches.subcommand() {
        for operation in operations {
            if operation.name() == subcommand_name {
                let output = operation.execute_from_args(sub_matches).await?;
                println!("{}", output);
                return Ok(());
            }
        }
        return Err(format!("Unknown command: {}", subcommand_name).into());
    }

    Err("No command specified".into())
}
