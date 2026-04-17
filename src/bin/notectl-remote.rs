use notectl_core::config::Config;
use notectl_core::operation::Operation;
use notectl_daily_notes::{DailyNoteCapability, GetDailyNoteOperation, SearchDailyNotesOperation};
use notectl_files::{FileCapability, ListFilesOperation, ReadFilesOperation};
use notectl_outline::{
    GetOutlineOperation, GetSectionOperation, OutlineCapability, SearchHeadingsOperation,
};
use notectl_tags::{ExtractTagsOperation, ListTagsOperation, SearchByTagsOperation, TagCapability};
use notectl_tasks::{SearchTasksOperation, TaskCapability};
use std::path::PathBuf;
use std::sync::Arc;

fn create_operations() -> Vec<Arc<dyn Operation>> {
    let config = Arc::new(Config::default());
    // Use a placeholder path — capabilities won't execute locally in this binary
    let base_path = PathBuf::from(".");

    let file_capability = Arc::new(FileCapability::new(base_path.clone(), Arc::clone(&config)));
    let daily_note_capability = Arc::new(DailyNoteCapability::new(
        base_path.clone(),
        Arc::clone(&config),
        Arc::clone(&file_capability),
    ));

    vec![
        Arc::new(SearchTasksOperation::new(Arc::new(TaskCapability::new(
            base_path.clone(),
            Arc::clone(&config),
        )))),
        Arc::new(ExtractTagsOperation::new(Arc::new(TagCapability::new(
            base_path.clone(),
            Arc::clone(&config),
        )))),
        Arc::new(ListTagsOperation::new(Arc::new(TagCapability::new(
            base_path.clone(),
            Arc::clone(&config),
        )))),
        Arc::new(SearchByTagsOperation::new(Arc::new(TagCapability::new(
            base_path.clone(),
            Arc::clone(&config),
        )))),
        Arc::new(ListFilesOperation::new(Arc::clone(&file_capability))),
        Arc::new(ReadFilesOperation::new(Arc::clone(&file_capability))),
        Arc::new(GetDailyNoteOperation::new(Arc::clone(
            &daily_note_capability,
        ))),
        Arc::new(SearchDailyNotesOperation::new(Arc::clone(
            &daily_note_capability,
        ))),
        Arc::new(GetOutlineOperation::new(Arc::new(OutlineCapability::new(
            base_path.clone(),
            Arc::clone(&config),
        )))),
        Arc::new(GetSectionOperation::new(Arc::new(OutlineCapability::new(
            base_path.clone(),
            Arc::clone(&config),
        )))),
        Arc::new(SearchHeadingsOperation::new(Arc::new(
            OutlineCapability::new(base_path, Arc::clone(&config)),
        ))),
    ]
}

fn build_cli(operations: &[Arc<dyn Operation>]) -> clap::Command {
    let mut cmd = clap::Command::new("notectl-remote")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Send notectl commands to a remote HTTP server")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .arg(
            clap::Arg::new("server")
                .long("server")
                .short('s')
                .env("NOTECTL_SERVER")
                .help("Remote server URL (e.g. http://host:8000)")
                .global(true),
        );

    for operation in operations {
        // Exclude operations with no HTTP path (e.g. serve)
        if !operation.path().is_empty() {
            cmd = cmd.subcommand(operation.get_remote_command());
        }
    }

    cmd = cmd.subcommand(notectl::prime::command());

    cmd
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let operations = create_operations();
    let cli = build_cli(&operations);
    let matches = cli.get_matches();

    if let Some(("prime", _)) = matches.subcommand() {
        print!("{}", notectl::prime::generate_for_current_binary());
        return Ok(());
    }

    let server = matches
        .get_one::<String>("server")
        .ok_or("Server URL is required. Use --server <url> or set NOTECTL_SERVER.")?
        .clone();

    let server = server.trim_end_matches('/').to_string();

    if let Some((subcommand_name, sub_matches)) = matches.subcommand() {
        for operation in &operations {
            if operation.name() == subcommand_name {
                let json = operation.args_to_json(sub_matches)?;
                let url = format!("{}{}", server, operation.path());

                let client = reqwest::Client::new();
                let response = client
                    .post(&url)
                    .json(&json)
                    .send()
                    .await
                    .map_err(|e| format!("Failed to connect to {}: {}", url, e))?;

                let status = response.status();
                let body = response.text().await?;

                if status.is_success() {
                    println!("{}", body);
                } else {
                    eprintln!("Error {}: {}", status, body);
                    std::process::exit(1);
                }

                return Ok(());
            }
        }
        return Err(format!("Unknown command: {}", subcommand_name).into());
    }

    Err("No command specified".into())
}
