use crate::config::Config;
use std::fs;
use std::path::{Path, PathBuf};

/// Recursively collect all markdown files in a directory, respecting config exclusions.
pub fn collect_markdown_files(
    dir: &Path,
    config: &Config,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    collect_recursive(dir, config, &mut files)?;
    Ok(files)
}

fn collect_recursive(
    dir: &Path,
    config: &Config,
    files: &mut Vec<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !dir.is_dir() {
        return Ok(());
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        // Skip excluded paths
        if config.should_exclude(&path) {
            continue;
        }

        if path.is_dir() {
            collect_recursive(&path, config, files)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("md") {
            files.push(path);
        }
    }

    Ok(())
}
