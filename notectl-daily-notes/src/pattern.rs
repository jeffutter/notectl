//! Pattern matching and file discovery for daily notes
//!
//! Handles pattern substitution (YYYY/MM/DD) and file discovery with security checks.

use crate::date_utils::parse_date;
use notectl_core::config::Config;
use std::path::{Path, PathBuf};

/// Apply a pattern by substituting YYYY, MM, DD placeholders with date components
///
/// Example: "YYYY/MM/DD.md" with date "2025-01-20" → "2025/01/20.md"
pub fn apply_pattern(pattern: &str, date: &str) -> Option<String> {
    let (year, month, day) = parse_date(date)?;

    let result = pattern
        .replace("YYYY", &format!("{:04}", year))
        .replace("MM", &format!("{:02}", month))
        .replace("DD", &format!("{:02}", day));

    Some(result)
}

/// Find a daily note file for a specific date
///
/// Tries each configured pattern in order and returns the first match.
/// If multiple patterns match different files, returns an error.
pub fn find_daily_note(
    base_path: &Path,
    date: &str,
    patterns: &[String],
    config: &Config,
) -> Result<Option<PathBuf>, String> {
    let mut found_paths: Vec<PathBuf> = Vec::new();

    for pattern in patterns {
        let substituted =
            apply_pattern(pattern, date).ok_or_else(|| format!("Invalid pattern: {}", pattern))?;

        let full_path = base_path.join(&substituted);

        // Check if file exists
        if full_path.exists() && full_path.is_file() {
            // Check if path should be excluded
            let relative_path = full_path.strip_prefix(base_path).unwrap_or(&full_path);
            if !config.should_exclude(relative_path) {
                // Security check: ensure path is within base directory
                match full_path.canonicalize() {
                    Ok(canonical_path) => {
                        let canonical_base = base_path
                            .canonicalize()
                            .map_err(|e| format!("Failed to resolve base path: {}", e))?;

                        if canonical_path.starts_with(&canonical_base) {
                            found_paths.push(full_path);
                        }
                    }
                    Err(_) => {
                        // Skip files that can't be canonicalized
                        continue;
                    }
                }
            }
        }
    }

    match found_paths.len() {
        0 => Ok(None),
        1 => Ok(Some(found_paths[0].clone())),
        _ => Err(format!(
            "Multiple daily notes found for date {}: {:?}",
            date, found_paths
        )),
    }
}

/// Get the relative path for a daily note (for use in FileCapability)
///
/// Returns None if no file is found
pub fn get_daily_note_relative_path(
    base_path: &Path,
    date: &str,
    patterns: &[String],
    config: &Config,
) -> Option<String> {
    let full_path = find_daily_note(base_path, date, patterns, config).ok()??;

    full_path
        .strip_prefix(base_path)
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_apply_pattern() {
        assert_eq!(
            apply_pattern("YYYY-MM-DD.md", "2025-01-20"),
            Some("2025-01-20.md".to_string())
        );
        assert_eq!(
            apply_pattern("Daily/YYYY/MM-DD.md", "2025-01-20"),
            Some("Daily/2025/01-20.md".to_string())
        );
        assert_eq!(
            apply_pattern("YYYY/MM/DD.md", "2025-01-20"),
            Some("2025/01/20.md".to_string())
        );
        assert_eq!(apply_pattern("YYYY-MM-DD.md", "invalid"), None);
    }

    #[test]
    fn test_find_daily_note() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Create test files
        fs::write(base_path.join("2025-01-20.md"), "# Jan 20").unwrap();
        fs::write(base_path.join("2025-01-21.md"), "# Jan 21").unwrap();

        let config = Config::default();
        let patterns = vec!["YYYY-MM-DD.md".to_string()];

        let result = find_daily_note(base_path, "2025-01-20", &patterns, &config);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some(base_path.join("2025-01-20.md")));

        let result = find_daily_note(base_path, "2025-01-22", &patterns, &config);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn test_find_daily_note_multiple_patterns() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Create test files in different patterns
        fs::write(base_path.join("2025-01-20.md"), "# Jan 20").unwrap();

        let daily_dir = base_path.join("Daily");
        fs::create_dir(&daily_dir).unwrap();
        fs::write(daily_dir.join("2025-01-21.md"), "# Jan 21").unwrap();

        let config = Config::default();
        let patterns = vec![
            "YYYY-MM-DD.md".to_string(),
            "Daily/YYYY-MM-DD.md".to_string(),
        ];

        let result = find_daily_note(base_path, "2025-01-20", &patterns, &config);
        assert_eq!(result.unwrap(), Some(base_path.join("2025-01-20.md")));

        let result = find_daily_note(base_path, "2025-01-21", &patterns, &config);
        assert_eq!(result.unwrap(), Some(daily_dir.join("2025-01-21.md")));
    }

    #[test]
    fn test_find_daily_note_with_exclusion() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Create test files, one in an excluded directory
        fs::write(base_path.join("2025-01-20.md"), "# Jan 20").unwrap();

        let archive_dir = base_path.join("Archive");
        fs::create_dir(&archive_dir).unwrap();
        fs::write(archive_dir.join("2025-01-20.md"), "# Archived Jan 20").unwrap();

        let config = Config {
            exclude_paths: vec!["Archive".to_string()],
            ..Default::default()
        };
        let patterns = vec![
            "YYYY-MM-DD.md".to_string(),
            "Archive/YYYY-MM-DD.md".to_string(),
        ];

        let result = find_daily_note(base_path, "2025-01-20", &patterns, &config);
        // Should find only the non-excluded one
        assert_eq!(result.unwrap(), Some(base_path.join("2025-01-20.md")));
    }

    #[test]
    fn test_find_daily_note_multiple_matches_error() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Create two files that both match different patterns for the same date
        fs::write(base_path.join("2025-01-20.md"), "# Jan 20 v1").unwrap();
        fs::write(base_path.join("2025_01_20.md"), "# Jan 20 v2").unwrap();

        let config = Config::default();
        let patterns = vec!["YYYY-MM-DD.md".to_string(), "YYYY_MM_DD.md".to_string()];

        let result = find_daily_note(base_path, "2025-01-20", &patterns, &config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Multiple daily notes found"));
    }

    #[test]
    fn test_get_daily_note_relative_path() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        fs::write(base_path.join("2025-01-20.md"), "# Jan 20").unwrap();

        let config = Config::default();
        let patterns = vec!["YYYY-MM-DD.md".to_string()];

        assert_eq!(
            get_daily_note_relative_path(base_path, "2025-01-20", &patterns, &config),
            Some("2025-01-20.md".to_string())
        );

        assert_eq!(
            get_daily_note_relative_path(base_path, "2025-01-21", &patterns, &config),
            None
        );
    }
}
