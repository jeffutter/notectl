use notectl_core::config::Config;
use notectl_core::{CapabilityResult, error::internal_error};
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Extractor for YAML frontmatter tags
#[derive(Clone)]
pub struct TagExtractor {
    config: Arc<Config>,
}

/// Tag with occurrence statistics
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TagCount {
    /// The tag name (without # prefix)
    pub tag: String,
    /// Number of documents containing this tag
    pub document_count: usize,
}

/// Represents a file that matches tag search criteria
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TaggedFile {
    /// Absolute path to the file
    pub file_path: String,
    /// File name without path
    pub file_name: String,
    /// Tags that matched the search criteria
    pub matched_tags: Vec<String>,
    /// All tags found in the file's frontmatter
    pub all_tags: Vec<String>,
}

/// Recursively collect all markdown files in a directory
fn collect_markdown_files(
    dir: &Path,
    config: &Config,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut files = Vec::new();

    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            // Skip excluded paths
            if config.should_exclude(&path) {
                continue;
            }

            if path.is_dir() {
                files.extend(collect_markdown_files(&path, config)?);
            } else if path.extension().and_then(|s| s.to_str()) == Some("md") {
                files.push(path);
            }
        }
    }

    Ok(files)
}

impl TagExtractor {
    pub fn new(config: Arc<Config>) -> Self {
        Self { config }
    }

    pub async fn extract_tags(&self, path: &Path) -> CapabilityResult<Vec<String>> {
        let me = self.clone();
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || {
            me.extract_tags_blocking(&path)
                .map_err(|e| internal_error(format!("Failed to extract tags: {}", e)))
        })
        .await
        .map_err(|e| internal_error(format!("Tag extraction panicked: {}", e)))?
    }

    fn extract_tags_blocking(
        &self,
        path: &Path,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let files = if path.is_file() {
            vec![path.to_path_buf()]
        } else {
            collect_markdown_files(path, &self.config)?
        };

        // Use a BTreeSet to automatically sort and deduplicate tags
        let tags: BTreeSet<String> = files
            .par_iter()
            .filter_map(|file_path| self.extract_tags_from_file(file_path).ok())
            .flatten()
            .collect();

        Ok(tags.into_iter().collect())
    }

    /// Extract tags from a single markdown file
    fn extract_tags_from_file(
        &self,
        file_path: &Path,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(file_path)?;
        self.extract_tags_from_content(&content)
    }

    /// Extract tags from markdown content by parsing YAML frontmatter
    fn extract_tags_from_content(
        &self,
        content: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let frontmatter = self.extract_frontmatter(content)?;

        if let Some(fm) = frontmatter {
            self.parse_tags_from_frontmatter(&fm)
        } else {
            Ok(vec![])
        }
    }

    /// Extract YAML frontmatter from markdown content
    /// Frontmatter is expected to be at the start of the file between --- delimiters
    fn extract_frontmatter(
        &self,
        content: &str,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        let lines: Vec<&str> = content.lines().collect();

        // Check if the file starts with ---
        if lines.is_empty() || lines[0].trim() != "---" {
            return Ok(None);
        }

        // Find the closing ---
        let mut end_index = None;
        for (i, line) in lines.iter().enumerate().skip(1) {
            if line.trim() == "---" {
                end_index = Some(i);
                break;
            }
        }

        if let Some(end) = end_index {
            let frontmatter_lines = &lines[1..end];
            Ok(Some(frontmatter_lines.join("\n")))
        } else {
            Ok(None)
        }
    }

    /// Parse tags from YAML frontmatter
    fn parse_tags_from_frontmatter(
        &self,
        frontmatter: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        // Parse YAML frontmatter
        let yaml: serde_yaml::Value = serde_yaml::from_str(frontmatter)?;

        // Extract tags field
        if let Some(tags_value) = yaml.get("tags") {
            match tags_value {
                // Handle array of tags
                serde_yaml::Value::Sequence(seq) => {
                    let tags: Vec<String> = seq
                        .iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .filter(|s| !s.trim().is_empty())
                        .collect();
                    Ok(tags)
                }
                // Handle single tag as string
                serde_yaml::Value::String(s) => {
                    if s.trim().is_empty() {
                        Ok(vec![])
                    } else {
                        Ok(vec![s.clone()])
                    }
                }
                _ => Ok(vec![]),
            }
        } else {
            Ok(vec![])
        }
    }

    pub async fn extract_tags_with_counts(&self, path: &Path) -> CapabilityResult<Vec<TagCount>> {
        let me = self.clone();
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || {
            me.extract_tags_with_counts_blocking(&path)
                .map_err(|e| internal_error(format!("Failed to list tags: {}", e)))
        })
        .await
        .map_err(|e| internal_error(format!("Tag list panicked: {}", e)))?
    }

    fn extract_tags_with_counts_blocking(
        &self,
        path: &Path,
    ) -> Result<Vec<TagCount>, Box<dyn std::error::Error>> {
        let files = if path.is_file() {
            vec![path.to_path_buf()]
        } else {
            collect_markdown_files(path, &self.config)?
        };

        // Track which documents contain each tag
        // Key: tag name, Value: set of file paths that contain this tag
        use std::collections::{HashMap, HashSet};
        let tag_documents: HashMap<String, HashSet<PathBuf>> = files
            .par_iter()
            .filter_map(|file_path| {
                self.extract_tags_from_file(file_path)
                    .ok()
                    .map(|tags| (file_path.clone(), tags))
            })
            .fold(
                HashMap::new,
                |mut acc: HashMap<String, HashSet<PathBuf>>, (file_path, tags)| {
                    // Deduplicate tags within the same file (a file counts once per tag)
                    let unique_tags: HashSet<String> = tags.into_iter().collect();
                    for tag in unique_tags {
                        acc.entry(tag).or_default().insert(file_path.clone());
                    }
                    acc
                },
            )
            .reduce(HashMap::new, |mut a, b| {
                for (tag, files) in b {
                    a.entry(tag).or_insert_with(HashSet::new).extend(files);
                }
                a
            });

        // Convert to Vec<TagCount> sorted by document_count desc, then tag name asc
        let mut result: Vec<TagCount> = tag_documents
            .into_iter()
            .map(|(tag, files)| TagCount {
                tag,
                document_count: files.len(),
            })
            .collect();

        result.sort_by(|a, b| {
            b.document_count
                .cmp(&a.document_count)
                .then_with(|| a.tag.cmp(&b.tag))
        });

        Ok(result)
    }

    pub async fn search_by_tags(
        &self,
        path: &Path,
        tags: &[String],
        match_all: bool,
    ) -> CapabilityResult<Vec<TaggedFile>> {
        let me = self.clone();
        let path = path.to_path_buf();
        let tags = tags.to_vec();
        tokio::task::spawn_blocking(move || {
            me.search_by_tags_blocking(&path, &tags, match_all)
                .map_err(|e| internal_error(format!("Failed to search by tags: {}", e)))
        })
        .await
        .map_err(|e| internal_error(format!("Tag search panicked: {}", e)))?
    }

    fn search_by_tags_blocking(
        &self,
        path: &Path,
        tags: &[String],
        match_all: bool,
    ) -> Result<Vec<TaggedFile>, Box<dyn std::error::Error>> {
        let files = if path.is_file() {
            vec![path.to_path_buf()]
        } else {
            collect_markdown_files(path, &self.config)?
        };

        // Normalize search tags to lowercase for case-insensitive comparison
        let search_tags: Vec<String> = tags.iter().map(|t| t.to_lowercase()).collect();

        let results: Vec<TaggedFile> = files
            .par_iter()
            .filter_map(|file_path| {
                // Extract tags from file
                let all_tags = self.extract_tags_from_file(file_path).ok()?;

                if all_tags.is_empty() {
                    return None;
                }

                // Normalize file tags for comparison
                let normalized_tags: Vec<String> =
                    all_tags.iter().map(|t| t.to_lowercase()).collect();

                // Find which search tags match this file
                let matched_tags: Vec<String> = search_tags
                    .iter()
                    .filter(|search_tag| normalized_tags.contains(search_tag))
                    .cloned()
                    .collect();

                // Apply match logic
                let matches = if match_all {
                    // AND logic: all search tags must be present
                    matched_tags.len() == search_tags.len()
                } else {
                    // OR logic: at least one search tag must be present
                    !matched_tags.is_empty()
                };

                if matches {
                    Some(TaggedFile {
                        file_path: file_path.to_string_lossy().to_string(),
                        file_name: file_path.file_name()?.to_string_lossy().to_string(),
                        matched_tags,
                        all_tags,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_test_config() -> Arc<Config> {
        Arc::new(Config::default())
    }

    fn create_test_file(dir: &std::path::Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn test_extract_frontmatter() {
        let extractor = TagExtractor::new(create_test_config());

        let content = r#"---
title: My Document
tags:
  - rust
  - programming
---

# Content here
"#;

        let frontmatter = extractor.extract_frontmatter(content).unwrap();
        assert!(frontmatter.is_some());
        assert!(frontmatter.unwrap().contains("tags:"));
    }

    #[test]
    fn test_parse_tags_array() {
        let extractor = TagExtractor::new(create_test_config());

        let frontmatter = r#"title: My Document
tags:
  - rust
  - programming
  - cli
"#;

        let tags = extractor.parse_tags_from_frontmatter(frontmatter).unwrap();
        assert_eq!(tags.len(), 3);
        assert!(tags.contains(&"rust".to_string()));
        assert!(tags.contains(&"programming".to_string()));
        assert!(tags.contains(&"cli".to_string()));
    }

    #[test]
    fn test_parse_tags_single_string() {
        let extractor = TagExtractor::new(create_test_config());

        let frontmatter = r#"title: My Document
tags: single-tag
"#;

        let tags = extractor.parse_tags_from_frontmatter(frontmatter).unwrap();
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0], "single-tag");
    }

    #[test]
    fn test_extract_tags_from_content() {
        let extractor = TagExtractor::new(create_test_config());

        let content = r#"---
title: My Document
tags:
  - rust
  - programming
---

# My Document

Some content here.
"#;

        let tags = extractor.extract_tags_from_content(content).unwrap();
        assert_eq!(tags.len(), 2);
        assert!(tags.contains(&"rust".to_string()));
        assert!(tags.contains(&"programming".to_string()));
    }

    #[test]
    fn test_no_frontmatter() {
        let extractor = TagExtractor::new(create_test_config());

        let content = r#"# My Document

Some content here without frontmatter.
"#;

        let tags = extractor.extract_tags_from_content(content).unwrap();
        assert_eq!(tags.len(), 0);
    }

    #[test]
    fn test_empty_tags_filtered() {
        let extractor = TagExtractor::new(create_test_config());

        let frontmatter = r#"title: My Document
tags:
  - rust
  - ""
  - programming
  - "  "
  - cli
"#;

        let tags = extractor.parse_tags_from_frontmatter(frontmatter).unwrap();
        assert_eq!(tags.len(), 3);
        assert!(tags.contains(&"rust".to_string()));
        assert!(tags.contains(&"programming".to_string()));
        assert!(tags.contains(&"cli".to_string()));
        assert!(!tags.contains(&"".to_string()));
        assert!(!tags.contains(&"  ".to_string()));
    }

    #[test]
    fn test_empty_string_tag_filtered() {
        let extractor = TagExtractor::new(create_test_config());

        let frontmatter = r#"title: My Document
tags: ""
"#;

        let tags = extractor.parse_tags_from_frontmatter(frontmatter).unwrap();
        assert_eq!(tags.len(), 0);
    }

    #[tokio::test]
    async fn test_extract_tags_with_counts_single_file() {
        use tempfile::TempDir;
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config();
        let extractor = TagExtractor::new(config);

        let content = r#"---
tags:
  - rust
  - programming
---
# Content
"#;
        create_test_file(temp_dir.path(), "test1.md", content);

        let counts = extractor
            .extract_tags_with_counts(temp_dir.path())
            .await
            .unwrap();

        assert_eq!(counts.len(), 2);
        assert!(
            counts
                .iter()
                .any(|t| t.tag == "rust" && t.document_count == 1)
        );
        assert!(
            counts
                .iter()
                .any(|t| t.tag == "programming" && t.document_count == 1)
        );
    }

    #[tokio::test]
    async fn test_extract_tags_with_counts_multiple_files() {
        use tempfile::TempDir;
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config();
        let extractor = TagExtractor::new(config);

        // File 1: has rust and programming tags
        let content1 = r#"---
tags:
  - rust
  - programming
---
"#;
        create_test_file(temp_dir.path(), "file1.md", content1);

        // File 2: has rust and cli tags
        let content2 = r#"---
tags:
  - rust
  - cli
---
"#;
        create_test_file(temp_dir.path(), "file2.md", content2);

        let counts = extractor
            .extract_tags_with_counts(temp_dir.path())
            .await
            .unwrap();

        // rust appears in 2 documents, programming and cli in 1 each
        let rust = counts.iter().find(|t| t.tag == "rust").unwrap();
        assert_eq!(rust.document_count, 2);

        let programming = counts.iter().find(|t| t.tag == "programming").unwrap();
        assert_eq!(programming.document_count, 1);

        let cli = counts.iter().find(|t| t.tag == "cli").unwrap();
        assert_eq!(cli.document_count, 1);

        // Should be sorted by count desc
        assert_eq!(counts[0].tag, "rust");
    }

    #[tokio::test]
    async fn test_extract_tags_with_counts_duplicate_in_same_file() {
        use tempfile::TempDir;
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config();
        let extractor = TagExtractor::new(config);

        // File with duplicate tag (should only count once per document)
        let content = r#"---
tags:
  - rust
  - rust
  - programming
---
"#;
        create_test_file(temp_dir.path(), "file.md", content);

        let counts = extractor
            .extract_tags_with_counts(temp_dir.path())
            .await
            .unwrap();

        let rust = counts.iter().find(|t| t.tag == "rust").unwrap();
        assert_eq!(rust.document_count, 1); // Should be 1, not 2
    }

    #[tokio::test]
    async fn test_search_by_tags_or_logic() {
        use tempfile::TempDir;
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config();
        let extractor = TagExtractor::new(config);

        // Create test files
        create_test_file(
            temp_dir.path(),
            "file1.md",
            "---\ntags:\n  - rust\n  - cli\n---\n# File 1",
        );
        create_test_file(
            temp_dir.path(),
            "file2.md",
            "---\ntags:\n  - python\n  - cli\n---\n# File 2",
        );
        create_test_file(
            temp_dir.path(),
            "file3.md",
            "---\ntags:\n  - java\n---\n# File 3",
        );

        // Search with OR logic (default)
        let results = extractor
            .search_by_tags(
                temp_dir.path(),
                &["rust".to_string(), "python".to_string()],
                false,
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|f| f.file_name == "file1.md"));
        assert!(results.iter().any(|f| f.file_name == "file2.md"));
    }

    #[tokio::test]
    async fn test_search_by_tags_and_logic() {
        use tempfile::TempDir;
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config();
        let extractor = TagExtractor::new(config);

        // Create test files
        create_test_file(
            temp_dir.path(),
            "file1.md",
            "---\ntags:\n  - rust\n  - cli\n---\n# File 1",
        );
        create_test_file(
            temp_dir.path(),
            "file2.md",
            "---\ntags:\n  - rust\n---\n# File 2",
        );

        // Search with AND logic
        let results = extractor
            .search_by_tags(
                temp_dir.path(),
                &["rust".to_string(), "cli".to_string()],
                true,
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_name, "file1.md");
    }

    #[tokio::test]
    async fn test_search_by_tags_case_insensitive() {
        use tempfile::TempDir;
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config();
        let extractor = TagExtractor::new(config);

        // Create test file with mixed case tags
        create_test_file(
            temp_dir.path(),
            "file1.md",
            "---\ntags:\n  - Rust\n  - CLI\n---\n# File 1",
        );

        // Search with lowercase
        let results = extractor
            .search_by_tags(temp_dir.path(), &["rust".to_string()], false)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);

        // Search with uppercase
        let results = extractor
            .search_by_tags(temp_dir.path(), &["RUST".to_string()], false)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_search_by_tags_empty_result() {
        use tempfile::TempDir;
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config();
        let extractor = TagExtractor::new(config);

        // Create test file
        create_test_file(
            temp_dir.path(),
            "file1.md",
            "---\ntags:\n  - rust\n---\n# File 1",
        );

        // Search for non-existent tag
        let results = extractor
            .search_by_tags(temp_dir.path(), &["nonexistent".to_string()], false)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_by_tags_respects_exclusions() {
        use tempfile::TempDir;
        let temp_dir = TempDir::new().unwrap();
        let config = Arc::new(Config {
            exclude_paths: vec!["excluded".to_string()],
            daily_note_patterns: notectl_core::config::default_daily_note_patterns(),
        });
        let extractor = TagExtractor::new(config);

        // Create test files
        create_test_file(
            temp_dir.path(),
            "file1.md",
            "---\ntags:\n  - rust\n---\n# File 1",
        );

        // Create excluded directory
        let excluded_dir = temp_dir.path().join("excluded");
        std::fs::create_dir(&excluded_dir).unwrap();
        create_test_file(
            &excluded_dir,
            "file2.md",
            "---\ntags:\n  - rust\n---\n# File 2",
        );

        // Search should not include excluded file
        let results = extractor
            .search_by_tags(temp_dir.path(), &["rust".to_string()], false)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_name, "file1.md");
    }

    #[tokio::test]
    async fn test_tagged_file_contains_all_tags() {
        use tempfile::TempDir;
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config();
        let extractor = TagExtractor::new(config);

        // Create test file with multiple tags
        create_test_file(
            temp_dir.path(),
            "file1.md",
            "---\ntags:\n  - rust\n  - cli\n  - tool\n---\n# File 1",
        );

        // Search for one tag
        let results = extractor
            .search_by_tags(temp_dir.path(), &["rust".to_string()], false)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].matched_tags, vec!["rust".to_string()]);
        assert_eq!(
            results[0].all_tags,
            vec!["rust".to_string(), "cli".to_string(), "tool".to_string()]
        );
    }
}
