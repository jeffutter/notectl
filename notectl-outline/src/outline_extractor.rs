use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Represents a heading found in a markdown file
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Heading {
    pub title: String,
    pub level: u8,
    pub line_number: usize,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub children: Vec<Heading>,
}

/// Represents a section in a markdown file (heading + content)
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Section {
    pub heading: Heading,
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
}

/// Represents a heading match across multiple files
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct HeadingMatch {
    pub heading: Heading,
    pub file_path: String,
    pub file_name: String,
}

/// Extracts outline structure from markdown files
pub struct OutlineExtractor {
    heading_pattern: Regex,
}

impl OutlineExtractor {
    pub fn new() -> Self {
        Self {
            // Match ATX-style headings: # to ###### followed by space and title
            // Supports Obsidian heading IDs: ## Title {#custom-id}
            heading_pattern: Regex::new(r"^(#{1,6})\s+(.+?)(?:\s*\{#[^}]*\})?\s*$").unwrap(),
        }
    }

    /// Parse a single heading from a line
    fn parse_heading(&self, line: &str, line_number: usize) -> Option<Heading> {
        let caps = self.heading_pattern.captures(line)?;
        let hashes = caps.get(1)?.as_str();
        let title = caps.get(2)?.as_str().trim();

        Some(Heading {
            title: title.to_string(),
            level: hashes.len() as u8,
            line_number,
            children: Vec::new(),
        })
    }

    /// Extract all headings from file content, filtering out headings in code blocks
    pub fn extract_headings(&self, content: &str) -> Vec<Heading> {
        let mut headings = Vec::new();
        let mut in_code_block = false;
        let mut code_fence: Option<&str> = None;

        for (line_num, line) in content.lines().enumerate() {
            let trimmed = line.trim();

            // Track code blocks (both ``` and ~~~ style)
            if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
                if in_code_block {
                    // Check if closing the same fence type
                    let fence = code_fence.unwrap_or("```");
                    if trimmed.starts_with(fence) {
                        in_code_block = false;
                        code_fence = None;
                    }
                } else {
                    // Opening a new code block
                    in_code_block = true;
                    code_fence = if trimmed.starts_with("```") {
                        Some("```")
                    } else {
                        Some("~~~")
                    };
                }
                continue;
            }

            // Skip headings inside code blocks
            if in_code_block {
                continue;
            }

            // Try to parse this line as a heading
            if let Some(heading) = self.parse_heading(line, line_num + 1) {
                headings.push(heading);
            }
        }

        headings
    }

    /// Build hierarchical tree from flat list of headings
    /// Uses indices instead of references to avoid borrow checker issues
    pub fn build_hierarchy(&self, headings: &[Heading]) -> Vec<Heading> {
        if headings.is_empty() {
            return Vec::new();
        }

        // We'll build the tree using a stack of indices
        // Each element is (index_in_parent, level) where index_in_parent is where this node lives
        // in its parent's children vector
        let mut result: Vec<Heading> = Vec::new();

        // Stack stores (parent_path, level) where parent_path is a Vec of indices
        // Empty parent_path means root level
        let mut stack: Vec<(Vec<usize>, u8)> = Vec::new();

        for heading in headings {
            let new_heading = Heading {
                title: heading.title.clone(),
                level: heading.level,
                line_number: heading.line_number,
                children: Vec::new(),
            };

            // Pop from stack until we find appropriate parent
            while let Some((_, parent_level)) = stack.last() {
                if *parent_level < heading.level {
                    break;
                }
                stack.pop();
            }

            // Add heading to appropriate parent
            if let Some((parent_path, _)) = stack.last() {
                // Navigate to the parent and add the child
                let parent = Self::get_mut_node_at_path(&mut result, parent_path);
                parent.children.push(new_heading);

                // Build new path for this node
                let mut new_path = parent_path.clone();
                new_path.push(parent.children.len() - 1);
                stack.push((new_path, heading.level));
            } else {
                // Add to root
                result.push(new_heading);
                stack.push((vec![result.len() - 1], heading.level));
            }
        }

        result
    }

    /// Helper to get a mutable reference to a node at a given path
    fn get_mut_node_at_path<'a>(root: &'a mut [Heading], path: &[usize]) -> &'a mut Heading {
        if path.is_empty() {
            panic!("Empty path");
        }

        let mut current = &mut root[path[0]];
        for &index in &path[1..] {
            current = &mut current.children[index];
        }
        current
    }

    /// Get outline from a file (returns flat or hierarchical based on flag)
    pub fn get_outline(
        &self,
        file_path: &Path,
        hierarchical: bool,
    ) -> Result<Vec<Heading>, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(file_path)
            .map_err(|e| format!("Failed to read file {:?}: {}", file_path, e))?;

        let headings = self.extract_headings(&content);

        if hierarchical {
            Ok(self.build_hierarchy(&headings))
        } else {
            Ok(headings)
        }
    }

    /// Extract section content under a specific heading
    pub fn get_section(
        &self,
        file_path: &Path,
        target_heading: &str,
        include_subsections: bool,
    ) -> Result<Vec<Section>, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(file_path)
            .map_err(|e| format!("Failed to read file {:?}: {}", file_path, e))?;

        let lines: Vec<&str> = content.lines().collect();
        let headings = self.extract_headings(&content);
        let mut sections = Vec::new();

        // Find all headings matching the target
        let matching_indices: Vec<usize> = headings
            .iter()
            .enumerate()
            .filter(|(_, h)| h.title.to_lowercase() == target_heading.to_lowercase())
            .map(|(i, _)| i)
            .collect();

        for idx in matching_indices {
            let heading = &headings[idx];
            let start_line = heading.line_number;

            // Determine end line
            let end_line = if include_subsections {
                // Include until next heading of same or higher level
                // "Higher level" means smaller number (H1 > H2 > H3)
                headings
                    .iter()
                    .skip(idx + 1)
                    .find(|h| h.level <= heading.level)
                    .map(|h| h.line_number - 1)
                    .unwrap_or(lines.len())
            } else {
                // Exclude subsections - stop at the next heading of any level
                // This cuts off at the subsection heading itself
                headings
                    .get(idx + 1)
                    .map(|h| h.line_number - 1)
                    .unwrap_or(lines.len())
            };

            // Extract content (skip the heading line itself)
            let section_content = if start_line < lines.len() && end_line <= lines.len() {
                lines[start_line..end_line].join("\n")
            } else {
                String::new()
            };

            sections.push(Section {
                heading: Heading {
                    title: heading.title.clone(),
                    level: heading.level,
                    line_number: heading.line_number,
                    children: Vec::new(),
                },
                content: section_content.trim().to_string(),
                start_line,
                end_line,
            });
        }

        Ok(sections)
    }

    /// Search for headings matching a pattern across files in a directory
    pub fn search_headings(
        &self,
        dir_path: &Path,
        pattern: &str,
        min_level: Option<u8>,
        max_level: Option<u8>,
        limit: Option<usize>,
        config: &markdown_todo_extractor_core::config::Config,
    ) -> Result<Vec<HeadingMatch>, Box<dyn std::error::Error>> {
        let mut matches = Vec::new();
        let pattern_lower = pattern.to_lowercase();

        // Collect all markdown files
        let mut files_to_search = Vec::new();
        self.collect_markdown_files(dir_path, &mut files_to_search, config)?;

        // Search each file
        for file_path in files_to_search {
            let content = match fs::read_to_string(&file_path) {
                Ok(c) => c,
                Err(_) => continue, // Skip files we can't read
            };

            let headings = self.extract_headings(&content);

            for heading in headings {
                // Filter by level if specified
                if let Some(min) = min_level
                    && heading.level < min
                {
                    continue;
                }
                if let Some(max) = max_level
                    && heading.level > max
                {
                    continue;
                }

                // Case-insensitive substring match
                if heading.title.to_lowercase().contains(&pattern_lower) {
                    let file_name = file_path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();

                    matches.push(HeadingMatch {
                        heading: Heading {
                            title: heading.title,
                            level: heading.level,
                            line_number: heading.line_number,
                            children: Vec::new(),
                        },
                        file_path: file_path.to_string_lossy().to_string(),
                        file_name,
                    });

                    // Check limit
                    if let Some(lim) = limit
                        && matches.len() >= lim
                    {
                        return Ok(matches);
                    }
                }
            }
        }

        Ok(matches)
    }

    /// Recursively collect all markdown files in a directory
    fn collect_markdown_files(
        &self,
        dir: &Path,
        files: &mut Vec<std::path::PathBuf>,
        config: &markdown_todo_extractor_core::config::Config,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            // Skip excluded paths
            if config.should_exclude(&path) {
                continue;
            }

            if path.is_dir() {
                self.collect_markdown_files(&path, files, config)?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("md") {
                files.push(path);
            }
        }

        Ok(())
    }
}

impl Default for OutlineExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_extractor() -> OutlineExtractor {
        OutlineExtractor::new()
    }

    mod parse_heading {
        use super::*;

        #[test]
        fn test_h1_heading() {
            let extractor = create_test_extractor();
            let heading = extractor.parse_heading("# Title", 1);

            assert!(heading.is_some());
            let h = heading.unwrap();
            assert_eq!(h.title, "Title");
            assert_eq!(h.level, 1);
            assert_eq!(h.line_number, 1);
        }

        #[test]
        fn test_h6_heading() {
            let extractor = create_test_extractor();
            let heading = extractor.parse_heading("###### Deep Title", 1);

            assert!(heading.is_some());
            let h = heading.unwrap();
            assert_eq!(h.title, "Deep Title");
            assert_eq!(h.level, 6);
        }

        #[test]
        fn test_heading_with_obsidian_id() {
            let extractor = create_test_extractor();
            let heading = extractor.parse_heading("## Title {#custom-id}", 1);

            assert!(heading.is_some());
            let h = heading.unwrap();
            assert_eq!(h.title, "Title");
            assert_eq!(h.level, 2);
        }

        #[test]
        fn test_not_a_heading_no_space() {
            let extractor = create_test_extractor();
            let heading = extractor.parse_heading("#NoSpace", 1);

            assert!(heading.is_none());
        }

        #[test]
        fn test_too_many_hashes() {
            let extractor = create_test_extractor();
            let heading = extractor.parse_heading("####### Too many", 1);

            assert!(heading.is_none());
        }

        #[test]
        fn test_regular_text() {
            let extractor = create_test_extractor();
            let heading = extractor.parse_heading("Just some text", 1);

            assert!(heading.is_none());
        }

        #[test]
        fn test_heading_with_unicode() {
            let extractor = create_test_extractor();
            let heading = extractor.parse_heading("## 日本語タイトル", 1);

            assert!(heading.is_some());
            let h = heading.unwrap();
            assert_eq!(h.title, "日本語タイトル");
        }
    }

    mod extract_headings {
        use super::*;

        #[test]
        fn test_simple_document() {
            let extractor = create_test_extractor();
            let content = r"# Title
## Section 1
Some text
### Subsection
More text
## Section 2
Final text";

            let headings = extractor.extract_headings(content);
            assert_eq!(headings.len(), 4);
            assert_eq!(headings[0].title, "Title");
            assert_eq!(headings[0].level, 1);
            assert_eq!(headings[1].title, "Section 1");
            assert_eq!(headings[1].level, 2);
        }

        #[test]
        fn test_headings_in_code_blocks_ignored() {
            let extractor = create_test_extractor();
            let content = r"# Real Heading
```markdown
# Fake Heading in code
```
## Another Real Heading";

            let headings = extractor.extract_headings(content);
            assert_eq!(headings.len(), 2);
            assert_eq!(headings[0].title, "Real Heading");
            assert_eq!(headings[1].title, "Another Real Heading");
        }

        #[test]
        fn test_nested_code_blocks() {
            let extractor = create_test_extractor();
            let content = r"# Real Heading
```
Some code
~~~
# Inside nested
~~~
```
## After Code";

            let headings = extractor.extract_headings(content);
            assert_eq!(headings.len(), 2);
        }
    }

    mod build_hierarchy {
        use super::*;

        #[test]
        fn test_simple_hierarchy() {
            let extractor = create_test_extractor();
            let content = r"# Title
## Section 1
### Subsection 1.1
## Section 2";

            let flat_headings = extractor.extract_headings(content);
            let hierarchical = extractor.build_hierarchy(&flat_headings);

            assert_eq!(hierarchical.len(), 1);
            assert_eq!(hierarchical[0].title, "Title");
            assert_eq!(hierarchical[0].children.len(), 2);
            assert_eq!(hierarchical[0].children[0].title, "Section 1");
            assert_eq!(hierarchical[0].children[0].children.len(), 1);
        }

        #[test]
        fn test_level_skipping() {
            let extractor = create_test_extractor();
            let content = r"# Title
### Deep Section
#### Deeper";

            let flat_headings = extractor.extract_headings(content);
            let hierarchical = extractor.build_hierarchy(&flat_headings);

            assert_eq!(hierarchical.len(), 1);
            assert_eq!(hierarchical[0].children.len(), 1);
        }
    }

    mod get_section {
        use super::*;
        use std::io::Write;
        use tempfile::NamedTempFile;

        #[test]
        fn test_get_section_basic() {
            let extractor = create_test_extractor();
            let mut temp_file = NamedTempFile::new().unwrap();
            write!(
                temp_file,
                r"# Title
## Target Section
Content here
More content
## Next Section
Other content"
            )
            .unwrap();

            let sections = extractor
                .get_section(temp_file.path(), "Target Section", false)
                .unwrap();
            assert_eq!(sections.len(), 1);
            assert_eq!(sections[0].content, "Content here\nMore content");
        }

        #[test]
        fn test_get_section_with_subsections() {
            let extractor = create_test_extractor();
            let mut temp_file = NamedTempFile::new().unwrap();
            write!(
                temp_file,
                r"# Title
## Target Section
Content
### Subsection
Sub content
## Next Section
Other"
            )
            .unwrap();

            let sections = extractor
                .get_section(temp_file.path(), "Target Section", true)
                .unwrap();
            assert_eq!(sections.len(), 1);
            assert!(sections[0].content.contains("Sub content"));
        }

        #[test]
        fn test_get_section_without_subsections() {
            let extractor = create_test_extractor();
            let mut temp_file = NamedTempFile::new().unwrap();
            write!(
                temp_file,
                r"# Title
## Target Section
Content
### Subsection
Sub content
## Next Section
Other"
            )
            .unwrap();

            let sections = extractor
                .get_section(temp_file.path(), "Target Section", false)
                .unwrap();
            assert_eq!(sections.len(), 1);
            assert!(!sections[0].content.contains("Sub content"));
        }

        #[test]
        fn test_multiple_matching_sections() {
            let extractor = create_test_extractor();
            let mut temp_file = NamedTempFile::new().unwrap();
            write!(
                temp_file,
                r"# Title
## Duplicate
First content
## Other
Different
## Duplicate
Second content"
            )
            .unwrap();

            let sections = extractor
                .get_section(temp_file.path(), "Duplicate", false)
                .unwrap();
            assert_eq!(sections.len(), 2);
        }
    }

    mod search_headings {
        use super::*;
        use std::io::Write;

        use tempfile::TempDir;

        #[test]
        fn test_search_across_files() {
            let extractor = create_test_extractor();
            let temp_dir = TempDir::new().unwrap();
            let config = markdown_todo_extractor_core::config::Config::default();

            let mut file1 = std::fs::File::create(temp_dir.path().join("file1.md")).unwrap();
            write!(file1, "# Introduction\n## Search Target").unwrap();

            let mut file2 = std::fs::File::create(temp_dir.path().join("file2.md")).unwrap();
            write!(file2, "## Other Section\n# Search Target").unwrap();

            let matches = extractor
                .search_headings(temp_dir.path(), "Search Target", None, None, None, &config)
                .unwrap();
            assert_eq!(matches.len(), 2);
        }

        #[test]
        fn test_search_with_level_filter() {
            let extractor = create_test_extractor();
            let temp_dir = TempDir::new().unwrap();
            let config = markdown_todo_extractor_core::config::Config::default();

            let mut file = std::fs::File::create(temp_dir.path().join("file.md")).unwrap();
            write!(file, "# Target\n## Target\n### Target").unwrap();

            let matches = extractor
                .search_headings(temp_dir.path(), "Target", Some(2), Some(2), None, &config)
                .unwrap();
            assert_eq!(matches.len(), 1);
            assert_eq!(matches[0].heading.level, 2);
        }

        #[test]
        fn test_search_limit() {
            let extractor = create_test_extractor();
            let temp_dir = TempDir::new().unwrap();
            let config = markdown_todo_extractor_core::config::Config::default();

            let mut file = std::fs::File::create(temp_dir.path().join("file.md")).unwrap();
            write!(file, "# Target 1\n# Target 2\n# Target 3").unwrap();

            let matches = extractor
                .search_headings(temp_dir.path(), "Target", None, None, Some(2), &config)
                .unwrap();
            assert_eq!(matches.len(), 2);
        }

        #[test]
        fn test_case_insensitive_search() {
            let extractor = create_test_extractor();
            let temp_dir = TempDir::new().unwrap();
            let config = markdown_todo_extractor_core::config::Config::default();

            let mut file = std::fs::File::create(temp_dir.path().join("file.md")).unwrap();
            write!(file, "# UPPERCASE\n# lowercase\n# MixedCase").unwrap();

            let matches = extractor
                .search_headings(temp_dir.path(), "case", None, None, None, &config)
                .unwrap();
            assert_eq!(matches.len(), 3);
        }
    }
}
