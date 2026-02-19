use markdown_todo_extractor_core::config::Config;
use rayon::prelude::*;
use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::sync::Arc;

/// Represents a task found in a markdown file
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Task {
    pub content: String,
    pub status: String,
    pub file_path: String,
    pub file_name: String,
    pub line_number: usize,
    pub raw_line: String,
    pub tags: Vec<String>,
    pub sub_items: Vec<String>,
    pub summary: Option<String>,
    pub due_date: Option<String>,
    pub priority: Option<String>,
    pub created_date: Option<String>,
    pub completed_date: Option<String>,
}

/// Extracts tasks from markdown files
pub struct TaskExtractor {
    task_incomplete: Regex,
    task_completed: Regex,
    task_cancelled: Regex,
    task_other: Regex,
    tag_pattern: Regex,
    due_date_patterns: Vec<Regex>,
    priority_pattern: Regex,
    created_patterns: Vec<Regex>,
    completion_patterns: Vec<Regex>,
    // Cleaning patterns (moved from clean_content())
    timestamp_pattern: Regex,
    priority_emoji_pattern: Regex,
    priority_text_pattern: Regex,
    whitespace_pattern: Regex,
    // Sub-item pattern (moved from parse_sub_item())
    checkbox_pattern: Regex,
    // Configuration for path exclusion
    config: Arc<Config>,
}

impl TaskExtractor {
    pub fn new(config: Arc<Config>) -> Self {
        TaskExtractor {
            task_incomplete: Regex::new(r"^(\s*)-\s*\[\s\]\s*(.+)$").unwrap(),
            task_completed: Regex::new(r"(?i)^(\s*)-\s*\[x\]\s*(.+)$").unwrap(),
            task_cancelled: Regex::new(r"^(\s*)-\s*\[-\]\s*(.+)$").unwrap(),
            task_other: Regex::new(r"^(\s*)-\s*\[(.)\]\s*(.+)$").unwrap(),
            tag_pattern: Regex::new(r"#(\w+)").unwrap(),
            due_date_patterns: vec![
                Regex::new(r"📅\s*(\d{4}-\d{2}-\d{2})").unwrap(),
                Regex::new(r"due:\s*(\d{4}-\d{2}-\d{2})").unwrap(),
                Regex::new(r"@due\((\d{4}-\d{2}-\d{2})\)").unwrap(),
            ],
            priority_pattern: Regex::new(r"[⏫🔼🔽⏬]|priority:\s*(high|medium|low)").unwrap(),
            created_patterns: vec![
                Regex::new(r"➕\s*(\d{4}-\d{2}-\d{2})").unwrap(),
                Regex::new(r"created:\s*(\d{4}-\d{2}-\d{2})").unwrap(),
            ],
            completion_patterns: vec![
                Regex::new(r"✅\s*(\d{4}-\d{2}-\d{2})").unwrap(),
                Regex::new(r"completed:\s*(\d{4}-\d{2}-\d{2})").unwrap(),
            ],
            // Cleaning patterns
            timestamp_pattern: Regex::new(r"^\d{2}:\d{2} ").unwrap(),
            priority_emoji_pattern: Regex::new(r"[⏫🔼🔽⏬]").unwrap(),
            priority_text_pattern: Regex::new(r"(?i)priority:\s*(high|medium|low)").unwrap(),
            whitespace_pattern: Regex::new(r"\s+").unwrap(),
            // Sub-item pattern
            checkbox_pattern: Regex::new(r"^-\s*\[.\]\s*(.+)$").unwrap(),
            config,
        }
    }

    fn extract_tags(&self, content: &str) -> Vec<String> {
        self.tag_pattern
            .captures_iter(content)
            .map(|cap| cap.get(1).unwrap().as_str().to_string())
            .collect()
    }

    fn extract_due_date(&self, content: &str) -> Option<String> {
        for pattern in &self.due_date_patterns {
            if let Some(caps) = pattern.captures(content) {
                return Some(caps.get(1).unwrap().as_str().to_string());
            }
        }
        None
    }

    fn extract_priority(&self, content: &str) -> Option<String> {
        if let Some(caps) = self.priority_pattern.captures(content) {
            let matched = caps.get(0).unwrap().as_str();
            match matched {
                "⏫" => Some("urgent".to_string()),
                "🔼" => Some("high".to_string()),
                "🔽" => Some("low".to_string()),
                "⏬" => Some("lowest".to_string()),
                _ => caps.get(1).map(|m| m.as_str().to_lowercase()),
            }
        } else {
            None
        }
    }

    fn extract_created_date(&self, content: &str) -> Option<String> {
        for pattern in &self.created_patterns {
            if let Some(caps) = pattern.captures(content) {
                return Some(caps.get(1).unwrap().as_str().to_string());
            }
        }
        None
    }

    fn extract_completed_date(&self, content: &str) -> Option<String> {
        for pattern in &self.completion_patterns {
            if let Some(caps) = pattern.captures(content) {
                return Some(caps.get(1).unwrap().as_str().to_string());
            }
        }
        None
    }

    fn clean_content(&self, content: &str) -> String {
        use std::borrow::Cow;

        // Start with borrowed content
        let mut cleaned = Cow::Borrowed(content);

        // Remove due date patterns
        for pattern in &self.due_date_patterns {
            if let Cow::Owned(s) = pattern.replace_all(&cleaned, "") {
                cleaned = Cow::Owned(s);
            }
        }

        // Remove timestamp prefix
        if let Cow::Owned(s) = self.timestamp_pattern.replace_all(&cleaned, " ") {
            cleaned = Cow::Owned(s);
        }

        // Remove priority indicators
        if let Cow::Owned(s) = self.priority_emoji_pattern.replace_all(&cleaned, "") {
            cleaned = Cow::Owned(s);
        }
        if let Cow::Owned(s) = self.priority_text_pattern.replace_all(&cleaned, "") {
            cleaned = Cow::Owned(s);
        }

        // Remove created date patterns
        for pattern in &self.created_patterns {
            if let Cow::Owned(s) = pattern.replace_all(&cleaned, "") {
                cleaned = Cow::Owned(s);
            }
        }

        // Remove completed date patterns
        for pattern in &self.completion_patterns {
            if let Cow::Owned(s) = pattern.replace_all(&cleaned, "") {
                cleaned = Cow::Owned(s);
            }
        }

        // Clean up extra whitespace
        if let Cow::Owned(s) = self.whitespace_pattern.replace_all(&cleaned, " ") {
            cleaned = Cow::Owned(s);
        }

        // Final trim and convert to owned String
        cleaned.trim().to_string()
    }

    fn is_sub_item(&self, line: &str, parent_line: &str) -> bool {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return false;
        }

        // Get indentation levels
        let parent_indent = parent_line.len() - parent_line.trim_start().len();
        let line_indent = line.len() - line.trim_start().len();

        // Sub-item must be more indented than parent
        if line_indent <= parent_indent {
            return false;
        }

        // Check if it's a list item (starts with - or *)
        let stripped = line.trim_start();
        stripped.starts_with('-')
            || stripped.starts_with('*')
            || stripped.starts_with("- [")
            || stripped.starts_with("* [")
    }

    fn parse_sub_item(&self, line: &str) -> Option<String> {
        let stripped = line.trim();

        // Handle checkbox sub-items
        if stripped.starts_with("- [")
            && let Some(caps) = self.checkbox_pattern.captures(stripped)
        {
            return Some(caps.get(1).unwrap().as_str().trim().to_string());
        }

        // Handle regular list items
        if stripped.starts_with('-') || stripped.starts_with('*') {
            return Some(stripped[1..].trim().to_string());
        }

        None
    }

    fn extract_tasks_from_file(
        &self,
        file_path: &Path,
    ) -> Result<Vec<Task>, Box<dyn std::error::Error>> {
        // Read file as bytes and validate UTF-8 with SIMD
        let bytes = fs::read(file_path)?;
        let content = simdutf8::basic::from_utf8(&bytes)
            .map_err(|e| format!("Invalid UTF-8 in {:?}: {}", file_path, e))?;
        let mut tasks = Vec::new();

        // Use iterator instead of collecting into Vec
        let mut lines = content.lines().enumerate().peekable();

        while let Some((line_num, line)) = lines.next() {
            if let Some(mut task) = self.parse_task_line(line, file_path, line_num + 1) {
                // Look ahead for sub-items on subsequent lines
                while let Some(&(_, next_line)) = lines.peek() {
                    if self.is_sub_item(next_line, &task.raw_line) {
                        if let Some(sub_item) = self.parse_sub_item(next_line) {
                            task.sub_items.push(sub_item);
                        }
                        lines.next(); // Consume the sub-item line
                    } else {
                        break;
                    }
                }
                tasks.push(task);
            }
        }

        Ok(tasks)
    }

    pub fn extract_tasks(&self, path: &Path) -> Result<Vec<Task>, Box<dyn std::error::Error>> {
        if path.is_file() {
            // Single file
            if path.extension().and_then(|s| s.to_str()) == Some("md") {
                self.extract_tasks_from_file(path)
            } else {
                Ok(Vec::new())
            }
        } else if path.is_dir() {
            // Directory - recursively find all .md files in parallel
            self.extract_tasks_from_dir(path)
        } else {
            Err(format!("Path does not exist: {}", path.display()).into())
        }
    }

    fn extract_tasks_from_dir(&self, dir: &Path) -> Result<Vec<Task>, Box<dyn std::error::Error>> {
        // Collect all directory entries
        let entries: Vec<_> = fs::read_dir(dir)?.collect::<Result<Vec<_>, _>>()?;

        // Process entries in parallel
        let tasks: Vec<Task> = entries
            .par_iter()
            .flat_map(|entry| {
                let path = entry.path();

                // Check if this path should be excluded
                if self.config.should_exclude(&path) {
                    return Vec::new();
                }

                if path.is_file() {
                    if path.extension().and_then(|s| s.to_str()) == Some("md") {
                        match self.extract_tasks_from_file(&path) {
                            Ok(file_tasks) => file_tasks,
                            Err(e) => {
                                eprintln!("Warning: Could not read {:?}: {}", path, e);
                                Vec::new()
                            }
                        }
                    } else {
                        Vec::new()
                    }
                } else if path.is_dir() {
                    // Recursively process subdirectories
                    match self.extract_tasks_from_dir(&path) {
                        Ok(dir_tasks) => dir_tasks,
                        Err(e) => {
                            eprintln!("Warning: Could not read directory {:?}: {}", path, e);
                            Vec::new()
                        }
                    }
                } else {
                    Vec::new()
                }
            })
            .collect();

        Ok(tasks)
    }

    fn parse_task_line(&self, line: &str, file_path: &Path, line_number: usize) -> Option<Task> {
        let line = line.trim_end_matches(&['\n', '\r'][..]);

        // Try incomplete pattern
        if let Some(caps) = self.task_incomplete.captures(line) {
            let content = caps.get(2).unwrap().as_str().to_string();
            return Some(self.create_task(
                content,
                "incomplete".to_string(),
                line,
                file_path,
                line_number,
            ));
        }

        // Try completed pattern
        if let Some(caps) = self.task_completed.captures(line) {
            let content = caps.get(2).unwrap().as_str().to_string();
            return Some(self.create_task(
                content,
                "completed".to_string(),
                line,
                file_path,
                line_number,
            ));
        }

        // Try cancelled pattern
        if let Some(caps) = self.task_cancelled.captures(line) {
            let content = caps.get(2).unwrap().as_str().to_string();
            return Some(self.create_task(
                content,
                "cancelled".to_string(),
                line,
                file_path,
                line_number,
            ));
        }

        // Try other pattern
        if let Some(caps) = self.task_other.captures(line) {
            let char = caps.get(2).unwrap().as_str();
            let content = caps.get(3).unwrap().as_str().to_string();

            // Skip if it matches standard patterns
            if char == "x" || char == "X" || char == " " || char == "-" {
                return None;
            }

            return Some(self.create_task(
                content,
                format!("other_{}", char),
                line,
                file_path,
                line_number,
            ));
        }

        None
    }

    fn create_task(
        &self,
        content: String,
        status: String,
        raw_line: &str,
        file_path: &Path,
        line_number: usize,
    ) -> Task {
        // Extract metadata from content
        let tags = self.extract_tags(&content);
        let due_date = self.extract_due_date(&content);
        let priority = self.extract_priority(&content);
        let created_date = self.extract_created_date(&content);
        let completed_date = self.extract_completed_date(&content);

        // Clean content by removing metadata
        let clean_content = self.clean_content(&content);

        Task {
            content: clean_content,
            status,
            file_path: file_path.to_string_lossy().to_string(),
            file_name: file_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            line_number,
            raw_line: raw_line.to_string(),
            tags,
            sub_items: Vec::new(),
            summary: None,
            due_date,
            priority,
            created_date,
            completed_date,
        }
    }
}

impl Default for TaskExtractor {
    fn default() -> Self {
        Self::new(Arc::new(Config::default()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn create_test_extractor() -> TaskExtractor {
        TaskExtractor::new(Arc::new(Config::default()))
    }

    mod parse_task_line {
        use super::*;

        #[test]
        fn test_unchecked_task() {
            let extractor = create_test_extractor();
            let path = PathBuf::from("test.md");
            let task = extractor.parse_task_line("- [ ] Test task", &path, 1);

            assert!(task.is_some());
            let task = task.unwrap();
            assert_eq!(task.status, "incomplete");
            assert_eq!(task.content, "Test task");
            assert_eq!(task.line_number, 1);
        }

        #[test]
        fn test_completed_task() {
            let extractor = create_test_extractor();
            let path = PathBuf::from("test.md");
            let task = extractor.parse_task_line("- [x] Completed task", &path, 1);

            assert!(task.is_some());
            let task = task.unwrap();
            assert_eq!(task.status, "completed");
            assert_eq!(task.content, "Completed task");
        }

        #[test]
        fn test_completed_task_uppercase() {
            let extractor = create_test_extractor();
            let path = PathBuf::from("test.md");
            let task = extractor.parse_task_line("- [X] Completed task", &path, 1);

            assert!(task.is_some());
            let task = task.unwrap();
            assert_eq!(task.status, "completed");
            assert_eq!(task.content, "Completed task");
        }

        #[test]
        fn test_cancelled_task() {
            let extractor = create_test_extractor();
            let path = PathBuf::from("test.md");
            let task = extractor.parse_task_line("- [-] Cancelled task", &path, 1);

            assert!(task.is_some());
            let task = task.unwrap();
            assert_eq!(task.status, "cancelled");
            assert_eq!(task.content, "Cancelled task");
        }

        #[test]
        fn test_other_status_task() {
            let extractor = create_test_extractor();
            let path = PathBuf::from("test.md");
            let task = extractor.parse_task_line("- [?] Unknown status", &path, 1);

            assert!(task.is_some());
            let task = task.unwrap();
            assert_eq!(task.status, "other_?");
            assert_eq!(task.content, "Unknown status");
        }

        #[test]
        fn test_task_with_leading_whitespace() {
            let extractor = create_test_extractor();
            let path = PathBuf::from("test.md");
            let task = extractor.parse_task_line("  - [ ] Indented task", &path, 1);

            assert!(task.is_some());
            let task = task.unwrap();
            assert_eq!(task.status, "incomplete");
            assert_eq!(task.content, "Indented task");
        }

        #[test]
        fn test_not_a_task() {
            let extractor = create_test_extractor();
            let path = PathBuf::from("test.md");
            let task = extractor.parse_task_line("This is just text", &path, 1);

            assert!(task.is_none());
        }

        #[test]
        fn test_regular_list_item() {
            let extractor = create_test_extractor();
            let path = PathBuf::from("test.md");
            let task = extractor.parse_task_line("- Regular list item", &path, 1);

            assert!(task.is_none());
        }
    }

    mod metadata_extraction {
        use super::*;

        #[test]
        fn test_extract_single_tag() {
            let extractor = create_test_extractor();
            let tags = extractor.extract_tags("Test task #work");

            assert_eq!(tags.len(), 1);
            assert_eq!(tags[0], "work");
        }

        #[test]
        fn test_extract_multiple_tags() {
            let extractor = create_test_extractor();
            let tags = extractor.extract_tags("Test task #work #urgent #project1");

            assert_eq!(tags.len(), 3);
            assert_eq!(tags[0], "work");
            assert_eq!(tags[1], "urgent");
            assert_eq!(tags[2], "project1");
        }

        #[test]
        fn test_extract_tags_with_numbers() {
            let extractor = create_test_extractor();
            let tags = extractor.extract_tags("Task #project2024 #v1");

            assert_eq!(tags.len(), 2);
            assert_eq!(tags[0], "project2024");
            assert_eq!(tags[1], "v1");
        }

        #[test]
        fn test_no_tags() {
            let extractor = create_test_extractor();
            let tags = extractor.extract_tags("Task with no tags");

            assert_eq!(tags.len(), 0);
        }

        #[test]
        fn test_hashtag_alone_no_match() {
            let extractor = create_test_extractor();
            let tags = extractor.extract_tags("Task with # alone");

            assert_eq!(tags.len(), 0);
        }

        #[test]
        fn test_extract_due_date_emoji() {
            let extractor = create_test_extractor();
            let date = extractor.extract_due_date("Task 📅 2025-12-10");

            assert!(date.is_some());
            assert_eq!(date.unwrap(), "2025-12-10");
        }

        #[test]
        fn test_extract_due_date_text() {
            let extractor = create_test_extractor();
            let date = extractor.extract_due_date("Task due: 2025-12-10");

            assert!(date.is_some());
            assert_eq!(date.unwrap(), "2025-12-10");
        }

        #[test]
        fn test_extract_due_date_function() {
            let extractor = create_test_extractor();
            let date = extractor.extract_due_date("Task @due(2025-12-10)");

            assert!(date.is_some());
            assert_eq!(date.unwrap(), "2025-12-10");
        }

        #[test]
        fn test_no_due_date() {
            let extractor = create_test_extractor();
            let date = extractor.extract_due_date("Task with no date");

            assert!(date.is_none());
        }

        #[test]
        fn test_extract_created_date_emoji() {
            let extractor = create_test_extractor();
            let date = extractor.extract_created_date("Task ➕ 2025-12-01");

            assert!(date.is_some());
            assert_eq!(date.unwrap(), "2025-12-01");
        }

        #[test]
        fn test_extract_created_date_text() {
            let extractor = create_test_extractor();
            let date = extractor.extract_created_date("Task created: 2025-12-01");

            assert!(date.is_some());
            assert_eq!(date.unwrap(), "2025-12-01");
        }

        #[test]
        fn test_extract_completed_date_emoji() {
            let extractor = create_test_extractor();
            let date = extractor.extract_completed_date("Task ✅ 2025-12-15");

            assert!(date.is_some());
            assert_eq!(date.unwrap(), "2025-12-15");
        }

        #[test]
        fn test_extract_completed_date_text() {
            let extractor = create_test_extractor();
            let date = extractor.extract_completed_date("Task completed: 2025-12-15");

            assert!(date.is_some());
            assert_eq!(date.unwrap(), "2025-12-15");
        }

        #[test]
        fn test_extract_priority_urgent_emoji() {
            let extractor = create_test_extractor();
            let priority = extractor.extract_priority("Task ⏫");

            assert!(priority.is_some());
            assert_eq!(priority.unwrap(), "urgent");
        }

        #[test]
        fn test_extract_priority_high_emoji() {
            let extractor = create_test_extractor();
            let priority = extractor.extract_priority("Task 🔼");

            assert!(priority.is_some());
            assert_eq!(priority.unwrap(), "high");
        }

        #[test]
        fn test_extract_priority_low_emoji() {
            let extractor = create_test_extractor();
            let priority = extractor.extract_priority("Task 🔽");

            assert!(priority.is_some());
            assert_eq!(priority.unwrap(), "low");
        }

        #[test]
        fn test_extract_priority_lowest_emoji() {
            let extractor = create_test_extractor();
            let priority = extractor.extract_priority("Task ⏬");

            assert!(priority.is_some());
            assert_eq!(priority.unwrap(), "lowest");
        }

        #[test]
        fn test_extract_priority_text_high() {
            let extractor = create_test_extractor();
            let priority = extractor.extract_priority("Task priority: high");

            assert!(priority.is_some());
            assert_eq!(priority.unwrap(), "high");
        }

        #[test]
        fn test_extract_priority_text_medium() {
            let extractor = create_test_extractor();
            let priority = extractor.extract_priority("Task priority: medium");

            assert!(priority.is_some());
            assert_eq!(priority.unwrap(), "medium");
        }

        #[test]
        fn test_extract_priority_text_low() {
            let extractor = create_test_extractor();
            let priority = extractor.extract_priority("Task priority: low");

            assert!(priority.is_some());
            assert_eq!(priority.unwrap(), "low");
        }

        #[test]
        fn test_no_priority() {
            let extractor = create_test_extractor();
            let priority = extractor.extract_priority("Task with no priority");

            assert!(priority.is_none());
        }
    }

    mod clean_content {
        use super::*;

        #[test]
        fn test_removes_due_date_emoji() {
            let extractor = create_test_extractor();
            let cleaned = extractor.clean_content("Task 📅 2025-12-10");

            assert_eq!(cleaned, "Task");
        }

        #[test]
        fn test_removes_due_date_text() {
            let extractor = create_test_extractor();
            let cleaned = extractor.clean_content("Task due: 2025-12-10");

            assert_eq!(cleaned, "Task");
        }

        #[test]
        fn test_removes_tags_preserved() {
            let extractor = create_test_extractor();
            let cleaned = extractor.clean_content("Task #work #urgent");

            // Tags are NOT removed by clean_content - they're part of the task description
            assert_eq!(cleaned, "Task #work #urgent");
        }

        #[test]
        fn test_removes_priority_emoji() {
            let extractor = create_test_extractor();
            let cleaned = extractor.clean_content("Task ⏫");

            assert_eq!(cleaned, "Task");
        }

        #[test]
        fn test_removes_priority_text() {
            let extractor = create_test_extractor();
            let cleaned = extractor.clean_content("Task priority: high");

            assert_eq!(cleaned, "Task");
        }

        #[test]
        fn test_removes_created_date() {
            let extractor = create_test_extractor();
            let cleaned = extractor.clean_content("Task ➕ 2025-12-01");

            assert_eq!(cleaned, "Task");
        }

        #[test]
        fn test_removes_completed_date() {
            let extractor = create_test_extractor();
            let cleaned = extractor.clean_content("Task ✅ 2025-12-15");

            assert_eq!(cleaned, "Task");
        }

        #[test]
        fn test_removes_timestamp() {
            let extractor = create_test_extractor();
            let cleaned = extractor.clean_content("10:30 Task description");

            assert_eq!(cleaned, "Task description");
        }

        #[test]
        fn test_removes_multiple_metadata() {
            let extractor = create_test_extractor();
            let cleaned = extractor.clean_content("Task 📅 2025-12-10 ⏫ #work");

            assert_eq!(cleaned, "Task #work");
        }

        #[test]
        fn test_cleans_extra_whitespace() {
            let extractor = create_test_extractor();
            let cleaned = extractor.clean_content("Task   with    extra    spaces");

            assert_eq!(cleaned, "Task with extra spaces");
        }

        #[test]
        fn test_preserves_task_text() {
            let extractor = create_test_extractor();
            let cleaned = extractor.clean_content("Write documentation for API");

            assert_eq!(cleaned, "Write documentation for API");
        }
    }

    mod sub_items {
        use super::*;

        #[test]
        fn test_is_sub_item_with_indent() {
            let extractor = create_test_extractor();
            let parent = "- [ ] Main task";
            let sub = "  - Sub item";

            assert!(extractor.is_sub_item(sub, parent));
        }

        #[test]
        fn test_is_sub_item_with_checkbox() {
            let extractor = create_test_extractor();
            let parent = "- [ ] Main task";
            let sub = "  - [ ] Sub task";

            assert!(extractor.is_sub_item(sub, parent));
        }

        #[test]
        fn test_is_not_sub_item_same_indent() {
            let extractor = create_test_extractor();
            let parent = "- [ ] Main task";
            let other = "- [ ] Another task";

            assert!(!extractor.is_sub_item(other, parent));
        }

        #[test]
        fn test_is_not_sub_item_empty_line() {
            let extractor = create_test_extractor();
            let parent = "- [ ] Main task";
            let empty = "";

            assert!(!extractor.is_sub_item(empty, parent));
        }

        #[test]
        fn test_is_sub_item_with_asterisk() {
            let extractor = create_test_extractor();
            let parent = "- [ ] Main task";
            let sub = "  * Sub item";

            assert!(extractor.is_sub_item(sub, parent));
        }

        #[test]
        fn test_parse_sub_item_regular_list() {
            let extractor = create_test_extractor();
            let sub = "  - Sub item text";

            let parsed = extractor.parse_sub_item(sub);
            assert!(parsed.is_some());
            assert_eq!(parsed.unwrap(), "Sub item text");
        }

        #[test]
        fn test_parse_sub_item_checkbox() {
            let extractor = create_test_extractor();
            let sub = "  - [ ] Sub task text";

            let parsed = extractor.parse_sub_item(sub);
            assert!(parsed.is_some());
            assert_eq!(parsed.unwrap(), "Sub task text");
        }

        #[test]
        fn test_parse_sub_item_with_asterisk() {
            let extractor = create_test_extractor();
            let sub = "  * Sub item with asterisk";

            let parsed = extractor.parse_sub_item(sub);
            assert!(parsed.is_some());
            assert_eq!(parsed.unwrap(), "Sub item with asterisk");
        }

        #[test]
        fn test_parse_sub_item_completed_checkbox() {
            let extractor = create_test_extractor();
            let sub = "  - [x] Completed sub task";

            let parsed = extractor.parse_sub_item(sub);
            assert!(parsed.is_some());
            assert_eq!(parsed.unwrap(), "Completed sub task");
        }

        #[test]
        fn test_parse_sub_item_not_list() {
            let extractor = create_test_extractor();
            let sub = "  Just some text";

            let parsed = extractor.parse_sub_item(sub);
            assert!(parsed.is_none());
        }
    }

    mod integration {
        use super::*;

        #[test]
        fn test_full_task_with_all_metadata() {
            let extractor = create_test_extractor();
            let path = PathBuf::from("test.md");
            let line = "- [ ] Write tests #testing ⏫ 📅 2025-12-10 ➕ 2025-12-01";

            let task = extractor.parse_task_line(line, &path, 5);
            assert!(task.is_some());

            let task = task.unwrap();
            assert_eq!(task.status, "incomplete");
            assert_eq!(task.content, "Write tests #testing");
            assert_eq!(task.line_number, 5);
            assert_eq!(task.tags, vec!["testing"]);
            assert_eq!(task.priority, Some("urgent".to_string()));
            assert_eq!(task.due_date, Some("2025-12-10".to_string()));
            assert_eq!(task.created_date, Some("2025-12-01".to_string()));
        }

        #[test]
        fn test_completed_task_with_completion_date() {
            let extractor = create_test_extractor();
            let path = PathBuf::from("test.md");
            let line = "- [x] Finished task ✅ 2025-12-15 #done";

            let task = extractor.parse_task_line(line, &path, 1);
            assert!(task.is_some());

            let task = task.unwrap();
            assert_eq!(task.status, "completed");
            assert_eq!(task.content, "Finished task #done");
            assert_eq!(task.completed_date, Some("2025-12-15".to_string()));
            assert_eq!(task.tags, vec!["done"]);
        }

        #[test]
        fn test_task_preserves_raw_line() {
            let extractor = create_test_extractor();
            let path = PathBuf::from("test.md");
            let line = "- [ ] Task with metadata 📅 2025-12-10 #work";

            let task = extractor.parse_task_line(line, &path, 1);
            assert!(task.is_some());

            let task = task.unwrap();
            assert_eq!(task.raw_line, line);
            // Content should be cleaned
            assert_eq!(task.content, "Task with metadata #work");
        }

        #[test]
        fn test_file_path_and_name() {
            let extractor = create_test_extractor();
            let path = PathBuf::from("/path/to/tasks.md");
            let line = "- [ ] Test task";

            let task = extractor.parse_task_line(line, &path, 1);
            assert!(task.is_some());

            let task = task.unwrap();
            assert_eq!(task.file_name, "tasks.md");
            assert!(task.file_path.contains("tasks.md"));
        }
    }
}
