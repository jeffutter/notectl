use crate::extractor::Task;
use serde::{Deserialize, Serialize};

/// Filter options for task search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterOptions {
    pub status: Option<String>,
    pub due_on: Option<String>,
    pub due_before: Option<String>,
    pub due_after: Option<String>,
    pub completed_on: Option<String>,
    pub completed_before: Option<String>,
    pub completed_after: Option<String>,
    pub tags: Option<Vec<String>>,
    pub exclude_tags: Option<Vec<String>>,
}

pub fn filter_tasks(tasks: Vec<Task>, options: &FilterOptions) -> Vec<Task> {
    tasks
        .into_iter()
        .filter(|task| {
            // Filter by status
            if let Some(ref status) = options.status
                && &task.status != status
            {
                return false;
            }

            // Filter by exact due date
            if let Some(ref due_on) = options.due_on
                && task.due_date.as_ref() != Some(due_on)
            {
                return false;
            }

            // Filter by due before date
            if let Some(ref due_before) = options.due_before {
                if let Some(ref due_date) = task.due_date {
                    if due_date >= due_before {
                        return false;
                    }
                } else {
                    return false;
                }
            }

            // Filter by due after date
            if let Some(ref due_after) = options.due_after {
                if let Some(ref due_date) = task.due_date {
                    if due_date <= due_after {
                        return false;
                    }
                } else {
                    return false;
                }
            }

            // Filter by exact completed date
            if let Some(ref completed_on) = options.completed_on
                && task.completed_date.as_ref() != Some(completed_on)
            {
                return false;
            }

            // Filter by completed before date
            if let Some(ref completed_before) = options.completed_before {
                if let Some(ref completed_date) = task.completed_date {
                    if completed_date >= completed_before {
                        return false;
                    }
                } else {
                    return false;
                }
            }

            // Filter by completed after date
            if let Some(ref completed_after) = options.completed_after {
                if let Some(ref completed_date) = task.completed_date {
                    if completed_date <= completed_after {
                        return false;
                    }
                } else {
                    return false;
                }
            }

            // Filter by tags (must have all specified tags)
            if let Some(ref tags) = options.tags
                && !tags.iter().all(|tag| task.tags.contains(tag))
            {
                return false;
            }

            // Filter by excluded tags (must not have any specified tags)
            if let Some(ref exclude_tags) = options.exclude_tags
                && exclude_tags.iter().any(|tag| task.tags.contains(tag))
            {
                return false;
            }

            true
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_task(
        content: &str,
        status: &str,
        due_date: Option<&str>,
        completed_date: Option<&str>,
        tags: Vec<&str>,
    ) -> Task {
        Task {
            content: content.to_string(),
            status: status.to_string(),
            file_path: "test.md".to_string(),
            file_name: "test.md".to_string(),
            line_number: 1,
            raw_line: format!(
                "- [{}] {}",
                if status == "incomplete" { " " } else { "x" },
                content
            ),
            tags: tags.iter().map(|s| s.to_string()).collect(),
            sub_items: vec![],
            summary: None,
            due_date: due_date.map(String::from),
            priority: None,
            created_date: None,
            completed_date: completed_date.map(String::from),
        }
    }

    #[test]
    fn test_no_filters_returns_all_tasks() {
        let tasks = vec![
            create_test_task("Task 1", "incomplete", None, None, vec![]),
            create_test_task("Task 2", "complete", None, None, vec![]),
        ];

        let options = FilterOptions {
            status: None,
            due_on: None,
            due_before: None,
            due_after: None,
            completed_on: None,
            completed_before: None,
            completed_after: None,
            tags: None,
            exclude_tags: None,
        };

        let filtered = filter_tasks(tasks.clone(), &options);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_empty_task_list() {
        let tasks: Vec<Task> = vec![];
        let options = FilterOptions {
            status: Some("incomplete".to_string()),
            due_on: None,
            due_before: None,
            due_after: None,
            completed_on: None,
            completed_before: None,
            completed_after: None,
            tags: None,
            exclude_tags: None,
        };

        let filtered = filter_tasks(tasks, &options);
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_status_filter_incomplete() {
        let tasks = vec![
            create_test_task("Task 1", "incomplete", None, None, vec![]),
            create_test_task("Task 2", "complete", None, None, vec![]),
            create_test_task("Task 3", "incomplete", None, None, vec![]),
        ];

        let options = FilterOptions {
            status: Some("incomplete".to_string()),
            due_on: None,
            due_before: None,
            due_after: None,
            completed_on: None,
            completed_before: None,
            completed_after: None,
            tags: None,
            exclude_tags: None,
        };

        let filtered = filter_tasks(tasks, &options);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|t| t.status == "incomplete"));
    }

    #[test]
    fn test_single_tag_filter() {
        let tasks = vec![
            create_test_task("Task 1", "incomplete", None, None, vec!["work"]),
            create_test_task("Task 2", "incomplete", None, None, vec!["personal"]),
            create_test_task("Task 3", "incomplete", None, None, vec!["work", "urgent"]),
        ];

        let options = FilterOptions {
            status: None,
            due_on: None,
            due_before: None,
            due_after: None,
            completed_on: None,
            completed_before: None,
            completed_after: None,
            tags: Some(vec!["work".to_string()]),
            exclude_tags: None,
        };

        let filtered = filter_tasks(tasks, &options);
        assert_eq!(filtered.len(), 2);
        assert!(
            filtered
                .iter()
                .all(|t| t.tags.contains(&"work".to_string()))
        );
    }
}
