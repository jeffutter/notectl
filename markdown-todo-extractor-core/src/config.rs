use glob::Pattern;
use serde::Deserialize;
use std::fs;
use std::path::Path;

pub fn default_daily_note_patterns() -> Vec<String> {
    vec!["YYYY-MM-DD.md".to_string()]
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub exclude_paths: Vec<String>,

    #[serde(default = "default_daily_note_patterns")]
    pub daily_note_patterns: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            exclude_paths: Vec::new(),
            daily_note_patterns: default_daily_note_patterns(),
        }
    }
}

impl Config {
    /// Load configuration from a file at the specified path
    /// Falls back to default config if file doesn't exist or can't be read
    pub fn load_from_file(config_path: &Path) -> Self {
        if !config_path.exists() {
            return Config::default();
        }

        match fs::read_to_string(config_path) {
            Ok(content) => toml::from_str(&content).unwrap_or_default(),
            Err(_) => Config::default(),
        }
    }

    /// Load configuration from the base path (looks for .markdown-todo-extractor.toml)
    /// Also merges configuration from environment variables
    pub fn load_from_base_path(base_path: &Path) -> Self {
        let config_path = base_path.join(".markdown-todo-extractor.toml");
        let mut config = Self::load_from_file(&config_path);

        // Merge in environment variable configuration
        config.merge_from_env();

        config
    }

    /// Merge configuration from environment variables
    /// MARKDOWN_TODO_EXTRACTOR_EXCLUDE_PATHS: comma-separated list of exclusion patterns
    /// MARKDOWN_TODO_EXTRACTOR_DAILY_NOTE_PATTERNS: comma-separated list of daily note patterns
    fn merge_from_env(&mut self) {
        self.merge_from_env_var("MARKDOWN_TODO_EXTRACTOR_EXCLUDE_PATHS");

        // Merge daily note patterns from environment variable
        if let Ok(env_patterns) = std::env::var("MARKDOWN_TODO_EXTRACTOR_DAILY_NOTE_PATTERNS") {
            let env_daily_patterns: Vec<String> = env_patterns
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            // Extend existing patterns with env var patterns
            self.daily_note_patterns.extend(env_daily_patterns);
        }
    }

    /// Merge configuration from a specific environment variable
    fn merge_from_env_var(&mut self, var_name: &str) {
        if let Ok(env_excludes) = std::env::var(var_name) {
            let env_patterns: Vec<String> = env_excludes
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            // Extend existing patterns with env var patterns
            self.exclude_paths.extend(env_patterns);
        }
    }

    /// Check if a given path should be excluded based on configured patterns
    pub fn should_exclude(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();

        for pattern_str in &self.exclude_paths {
            // Try to compile the pattern
            if let Ok(pattern) = Pattern::new(pattern_str)
                && pattern.matches(&path_str)
            {
                return true;
            }

            // Also check if the path contains the pattern as a substring
            // This handles simple cases like "Template" or "Recipes"
            if path_str.contains(pattern_str) {
                return true;
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_should_exclude_substring() {
        let config = Config {
            exclude_paths: vec!["Template".to_string(), "Recipes".to_string()],
            daily_note_patterns: default_daily_note_patterns(),
        };

        assert!(config.should_exclude(&PathBuf::from("/vault/Templates/note.md")));
        assert!(config.should_exclude(&PathBuf::from("/vault/Recipes/recipe.md")));
        assert!(!config.should_exclude(&PathBuf::from("/vault/Notes/note.md")));
    }

    #[test]
    fn test_should_exclude_glob_pattern() {
        let config = Config {
            exclude_paths: vec!["**/Template/**".to_string(), "**/Recipes/**".to_string()],
            daily_note_patterns: default_daily_note_patterns(),
        };

        assert!(config.should_exclude(&PathBuf::from("/vault/Template/note.md")));
        assert!(config.should_exclude(&PathBuf::from("/vault/Recipes/recipe.md")));
        assert!(config.should_exclude(&PathBuf::from("/vault/sub/Template/note.md")));
        assert!(!config.should_exclude(&PathBuf::from("/vault/Notes/note.md")));
    }

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.exclude_paths.is_empty());
        assert!(!config.should_exclude(&PathBuf::from("/vault/anything.md")));
    }

    #[test]
    fn test_merge_from_env() {
        // Use a unique env var name for this test to avoid parallel test conflicts
        const TEST_VAR: &str = "MARKDOWN_TODO_EXTRACTOR_TEST_MERGE_FROM_ENV";

        // Set env var
        unsafe {
            std::env::set_var(TEST_VAR, "Archive, Backup, **/tmp/**");
        }

        let mut config = Config {
            exclude_paths: vec!["Template".to_string()],
            daily_note_patterns: default_daily_note_patterns(),
        };

        config.merge_from_env_var(TEST_VAR);

        // Should have both original and env var patterns
        assert_eq!(config.exclude_paths.len(), 4);
        assert!(config.exclude_paths.contains(&"Template".to_string()));
        assert!(config.exclude_paths.contains(&"Archive".to_string()));
        assert!(config.exclude_paths.contains(&"Backup".to_string()));
        assert!(config.exclude_paths.contains(&"**/tmp/**".to_string()));

        // Clean up
        unsafe {
            std::env::remove_var(TEST_VAR);
        }
    }

    #[test]
    fn test_env_with_empty_patterns() {
        // Use a unique env var name for this test to avoid parallel test conflicts
        const TEST_VAR: &str = "MARKDOWN_TODO_EXTRACTOR_TEST_EMPTY_PATTERNS";

        // Test that empty strings are filtered out
        unsafe {
            std::env::set_var(TEST_VAR, "Archive, , Backup,  ,");
        }

        let mut config = Config::default();
        config.merge_from_env_var(TEST_VAR);

        assert_eq!(config.exclude_paths.len(), 2);
        assert!(config.exclude_paths.contains(&"Archive".to_string()));
        assert!(config.exclude_paths.contains(&"Backup".to_string()));

        // Clean up
        unsafe {
            std::env::remove_var(TEST_VAR);
        }
    }
}
