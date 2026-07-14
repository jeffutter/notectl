use std::path::Path;

/// A text chunk produced by the chunker
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Chunk {
    /// Unique identifier for this chunk (file_path:line_offset)
    pub id: String,
    /// The markdown file this chunk came from
    pub source_file: String,
    /// Starting line offset in the source file
    pub line_start: usize,
    /// Ending line offset in the source file
    pub line_end: usize,
    /// The heading that anchors this section (if any)
    pub heading: Option<String>,
    /// The raw text content of this chunk
    pub text: String,
}

/// Configuration for the chunker
#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    /// Maximum tokens per chunk (default: 512)
    pub max_tokens: usize,
    /// Minimum tokens to keep a chunk (default: 50)
    pub min_chunk_size: usize,
    /// Split by heading level (1-6, default: all levels)
    pub split_by_heading: bool,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            min_chunk_size: 50,
            split_by_heading: true,
        }
    }
}

/// Markdown heading regex pattern
const HEADING_PATTERN: &str = r"^(#{1,6})\s+(.+)$";

/// Chunker that splits markdown files into searchable chunks
pub struct Chunker {
    config: ChunkerConfig,
}

impl Chunker {
    pub fn new(config: ChunkerConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(ChunkerConfig::default())
    }

    /// Chunk a single markdown file's contents
    pub fn chunk_file(&self, path: &Path, content: &str) -> Vec<Chunk> {
        if !self.config.split_by_heading {
            return self.chunk_by_size(content, path.to_string_lossy().to_string());
        }

        let sections = self.split_by_headings(content);
        let mut chunks = Vec::new();

        for (heading, text) in sections {
            let file_name = path.to_string_lossy().to_string();
            let line_start = content[..content.find(text.as_str()).unwrap_or(0)]
                .lines()
                .count();

            if text.len() < self.config.min_chunk_size {
                // Too small - skip or merge with previous (simplified: skip)
                continue;
            }

            let chunk_text = if text.len() > self.config.max_tokens * 4 {
                // Long section - split by size
                self.split_long_text(&text, &file_name, line_start)
            } else {
                vec![text]
            };

            for (i, part) in chunk_text.iter().enumerate() {
                chunks.push(Chunk {
                    id: format!(
                        "{}:{}:{}",
                        file_name,
                        line_start + i,
                        heading.as_deref().unwrap_or("unknown")
                    ),
                    source_file: file_name.clone(),
                    line_start,
                    line_end: line_start + part.lines().count(),
                    heading: heading.clone(),
                    text: part.clone(),
                });
            }
        }

        chunks
    }

    /// Split content by markdown headings
    fn split_by_headings(&self, content: &str) -> Vec<(Option<String>, String)> {
        let re = regex::Regex::new(HEADING_PATTERN).unwrap();
        let mut sections = Vec::new();
        let mut current_heading: Option<String> = None;
        let mut current_text = String::new();

        for line in content.lines() {
            if let Some(captures) = re.captures(line) {
                // Save previous section
                if !current_text.is_empty() {
                    sections.push((current_heading.clone(), current_text.clone()));
                }
                current_heading = Some(captures[2].to_string());
                current_text = String::new();
            } else {
                if !current_text.is_empty() {
                    current_text.push('\n');
                }
                current_text.push_str(line);
            }
        }

        // Don't forget the last section
        if !current_text.is_empty() {
            sections.push((current_heading, current_text));
        }

        sections
    }

    /// Split text into fixed-size chunks (fallback for long sections)
    fn chunk_by_size(&self, content: &str, file_name: String) -> Vec<Chunk> {
        let words: Vec<&str> = content.split_whitespace().collect();
        let mut chunks = Vec::new();

        for chunk_words in words.chunks(self.config.max_tokens) {
            let text = chunk_words.join(" ");
            if text.len() >= self.config.min_chunk_size {
                let line_start = content
                    .find(&text)
                    .map_or(0, |pos| content[..pos].lines().count());
                chunks.push(Chunk {
                    id: format!("{file_name}:{line_start}"),
                    source_file: file_name.clone(),
                    line_start,
                    line_end: line_start + text.lines().count(),
                    heading: None,
                    text,
                });
            }
        }

        chunks
    }

    /// Split a long text into multiple chunks respecting word boundaries
    fn split_long_text(&self, text: &str, _file_name: &str, _line_start: usize) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut parts = Vec::new();
        let mut current = String::new();

        for word in words {
            if current.len() + word.len() + 1 > self.config.max_tokens * 4 && !current.is_empty() {
                parts.push(current.clone());
                current.clear();
            }
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(word);
        }

        if !current.is_empty() {
            parts.push(current);
        }

        parts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_by_headings() {
        let chunker = Chunker::default_config();
        let content = "# Title\nSome text.\n## Section 1\nMore text.\n## Section 2\nEven more.";
        let sections = chunker.split_by_headings(content);

        assert_eq!(sections.len(), 3);
        assert_eq!(sections[0].0, Some("Title".to_string()));
        assert_eq!(sections[1].0, Some("Section 1".to_string()));
        assert_eq!(sections[2].0, Some("Section 2".to_string()));
    }

    #[test]
    fn test_chunk_file_basic() {
        let config = ChunkerConfig {
            min_chunk_size: 5,
            ..Default::default()
        };
        let chunker = Chunker::new(config);
        let content =
            "# My Note\nHello world. This is a longer piece of text for testing chunking.";
        let chunks = chunker.chunk_file(Path::new("test.md"), content);

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunk_by_size_fallback() {
        let config = ChunkerConfig {
            split_by_heading: false,
            max_tokens: 10,
            min_chunk_size: 5,
        };
        let chunker = Chunker::new(config);
        let content = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12";
        let chunks = chunker.chunk_file(Path::new("test.md"), content);

        assert!(!chunks.is_empty());
    }
}
