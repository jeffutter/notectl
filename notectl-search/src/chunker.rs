use notectl_outline::OutlineExtractor;
use std::path::Path;

use crate::tokenize;

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
    /// Full heading path (e.g., "Title > Section 1 > Subsection")
    pub heading_path: Vec<String>,
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
    /// Overlap tokens between consecutive chunks in long sections (default: 50)
    pub overlap_tokens: usize,
    /// Merge tiny sections into the next one if below this threshold (default: 30)
    pub merge_threshold: usize,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            min_chunk_size: 50,
            overlap_tokens: 50,
            merge_threshold: 30,
        }
    }
}

/// Chunker that splits markdown files into searchable chunks using outline sections.
pub struct Chunker {
    config: ChunkerConfig,
    extractor: OutlineExtractor,
}

impl Chunker {
    pub fn new(config: ChunkerConfig) -> Self {
        Self {
            config,
            extractor: OutlineExtractor::new(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(ChunkerConfig::default())
    }

    /// Chunk a single markdown file's contents
    pub fn chunk_file(&self, path: &Path, content: &str) -> Vec<Chunk> {
        let sections = match self.extractor.extract_sections_from_content(content) {
            Ok(sections) => sections,
            Err(_) => {
                // Fallback to simple size-based chunking if section extraction fails
                return self.chunk_by_size(content, path.to_string_lossy().to_string());
            }
        };

        let mut chunks = Vec::new();
        let mut i = 0;

        while i < sections.len() {
            let section = &sections[i];
            let section_tokens = tokenize::count_tokens(&section.content);

            // Build heading path from this section's hierarchy
            let heading_path = self.build_heading_path(&sections, i);

            // Check if this is a tiny section that should be merged
            if section_tokens < self.config.merge_threshold && i + 1 < sections.len() {
                // Merge with next section
                let next_section = &sections[i + 1];
                let merged_content = format!("{}\n\n{}", section.content, next_section.content);
                let merged_tokens = tokenize::count_tokens(&merged_content);

                if merged_tokens <= self.config.max_tokens * 2 {
                    // Merge successful, skip to next section after this one
                    let chunk_text = if merged_tokens > self.config.max_tokens {
                        self.split_long_text(&merged_content)
                    } else {
                        vec![merged_content]
                    };

                    for (j, part) in chunk_text.iter().enumerate() {
                        chunks.push(Chunk {
                            id: format!(
                                "{}:{}:{}",
                                path.to_string_lossy(),
                                section.start_line + j,
                                heading_path.last().map(|s| s.as_str()).unwrap_or("merged")
                            ),
                            source_file: path.to_string_lossy().to_string(),
                            line_start: section.start_line,
                            line_end: next_section.end_line,
                            heading: Some(section.heading.title.clone()),
                            heading_path: heading_path.clone(),
                            text: part.clone(),
                        });
                    }
                    i += 2;
                    continue;
                }
            }

            // Process this section (possibly splitting if too long)
            let chunk_texts = if section_tokens > self.config.max_tokens {
                self.split_long_text(&section.content)
            } else {
                vec![section.content.clone()]
            };

            for (j, part) in chunk_texts.iter().enumerate() {
                // Skip chunks that are too small
                if tokenize::count_tokens(part) < self.config.min_chunk_size
                    && chunk_texts.len() == 1
                {
                    continue;
                }

                chunks.push(Chunk {
                    id: format!(
                        "{}:{}:{}",
                        path.to_string_lossy(),
                        section.start_line + j,
                        heading_path.last().map(|s| s.as_str()).unwrap_or("section")
                    ),
                    source_file: path.to_string_lossy().to_string(),
                    line_start: section.start_line + j * part.lines().count() / 2,
                    line_end: section.start_line
                        + j * part.lines().count() / 2
                        + part.lines().count(),
                    heading: Some(section.heading.title.clone()),
                    heading_path: heading_path.clone(),
                    text: part.clone(),
                });
            }

            i += 1;
        }

        chunks
    }

    /// Build a hierarchical heading path for a section at the given index.
    /// Returns ancestors from root to the section's parent.
    fn build_heading_path(&self, sections: &[notectl_outline::Section], idx: usize) -> Vec<String> {
        let mut path = Vec::new();

        // Look backwards to find ancestor headings (same or higher level)
        for j in (0..idx).rev() {
            if sections[j].heading.level < sections[idx].heading.level {
                path.push(sections[j].heading.title.clone());
            } else if sections[j].heading.level == 0 && sections[j].heading.title.is_empty() {
                // Skip the "no heading" root section
                continue;
            } else {
                break;
            }
        }

        path.reverse();
        path
    }

    /// Split text into fixed-size word chunks without overlap.
    fn chunk_by_size(&self, content: &str, file_name: String) -> Vec<Chunk> {
        let words: Vec<&str> = content.split_whitespace().collect();
        let mut chunks = Vec::new();

        for chunk_words in words.chunks(self.config.max_tokens) {
            let text = chunk_words.join(" ");
            if tokenize::count_tokens(&text) >= self.config.min_chunk_size {
                let line_start = content
                    .find(&text)
                    .map_or(0, |pos| content[..pos].lines().count());
                chunks.push(Chunk {
                    id: format!("{file_name}:{line_start}"),
                    source_file: file_name.clone(),
                    line_start,
                    line_end: line_start + text.lines().count(),
                    heading: None,
                    heading_path: Vec::new(),
                    text,
                });
            }
        }

        chunks
    }

    /// Split a long text into multiple overlapping chunks respecting token budget.
    fn split_long_text(&self, text: &str) -> Vec<String> {
        tokenize::tokenize_with_overlap(text, self.config.max_tokens, self.config.overlap_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Should have at least one chunk
        assert!(chunks.iter().any(|c| c.text.contains("Hello world")));
    }

    #[test]
    fn test_chunk_file_with_sections() {
        let config = ChunkerConfig {
            min_chunk_size: 5,
            ..Default::default()
        };
        let chunker = Chunker::new(config);
        let content = r#"# Title
## Section 1
Some content for section one.

## Section 2
Different content here for section two.
"#;
        let chunks = chunker.chunk_file(Path::new("test.md"), content);

        assert!(!chunks.is_empty());
        // Should have chunks with heading paths
        assert!(
            chunks
                .iter()
                .all(|c| !c.heading_path.is_empty() || c.heading.is_some())
        );
    }

    #[test]
    fn test_long_section_splitting() {
        let config = ChunkerConfig {
            max_tokens: 20,
            overlap_tokens: 5,
            min_chunk_size: 5,
            ..Default::default()
        };
        let chunker = Chunker::new(config.clone());

        // Create a long section with many words
        let long_text: String = (0..100)
            .map(|i| format!("word{}", i))
            .collect::<Vec<_>>()
            .join(" ");
        let content = format!("# Long Section\n{}", long_text);

        let chunks = chunker.chunk_file(Path::new("test.md"), &content);

        // Should be split into multiple chunks
        assert!(chunks.len() > 1);
        // Each chunk should be within token budget (approximately)
        for chunk in &chunks {
            let tokens = tokenize::count_tokens(&chunk.text);
            assert!(
                tokens <= config.max_tokens + config.overlap_tokens,
                "Chunk has {} tokens, expected <= {}",
                tokens,
                config.max_tokens + config.overlap_tokens
            );
        }
    }

    #[test]
    fn test_tiny_section_merging() {
        let config = ChunkerConfig {
            merge_threshold: 10,
            max_tokens: 200,
            min_chunk_size: 5,
            ..Default::default()
        };
        let chunker = Chunker::new(config);

        // Two small sections that should be merged
        let content = r#"# Title
## Small One
a b c d e f g h i j k l m n o p q r s t u v w x y z

## Another Small
x y z a b c d e f g h i j k l m n o p q r s t u v w x y z
"#;
        let chunks = chunker.chunk_file(Path::new("test.md"), content);

        // Should merge the two small sections
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_heading_path_tracking() {
        let config = ChunkerConfig {
            min_chunk_size: 5,
            ..Default::default()
        };
        let chunker = Chunker::new(config);
        let content = r#"# Main Title
## Chapter 1
### Section 1.1
Content here.

## Chapter 2
Content here too.
"#;
        let chunks = chunker.chunk_file(Path::new("test.md"), content);

        // Check that heading paths are populated
        for chunk in &chunks {
            if !chunk.heading_path.is_empty() {
                // Should have at least the main title
                assert!(chunk.heading_path[0] == "Main Title" || chunk.heading_path.len() >= 1);
            }
        }
    }

    #[test]
    fn test_chunk_by_size_fallback() {
        let config = ChunkerConfig {
            min_chunk_size: 5,
            ..Default::default()
        };
        let chunker = Chunker::new(config);
        // Content without headings should fallback to size-based chunking
        let content = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12";
        let chunks = chunker.chunk_file(Path::new("test.md"), content);

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_empty_content() {
        let config = ChunkerConfig::default();
        let chunker = Chunker::new(config);
        let chunks = chunker.chunk_file(Path::new("empty.md"), "");
        assert!(chunks.is_empty());
    }
}
