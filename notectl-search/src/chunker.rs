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

impl ChunkerConfig {
    /// Build a ChunkerConfig from the authoritative SearchConfig.
    pub fn from_search_config(sc: &notectl_core::config::SearchConfig) -> Self {
        Self {
            max_tokens: sc.max_seq_tokens,
            overlap_tokens: sc.chunk_overlap_tokens,
            min_chunk_size: sc.min_chunk_tokens,
            merge_threshold: sc.merge_threshold,
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
        // Stack of (level, title) tracking current ancestor headings.
        // For each section we pop entries with level >= current to find
        // the true ancestors, then snapshot the remaining stack as the path.
        let mut ancestor_stack: Vec<(usize, String)> = Vec::new();

        while i < sections.len() {
            let section = &sections[i];
            let section_tokens = tokenize::count_tokens(&section.content);

            // Compute heading path via forward stack pass
            // Pop all ancestors at same or deeper level than current section
            while ancestor_stack
                .last()
                .is_some_and(|&(lvl, _)| lvl >= section.heading.level as usize)
            {
                ancestor_stack.pop();
            }
            let heading_path: Vec<String> = ancestor_stack
                .iter()
                .filter(|(_, title)| !title.is_empty())
                .map(|(_, title)| title.clone())
                .collect();
            // Push current section onto the stack for future sections' use
            if !section.heading.title.is_empty() {
                ancestor_stack.push((
                    section.heading.level as usize,
                    section.heading.title.clone(),
                ));
            }

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
        // Disable merging so each section produces its own chunk for path testing.
        let config = ChunkerConfig {
            merge_threshold: 0,
            min_chunk_size: 1,
            ..Default::default()
        };
        let chunker = Chunker::new(config);
        let content = r#"# Main Title
## Chapter 1
### Section 1.1
Some content for section 1.1 to make it non-trivial.

## Chapter 2
Some content for chapter 2 as well.
"#;
        let chunks = chunker.chunk_file(Path::new("test.md"), content);

        // Build a map of heading -> path from the chunks for inspection
        let mut heading_to_path: std::collections::HashMap<&str, Vec<String>> =
            std::collections::HashMap::new();
        for chunk in &chunks {
            if let Some(ref h) = chunk.heading {
                heading_to_path.insert(h.as_str(), chunk.heading_path.clone());
            }
        }

        // Section 1.1 (H3) should have path ["Main Title", "Chapter 1"]
        let s11_path = heading_to_path.get("Section 1.1").unwrap();
        assert_eq!(
            s11_path,
            &["Main Title".to_string(), "Chapter 1".to_string()]
        );

        // Chapter 2 (H2) should have path ["Main Title"] — this is the bug fix.
        // The old backward-walk algorithm failed here because it hit ### Section 1.1
        // (level 3 >= level 2) and stopped before reaching Main Title.
        let ch2_path = heading_to_path.get("Chapter 2").unwrap();
        assert_eq!(ch2_path, &["Main Title".to_string()]);
    }

    #[test]
    fn test_heading_path_sibling_sections() {
        // H1(A), H2(B), H1(C): C's path should NOT include A or B.
        let config = ChunkerConfig {
            merge_threshold: 0,
            min_chunk_size: 1,
            ..Default::default()
        };
        let chunker = Chunker::new(config);
        let content = r#"# Alpha
## Beta
Some beta content here for testing.

# Charlie
Charlie content goes here for testing.
"#;
        let chunks = chunker.chunk_file(Path::new("test.md"), content);

        let mut heading_to_path: std::collections::HashMap<&str, Vec<String>> =
            std::collections::HashMap::new();
        for chunk in &chunks {
            if let Some(ref h) = chunk.heading {
                heading_to_path.insert(h.as_str(), chunk.heading_path.clone());
            }
        }

        // Charlie (H1) is its own root — no ancestors.
        let charlie_path = heading_to_path.get("Charlie").unwrap();
        assert_eq!(charlie_path, &Vec::<String>::new());

        // Beta (H2 under Alpha) should have ["Alpha"]
        let beta_path = heading_to_path.get("Beta").unwrap();
        assert_eq!(beta_path, &["Alpha".to_string()]);
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

    #[test]
    fn test_chunker_config_from_search_config() {
        use notectl_core::config::SearchConfig;

        let sc = SearchConfig {
            max_seq_tokens: 1024,
            chunk_overlap_tokens: 128,
            min_chunk_tokens: 64,
            merge_threshold: 50,
            ..Default::default()
        };

        let cc = ChunkerConfig::from_search_config(&sc);
        assert_eq!(cc.max_tokens, 1024);
        assert_eq!(cc.overlap_tokens, 128);
        assert_eq!(cc.min_chunk_size, 64);
        assert_eq!(cc.merge_threshold, 50);
    }
}
