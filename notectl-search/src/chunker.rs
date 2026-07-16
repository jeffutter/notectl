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

        // If no real headings were found, fall back to size-based chunking.
        if sections.iter().all(|s| s.heading.title.is_empty()) {
            return self.chunk_by_size(content, path.to_string_lossy().to_string());
        }

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
                    // Count words in each original section to map merged-word indices back
                    // to the correct file-line offset.
                    let first_section_words: usize = section.content.split_whitespace().count();
                    // Byte offset of next_section's content within merged_content
                    // (section.content + "\n\n" + next_section.content).
                    let next_section_byte_offset = section.content.len() + 2;

                    // Build word-span map for merged_content before potentially moving it.
                    let merged_word_spans = Self::word_spans(&merged_content);

                    let chunk_parts: Vec<(String, usize, usize)> =
                        if merged_tokens > self.config.max_tokens {
                            self.split_long_text(&merged_content)
                        } else {
                            vec![(merged_content.clone(), 0, merged_tokens)]
                        };

                    for (j, &(ref part, start_idx, end_idx)) in chunk_parts.iter().enumerate() {
                        let (line_start, line_end) = if !merged_word_spans.is_empty()
                            && start_idx < merged_word_spans.len()
                        {
                            let first_span = &merged_word_spans[start_idx];
                            let last_idx =
                                end_idx.saturating_sub(1).min(merged_word_spans.len() - 1);
                            let last_span = &merged_word_spans[last_idx];

                            // Determine which original section each end falls in.
                            let first_in_first_section = start_idx < first_section_words;
                            let last_in_first_section = last_idx < first_section_words;

                            // start_line is 1-indexed heading line; content starts one line after
                            // the heading, so start_line equals the 0-indexed file line of the
                            // first content line.
                            let ls = if first_in_first_section {
                                section.start_line
                                    + Self::line_at(&section.content, first_span.start)
                            } else {
                                next_section.start_line
                                    + Self::line_at(
                                        &next_section.content,
                                        first_span.start - next_section_byte_offset,
                                    )
                            };

                            let le = if last_in_first_section {
                                section.start_line + Self::line_at(&section.content, last_span.end)
                            } else {
                                next_section.start_line
                                    + Self::line_at(
                                        &next_section.content,
                                        last_span.end - next_section_byte_offset,
                                    )
                            };

                            (ls, le)
                        } else {
                            (section.start_line, next_section.end_line)
                        };

                        chunks.push(Chunk {
                            id: format!(
                                "{}:{}:{}",
                                path.to_string_lossy(),
                                section.start_line + j,
                                heading_path.last().map(|s| s.as_str()).unwrap_or("merged")
                            ),
                            source_file: path.to_string_lossy().to_string(),
                            line_start,
                            line_end,
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
            let chunk_parts: Vec<(String, usize, usize)> =
                if section_tokens > self.config.max_tokens {
                    self.split_long_text(&section.content)
                } else {
                    vec![(section.content.clone(), 0, section_tokens)]
                };

            // Build word-span map for the section content so we can compute accurate
            // line numbers from character positions in the original file.
            let section_word_spans = Self::word_spans(&section.content);

            for (j, &(ref part, start_idx, end_idx)) in chunk_parts.iter().enumerate() {
                // Skip chunks that are too small
                if tokenize::count_tokens(part) < self.config.min_chunk_size
                    && chunk_parts.len() == 1
                {
                    continue;
                }

                let part_word_count = end_idx.saturating_sub(start_idx);

                // start_line is 1-indexed heading line; content starts one line after
                // the heading, so start_line equals the 0-indexed file line of the first
                // content line.
                let (line_start, line_end) =
                    if start_idx < section_word_spans.len() && part_word_count > 0 {
                        let first_span = &section_word_spans[start_idx];
                        let last_idx = end_idx.saturating_sub(1).min(section_word_spans.len() - 1);
                        let last_span = &section_word_spans[last_idx];
                        (
                            section.start_line + Self::line_at(&section.content, first_span.start),
                            section.start_line + Self::line_at(&section.content, last_span.end),
                        )
                    } else {
                        (section.start_line, section.end_line)
                    };

                chunks.push(Chunk {
                    id: format!(
                        "{}:{}:{}",
                        path.to_string_lossy(),
                        section.start_line + j,
                        heading_path.last().map(|s| s.as_str()).unwrap_or("section")
                    ),
                    source_file: path.to_string_lossy().to_string(),
                    line_start,
                    line_end,
                    heading: Some(section.heading.title.clone()),
                    heading_path: heading_path.clone(),
                    text: part.clone(),
                });
            }

            i += 1;
        }

        chunks
    }

    /// Compute the 0-indexed line number at a given character position in `content`.
    ///
    /// Counts newline bytes before `pos`, which is the correct 0-indexed line
    /// number for any position — not just positions that fall exactly on a
    /// line boundary. Using `.lines().count()` would over-count by 1 for
    /// mid-line positions because it treats a partial trailing line as a
    /// full line.
    fn line_at(content: &str, pos: usize) -> usize {
        let clamped = pos.min(content.len());
        content.as_bytes()[..clamped]
            .iter()
            .filter(|&&b| b == b'\n')
            .count()
    }

    /// Collect byte-span ranges for each whitespace-delimited word in `content`.
    fn word_spans(content: &str) -> Vec<std::ops::Range<usize>> {
        let bytes = content.as_bytes();
        let mut spans = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            // Skip whitespace.
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= bytes.len() {
                break;
            }
            let start = i;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            spans.push(start..i);
        }
        spans
    }

    /// Split text into fixed-size word chunks without overlap.
    fn chunk_by_size(&self, content: &str, file_name: String) -> Vec<Chunk> {
        // Pre-compute character spans of each whitespace-delimited word in the original content.
        let word_spans = Self::word_spans(content);
        let mut chunks = Vec::new();

        for chunk_words in word_spans.chunks(self.config.max_tokens) {
            let text: String = chunk_words
                .iter()
                .map(|s| &content[s.start..s.end])
                .collect::<Vec<_>>()
                .join(" ");
            if tokenize::count_tokens(&text) >= self.config.min_chunk_size {
                let first_pos = chunk_words.first().unwrap().start;
                let last_pos = chunk_words.last().unwrap().end;
                let line_start = Self::line_at(content, first_pos);
                let line_end = Self::line_at(content, last_pos);
                chunks.push(Chunk {
                    id: format!("{file_name}:{line_start}"),
                    source_file: file_name.clone(),
                    line_start,
                    line_end,
                    heading: None,
                    heading_path: Vec::new(),
                    text,
                });
            }
        }

        chunks
    }

    /// Split a long text into multiple overlapping chunks respecting token budget.
    /// Returns `(chunk_text, start_word_idx, end_word_idx_exclusive)` for each part,
    /// where indices are relative to the input text's word array and correctly account
    /// for overlap re-emission.
    fn split_long_text(&self, text: &str) -> Vec<(String, usize, usize)> {
        tokenize::tokenize_with_overlap_indexed(
            text,
            self.config.max_tokens,
            self.config.overlap_tokens,
        )
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

    // --- Acceptance criterion #3: multi-line content with no headings (chunk_by_size fallback) ---
    #[test]
    fn test_chunk_by_size_fallback_multi_line_line_numbers() {
        let config = ChunkerConfig {
            max_tokens: 4,
            min_chunk_size: 3,
            ..Default::default()
        };
        let chunker = Chunker::new(config);

        // Build multi-line content with no headings so it hits the size-fallback path.
        // Each line has exactly 2 words; max_tokens=4 means each chunk spans ~2 lines,
        // guaranteeing distinct line numbers across chunks.
        let mut lines: Vec<String> = Vec::new();
        for i in 0..20 {
            lines.push(format!("wordA{} wordB{}", i, i));
        }
        let content = lines.join("\n");

        let chunks = chunker.chunk_file(Path::new("test.md"), &content);
        assert!(
            chunks.len() > 1,
            "Expected multiple chunks from multi-line fallback content"
        );

        // Verify line_start values are monotonically non-decreasing.
        for w in chunks.windows(2) {
            assert!(
                w[1].line_start >= w[0].line_start,
                "line_start must be non-decreasing: {:?} then {:?}",
                w[0].line_start,
                w[1].line_start
            );
        }

        // Verify line_end > line_start for every chunk (valid span).
        for chunk in &chunks {
            assert!(
                chunk.line_end > chunk.line_start,
                "line_end must be > line_start: chunk {:?}",
                chunk.id
            );
        }

        // Verify the first chunk starts at line 0.
        assert_eq!(
            chunks[0].line_start, 0,
            "First chunk should start at line 0"
        );

        // Verify the last chunk's line_end reaches the last line of content.
        // line_end is the 0-indexed line number of the last word (inclusive).
        let expected_last_line = content.lines().count().saturating_sub(1);
        assert_eq!(
            chunks.last().unwrap().line_end,
            expected_last_line,
            "Last chunk should extend to last line of content"
        );
    }

    // --- Acceptance criterion #4: long section split into 3+ parts with distinct line spans ---
    #[test]
    fn test_long_section_split_distinct_line_spans() {
        let config = ChunkerConfig {
            max_tokens: 10,
            overlap_tokens: 0, // no overlap so each chunk has a unique line span
            min_chunk_size: 5,
            merge_threshold: 0, // disable merging
        };
        let chunker = Chunker::new(config);

        // Build a multi-line section with enough content to split into 3+ parts.
        // Each line has 3 words; max_tokens=10 means ~3-4 lines per chunk.
        let mut section_lines: Vec<String> = Vec::new();
        for i in 0..30 {
            section_lines.push(format!("wordA{} wordB{} wordC{}", i, i, i));
        }
        let section_content = section_lines.join("\n");
        let content = format!("# Big Section\n{}", section_content);

        let chunks = chunker.chunk_file(Path::new("test.md"), &content);

        // Filter to only the chunks from the "Big Section".
        let section_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| c.heading.as_deref() == Some("Big Section"))
            .collect();

        // Should produce 3+ chunks from splitting.
        assert!(
            section_chunks.len() >= 3,
            "Expected at least 3 split chunks, got {} (total chunks: {})",
            section_chunks.len(),
            chunks.len()
        );

        // All line_start values must be distinct (no collapsed/repeated values).
        let line_starts: Vec<usize> = section_chunks.iter().map(|c| c.line_start).collect();
        let unique_starts: std::collections::HashSet<usize> = line_starts.iter().copied().collect();
        assert_eq!(
            unique_starts.len(),
            line_starts.len(),
            "All chunks must have distinct line_start values"
        );

        // All line_end values must be distinct.
        let line_ends: Vec<usize> = section_chunks.iter().map(|c| c.line_end).collect();
        let unique_ends: std::collections::HashSet<usize> = line_ends.iter().copied().collect();
        assert_eq!(
            unique_ends.len(),
            line_ends.len(),
            "All chunks must have distinct line_end values"
        );

        // Line spans must be correctly ordered (non-decreasing).
        for w in section_chunks.windows(2) {
            assert!(
                w[1].line_start >= w[0].line_start,
                "Chunks must be in file order: chunk 0 starts at line {}, chunk 1 starts at line {}",
                w[0].line_start,
                w[1].line_start
            );
        }

        // First split chunk should start on file line 1 (0-indexed), which is the
        // second line of the file (the heading is on line 0).
        assert_eq!(
            section_chunks[0].line_start, 1,
            "First chunk of section should start at file line 1 (0-indexed)"
        );
    }

    // --- TASK-4 AC#1/#2/#3: section-split path with nonzero overlap_tokens ---
    #[test]
    fn test_long_section_split_overlap_nonzero_line_spans() {
        let config = ChunkerConfig {
            max_tokens: 10,
            overlap_tokens: 3, // nonzero overlap — word-window advances by 7 (not a multiple of 3 words/line)
            min_chunk_size: 5,
            merge_threshold: 0, // disable merging
        };
        let chunker = Chunker::new(config.clone());

        // Build a multi-line section with UNIQUE words per position so we can trace
        // exactly which word-index maps to which line.
        // Each line has 3 words; max_tokens=10 means ~3 lines per chunk, overlap=3.
        let mut section_lines: Vec<String> = Vec::new();
        for i in 0..30 {
            section_lines.push(format!("w{:02} x{:02} y{:02}", i, i, i));
        }
        let section_content = section_lines.join("\n");
        let content = format!("# Big Section\n{}", section_content);

        let chunks = chunker.chunk_file(Path::new("test.md"), &content);

        let section_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| c.heading.as_deref() == Some("Big Section"))
            .collect();

        assert!(
            section_chunks.len() >= 3,
            "Expected at least 3 split chunks, got {}",
            section_chunks.len()
        );

        // Independently compute expected line numbers WITHOUT calling Chunker::line_at.
        // For each chunk, find its first and last word, then locate those words in the
        // original file content by scanning lines directly.
        for chunk in &section_chunks {
            let first_word = chunk.text.split_whitespace().next().unwrap();
            let last_word = chunk.text.split_whitespace().last().unwrap();

            // True 0-indexed line of the first word in the full file content.
            // The heading is on line 0; section content starts on line 1.
            let true_line_start = content
                .lines()
                .position(|l| l.split_whitespace().any(|w| w == first_word))
                .unwrap();
            let true_line_end = content
                .lines()
                .position(|l| l.split_whitespace().any(|w| w == last_word))
                .unwrap();

            assert_eq!(
                chunk.line_start, true_line_start,
                "Chunk {:?} line_start mismatch: got {}, expected {} (first word '{}')",
                chunk.id, chunk.line_start, true_line_start, first_word,
            );
            assert_eq!(
                chunk.line_end, true_line_end,
                "Chunk {:?} line_end mismatch: got {}, expected {} (last word '{}')",
                chunk.id, chunk.line_end, true_line_end, last_word,
            );
        }

        // Line spans must be non-decreasing (chunks in file order).
        for w in section_chunks.windows(2) {
            assert!(
                w[1].line_start >= w[0].line_start,
                "Chunks must be in file order: chunk 0 starts at line {}, chunk 1 starts at line {}",
                w[0].line_start,
                w[1].line_start
            );
        }
    }

    // --- TASK-4 AC#1/#2/#3: merge-split path with nonzero overlap_tokens ---
    #[test]
    fn test_merged_section_split_overlap_nonzero_line_spans() {
        let config = ChunkerConfig {
            max_tokens: 10,
            overlap_tokens: 3, // nonzero overlap
            min_chunk_size: 3,
            merge_threshold: 8, // both sections below threshold → merge
        };
        let chunker = Chunker::new(config);

        // Two small sections with UNIQUE words that will be merged.
        // Section A: 6 words (< merge_threshold=8 tokens)
        // Section B: 6 words (< merge_threshold=8 tokens)
        // Merged = 12 words + "\n\n" separator, which exceeds max_tokens=10 → split.
        let content = r"# Title
## Small A
sa0 sa1 sa2
sa3 sa4 sa5

## Small B
sb0 sb1 sb2
sb3 sb4 sb5
";
        let chunks = chunker.chunk_file(Path::new("test.md"), content);

        // Should have produced multiple chunks from the merged sections.
        let merged_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| c.heading.as_deref() == Some("Small A"))
            .collect();

        assert!(
            merged_chunks.len() >= 2,
            "Expected at least 2 chunks from merged sections, got {}. Chunks: {:?}",
            merged_chunks.len(),
            merged_chunks.iter().map(|c| &c.text).collect::<Vec<_>>()
        );

        // Exact-match assertions: independently verify line_start/line_end against
        // the true position of each chunk's first and last word in the source content.
        for chunk in &merged_chunks {
            let first_word = chunk.text.split_whitespace().next().unwrap();
            let last_word = chunk.text.split_whitespace().last().unwrap();

            let true_line_start = content
                .lines()
                .position(|l| l.split_whitespace().any(|w| w == first_word))
                .unwrap();
            let true_line_end = content
                .lines()
                .position(|l| l.split_whitespace().any(|w| w == last_word))
                .unwrap();

            assert_eq!(
                chunk.line_start, true_line_start,
                "Merged chunk {:?} line_start mismatch: got {}, expected {} (first word '{}')",
                chunk.id, chunk.line_start, true_line_start, first_word,
            );
            assert_eq!(
                chunk.line_end, true_line_end,
                "Merged chunk {:?} line_end mismatch: got {}, expected {} (last word '{}')",
                chunk.id, chunk.line_end, true_line_end, last_word,
            );
        }

        // Verify line spans are non-decreasing (chunks in file order).
        for w in merged_chunks.windows(2) {
            assert!(
                w[1].line_start >= w[0].line_start,
                "Merged chunks must be in file order: {} then {}",
                w[0].line_start,
                w[1].line_start
            );
        }

        // Verify each chunk's line_start is within valid range of the file.
        let total_lines = content.lines().count();
        for (i, chunk) in merged_chunks.iter().enumerate() {
            assert!(
                chunk.line_start < total_lines,
                "Chunk {} line_start={} exceeds total lines {}",
                i,
                chunk.line_start,
                total_lines
            );
            assert!(
                chunk.line_end <= total_lines,
                "Chunk {} line_end={} exceeds total lines {}",
                i,
                chunk.line_end,
                total_lines
            );
            assert!(
                chunk.line_end > chunk.line_start,
                "Chunk {} has invalid span: line_start={} >= line_end={}",
                i,
                chunk.line_start,
                chunk.line_end
            );
        }

        // Verify the first chunk starts at the beginning of section A's content.
        // "## Small A" is on line 1 (0-indexed), content starts on line 2.
        assert_eq!(
            merged_chunks[0].line_start, 2,
            "First merged chunk should start at line 2 (content after '## Small A' heading)"
        );
    }
}
