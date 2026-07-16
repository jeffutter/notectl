//! Simple word-based tokenizer for estimating chunk sizes in tokens.
//!
//! This is a lightweight approximation suitable for the chunker's token budget logic.
//! It counts whitespace-separated words as tokens, which is a reasonable approximation
//! for embedding models that use subword tokenization (typically 1-2 tokens per word).

/// Count approximate tokens in text by counting whitespace-separated words.
/// This is an approximation - actual tokenizers may produce different counts.
pub fn count_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Split text into overlapping chunks, returning each chunk's text along with its
/// start and end word indices (relative to the input's word array).
///
/// Returns `Vec<(chunk_text, start_word_idx, end_word_idx_exclusive)>` where
/// `start..end` covers exactly the words that produced the chunk.
/// Unlike the plain variant, the indices reflect the actual windowing logic,
/// so callers can map each chunk back to precise positions in the source.
pub fn tokenize_with_overlap_indexed(
    text: &str,
    max_tokens: usize,
    overlap_tokens: usize,
) -> Vec<(String, usize, usize)> {
    if max_tokens == 0 {
        return vec![(String::new(), 0, 0)];
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![(String::new(), 0, 0)];
    }

    // Clamp overlap so that each chunk advances by at least one word.
    let overlap = overlap_tokens.min(max_tokens.saturating_sub(1));

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < words.len() {
        let end = std::cmp::min(start + max_tokens, words.len());
        let chunk: String = words[start..end].join(" ");
        chunks.push((chunk, start, end));

        // Move forward, but leave overlap for context
        let advance = if end >= words.len() {
            words.len()
        } else {
            end - overlap
        };

        start = if advance <= start { end } else { advance };
    }

    chunks
}

/// Split text into chunks respecting a maximum token budget.
/// Splits at word boundaries and produces overlapping windows when the text exceeds the budget.
///
/// # Arguments
/// * `text` - The text to split
/// * `max_tokens` - Maximum tokens per chunk
/// * `overlap_tokens` - Number of tokens to overlap between consecutive chunks (for context)
///
/// # Returns
/// A vector of text chunks, each within the token budget.
pub fn tokenize_with_overlap(text: &str, max_tokens: usize, overlap_tokens: usize) -> Vec<String> {
    tokenize_with_overlap_indexed(text, max_tokens, overlap_tokens)
        .into_iter()
        .map(|(t, _, _)| t)
        .collect()
}

/// Split text into fixed-size word chunks without overlap.
pub fn tokenize_fixed(text: &str, max_tokens: usize) -> Vec<String> {
    if max_tokens == 0 {
        return vec![String::new()];
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![String::new()];
    }

    words
        .chunks(max_tokens)
        .map(|chunk| chunk.join(" "))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_tokens_simple() {
        assert_eq!(count_tokens("hello world"), 2);
        assert_eq!(count_tokens("one two three four"), 4);
        assert_eq!(count_tokens(""), 0);
        assert_eq!(count_tokens("   "), 0);
    }

    #[test]
    fn test_count_tokens_with_extra_whitespace() {
        assert_eq!(count_tokens("hello   world"), 2);
        assert_eq!(count_tokens("\thello\tworld\n"), 2);
    }

    #[test]
    fn test_tokenize_fixed_simple() {
        let text = "one two three four five six seven eight";
        let chunks = tokenize_fixed(text, 4);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "one two three four");
        assert_eq!(chunks[1], "five six seven eight");
    }

    #[test]
    fn test_tokenize_fixed_remainder() {
        let text = "a b c d e f";
        let chunks = tokenize_fixed(text, 4);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "a b c d");
        assert_eq!(chunks[1], "e f");
    }

    #[test]
    fn test_tokenize_with_overlap() {
        let text = "a b c d e f g h i j";
        let chunks = tokenize_with_overlap(text, 4, 2);
        // First chunk: a b c d
        // Second chunk starts at index 4-2=2: c d e f
        // Third chunk starts at index 6-2=4: e f g h
        // Fourth chunk starts at index 8-2=6: g h i j
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], "a b c d");
        assert_eq!(chunks[1], "c d e f");
        assert_eq!(chunks[2], "e f g h");
        assert_eq!(chunks[3], "g h i j");
    }

    #[test]
    fn test_tokenize_with_overlap_no_remainder() {
        let text = "a b c d e f";
        let chunks = tokenize_with_overlap(text, 3, 1);
        // First: a b c
        // Second starts at 3-1=2: c d e
        // Third starts at 5-1=4: e f (two words left)
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "a b c");
        assert_eq!(chunks[1], "c d e");
        assert_eq!(chunks[2], "e f");
    }

    #[test]
    fn test_tokenize_empty() {
        assert_eq!(tokenize_fixed("", 4), vec![String::new()]);
        assert_eq!(tokenize_with_overlap("", 4, 0), vec![String::new()]);
    }

    #[test]
    fn test_tokenize_zero_max() {
        assert_eq!(tokenize_fixed("hello", 0), vec![String::new()]);
        assert_eq!(tokenize_with_overlap("hello", 0, 0), vec![String::new()]);
    }

    #[test]
    fn test_overlap_ge_max_tokens() {
        // overlap_tokens (10) > max_tokens (4): should clamp to 3 and produce chunks without panicking.
        // With overlap clamped to 3, each chunk advances by 1 word.
        let text = "a b c d e f g h i j";
        let chunks = tokenize_with_overlap(text, 4, 10);
        assert_eq!(chunks.len(), 7);
        assert_eq!(chunks[0], "a b c d");
        assert_eq!(chunks[1], "b c d e");
        assert_eq!(chunks[6], "g h i j");
    }

    #[test]
    fn test_overlap_max_one() {
        // max_tokens=1, overlap_tokens=5: should produce one-word chunks without panicking.
        let text = "a b c";
        let chunks = tokenize_with_overlap(text, 1, 5);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "a");
        assert_eq!(chunks[1], "b");
        assert_eq!(chunks[2], "c");
    }

    // --- Indexed variant tests ---

    #[test]
    fn test_tokenize_with_overlap_indexed_basic() {
        let text = "a b c d e f g h i j";
        let chunks = tokenize_with_overlap_indexed(text, 4, 2);
        assert_eq!(chunks.len(), 4);
        // Chunk 0: words[0..4] = a b c d
        assert_eq!(chunks[0], ("a b c d".to_string(), 0, 4));
        // Chunk 1: words[2..6] = c d e f (starts at 4-2=2)
        assert_eq!(chunks[1], ("c d e f".to_string(), 2, 6));
        // Chunk 2: words[4..8] = e f g h (starts at 6-2=4)
        assert_eq!(chunks[2], ("e f g h".to_string(), 4, 8));
        // Chunk 3: words[6..10] = g h i j (starts at 8-2=6)
        assert_eq!(chunks[3], ("g h i j".to_string(), 6, 10));
    }

    #[test]
    fn test_tokenize_with_overlap_indexed_no_overlap() {
        let text = "a b c d e f";
        let chunks = tokenize_with_overlap_indexed(text, 3, 0);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], ("a b c".to_string(), 0, 3));
        assert_eq!(chunks[1], ("d e f".to_string(), 3, 6));
    }

    #[test]
    fn test_tokenize_with_overlap_indexed_remainder() {
        let text = "a b c d e f";
        let chunks = tokenize_with_overlap_indexed(text, 3, 1);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], ("a b c".to_string(), 0, 3));
        assert_eq!(chunks[1], ("c d e".to_string(), 2, 5));
        assert_eq!(chunks[2], ("e f".to_string(), 4, 6));
    }

    #[test]
    fn test_tokenize_with_overlap_indexed_empty() {
        let chunks = tokenize_with_overlap_indexed("", 4, 0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], (String::new(), 0, 0));
    }

    #[test]
    fn test_tokenize_with_overlap_indexed_zero_max() {
        let chunks = tokenize_with_overlap_indexed("hello", 0, 0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], (String::new(), 0, 0));
    }

    #[test]
    fn test_tokenize_with_overlap_indexed_consistency() {
        // The indexed and non-indexed variants must produce identical text output.
        let cases = [
            ("a b c d e f g h i j", 4, 2),
            ("a b c d e f", 3, 1),
            ("hello world", 10, 0),
            ("", 4, 0),
            ("single", 4, 0),
            ("a b c", 1, 5),
        ];
        for (text, max, overlap) in cases {
            let plain = tokenize_with_overlap(text, max, overlap);
            let indexed = tokenize_with_overlap_indexed(text, max, overlap);
            assert_eq!(
                plain.len(),
                indexed.len(),
                "Length mismatch for ({}, {}, {}): plain={}, indexed={}",
                text,
                max,
                overlap,
                plain.len(),
                indexed.len()
            );
            for (i, (p, (_, _, _))) in plain.iter().zip(indexed.iter()).enumerate() {
                assert_eq!(
                    p, &indexed[i].0,
                    "Text mismatch at index {} for ({}, {}, {})",
                    i, text, max, overlap
                );
            }
        }
    }
}
