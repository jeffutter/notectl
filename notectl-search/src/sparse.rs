use crate::bm25::Bm25Indexer;
use crate::chunker::Chunk;

/// Thin in-memory wrapper around [`Bm25Indexer`] that indexes [`Chunk`] text and
/// returns ranked `(chunk_index, score)` pairs for query strings.
///
/// Built at search time from the chunk list; no persistence needed.
///
/// # Example
/// ```
/// use notectl_search::sparse::SparseIndexer;
/// use notectl_search::chunker::Chunk;
///
/// let chunks = vec![
///     Chunk {
///         id: "c1".into(),
///         source_file: "note.md".into(),
///         line_start: 0,
///         line_end: 10,
///         heading: None,
///         heading_path: Vec::new(),
///         text: "Rust is a systems programming language".to_string(),
///     },
/// ];
/// let indexer = SparseIndexer::index_chunks(&chunks);
/// let results = indexer.score_query("rust programming");
/// assert!(!results.is_empty());
/// ```
pub struct SparseIndexer {
    inner: Bm25Indexer,
}

impl SparseIndexer {
    /// Build a BM25 index from the given chunks.
    ///
    /// Each chunk's `text` field is tokenized and added as a document. The
    /// indexer is finalized (IDF computed) before returning so it is ready to
    /// score queries immediately.
    pub fn index_chunks(chunks: &[Chunk]) -> Self {
        let mut inner = Bm25Indexer::default_params();

        for chunk in chunks {
            let tokens = Bm25Indexer::tokenize(&chunk.text);
            inner.add_document(&tokens);
        }

        inner.finalize();
        Self { inner }
    }

    /// Score a free-text query against the indexed chunks.
    ///
    /// Returns a vector of `(chunk_index, score)` pairs sorted by descending
    /// score. Only chunks with a positive score are included.
    ///
    /// # Arguments
    /// * `query` - Free-text query string (will be tokenized internally)
    ///
    /// # Returns
    /// Ranked `(chunk_index, score)` — `chunk_index` corresponds to the index
    /// of the chunk in the original slice passed to [`SparseIndexer::index_chunks`].
    pub fn score_query(&self, query: &str) -> Vec<(usize, f64)> {
        let tokens = Bm25Indexer::tokenize(query);
        self.inner
            .score_query(&tokens)
            .into_iter()
            .map(|r| (r.doc_index, r.score))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a Chunk with minimal fields for testing.
    fn make_chunk(index: usize, text: &str) -> Chunk {
        Chunk {
            id: format!("test:{index}"),
            source_file: "test.md".into(),
            line_start: 0,
            line_end: 0,
            heading: None,
            heading_path: Vec::new(),
            text: text.to_string(),
        }
    }

    #[test]
    fn test_index_and_score_basic() {
        let chunks = vec![
            make_chunk(0, "Rust is a systems programming language"),
            make_chunk(1, "Python is great for data science"),
            make_chunk(2, "Rust has excellent memory safety guarantees"),
        ];

        let indexer = SparseIndexer::index_chunks(&chunks);
        let results = indexer.score_query("rust memory");

        // Should get non-empty results
        assert!(!results.is_empty());

        // Chunk 2 ("Rust has excellent memory safety") should rank highest
        // because it contains both "rust" and "memory"
        assert_eq!(results[0].0, 2);
        assert!(results[0].1 > 0.0);
    }

    #[test]
    fn test_empty_corpus() {
        let chunks: Vec<Chunk> = vec![];
        let indexer = SparseIndexer::index_chunks(&chunks);
        let results = indexer.score_query("anything");
        assert!(results.is_empty());
    }

    #[test]
    fn test_multi_term_ranking() {
        let chunks = vec![
            make_chunk(0, "GraphQL federation allows composing multiple subgraphs"),
            make_chunk(1, "Apollo Router runs federated supergraphs"),
            make_chunk(2, "Rust programming language is fast and safe"),
            make_chunk(3, "GraphQL schema design best practices for APIs"),
        ];

        let indexer = SparseIndexer::index_chunks(&chunks);
        let results = indexer.score_query("graphql federation schema");

        // Should return results for chunks containing graphql/federation/schema terms
        assert!(!results.is_empty());

        // Results should be sorted by descending score
        for window in results.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "Results must be sorted descending by score"
            );
        }

        // Chunk 0 (has "graphql" + "federation") and chunk 3 (has "graphql" + "schema")
        // should appear in top results
        let top_indices: Vec<usize> = results.iter().map(|&(idx, _)| idx).collect();
        assert!(
            top_indices.contains(&0),
            "Chunk 0 (graphql federation) should be in results"
        );
        assert!(
            top_indices.contains(&3),
            "Chunk 3 (graphql schema) should be in results"
        );
    }

    #[test]
    fn test_empty_query() {
        let chunks = vec![
            make_chunk(0, "Rust is great"),
            make_chunk(1, "Python is also great"),
        ];
        let indexer = SparseIndexer::index_chunks(&chunks);
        let results = indexer.score_query("");
        assert!(results.is_empty(), "Empty query should return no results");
    }

    #[test]
    fn test_single_chunk() {
        let chunks = vec![make_chunk(0, "hello world foo bar")];
        let indexer = SparseIndexer::index_chunks(&chunks);

        // Any matching term should return the single chunk.
        let results = indexer.score_query("hello");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 > 0.0);
    }
}
