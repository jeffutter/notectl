use std::collections::HashMap;

/// BM25 scoring parameters
#[derive(Debug, Clone)]
pub struct Bm25Params {
    /// Term frequency saturation parameter (default: 1.2)
    pub k1: f64,
    /// Document length normalization parameter (default: 0.75)
    pub b: f64,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// A single BM25 result with score and document index
#[derive(Debug, Clone)]
pub struct Bm25Result {
    pub doc_index: usize,
    pub score: f64,
}

/// In-memory BM25 indexer and scorer.
///
/// This is a lightweight implementation that stores an inverted index
/// (term → postings) for efficient query-time scoring. It does not require external crates.
pub struct Bm25Indexer {
    params: Bm25Params,
    /// Inverted index: term → [(doc_index, term_frequency)]
    postings: HashMap<String, Vec<(usize, u32)>>,
    /// Document lengths (number of tokens)
    doc_lengths: Vec<usize>,
    /// Running total of all token counts across documents
    total_tokens: usize,
    /// Average document length (recomputed after each add_document)
    avg_doc_length: f64,
    /// Number of documents
    doc_count: usize,
    /// Inverse document frequency per term: term -> idf
    idf: HashMap<String, f64>,
}

impl Bm25Indexer {
    pub fn new(params: Bm25Params) -> Self {
        Self {
            params,
            postings: HashMap::new(),
            doc_lengths: Vec::new(),
            total_tokens: 0,
            avg_doc_length: 0.0,
            doc_count: 0,
            idf: HashMap::new(),
        }
    }

    pub fn default_params() -> Self {
        Self::new(Bm25Params::default())
    }

    /// Add a document to the index.
    ///
    /// Builds the inverted index in-place: each unique term in the document
    /// is pushed onto its postings list with (doc_index, term_frequency).
    pub fn add_document(&mut self, tokens: &[String]) {
        let doc_index = self.doc_count;
        let length = tokens.len();

        // Count term frequencies for this document
        let mut term_counts: HashMap<String, u32> = HashMap::new();
        for token in tokens {
            *term_counts.entry(token.clone()).or_insert(0) += 1;
        }

        // Insert into inverted index (term → postings)
        for (term, count) in &term_counts {
            self.postings
                .entry(term.clone())
                .or_default()
                .push((doc_index, *count));
        }

        self.doc_lengths.push(length);
        self.total_tokens += length;
        self.doc_count += 1;
        self.avg_doc_length = self.total_tokens as f64 / self.doc_count as f64;
    }

    /// Compute IDF for all terms after all documents have been added.
    ///
    /// Document frequency (DF) is derived from the inverted index: each term's
    /// postings list length equals the number of distinct documents containing it.
    pub fn finalize(&mut self) {
        // Compute IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        for (term, postings) in &self.postings {
            let df = postings.len();
            let idf = ((self.doc_count as f64 - df as f64 + 0.5) / (df as f64 + 0.5) + 1.0).ln();
            self.idf.insert(term.clone(), idf);
        }
    }

    /// Score a query against all indexed documents.
    ///
    /// Uses the inverted index: for each query token, only the documents in its
    /// postings list are scored — no full corpus scan.
    pub fn score_query(&self, query_tokens: &[String]) -> Vec<Bm25Result> {
        let mut scores: HashMap<usize, f64> = HashMap::new();

        for token in query_tokens {
            let idf = match self.idf.get(token) {
                Some(idf) => *idf,
                None => continue, // Term not in index, skip
            };

            // Iterate only over documents that contain this term
            if let Some(postings) = self.postings.get(token.as_str()) {
                for &(doc_index, tf_raw) in postings {
                    let tf = tf_raw as f64;
                    let doc_len = self.doc_lengths[doc_index] as f64;
                    let k1 = self.params.k1;
                    let b = self.params.b;

                    // BM25 formula: IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * doc_len / avg_dl))
                    let numerator = tf * (k1 + 1.0);
                    let denominator = tf + k1 * (1.0 - b + b * doc_len / self.avg_doc_length);
                    let score = idf * numerator / denominator;

                    *scores.entry(doc_index).or_insert(0.0) += score;
                }
            }
        }

        // Convert to sorted results
        let mut results: Vec<Bm25Result> = scores
            .into_iter()
            .map(|(doc_index, score)| Bm25Result { doc_index, score })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Tokenize text into lowercase tokens (simple whitespace + punctuation split)
    pub fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokens = Bm25Indexer::tokenize("Hello, World! Hello.");
        assert_eq!(tokens, vec!["hello", "world", "hello"]);
    }

    #[test]
    fn test_basic_scoring() {
        let mut indexer = Bm25Indexer::default_params();

        // Add two documents of different lengths
        indexer.add_document(&["rust".into(), "is".into(), "great".into()]);
        indexer.add_document(&[
            "rust".into(),
            "programming".into(),
            "language".into(),
            "is".into(),
            "also".into(),
            "amazing".into(),
        ]);
        indexer.finalize();

        // Query for "rust" - both docs should match
        let results = indexer.score_query(&["rust".into()]);
        assert_eq!(results.len(), 2);
        // Shorter doc (doc 0) should rank higher due to length normalization
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_unrelated_query_returns_empty() {
        let mut indexer = Bm25Indexer::default_params();
        indexer.add_document(&["rust".into(), "is".into(), "great".into()]);
        indexer.finalize();

        let results = indexer.score_query(&["nonexistent_term_xyz".into()]);
        assert!(results.is_empty());
    }
}
