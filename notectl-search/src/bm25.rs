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
/// This is a lightweight implementation that stores term frequencies
/// and document lengths for scoring. It does not require external crates.
pub struct Bm25Indexer {
    params: Bm25Params,
    /// Term frequency per document: doc_index -> term -> count
    tf: HashMap<usize, HashMap<String, u32>>,
    /// Document lengths (number of tokens)
    doc_lengths: Vec<usize>,
    /// Average document length
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
            tf: HashMap::new(),
            doc_lengths: Vec::new(),
            avg_doc_length: 0.0,
            doc_count: 0,
            idf: HashMap::new(),
        }
    }

    pub fn default_params() -> Self {
        Self::new(Bm25Params::default())
    }

    /// Add a document to the index
    pub fn add_document(&mut self, tokens: &[String]) {
        let doc_index = self.doc_count;
        let length = tokens.len();

        // Count term frequencies
        let mut term_counts: HashMap<String, u32> = HashMap::new();
        for token in tokens {
            *term_counts.entry(token.clone()).or_insert(0) += 1;
        }

        self.tf.insert(doc_index, term_counts);
        self.doc_lengths.push(length);
        self.doc_count += 1;

        // Update average document length
        let total_length: usize = self.doc_lengths.iter().sum();
        self.avg_doc_length = total_length as f64 / self.doc_count as f64;
    }

    /// Compute IDF for all terms after all documents have been added
    pub fn finalize(&mut self) {
        // Count document frequency for each term
        let mut df: HashMap<String, usize> = HashMap::new();
        for term_counts in self.tf.values() {
            for term in term_counts.keys() {
                *df.entry(term.clone()).or_insert(0) += 1;
            }
        }

        // Compute IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        for (term, count) in &df {
            let idf =
                ((self.doc_count as f64 - *count as f64 + 0.5) / (*count as f64 + 0.5) + 1.0).ln();
            self.idf.insert(term.clone(), idf);
        }
    }

    /// Score a query against all indexed documents
    pub fn score_query(&self, query_tokens: &[String]) -> Vec<Bm25Result> {
        let mut scores: HashMap<usize, f64> = HashMap::new();

        for token in query_tokens {
            let idf = match self.idf.get(token) {
                Some(idf) => *idf,
                None => continue, // Term not in index, skip
            };

            for (&doc_index, term_counts) in &self.tf {
                let tf = match term_counts.get(token) {
                    Some(&count) => count as f64,
                    None => 0.0,
                };

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
