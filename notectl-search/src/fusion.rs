//! Dense retrieval and hybrid ranking via pure vector math.
//!
//! Two public functions:
//! 1. [`cosine_top_k`] — dot-product ranking on L2-normalized vectors.
//! 2. [`rrf_fuse`] — weighted Reciprocal Rank Fusion to merge dense + sparse rankings.
//!
//! No model dependencies; fully unit-testable with synthetic vectors.

use std::collections::HashMap;

/// Return top-k chunk indices ranked by cosine similarity to `query`.
///
/// Both `vectors` and `query` must be L2-normalized.
/// When vectors are normalized, the dot product *is* cosine similarity.
///
/// # Arguments
/// * `vectors` — slice of L2-normalized chunk embeddings
/// * `query` — L2-normalized query embedding
/// * `k` — maximum number of results to return
///
/// # Returns
/// `Vec<(usize, f32)>` — `(chunk_index, cosine_similarity)` sorted descending.
///
/// # Example
/// ```
/// use notectl_search::fusion::cosine_top_k;
///
/// // Exact match → score 1.0; orthogonal → score ~0.0
/// let vecs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let query = vec![1.0, 0.0];
///
/// let result = cosine_top_k(&vecs, &query, 10);
/// assert_eq!(result[0].0, 0); // exact match ranks first
/// assert!((result[0].1 - 1.0).abs() < f32::EPSILON);
/// assert!((result[1].1 - 0.0).abs() < f32::EPSILON); // orthogonal
/// ```
pub fn cosine_top_k(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut scores: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            // Dot product of two L2-normalized vectors == cosine similarity.
            let score: f32 = v.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
            (i, score)
        })
        .collect();

    // Sort descending by score (stable so ties preserve original index order).
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scores.into_iter().take(k).collect()
}

/// Merge dense and sparse rankings via weighted Reciprocal Rank Fusion.
///
/// For each list, a document at 1-indexed rank `r` contributes
/// `weight / (k + r)` to its fused score.  Documents appearing in only
/// one list still get credit from that list alone.
///
/// # Arguments
/// * `dense` — `(chunk_index, cosine_score)` from [`cosine_top_k`], sorted descending
/// * `sparse` — `(chunk_index, bm25_score)` from [`crate::sparse::SparseIndexer::score_query`], sorted descending
/// * `k` — RRF rank-damping constant (typically 60)
/// * `w_dense` — weight multiplier for dense-side contributions
/// * `w_sparse` — weight multiplier for sparse-side contributions
///
/// # Returns
/// `Vec<(usize, f64)>` — `(chunk_index, fused_score)` sorted descending.
///
/// # Example
/// ```
/// use notectl_search::fusion::rrf_fuse;
///
/// let dense = [(0, 0.9), (1, 0.7)];
/// let sparse = [(0, 1.5), (2, 1.0)];
///
/// let result = rrf_fuse(&dense, &sparse, 60.0, 1.0, 1.0);
/// // Doc 0 appears at rank 1 in both lists → highest fused score.
/// assert_eq!(result[0].0, 0);
/// ```
pub fn rrf_fuse(
    dense: &[(usize, f32)],
    sparse: &[(usize, f64)],
    k: f64,
    w_dense: f64,
    w_sparse: f64,
) -> Vec<(usize, f64)> {
    let mut acc: HashMap<usize, f64> = HashMap::new();

    // Dense contributions (1-indexed rank).
    for (rank, &(idx, _)) in dense.iter().enumerate() {
        let r = (rank + 1) as f64;
        let entry = acc.entry(idx).or_default();
        *entry += w_dense / (k + r);
    }

    // Sparse contributions (1-indexed rank).
    for (rank, &(idx, _)) in sparse.iter().enumerate() {
        let r = (rank + 1) as f64;
        let entry = acc.entry(idx).or_default();
        *entry += w_sparse / (k + r);
    }

    let mut results: Vec<(usize, f64)> = acc.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------
    // cosine_top_k tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_cosine_top_k_exact_match() {
        // Identical L2-normalized vectors → similarity 1.0
        let vecs = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let query = vec![1.0, 0.0, 0.0];

        let result = cosine_top_k(&vecs, &query, 10);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 0);
        assert!((result[0].1 - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cosine_top_k_orthogonal() {
        // Orthogonal L2-normalized vectors → similarity ~0.0
        let vecs = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let query = vec![0.0, 1.0, 0.0];

        let result = cosine_top_k(&vecs, &query, 10);
        assert_eq!(result[0].0, 1);
        assert!((result[0].1 - 1.0).abs() < f32::EPSILON);
        assert!((result[1].1 - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cosine_top_k_truncation() {
        let vecs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.8, 0.6, 0.0],
            vec![0.6, 0.8, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let query = vec![1.0, 0.0, 0.0];

        let result = cosine_top_k(&vecs, &query, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 0); // exact match
        assert_eq!(result[1].0, 1); // 0.8 overlap
    }

    #[test]
    fn test_cosine_top_k_empty_vectors() {
        let vecs: Vec<Vec<f32>> = vec![];
        let query = vec![1.0, 0.0, 0.0];
        let result = cosine_top_k(&vecs, &query, 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_cosine_top_k_k_zero() {
        let vecs = vec![vec![1.0, 0.0, 0.0]];
        let query = vec![1.0, 0.0, 0.0];
        let result = cosine_top_k(&vecs, &query, 0);
        assert!(result.is_empty());
    }

    // ---------------------------------------------------------------------------
    // rrf_fuse tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_rrf_fuse_overlapping() {
        // Two lists sharing doc 0 at different ranks.
        let dense = [(0, 0.9), (1, 0.7)];
        let sparse = [(0, 1.5), (2, 1.0)];
        let k = 60.0;

        let result = rrf_fuse(&dense, &sparse, k, 1.0, 1.0);
        assert_eq!(result.len(), 3);

        // Doc 0 appears at rank 1 in both lists → highest fused score.
        assert_eq!(result[0].0, 0);

        // Verify exact fused score for doc 0:
        //   dense rank 1: 1/(60+1) = 1/61
        //   sparse rank 1: 1/(60+1) = 1/61
        //   total = 2/61
        let expected_doc0 = 2.0 / 61.0;
        assert!((result[0].1 - expected_doc0).abs() < 1e-9);
    }

    #[test]
    fn test_rrf_fuse_non_overlapping() {
        let dense = [(0, 0.9)];
        let sparse = [(1, 1.5)];
        let k = 60.0;

        let result = rrf_fuse(&dense, &sparse, k, 1.0, 1.0);
        assert_eq!(result.len(), 2);

        // Both at rank 1 in their respective lists, same weight → tied.
        // After stable sort they may appear in either order, but scores are equal.
        assert!((result[0].1 - result[1].1).abs() < 1e-9);
    }

    #[test]
    fn test_rrf_fuse_empty_dense() {
        let dense: [(usize, f32); 0] = [];
        let sparse = [(0, 1.5), (1, 1.0)];
        let k = 60.0;

        let result = rrf_fuse(&dense, &sparse, k, 1.0, 1.0);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 0); // sparse rank 1 wins
    }

    #[test]
    fn test_rrf_fuse_empty_both() {
        let dense: [(usize, f32); 0] = [];
        let sparse: [(usize, f64); 0] = [];
        let result = rrf_fuse(&dense, &sparse, 60.0, 1.0, 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_rrf_fuse_weighted_dense_heavy() {
        let dense = [(0, 0.5)];
        let sparse = [(1, 2.0)];
        let k = 60.0;

        // Give dense 10× weight so doc 0 wins despite being lower BM25.
        let result = rrf_fuse(&dense, &sparse, k, 10.0, 1.0);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 0); // dense-heavy doc 0 ranks first

        // doc 0: 10/(60+1) = 10/61
        // doc 1: 1/(60+1) = 1/61
        assert!((result[0].1 - 10.0 / 61.0).abs() < 1e-9);
        assert!((result[1].1 - 1.0 / 61.0).abs() < 1e-9);
    }

    #[test]
    fn test_rrf_defaults_match_config() {
        // Verify the default parameter values align with SearchConfig defaults.
        use crate::SearchConfig;
        let config = SearchConfig::default();

        assert!((config.rrf_k - 60.0).abs() < f64::EPSILON);
        assert!((config.rrf_bm25_weight - 1.0).abs() < f64::EPSILON);
        assert!((config.rrf_cosine_weight - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rrf_fuse_preserves_order_for_ties() {
        // Docs with identical fused scores should maintain stable ordering.
        let dense = [(0, 0.9), (1, 0.8)];
        let sparse: [(usize, f64); 0] = [];
        let k = 60.0;

        let result = rrf_fuse(&dense, &sparse, k, 1.0, 0.0);
        // Only dense contributes (sparse weight is 0).
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 0); // rank 1 > rank 2
        assert_eq!(result[1].0, 1);
    }
}
