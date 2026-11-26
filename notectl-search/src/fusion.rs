//! Dense retrieval and score combination via pure vector math.
//!
//! Two public functions:
//! 1. [`cosine_top_k`] — dot-product ranking on L2-normalized vectors.
//! 2. [`combine_scores`] — weighted mean of cosine similarity and scaled BM25.
//!
//! No model dependencies; fully unit-testable with synthetic vectors.

use std::collections::BTreeMap;

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

/// Combine each chunk's dense and sparse scores into one score in `0..=1`.
///
/// The result is the weighted mean `(w_dense * cosine + w_sparse * bm25) /
/// (w_dense + w_sparse)`. A chunk missing from one list counts as 0 there, and
/// negative similarities count as 0. Unlike a rank-based fusion, the value says
/// how well a chunk matches, so an unrelated query scores low.
///
/// # Arguments
/// * `dense` — `(chunk_index, cosine_similarity)` pairs, in any order
/// * `sparse` — `(chunk_index, bm25)` pairs scaled to `0..=1`, as returned by
///   [`crate::sparse::SparseIndexer::score_query`]
/// * `w_dense` — weight of the dense side; negative values count as 0
/// * `w_sparse` — weight of the sparse side; negative values count as 0
///
/// # Returns
/// `Vec<(usize, f64)>` — every chunk in either list, highest score first, ties
/// in chunk order.
///
/// # Example
/// ```
/// use notectl_search::fusion::combine_scores;
///
/// let dense = [(0, 0.8), (1, 0.6)];
/// let sparse = [(0, 0.4)];
///
/// let result = combine_scores(&dense, &sparse, 1.0, 1.0);
/// assert_eq!(result[0].0, 0);
/// assert!((result[0].1 - 0.6).abs() < 1e-6);
/// assert!((result[1].1 - 0.3).abs() < 1e-6);
/// ```
pub fn combine_scores(
    dense: &[(usize, f32)],
    sparse: &[(usize, f64)],
    w_dense: f64,
    w_sparse: f64,
) -> Vec<(usize, f64)> {
    let (w_dense, w_sparse) = (w_dense.max(0.0), w_sparse.max(0.0));
    let total = w_dense + w_sparse;
    let share = |weight: f64| if total > 0.0 { weight / total } else { 0.0 };

    let dense = dense
        .iter()
        .map(|&(idx, score)| (idx, share(w_dense) * f64::from(score).clamp(0.0, 1.0)));
    let sparse = sparse
        .iter()
        .map(|&(idx, score)| (idx, share(w_sparse) * score.clamp(0.0, 1.0)));

    let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
    for (idx, part) in dense.chain(sparse) {
        *acc.entry(idx).or_default() += part;
    }

    let mut results: Vec<(usize, f64)> = acc.into_iter().collect();
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
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
    // combine_scores tests
    // ---------------------------------------------------------------------------

    fn assert_scores(actual: &[(usize, f64)], expected: &[(usize, f64)]) {
        assert_eq!(actual.len(), expected.len(), "{actual:?}");
        for (a, e) in actual.iter().zip(expected) {
            assert_eq!(a.0, e.0, "{actual:?}");
            assert!((a.1 - e.1).abs() < 1e-6, "{actual:?}");
        }
    }

    #[test]
    fn combine_is_a_weighted_mean_with_missing_sides_as_zero() {
        let result = combine_scores(&[(0, 0.8), (1, 0.6)], &[(2, 1.0)], 3.0, 1.0);
        assert_scores(&result, &[(0, 0.6), (1, 0.45), (2, 0.25)]);
    }

    #[test]
    fn combine_with_one_side_weighted_keeps_that_score() {
        let result = combine_scores(&[(0, 0.7)], &[], 1.0, 0.0);
        assert_scores(&result, &[(0, 0.7)]);
    }

    #[test]
    fn combine_stays_within_zero_and_one() {
        let result = combine_scores(&[(0, 1.000_001), (1, -0.4)], &[], 1.0, -1.0);
        assert_scores(&result, &[(0, 1.0), (1, 0.0)]);
    }

    #[test]
    fn combine_with_no_weight_scores_zero() {
        let result = combine_scores(&[(0, 0.9)], &[(1, 0.9)], 0.0, 0.0);
        assert_scores(&result, &[(0, 0.0), (1, 0.0)]);
    }

    #[test]
    fn combine_breaks_ties_by_chunk_order() {
        let result = combine_scores(&[(2, 0.5), (0, 0.5), (1, 0.5)], &[], 1.0, 0.0);
        assert_scores(&result, &[(0, 0.5), (1, 0.5), (2, 0.5)]);
    }

    #[test]
    fn default_weights_are_equal() {
        let config = crate::SearchConfig::default();
        assert!((config.rrf_bm25_weight - config.rrf_cosine_weight).abs() < f64::EPSILON);
    }
}
