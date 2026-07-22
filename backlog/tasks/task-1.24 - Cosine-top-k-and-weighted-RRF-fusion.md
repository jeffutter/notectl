---
id: TASK-1.24
title: Cosine top-k and weighted RRF fusion
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-16 04:40'
updated_date: '2026-07-16 05:09'
labels:
  - planned
dependencies:
  - TASK-1.7
parent_task_id: TASK-1.6
priority: high
type: task
ordinal: 25000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add `notectl-search/src/fusion.rs` — pure vector math for dense retrieval and hybrid ranking. Two public functions: (1) cosine top-k via dot product on L2-normalized vectors, (2) weighted Reciprocal Rank Fusion to merge dense + sparse rankings. No model dependencies; fully unit-testable with synthetic vectors.

Consumes L2-normalized embeddings from TASK-1.7's `normalize_embedding()` output and BM25 scores from TASK-1.23's `SparseIndexer`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [ ] #1 `cosine_top_k(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(usize, f32)>` returns top-k (chunk_index, similarity) sorted descending
- [ ] #2 Uses dot product of L2-normalized vectors (equivalent to cosine for unit vectors)
- [ ] #3 `rrf_fuse(dense: &[(usize, f32)], sparse: &[(usize, f64)], k: f64, w_dense: f64, w_sparse: f64) -> Vec<(usize, f64)>` implements weighted RRF
- [ ] #4 RRF formula: `score = w_dense / (k + rank) + w_sparse / (k + rank)` where rank is 1-indexed position in input list
- [ ] #5 Handles edge cases: empty inputs, single result, overlapping/non-overlapping doc sets
- [ ] #6 Unit tests cover cosine correctness, RRF ranking order, config defaults matching SearchConfig
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->

## Definition of Done

<!-- SECTION:DEFINITION_OF_DONE:BEGIN -->
- [ ] Code compiles and passes `cargo test --workspace`
- [ ] All acceptance criteria met
<!-- SECTION:DEFINITION_OF_DONE:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
<!-- SECTION:IMPLEMENTATION_PLAN:BEGIN -->

### Structure

```rust
/// Return top-k chunk indices ranked by cosine similarity to query.
/// Vectors must be L2-normalized (they come from normalize_embedding).
pub fn cosine_top_k(
    vectors: &[Vec<f32>],
    query: &[f32],
    k: usize,
) -> Vec<(usize, f32)> { ... }

/// Merge dense and sparse rankings via weighted Reciprocal Rank Fusion.
pub fn rrf_fuse(
    dense: &[(usize, f32)],
    sparse: &[(usize, f64)],
    k: f64,
    w_dense: f64,
    w_sparse: f64,
) -> Vec<(usize, f64)> { ... }
```

### `cosine_top_k`
1. For each vector at index i, compute dot product with query: `v.iter().zip(query.iter()).map(|(a,b)| a*b).sum()`
2. Collect `(i, score)` pairs
3. Sort descending by score
4. Truncate to k results
5. Return `Vec<(usize, f32)>`

Note: Brute-force O(n*d) is acceptable for typical vault sizes (< 10K chunks, dim ~768). No ANN approximation needed.

### `rrf_fuse`
1. Create `HashMap<usize, f64>` accumulator
2. For each entry in dense list at 1-indexed rank r: `acc[idx] += w_dense / (k + r as f64)`
3. For each entry in sparse list at 1-indexed rank r: `acc[idx] += w_sparse / (k + r as f64)`
4. Docs appearing in only one list get contribution from that list alone
5. Convert HashMap to Vec, sort descending by fused score
6. Return `Vec<(usize, f64)>`

Default parameters from `SearchConfig`:
- `k = 60` (rrf_k)
- `w_dense = 1.0` (rrf_cosine_weight)
- `w_sparse = 1.0` (rrf_bm25_weight)

### Tests
- `test_cosine_top_k_exact_match`: identical normalized vectors → similarity 1.0
- `test_cosine_top_k_orthogonal`: orthogonal vectors → similarity ~0.0
- `test_cosine_top_k_truncation`: k=2 on 5 vectors → exactly 2 results
- `test_rrf_fuse_overlapping`: two lists sharing docs → correct fused scores
- `test_rrf_fuse_non_overlapping`: disjoint lists → both docs appear
- `test_rrf_fuse_empty_dense`: only sparse results → sparse-only ranking
- `test_rrf_fuse_empty_both`: both empty → empty result
- `test_rrf_defaults_match_config`: verify default k/w values match SearchConfig

### Dependencies
None — pure Rust math, no external crates beyond std

### No new Cargo.toml dependencies needed
<!-- SECTION:IMPLEMENTATION_PLAN:END -->
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented fusion.rs with:
- cosine_top_k(vectors, query, k) → Vec<(usize, f32)> sorted descending
- rrf_fuse(dense, sparse, k, w_dense, w_sparse) → Vec<(usize, f64)> sorted descending
- 12 unit tests covering exact match, orthogonal, truncation, empty inputs, overlapping/non-overlapping RRF, weighted fusion, config default alignment, and tie-breaking
- No external crate deps beyond std (HashMap only)
- Module registered in lib.rs
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created notectl-search/src/fusion.rs with cosine_top_k and rrf_fuse functions. Pure Rust vector math, no external deps, 12 passing tests, clippy clean.
<!-- SECTION:FINAL_SUMMARY:END -->
