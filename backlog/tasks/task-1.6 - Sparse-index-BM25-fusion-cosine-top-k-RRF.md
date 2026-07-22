---
id: TASK-1.6
title: 'Sparse index (BM25) + fusion (cosine top-k, RRF)'
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-14 02:22'
updated_date: '2026-07-16 05:12'
labels:
  - planned
dependencies:
  - TASK-1.2
  - TASK-1.21
  - TASK-1.22
parent_task_id: TASK-1
priority: medium
type: task
ordinal: 7000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add notectl-search/src/sparse.rs (BM25 wrapper built from chunk texts, in-memory, rebuilt at load rather than persisted) and fusion.rs (cosine top-k over normalized vectors via dot product; reciprocal rank fusion combining dense + sparse rankings, weighted per SearchConfig). Unit-testable independently of the embedding model.

Delivered as two child tickets:
- **TASK-1.23**: `sparse.rs` — `SparseIndexer` wrapping `Bm25Indexer`
- **TASK-1.24**: `fusion.rs` — cosine top-k + weighted RRF
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [ ] #1 Add `sparse.rs` module with `SparseIndexer` struct wrapping `Bm25Indexer`
- [ ] #2 `index_chunks(chunks: &[Chunk]) -> SparseIndexer` builds BM25 index from chunk texts
- [ ] #3 `score_query(&self, query: &str) -> Vec<(usize, f64)>` returns ranked (chunk_index, score) pairs
- [ ] #4 Add `fusion.rs` module with pure vector math (no model deps)
- [ ] #5 `cosine_top_k(vectors, query, k)` returns top-k (chunk_index, similarity) via L2-normalized dot product
- [ ] #6 `rrf_fuse(dense, sparse, k, w_dense, w_sparse)` implements weighted RRF using config defaults
- [ ] #7 Both modules fully unit-testable without embedding model or external services
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->

## Definition of Done

<!-- SECTION:DEFINITION_OF_DONE:BEGIN -->
- [ ] Code compiles and passes `cargo test --workspace`
- [ ] All acceptance criteria met
- [ ] Child tickets TASK-1.23 and TASK-1.24 created and marked Dev Ready
<!-- SECTION:DEFINITION_OF_DONE:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
<!-- SECTION:IMPLEMENTATION_PLAN:BEGIN -->

### Overview

Two focused modules, each ~100-150 lines, no external dependencies beyond existing workspace crates.

**Module 1: `sparse.rs`** (TASK-1.23)
- Wraps the existing `Bm25Indexer` (from TASK-1.22, already has inverted index + running totals)
- Bridges `Chunk` text → BM25 doc indexing → ranked results
- In-memory only; rebuilt at search time from chunk list

**Module 2: `fusion.rs`** (TASK-1.24)
- Pure Rust vector math — no candle, no tokenizers, no model loading
- Cosine similarity via dot product on L2-normalized vectors (embeddings are already L2-normalized by `normalize_embedding`)
- Weighted Reciprocal Rank Fusion to merge dense + sparse rankings

### Module 1: `sparse.rs`

```rust
pub struct SparseIndexer {
    inner: Bm25Indexer,
}

impl SparseIndexer {
    /// Build an in-memory BM25 index from a slice of chunks.
    /// Each chunk's `.text` is added as a document.
    /// Calls finalize() internally so the index is ready for scoring.
    pub fn index_chunks(chunks: &[Chunk]) -> Self { ... }

    /// Score all chunks against a query string.
    /// Returns (chunk_index, bm25_score) sorted descending by score.
    /// Empty results if no chunks match any query term.
    pub fn score_query(&self, query: &str) -> Vec<(usize, f64)> { ... }
}
```

**Implementation details:**
- `index_chunks`: iterate chunks, tokenize each text via `Bm25Indexer::tokenize()`, call `add_document()`, then `finalize()`
- `score_query`: delegate to `Bm25Indexer::score_query()`, map result indices back to chunk positions
- No persistence — caller passes chunk slice each time

**Tests:**
- `test_index_and_score_basic`: 3 chunks, query matching one, verify ordering
- `test_empty_corpus`: zero chunks, query returns empty
- `test_multi_term_query`: query spanning multiple chunks, verify ranking

### Module 2: `fusion.rs`

```rust
/// Return top-k chunk indices ranked by cosine similarity to query.
/// Vectors must be L2-normalized (they come from normalize_embedding).
/// Uses dot product which equals cosine for unit vectors.
pub fn cosine_top_k(
    vectors: &[Vec<f32>],
    query: &[f32],
    k: usize,
) -> Vec<(usize, f32)> { ... }

/// Merge dense and sparse rankings via weighted Reciprocal Rank Fusion.
/// RRF score for doc d: w_dense / (k + dense_rank) + w_sparse / (k + sparse_rank)
/// Docs appearing in only one list get their contribution from that list alone.
/// Returns (chunk_index, fused_score) sorted descending.
pub fn rrf_fuse(
    dense: &[(usize, f32)],
    sparse: &[(usize, f64)],
    k: f64,
    w_dense: f64,
    w_sparse: f64,
) -> Vec<(usize, f64)> { ... }
```

**Implementation details:**

`cosine_top_k`:
- Brute-force dot product over all vectors (acceptable for typical vault sizes < 10K chunks)
- Collect scores, sort descending, truncate to k
- Returns `(chunk_index, similarity)` where similarity \in [-1, 1] (typically [0, 1] for normalized embeddings)

`rrf_fuse`:
- Build a `HashMap<usize, f64>` accumulator
- For each entry in dense list at rank r (1-indexed): `acc[idx] += w_dense / (k + r as f64)`
- For each entry in sparse list at rank r (1-indexed): `acc[idx] += w_sparse / (k + r as f64)`
- Convert to Vec, sort descending, return
- Default parameters from `SearchConfig`: `k = 60`, `w_dense = 1.0`, `w_sparse = 1.0`

**Tests:**
- `test_cosine_top_k_exact_match`: identical vectors return similarity 1.0
- `test_cosine_top_k_orthogonal`: orthogonal vectors return ~0.0
- `test_cosine_top_k_truncation`: k=2 on 5 vectors returns exactly 2 results
- `test_rrf_fuse_basic`: two overlapping lists, verify fused scores
- `test_rrf_fuse_non_overlapping`: disjoint lists, both docs appear
- `test_rrf_fuse_empty_inputs`: empty dense or sparse handled gracefully
- `test_rrf_defaults_match_config`: verify default k/w values match SearchConfig defaults

### Integration Points

Both modules will be consumed by TASK-1.8 (index pipeline) and TASK-1.9 (search pipeline):

- **TASK-1.8** (`index.rs`): Will call `SparseIndexer::index_chunks()` during indexing (though sparse index is in-memory, the indexed chunk metadata persists)
- **TASK-1.9** (`search.rs`): Will call both `cosine_top_k()` and `score_query()` then merge via `rrf_fuse()`

The modules are registered in `lib.rs` alongside existing modules:
```rust
pub mod sparse;
pub mod fusion;
```

No changes to `Cargo.toml` needed — no new dependencies.

### File Changes

| File | Action | Lines |
|------|--------|-------|
| `notectl-search/src/sparse.rs` | Create | ~120 |
| `notectl-search/src/fusion.rs` | Create | ~150 |
| `notectl-search/src/lib.rs` | Edit | +2 (mod declarations) |

<!-- SECTION:IMPLEMENTATION_PLAN:END -->
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both child tickets (TASK-1.23 sparse.rs, TASK-1.24 fusion.rs) were already completed. Verified: modules registered in lib.rs, all 67 notectl-search tests pass, clippy clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Parent task completed via child tickets TASK-1.23 (sparse.rs — SparseIndexer wrapping Bm25Indexer) and TASK-1.24 (fusion.rs — cosine_top_k + rrf_fuse). Both modules registered in lib.rs, 67 tests passing, clippy clean.
<!-- SECTION:FINAL_SUMMARY:END -->
