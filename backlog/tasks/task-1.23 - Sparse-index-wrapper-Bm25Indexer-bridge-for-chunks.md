---
id: TASK-1.23
title: 'Sparse index wrapper: Bm25Indexer bridge for chunks'
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-16 04:40'
updated_date: '2026-07-16 04:57'
labels:
  - planned
dependencies:
  - TASK-1.22
parent_task_id: TASK-1.6
priority: high
type: task
ordinal: 24000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add `notectl-search/src/sparse.rs` — a thin in-memory wrapper around the existing `Bm25Indexer` that indexes `Chunk` text and returns ranked `(chunk_index, score)` pairs for query strings. Built at search time from the chunk list; no persistence needed.

Leverages the inverted index improvements from TASK-1.22 (postings list, running total_tokens).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [ ] #1 Add `sparse.rs` module with `SparseIndexer` struct wrapping `Bm25Indexer`
- [ ] #2 `index_chunks(chunks: &[Chunk]) -> SparseIndexer` builds BM25 index from chunk texts, calls `finalize()` internally
- [ ] #3 `score_query(&self, query: &str) -> Vec<(usize, f64)>` returns (chunk_index, score) sorted descending by score
- [ ] #4 Empty corpus edge case handled (zero chunks → empty results)
- [ ] #5 Unit tests: multi-chunk indexing, scoring ordering, empty corpus
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
pub struct SparseIndexer {
    inner: Bm25Indexer,
}

impl SparseIndexer {
    pub fn index_chunks(chunks: &[Chunk]) -> Self { ... }
    pub fn score_query(&self, query: &str) -> Vec<(usize, f64)> { ... }
}
```

### `index_chunks`
1. Create `Bm25Indexer::new()`
2. For each chunk, call `inner.add_document(chunk.text.clone())`
3. Call `inner.finalize()` to compute IDF scores
4. Return wrapped `SparseIndexer`

### `score_query`
1. Delegate to `self.inner.score_query(query)`
2. Map result indices back to chunk positions
3. Sort descending (already done by Bm25Indexer)
4. Return `Vec<(usize, f64)>`

### Tests
- `test_index_and_score_basic`: 3 chunks, query matching one, verify top result
- `test_empty_corpus`: zero chunks, any query returns empty vec
- `test_multi_term_ranking`: multiple matching chunks, verify ordering by relevance

### Dependencies
- `crate::chunker::Chunk` (existing)
- `crate::bm25::Bm25Indexer` (existing, improved in TASK-1.22)

### No new Cargo.toml dependencies needed
<!-- SECTION:IMPLEMENTATION_PLAN:END -->
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created notectl-search/src/sparse.rs with SparseIndexer struct wrapping Bm25Indexer. Two public methods: index_chunks(chunks) builds the BM25 index from Chunk text, and score_query(query) returns ranked (chunk_index, score) pairs sorted descending. Module registered in lib.rs. Three tests cover basic scoring, empty corpus, and multi-term ranking. All 55 tests in notectl-search pass; clippy and rustfmt clean.
<!-- SECTION:FINAL_SUMMARY:END -->
