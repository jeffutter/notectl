---
id: TASK-1.22
title: Resolve BM25 crate-vs-hand-rolled duplication and add an inverted index
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-14 11:12'
updated_date: '2026-07-15 22:55'
labels:
  - planned
dependencies:
  - TASK-1.2
parent_task_id: TASK-1
priority: medium
type: task
ordinal: 23000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
notectl-search/Cargo.toml declares a dependency on the `bm25` crate (gated behind the `embeddings` feature), but notectl-search/src/bm25.rs never uses it (`grep -rn "bm25::" notectl-search/src` returns nothing) — it hand-rolls its own BM25 indexer/scorer/tokenizer instead, unconditionally available regardless of feature flags. TASK-1.1's research spike evaluated the `bm25` crate as a candidate; the current code doesn't use what was researched, so this dependency and that decision should be reconciled (drop the unused dependency, or actually use the crate and drop the hand-rolled version).

Separately, `Bm25Index::score_query` (bm25.rs, ~line 107) has no term→postings (inverted) index: every query scans every document in the corpus regardless of how few documents contain the query terms, and `add_document` (~line 74) re-sums every document's length on each insert, making index construction O(n²) in corpus size. For a feature named "sparse index," a real index structure (term → list of documents) is closer to the intended design than a brute-force scan, and is TASK-1.6's stated deliverable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The unused bm25 crate dependency is either removed from Cargo.toml, or actually adopted in place of the hand-rolled implementation — pick one and remove the other
- [ ] #2 Bm25Index builds and queries via a term-to-postings structure rather than a full corpus scan per query term
- [ ] #3 add_document does not re-sum all document lengths on every insert (track a running total instead)
- [ ] #4 Existing BM25 scoring unit tests still pass with equivalent (or documented, intentionally different) rankings
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Approach

Keep the hand-rolled BM25 implementation (~130 lines, simple, well-tested). Remove the unused bm25 crate dependency. Fix the three performance/design issues in a single pass.

## Changes (all in notectl-search/src/bm25.rs + Cargo.toml)

### 1. Remove unused bm25 crate dependency
- Remove `bm25 = { workspace = true, optional = true }` from notectl-search/Cargo.toml
- Remove `"bm25"` from the `embeddings` feature list

### 2. Add inverted index (term → postings) — Criterion #2
Replace:
- `tf: HashMap<usize, HashMap<String, u32>>` (doc_index → term → count)

With:
- `postings: HashMap<String, Vec<(usize, u32)>>` (term → [(doc_index, tf)])

In add_document(): after counting term frequencies for the new document, insert each term into postings by pushing (doc_index, count) into the postings list.

In score_query(): for each query token, look up postings[token] and iterate only that postings list instead of scanning all documents. O(|matching docs| × |terms|) instead of O(|all docs| × |terms|).

### 3. Track running total for avg doc length — Criterion #3
Add a `total_tokens: usize` field to Bm25Indexer. In add_document(), update it by `+= tokens.len()` instead of calling `self.doc_lengths.iter().sum()`. Compute avg_doc_length after the increment. Eliminates O(n²) index construction.

### 4. Preserve IDF computation in finalize()
With the inverted index, derive DF from postings list length per term (length = number of docs containing that term). Update finalize() to iterate postings instead of tf. The IDF formula stays identical.

### 5. Preserve score_query() semantics
The BM25 scoring formula is already canonical Okapi BM25. The per-term scoring loop changes from "iterate all docs" to "iterate postings list for this term." Numerator/denominator math is identical. Accumulate into HashMap<usize, f64> as before, sort descending.

### 6. Verify tests pass
All three existing unit tests (test_tokenize, test_basic_scoring, test_unrelated_query_returns_empty) must still pass with equivalent rankings. The scoring formula does not change — only the data structure backing it.

## Files touched
- notectl-search/src/bm25.rs — inverted index refactor, running total
- notectl-search/Cargo.toml — remove bm25 dep
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
## Implementation Notes

### Data structure changes in :

1. **Inverted index**: Replaced  (doc_index → term → count) with  (term → [(doc_index, tf)]). This is the core change — queries now iterate only over documents that actually contain each query term.

2. **Running total**: Added  field.  increments it by  instead of summing  on every insert.  is computed from the running total, making index construction O(n) instead of O(n²).

3. **IDF derivation**:  now iterates  and derives DF from each postings list's length (number of distinct documents containing that term). IDF formula unchanged.

4. **Query scoring**:  looks up  for each query term and iterates only the matching postings list instead of scanning all documents. Accumulates into  as before, sorts descending.

### Removed files/dependencies:
-  from notectl-search/Cargo.toml
-  from the  feature list in notectl-search/Cargo.toml
-  from workspace root Cargo.toml

### Test results:
All 3 existing BM25 unit tests pass unchanged (test_tokenize, test_basic_scoring, test_unrelated_query_returns_empty). The scoring formula is identical — only the backing data structure changed.

Implementation: Replaced tf HashMap with term->postings inverted index (HashMap<String, Vec<(usize,u32)>>). Added running total_tokens field for O(1) avg_doc_length updates. finalize() derives DF from postings list lengths. score_query() iterates only matching docs per query term. Removed bm25 crate from notectl-search/Cargo.toml and workspace root Cargo.toml. All 3 existing BM25 tests pass unchanged.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed all 4 acceptance criteria: (1) Removed unused bm25 crate dependency from both notectl-search/Cargo.toml and workspace root Cargo.toml, keeping the hand-rolled implementation. (2) Replaced full corpus scan with term→postings inverted index — score_query() now iterates only matching documents per query term. (3) Added running total_tokens field so add_document() updates avg_doc_length in O(1) instead of O(n), eliminating O(n²) index construction. (4) All 3 existing BM25 unit tests pass with identical scoring semantics — the Okapi BM25 formula is unchanged, only the backing data structure improved.
<!-- SECTION:FINAL_SUMMARY:END -->
