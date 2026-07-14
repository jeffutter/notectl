---
id: TASK-1.22
title: Resolve BM25 crate-vs-hand-rolled duplication and add an inverted index
status: To Do
assignee: []
created_date: '2026-07-14 11:12'
labels: []
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
