---
id: TASK-1.17
title: Vector storage round-trip corrupts embeddings on reload
status: To Do
assignee: []
created_date: '2026-07-14 11:11'
labels: []
dependencies:
  - TASK-1.2
parent_task_id: TASK-1
priority: high
type: bug
ordinal: 18000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`SearchIndex::write_vectors` and `SearchIndex::read_vectors` in notectl-search/src/storage.rs (lines ~112-171) use incompatible wire formats. `write_vectors` writes `[count: u64][vec0 bytes][vec1 bytes]...` with no per-file dimension field. `read_vectors` reads the 8-byte count, then reads the *next* 4 bytes and interprets them as a `u32` embedding dimension — but those 4 bytes are actually the leading bytes of `vectors[0][0]` (the first float of the first vector) reinterpreted as an integer, and they are never restored into the first vector's data. Depending on the bit pattern of that first float, `dim` comes out as garbage: either an enormous number (causing a huge/failing allocation or a panic) or a plausible-looking wrong dimension that misaligns every subsequent read, silently corrupting every embedding retrieved.

No existing test exercises this round trip — the current storage.rs tests only cover the manifest and chunk text, not `write_vectors`/`read_vectors` together. TASK-1.5 (storage: manifest + vectors + staleness diff) owns this file going forward and should not build the staleness-diff logic on top of a broken vector format.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 write_vectors/read_vectors use a self-consistent on-disk format such that writing N vectors and reading them back returns the original N vectors unchanged
- [ ] #2 Add a unit test that writes a set of vectors (multiple vectors, non-trivial dimension) via write_vectors and asserts read_vectors returns equal vectors
- [ ] #3 Add a test for the empty-index case (0 vectors)
<!-- AC:END -->
