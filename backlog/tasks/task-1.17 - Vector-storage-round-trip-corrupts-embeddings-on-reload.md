---
id: TASK-1.17
title: Vector storage round-trip corrupts embeddings on reload
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-14 11:11'
updated_date: '2026-07-15 11:44'
labels:
  - planned
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Implementation Plan

### Root Cause
 serializes  with no dimension metadata.  reads count, then consumes the next 4 bytes as a u32 dimension guess — but those are the leading float of , not a stored dim field. This corrupts every read.

### Fix: Store dimension in header (storage.rs lines ~112-156)

**write_vectors changes:**
After writing count, write  as u32 LE from the first non-empty vector (or 0 if empty). Format becomes: .

**read_vectors changes:**
After reading count, read 4 more bytes as . Handle the count==0 / empty-file case gracefully (return early if count is 0, or dim file may be 0 bytes after count). Then loop reading  floats per vector.

### Tests (in storage.rs #[cfg(test)] module)

1. **** — Write N=5 vectors of dim=768 with known f32 values, read back, assert element-wise equality. Use a deterministic pattern like  so any corruption is detectable.

2. **** — Write empty slice, read back, assert .

3. **** — Edge case: 1 vector to confirm no off-by-one.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed wire format incompatibility between write_vectors and read_vectors in notectl-search/src/storage.rs. Both functions now use a consistent format: [count: u64 LE][dim: u32 LE][vec data]. write_vectors writes the dimension after count (0 if no vectors/empty). read_vectors reads the explicit dim field instead of misinterpreting the first float bytes as dimension. Added 3 unit tests: round-trip with N=5 vectors dim=768, empty index case, and single-vector edge case.
<!-- SECTION:NOTES:END -->
