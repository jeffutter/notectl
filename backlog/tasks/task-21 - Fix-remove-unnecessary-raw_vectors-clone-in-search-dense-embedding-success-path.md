---
id: TASK-21
title: >-
  Fix: remove unnecessary raw_vectors clone in search() dense-embedding success
  path
status: Needs Plan
assignee: []
created_date: '2026-07-16 17:20'
updated_date: '2026-07-18 14:20'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-8
priority: high
ordinal: 100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-8 (notectl-search/src/search.rs:293). In the Ok(qvec) arm of the embed_single match, dense_data is built with `Some((qvec, raw_vectors.clone()))`. raw_vectors has no use after this match (the Err arm drops it unused, and nothing later in search() reads it), so the .clone() is an unneeded full deep-copy of the entire dense vector matrix (Vec<Vec<f32>>, one f32 vector per chunk) on every successful Dense/Hybrid query. This is a Concise/efficiency-axis regression directly in the hot path TASK-8 was filed to make cheaper: TASK-8 eliminated a redundant disk read of vectors.bin, but left behind a redundant in-memory copy of the same data on every request. Verified locally that replacing the clone with a plain move (`raw_vectors` instead of `raw_vectors.clone()`) compiles cleanly with cargo check -p notectl-search --all-features, confirming the clone is not required by the borrow checker.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 notectl-search/src/search.rs::search() no longer calls .clone() on raw_vectors; the Ok(qvec) arm moves raw_vectors directly into the tuple instead
- [ ] #2 All existing tests in notectl-search/src/search.rs pass without modification to their assertions
- [ ] #3 nix develop -c cargo test -p notectl-search --all-features passes
- [ ] #4 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust+WebAssembly core (crates/gql-core) with a
TypeScript/React web app (web/). ALL commands must run inside the Nix dev
shell: either run 'direnv allow' once, or prefix every command with
'nix develop -c'. Work from the repository root unless told otherwise. Do not
change pinned dependency versions.

(This repo is notectl; the crate under test is notectl-search. The same Nix-shell rule applies.)

1. Open notectl-search/src/search.rs and find the dense_data construction inside search() (around line 276-308), specifically the match on the result of embedder.embed_single(...).await (around line 289-298).

2. Change:
   Ok(qvec) => Some((qvec, raw_vectors.clone())),
   to:
   Ok(qvec) => Some((qvec, raw_vectors)),

   This moves raw_vectors instead of cloning it. Confirm raw_vectors is not referenced anywhere later in the function (it is not — its only other use is the has_vectors computation earlier at line 248, which borrows it and completes before this match runs).

3. Run: nix develop -c cargo check -p notectl-search --all-features
   Confirm it compiles with no borrow-checker errors (the Err arm of the same match does not use raw_vectors, so Rust allows the move in the Ok arm alone).

4. Run: nix develop -c cargo test -p notectl-search --all-features
   Confirm all existing tests pass unmodified, in particular test_dense_mode_degrades_to_sparse_when_embedding_unavailable and test_search_mode_used_reflects_degradation.

5. Run: nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings
   Fix any warnings (there should be none expected from this change).
<!-- SECTION:PLAN:END -->
