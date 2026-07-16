---
id: TASK-13
title: >-
  Fix: SearchResponse.mode_used always echoes requested mode, never reflects
  real auto-degradation
status: Done
assignee: []
created_date: '2026-07-16 14:38'
updated_date: '2026-07-16 16:16'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-1.10
priority: high
type: bug
ordinal: 110
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-1.10 (notectl-search/src/capability.rs:199-233, SearchCapability::do_search). SearchResponse.mode_used is documented as 'Effective search mode used (may differ from requested due to auto-degradation)' (capability.rs:114-115), but do_search() computes it purely from the caller-supplied options.mode, before calling crate::search::search(). The search() pipeline (notectl-search/src/search.rs:237-324) independently computes effective_mode and final_mode, auto-degrading Dense/Hybrid to Sparse when vectors are missing or query embedding fails at runtime — but search() only returns Vec<RankedChunk>, discarding that decision. So mode_used in the HTTP/MCP/CLI response always reports back exactly what the caller asked for, even when the search actually ran in a degraded mode. This is a Correctness-axis bug: the response field is misleading, and any client relying on mode_used to detect degradation (e.g. to warn a user that dense search silently fell back to keyword search) will never see it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 search::search() (or a new wrapper) returns the effective/final SearchMode alongside the ranked results, e.g. by changing its return type to a small struct/tuple (results, effective_mode) or an out-parameter
- [ ] #2 SearchCapability::do_search sets SearchResponse.mode_used from the actual effective mode returned by the search pipeline, not from the originally requested options.mode
- [ ] #3 A new test in notectl-search/src/search.rs or notectl-search/src/capability.rs builds an index with no vectors available, requests SearchMode::Hybrid or SearchMode::Dense, and asserts the returned effective mode is Sparse (not the originally requested mode)
- [ ] #4 nix develop -c cargo test -p notectl-search --all-features passes
- [ ] #5 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust+WebAssembly core (crates/gql-core) with a TypeScript/React web app (web/). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-search/src/search.rs. Find `pub async fn search(...) -> SearchResult<Vec<RankedChunk>>` (~line 143). Inside, `final_mode` (~line 314-324) already holds the true effective mode after all degradation decisions.
2. Change search()'s return type to expose final_mode alongside the results. Simplest approach: define a small struct near RankedChunk's usage, e.g. `pub struct SearchOutcome { pub results: Vec<RankedChunk>, pub mode_used: SearchMode }` and change search() to return `SearchResult<SearchOutcome>` instead of `SearchResult<Vec<RankedChunk>>`.
3. Update every existing caller of search::search() to use the new return shape: notectl-search/src/lib.rs SearchEngine::search() (or wherever it lives after TASK-1.10's dead-code cleanup, see the sibling review-followup ticket about removing SearchEngine — coordinate so this doesn't conflict), and notectl-search/src/capability.rs SearchCapability::do_search().
4. In SearchCapability::do_search (capability.rs ~line 199-233), set SearchResponse.mode_used from outcome.mode_used (converted to the existing lowercase string form), not from options.mode.
5. Update all existing tests in notectl-search/src/search.rs that currently do `let results = search(...).await.unwrap();` to destructure the new return type (e.g. `let outcome = search(...).await.unwrap(); let results = outcome.results;`).
6. Add a new test, e.g. test_search_mode_used_reflects_degradation: build an index without the embeddings feature enabled (or with vectors.bin absent), call search() with `SearchOptions { mode: SearchMode::Hybrid, .. }`, and assert the returned mode_used/effective mode is SearchMode::Sparse.
7. Run: nix develop -c cargo test -p notectl-search --all-features
8. Run: nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings
9. Run: nix develop -c cargo fmt -p notectl-search -- --check (fix formatting if needed).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation complete. Changed search() return type from SearchResult<Vec<RankedChunk>> to SearchResult<SearchOutcome> where SearchOutcome bundles results + mode_used (the actual effective SearchMode after all auto-degradation decisions). Updated SearchCapability::do_search, SearchEngine::search, and all 7 existing tests. Added new regression test test_search_mode_used_reflects_degradation that verifies Hybrid→Sparse and Dense→Sparse degradation is correctly reported. All 125 tests pass, clippy clean, fmt clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added SearchOutcome struct to search.rs that bundles Vec<RankedChunk> + effective SearchMode. Changed search() return type, updated all callers (SearchCapability, SearchEngine), rewrote 7 existing tests for new return shape, and added regression test. mode_used in SearchResponse now correctly reflects auto-degradation.
<!-- SECTION:FINAL_SUMMARY:END -->
