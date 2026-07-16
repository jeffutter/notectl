---
id: TASK-16
title: >-
  Remove dead SearchEngine struct that duplicates SearchCapability's
  index/search orchestration
status: Dev Ready
assignee: []
created_date: '2026-07-16 14:40'
updated_date: '2026-07-16 14:40'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-1.10
priority: high
type: chore
ordinal: 140
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-1.10 (notectl-search/src/lib.rs:99-148). TASK-1.10 renamed the pre-existing SearchCapability struct to SearchEngine (to free up the SearchCapability name for the new capability.rs struct) instead of removing it. SearchEngine::search() and SearchEngine::index() reimplement near-identical orchestration to SearchCapability::do_search()/build_index() in the same commit's new capability.rs -- both build a synthetic Config from a SearchConfig and call the same crate::search::search()/crate::index::build_index() free functions. Confirmed via grep across the whole workspace: SearchEngine is referenced nowhere outside notectl-search/src/lib.rs itself (its own definition plus one unit test) -- it is dead code that duplicates a real, actively-used abstraction. This is a Concise-axis finding: two parallel public entry points to the same pipeline double the maintenance surface (e.g. a change to search::SearchOptions construction now needs updating in two places) for no value.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 SearchEngine struct and its impl block are removed from notectl-search/src/lib.rs
- [ ] #2 The test coverage SearchEngine::search() provided (sparse-only search without the embeddings feature) is preserved by rewriting it against SearchCapability instead of being silently dropped
- [ ] #3 grep -rn "SearchEngine" across the repo returns no matches
- [ ] #4 nix develop -c cargo build (workspace-wide) succeeds
- [ ] #5 nix develop -c cargo test -p notectl-search --all-features passes
- [ ] #6 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust+WebAssembly core (crates/gql-core) with a TypeScript/React web app (web/). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Confirm SearchEngine is unused outside its own module before deleting anything: run `nix develop -c cargo build` (workspace-wide, all default features) then `grep -rn "SearchEngine" --include=*.rs .` from the repo root. As of this review, SearchEngine is referenced only in notectl-search/src/lib.rs (its own definition, impl block, and one unit test) -- nothing in src/ (the main binary), web/, or any other crate uses it. If this has changed (e.g. a task landed between this ticket being filed and being picked up that added a new caller), stop and re-scope: migrate that caller to SearchCapability instead of deleting SearchEngine out from under it.
2. Open notectl-search/src/lib.rs. Delete the `SearchEngine` struct and its `impl SearchEngine` block (~line 98-148): the `new`, `search`, and `index` methods duplicate SearchCapability::do_search / SearchCapability::build_index in notectl-search/src/capability.rs (both wrap crate::search::search() / crate::index::build_index() with near-identical Config construction from a SearchConfig).
3. Delete or rewrite the one test that references SearchEngine: `test_search_without_embeddings_runs_sparse_only` (~line 182-214, under `#[cfg(not(feature = "embeddings"))]`). Since it exercises real behavior (search() in sparse-only mode without the embeddings feature) that's still worth covering, rewrite it to go through `SearchCapability` instead: construct `SearchCapability::new(base, Arc::new(Config { exclude_paths: Vec::new(), daily_note_patterns: vec!["YYYY-MM-DD.md".to_string()], search: SearchConfig::default() }))` and call `.do_search("test document", 50, SearchMode::Sparse, false)`, asserting `Ok`. Do not just delete the test outright -- the underlying behavior (search works without the embeddings feature) is worth keeping covered somewhere.
4. Remove the now-unused `use std::path::PathBuf;` import from lib.rs if nothing else in the file still needs it (check remaining usages first -- SearchError variants and other code may still reference PathBuf).
5. Run `nix develop -c cargo build` (workspace-wide) and confirm nothing outside notectl-search referenced SearchEngine (this should already be a compile error if something did, per step 1's grep).
6. Run: nix develop -c cargo test -p notectl-search --all-features
7. Run: nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings
8. Run: nix develop -c cargo fmt -p notectl-search -- --check (fix formatting if needed).
9. Run: nix develop -c cargo build (workspace-wide, all features off and on) to confirm nothing else in the workspace broke.
<!-- SECTION:PLAN:END -->
