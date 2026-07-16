---
id: TASK-15
title: >-
  Fix: --reindex cleanup in SearchCapability::build_index duplicates
  storage.rs's persistence-format knowledge via raw fs calls
status: Dev Ready
assignee: []
created_date: '2026-07-16 14:39'
updated_date: '2026-07-16 14:40'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-1.10
priority: high
type: bug
ordinal: 130
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-1.10 (notectl-search/src/capability.rs:156-181, SearchCapability::build_index). The --reindex cleanup path manually deletes manifest.json, chunks/, and vectors.bin via raw std::fs::remove_file/remove_dir_all calls against paths capability.rs constructs itself. But notectl-search/src/storage.rs already owns exactly this responsibility -- its own doc comment says 'SearchIndex owns persistence: open_or_create, save_manifest, write_chunks, write_vectors, read_vectors, read_chunk, clear_chunks, remove_chunks, reset' -- and SearchIndex::clear_chunks() already does the identical chunks/ removal. capability.rs re-encoding the artifact filenames (manifest.json, vectors.bin) is information leakage per the project's design philosophy (same persistence-format knowledge duplicated across module boundaries): if storage.rs's on-disk layout ever changes (e.g. renaming vectors.bin, splitting manifest.json), capability.rs's --reindex path silently breaks or leaves stale artifacts, since nothing would fail to compile to catch it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 notectl-search/src/storage.rs gains SearchIndex::remove_manifest() and SearchIndex::remove_vectors() methods, following clear_chunks()'s existing style
- [ ] #2 SearchCapability::build_index's --reindex cleanup in capability.rs calls SearchIndex::remove_manifest/clear_chunks/remove_vectors instead of raw std::fs calls, and no longer hardcodes manifest.json/vectors.bin filenames
- [ ] #3 A test in storage.rs covers remove_manifest and remove_vectors (removes existing files, no-ops cleanly when absent)
- [ ] #4 A test in capability.rs (or storage.rs) confirms --reindex preserves the models/ directory while removing manifest/chunks/vectors
- [ ] #5 nix develop -c cargo test -p notectl-search --all-features passes
- [ ] #6 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust+WebAssembly core (crates/gql-core) with a TypeScript/React web app (web/). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-search/src/capability.rs. Locate `SearchCapability::build_index` (~line 139-196), specifically the `if reindex { ... }` block (~line 157-181) that manually deletes `manifest.json`, the `chunks/` directory, and `vectors.bin` via raw `std::fs::remove_file`/`std::fs::remove_dir_all` calls against paths it constructs itself (`index_dir.join("manifest.json")`, `index_dir.join("chunks")`, `index_dir.join("vectors.bin")`).
2. Open notectl-search/src/storage.rs and review `SearchIndex::clear_chunks` (~line 541-548, removes the `chunks/` dir) and `SearchIndex::reset` (~line 571-575, removes the entire base_dir including `models/`). Note storage.rs's own doc comment (~line 583-584) states it is the owner of persistence: "SearchIndex owns persistence: open_or_create, save_manifest, write_chunks, write_vectors, read_vectors, read_chunk, clear_chunks, remove_chunks, reset." capability.rs's manual `std::fs` calls duplicate that ownership and re-encode the artifact filenames (`manifest.json`, `vectors.bin`) that only storage.rs should know about.
3. Add two new methods to `impl SearchIndex` in storage.rs: `remove_manifest(&self) -> Result<(), SearchError>` (removes `self.base_dir.join("manifest.json")` if it exists) and `remove_vectors(&self) -> Result<(), SearchError>` (removes `self.base_dir.join("vectors.bin")` if it exists) -- follow the existing style of `clear_chunks` (check `.exists()`, map errors to `SearchError::Storage`).
4. In capability.rs's `build_index`, replace the manual `std::fs` block with: open (or construct) a `SearchIndex` for `index_dir` and call `index.remove_manifest()?`, `index.clear_chunks()?`, `index.remove_vectors()?` -- propagating errors via `internal_error(...)` as the surrounding code already does for other fallible calls. Do NOT use `SearchIndex::reset()` since it deletes the entire base_dir including `models/`, which --reindex must preserve (per the existing doc comment on build_index, ~line 136-138).
5. Confirm `index_dir.exists()` guard behavior is preserved (skip cleanup entirely if the index directory doesn't exist yet).
6. Add/update a test in notectl-search/src/storage.rs for the two new methods (e.g. write a manifest/vectors file, call remove_manifest/remove_vectors, assert files are gone and no error is returned when the files don't exist).
7. Add/update a test in notectl-search/src/capability.rs (new `#[cfg(test)] mod tests` block if none exists) exercising `SearchCapability::build_index(true, None, None)` against a vault with an existing index, asserting `models/` survives while manifest/chunks/vectors are removed and rebuilt.
8. Run: nix develop -c cargo test -p notectl-search --all-features
9. Run: nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings
10. Run: nix develop -c cargo fmt -p notectl-search -- --check (fix formatting if needed).
<!-- SECTION:PLAN:END -->
