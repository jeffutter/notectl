---
id: TASK-7
title: 'Cleanup: remove dead duplicate SearchIndex::build_index in storage.rs'
status: To Do
assignee: []
created_date: '2026-07-16 07:21'
updated_date: '2026-07-16 07:22'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-1.8
priority: high
type: chore
ordinal: 120
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-1.8 (notectl-search/src/storage.rs:585-706, `SearchIndex::build_index`). TASK-1.8's own accepted implementation plan said it would "Refactor SearchIndex.build_index() out of its orchestration role (keep as storage layer only)" — but the method was left fully intact, unchanged, and still implements the complete walk -> diff -> chunk -> persist pipeline (minus embedding), duplicating ~120 lines of logic that now also lives in `IndexBuilder::build` (notectl-search/src/index.rs).

Confirmed via grep that `SearchIndex::build_index` has zero production callers — `crate::index::build_index` (the new IndexBuilder-based entrypoint) is what `notectl-search/src/lib.rs` actually calls. The old method's doc comment even says "This synchronous method is kept for backward compatibility and testing," but there is no external caller depending on it for backward compatibility (it isn't part of any public CLI/API surface yet — TASK-1.10, the first consumer, hasn't landed). Its only remaining callers are its own ~15 near-duplicate unit tests in storage.rs.

This is a Concise/Organization-axis finding: two independent implementations of the same pipeline is a real maintenance cost — any future change to the walk/diff/chunk/persist logic (e.g. a bugfix) has to be applied twice or it silently drifts, and a new reader has no way to know which one is authoritative without reading both. (As one concrete example of the drift risk, the old build_index passes the same absolute-path Chunk::source_file bug described in a sibling ticket into its own chunk_file call — a fix applied only to IndexBuilder would leave this dead copy still wrong.)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 SearchIndex::build_index and its EmbedderRef parameter/type (notectl-search/src/storage.rs) are removed, along with the ~15 tests in storage.rs that exist solely to exercise it (grep for '.build_index(' in storage.rs test module to find them)
- [ ] #2 Any assertions in the removed tests that covered behavior NOT otherwise covered by notectl-search/src/index.rs's IndexBuilder tests (e.g. specific staleness-diff edge cases) are ported to storage.rs's compute_staleness_diff tests or index.rs's IndexBuilder tests instead of being silently dropped — check this by diffing the removed test names/assertions against existing coverage before deleting
- [ ] #3 notectl-search compiles and all remaining tests pass with no reference to SearchIndex::build_index anywhere in notectl-search/src
- [ ] #4 nix develop -c cargo test -p notectl-search --all-features passes
- [ ] #5 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust+WebAssembly core (crates/gql-core) with a
TypeScript/React web app (web/). ALL commands must run inside the Nix dev
shell: either run 'direnv allow' once, or prefix every command with
'nix develop -c'. Work from the repository root unless told otherwise. Do not
change pinned dependency versions.

(This repo is notectl; the crate under test is notectl-search. The same Nix-shell rule applies.)

1. Open notectl-search/src/storage.rs and locate `pub fn build_index` on `impl SearchIndex` (~line 585-706), and the `EmbedderRef` type it takes as a parameter (search for `EmbedderRef` definition, likely just above or near this method).

2. Before deleting anything, run `grep -n "\.build_index(\|EmbedderRef" notectl-search/src/storage.rs` and read every one of the ~15 test functions that call `.build_index(...)` (they're all inside `storage.rs`'s own `#[cfg(test)] mod tests`, roughly lines 1080-1660). For each test, note what specific behavior it's asserting (e.g. staleness diff on model change, incremental add/remove, exclusion filtering, up-to-date detection).

3. Cross-reference each behavior from step 2 against the tests already in `notectl-search/src/index.rs`'s `#[cfg(test)] mod tests` (which exercise the same scenarios through `IndexBuilder::build`/`run_build`) and against `compute_staleness_diff`'s own tests in storage.rs (search for `test_staleness_diff_` — these test the diff computation directly, independent of `build_index`). For any storage.rs `build_index`-based test whose exact scenario is NOT already covered by an `index.rs` test or a `test_staleness_diff_*` test, port that scenario as a new test in `notectl-search/src/index.rs` using the existing `run_build` helper there, BEFORE deleting the original.

4. Delete `SearchIndex::build_index` (storage.rs) and the `EmbedderRef` type/enum if it has no other callers after the deletion (re-check with grep). Delete all ~15 test functions in storage.rs that called `.build_index(...)` — they no longer compile once the method is gone.

5. Run `nix develop -c cargo build -p notectl-search --all-features` to confirm the crate still compiles with the method removed (catches any other caller you missed).

6. Run `nix develop -c cargo test -p notectl-search --all-features` — confirm the total test count only dropped by however many storage.rs tests were deleted minus however many were ported to index.rs in step 3, and that everything passes.

7. Run `nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings` and fix any warnings (e.g. now-unused imports in storage.rs after the deletion).

8. In the task's Implementation Notes, list which (if any) test scenarios were ported to index.rs per step 3, so the coverage trail is documented.
<!-- SECTION:PLAN:END -->
