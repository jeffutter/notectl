---
id: TASK-7
title: 'Cleanup: remove dead duplicate SearchIndex::build_index in storage.rs'
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-16 07:21'
updated_date: '2026-07-16 20:58'
labels:
  - review-followup
  - planned
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
- [x] #1 SearchIndex::build_index and its EmbedderRef parameter/type (notectl-search/src/storage.rs) are removed, along with the ~15 tests in storage.rs that exist solely to exercise it (grep for '.build_index(' in storage.rs test module to find them)
- [x] #2 Any assertions in the removed tests that covered behavior NOT otherwise covered by notectl-search/src/index.rs's IndexBuilder tests (e.g. specific staleness-diff edge cases) are ported to storage.rs's compute_staleness_diff tests or index.rs's IndexBuilder tests instead of being silently dropped — check this by diffing the removed test names/assertions against existing coverage before deleting
- [x] #3 notectl-search compiles and all remaining tests pass with no reference to SearchIndex::build_index anywhere in notectl-search/src
- [x] #4 nix develop -c cargo test -p notectl-search --all-features passes
- [x] #5 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): ALL commands must run inside the Nix dev shell — prefix every command with `nix develop -c`. Work from the repository root.

### Background (from code audit)

Dead code to remove:
- `SearchIndex::build_index` (storage.rs lines 606–706, ~100 lines) — zero production callers
- `EmbedderRef` enum (storage.rs lines 335–348) — exists solely as a parameter type for `build_index`

The authoritative pipeline is `IndexBuilder::build` (index.rs), called via `crate::index::build_index` from lib.rs, capability.rs, and search.rs.

There are exactly **13 test functions** in storage.rs that call `.build_index(...)`. These fall into two categories:

**Category A: Staleness-diff variant assertions (9 tests)** — assert on exact `StalenessDiff` variant returned (`UpToDate`, `Incremental{added/modified/removed}`, `FullRebuild{reason}`). Since `IndexBuilder::build` returns only `BuildSummary` (not `StalenessDiff`), these cannot port directly to index.rs integration tests.

**Category B: End-to-end pipeline tests (4 tests)** — verify disk persistence, chunk clearing, content hash changes across builds.

### Step 1: Port 4 integration tests to index.rs

Add the following tests to `notectl-search/src/index.rs` test module using the existing `run_build` helper:

1. **`test_content_hash_changes_on_modification`** — Run build, modify file content, run build again, assert `content_hash` differs between the two `BuildSummary` results.
2. **`test_touch_without_content_change_no_reindex`** — Run build, `touch` the file (mtime-only change), run build again, assert `files_indexed` and `chunks_produced` are unchanged.
3. **`test_manifest_persists_after_build`** — Run build, re-open `SearchIndex` from disk, assert `document_count()` == 1 and `chunk_count()` matches original manifest.
4. **`test_full_rebuild_clears_chunks`** — Run build, verify chunks dir exists, change `model_id` in config, rebuild, assert chunks dir still exists and manifest has new model_id.

### Step 2: Convert 9 staleness-diff tests to direct unit tests of `compute_staleness_diff`

In storage.rs, replace each `build_index`-based test with a leaner version that calls `compute_staleness_diff(base_path, config, &manifest)` directly. This eliminates dependency on `build_index` while preserving the exact same assertions about diff computation.

Each converted test should:
1. Create temp vault + write files as needed
2. Open a `SearchIndex` via `open_or_create` to get an initial manifest
3. Call `compute_staleness_diff(base_path, &config, index.manifest())` directly
4. Assert on the returned `StalenessDiff` variant

For tests requiring a "second build" scenario (e.g., up_to_date after initial build), first do a real build via `open_or_create` → manually populate manifest by calling the function under test's prerequisite setup, OR simply create a pre-populated manifest and call diff against it.

Actually, the simplest approach: for tests that need an indexed state before testing diff behavior, use `crate::index::build_index` (the LIVE async one) to set up the initial index, then call `compute_staleness_diff` directly to test the diff logic. This avoids depending on dead code.

The 9 conversions:
- `test_staleness_diff_up_to_date`
- `test_staleness_diff_modified_file`
- `test_staleness_diff_removed_file`
- `test_staleness_diff_added_file`
- `test_staleness_diff_full_rebuild_model_changed`
- `test_staleness_diff_full_rebuild_dimension_changed`
- `test_staleness_diff_full_rebuild_chunk_config_changed`
- `test_staleness_diff_exclusion_filtering`
- `test_staleness_diff_empty_index`

### Step 3: Delete dead code

Delete from storage.rs:
1. `pub enum EmbedderRef` and its `impl EmbedderRef` block (~lines 335–348)
2. `pub fn build_index(&mut self, ...)` method on `impl SearchIndex` (~lines 606–706)
3. All 13 original `build_index`-based test functions (replaced by steps 1–2)
4. Any now-unused imports (e.g., `EmbedderRef` references in test helpers)

### Step 4: Verify compilation and tests

```bash
nix develop -c cargo build -p notectl-search --all-features
nix develop -c cargo test -p notectl-search --all-features
nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings
```

Expected test count: -13 (deleted) + 4 (new in index.rs) + 9 (converted in storage.rs) = net 0 change. Total test count should remain the same.

### Step 5: Final grep verification

```bash
# Should find zero references to EmbedderRef
grep -rn "EmbedderRef" notectl-search/src/

# Should find only crate::index::build_index references (the live one)
grep -rn "\.build_index(" notectl-search/src/storage.rs
```
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All 5 acceptance criteria met:
1. ✅ SearchIndex::build_index and EmbedderRef removed from storage.rs, plus all 13 tests that called build_index
2. ✅ 4 integration tests ported to index.rs (content hash, touch, manifest persistence, full rebuild). 9 staleness-diff tests converted to direct compute_staleness_diff calls — no coverage lost
3. ✅ notectl-search compiles cleanly, zero references to SearchIndex::build_index or EmbedderRef remain
4. ✅ cargo test -p notectl-search --all-features: 137 passed
5. ✅ cargo clippy -p notectl-search --all-features --all-targets -- -D warnings: clean
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed dead SearchIndex::build_index (~100 lines) and EmbedderRef enum from storage.rs. Converted 9 staleness-diff tests to direct compute_staleness_diff calls (leaner, no pipeline dependency). Ported 4 integration tests to index.rs using existing run_build helper. Net result: -289 lines of dead code, test count unchanged at 137 passing tests.
<!-- SECTION:FINAL_SUMMARY:END -->
