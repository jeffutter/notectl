---
id: TASK-11
title: >-
  Fix: TASK-6's relative-path chunk IDs are not backfilled for already-indexed
  files, leaving mixed absolute/relative paths after upgrade
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-16 13:39'
updated_date: '2026-07-16 21:33'
labels:
  - planned
milestone: Active
dependencies:
  - TASK-6
priority: high
type: bug
ordinal: 150
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-6 (notectl-search/src/index.rs:165, IndexBuilder::build). TASK-6 correctly fixed the chunker call site to pass rel_path instead of abs_path going forward, so newly-indexed and modified files now get vault-relative Chunk.source_file/Chunk.id values.

However, IndexBuilder::build only regenerates chunks for files that compute_staleness_diff flags as added/modified (storage.rs). Files that were already indexed under the OLD binary (pre-TASK-6, absolute-path chunk IDs/source_file) and have not changed since are never re-chunked — file_hashes/file_info_map are keyed by rel_path and content hash, which are unaffected by this fix, so those files' content_hash matches what's already in the manifest and they are treated as up-to-date. Their old ChunkEntry.source_file/id values (containing the full local absolute path) remain in the manifest untouched.

Net effect: any user who upgrades to a build containing the TASK-6 fix, without doing a full rebuild, ends up with a manifest that mixes relative-path chunk entries (for files touched since upgrade) and absolute-path chunk entries (for untouched files) indefinitely. This silently defeats the exact portability guarantee TASK-6 set out to establish (chunk ids stable/portable across machines and mount points), and re-leaks local filesystem paths for any file that simply hasn't changed recently.

The project already has a mechanism for exactly this class of problem: notectl-search/src/storage.rs's INDEX_FORMAT_VERSION / version-mismatch handling (see storage.rs:13-16, 376-389) treats a manifest with a stale format version as empty on open, forcing a full rebuild. This is a Resilient/Correctness-axis finding: the chunk id/path scheme changed in a way that makes old on-disk manifests semantically incompatible, but nothing forces those manifests to be rebuilt.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 INDEX_FORMAT_VERSION in notectl-search/src/storage.rs is bumped (e.g. 2 -> 3), forcing any on-disk manifest written before the TASK-6 relative-path fix to be treated as empty on next open, triggering a full rebuild with correct relative-path chunk ids
- [ ] #2 A new or updated test in notectl-search/src/storage.rs constructs a manifest JSON with the OLD (pre-bump) version number on disk, opens it via SearchIndex::open_or_create, and asserts the resulting in-memory manifest is empty (forcing a rebuild) — following the existing pattern in test_open_or_create_version_mismatch
- [ ] #3 Existing tests that assert manifest.version == INDEX_FORMAT_VERSION continue to pass by referencing the constant (not a hardcoded literal), so they remain correct after the bump
- [ ] #4 nix develop -c cargo test -p notectl-search --all-features passes
- [ ] #5 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Scope: Only notectl-search/src/storage.rs. One constant bump + one new test.

**Step 1: Bump INDEX_FORMAT_VERSION (line 13)**
Change `pub const INDEX_FORMAT_VERSION: u32 = 2;` to `= 3`. Update doc comment:
```rust
/// Current manifest format version. Bumped from 2 -> 3 because
/// IndexBuilder::build now writes vault-relative Chunk.source_file/id
/// values instead of absolute paths (TASK-6); old v2 manifests may
/// contain absolute-path chunk entries for files that haven't changed
/// since upgrade, so they must be treated as stale and rebuilt.
pub const INDEX_FORMAT_VERSION: u32 = 3;
```

**Step 2: Confirm version-mismatch path needs no changes**
Lines 359-370 already handle `parsed.version != INDEX_FORMAT_VERSION` by returning an empty manifest. No logic changes needed — bumping the constant alone routes old v2 manifests through this path automatically.

**Step 3: Add regression test alongside test_open_or_create_version_mismatch (~line 770)**
Name: `test_open_or_create_v2_manifest_with_absolute_paths_is_rebuilt`
- Write a v2-format manifest JSON with `"version": 2` (now stale) and at least one chunk entry with an absolute-looking source_file (e.g. `"/home/alice/vault/note.md"`)
- Call `SearchIndex::open_or_create`
- Assert: `manifest.chunks.is_empty()` AND `manifest.version == INDEX_FORMAT_VERSION` (new value 3)
- Follow the exact pattern of `test_open_or_create_version_mismatch` (same TempDir setup, same open_or_create call signature)

**Step 4: Verify no hardcoded literals**
Grep confirmed: no hardcoded `2` used as manifest version anywhere outside storage.rs. All tests reference `INDEX_FORMAT_VERSION` constant, so they self-adjust.

**Step 5: Quality gates**
- `nix develop -c cargo test -p notectl-search --all-features`
- `nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings`

**Why this works**: When a user upgrades to the binary containing both TASK-6 (rel_path fix) and this version bump, their on-disk v2 manifest has `version: 2`. On next `open_or_create`, the version check fails, the manifest is treated as empty, and `IndexBuilder::build` does a full walk/rechunk of every file using the corrected rel_path path from TASK-6. After rebuild completes, the saved manifest has `version: 3` with all relative-path chunk IDs.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Bumped INDEX_FORMAT_VERSION from 2 to 3 in notectl-search/src/storage.rs, forcing any on-disk v2 manifest (with potentially absolute-path chunk IDs) to be treated as empty on open, triggering a full rebuild with correct relative-path chunk IDs. Added regression test test_open_or_create_v2_manifest_with_absolute_paths_is_rebuilt that writes a v2 manifest with absolute paths and verifies it is discarded for a fresh v3 manifest. All 138 tests pass, clippy clean.
<!-- SECTION:FINAL_SUMMARY:END -->
