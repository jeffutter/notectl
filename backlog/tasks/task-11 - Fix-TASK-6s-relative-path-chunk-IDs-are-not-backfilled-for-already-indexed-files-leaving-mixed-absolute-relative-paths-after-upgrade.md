---
id: TASK-11
title: >-
  Fix: TASK-6's relative-path chunk IDs are not backfilled for already-indexed
  files, leaving mixed absolute/relative paths after upgrade
status: To Do
assignee: []
created_date: '2026-07-16 13:39'
updated_date: '2026-07-16 13:39'
labels:
  - review-followup
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
SETUP (read first): This is a Rust+WebAssembly core (crates/gql-core) with a
TypeScript/React web app (web/). ALL commands must run inside the Nix dev
shell: either run 'direnv allow' once, or prefix every command with
'nix develop -c'. Work from the repository root unless told otherwise. Do not
change pinned dependency versions.

(This repo is notectl; the crate under test is notectl-search. The same Nix-shell rule applies.)

**Scope**: Only notectl-search/src/storage.rs.

1. Open notectl-search/src/storage.rs and locate the INDEX_FORMAT_VERSION constant (~line 13-14):
   ```rust
   /// Current manifest format version. Bumped from 1 -> 2 for the richer schema.
   pub const INDEX_FORMAT_VERSION: u32 = 2;
   ```
   Bump it to 3 and update the doc comment to explain why:
   ```rust
   /// Current manifest format version. Bumped from 2 -> 3 because
   /// IndexBuilder::build now writes vault-relative Chunk.source_file/id
   /// values instead of absolute paths (see TASK-6); old v2 manifests may
   /// contain absolute-path chunk entries for files that haven't changed
   /// since the upgrade, so they must be treated as stale and rebuilt.
   pub const INDEX_FORMAT_VERSION: u32 = 3;
   ```

2. Read the version-mismatch handling around storage.rs:376-389 (inside SearchIndex::open_or_create or wherever the manifest is parsed) to confirm the existing behavior: on version != INDEX_FORMAT_VERSION, the manifest is treated as empty (forcing IndexBuilder::build to do a full walk + rechunk of every file, since compute_staleness_diff will see no prior file_hashes). No logic changes are needed here — bumping the constant alone is sufficient to route old manifests through this existing path. Confirm this by reading, do not change this function.

3. Find test_open_or_create_version_mismatch (~storage.rs:913-947) and add a new test alongside it, e.g. test_open_or_create_v2_manifest_with_absolute_paths_is_rebuilt: construct a manifest JSON literal with "version": 2 (the OLD value, now stale) and at least one chunk entry whose source_file is an absolute-looking path (e.g. "/home/alice/vault/note.md"), write it to the expected manifest.json location in a TempDir index dir, call SearchIndex::open_or_create with the same signature used elsewhere in this test module, and assert: (a) the returned manifest.chunks is empty, (b) manifest.version == INDEX_FORMAT_VERSION (the new value), matching the existing test_open_or_create_version_mismatch pattern but specifically naming the TASK-6 migration scenario in the test name/doc comment.

4. Grep notectl-search/src for any other place that hardcodes the literal 2 where it means 'current manifest version' (as opposed to unrelated uses of the number 2) — there should be none outside storage.rs; all should reference INDEX_FORMAT_VERSION. Fix any that don't.

5. Run: nix develop -c cargo test -p notectl-search --all-features and nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings. Fix any failures — existing tests asserting manifest.version == INDEX_FORMAT_VERSION should keep passing unmodified since they reference the constant.

6. In the task's Implementation Notes, state the version bump (2 -> 3) and confirm no other code needed to change because open_or_create's existing version-mismatch path already does the right thing.
<!-- SECTION:PLAN:END -->
