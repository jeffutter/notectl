---
id: TASK-24
title: >-
  Fix: version-mismatch rebuild leaves orphaned absolute-path chunk files and
  vectors.bin on disk
status: Needs Plan
assignee: []
created_date: '2026-07-16 22:10'
updated_date: '2026-07-16 23:47'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-11
priority: high
ordinal: 102
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-11 (notectl-search/src/storage.rs:362-378, notectl-search/src/index.rs:105-131). TASK-11 bumped INDEX_FORMAT_VERSION from 2 to 3 so that on-disk manifests written before the TASK-6 relative-path fix are discarded and rebuilt with correct vault-relative chunk IDs. This works for manifest.json itself: SearchIndex::open_or_create (storage.rs:362-378) replaces a stale-version manifest with SearchManifest::new_empty(...), which has empty files/chunks lists.

However, IndexBuilder::build (index.rs:96-244) branches on StalenessDiff, and the version-bump path never produces StalenessDiff::FullRebuild — it produces StalenessDiff::Incremental{added: <every current file>, modified: [], removed: []}, because compute_staleness_diff (storage.rs:164-268) just compares the current vault's files against manifest.files, which is now empty, so every file looks 'added', not 'removed'. Only the FullRebuild branch (index.rs:115-124) calls self.index.clear_chunks() and removes vectors.bin; the Incremental branch (index.rs:125-130) only removes chunks for files in `removed`, which is empty here since the underlying markdown files never went away.

Net effect: after an upgrade, the first index/search run writes a fresh manifest.json with correct relative-path chunk IDs and a fresh vectors.bin (write_vectors always overwrites the whole file — confirmed safe), but the OLD chunk text files under <index_dir>/chunks/, whose filenames are derived from the old absolute-path chunk IDs via `id.replace(['/', '\\', ':'], "_")` (storage.rs write_chunks/read_chunk), are never deleted. Those filenames literally embed the pre-upgrade local absolute path (e.g. a chunk id '/home/alice/vault/note.md:0:intro' becomes the file '_home_alice_vault_note.md_0_intro.txt'). This is exactly the portability/privacy leak TASK-6 and TASK-11 were filed to eliminate — it just moved from manifest.json into permanently-orphaned files in chunks/. This is a Resilient/Correctness-axis finding: TASK-11's own description states the risk as 're-leaks local filesystem paths for any file that simply hasn't changed recently', and the shipped fix does not fully close that gap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After SearchIndex::open_or_create discards a manifest due to INDEX_FORMAT_VERSION mismatch (or a parse error), the on-disk chunks/ directory and vectors.bin under that index's base_dir are removed as part of the same open_or_create call (or, alternatively, the returned state clearly signals a full rebuild so IndexBuilder::build's FullRebuild branch — not the Incremental branch — runs on the next build() call; pick whichever approach requires the smaller, more localized change and document the choice in the task's Implementation Notes)
- [ ] #2 A new regression test in notectl-search/src/storage.rs (or index.rs, whichever ends up owning the fix) writes an old chunk text file with an absolute-path-derived filename plus a vectors.bin file into a temp index dir alongside a stale-version manifest.json, then opens/builds the index at the current version, and asserts the old chunk file and vectors.bin no longer exist afterward
- [ ] #3 Existing tests test_open_or_create_v2_manifest_with_absolute_paths_is_rebuilt (storage.rs) and test_full_rebuild_clears_chunks (index.rs) continue to pass unmodified
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

(This repo is notectl; the crate under fix is notectl-search. The same Nix-shell rule applies.)

1. Read notectl-search/src/storage.rs:343-387 (SearchIndex::open_or_create) and notectl-search/src/index.rs:96-131 (IndexBuilder::build's StalenessDiff match) in full before changing anything, to confirm the current control flow described above still matches reality.

2. Preferred fix (localized to storage.rs, mirrors the existing --reindex cleanup pattern in SearchCapability::build_index at notectl-search/src/capability.rs:156-182): in SearchIndex::open_or_create, in the two branches that currently return SearchManifest::new_empty(...) after logging a warning (storage.rs ~364-372 for version mismatch, ~373-377 for parse error), also wipe the stale on-disk artifacts for that index dir before returning:
   - Remove base_dir.join("chunks") recursively if it exists (same as SearchIndex::clear_chunks, but clear_chunks takes &self and SearchIndex isn't constructed yet at this point — either inline the fs::remove_dir_all call the same way IndexBuilder::build already does for vectors.bin at index.rs:118-123, or restructure to construct the SearchIndex first with the empty manifest and then call self.clear_chunks()/self.remove_vectors() before returning Ok(Self { ... }))
   - Remove base_dir.join("vectors.bin") if it exists
   Do NOT remove base_dir.join("models") — that directory holds the downloaded embedding model and must be preserved, exactly like the --reindex cleanup path already preserves it (see test_reindex_cleanup_preserves_models_dir in storage.rs for the existing convention).

3. If step 2 turns out to be awkward given open_or_create's current return-early control flow, the alternative is to make compute_staleness_diff / StalenessDiff aware that the manifest was just reset due to a version mismatch (wire up the existing-but-dead RebuildReason::VersionMismatch variant at storage.rs:124 so it's actually constructed and returned as StalenessDiff::FullRebuild(RebuildReason::VersionMismatch) the first time build() runs against a freshly-reset empty manifest), so IndexBuilder::build's FullRebuild branch (index.rs:115-124) does the cleanup instead. This requires plumbing a 'was this manifest just reset due to version mismatch' flag from SearchIndex through to compute_staleness_diff's caller. Only take this path if step 2 proves impractical; prefer step 2 for its smaller blast radius.

4. Add the regression test described in AC #2: in notectl-search/src/storage.rs's #[cfg(test)] mod tests, near test_open_or_create_v2_manifest_with_absolute_paths_is_rebuilt (~line 776), write a variant (e.g. test_open_or_create_v2_manifest_cleans_up_orphaned_chunk_files) that additionally: creates the chunks/ subdirectory in the TempDir index dir and writes a dummy .txt file into it with an absolute-path-derived name (e.g. '_home_alice_vault_note.md_0_intro.txt'), writes a dummy vectors.bin file (any non-empty byte content is fine — the test only checks existence, not contents), then calls SearchIndex::open_or_create with the current INDEX_FORMAT_VERSION, and asserts both the dummy chunk file and vectors.bin no longer exist on disk after the call returns. Also create a models/ subdir with a placeholder file first and assert it still exists afterward, mirroring test_reindex_cleanup_preserves_models_dir's preservation check.

5. Run: nix develop -c cargo test -p notectl-search --all-features — confirm the new test passes and all 138+ existing tests still pass, in particular test_open_or_create_v2_manifest_with_absolute_paths_is_rebuilt, test_full_rebuild_clears_chunks, and test_reindex_cleanup_preserves_models_dir.

6. Run: nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings — fix any warnings.

7. In the task's Implementation Notes, state which of step 2 or step 3 was taken and why.
<!-- SECTION:PLAN:END -->
