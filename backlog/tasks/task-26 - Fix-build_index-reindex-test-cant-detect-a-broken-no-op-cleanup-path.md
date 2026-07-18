---
id: TASK-26
title: 'Fix: build_index --reindex test can''t detect a broken/no-op cleanup path'
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-18 05:49'
updated_date: '2026-07-18 21:36'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-20
priority: high
ordinal: 100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-20 (notectl-search/src/capability.rs:625-689, test build_index_reindex_removes_and_rebuilds_artifacts_preserves_models). The test reuses the exact same unchanged markdown file across both the initial build_index(false,...) call and the build_index(true,...) reindex call. Verified empirically: temporarily changing 'if reindex {' to 'if reindex && false {' in SearchCapability::build_index (capability.rs:157) -- i.e. completely disabling the manifest/chunks/vectors wipe -- and re-running 'nix develop -c cargo test -p notectl-search --all-features build_index_tests' still passes both tests with zero failures. This happens because compute_staleness_diff (storage.rs) sees no file changes and returns UpToDate, so IndexBuilder::build short-circuits without needing the cleanup to have run; every assertion in the test (manifest.json exists, chunks/ is a dir, content_hash non-empty, files_indexed >= 1, models/model.bin exists) is still true purely from the untouched first build. This is exactly the regression class TASK-20 itself was filed to catch ('a regression introduced directly in build_index's reindex block... would not be caught by any existing test') -- and after TASK-20 shipped, it still isn't caught. This is a Correctness-axis gap: the delivered test doesn't achieve what its own AC #2 required. Note also that TASK-20's backlog file was marked Done with all 5 AC checkboxes still unchecked and no Implementation Notes/Final Summary recorded -- a process gap worth keeping in mind, though this ticket only covers the code-level test fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 build_index_reindex_removes_and_rebuilds_artifacts_preserves_models (or a renamed/split test) proves the wipe actually happened by asserting something that ONLY the reindex cleanup path would change -- e.g. a sentinel file planted in <index_dir>/chunks/ after the initial build is confirmed gone after the reindex call
- [x] #2 Temporarily disabling the cleanup block in SearchCapability::build_index (capability.rs ~157-183, e.g. changing 'if reindex {' to 'if reindex && false {') causes the updated test to FAIL when verified manually during implementation; the temporary change is reverted before committing
- [x] #3 nix develop -c cargo test -p notectl-search --all-features passes
- [x] #4 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust CLI workspace (notectl-core, notectl-outline, notectl-search, notectl-files, notectl-tags, notectl-tasks, notectl-daily-notes, plus the main notectl binary). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-search/src/capability.rs and locate the build_index_tests module (~line 617-719, added by TASK-20).

2. In build_index_reindex_removes_and_rebuilds_artifacts_preserves_models (~line 625), after the initial build_index(false, None, None) call and its existence assertions (~line 649-651: manifest.json exists, chunks/ is a dir), and before creating the models/ dir, write a sentinel file directly into the chunks/ directory that a real chunker-driven rebuild would never produce, e.g.:
   fs::write(index_dir.join("chunks").join("_stale_sentinel.txt"), b"stale").unwrap();

3. After the reindex call succeeds and its existing assertions (~line 660-664), add:
   assert!(!index_dir.join("chunks").join("_stale_sentinel.txt").exists(), "stale sentinel chunk file must be removed by --reindex cleanup, proving the wipe actually ran");

4. Verify the test is now actually rigorous: temporarily edit capability.rs line ~157 from 'if reindex {' to 'if reindex && false {', run 'nix develop -c cargo test -p notectl-search --all-features capability::build_index_tests', and confirm the updated test now FAILS on the new sentinel assertion (proving it would have caught the exact regression TASK-20 was filed to prevent). Then revert that temporary edit back to 'if reindex {' -- do not commit the temporary change.

5. Run: nix develop -c cargo test -p notectl-search --all-features -- confirm everything passes with the temporary edit reverted.

6. Run: nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings.

7. Run: nix develop -c cargo fmt -p notectl-search -- --check (fix formatting if needed).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation Notes: Added sentinel file (_stale_sentinel.txt) planted in chunks/ after initial build. After reindex, assert it's gone — proving clear_chunks() (fs::remove_dir_all) actually ran. Verified rigor by temporarily disabling cleanup block (if reindex && false): test correctly fails on sentinel assertion. Reverted temporary change before committing.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added sentinel file assertion to build_index_reindex_removes_and_rebuilds_artifacts_preserves_models test. The sentinel (_stale_sentinel.txt) is planted in chunks/ after initial build and asserted gone after reindex, proving clear_chunks() actually ran. Previously the test passed even with cleanup disabled because compute_staleness_diff returned UpToDate.
<!-- SECTION:FINAL_SUMMARY:END -->
