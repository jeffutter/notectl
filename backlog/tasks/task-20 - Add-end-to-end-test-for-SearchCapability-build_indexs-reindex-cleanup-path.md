---
id: TASK-20
title: 'Add end-to-end test for SearchCapability::build_index''s --reindex cleanup path'
status: To Do
assignee: []
created_date: '2026-07-16 16:53'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-15
priority: high
type: task
ordinal: 103
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-15 (notectl-search/src/capability.rs SearchCapability::build_index, ~line 139-198, and notectl-search/src/storage.rs). TASK-15's AC #4 required 'A test in capability.rs (or storage.rs) confirms --reindex preserves the models/ directory while removing manifest/chunks/vectors', and TASK-15's own implementation plan step 7 specified exercising it via SearchCapability::build_index(true, None, None) end-to-end. What was actually delivered is storage.rs's test_reindex_cleanup_preserves_models_dir (storage.rs ~line 1829), which calls index.remove_manifest()/clear_chunks()/remove_vectors() directly in sequence -- the same three calls capability.rs's build_index makes, but NOT through build_index itself. There is still zero test coverage of SearchCapability::build_index's actual reindex branch (capability.rs ~line 156-183): the index_dir.exists() guard, the SearchIndex::open_or_create(&index_dir, config.search.model_id.clone(), ...) call using the (possibly model/dim-overridden) config, and the internal_error(...) wrapping on each fallible call. This is a Correctness-axis gap: the AC was checked off, but the test substituting for it exercises the same effect via a different, lower-level code path than the one TASK-15 actually modified, so a regression introduced directly in build_index's reindex block (e.g. wrong index_dir computation, wrong config passed to open_or_create, calls reordered, or the exists() guard removed) would not be caught by any existing test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A new test in notectl-search/src/capability.rs (new #[cfg(test)] mod tests block, since none currently exists in that file outside remote_command_tests) constructs a SearchCapability against a temp vault with an existing index (write real chunks/manifest/vectors.bin via crate::index::build_index or SearchIndex helpers, plus a models/ dir with a placeholder file) and calls capability.build_index(true, None, None).await
- [ ] #2 The test asserts the call returns Ok, that manifest.json/chunks//vectors.bin are removed and then rebuilt (post-call state reflects a fresh index, not the pre-cleanup stale one), and that models/model.bin still exists afterward
- [ ] #3 A second small test in the same module calls capability.build_index(true, None, None).await against a vault with NO existing index directory and asserts it succeeds without error (covers the index_dir.exists() guard's false branch)
- [ ] #4 nix develop -c cargo test -p notectl-search --all-features passes
- [ ] #5 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust CLI workspace (notectl-core, notectl-outline, notectl-search, notectl-files, notectl-tags, notectl-tasks, notectl-daily-notes, plus the main notectl binary). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-search/src/capability.rs. Confirm there is currently no #[cfg(test)] mod tests block in this file (only remote_command_tests, added by TASK-14, which covers get_remote_command/args_to_json, not build_index). Add a new module near the bottom of the file: #[cfg(test)] mod tests { use super::*; use tempfile::TempDir; ... }.
2. Write test build_index_reindex_removes_and_rebuilds_artifacts_preserves_models: create a TempDir, write a markdown file (e.g. base.join("hello.md") with a heading and a paragraph, following the pattern in notectl-search/src/lib.rs's test_search_without_embeddings_runs_sparse_only for how to set up a minimal vault). Construct a SearchCapability::new(base.clone(), Arc::new(Config::default())) and call capability.build_index(false, None, None).await.unwrap() once to create a real initial index (this exercises the normal, non-reindex path and populates manifest.json/chunks//vectors.bin under the config's resolved index dir -- use config.search.resolve_index_dir(&base) to find where artifacts land, mirroring capability.rs's own build_index method, ~line 158).
3. After the initial build, manually create a models/ directory under the resolved index dir with a placeholder file (fs::create_dir_all + fs::write), simulating a downloaded embedding model that must survive --reindex.
4. Call capability.build_index(true, None, None).await and assert it returns Ok(_). Assert the models/ directory and its placeholder file still exist (fs::exists checks against the resolved index dir).
5. Assert the index was actually rebuilt: e.g. check content_hash on the returned IndexResponse is non-empty / files_indexed >= 1, or re-open a SearchIndex against the resolved dir and confirm the manifest reflects the rebuilt state (files_indexed matches the single markdown file written in step 2).
6. Write a second, smaller test build_index_reindex_when_no_existing_index_succeeds: a fresh TempDir with a markdown file but NO prior build_index call (so the index dir doesn't exist yet), call capability.build_index(true, None, None).await, and assert Ok(_) -- this exercises the 'if index_dir.exists()' guard's false branch (capability.rs ~line 159), confirming --reindex on a brand-new vault doesn't error out trying to clean up artifacts that were never created.
7. Run: nix develop -c cargo test -p notectl-search --all-features
8. Run: nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings
9. Run: nix develop -c cargo fmt -p notectl-search -- --check (fix formatting if needed).
<!-- SECTION:PLAN:END -->
