---
id: TASK-10
title: >-
  Fix: search() returns error instead of degrading to sparse when Dense-mode
  embedding is unavailable at query time
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-16 13:38'
updated_date: '2026-07-16 16:06'
labels:
  - review-followup
  - planned
milestone: Active
dependencies:
  - TASK-9
priority: high
type: bug
ordinal: 145
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-9 (notectl-search/src/search.rs:244-342). TASK-9 replaced six .expect() calls with a typed ScoreInputs enum, converting a panic into a graceful SearchError — a real improvement. But it did not fix the underlying degradation bug the .expect()s were papering over.

sparse_indexer (built ~line 244) is constructed based on effective_mode.needs_sparse(), which is the mode BEFORE query-time embedding is attempted. final_mode (determined ~line 303, after embedding is attempted) can independently degrade Dense -> Sparse when the model is not ready (index deleted/moved after vectors were written) or embed_single() errors at query time (line 274-291). When a user explicitly requests SearchMode::Dense, vectors.bin exists on disk (has_vectors=true, so effective_mode stays Dense, not auto-degraded), but the embedding call then fails or the model isn't ready: dense_data becomes None, final_mode falls back to Sparse (with a tracing::warn! that says 'Degrading to sparse-only search' / 'Query embedding failed... Degrading to sparse'), yet sparse_indexer was never built (effective_mode was Dense, which does not need sparse). The ScoreInputs match's catch-all arm then fires and search() returns Err(SearchError::Other("Inconsistent search state...")) instead of actually degrading to sparse as the log messages promise.

This is reachable in practice: e.g. a vault was indexed with embeddings, the model cache directory is later deleted to free disk space (vectors.bin remains), and the user runs an explicit Dense-mode search — every such search now hard-errors instead of falling back to BM25 as designed. This is a Correctness-axis bug: the log messages and the ScoreInputs enum's whole purpose (guarantee correct data per mode) are undermined by sparse_indexer being gated on the wrong mode variable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 sparse_indexer in notectl-search/src/search.rs is constructed using final_mode.needs_sparse() (or is otherwise guaranteed to be Some whenever final_mode ends up Sparse via degradation), not effective_mode.needs_sparse()
- [ ] #2 A new test in notectl-search/src/search.rs (feature = "embeddings") reproduces the bug: build an index with embeddings enabled, write vectors.bin directly via index.write_vectors() to simulate previously-computed vectors, then delete/rename the model cache directory (or otherwise force Embedder::is_ready() == false) so has_vectors=true but embedding is unavailable at query time; call search() with SearchOptions { mode: SearchMode::Dense, .. } and assert it returns Ok with sparse-scored results, NOT Err(SearchError::Other(...))
- [ ] #3 The catch-all arm of the ScoreInputs match in search() remains reachable only for genuinely-impossible combinations, not for the Dense-embedding-unavailable case
- [ ] #4 nix develop -c cargo test -p notectl-search --all-features passes
- [ ] #5 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Single-file fix in notectl-search/src/search.rs. No sub-tickets needed — the code change and test are tightly coupled and must ship together.

## Code Change (lines ~255-273 → after line ~323)

**Problem**: `sparse_indexer` (line 255) is gated on `effective_mode.needs_sparse()`, but `final_mode` (line 314) can independently degrade Dense→Sparse after embedding fails. When user requests Dense + vectors.bin exists + model not ready: effective_mode=Dense (no sparse built), final_mode=Sparse (after embed fail), ScoreInputs catch-all fires → hard error.

**Fix**: Move the sparse_indexer construction block (currently lines 255-273) to AFTER the final_mode determination (line 314+), and change its condition from `effective_mode.needs_sparse()` to `final_mode.needs_sparse()`.

This guarantees sparse_indexer is Some whenever final_mode ends up Sparse or Hybrid, regardless of whether that was requested or a runtime degradation. Safe because sparse_indexer only reads manifest, chunk_texts, and query — none depend on dense_data or final_mode.

## Test (new #[cfg(feature = "embeddings")] test in search.rs tests module)

Name: `test_dense_mode_degrades_to_sparse_when_embedding_unavailable`

Setup:
1. Create temp vault with markdown files (sufficient content for chunks).
2. Run `build_index(&base, &config)` — without model downloaded, no real vectors produced.
3. Re-open the index via `SearchIndex::open_or_create(...)`.
4. Get the manifest to determine chunk count.
5. Call `index.write_vectors(&fake_vectors)` where fake_vectors is `vec![vec![0.1f32; 256]; chunk_count]` (256 is default embedding_dim). This simulates "vectors exist on disk from a previous indexing run."
6. Call `search(&base, &config, "query text", SearchOptions { mode: SearchMode::Dense, ..Default::default() })`.

Assertions:
- Result is `Ok(...)` — NOT `Err(SearchError::Other("Inconsistent search state..."))`.
- Results are non-empty (sparse scoring ran successfully).
- Top result has positive score.

The test exercises the exact failure path: mode=Dense + has_vectors=true (from fake vectors.bin) + is_ready()=false (no model in temp dir) → should degrade to sparse and return results.

## Verification

Run:
```bash
nix develop -c cargo test -p notectl-search --all-features
nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings
```

Before fix: new test fails with `Err(SearchError::Other("Inconsistent search state: mode=Sparse, has_dense=false, has_sparse=false"))`.
After fix: new test passes with Ok containing sparse-scored results.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation: Moved sparse_indexer construction from effective_mode.needs_sparse() to final_mode.needs_sparse() in notectl-search/src/search.rs. This ensures BM25 indexer is built whenever final_mode ends up Sparse via query-time degradation (Dense -> Sparse when embedding unavailable). Added test_dense_mode_degrades_to_sparse_when_embedding_unavailable that writes fake vectors.bin, forces Embedder::is_ready()==false, and asserts search(mode=Dense) returns Ok with sparse results instead of Err(SearchError::Other(...)).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed query-time degradation bug: sparse_indexer now gated on final_mode.needs_sparse() instead of effective_mode.needs_sparse(). When Dense mode is requested but embedding fails at query time (model missing, embed_single error), search() correctly degrades to BM25 sparse scoring and returns Ok results instead of hard-erroring with 'Inconsistent search state'. Added regression test that reproduces the exact failure path. All 123 tests pass with --all-features, clippy clean.
<!-- SECTION:FINAL_SUMMARY:END -->
