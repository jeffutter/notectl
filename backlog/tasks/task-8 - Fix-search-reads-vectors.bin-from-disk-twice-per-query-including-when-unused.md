---
id: TASK-8
title: >-
  Fix: search() reads vectors.bin from disk twice per query, including when
  unused
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-16 07:22'
updated_date: '2026-07-16 17:12'
labels:
  - review-followup
  - planned
milestone: Active
dependencies:
  - TASK-1.9
priority: high
type: bug
ordinal: 140
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-1.9 (notectl-search/src/search.rs). `search()` reads and deserializes `vectors.bin` from disk twice on a single call, and once unconditionally regardless of the requested mode:

1. Lines 186-196 (`has_vectors` computation) call `index.read_vectors()` unconditionally for every search — including when `options.mode == SearchMode::Sparse` was explicitly requested — just to check whether the result is non-empty, then discard the actual vectors. Note that for `SearchMode::Sparse`, `effective_mode` (the match at lines 199-213) has no arm for `(Sparse, _)` other than the catch-all `(mode, _) => mode`, so `has_vectors` can never change the outcome for an explicit Sparse query — the read is pure waste on that path.
2. Lines 290-301 (`dense_vectors` computation) call `index.read_vectors()` again, this time to actually use the result, whenever `final_mode.needs_dense()`.

For a Dense or Hybrid query this means `vectors.bin` — which can be large for a big vault (one f32 vector per chunk) — is read and deserialized from disk twice per search call. For a Sparse-only query it's read once for no reason at all. This is a Concise/efficiency-axis finding: real, recurring cost on a hot path (every single search request), not a one-time cost.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 notectl-search/src/search.rs::search() calls index.read_vectors() at most once per invocation
- [x] #2 When options.mode == SearchMode::Sparse, index.read_vectors() is not called at all (has_vectors/auto-degradation logic is restructured so it only reads vectors when the requested mode could actually use them, i.e. Dense or Hybrid)
- [x] #3 The single read result is reused for both the has_vectors/auto-degradation check and the actual cosine_top_k scoring, rather than being read once and discarded then read again
- [x] #4 Existing behavior is unchanged for all modes: auto-degradation from Dense/Hybrid to Sparse when vectors are missing or empty still works, and all existing tests in notectl-search/src/search.rs pass without modification to their assertions
- [x] #5 nix develop -c cargo test -p notectl-search --all-features passes
- [x] #6 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Single-file refactoring in notectl-search/src/search.rs: hoist the vector read to be called at most once per search() invocation, guarded by options.mode.needs_dense().

## Implementation Plan

### Step 1: Hoist vector read behind needs_dense() guard

Replace the unconditional `has_vectors` block (~lines 186-196) with an early conditional read:

```rust
// Read dense vectors ONCE if the requested mode could use them.
// For Sparse mode, skip entirely — has_vectors cannot affect the outcome.
let raw_vectors: Vec<Vec<f32>> = if options.mode.needs_dense() {
    #[cfg(feature = "embeddings")]
    {
        index.read_vectors().unwrap_or_default()
    }
    #[cfg(not(feature = "embeddings"))]
    {
        Vec::new()
    }
} else {
    Vec::new()
};

let has_vectors = !raw_vectors.is_empty() && raw_vectors.len() == manifest.chunks.len();
```

This eliminates the wasteful read on the Sparse path (acceptance criterion #2) and ensures at most one read total.

### Step 2: Reuse raw_vectors in dense_data construction

In the `dense_data` block (~line 240+), replace the second `index.read_vectors()` call with the already-loaded `raw_vectors`:

In the `Ok(qvec)` arm inside `dense_data`, change:
```rust
let vectors = index.read_vectors().unwrap_or_default();
Some((qvec, vectors))
```
to:
```rust
Some((qvec, raw_vectors))
```

Note: `raw_vectors` is moved here. If the embedding fails (Err arm) or model is not ready, `raw_vectors` is dropped without being used — that's fine, we only paid the disk read cost once.

However, since `raw_vectors` is consumed in the `Ok` arm but might not exist in other branches, we need to handle ownership. Two options:

**Option A (preferred — clone only on success):** Keep `raw_vectors` as-is and clone it in the Ok arm:
```rust
Ok(qvec) => Some((qvec, raw_vectors.clone())),
```
This means we clone the vectors Vec only when embedding succeeds. The original is dropped in failure paths. Cost of one Vec clone (pointer + length + capacity, plus the inner vecs which are shared via clone) is negligible compared to the fs::read we're saving.

**Option B (reference-based):** Store `raw_vectors` in a local that outlives the dense_data block and take references. More complex due to lifetime constraints across the async .await point.

Go with Option A for simplicity and clarity.

### Step 3: Verify auto-degradation chain is preserved

The three degradation points must fire identically:

1. **Vectors missing on disk → Dense/Hybrid degrades to Sparse**: `has_vectors` is computed from `raw_vectors` (same boolean check). The `effective_mode` match (~lines 199-213) uses `has_vectors` unchanged. Warn message identical.

2. **Model not downloaded/embedding fails → dense_data = None**: Unchanged. The embedder creation and embed_single call are untouched. When embedding fails, `dense_data = None`, triggering the second degradation in `final_mode`.

3. **Both dense paths fail → final_mode degrades**: Unchanged.

### Step 4: Run tests and clippy

```bash
nix develop -c cargo test -p notectl-search --all-features
nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings
```

All 121+ existing tests should pass without modification. Key tests to watch:
- `test_search_sparse_only` — Sparse mode should work (no vector read at all now)
- `test_auto_degrade_to_sparse_without_embeddings` — auto-degradation still works
- `test_dense_mode_degrades_to_sparse_when_embedding_unavailable` — two-level degradation chain
- `test_search_mode_used_reflects_degradation` — mode_used field correct after degradation

### Step 5: Sanity-check read count (temporary instrumentation)

Temporarily add `tracing::debug!("read_vectors called");` at the top of `SearchIndex::read_vectors()` in storage.rs. Run a Dense-mode test and confirm exactly 1 debug line (was 2 before). Run a Sparse-mode test and confirm 0 lines (was 1 before). Remove the instrumentation before committing.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Single-file refactoring in notectl-search/src/search.rs: hoisted index.read_vectors() to at most one call per search() invocation, guarded by options.mode.needs_dense(). Sparse mode now skips vector read entirely (was 1 read). Dense/Hybrid mode reads vectors once instead of twice. Verified with temporary instrumentation: 0 reads for Sparse, 1 read for Dense. All 137 tests pass, clippy clean.
<!-- SECTION:FINAL_SUMMARY:END -->
