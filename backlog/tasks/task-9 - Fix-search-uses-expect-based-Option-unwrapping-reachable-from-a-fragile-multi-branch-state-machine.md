---
id: TASK-9
title: >-
  Fix: search() uses expect()-based Option unwrapping reachable from a fragile
  multi-branch state machine
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-16 07:22'
updated_date: '2026-07-16 12:13'
labels:
  - review-followup
  - planned
milestone: Active
dependencies:
  - TASK-1.9
priority: high
type: bug
ordinal: 130
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-1.9 (notectl-search/src/search.rs:303-350, the "Score & rank" match on `final_mode`). Six `.expect()` calls unwrap `Option<Vec<f32>>` / `Option<SparseIndexer>` values in production code:

```rust
let qvec = query_vec.expect("query_vec should be present for Dense mode");       // line 305
let vectors = dense_vectors.as_ref().expect("vectors should exist for Dense mode"); // line 308
let indexer = sparse_indexer.expect("sparse_indexer should exist for Sparse mode"); // line 320
let qvec = query_vec.expect("query_vec should be present for Hybrid mode");      // line 332
let vectors = dense_vectors.as_ref().expect("vectors should exist for Hybrid mode"); // line 335
let indexer = sparse_indexer.expect("sparse_indexer should exist for Hybrid mode"); // line 339
```

Whether these can actually be `None` depends on a fragile, manually-maintained invariant spanning ~150 lines: `effective_mode` (derived from `options.mode` + `has_vectors`, lines 199-213) feeds into whether `query_vec` gets computed (lines 237-272), which feeds into `final_mode` (lines 275-285), which is what the match at line 303 actually switches on. Today the invariant holds — every branch that lets `final_mode` become `Dense`/`Hybrid` also guarantees `query_vec`/`dense_vectors`/`sparse_indexer` are `Some` — but nothing in the type system enforces this. A future change to any one of those three intermediate match blocks (e.g. adding a new SearchMode variant, or changing the auto-degradation logic) can silently reintroduce a panic reachable from a user-facing `search()` call, which the project convention (and this repo's Resilient-axis review criteria) requires be an error return instead: "No panic!/unwrap()/expect() outside tests — errors are returned as values across the boundary."

Additionally, the current structure allocates unnecessary Option wrapping only to unconditionally unwrap it several branches later — e.g. `dense_vectors` is always `Some(vec)` (via `.unwrap_or_default()`) whenever `final_mode.needs_dense()`, so the Option adds no information and the `.expect()` is pure ceremony hiding the real invariant rather than encoding it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Score & rank section of notectl-search/src/search.rs::search() no longer contains any .expect()/.unwrap() calls outside #[cfg(test)] code
- [ ] #2 The dense/sparse data needed per SearchMode is modeled so the type system guarantees the right data is present for the mode being scored, e.g. an enum ScoreInputs{Dense(...),Sparse(...),Hybrid(...)} built once and matched exhaustively — or an equivalent restructuring
- [ ] #3 Any genuinely-unreachable-but-not-type-proven case returns a SearchError variant instead of panicking, rather than being asserted away with expect()
- [ ] #4 All existing tests in notectl-search/src/search.rs pass unmodified in their observable behavior (same results for the same inputs)
- [ ] #5 nix develop -c cargo test -p notectl-search --all-features passes
- [ ] #6 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Refactor notectl-search/src/search.rs::search() to eliminate six .expect() calls (lines 305, 308, 320, 332, 335, 339) by encoding scoring data requirements in the type system.

## Step 1: Introduce ScoreInputs enum (~line 56, after SearchMode impl block)

Add a private enum that carries exactly the data needed per mode:

```rust
/// Data required to score results in a given search mode.
/// Each variant owns its required inputs — no Option plumbing needed.
enum ScoreInputs {
    Dense { query_vec: Vec<f32>, vectors: Vec<Vec<f32>> },
    Sparse { indexer: SparseIndexer },
    Hybrid { query_vec: Vec<f32>, vectors: Vec<Vec<f32>>, indexer: SparseIndexer },
}
```

Do NOT mark `#[non_exhaustive]` — this is internal-only. `SearchMode` remains `#[non_exhaustive]` for external API stability.

## Step 2: Build ScoreInputs once after all degradation decisions (~line 287)

Replace the current sequence (effective_mode → query_vec → final_mode → dense_vectors → six .expect()s) with a single construction point. Keep ALL existing auto-degradation logic identical — only change how the results are packaged.

Concrete approach: after the `final_mode` determination (~line 285), construct `ScoreInputs` in one match:

```rust
let inputs = match (final_mode, query_vec, dense_vectors, sparse_indexer) {
    // Dense path
    (SearchMode::Dense, Some(qv), Some(vects), _) => {
        ScoreInputs::Dense { query_vec: qv, vectors: vects }
    }
    // Sparse path
    (SearchMode::Sparse, _, _, Some(idx)) => {
        ScoreInputs::Sparse { indexer: idx }
    }
    // Hybrid path
    (SearchMode::Hybrid, Some(qv), Some(vects), Some(idx)) => {
        ScoreInputs::Hybrid { query_vec: qv, vectors: vects, indexer: idx }
    }
    // Structurally impossible combinations → return error instead of panic
    _ => return Err(SearchError::Other(
        format!("Inconsistent search state: mode={:?}, has_query_vec={}, has_vectors={}, has_sparse={}",
            final_mode,
            query_vec.is_some(),
            dense_vectors.is_some(),
            sparse_indexer.is_some(),
        )
    ).into()),  // Note: may need .map_err or direct return depending on SearchResult type
};
```

IMPORTANT: The `dense_vectors` read (~line 290) must retain its `#[cfg(feature = "embeddings")]` / `#[cfg(not(...))]` branches because `notcfg` builds can never produce Dense/Hybrid variants. When `embeddings` is disabled, `final_mode` will always be `Sparse`, so only the Sparse arm is reachable — but the compiler still needs the `#[cfg]` guards on any code that references embedding types.

Handle this by keeping the cfg-gated dense_vectors read before the ScoreInputs construction, OR by folding the read into the construction match itself with inline cfg guards. The former is simpler and clearer.

## Step 3: Rewrite "Score & rank" match (~line 303) to match on ScoreInputs

Replace `match final_mode { ... }` with `match inputs { ... }`:

```rust
let fused: Vec<(usize, f64)> = match inputs {
    ScoreInputs::Dense { query_vec, vectors } => {
        let dense_scores = cosine_top_k(&vectors, &query_vec, options.max_results);
        rrf_fuse(&dense_scores, &[], options.rrf_k, options.rrf_cosine_weight, 0.0)
    }
    ScoreInputs::Sparse { indexer } => {
        let sparse_scores = indexer.score_query(query);
        rrf_fuse(&[], &sparse_scores, options.rrf_k, 0.0, options.rrf_bm25_weight)
    }
    ScoreInputs::Hybrid { query_vec, vectors, indexer } => {
        let dense_scores = cosine_top_k(&vectors, &query_vec, options.max_results * 2);
        let sparse_scores = indexer.score_query(query);
        rrf_fuse(&dense_scores, &sparse_scores, options.rrf_k, options.rrf_cosine_weight, options.rrf_bm25_weight)
    }
};
```

Note: `cosine_top_k` takes `&Vec<Vec<f32>>` (ref) — adjust borrowing as needed since ScoreInputs now owns the data directly. Check the function signature in fusion.rs.

## Step 4: Clean up dead code

After steps 2-3, verify which variables are no longer needed:
- `effective_mode` — may still be needed for the sparse_indexer/query_vec conditional compilation. If fully absorbed into ScoreInputs construction, remove.
- `final_mode` — removed (replaced by ScoreInputs discriminant)
- `query_vec: Option<Vec<f32>>` — removed (data flows through ScoreInputs)
- `dense_vectors: Option<Vec<Vec<f32>>>` — removed (data flows through ScoreInputs)
- `sparse_indexer: Option<SparseIndexer>` — removed (data flows through ScoreInputs)

Check `SearchMode::needs_dense()` and `SearchMode::needs_sparse()` usage:
- Production callers: lines 216, 237, 290 — all eliminated if absorbed into ScoreInputs construction
- Test callers: lines 416-426 — still used in unit tests
- If ONLY tests call them, keep them (tests are valid consumers). Remove only if truly unused everywhere.

## Step 5: Handle cfg(feature = "embeddings") carefully

The `query_vec` computation (~line 237) and `dense_vectors` read (~line 290) both have `#[cfg(feature = "embeddings")]` guards. When constructing ScoreInputs:

- Without `embeddings` feature: `final_mode` is always `Sparse`, so only `ScoreInputs::Sparse` is reachable
- With `embeddings` feature: all three variants are possible
- The catch-all `_ => Err(...)` arm handles any unreachable combination gracefully

Make sure the `#[cfg]` branches compile cleanly in BOTH configurations. Run:
```bash
nix develop -c cargo check -p notectl-search  # without embeddings
nix develop -c cargo check -p notectl-search --features embeddings  # with embeddings
```

## Step 6: Verify behavior preservation

Run the full test suite:
```bash
nix develop -c cargo test -p notectl-search --all-features
```

Every existing test must pass with unchanged assertions. This is a pure internal restructuring — no behavioral change.

## Step 7: Clippy clean

```bash
nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings
```

Fix any warnings, especially around unused variables or dead branches.

## TASK-8 interaction note

If TASK-8 (double vectors.bin read fix) has landed before this ticket, reuse whatever single-read vectors representation it introduced when constructing `ScoreInputs::Dense`/`Hybrid`. If TASK-8 has NOT landed yet (current status: To Do), do NOT try to fix the double-read here — that is out of scope. Just ensure you dont make it worse (call `index.read_vectors()` the same number of times as current code).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation complete. All 6 acceptance criteria met:

1. Zero .expect()/.unwrap() calls in production code (only in tests)
2. ScoreInputs{Dense,Sparse,Hybrid} enum encodes per-mode data requirements in the type system
3. Impossible combinations return SearchError::Other with diagnostic message
4. All existing tests pass unmodified — 98 tests without embeddings, 121 with --all-features
5. cargo test -p notectl-search --all-features passes
6. cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes

Note: Also combined query embedding + vector read into a single dense_data tuple to avoid the separate query_vec/dense_vectors Option plumbing.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Introduced ScoreInputs enum to replace six .expect() calls in search(). The type system now guarantees correct data availability per scoring mode. Impossible states return SearchError instead of panicking. All tests pass (121 with --all-features), clippy clean.
<!-- SECTION:FINAL_SUMMARY:END -->
