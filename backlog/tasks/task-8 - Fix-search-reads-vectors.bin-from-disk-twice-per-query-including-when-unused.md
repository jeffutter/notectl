---
id: TASK-8
title: >-
  Fix: search() reads vectors.bin from disk twice per query, including when
  unused
status: Needs Plan
assignee: []
created_date: '2026-07-16 07:22'
updated_date: '2026-07-16 16:31'
labels:
  - review-followup
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
- [ ] #1 notectl-search/src/search.rs::search() calls index.read_vectors() at most once per invocation
- [ ] #2 When options.mode == SearchMode::Sparse, index.read_vectors() is not called at all (has_vectors/auto-degradation logic is restructured so it only reads vectors when the requested mode could actually use them, i.e. Dense or Hybrid)
- [ ] #3 The single read result is reused for both the has_vectors/auto-degradation check and the actual cosine_top_k scoring, rather than being read once and discarded then read again
- [ ] #4 Existing behavior is unchanged for all modes: auto-degradation from Dense/Hybrid to Sparse when vectors are missing or empty still works, and all existing tests in notectl-search/src/search.rs pass without modification to their assertions
- [ ] #5 nix develop -c cargo test -p notectl-search --all-features passes
- [ ] #6 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust+WebAssembly core (crates/gql-core) with a
TypeScript/React web app (web/). ALL commands must run inside the Nix dev
shell: either run 'direnv allow' once, or prefix every command with
'nix develop -c'. Work from the repository root unless told otherwise. Do not
change pinned dependency versions.

(This repo is notectl; the crate under test is notectl-search. The same Nix-shell rule applies.)

1. Open notectl-search/src/search.rs and read the whole `search()` function (~line 112-382) to see the full current structure of `has_vectors` (~186-196), `effective_mode` (~199-213), and `dense_vectors` (~290-301).

2. Restructure so `index.read_vectors()` (feature-gated behind `#[cfg(feature = "embeddings")]`) is called at most once. Suggested approach:
   - Immediately after Step 2's manifest-empty check, if `options.mode.needs_dense()` (add/reuse a `needs_dense()` check on `options.mode`, NOT a mode derived later), read vectors once into a local `let raw_vectors: Vec<Vec<f32>> = ...` (feature-gated; `Vec::new()` when the `embeddings` feature is off or mode doesn't need dense).
   - Compute `has_vectors` from `raw_vectors` directly (`!raw_vectors.is_empty() && raw_vectors.len() == manifest.chunks.len()`) instead of calling `read_vectors()` again.
   - When `options.mode == SearchMode::Sparse`, skip the read entirely (`raw_vectors` stays empty, `has_vectors` stays `false`, matching today's behavior for that path since `effective_mode` never changes for `Sparse` regardless of `has_vectors`).
   - Downstream, in the "Score & rank" step, reuse `raw_vectors` (renamed appropriately, e.g. as part of the `ScoreInputs` restructuring from the companion ticket if it has already landed — if not, just reuse the same `Vec<Vec<f32>>` local variable) instead of calling `index.read_vectors()` a second time.

3. Preserve existing auto-degradation behavior exactly: if `options.mode` is `Dense`/`Hybrid` but `raw_vectors` ends up empty or mismatched in length, the existing warn-and-degrade-to-Sparse logic (currently at ~lines 199-213) must still trigger.

4. Do not change the public `SearchOptions`/`SearchMode` API — this is an internal restructuring only.

5. Run `nix develop -c cargo test -p notectl-search --all-features` and confirm every existing test still passes without modifying their assertions (in particular `test_search_sparse_only`, `test_auto_degrade_to_sparse_without_embeddings`, and any hybrid/dense-mode tests).

6. As a sanity check that the read count actually dropped, temporarily add a `tracing::debug!` (or use `eprintln!` locally, not committed) inside `SearchIndex::read_vectors` in storage.rs to count invocations during a single `search()` test call, confirm it's now 1 instead of 2 for a Dense/Hybrid-mode search and 0 for a Sparse-mode search, then remove the temporary instrumentation before committing.

7. Run `nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings` and fix any warnings.
<!-- SECTION:PLAN:END -->
