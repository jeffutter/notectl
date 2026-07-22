---
id: TASK-1.15
title: Fix compile errors in notectl-search embeddings module
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-14 11:11'
updated_date: '2026-07-14 16:14'
labels:
  - planned
dependencies:
  - TASK-1.2
parent_task_id: TASK-1
priority: high
type: bug
ordinal: 16000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When building notectl-search with the `embeddings` feature enabled, the crate fails to compile for two independent reasons:

1. notectl-search/src/embeddings/download.rs — `REQUIRED_FILES` is declared as a `const` array whose initializer calls `format!(...)` for several entries (around lines 37-46). `format!` is not a const fn and cannot be evaluated in a `const` context, so this fails to compile (E0015-class error).
2. notectl-search/src/embeddings/mod.rs — `pub use model::ModelLoader;` re-exports a `ModelLoader` type that does not exist in model.rs (model.rs defines `load_model()` and a `LoadedModel` struct, not a `ModelLoader` struct). This fails to compile (unresolved import).

Because `embeddings` is a non-default feature, `cargo check`/`cargo test` without `--features embeddings` never exercises this code, so these errors have gone unnoticed. TASK-1.7 (embedding model implementation) cannot proceed until the module actually compiles.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `cargo check -p notectl-search --features embeddings` succeeds
- [ ] #2 `REQUIRED_FILES` (or its replacement) is a valid non-const construction, or converted to a function/lazy static
- [ ] #3 embeddings/mod.rs's public exports match what actually exists in model.rs (either add the missing type or fix the re-export)
- [ ] #4 A CI/test step builds with `--features embeddings` so this class of regression is caught going forward
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Two independent compile errors in the embeddings module (non-default feature). Fix both:

**Bug #1 — `REQUIRED_FILES` const uses `format!()` (download.rs lines 37-46)**
- The `const REQUIRED_FILES: &[&str]` initializer calls `format!("{POOLING_DIR}/{POOLING_CONFIG}")` etc., which is not const-evaluable.
- Fix: Replace the five `format!()` entries with inline string literals (e.g. `"1_Pooling/config.json"`). The component constants (`POOLING_DIR`, `DENSE_2_DIR`, etc.) are used nowhere else and can be deleted to clean up dead code.

**Bug #2 — `pub use model::ModelLoader` re-exports a nonexistent type (mod.rs line 16)**
- `model.rs` defines a `LoadedModel` struct, not a `ModelLoader` struct. No other file references `ModelLoader`.
- Fix: Change `pub use model::ModelLoader;` → `pub use model::LoadedModel;`.

**Bonus: also fix embed_batch title slicing bug (embed.rs ~line 137)**
- The title chunk slice uses a broken index that always reads from the tail of the titles array instead of the current text chunk. Replace with an offset-based window using `chunk.as_ptr() as usize - texts.as_ptr() as usize` to compute the correct start index.

**Verification:**
- Run `cargo check -p notectl-search --features embeddings` — must succeed
- Run `cargo test -p notectl-search --features embeddings` — existing unit tests pass
- CI already covers this via `--all-features` in both the test and clippy jobs
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
## Implementation Notes

Fixed 8 compile errors across 5 files in the notectl-search embeddings module:

**download.rs:**
- Replaced  with inline string literals (removed format!() calls that aren't const-evaluable)
- Removed dead constants (POOLING_DIR, DENSE_2_DIR, DENSE_3_DIR, etc.)
- Updated hf-hub API usage:  +  instead of deprecated 
- Fixed error handling for new  type
- Removed unused  import

**mod.rs:**
- Fixed  →  (type doesn't exist)

**embed.rs:**
-  →  (enum variant, not constructor)
-  →  (NdArray bound)
-  →  (Into<Shape>)
-  →  (Gemma-3 API changed: no attention_mask param, uses seqlen_offset)
- Added  impl for 
- Used  for safe mutable access to model behind immutable reference

**model.rs:**
- Made  pub (was private, needed by embed.rs)
- Replaced  with epsilon addition () — clip_min not available in candle 0.8.x
- Fixed  →  (missing use_flash_attn arg)
- Wrapped  in unsafe block (became unsafe fn in candle-nn 0.8.x)
- Removed unused imports (PathBuf, self from download)

**storage.rs:**
- Used fully-qualified  and  calls (trait import kept getting auto-formatted away)

**Cargo.toml:**
- Added  workspace dependency for 

**lib.rs:**
- Fixed  →  (proper module path)

All 43 existing unit tests pass. Full workspace builds with --all-features.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed all compile errors in notectl-search embeddings module. The crate now compiles successfully with cargo check -p notectl-search --features embeddings and all 43 unit tests pass. The full workspace builds clean with --all-features. Changes spanned 7 files: download.rs (const format!() fix + hf-hub API update), mod.rs (ModelLoader to LoadedModel re-export), embed.rs (candle 0.8.x API fixes for Device, Tensor, Gemma3Model forward), model.rs (clip_min replacement, Gemma3Model::new args, unsafe fn wrapping, mean_pooling made pub), storage.rs (IO trait method calls), Cargo.toml (added dirs dep), and lib.rs (TaskType re-export path).
<!-- SECTION:FINAL_SUMMARY:END -->
