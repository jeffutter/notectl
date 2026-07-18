---
id: TASK-27
title: >-
  Fix: integration test get_embedding() skips L2-normalization, will fail once
  real model is available
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-18 07:10'
updated_date: '2026-07-18 14:44'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-1.14.2
priority: high
ordinal: 100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-1.14.2 (notectl-search/src/embeddings/model.rs:847-915, the get_embedding() helper added/refactored in commit b161180). Production embedding (inner_embed_text, notectl-search/src/embeddings/embed.rs:279-359) always ends with 'Ok(normalize_embedding(&embedding_f32, output_dim))' (embed.rs:358) — matryoshka truncation + L2 normalization. The integration test's get_embedding() hand-reimplements every step of that pipeline (tokenize, pad, attention mask, forward, mean_pooling, projection) but never calls normalize_embedding, returning the raw projected vector instead. assert_embedding_properties() (model.rs:918-931) then asserts the returned vector's L2 norm is ~1.0 (within 1e-4) — an assertion on raw (non-normalized) projection output that will fail the first time this test is actually run against a downloaded model with HF_TOKEN set. This is a Correctness-axis bug: the gated integration test does not validate the real production embedding path it claims to validate, and would block TASK-1.14.2.1 (populate REFERENCE_EMBEDDING/DOC_REFERENCE_EMBEDDING) from ever capturing correct (normalized) reference values.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 grep -n normalize_embedding notectl-search/src/embeddings/model.rs shows get_embedding() calling normalize_embedding(...) on its output before returning, mirroring inner_embed_text's normalize_embedding call at embed.rs:358
- [ ] #2 A doc comment on get_embedding() notes it must mirror embed.rs::inner_embed_text's steps (tokenize/pad/forward/pool/project/normalize) to prevent future drift between the two implementations
- [ ] #3 nix develop -c cargo test -p notectl-search --features integration passes, and both integration tests still print a graceful skip message and succeed when HF_TOKEN/model is unavailable (no change to skip behavior)
- [ ] #4 nix develop -c cargo clippy -p notectl-search --features integration --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust+WebAssembly core (crates/gql-core) with a TypeScript/React web app (web/). Note: for THIS repo (notectl) it is actually a Rust CLI workspace (notectl-core, notectl-outline, notectl-search, notectl-files, notectl-tags, notectl-tasks, notectl-daily-notes, plus the main notectl binary) — ignore the gql-core/web mention, it does not apply here. ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-search/src/embeddings/embed.rs and read inner_embed_text (lines 279-359). Note it always finishes with 'Ok(normalize_embedding(&embedding_f32, output_dim))' at line 358 — matryoshka truncation + L2 normalization, applied via the pub fn normalize_embedding defined in notectl-search/src/embeddings/model.rs (~line 732).

2. Open notectl-search/src/embeddings/model.rs and locate the integration_tests module (starts ~line 816). Find get_embedding() (~line 847-915). Its final two lines currently read:
   let embedding = projected.squeeze(0).unwrap();
   embedding.to_dtype(DType::F32).unwrap().to_vec1().unwrap()

3. Change those final lines to capture the raw vector, then normalize it exactly like production does, e.g.:
   let raw: Vec<f32> = embedding.to_dtype(DType::F32).unwrap().to_vec1().unwrap();
   normalize_embedding(&raw, 768)
   (normalize_embedding is already in scope via 'use super::*;' at model.rs:818 — no new import needed.)

4. Add a short doc comment directly above get_embedding()'s definition, e.g.: '// Mirrors embed.rs::inner_embed_text (tokenize -> pad -> forward -> mean_pooling -> projection -> normalize_embedding). Keep the two in sync -- this duplication is what let the missing normalize_embedding call go unnoticed.' This is a precision comment (why + drift risk), not a restatement of the code.

5. Run: nix develop -c cargo build -p notectl-search --features integration (verify it compiles without needing network access or a downloaded model).

6. Run: nix develop -c cargo test -p notectl-search --features integration -- integration_tests (confirm both tests still print 'Skipping integration test' and pass, since no HF_TOKEN/model is available in this environment -- the fix must not change skip behavior).

7. Run: nix develop -c cargo clippy -p notectl-search --features integration --all-targets -- -D warnings.

8. Run: nix develop -c cargo fmt -p notectl-search -- --check (fix formatting if needed).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
## Implementation Notes

Fixed two issues in notectl-search/src/embeddings/model.rs:

1. Added `normalize_embedding(&raw, 768)` call at end of `get_embedding()` to match production `inner_embed_text` behavior (embed.rs:358)
2. Added doc comment above `get_embedding()` noting it must mirror embed.rs::inner_embed_text steps to prevent future drift

All quality gates passed:
- cargo build -p notectl-search --features integration ✓
- cargo test -p notectl-search --features integration -- integration_tests ✓ (2 tests pass with graceful skip)
- cargo clippy -p notectl-search --features integration --all-targets -- -D warnings ✓
- cargo fmt --check ✓
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added missing normalize_embedding call to integration test's get_embedding() helper. The test now mirrors production inner_embed_text exactly (tokenize → pad → forward → mean_pooling → projection → normalize_embedding), preventing the L2 norm assertion from failing against real models.
<!-- SECTION:FINAL_SUMMARY:END -->
