---
id: TASK-31
title: >-
  Fix: integration test get_embedding() lacks token truncation, risks usize
  underflow panic and contradicts its own 'mirrors production' doc comment
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-18 17:33'
updated_date: '2026-07-18 21:30'
labels:
  - review-followup
  - planned
milestone: Active
dependencies:
  - TASK-27
priority: high
type: bug
ordinal: 90
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-27 (notectl-search/src/embeddings/model.rs:854-878, get_embedding() in the integration_tests module). Production's inner_embed_text (notectl-search/src/embeddings/embed.rs:299) computes 'actual_len = token_ids.len().min(model.embedding_config.max_seq_len)' and pads/truncates using 'token_ids[..actual_len]', so oversized input is safely truncated with a warning log. get_embedding()'s test helper never truncates: it does 'let mut padded = token_ids; padded.extend(std::iter::repeat_n(pad_id, max_len - padded.len()))' using the raw, unbounded token_ids. If a test input ever tokenizes to more than max_seq_len (2048) tokens, 'max_len - padded.len()' underflows (usize subtraction) and panics with 'attempt to subtract with overflow' in debug builds instead of gracefully truncating like production. This directly contradicts the doc comment TASK-27 added at model.rs:854 ('Mirrors embed.rs::inner_embed_text (tokenize -> pad -> forward -> mean_pooling -> projection -> normalize_embedding). Keep the two in sync'), which now overpromises: the two diverge on truncation behavior. Correctness/Resilience-axis gap: a latent panic reachable the moment QUERY_TEST_INPUT/DOC_TEST_INPUT (or any future integration test reusing this helper) exceeds 2048 tokens.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 get_embedding() in notectl-search/src/embeddings/model.rs truncates token_ids to max_seq_len before padding, mirroring inner_embed_text's 'actual_len = token_ids.len().min(max_seq_len)' + 'token_ids[..actual_len]' logic (embed.rs:299,314), eliminating the usize underflow risk
- [ ] #2 A new unit test (outside the #[cfg(feature = "integration")] gate, e.g. in a plain #[cfg(test)] mod near get_embedding, or a pure-function extraction of the truncate+pad step) proves that input longer than max_seq_len does not panic and produces a padded vector of exactly max_seq_len tokens
- [ ] #3 nix develop -c cargo test -p notectl-search --features integration passes
- [ ] #4 nix develop -c cargo clippy -p notectl-search --features integration --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Fix usize underflow in integration test get_embedding() when token_ids exceed max_seq_len (2048).

## Changes (single file: notectl-search/src/embeddings/model.rs)

### 1. Fix truncation in get_embedding() (~line 998-999)

Replace the unsafe padding block:
  let mut padded = token_ids;
  padded.extend(std::iter::repeat_n(pad_id, max_len - padded.len()));

With production-mirrored truncation+padding (matching embed.rs inner_embed_text):
  let actual_len = token_ids.len().min(max_len);
  let mut padded = Vec::with_capacity(max_len);
  padded.extend_from_slice(&token_ids[..actual_len]);
  padded.extend(std::iter::repeat_n(pad_id, max_len - actual_len));

This eliminates the "max_len - padded.len()" underflow when token_ids.len() > max_len.

### 2. Extract pure helper for testability

Extract the truncate+pad logic into a standalone pure function near the top of the file (outside the integration_tests module):
  pub(crate) fn truncate_and_pad(token_ids: Vec<u32>, max_len: usize, pad_id: u32) -> Vec<u32> {
      let actual_len = token_ids.len().min(max_len);
      let mut padded = Vec::with_capacity(max_len);
      padded.extend_from_slice(&token_ids[..actual_len]);
      padded.extend(std::iter::repeat_n(pad_id, max_len - actual_len));
      padded
  }

Call this from get_embedding() instead of inline logic. This keeps the integration test behavior correct while enabling a plain unit test.

### 3. Add unit test (plain #[cfg(test)], no integration feature gate)

In the existing #[cfg(test)] mod (not inside integration_tests), add two tests:
  - test_truncate_and_pad_over_length_does_not_panic: vec![1u32; 2049], assert result.len() == 2048
  - test_truncate_and_pad_under_length_pads: vec![1u32; 10], assert result.len() == 2048, first 10 are 1, rest are pad_id

These tests run without the integration feature or downloaded model.

### 4. Quality gates

Run all commands inside nix dev shell:
  cargo test -p notectl-search (new unit tests pass, no integration feature needed)
  cargo test -p notectl-search --features integration -- integration_tests (integration tests still skip gracefully)
  cargo clippy -p notectl-search --features integration --all-targets -- -D warnings
  cargo fmt -p notectl-search -- --check

### Why no sub-tickets

All changes are in a single function (+ pure extraction) in one file. The truncate fix and unit test are tightly coupled — extracting the helper IS how we enable the test. Total change is ~15 lines of code + ~15 lines of tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
## Implementation Notes

Fixed in single commit (582b988). Changes in notectl-search/src/embeddings/model.rs:

1. **Extracted truncate_and_pad()** as pub(crate) pure helper (~line 774) that mirrors inner_embed_text's truncation+padding logic from embed.rs. Eliminates usize underflow risk.

2. **Updated get_embedding()** (~line 1028) to call truncate_and_pad() instead of inline padding. Now safely handles oversized token input.

3. **Added two unit tests** in plain #[cfg(test)] mod (no integration feature needed):
   - test_truncate_and_pad_over_length_does_not_panic: 2049 tokens → 2048 output, no panic
   - test_truncate_and_pad_under_length_pads: 10 tokens → 2048 output with correct pad values

4. **Quality gates passed**: cargo test (154 tests), cargo clippy (-D warnings), cargo fmt --check

All acceptance criteria met.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed usize underflow panic risk in integration test get_embedding() by extracting truncate_and_pad() helper that mirrors production's inner_embed_text truncation+padding logic. Added two unit tests proving over-length input does not panic and under-length input pads correctly. All 154 tests pass, clippy clean, docs build clean. Pushed to embedding branch (dc76188).
<!-- SECTION:FINAL_SUMMARY:END -->
