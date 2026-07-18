---
id: TASK-34
title: >-
  Fix: inner_embed_text duplicates truncate_and_pad's truncation logic instead
  of calling it
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-18 21:41'
updated_date: '2026-07-18 21:52'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-31
priority: high
type: chore
ordinal: 100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-31/TASK-33 (notectl-search/src/embeddings/embed.rs:310-315 vs notectl-search/src/embeddings/model.rs:775-781). TASK-31 extracted truncate_and_pad() as a pub(crate) pure helper specifically so the test-only get_embedding() integration helper could mirror production's truncation+padding logic without risking a usize underflow. But inner_embed_text (embed.rs) was left with its own hand-inlined copy of the exact same actual_len/extend_from_slice/repeat_n sequence instead of being refactored to call the new shared helper. This is a Concise/Organized-axis gap: the same truncation-policy knowledge now exists in two places, which is precisely the failure mode that produced TASK-27 (missing normalize_embedding call), TASK-31 (missing truncation), and TASK-33 (all-ones mask instead of pad_tensor) in the first place -- get_embedding()'s doc comment even says 'Mirrors embed.rs::inner_embed_text ... Keep the two in sync' as a manual, unenforced convention. Calling the shared helper from both sites removes the possibility of the two drifting again.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 notectl-search/src/embeddings/embed.rs::inner_embed_text calls super::model::truncate_and_pad(token_ids, max_len, pad_id) instead of its own inline actual_len/extend_from_slice/repeat_n block
- [ ] #2 The #[allow(dead_code)] attribute on truncate_and_pad in model.rs is removed since it is now genuinely used by production code
- [ ] #3 The tracing::warn! truncation-length warning currently logged in inner_embed_text before calling the padding logic is preserved (moved above the truncate_and_pad call, still comparing token_ids.len() against max_seq_len)
- [ ] #4 nix develop -c cargo test -p notectl-search --all-features passes
- [ ] #5 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust CLI workspace (notectl-core, notectl-outline, notectl-search, notectl-files, notectl-tags, notectl-tasks, notectl-daily-notes, plus the main notectl binary). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-search/src/embeddings/embed.rs and locate inner_embed_text (~lines 287-321). Note the existing tracing::warn! block (~lines 301-307) that logs when token_ids.len() > model.embedding_config.max_seq_len -- this must be kept, it happens before the padding logic.

2. Replace the inline padding block (~lines 313-315):
   let mut padded_ids = Vec::with_capacity(max_len);
   padded_ids.extend_from_slice(&token_ids[..actual_len]);
   padded_ids.extend(std::iter::repeat_n(pad_id, max_len - actual_len));
   with a single call to the shared helper:
   let padded_ids = super::model::truncate_and_pad(token_ids, max_len, pad_id);
   Note token_ids is currently borrowed as &[u32] via encoding.get_ids() (~line 298) and actual_len is computed from it (~line 299) purely for the warning check -- keep that computation (or inline the comparison directly into the warn condition) but pass an owned Vec<u32> (encoding.get_ids().to_vec()) into truncate_and_pad since it takes ownership. Remove the now-unused actual_len variable if nothing else references it after this change (the warning condition can use token_ids.len() directly).

3. Add super::model::truncate_and_pad to the existing 'use super::model::{...}' import list (~line 19-21) instead of qualifying it inline, matching the style of the existing normalize_embedding import.

4. Open notectl-search/src/embeddings/model.rs and locate the truncate_and_pad definition (~line 775, has #[allow(dead_code)] on the line directly above it). Remove the #[allow(dead_code)] attribute -- it is now used by production code (embed.rs) as well as tests, so the lint no longer fires.

5. Run: nix develop -c cargo build -p notectl-search --all-features (confirm no dead_code warning reappears and no compile errors).

6. Run: nix develop -c cargo test -p notectl-search --all-features (confirm all 154+ existing tests still pass, including test_truncate_and_pad_over_length_does_not_panic, test_truncate_and_pad_under_length_pads, and the embeddings integration tests).

7. Run: nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings.

8. Run: nix develop -c cargo fmt -p notectl-search -- --check (fix formatting if needed).

9. In the task's Implementation Notes, confirm the tracing::warn! for oversized input is still reachable from inner_embed_text's production path (manually trace the call, no need for a new test).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Refactored inner_embed_text in embed.rs to call super::model::truncate_and_pad() instead of hand-inlined actual_len/extend_from_slice/repeat_n block. Removed #[allow(dead_code)] from truncate_and_pad since it is now used by production code. Updated doc comment to reflect shared usage. Preserved tracing::warn! for oversized input above the truncate_and_pad call. Kept pad_id variable for attention mask computation. All 154 tests pass, clippy clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Deduplicated truncation+padding logic by replacing inline code in inner_embed_text with call to shared truncate_and_pad helper. Removed #[allow(dead_code)] since helper is now used by production code.
<!-- SECTION:FINAL_SUMMARY:END -->
