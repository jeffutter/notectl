---
id: TASK-35
title: >-
  Fix: truncate_and_pad takes ownership of Vec<u32>, forcing an unneeded clone
  in inner_embed_text's hot path
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-18 21:56'
updated_date: '2026-07-18 22:25'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-34
priority: high
ordinal: 100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-34 (notectl-search/src/embeddings/model.rs:775, notectl-search/src/embeddings/embed.rs:311-315). TASK-34 deduplicated the truncation+padding logic by extracting truncate_and_pad(token_ids: Vec<u32>, max_len: usize, pad_id: u32) -> Vec<u32> and having inner_embed_text call it. But the function only ever reads from token_ids (it slices token_ids[..actual_len] and extend_from_slice's into a freshly-allocated Vec) -- it never needs ownership of the input buffer. Because the signature takes Vec<u32> by value, the only production caller (inner_embed_text in embed.rs, which holds token_ids as a &[u32] borrowed from encoding.get_ids()) is forced to call token_ids.to_vec() before passing it in (embed.rs:312), allocating and copying the entire token buffer on every single embed call. Before TASK-34's refactor, inner_embed_text built padded_ids directly via extend_from_slice(&token_ids[..actual_len]) with no intermediate clone at all -- so this refactor is a Concise/efficiency-axis regression: it introduced a heap allocation+copy that did not exist in the pre-refactor code, purely to satisfy an overly-strict function signature. Per this repo's Rust guidelines (prefer &[T] over Vec<T> in function parameters when ownership isn't required), truncate_and_pad should take token_ids: &[u32] instead.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 notectl-search/src/embeddings/model.rs::truncate_and_pad takes token_ids: &[u32] instead of Vec<u32>, and its body no longer requires an owned input (it already only reads via slicing and extend_from_slice)
- [x] #2 notectl-search/src/embeddings/embed.rs::inner_embed_text passes token_ids directly (a &[u32]) to truncate_and_pad without calling .to_vec()
- [x] #3 All existing call sites in notectl-search/src/embeddings/model.rs's test module (test_truncate_and_pad_over_length_does_not_panic, test_truncate_and_pad_under_length_pads, and the integration test around line 1041) are updated to pass a slice (e.g. &input or &token_ids) instead of an owned Vec, without changing what each test asserts
- [x] #4 nix develop -c cargo test -p notectl-search --all-features passes
- [x] #5 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust CLI workspace (notectl-core, notectl-outline, notectl-search, notectl-files, notectl-tags, notectl-tasks, notectl-daily-notes, plus the main notectl binary). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-search/src/embeddings/model.rs and locate truncate_and_pad (~line 775):
   pub(crate) fn truncate_and_pad(token_ids: Vec<u32>, max_len: usize, pad_id: u32) -> Vec<u32> {
       let actual_len = token_ids.len().min(max_len);
       let mut padded = Vec::with_capacity(max_len);
       padded.extend_from_slice(&token_ids[..actual_len]);
       padded.extend(std::iter::repeat_n(pad_id, max_len - actual_len));
       padded
   }
   Change the parameter type from token_ids: Vec<u32> to token_ids: &[u32]. The body needs no other changes -- token_ids[..actual_len] and token_ids.len() both work identically on a slice.

2. Update the doc comment directly above (~line 770-774) if it references taking ownership; it currently just says 'Truncate token IDs to max_len and pad with pad_id' plus the shared-helper rationale, so likely no wording change is needed beyond re-reading it for accuracy.

3. Open notectl-search/src/embeddings/embed.rs and locate the call site in inner_embed_text (~line 309-315):
   let pad_id = model.pad_token_id;
   let padded_ids = truncate_and_pad(
       token_ids.to_vec(),
       model.embedding_config.max_seq_len,
       pad_id,
   );
   Change token_ids.to_vec() to token_ids (token_ids is already a &[u32] from encoding.get_ids() at ~line 299, so this now passes the borrow directly with no allocation).

4. Open notectl-search/src/embeddings/model.rs and update the three test call sites:
   - test_truncate_and_pad_over_length_does_not_panic (~line 950-956): change 'let input = vec![1u32; 2049]; let result = truncate_and_pad(input, 2048, 0);' to pass '&input' instead of 'input' (keep the vec! binding so .len() assertions below still have something to reference if needed, or inline as '&vec![1u32; 2049]' -- prefer keeping the named binding and just changing the call to truncate_and_pad(&input, 2048, 0)).
   - test_truncate_and_pad_under_length_pads (~line 959-964): same change, truncate_and_pad(&input, 2048, 50256).
   - The integration test around line 1037-1041 (inside the gated embeddings integration test module): 'let token_ids: Vec<u32> = encoding.get_ids().to_vec(); ... let padded = truncate_and_pad(token_ids, max_len, pad_id);' -- change the final call to truncate_and_pad(&token_ids, max_len, pad_id). Keep the to_vec() here since token_ids is reused/owned locally in this test scope; only the call argument becomes a borrow.

5. Run: nix develop -c cargo build -p notectl-search --all-features -- confirm it compiles with no warnings.

6. Run: nix develop -c cargo test -p notectl-search --all-features -- confirm all tests pass, including test_truncate_and_pad_over_length_does_not_panic and test_truncate_and_pad_under_length_pads.

7. Run: nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings.

8. Run: nix develop -c cargo fmt -p notectl-search -- --check (fix formatting if needed).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Changed truncate_and_pad signature from Vec<u32> to &[u32] in model.rs:775. Removed .to_vec() allocation at call site in embed.rs (inner_embed_text hot path). Updated 3 test call sites to pass slices. All 154 tests pass, clippy clean, fmt clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Changed truncate_and_pad parameter from Vec<u32> to &[u32] in model.rs, eliminating an unnecessary .to_vec() clone in inner_embed_text's hot path (embed.rs). Updated 3 test call sites to pass slices. All 154 tests pass, clippy clean, fmt clean.
<!-- SECTION:FINAL_SUMMARY:END -->
