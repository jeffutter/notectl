---
id: TASK-33
title: >-
  Fix: mean_pooling is called with an all-ones mask, never excluding padding
  tokens from the pooled average
status: To Do
assignee: []
created_date: '2026-07-18 18:18'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-27
priority: high
ordinal: 55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-27 (notectl-search/src/embeddings/model.rs:915, and the production path it mirrors at notectl-search/src/embeddings/embed.rs:341). mean_pooling's own doc comment (model.rs:458) states its attention_mask parameter must be '(batch, seq_len) - 1 for real tokens, 0 for padding' so the average is computed only over real-token positions. Both call sites instead construct a fresh all-ones mask and pass that: embed.rs:341 'let pooling_mask = Tensor::ones(input_ids.shape().clone(), DType::F32, &model.device)...' and model.rs:915 'let pooling_mask = Tensor::ones(input_ids.shape().clone(), DType::F32, &device).unwrap();' — even though the real per-token attention mask (pad_tensor, correctly built two lines earlier at embed.rs:329 and model.rs:889, with 0.0 for padding) is already in scope and unused for this call. Since inputs are always padded out to max_seq_len (2048) regardless of actual content length, this means every embedding is mean-pooled over up to 2048 positions when only a handful are real tokens, drowning the real content's contribution under ~2000+ padding-position hidden states. This is a Correctness-axis bug affecting every embedding the encoder produces, independent of and in addition to the attention-masking bug filed as TASK-32. It also flows into TASK-29's plan to capture REFERENCE_EMBEDDING/DOC_REFERENCE_EMBEDDING from this exact code path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 embed.rs::inner_embed_text passes the existing pad_tensor (not a freshly-constructed all-ones tensor) as the attention_mask argument to mean_pooling
- [ ] #2 model.rs::integration_tests::get_embedding passes the existing pad_tensor (not a freshly-constructed all-ones tensor) as the attention_mask argument to mean_pooling, keeping it mirrored with inner_embed_text per the doc comment TASK-27 added
- [ ] #3 A new unit test calls mean_pooling directly with a synthetic hidden_states tensor and a pad_mask containing both real (1.0) and padding (0.0) positions, and asserts the result equals the mean of only the real-token positions (i.e. padding positions do not affect the output)
- [ ] #4 nix develop -c cargo test -p notectl-search --features integration passes
- [ ] #5 nix develop -c cargo clippy -p notectl-search --features integration --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust CLI workspace (notectl-core, notectl-outline, notectl-search, notectl-files, notectl-tags, notectl-tasks, notectl-daily-notes, plus the main notectl binary). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-search/src/embeddings/embed.rs and locate inner_embed_text (lines 287-367). Find the pad_tensor construction (around line 329-332):
   let pad_tensor = Tensor::new(attention_mask.as_slice(), &model.device)?.unsqueeze(0)?;
   Then find the pooling_mask construction a few lines later (around line 341):
   let pooling_mask = Tensor::ones(input_ids.shape().clone(), DType::F32, &model.device).map_err(...)?;
   let pooled = super::model::mean_pooling(&hidden_states, &pooling_mask)...;
   Replace the pooling_mask line: delete the Tensor::ones(...) construction and pass pad_tensor directly to mean_pooling instead: 'let pooled = super::model::mean_pooling(&hidden_states, &pad_tensor).map_err(...)?;' Remove the now-unused pooling_mask variable and its construction entirely.

2. Open notectl-search/src/embeddings/model.rs and locate get_embedding() in the integration_tests module (around lines 856-925). Find the equivalent pad_tensor construction (around line 889-892) and the pooling_mask construction (around line 915):
   let pooling_mask = Tensor::ones(input_ids.shape().clone(), DType::F32, &device).unwrap();
   let pooled = mean_pooling(&hidden_states, &pooling_mask).expect(...);
   Apply the same fix: delete the Tensor::ones(...) pooling_mask and pass pad_tensor directly to mean_pooling.

3. Double check there is no remaining use of the deleted pooling_mask variables in either function (cargo build will catch unused-variable warnings if something is missed).

4. Add a unit test near mean_pooling's existing tests in model.rs's #[cfg(test)] mod tests (plain, not gated behind feature = "integration") that:
   - Builds a small synthetic hidden_states Tensor of shape (1, 4, 2) (batch=1, seq_len=4, hidden_dim=2) with distinct, known values per position, e.g. positions 0,1 = real tokens with values [1.0, 1.0] and [3.0, 3.0], positions 2,3 = padding with values [100.0, 100.0] and [200.0, 200.0] (deliberately large/outlier values so any leakage is obvious).
   - Builds a pad_mask Tensor of shape (1, 4) with values [1.0, 1.0, 0.0, 0.0].
   - Calls mean_pooling(&hidden_states, &pad_mask) and asserts the result equals the mean of only the two real positions: [2.0, 2.0] (i.e. (1+3)/2), NOT influenced by the 100.0/200.0 padding values.
   - Also assert that if an all-ones mask were used instead (for contrast/documentation, or just as a second assertion in the same test) the result would differ substantially — this documents why the fix matters, but the core assertion is the pad_mask-respecting result above.

5. Run: nix develop -c cargo test -p notectl-search (confirm the new mean_pooling test passes without needing the integration feature or a downloaded model).

6. Run: nix develop -c cargo test -p notectl-search --features integration -- integration_tests (confirm both integration tests still gracefully skip when the model isn't downloaded, and don't panic or change behavior otherwise).

7. Run: nix develop -c cargo clippy -p notectl-search --features integration --all-targets -- -D warnings.

8. Run: nix develop -c cargo fmt -p notectl-search -- --check (fix formatting if needed).

9. In the task's Implementation Notes, note that this fix is independent of and complementary to TASK-32 (which fixes attention-level padding exclusion) — both are needed for padding to be fully excluded from the final embedding: TASK-32 ensures padded positions don't distort what real tokens attend to, this ticket ensures padded positions themselves don't get averaged into the final pooled vector.
<!-- SECTION:PLAN:END -->
