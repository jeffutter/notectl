---
id: TASK-32
title: >-
  Fix: padding key positions get a positive attention bias instead of being
  excluded in Gemma3Encoder::forward
status: To Do
assignee: []
created_date: '2026-07-18 18:18'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-27
priority: high
ordinal: 50
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-27 (notectl-search/src/embeddings/model.rs:409-420, Gemma3Encoder::forward). The per-layer additive attention mask combines a structural component (0.0 = attend, f32::NEG_INFINITY = blocked, used correctly for sliding-window exclusion at lines 392-407) with a padding component built at lines 413-420: 'let inv_pad = (&one - &pad_f32)?...; layer_mask = layer_mask.broadcast_add(&pad_contrib)?;'. inv_pad is 0.0 for real key positions and 1.0 for padded key positions, and this raw 0/1 value is added directly to the logits — it does NOT reuse the NEG_INFINITY convention used two lines above for structural masking. The result: padded key positions get a small POSITIVE +1.0 logit bump relative to real key positions (which get +0.0), the opposite of exclusion. Since inputs are always padded out to max_seq_len (2048, see embed.rs:310 and model.rs:875) regardless of actual content length, and 'full attention' layers (every sliding_window_pattern-th layer, is_full=true at line 391) attend across all 2048 positions with zero structural blocking, this means padding tokens are never excluded from self-attention and are mildly favored over real content. This is a Correctness-axis bug that corrupts every embedding produced by the encoder for any input shorter than max_seq_len (i.e. virtually all real usage), and it flows straight into TASK-29's plan to capture REFERENCE_EMBEDDING/DOC_REFERENCE_EMBEDDING from this exact forward pass — locking in wrong output as 'ground truth' if not fixed first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 layer_mask's padding contribution excludes padded key positions using a large negative bias (e.g. multiply inv_pad by a large negative finite constant such as -1e9, NOT literal f32::NEG_INFINITY or f32::INFINITY multiplication, since 0.0 * NEG_INFINITY = NaN would corrupt real-position logits) instead of adding the raw 0/1 inv_pad value directly
- [ ] #2 A new unit test isolates the mask-construction logic (extract it into a small pure function, e.g. fn padding_bias(pad_mask: &Tensor, seq_len: usize, dtype: DType) -> CandleResult<Tensor>) and asserts real-token key positions get ~0.0 bias while padded key positions get a large negative bias, with no NaN values anywhere in the output
- [ ] #3 nix develop -c cargo test -p notectl-search --features integration passes
- [ ] #4 nix develop -c cargo clippy -p notectl-search --features integration --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust CLI workspace (notectl-core, notectl-outline, notectl-search, notectl-files, notectl-tags, notectl-tasks, notectl-daily-notes, plus the main notectl binary). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-search/src/embeddings/model.rs and read Gemma3Encoder::forward (around lines 373-428), focusing on the per-layer mask construction loop (lines 389-423). Note the structural mask convention established at lines 392-407: 0.0_f32 means 'attend', f32::NEG_INFINITY means 'blocked' — this is the additive-logit-mask convention the rest of the function must follow.

2. Locate the padding contribution block (around lines 413-420):
   if let Some(pad_mask) = attention_mask {
       let pad_f32 = pad_mask.to_dtype(DType::F32)?;
       let one = Tensor::new(1.0f32, pad_mask.device())?;
       let inv_pad = (&one - &pad_f32)?.to_dtype(self.dtype)?;
       let inv_pad_4d = inv_pad.unsqueeze(1)?.unsqueeze(1)?;
       let pad_contrib = inv_pad_4d.broadcast_as((_b_size, 1, seq_len, seq_len))?;
       layer_mask = layer_mask.broadcast_add(&pad_contrib)?;
   }
   inv_pad is 0.0 for real positions, 1.0 for padded positions — this is added RAW, giving padded keys a +1.0 logit bump instead of being excluded.

3. Fix the magnitude/sign: scale inv_pad by a large negative FINITE constant before adding, so padded keys get pushed to ~0 softmax weight without producing NaN. Do NOT multiply by f32::NEG_INFINITY or f32::INFINITY — since inv_pad contains literal 0.0 values for real positions, 0.0 * NEG_INFINITY = NaN in IEEE-754, which would corrupt every real-token logit. Use a large finite negative value instead, e.g.:
   const PADDING_MASK_BIAS: f32 = -1e9;
   ...
   let pad_contrib = (inv_pad_4d.broadcast_as((_b_size, 1, seq_len, seq_len))? * PADDING_MASK_BIAS as f64)?;
   layer_mask = layer_mask.broadcast_add(&pad_contrib)?;
   (adjust exact candle Tensor arithmetic API calls as needed — Tensor supports scalar multiplication via the Mul trait with a scalar reference or ; use whichever is idiomatic given the surrounding code's existing patterns in this file.)

4. Extract the mask-construction logic (steps computing inv_pad -> scaled pad_contrib) into a small pure, independently testable function near mean_pooling (e.g. just above or below it in model.rs), something like:
   fn padding_bias(pad_mask: &Tensor, seq_len: usize, dtype: DType) -> CandleResult<Tensor>
   that takes the (batch, seq_len) pad_mask and returns the (batch, 1, seq_len, seq_len) additive bias tensor. Update Gemma3Encoder::forward's per-layer loop to call this function instead of inlining the logic.

5. Add a #[cfg(test)] unit test (plain, not gated behind feature = "integration", since it only needs a small synthetic pad_mask tensor, no downloaded model) that:
   - Builds a small pad_mask, e.g. Tensor::new(&[1.0f32, 1.0, 0.0, 0.0], &Device::Cpu)?.unsqueeze(0)? for a batch of 1, seq_len 4 (2 real tokens, 2 padding).
   - Calls padding_bias(&pad_mask, 4, DType::F32).
   - Asserts positions corresponding to real keys (indices 0,1 in the last dim) are ~0.0 (within a small epsilon) for every query row, and positions corresponding to padded keys (indices 2,3) are a large negative value (e.g. < -1e6) for every query row.
   - Asserts no NaN or positive-infinity values appear anywhere in the output tensor (extract to Vec<f32> and check with is_finite() and is_nan()).

6. Run: nix develop -c cargo test -p notectl-search --features integration -- integration_tests (confirm both integration tests still gracefully skip when the model isn't downloaded, and don't panic).

7. Run: nix develop -c cargo test -p notectl-search (confirm the new padding_bias unit test passes without needing the integration feature or a downloaded model).

8. Run: nix develop -c cargo clippy -p notectl-search --features integration --all-targets -- -D warnings.

9. Run: nix develop -c cargo fmt -p notectl-search -- --check (fix formatting if needed).

10. In the task's Implementation Notes, record the exact bias magnitude chosen and why (e.g. -1e9 is large enough that exp(-1e9) underflows to 0.0 in f32 softmax, fully excluding padded keys, while staying finite to avoid NaN propagation through the 0.0 * bias multiplication for real positions).
<!-- SECTION:PLAN:END -->
