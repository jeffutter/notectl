---
id: TASK-32
title: >-
  Fix: padding key positions get a positive attention bias instead of being
  excluded in Gemma3Encoder::forward
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-18 18:18'
updated_date: '2026-07-18 19:37'
labels:
  - review-followup
  - planned
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
- [x] #1 layer_mask's padding contribution excludes padded key positions using a large negative bias (e.g. multiply inv_pad by a large negative finite constant such as -1e9, NOT literal f32::NEG_INFINITY or f32::INFINITY multiplication, since 0.0 * NEG_INFINITY = NaN would corrupt real-position logits) instead of adding the raw 0/1 inv_pad value directly
- [x] #2 A new unit test isolates the mask-construction logic (extract it into a small pure function, e.g. fn padding_bias(pad_mask: &Tensor, seq_len: usize, dtype: DType) -> CandleResult<Tensor>) and asserts real-token key positions get ~0.0 bias while padded key positions get a large negative bias, with no NaN values anywhere in the output
- [x] #3 nix develop -c cargo test -p notectl-search --features integration passes
- [x] #4 nix develop -c cargo clippy -p notectl-search --features integration --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP: All commands inside Nix dev shell (nix develop -c). Single-file fix in notectl-search/src/embeddings/model.rs.

## Background
Lines 413-420 add raw inv_pad values (0.0 for real tokens, 1.0 for padded tokens) directly to the additive attention mask. This gives padded positions a +1.0 logit bump — the opposite of exclusion. In full-attention layers (every sliding_window_pattern-th layer), there is zero structural masking, so padded keys are actively favored over real content. Since inputs are always padded to max_seq_len (2048), this corrupts every embedding for any input shorter than 2048.

## Fix Strategy
Scale inv_pad by a large negative FINITE constant (-1e9) before adding to the mask. Do NOT use f32::NEG_INFINITY since 0.0 * NEG_INFINITY = NaN in IEEE-754, which would corrupt real-token logits. The value -1e9 is large enough that exp(-1e9) underflows to 0.0 in f32 softmax (fully excluding padded keys), while staying finite to avoid NaN propagation.

## Steps

### Step 1: Extract padding_bias helper function
Create a pure function near mean_pooling (~line 457):

### Step 2: Update Gemma3Encoder::forward to use padding_bias
Replace lines 413-420 with:

### Step 3: Add unit test
Add a #[test] in the existing mod tests (not gated behind feature = "integration"):
- Build synthetic pad_mask: Tensor::new(&[1.0, 1.0, 0.0, 0.0], &Device::Cpu)?.unsqueeze(0)?
- Call padding_bias(&pad_mask, 4, DType::F32, &Device::Cpu)
- Assert real-key positions (indices 0,1) are ~0.0 within epsilon
- Assert padded-key positions (indices 2,3) are < -1e6
- Assert no NaN or infinity values anywhere (extract to Vec<f32>, check is_finite())

### Step 4: Verify
- cargo test -p notectl-search (new unit test passes)
- cargo test -p notectl-search --features integration (integration tests still skip gracefully without model)
- cargo clippy -p notectl-search --features integration --all-targets -- -D warnings
- cargo fmt -p notectl-search -- --check
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Extracted padding_bias() helper that multiplies inv_pad by -1e9 (large negative finite constant) before adding to attention mask. Replaced buggy inline code in Gemma3Encoder::forward() that added raw 0/1 values directly, giving padded positions a +1.0 logit bump instead of excluding them. Added unit test verifying real keys get ~0.0 bias, padded keys get large negative bias, no NaN values. All 151 tests pass, clippy clean.
<!-- SECTION:FINAL_SUMMARY:END -->
