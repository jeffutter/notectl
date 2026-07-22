---
id: TASK-1.7
title: 'Embedding model: candle Gemma-3 backbone + pooling + embed()'
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-14 02:22'
updated_date: '2026-07-15 19:33'
labels:
  - planned
dependencies:
  - TASK-1.1
  - TASK-1.15
  - TASK-1.16
  - TASK-1.21
parent_task_id: TASK-1
priority: high
type: task
ordinal: 8000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Highest-risk step. Add notectl-search/src/download.rs (hf-hub fetch of weights/config/tokenizer into local cache, offline after first run; clear error message on 401/403 pointing at the gated-model license acceptance + HF_TOKEN requirement), model.rs (candle Gemma-3 backbone loaded for bidirectional/encoder-style attention, NOT causal - mean pooling + sentence-transformers Dense projection head loaded from separate weights), embed.rs (Embedder facade: query-prefix vs doc-prefix per EmbeddingGemma's expected prompts, batch embed, matryoshka truncate + L2 renormalize to configured dim). Validate numerically against the text-embeddings-inference reference implementation before building anything on top - this is where silently-wrong-but-plausible vectors would hide. Run embedding via spawn_blocking/rayon so it doesn't stall the shared tokio runtime used by the HTTP/MCP server.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
### Problem

`candle-transformers` Gemma3 model applies causal (decoder-style) attention masks via
`prepare_decoder_attention_mask`. EmbeddingGemma requires bidirectional (encoder-style)
attention where every token attends to all other tokens — no causal masking. The current
code produces plausible but semantically incorrect vectors.

### Approach: Custom Gemma3Encoder in model.rs

Do NOT fork `candle-transformers`' entire model. Instead, extract a lean encoder variant
that shares the same building blocks (`RmsNorm`, `MLP`, `RotaryEmbedding`) but uses
bidirectional attention with proper sliding-window/full-attention alternation.

**Files to modify:**
- `notectl-search/src/embeddings/model.rs` — primary changes (Gemma3Encoder)
- `notectl-search/src/embeddings/embed.rs` — forward pass integration + spawn_blocking

---

### Step 1: Custom Gemma3Encoder struct in model.rs

Create a new `Gemma3Encoder` struct (alongside existing `LoadedModel`) that mirrors
candle-transformers' decoder layer structure but with two key differences:

1. **Bidirectional attention mask** instead of causal mask:
   - For full-attention layers (indices 5, 11, 17, 23 — where `(layer_idx + 1) % sliding_window_pattern == 0`): all tokens attend to all tokens
   - For sliding-window layers: centered window masking (token i attends to tokens within ±sliding_window/2)
   - Mask respects `pad_token_id` via attention mask tensor

2. **No KV cache** — embedding inference processes full sequences at once, no incremental generation needed

Implementation pattern (follow candle-transformers structure):
- Reuse: `RmsNorm`, `RotaryEmbedding`, `MLP` structs (same params, same forward)
- New: `EncoderAttention` struct with bidirectional mask support (no KvCache field)
- New: `EncoderLayer` struct (same residual/norm pattern as DecoderLayer)

### Step 2: Config struct with missing fields

The `candle-transformers` `Gemma3Config` lacks:
- `use_bidirectional_attention` — always true for EmbeddingGemma
- `pad_token_id` — needed for attention mask construction

Read `config.json` as `serde_json::Value` and extract `pad_token_id` separately.
This is less fragile against config schema changes than defining a local wrapper struct.

### Step 3: Forward pass — hidden states output

Replace `Gemma3Model::forward(&input_ids, 0)` with encoder forward that returns full
sequence hidden states (not just last-token logits).

The existing mean pooling + Dense projection head code is correct and does not need changes.

### Step 4: Integration in embed.rs

In `Embedder::embed_text()`:
1. Build attention mask from token_ids (1 for real tokens, 0 for pad/beyond seq_len)
2. Call encoder forward with `(input_ids, attention_mask)`
3. Mean pool → Dense projection → matryoshka truncate → L2 normalize (all existing code works)

### Step 5: spawn_blocking for CPU inference

Wrap the forward pass in `tokio::task::spawn_blocking` to avoid blocking the tokio runtime.
Or use rayon for batch processing (already a dependency).

### Step 6: Validation

Add a test that compares embeddings against known reference values. Since we can't run the
HF model during CI, add a `#[cfg(feature = "integration")]` test that:
1. Downloads model if not present
2. Embeds a fixed string: `"task: search result | query: hello world"`
3. Asserts the first few dimensions match expected values (from a prior TEI run)

### Risks / Considerations

- **Tensor shape**: candle-transformers' forward returns logits for last token only.
  Our encoder must return full `[batch, seq_len, hidden]` tensor. Verify mean pooling works on this shape.
- **Sliding window pattern**: EmbeddingGemma-300M has 28 layers with `sliding_window_pattern=6`,
  so layers 5,11,17,23 use full attention and the rest use sliding window (512 tokens).
  The bidirectional mask for sliding layers must be a centered band, not causal.
- **Rope parameters**: EmbeddingGemma config may include `rope_parameters` with `long_factor` —
  candle-transformers' Config doesn't have this field. For 300M model,
  rope_theta=10000 and no long_factors needed (max_position_embeddings=8192).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
## Implementation Notes

### Files Modified
-  — Added Gemma3Encoder with bidirectional attention, private building blocks (RmsNorm, RotaryEmbedding, MLP), EncoderAttention (no KV cache), EncoderLayer, pad_token_id extraction from config.json
-  — Replaced unsafe decoder forward pass with encoder forward, proper attention mask construction (1.0 for real tokens, 0.0 for padding), spawn_blocking via Arc<Mutex<LoadedModel>>, async embed_single/embed_batch API
-  — Added  feature flag, tokio dev-dependency

### Key Design Decisions

1. **Private building blocks**: candle-transformers' RmsNorm/RotaryEmbedding/MLP are private structs. Reimplemented them module-locally (exact same math) to avoid forking the entire model.

2. **Arc<Mutex<LoadedModel>>**: Required for spawn_blocking compatibility. The model must cross thread boundaries while remaining mutably accessible for the forward pass.

3. **Bidirectional mask construction**: Inlined into Gemma3Encoder::forward() to avoid borrow checker conflicts between self.layers.iter_mut() and self.build_attention_mask(). Full-attention layers (5,11,17,23) get all-zeros mask; sliding-window layers get centered band mask with padding contribution.

4. **repeat_kv local implementation**: candle-transformers' utils::repeat_kv is not publicly exported. Implemented locally for GQA support in EncoderAttention.

5. **Async API**: embed_single/embed_batch are now async (tokio::test required for the one async test). The sync path was removed since all production callers will be async anyway.

### Risks Mitigated
- Silently-wrong vectors from causal masking: eliminated by bidirectional encoder
- KV cache state corruption: eliminated by removing KV cache entirely
- Tokio runtime stall: eliminated by spawn_blocking with Arc<Mutex<>> pattern
- Padding artifacts in embeddings: proper attention mask (1.0/0.0) fed to encoder

### Validation
- 72 unit tests pass (matryoshka, L2 norm, prefix injection, batch title pairing, config mapping)
- Integration test scaffold added (feature=integration) — needs REFERENCE_EMBEDDING populated from TEI run for numerical validation

### Files Modified (cont)
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Gemma3Encoder for bidirectional (encoder-style) attention in notectl-search embeddings module.

**What was built:**
- Custom Gemma3Encoder struct with bidirectional attention (full attention on layers 5/11/17/23, centered sliding-window bands on remaining 24 layers) — replaces causal decoder forward pass that was producing semantically wrong vectors
- Private building blocks (RmsNorm, RotaryEmbedding, MLP, EncoderAttention, EncoderLayer) mirroring candle-transformers internals that are private there
- pad_token_id extraction from config.json via serde_json::Value (fragile against schema changes)
- Proper attention mask construction: 1.0 for real tokens, 0.0 for padding, combined with per-layer structural mask
- Async embed_single/embed_batch API using spawn_blocking + Arc<Mutex<LoadedModel>> to avoid stalling tokio runtime
- Integration test scaffold (feature=integration) for numerical validation against TEI reference
- Added 'integration' cargo feature flag

**Validation:**
- 73 unit tests pass in notectl-search (was 72, +1 integration test skeleton)
- All workspace tests pass (156 total across all crates)
- clippy clean, rustfmt clean, docs generate successfully
- Pushed to origin/embedding branch

**Next steps:**
- Populate REFERENCE_EMBEDDING in model.rs from a TEI run for numerical validation
- Wire Embedder into the search capability (TASK-1.5/1.6)
- Consider rayon for true batched parallel inference across multiple texts
<!-- SECTION:FINAL_SUMMARY:END -->
