---
id: TASK-1.14.2.1
title: Populate REFERENCE_EMBEDDING and DOC_REFERENCE_EMBEDDING vectors
status: To Do
assignee:
  - '@ralph'
created_date: '2026-07-18 05:30'
updated_date: '2026-07-18 21:19'
labels:
  - planned
dependencies:
  - TASK-27
  - TASK-28
  - TASK-30
parent_task_id: TASK-1.14.2
ordinal: 27000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The integration test infrastructure in `notectl-search/src/embeddings/model.rs` is complete and both tests skip gracefully when the model is unavailable. However, the reference embedding vectors are still stubs (all zeros).

### What's needed
Run the encoder model on the two test inputs and capture the output vectors:

**Query input:** `"task: search result | query: hello world"`
**Document input:** `"title: My Note | text: hello world"`

Paste first ~50 dimensions into `REFERENCE_EMBEDDING` and `DOC_REFERENCE_EMBEDDING` constants respectively.

### How to capture
Option A: Run the test with debug output added to print the vector, then paste values.
Option B: Use TEI container directly.

Requires: `HF_TOKEN` environment variable set with accepted license for `google/embeddinggemma-300m`.

### Acceptance Criteria
- [ ] `REFERENCE_EMBEDDING` populated with real f32 values (~50 dims minimum)
- [ ] `DOC_REFERENCE_EMBEDDING` populated with real f32 values (~50 dims minimum)  
- [ ] Both integration tests assert against reference vectors (not skipping numerical check)
- [ ] `cargo test -p notectl-search --features integration` passes with model available
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Single file edit: notectl-search/src/embeddings/model.rs

### Goal
Replace zero-stub constants REFERENCE_EMBEDDING and DOC_REFERENCE_EMBEDDING with real f32 values from Google EmbeddingGemma-300M, enabling numerical validation in integration tests.

### Approach: In-process capture via get_embedding() (NOT TEI)
Use the Candle inference path directly — the test module's get_embedding() function mirrors production (embed.rs::inner_embed_text). Research confirms TEI produces *different* vectors from the same model weights (first ~10 dims diverge significantly), so capturing via TEI would produce references that fail the 1e-4 tolerance against our own implementation.

### Step 1: Capture Query Embedding
Add temporary debug output to print the embedding vector, then run the test:

```rust
// Temporarily add before the assertion in test_encoder_produces_correct_dimension:
eprintln!("Query embedding (first 50): {:?}", &embedding[..50]);
```

```bash
HF_TOKEN=<token> cargo test -p notectl-search --features integration \
    -- integration_tests::test_encoder_produces_correct_dimension --nocapture
```

Copy first ~50 f32 dimensions into REFERENCE_EMBEDDING constant.

### Step 2: Capture Document Embedding
Similarly add debug output to test_document_embedding_matches_reference:

```rust
eprintln!("Doc embedding (first 50): {:?}", &embedding[..50]);
```

```bash
HF_TOKEN=<token> cargo test -p notectl-search --features integration \
    -- integration_tests::test_document_embedding_matches_reference --nocapture
```

Copy first ~50 f32 dimensions into DOC_REFERENCE_EMBEDDING constant.

### Step 3: Update Constants and Flags
In model.rs, replace the zero stubs with captured values and flip both boolean sentinels:

```rust
const REFERENCE_EMBEDDING: &[f32] = &[/* captured values */];
const REFERENCE_EMBEDDING_POPULATED: bool = true;
const DOC_REFERENCE_EMBEDDING: &[f32] = &[/* captured values */];
const DOC_REFERENCE_EMBEDDING_POPULATED: bool = true;
```

Remove temporary eprintln! calls.

### Step 4: Verify
Run both integration tests — they should now assert numerically (not skip):

```bash
cargo test -p notectl-search --features integration -- integration_tests
```

Both must pass with numerical assertions active (no "not populated" warning messages).

### Requirements
- HF_TOKEN env var with accepted license for google/embeddinggemma-300m (see TASK-30)
- Model weights (~600MB) downloaded on first run via hf-hub cache
- CPU inference takes several minutes per embedding
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
BLOCKER: Cannot populate REFERENCE_EMBEDDING/DOC_REFERENCE_EMBEDDING without HF_TOKEN. Model google/embeddinggemma-300m is gated (manual access required). All HF inference API subdomains fail DNS resolution on this network. Docker not running for TEI container option. Need HF_TOKEN set with accepted license before this can proceed.

Attempted execution on 2026-07-18: verified HF_TOKEN is not set, no ~/.huggingface/token file exists, Docker not running (TEI container unavailable). huggingface.co/api resolves but model files return 401 (gated/manual license). Cannot proceed without user providing HF_TOKEN with accepted license for google/embeddinggemma-300m.

Re-attempted 2026-07-18: still blocked by missing HF_TOKEN. No token in env, no ~/.huggingface/token, Docker daemon not running (TEI container unavailable), huggingface.co returns 401 for gated model google/embeddinggemma-300m. Dependencies TASK-27 and TASK-28 are Done. To unblock: set HF_TOKEN with accepted license, then run 'cargo test -p notectl-search --features integration -- integration_tests --nocapture' to capture embedding vectors.

Attempted execution 2026-07-18 (session 3): still blocked by TASK-30 (HF_TOKEN setup). No token in env, no ~/.huggingface/token, no ~/.config/huggingface/token, Docker not running (TEI fallback unavailable), model weights not cached. This requires manual human action: accept EmbeddingGemma license on huggingface.co and set HF_TOKEN. Reverted to Dev Ready per backlog-execute skill instructions for blocked tickets.

Attempted execution 2026-07-18 (session 4): still blocked by TASK-30 (HF_TOKEN setup). No token in env, no ~/.huggingface/token, Docker not running (TEI fallback unavailable), model weights not cached. Requires manual human action: accept EmbeddingGemma license on huggingface.co and set HF_TOKEN. Reverted to To Do per backlog-execute skill instructions for blocked tickets.
<!-- SECTION:NOTES:END -->
