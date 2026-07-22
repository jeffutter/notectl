---
id: TASK-1.14.2.1
title: Populate REFERENCE_EMBEDDING and DOC_REFERENCE_EMBEDDING vectors
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-18 05:30'
updated_date: '2026-07-19 14:38'
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
## Implementation Plan (Already Complete)

### Status Verification
All acceptance criteria verified as of 2026-07-19:

- [x] REFERENCE_EMBEDDING populated with real 768-dim f32 values (not zeros)
- [x] DOC_REFERENCE_EMBEDDING populated with real 768-dim f32 values (not zeros)
- [x] REFERENCE_EMBEDDING_POPULATED = true
- [x] DOC_REFERENCE_EMBEDDING_POPULATED = true
- [x] Integration tests assert numerically against reference vectors (not skipping)
- [x] cargo test -p notectl-search --features integration passes both tests (~210s)

### How Values Were Captured
Values were captured using the in-process Candle inference path (test module get_embedding() mirroring production inner_embed_text), NOT TEI container. This matches the approach documented in the original plan — TEI produces different first ~10 dimensions due to internal preprocessing differences.

### Files Modified
- notectl-search/src/embeddings/model.rs — REFERENCE_EMBEDDING (line ~1014, 768 dims), DOC_REFERENCE_EMBEDDING (line ~1794, 768 dims), both _POPULATED flags set to true

### Note on TASK-29
TASK-29 appears to be a duplicate/parallel ticket for the same work. Since this work is complete via TASK-1.14.2.1, TASK-29 should also be marked Done or deprecated.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
BLOCKER: Cannot populate REFERENCE_EMBEDDING/DOC_REFERENCE_EMBEDDING without HF_TOKEN. Model google/embeddinggemma-300m is gated (manual access required). All HF inference API subdomains fail DNS resolution on this network. Docker not running for TEI container option. Need HF_TOKEN set with accepted license before this can proceed.

Attempted execution on 2026-07-18: verified HF_TOKEN is not set, no ~/.huggingface/token file exists, Docker not running (TEI container unavailable). huggingface.co/api resolves but model files return 401 (gated/manual license). Cannot proceed without user providing HF_TOKEN with accepted license for google/embeddinggemma-300m.

Re-attempted 2026-07-18: still blocked by missing HF_TOKEN. No token in env, no ~/.huggingface/token, Docker daemon not running (TEI container unavailable), huggingface.co returns 401 for gated model google/embeddinggemma-300m. Dependencies TASK-27 and TASK-28 are Done. To unblock: set HF_TOKEN with accepted license, then run 'cargo test -p notectl-search --features integration -- integration_tests --nocapture' to capture embedding vectors.

Attempted execution 2026-07-18 (session 3): still blocked by TASK-30 (HF_TOKEN setup). No token in env, no ~/.huggingface/token, no ~/.config/huggingface/token, Docker not running (TEI fallback unavailable), model weights not cached. This requires manual human action: accept EmbeddingGemma license on huggingface.co and set HF_TOKEN. Reverted to Dev Ready per backlog-execute skill instructions for blocked tickets.

Attempted execution 2026-07-18 (session 4): still blocked by TASK-30 (HF_TOKEN setup). No token in env, no ~/.huggingface/token, Docker not running (TEI fallback unavailable), model weights not cached. Requires manual human action: accept EmbeddingGemma license on huggingface.co and set HF_TOKEN. Reverted to To Do per backlog-execute skill instructions for blocked tickets.
<!-- SECTION:NOTES:END -->
