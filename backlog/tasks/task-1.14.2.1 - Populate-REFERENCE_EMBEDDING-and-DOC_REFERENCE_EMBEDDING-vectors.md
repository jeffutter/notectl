---
id: TASK-1.14.2.1
title: Populate REFERENCE_EMBEDDING and DOC_REFERENCE_EMBEDDING vectors
status: To Do
assignee:
  - '@ralph'
created_date: '2026-07-18 05:30'
updated_date: '2026-07-18 19:23'
labels:
  - planned
dependencies:
  - TASK-27
  - TASK-28
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

### Step 1: Capture Query Embedding
Run the existing integration test with temporary debug output to print the full embedding vector:

```bash
# Add eprintln!("Query embedding: {:?}", &embedding[..50]) before the assertion in test_encoder_produces_correct_dimension
cargo test -p notectl-search --features integration -- integration_tests::test_encoder_produces_correct_dimension
```

Copy first ~50 f32 dimensions from output into REFERENCE_EMBEDDING constant.

### Step 2: Capture Document Embedding
Similarly add debug output to test_document_embedding_matches_reference and run:

```bash
cargo test -p notectl-search --features integration -- integration_tests::test_document_embedding_matches_reference
```

Copy first ~50 f32 dimensions into DOC_REFERENCE_EMBEDDING constant.

### Step 3: Remove Debug Output and Verify
Remove temporary eprintln! calls. Run both tests to confirm they now assert against reference vectors (not skipping numerical check):

```bash
cargo test -p notectl-search --features integration -- integration_tests
```

Both should pass with numerical assertions active (no "REFERENCE_EMBEDDING not populated" warning).

### Requirements
- HF_TOKEN env var must be set with accepted license for google/embeddinggemma-300m
- Model weights (~600MB) will be downloaded on first run via hf-hub cache
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
BLOCKER: Cannot populate REFERENCE_EMBEDDING/DOC_REFERENCE_EMBEDDING without HF_TOKEN. Model google/embeddinggemma-300m is gated (manual access required). All HF inference API subdomains fail DNS resolution on this network. Docker not running for TEI container option. Need HF_TOKEN set with accepted license before this can proceed.

Attempted execution on 2026-07-18: verified HF_TOKEN is not set, no ~/.huggingface/token file exists, Docker not running (TEI container unavailable). huggingface.co/api resolves but model files return 401 (gated/manual license). Cannot proceed without user providing HF_TOKEN with accepted license for google/embeddinggemma-300m.

Re-attempted 2026-07-18: still blocked by missing HF_TOKEN. No token in env, no ~/.huggingface/token, Docker daemon not running (TEI container unavailable), huggingface.co returns 401 for gated model google/embeddinggemma-300m. Dependencies TASK-27 and TASK-28 are Done. To unblock: set HF_TOKEN with accepted license, then run 'cargo test -p notectl-search --features integration -- integration_tests --nocapture' to capture embedding vectors.
<!-- SECTION:NOTES:END -->
