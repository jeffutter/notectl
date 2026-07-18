---
id: TASK-1.14.2
title: 'Integration test: populate REFERENCE_EMBEDDING from TEI run'
status: To Do
assignee:
  - '@ralph'
created_date: '2026-07-17 00:30'
updated_date: '2026-07-18 05:31'
labels:
  - planned
dependencies: []
parent_task_id: TASK-1.14
priority: medium
type: task
ordinal: 14200
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the gated integration test in `notectl-search/src/embeddings/model.rs` by populating `REFERENCE_EMBEDDING` with actual values and adding a document-text test case.

### What exists
The test scaffold (`#[cfg(all(test, feature = "integration"))]`) is structurally complete:
- Downloads model if needed (hf-hub)
- Tokenizes input, runs encoder forward pass
- Mean pools, projects through Dense head
- Validates shape (768-dim), L2 norm (~1.0)
- Compares against `REFERENCE_EMBEDDING` within `1e-4` tolerance per dimension
- Gracefully skips when model not downloaded

### What's needed

**1. Populate `REFERENCE_EMBEDDING`**

Run the encoder on `"task: search result | query: hello world"` and capture the output vector. Options:

```bash
# Option A: Use this test itself — it prints the embedding when you add debug output
HF_TOKEN=<token> cargo test -p notectl-search --features integration \
    -- integration_tests::test_encoder_produces_correct_dimension

# Option B: Via TEI container
docker run --rm -v $PWD/output:/output \
    ghcr.io/huggingface/text-embeddings-inference \
    --model-id google/embeddinggemma-300m
```

Paste first ~50 dimensions into `REFERENCE_EMBEDDING` (full 768 optional; test checks as many as provided).

**2. Add document-text test case**

Add a second test validating `RetrievalDocument` prefix path:
```rust
const DOC_REFERENCE_EMBEDDING: &[f32] = &[...];
const DOC_TEST_INPUT: &str = "title: My Note | text: hello world";
```

**3. CI behavior**

Test must skip gracefully (not fail) when `HF_TOKEN` unset or model not cached. Current implementation already does this via `download::is_model_ready()` check.

### Acceptance Criteria
- [ ] `cargo test -p notectl-search --features integration` passes when model available
- [ ] Test gracefully skips (no error) when model unavailable
- [ ] Both query-text and document-text reference vectors populated
- [ ] Numerical assertions use ≤1e-4 tolerance per dimension
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Implementation Plan

### Scope
Single file change: `notectl-search/src/embeddings/model.rs` — update the `integration_tests` module.

### Step 1: Capture Query Reference Embedding

Add temporary debug output to the existing test, run once, capture values:

Run: `cargo test -p notectl-search --features integration -- integration_tests::test_encoder_produces_correct_dimension`

Copy first ~50 dimensions into `REFERENCE_EMBEDDING` constant. Remove temp debug output.

### Step 2: Refactor Test Into Helper Function

Extract shared setup (model loading, tokenization, forward pass, pooling, projection) into a helper:

This avoids duplicating ~40 lines of setup between the two test cases.

### Step 3: Add Document Text Test Case

Add new constants and test:

The document prompt template differs from query (`title: {title} | text: {content}` vs `task: search result | query: {content}`), producing a meaningfully different embedding.

### Step 4: Verify Graceful Skip

Confirm both tests still skip gracefully when model unavailable:
`cargo test -p notectl-search --features integration` (without HF_TOKEN / cached model)

Both should print \"Skipping integration test\" and return without error.

### File Changes

**`notectl-search/src/embeddings/model.rs`** (~100 line diff):
- Replace stub `REFERENCE_EMBEDDING` with real values (~50 dims)
- Add `DOC_REFERENCE_EMBEDDING` constant with real values (~50 dims)  
- Add `DOC_TEST_INPUT` constant
- Extract `get_embedding()` helper function
- Update existing test to use helper + assert against reference
- Add `test_document_embedding_matches_reference()` test
- Both tests share skip logic via `download::is_model_ready()`

### Risks / Notes
- Requires HF_TOKEN and model download to capture reference values (one-time cost)
- Candle f32/CPU is deterministic within same architecture — safe for CI
- 1e-4 tolerance per dimension accounts for floating-point precision differences across hardware
- If CI runs on different CPU arch, might need to regenerate reference vectors there
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
### Implementation Notes (2026-07-18)

**Completed structural work:**
- Extracted `get_embedding()` helper function that handles model loading, tokenization, forward pass, mean pooling, and projection
- Added `DOC_REFERENCE_EMBEDDING` constant and `DOC_TEST_INPUT` for document-text test case  
- Added `assert_embedding_properties()` and `assert_matches_reference()` shared assertion helpers
- Added `skip_if_model_not_ready()` shared skip logic
- Both tests (`test_encoder_produces_correct_dimension`, `test_document_embedding_matches_reference`) compile, run, and skip gracefully when model unavailable
- All 122 tests in notectl-search pass

**Blocked:** Cannot populate `REFERENCE_EMBEDDING` or `DOC_REFERENCE_EMBEDDING` with actual values — requires `HF_TOKEN` with accepted license for `google/embeddinggemma-300m`. Follow-up ticket TASK-1.14.2.1 filed for this work.
<!-- SECTION:NOTES:END -->
