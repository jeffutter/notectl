---
id: TASK-1.14.2
title: 'Integration test: populate REFERENCE_EMBEDDING from TEI run'
status: To Do
assignee: []
created_date: '2026-07-17 00:30'
updated_date: '2026-07-17 00:30'
labels: []
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