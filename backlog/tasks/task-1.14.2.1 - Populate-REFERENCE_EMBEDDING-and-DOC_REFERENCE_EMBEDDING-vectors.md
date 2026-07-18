---
id: TASK-1.14.2.1
title: Populate REFERENCE_EMBEDDING and DOC_REFERENCE_EMBEDDING vectors
status: To Do
assignee: []
created_date: '2026-07-18 05:30'
labels: []
dependencies: []
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
