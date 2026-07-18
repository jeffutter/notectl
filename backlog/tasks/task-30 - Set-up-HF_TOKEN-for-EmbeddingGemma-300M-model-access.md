---
id: TASK-30
title: Set up HF_TOKEN for EmbeddingGemma-300M model access
status: To Do
assignee: []
created_date: '2026-07-18 16:59'
labels:
  - infra
  - blocker
dependencies: []
priority: high
type: task
ordinal: 29000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-29 (populating REFERENCE_EMBEDDING constants) is blocked because HF_TOKEN is not available. The google/embeddinggemma-300m model is gated on Hugging Face and requires:

1. A Hugging Face account
2. Accepted license agreement for google/embeddinggemma-300m  
3. HF_TOKEN environment variable set with a valid token

Once HF_TOKEN is configured, run:
```bash
cargo run --features embeddings -p notectl-search --example print_embedding
```
This will download the model and output Rust-ready constants for REFERENCE_EMBEDDING and DOC_REFERENCE_EMBEDDING.
<!-- SECTION:DESCRIPTION:END -->
