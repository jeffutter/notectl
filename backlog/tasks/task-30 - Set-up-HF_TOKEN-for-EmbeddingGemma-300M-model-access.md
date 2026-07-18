---
id: TASK-30
title: Set up HF_TOKEN for EmbeddingGemma-300M model access
status: To Do
assignee:
  - '@ralph'
created_date: '2026-07-18 16:59'
updated_date: '2026-07-18 19:56'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Attempted automated setup via agent-browser but Hugging Face returned 403 (anti-bot blocking). No existing HF_TOKEN found in environment, ~/.huggingface/, .envrc, or any config files. This task requires manual human action:

Manual steps required:
1. Visit https://huggingface.co/google/embeddinggemma-300m and log in (create account if needed)
2. Accept the model license agreement
3. Generate an access token at https://huggingface.co/settings/tokens (read permission on models is sufficient)
4. Export HF_TOKEN="hf_xxxxxxxx" in your shell or add to .envrc
5. Run: cargo run --features embeddings -p notectl-search --example print_embedding
6. Paste the output constants into notectl-search/src/embeddings/model.rs per TASK-29
<!-- SECTION:NOTES:END -->
