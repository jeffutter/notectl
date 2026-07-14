---
id: TASK-1.7
title: 'Embedding model: candle Gemma-3 backbone + pooling + embed()'
status: To Do
assignee: []
created_date: '2026-07-14 02:22'
updated_date: '2026-07-14 11:12'
labels: []
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
