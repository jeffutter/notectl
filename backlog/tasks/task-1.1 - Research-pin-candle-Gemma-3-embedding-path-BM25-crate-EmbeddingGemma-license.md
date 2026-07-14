---
id: TASK-1.1
title: >-
  Research: pin candle Gemma-3 embedding path, BM25 crate, EmbeddingGemma
  license
status: To Do
assignee: []
created_date: '2026-07-14 02:21'
labels: []
dependencies: []
parent_task_id: TASK-1
priority: high
type: spike
ordinal: 2000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Confirm candle-transformers exposes a Gemma-3 model usable for EmbeddingGemma (sliding+full attention). Pick a BM25 crate (or evaluate tantivy's built-in scoring as an alternative). Verify EmbeddingGemma's HF gating/license terms and what the first-run auth flow needs to look like. Study the text-embeddings-inference Rust reference implementation for the exact pooling + 2_Dense projection head and the query/document prompt prefixes EmbeddingGemma expects.
<!-- SECTION:DESCRIPTION:END -->
