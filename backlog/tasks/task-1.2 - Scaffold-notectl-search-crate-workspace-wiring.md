---
id: TASK-1.2
title: Scaffold notectl-search crate + workspace wiring
status: To Do
assignee: []
created_date: '2026-07-14 02:21'
labels: []
dependencies:
  - TASK-1.1
parent_task_id: TASK-1
priority: high
type: task
ordinal: 3000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create notectl-search/Cargo.toml and src/lib.rs. Gate candle/hf-hub/tokenizers/bm25 deps behind an 'embeddings' feature on the crate and a 'search' feature on the root notectl package. Add to workspace members and [workspace.dependencies]. Without the feature, the crate should still compile (chunker/BM25/storage only; dense search returns a clear 'feature disabled' error).
<!-- SECTION:DESCRIPTION:END -->
