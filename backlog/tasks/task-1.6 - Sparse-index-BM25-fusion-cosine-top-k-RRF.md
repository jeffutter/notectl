---
id: TASK-1.6
title: 'Sparse index (BM25) + fusion (cosine top-k, RRF)'
status: To Do
assignee: []
created_date: '2026-07-14 02:22'
labels: []
dependencies:
  - TASK-1.2
parent_task_id: TASK-1
priority: medium
type: task
ordinal: 7000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add notectl-search/src/sparse.rs (BM25 wrapper built from chunk texts, in-memory, rebuilt at load rather than persisted) and fusion.rs (cosine top-k over normalized vectors via dot product; reciprocal rank fusion combining dense + sparse rankings, weighted per SearchConfig). Unit-testable independently of the embedding model.
<!-- SECTION:DESCRIPTION:END -->
