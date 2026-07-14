---
id: TASK-1.3
title: Add SearchConfig to notectl-core
status: To Do
assignee: []
created_date: '2026-07-14 02:21'
labels: []
dependencies: []
parent_task_id: TASK-1
priority: medium
type: task
ordinal: 4000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend notectl-core/src/config.rs with a #[serde(default)] SearchConfig: model id/revision, embedding_dim (matryoshka truncation), max_seq_tokens, chunk_overlap_tokens, min_chunk_tokens, rrf_k, optional dense/sparse weights, cache_dir (default .notectl/search). Mirror the existing env-merge pattern. Add unit tests matching existing Config test style.
<!-- SECTION:DESCRIPTION:END -->
