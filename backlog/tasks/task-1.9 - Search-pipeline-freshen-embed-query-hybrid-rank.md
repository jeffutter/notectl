---
id: TASK-1.9
title: 'Search pipeline: freshen -> embed query -> hybrid rank'
status: To Do
assignee: []
created_date: '2026-07-14 02:22'
updated_date: '2026-07-14 02:22'
labels: []
dependencies:
  - TASK-1.7
  - TASK-1.8
parent_task_id: TASK-1
priority: high
type: task
ordinal: 10000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add notectl-search/src/search.rs. Cheap staleness freshen unless --no-reindex (only loads the model if something actually needs re-embedding). Load manifest + vectors + texts, rebuild BM25 in-memory. Embed query with the query prompt prefix, truncate+normalize. Dense cosine top-k + BM25 score + RRF fusion (or dense-only/sparse-only per --mode). Map chunk ids back to file path/heading path/line span/snippet.
<!-- SECTION:DESCRIPTION:END -->
