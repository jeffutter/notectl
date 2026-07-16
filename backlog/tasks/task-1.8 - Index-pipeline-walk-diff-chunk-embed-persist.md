---
id: TASK-1.8
title: 'Index pipeline: walk -> diff -> chunk -> embed -> persist'
status: To Do
assignee: []
created_date: '2026-07-14 02:22'
updated_date: '2026-07-15 23:56'
labels: []
dependencies:
  - TASK-1.4
  - TASK-1.5
  - TASK-1.6
  - TASK-1.18
  - TASK-1.19
  - TASK-1.20
  - TASK-4
parent_task_id: TASK-1
priority: high
type: task
ordinal: 9000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add notectl-search/src/index.rs (IndexBuilder). Use collect_markdown_files from notectl-core. For each file: mtime pre-check, blake3 hash if changed, chunk via chunker, embed via embed::embed_documents with the document prompt prefix. Drop chunks for removed files. Reassign chunk ids, write vectors.bin/texts/manifest.json atomically.
<!-- SECTION:DESCRIPTION:END -->
