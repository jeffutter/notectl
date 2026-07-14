---
id: TASK-1.4
title: 'Chunker: section splitting + token-budget fallback'
status: To Do
assignee: []
created_date: '2026-07-14 02:21'
labels: []
dependencies:
  - TASK-1.2
parent_task_id: TASK-1
priority: high
type: task
ordinal: 5000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add notectl-search/src/chunker.rs + tokenize.rs. Add an extract_sections helper to notectl-outline (generalizing the get_section span logic to return all sections at once) rather than duplicating span logic. Leaf sections become chunks; sections exceeding max_seq_tokens split into overlapping windows; tiny sections may merge forward. Each chunk carries heading_path, start_line, end_line. Pure logic, unit-testable without the embedding model.
<!-- SECTION:DESCRIPTION:END -->
