---
id: TASK-1.5
title: 'Storage: manifest + vectors + staleness diff'
status: To Do
assignee: []
created_date: '2026-07-14 02:21'
labels: []
dependencies:
  - TASK-1.2
parent_task_id: TASK-1
priority: high
type: task
ordinal: 6000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add notectl-search/src/store.rs: manifest.json (schema version, model id/dim/chunk params, per-file content_hash/mtime/chunk_ids, chunk list with heading_path/line spans), vectors.bin (flat row-major f32), chunk texts. Atomic write (temp file + rename). Staleness diff: mtime pre-check, blake3 hash as source of truth; drop chunks for removed files; force full rebuild on model/dim/param mismatch. Unit-testable with fake vectors, no model needed.
<!-- SECTION:DESCRIPTION:END -->
