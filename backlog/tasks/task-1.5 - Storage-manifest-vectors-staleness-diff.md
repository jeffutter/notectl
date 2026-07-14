---
id: TASK-1.5
title: 'Storage: manifest + vectors + staleness diff'
status: To Do
assignee: []
created_date: '2026-07-14 02:21'
updated_date: '2026-07-14 11:12'
labels: []
dependencies:
  - TASK-1.2
  - TASK-1.17
  - TASK-1.21
parent_task_id: TASK-1
priority: high
type: task
ordinal: 6000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add notectl-search/src/store.rs: manifest.json (schema version, model id/dim/chunk params, per-file content_hash/mtime/chunk_ids, chunk list with heading_path/line spans), vectors.bin (flat row-major f32), chunk texts. Atomic write (temp file + rename). Staleness diff: mtime pre-check, blake3 hash as source of truth; drop chunks for removed files; force full rebuild on model/dim/param mismatch. Unit-testable with fake vectors, no model needed.
<!-- SECTION:DESCRIPTION:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-07-14 11:12
---
Review note (branch `embedding` vs `main`): the existing `collect_file_info` helper in storage.rs (used by `compute_content_hash`) walks the whole directory tree with no exclusion filtering, unlike `notectl_core::file_walker::collect_markdown_files` which honors `config.should_exclude()`. When implementing the real staleness-diff/content-hash logic here, make sure excluded paths stay excluded — don't just adapt the existing walker as-is.
---
<!-- COMMENTS:END -->
