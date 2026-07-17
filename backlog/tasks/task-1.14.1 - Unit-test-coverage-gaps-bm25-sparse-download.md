---
id: TASK-1.14.1
title: 'Unit test coverage gaps: bm25, sparse, download'
status: To Do
assignee: []
created_date: '2026-07-17 00:30'
updated_date: '2026-07-17 00:30'
labels: []
dependencies: []
parent_task_id: TASK-1.14
priority: medium
type: task
ordinal: 14100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add 8 unit tests to fill coverage gaps in bm25.rs (4), sparse.rs (2), and download.rs (2). No model required — all tests run without any cargo features.

### bm25.rs (+4 tests)
- `test_single_document_corpus` — IDF computes correctly with one doc
- `test_identical_documents` — identical docs score equally
- `test_extreme_params` — k1=0 or b=1 produces valid scores (no NaN/inf)
- `test_long_document_vs_short` — length normalization favors short doc with same terms

### sparse.rs (+2 tests)
- `test_empty_query` — empty string returns empty results
- `test_single_chunk` — single chunk corpus returns it for matching query

### download.rs (+2 tests)
- `test_is_model_ready_missing_dir` — false when cache dir absent
- `test_is_model_ready_partial_files` — false when some required files missing

All tests go inline in existing `#[cfg(test)]` modules. No new dependencies needed.
<!-- SECTION:DESCRIPTION:END -->