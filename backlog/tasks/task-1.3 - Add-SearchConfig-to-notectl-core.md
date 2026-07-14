---
id: TASK-1.3
title: Add SearchConfig to notectl-core
status: Done
assignee: []
created_date: '2026-07-14 02:21'
updated_date: '2026-07-14 02:58'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added `SearchConfig` struct with all required fields and default values
- Default model_id: "google/embedding-gemma-300m"
- Default embedding_dim: 256 (matryoshka truncation)
- Default max_seq_tokens: 512
- Default chunk_overlap_tokens: 64
- Default min_chunk_tokens: 32
- Default rrf_k: 60.0
- Default cache_dir: ".notectl/search"
- Added env var support: NOTECTL_SEARCH_CACHE_DIR, NOTECTL_SEARCH_EMBEDDING_DIM, NOTECTL_SEARCH_MAX_SEQ_TOKENS
- Added unit tests for default config, TOML parsing, and env var merging
- All 8 tests passing
<!-- SECTION:NOTES:END -->
