---
id: TASK-1.2
title: Scaffold notectl-search crate + workspace wiring
status: Done
assignee: '@ralph'
created_date: '2026-07-14 02:21'
updated_date: '2026-07-14 03:15'
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

## Implementation Notes

- Created `notectl-search/` workspace crate with three core modules:
  - `chunker.rs` - Splits markdown by heading sections with fixed-size fallback for long sections
  - `bm25.rs` - In-memory BM25 indexer/scorer (no external crate dependency)
  - `storage.rs` - Manifest management (manifest.json), chunk text storage, and binary vector I/O
- Feature gating:
  - Crate-level `embeddings` feature gates candle/hf-hub/tokenizers/bm25 deps
  - Root package `search` feature enables notectl-search (sparse only)
  - Root package `search-dense` feature enables notectl-search with embeddings
- `SearchError::EmbeddingsNotEnabled` provides a clear error message when dense search is attempted without the feature
- All types use serde for serialization (manifest, chunks, config)
- Tests cover chunker splitting, BM25 scoring, storage round-trip, and content hashing
- Workspace wiring: added to members, [workspace.dependencies], and root package optional dep

## Acceptance Criteria

- [x] Create notectl-search/Cargo.toml and src/lib.rs
- [x] Gate candle/hf-hub/tokenizers/bm25 deps behind 'embeddings' feature on the crate
- [x] Add 'search' and 'search-dense' features on root notectl package
- [x] Add to workspace members and [workspace.dependencies]
- [x] Crate compiles without the embeddings feature (chunker/BM25/storage work)
- [x] Dense search returns clear 'feature disabled' error without embeddings
