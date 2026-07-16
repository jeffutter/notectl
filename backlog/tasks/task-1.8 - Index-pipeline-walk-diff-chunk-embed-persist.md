---
id: TASK-1.8
title: 'Index pipeline: walk -> diff -> chunk -> embed -> persist'
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-14 02:22'
updated_date: '2026-07-16 06:28'
labels:
  - planned
dependencies:
  - TASK-1.8.1
parent_task_id: TASK-1
priority: high
type: task
ordinal: 9000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add notectl-search/src/index.rs (IndexBuilder). Use collect_markdown_files from notectl-core. For each file: mtime pre-check, blake3 hash if changed, chunk via chunker, embed via embed::embed_documents with the document prompt prefix. Drop chunks for removed files. Reassign chunk ids, write vectors.bin/texts/manifest.json atomically.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Orchestration Plan

### Architecture

Create notectl-search/src/index.rs with IndexBuilder — the top-level orchestrator that wires together SearchIndex (storage), Chunker, Embedder, and SparseIndexer into a single build pipeline.

### How sub-tickets fit together

**TASK-1.8.1** (planned): Create index.rs module with IndexBuilder orchestration. This is the sole implementation ticket covering:
1. New index.rs module with IndexBuilder struct holding SearchIndex, Chunker, optional Embedder
2. Async build() method implementing walk→diff→chunk→embed→persist pipeline
3. Refactoring SearchIndex.build_index() out of its orchestration role (keep as storage layer only)
4. Wire through SearchCapability.index()

### Integration steps

After TASK-1.8.1 completes:
- SearchCapability.index() calls IndexBuilder::new().build()
- IndexBuilder delegates to existing compute_staleness_diff, chunker.chunk_file(), embedder.embed_batch(), and SearchIndex persistence methods
- No new dependencies or breaking API changes

### Verification

- Run cargo test --features embeddings on notectl-search
- Verify build_index produces vectors.bin alongside chunks/ and manifest.json
- Verify incremental updates re-embed only changed files' chunks
- Verify full rebuild clears old vectors on model/dim/config change
- Verify has_embeddings flag in manifest reflects whether embedder was available

### Remaining work after sub-tickets

None — TASK-1.9 (search pipeline) and TASK-1.10 (capability operations) are sibling tickets that consume the index built by this one.
<!-- SECTION:PLAN:END -->
