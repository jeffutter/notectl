---
id: TASK-1.8.1
title: Create index.rs module with IndexBuilder orchestration
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-16 06:07'
updated_date: '2026-07-16 06:27'
labels:
  - planned
dependencies: []
parent_task_id: TASK-1.8
priority: high
ordinal: 26000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add notectl-search/src/index.rs containing IndexBuilder — the top-level orchestrator that wires together SearchIndex (storage), Chunker, Embedder, and SparseIndexer into a single build pipeline.

IndexBuilder holds SearchIndex, Chunker, and an optional Embedder. Its async build() method:
1. Calls compute_staleness_diff to classify added/modified/removed files
2. For full rebuilds: clears old chunks/vectors, processes all files
3. For incremental: drops chunks for removed files, re-chunks only changed files
4. Derives document titles from heading_path.join(" > ") for each chunk
5. Calls embed_batch with TaskType::RetrievalDocument when embedder is available
6. Maintains vector-chunk 1:1 alignment by rebuilding the complete vector array after any change
7. Writes vectors.bin, chunk texts, and manifest.json atomically (temp-file-rename)

Refactor SearchIndex.build_index() to delegate its orchestration role to IndexBuilder, keeping SearchIndex as the storage/persistence layer only. Wire through SearchCapability.index().
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Implementation Plan

### 1. Create notectl-search/src/index.rs

Define IndexBuilder struct:

Constructor takes base_path and Config; resolves index_dir via config.search.resolve_index_dir(base_path). Creates SearchIndex::open_or_create, Chunker::new(ChunkerConfig::from_search_config(&config.search)), and optionally Embedder::new(model_cache_dir, EmbeddingConfig::from_search_config(&config.search)).

### 2. Implement async build(&mut self, base_path: &Path, config: &Config) -> Result<StalenessDiff, SearchError>

- Call compute_staleness_diff (already public in storage.rs)
- Match on StalenessDiff:
  - UpToDate: return early
  - FullRebuild: call index.clear_chunks(), remove vectors.bin
  - Incremental: collect removed chunk_ids from manifest's FileInfo entries, call index.remove_chunks()
- Collect current markdown files via collect_markdown_files (honors exclusion patterns)
- For each file: read content, blake3 hash, chunk via chunker.chunk_file()
- Build unified chunk list sorted by source_file path for deterministic ordering
- If embedder is Some: derive titles from heading_path.join(" > "), call embed_batch(texts, titles, RetrievalDocument), write vectors via index.write_vectors(vectors), set manifest.has_embeddings = true
- Write chunk texts via index.write_chunks()
- Update manifest fields (files, chunks, content_hash, last_indexed, has_embeddings)
- Save manifest atomically via index.save_manifest()

### 3. Refactor SearchIndex.build_index()

Remove the orchestration logic from build_index(). Keep it as a simple wrapper or remove it entirely. The method signature currently takes _embedder which is unused — this confirms the orchestration belongs elsewhere. After refactoring, SearchIndex owns only persistence: open_or_create, save_manifest, write_chunks, write_vectors, read_vectors, read_chunk, clear_chunks, remove_chunks, reset.

### 4. Update lib.rs

Add  and .

### 5. Wire SearchCapability.index()

Replace the todo! with IndexBuilder creation + .build() call.

### Key Design Decisions

- **Vector-chunk alignment**: On any incremental update where chunks change, we rebuild the entire vector array. We cannot patch individual vectors because chunk IDs may shift and the binary format is positional. This is safe because embedding is batched and the changed subset is typically small.
- **Title derivation**: Use heading_path.join(" > ") for the document title in RetrievalDocument prefix. Falls back to filename stem if heading_path is empty.
- **Feature gating**: Embedding step is behind #[cfg(feature = "embeddings")]. Without the feature, build() still works but sets has_embeddings = false.
- **Error resilience**: Per-file errors (read failures, chunk panics) should be collected as warnings without aborting the entire pipeline. One bad file shouldn't block indexing the rest of the vault.
,
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation complete:

**Created notectl-search/src/index.rs** with:
-  struct holding SearchIndex, Chunker, and optional Embedder (feature-gated)
-  method implementing walk→diff→chunk→embed→persist pipeline
-  return type with files_indexed, chunks_produced, has_embeddings, content_hash
- Title derivation from heading_path.join(' > ') for RetrievalDocument embeddings
- Per-file error resilience (skips unreadable files with warning)
- Deterministic chunk ordering (sorted by source_file path)
- Convenience  function that creates IndexBuilder from base_path + Config
- 9 comprehensive tests covering initial build, up-to-date detection, incremental updates, full rebuilds, exclusion patterns, and empty vault

**Updated notectl-search/src/storage.rs**:
- Made helper functions pub(crate): blake3_hash_str, atomic_write_json, atomic_write_binary, chrono_now_rfc3339
- Fixed rel_path ownership issue in build_index() 
- Kept SearchIndex.build_index() as synchronous method for backward compat with existing tests

**Updated notectl-search/src/lib.rs**:
- Added index module export
- Wired SearchCapability.index() to use crate::index::build_index()

All 83 tests pass. Clippy clean.
<!-- SECTION:NOTES:END -->
