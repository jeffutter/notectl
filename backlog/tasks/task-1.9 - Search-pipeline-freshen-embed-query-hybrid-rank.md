---
id: TASK-1.9
title: 'Search pipeline: freshen -> embed query -> hybrid rank'
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-14 02:22'
updated_date: '2026-07-16 06:56'
labels:
  - planned
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Implementation Plan for search.rs

### Overview
Create `notectl-search/src/search.rs` implementing the end-to-end search pipeline.
All building blocks exist: `fusion.rs` (cosine + RRF), `sparse.rs` (BM25),
`storage.rs` (manifest/vectors/chunks), `embed.rs` (query embedding),
`index.rs` (reindex pipeline). This module orchestrates them.

### Step 1: Define public types
- `enum SearchMode { Hybrid, Dense, Sparse }` — controls which scoring paths run
- `struct SearchOptions { mode: SearchMode, max_results: usize, rrf_k: f64, rrf_bm25_weight: f64, rrf_cosine_weight: f64, no_reindex: bool }`
- Re-export `RankedChunk` from lib.rs (already defined there)

### Step 2: Implement `pub async fn search(\&self, query: &str, options: SearchOptions) -> SearchResult<Vec<RankedChunk>>`
Replace the `todo!()` stub in `SearchCapability::search()`.

**Step 2a — Freshen (staleness check + conditional reindex)**
- Resolve index_dir from config
- Open manifest via `SearchIndex::open_or_create()`
- Run `compute_staleness_diff(base_path, config, manifest)`
- If stale (Incremental or FullRebuild) and !no_reindex:
  - Create Embedder + Chunker, call `build_index()` to update index
  - Log info message with summary
- If stale and no_reindex: log warning, proceed with existing index
- If UpToDate: skip model load entirely (cold-start optimization)

**Step 2b — Load index artifacts**
- Read manifest chunks list (`manifest.chunks`: Vec<ChunkEntry>)
- Read chunk texts: iterate manifest chunks, call `index.read_chunk(id)` for each
  - Build a Vec<String> preserving manifest order (deterministic, matches vector positions)
- Read dense vectors: `index.read_vectors()` (returns empty Vec if vectors.bin missing)
  - If vectors missing but chunks exist: log warning, auto-degrade to sparse-only
- Rebuild BM25 in-memory: construct a temporary `Vec<Chunk>` from manifest entries + loaded texts, pass to `SparseIndexer::index_chunks()`

**Step 2c — Embed query**
- Only when mode is Dense or Hybrid
- Create Embedder (lazy-load model; it caches internally)
- Call `embedder.embed_single(query, None, TaskType::RetrievalQuery)`
- Result is already L2-normalized at configured matryoshka dim

**Step 2d — Score & rank**
Match mode:
- `Dense`: `cosine_top_k(&vectors, &query_vec, max_results)` → fuse with empty sparse
- `Sparse`: `sparse_indexer.score_query(query)` → fuse with empty dense
- `Hybrid`: `cosine_top_k(&vectors, &query_vec, max_results * 2)` + `sparse_indexer.score_query(query)` → `rrf_fuse(dense, sparse, rrf_k, w_dense, w_sparse)`
  - Use max_results*2 for cosine top-k to give BM25 long-tail terms a chance before truncation
- Truncate fused results to max_results

**Step 2e — Map to RankedChunk**
- For each (chunk_index, fused_score) in fused results:
  - Look up ChunkEntry at that index in manifest.chunks
  - Read preview: extract ~200 chars from chunk text (first 200 chars, trimmed)
  - Construct RankedChunk { id, source_file, score, heading: heading_path.join(" > "), preview }
- Return sorted by descending score

### Step 3: Wire into SearchCapability
In lib.rs, replace the `todo!()` in `SearchCapability::search()`:
- Construct a Config from self.config (SearchConfig) for the core Config struct
- Call the new search function with default SearchOptions
- For non-embeddings feature: keep BM25-only fallback (but note: current code returns EmbeddingsNotEnabled error — we should allow sparse-only search even without embeddings feature)

### Step 4: Add tests
Unit tests in search.rs mod tests:
- Test with synthetic vectors (no model needed): mock the pipeline path
- Test SearchMode enum serialization
- Test result mapping from chunk_index to RankedChunk
- Integration test: build small index, search it, verify results match expectations

### Error Handling
- Index not found: return SearchError::IndexNotFound
- Vectors missing with dense mode: degrade to sparse with warning log (not an error)
- Embedding fails: return clear error mentioning model download
- Empty corpus: return empty Vec gracefully
,
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created notectl-search/src/search.rs implementing the full search pipeline:

1. **SearchMode enum** (Hybrid, Dense, Sparse) with needs_dense/needs_sparse helpers
2. **SearchOptions struct** configurable from SearchConfig
3. **search() function** orchestrating: freshen (staleness + conditional reindex), load index artifacts, embed query with RetrievalQuery prefix, score & rank (cosine_top_k + BM25 + RRF fusion), map to RankedChunk
4. **Auto-degradation**: gracefully falls back to sparse-only when vectors/model unavailable
5. **Wired into SearchCapability::search()**, replacing the todo!() stub
6. **Without embeddings feature**: search runs sparse-only via BM25 (no longer returns EmbeddingsNotEnabled error)

Added 12 new tests covering: SearchMode flags, SearchOptions defaults/config, extract_preview, sparse-only end-to-end search, empty vault, max_results limit, no_reindex flag, result sorting, RankedChunk field population, and auto-degradation. All 121 tests pass.
<!-- SECTION:FINAL_SUMMARY:END -->
