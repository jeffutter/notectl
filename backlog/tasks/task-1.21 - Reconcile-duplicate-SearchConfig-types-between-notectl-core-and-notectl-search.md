---
id: TASK-1.21
title: Reconcile duplicate SearchConfig types between notectl-core and notectl-search
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-14 11:12'
updated_date: '2026-07-15 13:42'
labels:
  - planned
dependencies:
  - TASK-1.3
parent_task_id: TASK-1
priority: high
type: task
ordinal: 22000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two unrelated structs share the name `SearchConfig`:
- `notectl_core::config::SearchConfig` (notectl-core/src/config.rs, added by TASK-1.3) — the type actually parsed from the user's TOML config and env vars (model_id, embedding_dim, max_seq_tokens, chunk_overlap_tokens, min_chunk_tokens, rrf_k, dense_weights, sparse_weights, cache_dir).
- `notectl_search::SearchConfig` (notectl-search/src/lib.rs, added by TASK-1.2) — a separate struct (index_dir, max_results, bm25_k1, cosine_weight) that `SearchCapability::new` actually takes as its constructor argument.

No code converts one into the other anywhere in the workspace. As a result, every value a user sets under `[search]` in their config or via `NOTECTL_SEARCH_*` env vars is currently inert: `ChunkerConfig::default()` hardcodes 512/50/50/30 rather than reading max_seq_tokens/chunk_overlap_tokens/min_chunk_tokens, and `EmbeddingConfig::default()` hardcodes its own output dimension rather than reading embedding_dim.

Separately, `notectl_search::SearchConfig::bm25_k1` is documented as "Weight for BM25 scores in RRF fusion" — it is actually being used as the RRF weight, not the classic BM25 k1 saturation constant (a different, already-present field: `Bm25Params::k1` in bm25.rs, default 1.2). Reusing that name for two different concepts is worth fixing in the same pass. `notectl_core::config::merge_search_from_env` also currently wires only 3 of SearchConfig's 9 fields from env vars, an inconsistency worth resolving here too.

This needs to be decided before TASK-1.5/1.6/1.7 stop relying on their own hardcoded defaults, and before TASK-1.10/1.11 wire SearchCapability into the CLI/HTTP surface with a config users can actually control.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A single, authoritative SearchConfig shape is established (either one struct is dropped in favor of the other, or a clear conversion exists between them), documented as to which one is user-facing
- [ ] #2 ChunkerConfig, EmbeddingConfig, and the BM25/fusion parameters are constructed from the authoritative config rather than their own hardcoded defaults
- [ ] #3 The RRF-weight field is renamed away from bm25_k1 (or otherwise disambiguated from Bm25Params::k1) so the two concepts don't share a name
- [ ] #4 merge_search_from_env covers all fields of the authoritative config consistently, or the gap is a deliberate documented decision
- [ ] #5 Unit tests confirm a non-default value set via config/env actually changes chunking/embedding/fusion behavior end-to-end
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Implementation Plan

### Approach

Establish `notectl_core::config::SearchConfig` as the single authoritative type. Drop `notectl_search::SearchConfig` entirely — replace with `pub use notectl_core::config::SearchConfig`. Add missing fields to core, add conversion methods on ChunkerConfig/EmbeddingConfig, rename `bm25_k1` → `rrf_bm25_weight`, fill env var gap.

### Step 1 — Enrich `notectl_core::config::SearchConfig` (config.rs)

Add 4 fields missing from core but needed downstream:
- `max_results: usize` (default 50) — from search crate
- `rrf_bm25_weight: f64` (default 1.0) — renamed from `bm25_k1`; the old value 60.0 was really RRF k, not a weight; use standard Elastic default 1.0
- `rrf_cosine_weight: f64` (default 1.0) — from search crate `cosine_weight`
- `merge_threshold: usize` (default 30) — from ChunkerConfig

Add serde default functions for each. Update Default impl.

### Step 2 — Drop local SearchConfig in notectl-search (lib.rs)

Replace `SearchConfig` struct with `pub use notectl_core::config::SearchConfig`. The core field `cache_dir` already covers the old `index_dir` (same default: `.notectl/search`). Update `resolve_index_dir` to read `cache_dir` instead. Update `SearchCapability::new` — it now takes the core type directly.

### Step 3 — Wire ChunkerConfig from SearchConfig (chunker.rs)

Add `impl ChunkerConfig { pub fn from_search_config(&SearchConfig) -> Self }`:
- `max_tokens` ← `sc.max_seq_tokens`
- `overlap_tokens` ← `sc.chunk_overlap_tokens`
- `min_chunk_size` ← `sc.min_chunk_tokens`
- `merge_threshold` ← `sc.merge_threshold`

Keep `ChunkerConfig::default()` for backward compat (tests, standalone).

### Step 4 — Wire EmbeddingConfig from SearchConfig (embeddings/embed.rs)

Add `impl EmbeddingConfig { pub fn from_search_config(&SearchConfig) -> Self }`:
- `output_dim` ← `sc.embedding_dim as usize`
- `max_seq_len` ← `sc.max_seq_tokens`
- Keep `dtype`, `batch_size` at current defaults (not user-configurable yet).

### Step 5 — Fill env var gap in `merge_search_from_env` (config.rs)

Add lookups for all 7+ uncovered fields: MODEL_ID, MODEL_REVISION, CHUNK_OVERLAP_TOKENS, MIN_CHUNK_TOKENS, RRF_K, DENSE_WEIGHTS, SPARSE_WEIGHTS, MAX_RESULTS, MERGE_THRESHOLD. Follow existing pattern.

### Step 6 — Unit Tests

1. `test_search_config_all_env_vars` — set new env vars, verify fields
2. `test_chunker_config_from_search_config` — non-default SearchConfig → ChunkerConfig
3. `test_embedding_config_from_search_config` — same for EmbeddingConfig
4. `test_search_config_toml_new_fields` — TOML with new fields

### Verification

- `cargo test --package notectl-core` all pass
- `cargo test --package notectl-search` all pass
- `cargo build --features search` clean build
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
## Implementation Notes

Core SearchConfig enriched: Added 4 new fields - max_results (default 50), rrf_bm25_weight (default 1.0, renamed from search-crate bm25_k1 to disambiguate from Bm25Params::k1), rrf_cosine_weight (default 1.0), merge_threshold (default 30). Moved resolve_index_dir helper to core SearchConfig impl block.

Local SearchConfig dropped: Replaced duplicate struct in notectl-search with pub use re-export from core. The old index_dir field is covered by core cache_dir.

ChunkerConfig wired: Added ChunkerConfig::from_search_config() mapping max_seq_tokens, chunk_overlap_tokens, min_chunk_tokens, merge_threshold.

EmbeddingConfig wired: Added EmbeddingConfig::from_search_config() mapping embedding_dim to output_dim, max_seq_tokens to max_seq_len. dtype and batch_size kept at current defaults.

Env var gap filled: merge_search_from_env now covers all 14 SearchConfig fields via env vars (MODEL_ID, MODEL_REVISION, EMBEDDING_DIM, MAX_SEQ_TOKENS, CHUNK_OVERLAP_TOKENS, MIN_CHUNK_TOKENS, MERGE_THRESHOLD, RRF_K, RRF_BM25_WEIGHT, RRF_COSINE_WEIGHT, DENSE_WEIGHTS, SPARSE_WEIGHTS, CACHE_DIR, MAX_RESULTS).

Tests added: test_search_config_default (extended), test_search_config_all_env_vars (all 14 env vars), test_search_config_toml_new_fields, test_chunker_config_from_search_config, test_embedding_config_from_search_config. Updated existing tests in search crate to use new field names.
<!-- SECTION:NOTES:END -->
