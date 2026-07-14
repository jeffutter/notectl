---
id: TASK-1.21
title: Reconcile duplicate SearchConfig types between notectl-core and notectl-search
status: To Do
assignee: []
created_date: '2026-07-14 11:12'
labels: []
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
