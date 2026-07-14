---
id: TASK-1
title: Semantic search over full notes (notectl-search)
status: To Do
assignee: []
created_date: '2026-07-14 02:21'
labels: []
dependencies: []
priority: high
type: feature
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a new capability for semantic search over full Obsidian note content (not just extracted tasks), following the plan drafted with Opus.

Architecture: new notectl-search workspace crate, gated behind a cargo feature (candle deps are heavy).
- Chunking: split notes by heading section (extend notectl-outline), fixed-size fallback for long leaf sections.
- Dense embeddings: local, via candle + candle-transformers, using Google's EmbeddingGemma-300M (Gemma 3 backbone). Weights fetched once via hf-hub, cached, offline after.
- Sparse: BM25 (not BM42 - evaluated and rejected as contested/experimental).
- Retrieval: brute-force cosine over stored vectors (no ANN index - vault scale doesn't need it) + BM25, combined via reciprocal rank fusion.
- Storage: .notectl/search/ cache dir - manifest.json + vectors.bin (flat f32) + chunk texts. No sqlite/sled.
- Incremental reindex: mtime pre-check, blake3 content hash as source of truth.
- New CLI/MCP/HTTP surface: `index` and `search` operations following the existing Operation pattern.

GPU: no viable path today - candle has no mature AMD backend (ROCm support is WIP/unmerged) and the target hardware (Radeon 890M) isn't officially ROCm-supported anyway. CPU inference is fine at personal-vault scale.

This is the parent ticket; see subtasks for the ordered implementation sequence.
<!-- SECTION:DESCRIPTION:END -->
