---
id: DRAFT-1
title: Integrate file-level tags into search chunks
status: Draft
assignee: []
created_date: '2026-07-20 00:29'
labels: []
dependencies: []
priority: medium
type: feature
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extract YAML frontmatter tags during chunking and attach them to each Chunk. Store tags in the manifest (ChunkEntry), inject them into BM25 index text for searchability, and expose them in RankedChunk results. Add optional `--tags` filter to search operations.

**Changes needed:**
1. Add `tags: Vec<String>` to `Chunk` struct (chunker.rs)
2. Extract frontmatter tags in `chunk_file()` — reuse tag extraction logic or parse inline
3. Add `tags` field to `ChunkEntry` in storage.rs (manifest persistence)
4. Inject tags into BM25 index text when building SparseIndexer (search.rs)
5. Add `tags: Vec<String>` to `RankedChunk` (lib.rs)
6. Add `tags` filter parameter to `SearchOptions` and filter results post-scoring
7. Update CLI/MCP/HTTP capabilities to accept `--tags` parameter
8. Tests for tag extraction, indexing, searching, and filtering
<!-- SECTION:DESCRIPTION:END -->
