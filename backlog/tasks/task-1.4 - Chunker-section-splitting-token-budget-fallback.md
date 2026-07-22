---
id: TASK-1.4
title: 'Chunker: section splitting + token-budget fallback'
status: Done
assignee: ralph
created_date: '2026-07-14 02:21'
labels: []
dependencies:
  - TASK-1.2
parent_task_id: TASK-1
priority: high
type: task
ordinal: 5000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add notectl-search/src/chunker.rs + tokenize.rs. Add an extract_sections helper to notectl-outline (generalizing the get_section span logic to return all sections at once) rather than duplicating span logic. Leaf sections become chunks; sections exceeding max_seq_tokens split into overlapping windows; tiny sections may merge forward. Each chunk carries heading_path, start_line, end_line. Pure logic, unit-testable without the embedding model.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

Completed implementation of the chunker system with the following components:

### 1. extract_sections helper in notectl-outline

- Added `extract_sections()` and `extract_sections_from_content()` methods to `OutlineExtractor`
- Returns all sections in a markdown file with their spans (start_line, end_line)
- Generalizes the span logic from `get_section` to return every section at once
- Handles edge case of files with no headings (treats entire file as one section)

### 2. tokenize.rs module

- Created `notectl-search/src/tokenize.rs` with lightweight tokenization utilities
- `count_tokens()`: Counts whitespace-separated words as approximate token count
- `tokenize_with_overlap()`: Splits text into overlapping chunks respecting token budget
- `tokenize_fixed()`: Splits text into fixed-size word chunks without overlap
- All functions are pure logic and unit-testable without embedding model

### 3. Enhanced chunker.rs

- Refactored to use `extract_sections_from_content()` from notectl-outline
- Added `heading_path: Vec<String>` field to `Chunk` struct for full hierarchical path tracking
- Implemented tiny section merging (configurable threshold, default 30 tokens)
- Implemented overlapping window splitting for long sections (configurable overlap, default 50 tokens)
- Added fallback to size-based chunking if section extraction fails
- Each chunk now carries: id, source_file, line_start, line_end, heading, heading_path, text

### 4. Configuration

- `ChunkerConfig` with fields:
  - `max_tokens`: Maximum tokens per chunk (default: 512)
  - `min_chunk_size`: Minimum tokens to keep a chunk (default: 50)
  - `overlap_tokens`: Overlap between consecutive chunks in long sections (default: 50)
  - `merge_threshold`: Threshold for merging tiny sections (default: 30)

### 5. Dependencies

- Added `notectl-outline.workspace = true` to notectl-search/Cargo.toml

### Tests

- All existing tests pass (166 tests across workspace)
- Added comprehensive unit tests for:
  - Basic chunking with sections
  - Long section splitting with overlap
  - Tiny section merging
  - Heading path tracking
  - Tokenization functions
  - Edge cases (empty content, zero max tokens, etc.)

## Acceptance Criteria

- [x] Add notectl-search/src/chunker.rs + tokenize.rs
- [x] Add extract_sections helper to notectl-outline
- [x] Leaf sections become chunks
- [x] Sections exceeding max_seq_tokens split into overlapping windows
- [x] Tiny sections may merge forward
- [x] Each chunk carries heading_path, start_line, end_line
- [x] Pure logic, unit-testable without the embedding model
