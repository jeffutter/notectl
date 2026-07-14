---
id: TASK-1.20
title: tokenize_with_overlap panics when overlap_tokens exceeds the current window
status: To Do
assignee: []
created_date: '2026-07-14 11:12'
labels: []
dependencies:
  - TASK-1.4
parent_task_id: TASK-1
priority: medium
type: bug
ordinal: 21000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`tokenize::tokenize_with_overlap` (notectl-search/src/tokenize.rs, ~line 45) computes `end - overlap_tokens` (both `usize`) with no guard that `overlap_tokens <= end`. On the first iteration, `end == max_tokens` (assuming enough words); if a caller configures `overlap_tokens > max_tokens`, this subtraction underflows and panics in debug builds.

Nothing in `ChunkerConfig` or `notectl_core::config::SearchConfig` validates that `overlap_tokens < max_tokens` (equivalently `chunk_overlap_tokens < max_seq_tokens`), so this is reachable through ordinary misconfiguration once the search config is wired into the chunker (see the sibling task reconciling the two SearchConfig types).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 tokenize_with_overlap cannot panic from overlap_tokens >= max_tokens; either clamp/guard inside the function, or validate the invariant where ChunkerConfig is constructed
- [ ] #2 Add a unit test that calls tokenize_with_overlap with overlap_tokens >= max_tokens and asserts it returns a sane result (or a clear error) instead of panicking
<!-- AC:END -->
