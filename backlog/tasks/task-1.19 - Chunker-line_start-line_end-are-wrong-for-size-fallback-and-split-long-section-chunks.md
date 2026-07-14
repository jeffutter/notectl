---
id: TASK-1.19
title: >-
  Chunker line_start/line_end are wrong for size-fallback and split long-section
  chunks
status: To Do
assignee: []
created_date: '2026-07-14 11:12'
labels: []
dependencies:
  - TASK-1.4
parent_task_id: TASK-1
priority: medium
type: bug
ordinal: 20000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two separate places in notectl-search/src/chunker.rs compute incorrect line spans for chunks:

1. `chunk_by_size` (the no-heading fallback path, ~lines 185-208) computes `line_start` via `content.find(&text)`, where `text` is built by re-joining `split_whitespace()` words with single spaces. For any multi-line input (the normal case — this fallback exists specifically for content without headings, which is still typically multi-line), the rejoined text no longer matches the original text's newlines, so `content.find(&text)` returns `None` and `line_start` silently defaults to `0` for every chunk produced by this path.

2. In `chunk_file`'s section-splitting branch (~lines 147-150), when a long section is split into multiple parts by `split_long_text`, each part's `line_start`/`line_end` is computed as `section.start_line + j * part.lines().count() / 2`. Because `split_long_text` builds each part via `words.join(" ")` (no newlines), `part.lines().count()` is always 1, so this formula collapses to the same (wrong) value for every split part instead of tracking each part's actual position within the section — all split chunks from the same section report near-identical, incorrect line spans.

Nothing currently consumes these fields at runtime (no active breakage today), but TASK-1.8's persisted chunk records and TASK-1.9's "map chunk ids back to ... line span" both depend on these being correct.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Chunks produced by the no-heading fallback path (chunk_by_size) carry accurate line_start/line_end for multi-line input
- [ ] #2 Chunks produced by splitting a long section carry distinct, accurate line_start/line_end per split part instead of a repeated/collapsed value
- [ ] #3 Add a unit test for multi-line content with no headings asserting correct line_start for chunks past the first
- [ ] #4 Add a unit test for a section long enough to split into 3+ parts asserting each part's line span is distinct and correctly ordered
<!-- AC:END -->
