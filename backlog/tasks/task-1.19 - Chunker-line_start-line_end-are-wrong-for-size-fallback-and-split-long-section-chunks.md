---
id: TASK-1.19
title: >-
  Chunker line_start/line_end are wrong for size-fallback and split long-section
  chunks
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-14 11:12'
updated_date: '2026-07-15 22:29'
labels:
  - planned
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
- [x] #1 Chunks produced by the no-heading fallback path (chunk_by_size) carry accurate line_start/line_end for multi-line input
- [x] #2 Chunks produced by splitting a long section carry distinct, accurate line_start/line_end per split part instead of a repeated/collapsed value
- [x] #3 Add a unit test for multi-line content with no headings asserting correct line_start for chunks past the first
- [x] #4 Add a unit test for a section long enough to split into 3+ parts asserting each part's line span is distinct and correctly ordered
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Implementation Plan

### Overview
Fix two bugs in notectl-search/src/chunker.rs where line_start/line_end are computed incorrectly for chunks produced by (a) the no-heading size-fallback path and (b) the long-section split path. Both bugs stem from trying to infer positions from space-joined text that has lost its original newlines.

### Root Cause
- **chunk_by_size:** Joins words with " ", then calls content.find(&text) — fails on multi-line input, defaults line_start to 0 for every chunk.
- **split_long_text output:** tokenize_with_overlap produces single-line strings (words joined by spaces). part.lines().count() is always 1, so the formula section.start_line + j * 1 / 2 collapses to the same value for every part.

### Fix Strategy: Track word indices, not text search

**Step 1 — Fix chunk_by_size (lines ~172-194)**
Replace the content.find(&text) approach with a running word cursor:
- Maintain  that advances by chunk size each iteration
- For line_start: count newlines in content[..first_char_of_word_at_word_idx]
- For line_end: count newlines up to first char of next chunk's word (or content.len())
- Use a helper  = content[..char_pos].lines().count()

**Step 2 — Fix split_long_text line computation (lines ~139-150)**
split_long_text delegates to tokenize_with_overlap which returns Vec<String>. We need word-range info too. Two options:
- **Option A (preferred):** Change split_long_text to return Vec<(String, usize, usize)> where the tuple includes (text, first_word_index_in_original, last_word_index_in_original). The caller uses these indices against the original section content to compute line_start/line_end.
- **Option B:** Have the main loop track a running word cursor itself, advancing by (max_tokens - overlap) per chunk (accounting for the final chunk being shorter). Use the cursor + part word count to index into original section words.

Go with Option B — it avoids changing tokenize_with_overlap's signature and keeps the fix localized to the caller in chunk_file.

In chunk_file, after calling split_long_text:
- Split section.content into words once: let section_words = section.content.split_whitespace().collect::<Vec<_>>()
- Track  starting at 0
- For each part (index j): compute line_start from section_words[word_cursor]'s position in content, line_end from the last word's position
- Advance word_cursor by (max_tokens - overlap) capped at remaining words

**Step 3 — Same fix for merged-section split path (lines ~115-134)**
The merge path also calls split_long_text on merged_content. Apply the same word-cursor tracking pattern there, using the merged content's words.

**Step 4 — Add unit tests (acceptance criteria #3 and #4)**
- Test: multi-line content with NO headings → verify chunk_by_size produces distinct, monotonically increasing line_start values
- Test: section with enough words to split into 3+ parts → verify each part's line_start/line_end is distinct and correctly ordered

### Files Changed
- notectl-search/src/chunker.rs (fix + tests)

### Risks
- Low. The tokenize_with_overlap overlap semantics are already tested. We just need to correctly map word indices back to line numbers in the original content.
- Edge case: section.content may have leading/trailing whitespace before/after words — using char positions from the original content handles this naturally since we search for each word's first occurrence from a known offset.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
## Implementation Notes

### Root Cause
Both bugs shared the same underlying problem: trying to infer line positions from space-joined text that had lost its original newlines.

**Bug 1 — chunk_by_size:** Used content.find(&text) where text was words re-joined with single spaces. For multi-line input, the rejoined text never matched (newlines gone), so find returned None and line_start silently defaulted to 0.

**Bug 2 — Section splitting:** Used part.lines().count() but split_long_text returns single-line strings (words joined by spaces), so .lines().count() was always 1. The formula collapsed to the same value for every split part.

### Fix: Word-span tracking with character-to-line conversion

Added two helper methods on Chunker:
- line_at(content, pos) - converts a character position to a 0-indexed line number
- word_spans(content) - collects byte-span ranges for each whitespace-delimited word

chunk_by_size fix: Replaced content.find(&text) with pre-computed word spans. Each chunk's line_start/line_end is derived from the character positions of its first and last words in the original content.

Section splitting fix: After calling split_long_text, count words in each part, advance a word cursor into the section's word-span map, and compute line numbers from character positions. Same approach for the merge-split path, with extra logic to map merged-content word indices back to the correct original section (first vs second) for proper file-line offsets.

Additional fix: Made chunk_by_size the true no-heading fallback by checking if all sections have empty heading titles and routing to chunk_by_size instead of processing a catch-all section through the buggy section-splitting path.

Line number convention: All line numbers are now 0-indexed. The outline extractor returns 1-indexed start_line values, but since section content starts one line after the heading, start_line directly equals the 0-indexed file line of the first content line.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed two bugs in notectl-search/src/chunker.rs where line_start/line_end were computed incorrectly:

1. chunk_by_size (no-heading fallback): replaced content.find(&text) with word-span tracking. The old approach failed on multi-line input because re-joining words with spaces lost newlines, causing find to return None and line_start to default to 0.

2. Section splitting (long sections split into parts): replaced part.lines().count() (always 1 for single-line tokenized output) with word-cursor tracking against pre-computed word spans in the original section content. Same fix applied to the merge-split path with extra logic to map merged-content word indices back to the correct original section.

Also made chunk_by_size the true no-heading fallback by checking if all sections have empty heading titles. All line numbers normalized to 0-indexed. Added two unit tests verifying correct line spans for both code paths.
<!-- SECTION:FINAL_SUMMARY:END -->
