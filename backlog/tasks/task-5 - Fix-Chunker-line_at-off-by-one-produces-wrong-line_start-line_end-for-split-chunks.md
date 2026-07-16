---
id: TASK-5
title: >-
  Fix: Chunker::line_at off-by-one produces wrong line_start/line_end for split
  chunks
status: To Do
assignee: []
created_date: '2026-07-16 07:21'
updated_date: '2026-07-16 07:21'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-4
priority: high
type: bug
ordinal: 101
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-4 (notectl-search/src/chunker.rs:277-283, `Chunker::line_at`). TASK-4 fixed the word_cursor/index bookkeeping for split chunks (section-split and merge-split paths), but a separate, pre-existing off-by-one bug in `line_at` was not caught by TASK-4's own regression tests and remains live.

`line_at(content, pos)` computes `content[..pos].lines().count()`. This is only correct when `pos` is exactly at the start of a line (position 0, or immediately after a `\n`). For any `pos` that falls strictly *inside* a line (the common case for split-chunk boundaries, since word-window advances rarely land on a line's first word), `.lines().count()` over-counts by exactly 1, because a partial trailing line is still counted as "one more line" even though the position hasn't crossed into the next line yet.

Verified empirically with an independent ground-truth check (searching raw file content for the true line containing each chunk's first word, NOT reusing `line_at`): in a 30-line/3-words-per-line test with `max_tokens=10, overlap_tokens=3` (word window advances by 7, not a multiple of 3), 8 of 13 split chunks reported a `line_start` exactly one line later than the true line — both in the section-split path (chunker.rs ~227-251) and the merge-split path (chunker.rs ~153-193, the merged_word_spans branch). This directly violates TASK-4 AC #1 ('produces line_start/line_end for each part that correctly reflects that part's true first/last word position'), despite TASK-4 being marked Done with that AC checked off.

Root cause of the false confidence: TASK-4's two new regression tests (`test_long_section_split_overlap_nonzero_line_spans` and `test_merged_section_split_overlap_nonzero_line_spans` in chunker.rs) do not independently verify ground truth. The section-split test computes its 'expected' line numbers by calling `Chunker::line_at` itself — the very function under test — so any bug in `line_at` passes trivially (tautological assertion, not a real check). The merge-split test only asserts weaker invariants (non-decreasing order, bounds) and never compares against a true position at all. This violates TASK-4 AC #3, which required tests that assert against 'the true position of its first/last word in the source content'.

This is a Correctness-axis finding: search results built on top of these chunks (TASK-1.9's search pipeline) will report wrong line numbers to users for the majority of split chunks whenever overlap_tokens > 0 (which is the default: ChunkerConfig::default().overlap_tokens = 50).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Chunker::line_at(content, pos) returns the correct 0-indexed line number for ANY pos, not just positions at a line's start — e.g. by counting '\n' occurrences in content[..pos] instead of counting content[..pos].lines()
- [ ] #2 test_long_section_split_overlap_nonzero_line_spans is rewritten so its 'expected' line numbers are computed independently of Chunker::line_at (e.g. by locating each chunk's first/last word directly in the original file content, not by re-deriving via the function under test)
- [ ] #3 test_merged_section_split_overlap_nonzero_line_spans is strengthened to assert exact line_start/line_end equality against an independently computed true position for every merged chunk, not just ordering/bounds invariants
- [ ] #4 A new or adjusted test covers the case where the word-window advance (max_tokens - overlap_tokens) is NOT a multiple of the content's words-per-line, so the split boundary lands mid-line (this is the case that exposed the bug)
- [ ] #5 nix develop -c cargo test -p notectl-search --all-features passes
- [ ] #6 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust+WebAssembly core (crates/gql-core) with a
TypeScript/React web app (web/). ALL commands must run inside the Nix dev
shell: either run 'direnv allow' once, or prefix every command with
'nix develop -c'. Work from the repository root unless told otherwise. Do not
change pinned dependency versions.

(This repo is notectl; the crate under test is notectl-search. The same Nix-shell rule applies.)

1. Open notectl-search/src/chunker.rs and locate `fn line_at` (~line 277):
   ```rust
   fn line_at(content: &str, pos: usize) -> usize {
       if pos >= content.len() {
           content.lines().count()
       } else {
           content[..pos].lines().count()
       }
   }
   ```
   Replace the body with a count of newline bytes preceding `pos`, which is the actual 0-indexed line number:
   ```rust
   fn line_at(content: &str, pos: usize) -> usize {
       let clamped = pos.min(content.len());
       content.as_bytes()[..clamped].iter().filter(|&&b| b == b'\n').count()
   }
   ```
   Verify by hand: for content = "aaa\nbbb\nccc", line_at(content, 0) == 0, line_at(content, 4) == 1 (start of "bbb"), line_at(content, 5) == 1 (mid "bbb" — this case was incorrectly 2 before the fix), line_at(content, 8) == 2 (start of "ccc").

2. Search notectl-search/src/chunker.rs for all callers of `line_at` (merge-split loop ~line 154-188, section-split loop ~line 240-247, and `chunk_by_size` ~line 322-323). Confirm none need to change — they already add `section.start_line` / `next_section.start_line` as the base offset and expect `line_at` to return a correct *relative* line offset within that section's content.

3. In `#[cfg(test)] mod tests`, rewrite `test_long_section_split_overlap_nonzero_line_spans` (~line 705) so the 'expected' value is computed WITHOUT calling `Chunker::line_at` or any other production helper. For each chunk: take its first word (`c.text.split_whitespace().next()`), and independently determine the true 0-indexed line in `content` by finding which line's `split_whitespace()` word list contains that exact word (the test fixture already uses unique words per position, e.g. "w07 x07 y07"). Example:
   ```rust
   let true_line = content
       .lines()
       .position(|l| l.split_whitespace().any(|w| w == first_word))
       .unwrap();
   assert_eq!(chunk.line_start, true_line, "chunk {:?} line_start mismatch", chunk.id);
   ```
   Do the same for the chunk's last word against `line_end`.

4. Similarly rewrite `test_merged_section_split_overlap_nonzero_line_spans` (~line 788) to assert exact `line_start`/`line_end` equality against the same kind of independent ground truth, for every chunk produced by the merge — not just ordering and bounds. Keep the existing ordering/bounds assertions in addition to the new exact-match assertions.

5. Confirm at least one chunk in each of the two tests above has a split boundary that is NOT aligned to a line start (i.e. its first word is not the first word on its line) — the existing `max_tokens: 10, overlap_tokens: 3` fixtures with 3-words-per-line content already produce this (advance = 7, not a multiple of 3), so this should already be exercised once steps 3-4 use real ground truth. If it turns out not to be exercised, adjust `max_tokens`/`overlap_tokens` so at least one split lands mid-line, and note this in the Implementation Notes.

6. Run: `nix develop -c cargo test -p notectl-search --all-features` and `nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings`. Fix any failures — do not weaken assertions to make tests pass; if a test fails, the underlying bug is not yet fixed.

7. In the task's Implementation Notes, record a concrete before/after example (e.g. line_at(content, 5) returning 2 before the fix and 1 after) so future readers understand why the change was needed.
<!-- SECTION:PLAN:END -->
