---
id: TASK-4
title: 'Fix: chunker line_start/line_end drift when overlap_tokens > 0'
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-15 23:56'
updated_date: '2026-07-16 05:58'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-1.19
priority: high
type: bug
ordinal: 90
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-1.19 (notectl-search/src/chunker.rs:150-189 and 223-251). Both the merged-section split path and the long-section split path advance word_cursor by each part's FULL word count (part_word_count) after processing it. But tokenize_with_overlap (notectl-search/src/tokenize.rs:23-57) advances the real window by (max_tokens - overlap_tokens) per chunk, re-emitting the last overlap_tokens words at the start of the next part. Since word_cursor ignores this overlap, it overshoots the true word position by overlap_tokens per split part, and the error compounds across parts. Confirmed empirically: a section split with max_tokens=10, overlap_tokens=3 reports line_start values of 1,5,8,11,15,18,21,25,28,1,1,1,1 while the TRUE line for those chunks' first words is 1,-,-,8,-,-,15,-,-,22,-,-,29 (drift growing each chunk, then collapsing back to section.start_line once word_cursor runs past the section's word-span array). Production is affected: ChunkerConfig::default() sets overlap_tokens=50, and storage.rs wires config.search.chunk_overlap_tokens through to every Chunker::new() call, so this is not a theoretical edge case — it is the default configuration path. This violates AC #2 of TASK-1.19 ('distinct, accurate line_start/line_end per split part'). Both new unit tests added by TASK-1.19 (test_long_section_split_distinct_line_spans) explicitly set overlap_tokens: 0, which sidesteps the exact code path where the bug lives — the tests should also cover overlap_tokens > 0 once fixed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A section split into 3+ parts with overlap_tokens > 0 (e.g. matching ChunkerConfig::default()'s overlap_tokens=50 or a smaller nonzero test value) produces line_start/line_end for each part that correctly reflects that part's true first/last word position in the original file, not a value that overshoots by the overlap amount
- [x] #2 The word cursor advance in the section-split loop (chunker.rs ~line 251) and the merge-split loop (chunker.rs ~line 189) accounts for tokenize_with_overlap's actual re-emission of the last overlap_tokens words at the start of each subsequent part, e.g. by tracking the same (end - overlap) advance logic tokenize_with_overlap uses, or by having split_long_text return word-index ranges alongside each part instead of re-deriving them from word counts
- [x] #3 A new or updated unit test exercises a nonzero overlap_tokens value (not 0) for both the section-split and merge-split paths and asserts each part's line_start/line_end matches the true position of its first/last word in the source content
- [x] #4 nix develop -c cargo test -p notectl-search passes
- [x] #5 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Implementation Plan

### Bug Recap
`tokenize_with_overlap()` advances its internal window by `(end - overlap)` words per iteration, re-emitting the last `overlap` words at the start of each subsequent chunk. But both split loops in `chunk_file()` advance `word_cursor` by each part's *full* word count (`part_word_count`), ignoring that overlapping words are reused. This causes `word_cursor` to overshoot by exactly `overlap_tokens` per split part, compounding linearly and producing incorrect `line_start`/`line_end` values.

### Strategy: Return indexed tuples from tokenizer
The universal pattern from prior art (scikit-plots, chunkedrs): the component that owns the windowing logic should emit positional metadata. We will add an indexed variant of `tokenize_with_overlap` that returns `Vec<(String, usize, usize)>` — each tuple is `(chunk_text, start_word_idx, end_word_idx_exclusive)` relative to the input text's word array. The existing `tokenize_with_overlap` delegates to it and discards indices for backward compatibility. Then both split loops in `chunk_file()` use the returned indices directly instead of maintaining a hand-rolled `word_cursor`.

### Step 1: Add `tokenize_with_overlap_indexed` to `notectl-search/src/tokenize.rs`
- Create `pub fn tokenize_with_overlap_indexed(text: &str, max_tokens: usize, overlap_tokens: usize) -> Vec<(String, usize, usize)>` that tracks `start` and `end` word indices alongside each chunk string, returning `(chunk_text, start, end)` where `end` is exclusive (matches `words[start..end]`).
- Refactor existing `tokenize_with_overlap` to delegate to `_indexed` and map away the indices: `tokenize_with_overlap_indexed(text, max_tokens, overlap_tokens).into_iter().map(|(t, _, _)| t).collect()`.
- Add unit tests for the indexed variant verifying start/end indices match expected window boundaries.

### Step 2: Update `split_long_text` to return indexed chunks
- Change signature from `fn split_long_text(&self, text: &str) -> Vec<String>` to `fn split_long_text(&self, text: &str) -> Vec<(String, usize, usize)>`.
- Call `tokenize::tokenize_with_overlap_indexed(...)` instead of `tokenize::tokenize_with_overlap(...)`.

### Step 3: Fix section-split loop in `chunk_file()` (~lines 225-260)
- Replace `let chunk_texts = if ... self.split_long_text(...) else { vec![section.content.clone()] }` with indexed output: single-chunk case yields `vec![(section.content.clone(), 0, section_tokens)]`.
- Remove `word_cursor` accumulator entirely.
- For each `(part, start_idx, end_idx)` tuple, use `start_idx` as `word_cursor` and `(end_idx - start_idx)` as `part_word_count` when indexing into `section_word_spans`.
- Verify `last_idx` computation uses `end_idx.saturating_sub(1)` clamped to span bounds.

### Step 4: Fix merge-split loop in `chunk_file()` (~lines 155-195)
- Same pattern: replace `chunk_text: Vec<String>` with `Vec<(String, usize, usize)>`, single-chunk case yields `vec![(merged_content.clone(), 0, merged_tokens)]`.
- Remove `word_cursor` accumulator. Use `start_idx` from the tuple directly against `merged_word_spans`.
- The `first_in_first_section` / `last_in_first_section` checks remain unchanged since they compare word index against `first_section_words`.

### Step 5: Add regression tests with nonzero `overlap_tokens`
- **Test 1** (section-split path): Reuse pattern from `test_long_section_split_distinct_line_spans` but set `overlap_tokens: 3` (nonzero). Build multi-line section content (each line = 3 words, `max_tokens: 10`). For each resulting chunk, extract first word via `c.text.split_whitespace().next()`, locate that word's true line in original content by searching `content.lines()`, assert it equals `c.line_start`. Also verify monotonic ordering and that no two adjacent chunks share identical `(line_start, line_end)`.
- **Test 2** (merge-split path): Two tiny sections below `merge_threshold`, merged content exceeds `max_tokens`, with `overlap_tokens: 3`. Same verification: each chunk's `line_start` matches the true line of its first word.

### Step 6: Quality gates
```bash
nix develop -c cargo test -p notectl-search
nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings
```

### Files Modified
| File | Change |
|------|--------|
| `notectl-search/src/tokenize.rs` | Add `tokenize_with_overlap_indexed`, refactor existing function to delegate |
| `notectl-search/src/chunker.rs` | Change `split_long_text` return type, fix both split loops to use indexed tuples, add 2 regression tests |

### Blast Radius Assessment
- `tokenize_with_overlap`: existing callers (tests only, plus one production caller via `split_long_text`) — backward compatible since we keep the original signature delegating to `_indexed`.
- `split_long_text`: private method (`fn`, not `pub`) with exactly two call sites in the same file — changing return type affects only local code.
- No public API surface changes; no external crate dependencies affected.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
## Implementation Notes

Fixed the word_cursor drift bug in chunker.rs by having the tokenizer emit indexed tuples instead of maintaining a hand-rolled cursor. Key changes:

1. **tokenize.rs**: Added `tokenize_with_overlap_indexed()` that returns `Vec<(String, usize, usize)>` — each tuple is (chunk_text, start_word_idx, end_word_idx_exclusive). Refactored existing `tokenize_with_overlap()` to delegate to it for backward compatibility.

2. **chunker.rs**: Changed `split_long_text()` to return indexed tuples. Updated both the section-split loop and merge-split loop to use the returned indices directly instead of advancing a `word_cursor` accumulator. This eliminates the drift caused by overlap_tokens being ignored during cursor advancement.

3. **Bonus fix**: Discovered and fixed a pre-existing bug in the merge-split path where byte positions from merged_content were used directly against individual section content without adjusting for the '\n\n' separator offset. Added `next_section_byte_offset` to correctly map byte positions.

4. **Tests**: Added 6 new unit tests for the indexed tokenizer variant and 2 regression tests exercising nonzero overlap_tokens for both split paths.
<!-- SECTION:NOTES:END -->
