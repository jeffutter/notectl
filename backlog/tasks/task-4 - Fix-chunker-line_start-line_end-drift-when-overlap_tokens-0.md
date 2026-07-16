---
id: TASK-4
title: 'Fix: chunker line_start/line_end drift when overlap_tokens > 0'
status: To Do
assignee: []
created_date: '2026-07-15 23:56'
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
- [ ] #1 A section split into 3+ parts with overlap_tokens > 0 (e.g. matching ChunkerConfig::default()'s overlap_tokens=50 or a smaller nonzero test value) produces line_start/line_end for each part that correctly reflects that part's true first/last word position in the original file, not a value that overshoots by the overlap amount
- [ ] #2 The word cursor advance in the section-split loop (chunker.rs ~line 251) and the merge-split loop (chunker.rs ~line 189) accounts for tokenize_with_overlap's actual re-emission of the last overlap_tokens words at the start of each subsequent part, e.g. by tracking the same (end - overlap) advance logic tokenize_with_overlap uses, or by having split_long_text return word-index ranges alongside each part instead of re-deriving them from word counts
- [ ] #3 A new or updated unit test exercises a nonzero overlap_tokens value (not 0) for both the section-split and merge-split paths and asserts each part's line_start/line_end matches the true position of its first/last word in the source content
- [ ] #4 nix develop -c cargo test -p notectl-search passes
- [ ] #5 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust+WebAssembly core (crates/gql-core) with a TypeScript/React web app (web/). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-search/src/chunker.rs and notectl-search/src/tokenize.rs side by side. Re-read tokenize_with_overlap (tokenize.rs:23-57): it slides a window of max_tokens words, advancing by (end - overlap) each iteration (where overlap = overlap_tokens.min(max_tokens.saturating_sub(1))), so consecutive parts share `overlap` words at the boundary.

2. In chunk_file's section-split loop (chunker.rs, the block starting around 'let chunk_texts = if section_tokens > self.config.max_tokens' through the 'word_cursor += part_word_count;' line), the current code advances word_cursor by the FULL word count of each part, which double-counts the overlapping words. Replace this with logic that mirrors tokenize_with_overlap's real advance: track word_cursor the same way tokenize_with_overlap tracks `start` — first part starts at word_cursor=0 and covers max_tokens words (or fewer, for the last part); each subsequent part's true starting word index is (previous window end - overlap), not (previous word_cursor + previous part's word count).

   Concretely: instead of recomputing part_word_count from part.split_whitespace().len() and advancing by that, compute the same end/advance sequence tokenize_with_overlap uses directly in chunk_file (or, preferably, change split_long_text to return Vec<(String, usize, usize)> of (text, start_word_idx, end_word_idx) so chunk_file does not need to re-derive word positions at all — this removes the duplicated windowing logic and the class of bug entirely). Prefer this second approach since it stops chunker.rs from re-implementing tokenize_with_overlap's windowing math a second time.

3. Update tokenize_with_overlap's signature (or add a sibling function) in notectl-search/src/tokenize.rs to return word index ranges alongside each text part, e.g. `pub fn tokenize_with_overlap_indexed(text: &str, max_tokens: usize, overlap_tokens: usize) -> Vec<(String, usize, usize)>` returning (chunk_text, start_word_idx, end_word_idx_exclusive). Keep the existing tokenize_with_overlap for any other callers, or have it delegate to the new indexed version and discard the indices — check for other callers first with `grep -rn 'tokenize_with_overlap' notectl-search/src`.

4. Update chunk_file's section-split loop and merge-split loop (chunker.rs ~150-189 and ~223-251) to call the indexed variant and use the returned start_word_idx/end_word_idx directly against section_word_spans / merged_word_spans instead of maintaining a hand-rolled word_cursor that assumes no overlap.

5. Add a unit test in chunker.rs's #[cfg(test)] mod tests: build multi-line section content (reuse the pattern from test_long_section_split_distinct_line_spans — lines of 3 words each), but set overlap_tokens to a nonzero value (e.g. 3) instead of 0. For each resulting chunk, extract its first word via c.text.split_whitespace().next(), locate that word's true line in the original content (search content.lines() for the line whose first word matches), and assert it equals c.line_start. Add a second test doing the same for the merge-split path (small sections below merge_threshold with a nonzero overlap_tokens on the merged content when it also exceeds max_tokens).

6. Run: nix develop -c cargo test -p notectl-search (all tests, including the two new ones, must pass) and nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings (must be clean).
<!-- SECTION:PLAN:END -->
