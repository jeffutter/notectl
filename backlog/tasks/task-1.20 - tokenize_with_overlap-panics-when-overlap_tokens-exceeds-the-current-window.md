---
id: TASK-1.20
title: tokenize_with_overlap panics when overlap_tokens exceeds the current window
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-14 11:12'
updated_date: '2026-07-15 12:09'
labels:
  - planned
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Implementation Plan

### 1. Fix the panic in `tokenize_with_overlap` (notectl-search/src/tokenize.rs, line ~56)

**Problem:** `end - overlap_tokens` underflows when `overlap_tokens >= max_tokens`. The guard on line 58 (`if advance <= start { end } else { advance }`) checks the result *after* subtraction.

**Fix:** Clamp `overlap_tokens` at the top of the function so it can never exceed `max_tokens - 1`:

```rust
pub fn tokenize_with_overlap(text: &str, max_tokens: usize, overlap_tokens: usize) -> Vec<String> {
    if max_tokens == 0 {
        return vec![String::new()];
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![String::new()];
    }

    // Clamp overlap so it is always strictly less than max_tokens
    let overlap = overlap_tokens.min(max_tokens.saturating_sub(1));
    // ... rest uses `overlap` instead of `overlap_tokens`
}
```

Using `max_tokens.saturating_sub(1)` handles the `max_tokens=0` early return path and ensures `overlap < max_tokens` for all valid paths. The clamp is cheap (one comparison) and preserves all existing behavior for correctly-configured callers.

### 2. Add unit test

Add two test cases to the existing `tests` module in tokenize.rs:

- **`test_overlap_ge_max_tokens`**: Call with `overlap_tokens > max_tokens` (e.g. `tokenize_with_overlap("a b c d e f g h i j", 4, 10)`) — should produce chunks without panicking. With overlap clamped to 3, expect non-overlapping chunks: `["a b c d", "e f g h", "i j"]`.
- **`test_overlap_max_one`**: Call with `max_tokens=1, overlap_tokens=5` — should produce one-word-per-chunk result without panicking.

### Files Changed
- `notectl-search/src/tokenize.rs` — clamp + tests (~10 lines changed, ~15 lines added)

### Verification
- `cargo test -p notectl-search -- tokenize` — all existing tests pass plus new ones
- No changes to chunker.rs or config types needed; the clamp is defense-in-depth at the function level
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
## Implementation Notes

### Fix Applied
Clamped  at the top of  to prevent underflow:

This ensures  for all code paths, preventing the panic when a caller configures .

### Test Cases Added
1. **test_overlap_ge_max_tokens**: Verifies no panic with . Produces 7 overlapping chunks (each advancing by 1 word).
2. **test_overlap_max_one**: Verifies no panic with . Produces 3 single-word chunks.

### Verification
- 
running 11 tests
test bm25::tests::test_tokenize ... ok
test tokenize::tests::test_count_tokens_simple ... ok
test tokenize::tests::test_overlap_ge_max_tokens ... ok
test tokenize::tests::test_count_tokens_with_extra_whitespace ... ok
test tokenize::tests::test_overlap_max_one ... ok
test tokenize::tests::test_tokenize_empty ... ok
test tokenize::tests::test_tokenize_fixed_remainder ... ok
test tokenize::tests::test_tokenize_with_overlap ... ok
test tokenize::tests::test_tokenize_fixed_simple ... ok
test tokenize::tests::test_tokenize_with_overlap_no_remainder ... ok
test tokenize::tests::test_tokenize_zero_max ... ok

test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 16 filtered out; finished in 0.00s

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s — all 11 tests pass (9 existing + 2 new)
- No changes to chunker.rs or config types needed; the clamp is defense-in-depth at the function level

## Implementation Notes

Fix: Clamped overlap_tokens at the top of tokenize_with_overlap using .min(max_tokens.saturating_sub(1)) to prevent underflow panic when overlap_tokens >= max_tokens.

Tests added: test_overlap_ge_max_tokens (overlap=10 > max=4) and test_overlap_max_one (max=1, overlap=5). Both verify no panic and produce sane chunks.

All 11 tests pass.
<!-- SECTION:NOTES:END -->
