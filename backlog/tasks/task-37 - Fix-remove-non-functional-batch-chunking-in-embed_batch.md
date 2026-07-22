---
id: TASK-37
title: 'Fix: remove non-functional batch chunking in embed_batch'
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-19 00:46'
updated_date: '2026-07-19 01:04'
labels:
  - review-followup
  - planned
milestone: Active
dependencies:
  - TASK-36
priority: high
ordinal: 100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-36 (notectl-search/src/embeddings/embed.rs:243-273). TASK-36's fix made the doc/inline comments on embed_batch accurate, but doing so exposed that the two-level chunking loop (outer texts.chunks(self.config.batch_size), inner per-item loop) provides no real function: there is no concurrency (every spawn_blocking is awaited before the next starts, confirmed by TASK-36), no memory benefit (results is pre-allocated to full length via Vec::with_capacity(texts.len()) before the loop starts, regardless of batch_size), and batch_size is 'not user-configurable yet' per TASK-1.21's notes so there is no external knob relying on this shape. This chunking structure is also the exact root cause of a real historical bug: TASK-1.16 (embed_batch pairs texts with wrong titles across batches), where the start/end index arithmetic across chunk boundaries silently corrupted title-to-text pairing. The structure has now caused three review cycles of churn (TASK-1.16, TASK-2, TASK-36) for a code shape that does nothing. This is a Concise/Organized-axis violation per CLAUDE.md's value>>cost test and 'define errors out of existence': flattening embed_batch into a single loop over texts.iter().zip(titles.iter()) removes the chunk-boundary index arithmetic entirely, structurally eliminating the whole class of pairing bugs instead of re-verifying it by hand in tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 EmbeddingConfig no longer has a batch_size field (removed from the struct at embed.rs:50-59, its Default impl at 61-70, and from_search_config at 72-82)
- [ ] #2 Embedder::embed_batch (embed.rs:243-273) is a single flat loop over texts.iter().zip(titles.iter()) with no chunking, no batch_idx/start/end arithmetic, and no config.batch_size reference
- [ ] #3 The doc comment and inline comment on embed_batch describe the flat sequential loop only — no mention of batching, chunk boundaries, or batch_size
- [ ] #4 The embed_batch_prefixed test helper and its batch-boundary-specific tests (test_embed_batch_multi_batch_title_pairing, test_embed_batch_multi_batch_exact_boundary) are collapsed into a single title-pairing test that has no batch_size parameter, since there is no batch boundary left to test; test_embed_batch_multi_batch_with_none_titles is updated the same way and kept
- [ ] #5 nix develop -c cargo test -p notectl-search --all-features passes
- [ ] #6 nix develop -c cargo clippy -p notectl-search -- -D warnings passes
- [ ] #7 #1 EmbeddingConfig no longer has a batch_size field (removed from the struct at embed.rs:50-59, its Default impl at 61-70, and from_search_config at 72-82)=done
- [ ] #8 #2 Embedder::embed_batch (embed.rs:243-273) is a single flat loop over texts.iter().zip(titles.iter()) with no chunking, no batch_idx/start/end arithmetic, and no config.batch_size reference=done
- [ ] #9 #3 The doc comment and inline comment on embed_batch describe the flat sequential loop only — no mention of batching, chunk boundaries, or batch_size=done
- [ ] #10 #4 The embed_batch_prefixed test helper and its batch-boundary-specific tests (test_embed_batch_multi_batch_title_pairing, test_embed_batch_multi_batch_exact_boundary) are collapsed into a single title-pairing test that has no batch_size parameter, since there is no batch boundary left to test; test_embed_batch_multi_batch_with_none_titles is updated the same way and kept=done
- [ ] #11 #5 nix develop -c cargo test -p notectl-search --all-features passes=done
- [ ] #12 #6 nix develop -c cargo clippy -p notectl-search -- -D warnings passes=done
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
All changes are confined to a single file: `notectl-search/src/embeddings/embed.rs`. No other files reference `batch_size` — callers construct `EmbeddingConfig` via `default()` or `from_search_config()` and only access `output_dim`, `max_seq_len`, and `dtype`. Build inside the Nix dev shell (`nix develop -c <command>`).

### Step 1: Remove `batch_size` from `EmbeddingConfig`

1. Delete the `batch_size` field and its doc comment from the struct (lines 57–58):
   ```rust
   // REMOVE THESE TWO LINES:
   /// Batch size for embedding (higher = faster but more memory)
   pub batch_size: usize,
   ```

2. Delete `batch_size: 32,` from `impl Default for EmbeddingConfig` (line 67).

3. Delete `batch_size: 32,` from `EmbeddingConfig::from_search_config` (line 79).

### Step 2: Remove dead `EmbedError::BatchExceeded` variant

The `BatchExceeded` variant (line 98) and its `Display` arm (lines 115–117) are never constructed anywhere in the codebase. Remove them:

4. Delete `BatchExceeded(usize),` from the enum (line 98).

5. Delete the corresponding match arm from `impl Display for EmbedError` (lines 115–117):
   ```rust
   // REMOVE THIS ARM:
   EmbedError::BatchExceeded(size) => {
       write!(f, "Batch size {size} exceeds maximum")
   }
   ```

### Step 3: Flatten `embed_batch`

Replace the entire `embed_batch` method (lines 234–273) with a flat sequential loop:

```rust
/// Async entry point: embed a batch of texts with the specified task type.
///
/// Each text is embedded via its own `spawn_blocking` call, awaited to completion
/// before the next begins — processing is fully sequential. Returns a vector of
/// embedding vectors in the same order as the input.
pub async fn embed_batch(
    &mut self,
    texts: &[String],
    titles: &[Option<String>],
    task: TaskType,
) -> Result<Vec<Vec<f32>>, EmbedError> {
    if texts.len() != titles.len() {
        return Err(EmbedError::Inference(
            "texts and titles must have the same length".to_string(),
        ));
    }

    let mut results = Vec::with_capacity(texts.len());
    for (text, title) in texts.iter().zip(titles.iter()) {
        let embedding = self.embed_single(text, title.as_deref(), task).await?;
        results.push(embedding);
    }

    Ok(results)
}
```

Key structural change: `texts.iter().zip(titles.iter())` replaces the two-level `chunks(batch_size)` + `start/end` index arithmetic. This structurally eliminates the whole class of title/text pairing bugs (TASK-1.16, TASK-2, TASK-36).

### Step 4: Update config tests

6. In `test_default_embedding_config` (~line 389–395), delete:
   ```rust
   assert_eq!(config.batch_size, 32);
   ```

7. In `test_embedding_config_from_search_config` (~line 421–436), delete:
   ```rust
   assert_eq!(ec.batch_size, 32);
   ```

### Step 5: Simplify test helper and tests

8. Replace `embed_batch_prefixed` (~lines 457–486) — drop the `batch_size` parameter and flatten to a simple iterator chain:
   ```rust
   /// Test helper: simulate embed_batch's text/title pairing and return prefixed strings.
   /// This lets us verify title-to-text pairing without loading the model.
   fn embed_batch_prefixed(
       texts: &[String],
       titles: &[Option<String>],
       task: TaskType,
   ) -> Vec<String> {
       assert_eq!(
           texts.len(),
           titles.len(),
           "texts and titles must have the same length"
       );

       texts
           .iter()
           .zip(titles.iter())
           .map(|(text, title)| task.apply_prefix(text, title.as_deref()))
           .collect()
   }
   ```

9. Update ALL call sites of `embed_batch_prefixed` to drop the trailing `batch_size` argument (currently passed as `32` or `2`).

10. **Delete** `test_embed_batch_multi_batch_exact_boundary` entirely (~lines 538–556). It only exercised chunk-boundary arithmetic that no longer exists.

11. **Rename** `test_embed_batch_multi_batch_title_pairing` → `test_embed_batch_title_pairing`. Update the comment to drop the "force multiple batches with batch_size=2" framing. Keep the 5-text/5-title assertions — they still validate correct index-wise pairing.

12. Update the comment in `test_embed_batch_multi_batch_with_none_titles` to drop the "multi-batch" framing. The test itself is fine — it verifies `None` titles get the `"none"` fallback.

### Step 6: Verify

13. Scan the full file for any remaining references to `batch_size`, `chunk`, `batch_idx`, `BatchExceeded`, or misleading batching language.

14. Run: `nix develop -c cargo test -p notectl-search --all-features`

15. Run: `nix develop -c cargo clippy -p notectl-search -- -D warnings`

16. Run: `nix develop -c cargo build && nix develop -c cargo build --features search` (verifies both build configurations compile, since `EmbeddingConfig::from_search_config` is called from `index.rs` and `search.rs`)
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Flattened Embedder::embed_batch into a single texts.iter().zip(titles.iter()) loop, removing the non-functional two-level chunking structure that caused title/text pairing bugs across TASK-1.16, TASK-2, and TASK-36. Removed batch_size field from EmbeddingConfig (struct, Default, from_search_config), dead EmbedError::BatchExceeded variant, and collapsed batch-boundary-specific tests into concise title-pairing verification tests. Net change: -93 lines +17 lines in embed.rs.
<!-- SECTION:FINAL_SUMMARY:END -->
