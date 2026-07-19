---
id: TASK-37
title: 'Fix: remove non-functional batch chunking in embed_batch'
status: To Do
assignee: []
created_date: '2026-07-19 00:46'
labels:
  - review-followup
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust+WebAssembly core (crates/gql-core) with a TypeScript/React web app (web/). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-search/src/embeddings/embed.rs.

2. In the EmbeddingConfig struct (currently lines 48-59), delete the 'batch_size' field and its doc comment (line 57-58).

3. In 'impl Default for EmbeddingConfig' (currently lines 61-70), delete the 'batch_size: 32,' line (line 67).

4. In 'EmbeddingConfig::from_search_config' (currently lines 72-82), delete the 'batch_size: 32,' line (line 79).

5. Replace the embed_batch method body (currently lines 237-273) with:
   '/// Async entry point: embed a batch of texts with the specified task type.
   ///
   /// Each text is embedded via its own spawn_blocking call, awaited to completion
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
   }'

6. In the '#[cfg(test)] mod tests' block, update 'test_default_embedding_config' (currently around line 389-395) to remove the 'assert_eq!(config.batch_size, 32);' line.

7. Update 'test_embedding_config_from_search_config' (currently around line 421-436) to remove the 'assert_eq!(ec.batch_size, 32);' line.

8. Replace the 'embed_batch_prefixed' test helper (currently lines 457-486) with a version that drops the batch_size parameter and chunking loop:
   '/// Test helper: simulate embed_batch's text/title pairing and return prefixed strings.
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
   }'

9. Update all call sites of embed_batch_prefixed to drop the trailing batch_size argument (currently in test_embed_batch_single_batch_no_title_swap ~line 498, test_embed_batch_multi_batch_title_pairing ~line 525, test_embed_batch_multi_batch_exact_boundary ~line 549, test_embed_batch_multi_batch_with_none_titles ~line 564).

10. Delete test_embed_batch_multi_batch_exact_boundary entirely (currently lines 538-556) — it only exercised chunk-boundary arithmetic that no longer exists, and is now redundant with the pairing test from step 11.

11. Rename test_embed_batch_multi_batch_title_pairing (currently lines 506-536) to test_embed_batch_title_pairing and update its comment to describe pairing correctness generally (drop the 'force multiple batches with batch_size=2' framing since there is no batch_size parameter anymore) — keep its 5-text/5-title assertions as-is, they still validate correct index-wise pairing.

12. Update the comment in test_embed_batch_multi_batch_with_none_titles (currently line 559-561) to drop the 'multi-batch' framing.

13. Read the full embed.rs test module afterward to confirm no remaining reference to batch_size or chunk/batch framing outside of what's needed.

14. Run: nix develop -c cargo test -p notectl-search --all-features (verify all tests pass, including the renamed/updated ones).

15. Run: nix develop -c cargo clippy -p notectl-search -- -D warnings (verify clean).

16. Run: nix develop -c cargo build && nix develop -c cargo build --features search (verify both build configurations still compile, since EmbeddingConfig is constructed elsewhere via from_search_config).
<!-- SECTION:PLAN:END -->
