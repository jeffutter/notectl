---
id: TASK-31
title: >-
  Fix: integration test get_embedding() lacks token truncation, risks usize
  underflow panic and contradicts its own 'mirrors production' doc comment
status: To Do
assignee: []
created_date: '2026-07-18 17:33'
updated_date: '2026-07-18 17:33'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-27
priority: high
type: bug
ordinal: 90
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-27 (notectl-search/src/embeddings/model.rs:854-878, get_embedding() in the integration_tests module). Production's inner_embed_text (notectl-search/src/embeddings/embed.rs:299) computes 'actual_len = token_ids.len().min(model.embedding_config.max_seq_len)' and pads/truncates using 'token_ids[..actual_len]', so oversized input is safely truncated with a warning log. get_embedding()'s test helper never truncates: it does 'let mut padded = token_ids; padded.extend(std::iter::repeat_n(pad_id, max_len - padded.len()))' using the raw, unbounded token_ids. If a test input ever tokenizes to more than max_seq_len (2048) tokens, 'max_len - padded.len()' underflows (usize subtraction) and panics with 'attempt to subtract with overflow' in debug builds instead of gracefully truncating like production. This directly contradicts the doc comment TASK-27 added at model.rs:854 ('Mirrors embed.rs::inner_embed_text (tokenize -> pad -> forward -> mean_pooling -> projection -> normalize_embedding). Keep the two in sync'), which now overpromises: the two diverge on truncation behavior. Correctness/Resilience-axis gap: a latent panic reachable the moment QUERY_TEST_INPUT/DOC_TEST_INPUT (or any future integration test reusing this helper) exceeds 2048 tokens.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 get_embedding() in notectl-search/src/embeddings/model.rs truncates token_ids to max_seq_len before padding, mirroring inner_embed_text's 'actual_len = token_ids.len().min(max_seq_len)' + 'token_ids[..actual_len]' logic (embed.rs:299,314), eliminating the usize underflow risk
- [ ] #2 A new unit test (outside the #[cfg(feature = "integration")] gate, e.g. in a plain #[cfg(test)] mod near get_embedding, or a pure-function extraction of the truncate+pad step) proves that input longer than max_seq_len does not panic and produces a padded vector of exactly max_seq_len tokens
- [ ] #3 nix develop -c cargo test -p notectl-search --features integration passes
- [ ] #4 nix develop -c cargo clippy -p notectl-search --features integration --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust CLI workspace (notectl-core, notectl-outline, notectl-search, notectl-files, notectl-tags, notectl-tasks, notectl-daily-notes, plus the main notectl binary). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-search/src/embeddings/model.rs, integration_tests module, get_embedding() (around line 856-925). Locate the tokenize/pad block (around line 872-878):
   let encoding = tokenizer.encode(input, false).expect("Tokenization failed");
   let token_ids: Vec<u32> = encoding.get_ids().to_vec();
   let max_len = embedding_config.max_seq_len;
   let pad_id = loaded.pad_token_id;
   let mut padded = token_ids;
   padded.extend(std::iter::repeat_n(pad_id, max_len - padded.len()));

2. Compare against the production reference in notectl-search/src/embeddings/embed.rs, inner_embed_text (around line 298-316):
   let token_ids = encoding.get_ids();
   let actual_len = token_ids.len().min(model.embedding_config.max_seq_len);
   (warn! if token_ids.len() > max_seq_len)
   let mut padded_ids = Vec::with_capacity(max_len);
   padded_ids.extend_from_slice(&token_ids[..actual_len]);
   padded_ids.extend(std::iter::repeat_n(pad_id, max_len - actual_len));

3. Change get_embedding()'s block to match this truncation logic exactly, e.g.:
   let token_ids: Vec<u32> = encoding.get_ids().to_vec();
   let max_len = embedding_config.max_seq_len;
   let pad_id = loaded.pad_token_id;
   let actual_len = token_ids.len().min(max_len);
   let mut padded = Vec::with_capacity(max_len);
   padded.extend_from_slice(&token_ids[..actual_len]);
   padded.extend(std::iter::repeat_n(pad_id, max_len - actual_len));
   Keep the rest of the function (attention_mask construction, tensors, forward pass, pooling, projection, normalize_embedding) unchanged -- it already operates on the local variable 'padded' correctly once this fix is applied.

4. Re-read the doc comment above get_embedding() (around line 854-855) to confirm it still accurately describes the mirrored steps after this fix; no change should be needed.

5. Add a unit test proving the truncation is safe. Because get_embedding() itself requires a downloaded model (gated behind feature = "integration" and skip_if_model_not_ready()), extract the truncate+pad token logic into a small pure helper function (e.g. fn truncate_and_pad(token_ids: Vec<u32>, max_len: usize, pad_id: u32) -> Vec<u32>) that get_embedding() calls, and add a plain #[cfg(test)] test (not gated behind the integration feature) that calls this helper with a token_ids Vec longer than max_len and asserts: (a) it does not panic, (b) the result length equals max_len, (c) the first max_len tokens match the input's first max_len tokens.

6. Run: nix develop -c cargo test -p notectl-search (confirm the new pure-function test passes without needing the integration feature or a downloaded model).

7. Run: nix develop -c cargo test -p notectl-search --features integration -- integration_tests (confirm both integration tests still pass with graceful skip messages, unchanged behavior for the existing short QUERY_TEST_INPUT/DOC_TEST_INPUT).

8. Run: nix develop -c cargo clippy -p notectl-search --features integration --all-targets -- -D warnings.

9. Run: nix develop -c cargo fmt -p notectl-search -- --check (fix formatting if needed).
<!-- SECTION:PLAN:END -->
