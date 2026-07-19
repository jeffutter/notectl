---
id: TASK-36
title: >-
  Fix: embed_batch comment still misleading — falsely claims cross-batch
  parallelism
status: To Do
assignee: []
created_date: '2026-07-18 23:36'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-2
priority: high
type: bug
ordinal: 100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-2 (notectl-search/src/embeddings/embed.rs:239-240,261-263). TASK-2 fixed a comment that falsely claimed embed_batch used rayon-based parallelism, but the replacement comment introduces a new false claim: 'parallelism comes from overlapping batches across different spawn_blocking threads on the tokio runtime.' In reality embed_batch has zero concurrency anywhere — the outer batch loop (line 255) and inner item loop (line 264) both call .await? on each embed_single before starting the next, so every spawn_blocking call runs strictly one at a time, never overlapping with any other. The function-level doc comment on line 239-240 ('Processes in batches via spawn_blocking to manage both memory and reactor responsiveness') is similarly misleading — chunking into batch_size groups provides no memory or reactor benefit beyond what each individual spawn_blocking call already provides per item, since results are pre-allocated to full length up front and every item already yields to the reactor independently of chunk boundaries. This violates the Correct/Clear axis: the whole point of TASK-2 was to make this documentation accurate, and it is still inaccurate, just in a new way.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The doc comment on embed_batch (currently lines 239-240) does not claim batching provides memory or reactor-responsiveness benefits it does not actually provide
- [ ] #2 The inline comment inside embed_batch (currently lines 261-263) does not claim any parallelism or overlap occurs between batches or between items — accurately states that all spawn_blocking calls in embed_batch run strictly sequentially, one at a time
- [ ] #3 nix develop -c cargo clippy -p notectl-search passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust+WebAssembly core (crates/gql-core) with a TypeScript/React web app (web/). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-search/src/embeddings/embed.rs.
2. Replace the function doc comment above embed_batch (currently lines 239-240):
   OLD: '/// Processes in batches via `spawn_blocking` to manage both memory and reactor
   /// responsiveness. Returns a vector of embedding vectors.'
   NEW: '/// Each text is embedded via its own `spawn_blocking` call, awaited to completion
   /// before the next begins — processing is fully sequential, both within a batch and
   /// across batches. `batch_size` only controls how many titles are sliced per
   /// iteration; it does not introduce concurrency. Returns a vector of embedding vectors.'
3. Replace the inline comment inside the batch loop (currently lines 261-263):
   OLD: '// Each item in the chunk gets its own spawn_blocking call. Items within a
               // batch are awaited sequentially; parallelism comes from overlapping
               // batches across different spawn_blocking threads on the tokio runtime.'
   NEW: '// Each item gets its own spawn_blocking call, but every call is awaited here
               // before the next starts — there is no concurrency within or across batches.'
4. Read the full embed_batch function afterward to confirm the two comments no longer contradict each other or the actual control flow (sequential for loops, no join_all/FuturesUnordered/spawn anywhere).
5. Run: nix develop -c cargo clippy -p notectl-search (verify clean)
6. Run: nix develop -c cargo test -p notectl-search (verify existing tests still pass)
<!-- SECTION:PLAN:END -->
