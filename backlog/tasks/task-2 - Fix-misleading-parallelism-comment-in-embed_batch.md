---
id: TASK-2
title: 'Fix: misleading parallelism comment in embed_batch'
status: Needs Plan
assignee: []
created_date: '2026-07-15 21:49'
updated_date: '2026-07-18 14:20'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-1.16
priority: high
ordinal: 100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-1.16 (notectl-search/src/embeddings/embed.rs:252-253). The comment claims 'multiple CPU cores can be utilized via rayon's thread pool' but the batch loop calls self.embed_single().await? sequentially — only one spawn_blocking runs at a time per batch, so no actual parallelism occurs. This is misleading documentation (Clear axis).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Comment on line 252-253 of embed.rs accurately describes the actual concurrency behavior (sequential within-batch processing via spawn_blocking).
- [ ] #2 nix develop -c cargo clippy -p notectl-search passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust+WebAssembly core (crates/gql-core) with a TypeScript/React web app (web/). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-search/src/embeddings/embed.rs and locate lines 252-253 (inside the embed_batch method, the inner loop comment).
2. Replace the comment:
   OLD: '// Each item in the chunk gets its own spawn_blocking call so that\n            // multiple CPU cores can be utilized via rayon's thread pool.'
   NEW: '// Each item in the chunk gets its own spawn_blocking call. Items within a\n            // batch are awaited sequentially; parallelism comes from overlapping\n            // batches across different spawn_blocking threads on the tokio runtime.'
3. Run: nix develop -c cargo clippy -p notectl-search (verify clean)
<!-- SECTION:PLAN:END -->
