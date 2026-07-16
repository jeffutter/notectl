---
id: TASK-3
title: 'Fix: fragile unwrap() on model/tokenizer in embed_single'
status: Needs Plan
assignee: []
created_date: '2026-07-15 21:50'
updated_date: '2026-07-15 22:58'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-1.16
priority: high
ordinal: 110
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-1.16 (notectl-search/src/embeddings/embed.rs:215-216). embed_single() calls self.model.as_ref().unwrap() and self.tokenizer.as_ref().unwrap() in production code after ensure_loaded(). While currently safe (ensure_loaded sets both), this violates the project's errors-as-values principle and is fragile if ensure_loaded's contract changes. A panic here would crash the process rather than returning an error.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 embed_single uses Result-aware access to model/tokenizer instead of unwrap()
- [ ] #2 nix develop -c cargo clippy -p notectl-search passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust+WebAssembly core (crates/gql-core) with a TypeScript/React web app (web/). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-search/src/embeddings/embed.rs and locate the embed_single method (around lines 202-227).
2. Replace the two unwrap() calls with a single destructuring match that propagates errors:
   OLD code (lines 215-216):
   
   NEW code:
   
   Actually, since Arc\<Mutex\<T\>\> doesn't implement Clone from &Arc directly in the way needed, a simpler approach:
   
   Then clone them inside the spawn_blocking closure or before it.
3. Verify the function signature already returns Result — it does (Result<Vec<f32>, EmbedError>).
4. Run: nix develop -c cargo clippy -p notectl-search (verify clean)
<!-- SECTION:PLAN:END -->
