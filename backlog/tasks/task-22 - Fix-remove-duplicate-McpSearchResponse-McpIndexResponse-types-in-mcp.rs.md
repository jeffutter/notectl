---
id: TASK-22
title: 'Fix: remove duplicate McpSearchResponse/McpIndexResponse types in mcp.rs'
status: Needs Plan
assignee: []
created_date: '2026-07-16 22:09'
updated_date: '2026-07-16 23:49'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-1.12
priority: high
ordinal: 100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-1.12 (src/mcp.rs:227-232, 288-295). McpSearchResponse and McpIndexResponse are hand-written structs whose fields (results/total_count/mode_used and files_indexed/chunks_produced/has_embeddings/content_hash/duration_ms) are byte-for-byte duplicates of notectl_search::capability::SearchResponse (notectl-search/src/capability.rs:108-116) and IndexResponse (capability.rs:58-70), which already derive Serialize + JsonSchema — the only bounds ToolBase::Output requires. SearchTool::invoke and IndexTool::invoke (mcp.rs:252-274, 315-333) manually copy each field from the real response into the shadow struct for no functional reason. This is a Concise-axis violation (CLAUDE.md: pass-through duplication, information leakage) — two places now encode the same response schema, and they will silently drift out of sync the next time SearchResponse or IndexResponse gains/loses a field.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 SearchTool::Output is notectl_search::SearchResponse and IndexTool::Output is notectl_search::IndexResponse (no McpSearchResponse/McpIndexResponse types remain in mcp.rs)
- [ ] #2 SearchTool::invoke and IndexTool::invoke return the capability's response value directly (no manual field-by-field reconstruction)
- [ ] #3 nix develop -c cargo build --features search succeeds and nix develop -c cargo build succeeds (feature-off build still compiles)
- [ ] #4 nix develop -c cargo clippy --features search --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust+WebAssembly core (crates/gql-core) with a
TypeScript/React web app (web/). ALL commands must run inside the Nix dev
shell: either run 'direnv allow' once, or prefix every command with
'nix develop -c'. Work from the repository root unless told otherwise. Do not
change pinned dependency versions.

(This repo is notectl; the file under fix is src/mcp.rs, in the 'search'-feature-gated search_tools module. The same Nix-shell rule applies.)

1. Open src/mcp.rs and locate the search_tools module (starts around line 206, #[cfg(feature = "search")] mod search_tools).

2. Add 'use notectl_search::{IndexResponse, SearchResponse};' near the top of the module (alongside the existing 'use notectl_search::{RankedChunk, SearchMode};' import at line 209) — RankedChunk and SearchMode may become unused after this change; if so remove them and re-check what is still needed (SearchMode is still used by McpSearchParams::mode, so only add the two response types).

3. Delete the McpSearchResponse struct definition (around line 227-232).

4. In 'impl ToolBase for SearchTool' (around line 236-250), change 'type Output = McpSearchResponse;' to 'type Output = SearchResponse;'.

5. In 'impl AsyncTool<TaskSearchService> for SearchTool' (around line 252-274), simplify the invoke body's tail from:
   Ok(McpSearchResponse { results: response.results, total_count: response.total_count, mode_used: response.mode_used })
   to:
   Ok(response)

6. Delete the McpIndexResponse struct definition (around line 288-295).

7. In 'impl ToolBase for IndexTool' (around line 299-313), change 'type Output = McpIndexResponse;' to 'type Output = IndexResponse;'.

8. In 'impl AsyncTool<TaskSearchService> for IndexTool' (around line 315-333), simplify the invoke body's tail from:
   Ok(McpIndexResponse { files_indexed: response.files_indexed, chunks_produced: response.chunks_produced, has_embeddings: response.has_embeddings, content_hash: response.content_hash, duration_ms: response.duration_ms })
   to:
   Ok(response)

9. Run: nix develop -c cargo build --features search — fix any unused-import warnings from step 2.
10. Run: nix develop -c cargo build (feature off) — must still compile since search_tools is fully #[cfg(feature = "search")]-gated.
11. Run: nix develop -c cargo clippy --features search --all-targets -- -D warnings — fix any warnings.
12. Run: nix develop -c cargo test -p notectl-search --all-features — confirm no regressions (this change is purely in the top-level notectl crate, notectl-search tests are unaffected but should still pass).
<!-- SECTION:PLAN:END -->
