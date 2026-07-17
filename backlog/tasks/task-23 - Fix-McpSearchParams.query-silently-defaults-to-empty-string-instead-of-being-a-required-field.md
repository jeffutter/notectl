---
id: TASK-23
title: >-
  Fix: McpSearchParams.query silently defaults to empty string instead of being
  a required field
status: Needs Plan
assignee: []
created_date: '2026-07-16 22:09'
updated_date: '2026-07-16 23:45'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-1.12
priority: high
ordinal: 101
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-1.12 (src/mcp.rs:216-225, 257). McpSearchParams.query is Option<String>, and SearchTool::invoke calls params.query.unwrap_or_default() (mcp.rs:257), so an MCP 'search' call that omits 'query' silently executes a search with an empty string instead of failing. This is inconsistent with the same operation's HTTP/CLI surface: notectl_search::capability::SearchRequest.query is a plain non-Optional String (notectl-search/src/capability.rs:85) with clap 'required = true' (capability.rs:83), so omitting it there is a hard error at parse/deserialize time. TASK-1.12's Implementation Notes justify the Option<String> choice as 'all-optional fields (Default required by ToolBase)', but rmcp's ToolBase::Parameter bound is only '+ Default' (rmcp-2.2.0 tool_traits.rs:27), not '+ Option fields' — String already implements Default (empty string), so 'query: String' satisfies #[derive(Default)] on the struct while still being a serde-required field. This is a Correctness/Clarity-axis finding: the same logical operation enforces 'query is required' on two transports (HTTP, CLI) but silently no-ops on the third (MCP), and the stated rationale for the current design doesn't hold up.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 McpSearchParams.query in src/mcp.rs is of type String (not Option<String>)
- [ ] #2 SearchTool::invoke no longer calls .unwrap_or_default() on query; it uses params.query directly
- [ ] #3 A new test in src/mcp.rs (or notectl-search integration test, whichever the implementer's investigation shows is reachable) demonstrates that a JSON-RPC call to the 'search' MCP tool missing the 'query' field is rejected before reaching SearchCapability::do_search, matching SearchRequest's existing required-query behavior on the HTTP/CLI path
- [ ] #4 nix develop -c cargo build --features search succeeds
- [ ] #5 nix develop -c cargo clippy --features search --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust+WebAssembly core (crates/gql-core) with a
TypeScript/React web app (web/). ALL commands must run inside the Nix dev
shell: either run 'direnv allow' once, or prefix every command with
'nix develop -c'. Work from the repository root unless told otherwise. Do not
change pinned dependency versions.

(This repo is notectl; the file under fix is src/mcp.rs, in the 'search'-feature-gated search_tools module. The same Nix-shell rule applies.)

1. Open src/mcp.rs and locate McpSearchParams (around line 215-225):
   #[derive(Debug, Deserialize, JsonSchema, Default)]
   pub struct McpSearchParams {
       pub query: Option<String>,
       ...
   }
   Change 'pub query: Option<String>' to 'pub query: String'. Keep the '/// The text to search for' doc comment above it (it still becomes the schemars description). Confirm the struct still derives Default cleanly — String::default() is "", so #[derive(Default)] still works with no further changes.

2. In 'impl AsyncTool<TaskSearchService> for SearchTool::invoke' (around line 252-274), change:
   let query = params.query.unwrap_or_default();
   to:
   let query = params.query;
   (or inline params.query directly into the do_search call, matching whichever reads more naturally next to the surrounding code).

3. Confirm rmcp's JSON-RPC parameter deserialization actually rejects a missing 'query' key now that the field is a plain String (not Option<String>) — read how Parameters<T>/ToolBase deserializes incoming arguments (grep the rmcp crate vendored at ~/.cargo/registry/src/*/rmcp-2.2.0/src/handler/server/router/tool/ for how tool call arguments are deserialized) to confirm a missing required field produces a JSON-RPC invalid-params style error rather than silently defaulting. If rmcp's tool-call deserialization path has its own default-filling behavior that bypasses serde's normal missing-field error (e.g. it merges over a Default::default() instance instead of deserializing directly), that must be documented in the task's Implementation Notes and the AC's test adjusted accordingly to reflect actual behavior — do not silently assume.

4. Add a test that exercises this. Prefer a test near the existing search_tools code if mcp.rs already has a #[cfg(test)] section reachable from there (it does not yet — see TASK-1.12's review notes that mcp.rs has zero tests today, matching the rest of the file's existing convention). Given that, the most direct place to add coverage is a small integration-style test: construct McpSearchParams via serde_json::from_value(serde_json::json!({})) (an empty object, i.e. 'query' omitted) and assert the deserialization fails with a missing-field error. This directly proves the schema now requires 'query' without needing to spin up a full MCP server round-trip.

5. Run: nix develop -c cargo build --features search and nix develop -c cargo test --features search (or the appropriate test invocation covering src/mcp.rs) to confirm the new test passes and nothing else regresses.

6. Run: nix develop -c cargo clippy --features search --all-targets -- -D warnings — fix any warnings.

7. In the task's Implementation Notes, record what rmcp actually does with a missing required parameter (invalid-params JSON-RPC error vs. some other failure mode) so this is documented for future MCP tool authors in this codebase.
<!-- SECTION:PLAN:END -->
