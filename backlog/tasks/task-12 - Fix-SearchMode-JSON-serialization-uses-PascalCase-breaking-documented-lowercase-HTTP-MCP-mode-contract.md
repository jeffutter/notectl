---
id: TASK-12
title: >-
  Fix: SearchMode JSON serialization uses PascalCase, breaking documented
  lowercase HTTP/MCP mode contract
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-16 14:37'
updated_date: '2026-07-16 16:08'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-1.10
priority: high
type: bug
ordinal: 100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-1.10 (notectl-search/src/search.rs:44, SearchMode enum). SearchMode derives plain serde::Serialize/Deserialize with no #[serde(rename_all)], so JSON (de)serialization uses PascalCase variant names ('Hybrid', 'Dense', 'Sparse'). But every other surface of this type uses lowercase: the clap value_enum CLI accepts 'hybrid'/'dense'/'sparse' (confirmed working), SearchRequest's schemars description documents 'hybrid|dense|sparse', and SearchOperation::execute_json deserializes SearchRequest (which contains mode: Option<SearchMode>) directly from an HTTP/MCP JSON body via serde_json::from_value. Confirmed empirically: serde_json::to_string(&SearchMode::Hybrid) == \"Hybrid\", and serde_json::from_str::<SearchMode>(\"\\\"hybrid\\\"\") fails. Any HTTP or MCP client sending {"mode": "hybrid"} per the documented convention gets a deserialize error on the /api/search endpoint. This is a Correctness-axis bug: the API contract documented in the schema does not match what the wire format actually accepts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 SearchMode has #[serde(rename_all = "lowercase")] (or equivalent) so serde_json::to_string(&SearchMode::Hybrid) == "\"hybrid\""
- [ ] #2 serde_json::from_str::<SearchMode>("\"hybrid\"") and "\"dense\"" and "\"sparse\"" all succeed
- [ ] #3 A new test in notectl-search/src/search.rs asserts SearchMode's JSON round-trip uses lowercase variant names
- [ ] #4 nix develop -c cargo test -p notectl-search --all-features passes
- [ ] #5 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust+WebAssembly core (crates/gql-core) with a TypeScript/React web app (web/). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-search/src/search.rs. Locate the SearchMode enum definition (~line 31-52), which derives Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize, schemars::JsonSchema, clap::ValueEnum.
2. Add #[serde(rename_all = "lowercase")] directly above the #[non_exhaustive] attribute (or combined into the same derive attribute block), so the three variants Hybrid/Dense/Sparse serialize as "hybrid"/"dense"/"sparse".
3. Verify clap's ValueEnum derive is unaffected (it already produces lowercase kebab-case values independently of serde attributes) by running the existing CLI tests.
4. Add a new test in the '#[cfg(test)] mod tests' block in search.rs, e.g. test_search_mode_json_uses_lowercase, that asserts: serde_json::to_string(&SearchMode::Hybrid).unwrap() == "\"hybrid\""; and serde_json::from_str::<SearchMode>("\"sparse\"").unwrap() == SearchMode::Sparse (and similarly for dense/hybrid).
5. Run: nix develop -c cargo test -p notectl-search --all-features
6. Run: nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings
7. Run: nix develop -c cargo fmt -p notectl-search -- --check (fix formatting if needed).
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added #[serde(rename_all = "lowercase")] to SearchMode enum in notectl-search/src/search.rs so JSON serialization produces lowercase variant names ("hybrid", "dense", "sparse") matching the documented HTTP/MCP schema contract. Added test_search_mode_json_uses_lowercase test asserting bidirectional round-trip for all three variants. All 124 tests pass, clippy clean, fmt clean.
<!-- SECTION:FINAL_SUMMARY:END -->
