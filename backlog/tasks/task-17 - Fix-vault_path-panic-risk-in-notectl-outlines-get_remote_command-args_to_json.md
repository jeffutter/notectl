---
id: TASK-17
title: >-
  Fix: vault_path panic risk in notectl-outline's
  get_remote_command/args_to_json
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-16 16:51'
updated_date: '2026-07-16 20:36'
labels:
  - review-followup
  - planned
milestone: Active
dependencies:
  - TASK-14
priority: high
type: bug
ordinal: 100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-14 (notectl-search/src/capability.rs), which fixed the identical bug for notectl-search and explicitly flagged (in a code comment above IndexOperation::get_remote_command and SearchOperation::get_remote_command) that the same defect exists in notectl-outline, notectl-tags, notectl-files, notectl-daily-notes, and notectl-tasks, deferring the fix as out of scope. Investigation confirms notectl-outline is affected: GetOutlineOperation, GetSectionOperation, and SearchHeadingsOperation in notectl-outline/src/capability.rs (get_remote_command at ~line 356, ~437, ~524) each manually rebuild a clap::Command from scratch that omits the vault_path arg entirely, while their args_to_json methods (~line 413, ~500, ~593) call e.g. GetOutlineRequest::from_arg_matches(matches) unconditionally -- the derive-generated FromArgMatches impl looks up vault_path by id, which panics with 'Mismatch between definition and access of `vault_path`' when the arg was never registered in the hand-built Command (this exact panic was reproduced live for notectl-outline itself in TASK-14's own investigation: building notectl-remote and running './target/debug/notectl-remote --server http://localhost:1 outline foo.md' panics today). This is a Correctness/Resilience-axis bug: a live, reproducible panic reachable from the notectl-remote CLI boundary the moment outline operations are invoked without a local vault_path. Unlike notectl-search's SearchOperation (which genuinely needs a hand-built Command because clap rejects an optional positional preceding a required one -- confirmed empirically: clap_builder panics with 'Found non-required positional argument with a lower index than a required positional argument' when vault_path index=1 is optional and a second positional follows at index=2), none of notectl-outline's three operations have this constraint: vault_path is each operation's *only* positional (file_path/heading/pattern are named --long options or, for GetOutlineOperation, a second positional that comes after vault_path already in the local grammar -- verify per-operation during implementation). Prefer the simpler, already-proven fix used by notectl-tags, notectl-daily-notes, and notectl-tasks: self.get_command().mut_arg("vault_path", |a| a.required(false).hide(true)) for get_remote_command(), plus Request::from_arg_matches(matches)? + request.vault_path = None for args_to_json -- this eliminates the duplicate hand-rolled Command/JSON construction entirely rather than reproducing notectl-search's more verbose field-by-field workaround.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 GetOutlineOperation, GetSectionOperation, and SearchHeadingsOperation's get_remote_command() methods in notectl-outline/src/capability.rs use self.get_command().mut_arg("vault_path", |a| a.required(false).hide(true)) instead of manually rebuilding clap::Command (or, for any operation where mut_arg provably cannot work due to positional ordering -- verify empirically with a small clap probe before assuming this -- keep the manual rebuild but add the vault_path arg to it as hidden+optional)
- [ ] #2 Their args_to_json() methods no longer need special-casing once get_remote_command() registers vault_path (hidden/optional); verify Request::from_arg_matches(matches)? + vault_path = None still works and simplify back to that form if a manual rebuild is no longer needed
- [ ] #3 A new test exercises get_remote_command() end-to-end for all three outline operations without a vault_path arg present (mirroring notectl-search/src/capability.rs's remote_command_tests module added in TASK-14) and asserts args_to_json does not panic and produces valid JSON
- [ ] #4 nix develop -c cargo test -p notectl-outline --all-features passes
- [ ] #5 nix develop -c cargo clippy -p notectl-outline --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Single-file fix in notectl-outline/src/capability.rs. All three operations need Pattern B (manual rebuild + field-by-field JSON) because each has a required positional immediately after vault_path, making mut_arg impossible due to clap 4.6.x positional ordering debug_assert.

## Root Cause
Each get_remote_command() manually rebuilds clap::Command omitting vault_path, while args_to_json() calls Request::from_arg_matches(matches) whose derive-generated impl looks up vault_path by ID and panics with "Mismatch between definition and access of vault_path".

## Changes per operation

### 1. GetOutlineOperation (~line 340-420)
- get_remote_command(): KEEP existing manual rebuild (already correct — no vault_path). No change needed.
- args_to_json(): REPLACE from_arg_matches call with field-by-field serde_json::Map construction:
  - file_path: matches.get_one::<String>("file_path") → String
  - hierarchical: matches.get_one::<bool>("hierarchical") → bool
  - Build serde_json::Map, return Ok(serde_json::Value::Object(obj))

### 2. GetSectionOperation (~line 420-520)
- get_remote_command(): KEEP existing manual rebuild. No change needed.
- args_to_json(): REPLACE from_arg_matches with field-by-field:
  - file_path: get_one::<String>("file_path")
  - heading: get_one::<String>("heading")
  - include_subsections: get_one::<bool>("include_subsections")

### 3. SearchHeadingsOperation (~line 520-620)
- get_remote_command(): KEEP existing manual rebuild. No change needed.
- args_to_json(): REPLACE from_arg_matches with field-by-field:
  - pattern: get_one::<String>("pattern")
  - min_level: get_one::<u8>("min_level")
  - max_level: get_one::<u8>("max_level")
  - limit: get_one::<usize>("limit")

### 4. Add remote_command_tests module (end of file)
Mirror notectl-search/src/capability.rs test module (added in TASK-14):
- dummy_outline_capability() helper
- For each operation (3 ops × 2 tests = 6 tests minimum):
  - args_to_json_no_vault_path_panic: parse via get_remote_command without vault_path, assert args_to_json returns Ok with expected fields
  - boolean_flag_grammar_consistency: for hierarchical/include_subsections, verify --flag true and --flag false both succeed, bare --flag fails (value_parser(bool) not SetTrue)
- Total: ~8 tests (3 panic tests + 3 full-options tests + 2 boolean grammar tests)

### 5. Quality gates
- cargo test -p notectl-outline --all-features
- cargo clippy -p notectl-outline --all-features --all-targets -- -D warnings
- cargo fmt -p notectl-outline -- --check
- cargo build (workspace-wide sanity check)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed all three operations (GetOutlineOperation, GetSectionOperation, SearchHeadingsOperation) in notectl-outline/src/capability.rs by replacing Request::from_arg_matches() with field-by-field serde_json::Map construction in args_to_json(). This prevents the 'Mismatch between definition and access of vault_path' panic when called from get_remote_command. Added 10 new tests in remote_command_tests module mirroring notectl-search's tests from TASK-14.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replaced panic-prone from_arg_matches calls with field-by-field JSON construction in args_to_json() for all 3 outline operations. Added 10 regression tests. All quality gates pass (tests, clippy, fmt, pre-push hooks).
<!-- SECTION:FINAL_SUMMARY:END -->
