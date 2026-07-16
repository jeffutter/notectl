---
id: TASK-14
title: >-
  Fix: get_remote_command() boolean-flag grammar mismatch and vault_path-missing
  panic risk for IndexOperation/SearchOperation
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-16 14:39'
updated_date: '2026-07-16 16:29'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-1.10
priority: high
type: bug
ordinal: 120
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-1.10 (notectl-search/src/capability.rs). Two related defects in get_remote_command() for both new operations: (1) IndexOperation::get_remote_command (line ~284-289) and SearchOperation::get_remote_command (line ~399-403) declare reindex/no_reindex as clap::ArgAction::SetTrue bare flags, while the corresponding IndexRequest.reindex/SearchRequest.no_reindex fields are Option<bool> which clap-derive's get_command() treats as value-required (--reindex true), not a bare flag -- confirmed with a clap 4.6 repro. No other capability file in the workspace uses SetTrue for an Option<bool> field; outline/tags/files all correctly use value_parser(bool) in both get_command and get_remote_command. (2) Both get_remote_command impls omit the vault_path arg entirely, but args_to_json calls Request::from_arg_matches(matches), whose derive-generated impl unconditionally looks up vault_path by id -- this panics when matches come from get_remote_command. Reproduced live: building notectl-remote and running './target/debug/notectl-remote --server http://localhost:1 outline foo.md' panics today with 'Mismatch between definition and access of vault_path' via the identical pre-existing pattern in notectl-outline. This confirms notectl-search's new capability.rs inherited a real, systemic crash risk that will trigger the moment search operations are registered in notectl-remote.rs's create_operations() (they are not yet). Both are Correctness-axis bugs: a CLI grammar inconsistency between local and remote command surfaces, and a live, reproducible panic reachable from a CLI boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 IndexOperation::get_remote_command's reindex arg and SearchOperation::get_remote_command's no_reindex arg use value_parser(clap::value_parser!(bool)) instead of ArgAction::SetTrue, matching get_command()'s derived grammar
- [ ] #2 Parsing args_to_json's input ArgMatches (as produced by get_remote_command(), which omits vault_path) via IndexRequest::from_arg_matches / SearchRequest::from_arg_matches no longer panics
- [ ] #3 A new test exercises get_remote_command() end-to-end for both operations (parse, then args_to_json) without a vault_path arg present, and asserts no panic and valid JSON output
- [ ] #4 nix develop -c cargo test -p notectl-search --all-features passes
- [ ] #5 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust+WebAssembly core (crates/gql-core) with a TypeScript/React web app (web/). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-search/src/capability.rs. Locate `IndexOperation::get_remote_command` (~line 280-302) and `SearchOperation::get_remote_command` (~line 375-405).
2. Fix the `reindex` arg (IndexOperation::get_remote_command, ~line 284-289): change `.action(clap::ArgAction::SetTrue)` to `.value_parser(clap::value_parser!(bool))`, matching how `IndexRequest.reindex: Option<bool>` is parsed by the derived `get_command()` (clap-derive's default action for `Option<bool>` fields requires an explicit `true`/`false` value — confirmed empirically). This also matches the established convention in notectl-outline/src/capability.rs's `get_remote_command` (e.g. its `hierarchical` arg uses `.value_parser(clap::value_parser!(bool))`, never `SetTrue`).
3. Apply the same fix to the `no_reindex` arg in `SearchOperation::get_remote_command` (~line 399-403).
4. Separately, both `get_remote_command` implementations omit the `vault_path` positional entirely (by design — it's CLI-only local, per the "Rebuild without the vault_path positional" comment), but `IndexRequest`/`SearchRequest` declare `vault_path` with `#[arg(index = 1, required = true, ...)]`. When `notectl-remote` (src/bin/notectl-remote.rs) parses args against `get_remote_command()` and later calls `args_to_json(sub_matches)`, `args_to_json` does `Request::from_arg_matches(matches)` — the clap-derive-generated `FromArgMatches` impl unconditionally looks up `vault_path` by id, which was never registered in `get_remote_command()`'s hand-built Command, and this panics ("Mismatch between definition and access of `vault_path`"). Reproduce this by building `notectl-remote` (`nix develop -c cargo build --bin notectl-remote`) and running `./target/debug/notectl-remote --server http://localhost:1 outline foo.md` — the same panic is already live today via notectl-outline's identical pattern, confirming this is a real, systemic risk that notectl-search's new capability.rs inherits verbatim. Fix it for search specifically: add a hidden/optional `vault_path` arg to both `get_remote_command()` implementations (e.g. `.hide(true)` with no `required(true)`, so `from_arg_matches` finds an arg named `vault_path` that is simply absent/None rather than undefined) OR rewrite `IndexOperation::args_to_json`/`SearchOperation::args_to_json` (capability.rs ~347-354, ~461-467) to build the JSON value field-by-field from `matches` directly instead of routing through `Request::from_arg_matches`, so it no longer depends on an arg id that get_remote_command doesn't define. Prefer the "add a hidden optional vault_path arg" approach — it is the smaller, more consistent fix and requires no change to args_to_json's shape.
5. Note in a code comment above both get_remote_command fns that this same panic-on-missing-arg risk applies to every other capability file (notectl-outline, notectl-tags, notectl-files, notectl-daily-notes, notectl-tasks) since they all follow the identical pattern — flag this for a possible follow-up systemic ticket, but do not attempt to fix those other files as part of this ticket (out of scope; keep this fix limited to notectl-search).
6. Add or update a test that exercises get_remote_command() argument parsing end-to-end for both operations: parse `IndexOperation::default().get_remote_command()` (via `try_get_matches_from`) against `["index", "--reindex", "true"]` and against `["index", "--reindex"]` alone (confirm the bare-flag form now correctly fails, or the value form succeeds, whichever the fix settles on — the point is the local (`get_command()`) and remote (`get_remote_command()`) grammars now agree). Also parse `["search", "some query"]` (no vault_path) through `args_to_json` and assert it does NOT panic and produces valid JSON without a vault_path key.
7. Run: nix develop -c cargo test -p notectl-search --all-features
8. Run: nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings
9. Run: nix develop -c cargo fmt -p notectl-search -- --check (fix formatting if needed).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed both bugs in notectl-search/src/capability.rs:

1. Boolean flag grammar mismatch: Changed IndexOperation::get_remote_command's reindex arg and SearchOperation::get_remote_command's no_reindex arg from ArgAction::SetTrue to value_parser(value_parser!(bool)), matching the derived get_command() grammar for Option<bool> fields. This aligns with the established convention in other capability files (outline, tags, files).

2. vault_path panic risk: Rewrote both args_to_json methods to build JSON field-by-field from matches directly instead of routing through Request::from_arg_matches, which panics when vault_path is absent from get_remote_command's hand-built Command. This avoids adding a hidden optional vault_path positional (which clap rejects before required positionals in the search case).

Added 6 tests in a new remote_command_tests module exercising both operations end-to-end. Added code comments noting this systemic risk exists across all capability files.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed two defects in notectl-search/src/capability.rs: (1) Changed get_remote_command boolean args from ArgAction::SetTrue to value_parser(bool) for grammar consistency with derived get_command(), and (2) rewrote both args_to_json methods to build JSON field-by-field instead of panicking on missing vault_path via from_arg_matches. Added 6 tests covering both operations. All quality gates pass (tests, clippy, fmt).
<!-- SECTION:FINAL_SUMMARY:END -->
