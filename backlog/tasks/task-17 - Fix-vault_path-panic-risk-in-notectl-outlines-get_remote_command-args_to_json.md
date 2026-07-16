---
id: TASK-17
title: >-
  Fix: vault_path panic risk in notectl-outline's
  get_remote_command/args_to_json
status: To Do
assignee: []
created_date: '2026-07-16 16:51'
labels:
  - review-followup
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
SETUP (read first): This is a Rust CLI workspace (notectl-core, notectl-outline, notectl-search, notectl-files, notectl-tags, notectl-tasks, notectl-daily-notes, plus the main notectl binary). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-outline/src/capability.rs. Locate the three Operation impls: GetOutlineOperation (~line 340-417), GetSectionOperation (~line 420-520), SearchHeadingsOperation (~line 522-620 approx -- exact line numbers may have shifted, search for 'impl notectl_core::operation::Operation for').
2. For each, read the local get_command() (derived from the Request struct's #[derive(clap::Parser)]) and the current get_remote_command() (a manually rebuilt clap::Command that omits vault_path). Compare against notectl-tags/src/capability.rs's ExtractTagsOperation::get_remote_command (~line 300): 'self.get_command().mut_arg("path", |a| a.required(false).hide(true))'.
3. Before changing anything, write a throwaway clap probe (a temporary #[test] or a scratch binary under /tmp, NOT committed) to confirm mut_arg works for each operation's specific positional layout: does the operation have any OTHER positional arg after vault_path (e.g. GetOutlineOperation's file_path, GetSectionOperation's file_path + heading, SearchHeadingsOperation's pattern)? Reproduce notectl-search's finding: 'clap panics with debug_assert 'Found non-required positional argument with a lower index than a required positional argument' if an optional/hidden positional precedes a required one.' Check each Request struct's #[arg(index = N, ...)] attributes in notectl-outline/src/capability.rs to determine each operation's positional ordering.
4. For each operation where mut_arg is safe (no required positional follows vault_path, or the local grammar already has vault_path as the ONLY positional with everything else as --long options), replace get_remote_command()'s manual clap::Command rebuild with 'self.get_command().mut_arg("vault_path", |a| a.required(false).hide(true))'. Delete the now-redundant manual .arg(...) calls.
5. Simplify each corresponding args_to_json() back to the simple two-line form already used elsewhere (e.g. notectl-tags/src/capability.rs's args_to_json): 'let mut request = XRequest::from_arg_matches(matches)?; request.vault_path = None; Ok(serde_json::to_value(request)?)'. This works once vault_path is a registered (if hidden/optional) arg id, since from_arg_matches will just find it as None rather than panicking on an unregistered id.
6. For any operation where the probe in step 3 shows mut_arg is NOT safe (a required positional follows vault_path), keep the manual clap::Command rebuild but ensure it explicitly registers vault_path (e.g. as a hidden, non-required Arg at the correct index) so from_arg_matches does not panic, following notectl-search's SearchOperation::get_remote_command pattern in notectl-search/src/capability.rs (~line 397-427) as a reference, including its field-by-field args_to_json workaround if from_arg_matches would still panic for other reasons in that case.
7. Add a  module (or extend an existing tests module) in notectl-outline/src/capability.rs mirroring notectl-search/src/capability.rs's remote_command_tests (~line 510-645, added in TASK-14): construct a dummy capability, call get_remote_command(), try_get_matches_from(...) without any vault_path argument present, call args_to_json(&matches), and assert it returns Ok(...) with the expected fields -- one test per operation (3 total), plus a bare-flag-vs-value test for any boolean args following notectl-search's grammar-consistency pattern if notectl-outline has any Option<bool> fields (hierarchical, include_subsections -- check these already use value_parser(bool) correctly per TASK-14's investigation notes, so this may already be fine and just needs a regression test).
8. Run: nix develop -c cargo test -p notectl-outline --all-features
9. Run: nix develop -c cargo clippy -p notectl-outline --all-features --all-targets -- -D warnings
10. Run: nix develop -c cargo fmt -p notectl-outline -- --check (fix formatting if needed).
11. Run: nix develop -c cargo build (workspace-wide) to confirm nothing else broke.
<!-- SECTION:PLAN:END -->
