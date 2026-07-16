---
id: TASK-18
title: >-
  Fix: vault_path panic risk in notectl-files ReadFilesOperation's
  get_remote_command/args_to_json
status: To Do
assignee: []
created_date: '2026-07-16 16:52'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-14
priority: high
type: bug
ordinal: 101
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-14 (notectl-search/src/capability.rs), which fixed the identical bug for notectl-search and flagged in a code comment that the same defect exists in other capability files, deferring the fix as out of scope. Investigation confirms notectl-files' ReadFilesOperation is affected: ReadFilesOperation::get_remote_command (notectl-files/src/capability.rs ~line 435-452) manually rebuilds a clap::Command from scratch that omits the vault_path arg entirely, while ReadFilesOperation::args_to_json (~line 489-493) calls ReadFilesRequest::from_arg_matches(matches) unconditionally -- this panics with a 'Mismatch between definition and access' error when vault_path was never registered in the hand-built Command, per the identical live-reproduced panic pattern documented in TASK-14. Note ListFilesOperation and RecentFilesOperation in the same file are NOT affected: they already use the safe self.get_command().mut_arg("path", |a| a.required(false).hide(true)) pattern (confirmed at notectl-files/src/capability.rs ~line 364-367 and ~853-856). This is a Correctness/Resilience-axis bug: a live, reproducible panic reachable from the notectl-remote CLI boundary the moment read-files is invoked without a local vault_path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ReadFilesOperation::get_remote_command in notectl-files/src/capability.rs uses self.get_command().mut_arg("vault_path", |a| a.required(false).hide(true)) instead of manually rebuilding clap::Command, matching ListFilesOperation and RecentFilesOperation's existing pattern in the same file -- first verify via a throwaway clap probe that ReadFilesRequest's file_paths positional (index 2, following vault_path at index 1) does not hit the 'non-required positional before required positional' clap panic; if it does, keep the manual rebuild but explicitly register a hidden non-required vault_path arg instead
- [ ] #2 ReadFilesOperation::args_to_json is simplified back to ReadFilesRequest::from_arg_matches(matches)? + request.vault_path = None (matching ListFilesOperation/RecentFilesOperation's args_to_json in the same file), unless the clap probe in the previous AC shows this is not possible, in which case keep field-by-field JSON construction
- [ ] #3 A new test exercises ReadFilesOperation::get_remote_command() end-to-end without a vault_path arg present and asserts args_to_json does not panic and produces valid JSON containing file_paths and continue_on_error
- [ ] #4 nix develop -c cargo test -p notectl-files --all-features passes
- [ ] #5 nix develop -c cargo clippy -p notectl-files --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust CLI workspace (notectl-core, notectl-outline, notectl-search, notectl-files, notectl-tags, notectl-tasks, notectl-daily-notes, plus the main notectl binary). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-files/src/capability.rs. Locate ReadFilesOperation's impl block (search for 'impl notectl_core::operation::Operation for ReadFilesOperation', ~line 425-495). Note its get_remote_command (~line 435-452) manually rebuilds clap::Command with a 'file_paths' positional at index 1 and a 'continue_on_error' long flag, omitting vault_path entirely -- compare against ReadFilesRequest's derived grammar where vault_path is index=1 and file_paths is index=2 (check the struct definition near the top of the file for #[arg(index = N)] attributes to confirm).
2. Before changing anything, write a throwaway clap probe (temporary #[test] or scratch binary under /tmp, not committed) reproducing notectl-search's TASK-14 finding: does clap panic with 'Found non-required positional argument with a lower index than a required positional argument' if vault_path (index 1, optional/hidden) precedes file_paths (index 2, required)? This is the same shape as notectl-search's SearchRequest (vault_path index 1, query index 2), which DID hit this panic and required a manual rebuild -- so ReadFilesRequest likely has the same constraint. Confirm empirically before choosing an approach.
3. If the probe shows mut_arg is unsafe (matching the search case), keep get_remote_command as a manual clap::Command rebuild but add vault_path back in as a hidden, non-required arg at index 1 (shifting nothing else), so ReadFilesRequest::from_arg_matches(matches) finds a registered (if empty) vault_path field and does not panic. If the probe shows mut_arg IS safe, replace the manual rebuild with self.get_command().mut_arg("vault_path", |a| a.required(false).hide(true)) and delete the redundant .arg() calls, matching ListFilesOperation's pattern (~line 364-367).
4. Update ReadFilesOperation::args_to_json (~line 489-493) accordingly: if mut_arg was used, simplify to 'let mut request = ReadFilesRequest::from_arg_matches(matches)?; request.vault_path = None; Ok(serde_json::to_value(request)?)'. If the manual-rebuild-with-hidden-arg approach was needed instead, this same from_arg_matches form should still work (since vault_path is now a registered arg id) -- prefer it over notectl-search's more verbose field-by-field workaround unless from_arg_matches still panics for some other reason, in which case fall back to field-by-field JSON construction as SearchOperation did in notectl-search/src/capability.rs (~line 485-503).
5. Add a test in notectl-files/src/capability.rs (new #[cfg(test)] module if none exists near ReadFilesOperation, or extend an existing one) that builds ReadFilesOperation::get_remote_command(), parses args without vault_path (e.g. try_get_matches_from(["read-files", "a.md,b.md", "--continue-on-error", "true"])), calls args_to_json, and asserts it returns Ok with file_paths and continue_on_error present and no panic.
6. Run: nix develop -c cargo test -p notectl-files --all-features
7. Run: nix develop -c cargo clippy -p notectl-files --all-features --all-targets -- -D warnings
8. Run: nix develop -c cargo fmt -p notectl-files -- --check (fix formatting if needed).
9. Run: nix develop -c cargo build (workspace-wide) to confirm nothing else broke.
<!-- SECTION:PLAN:END -->
