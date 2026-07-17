---
id: TASK-19
title: Simplify IndexOperation get_remote_command/args_to_json to use mut_arg pattern
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-16 16:52'
updated_date: '2026-07-17 00:14'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-14
priority: high
type: chore
ordinal: 102
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-14 (notectl-search/src/capability.rs). IndexOperation::get_remote_command (~line 287-309) manually rebuilds a clap::Command field-by-field (reindex, model, dim), and IndexOperation::args_to_json (~line 356-371) manually re-extracts each field from ArgMatches by name/type -- both duplicating knowledge already encoded once in the derived IndexRequest struct (#[derive(clap::Parser)] with #[arg(...)] attributes). This differs from SearchOperation in the same file, which genuinely needs the manual-rebuild approach: SearchRequest's vault_path (index 1) precedes a required query positional (index 2), and empirically (verified via a clap probe during this review) clap_builder panics with 'Found non-required positional argument with a lower index than a required positional argument' if vault_path is added back as a hidden/optional positional ahead of a required one. IndexRequest has NO such constraint -- vault_path is its ONLY positional (reindex/model/dim are all --long options) -- confirmed via the same clap probe that notectl-tags/notectl-daily-notes/notectl-tasks's established pattern (self.get_command().mut_arg("path"/"vault_path", |a| a.required(false).hide(true))) works cleanly for a single-positional operation. This is a Concise/Organized-axis finding: IndexOperation's current approach means a new field added to IndexRequest (e.g. a hypothetical --batch-size) must be manually mirrored in THREE places (the struct, get_remote_command's arg list, and args_to_json's extraction) instead of one, and a missed mirror silently drops the field from the remote/HTTP/MCP path while the local CLI path (which uses IndexRequest::from_arg_matches directly) keeps working -- a footgun for future maintainers. Do NOT apply this simplification to SearchOperation; its manual rebuild is a validated necessity, not duplication for its own sake.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 IndexOperation::get_remote_command in notectl-search/src/capability.rs uses self.get_command().mut_arg("vault_path", |a| a.required(false).hide(true)) instead of manually rebuilding clap::Command with individual .arg() calls for reindex/model/dim
- [x] #2 IndexOperation::args_to_json is simplified to: let mut request = IndexRequest::from_arg_matches(matches)?; request.vault_path = None; Ok(serde_json::to_value(request)?) -- matching the pattern in notectl-tags/src/capability.rs's args_to_json
- [x] #3 SearchOperation::get_remote_command and SearchOperation::args_to_json are left unchanged (they require the manual rebuild due to positional ordering -- do not touch them in this ticket)
- [x] #4 Existing remote_command_tests in notectl-search/src/capability.rs (index_remote_command_reindex_accepts_bool_value, index_remote_command_reindex_bare_flag_fails, index_remote_command_args_to_json_no_vault_path_panic) still pass unmodified against the simplified implementation, proving the externally observable grammar and JSON shape are unchanged
- [x] #5 nix develop -c cargo test -p notectl-search --all-features passes
- [x] #6 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust CLI workspace (notectl-core, notectl-outline, notectl-search, notectl-files, notectl-tags, notectl-tasks, notectl-daily-notes, plus the main notectl binary). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-search/src/capability.rs. Locate IndexOperation::get_remote_command (~line 287-309): it currently rebuilds clap::Command::new("index") from scratch with three .arg() calls for reindex, model, dim.
2. Replace the entire method body with: self.get_command().mut_arg("vault_path", |a| a.required(false).hide(true)) -- this reuses IndexRequest's derived Command (which already has reindex/model/dim correctly declared with value_parser(bool) for reindex per TASK-14's fix) and simply makes the vault_path positional optional and hidden for the remote grammar, exactly matching notectl-tags/src/capability.rs's ExtractTagsOperation::get_remote_command.
3. Locate IndexOperation::args_to_json (~line 356-371): it currently builds a serde_json::Map field-by-field from matches.get_one::<T>(name) calls for reindex/model/dim.
4. Replace the entire method body with: let mut request = IndexRequest::from_arg_matches(matches)?; request.vault_path = None; Ok(serde_json::to_value(request)?) -- this now works safely because step 2 registered vault_path as a real (hidden, optional) arg id, so from_arg_matches finds it as None instead of panicking on an unregistered id.
5. Remove the now-unneeded 'reindex'/'model'/'dim' manual extraction code and any now-unused imports if applicable (check clap::FromArgMatches is still imported/used elsewhere in the file -- it is, via SearchOperation and other operations, so no import changes needed).
6. Leave SearchOperation::get_remote_command and SearchOperation::args_to_json completely untouched -- they are correct as-is due to the positional-ordering constraint documented in this ticket's description.
7. Run the existing remote_command_tests module (index_remote_command_reindex_accepts_bool_value, index_remote_command_reindex_bare_flag_fails, index_remote_command_args_to_json_no_vault_path_panic) and confirm they still pass without modification -- these tests assert on ArgMatches values and JSON output shape, which should be identical whether the Command is hand-built or derived+mut_arg'd, since the same 'reindex'/'model'/'dim' arg ids and types are ultimately registered either way.
8. Run: nix develop -c cargo test -p notectl-search --all-features
9. Run: nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings
10. Run: nix develop -c cargo fmt -p notectl-search -- --check (fix formatting if needed).
11. Run: nix develop -c cargo build (workspace-wide) to confirm nothing else broke.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced IndexOperation::get_remote_command manual clap::Command rebuild with self.get_command().mut_arg("vault_path", |a| a.required(false).hide(true)). Replaced IndexOperation::args_to_json field-by-field extraction with IndexRequest::from_arg_matches + vault_path = None. SearchOperation left untouched per AC #3. All 140 tests pass, clippy clean, fmt clean, workspace build clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Simplified IndexOperation::get_remote_command and args_to_json in notectl-search/src/capability.rs to use the mut_arg pattern (matching notectl-tags). Removed 26 lines of duplicated clap Command field-by-field rebuild code. SearchOperation left untouched per requirements. All 140 tests pass, clippy/fmt clean.
<!-- SECTION:FINAL_SUMMARY:END -->
