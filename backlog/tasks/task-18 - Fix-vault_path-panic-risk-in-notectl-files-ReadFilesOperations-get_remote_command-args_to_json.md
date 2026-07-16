---
id: TASK-18
title: >-
  Fix: vault_path panic risk in notectl-files ReadFilesOperation's
  get_remote_command/args_to_json
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-16 16:52'
updated_date: '2026-07-16 21:16'
labels:
  - review-followup
  - planned
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
Single-file fix in notectl-files/src/capability.rs. ReadFilesOperation has two positional args (vault_path at index 1, file_paths at index 2), so clap probe is required before choosing approach.

## Root Cause
ReadFilesOperation::get_remote_command() manually rebuilds clap::Command omitting vault_path entirely. When notectl-remote calls args_to_json(sub_matches), it invokes ReadFilesRequest::from_arg_matches(matches) whose derive-generated impl looks up vault_path by ID — panics with "Mismatch between definition and access of vault_path" since the arg was never registered. Additionally, the current manual rebuild shifts file_paths from index 2 to index 1, which would cause a second mismatch if from_arg_matches were ever to succeed.

## Clap Probe (Step 0 — throwaway test, not committed)
Before touching production code, write a small scratch test to determine which fix path is viable:

```rust
// Does mut_arg work when optional positional (index 1) precedes required positional (index 2)?
let cmd = ReadFilesRequest::command()
    .mut_arg("vault_path", |a| a.required(false).hide(true));
// If this debug_assert panics with "non-required positional before required positional",
// then mut_arg is NOT viable → use field-by-field JSON (Path B).
// If it succeeds, mut_arg IS viable → simpler fix (Path A).
```

This mirrors what TASK-14 did for SearchRequest (same shape: vault_path + query as two positionals) which DID hit the panic. However, empirical confirmation is preferred over extrapolation.

## Path A — mut_arg works (preferred if probe passes)
If the probe shows no clap panic:

### 1. ReadFilesOperation::get_remote_command (~line 435-452)
Replace the entire manual rebuild with:
```rust
fn get_remote_command(&self) -> clap::Command {
    self.get_command()
        .mut_arg("vault_path", |a| a.required(false).hide(true))
}
```
This matches ListFilesOperation and RecentFilesOperation in the same file.

### 2. ReadFilesOperation::args_to_json (~line 489-493)
No change needed — the existing form already does the right thing:
```rust
fn args_to_json(&self, matches: &clap::ArgMatches) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let mut request = ReadFilesRequest::from_arg_matches(matches)?;
    request.vault_path = None;
    Ok(serde_json::to_value(request)?)
}
```
This already works because from_arg_matches finds vault_path (now registered but absent → None) and file_paths at its correct index.

## Path B — mut_arg fails due to clap positional constraint (expected based on TASK-14)
If the probe panics, follow the pattern established by TASK-14 (SearchOperation) and TASK-17 (GetOutlineOperation):

### 1. ReadFilesOperation::get_remote_command (~line 435-452)
Keep the manual rebuild but FIX the indexing — keep file_paths at index 2 (matching struct definition) and add vault_path as hidden optional at index 1:
```rust
fn get_remote_command(&self) -> clap::Command {
    clap::Command::new("read-files")
        .about("Read one or more markdown files")
        .arg(
            clap::Arg::new("vault_path")
                .index(1)
                .required(false)
                .hide(true)
                .help("Vault path (CLI only)"),
        )
        .arg(
            clap::Arg::new("file_paths")
                .index(2)
                .required(true)
                .value_delimiter(',')
                .help("Comma-separated file paths relative to vault root"),
        )
        .arg(
            clap::Arg::new("continue_on_error")
                .long("continue-on-error")
                .value_parser(clap::value_parser!(bool))
                .help("Continue reading files even if some fail"),
        )
}
```
Key difference from current code: vault_path is added back at index 1 (hidden, optional), and file_paths stays at index 2 (not shifted to index 1). This lets from_arg_matches find both fields correctly.

### 2. ReadFilesOperation::args_to_json (~line 489-493)
The EXISTING form should now work since vault_path is registered:
```rust
fn args_to_json(&self, matches: &clap::ArgMatches) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let mut request = ReadFilesRequest::from_arg_matches(matches)?;
    request.vault_path = None;
    Ok(serde_json::to_value(request)?)
}
```
NO CHANGE needed if vault_path is added to get_remote_command. If from_arg_matches still fails for any reason, fall back to field-by-field JSON construction matching TASK-14/17 pattern:
```rust
fn args_to_json(&self, matches: &clap::ArgMatches) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let mut obj = serde_json::Map::new();
    if let Some(v) = matches.get_one::<Vec<String>>("file_paths") {
        obj.insert("file_paths".into(), serde_json::json!(v));
    }
    if let Some(v) = matches.get_one::<bool>("continue_on_error") {
        obj.insert("continue_on_error".into(), serde_json::Value::Bool(*v));
    }
    Ok(serde_json::Value::Object(obj))
}
```

## Test (either path)
Add a new #[cfg(test)] module (or extend existing tests) in notectl-files/src/capability.rs:

```rust
#[cfg(test)]
mod remote_command_tests {
    use super::*;
    use notectl_core::operation::Operation;

    fn dummy_capability() -> Arc<FileCapability> {
        Arc::new(FileCapability::new(PathBuf::from("/tmp"), Arc::new(Config::default())))
    }

    #[test]
    fn read_files_remote_command_args_to_json_no_vault_path_panic() {
        let op = ReadFilesOperation::new(dummy_capability());
        let cmd = op.get_remote_command();
        let matches = cmd.try_get_matches_from(["read-files", "a.md,b.md", "--continue-on-error", "true"])
            .expect("should parse remote command args");
        let json = op.args_to_json(&matches)
            .expect("args_to_json must not panic without vault_path");
        // Assert file_paths and continue_on_error are present
        assert!(json.get("file_paths").is_some());
        assert!(json.get("continue_on_error").is_some());
        assert!(json.get("vault_path").is_none());
    }

    #[test]
    fn read_files_remote_command_minimal_args() {
        let op = ReadFilesOperation::new(dummy_capability());
        let cmd = op.get_remote_command();
        let matches = cmd.try_get_matches_from(["read-files", "single.md"])
            .expect("should parse minimal args");
        let json = op.args_to_json(&matches)
            .expect("args_to_json must not panic");
        assert!(json.get("file_paths").is_some());
    }
}
```

## Quality Gates
1. `nix develop -c cargo test -p notectl-files --all-features`
2. `nix develop -c cargo clippy -p notectl-files --all-features --all-targets -- -D warnings`
3. `nix develop -c cargo fmt -p notectl-files -- --check`
4. `nix develop -c cargo build` (workspace-wide sanity check)

## Note on Scope
This ticket fixes ONLY ReadFilesOperation in notectl-files. ListFilesOperation and RecentFilesOperation in the same file already use the safe `mut_arg` pattern and are unaffected. Other capability crates (outline, search) have their own tickets (TASK-17, TASK-14 respectively).
<!-- SECTION:PLAN:END -->
