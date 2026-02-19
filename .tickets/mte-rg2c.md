---
id: mte-rg2c
status: closed
deps: []
links: []
created: 2026-02-19T01:26:00Z
type: epic
priority: 1
assignee: Jeffery Utter
tags: []
---
# Refactor into workspace

Refactor the project into a rust workspace. The intention here is to create separate libraries for different types of markdown processing, perhaps with a -core library for some common things. I would still like to preserve the overall binary application here as-is (not sure if it makes more sense to put it in a workspace as well). Each sub-library should provide a slice of markdown processing. Ex:
1. Daily note navigation
2. Todo Extraction
3. Tag search
4. File navigation
5. Section extraction

Any common code should go in a -core crate
Each workspace can still implement capabilities

Following this refactor we'll add small binaries for each different area in addition to the one large main binary (with the mcp and all)

# Design

## Target Structure

```
markdown-todo-extractor/                    (workspace root + main binary)
  Cargo.toml                                ([workspace] + [package])
  src/
    main.rs                                 (unchanged, import paths updated)
    mcp.rs                                  (unchanged, import paths updated)
    http_router.rs                          (stays here)
    cli.rs                                  (stays here)
    cli_router.rs                           (stays here)
    capabilities/mod.rs                     (CapabilityRegistry - the integration point)

  markdown-todo-extractor-core/             (shared types, traits, config)
    Cargo.toml
    src/
      lib.rs
      config.rs                             (from src/config.rs)
      error.rs                              (from src/error.rs)
      operation.rs                          (from src/operation.rs, modified)
      file_walker.rs                        (NEW - unified from 3 duplicates)

  markdown-todo-extractor-tasks/
    Cargo.toml
    src/
      lib.rs
      extractor.rs                          (from src/extractor.rs)
      filter.rs                             (from src/filter.rs)
      capability.rs                         (from src/capabilities/tasks.rs)

  markdown-todo-extractor-tags/
    Cargo.toml
    src/
      lib.rs
      tag_extractor.rs                      (from src/tag_extractor.rs)
      capability.rs                         (from src/capabilities/tags.rs)

  markdown-todo-extractor-files/
    Cargo.toml
    src/
      lib.rs
      capability.rs                         (from src/capabilities/files.rs)

  markdown-todo-extractor-daily-notes/
    Cargo.toml
    src/
      lib.rs
      capability.rs                         (from src/capabilities/daily_notes.rs)
      date_utils.rs                         (from src/capabilities/daily_notes/date_utils.rs)
      pattern.rs                            (from src/capabilities/daily_notes/pattern.rs)

  markdown-todo-extractor-outline/
    Cargo.toml
    src/
      lib.rs
      outline_extractor.rs                  (from src/outline_extractor.rs)
      capability.rs                         (from src/capabilities/outline.rs)
```

## Dependency Graph (no cycles)

```
core
  ^
  |--- tasks (core)
  |--- tags (core)
  |--- files (core)
  |--- outline (core)
  |--- daily-notes (core, files)
  |--- binary (core, tasks, tags, files, daily-notes, outline)
```

## Key Design Decisions

### 1. Remove `registry` param from `Operation::execute_from_args`

The `Operation` trait lives in `core` but references `CapabilityRegistry` (which lives in the binary and depends on all domain crates). This creates a circular dependency. Remove the `registry: &CapabilityRegistry` parameter - it is `_registry` (unused) in all 13 implementations. Each operation already holds its own `Arc<XxxCapability>` and creates temporary capabilities when CLI path is provided.

### 2. Core depends on `rmcp`

`CapabilityResult<T>` = `Result<T, rmcp::model::ErrorData>` is used everywhere. Rather than wrapping in an intermediate error type (adding a shallow conversion layer), core depends on `rmcp` directly. The `error.rs` helpers also produce `ErrorData`.

### 3. Unified file walking in core

Three near-identical `collect_markdown_files` implementations exist. Core provides a single `file_walker::collect_markdown_files(dir, config) -> Vec<PathBuf>`. The task extractor keeps its own rayon-based parallel walk since it streams extraction during traversal (different pattern).

### 4. CapabilityRegistry stays in the binary

It's the integration point that wires all capabilities together. It imports from all domain crates. This is appropriate - it belongs where everything comes together.

### 5. `[workspace.dependencies]` for version alignment

All shared dependencies are declared once in the root `Cargo.toml` under `[workspace.dependencies]`, and members use `dep.workspace = true`.

## Implementation Steps

### Step 1: Set up workspace Cargo.toml

Transform root `Cargo.toml` to add `[workspace]` section with members list, `[workspace.package]` for shared metadata, and `[workspace.dependencies]` for all dependencies. Keep the existing `[package]` section for the binary.

### Step 2: Create `markdown-todo-extractor-core`

- Create `markdown-todo-extractor-core/Cargo.toml`
  - Deps: `rmcp`, `clap`, `serde`, `serde_json`, `schemars`, `glob`, `toml`, `async-trait`
- Move `src/config.rs` -> `core/src/config.rs`
- Move `src/error.rs` -> `core/src/error.rs`
- Move `src/operation.rs` -> `core/src/operation.rs`
  - Remove `registry` param from `execute_from_args`
  - Remove `use crate::capabilities::CapabilityRegistry`
  - Move `CapabilityResult<T>` type alias here (from `capabilities/mod.rs`)
- Create `core/src/file_walker.rs` - extract from `tag_extractor.rs::collect_markdown_files`
- Create `core/src/lib.rs` with `pub mod config, error, operation, file_walker`

### Step 3: Create `markdown-todo-extractor-tasks`

- Create `Cargo.toml` - deps: core, `rayon`, `regex`, `serde`, `serde_json`, `simdutf8`, `schemars`, `clap`, `async-trait`
- Move `src/extractor.rs` -> update `Config` import to core
- Move `src/filter.rs` -> update `Task` import (same crate, no change needed)
- Move `src/capabilities/tasks.rs` -> `capability.rs`, update all imports
- `lib.rs`: pub re-export key types (`TaskCapability`, `SearchTasksOperation`, request/response types, `Task`, `FilterOptions`)

### Step 4: Create `markdown-todo-extractor-tags`

- Create `Cargo.toml` - deps: core, `serde`, `serde_json`, `serde_yaml`, `schemars`, `clap`, `async-trait`
- Move `src/tag_extractor.rs` -> replace internal `collect_markdown_files` with `core::file_walker`
- Move `src/capabilities/tags.rs` -> `capability.rs`, update imports
- `lib.rs`: pub re-export key types

### Step 5: Create `markdown-todo-extractor-files`

- Create `Cargo.toml` - deps: core, `serde`, `serde_json`, `schemars`, `clap`, `async-trait`
- Move `src/capabilities/files.rs` -> `capability.rs`
- `lib.rs`: pub re-export `FileCapability`, request/response types (needed by daily-notes)

### Step 6: Create `markdown-todo-extractor-daily-notes`

- Create `Cargo.toml` - deps: core, files, `serde`, `serde_json`, `schemars`, `clap`, `async-trait`
- Move `src/capabilities/daily_notes.rs` -> `capability.rs`
- Move `src/capabilities/daily_notes/date_utils.rs` and `pattern.rs`
- Update `FileCapability` import to come from `markdown-todo-extractor-files`

### Step 7: Create `markdown-todo-extractor-outline`

- Create `Cargo.toml` - deps: core, `regex`, `serde`, `serde_json`, `schemars`, `clap`, `async-trait`
- Move `src/outline_extractor.rs` -> replace internal `collect_markdown_files` with `core::file_walker`
- Move `src/capabilities/outline.rs` -> `capability.rs`

### Step 8: Update root binary package

- Update root `Cargo.toml` `[dependencies]` to include all workspace member crates
- Remove moved files from `src/` (`config.rs`, `error.rs`, `operation.rs`, `extractor.rs`, `filter.rs`, `tag_extractor.rs`, `outline_extractor.rs`)
- Remove `src/capabilities/tasks.rs`, `tags.rs`, `files.rs`, `daily_notes.rs`, `daily_notes/`, `outline.rs`
- Update `src/main.rs` module declarations and imports
- Update `src/capabilities/mod.rs` to import from domain crates
- Update `src/mcp.rs` imports (capability types and request/response types)
- Update `src/cli_router.rs` to remove `registry` param from `execute_cli`
- Update `src/cli.rs` `ServeOperation` to match new `execute_from_args` signature
- Update `src/http_router.rs` imports if needed

### Step 9: Update CLAUDE.md

Update the Architecture section to reflect the workspace structure.

## Verification

```bash
cargo build                # Build entire workspace
cargo test                 # Run all tests
cargo run -- search-tasks /path/to/vault --status incomplete
cargo run -- list-tags /path/to/vault
cargo run -- list-files /path/to/vault
cargo run -- get-outline /path/to/vault/note.md
cargo run -- get-daily-note /path/to/vault --date 2025-01-01
cargo run -- serve stdio /path/to/vault
cargo build --release
```
