---
id: mte-jeur
status: open
deps: [mte-46gp, mte-23h3, mte-urmv]
links: []
created: 2026-02-19T14:52:46Z
type: chore
priority: 2
assignee: Jeffery Utter
tags: [planned]
---
# Rename project from markdown-todo-extractor to notectl

Rename all references to the project from 'markdown-todo-extractor' to 'notectl'.

## Scope

- All Cargo.toml package names (root + workspace crates)
- Workspace crate directory names and their internal references
- Rust code: crate names, module paths, use statements
- Config file name: .markdown-todo-extractor.toml → .notectl.toml
- Environment variable: MARKDOWN_TODO_EXTRACTOR_EXCLUDE_PATHS → NOTECTL_EXCLUDE_PATHS
- README.md, CLAUDE.md, CHANGELOG.md, CONTRIBUTING.md
- Any other documentation or comments referencing the old name

## Out of Scope

- GitHub repo rename (user will do manually)
- crates.io publish (separate step)

## Crates to rename

- markdown-todo-extractor (binary) → notectl
- markdown-todo-extractor-core → notectl-core
- markdown-todo-extractor-tasks → notectl-tasks
- markdown-todo-extractor-tags → notectl-tags
- markdown-todo-extractor-files → notectl-files
- markdown-todo-extractor-daily-notes → notectl-daily-notes
- markdown-todo-extractor-outline → notectl-outline

## Acceptance criteria

- cargo build succeeds with no errors
- All references to old name are replaced
- Config file name and env var name updated

## Design

### Overview

This is a mechanical rename across the entire Rust workspace. The project currently has 7 crates (1 binary + 6 libraries) and all need their package names, directory names, Rust module imports, and documentation updated.

### Sub-ticket Breakdown

The work is split into 3 sequential tasks:

1. **mte-46gp** — Rename Cargo.toml package names and workspace structure (FIRST)
   - Physical directory renames via `git mv`
   - All Cargo.toml updates (package names, workspace members, dependencies)
   - Cargo.lock regeneration
   - This must complete before any Rust source or docs work

2. **mte-23h3** — Update Rust source file imports and module references (SECOND, depends on mte-46gp)
   - Update `use` statements: `markdown_todo_extractor_*` → `notectl_*`
   - Update config file lookup: `.markdown-todo-extractor.toml` → `.notectl.toml`
   - Update env var: `MARKDOWN_TODO_EXTRACTOR_EXCLUDE_PATHS` → `NOTECTL_EXCLUDE_PATHS`
   - Files: src/capabilities/mod.rs, src/mcp.rs, src/main.rs, src/cli.rs, src/cli_router.rs, src/http_router.rs

3. **mte-urmv** — Update CI/CD workflow and documentation (can run parallel to mte-23h3 after mte-46gp)
   - .github/workflows/cd.yml: binary name, release artifact glob patterns
   - README.md, CONTRIBUTING.md, CLAUDE.md, CHANGELOG.md

### Key Risks

- **Cargo.lock**: Must be deleted and regenerated after Cargo.toml renames
- **Module path conversion**: Rust uses underscores in module paths (notectl-tasks → notectl_tasks)
- **Config file migration**: Users with `.markdown-todo-extractor.toml` must rename to `.notectl.toml`
- **GitHub repo URLs**: Cargo.toml repository field kept pointing to old GitHub URL (repo rename is out of scope)

### Verification Steps

After all sub-tickets complete:
```bash
cargo check --all      # All 7 crates resolve imports
cargo build --release  # Binary builds as 'notectl'
cargo test --all       # All tests pass
./target/release/notectl --help  # Binary runs
```

### Note on Ticket Filenames

Existing tickets use the `markdown-todo-extractor-*` filename prefix. These are NOT renamed as part of this work — the ticket system tracks work history and renaming would require careful coordination. New tickets going forward will use the `mte-` short prefix (already established).
