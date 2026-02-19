---
id: mte-23h3
status: open
deps: [mte-46gp]
links: []
created: 2026-02-19T15:03:44Z
type: task
priority: 2
assignee: Jeffery Utter
parent: mte-jeur
tags: [planned]
---
# Update Rust source file imports and module references

Update all Rust source files that reference the old crate names with use/extern crate statements. Crate names with hyphens become underscores in Rust module paths (notectl-tasks → notectl_tasks). Files to update: src/capabilities/mod.rs, src/mcp.rs, src/main.rs, src/cli.rs, src/cli_router.rs, src/http_router.rs, plus any source files in sub-crates that cross-reference siblings.

## Design

Pattern change: markdown_todo_extractor_* → notectl_*

Files to update:
- src/capabilities/mod.rs: pub use markdown_todo_extractor_* → pub use notectl_*
- src/mcp.rs: use statements
- src/main.rs: any use statements
- src/cli.rs: any use statements
- src/cli_router.rs: any use statements
- src/http_router.rs: any use statements
- markdown-todo-extractor-daily-notes (now notectl-daily-notes) src: references to notectl-files

Also update config file lookup in notectl-core/src/config.rs:
- .markdown-todo-extractor.toml → .notectl.toml
- MARKDOWN_TODO_EXTRACTOR_EXCLUDE_PATHS env var → NOTECTL_EXCLUDE_PATHS

Run: cargo build to verify

