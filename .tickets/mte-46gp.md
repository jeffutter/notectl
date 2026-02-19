---
id: mte-46gp
status: open
deps: []
links: []
created: 2026-02-19T15:03:35Z
type: task
priority: 2
assignee: Jeffery Utter
parent: mte-jeur
tags: [planned]
---
# Rename Cargo.toml package names and workspace structure

Rename all 7 Cargo.toml files: update package names (markdown-todo-extractor* → notectl*), workspace members list, workspace.dependencies entries, and inter-crate dependency references. Also rename physical directories using git mv. Delete and regenerate Cargo.lock. Verify with cargo check --all.

## Design

1. Update root Cargo.toml: name = 'notectl', update [workspace] members paths, update [workspace.dependencies] entry names, update [dependencies] crate names
2. git mv each of the 6 member crate directories:
   - markdown-todo-extractor-core → notectl-core
   - markdown-todo-extractor-tasks → notectl-tasks
   - markdown-todo-extractor-tags → notectl-tags
   - markdown-todo-extractor-files → notectl-files
   - markdown-todo-extractor-daily-notes → notectl-daily-notes
   - markdown-todo-extractor-outline → notectl-outline
3. Update each crate's Cargo.toml: package name, dependency references to other workspace crates
4. Delete Cargo.lock and run cargo build to regenerate
5. Verify: cargo check --all passes

