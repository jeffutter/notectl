---
id: mte-jeur
status: open
deps: []
links: []
created: 2026-02-19T14:52:46Z
type: chore
priority: 2
assignee: Jeffery Utter
tags: [needs-plan]
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

