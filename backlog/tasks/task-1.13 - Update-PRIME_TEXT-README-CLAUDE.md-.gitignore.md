---
id: TASK-1.13
title: 'Update PRIME_TEXT, README, CLAUDE.md, .gitignore'
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-14 02:22'
updated_date: '2026-07-18 05:42'
labels:
  - planned
dependencies:
  - TASK-1.11
parent_task_id: TASK-1
priority: medium
type: docs
ordinal: 14000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Per CLAUDE.md's mandatory rule, update src/prime.rs PRIME_TEXT for the new index and search commands and their flags. Update README.md, and CLAUDE.md's workspace map + dependency graph to include notectl-search. Add .notectl/ to .gitignore (document that vault users should gitignore their own .notectl/ cache dir too).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Update 4 documentation artifacts to reflect the new notectl-search commands (index, search). No code changes.

### File 1: src/prime.rs (PRIME_TEXT)

Add a "### Search" section before "## General notes" in the format! macro. Two subsections:

**Index command:**
```
\`\`{bin} index{vp}\`\`
  --reindex true|false       force full rebuild (wipe manifest.json, chunks/, vectors.bin; preserve models/)
  --model <id>               override embedding model ID (default: google/embedding-gemma-300m)
  --dim <N>                  override embedding dimension (default: 256)

Examples:
  {bin} index{v}
  {bin} index{v} --reindex true
```

**Search command:**
```
\`\`{bin} search{vp} <query>\`\`
  --limit N                  max results (default 50)
  --mode hybrid|dense|sparse scoring mode (default: hybrid)
  --no-reindex true|false    skip staleness check/rebuild

Output fields per result: id, source_file, score, heading (optional), preview

Examples:
  {bin} search{v} "project timeline"
  {bin} search{v} "deployment steps" --mode dense --limit 10
```

Follow the existing pattern: use `{vp}` for vault_path positionals, `{v}` for examples, match formatting of other capability sections.

### File 2: README.md

Add a "### Search Operations" section after "Document Outline Operations" and before "MCP Server Mode". Include:

- Brief description paragraph
- `notectl index path/to/vault` with `--reindex`, `--model`, `--dim` flags and examples
- `notectl search path/to/vault <query>` with `--limit`, `--mode`, `--no-reindex` flags and examples
- Note about `.notectl/search/` cache directory and that it should be gitignored
- Match the style of existing sections (bash code blocks, flag descriptions inline)

### File 3: AGENTS.md (CLAUDE.md)

Three targeted edits:

1. **Workspace structure**: Add `notectl-search/` block after `notectl-outline/` showing key source files (capability.rs, chunker.rs, index.rs, search.rs, storage.rs, embeddings/)
2. **Dependency graph**: Update binary line to include `+ search (core, files)` or add a separate arrow showing search depends on core
3. **Adding New Features section**: Add a note about `#[cfg(feature = "search")]` conditional compilation used throughout capabilities/mod.rs, mcp.rs, and main.rs. Also document that search MCP tools use manual `with_async_tool` registration rather than `#[tool_router]` macro (per mcp.rs::search_tools module)

### File 4: .gitignore

Append `/.notectl/` to the existing entries. This covers the default cache directory (`.notectl/search/`) where index artifacts (manifest.json, chunks/, vectors.bin, models/) are stored.

### Execution order

Any order works since changes are independent. Suggested: .gitignore → prime.rs → README.md → AGENTS.md (smallest to most complex).

### Verification

- `cargo build --features search` must still compile
- `cargo run --features search -- prime` outputs the new Search section
- `cargo run --features search -- index /tmp/test-vault --help` shows correct flags
- `cargo run --features search -- search /tmp/test-vault "test" --help` shows correct flags
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated 4 documentation artifacts for the notectl-search feature:

1. .gitignore — added /.notectl/ to exclude search cache directory
2. src/prime.rs — added Search section with index/search commands, flags, and examples
3. README.md — added Search Operations section with code examples and gitignore note
4. AGENTS.md — updated workspace structure (notectl-search crate files), dependency graph (+ search?), and documented #[cfg(feature = "search")] conditional compilation across capabilities/mod.rs, mcp.rs, and main.rs
<!-- SECTION:FINAL_SUMMARY:END -->
