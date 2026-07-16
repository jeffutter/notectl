---
id: TASK-6
title: >-
  Fix: IndexBuilder chunks files by absolute path, leaking local filesystem
  paths into source_file
status: To Do
assignee: []
created_date: '2026-07-16 07:21'
updated_date: '2026-07-16 07:22'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-1.8
priority: high
type: bug
ordinal: 110
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-1.8 (notectl-search/src/index.rs:165, `IndexBuilder::build`). Inside the file-processing loop, the builder computes a vault-relative path (`rel_path`, used for `FileInfo.path`, `FileInfo` is documented "Relative path from base_path") but then calls the chunker with the ABSOLUTE path instead:

```rust
let chunks = self.chunker.chunk_file(abs_path, &content);
```

`Chunker::chunk_file` uses whatever `Path` it's given verbatim to populate `Chunk.source_file` (via `path.to_string_lossy()`) and `Chunk.id` (`format!("{}:{}:{}", path.to_string_lossy(), ...)`). Since `abs_path` is absolute, every `ChunkEntry.source_file` written to the manifest, and every chunk id, ends up containing the full local filesystem path (e.g. "/home/alice/vault/notes/foo.md") instead of the vault-relative path (e.g. "notes/foo.md").

This is inconsistent with `FileInfo.path`, which is the relative path for the same file, and it will leak the user's local absolute filesystem path (home directory name, etc.) into search results once TASK-1.10 (Capability + Operations) surfaces `RankedChunk.source_file` over the CLI/HTTP API. It also makes the manifest/chunk-id scheme non-portable: the same vault indexed from two different absolute mount points, or moved to a different machine, produces entirely different chunk ids for identical content, defeating incremental-update matching that relies on chunk ids being stable.

This is a Correctness/Resilient-axis finding: the manifest's own documentation promises relative paths, and the field silently doesn't deliver that promise for chunk-level data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 IndexBuilder::build (notectl-search/src/index.rs) calls chunker.chunk_file with the vault-relative path (rel_path), not abs_path, so every resulting Chunk.source_file and Chunk.id is relative to base_path
- [ ] #2 A new or updated test in notectl-search/src/index.rs builds an index for a vault at a TempDir path and asserts that manifest.chunks[].source_file does NOT contain the TempDir's absolute path prefix and DOES equal the expected relative path (e.g. "note.md" or "sub/note.md")
- [ ] #3 Existing tests in notectl-search/src/index.rs and notectl-search/src/search.rs continue to pass unmodified in their assertions (source_file.contains("rust") style checks remain valid for relative paths)
- [ ] #4 nix develop -c cargo test -p notectl-search --all-features passes
- [ ] #5 nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust+WebAssembly core (crates/gql-core) with a
TypeScript/React web app (web/). ALL commands must run inside the Nix dev
shell: either run 'direnv allow' once, or prefix every command with
'nix develop -c'. Work from the repository root unless told otherwise. Do not
change pinned dependency versions.

(This repo is notectl; the crate under test is notectl-search. The same Nix-shell rule applies.)

1. Open notectl-search/src/index.rs and find the file-processing loop inside `IndexBuilder::build` (~line 142-179). Note the existing computation:
   ```rust
   let rel_path = abs_path
       .strip_prefix(base_path)
       .unwrap_or(abs_path.as_path())
       .to_string_lossy()
       .to_string();
   ```
   and the buggy call a few lines below (~line 165):
   ```rust
   let chunks = self.chunker.chunk_file(abs_path, &content);
   ```

2. Change the `chunk_file` call to pass a `Path` built from `rel_path` instead of `abs_path`:
   ```rust
   let chunks = self.chunker.chunk_file(Path::new(&rel_path), &content);
   ```
   `std::path::Path` is already imported at the top of index.rs. Do not change `chunker.chunk_file`'s signature — only the call site.

3. Double check `FileInfo` construction later in the same loop still uses `rel_path.clone()` (it already does) — no change needed there.

4. In notectl-search/src/index.rs's `#[cfg(test)] mod tests`, add a new test (e.g. `test_chunk_source_file_is_relative_not_absolute`) that:
   - Creates a TempDir vault with a nested file, e.g. `base.join("sub").join("note.md")`, with enough content to produce at least one chunk.
   - Runs `run_build` (the existing test helper) to build the index.
   - Re-opens the `SearchIndex` (via `SearchIndex::open_or_create` with the same `ChunkConfigSnapshot`/model_id/dim used by `test_config()`) and asserts `index.manifest().chunks` is non-empty, and for every `ChunkEntry`, `entry.source_file` does NOT contain the TempDir's absolute path string (`tmp.path().to_string_lossy()`) and DOES equal `"sub/note.md"` (or the appropriate relative path for the fixture used).

5. Do NOT modify `notectl-search/src/storage.rs`'s old `SearchIndex::build_index` method as part of this ticket — that dead-code duplicate (including its own copy of this same bug) is handled by a separate ticket (TASK-7) which removes it entirely. Touching it here risks a merge conflict with that ticket; leave it untouched.

6. Run `nix develop -c cargo test -p notectl-search --all-features` and confirm all existing tests in `search.rs` that do substring checks like `results[0].source_file.contains("rust")` still pass (they will, since `"rust.md"` is a substring of both `"rust.md"` and `"/abs/path/rust.md"` — but the goal is the actual value is now the relative one).

7. Run `nix develop -c cargo clippy -p notectl-search --all-features --all-targets -- -D warnings` and fix any warnings.
<!-- SECTION:PLAN:END -->
