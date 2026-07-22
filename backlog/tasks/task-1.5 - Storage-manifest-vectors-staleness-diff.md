---
id: TASK-1.5
title: 'Storage: manifest + vectors + staleness diff'
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-14 02:21'
updated_date: '2026-07-15 16:19'
labels:
  - planned
dependencies:
  - TASK-1.2
  - TASK-1.17
  - TASK-1.21
parent_task_id: TASK-1
priority: high
type: task
ordinal: 6000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add notectl-search/src/store.rs: manifest.json (schema version, model id/dim/chunk params, per-file content_hash/mtime/chunk_ids, chunk list with heading_path/line spans), vectors.bin (flat row-major f32), chunk texts. Atomic write (temp file + rename). Staleness diff: mtime pre-check, blake3 hash as source of truth; drop chunks for removed files; force full rebuild on model/dim/param mismatch. Unit-testable with fake vectors, no model needed.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Implementation Plan

### Overview

Rewrite `notectl-search/src/storage.rs` to produce a richer manifest with model/chunking params and per-file metadata, add staleness-diff logic (mtime pre-check → blake3 content hash), use atomic writes for all persistence files, and drop chunks/vectors for removed files. Force full rebuild when model/dim/chunk params change.

### Step 1: Add `blake3` workspace dependency

- Add `blake3 = "1"` to root `Cargo.toml` under `[workspace.dependencies]`
- Add `blake3.workspace = true` to `notectl-search/Cargo.toml` (unconditional — needed for hashing even without embeddings)

### Step 2: Redesign `SearchManifest` struct

Replace the current minimal manifest with a richer schema:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchManifest {
    pub version: u32,                        // format version (bump to 2)
    pub model_id: String,                   // e.g. "google/embedding-gemma-300m"
    pub embedding_dim: u32,                 // for matryoshka truncation
    pub chunk_config: ChunkConfigSnapshot,   // max_tokens, overlap, min_chunk, merge_threshold
    pub files: Vec<FileInfo>,               // per-file metadata
    pub chunks: Vec<ChunkEntry>,            // chunk list with heading_path, line spans
    pub content_hash: String,               // blake3 hex of combined file hashes
    pub last_indexed: Option<String>,       // RFC3339 timestamp
    pub has_embeddings: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub path: String,                       // relative to base_path
    pub content_hash: String,               // blake3 of file content (hex)
    pub mtime: u64,                         // seconds since epoch
    pub chunk_ids: Vec<String>,             // chunk IDs belonging to this file
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkEntry {
    pub id: String,
    pub source_file: String,
    pub line_start: usize,
    pub line_end: usize,
    pub heading: Option<String>,
    pub heading_path: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkConfigSnapshot {
    pub max_tokens: usize,
    pub overlap_tokens: usize,
    pub min_chunk_size: usize,
    pub merge_threshold: usize,
}
```

Key changes from current code:

- `chunk_count` / `document_count` removed — derivable from `chunks.len()` and `files.len()`
- Per-file `content_hash` enables fine-grained staleness detection (which files changed)
- Chunk entries stored in manifest (not just chunk text files on disk)
- `ChunkConfigSnapshot` enables param-mismatch detection

### Step 3: Atomic write helper

Add a helper using `tempfile::NamedTempFile`:

```rust
fn atomic_write_json(path: &Path, data: &str) -> Result<(), SearchError> {
    let dir = path.parent().unwrap();
    let mut tmp = NamedTempFile::new_in(dir)?;
    tmp.write_all(data.as_bytes())?;
    tmp.flush()?;
    tmp.persist(path)?;
    Ok(())
}
```

Use this for `manifest.json` and `vectors.bin`. For vectors, write to a temp file in the same directory then persist.

### Step 4: Staleness diff logic — `compute_staleness_diff`

Replace `compute_content_hash` / `needs_reindex` with a richer diff function:

```rust
pub fn compute_staleness_diff(
    base_path: &Path,
    config: &Config,
    manifest: &SearchManifest,
) -> Result<StalenessDiff, SearchError>
```

Where `StalenessDiff` has variants:

- `UpToDate` — nothing changed
- `Incremental { added: Vec<PathBuf>, modified: Vec<PathBuf>, removed: Vec<String> }` — some files changed
- `FullRebuild(Reason)` — model/dim/chunk params mismatch, or version mismatch

Algorithm:

1. **Param check first**: Compare manifest.model_id, embedding_dim, chunk_config against current config. Mismatch → `FullRebuild`.
2. **Walk files**: Use `notectl_core::file_walker::collect_markdown_files(base_path, config)` which honors exclusion patterns (critical fix — current code walks everything including `.notectl/`).
3. **mtime pre-check**: For each file, compare mtime against manifest FileInfo. If all mtimes match → `UpToDate`.
4. **blake3 content hash**: For files with changed mtime, compute blake3(content) and compare against stored hash. Only count as modified if hash differs (handles `touch` without content change).
5. **Detect removed files**: Any file in manifest.files not on disk → added to `removed` list.
6. **Compute overall content hash**: blake3 of all per-file hashes (sorted by path) for the manifest-level content_hash.

### Step 5: Rewrite `SearchIndex` methods

- **`open_or_create`**: Same API but deserializes new manifest format. On version mismatch (old v1 → new v2), treat as empty manifest and log a warning.
- **`save_manifest`**: Use atomic write helper.
- **`write_vectors`**: Already fixed by TASK-1.17 (header has count + dim). Add atomic write wrapper.
- **`write_chunks`**: Keep per-file chunk storage but also persist chunk metadata in the manifest chunk list.
- **New `build_index` method**: Orchestrate the full index build:
  1. Compute staleness diff
  2. On full rebuild: clear chunks dir, clear vectors, rebuild everything
  3. On incremental: remove chunks for deleted files, re-chunk modified/added files, append new vectors
  4. Update manifest with new FileInfo entries, chunk list, and overall hash
  5. Persist atomically

### Step 6: Drop removed file chunks

When staleness diff reports removed files:

1. Look up chunk_ids from the old FileInfo
2. Delete `chunks/<safe_id>.txt` for each chunk ID
3. If vectors exist, note which positions are being removed (will need vector rebuild since flat storage has no sparse delete)

For incremental updates with removed chunks, rebuild vectors.bin in manifest-chunk-list order.

### Step 7: Unit tests

Add tests for:

- **Staleness diff — up-to-date**: Create index, verify second call returns UpToDate
- **Staleness diff — modified file**: Change a file, verify Incremental with that file listed
- **Staleness diff — removed file**: Delete a file, verify it appears in removed list
- **Staleness diff — added file**: Add new .md file, verify it appears in added list
- **Full rebuild on param mismatch**: Change model_id or embedding_dim, verify FullRebuild
- **Exclusion filtering**: Create files in excluded paths, verify they are not hashed
- **Atomic write**: Write manifest, verify temp file is cleaned up
- **Vector round-trip with new manifest**: Write vectors, read back, verify equality (already covered by TASK-1.17 test but ensure it still passes)
- **Empty index staleness**: Fresh index with no files returns UpToDate

### File Changes

- `Cargo.toml` — add blake3 to workspace deps
- `notectl-search/Cargo.toml` — add blake3 dependency
- `notectl-search/src/storage.rs` — major rewrite (manifest schema, staleness diff, atomic writes, build orchestration)
<!-- SECTION:PLAN:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-07-14 11:12
---

Review note (branch `embedding` vs `main`): the existing `collect_file_info` helper in storage.rs (used by `compute_content_hash`) walks the whole directory tree with no exclusion filtering, unlike `notectl_core::file_walker::collect_markdown_files` which honors `config.should_exclude()`. When implementing the real staleness-diff/content-hash logic here, make sure excluded paths stay excluded — don't just adapt the existing walker as-is
---

## Implementation Notes

<!-- NOTES:BEGIN -->
Implemented all 7 steps from the plan:

1. **blake3 dependency**: Added to workspace Cargo.toml and notectl-search/Cargo.toml (unconditional, needed for hashing even without embeddings).

2. **Richer manifest schema (v2)**: SearchManifest now includes version, model_id, embedding_dim, chunk_config snapshot, per-file FileInfo (path, content_hash, mtime, chunk_ids), ChunkEntry list (id, source_file, line spans, heading_path), overall content_hash, last_indexed timestamp, and has_embeddings flag. Old v1 manifests are handled gracefully (treated as empty).

3. **Atomic write helper**: atomic_write_json and atomic_write_binary functions use tempfile::NamedTempFile for crash-safe writes.

4. **Staleness diff logic**: compute_staleness_diff implements the full algorithm:
   - Param check first (model_id, embedding_dim, chunk_config mismatch → FullRebuild)
   - Walk files using exclusion-aware collect_markdown_files from core
   - Always hash and compare content (mtime alone is unreliable on fast/test filesystems)
   - Detect removed files (in manifest but not on disk)
   - Compute overall content_hash (blake3 of sorted per-file hashes)

5. **SearchIndex methods**: open_or_create handles version mismatch gracefully, save_manifest uses atomic write, build_index orchestrates the full index build with proper param updates.

6. **Drop removed file chunks**: remove_chunks deletes chunk files for removed files. Full rebuild clears all chunks and vectors.

7. **Unit tests**: 49 tests total, all passing. Covers manifest serialization, open_or_create (new/existing/version mismatch), atomic writes, chunk read/write, staleness diff (up-to-date/modified/removed/added/full-rebuild scenarios), exclusion filtering, empty index, touch-without-content-change, content hash changes, and RFC3339 formatting.

Key fixes applied:

- Fixed version mismatch handling to gracefully handle old v1 manifests that can't deserialize to v2 schema
- Changed staleness detection to always hash content (mtime-only check is unreliable in tests)
- Added param updates (model_id, embedding_dim, chunk_config) to manifest after full rebuild
- Fixed incremental update logging to use actual added/modified counts
- Fixed date algorithm parentheses warnings
<!-- NOTES:END -->
---
<!-- COMMENTS:END -->
