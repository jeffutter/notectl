---
id: TASK-1.10
title: 'Capability + Operations (IndexOperation, SearchOperation)'
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-14 02:22'
updated_date: '2026-07-16 14:04'
labels:
  - planned
dependencies:
  - TASK-1.8
  - TASK-1.9
  - TASK-1.21
  - TASK-6
parent_task_id: TASK-1
priority: high
type: task
ordinal: 11000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add notectl-search/src/capability.rs: SearchCapability, IndexRequest/Response, SearchRequest/Response, IndexOperation, SearchOperation implementing notectl_core::operation::Operation. Mirror notectl-outline/src/capability.rs: get_command/get_remote_command/execute_json/execute_from_args/input_schema/args_to_json, including the vault_path-as-CLI-only-positional convention (throwaway capability built in execute_from_args when a path is given). IndexOperation: CLI 'index', HTTP /api/search/index, args vault_path/--reindex/--model/--dim. SearchOperation: CLI 'search', HTTP /api/search, args vault_path/query/--limit (default 50)/--mode (hybrid|dense|sparse)/--no-reindex.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Create notectl-search/src/capability.rs with SearchCapability, IndexOperation, SearchOperation implementing Operation trait. Mirror notectl-outline pattern (manual get_remote_command, vault_path CLI-only positional, throwaway capability in execute_from_args).

### Files Modified

**notectl-search/src/lib.rs:**
- Rename existing SearchCapability struct → SearchEngine (avoids naming collision with new capability)
- Add serde::Serialize + schemars::JsonSchema derives to BuildSummary (in index.rs)
- Add schemars::JsonSchema derive to RankedChunk (in lib.rs)
- Add serde::Serialize, Deserialize, JsonSchema derives to SearchMode (in search.rs)
- Add pub mod capability; pub use capability::*;

**notectl-search/src/capability.rs (NEW):**
- Metadata modules: index { DESCRIPTION, CLI_NAME="/api/search/index" }, search { DESCRIPTION, CLI_NAME="search", HTTP_PATH="/api/search" }
- IndexRequest: vault_path (CLI-only positional), --reindex, --model, --dim
- SearchRequest: vault_path (CLI-only positional), query (positional), --limit (default 50), --mode (hybrid|dense|sparse), --no-reindex
- IndexResponse: files_indexed, chunks_produced, has_embeddings, content_hash, duration_ms
- SearchResponse: results (Vec<RankedChunk>), total_count, mode_used
- SearchCapability(base_path, Arc<Config>) wrapping SearchEngine
- Methods: build_index(reindex, model_override, dim_override), search(query, limit, mode, no_reindex)
- IndexOperation + SearchOperation with full Operation trait impls (9 methods each)

### Key Decisions
1. --reindex deletes index artifacts (manifest.json, chunks/, vectors.bin) preserving models/ to avoid redownloads
2. --model/--dim clone config and apply overrides before build_index call
3. SearchCapability uses Arc<Config> consistent with other capabilities
4. No sub-tickets — single cohesive unit, all changes compile together
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented SearchCapability with IndexOperation and SearchOperation in notectl-search/src/capability.rs. Renamed existing SearchCapability to SearchEngine. Added serde/JsonSchema derives to BuildSummary, RankedChunk, SearchMode. Registered behind cfg(feature = "search") in CapabilityRegistry. All 99 tests pass, clippy clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created notectl-search/src/capability.rs with SearchCapability, IndexOperation, and SearchOperation implementing the Operation trait. Renamed existing SearchCapability to SearchEngine to avoid naming collision. Added serde/JsonSchema derives to BuildSummary (index.rs), RankedChunk (lib.rs), and SearchMode (search.rs) for JSON serialization and schema generation. Registered search operations in CapabilityRegistry behind cfg(feature = "search") since notectl-search is an optional workspace dependency.
<!-- SECTION:FINAL_SUMMARY:END -->
