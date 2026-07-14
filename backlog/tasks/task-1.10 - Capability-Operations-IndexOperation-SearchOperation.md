---
id: TASK-1.10
title: 'Capability + Operations (IndexOperation, SearchOperation)'
status: To Do
assignee: []
created_date: '2026-07-14 02:22'
updated_date: '2026-07-14 11:12'
labels: []
dependencies:
  - TASK-1.8
  - TASK-1.9
  - TASK-1.21
parent_task_id: TASK-1
priority: high
type: task
ordinal: 11000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add notectl-search/src/capability.rs: SearchCapability, IndexRequest/Response, SearchRequest/Response, IndexOperation, SearchOperation implementing notectl_core::operation::Operation. Mirror notectl-outline/src/capability.rs: get_command/get_remote_command/execute_json/execute_from_args/input_schema/args_to_json, including the vault_path-as-CLI-only-positional convention (throwaway capability built in execute_from_args when a path is given). IndexOperation: CLI 'index', HTTP /api/search/index, args vault_path/--reindex/--model/--dim. SearchOperation: CLI 'search', HTTP /api/search, args vault_path/query/--limit (default 50)/--mode (hybrid|dense|sparse)/--no-reindex.
<!-- SECTION:DESCRIPTION:END -->
