---
id: TASK-1.12
title: Wire MCP tools for search + build_search_index
status: To Do
assignee: []
created_date: '2026-07-14 02:22'
labels: []
dependencies:
  - TASK-1.11
parent_task_id: TASK-1
priority: medium
type: task
ordinal: 13000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In src/mcp.rs add #[tool] methods 'search' and 'build_search_index' delegating to capability_registry.search(), following the existing tool_router pattern. Update the get_info MCP server instructions text to describe the new tools.
<!-- SECTION:DESCRIPTION:END -->
