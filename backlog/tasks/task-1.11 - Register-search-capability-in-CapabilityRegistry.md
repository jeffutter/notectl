---
id: TASK-1.11
title: Register search capability in CapabilityRegistry
status: To Do
assignee: []
created_date: '2026-07-14 02:22'
labels: []
dependencies:
  - TASK-1.10
parent_task_id: TASK-1
priority: medium
type: task
ordinal: 12000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In src/capabilities/mod.rs: pub use notectl_search::{SearchCapability, IndexOperation, SearchOperation}; add a search_capability field + getter to CapabilityRegistry; add both operations to create_operations() so HTTP and CLI pick them up automatically.
<!-- SECTION:DESCRIPTION:END -->
