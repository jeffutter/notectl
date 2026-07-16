---
id: TASK-1.11
title: Register search capability in CapabilityRegistry
status: Done
assignee: []
created_date: '2026-07-14 02:22'
updated_date: '2026-07-16 14:10'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
No new code required. The CapabilityRegistry registration described in this ticket was already implemented as part of TASK-1.10's scope. Verified: re-exports (line 14-15), struct field (28-29), getter (75-77), operations in create_operations() (94-97), and construction in new() (50-51) are all present behind cfg(feature = "search"). cargo check --features search passes clean.
<!-- SECTION:PLAN:END -->
