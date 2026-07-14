---
id: TASK-1.14
title: 'Tests: unit coverage + gated integration test'
status: To Do
assignee: []
created_date: '2026-07-14 02:22'
labels: []
dependencies:
  - TASK-1.4
  - TASK-1.5
  - TASK-1.6
  - TASK-1.7
parent_task_id: TASK-1
priority: medium
type: task
ordinal: 15000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Unit tests for chunker/store/fusion/sparse that need no model. A #[cfg(feature = "embeddings")] #[ignore] integration test that downloads the model once and embeds a known string, asserting against a reference vector (from text-embeddings-inference) within a tolerance. Manual smoke test: cargo run --features search -- index ~/vault then search ~/vault "query".
<!-- SECTION:DESCRIPTION:END -->
