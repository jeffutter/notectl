---
id: TASK-1.16
title: embed_batch pairs texts with wrong titles across batches
status: To Do
assignee: []
created_date: '2026-07-14 11:11'
labels: []
dependencies:
  - TASK-1.2
parent_task_id: TASK-1
priority: high
type: bug
ordinal: 17000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Embedder::embed_batch` in notectl-search/src/embeddings/embed.rs (around lines 194-225) is supposed to pair each text with its own title before applying the retrieval-document prompt prefix. The per-batch title slice is computed as `titles[texts.len() - texts.len() + chunk.len() - chunk.len()..texts.len()]`, which algebraically always evaluates to `titles[0..texts.len()]` — the entire titles vector — regardless of which batch of the `for chunk in texts.chunks(self.config.batch_size)` loop is being processed. Inside the loop, `title_chunk.get(i)` is then indexed by the chunk-local loop counter `i` (which restarts at 0 every batch), so every batch after the first is embedded with `titles[0..chunk.len()]` — the titles belonging to the *first* batch — instead of its own titles.

Since `TaskType::apply_prefix` bakes the title into the text sent to the model ("title: {title} | text: {content}"), this silently corrupts the semantic content of every embedding beyond the first `batch_size` (default 32) chunks in any indexing run, with no error or panic to surface it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 embed_batch pairs each text with its own corresponding title, verified for input larger than one batch (texts.len() > batch_size)
- [ ] #2 Add a unit test with more texts/titles than batch_size that asserts the correct title is used for chunks past the first batch
- [ ] #3 No regression for the single-batch case
<!-- AC:END -->
