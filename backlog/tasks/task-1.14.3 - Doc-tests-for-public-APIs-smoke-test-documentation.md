---
id: TASK-1.14.3
title: Doc-tests for public APIs + smoke test documentation
status: Done
assignee: []
created_date: '2026-07-17 00:30'
updated_date: '2026-07-17 00:48'
labels: []
dependencies: []
parent_task_id: TASK-1.14
priority: medium
type: task
ordinal: 14300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add 5 runnable doc-tests to key public functions and document the manual smoke test procedure.

### Doc-tests (5)

Add `/// ``` ... /// ```` examples with `fn main()` to:

1. **`Bm25Indexer::tokenize`** — show tokenization of mixed punctuation text
   ```rust
   let tokens = Bm25Indexer::tokenize("Hello, World!");
   assert_eq!(tokens, vec!["hello", "world"]);
   ```

2. **`SparseIndexer::index_chunks`** — show indexing + scoring round-trip with a single chunk

3. **`cosine_top_k`** — show exact match vs orthogonal vectors returning different scores

4. **`rrf_fuse`** — show two ranked lists being fused into one

5. **`normalize_embedding`** — show truncation + L2 normalization on a small vector

Each example must compile as a standalone doctest (`cargo test --doc`).

### Smoke test documentation

Add a "Smoke Test" section to `notectl-search/README.md` (create if needed):

```markdown
## Smoke Test

```bash
# Index a vault
cargo run --features search -- index /path/to/vault

# Search
cargo run --features search -- search /path/to/vault "your query"

# Verify JSON output
cargo run --features search -- search /path/to/vault "query" | jq '.results[0]'
```
```

### Acceptance Criteria
- [ ] `cargo test -p notectl-search --doc` passes (5 new tests)
- [ ] No regressions in existing test suite
- [ ] Smoke test procedure documented in crate README or AGENTS.md
<!-- SECTION:DESCRIPTION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Already implemented as part of parent TASK-1.14's commit 7143e81 (test(notectl-search): add unit tests, doc-tests, and smoke test docs) -- all 5 doc-tests listed in this subtask's description (Bm25Indexer::tokenize, SparseIndexer::index_chunks, cosine_top_k, rrf_fuse, normalize_embedding) exist verbatim and pass via 'cargo test --doc'. notectl-search/README.md's Smoke Test section was also added, matching this subtask's description. NOTE: the smoke test commands as documented (and as embedded verbatim in this subtask's own description) are missing '--bin notectl' and fail with an ambiguous-binary error since the workspace also has a notectl-remote binary; that gap is now tracked separately as TASK-25. This subtask was left in To Do by mistake when the parent was marked Done; closing to prevent duplicate/conflicting re-implementation on a future pi round.
<!-- SECTION:FINAL_SUMMARY:END -->
