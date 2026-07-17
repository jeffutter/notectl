---
id: TASK-1.14.3
title: 'Doc-tests for public APIs + smoke test documentation'
status: To Do
assignee: []
created_date: '2026-07-17 00:30'
updated_date: '2026-07-17 00:30'
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