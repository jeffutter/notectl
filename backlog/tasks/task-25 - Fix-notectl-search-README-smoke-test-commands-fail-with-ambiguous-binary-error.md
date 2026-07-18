---
id: TASK-25
title: >-
  Fix: notectl-search README smoke test commands fail with ambiguous binary
  error
status: Dev Ready
assignee: []
created_date: '2026-07-17 00:47'
updated_date: '2026-07-17 00:55'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-1.14
priority: high
type: docs
ordinal: 100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-1.14 (notectl-search/README.md:22-31, added in commit 7143e81). The documented Smoke Test commands `cargo run --features search -- index /path/to/vault` and `cargo run --features search -- search /path/to/vault "your query here"` fail immediately with `error: could not determine which binary to run. Use the --bin option to specify a binary... available binaries: notectl, notectl-remote` -- the workspace root package defines two binaries (src/main.rs -> notectl, src/bin/notectl-remote.rs -> notectl-remote, added earlier for remote HTTP command dispatch) and cargo cannot disambiguate without --bin. Verified directly: `nix develop -c cargo run --features search -- index <vault>` fails with that error; `nix develop -c cargo run --bin notectl --features search -- index <vault>` succeeds and produces the expected JSON summary. This is a Correct-axis finding: the README instructs any reader to run a command that fails verbatim.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 notectl-search/README.md's Smoke Test section uses 'cargo run --bin notectl --features search -- index ...' and '... search ...' instead of the ambiguous 'cargo run --features search -- ...' form
- [ ] #2 Running the corrected commands from the repo root against a scratch vault (a temp dir with one .md file) succeeds with no 'could not determine which binary to run' error and prints the expected JSON
- [ ] #3 nix develop -c cargo test -p notectl-search --all-features still passes (README-only change, sanity check)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Implementation Plan

**Scope:** Docs-only change — 3 lines in `notectl-search/README.md`.

### Step 1: Fix Smoke Test commands (notectl-search/README.md)

Add `--bin notectl` to all three `cargo run` commands in the Smoke Test section (~lines 25-31):

```diff
-cargo run --features search -- index /path/to/vault
+cargo run --bin notectl --features search -- index /path/to/vault

-cargo run --features search -- search /path/to/vault "your query here"
+cargo run --bin notectl --features search -- search /path/to/vault "your query here"

-cargo run --features search -- search /path/to/vault "query" | jq ".results[0]"
+cargo run --bin notectl --features search -- search /path/to/vault "query" | jq ".results[0]"
```

### Step 2: Verify manually

Create a scratch directory with one `.md` file containing at least a full paragraph of content (short notes produce 0 chunks due to `min_chunk_tokens`), then run from repo root:

```bash
nix develop -c cargo run --bin notectl --features search -- index /tmp/scratch-vault
# Expect: JSON summary with files_indexed > 0, chunks_produced > 0, no error

nix develop -c cargo run --bin notectl --features search -- search /tmp/scratch-vault "some word from note"
# Expect: JSON response with results array, no "could not determine which binary" error
```

### Step 3: Sanity test

```bash
nix develop -c cargo test -p notectl-search --all-features
# Must still pass (docs-only change)
```

### Why this works

When a Cargo package declares multiple `[[bin]]` targets (auto-discovered `src/main.rs` + explicit `src/bin/notectl-remote.rs`), `cargo run` cannot disambiguate without `--bin`. There is no workspace-level `default-run` setting, so `--bin notectl` is the only reliable approach for documented commands.

### Risk assessment

- **Risk:** None. Purely editorial — no code changes.
- **Dependencies:** TASK-1.14 (Done) — the README was created there.
- **Child tickets:** None needed.
<!-- SECTION:PLAN:END -->
