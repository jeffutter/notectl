---
id: mte-r91g
status: open
deps: [mte-c5oz, mte-uosl]
links: []
created: 2026-04-12T03:44:46Z
type: feature
priority: 2
assignee: Jeffery Utter
tags: [planned]
---
# Implement notectl-remote binary

Add a second bin target src/bin/notectl-remote.rs that sends CLI commands to a remote HTTP server instead of accessing the vault locally. See docs/plans/2026-04-11-notectl-remote-design.md for full design.

## Design

### Approach

Two-phase implementation:

1. **Foundation (mte-c5oz):** Add `args_to_json()` to the `Operation` trait so CLI arg parsing can be decoupled from execution. Every operation gets a mechanical implementation: parse from ArgMatches, clear CLI-only path field, serialize to JSON.

2. **Binary (mte-uosl):** Create `src/bin/notectl-remote.rs` that builds the same CLI as `notectl` (minus `serve`), but instead of executing locally, POSTs the JSON to `{server}{operation.path()}` via reqwest.

### How sub-tickets fit together

```
mte-c5oz: args_to_json() trait method  (foundation, no external deps)
    ↓
mte-uosl: notectl-remote binary        (uses args_to_json, adds reqwest)
    ↓
mte-r91g: Feature complete             (verify end-to-end)
```

### Integration and verification

After both tasks complete:
1. Build both binaries: `cargo build`
2. Start HTTP server: `notectl serve http /path/to/vault --port 8000`
3. Run remote commands: `notectl-remote --server http://localhost:8000 tasks --status incomplete`
4. Verify output matches local CLI output for the same queries
5. Test error cases: no server flag, unreachable server, non-2xx responses

