---
id: mte-r91g
status: open
deps: []
links: []
created: 2026-04-12T03:44:46Z
type: feature
priority: 2
assignee: Jeffery Utter
---
# Implement notectl-remote binary

Add a second bin target src/bin/notectl-remote.rs that sends CLI commands to a remote HTTP server instead of accessing the vault locally. See docs/plans/2026-04-11-notectl-remote-design.md for full design.

