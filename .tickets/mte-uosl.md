---
id: mte-uosl
status: open
deps: [mte-c5oz]
links: []
created: 2026-04-12T03:51:49Z
type: task
priority: 2
assignee: Jeffery Utter
parent: mte-r91g
---
# Create notectl-remote binary

Create src/bin/notectl-remote.rs - a CLI binary that sends commands to a remote HTTP server instead of executing locally. Reuses existing Operation trait for arg parsing and args_to_json() for serialization. POSTs JSON to the operation's HTTP path on the remote server via reqwest. Excludes ServeOperation (path returns empty string). Server URL from --server flag or NOTECTL_SERVER env var.

