# notectl-remote: Remote HTTP CLI Client

## Overview

A second binary `notectl-remote` that exposes the same CLI commands as `notectl` but sends requests to a remote HTTP server instead of accessing the vault locally. Intended for use by LLMs or scripts on machines that don't have direct vault access.

## Motivation

The existing `notectl` binary accesses the vault directly via the filesystem. When an LLM agent needs to query a vault hosted on a remote machine, it connects via MCP (HTTP transport) or could use a CLI client that hits the HTTP API. A separate binary eliminates ambiguity — the LLM only knows one binary with no local fallback possible.

## Binary Shape

A second `[[bin]]` target inside the existing `notectl` crate:

```
src/bin/notectl-remote.rs
```

No new workspace crate needed. Shares all existing dependencies (capability crates, `Operation` trait, request structs, `cli_router`).

## Server URL Configuration

Resolved in order:

1. `--server` / `-s` flag per invocation
2. `NOTECTL_SERVER` environment variable

Hard error at startup if neither is set.

## Request Flow

For each subcommand:

1. Parse args using the same clap definitions as the local CLI (`operation.get_command()`)
2. Serialize parsed args to JSON via a new `Operation::args_to_json()` trait method
3. `POST {server}{operation.path()}` with JSON body using `reqwest`
4. Pretty-print response body to stdout on success
5. Print error body to stderr and exit non-zero on non-2xx

The `serve` operation is excluded (its `path()` returns `""`).

## New Trait Method

Add to `notectl_core::operation::Operation`:

```rust
fn args_to_json(&self, matches: &clap::ArgMatches) -> Result<serde_json::Value, Box<dyn Error>>;
```

Each capability implements it by parsing its request struct from `ArgMatches` and serializing to JSON. This separates "parse args" from "execute" so the remote client can reuse the parsing logic without running the capability.

## Dependencies

- `reqwest` (with `json` and `rustls-tls` features) added to the main `notectl` crate only — not workspace-wide.

## Output Format

- Success: pretty-printed JSON to stdout (same as local CLI)
- Error: error message to stderr, non-zero exit code

## Authentication

None for now. To be added later when server-side auth is implemented.

## Usage

```bash
# With flag
notectl-remote tasks --server http://host:8000 --status incomplete

# With env var
export NOTECTL_SERVER=http://host:8000
notectl-remote tasks --status incomplete
notectl-remote list-tags --min-count 2
```
