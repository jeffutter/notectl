//! End-to-end contract test for the MCP stdio server.
//!
//! Registering a tool via `.with_async_tool::<T>()` in `TaskSearchService::new`
//! and describing it in `get_info()` is not enough to guarantee it is actually
//! reachable: rmcp's `#[tool_handler]` macro defaults its `list_tools`/`call_tool`
//! dispatch to a *freshly rebuilt* `Self::tool_router()`, silently ignoring the
//! `self.tool_router` field unless the macro is told `router = self.tool_router.clone()`.
//! A tool can therefore compile, appear in the human-readable instructions, and
//! still be completely unreachable at runtime. This test drives the real binary
//! over stdio and checks the live `tools/list`/`tools/call` results, not just that
//! the source registers the tool.

use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

const EXPECTED_TOOLS: &[&str] = &[
    "search",
    "build_search_index",
    "search_tasks",
    "extract_tags",
    "list_tags",
    "search_by_tags",
    "list_files",
    "read_files",
    "recent_files",
    "get_daily_note",
    "search_daily_notes",
];

fn send(stdin: &mut impl Write, value: &Value) {
    let mut line = serde_json::to_string(value).unwrap();
    line.push('\n');
    stdin.write_all(line.as_bytes()).unwrap();
    stdin.flush().unwrap();
}

fn recv_response(rx: &mpsc::Receiver<Value>, id: i64) -> Value {
    loop {
        let value = rx
            .recv_timeout(Duration::from_secs(15))
            .unwrap_or_else(|_| panic!("timed out waiting for response id={id}"));
        if value.get("id").and_then(Value::as_i64) == Some(id) {
            return value;
        }
    }
}

#[test]
fn mcp_stdio_exposes_and_dispatches_all_registered_tools() {
    let vault = tempfile::tempdir().unwrap();
    std::fs::write(
        vault.path().join("note.md"),
        "---\ntitle: test\n---\nhello world\n",
    )
    .unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_notectl"))
        .args(["serve", "stdio", vault.path().to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn notectl");

    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if let Ok(value) = serde_json::from_str::<Value>(&line)
                && tx.send(value).is_err()
            {
                break;
            }
        }
    });

    send(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcp_tools_test", "version": "1.0"}
            }
        }),
    );
    recv_response(&rx, 1);

    send(
        &mut stdin,
        &json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
    );

    send(
        &mut stdin,
        &json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    );
    let list_response = recv_response(&rx, 2);
    let listed_names: Vec<String> = list_response["result"]["tools"]
        .as_array()
        .expect("tools/list must return a tools array")
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_string())
        .collect();

    for expected in EXPECTED_TOOLS {
        assert!(
            listed_names.iter().any(|n| n == expected),
            "tool `{expected}` registered in TaskSearchService::new is missing from a live \
             tools/list response (only saw: {listed_names:?}) — a tool can be wired into the \
             router and described in get_info() instructions yet still be unreachable"
        );
    }

    // Prove `search` isn't just listed but actually dispatchable, not "tool not found".
    send(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "hello", "mode": "sparse"}}
        }),
    );
    let call_response = recv_response(&rx, 3);
    assert!(
        call_response.get("error").is_none(),
        "tools/call for `search` returned an error even though it appeared in tools/list: {call_response:?}"
    );

    let _ = child.kill();
    let _ = child.wait();
}
