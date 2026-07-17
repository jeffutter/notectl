---
id: TASK-1.12
title: Wire MCP tools for search + build_search_index
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-14 02:22'
updated_date: '2026-07-16 22:02'
labels:
  - planned
dependencies:
  - TASK-1.11
parent_task_id: TASK-1
priority: medium
type: task
ordinal: 13000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In src/mcp.rs add #[tool] methods 'search' and 'build_search_index' delegating to capability_registry.search(), following the existing tool_router pattern. Update the get_info MCP server instructions text to describe the new tools.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Implementation Plan: Wire MCP tools for search + build_search_index

### Overview
Add two `#[tool]` methods to `src/mcp.rs` inside the existing `#[tool_router] impl TaskSearchService` block, gated behind `#[cfg(feature = "search")]`. Update `get_info()` instructions to include the new tool descriptions.

### Files to Modify
- **`src/mcp.rs`** — only file changed

### Step 1: Add imports (top of `mcp.rs`)

Add conditional imports for search types alongside existing imports:

```rust
#[cfg(feature = "search")]
use notectl_search::{IndexRequest, IndexResponse, SearchRequest, SearchResponse};
```

### Step 2: Add `search` MCP tool method

Inside the `#[tool_router] impl TaskSearchService` block, add after the last existing tool (`search_daily_notes`):

```rust
    #[cfg(feature = "search")]
    #[tool(
        description = "Search across all indexed notes using hybrid (dense + sparse), dense-only, or sparse-only scoring. Auto-degrades when vectors are unavailable."
    )]
    async fn search(
        &self,
        Parameters(request): Parameters<SearchRequest>,
    ) -> Result<Json<SearchResponse>, ErrorData> {
        let response = self
            .capability_registry
            .search()
            .do_search(
                &request.query,
                request.limit.unwrap_or(50),
                request.mode.unwrap_or_default(),
                request.no_reindex.unwrap_or(false),
            )
            .await?;
        Ok(Json(response))
    }
```

### Step 3: Add `build_search_index` MCP tool method

After the `search` method:

```rust
    #[cfg(feature = "search")]
    #[tool(
        description = "Build or update the search index for all markdown files in the vault. Computes chunks, optional embeddings, and persists index artifacts."
    )]
    async fn build_search_index(
        &self,
        Parameters(request): Parameters<IndexRequest>,
    ) -> Result<Json<IndexResponse>, ErrorData> {
        let response = self
            .capability_registry
            .search()
            .build_index(
                request.reindex.unwrap_or(false),
                request.model,
                request.dim,
            )
            .await?;
        Ok(Json(response))
    }
```

### Step 4: Update `get_info()` instructions

In the `ServerHandler` impl's `get_info()` method, modify the instructions format string to conditionally include search operation descriptions. Replace the current `format!` block with one that builds instructions dynamically:

```rust
fn get_info(&self) -> ServerInfo {
    let mut instructions = vec![
        "A Markdown task extraction service. Available operations:".to_string(),
        format!("- {}", notectl_tasks::capability::search_tasks::DESCRIPTION),
        format!("- {}", notectl_tags::extract_tags::DESCRIPTION),
        format!("- {}", notectl_tags::list_tags::DESCRIPTION),
        format!("- {}", notectl_tags::search_by_tags::DESCRIPTION),
        format!("- {}", notectl_files::list_files::DESCRIPTION),
        format!("- {}", notectl_files::read_files::DESCRIPTION),
        format!("- {}", notectl_daily_notes::get_daily_note::DESCRIPTION),
        format!("- {}", notectl_daily_notes::search_daily_notes::DESCRIPTION),
    ];

    #[cfg(feature = "search")]
    {
        instructions.push(format!("- {}", notectl_search::index::DESCRIPTION));
        instructions.push(format!("- {}", notectl_search::search::DESCRIPTION));
    }

    ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
        .with_instructions(instructions.join("\n"))
}
```

This approach switches from a single `format!` macro call to building a `Vec<String>` so that `#[cfg]` can be applied cleanly without complex conditional formatting.

### Design Decisions
1. **Inline `#[cfg]` on methods** — simplest approach, consistent with how `CapabilityRegistry` gates search behind the feature flag. No need for separate router merging since these are just 2 methods.
2. **Tool names: `search` and `build_search_index`** — matches CLI command names (`search` and `index`). Using `build_search_index` instead of just `index` avoids potential confusion with other indexing concepts and makes the tool name more descriptive in MCP tool listings.
3. **Descriptions copied from DESCRIPTION constants** — keeps them in sync with HTTP/CLI documentation.
4. **Error handling** — already handled by existing `From<SearchError> for ErrorData` impl in `notectl-search/src/lib.rs`.

### Verification
```bash
# Build with search feature
cargo build --features search

# Verify MCP tools are registered (start server and check)
cargo run --features search -- serve stdio /path/to/vault

# Build without search feature (should still compile)
cargo build
```
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation complete. Added two MCP tools (search, build_search_index) via trait-based AsyncTool pattern since #[tool_router] macro doesn't support #[cfg]-gated methods. Created McpSearchParams/McpIndexParams with all-optional fields (Default required by ToolBase). Updated get_info() instructions to include search tool descriptions when feature is enabled.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added two MCP tools (search, build_search_index) gated behind the 'search' feature flag. Used trait-based AsyncTool pattern since #[tool_router] macro doesn't support conditional methods. Created McpSearchParams/McpIndexParams types with all-optional fields for Default compliance. Updated get_info() instructions to include search tool descriptions when feature is enabled. Both builds (with/without search) compile cleanly with no warnings.
<!-- SECTION:FINAL_SUMMARY:END -->
