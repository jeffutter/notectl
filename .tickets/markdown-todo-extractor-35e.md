---
id: markdown-todo-extractor-35e
status: closed
deps: []
links: []
created: 2026-01-21T20:52:03.622063164-06:00
type: feature
priority: 2
tags: ["planned"]
---
# Time-based queries capability

Query notes based on file system timestamps (created, modified). Methods: recently_modified(limit, days), created_in_range(start, end), activity_timeline().

## Design

### Implementation Plan: Time-Based Queries Capability

#### Overview

Add a new `TimeQueryCapability` to query markdown files by filesystem timestamps (modified, created). Expose three operations via HTTP, CLI, and MCP:
1. `recently-modified` - Get files modified in last N days
2. `created-in-range` - Get files created between two dates
3. `activity-timeline` - Get aggregated daily activity counts

#### Architecture

Following the established capability-based architecture:
- Single capability struct with three operation methods
- Three operation wrappers implementing the `Operation` trait
- Automatic registration for HTTP, CLI, and MCP interfaces
- Uses `chrono` for date/time handling
- Follows patterns from `TagCapability` (3 operations) and `FileCapability` (traversal/security)

#### Key Design Decisions

### 1. Three Separate Operations (not a unified query operation)
- **Rationale**: Follows existing patterns (tags has 3 operations, files has 2)
- Each operation has distinct use cases with minimal parameter overlap
- Simpler, more discoverable APIs

### 2. Date/Time Library: chrono
- Industry standard for Rust date/time handling
- Excellent YYYY-MM-DD parsing support
- RFC 3339 output for JSON serialization
- Already in dependency tree

### 3. Creation Time Handling
- **Challenge**: Creation time not available on all filesystems (notably Linux ext4)
- **Solution**: Gracefully handle with `created: Option<String>`
- Track `unavailable_count` in responses where relevant
- Document limitation in API descriptions

### 4. Sorting Strategy
- Default: Most recent first (intuitive for "recent" queries)
- `created_in_range` supports sort parameter: "newest" or "oldest"

#### Implementation Steps

### Step 1: Add Dependencies
**File**: `Cargo.toml`
```toml
[dependencies]
chrono = { version = "0.4", features = ["serde"] }
```

### Step 2: Create Capability Module
**File**: `src/capabilities/time_queries.rs` (NEW, ~600-700 lines)

Structure:
1. **Metadata modules** (3 modules with DESCRIPTION, CLI_NAME, HTTP_PATH constants)
2. **Request/Response structs** (6 structs total, all with `#[derive(Parser, Serialize, Deserialize, JsonSchema)]`)
3. **TimeQueryCapability struct** with three async methods
4. **Helper functions** (`collect_file_metadata`, `extract_file_metadata`, `system_time_to_string`, `parse_date`)
5. **Three Operation structs** implementing `Operation` trait

Key data structures:
- FileMetadata: path, name, size_bytes, modified (RFC 3339), created (Optional RFC 3339)
- RecentlyModifiedRequest: path (CLI only), days (default 7), limit (default 50)
- CreatedInRangeRequest: path (CLI only), start_date, end_date, limit, sort
- ActivityTimelineRequest: path (CLI only), days (default 30)
- DayActivity: date (YYYY-MM-DD), modified_count, created_count

### Step 3: File Scanning Implementation

**Pattern**: Reuse `FileCapability` traversal logic adapted for metadata collection
- Check exclusions via config.should_exclude()
- Skip hidden files (starting with '.')
- Only process .md files
- Recurse into directories
- Extract metadata via std::fs::metadata()

**Security**: Follow `FileCapability` patterns:
- Canonicalize paths before use
- Validate paths stay within base directory
- Respect config exclusions
- Skip hidden files

### Step 4: Operation Implementations

#### Recently Modified Algorithm
1. Collect all `.md` file metadata
2. Calculate cutoff: `now - days * 24h` using chrono
3. Filter where `modified >= cutoff`
4. Sort by modified (most recent first)
5. Apply limit, track truncation

#### Created in Range Algorithm
1. Parse start/end dates (YYYY-MM-DD → `NaiveDate`)
2. Validate range (start <= end)
3. Collect all file metadata
4. Filter by creation date, track `unavailable_count` for files without creation time
5. Sort by created time (respecting sort parameter)
6. Apply limit

#### Activity Timeline Algorithm
1. Calculate date range: `now - days` to `now`
2. Initialize HashMap with all dates (zero counts)
3. Collect all file metadata
4. Bucket files by date for both modified and created
5. Convert to sorted vector (most recent first)

### Step 5: Register in Capability Registry
**File**: `src/capabilities/mod.rs`

Changes:
1. Add `pub mod time_queries;`
2. Import: `use self::time_queries::TimeQueryCapability;`
3. Add field: `time_query_capability: Arc<TimeQueryCapability>`
4. Initialize in `new()`: `time_query_capability: Arc::new(TimeQueryCapability::new(...))`
5. Add getter: `pub fn time_queries(&self) -> Arc<TimeQueryCapability>`
6. Add 3 operations to `create_operations()`

### Step 6: Build and Test
1. `cargo build` - verify compilation
2. Run unit tests (if implemented)
3. Manual CLI testing
4. Manual HTTP testing (if running server)
5. Manual MCP testing (via inspector)

#### Critical Files

### Files to Create
- `src/capabilities/time_queries.rs` (NEW, ~600-700 lines)

### Files to Modify
- `Cargo.toml` - Add chrono dependency
- `src/capabilities/mod.rs` - Register capability and operations

### Reference Files (read-only)
- `src/capabilities/files.rs` - Security patterns, traversal logic
- `src/capabilities/tags.rs` - Multi-operation capability pattern
- `src/operation.rs` - Operation trait definition
- `src/http_router.rs` - `execute_json_operation` helper
- `src/error.rs` - Error handling (`internal_error`, `invalid_params`)

#### Error Handling

Follow existing patterns:
- `invalid_params()` for invalid dates, invalid date ranges
- `internal_error()` for I/O errors, path resolution failures
- Continue on individual file errors (don't fail entire scan)
- Document filesystem limitations in responses

#### Testing Strategy

### Manual CLI Testing
```bash
cargo run -- recently-modified /path/to/vault --days 7 --limit 10
cargo run -- created-in-range /path/to/vault --start-date 2025-01-01 --end-date 2025-01-20 --sort newest --limit 20
cargo run -- activity-timeline /path/to/vault --days 30
```

### Manual HTTP Testing
```bash
curl "http://localhost:3000/api/time-queries/recently-modified?days=7&limit=10"
curl -X POST http://localhost:3000/api/time-queries/created-in-range -H "Content-Type: application/json" -d '{"start_date": "2025-01-01", "end_date": "2025-01-20"}'
curl "http://localhost:3000/api/time-queries/activity-timeline?days=30"
```

### Verification Checklist
- [ ] `cargo build` succeeds
- [ ] CLI commands return valid JSON
- [ ] HTTP endpoints accessible and return correct data
- [ ] Respects config exclusions
- [ ] Skips hidden files
- [ ] Only processes `.md` files
- [ ] Handles missing creation time gracefully
- [ ] Date parsing validates YYYY-MM-DD format
- [ ] Invalid date ranges return errors
- [ ] Results sorted correctly
- [ ] Limits applied correctly

#### Performance Considerations

**Current approach**: Single-threaded file traversal
- Sufficient for most vaults (<10k files)
- Profile if performance issues arise
- Consider parallelization with `rayon` if needed

#### Success Criteria

1. Three new CLI commands work and output valid JSON
2. Three new HTTP endpoints accessible and functional
3. Three new MCP tools exposed (automatic via capability registry)
4. Follows established architecture patterns exactly
5. Handles edge cases gracefully (missing creation time, invalid dates)
6. Respects configuration (exclusions, hidden files)


