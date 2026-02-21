# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust CLI tool that extracts todo items from Markdown files in Obsidian vaults. It parses task checkboxes, extracts metadata (tags, dates, priorities), and outputs structured JSON.

## Project Management

This project uses **tk** (tickets) for issue tracking. Run `tk help` for command reference.

### Quick Reference

```bash
tk ready              # Find available work
tk show <id>          # View ticket details
tk start <id>         # Claim work (sets status to in_progress)
tk close <id>         # Complete work
tk dep <id> <dep-id>  # Add dependency (id depends on dep-id)
tk list               # List all tickets
tk blocked            # List blocked tickets
```

### Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File tickets for remaining work** - Create tickets for anything that needs follow-up using `tk create`
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update ticket status** - Close finished work with `tk close <id>`, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git add .tickets/
   git commit -m "Update tickets"
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- Tickets are stored in `.tickets/` as markdown files and must be committed to git
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds


## Build and Development Commands

```bash
# Build debug version
cargo build

# Build release version
cargo build --release

# Run task search with arguments
cargo run -- tasks path/to/file.md
cargo run -- tasks path/to/vault --status incomplete --tags work --limit 20

# Start MCP server (stdio)
cargo run -- serve stdio path/to/vault

# Start HTTP server
cargo run -- serve http path/to/vault --port 8000

# Test the tool manually
echo "- [ ] Test task #tag 📅 2025-12-10" > test.md
cargo run -- tasks test.md
```

## Configuration

The tool supports configuration via a `.notectl.toml` file placed in the vault's root directory.

### Path Exclusions

You can exclude specific paths or path patterns from being scanned for tasks. This is useful for ignoring template directories, recipe folders, or any other content you don't want to include in task searches.

**Option 1: Configuration file `.notectl.toml`**

```toml
# Path exclusion patterns
# Supports both substring matching and glob patterns
exclude_paths = [
    "Template",      # Excludes any path containing "Template"
    "Recipes",       # Excludes any path containing "Recipes"
    "**/Archive/**"  # Glob pattern for Archive directories
]
```

**Option 2: Environment variable**

```bash
# Comma-separated list of exclusion patterns
export NOTECTL_EXCLUDE_PATHS="Template,Recipes,**/Archive/**"

# Start the server with exclusions
cargo run -- --mcp-stdio /path/to/vault
```

**How it works:**
- The configuration is loaded automatically from the base path when the server starts or CLI runs
- Environment variables are merged with TOML config (both sources are combined)
- Exclusion patterns support both:
  - **Substring matching**: Any path containing the pattern string will be excluded
  - **Glob patterns**: Standard glob patterns like `**/folder/**`, `*.backup`, etc.
- Excluded paths are skipped during directory traversal in `extract_tasks_from_dir`
- No MCP parameter needed - this is a server-side configuration only

## Architecture

### Workspace Structure

The project is organized as a Rust workspace with separate crates for each functional area:

```
notectl/                              (workspace root + main binary)
  Cargo.toml                          ([workspace] + [package])
  src/
    main.rs                           (entry point)
    mcp.rs                            (MCP server adapter)
    http_router.rs                    (HTTP routing helpers)
    cli.rs                            (serve command + ServeOperation)
    cli_router.rs                     (CLI dispatch)
    capabilities/mod.rs               (CapabilityRegistry - integration point)

  notectl-core/                       (shared types, traits, config)
    src/
      lib.rs
      config.rs                       (Config struct, load_from_base_path)
      error.rs                        (internal_error, invalid_params helpers)
      operation.rs                    (Operation trait for HTTP/CLI/MCP)
      file_walker.rs                  (collect_markdown_files utility)

  notectl-tasks/                      (task extraction)
    src/
      lib.rs
      extractor.rs                    (TaskExtractor, Task struct)
      filter.rs                       (FilterOptions, filter_tasks)
      capability.rs                   (TaskCapability, SearchTasksOperation)

  notectl-tags/                       (tag operations)
    src/
      lib.rs
      tag_extractor.rs                (TagExtractor)
      capability.rs                   (TagCapability, Extract/List/SearchByTagsOperation)

  notectl-files/                      (file operations)
    src/
      lib.rs
      capability.rs                   (FileCapability, ListFilesOperation, ReadFilesOperation)

  notectl-daily-notes/                (daily note operations)
    src/
      lib.rs
      date_utils.rs                   (validate_date, date_range, today)
      pattern.rs                      (apply_pattern, find_daily_note)
      capability.rs                   (DailyNoteCapability, GetDailyNoteOperation, SearchDailyNotesOperation)

  notectl-outline/                    (document outline/structure)
    src/
      lib.rs
      outline_extractor.rs            (OutlineExtractor, heading parsing)
      capability.rs                   (OutlineCapability, GetOutline/GetSection/SearchHeadingsOperation)
```

### Dependency Graph (no cycles)

```
core
  ^
  |--- tasks (core)
  |--- tags (core)
  |--- files (core)
  |--- outline (core)
  |--- daily-notes (core, files)
  |--- binary (core, tasks, tags, files, daily-notes, outline)
```

### Capability-Based Architecture

Each workspace crate provides one or more **capabilities** that encapsulate a functional area. Capabilities can be exposed via multiple interfaces (MCP, HTTP, CLI) through the **Operation** trait defined in core.

**Key components:**

- **`notectl_core::operation::Operation`**: Unified trait for all operations
  - `name()`: CLI command name
  - `path()`: HTTP endpoint path
  - `execute_json()`: HTTP/MCP execution with JSON I/O
  - `execute_from_args()`: CLI execution from clap ArgMatches
  - `input_schema()`: JSON schema for the operation's input

- **`src/capabilities/mod.rs`**: `CapabilityRegistry` - the integration point
  - Instantiates all capabilities from workspace crates
  - `create_operations()` returns all registered operations for automatic HTTP/CLI/MCP registration

**Interface Adapters (in binary `src/`):**

- **`src/mcp.rs`**: MCP server adapter using rmcp's `#[tool_router]` macro
- **`src/http_router.rs`**: HTTP route registration helpers
- **`src/cli_router.rs`**: CLI command dispatch
- **`src/cli.rs`**: Serve command definition
- **`src/main.rs`**: Entry point, wires everything together

**Core Extractors (in domain crates):**

- **`notectl-tasks`**: Task extraction pipeline (regex patterns, parsing, filtering)
- **`notectl-tags`**: YAML frontmatter tag extraction
- **`notectl-files`**: File tree listing and content reading
- **`notectl-daily-notes`**: Daily note lookup by date pattern
- **`notectl-outline`**: Heading hierarchy extraction

### Task Extraction Pipeline

All task extraction lives in `notectl-tasks/src/`:

1. **File Discovery**: `extract_tasks()` → `extract_tasks_from_dir()` recursively finds `.md` files
2. **Line Parsing**: `extract_tasks_from_file()` → `parse_task_line()` matches task patterns
3. **Sub-item Detection**: `is_sub_item()` + `parse_sub_item()` handle indented list items
4. **Metadata Extraction**: Multiple `extract_*()` methods parse tags, dates, priorities from task content
5. **Content Cleaning**: `clean_content()` removes all metadata markers to produce clean task text
6. **Filtering**: `filter_tasks()` applies user-specified filters (status, dates, tags)
7. **JSON Output**: Serde serializes filtered tasks

### Regex Pattern System

The `TaskExtractor` (in `extractor.rs`) holds compiled regex patterns that are reused across all files:

- **Task patterns**: Detect `- [ ]`, `- [x]`, `- [-]`, `- [?]` checkboxes with various statuses
- **Metadata patterns**: Extract dates (`📅 YYYY-MM-DD`, `due: YYYY-MM-DD`), priorities (emoji or text), tags (`#tag`)
- **Cleaning patterns**: Remove metadata from content to get clean task descriptions

The cleaning step is critical: content is extracted first with all metadata intact, then cleaned separately after metadata extraction to avoid losing information.

## Supported Metadata Formats

**Task Statuses**:
- `- [ ]` → `"incomplete"`
- `- [x]` or `- [X]` → `"completed"`
- `- [-]` → `"cancelled"`
- `- [?]` or any other char → `"other_?"` (the char is embedded in the status string)

**Dates** (YYYY-MM-DD format):
- Due: `📅 2025-12-10`, `due: 2025-12-10`, `@due(2025-12-10)`
- Created: `➕ 2025-12-10`, `created: 2025-12-10`
- Completed: `✅ 2025-12-10`, `completed: 2025-12-10`

**Priority**:
- Emojis: `⏫` (urgent), `🔼` (high), `🔽` (low), `⏬` (lowest)
- Text: `priority: high/medium/low`

**Tags**: `#tagname` (alphanumeric/underscore only; tags are preserved in `content` after cleaning)

**Result Limit**: Results default to 50 (override with `--limit` flag or `NOTECTL_DEFAULT_LIMIT` env var)

## Adding New Features

### Adding a New Capability (as a new workspace crate)

To add a new capability (e.g., for bookmarks):

1. **Create a new workspace crate** `notectl-bookmarks/`:
   - `Cargo.toml`: depend on `notectl-core.workspace = true` plus any needed deps
   - `src/lib.rs`: `pub mod capability; pub use capability::*;`
   - `src/capability.rs`: implement the capability struct and `Operation` structs

2. **Capability pattern** (see `notectl-files/src/capability.rs` for reference):
   ```rust
   use notectl_core::{CapabilityResult, config::Config};
   use notectl_core::error::{internal_error, invalid_params};

   pub struct NewCapability { base_path: PathBuf, config: Arc<Config> }

   impl NewCapability {
       pub fn new(base_path: PathBuf, config: Arc<Config>) -> Self { ... }
       pub async fn do_thing(&self, request: Request) -> CapabilityResult<Response> { ... }
   }

   pub struct DoThingOperation { capability: Arc<NewCapability> }

   #[async_trait::async_trait]
   impl notectl_core::operation::Operation for DoThingOperation {
       fn name(&self) -> &'static str { "do-thing" }
       fn path(&self) -> &'static str { "/api/things" }
       fn description(&self) -> &'static str { "..." }
       fn get_command(&self) -> clap::Command { RequestStruct::command() }
       async fn execute_json(&self, json: serde_json::Value) -> Result<serde_json::Value, ErrorData> { ... }
       async fn execute_from_args(&self, matches: &clap::ArgMatches) -> Result<String, Box<dyn Error>> { ... }
       fn input_schema(&self) -> serde_json::Value { serde_json::to_value(schema_for!(RequestStruct)).unwrap() }
   }
   ```

3. **Add to workspace** in root `Cargo.toml`:
   - Add `"notectl-bookmarks"` to `[workspace] members`
   - Add `notectl-bookmarks = { path = "notectl-bookmarks" }` under `[workspace.dependencies]`
   - Add `notectl-bookmarks.workspace = true` to `[dependencies]`

4. **Register in `src/capabilities/mod.rs`**:
   - Add `pub use notectl_bookmarks::{NewCapability, DoThingOperation};`
   - Add field to `CapabilityRegistry` and getter method
   - Add operation to `create_operations()`

5. **Add MCP tool in `src/mcp.rs`**:
   ```rust
   #[tool(description = "...")]
   async fn do_thing(&self, Parameters(req): Parameters<Request>) -> Result<Json<Response>, ErrorData> {
       Ok(Json(self.capability_registry.bookmarks().do_thing(req).await?))
   }
   ```

> **Note**: As of now, `notectl-outline` operations (`get_outline`, `get_section`, `search_headings`) are registered for HTTP and CLI but **not** exposed as MCP tools in `src/mcp.rs`. They are available via HTTP and CLI only.

### Adding New Metadata Types or Task Statuses

In `notectl-tasks/src/extractor.rs`:
- Add regex pattern to `TaskExtractor::new()`
- Add extraction method and call it in `create_task()`
- Add cleaning logic to `clean_content()`
- Add field to `Task` struct

If filtering is needed, in `notectl-tasks/src/`:
- Add field to `FilterOptions` in `filter.rs`
- Add filter logic in `filter_tasks()` in `filter.rs`
- Update `SearchTasksRequest` in `capability.rs`

### Operation Pattern (request struct doubles as CLI args)

Request structs implement both `serde::Deserialize` (for HTTP/MCP) and `clap::Parser` (for CLI). CLI-only fields use `#[serde(skip)]` and `#[schemars(skip)]`.

```rust
#[derive(Debug, Deserialize, Serialize, JsonSchema, clap::Parser)]
#[command(name = "list-tags", about = "List all tags with document counts")]
pub struct ListTagsRequest {
    #[arg(index = 1, required = true)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub path: Option<PathBuf>,  // CLI only

    #[arg(long)]
    pub min_count: Option<usize>,
}
```

In `execute_from_args`, if a path is provided, create a temporary capability with that path:
```rust
async fn execute_from_args(&self, matches: &clap::ArgMatches) -> Result<String, Box<dyn Error>> {
    let request = ListTagsRequest::from_arg_matches(matches)?;
    let response = if let Some(ref path) = request.path {
        let config = Arc::new(Config::load_from_base_path(path));
        let capability = TagCapability::new(path.clone(), config);
        capability.list_tags(request).await?
    } else {
        self.capability.list_tags(request).await?
    };
    Ok(serde_json::to_string_pretty(&response)?)
}
```

**Testing**:
```bash
# Test the command
cargo run -- list-tags /path/to/vault --min-count 2

# Test help text
cargo run -- list-tags --help
```
