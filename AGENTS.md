# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust CLI tool that extracts todo items from Markdown files in Obsidian vaults. It parses task checkboxes, extracts metadata (tags, dates, priorities), and outputs structured JSON.

## Project Management

<!-- BACKLOG.MD MCP GUIDELINES START -->

<CRITICAL_INSTRUCTION>

## BACKLOG WORKFLOW INSTRUCTIONS

This project uses Backlog.md MCP for all task and project management activities.

**CRITICAL GUIDANCE**

- If your client supports MCP resources, read `backlog://workflow/overview` to understand when and how to use Backlog for this project.
- If your client only supports tools or the above request fails, call `backlog.get_backlog_instructions()` to load the tool-oriented overview. Use the `instruction` selector when you need `task-creation`, `task-execution`, or `task-finalization`.

- **First time working here?** Read the overview resource IMMEDIATELY to learn the workflow
- **Already familiar?** You should have the overview cached ("## Backlog.md Overview (MCP)")
- **When to read it**: BEFORE creating tasks, or when you're unsure whether to track work

These guides cover:

- Decision framework for when to create tasks
- Search-first workflow to avoid duplicates
- Links to detailed guides for task creation, execution, and finalization
- MCP tools reference

You MUST read the overview resource to understand the complete workflow. The information is NOT summarized here.

</CRITICAL_INSTRUCTION>

<!-- BACKLOG.MD MCP GUIDELINES END -->

### Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File tasks for remaining work** - Create Backlog tasks for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update task status** - Complete finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:

   ```bash
   git add backlog/
   git commit -m "Update tasks"
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

# The package defines two binaries (notectl, notectl-remote) — `cargo run`
# needs --bin notectl to disambiguate:

# Run task search with arguments
cargo run --bin notectl -- tasks path/to/file.md
cargo run --bin notectl -- tasks path/to/vault --status incomplete --tags work --limit 20

# Start MCP server (stdio)
cargo run --bin notectl -- serve stdio path/to/vault

# Start HTTP server
cargo run --bin notectl -- serve http path/to/vault --port 8000

# Search operations
cargo run --bin notectl -- index path/to/vault
cargo run --bin notectl -- search path/to/vault "query text"
cargo run --bin notectl -- search path/to/vault "query text" --mode dense

# Test the tool manually
echo "- [ ] Test task #tag 📅 2025-12-10" > test.md
cargo run --bin notectl -- tasks test.md
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

### Search Configuration

The `[search]` section in `.notectl.toml` controls indexing and search behavior:

```toml
[search]
model_id = "qwen3-embedding:0.6b"            # Model name sent to the embedding API
embedding_api_base = "https://your-server/v1" # OpenAI-compatible endpoint; unset = BM25-only
embedding_dim = 4096                         # Embedding dimension ceiling (matryoshka truncation)
max_seq_tokens = 512                         # Max sequence tokens for chunking
chunk_overlap_tokens = 64                    # Token overlap between adjacent chunks
min_chunk_tokens = 32                        # Min tokens per chunk before merging forward
rrf_k = 60.0                                 # RRF k parameter
rrf_bm25_weight = 1.0                        # BM25 weight in RRF fusion
rrf_cosine_weight = 1.0                      # Cosine weight in RRF fusion
max_results = 50                             # Max results per query
merge_threshold = 30                         # Merge tiny sections below this token count
cache_dir = ".notectl/search"                # Index cache directory
```

No model runs in-process — dense search calls out to whatever OpenAI-compatible
`/v1/embeddings` server `embedding_api_base` points at. If unset or unreachable,
search degrades to BM25 keyword-only automatically.

All `[search]` keys can be overridden via `NOTECTL_SEARCH_<KEY>` environment variables (e.g., `NOTECTL_SEARCH_MODEL_ID`, `NOTECTL_SEARCH_RRF_K`). See README.md for the full table.

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

  notectl-search/                     (semantic + keyword search)
    src/
      lib.rs
      capability.rs                   (SearchCapability, IndexOperation, SearchOperation)
      chunker.rs                      (markdown chunking, token-budget bounding)
      index.rs                        (streaming index build/incremental update)
      search.rs                       (search execution, result ranking)
      bm25.rs                         (BM25 sparse scoring)
      sparse.rs                       (sparse vector storage)
      fusion.rs                       (RRF fusion of dense + sparse results)
      storage.rs                      (persistent index storage, streaming VectorWriter)
      tokenize.rs                     (text tokenization)
      embeddings/mod.rs               (HTTP client for any OpenAI-compatible /v1/embeddings endpoint)
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
  |--- search (core, files)
  |--- binary (core, tasks, tags, files, daily-notes, outline, search)
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
- **`notectl-search`**: Semantic + keyword search (chunking, BM25 sparse scoring, dense embeddings via an OpenAI-compatible HTTP endpoint, RRF fusion, persistent index storage). Feature-gated behind `search`.

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

## Keeping `prime` Up to Date

The `prime` command (`src/prime.rs`) outputs a static LLM reference for all CLI commands. **You must update `PRIME_TEXT` whenever you:**

- Add, rename, or remove a command
- Add, rename, or remove an option on any command
- Change argument names or positional argument order
- Change a default value (e.g. default `--limit`)
- Change path conventions (vault_path vs path, relative vs absolute)

There are no automated tests for this — it is purely editorial. After any CLI surface change, open `src/prime.rs` and update the relevant section before committing.

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

### Search MCP Tool Registration

Unlike other capabilities, search's MCP tools aren't registered via the
`#[tool_router]` macro — `src/mcp.rs`'s `search_tools` module manually
registers `SearchTool`/`IndexTool` via `with_async_tool::<...>()` in
`TaskSearchService::new`. `notectl-search` itself is a regular, always-on
workspace dependency (not feature-gated) — see `src/capabilities/mod.rs`.

**Gotcha:** rmcp's `#[tool_handler]` macro defaults its `list_tools`/
`call_tool`/`get_tool` dispatch to a freshly rebuilt `Self::tool_router()`
(the `#[tool_router]`-macro-generated function containing only `#[tool]`
methods) — it does **not** use the `self.tool_router` field by default.
Since the manually-registered `SearchTool`/`IndexTool` only exist on the
field built in `new()`, the `#[tool_handler]` attribute on
`impl ServerHandler for TaskSearchService` **must** keep its
`router = self.tool_router.clone()` argument, or those tools compile fine,
show up in `get_info()`'s instructions text, and are still completely
unreachable via `tools/list`/`tools/call` — this happened once already
(see `tests/mcp_tools.rs`'s doc comment for the full story). That test
drives the real stdio server end-to-end and asserts every expected tool
name is both listed and dispatchable — run it (`cargo test --test
mcp_tools`) after any change to `src/mcp.rs`'s tool wiring, and extend its
`EXPECTED_TOOLS` list when adding a new tool. Compiling and starting the
server is not sufficient verification for MCP tool wiring — only a live
`tools/list`/`tools/call` round trip catches this class of bug.

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
