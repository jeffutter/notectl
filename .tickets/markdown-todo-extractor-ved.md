---
id: markdown-todo-extractor-ved
status: closed
deps: []
links: []
created: 2026-01-21T20:52:03.198816314-06:00
type: feature
priority: 2
tags: ["planned"]
---
# Full-text search capability

Implement content search across vault with regex/fuzzy matching. Need to design efficient implementation strategy. Methods: search_content(query, filters).

## Design

### Implementation Plan: Full-Text Search Capability

**Issue**: markdown-todo-extractor-ved - Full-text search capability

#### Overview

Implement a content search capability that allows searching across markdown files in the vault with support for:
- Regex pattern matching
- Case-insensitive text search
- Fuzzy/approximate matching
- Configurable context lines around matches
- File path patterns (glob), modified date filters, and result limits

#### Architecture Approach

Follow the established capability-based architecture pattern where a single operation automatically exposes functionality via HTTP, CLI, and MCP interfaces.

**Components to create:**
1. `src/search_extractor.rs` - Core search logic and file processing
2. `src/capabilities/search.rs` - Capability and operation definitions
3. Update `src/capabilities/mod.rs` - Register new capability
4. Update `Cargo.toml` - Add fuzzy matching dependency

#### Critical Files

### New Files
- `src/search_extractor.rs` - SearchExtractor struct with search methods
- `src/capabilities/search.rs` - SearchCapability and SearchContentOperation

### Modified Files
- `src/capabilities/mod.rs` - Add search module and register capability
- `Cargo.toml` - Add `nucleo-matcher` crate for fuzzy matching

#### Detailed Design

### 1. Search Extractor (`src/search_extractor.rs`)

**Purpose**: Core search logic, file traversal, pattern matching

**Key components**:

```rust
pub struct SearchExtractor {
    config: Arc<Config>,
}

pub struct SearchMatch {
    pub file_path: String,
    pub file_name: String,
    pub line_number: usize,
    pub matched_line: String,
    pub context_before: Vec<String>,  // Lines before match
    pub context_after: Vec<String>,   // Lines after match
    pub match_type: MatchType,        // Regex, Text, or Fuzzy
}

pub struct SearchResult {
    pub matches: Vec<SearchMatch>,
    pub total_matches: usize,
    pub files_searched: usize,
    pub truncated: bool,  // True if results were limited
}

pub enum MatchType {
    Regex,
    Text,
    Fuzzy,
}
```

**Methods**:
- `search_content()` - Main entry point, delegates to specific search type
- `search_regex()` - Regex pattern search with compiled patterns
- `search_text()` - Simple case-insensitive substring search
- `search_fuzzy()` - Fuzzy matching using nucleo-matcher
- `process_file()` - Read file, search lines, extract context
- `collect_markdown_files()` - Reuse pattern from tag_extractor.rs

**Implementation notes**:
- Use `rayon::par_iter()` for parallel file processing
- Compile regex patterns once and pass to worker threads (Arc<Regex>)
- Respect `config.should_exclude()` for path filtering
- Read files with `fs::read_to_string()` like existing extractors
- For context extraction: collect N lines before/after maintaining line numbers

### 2. Capability Module (`src/capabilities/search.rs`)

**Structure** (following established pattern):

```rust
// Request struct - serves as CLI args, HTTP params, and MCP input
#[derive(Debug, Deserialize, Serialize, JsonSchema, Parser)]
#[command(name = "search-content", about = "Search for content across markdown files")]
pub struct SearchContentRequest {
    // CLI-only path field
    #[arg(index = 1)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub path: Option<PathBuf>,

    // Search query
    #[arg(long, short = 'q', required = true)]
    #[schemars(description = "Search query (text, regex, or fuzzy pattern)")]
    pub query: String,

    // Search mode
    #[arg(long, default_value = "text")]
    #[schemars(description = "Search mode: 'text', 'regex', or 'fuzzy'")]
    pub mode: Option<String>,  // "text", "regex", "fuzzy"

    // Case sensitivity (for text mode)
    #[arg(long)]
    #[schemars(description = "Case-sensitive search (default: false)")]
    pub case_sensitive: Option<bool>,

    // Context lines
    #[arg(long, short = 'C', default_value = "2")]
    #[schemars(description = "Number of context lines before and after match")]
    pub context: Option<usize>,

    // File path filter (glob pattern)
    #[arg(long)]
    #[schemars(description = "File path pattern (glob) to filter search")]
    pub file_pattern: Option<String>,

    // Modified date filter
    #[arg(long)]
    #[schemars(description = "Only search files modified after this date (YYYY-MM-DD)")]
    pub modified_after: Option<String>,

    #[arg(long)]
    #[schemars(description = "Only search files modified before this date (YYYY-MM-DD)")]
    pub modified_before: Option<String>,

    // Result limits
    #[arg(long, default_value = "100")]
    #[schemars(description = "Maximum total matches to return")]
    pub max_results: Option<usize>,

    #[arg(long, default_value = "10")]
    #[schemars(description = "Maximum matches per file")]
    pub max_per_file: Option<usize>,

    // Subpath (for HTTP/MCP - alternative to CLI path)
    #[arg(skip)]
    #[schemars(description = "Optional subpath within base path to search")]
    pub subpath: Option<PathBuf>,
}

// Response struct
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SearchContentResponse {
    pub matches: Vec<SearchMatch>,
    pub total_matches: usize,
    pub files_searched: usize,
    pub truncated: bool,
}

// Capability struct
pub struct SearchCapability {
    base_path: PathBuf,
    search_extractor: Arc<SearchExtractor>,
}

impl SearchCapability {
    pub fn new(base_path: PathBuf, config: Arc<Config>) -> Self {
        Self {
            base_path,
            search_extractor: Arc::new(SearchExtractor::new(config)),
        }
    }

    pub async fn search_content(
        &self,
        request: SearchContentRequest,
    ) -> CapabilityResult<SearchContentResponse> {
        // Resolve search path
        let search_path = if let Some(ref subpath) = request.subpath {
            self.base_path.join(subpath)
        } else {
            self.base_path.clone()
        };

        // Validate and execute search based on mode
        let mode = request.mode.as_deref().unwrap_or("text");
        let result = match mode {
            "regex" => self.search_extractor.search_regex(&search_path, &request),
            "fuzzy" => self.search_extractor.search_fuzzy(&search_path, &request),
            _ => self.search_extractor.search_text(&search_path, &request),
        }.map_err(|e| internal_error(format!("Search failed: {}", e)))?;

        Ok(SearchContentResponse {
            matches: result.matches,
            total_matches: result.total_matches,
            files_searched: result.files_searched,
            truncated: result.truncated,
        })
    }
}

// Operation struct
pub struct SearchContentOperation {
    capability: Arc<SearchCapability>,
}

impl SearchContentOperation {
    pub fn new(capability: Arc<SearchCapability>) -> Self {
        Self { capability }
    }
}

// Operation metadata module
pub mod search_content {
    pub const DESCRIPTION: &str = "Search for content across markdown files with regex, text, or fuzzy matching";
    pub const CLI_NAME: &str = "search-content";
    pub const HTTP_PATH: &str = "/api/search/content";
}

// Implement Operation trait
#[async_trait]
impl Operation for SearchContentOperation {
    fn name(&self) -> &'static str {
        search_content::CLI_NAME
    }

    fn path(&self) -> &'static str {
        search_content::HTTP_PATH
    }

    fn description(&self) -> &'static str {
        search_content::DESCRIPTION
    }

    fn get_command(&self) -> clap::Command {
        SearchContentRequest::command()
    }

    async fn execute_json(&self, json: Value) -> Result<Value, ErrorData> {
        execute_json_operation(json, |req| self.capability.search_content(req)).await
    }

    async fn execute_from_args(
        &self,
        matches: &ArgMatches,
        _registry: &CapabilityRegistry,
    ) -> Result<String, Box<dyn Error>> {
        let request = SearchContentRequest::from_arg_matches(matches)?;

        // Handle CLI-specific path
        let response = if let Some(ref path) = request.path {
            let config = Arc::new(Config::load_from_base_path(path.as_path()));
            let capability = SearchCapability::new(path.clone(), config);
            let mut req_without_path = request;
            req_without_path.path = None;
            capability.search_content(req_without_path).await?
        } else {
            self.capability.search_content(request).await?
        };

        Ok(serde_json::to_string_pretty(&response)?)
    }

    fn input_schema(&self) -> serde_json::Value {
        schemars::schema_for!(SearchContentRequest)
    }
}
```

### 3. Registration (`src/capabilities/mod.rs`)

Add to the CapabilityRegistry:

```rust
pub mod search;

pub struct CapabilityRegistry {
    // ... existing fields ...
    search_capability: Arc<SearchCapability>,
}

impl CapabilityRegistry {
    pub fn new(base_path: PathBuf, config: Arc<Config>) -> Self {
        Self {
            // ... existing initialization ...
            search_capability: Arc::new(SearchCapability::new(base_path.clone(), config.clone())),
        }
    }

    pub fn search(&self) -> Arc<SearchCapability> {
        Arc::clone(&self.search_capability)
    }

    pub fn create_operations(&self) -> Vec<Arc<dyn Operation>> {
        vec![
            // ... existing operations ...
            Arc::new(search::SearchContentOperation::new(self.search())),
        ]
    }
}
```

### 4. Dependencies (`Cargo.toml`)

Add fuzzy matching support:

```toml
[dependencies]
### ... existing dependencies ...
nucleo-matcher = "0.3"  # Fast fuzzy matcher (used by Helix editor)
```

**Why nucleo-matcher?**
- High-performance fuzzy matching (used in production by Helix editor)
- Simple API
- No complex configuration needed
- Good for interactive search scenarios

#### Implementation Details

### File Filtering Strategy

1. **Path exclusions**: Use existing `config.should_exclude()` from Config
2. **Glob patterns**: Use `glob::Pattern` (already a dependency) for `file_pattern` matching
3. **Modified date**: Use `std::fs::metadata().modified()` with date comparison

### Performance Optimizations

1. **Parallel processing**: Use `rayon::par_iter()` for file-level parallelism
2. **Early termination**: Stop searching when `max_results` reached
3. **Compiled patterns**: Create regex patterns once, share via `Arc<Regex>` across threads
4. **File reading**: Use `std::fs::read_to_string()` (same as existing extractors)

### Search Mode Details

**Text mode** (default):
- Simple case-insensitive substring search using `.to_lowercase().contains()`
- Fast, no regex overhead
- Good for simple queries

**Regex mode**:
- Compile pattern with `Regex::new()`
- Use case-insensitive flag if specified: `(?i)pattern`
- Return error if pattern is invalid
- Good for complex pattern matching

**Fuzzy mode**:
- Use `nucleo_matcher::Matcher` with default config
- Match against each line
- Score matches and sort by relevance
- Good for approximate matching (typos, variations)

### Context Extraction Algorithm

For each match at line N with context size C:
1. Read file into `Vec<String>` (split by lines)
2. Find matching line at index N
3. Extract lines `[N-C .. N)` for context_before (handle bounds)
4. Extract lines `(N .. N+C]` for context_after (handle bounds)
5. Store with proper line numbers

### Result Limiting Strategy

1. **Per-file limit**: Stop processing file after `max_per_file` matches
2. **Total limit**: Track global match count, stop all processing when reached
3. **Set truncated flag**: Indicate if results were limited
4. **Files searched counter**: Track how many files were examined before stopping

#### MCP Integration

The operation will automatically be exposed via MCP when registered. Update `src/mcp.rs` tool instructions:

Add to `get_info()` instructions:
```markdown
### search-content
Search for content across markdown files with multiple matching modes.

**Parameters:**
- `query` (required): Search query string
- `mode`: Search mode - "text" (default), "regex", or "fuzzy"
- `case_sensitive`: Enable case-sensitive search (default: false)
- `context`: Number of context lines before/after match (default: 2)
- `file_pattern`: Glob pattern to filter files (e.g., "projects/**/*.md")
- `modified_after/before`: Filter by file modification date (YYYY-MM-DD)
- `max_results`: Maximum total matches (default: 100)
- `max_per_file`: Maximum matches per file (default: 10)
- `subpath`: Optional subpath within vault to search

**Returns:** List of matches with file paths, line numbers, content, and context

**Example:**
```json
{
  "query": "TODO",
  "mode": "text",
  "context": 3,
  "file_pattern": "projects/**",
  "max_results": 50
}
```
```

#### Verification Plan

### Unit Tests
1. Test each search mode (text, regex, fuzzy) with sample content
2. Test context extraction with edge cases (beginning/end of file)
3. Test file filtering (glob patterns, date ranges)
4. Test result limiting (per-file and total)
5. Test path exclusion integration

### Integration Tests
1. **CLI**: `cargo run -- search-content /path/to/vault -q "search term" --mode text --context 2`
2. **HTTP**: `curl "http://localhost:3000/api/search/content?query=TODO&mode=text&context=2"`
3. **MCP**: Test via MCP client calling search_content tool

### Manual Testing
1. Create test vault with sample markdown files
2. Test with various queries:
   - Simple text: "project"
   - Regex pattern: `\d{4}-\d{2}-\d{2}` (dates)
   - Fuzzy: "aproximate" finding "approximate"
3. Test filters:
   - File patterns: `projects/**/*.md`
   - Date ranges: files modified this week
   - Result limits: verify truncation
4. Test context display: verify before/after lines are correct
5. Test edge cases:
   - Empty query
   - Invalid regex
   - No matches
   - Matches at beginning/end of file

### Performance Testing
1. Test on large vault (1000+ files)
2. Verify parallel processing is working (CPU usage)
3. Ensure result limiting prevents runaway queries
4. Check memory usage with large result sets

#### Error Handling

Handle these scenarios gracefully:
- Invalid regex pattern → return ErrorData with clear message
- File read errors → log warning, skip file, continue search
- Invalid date format → return ErrorData with validation message
- Invalid mode value → default to "text" mode
- Empty query → return empty results (or error?)
- Path doesn't exist → return ErrorData

#### CLI Examples

```bash
### Simple text search
cargo run -- search-content /vault -q "TODO"

### Regex search for dates
cargo run -- search-content /vault -q '\d{4}-\d{2}-\d{2}' --mode regex

### Fuzzy search with more context
cargo run -- search-content /vault -q "aproximate" --mode fuzzy --context 5

### Search in specific folder with limits
cargo run -- search-content /vault -q "meeting" --file-pattern "projects/**" --max-results 20

### Search recent files
cargo run -- search-content /vault -q "sprint" --modified-after 2025-01-01
```

#### Implementation Order

1. Add `nucleo-matcher` to Cargo.toml
2. Create `src/search_extractor.rs` with core search logic
3. Create `src/capabilities/search.rs` with capability and operation
4. Update `src/capabilities/mod.rs` to register capability
5. Add unit tests for search_extractor
6. Test CLI manually
7. Test HTTP endpoint manually
8. Test MCP tool via client
9. Add integration tests
10. Update documentation (CLAUDE.md if needed)

#### Success Criteria

- [ ] All three search modes work correctly (text, regex, fuzzy)
- [ ] Context lines are extracted accurately
- [ ] File filtering works (glob patterns, date ranges)
- [ ] Result limits prevent overwhelming responses
- [ ] Operation auto-registers for HTTP, CLI, and MCP
- [ ] Path exclusions from config are respected
- [ ] Parallel processing provides good performance
- [ ] Error handling is robust and user-friendly
- [ ] Tests pass (unit and integration)
- [ ] Manual testing confirms expected behavior


