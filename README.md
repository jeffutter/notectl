# notectl

[![CI](https://github.com/jeffutter/notectl/workflows/CI/badge.svg)](https://github.com/jeffutter/notectl/actions)

A Rust CLI tool to extract todo items and manage content in Markdown files within an Obsidian vault.

## Features

- Extract tasks from single files or entire directories
- Support for multiple task statuses:
  - Incomplete: `- [ ]`
  - Completed: `- [x]`
  - Cancelled: `- [-]`
  - Custom statuses: `- [>]`, `- [!]`, etc. (output as `other_>`, `other_!`)
- Extract metadata:
  - Tags: `#tag`
  - Due dates: `📅 2025-12-10`, `due: 2025-12-10`, `@due(2025-12-10)`
  - Priority: `⏫ 🔼 🔽 ⏬` or `priority: high/medium/low`
  - Created dates: `➕ 2025-12-10`, `created: 2025-12-10`
  - Completed dates: `✅ 2025-12-10`, `completed: 2025-12-10`
- Filter tasks by status, dates, and tags; limit result count
- Parse sub-items (indented list items)
- Output as structured JSON
- Tag operations: extract, list, and search files by YAML frontmatter tags
- File operations: list vault directory tree and read markdown files
- Daily note operations: retrieve and search daily notes by date
- Document outline operations: extract headings, get sections, search headings
- **MCP server**: expose all capabilities via stdio or HTTP MCP protocol for AI assistants

## Installation

### Cargo

* Install the rust toolchain in order to have cargo installed by following
  [this](https://www.rust-lang.org/tools/install) guide.
* run `cargo install notectl`

### Build from source

```bash
cargo build --release
```

## Usage

All operations are available as subcommands. Run `notectl --help` to see the full list.

### Task Search

Extract all tasks from a file:
```bash
notectl tasks path/to/file.md
```

Extract all tasks from a directory (recursive):
```bash
notectl tasks path/to/vault
```

#### Filtering Options

Filter by status:
```bash
notectl tasks path/to/vault --status incomplete
notectl tasks path/to/vault --status completed
notectl tasks path/to/vault --status cancelled
```

Filter by due date:
```bash
# Tasks due on a specific date
notectl tasks path/to/vault --due-on 2025-12-10

# Tasks due before a date
notectl tasks path/to/vault --due-before 2025-12-31

# Tasks due after a date
notectl tasks path/to/vault --due-after 2025-12-01
```

Filter by completed date:
```bash
# Tasks completed on a specific date
notectl tasks path/to/vault --completed-on 2025-12-01

# Tasks completed before a date
notectl tasks path/to/vault --completed-before 2025-12-31

# Tasks completed after a date
notectl tasks path/to/vault --completed-after 2025-12-01
```

Filter by tags:
```bash
# Tasks with specific tags (must have all specified tags)
notectl tasks path/to/vault --tags work,urgent

# Exclude tasks with certain tags
notectl tasks path/to/vault --exclude-tags archive,done
```

Limit results:
```bash
# Return at most 10 tasks (default: 50, configurable via NOTECTL_DEFAULT_LIMIT)
notectl tasks path/to/vault --limit 10
```

#### Combining Filters

```bash
notectl tasks path/to/vault \
  --status incomplete \
  --tags work \
  --due-before 2025-12-31 \
  --limit 20
```

### Tag Operations

Extract all unique tags from YAML frontmatter:
```bash
notectl extract-tags path/to/vault
notectl extract-tags path/to/vault --subpath Notes/
```

List tags with document counts:
```bash
notectl list-tags path/to/vault
notectl list-tags path/to/vault --min-count 2 --limit 50
```

Search files by tags (AND/OR logic):
```bash
# Files with any of the specified tags (OR)
notectl search-by-tags path/to/vault --tags work,personal

# Files with all of the specified tags (AND)
notectl search-by-tags path/to/vault --tags work,urgent --match-all
```

### File Operations

List the vault directory tree:
```bash
notectl list-files path/to/vault
notectl list-files path/to/vault --subpath Notes/ --max-depth 2 --include-sizes
```

Read one or more markdown files:
```bash
notectl read-files path/to/vault Notes/my-note.md
notectl read-files path/to/vault Notes/a.md,Notes/b.md
notectl read-files path/to/vault Notes/a.md --continue-on-error
```

### Daily Note Operations

Get a specific daily note:
```bash
notectl get-daily-note path/to/vault --date 2025-12-10
```

Search daily notes by date range:
```bash
notectl search-daily-notes path/to/vault
notectl search-daily-notes path/to/vault --start-date 2025-12-01 --end-date 2025-12-31
notectl search-daily-notes path/to/vault --include-content --sort asc --limit 7
```

### Document Outline Operations

Get the heading outline of a file:
```bash
notectl get-outline path/to/vault --file-path Notes/my-note.md
notectl get-outline path/to/vault --file-path Notes/my-note.md --hierarchical
```

Get the content of a specific section:
```bash
notectl get-section path/to/vault --file-path Notes/my-note.md --heading "Introduction"
notectl get-section path/to/vault --file-path Notes/my-note.md --heading "Setup" --include-subsections
```

Search headings across the vault:
```bash
notectl search-headings path/to/vault --pattern "TODO"
notectl search-headings path/to/vault --pattern "Setup" --min-level 2 --max-level 3
```

### MCP Server Mode

Start an MCP server to expose all capabilities to AI assistants like Claude:

```bash
# stdio transport (for Claude Desktop or other MCP clients)
notectl serve stdio path/to/vault

# HTTP transport
notectl serve http path/to/vault
notectl serve http path/to/vault --port 8080
```

The HTTP server exposes:
- `GET/POST /api/tasks` - task search
- `GET/POST /api/tags/extract` - tag extraction
- `GET/POST /api/tags` - tag listing
- `GET/POST /api/tags/search` - tag search
- `GET/POST /api/files` - file listing
- `GET/POST /api/files/read` - file reading
- `GET/POST /api/daily-notes` - daily note lookup
- `GET/POST /api/daily-notes/search` - daily note search
- `GET/POST /api/outline` - outline extraction
- `GET /tools` - list all available tools with schemas
- `POST /mcp` - MCP protocol endpoint

## Output Format

The task search outputs JSON with the following structure:

```json
[
  {
    "content": "Task description #tag",
    "status": "incomplete",
    "file_path": "path/to/file.md",
    "file_name": "file.md",
    "line_number": 5,
    "raw_line": "- [ ] Task description #tag 📅 2025-12-10",
    "tags": ["tag"],
    "sub_items": ["Sub-item 1", "Sub-item 2"],
    "summary": null,
    "due_date": "2025-12-10",
    "priority": "high",
    "created_date": null,
    "completed_date": null
  }
]
```

Notes:
- `content` has metadata markers removed but preserves `#tags`
- `status` is `"incomplete"`, `"completed"`, `"cancelled"`, or `"other_X"` for custom checkbox chars
- Results are limited to 50 by default (override with `--limit` or `NOTECTL_DEFAULT_LIMIT`)

## Example

Given a markdown file:

```markdown
# My Tasks

- [ ] Buy groceries #shopping 📅 2025-12-10
- [ ] Write report #work 🔼 due: 2025-12-15
  - Research topic
  - Outline structure
- [x] Finish project #work ✅ 2025-12-01
```

Running:
```bash
notectl tasks file.md --status incomplete --tags work
```

Will output only the "Write report" task with its sub-items.

## Configuration

The tool supports configuration via a `.notectl.toml` file placed in the vault's root directory.

### Path Exclusions

Exclude specific paths from task scanning and file listing:

```toml
# .notectl.toml
exclude_paths = [
    "Template",      # Excludes any path containing "Template"
    "Recipes",       # Excludes any path containing "Recipes"
    "**/Archive/**"  # Glob pattern for Archive directories
]
```

You can also set exclusions via environment variable:

```bash
export NOTECTL_EXCLUDE_PATHS="Template,Recipes,**/Archive/**"
```

Both sources are merged; patterns support substring matching and standard globs.

### Environment Variables

- `NOTECTL_EXCLUDE_PATHS` - Comma-separated path exclusion patterns
- `NOTECTL_DEFAULT_LIMIT` - Default task result limit (default: `50`)
