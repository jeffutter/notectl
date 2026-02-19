---
id: markdown-todo-extractor-afx
status: closed
deps: []
links: []
created: 2026-01-21T20:52:02.753910016-06:00
type: feature
priority: 2
tags: ["planned"]
---
# Links and Backlinks capability

Implement capability to extract [[wikilinks]], get backlinks for a list of notes, and generate link graph. Methods: search_links(), get_backlinks(notes[]), get_link_graph().

## Design

### Links and Backlinks Capability - Design Document

#### Overview

This capability will extract [[wikilinks]] from markdown files, compute backlinks (which notes reference a given note), and generate a link graph for understanding note relationships.

#### Data Structures

### LinkInfo
Represents a wikilink found in a markdown file:
- `target`: String - The target note name (without brackets)
- `display_text`: Option<String> - Custom display text if using [[target|display]]
- `source_file`: String - File containing this link
- `source_file_name`: String - Just the filename
- `line_number`: usize - Line where link appears

### BacklinkInfo
Represents a note that links to a target note:
- `source_file`: String - File path that contains the link
- `source_file_name`: String - Just the filename
- `link_count`: usize - How many times source links to target
- `links`: Vec<LinkInfo> - Individual link occurrences

### LinkGraphNode
Represents a node in the link graph:
- `file_path`: String
- `file_name`: String
- `outgoing_links`: Vec<String> - Notes this file links to
- `incoming_links`: Vec<String> - Notes that link to this file
- `outgoing_count`: usize
- `incoming_count`: usize

### LinkGraphEdge
Represents an edge in the graph:
- `source`: String - Source file path
- `target`: String - Target note name
- `weight`: usize - Number of links from source to target

#### Operations

### 1. search_links (search for wikilinks)
**Purpose**: Extract all wikilinks from markdown files with filtering options.

**Request (SearchLinksRequest)**:
- `path`: Option<PathBuf> - CLI path override (skipped in HTTP/MCP)
- `subpath`: Option<String> - Subpath within vault to search
- `target`: Option<String> - Filter to links pointing to specific note
- `source_file`: Option<String> - Filter to links from specific file
- `limit`: Option<usize> - Max results to return

**Response (SearchLinksResponse)**:
- `links`: Vec<LinkInfo>
- `total_count`: usize
- `truncated`: bool

**CLI**: `links <path> [--target <note>] [--source <file>] [--limit <n>]`
**HTTP**: `POST /api/links`

### 2. get_backlinks
**Purpose**: Find all notes that link to specified target notes.

**Request (GetBacklinksRequest)**:
- `path`: Option<PathBuf> - CLI path override
- `targets`: Vec<String> - Note names to find backlinks for
- `subpath`: Option<String> - Subpath within vault to search
- `include_link_details`: Option<bool> - Include individual link occurrences

**Response (GetBacklinksResponse)**:
- `backlinks`: HashMap<String, Vec<BacklinkInfo>> - Map from target to backlinks
- `total_files_scanned`: usize

**CLI**: `backlinks <path> --targets <note1,note2,...> [--include-details]`
**HTTP**: `POST /api/links/backlinks`

### 3. get_link_graph
**Purpose**: Generate a graph representation of all note links.

**Request (GetLinkGraphRequest)**:
- `path`: Option<PathBuf> - CLI path override
- `subpath`: Option<String> - Subpath within vault
- `include_orphans`: Option<bool> - Include notes with no links (default: false)
- `max_depth`: Option<usize> - Max traversal depth from root (if specified)
- `root`: Option<String> - Start from specific note (filters graph)

**Response (GetLinkGraphResponse)**:
- `nodes`: Vec<LinkGraphNode>
- `edges`: Vec<LinkGraphEdge>
- `total_nodes`: usize
- `total_edges`: usize
- `orphan_count`: usize

**CLI**: `link-graph <path> [--include-orphans] [--root <note>] [--max-depth <n>]`
**HTTP**: `POST /api/links/graph`

#### Implementation Components

### LinkExtractor (new file: src/link_extractor.rs)
Core extraction logic, following TagExtractor pattern:

```rust
pub struct LinkExtractor {
    config: Arc<Config>,
    wikilink_pattern: Regex,  // \[\[([^\]|]+)(?:\|([^\]]+))?\]\]
}
```

Methods:
- `extract_links_from_file(path) -> Result<Vec<LinkInfo>>`
- `extract_links_from_content(content, path) -> Vec<LinkInfo>`
- `extract_links(path) -> Result<Vec<LinkInfo>>` (dir traversal)
- `find_backlinks(path, targets) -> Result<HashMap<String, Vec<BacklinkInfo>>>`
- `build_link_graph(path) -> Result<(Vec<LinkGraphNode>, Vec<LinkGraphEdge>)>`

### LinkCapability (new file: src/capabilities/links.rs)
Following existing capability pattern:

- Operation metadata modules: `search_links`, `get_backlinks`, `get_link_graph`
- Request/Response structs with JsonSchema, clap::Parser derives
- LinkCapability struct with async methods
- Operation structs implementing `crate::operation::Operation` trait

### Integration Points

1. **CapabilityRegistry** (src/capabilities/mod.rs):
   - Add `LinkCapability` field
   - Add `links()` getter method
   - Add operations to `create_operations()`

2. **MCP Server** (src/mcp.rs):
   - Add three new `#[tool]` methods delegating to LinkCapability
   - Update ServerInfo instructions

#### Wikilink Parsing Rules

Support standard Obsidian wikilink formats:
- `[[Note Name]]` - Basic link
- `[[Note Name|Display Text]]` - Link with custom display
- `[[folder/Note Name]]` - Link with path
- `[[Note Name#Heading]]` - Link to heading (extract both note and heading)
- `[[Note Name#^block-id]]` - Link to block (extract both note and block)

The regex pattern will be:
```regex
\[\[([^\]|#^]+)(?:#([^\]|^]+))?(?:\^([^\]|]+))?(?:\|([^\]]+))?\]\]
```

#### Testing Strategy

Unit tests in link_extractor.rs:
- Wikilink pattern parsing (basic, with display, with path)
- Heading and block reference extraction
- Multi-link extraction from single line
- Edge cases (empty content, malformed links)

Unit tests in capabilities/links.rs:
- search_links filtering by target/source
- get_backlinks single and multiple targets
- get_link_graph node/edge generation
- Orphan node handling

Integration tests:
- Full vault scanning with rayon parallelization
- Path exclusion configuration
- Subpath filtering

#### Files to Create/Modify

**New Files:**
1. `src/link_extractor.rs` - Core link extraction logic
2. `src/capabilities/links.rs` - Capability and operations

**Modified Files:**
1. `src/capabilities/mod.rs` - Register LinkCapability
2. `src/mcp.rs` - Add MCP tool handlers
3. `src/main.rs` - (No changes needed - automatic via registry)
4. `src/lib.rs` or equivalent module declaration - Export link_extractor

#### Implementation Order

1. Create `src/link_extractor.rs` with LinkInfo struct and basic extraction
2. Create `src/capabilities/links.rs` with SearchLinksOperation
3. Register in CapabilityRegistry, add MCP handler
4. Implement backlinks functionality
5. Implement link graph functionality
6. Add comprehensive tests

#### Considerations

- **Performance**: Use rayon for parallel file scanning (following TagExtractor pattern)
- **Memory**: For large vaults, link graph could be large - consider streaming/pagination
- **Link resolution**: Just extract raw link targets, don't resolve to actual files (simpler, more flexible)
- **Case sensitivity**: Links should preserve original case but matching can be case-insensitive


