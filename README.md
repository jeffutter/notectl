# notectl

[![CI](https://github.com/jeffutter/notectl/workflows/CI/badge.svg)](https://github.com/jeffutter/notectl/actions)

A Rust CLI tool to extract todo items from markdown files in an Obsidian vault.

## Features

- Extract tasks from single files or entire directories
- Support for multiple task statuses:
  - Incomplete: `- [ ]`
  - Completed: `- [x]`
  - Cancelled: `- [-]`
  - Custom statuses: `- [>]`, `- [!]`, etc.
- Extract metadata:
  - Tags: `#tag`
  - Due dates: `📅 2025-12-10`, `due: 2025-12-10`, `@due(2025-12-10)`
  - Priority: `⏫ 🔼 🔽 ⏬` or `priority: high/medium/low`
  - Created dates: `➕ 2025-12-10`, `created: 2025-12-10`
  - Completed dates: `✅ 2025-12-10`, `completed: 2025-12-10`
- Parse sub-items (indented list items)
- Filter tasks by various criteria
- Output as structured JSON

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

### Basic Usage

Extract all tasks from a file:
```bash
notectl path/to/file.md
```

Extract all tasks from a directory (recursive):
```bash
notectl path/to/vault
```

### Filtering Options

Filter by status:
```bash
notectl path/to/vault --status incomplete
notectl path/to/vault --status completed
notectl path/to/vault --status cancelled
```

Filter by due date:
```bash
# Tasks due on a specific date
notectl path/to/vault --due-on 2025-12-10

# Tasks due before a date
notectl path/to/vault --due-before 2025-12-31

# Tasks due after a date
notectl path/to/vault --due-after 2025-12-01
```

Filter by completed date:
```bash
# Tasks completed on a specific date
notectl path/to/vault --completed-on 2025-12-01

# Tasks completed before a date
notectl path/to/vault --completed-before 2025-12-31

# Tasks completed after a date
notectl path/to/vault --completed-after 2025-12-01
```

Filter by tags:
```bash
# Tasks with specific tags (must have all specified tags)
notectl path/to/vault --tags work,urgent

# Exclude tasks with certain tags
notectl path/to/vault --exclude-tags archive,done
```

### Combining Filters

You can combine multiple filters:
```bash
notectl path/to/vault \
  --status incomplete \
  --tags work \
  --due-before 2025-12-31
```

## Output Format

The tool outputs JSON with the following structure:

```json
[
  {
    "content": "Task description",
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
notectl file.md --status incomplete --tags work
```

Will output only the "Write report" task with its sub-items.
