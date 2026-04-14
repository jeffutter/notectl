/// Static LLM priming text describing how to use notectl.
///
/// Keep this in sync with the actual CLI surface. Update it whenever you:
/// - Add, rename, or remove a command
/// - Add, rename, or remove an option on an existing command
/// - Change argument names or semantics
/// - Change default values (e.g. default limit)
pub const PRIME_TEXT: &str = r#"# notectl — Markdown Vault Assistant

notectl extracts structured data from Obsidian-style Markdown vaults. All output is JSON.

## Capabilities

### Tasks

Search and filter todo checkboxes across the vault.

`tasks <path>`
  --status incomplete|completed|cancelled|other_<char>
  --due-on YYYY-MM-DD          exact due date
  --due-before YYYY-MM-DD
  --due-after YYYY-MM-DD
  --completed-on YYYY-MM-DD
  --completed-before YYYY-MM-DD
  --completed-after YYYY-MM-DD
  --tags tag1,tag2             task must have ALL listed tags
  --exclude-tags tag1,tag2     task must have NONE of these tags
  --limit N                    max results (default 50)

Task statuses in output: "incomplete", "completed", "cancelled", "other_<char>"

### Tags

`tags <path>`                  extract unique tags from YAML frontmatter
`list-tags <path>`             list tags with document counts
  --min-count N                only tags appearing in at least N documents
  --subpath <subpath>          restrict to subdirectory
  --limit N

`search-tags <path>`           find files by YAML frontmatter tags
  --tags tag1,tag2
  --match-all true|false       AND vs OR logic (default: false = OR)
  --subpath <subpath>
  --limit N

### Files

`list-files <path>`            directory tree of the vault
  --subpath <subpath>          restrict to subdirectory
  --max-depth N
  --include-sizes true

`read-files <vault_path> <file1> [file2 ...]`   read file contents
  Paths are relative to vault root.
  --continue-on-error true     don't abort if a file is missing

### Daily Notes

`get-daily-note <vault_path> --date YYYY-MM-DD`

`search-daily-notes <vault_path>`
  --start-date YYYY-MM-DD
  --end-date YYYY-MM-DD
  --sort asc|desc              (default: desc)
  --include-content true       include full note text in results
  --limit N

### Document Structure

`outline <vault_path> <file_path>`   heading hierarchy for one file
  file_path is relative to vault root.
  --hierarchical true          nested tree instead of flat list

`section <vault_path> <file_path> <heading>`   content under a heading
  --include-subsections true   include nested sub-sections

`search-headings <vault_path> <pattern>`   search headings across all files
  --min-level N                minimum heading depth (1=H1 … 6=H6)
  --max-level N
  --limit N

## Path conventions

- `<path>` — file or directory to scan (many commands accept either)
- `<vault_path>` — the vault root directory
- `<file_path>` — always relative to vault root

## General notes

- All output is JSON.
- Default result limit is 50; override with --limit.
- Exclude paths via `.notectl.toml` `exclude_paths` array or the
  NOTECTL_EXCLUDE_PATHS environment variable (comma-separated patterns).
- Patterns support substring matching and glob syntax (e.g. **/Archive/**).
"#;

pub fn command() -> clap::Command {
    clap::Command::new("prime")
        .about("Print LLM priming instructions for using this tool")
        .long_about(
            "Outputs a concise reference of all available commands and options \
             suitable for injecting into an LLM context window.",
        )
}
