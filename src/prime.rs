/// Generate LLM priming text for the given binary name.
///
/// `remote` — when true, omits local path arguments (the server owns the vault
/// path) and instead documents the `--server` / `NOTECTL_SERVER` connection arg.
///
/// Keep this in sync with the actual CLI surface. Update it whenever you:
/// - Add, rename, or remove a command
/// - Add, rename, or remove an option on an existing command
/// - Change argument names or semantics
/// - Change default values (e.g. default limit)
pub fn generate(binary_name: &str, remote: bool) -> String {
    let bin = binary_name;

    // Positional path arguments shown in command signatures.
    let p = if remote { "" } else { " <path>" };
    let vp = if remote { "" } else { " <vault_path>" };

    // Vault prefix used in examples ("~/vault" locally, empty remotely).
    let v = if remote { "" } else { " ~/vault" };

    let connection_section = if remote {
        format!(
            r#"## Connection

Set the server URL with `--server <url>` (or `NOTECTL_SERVER` env var) on every
command. The vault path is configured on the server at startup.

  export NOTECTL_SERVER=http://host:8000
  {bin} tasks --status incomplete

"#
        )
    } else {
        String::new()
    };

    let path_conventions = if remote {
        String::new()
    } else {
        r#"## Path conventions

- `<path>` — file or directory to scan (many commands accept either)
- `<vault_path>` — the vault root directory
- `<file_path>` — always relative to vault root

"#
        .to_string()
    };

    let description = if remote {
        format!("{bin} forwards commands to a remote notectl HTTP server. All output is JSON.")
    } else {
        format!(
            "{bin} extracts structured data from Obsidian-style Markdown vaults. All output is JSON."
        )
    };

    format!(
        r#"# {bin} — Markdown Vault Assistant

{description}

{connection_section}## Capabilities

### Tasks

Search and filter todo checkboxes across the vault.

`{bin} tasks{p}`
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

Examples:
  {bin} tasks{v} --status incomplete --tags work --limit 20
  {bin} tasks{v} --due-before 2025-12-31 --status incomplete

### Tags

`{bin} tags{p}`              extract unique tags from YAML frontmatter

Examples:
  {bin} tags{v}
  {bin} tags{v} --subpath Projects

`{bin} list-tags{p}`         list tags with document counts
  --min-count N                only tags appearing in at least N documents
  --subpath <subpath>          restrict to subdirectory
  --limit N

Examples:
  {bin} list-tags{v} --min-count 3
  {bin} list-tags{v} --subpath Projects --limit 20

`{bin} search-tags{p}`       find files by YAML frontmatter tags
  --tags tag1,tag2
  --match-all true|false       AND vs OR logic (default: false = OR)
  --subpath <subpath>
  --limit N

Examples:
  {bin} search-tags{v} --tags work,urgent --match-all true
  {bin} search-tags{v} --tags meeting --subpath Meetings

### Files

`{bin} list-files{p}`        directory tree of the vault
  --subpath <subpath>          restrict to subdirectory
  --max-depth N
  --include-sizes true

`{bin} read-files{vp} <file1> [file2 ...]`   read file contents
  Paths are relative to vault root.
  --continue-on-error true     don't abort if a file is missing

`{bin} recent-files{p}`      recently modified files, newest first
  Uses frontmatter `updated:` field when present; falls back to filesystem mtime.
  --since YYYY-MM-DD           only files modified on or after this date
  --limit N                    max results (default 20)
  Output fields: file_path, file_name, updated_at (ISO 8601), date_source ("frontmatter"|"mtime"), total_found

Examples:
  {bin} list-files{v} --subpath Projects --max-depth 2
  {bin} read-files{v} Projects/Plan.md Daily/2025-01-15.md
  {bin} read-files{v} README.md --continue-on-error true
  {bin} recent-files{v} --limit 10
  {bin} recent-files{v} --since 2025-01-01 --limit 50

### Daily Notes

`{bin} get-daily-note{vp}`
  --date YYYY-MM-DD            (required)

`{bin} search-daily-notes{vp}`
  --start-date YYYY-MM-DD
  --end-date YYYY-MM-DD
  --sort asc|desc              (default: desc)
  --include-content true       include full note text in results
  --limit N

Examples:
  {bin} get-daily-note{v} --date 2025-06-01
  {bin} search-daily-notes{v} --start-date 2025-01-01 --end-date 2025-03-31
  {bin} search-daily-notes{v} --sort asc --include-content true --limit 7

### Document Structure

`{bin} outline{vp} <file_path>`   heading hierarchy for one file
  file_path is relative to vault root.
  --hierarchical true          nested tree instead of flat list

`{bin} section{vp} <file_path> <heading>`   content under a heading
  --include-subsections true   include nested sub-sections

`{bin} search-headings{vp} <pattern>`   search headings across all files
  --min-level N                minimum heading depth (1=H1 … 6=H6)
  --max-level N
  --limit N

Examples:
  {bin} outline{v} Projects/Plan.md --hierarchical true
  {bin} section{v} Projects/Plan.md "Implementation" --include-subsections true
  {bin} search-headings{v} "TODO" --max-level 3 --limit 20

{path_conventions}## General notes

- All output is JSON.
- Default result limit is 50; override with --limit.
- Exclude paths via `.notectl.toml` `exclude_paths` array or the
  NOTECTL_EXCLUDE_PATHS environment variable (comma-separated patterns).
- Patterns support substring matching and glob syntax (e.g. **/Archive/**).
"#
    )
}

pub fn command() -> clap::Command {
    clap::Command::new("prime")
        .about("Print LLM priming instructions for using this tool")
        .long_about(
            "Outputs a concise reference of all available commands and options \
             suitable for injecting into an LLM context window.",
        )
}

fn binary_name() -> String {
    std::env::args()
        .next()
        .and_then(|p| {
            std::path::Path::new(&p)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "notectl".to_string())
}

pub fn generate_for_current_binary() -> String {
    let name = binary_name();
    let remote = name.contains("remote");
    generate(&name, remote)
}
