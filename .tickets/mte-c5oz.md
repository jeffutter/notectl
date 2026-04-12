---
id: mte-c5oz
status: closed
deps: []
links: []
created: 2026-04-12T03:51:42Z
type: task
priority: 2
assignee: Jeffery Utter
parent: mte-r91g
tags: [planned]
---
# Add args_to_json() method to Operation trait

Add a new args_to_json(&self, matches: &clap::ArgMatches) -> Result<serde_json::Value, Box<dyn Error>> method to the Operation trait in notectl-core/src/operation.rs. Implement it in all 11 operation structs across 5 capability crates (notectl-tasks, notectl-tags, notectl-files, notectl-daily-notes, notectl-outline). The implementation pattern is identical for all: parse request struct from ArgMatches via FromArgMatches, set CLI-only path field to None, serialize to JSON via serde_json::to_value(). This separates arg parsing from execution so the remote client can reuse parsing logic.

## Design

1. Add args_to_json() method to Operation trait in notectl-core/src/operation.rs with signature: fn args_to_json(&self, matches: &clap::ArgMatches) -> Result<serde_json::Value, Box<dyn Error>>

2. Implement in each operation struct. Pattern (same for all 11):
   - Parse request: let mut request = RequestStruct::from_arg_matches(matches)?;
   - Clear CLI-only path: request.path = None; (or request.vault_path = None;)
   - Return JSON: Ok(serde_json::to_value(request)?)

3. Operations to update:
   - notectl-tasks: SearchTasksOperation
   - notectl-tags: ExtractTagsOperation, ListTagsOperation, SearchByTagsOperation
   - notectl-files: ListFilesOperation, ReadFilesOperation
   - notectl-daily-notes: GetDailyNoteOperation, SearchDailyNotesOperation
   - notectl-outline: GetOutlineOperation, GetSectionOperation, SearchHeadingsOperation

4. Do NOT implement for ServeOperation (CLI-only, excluded from remote)

5. Add unit tests verifying args_to_json produces correct JSON for at least one operation.

