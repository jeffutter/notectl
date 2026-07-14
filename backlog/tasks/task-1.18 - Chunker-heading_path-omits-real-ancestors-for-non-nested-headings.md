---
id: TASK-1.18
title: Chunker heading_path omits real ancestors for non-nested headings
status: To Do
assignee: []
created_date: '2026-07-14 11:12'
labels: []
dependencies:
  - TASK-1.4
parent_task_id: TASK-1
priority: medium
type: bug
ordinal: 19000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Chunker::build_heading_path` in notectl-search/src/chunker.rs (lines ~165-182) is meant to return the chain of ancestor heading titles for a given section. Walking backward through the flat `sections` list, it pushes any section whose level is strictly less than the current section's level, but `break`s as soon as it meets a section whose level is greater-than-or-equal, even when that section is a sibling or cousin rather than a true ancestor.

For a document like:
```
# Main Title
## Chapter 1
### Section 1.1
## Chapter 2
```
Chapter 2's computed `heading_path` comes out `[]` instead of the correct `["Main Title"]`, because the backward walk hits `### Section 1.1` (level 3, same-or-higher than Chapter 2's level 2) first and stops immediately instead of skipping past it to reach `Main Title`. The existing test (`test_heading_path_tracking`) doesn't catch this because its assertion is conditional on the path being non-empty.

`heading_path` is surfaced to end users in search results (TASK-1.9's "map chunk ids back to file path/heading path/line span/snippet") and used in chunk IDs, so wrong/missing paths degrade search result quality and chunk identity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 build_heading_path returns the true ancestor chain (root to immediate parent) for sections with intervening siblings/cousins at deeper or equal levels, not just strictly-nested documents
- [ ] #2 Add a unit test using a document with a sibling subsection between two top-level sections that asserts the second top-level section's heading_path includes the document title
- [ ] #3 Existing chunker tests still pass
<!-- AC:END -->
