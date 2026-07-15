---
id: TASK-1.18
title: Chunker heading_path omits real ancestors for non-nested headings
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-14 11:12'
updated_date: '2026-07-15 20:23'
labels:
  - planned
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Overview

Replace the O(n²) backward-walk  with an O(n) forward-pass stack algorithm that computes correct ancestor paths inline during section iteration. This eliminates the entire bug class (sibling/cousin headings blocking ancestor discovery) and aligns with proven implementations in breadchunks, markdown-vdb, and structchunk.

## Implementation Plan

### Step 1: Replace  with forward stack pass (chunker.rs ~lines 140-165)

**Current code:** Walks backward from , collecting levels , breaks on first level >= current. Fails when intervening siblings at deeper levels exist.

**New algorithm:** In the  loop, maintain a . For each section at index :
1. Pop all entries where 
2. Clone the stack as the heading path (titles only)
3. Push current section's  onto stack

This is a single forward pass — amortized O(n) total instead of O(n²) per-file.

Pseudocode:

**Edge cases handled by this approach:**
- Level-skip nesting (H1 → H3): stack naturally skips missing levels since it only pops on >= comparison
- Empty document / no headings: stack stays empty, path is 
- Sibling sections (H1(A), H2(B), H2(C)): C's path correctly includes A after B is popped
- The "no heading" root section (level=0, empty title): pushed as  but produces correct empty-ish paths for content-only files

### Step 2: Remove  method

Delete the  method entirely. It's no longer needed since heading paths are computed inline during the forward pass.

### Step 3: Update  (chunker.rs ~lines 250-270)

Replace the vacuous assertion () with precise checks:
- For : expect path  (or at minimum, contains "Main Title")
- For : expect path includes both "Main Title" and "Section 1.1"
- For : **assert path is ** — this is the bug case that currently fails

Add a new test:  with document structure H1(A), H2(B), H1(C) → verify C's path does NOT include A or B (C is its own root-level heading).

### Step 4: Verify all tests pass

Run 
running 8 tests
test chunker::tests::test_chunk_by_size_fallback ... ok
test chunker::tests::test_chunker_config_from_search_config ... ok
test chunker::tests::test_empty_content ... ok
test chunker::tests::test_chunk_file_basic ... ok
test chunker::tests::test_chunk_file_with_sections ... ok
test chunker::tests::test_tiny_section_merging ... ok
test chunker::tests::test_long_section_splitting ... ok
test chunker::tests::test_heading_path_tracking ... ok

test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 41 filtered out; finished in 0.01s and 
running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

running 9 tests
test config::tests::test_default_config ... ok
test config::tests::test_env_with_empty_patterns ... ok
test config::tests::test_search_config_default ... ok
test config::tests::test_merge_from_env ... ok
test config::tests::test_search_config_all_env_vars ... ok
test config::tests::test_search_config_from_toml ... ok
test config::tests::test_should_exclude_substring ... ok
test config::tests::test_should_exclude_glob_pattern ... ok
test config::tests::test_search_config_toml_new_fields ... ok

test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

running 29 tests
test capability::tests::test_get_daily_note_request_validation ... ok
test capability::tests::test_search_daily_notes_request_validation ... ok
test date_utils::tests::test_date_range ... ok
test date_utils::tests::test_date_range_single_day ... ok
test date_utils::tests::test_date_range_invalid_start ... ok
test date_utils::tests::test_date_range_cross_month ... ok
test date_utils::tests::test_date_range_start_after_end ... ok
test date_utils::tests::test_date_range_invalid_end ... ok
test date_utils::tests::test_leap_year ... ok
test date_utils::tests::test_days_in_month ... ok
test date_utils::tests::test_parse_date ... ok
test date_utils::tests::test_validate_date_invalid ... ok
test date_utils::tests::test_validate_date_valid ... ok
test pattern::tests::test_apply_pattern ... ok
test date_utils::tests::test_date_range_cross_year ... ok
test capability::tests::test_search_daily_notes_with_content ... ok
test capability::tests::test_get_daily_note_not_found ... ok
test capability::tests::test_get_daily_note_found ... ok
test capability::tests::test_search_daily_notes_date_range_too_large ... ok
test capability::tests::test_search_daily_notes_invalid_date_range ... ok
test capability::tests::test_validate_date ... ok
test capability::tests::test_search_daily_notes_limit ... ok
test pattern::tests::test_find_daily_note ... ok
test pattern::tests::test_find_daily_note_multiple_matches_error ... ok
test pattern::tests::test_get_daily_note_relative_path ... ok
test pattern::tests::test_find_daily_note_multiple_patterns ... ok
test pattern::tests::test_find_daily_note_with_exclusion ... ok
test capability::tests::test_search_daily_notes ... ok
test capability::tests::test_search_daily_notes_descending_sort ... ok

test result: ok. 29 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s

running 8 tests
test recent_files::tests::test_extract_updated_from_frontmatter ... ok
test recent_files::tests::test_extract_updated_quoted ... ok
test recent_files::tests::test_parse_iso8601_z ... ok
test recent_files::tests::test_extract_updated_no_frontmatter ... ok
test recent_files::tests::test_unix_roundtrip ... ok
test recent_files::tests::test_parse_iso8601_positive_offset ... ok
test recent_files::tests::test_extract_updated_missing ... ok
test recent_files::tests::test_parse_iso8601_with_offset ... ok

test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

running 20 tests
test outline_extractor::tests::extract_headings::test_nested_code_blocks ... ok
test outline_extractor::tests::extract_headings::test_simple_document ... ok
test outline_extractor::tests::build_hierarchy::test_simple_hierarchy ... ok
test outline_extractor::tests::extract_headings::test_headings_in_code_blocks_ignored ... ok
test outline_extractor::tests::get_section::test_get_section_basic ... ok
test outline_extractor::tests::build_hierarchy::test_level_skipping ... ok
test outline_extractor::tests::get_section::test_get_section_with_subsections ... ok
test outline_extractor::tests::get_section::test_get_section_without_subsections ... ok
test outline_extractor::tests::parse_heading::test_h1_heading ... ok
test outline_extractor::tests::parse_heading::test_not_a_heading_no_space ... ok
test outline_extractor::tests::parse_heading::test_h6_heading ... ok
test outline_extractor::tests::parse_heading::test_heading_with_obsidian_id ... ok
test outline_extractor::tests::parse_heading::test_regular_text ... ok
test outline_extractor::tests::search_headings::test_search_with_level_filter ... ok
test outline_extractor::tests::search_headings::test_search_limit ... ok
test outline_extractor::tests::search_headings::test_case_insensitive_search ... ok
test outline_extractor::tests::search_headings::test_search_across_files ... ok
test outline_extractor::tests::get_section::test_multiple_matching_sections ... ok
test outline_extractor::tests::parse_heading::test_too_many_hashes ... ok
test outline_extractor::tests::parse_heading::test_heading_with_unicode ... ok

test result: ok. 20 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s

running 49 tests
test bm25::tests::test_tokenize ... ok
test bm25::tests::test_unrelated_query_returns_empty ... ok
test bm25::tests::test_basic_scoring ... ok
test chunker::tests::test_chunker_config_from_search_config ... ok
test storage::tests::test_atomic_write_json ... ok
test storage::tests::test_manifest_serialization_round_trip ... ok
test storage::tests::test_manifest_new_empty ... ok
test storage::tests::test_atomic_write_no_temp_leak ... ok
test storage::tests::test_compute_overall_content_hash_deterministic ... ok
test chunker::tests::test_empty_content ... ok
test chunker::tests::test_chunk_file_basic ... ok
test chunker::tests::test_heading_path_tracking ... ok
test storage::tests::test_staleness_diff_exclusion_filtering ... ok
test storage::tests::test_remove_chunks ... ok
test tests::test_config_default ... ok
test tests::test_config_resolve_relative ... ok
test chunker::tests::test_chunk_by_size_fallback ... ok
test storage::tests::test_rfc3339_formatting ... ok
test chunker::tests::test_long_section_splitting ... ok
test tests::test_search_without_embeddings_returns_clear_error ... ok
test tokenize::tests::test_count_tokens_simple ... ok
test tokenize::tests::test_count_tokens_with_extra_whitespace ... ok
test tokenize::tests::test_overlap_ge_max_tokens ... ok
test tokenize::tests::test_overlap_max_one ... ok
test tests::test_config_resolve_absolute ... ok
test tokenize::tests::test_tokenize_empty ... ok
test tokenize::tests::test_tokenize_fixed_remainder ... ok
test tokenize::tests::test_tokenize_fixed_simple ... ok
test tokenize::tests::test_tokenize_with_overlap ... ok
test tokenize::tests::test_tokenize_with_overlap_no_remainder ... ok
test chunker::tests::test_tiny_section_merging ... ok
test tokenize::tests::test_tokenize_zero_max ... ok
test chunker::tests::test_chunk_file_with_sections ... ok
test storage::tests::test_open_or_create_new ... ok
test storage::tests::test_staleness_diff_empty_index ... ok
test storage::tests::test_write_and_read_chunks ... ok
test storage::tests::test_open_or_create_version_mismatch ... ok
test storage::tests::test_open_or_create_existing ... ok
test storage::tests::test_touch_without_content_change_no_reindex ... ok
test storage::tests::test_manifest_persists_after_build ... ok
test storage::tests::test_staleness_diff_removed_file ... ok
test storage::tests::test_staleness_diff_up_to_date ... ok
test storage::tests::test_staleness_diff_modified_file ... ok
test storage::tests::test_staleness_diff_full_rebuild_model_changed ... ok
test storage::tests::test_staleness_diff_added_file ... ok
test storage::tests::test_staleness_diff_full_rebuild_chunk_config_changed ... ok
test storage::tests::test_full_rebuild_clears_chunks ... ok
test storage::tests::test_staleness_diff_full_rebuild_dimension_changed ... ok
test storage::tests::test_content_hash_changes_on_modification ... ok

test result: ok. 49 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.04s

running 16 tests
test tag_extractor::tests::test_extract_frontmatter ... ok
test tag_extractor::tests::test_empty_string_tag_filtered ... ok
test tag_extractor::tests::test_extract_tags_from_content ... ok
test tag_extractor::tests::test_empty_tags_filtered ... ok
test tag_extractor::tests::test_no_frontmatter ... ok
test tag_extractor::tests::test_parse_tags_array ... ok
test tag_extractor::tests::test_parse_tags_single_string ... ok
test tag_extractor::tests::test_extract_tags_with_counts_duplicate_in_same_file ... ok
test tag_extractor::tests::test_extract_tags_with_counts_multiple_files ... ok
test tag_extractor::tests::test_search_by_tags_respects_exclusions ... ok
test tag_extractor::tests::test_tagged_file_contains_all_tags ... ok
test tag_extractor::tests::test_search_by_tags_case_insensitive ... ok
test tag_extractor::tests::test_search_by_tags_empty_result ... ok
test tag_extractor::tests::test_search_by_tags_and_logic ... ok
test tag_extractor::tests::test_extract_tags_with_counts_single_file ... ok
test tag_extractor::tests::test_search_by_tags_or_logic ... ok

test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s

running 60 tests
test extractor::tests::clean_content::test_removes_completed_date ... ok
test capability::tests::args_to_json_minimal_args ... ok
test extractor::tests::integration::test_file_path_and_name ... ok
test extractor::tests::clean_content::test_removes_created_date ... ok
test extractor::tests::metadata_extraction::test_extract_due_date_emoji ... ok
test extractor::tests::clean_content::test_removes_tags_preserved ... ok
test extractor::tests::metadata_extraction::test_extract_completed_date_emoji ... ok
test extractor::tests::metadata_extraction::test_extract_priority_high_emoji ... ok
test extractor::tests::clean_content::test_removes_priority_emoji ... ok
test extractor::tests::integration::test_full_task_with_all_metadata ... ok
test extractor::tests::clean_content::test_removes_priority_text ... ok
test extractor::tests::clean_content::test_removes_multiple_metadata ... ok
test extractor::tests::metadata_extraction::test_extract_multiple_tags ... ok
test capability::tests::args_to_json_strips_path_and_preserves_filters ... ok
test extractor::tests::clean_content::test_removes_due_date_text ... ok
test extractor::tests::metadata_extraction::test_extract_created_date_emoji ... ok
test extractor::tests::metadata_extraction::test_extract_created_date_text ... ok
test extractor::tests::metadata_extraction::test_extract_due_date_function ... ok
test extractor::tests::integration::test_task_preserves_raw_line ... ok
test extractor::tests::metadata_extraction::test_extract_completed_date_text ... ok
test extractor::tests::integration::test_completed_task_with_completion_date ... ok
test extractor::tests::clean_content::test_removes_due_date_emoji ... ok
test extractor::tests::metadata_extraction::test_extract_priority_low_emoji ... ok
test extractor::tests::clean_content::test_preserves_task_text ... ok
test extractor::tests::metadata_extraction::test_extract_priority_lowest_emoji ... ok
test extractor::tests::metadata_extraction::test_extract_priority_text_high ... ok
test extractor::tests::metadata_extraction::test_extract_priority_text_low ... ok
test extractor::tests::metadata_extraction::test_extract_priority_text_medium ... ok
test extractor::tests::clean_content::test_cleans_extra_whitespace ... ok
test extractor::tests::parse_task_line::test_cancelled_task ... ok
test extractor::tests::metadata_extraction::test_extract_tags_with_numbers ... ok
test extractor::tests::metadata_extraction::test_no_due_date ... ok
test extractor::tests::metadata_extraction::test_no_priority ... ok
test filter::tests::test_empty_task_list ... ok
test filter::tests::test_no_filters_returns_all_tasks ... ok
test filter::tests::test_single_tag_filter ... ok
test filter::tests::test_status_filter_incomplete ... ok
test extractor::tests::metadata_extraction::test_extract_priority_urgent_emoji ... ok
test extractor::tests::parse_task_line::test_completed_task ... ok
test extractor::tests::parse_task_line::test_regular_list_item ... ok
test extractor::tests::metadata_extraction::test_extract_single_tag ... ok
test extractor::tests::sub_items::test_is_not_sub_item_empty_line ... ok
test extractor::tests::parse_task_line::test_unchecked_task ... ok
test extractor::tests::metadata_extraction::test_hashtag_alone_no_match ... ok
test extractor::tests::sub_items::test_is_sub_item_with_asterisk ... ok
test extractor::tests::metadata_extraction::test_no_tags ... ok
test extractor::tests::sub_items::test_is_not_sub_item_same_indent ... ok
test extractor::tests::parse_task_line::test_not_a_task ... ok
test extractor::tests::sub_items::test_is_sub_item_with_indent ... ok
test extractor::tests::sub_items::test_parse_sub_item_completed_checkbox ... ok
test extractor::tests::sub_items::test_is_sub_item_with_checkbox ... ok
test extractor::tests::parse_task_line::test_task_with_leading_whitespace ... ok
test extractor::tests::sub_items::test_parse_sub_item_not_list ... ok
test extractor::tests::sub_items::test_parse_sub_item_checkbox ... ok
test extractor::tests::sub_items::test_parse_sub_item_regular_list ... ok
test extractor::tests::parse_task_line::test_completed_task_uppercase ... ok
test extractor::tests::metadata_extraction::test_extract_due_date_text ... ok
test extractor::tests::parse_task_line::test_other_status_task ... ok
test extractor::tests::sub_items::test_parse_sub_item_with_asterisk ... ok
test extractor::tests::clean_content::test_removes_timestamp ... ok

test result: ok. 60 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.27s

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s to confirm no regressions.

## Files Changed
-  (primary: algorithm replacement + tests)

## Risks
- **Merge path heading:** When two tiny sections merge, the merged chunk uses  from the first section. With the stack approach, this is still correct since the first section's ancestors are preserved at that point in the forward pass. No change needed.
- **Fallback :** Unaffected — it already returns empty heading paths.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced O(n²) backward-walk  with O(n) forward-pass stack algorithm. The stack tracks (level, title) pairs; for each section we pop entries with level >= current to find true ancestors, snapshot the remaining stack as the heading path, then push the current section. This eliminates the entire bug class where sibling/cousin headings at deeper levels blocked ancestor discovery.\n\nRemoved  method entirely. Heading paths are now computed inline during the section iteration in .\n\nAdded  test: H1(Alpha), H2(Beta), H1(Charlie) → verifies Charlie has empty path (own root) and Beta has path ["Alpha"].\n\nStrengthened : asserts Section 1.1 has path ["Main Title", "Chapter 1"] and Chapter 2 has path ["Main Title"] — the exact bug case.\n\nTests use merge_threshold=0 to prevent section merging from interfering with heading path assertions.
<!-- SECTION:NOTES:END -->
