---
id: TASK-28
title: >-
  Fix: REFERENCE_EMBEDDING zero-check silently disables numeric validation
  forever if a real dim-0 value is ever 0.0
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-18 07:10'
updated_date: '2026-07-18 16:09'
labels:
  - review-followup
milestone: Active
dependencies:
  - TASK-1.14.2
priority: high
ordinal: 110
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while reviewing TASK-1.14.2 (notectl-search/src/embeddings/model.rs:974 and :995). The two integration tests decide whether to run the numeric reference-vector comparison using 'REFERENCE_EMBEDDING[0] != 0.0' / 'DOC_REFERENCE_EMBEDDING[0] != 0.0' as a proxy for 'has this constant been populated with real values yet.' This is a Correctness/Clarity-axis gap: embedding dimensions are continuous floats, so a legitimately-populated reference vector whose first dimension happens to be 0.0 (or a copy-paste mistake that leaves index 0 at 0.0 while other dims are populated) would cause the test to silently keep skipping the numeric assertion forever, with only an eprintln! (invisible unless run with --nocapture) -- no test failure, no clear signal. The pattern is duplicated across both REFERENCE_EMBEDDING and DOC_REFERENCE_EMBEDDING (doubling the surface area) and will be relied on again by TASK-1.14.2.1 when populating real values.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 model.rs no longer branches on 'REFERENCE_EMBEDDING[0] != 0.0' or 'DOC_REFERENCE_EMBEDDING[0] != 0.0'; both numeric-check guards key off an explicit named boolean sentinel instead (e.g. a REFERENCE_EMBEDDING_POPULATED: bool const, flipped to true once real values are pasted in)
- [ ] #2 nix develop -c cargo test -p notectl-search --features integration passes, with both integration tests still gracefully skipping the numeric check (sentinel false) and printing a clear 'not populated' message
- [ ] #3 nix develop -c cargo clippy -p notectl-search --features integration --all-targets -- -D warnings passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SETUP (read first): This is a Rust CLI workspace (notectl-core, notectl-outline, notectl-search, notectl-files, notectl-tags, notectl-tasks, notectl-daily-notes, plus the main notectl binary). ALL commands must run inside the Nix dev shell: either run 'direnv allow' once, or prefix every command with 'nix develop -c'. Work from the repository root unless told otherwise. Do not change pinned dependency versions.

1. Open notectl-search/src/embeddings/model.rs, integration_tests module. Locate the REFERENCE_EMBEDDING const (~line 828) and DOC_REFERENCE_EMBEDDING const (~line 838).

2. Add two explicit boolean sentinel consts near them, e.g.:
   const REFERENCE_EMBEDDING_POPULATED: bool = false;
   const DOC_REFERENCE_EMBEDDING_POPULATED: bool = false;
   (Document that these must be flipped to true in the same commit that populates the corresponding vector with real values -- see TASK-1.14.2.1.)

3. In test_encoder_produces_correct_dimension (~line 964-982), replace the guard:
   if !REFERENCE_EMBEDDING.is_empty() && REFERENCE_EMBEDDING[0] != 0.0 {
   with:
   if REFERENCE_EMBEDDING_POPULATED {

4. In test_document_embedding_matches_reference (~line 985-1003), replace the equivalent guard for DOC_REFERENCE_EMBEDDING with DOC_REFERENCE_EMBEDDING_POPULATED.

5. Update the eprintln! messages in the else branches if needed so they still make sense (e.g. 'Populate REFERENCE_EMBEDDING and set REFERENCE_EMBEDDING_POPULATED = true to enable numerical validation.').

6. Run: nix develop -c cargo test -p notectl-search --features integration -- integration_tests (confirm both tests still skip the numeric check and pass).

7. Run: nix develop -c cargo clippy -p notectl-search --features integration --all-targets -- -D warnings.

8. Run: nix develop -c cargo fmt -p notectl-search -- --check (fix formatting if needed).
<!-- SECTION:PLAN:END -->
