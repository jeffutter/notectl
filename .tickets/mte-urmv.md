---
id: mte-urmv
status: closed
deps: [mte-46gp]
links: []
created: 2026-02-19T15:03:52Z
type: task
priority: 2
assignee: Jeffery Utter
parent: mte-jeur
tags: [planned]
---
# Update CI/CD workflow and documentation

Update .github/workflows/cd.yml (hardcoded binary name in packaging steps), README.md, CONTRIBUTING.md, CLAUDE.md, and CHANGELOG.md. All references to markdown-todo-extractor in docs, URLs, and CI artifact names should become notectl.

## Design

Files to update:

1. .github/workflows/cd.yml:
   - BINARY_NAME=markdown-todo-extractor → BINARY_NAME=notectl
   - RELEASE_NAME=markdown-todo-extractor-... → RELEASE_NAME=notectl-...
   - Glob patterns for release artifacts: markdown-todo-extractor-*.tar.gz → notectl-*.tar.gz

2. README.md:
   - Title line: # markdown-todo-extractor → # notectl
   - All command examples: markdown-todo-extractor → notectl
   - GitHub URLs (if repo is being renamed)

3. CONTRIBUTING.md:
   - Clone URL and issue tracker links
   - GitHub repo references

4. CLAUDE.md:
   - Architecture section workspace structure examples
   - Any example commands

5. CHANGELOG.md:
   - Add breaking changes entry noting config file rename and binary rename

