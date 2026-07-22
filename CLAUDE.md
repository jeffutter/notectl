# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See [AGENTS.md](AGENTS.md) for all repository guidance (commands, architecture,
toolchain, and gotchas).

**Reminder — `src/prime.rs`**: this generates the LLM-facing "skill file" text
(`notectl prime` / `notectl-remote prime`), and it is hand-maintained, not
generated from the CLI definitions. If your change touches a command, option,
default value, or config default (even indirectly, e.g. changing a
`SearchConfig` default in `notectl-core`) — check whether `src/prime.rs`'s
text needs updating too, even if the task at hand seems unrelated to the CLI
surface itself. See "Keeping `prime` Up to Date" in AGENTS.md for the full
checklist.
