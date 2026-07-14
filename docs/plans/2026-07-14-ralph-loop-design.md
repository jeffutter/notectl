# Ralph Loop: Autonomous Backlog-Churning pi Extension

## Overview

A project-local pi extension (`.pi/extensions/ralph/`) that drives an autonomous
loop over `backlog` tickets — plan them, execute them, periodically review the
work — until a bounded number of iterations is reached or there's no work left.
Modeled on the "Ralph" technique: each unit of real work runs in a fresh,
isolated `pi -p` subprocess rather than accumulating context in one long session.

## Motivation

Backlog tickets in this repo (`backlog/tasks/`) move through
`Needs Plan → Dev Ready → In Progress → Done`. Planning and executing them is
repetitive and well-specified enough to automate, but needs periodic human-grade
review to catch drift. This extension automates the churn while keeping a
check-in cadence, and exposes live progress so it can run alongside normal
interactive use of `pi`.

## Commands

- `/ralph [iterations] [reviewEvery]` — start the loop. Defaults: `iterations=16`,
  `reviewEvery=8`. Errors if a run is already active.
- `/ralph-stop` — request a graceful stop; the loop finishes its current step
  and exits before starting the next one.
- `/ralph-progress` — live dashboard (TUI) or a text summary (non-TUI) of the
  current run: counters, current step, ticket in flight, recent history.

## Lifetime & Execution Model

The loop runs as a background `async` task inside the *current* pi process
(started but not awaited by the `/ralph` command handler) — it stops if this
pi session exits, and `/ralph-progress` only has anything to report within that
same session. This was a deliberate simplification over a detached OS process:
no PID files, no orphan-process cleanup, and the loop can share `ctx.ui` for
live status directly.

Each iteration that requires judgment or does real work spawns a **fresh
headless `pi -p` subprocess** via `pi.exec("pi", [...])`, so no iteration's
context leaks into the next. All deterministic bookkeeping (counters, exit
conditions, `backlog task list` / `unblocked-todo.sh` queries) happens in plain
TypeScript — no LLM round-trip needed for control flow.

## State

In-memory `RalphState` is the source of truth while a run is active, mirrored
after every step to `.pi/ralph/state.json` (current snapshot) and appended to
`.pi/ralph/history.jsonl` (one JSON line per completed step) for post-hoc
inspection. These files are not read back to drive `/ralph-progress` — that
reads the in-memory state directly.

```ts
type RalphState = {
  status: "idle" | "running" | "stopping" | "stopped" | "done";
  iterations: number;        // requested max
  reviewEvery: number;
  loopCount: number;         // completed iterations
  reviewCount: number;       // iterations since last review
  currentStep?: string;      // human-readable, e.g. "executing TASK-42"
  startedAt: string;
  history: RalphHistoryEntry[]; // capped ring buffer, last 50, for the dashboard
};

type RalphHistoryEntry = {
  at: string;
  kind: "execute" | "plan" | "choose" | "review" | "exit";
  ticket?: string;
  outcome: "ok" | "failed" | "skipped";
  summary: string; // subprocess stdout tail or a short deterministic note
};
```

## Loop Algorithm

Directly implements the sequence supplied by the user, with judgment-heavy
steps delegated to subprocesses:

1. **Complete check** — `loopCount >= iterations` → stop.
2. **Review check** — `reviewCount >= reviewEvery` → step 3, else step 4.
3. **Review** — fresh `pi -p` call whose prompt: use the `herdr` skill to split
   a pane, run `claude "Run the review-pi-work skill for the last N tickets"`
   (N = tickets completed since last review), watch it via herdr pane
   read/wait, close the pane when done. Reset `reviewCount`, increment
   `loopCount`, continue.
4–5. Increment `loopCount`, `reviewCount`.
6. **In-progress/Dev-Ready check** — `backlog task list -s "In Progress" --plain`,
   then `-s "Dev Ready" --plain` (plain TS, no subprocess). First hit → step 7.
7. **Execute** — fresh `pi -p --skill backlog-execute "<ticket>"`. Record
   outcome; continue loop regardless of success/failure (see Error Handling).
8. **Needs-Plan check** — `backlog task list -s "Needs Plan" --plain`. Hit →
   step 9.
9. **Plan** — fresh `pi -p --model research` call to gather web context, then
   fresh `pi -p --model planning --skill backlog-planner "<ticket>"` fed the
   research output. Continue loop.
10. **Choose** — run `./backlog/unblocked-todo.sh`. Empty → stop (`done`, no
    tickets left). One candidate → deterministic pick. Multiple → fresh `pi -p`
    call given the candidate list, told to pick one and mark it `Needs Plan`.
11–12. Record a brief summary in history, continue loop.

## Subprocess Invocation

All worker calls go through one helper, `runHeadless(promptOrArgs, opts)`,
wrapping `pi.exec("pi", ["-p", ...flags, prompt], { timeout, signal })`.
Flags used: `--skill <path>` (backlog-execute / backlog-planner), `--model
<alias>` (`research`, `planning` — both pre-configured aliases in this user's
pi settings), `--no-session` (worker runs are one-shot, no need to persist).
Each call has a generous timeout (ticket work can run long); a timeout is
treated as a `failed` outcome, not a thrown error.

## Herdr Requirement (Review Step)

`/ralph` checks `process.env.HERDR_ENV === "1"` at start time and refuses to
start otherwise, since the review step is core to the design (not an optional
extra) and silently degrading it could hide review failures. The actual pane
orchestration (split, run `claude`, watch for completion, close) is delegated
to the prompted subprocess via the `herdr` skill rather than hand-rolled in the
extension — that judgment (parsing pane output, deciding "done") belongs in an
LLM turn, not deterministic string matching.

## Error Handling

A failed or timed-out subprocess records a `failed` history entry and the loop
proceeds to the next iteration — one bad ticket must not abort a 16-iteration
run. `/ralph-progress` surfaces failures prominently (recent history) so the
user notices a run that's failing repeatedly, but nothing auto-pauses the loop
on failure in v1.

## UI

- `ctx.ui.setWidget("ralph", ...)` — persistent one-line ambient status
  (`ralph: iter 4/16 · executing TASK-42`) while a run is active; cleared on
  stop/done.
- `/ralph-progress` — `ctx.ui.custom()` live dashboard in TUI mode (counters,
  current step, last ~10 history entries), polling in-memory state on an
  interval, closes on Escape. Falls back to a single `ctx.ui.notify()` summary
  outside TUI mode (e.g. print/JSON mode).

## Out of Scope (v1)

- Detached/OS-independent background execution (survives closing pi entirely).
- Auto-pause on repeated failures.
- Multiple concurrent runs.
- Non-herdr review fallback.
