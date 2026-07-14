/**
 * Ralph loop: autonomous backlog-churning extension.
 *
 * `/ralph [iterations] [reviewEvery]` drives tickets through
 * Needs Plan -> Dev Ready -> In Progress -> Done, periodically checkpointing
 * with a human-grade review, until `iterations` passes complete or the
 * backlog runs dry. See docs/plans/2026-07-14-ralph-loop-design.md.
 *
 * Each unit of real work (executing a ticket, planning one, choosing the next
 * one, reviewing recent work) runs in a fresh headless `pi -p` subprocess, so
 * no iteration's context leaks into the next — the "Ralph" technique. Only
 * bookkeeping (counters, exit conditions, backlog status queries) happens
 * here in plain TypeScript.
 *
 * Depends on skills already present as this user's global pi/Claude skills
 * (backlog-execute, backlog-planner, review-pi-work, herdr) and on "research"
 * / "planning" model aliases configured in pi's settings — see the design
 * doc for details. Requires HERDR_ENV=1 (the review step drives a herdr
 * pane) and the `backlog` CLI.
 */

import { appendFile, mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import { Key, matchesKey, truncateToWidth } from "@earendil-works/pi-tui";

// --- Types & constants ---------------------------------------------------

const STATE_DIR = ".pi/ralph";
const MAX_HISTORY = 50;

const DEFAULT_ITERATIONS = 16;
const DEFAULT_REVIEW_EVERY = 8;

const RESEARCH_TIMEOUT_MS = 20 * 60_000;
const PLAN_TIMEOUT_MS = 50 * 60_000;
const EXECUTE_TIMEOUT_MS = 60 * 60_000;
const CHOOSE_TIMEOUT_MS = 10 * 60_000;
const REVIEW_TIMEOUT_MS = 50 * 60_000;

/**
 * A step that fails this many times in a row (same kind + ticket) stops the
 * loop instead of retrying forever. Repeated identical failure is a signal
 * of a systemic problem (a hung subprocess, a broken tool), not a one-off
 * bad ticket — silently burning the iteration budget on it just hides that.
 */
const MAX_CONSECUTIVE_FAILURES = 2;

type RalphStatus = "running" | "stopping" | "stopped" | "done";

type StepKind = "execute" | "plan" | "choose" | "review";

type RalphHistoryEntry = {
  at: string;
  kind: StepKind;
  ticket?: string;
  outcome: "ok" | "failed";
  summary: string;
};

type RalphState = {
  status: RalphStatus;
  iterations: number;
  reviewEvery: number;
  loopCount: number;
  reviewCount: number;
  executedSinceReview: number;
  stopRequested: boolean;
  currentStep?: string;
  startedAt: string;
  history: RalphHistoryEntry[];
  /** Consecutive failures of the same (kind, ticket) step — see MAX_CONSECUTIVE_FAILURES. */
  failureStreak?: { key: string; count: number };
};

/** Records outcome `ok` under `key`; returns true once the streak hits the cap. */
function trackFailureStreak(state: RalphState, key: string, ok: boolean): boolean {
  if (ok) {
    state.failureStreak = undefined;
    return false;
  }
  state.failureStreak =
    state.failureStreak?.key === key
      ? { key, count: state.failureStreak.count + 1 }
      : { key, count: 1 };
  return state.failureStreak.count >= MAX_CONSECUTIVE_FAILURES;
}

type Ticket = { id: string; title: string };

/** The one loop this session is running, if any. Lifetime = this pi process. */
let activeState: RalphState | null = null;

// --- State persistence -----------------------------------------------------

function createState(iterations: number, reviewEvery: number): RalphState {
  return {
    status: "running",
    iterations,
    reviewEvery,
    loopCount: 0,
    reviewCount: 0,
    executedSinceReview: 0,
    stopRequested: false,
    currentStep: undefined,
    startedAt: new Date().toISOString(),
    history: [],
    failureStreak: undefined,
  };
}

async function ensureStateDir(cwd: string): Promise<void> {
  await mkdir(join(cwd, STATE_DIR), { recursive: true });
}

async function persist(cwd: string, state: RalphState): Promise<void> {
  await ensureStateDir(cwd);
  await writeFile(join(cwd, STATE_DIR, "state.json"), JSON.stringify(state, null, 2), "utf8");
}

async function recordHistory(
  cwd: string,
  state: RalphState,
  entry: Omit<RalphHistoryEntry, "at">,
): Promise<void> {
  const full: RalphHistoryEntry = { at: new Date().toISOString(), ...entry };
  state.history.push(full);
  if (state.history.length > MAX_HISTORY) state.history.shift();
  await ensureStateDir(cwd);
  await appendFile(join(cwd, STATE_DIR, "history.jsonl"), `${JSON.stringify(full)}\n`, "utf8");
}

// --- Deterministic backlog queries (no LLM involved) ------------------------

async function execCapture(
  pi: ExtensionAPI,
  cmd: string,
  args: string[],
  opts: { cwd: string; timeout?: number },
): Promise<{ ok: boolean; killed: boolean; stdout: string; stderr: string }> {
  const result = await pi.exec(cmd, args, opts);
  return {
    ok: result.code === 0 && !result.killed,
    killed: !!result.killed,
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
  };
}

function parsePlainTaskList(output: string): Ticket[] {
  const tasks: Ticket[] = [];
  for (const line of output.split("\n")) {
    const match = line.match(/^\s*\[[^\]]+\]\s*\[[^\]]+\]\s+(\S+)\s+-\s+(.+?)\s*$/);
    if (match) tasks.push({ id: match[1], title: match[2] });
  }
  return tasks;
}

function parseUnblockedList(output: string): Ticket[] {
  const tasks: Ticket[] = [];
  for (const line of output.split("\n")) {
    const match = line.match(/^(\S+)\s+-\s+(.+?)\s*$/);
    if (match) tasks.push({ id: match[1], title: match[2] });
  }
  return tasks;
}

async function findFirstByStatus(pi: ExtensionAPI, cwd: string, status: string): Promise<Ticket | undefined> {
  const { stdout } = await execCapture(pi, "backlog", ["task", "list", "-s", status, "--plain"], {
    cwd,
    timeout: 15_000,
  });
  return parsePlainTaskList(stdout)[0];
}

async function listUnblocked(pi: ExtensionAPI, cwd: string): Promise<Ticket[]> {
  const { stdout } = await execCapture(pi, "./backlog/unblocked-todo.sh", [], { cwd, timeout: 30_000 });
  return parseUnblockedList(stdout);
}

async function markNeedsPlan(pi: ExtensionAPI, cwd: string, ticketId: string): Promise<boolean> {
  const { ok } = await execCapture(pi, "backlog", ["task", "edit", ticketId, "-s", "Needs Plan"], {
    cwd,
    timeout: 15_000,
  });
  return ok;
}

// --- Headless pi worker calls -----------------------------------------------

function tailSummary(output: string, maxLen = 240): string {
  const collapsed = output.trim().replace(/\s+/g, " ");
  if (!collapsed) return "(no output)";
  return collapsed.length > maxLen ? `…${collapsed.slice(-maxLen)}` : collapsed;
}

async function runHeadless(
  pi: ExtensionAPI,
  cwd: string,
  prompt: string,
  opts: { timeout: number; model?: string },
): Promise<{ ok: boolean; killed: boolean; output: string }> {
  const args = ["-p", "--no-session"];
  if (opts.model) args.push("--model", opts.model);
  args.push(prompt);
  const { ok, killed, stdout, stderr } = await execCapture(pi, "pi", args, { cwd, timeout: opts.timeout });
  return { ok, killed, output: (stdout || stderr || "").trim() };
}

/** Prefixes a summary with a timeout marker when the subprocess was killed, so
 * .pi/ralph/history.jsonl distinguishes "hung until we killed it" from other failures. */
function summarize(result: { killed: boolean; output: string }, maxLen?: number): string {
  const prefix = result.killed ? "[timed out] " : "";
  return prefix + tailSummary(result.output, maxLen);
}

function extractTicketId(output: string, knownIds: string[]): string | undefined {
  const lines = output.trim().split("\n").reverse();
  for (const line of lines) {
    const trimmed = line.trim();
    if (knownIds.includes(trimmed)) return trimmed;
  }
  return knownIds.find((id) => output.includes(id));
}

// --- Step implementations ---------------------------------------------------

function setCurrentStep(ctx: ExtensionCommandContext, state: RalphState, text: string): void {
  state.currentStep = text;
  renderWidget(ctx, state);
}

async function doExecute(
  pi: ExtensionAPI,
  ctx: ExtensionCommandContext,
  cwd: string,
  state: RalphState,
  ticket: Ticket,
): Promise<boolean> {
  setCurrentStep(ctx, state, `executing ${ticket.id}`);
  const result = await runHeadless(pi, cwd, `/backlog-execute ${ticket.id}`, { timeout: EXECUTE_TIMEOUT_MS });
  if (result.ok) state.executedSinceReview += 1;
  await recordHistory(cwd, state, {
    kind: "execute",
    ticket: ticket.id,
    outcome: result.ok ? "ok" : "failed",
    summary: summarize(result),
  });
  return result.ok;
}

async function doPlan(
  pi: ExtensionAPI,
  ctx: ExtensionCommandContext,
  cwd: string,
  state: RalphState,
  ticket: Ticket,
): Promise<boolean> {
  setCurrentStep(ctx, state, `researching ${ticket.id}`);
  const researchPrompt = [
    `Research context to inform planning ticket ${ticket.id} ("${ticket.title}") in this repo.`,
    `Run \`backlog task ${ticket.id} --plain\` first to see the full ticket, then search the web for`,
    "relevant prior art, library documentation, or best practices that would help write a thorough",
    "implementation plan. Return a concise research summary (bullet points), not a plan.",
  ].join(" ");
  const research = await runHeadless(pi, cwd, researchPrompt, { timeout: RESEARCH_TIMEOUT_MS, model: "research" });
  await recordHistory(cwd, state, {
    kind: "plan",
    ticket: ticket.id,
    outcome: research.ok ? "ok" : "failed",
    summary: `research: ${summarize(research, 120)}`,
  });

  setCurrentStep(ctx, state, `planning ${ticket.id}`);
  const planPrompt = [
    `/backlog-planner ${ticket.id}`,
    "",
    "Research gathered before planning:",
    research.ok ? research.output : "(research step failed or returned nothing; plan from repo context alone)",
    "",
    `After planning completes (the ticket has a plan and, if applicable, is labeled planned), set its`,
    `status to Dev Ready: \`backlog task edit ${ticket.id} -s "Dev Ready"\`. If /backlog-planner instead`,
    "exited early because it found unplanned child tickets, leave the status as-is and explain why in",
    "your final message.",
  ].join("\n");
  const plan = await runHeadless(pi, cwd, planPrompt, { timeout: PLAN_TIMEOUT_MS, model: "planning" });
  await recordHistory(cwd, state, {
    kind: "plan",
    ticket: ticket.id,
    outcome: plan.ok ? "ok" : "failed",
    summary: summarize(plan),
  });
  return plan.ok;
}

async function doChoose(
  pi: ExtensionAPI,
  ctx: ExtensionCommandContext,
  cwd: string,
  state: RalphState,
  candidates: Ticket[],
): Promise<boolean> {
  if (candidates.length === 1) {
    const only = candidates[0];
    setCurrentStep(ctx, state, `queuing ${only.id} for planning`);
    const ok = await markNeedsPlan(pi, cwd, only.id);
    await recordHistory(cwd, state, {
      kind: "choose",
      ticket: only.id,
      outcome: ok ? "ok" : "failed",
      summary: `marked Needs Plan (${only.title})`,
    });
    return ok;
  }

  setCurrentStep(ctx, state, "choosing next ticket");
  const list = candidates.map((c) => `${c.id} - ${c.title}`).join("\n");
  const prompt = [
    "The following backlog tickets are unblocked (all dependencies Done) and waiting to be picked up:",
    "",
    list,
    "",
    "Pick exactly one to queue for planning next, using your judgment about priority, what unblocks the",
    'most future work, and risk. Then run: backlog task edit <chosen-id> -s "Needs Plan". End your final',
    "message with a line containing only the chosen ticket ID.",
  ].join("\n");
  const result = await runHeadless(pi, cwd, prompt, { timeout: CHOOSE_TIMEOUT_MS });
  const chosenId = extractTicketId(result.output, candidates.map((c) => c.id));
  const ok = result.ok && !!chosenId;
  await recordHistory(cwd, state, {
    kind: "choose",
    ticket: chosenId,
    outcome: ok ? "ok" : "failed",
    summary: summarize(result, 160),
  });
  return ok;
}

async function doReview(
  pi: ExtensionAPI,
  ctx: ExtensionCommandContext,
  cwd: string,
  state: RalphState,
): Promise<boolean> {
  const n = Math.max(state.executedSinceReview, 1);
  setCurrentStep(ctx, state, `reviewing last ${n} ticket(s)`);
  const prompt = [
    "You are the review checkpoint for pi's autonomous backlog loop. Use the herdr CLI (you are already",
    `running inside a herdr-managed pane) to have a fresh claude subagent audit the last ${n} completed`,
    "ticket(s):",
    "",
    "1. Split the current pane to open a new one for the review agent (e.g. `herdr pane split <this-pane-id>",
    "   --direction right --no-focus`); find <this-pane-id> via `herdr pane list`.",
    `2. Run in the new pane: \`claude "Run the review-pi-work skill for the last ${n} tickets"\`.`,
    "3. Wait until that pane's agent finishes (poll `herdr pane list` for the pane's `agent_status` field",
    "   becoming `done`; give it up to 20 minutes).",
    "4. Read its final output (`herdr pane read <new-pane-id> --source recent --lines 400`) and summarize",
    "   what it found, including any new follow-up ticket IDs it filed.",
    "5. Close the review pane (`herdr pane close <new-pane-id>`).",
    "",
    "Report back a concise summary of the review findings and any follow-up ticket IDs filed.",
  ].join("\n");
  const result = await runHeadless(pi, cwd, prompt, { timeout: REVIEW_TIMEOUT_MS });
  await recordHistory(cwd, state, {
    kind: "review",
    outcome: result.ok ? "ok" : "failed",
    summary: summarize(result, 300),
  });
  state.executedSinceReview = 0;
  return result.ok;
}

// --- Loop driver -------------------------------------------------------------

function finish(state: RalphState, status: RalphStatus, reason: string): void {
  state.status = status;
  state.currentStep = reason;
}

/** True if this step's failure streak just hit the cap; `finish()`s the state with an explanatory reason. */
function stoppedByFailureStreak(state: RalphState, key: string, ok: boolean): boolean {
  if (!trackFailureStreak(state, key, ok)) return false;
  finish(
    state,
    "stopped",
    `stopping: "${key}" failed ${MAX_CONSECUTIVE_FAILURES} times in a row. This looks like a systemic ` +
      "problem (a hung subprocess or broken tool), not a one-off bad ticket — check .pi/ralph/history.jsonl " +
      "before restarting.",
  );
  return true;
}

async function runLoop(pi: ExtensionAPI, ctx: ExtensionCommandContext, cwd: string, state: RalphState): Promise<void> {
  try {
    while (true) {
      if (state.stopRequested) {
        finish(state, "stopped", "stop requested");
        break;
      }
      if (state.loopCount >= state.iterations) {
        finish(state, "done", `reached ${state.iterations} iteration(s)`);
        break;
      }

      if (state.reviewCount >= state.reviewEvery) {
        const ok = await doReview(pi, ctx, cwd, state);
        state.reviewCount = 0;
        state.loopCount += 1;
        await persist(cwd, state);
        renderWidget(ctx, state);
        if (stoppedByFailureStreak(state, "review", ok)) break;
        continue;
      }

      state.loopCount += 1;
      state.reviewCount += 1;

      const active = (await findFirstByStatus(pi, cwd, "In Progress")) ?? (await findFirstByStatus(pi, cwd, "Dev Ready"));
      if (active) {
        const ok = await doExecute(pi, ctx, cwd, state, active);
        await persist(cwd, state);
        renderWidget(ctx, state);
        if (stoppedByFailureStreak(state, `execute:${active.id}`, ok)) break;
        continue;
      }

      const needsPlan = await findFirstByStatus(pi, cwd, "Needs Plan");
      if (needsPlan) {
        const ok = await doPlan(pi, ctx, cwd, state, needsPlan);
        await persist(cwd, state);
        renderWidget(ctx, state);
        if (stoppedByFailureStreak(state, `plan:${needsPlan.id}`, ok)) break;
        continue;
      }

      const unblocked = await listUnblocked(pi, cwd);
      if (unblocked.length === 0) {
        finish(state, "done", "no unblocked tickets remain");
        break;
      }
      const ok = await doChoose(pi, ctx, cwd, state, unblocked);
      await persist(cwd, state);
      renderWidget(ctx, state);
      if (stoppedByFailureStreak(state, "choose", ok)) break;
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    finish(state, "stopped", `unexpected error: ${message}`);
  } finally {
    await persist(cwd, state);
    renderWidget(ctx, state);
    ctx.ui.notify(
      `ralph ${state.status}: ${state.currentStep ?? ""} (${state.loopCount}/${state.iterations} iterations)`,
      state.status === "done" ? "info" : "warn",
    );
  }
}

// --- Progress UI ---------------------------------------------------------

function widgetLines(state: RalphState): string[] {
  const step = state.currentStep ? ` · ${state.currentStep}` : "";
  return [`ralph: ${state.status} · iter ${state.loopCount}/${state.iterations} · review ${state.reviewCount}/${state.reviewEvery}${step}`];
}

function renderWidget(ctx: ExtensionCommandContext, state: RalphState): void {
  ctx.ui.setWidget("ralph", widgetLines(state));
}

type DashboardTheme = { bold: (s: string) => string; fg: (color: string, s: string) => string };

const plainTheme: DashboardTheme = { bold: (s) => s, fg: (_c, s) => s };

function renderDashboardLines(state: RalphState, theme: DashboardTheme): string[] {
  const lines: string[] = [];
  lines.push(theme.bold(theme.fg("accent", "Ralph Loop")));
  lines.push(`status: ${state.status}`);
  lines.push(`iteration: ${state.loopCount} / ${state.iterations}`);
  lines.push(`since last review: ${state.reviewCount} / ${state.reviewEvery}`);
  if (state.currentStep) lines.push(`current: ${state.currentStep}`);
  lines.push("");
  lines.push(theme.bold("recent history"));
  const recent = state.history.slice(-10).reverse();
  if (recent.length === 0) {
    lines.push("  (none yet)");
  } else {
    for (const entry of recent) {
      const marker = entry.outcome === "ok" ? "✓" : "✗";
      const ticketPart = entry.ticket ? ` ${entry.ticket}` : "";
      lines.push(`  ${marker} [${entry.kind}]${ticketPart} — ${entry.summary}`);
    }
  }
  lines.push("");
  lines.push(theme.fg("muted", "Esc to close (updates live while ralph runs)"));
  return lines;
}

async function showProgressDashboard(ctx: ExtensionCommandContext, state: RalphState): Promise<void> {
  await ctx.ui.custom<void>((tui, theme, _keybindings, done) => {
    let cachedWidth: number | undefined;
    let cachedLines: string[] | undefined;

    const interval = setInterval(() => {
      cachedWidth = undefined;
      cachedLines = undefined;
      tui.requestRender();
    }, 1000);

    const close = () => {
      clearInterval(interval);
      done();
    };

    return {
      render(width: number): string[] {
        if (cachedWidth === width && cachedLines) return cachedLines;
        cachedLines = renderDashboardLines(state, theme).map((line) => truncateToWidth(line, width));
        cachedWidth = width;
        return cachedLines;
      },
      invalidate(): void {
        cachedWidth = undefined;
        cachedLines = undefined;
      },
      handleInput(data: string): void {
        if (matchesKey(data, Key.escape)) close();
      },
    };
  });
}

// --- Commands ----------------------------------------------------------------

function parsePositiveInt(token: string): number | undefined {
  const parsed = Number.parseInt(token, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
}

export default function (pi: ExtensionAPI) {
  pi.registerCommand("ralph", {
    description: "Start the autonomous backlog loop (plan/execute/review tickets until done or iteration limit)",
    handler: async (args, ctx) => {
      if (activeState && (activeState.status === "running" || activeState.status === "stopping")) {
        ctx.ui.notify(
          `ralph is already ${activeState.status} (iteration ${activeState.loopCount}/${activeState.iterations}). Use /ralph-stop first.`,
          "warn",
        );
        return;
      }
      if (process.env.HERDR_ENV !== "1") {
        ctx.ui.notify(
          "ralph requires running inside a herdr-managed pane (HERDR_ENV=1) because the review step drives a herdr pane.",
          "error",
        );
        return;
      }

      const tokens = (args ?? "").trim().split(/\s+/).filter(Boolean);
      let iterations = DEFAULT_ITERATIONS;
      let reviewEvery = DEFAULT_REVIEW_EVERY;

      if (tokens.length >= 1) {
        const parsed = parsePositiveInt(tokens[0]);
        if (parsed === undefined) {
          ctx.ui.notify(`Invalid iterations value: "${tokens[0]}"`, "error");
          return;
        }
        iterations = parsed;
      }
      if (tokens.length >= 2) {
        const parsed = parsePositiveInt(tokens[1]);
        if (parsed === undefined) {
          ctx.ui.notify(`Invalid reviewEvery value: "${tokens[1]}"`, "error");
          return;
        }
        reviewEvery = parsed;
      }

      const cwd = ctx.cwd;
      activeState = createState(iterations, reviewEvery);
      await persist(cwd, activeState);
      renderWidget(ctx, activeState);
      ctx.ui.notify(`ralph started: ${iterations} iteration(s), reviewing every ${reviewEvery}.`, "info");

      void runLoop(pi, ctx, cwd, activeState);
    },
  });

  pi.registerCommand("ralph-stop", {
    description: "Request a graceful stop of the running ralph loop",
    handler: async (_args, ctx) => {
      if (!activeState || activeState.status !== "running") {
        ctx.ui.notify("ralph is not currently running.", "info");
        return;
      }
      activeState.stopRequested = true;
      activeState.status = "stopping";
      renderWidget(ctx, activeState);
      ctx.ui.notify("ralph will stop after the current step finishes.", "info");
    },
  });

  pi.registerCommand("ralph-progress", {
    description: "Show the ralph loop's current progress",
    handler: async (_args, ctx) => {
      if (!activeState) {
        ctx.ui.notify("ralph has not been run yet in this session. Use /ralph to start it.", "info");
        return;
      }
      if (ctx.mode === "tui") {
        await showProgressDashboard(ctx, activeState);
      } else {
        ctx.ui.notify(renderDashboardLines(activeState, plainTheme).join("\n"), "info");
      }
    },
  });

  pi.on("session_shutdown", async () => {
    if (activeState && (activeState.status === "running" || activeState.status === "stopping")) {
      activeState.status = "stopped";
      activeState.currentStep = "session ended";
    }
  });
}
