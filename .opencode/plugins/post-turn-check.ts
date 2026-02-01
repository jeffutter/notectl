import { promises as fs } from "node:fs";
import { tmpdir } from "node:os";
import type { Plugin } from "@opencode-ai/plugin";

let hasEdited = false;
const cooldownMs = 15_000;
let lastRunAt = 0;

export const PostTurnCheck: Plugin = async ({ client, $ }) => {
  // Don't await - fire and forget to avoid blocking plugin initialization
  client.app.log({
    service: "post-turn-check",
    level: "info",
    message: "Plugin loaded and initialized",
  }).catch(() => {});

  return {
    "tool.execute.after": async (input) => {
      // Track when files are edited or created
      const editTools = ["write", "edit"];
      if (editTools.includes(input.tool)) {
        hasEdited = true;
        client.app.log({
          service: "post-turn-check",
          level: "info",
          message: `Edit detected: ${input.tool} on ${input.args?.filePath || "unknown"}`,
        }).catch(() => {});
      }
    },

    event: async ({ event }) => {
      if (event.type !== "session.idle") return;
      if (!hasEdited) {
        // Log when idle fires but no edits were made (for debugging)
        client.app.log({
          service: "post-turn-check",
          level: "debug",
          message: "Session idle but no edits detected, skipping checks",
        }).catch(() => {});
        return;
      }

      const now = Date.now();
      if (now - lastRunAt < cooldownMs) {
        client.app.log({
          service: "post-turn-check",
          level: "info",
          message: `Skipping check - cooldown active (${Math.round((cooldownMs - (now - lastRunAt)) / 1000)}s remaining)`,
        }).catch(() => {});
        return;
      }

      lastRunAt = now;
      hasEdited = false;

      client.app.log({
        service: "post-turn-check",
        level: "info",
        message: "🔍 Running post-turn cargo checks...",
      }).catch(() => {});

      const cargoFmtOutputFile = `${tmpdir()}/opencode-check-${Date.now()}-cargofmt.log`;
      const cargoClippyOutputFile = `${tmpdir()}/opencode-check-${Date.now()}-cargoclippy.log`;

      await $`sh -c ${"cargo fmt --all > " + cargoFmtOutputFile + " 2>&1 || true"}`;
      await $`sh -c ${"cargo clippy --all-targets --all-features --workspace -- -D warnings > " + cargoClippyOutputFile + " 2>&1 || true"}`;

      const cargoFmtOutput = await fs.readFile(cargoFmtOutputFile, "utf8").catch(() => "");
      const cargoClippyOutput = await fs.readFile(cargoClippyOutputFile, "utf8").catch(() => "");

      const message = `
Post-turn lint check completed.

--- BEGIN CARGO FMT OUTPUT ---
${cargoFmtOutput || "No issues found."}
--- END CARGO FMT OUTPUT ---

--- BEGIN CARGO CLIPPY OUTPUT ---
${cargoClippyOutput || "No issues found."}
--- END CARGO CLIPPY OUTPUT ---

If there are errors, fix them. If something's unclear, ask.
`.trim();

      await fs.unlink(cargoFmtOutputFile);
      await fs.unlink(cargoClippyOutputFile);

      const sessionID = event.properties.sessionID;
      if (sessionID) {
        await client.session.prompt({
          path: { id: sessionID },
          body: {
            parts: [{ type: "text", text: message }],
          },
        });
      }
    },
  };
};
