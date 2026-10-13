# Now — 2026-10-13

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.**
  - Step 6 (`MlpBrain`) is complete.
  - Step 7, part 1 (per-species brains) is complete, through 7.5 (`de14ff4`).
  - Step 5's deterministic neighbour lookups are complete (5.5–5.8). The scaling study is still
    deferred.
- **Now: step 7, part 2: record → train → export**
  ([`../agents-cloning-design.md`](../agents-cloning-design.md), `f1244cb`; phases 7.6–7.10).
  - **7.6 is committed:** `c2608d7`, docs `cf3f972`.
  - **7.7 is committed:** `0d3fb9f`. The trainer: `mlp-train.mjs`, `train.mjs`
    (`npm run agents:train`), `cli.mjs`, `trainingSet`, and U11.
    - **Both species meet the 0.05 bar at round 0:** prey turn 0.033, wolves turn 0.011.
    - **Settings (now the defaults):** 24→64→64→64→2 at 200 passes, step size 1e-2 to 1e-4; about
      45 minutes for prey and 20 for wolves, one core each.
  - **7.8 is committed:** `8afc6fa`. The copy in the running simulation:
    `evaluator.mjs`, `evaluate.mjs` (`npm run agents:evaluate`), `model.mjs`, U28, and U26's reader
    half.
    - **The catch bar is met:** the students against the rule without wander, +0.2% (AMD) and −0.6%
      (NVIDIA); the bar is 10%.
    - **No drift:** the students' error on their own runs equals their held-out error, so no round
      of correction was built.
    - **The rule without wander catches as many prey as the rule** (under 1% apart).
  - **7.9 is committed** (the commit after `8afc6fa`). The networks in the repository:
    - `tools/agents/models/herd-wolves/`: two `.mlp.json` files, their reports, the evaluation
      (476 KB; the draft said 50 KB);
    - U26 (writer, committed files), A26, A8; the guide's "A trained network";
      `tools/agents/README.md`;
    - **a retrain wrote both networks byte for byte;**
    - **trained weights are large (up to 178),** so the GPU differs from `mlpForward` by up to
      4.4e-5: beyond A4's bound for a few actions. A26 asserts 1e-4.
  - **Next: 7.10.** The `.onnx` writer; U12; A27 (ONNX Runtime against `mlpForward`).

## State of the repositories

- **null-graph** (`master`, not pushed). Nothing uncommitted after 7.9's commit.
- **The students and the recording** are in `tools/agents/out/` (ignored):
  - the recording: `herd-wolves/`, made at `cf3f972`;
  - the round-0 students: `e4-64x64x64-200-r01/`; the retrain that reproduced them: `r0-check/`;
  - the shape and schedule runs: the other folders.
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`),
  left from 5.6's validation. It holds a `node_modules` junction. Remove that first with
  `cmd /c rmdir`, which drops the link only, before `git worktree remove`, which may follow it.
- **Green:**
  - headless: agents (reference + 72);
  - `agent-tools` 15/15 on three host/GPU pairs. It now takes about a minute, and up to two in
    Chrome.
  - Not re-run this session, and untouched by it: core, agent-kit, the agents GPU suite (137).
- **Bundle:** unchanged by part 2 (agents 19,323 B; C2 ceiling 21,495 B).

## Open questions for the user

- [`plan.md`](./plan.md) §5, item 3 (the brain-swap stall) is deferred to step 9.
