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
  - **7.7 is built and measured, not committed.** The trainer: `mlp-train.mjs`, `train.mjs`
    (`npm run agents:train`), `cli.mjs`, `trainingSet`, and U11.
    - **Both species meet the 0.05 bar at round 0:** prey turn 0.033, wolves turn 0.011, every
      throttle at most 0.0002.
    - **Settings:** 24→64→64→64→2 at 200 passes, with the step size from 1e-2 to 1e-4. They are now
      the trainer's defaults.
    - **It took:** about 45 minutes for prey and 20 for wolves, one core each.
    - **The draft's start fell far short for prey:** 24→32→32→2 at 40 passes gave 0.177.
    - **Only the wall branch,** about 5% of rows, is above 0.05 (prey 0.101, wolves 0.058).
  - **Next: 7.8,** `evaluate.mjs`: the teacher, the rule without wander, and the students in the
    running simulation; catches within 10%; the students' error on their own runs.

## State of the repositories

- **null-graph** (`master`, not pushed). Uncommitted, all phase 7.7:
  - new: `tools/agents/mlp-train.mjs`, `train.mjs`, `cli.mjs`;
  - changed: `dataset.mjs` (`trainingSet`), `record.mjs` (uses `cli.mjs`),
    `test/agents-tools.test.mjs` (U11), `package.json` (`agents:train`);
  - docs: the cloning design, and `docs/work` (README, gotchas, plan, log, this file).
- **The students and the recording** are in `tools/agents/out/` (ignored):
  - the recording: `herd-wolves/`, made at `cf3f972`;
  - the round-0 students: `e4-64x64x64-200-r01/`;
  - the shape and schedule runs: the other folders.
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`),
  left from 5.6's validation. It holds a `node_modules` junction. Remove that first with
  `cmd /c rmdir`, which drops the link only, before `git worktree remove`, which may follow it.
- **Green:**
  - headless: agents (reference + 68), core, agent-kit 4/4;
  - `agent-tools` 10/10 on three host/GPU pairs (re-run at 7.6's commit);
  - the agents suite 137/137.
- **Bundle:** unchanged by part 2 (agents 19,323 B; C2 ceiling 21,495 B).

## Open questions for the user

- Commit 7.7?
- [`plan.md`](./plan.md) §5, item 3 (the brain-swap stall) is deferred to step 9.
