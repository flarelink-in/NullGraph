# Now — 2026-10-13

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.**
  - Step 6 (`MlpBrain`) is complete.
  - **Step 7, part 1 (per-species brains) is complete and committed**, through 7.5 (`de14ff4`).
  - **Step 5's deterministic neighbour lookups are complete and committed** (phases 5.5–5.8;
    5.7 `de87b3f`, 5.8 `8ed32a5`;
    [`../agents-ordered-grid-design.md`](../agents-ordered-grid-design.md)). On one device,
    lockstep reruns are bit-exact with or without grid overflow. The scaling study, the rest of
    step 5, is still deferred.
- **Now: step 7, part 2: record → train → export.** The first network that has learned Herd &
  Wolves.
  - **The design is decided (2026-10-13) and committed with this record:**
    [`../agents-cloning-design.md`](../agents-cloning-design.md). No code yet.
  - **Its main points:**
    - the demo's rule reads three things a network cannot see (the wolf's rest timer, a wander
      hash, index coin-flips), so the label is the rule without them;
    - labels come from a second think pass, spliced into the step, so they do not depend on who
      drives. A scratch probe showed it reproduces the system's own decisions word for word;
    - a plain-JS trainer, one network per species, files that today's `MlpBrain` loads;
    - the copy is measured decision by decision and in the running simulation, against the
      parent's A8 bars; `src/` is not touched.
  - **The user accepted the six recommendations** (the design's §14; [`plan.md`](./plan.md) §5,
    item 7), and kept the `.onnx` phase, 7.10, in this part.
  - **Next: phase 7.6,** asked for by the user: the scenario, the label pass, the dataset format
    and `record.mjs`, with U24, U25, U27, A24, A25 and the first measurements.

## State of the repositories

- **null-graph** (`master`, not pushed). Nothing is uncommitted after the design's commit.
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`),
  left from 5.6's validation. Remove it with `git worktree remove` when it is no longer wanted.
- **Raw results of step 5:** `%TEMP%/ng-policy/` and `…/after-the-change/`.
- **`dist/`** is a development build of the working tree, plus `dist/grid-policy-*.js`: bundles of
  `b7050b7` and `da262ad` for the spike (`node test/spikes/grid-policy/prepare.mjs` after a build).
- **Green today:** headless agents (reference + 63), core, agent-kit 4/4, TypeScript; the agents
  GPU suite, 137/137 with zero GPU errors, on Node/Dawn AMD, Node/Dawn NVIDIA and Chrome.
- **Bundle:** agents whole entry 19,323 B min+gzip; 2,172 B below the C2 ceiling.
- **Test-engine site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`):
  - its `node_modules/null-graph/dist` holds the limits build (`7f1acb0`), from before step 5;
    restart its dev server with `npm run dev -- --force` after any refresh
    ([`gotchas.md`](./gotchas.md));
  - its `HerdWolvesGPU/brain.ts` holds the rule that step 7, part 2 copies. It is not in this
    repository yet.

## Open questions for the user

- Not blocking, from step 5 ([`gotchas.md`](./gotchas.md) and the ordered-grid design's closure):
  a third form of the scan loop; the CPU `pointCell` (f64 against the GPU's f32); the acceptance
  spike's cost half; a comb instead of a window.
- [`plan.md`](./plan.md) §5, item 3 (the brain-swap stall) is deferred to step 9.
