# Now — 2026-10-13

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.**
  - Step 6 (`MlpBrain`) is complete.
  - **Step 7, part 1 (per-species brains) is complete and committed**, through 7.5 (`de14ff4`).
  - **Step 5's deterministic neighbour lookups are complete and committed** (phases 5.5–5.8;
    5.7 `de87b3f`, 5.8 `8ed32a5`). The scaling study, the rest of step 5, is still deferred.
- **Now: step 7, part 2: record → train → export**
  ([`../agents-cloning-design.md`](../agents-cloning-design.md), decided and committed, `f1244cb`).
  The first network that has learned Herd & Wolves. Phases 7.6–7.10.
  - **7.6 is committed** (`c2608d7`): the demo's scenario in this repository, the label pass, the
    dataset format and the recorder (`npm run agents:record`).
    - The scenario decides exactly what the site's rule decides: 0 rows differ over 600 steps.
    - A recording repeats byte for byte on one device: 5 episodes of 30 s in 22 s on the AMD iGPU.
    - **0.082 of the prey's turn is hidden input** (wander, coin-flips): a copy of the raw
      actions could not reach the 0.05 bar; a copy of the label can.
    - The teacher catches about 1,030 prey in 30 s, and seeds move that by about 2%.
    - 62% of wolf rows are frozen: about 147,000 usable wolf rows beside 360,000 prey rows.
    - Tests: U24, U25, U27; A24, A25 in a new GPU suite, `agent-tools`, 10/10 on three host/GPU
      pairs; 27 mutations seen failing.
  - **Now: 7.7, the trainer**, asked for by the user: `mlp-train.mjs` and `train.mjs` in plain
    JS; U11 and a gradient check; held-out error per species, action and reason, for three shapes.

## State of the repositories

- **null-graph** (`master`, not pushed): all committed, through 7.6.
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`),
  left from 5.6's validation. Remove it with `git worktree remove` when it is no longer wanted.
- **Raw results:** `%TEMP%/ng-cloning/` (7.6: the recordings' logs, the GPU suite's logs, the two
  scratch scripts); `%TEMP%/ng-policy/` (step 5).
- **`dist/`** is a development build of the working tree, plus `dist/grid-policy-*.js` for step
  5's spike (`node test/spikes/grid-policy/prepare.mjs` after a build).
- **Green today:** headless agents (reference + 66), core, agent-kit 4/4, TypeScript; the agents
  GPU suite 137/137 and the `agent-tools` suite 10/10, with zero GPU errors, on Node/Dawn AMD,
  Node/Dawn NVIDIA and Chrome.
- **Bundle:** unchanged by part 2: agents whole entry 19,323 B; 2,172 B below the C2 ceiling.
- **Test-engine site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`):
  - its `node_modules/null-graph/dist` holds the limits build (`7f1acb0`), from before step 5;
    restart its dev server with `npm run dev -- --force` after any refresh
    ([`gotchas.md`](./gotchas.md));
  - its `HerdWolvesGPU/brain.ts` is the rule this repository's scenario copies. If either changes,
    rerun the equivalence check (the design's 7.6 results).

## Open questions for the user

- Not blocking, from step 5 ([`gotchas.md`](./gotchas.md) and the ordered-grid design's closure):
  a third form of the scan loop; the CPU `pointCell` (f64 against the GPU's f32); the acceptance
  spike's cost half; a comb instead of a window.
- [`plan.md`](./plan.md) §5, item 3 (the brain-swap stall) is deferred to step 9.
