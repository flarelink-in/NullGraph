# Now — 2026-10-13

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.**
  - Step 6 (`MlpBrain`) is complete.
  - **Step 7, part 1 (per-species brains) is complete and committed**, through 7.5 (`de14ff4`)
    ([`../agents-species-brains-design.md`](../agents-species-brains-design.md)).
- **Step 5's deterministic neighbour lookups are complete** (phases 5.5–5.8,
  [`../agents-ordered-grid-design.md`](../agents-ordered-grid-design.md)). The scaling study, the
  rest of step 5, is still deferred.
  - **What it gives:** on one device, lockstep reruns are bit-exact with or without grid overflow.
    A stable radix sort orders each cell by index; a crowded cell yields a window that rotates
    per cell and tick.
  - **5.7 is committed** (`de87b3f`, validated alone in a worktree). A step is between −15% and
    +9.6% of the code before step 5 on the acceptance layouts, on three host/GPU pairs.
  - **5.8 is committed with this record:** the two guides, the parent designs and `docs/work`
    state the new claim, today's custom-sensor limits and the closure. No code changed.
- **Next: step 7, part 2,** asked for by the user: record the rule brain → train a copy per
  species → export. It has no design yet, so the design comes first ([`plan.md`](./plan.md) §6).

## State of the repositories

- **null-graph** (`master`, not pushed). Nothing is uncommitted after phase 5.8's commit.
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`),
  left from 5.6's validation. Remove it with `git worktree remove` when it is no longer wanted.
- **Raw results:** `%TEMP%/ng-policy/` (before the change) and `…/after-the-change/`.
- **`dist/`** is a development build of the working tree, plus `dist/grid-policy-*.js`: bundles of
  `b7050b7` and `da262ad` for the spike (`node test/spikes/grid-policy/prepare.mjs` after a build).
- **Green today:** headless agents (reference + 63), core, agent-kit 4/4, TypeScript; the agents
  GPU suite, 137/137 with zero GPU errors, on Node/Dawn AMD, Node/Dawn NVIDIA and Chrome.
- **Bundle:** agents whole entry 19,323 B min+gzip; 2,172 B below the C2 ceiling.
- **Test-engine site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`): its
  `node_modules/null-graph/dist` holds the limits build (`7f1acb0`), from before step 5. Refresh it
  to see the ordered grid in the demo; restart its dev server with `npm run dev -- --force` after
  any refresh ([`gotchas.md`](./gotchas.md)).

## Open questions for the user

- Not blocking, in the design's closure and [`gotchas.md`](./gotchas.md):
  - the scan loop costs the AMD iGPU 3 to 7 points at 65,536 flocked, where it gains 10 to 18 at
    4,096; a third form was not tried;
  - the CPU `pointCell` is not the GPU's (f64 against f32);
  - the acceptance spike's cost half cannot judge a before/after;
  - a comb would sample a very crowded cell more evenly than a window.
- [`plan.md`](./plan.md) §5, item 3 (the brain-swap stall) is deferred to step 9.
