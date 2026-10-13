# Now — 2026-10-13

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.**
  - Step 6 (`MlpBrain`) is complete.
  - **Step 7, part 1 (per-species brains) is complete and committed**, through 7.5 (`de14ff4`)
    ([`../agents-species-brains-design.md`](../agents-species-brains-design.md)).
- **Now: step 5, deterministic neighbour lookups only**
  ([`../agents-ordered-grid-design.md`](../agents-ordered-grid-design.md)), so that step 7, part 2
  can record reproducible teacher data. The scaling study comes later.
  - **5.5 and 5.6 are committed** (`e4eaad2`, `caaf930`): a stable radix sort by cell, a window
    for crowded cells, the CPU mirror. Reruns are bit-exact with overflow.
  - **5.7 is within the 10% rule, and committed with this record.**
    - First measured over the rule (`da262ad`). The follow-up found the sort cheap (at most +8%)
      and the cost in the window per querying agent: +15% to +26% at 65,536 agents flocked.
    - **The user revised decision 2 (2026-10-13): a window per cell and tick,** with a tighter
      scan loop. Every query of a crowded cell reads the same entries again.
    - **Measured after the change,** like for like on three host/GPU pairs: between −15% and
      +9.6% against the code before step 5. The worst case is Chrome at 65,536 flocked.
    - Agents GPU suite 137/137 on all three; headless agents reference + 63; bundle +18 B.
  - **Next: 5.8,** asked for by the user: the guides' determinism paragraphs, today's
    custom-sensor limits, the parent's claim, closure.
- **Then: step 7, part 2:** record → train → export, per species.

## State of the repositories

- **null-graph** (`master`, not pushed). Nothing is uncommitted after phase 5.7's commit.
  - `test/spikes/grid-policy/` is the like-for-like measurement (`prepare.mjs` first).
  - `test/spikes/grid-query/`, the earlier attribution spike, was deleted at the user's word.
- **Raw results:** `%TEMP%/ng-policy/` (before the change) and `…/after-the-change/`.
- **`dist/`** is a development build of the working tree, plus `dist/grid-policy-*.js`: bundles of
  `b7050b7` and `da262ad` for the spike (`node test/spikes/grid-policy/prepare.mjs` after a build).
- **Green today:** headless agents (reference + 63) and TypeScript; the agents GPU suite, 137/137
  with zero GPU errors, on Node/Dawn AMD, Node/Dawn NVIDIA and Chrome. Core and agent-kit: as of
  5.6, not rerun.
- **Bundle:** agents whole entry 19,323 B min+gzip; 2,172 B below the C2 ceiling.
- **Test-engine site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`): its
  `node_modules/null-graph/dist` holds the limits build (`7f1acb0`). Restart its dev server with
  `npm run dev -- --force` after any refresh ([`gotchas.md`](./gotchas.md)).

## Open questions for the user

- Not blocking, in the design's "Left open" and [`gotchas.md`](./gotchas.md):
  - the scan loop costs the AMD iGPU 3 to 7 points at 65,536 flocked, where it gains 10 to 18 at
    4,096; a third form was not tried;
  - the CPU `pointCell` is not the GPU's (f64 against f32);
  - the acceptance spike's cost half cannot judge a before/after.
- [`plan.md`](./plan.md) §5, item 3 (the brain-swap stall) is deferred to step 9.
