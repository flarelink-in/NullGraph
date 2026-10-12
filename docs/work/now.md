# Now — 2026-10-13

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.**
  - Step 6 (`MlpBrain`) is complete.
  - **Step 7, part 1 (per-species brains) is complete and committed**, through 7.5 (`de14ff4`)
    ([`../agents-species-brains-design.md`](../agents-species-brains-design.md)).
- **Now: step 5, deterministic neighbour lookups only.** The user resumed step 5 so that step 7,
  part 2 can record reproducible teacher data; the scaling study comes later.
  - **Design decided:** [`../agents-ordered-grid-design.md`](../agents-ordered-grid-design.md)
    ([`plan.md`](./plan.md) §5, item 6):
    - a stable radix sort by cell replaces the atomic scatter, as a single path;
    - crowded cells yield a window rotating per querying agent and tick;
    - the numbers come back to the user only if the sort adds more than about 10% to a step at
      4,096 or 65,536 agents, on any host.
  - **5.5 measured** (the spike `test/spikes/grid-order/`, committed in `e4eaad2`):
    - today's grid diverges on all three host/GPU pairs: about 16,300–19,300 of 65,536 record
      words differ after 200 identical overflowing steps;
    - the grid build is 0.3–3.4% of a step;
    - under the 10% rule the sort may add at most about 51–89 µs at 4,096 agents (the tightest is
      NVIDIA's 51 µs), and 1.4–12 ms at 65,536.
  - **5.6 correctness complete:** stable radix scatter, window/tick bindings and CPU mirror;
    exact A1 including AgentGrid ticks 1–5; overflowing A23 green and atomic-slot mutation
    watched failing on every host. Agents GPU: 137/137 on Node/AMD, Node/NVIDIA and Chrome/AMD,
    zero GPU errors. U23 includes allocation/binding failure cleanup.
  - **Next:** 5.7 timings and bundle, measured in isolation; apply the agreed 10% rule.
  - **Then:** 5.7 (cost after, with the same spike) and 5.8 (docs, including today's custom-sensor
    limits and the parent's claim).
- **Then: step 7, part 2:** record → train → export, per species.

## State of the repositories

- **null-graph** (`master`, not pushed): phase 5.5 and initial 5.6 committed as `e4eaad2`;
  phase 5.6 closure is recorded in the following commit.
  - **Uncommitted, not Claude's:** camera input work (`src/extras/cameras/OrbitalEventProxy.ts`,
    `docs/api/cameras.md`, `test/cameras-input.test.mjs`, a `test:cameras` script in
    `package.json`). Leave it to its owner, and validate commits alone in a worktree. Do not add
    npm scripts while `package.json` holds that change.
- **Green:**
  - headless: agents (reference + 63), core, agent-kit 4/4;
  - the agents GPU suite: 137/137 on Node/Dawn with the AMD iGPU and with the NVIDIA GPU, and in
    Chrome.
- **Bundle:** before ordered grid, `agents.js` 18,178 B min+gzip (C2 ceiling 21,495 B);
  ordered-grid measurement pending in 5.7.
- **Test-engine site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`): its
  `node_modules/null-graph/dist` holds the limits build (`7f1acb0`). Restart its dev server with
  `npm run dev -- --force` after any refresh ([`gotchas.md`](./gotchas.md)).

## Open questions for the user

- [`plan.md`](./plan.md) §5, item 3 (the brain-swap stall) is deferred to step 9.
