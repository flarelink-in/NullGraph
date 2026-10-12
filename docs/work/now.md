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
  - **5.7 measured; acceptance paused for the user:** several whole-step cases exceed ~10%
    (up to +51.6%). Grid-build increase alone is within 10% of the old step in every case.
    All three hosts now record zero differing words. Full numbers are in the design.
  - **Next:** discuss performance regressions; recommended investigation is interleaved
    baseline/after runs on fixed snapshots, isolating sort versus rotating-query cost.
    Do not advance to 5.8/teacher data or change the agreed semantics without that discussion.
- **Then: step 7, part 2:** record → train → export, per species.

## State of the repositories

- **null-graph** (`master`, not pushed): ordered grid committed as `e4eaad2` and `caaf930`;
  camera pinch/cleanup fix committed as `9ea8a33`, five tests + TypeScript green in a worktree.
  Phase 5.7 measurements documented; performance acceptance is not granted.
- **Green:**
  - headless: agents (reference + 63), core, agent-kit 4/4;
  - the agents GPU suite: 137/137 on Node/Dawn with the AMD iGPU and with the NVIDIA GPU, and in
    Chrome.
- **Bundle:** agents whole entry 19,305 B min+gzip (+1,127 B); 2,190 B below C2 ceiling.
  Rule path +825 B; MLP still tree-shakes from rules.
- **Test-engine site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`): its
  `node_modules/null-graph/dist` holds the limits build (`7f1acb0`). Restart its dev server with
  `npm run dev -- --force` after any refresh ([`gotchas.md`](./gotchas.md)).

## Open questions for the user

- [`plan.md`](./plan.md) §5, item 3 (the brain-swap stall) is deferred to step 9.
- Ordered-grid performance: whole-step regressions exceed the ~10% rule; see 5.7 results.
