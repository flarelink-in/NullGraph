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
  - **Next: phase 5.5**, a spike on today's code, on three host/GPU pairs:
    - A23's scenario with heavy overflow, counting the differing words;
    - the baseline step time and grid share at 4,096 and 65,536 agents, with Herd & Wolves' setup.
  - **Then:** 5.6 (build), 5.7 (cost after), 5.8 (docs, including today's custom-sensor limits and
    the parent's claim).
- **Then: step 7, part 2:** record → train → export, per species.

## State of the repositories

- **null-graph** (`master`, not pushed): Claude's work is all committed.
  - **Uncommitted, not Claude's:** camera input work (`src/extras/cameras/OrbitalEventProxy.ts`,
    `docs/api/cameras.md`, `test/cameras-input.test.mjs`, a `test:cameras` script in
    `package.json`). Leave it to its owner, and validate commits alone in a worktree. Do not add
    npm scripts while `package.json` holds that change.
- **Green:**
  - headless: agents (reference + 59), core, agent-kit 4/4;
  - the agents GPU suite: 124/124 on Node/Dawn with the AMD iGPU and with the NVIDIA GPU, and in
    Chrome.
- **Bundle:** `agents.js` 18,178 B min+gzip (C2 ceiling 21,495 B).
- **Test-engine site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`): its
  `node_modules/null-graph/dist` holds the limits build (`7f1acb0`). Restart its dev server with
  `npm run dev -- --force` after any refresh ([`gotchas.md`](./gotchas.md)).

## Open questions for the user

- [`plan.md`](./plan.md) §5, item 3 (the brain-swap stall) is deferred to step 9.
