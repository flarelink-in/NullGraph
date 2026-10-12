# Now — 2026-10-12

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.** Step 6 (`MlpBrain`) is complete. Commits: 6.0 `e345921`, 6.1 `dbaf884`,
  6.2 `b3424b1`, 6.3 `d8ad5b2`; core limits `7f1acb0`; the per-species design `ed195b3`; 7.1
  `1dc7e74`.
- **Step 7, part 1: per-species brains**
  ([`../agents-species-brains-design.md`](../agents-species-brains-design.md)):
  - **7.1 (rows) is committed.**
  - **7.2 is done but not committed:** A21 on both hosts, and costs at 2, 8 and 16 species.
  - **Decided, option B:** one think pass per brain, gated to its species
    (`setBrain(brain, { species })`). Rows stay for 8 or more species: they win at 16 everywhere.
    The composite is withdrawn.
  - **§1–13 were rewritten for B and wait for the user's review. Next: phase 7.3** (`AgentSystem`
    passes and the slot table).
- **For step 7, part 2's design:** record teacher data with overflow off, or pull step 5's
  grid-ordered queries forward (the user's call).
- Step 5 is deferred, not done.

## State of the repositories

- **null-graph** (`master`, not pushed): uncommitted 7.2 work (`test/agents-gpu/run.js` A21;
  `test/spikes/mlp-think-cost/run.js` with the rows, switch, per-species and 8/16-species
  kernels) and the docs above.
  - **Also uncommitted, not Claude's:** camera input work (`src/extras/cameras/OrbitalEventProxy.ts`,
    `docs/api/cameras.md`, `test/cameras-input.test.mjs`, a `test:cameras` script in
    `package.json`). Leave it to its owner.
- **Green:** headless agents 24 + 54, core, agent-kit; the agents GPU suite 122/122 on Node/Dawn and
  Chrome.
- **Bundle:** `agents.js` 17,827 B min+gzip (C2 ceiling 21,495 B).
- **Test-engine site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`): its
  `node_modules/null-graph/dist` holds the limits build (`7f1acb0`). Restart its dev server with
  `npm run dev -- --force` after any refresh ([`gotchas.md`](./gotchas.md)).

## Open questions for the user

- A review of the revised §6 (per-species passes, the slot table) before phase 7.3's code.
- [`plan.md`](./plan.md) §5, item 3 (the brain-swap stall) is deferred to step 9. Per-species passes
  shrink it: switching back to a brain that already has a stage compiles nothing.
