# Now — 2026-10-13

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.**
  - Step 6 (`MlpBrain`) is complete.
  - **Step 7, part 1 (per-species brains) is complete and committed**, through 7.5 (`de14ff4`)
    ([`../agents-species-brains-design.md`](../agents-species-brains-design.md)).
- **Now: step 5, resumed by the user** for deterministic neighbour lookups (grid-ordered queries),
  so that step 7, part 2 can record reproducible teacher data even when crowded cells overflow.
  - The rest of the scaling study ([`../agents-demo-design.md`](../agents-demo-design.md)) stays
    deferred. This is Claude's reading of "build step 5 first", recorded in [`plan.md`](./plan.md).
    The step 5 design asks the user to confirm it.
  - **Next action:** the design: where the grid's order comes from, the options and their costs,
    and the questions for the user.
- **Then: step 7, part 2:** record → train → export, per species (`tools/agents`).

## State of the repositories

- **null-graph** (`master`, not pushed): Claude's work is all committed.
  - **Uncommitted, not Claude's:** camera input work (`src/extras/cameras/OrbitalEventProxy.ts`,
    `docs/api/cameras.md`, `test/cameras-input.test.mjs`, a `test:cameras` script in
    `package.json`). Leave it to its owner, and validate commits alone in a worktree.
- **Green:**
  - headless: agents (reference + 59, including the guide's three examples), core, agent-kit 4/4;
  - the agents GPU suite: 124/124 on Node/Dawn with the AMD iGPU and with the NVIDIA GPU, and in
    Chrome.
- **Bundle:** `agents.js` 18,178 B min+gzip (C2 ceiling 21,495 B).
- **Test-engine site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`): its
  `node_modules/null-graph/dist` holds the limits build (`7f1acb0`). Restart its dev server with
  `npm run dev -- --force` after any refresh ([`gotchas.md`](./gotchas.md)).

## Open questions for the user

- Confirm step 5's scope: the deterministic lookups only, or the whole scaling study first?
- Custom sensors can read the neighbour lists, but not by any contract. Make that a documented
  input (generated layout constants, CPU-mirror access, maybe app buffers such as walls)?
- [`plan.md`](./plan.md) §5, item 3 (the brain-swap stall) is deferred to step 9.
