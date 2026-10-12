# Now — 2026-10-13

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.** Step 6 (`MlpBrain`) is complete.
- **Step 7, part 1: per-species brains, option B**
  ([`../agents-species-brains-design.md`](../agents-species-brains-design.md)):
  - **Committed:**
    - 7.1 rows, `1dc7e74`;
    - 7.2, `94069ea`;
    - the option-B design, `30336d1`;
    - 7.3 per-species passes, `681ccf4`;
    - 7.4: A22 `f19dbea` and the `bySpecies` message fix `3c3ad76`. A22 is green on three
      host/GPU pairs.
  - **Decided:** `bySpecies` keeps one network per declared species ([`plan.md`](./plan.md) §5,
    item 5).
  - **Now: phase 7.5:**
    - the guide's per-species section, with an executable example and the rule: passes below 8
      species, rows from 8;
    - refresh the guide's "a brain is required" and "swapping costs a compile" lines;
    - closure.
- **For step 7, part 2's design:** record teacher data with overflow off, or pull step 5's
  grid-ordered queries forward (the user's call).
- Step 5 is deferred, not done.

## State of the repositories

- **null-graph** (`master`, not pushed): Claude's work is all committed.
  - **Uncommitted, not Claude's:** camera input work (`src/extras/cameras/OrbitalEventProxy.ts`,
    `docs/api/cameras.md`, `test/cameras-input.test.mjs`, a `test:cameras` script in
    `package.json`). Leave it to its owner, and validate commits alone in a worktree.
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
