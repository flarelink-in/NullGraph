# Now — 2026-10-12

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.** Step 6 (`MlpBrain`) is complete and committed (6.0 `e345921`,
  6.1 `dbaf884`, 6.2 `b3424b1`, 6.3 `d8ad5b2`).
- **Step 7, part 1: per-species brains, designed and approved**
  ([`../agents-species-brains-design.md`](../agents-species-brains-design.md)):
  - same-shape networks as **weight-table rows** of one `MlpBrain` (`MlpBrain.bySpecies`, phases
    7.1–7.2);
  - then a composite `SpeciesBrain` to mix kinds (7.3–7.4);
  - then the guide (7.5).

  No fallback brain. **Next: phase 7.1.**
- **Core limit negotiation landed (uncommitted):** `PROGRESSION_LIMITS` raises three storage limits
  to the adapter's values on devices NullGraph creates. On both measured GPUs that is 16 buffers
  and ~2 GB bindings. `AgentSystem`'s cap of 3 brain buffers is gone.
- **Step 7, part 2** (record → train → export per species, the rest timer as an observation) gets
  its own design after part 1, from phase 7.6.
- Step 5 is deferred, not done.

## State of the repositories

- **null-graph** (`master`, not pushed): uncommitted. Changed: core (`WebGPUContext`, types), the
  `AgentSystem` cap, and tests (H8, C7, a U17 case); the new design; doc updates (core API guide,
  facts 16 and §12.3, the GPU-loop budget, pointers, `plan.md`). Green:
  - every headless suite;
  - every GPU suite on Node/Dawn (12) and Chrome (9), zero WebGPU errors.

  Core +89 B min+gzip.
- **Bundle:** `agents.js` 17,279 B min+gzip; C2 may reach 21,495 B.
- **Test-engine site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`): its
  `node_modules/null-graph/dist` was refreshed with this build and driven at 4K and 64K agents
  with no errors on a temporary Vite (:8083, stopped). Restart its dev server with
  `npm run dev -- --force` to pick the build up ([`gotchas.md`](./gotchas.md)).

## Open questions for the user

- None blocking. [`plan.md`](./plan.md) §5, item 3 (the brain-swap stall) is deferred to step 9.
