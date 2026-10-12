# Now — 2026-10-12

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.** Step 6 (`MlpBrain`) is complete and committed (6.0 `e345921`,
  6.1 `dbaf884`, 6.2 `b3424b1`, 6.3 `d8ad5b2`). Core limit negotiation: `7f1acb0`. Per-species
  design: `ed195b3`.
- **Step 7, part 1: per-species brains**
  ([`../agents-species-brains-design.md`](../agents-species-brains-design.md)):
  - **7.1 is done but not committed:** `MlpBrain.bySpecies`, one row per species in one weight
    table, chosen by species name; per-row `setWeights`; U21.
  - **Next: 7.2:** rows on the GPU (A21, both hosts) plus the lockstep measurement.
  - Then the composite `SpeciesBrain` (7.3–7.4) and the guide (7.5).
- **Docs corrected:** same-device determinism holds only while no grid cell
  overflows (measured: 19,990 of 64,000 words differed between identical runs at budget 64, none
  without overflow). "Overflow is rare" is false for flocks. The MLP design gained its integration
  facts, touchpoints and verbatim errors.
- **For step 7, part 2's design:** record teacher data with overflow off, or pull step 5's
  grid-ordered queries forward. That is the user's decision when part 2 is designed.
- Step 5 is deferred, not done.

## State of the repositories

- **README refreshed:** feature/subpath table and API backlinks; internal phase
  labels and unsupported absolute claims removed. Documentation-only; development position unchanged.

- **null-graph** (`master`, not pushed): documentation committed separately; 7.1 code and tests
  remain uncommitted.
  Green:
  - library build;
  - headless agents 24 + 54, core, agent-kit 4/4;
  - the agents GPU suite at 120/120 on Node/Dawn and Chrome.
- **Bundle:** `agents.js` 17,827 B min+gzip (C2 ceiling 21,495 B, 3,668 B left); an app with
  `MlpBrain` 14,314 B; a rule-only app 12,375 B, no MLP code.
- **Test-engine site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`): its
  `node_modules/null-graph/dist` holds the limits build (`7f1acb0`), not 7.1. Restart its dev server
  with `npm run dev -- --force` after any refresh ([`gotchas.md`](./gotchas.md)).

## Open questions for the user

- None blocking 7.2. [`plan.md`](./plan.md) §5, item 3 (the brain-swap stall) is deferred to step 9.
