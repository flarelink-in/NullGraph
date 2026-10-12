# Now — 2026-10-12

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2, step 6 (`MlpBrain`).** Phases 6.0 (`e345921`) and 6.1 (`dbaf884`) are
  committed. **Next: phase 6.2:** run a real network on both GPU hosts and check it against the CPU
  forward pass (A4: numbers, cadence gating, weight updates, replacement, lifetime). Then 6.3:
  executable guide example, size/timing report, closure. Spec:
  [`../agents-mlp-design.md`](../agents-mlp-design.md) §6 and §8.
- Step 5 is deferred, not done. Do not start it unless the user asks.
- After step 6: step 7, the first trained brain. Settle the species-input decision first
  ([`plan.md`](./plan.md) §5).

## State of the repositories

- **null-graph** (`master`, not pushed): clean after `dbaf884`. Builds; 24 reference + 47
  node:test agent checks and 17 core checks pass; agents GPU suite 106/106 on Node/Dawn (Chrome
  last run on phase 4.5's commit). The GPU suite does not exercise `MlpBrain` yet: that is 6.2.
- **Bundle:** `agents.js` 17,283 B min+gzip; C2 allowance left 2,212 B.
- **Test-engine site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`): GPU Herd &
  Wolves route with darts, picking, inspector, stress test. Uncommitted by design. Its dev server
  needs `npm run dev -- --force` after a `dist/` refresh ([`gotchas.md`](./gotchas.md)).

## Open questions for the user

See [`plan.md`](./plan.md) §5.
