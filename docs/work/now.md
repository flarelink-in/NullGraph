# Now — 2026-10-12

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2: step 6 (`MlpBrain`) is complete and committed:** 6.0 `e345921`,
  6.1 `dbaf884`, 6.2 `b3424b1` and 6.3 (the commit after it): the guide's network example and its
  test, `npm run spike:mlp-think` (cost) and `test/spikes/mlp-think-cost/size.mjs` (bundle).
  Results: [`../agents-mlp-design.md`](../agents-mlp-design.md), "Phase 6.3 implementation".
- **Next: step 7**, `tools/agents`: record the rule brain → train a copy → export (U11, U12). It
  needs its own design doc first, and one answer from the user ([`plan.md`](./plan.md) §5, item 1):
  how one network tells species apart.
- Step 5 is deferred, not done. Do not start it unless the user asks.

## State of the repositories

- **null-graph** (`master`, not pushed): clean after the 6.3 commit. Builds;
  agents GPU suite **120/120 on Node/Dawn and Chrome**, zero WebGPU errors; headless agents
  24 + 47, core 17, agent-kit 4/4.
- **Bundle:** `agents.js` 17,283 B min+gzip; C2 allowance left 2,212 B (for step 8). An app that
  adds `MlpBrain` grows 1,424 B; rule-only apps carry none of it.
- **Test-engine site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`): GPU Herd &
  Wolves route with darts, picking, inspector, stress test. Uncommitted by design. Its dev server
  needs `npm run dev -- --force` after a `dist/` refresh ([`gotchas.md`](./gotchas.md)).

## Open questions for the user

See [`plan.md`](./plan.md) §5: the species encoding (blocks step 7) and the bundle allowance
(step 8). The brain-swap stall is deferred to step 9 by the user.
