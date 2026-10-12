# Now — 2026-10-12

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2, step 6 (`MlpBrain`).** Phases 6.0 (`e345921`) and 6.1 (`dbaf884`) are
  committed, and so is **phase 6.2**: A4 passes on both GPU hosts. Its only change is 14 new
  checks in `test/agents-gpu/run.js`; no library source changed. Results:
  [`../agents-mlp-design.md`](../agents-mlp-design.md), "Phase 6.2 implementation".
- **Next: phase 6.3:** a small known-weight example in `docs/api/agents.md` (checked by
  `test/agents-api-example.test.mjs`), an inference-cost and bundle report, then step 6's closure.
- Step 5 is deferred, not done. Do not start it unless the user asks.
- After step 6: step 7, the first trained brain. Settle the species-input decision first
  ([`plan.md`](./plan.md) §5).

## State of the repositories

- **null-graph** (`master`, not pushed): clean after the phase 6.2 commit. Builds;
  agents GPU suite **120/120 on Node/Dawn and Chrome**, zero WebGPU errors (both hosts on the AMD
  GCN-5 iGPU; NVIDIA not exercised); headless agents 24 + 47, core 17, agent-kit 4/4.
- **Bundle:** `agents.js` 17,283 B min+gzip; C2 allowance left 2,212 B.
- **Test-engine site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`): GPU Herd &
  Wolves route with darts, picking, inspector, stress test. Uncommitted by design. Its dev server
  needs `npm run dev -- --force` after a `dist/` refresh ([`gotchas.md`](./gotchas.md)).

## Open questions for the user

See [`plan.md`](./plan.md) §5.
