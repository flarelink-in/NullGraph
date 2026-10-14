# Now — 2026-10-14

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.** Steps 6, 7 and 8 are built and committed (step 8's closure:
  `bc083b0`). Step 5's scaling study is still deferred.
- **Step 9, ONNX Runtime brains** ([`../agents-ort-design.md`](../agents-ort-design.md), phases
  9.0–9.6; decided with the user, [`plan.md`](./plan.md) §5, items 9 and 10):
  - **Committed:** 9.0 `a57cd76` (spikes, design); 9.1 `e2ac9a4` (`OrtRuntime`, `OrtModel`, the
    anchor); 9.2 `a4f4035` (weights in a second file, read both ways; buffers rounded up to 16 B;
    capture off for two-file models).
  - **9.3 is built, not committed:** `OrtBrain` (`src/extras/inference/OrtBrain.ts`) and async
    brains in `AgentSystem` (the run schedule, lockstep with a run, `destroy` after runs in
    flight, `obs` rounded up to 16 B).
    - U13 (agents suite) and U33 (inference suite); A5, A30, A31 in the `inference` GPU suite on
      three host/GPU pairs: prey on `OrtBrain` within 1.1e-5 of `MlpBrain` on the same step,
      capture on and off; two runs of 120 steps byte-identical; each attach seeded from `act`.
    - Agents entry about 19,932 B (+609 B); the adapter with `OrtBrain` 2,994 B of its 3,000 B.
    - Review fix before the commit: a brain re-attached while its run is in flight keeps that run
      (U13's sixth test).
- **Next action:** phase 9.4 (A6: the interactive
  schedule on the GPU, 300 frames, latency recorded on three hosts).

## State of the repositories

- **null-graph** (`master`, not pushed). Nothing uncommitted after 9.3's commit.
- **`package-lock.json` is ignored by git here.**
- **`venv/`** (ignored): PyTorch, onnx, onnxruntime, scikit-learn; `pip` there needs
  `--cert venv/pip-ca-bundle.pem` ([`gotchas.md`](./gotchas.md)).
- **The site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`, uncommitted): Herd &
  Wolves with a live Rules / Network choice. Its `null-graph/dist` is from 2026-10-13. Its worker
  makes the engine first, which ORT cannot use (step 10's design decides).
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`).
  Remove its `node_modules` junction first with `cmd /c rmdir`, then `git worktree remove`.
- **Green on 2026-10-14, with 9.3:** headless agents (reference + 79), inference (21), core,
  agent-kit (4); GPU `inference` 12/12 and the agents GPU suite 137/137 on Node/NVIDIA, Node/AMD and
  Chrome/AMD. Ten mutations caught headless, two on the GPU.
- **Bundle:** agents about 19,932 B (C2 ceiling 21,495 B); the inference entry: reader 2,846 B (line
  2,900 B), adapter with `OrtBrain` 2,994 B (line 3,000 B).

## Open questions for the user

- None.
