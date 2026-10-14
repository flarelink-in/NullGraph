# Now — 2026-10-14

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.** Steps 6, 7 and 8 are built and committed (step 8's closure:
  `bc083b0`). Step 5's scaling study is still deferred.
- **Step 9, ONNX Runtime brains** ([`../agents-ort-design.md`](../agents-ort-design.md), phases
  9.0–9.6; decided with the user, [`plan.md`](./plan.md) §5, item 9):
  - **9.0 is committed** (`a57cd76`): the spikes S1, S2, S4, S5 and the design.
  - **9.1 is committed** (the commit after `a57cd76`): `OrtRuntime`, `OrtModel` and the anchor session in
    `src/extras/inference/OrtRuntime.ts`; ORT an optional peer dependency.
    - U32, U14, size, types: 8 of 8 (`npm run test:inference`: 16 of 16).
    - A29 (`npm run test:gpu -- --only inference`): 6 of 6 on Node/NVIDIA, Node/AMD and
      Chrome/AMD. The device outlives every model; a captured model loaded after a release runs
      100 times on AMD, where it failed without an anchor.
    - Ten mutations caught by U32/U14; the anchor removed fails 6 of A29's checks.
    - The adapter is 1,925 B min+gzip (line 3,000 B).
- **Next action:** phase 9.2, under way: weights outside the file in `mlpFromOnnx`, the fixtures'
  data files, and both paths on the GPU (A32).

## State of the repositories

- **null-graph** (`master`, not pushed). Nothing uncommitted after 9.1's commit.
- **`package-lock.json` is ignored by git here.** `npm install --package-lock-only --offline`
  brought the local one up to date (it also still said version 1.1.0).
- **`venv/`** (ignored): PyTorch, onnx, onnxruntime, scikit-learn; `pip` there needs
  `--cert venv/pip-ca-bundle.pem` ([`gotchas.md`](./gotchas.md)). Phase 9.2 uses it to keep
  PyTorch's data files in the fixtures.
- **The site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`, uncommitted): Herd &
  Wolves with a live Rules / Network choice. Its `null-graph/dist` is from 2026-10-13. Its worker
  makes the engine first, which ORT cannot use (step 10's design decides).
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`).
  Remove its `node_modules` junction first with `cmd /c rmdir`, then `git worktree remove`.
- **Green on 2026-10-14, with 9.1:** headless inference (16), agents (reference + 73), core,
  agent-kit (4); GPU `inference` 6/6 on three host/GPU pairs, `onnx` 23/23 on Node/NVIDIA. The
  agents GPU suite was not re-run (nothing it runs changed).
- **Bundle:** agents 19,323 B (C2 ceiling 21,495 B); the inference entry 4,268 B (reader 2,527 B,
  line 2,600 B; adapter 1,925 B, line 3,000 B).

## Open questions for the user

- None.
