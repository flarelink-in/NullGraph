# Now — 2026-10-14

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.** Steps 6, 7 and 8 are built and committed (step 8's closure:
  `bc083b0`). Step 5's scaling study is still deferred.
- **Step 9, ONNX Runtime brains: design decided, phase 9.0 committed** (the commit after
  `bc083b0`): [`../agents-ort-design.md`](../agents-ort-design.md), phases 9.0–9.6.
  - **9.0, the spikes** (`test/spikes/ort/`, `npm run spike:s1` … `spike:s5`): ORT 1.29 runs only
    on a device it made itself; ORT costs about 3× (NVIDIA) to 7–10× (AMD iGPU) the built-in
    kernel and 0.8–1.6 s to start; its actions arrive one frame late, never torn; lockstep is
    exact. Numbers at the end of the design.
  - **The user's decisions** (design §14; [`plan.md`](./plan.md) §5, item 9): ORT creates the
    device and the engine adopts it; graph capture on; brains prepared at load and compiled in the
    background (`prepareBrain`, over a new core method, `createComputeStageAsync`); weights stored
    outside the `.onnx` file read by both ORT and `mlpFromOnnx`.
- **Next action: phase 9.1**: `OrtRuntime`, `OrtModel` and the anchor session in
  `null-graph/inference`, ORT as an optional peer dependency, the `OrtLike` type, the adapter's
  size line; tests U32 and A29 (design §13).

## State of the repositories

- **null-graph** (`master`, not pushed). Nothing uncommitted after 9.0's commit.
- **The spikes' raw logs** stayed in the session's scratchpad; the design holds the numbers.
- **`venv/`** (ignored): PyTorch, onnx, onnxruntime, scikit-learn; `pip` there needs
  `--cert venv/pip-ca-bundle.pem` ([`gotchas.md`](./gotchas.md)). Phase 9.2 uses it to keep
  PyTorch's data files in the fixtures.
- **The site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`, uncommitted): the Herd &
  Wolves demo with a live Rules / Network choice per species. Its `null-graph/dist` is from
  2026-10-13, with no `inference` entry. Its worker makes the engine first, which ORT cannot use
  (step 10's design decides how it gets ORT's device).
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`).
  Remove its `node_modules` junction first with `cmd /c rmdir`, then `git worktree remove`.
- **Green on 2026-10-14, with 8.3:** headless agents (reference + 73), inference (8), agent-kit
  (4), core; `onnx` 23/23 on three host/GPU pairs. Phase 9.0 changed no library code.
- **Bundle:** agents 19,323 B (C2 ceiling 21,495 B); the reader 2,525 B in `inference.js` (its line
  rises to about 2,900 B in phase 9.2).

## Open questions for the user

- None.
