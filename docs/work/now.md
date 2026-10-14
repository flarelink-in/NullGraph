# Now — 2026-10-14

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.** Steps 6, 7, 8 and 9 are built and committed. Step 9's phases: 9.0
  `a57cd76`, 9.1 `e2ac9a4`, 9.2 `a4f4035`, 9.3 `e1344ca`, 9.4 `f11ba4f`, 9.5 `35e5be1`, 9.6
  `4e19a0f` (the guides). Step 5's scaling study is still deferred.
- **Step 10, the live switch** ([`../agents-live-switch-design.md`](../agents-live-switch-design.md)):
  **design decided** with the user (`plan.md` §5, item 12) and committed, in the site's Herd &
  Wolves demo (`HerdWolvesGPU`): Rules / Network / ONNX Runtime per species, prepared ahead, with
  live timings. The worker makes its engine again on ONNX Runtime's device at the first ORT pick
  (D1); Network keeps `.mlp.json`, and ORT's `.onnx` is checked against it (D2).
- **Next action:** phase 10.0, spikes: ONNX Runtime in the site's worker through Vite; the device
  switch over one `OffscreenCanvas`; ONNX Runtime's time and memory at 4K–1M agents; the run
  bracket's bias.

## State of the repositories

- **null-graph** (`master`, not pushed). Nothing uncommitted after step 10's design.
- **ONNX Runtime 1.30.0 is out,** and the peer range `^1.29.0` admits it. Only the `inference`
  suite of 9.4 has run on it (Node/NVIDIA, 15/15, `NG_ORT_DIST`). The repository tests 1.29.0.
- **`package-lock.json` is ignored by git here.** `dist/index.js` and its map are tracked although
  `dist` is ignored: commit them when core changes.
- **`venv/`** (ignored): PyTorch, onnx, onnxruntime, scikit-learn; `pip` there needs
  `--cert venv/pip-ca-bundle.pem` ([`gotchas.md`](./gotchas.md)).
- **The site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`, uncommitted): Herd &
  Wolves with a live Rules / Network choice. Its `null-graph/dist` is from 2026-10-13.
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`).
  Remove its `node_modules` junction first with `cmd /c rmdir`, then `git worktree remove`.
- **Green on 2026-10-14, with 9.6:** headless agents (reference + 83), inference (23), core,
  agent-kit (4). With 9.5, on the GPU: `inference` 16/16 and agents 137/137 on Node/NVIDIA,
  Node/AMD and Chrome/AMD; every Node GPU suite (14/14). 9.6 changed no source.
- **Bundle:** core 6,191 B; agents 20,209 B (C2 ceiling 21,495 B); the inference entry: reader
  2,846 B (line 2,900 B), adapter with `OrtBrain` 2,988 B (line 3,000 B).

## Open questions for the user

- None.
