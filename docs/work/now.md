# Now — 2026-10-14

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.** Steps 6, 7 and 8 are built and committed (step 8's closure:
  `bc083b0`). Step 5's scaling study is still deferred.
- **Step 9, ONNX Runtime brains** ([`../agents-ort-design.md`](../agents-ort-design.md), phases
  9.0–9.6; decided with the user, [`plan.md`](./plan.md) §5, items 9–11): **every phase is built
  and committed.**
  - 9.0 `a57cd76`; 9.1 `e2ac9a4`; 9.2 `a4f4035`; 9.3 `e1344ca` (`OrtBrain`, async brains); 9.4
    `f11ba4f` (A6; graph capture off by default, item 11); 9.5 `35e5be1`
    (`createComputeStageAsync`, `prepareBrain`; A33).
  - 9.6, the commit after `35e5be1`: the guides. `docs/api/inference.md` gains "ONNX Runtime" (an
    example type-checked against the real `onnxruntime-web` and run on stand-ins);
    `docs/api/agents.md` gains "Switching brains without a hitch" (`prepareBrain`, a checked
    example); `docs/api/core.md`, `README.md` and the parent design say what step 9 built.
- **Next action:** step 10's design: the live Rules / Network / ONNX Runtime switch, with timings.

## State of the repositories

- **null-graph** (`master`, not pushed). Nothing uncommitted after 9.6's commit.
- **ONNX Runtime 1.30.0 is out,** and the peer range `^1.29.0` admits it. Only the `inference`
  suite of 9.4 has run on it (Node/NVIDIA, 15/15, `NG_ORT_DIST`). The repository tests 1.29.0.
- **`package-lock.json` is ignored by git here.** `dist/index.js` and its map are tracked although
  `dist` is ignored: commit them when core changes.
- **`venv/`** (ignored): PyTorch, onnx, onnxruntime, scikit-learn; `pip` there needs
  `--cert venv/pip-ca-bundle.pem` ([`gotchas.md`](./gotchas.md)).
- **The site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`, uncommitted): Herd &
  Wolves with a live Rules / Network choice. Its `null-graph/dist` is from 2026-10-13. Its worker
  makes the engine first, which ORT cannot use (step 10's design decides).
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`).
  Remove its `node_modules` junction first with `cmd /c rmdir`, then `git worktree remove`.
- **Green on 2026-10-14, with 9.6:** headless agents (reference + 83), inference (23), core,
  agent-kit (4). With 9.5, on the GPU: `inference` 16/16 and agents 137/137 on Node/NVIDIA,
  Node/AMD and Chrome/AMD; every Node GPU suite (14/14). 9.6 changed no source.
- **Bundle:** core 6,191 B; agents 20,209 B (C2 ceiling 21,495 B); the inference entry: reader
  2,846 B (line 2,900 B), adapter with `OrtBrain` 2,988 B (line 3,000 B).

## Open questions for the user

- None.
