# Now — 2026-10-14

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.** Steps 6, 7 and 8 are built and committed (step 8's closure:
  `bc083b0`). Step 5's scaling study is still deferred.
- **Step 9, ONNX Runtime brains** ([`../agents-ort-design.md`](../agents-ort-design.md), phases
  9.0–9.6; decided with the user, [`plan.md`](./plan.md) §5, items 9–11):
  - **Committed:** 9.0 `a57cd76`; 9.1 `e2ac9a4`; 9.2 `a4f4035`; 9.3 `e1344ca` (`OrtBrain`, async
    brains in `AgentSystem`); 9.4 `f11ba4f` (A6; graph capture off by default, item 11); 9.5, the
    commit after `f11ba4f`: core's `engine.createComputeStageAsync` and
    `agents.prepareBrain` (a brain's think pass compiled ahead, kept while the system lives).
    - U34 (core suite), U35 (agents suite), A33 (`inference` GPU suite) on three host/GPU pairs:
      `setBrain` of a prepared brain 0.11–0.65 ms; compiled inside `setBrain`, an MlpBrain froze a
      frame for 240 ms (Node/NVIDIA) and 1,759 ms (Node/AMD); prepared, frames went on. Chrome's
      main thread never stalled either way (it compiles in its GPU process).
    - Core's entry +126 B (estimate 60–100 B; 31 B are an error scope that puts the WGSL error in
      the message); agents +280 B (estimate 150–250 B), 20,209 B of C2's 21,495 B.
- **Next action:** phase 9.6, under way (the guides: ORT and external data in
  `docs/api/inference.md`, `prepareBrain` in the agents guide, the parent's facts 15–16, §9.4,
  §11, §12.3).

## State of the repositories

- **null-graph** (`master`, not pushed). Nothing uncommitted after 9.5's commit.
- **ONNX Runtime 1.30.0 is out,** and the peer range `^1.29.0` admits it. Only the `inference`
  suite of 9.4 has run on it (Node/NVIDIA, 15/15, `NG_ORT_DIST`). The repository tests 1.29.0.
- **`package-lock.json` is ignored by git here.**
- **`venv/`** (ignored): PyTorch, onnx, onnxruntime, scikit-learn; `pip` there needs
  `--cert venv/pip-ca-bundle.pem` ([`gotchas.md`](./gotchas.md)).
- **The site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`, uncommitted): Herd &
  Wolves with a live Rules / Network choice. Its `null-graph/dist` is from 2026-10-13. Its worker
  makes the engine first, which ORT cannot use (step 10's design decides).
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`).
  Remove its `node_modules` junction first with `cmd /c rmdir`, then `git worktree remove`.
- **Green on 2026-10-14, with 9.5:** headless core, agents (reference + 83), inference (21),
  agent-kit (4); GPU `inference` 16/16 and agents 137/137 on Node/NVIDIA, Node/AMD and Chrome/AMD;
  every Node GPU suite (14/14). Nine mutations caught (two U34, six U35, one A33).
- **Bundle:** core 6,191 B; agents 20,209 B (C2 ceiling 21,495 B); the inference entry: reader
  2,846 B (line 2,900 B), adapter with `OrtBrain` 2,988 B (line 3,000 B).

## Open questions for the user

- None.
