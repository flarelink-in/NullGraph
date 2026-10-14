# Now — 2026-10-14

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.** Steps 6, 7 and 8 are built and committed (step 8's closure:
  `bc083b0`). Step 5's scaling study is still deferred.
- **Step 9, ONNX Runtime brains** ([`../agents-ort-design.md`](../agents-ort-design.md), phases
  9.0–9.6; decided with the user, [`plan.md`](./plan.md) §5, items 9–11):
  - **Committed:** 9.0 `a57cd76` (spikes, design); 9.1 `e2ac9a4` (`OrtRuntime`, `OrtModel`, the
    anchor); 9.2 `a4f4035` (weights in a second file); 9.3 `e1344ca` (`OrtBrain`, async brains in
    `AgentSystem`); 9.4, the commit after `e1344ca`: A6 in the `inference` GPU suite, on three
    host/GPU pairs. 600 interactive frames, no `mapAsync` outside the inspector, no GPU error,
    actions one frame old (all frames but 1 of 300 in Chrome at the display's pace; S2's stall in
    Chrome when frames come as fast as they can).
    - A6 found two faults of ONNX Runtime under graph capture: a captured model can fail from its
      second run, depending on what ran between its first two (fault 3), and memory grows with
      every run (fault 4). **The user chose: capture off by default for every model** (`plan.md`
      §5, item 11). `npm run spike:ort-issues` shows both.
- **Next action:** phase 9.5, under way (prepared brains: core's `createComputeStageAsync`, then
  `agents.prepareBrain`; U34, U35, A33).

## State of the repositories

- **null-graph** (`master`, not pushed). Nothing uncommitted after 9.4's commit.
- **ONNX Runtime 1.30.0 is out,** and the peer range `^1.29.0` admits it. Only the `inference`
  suite has run on it, on Node/NVIDIA (15/15, `NG_ORT_DIST`). The repository still tests 1.29.0.
- **`package-lock.json` is ignored by git here.**
- **`venv/`** (ignored): PyTorch, onnx, onnxruntime, scikit-learn; `pip` there needs
  `--cert venv/pip-ca-bundle.pem` ([`gotchas.md`](./gotchas.md)).
- **The site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`, uncommitted): Herd &
  Wolves with a live Rules / Network choice. Its `null-graph/dist` is from 2026-10-13. Its worker
  makes the engine first, which ORT cannot use (step 10's design decides).
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`).
  Remove its `node_modules` junction first with `cmd /c rmdir`, then `git worktree remove`.
- **Green on 2026-10-14, with 9.4:** headless agents (reference + 79), inference (21), core,
  agent-kit (4); GPU `inference` 15/15 on Node/NVIDIA, Node/AMD and Chrome/AMD. Five mutations
  caught (four for A6, one for U32). The agents GPU suite was not re-run (nothing it runs changed).
- **Bundle:** agents 19,929 B (C2 ceiling 21,495 B); the inference entry: reader 2,846 B (line
  2,900 B), adapter with `OrtBrain` 2,988 B (line 3,000 B).

## Open questions for the user

- None.
