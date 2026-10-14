# Now — 2026-10-14

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.** Steps 6, 7 and 8 are built and committed (step 8's closure:
  `bc083b0`). Step 5's scaling study is still deferred.
- **Step 9, ONNX Runtime brains** ([`../agents-ort-design.md`](../agents-ort-design.md), phases
  9.0–9.6; decided with the user, [`plan.md`](./plan.md) §5, item 9):
  - **Committed:** 9.0 `a57cd76` (spikes, design); 9.1 `e2ac9a4` (`OrtRuntime`, `OrtModel`, the
    anchor; A29); 9.2, the commit after `e2ac9a4`: `mlpFromOnnx(bytes, { externalData })` reads
    weights stored in a second file; `make.py` keeps PyTorch's data files, with one new fixture
    (`torch-outside-batch`); U36; A28 and A32 read both two-file exports both ways on three
    host/GPU pairs. Reader 2,848 B (line 2,900 B).
  - **It found two faults of ONNX Runtime 1.29** (`npm run spike:ort-issues`):
    - its kernels write 16 B at a time, and on NVIDIA a write past a buffer's end is lost
      silently. Guarded: `OrtModel.bind` asks for buffers rounded up to 16 B;
    - under graph capture, a model whose weights are in a second file can fail from its second
      run, depending on what ran before. Avoided: `load` leaves capture off for such models unless
      asked (decided by the user, `plan.md` §5, item 10).
- **Next action:** phase 9.3, under way: async brains in `AgentSystem` (the run schedule, lockstep
  with a run) and `OrtBrain` with its copy pass (design §7, §8).

## State of the repositories

- **null-graph** (`master`, not pushed). Nothing uncommitted after 9.2's commit.
- **`package-lock.json` is ignored by git here.**
- **`venv/`** (ignored): PyTorch, onnx, onnxruntime, scikit-learn; `pip` there needs
  `--cert venv/pip-ca-bundle.pem` ([`gotchas.md`](./gotchas.md)).
- **The site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`, uncommitted): Herd &
  Wolves with a live Rules / Network choice. Its `null-graph/dist` is from 2026-10-13. Its worker
  makes the engine first, which ORT cannot use (step 10's design decides).
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`).
  Remove its `node_modules` junction first with `cmd /c rmdir`, then `git worktree remove`.
- **Green on 2026-10-14, with 9.2:** headless inference (17), agents (reference + 73), core,
  agent-kit (4); GPU `inference` 8/8 and `onnx` 25/25 on Node/NVIDIA, Node/AMD and Chrome/AMD.
  The agents GPU suite was not re-run (nothing it runs changed).
- **Bundle:** agents 19,323 B (C2 ceiling 21,495 B); the inference entry 4,635 B (reader 2,848 B,
  line 2,900 B; adapter 1,984 B, line 3,000 B).

## Open questions for the user

- None.
