# Now — 2026-10-14

**Rewrite this file at the end of every session; keep it under one screen.** History goes to the
log, not here.

## Position

- **Cycle C, milestone C2.**
  - Step 6 (`MlpBrain`) is complete.
  - **Step 7 is built and committed:** part 1 (per-species brains) through 7.5 (`de14ff4`); part 2
    (record → train → export, [`../agents-cloning-design.md`](../agents-cloning-design.md)) in
    7.6 `c2608d7`, 7.7 `0d3fb9f`, 7.8 `8afc6fa`, 7.9 `cd76a25`, 7.10 `834b360`.
  - Step 5's deterministic neighbour lookups are complete (5.5–5.8). The scaling study is still
    deferred.
- **Step 8, reading ONNX, is built:** every phase, 8.0 to 8.3
  ([`../agents-onnx-reader-design.md`](../agents-onnx-reader-design.md), its closure at the end).
  - **Committed:** the design `a8778ce`, 8.0 `7829f4f`, 8.1 `8aa243a`, 8.2 `874488d`, 8.3 (the
    commit after `874488d`).
  - **What exists:** `mlpFromOnnx(bytes)` in a new entry, `null-graph/inference`
    (`src/extras/inference/mlpOnnx.ts`), which returns an `MlpNetwork` for `MlpBrain`.
    - 13 files from PyTorch's two exporters, scikit-learn and `onnx.helper` read with their
      makers' weights bit for bit; 3 are refused by name (`test/onnx-fixtures/`).
    - ONNX Runtime agrees with the reader's `MlpBrain` within 4.5e-6 on 17 files, on three
      host/GPU pairs (A28).
    - The reader is 2,525 B min+gzip (line 2,600 B); the agents entry is unchanged at 19,323 B.
  - **8.3:** the guide `docs/api/inference.md` with a checked example
    (`test/inference-api-example.test.mjs`), the README row, the agents guide, the tools'
    README, the parent's wording.
  - **Decided by the user, to remember:** weights stored outside the `.onnx` file, which is
    what PyTorch's default call writes, are refused with the fix named; **step 9 looks again.**
- **Next action: step 9** (`OrtBrain` and the ONNX Runtime adapter, in the entry step 8
  created): spikes S1, S2, S4 and S5 come first ([`plan.md`](./plan.md) §3).

## State of the repositories

- **null-graph** (`master`, not pushed). Nothing uncommitted after 8.3's commit.
- **`venv/`** (ignored) gained PyTorch 2.14.1, onnx, onnxruntime, onnxscript, scikit-learn and
  skl2onnx on 2026-10-14 (973 MB in all). `pip` there needs `--cert venv/pip-ca-bundle.pem`
  ([`gotchas.md`](./gotchas.md)).
- **The site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`, uncommitted): the Herd &
  Wolves demo has a live Rules / Network choice per species, from copies of the two committed
  `.mlp.json` files. Its `node_modules/null-graph/dist` is this repository's dev `dist/` of
  2026-10-13, so it has no `inference` entry yet. Not checked by eye: the inspector's text for a
  network-driven agent.
- **The students and the recording** are in `tools/agents/out/` (ignored): the recording
  (`herd-wolves/`, made at `cf3f972`), the round-0 students (`e4-64x64x64-200-r01/`), their check
  retrain (`r0-check/`), and the exported `onnx/`.
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`),
  left from 5.6's validation. It holds a `node_modules` junction. Remove that first with
  `cmd /c rmdir`, which drops the link only, before `git worktree remove`, which may follow it.
- **Green on 2026-10-14, with 8.3:** headless agents (reference + 73), inference (8), agent-kit
  (4), core; `onnx` 23/23 on Node/NVIDIA.
  - With 8.2, the same day: `onnx` 23/23 on Node/AMD and Chrome/AMD too.
  - From 2026-10-13, not re-run: `agent-tools` 15/15 on three host/GPU pairs.
  - Not re-run, and untouched: the agents GPU suite (137) and the other GPU suites.
- **Bundle:** agents 19,323 B (C2 ceiling 21,495 B), unchanged by step 8. The reader is 2,525 B
  in its own entry.

## Open questions for the user

- For step 9, both already recorded in [`plan.md`](./plan.md) §5: the brain-swap stall (item 3),
  and weights stored outside the `.onnx` file (item 8).
