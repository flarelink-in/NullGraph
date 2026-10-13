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
- **Now: step 8, reading ONNX** ([`../agents-onnx-reader-design.md`](../agents-onnx-reader-design.md),
  phases 8.0–8.3).
  - **Every item is decided** (the user, 2026-10-14; the design's §14):
    - the reader is a function, `mlpFromOnnx(bytes)`, in a new entry `null-graph/inference`;
    - fixtures from other exporters, made in `venv/`; no TensorFlow fixtures now;
    - a `Cast` or `Flatten` before the first layer and a `Reshape` after the last are passed
      over, and an `Identity` of an initializer is followed;
    - weights outside the file are refused with the fix named; **look again in step 9;**
    - no "Use OrtBrain" in refusals until step 9; the reader's size line is 2,600 B.
  - **Committed 2026-10-14:** the design `a8778ce`, 8.0 `7829f4f`, 8.1 (the commit after `7829f4f`).
  - **8.0:** `test/onnx-fixtures/` (`make.py`, `fixtures.json`, `requirements.txt`): 16 files from
    PyTorch's two exporters, scikit-learn and `onnx.helper`.
    - **PyTorch's default call stores every weight tensor over 256 B outside the `.onnx` file.**
    - Exporters store equal tensors once; PyTorch's older one then writes `Identity` nodes.
  - **8.1:** `src/extras/inference/` (`mlpOnnx.ts`), the entry in `build.js` and `package.json`,
    `npm run test:inference` (7 of 7: U12's reader half, U29, U30, U31, U14, the size line).
    - 13 foreign files read with their makers' weights bit for bit; 3 are refused as listed.
    - 34 of 34 mutations were caught. The reader is 2,525 B min+gzip; the agents entry is
      unchanged at 19,323 B.
  - **Next action: 8.2,** A28 in the `onnx` GPU suite: the reader's `MlpBrain` against ONNX
    Runtime on the same bytes, on Node/NVIDIA, Node/AMD and Chrome. Then 8.3: the guide
    (`docs/api/inference.md`), the README row, the parent's wording.

## State of the repositories

- **null-graph** (`master`, not pushed). Nothing uncommitted after 8.1's commit.
  - **Stale until 8.3:** `docs/api/agents.md` and `tools/agents/README.md` still say that ONNX is
    not read; the README's table has no row for the new entry.
- **`venv/`** (ignored) gained PyTorch 2.14.1, onnx, onnxruntime, onnxscript, scikit-learn and
  skl2onnx on 2026-10-14 (973 MB in all). `pip` there needs `--cert venv/pip-ca-bundle.pem`
  ([`gotchas.md`](./gotchas.md)).
- **The site** (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`, uncommitted): the Herd &
  Wolves demo has a live Rules / Network choice per species, from copies of the two committed
  `.mlp.json` files. Its `node_modules/null-graph/dist` is this repository's dev `dist/`. Not
  checked by eye: the inspector's text for a network-driven agent.
- **The students and the recording** are in `tools/agents/out/` (ignored): the recording
  (`herd-wolves/`, made at `cf3f972`), the round-0 students (`e4-64x64x64-200-r01/`), their check
  retrain (`r0-check/`), and the exported `onnx/`.
- **Another worktree exists:** `D:/orion-realms/null-graph-56-validation` (detached at `9ea8a33`),
  left from 5.6's validation. It holds a `node_modules` junction. Remove that first with
  `cmd /c rmdir`, which drops the link only, before `git worktree remove`, which may follow it.
- **Green on 2026-10-14, on the state 8.1's commit holds:** headless agents (reference + 73),
  inference (7), agent-kit (4), core; `onnx` 5/5 on Node/NVIDIA.
  - From 2026-10-13, not re-run: `onnx` on Node/AMD and Chrome; `agent-tools` 15/15 on three
    host/GPU pairs.
  - Not re-run, and untouched: the agents GPU suite (137) and the other GPU suites.
- **Bundle:** agents 19,323 B (C2 ceiling 21,495 B), unchanged by step 8. The reader is 2,525 B
  in its own entry (line 2,600 B).

## Open questions for the user

- [`plan.md`](./plan.md) §5, item 3 (the brain-swap stall) is deferred to step 9.
- Step 9 also looks again at weights stored outside the `.onnx` file ([`plan.md`](./plan.md) §5,
  item 8).
