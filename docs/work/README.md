# Working on NullGraph — start here

For maintainers and coding agents (Claude, Cline, others) picking up work between sessions.
Library users want [`../agent/guide.md`](../agent/guide.md) and [`../api/`](../api/) instead.

## Read in this order

1. [`now.md`](./now.md): where things stand today and the next action. One screen.
2. [`plan.md`](./plan.md): the goal, the naming (cycle → milestone → step → phase), every step's
   purpose and status, the horizon goals and the decisions waiting on the user.
3. The design doc of the step you are working on (linked from `plan.md`).
4. [`gotchas.md`](./gotchas.md) before touching tests, shaders or the site.

Only if you need history: [`log/`](./log/) (session log, one file per month) and
[`../handoff.md`](../handoff.md), the detailed record up to 2026-10-12 (results, evidence,
decisions per cycle). The handoff is frozen: do not append to it.

## At the end of every session

| File | What to do |
|---|---|
| `now.md` | **Rewrite** it: position, repo state, next action. Stay under one screen. |
| `log/<yyyy-mm>.md` | **Append** one entry: who, what was asked, what was done, commits, what is next. 5–10 lines. |
| The step's design doc | Record phase results (numbers, tests, evidence) in its results section. |
| `plan.md` | Change **only** a status cell, or a scope change the user decided (dated, under "Scope changes"). |
| `gotchas.md` | Add a trap that cost time and will bite again. |

Detail belongs in the design doc, not in `now.md` or the log.

## Rules against drift

- Work only on the step `now.md` names, or what the user asks. If a request seems to change a
  step's goal, say so and ask; record the answer in `plan.md`.
- Never redefine, merge, skip or "complete" a step on your own. "Deferred" is not "done".
- Name things the same way everywhere: "step 6", "phase 6.2", "C2".
- Commit only when the user asks. One phase per commit; validate a commit alone (temporary worktree)
  when the tree holds other work. Never push or publish unasked.
- Report what was measured, how, and what was not run. Watch new tests fail before trusting them.

## Commands

| Command | What it does |
|---|---|
| `npm run build:lib` | Build `dist/`. **Every test imports `dist/`**: build after any `src/` change. `test:gpu` warns when `dist/` is older than `src/`. |
| `npm run test:agents` · `test:core` · `test:visibility` · … | Headless suites (mock devices, math, WGSL parsing). |
| `npm run test:inference` | Step 8: the ONNX reader of `null-graph/inference`, headless. Reads `test/onnx-fixtures/fixtures.json`; also holds the reader's size line. |
| `npm run test:gpu` | Every GPU suite in Node through Dawn, one process per suite, plus a render smoke test. `-- --only agents` to pick one; `-- --json-dir out` for JSON reports. |
| `npm run test:gpu:browser` | The same suites in headless Chrome. `-- --only agents`, `-- --chrome <path>`. |
| `NG_GPU_ARTIFACTS=<dir> npm run test:gpu` | Also save the frames harnesses render, as PNGs. |
| `npm run render -- <scene.mjs> --out shot.png` | Render a scene to a PNG headless; exit 1 on a WebGPU error or a blank frame. Contract in `test/gpu/render.mjs`. |
| `npm run drive -- <url> [--wait-for re] [--actions steps.json \| --interactive] [--out dir]` | Drive a page in headless Chrome: input, screenshots, eval; captures page and worker consoles and GPU errors. Exit 1 on any error. See `test/gpu/drive-page.mjs`, `../agent/validation.md`. |
| `npm run spike:s3` | Spike S3: think-kernel cost on every adapter Dawn offers. A measurement, not a suite. |
| `npm run agents:record` | Step 7, part 2: record what Herd & Wolves' rule decides, headless, into `tools/agents/out/` (ignored). `-- --episodes 2 --steps 360` for a short one. |
| `npm run agents:train` | Step 7, part 2: train one network per species on a recording, CPU only; writes `<species>.mlp.json` and `<species>.report.json` beside it. `-- --hidden 16x16 --epochs 2` for a quick one. |
| `npm run agents:evaluate` | Step 7, part 2: run the teacher, the rule without wander, and the students in the simulation, headless; prints catches and each driver's error against the label, and writes `evaluation.json` beside the students. `-- --students tools/agents/models/herd-wolves` judges the committed networks; `-- --episodes 1 --steps 300` for a quick one. |
| `npm run agents:export` | Step 7, part 2: write the trained networks as `.onnx` files into `tools/agents/out/onnx` (ignored). CPU only, under a second. `-- --students <dir>` picks other networks. |
| `venv/Scripts/python test/onnx-fixtures/make.py` | Step 8, phase 8.0: rewrite `test/onnx-fixtures/fixtures.json`, the small `.onnx` files that PyTorch, scikit-learn and `onnx.helper` write, with what each maker computed. `--list` prints what each file holds. Needs the packages of `requirements.txt` in `venv/` (see `gotchas.md` for `pip` there). The tests need no Python. A new case goes at the end of `torch_cases()`, or the earlier files change with the random stream. |

To refresh the test-engine site after library changes: `npm run yalc:push` here (it leaves a
**production** `dist/`; run `npm run build:lib` before committing), then restart the site with
`npm run dev -- --force`.
