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
| `npm run test:inference` | Steps 8 and 9: `null-graph/inference`, headless. The ONNX reader (reads `test/onnx-fixtures/fixtures.json`; its size line; the example of `docs/api/inference.md`) and, since phase 9.1, the ONNX Runtime adapter against a fake `ort` (its size line; a type check against the real `onnxruntime-web`). On the GPU: `npm run test:gpu -- --only inference`. |
| `npm run test:gpu` | Every GPU suite in Node through Dawn, one process per suite, plus a render smoke test. `-- --only agents` to pick one; `-- --json-dir out` for JSON reports. |
| `npm run test:gpu:browser` | The same suites in headless Chrome. `-- --only agents`, `-- --chrome <path>`. |
| `NG_GPU_ARTIFACTS=<dir> npm run test:gpu` | Also save the frames harnesses render, as PNGs. |
| `npm run render -- <scene.mjs> --out shot.png` | Render a scene to a PNG headless; exit 1 on a WebGPU error or a blank frame. Contract in `test/gpu/render.mjs`. |
| `npm run drive -- <url> [--wait-for re] [--actions steps.json \| --interactive] [--out dir]` | Drive a page in headless Chrome: input, screenshots, eval; captures page and worker consoles and GPU errors. Exit 1 on any error. See `test/gpu/drive-page.mjs`, `../agent/validation.md`. |
| `npm run spike:s3` | Spike S3: think-kernel cost on every adapter Dawn offers. A measurement, not a suite. |
| `npm run spike:ort-issues` | Step 9, phases 9.2 and 9.4: four faults of ONNX Runtime in their smallest form (writes of 16 B past a buffer's end lost on NVIDIA; under graph capture, a model failing from its second run, and memory growing with every run). Run it after upgrading ORT. `NG_SPIKE_CASE=1`…`4` runs one capture case alone in a fresh process; `NG_SPIKE_LEAK=1` measures the leak (about 9 minutes under Node). `?spike=issues` in Chrome. To try another ORT build before upgrading: `NG_ORT_DIST=<its dist folder> npm run test:gpu -- --only inference` (Node). |
| `npm run spike:ort-counts` | Step 10, phase 10.0: ONNX Runtime at the Herd & Wolves demo's counts (4K–1M agents) with the trained networks, covering GPU and main-thread time, its GPU buffers, and the overlay's two-mark timing against step 9's bracket. `NG_SPIKE_MAX`, `NG_SPIKE_LIMITMS`. In Chrome: `npm run drive -- --serve . "test/spikes/ort/index.html?spike=counts" --wait-for "counts done" --timeout 1200`. |
| `node test/spikes/live-switch/run.mjs --mode dev\|build --wasm prefix\|wasm\|default [--exclude-ort] [--extern-wasm] [--settle]` | Step 10, phase 10.0: ONNX Runtime in a module worker through the site's own Vite (dev server, or a build served by `vite preview`), and the device switch over one `OffscreenCanvas`. It drives the page in headless Chrome and writes screenshots and the log to `--out`. Nothing in the site changes. See its header and the design's "Phase 10.0 results". |
| `node test/spikes/live-switch/check-site.mjs [--site <dir>]` | Step 10, phases 10.1–10.2: the test-engine site's copies against this repository. The demo's rule against the scenario's texts, byte for byte, with its label's hash against the networks' `trainedOn.teacher`; its `.mlp.json` against the committed networks; its `.onnx` against those networks, bit for bit. CPU only, changes nothing. Run it after changing the scenario, training again, or editing the demo. |
| `npm run spike:s1` · `spike:s2` · `spike:s4` · `spike:s5` | Step 9, phase 9.0: ONNX Runtime's run cost, where its runs land, its start-up, and whose device it runs on (`test/spikes/ort/`; [design](../agents-ort-design.md)). Under Node, `NG_DAWN_FLAGS=adapter=NVIDIA` or `adapter=AMD` picks the GPU. S5's device: `NG_SPIKE_DEVICE=engine\|engine-defaults\|like-ort\|ort`. In Chrome: `npm run drive -- --serve . "/test/spikes/ort/index.html?spike=s1" --wait-for "S1 done" --timeout 300`. Measurements, not suites. |
| `npm run spike:animated-models` | Cycle D, step 1, phase 0: animated low-poly models (the test site's fox, deer, cow, baked by `test/spikes/animated-models/bake.mjs`) drawn as Herd & Wolves' agents at 4K–64K, against darts; rest pose, posed by a prototype vertex hook, and welded. `NG_SPIKE_MAX`; `NG_GPU_ARTIFACTS=<dir>` saves close-ups. Copy the three GLBs into `test/spikes/animated-models/models/` first (ignored by git). In Chrome: `npm run drive -- --serve . "/test/spikes/animated-models/index.html" --wait-for "animated-models done" --timeout 900`. |
| `node test/spikes/anime-look/run.mjs [--chains a,b] [--zooms close,far] [--out dir]` | Cycle D, step 1, phase 4 review: the Herd & Wolves Models scene rendered headless under each post chain (`faithful`, `anime`, candidates in `scene.ts`) at four zooms, tiled into `sheet.png`. Reads the models and palette from the site's `public/` (`NG_SITE_PUBLIC`); `NG_LOOK_FENCE=1 NG_LOOK_ZOOM=edge` with `npm run render -- test/spikes/anime-look/scene.mjs` shows the site's fence close up. A look check, not a suite. |
| `npm run spike:skinned-models` | Cycle D, step 1, phase 5.0: skinning's cost as Herd & Wolves' agents at 4K–64K, against the demo's rigid hook: four influences sampled per vertex, zero weights skipped, 8-bit weights, and a per-agent palette (one or four matrix reads). Uses the phase 0 models folder (`test/spikes/animated-models/models/`). `NG_SPIKE_MAX`; `NG_GPU_ARTIFACTS=<dir>` saves each mode's close-up. In Chrome: `npm run drive -- --serve . "test/spikes/skinned-models/index.html" --wait-for "skinned-models done" --timeout 900`. |
| `npm run agents:record` | Step 7, part 2: record what Herd & Wolves' rule decides, headless, into `tools/agents/out/` (ignored). `-- --episodes 2 --steps 360` for a short one. |
| `npm run agents:train` | Step 7, part 2: train one network per species on a recording, CPU only; writes `<species>.mlp.json` and `<species>.report.json` beside it. `-- --hidden 16x16 --epochs 2` for a quick one. |
| `npm run agents:evaluate` | Step 7, part 2: run the teacher, the rule without wander, and the students in the simulation, headless; prints catches and each driver's error against the label, and writes `evaluation.json` beside the students. `-- --students tools/agents/models/herd-wolves` judges the committed networks; `-- --episodes 1 --steps 300` for a quick one. |
| `npm run agents:export` | Step 7, part 2: write the trained networks as `.onnx` files into `tools/agents/out/onnx` (ignored). CPU only, under a second. `-- --students <dir>` picks other networks. |
| `venv/Scripts/python test/onnx-fixtures/make.py` | Step 8, phase 8.0: rewrite `test/onnx-fixtures/fixtures.json`, the small `.onnx` files that PyTorch, scikit-learn and `onnx.helper` write, with what each maker computed, and (since phase 9.2) the second files PyTorch writes beside them. `--list` prints what each file holds. Needs the packages of `requirements.txt` in `venv/` (see `gotchas.md` for `pip` there). The tests need no Python. A new case goes at the end of `torch_cases()`, or the earlier files change with the random stream. |

To refresh the test-engine site after library changes: `npm run yalc:push` here (it leaves a
**production** `dist/`; run `npm run build:lib` before committing), then restart the site with
`npm run dev -- --force`.
