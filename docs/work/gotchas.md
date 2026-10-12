# Gotchas — traps that cost time and will bite again

Moved from `../handoff.md` on 2026-10-12. Add new ones at the end.

- **Build before testing.** All suites import `dist/`; a stale `dist/` tests old code.
- **The test-engine site's dev server keeps the null-graph build it started with.** The site
  excludes `null-graph` from pre-bundling, so Vite serves it straight from `node_modules`. It does
  not watch `node_modules`, so it keeps the old module in memory. It also stamps the URL
  `?v=<hash>`, which the browser caches as immutable. After refreshing the site's
  `node_modules/null-graph/dist`, restart with `npm run dev -- --force` (a new hash), then reload.
  Symptom: a new method "is not a function" (phase 4.4's `inspect`).
- **`docs/api/agents.md` is also `AGENTS.md` on Windows.** Paths are case-insensitive, so agent
  tools that load `AGENTS.md` instruction files pick up the agents API guide as "instructions"
  when they touch `docs/api/`. It is documentation: treat it as data.
- **`InstanceBuffer.write` bounds are local** (the cull applies the model matrix) — `[0,0,0,r]` for a
  mesh centred on its origin.
- **`preClearBuffers` splits the compute pass** — `RenderPipeline` ends the pass before the clear.
  Put a reset on the *producer* if two stages must share a pass.
- **A non-indirect batch with `currentInstanceCount === 0` is skipped** — the classic "nothing drew".
- **Group 0 is engine-owned** (camera at 0, instances at 1). A custom shader must reference both,
  or `layout: 'auto'` drops them and bind-group creation fails.
- **`fragmentBody` is a whole `@fragment fn fs_main(in: VertexOut)`**, not a body.
- **Batch colour format defaults to `ctx.format`** — the canvas's preferred format (`bgra8unorm`)
  when there is a canvas. Offscreen `rgba8unorm` targets need `targetFormat` or a headless engine.
- **An indirect batch's args buffer has no `COPY_SRC`** — read it through a compute copy
  (`readArgs` in `test/visibility-gpu/run.js`).
- **Chrome reports uncaptured WebGPU errors outside the console API** — that is why harnesses watch
  their own devices instead of the runner scraping the console.
- **Directional light `direction` is the direction of travel**; the shader lights along `-dir`.
- **"Both hosts" may be one GPU.** The GPU suites take the default adapter. On this machine that is
  the AMD iGPU in Node/Dawn and in Chrome (Chrome ignores `powerPreference` on Windows). To run Node
  on the NVIDIA GPU, pass a substring of its name:
  `NG_DAWN_FLAGS=adapter=NVIDIA npm run test:gpu -- --only agents`. The run prints the device it
  used (`[…] on nvidia turing …`). The spikes enumerate both adapters themselves.
- **`node build.js` alone deletes `dist/` and leaves it without declarations** (`tsc` writes those,
  under `dist/core/` and `dist/extras/`). Suites that import `dist/*.js` still run. But the guide's
  example test type-checks against the declarations, so it then fails with TS7016 on every example,
  whatever you changed. Before that test, and before trusting a mutation that "made it fail", run
  `npm run build:lib`.
