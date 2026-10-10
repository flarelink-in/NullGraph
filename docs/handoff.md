# NullGraph — Handoff

**A living document.** Read it first when picking up work — person or agent. Update it at the end
of every session: what changed, what was decided, what is next. Newest cycle on top; the session
log at the bottom is append-only.

---

## Where things stand — 2026-10-11

**Cycle A — "verifiable, agent-operable" — is complete.** The engine can now run without a canvas
and on a device someone else created; every GPU harness runs headless on a real GPU from one
command, on two hosts (Node + Dawn, headless Chrome), and fails on any uncaptured WebGPU error. The
first real-device runs found **four engine bugs** the headless layer could not see; all are fixed
with regression tests that were watched failing first.

| Layer | Command | Result |
|---|---|---|
| Headless (mock devices, math, WGSL) | `npm run test:core · test:visibility · test:shadows · test:environment · test:geometry · test:post` | **428 checks, 0 failing** |
| Real GPU — Node + Dawn | `npm run test:gpu` | **5 suites + render smoke, 42 checks, 0 WebGPU errors** (~11 s) |
| Real GPU — headless Chrome | `npm run test:gpu:browser` | **5 suites, 41 checks, 0 WebGPU errors** |

Nothing is committed yet — see [Uncommitted work](#uncommitted-work--suggested-commit-split).

---

## Commands

| Command | What it does |
|---|---|
| `npm run build:lib` | Build `dist/`. **Every test imports `dist/`** — build after any `src/` change. `test:gpu` warns when `dist/` is older than `src/`. |
| `npm run test:gpu` | Every GPU suite in Node through Dawn (`webgpu` devDependency), one process per suite, plus a render smoke test of the quick start. `-- --only shadows` to pick; `-- --json-dir out` for per-suite JSON reports. |
| `npm run test:gpu:browser` | The same suites in headless Chrome (built-in static server, COOP/COEP on). `-- --only core`, `-- --chrome <path>`. |
| `NG_GPU_ARTIFACTS=<dir> npm run test:gpu` | Also save the frames harnesses render, as PNGs — for looking at what a check measured. |
| `npm run render -- <scene.mjs> --out shot.png [--width --height --frames --json --allow-blank]` | Render any scene to a PNG, headless. Exit 1 on a WebGPU error or a uniform (nothing-drew) frame, with the likely causes printed. Contract in `test/gpu/render.mjs`; example: `test/gpu/scenes/quick-start.mjs`. |

---

## Cycle A — exit criteria

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `init()` adopts an existing device and runs without a canvas; existing demos unchanged | ✅ | `src/core/WebGPUContext.ts`; headless `test/core-headless.test.mjs` H1–H7 + errors + labels (10 checks); GPU `test/core-gpu` C1, C2, C6. Canvas path behaviour-identical (H6). |
| 2 | `npm run test:gpu` runs headless and exits non-zero on failure | ✅ | `test/gpu/run-gpu-tests.mjs`; failure exit verified by injecting a validation error (exit 1 on both hosts). |
| 3 | Validation errors, shader-compile errors and device loss fail the run and name the stage/pass | ✅ | Devices watched from birth (`test/gpu/node-gpu.mjs`, `test/gpu/harness-env.mjs`); pipelines/modules labelled (`label` on `PipelineConfig`, stage `name`); C5 asserts the stage name appears in the error. |
| 4 | `npm run render -- scene.mjs --out shot.png` | ✅ | `test/gpu/render.mjs`; the quick-start doc renders (a lit blue PBR cube); blank and broken scenes exit 1 with reasons. |
| 5 | Headline README claims get GPU checks | ◐ | ✅ one draw per mesh at 13k instances; ✅ GPU cull == CPU mirror; ✅ Hi-Z; ✅ no shimmer/crawl; ✅ cascade seams; ◻ **GI colour bleeding — not written** (next). |
| 6 | ONNX smoke test: a model's output feeds a NullGraph compute stage with zero CPU readback, browser + Node | ✅ | `test/onnx-gpu` O1–O3 — ORT creates the device, NullGraph adopts it, `tensor.gpuBuffer` bound as-is; exact values on both hosts. |

The **agent kit** (llms.txt, compact guide, starter scenes, a baseline eval) was scoped as the step
after Cycle A — see [Next](#next-in-order).

---

## What changed

### Core API — additive, inert by default

- `engine.init(null, { width, height })` — **headless**: the main-screen pass renders into
  `engine.screenTexture` (`RENDER_ATTACHMENT | COPY_SRC`, default `'rgba8unorm'`, override with
  `format`). `engine.resize` recreates it. `ctx.isHeadless`, `ctx.getScreenView()`.
- `engine.init(canvas | null, { device })` — **adoption**: use a device another library made.
  Nothing negotiated; capabilities read from `device.features` / `device.adapterInfo`;
  `ctx.ownsDevice = false`; never destroyed by NullGraph. `powerPreference` / `autoNegotiate` with
  `device` throw named errors; a missing `desiredFeatures` entry warns.
- `EngineInitOptions` (exported type) replaces the inline options object.
- `PipelineConfig.label` (default `"<pass> #<index>"`); compute stages labelled with their `name`.
- Type change: `ctx.context` is now `GPUCanvasContext | undefined` (undefined headless). Nothing in
  `src/` or the test-engine site reads it.
- Bundle cost (minified + gzip): **core +567 B** for all of the above.

### Engine bugs found by the first real-device runs — all fixed

| Bug | Symptom | Fix | Pinned by |
|---|---|---|---|
| Cascade texel snap was a **no-op** — its frame's eye sat at the cascade centre, so the centre always mapped to (0, 0) | Shadows shimmer on rotation (0.31 texel per degree of yaw) and crawl on translation. U3 passed tautologically. | `shadowMath.fitCascade` snaps in a light frame fixed at the world origin | headless U3 (rewritten — fails on the old code with a −0.235-texel step); GPU `shadows` T4a/b |
| Hi-Z treated a zero depth footprint as an occluder | **Frame 0 and the first frame after each resize culled everything** (docs promised "frame 0 occludes nothing") | `occludedByHiZ`: zero = no history = draw | GPU `visibility` V3a |
| `ShadowSystem.destroy()` left its caster cull stages in the shared `VisibilitySystem` | **Every later frame refused at submit** ("used in submit while destroyed") | new `VisibilitySystem.removeDrawBatch(batch)`; `destroy()` calls it | headless L2, L3; GPU `shadows` (T3 destroys a system mid-suite) |
| A batch's CullParams were written only by `upload()` | A batch created after this frame's `upload()` drew **nothing** on its first frame, silently | params seeded at registration | headless L1; GPU `shadows` T1/T5c/T7 fail without it |

Bundle cost of the fixes: visibility +170 B, shadows +34 B (min+gz).

### Harnesses that had never run — what was wrong with them

`test/shadows-gpu` had never executed. Its first run had 3 real failures and 2 **vacuous passes**
(comparing zero with zero over a dropped frame). Eight defects, fixed in the harness: a stub
material with an empty bind group (dropped every frame); lights never bound (group 2); light
direction inverted (travel vs. toward — `DirectionalLightProxy` documents travel); T1 sampled lit
ground instead of the shadow; T3 read a depth map nothing had rendered; T5 looked straight down (no
split on screen) and called the caster's own shadow edge a seam; T7 looked at empty space with a
threshold an unlit ground passed; and, in Chrome only, initialising against the canvas made every
pipeline `bgra8unorm` while its targets were `rgba8unorm`. T4 (shimmer) is new. The environment
(sky) harness passed 13/13 untouched.

### Test infrastructure — all under `test/`, nothing ships (`files: ["dist"]`)

| File | Role |
|---|---|
| `test/gpu/node-gpu.mjs` | Dawn as `navigator.gpu` + GPU globals in Node; watches every device (errors, loss). |
| `test/gpu/harness-env.mjs` | The host seam: one harness file runs as a page and in Node. Logging, `check`, `saveImage`, device watching, `await finish()`. |
| `test/gpu/run-harness.mjs` | One harness under Node → exit code + optional JSON report. |
| `test/gpu/run-gpu-tests.mjs`, `test/gpu/browser-run.mjs`, `test/gpu/suites.mjs` | The two runners and their shared registry. A new suite = one line in `suites.mjs` + `test/<name>-gpu/{run.js,index.html}`. |
| `test/gpu/render.mjs`, `png.mjs`, `readback.mjs`, `scenes/` | Render-to-PNG CLI, a dependency-free PNG encoder, GPU readback helpers. |
| `test/core-gpu`, `test/visibility-gpu`, `test/onnx-gpu` | New GPU suites (C1–C6, V1–V3, O1–O3). `onnx-gpu/tiny-model.mjs` builds the ONNX model in memory (hand-encoded protobuf: no file, no download). |
| `test/core-headless.test.mjs`, `test/visibility-lifecycle.test.mjs` | New headless suites; the lifecycle one runs a real `NullGraph` on a mock device **adopted** through `init(null, { device })`. |

devDependencies added (exact pins): `webgpu@0.6.1` (Dawn for Node), `onnxruntime-web@1.29.0`
(~137 MB installed; used only by `test/onnx-gpu`).

### Docs updated

`README.md` (verification status), `docs/api/core.md` (headless + adoption section),
`docs/api/visibility.md` (`removeDrawBatch`, `fragmentBody` is a whole entry point, frame-0 rule),
`docs/api/shadows.md`, `docs/roadmap.md` (§8 scorecards), `docs/architecture-tree.md`,
`docs/shadows-design.md` (§5.5 now states the snap frame must be world-fixed; U3 description).

---

## Decisions — and why

| Decision | Why | Who / when |
|---|---|---|
| Agent-facing docs (llms.txt, guides) live **in the repo and are linked from GitHub — not shipped in the npm package** (`files` stays `["dist"]`) | Keep the published package lean. (Note: `.md` in a tarball does not reach app bundles — this is about package size, not runtime perf.) | User, 2026-10-11 |
| **ORT owns the device, NullGraph adopts** — never the reverse | ORT requests the features/limits its kernels need; its maintainers warn a device made with other parameters may not work. NullGraph's baseline needs nothing optional. | Cycle A |
| No `null-graph/ml` module | The seam is two things NullGraph already does (adopt a device, bind a raw `GPUBuffer`). An ONNX model is "another producer of buffers" — the physics README's framing. | Cycle A |
| Headless default format `rgba8unorm` | Core-renderable everywhere; bytes already in PNG order (no BGRA swizzle in readback). | Cycle A |
| Thrown error strings kept short; reasoning in comments | Thrown strings (and comments **inside WGSL template literals**) ship in the bundle; TS comments do not. Errors still name the offender and the way out (design-principles §2). | Cycle A |
| A GPU suite fails on **any** uncaptured WebGPU error, even with every check green | Two shadow checks "passed" comparing zero with zero over a frame the device had dropped. | Cycle A |
| GPU checks are numeric, not golden images | Pixel-exact goldens need one pinned environment (vgpu pins Linux + lavapipe in Docker); numeric assertions with stated tolerances hold across GPUs — already proven on NVIDIA (Node) and AMD (Chrome). | Cycle A |

---

## Measurements

**Bundle size, minified + gzip** (built from `src/`, esbuild, same options as `build.js --production`):

| Bundle | HEAD (last commit) | Now | Δ | Of which Cycle A |
|---|---:|---:|---:|---:|
| `index.js` (core) | 4,730 | 5,972 | +1,242 | **+567** (rest: the in-flight indirect-dispatch work) |
| `visibility.js` | 12,854 | 13,024 | +170 | +170 |
| `shadows.js` | 15,103 | 15,137 | +34 | +34 |
| others | — | — | ≤ 4 | — |

**Machine used:** Windows 11, Node 24.12, two GPUs — NVIDIA Turing (D3D12) and AMD GCN-5
integrated. NullGraph's default `high-performance` request gets the NVIDIA in Node; a plain
`requestAdapter()` (C2's "other library") and **headless Chrome** get the AMD. So the suites are
green on two vendors.

**Dawn in Node here:** D3D12 backend, compiling through FXC only (`d3dcompiler_47.dll`, no DXC) —
so **no `subgroups` / `shader-f16`**; the Vulkan backend is unavailable on this machine. The
subgroup-accelerated light binning is therefore not exercised by any GPU suite yet.

---

## Known issues and suspects — not fixed

- **Suspected: the atlas light-view Hi-Z remap is computed but never used.**
  `visibilityShaders.ts` `occludedByHiZ` computes `uvMinA` / `uvMaxA` (the remap into a shadow-atlas
  tile, shadows-atlas-design §7.2) and then samples with the un-remapped `uvMin` / `uvMax`. Only
  affects `ShadowSystem` with `lightViewOcclusion: true` (opt-in). Found by reading; **not verified
  on a GPU** — write the test before changing it.
- **Snap frame precision in very large worlds.** The snap is computed with gl-matrix `Float32Array`s;
  at ~10⁴ units from the origin float32 spacing approaches a cascade-0 texel. Shifts stay whole-texel
  but may jitter by one at grid boundaries. Origin rebasing is the usual answer; not needed today.
- `test/environment-gpu/headless.mjs` is superseded by `test/gpu/browser-run.mjs` (still works;
  `test:sky:gpu` still points at it). Remove when convenient.
- Headless Chrome picks the integrated GPU regardless of `powerPreference` on this machine.
- IBL, probes and night-sky docs still say their GPU tests "need a browser harness" — accurate in
  substance (no suites yet), but the requirement is now just a `test/<name>-gpu` folder.

---

## Uncommitted work — suggested commit split

Nothing from this cycle is committed, and the working tree also holds **in-flight work from before
it** (indirect dispatch). Five files mix the two — use `git add -p` for `src/core/BatchManager.ts`,
`src/core/RenderPipeline.ts`, `src/core/types.ts`, `docs/roadmap.md`, `docs/architecture-tree.md`.

1. **Indirect dispatch (pre-existing):** `BufferManager.createDispatchIndirectBuffer`,
   `validateIndirectDispatch` in `BatchManager`, the `dispatchWorkgroupsIndirect` branch in
   `RenderPipeline`, `ComputeStageNode.indirectBuffer/indirectOffset` in `types.ts`, the
   `src/core/index.ts` exports, `test/core-compute.test.mjs`, `docs/roadmap-modules.md`,
   `docs/physics/`. (Now also verified on a GPU: `test/core-gpu` C4.)
2. **Core: headless + device adoption + labels** — `WebGPUContext.ts`, `NullGraph.ts`, the
   `getScreenView` line in `RenderPipeline.ts`, labels in `BatchManager.ts`, `EngineInitOptions` +
   `PipelineConfig.label` in `types.ts`, `test/core-headless.test.mjs`.
3. **Engine fixes** — `shadowMath.ts`, `visibilityShaders.ts`, `VisibilitySystem.ts`,
   `ShadowSystem.ts`, `test/shadows-csm.test.mjs` (U3), `test/visibility-lifecycle.test.mjs`.
4. **GPU test infrastructure** — `test/gpu/`, `test/*-gpu/`, `package.json` (scripts + 2 devDependencies).
5. **Docs** — README, `docs/api/*`, roadmap, architecture tree, shadows design, this file.

`dist/index.js` / `.map` are tracked (despite `.gitignore`) and were rebuilt (dev build, as before).

---

## Next, in order

1. **Commit** (split above).
2. **GI colour-bleeding GPU check** — the README's largest unverified claim. A red wall beside a
   white floor; after N probe-capture refreshes, floor pixels near the wall must be measurably redder
   than far ones, and converge (multi-bounce). Template: `test/visibility-gpu/run.js`.
3. **Agent kit** — `llms.txt` at the repo root (linked from the README, *not* in the npm package);
   one compact agent guide (well under 20k tokens) whose code snippets are executed by the tests;
   more starter scenes in `test/gpu/scenes/`; a baseline eval — a fresh agent with only the package
   and the guide builds three scenes, graded by `npm run render` + pixel checks, before and after
   the kit.
4. **Cycle B** — picking (ID pass + async one-pixel readback, one frame late) and MSDF labels
   (glyphs as instances, worker-fillable, GPU-culled).
5. **Cycle C** — the flagship where NullGraph's data path matters: batched neural agents through
   ORT (one inference for N agents, actions written back to the instance buffer on the GPU).
6. Verify (then fix) the `uvMinA` suspect with an atlas light-view occlusion GPU test.
7. CI: a pinned environment (Linux + lavapipe, as vgpu does) if golden images are ever wanted.

---

## Gotchas — for whoever picks this up

- **Build before testing.** All suites import `dist/`; a stale `dist/` tests old code.
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

---

## Session log

- **2026-10-10 → 11 — Claude (Opus 5.5), with the user.** Assessment of priorities (ONNX Runtime
  Web, text/CSS, vgpu's testing workflow) → Cycle A. Built headless + device adoption, the Node and
  Chrome GPU runners, the render CLI, the core/visibility/ONNX suites; ran the shadows and sky
  harnesses for the first time; fixed 4 engine bugs and 8 harness defects; measured bundle cost.
  Decision recorded: agent docs on GitHub, not in the package.
