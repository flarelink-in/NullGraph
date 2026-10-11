# NullGraph — Handoff

**A living document.** Read it first when picking up work — person or agent. Update it at the end
of every session: what changed, what was decided, what is next. Newest cycle on top; the session
log at the bottom is append-only.

---

## Where things stand — 2026-10-11

**Cycle B — picking + text labels — is complete and committed** (seven commits; see below), with a live demo in the
test-engine site (*Axion Engine Loop → Picking & Labels*, see "Demo" below). **Cycle A —
"verifiable, agent-operable" — is complete and committed** (`c76fccc`..`e8c17c6` on `master`, not pushed).

| Layer | Command | Result |
|---|---|---|
| Headless (mock devices, math, WGSL) | `npm run test:core · test:visibility · test:shadows · test:environment · test:geometry · test:post · test:picking · test:text` | **458 checks, 0 failing** |
| Real GPU — Node + Dawn (NVIDIA) | `npm run test:gpu` | **8 suites: 64 checks + 1 render smoke, 0 WebGPU errors** |
| Real GPU — headless Chrome (AMD iGPU) | `npm run test:gpu:browser` | **8 suites, 65 checks, 0 WebGPU errors** |

### Cycle C — in progress: C1 steps 1–2 done (2026-10-11)

**C1 step 2 implemented and validated.** Shared XZ point-grid math lives in
`src/extras/spatial/gridMath.ts`: bin, exclusive prefix with a terminal offset, scatter, and excess
entries over the scan budget. `agentGridMath.ts` supplies the 3×3-cell query reference. U2/U3 add
five checks (4,096 positions, 64 prefix arrays, 512 random scenes compared with brute force,
scatter-order-independent ties, crowded-cell accounting). `npm run test:agents` now runs 29 checks.
GPU kernel generators now cover bin, hierarchical exclusive scan (terminal offset included),
scatter and bounded k-nearest queries. `test/agents-gpu/` is registered as the `agents` GPU suite.
`PointGrid` owns prebuilt shared stages/buffers; `AgentGrid` adds the nearest query output.
GPU reset dispatches make repeated builds in one submission safe (queue-side clears would not).
`null-graph/debug-ui` now exports `groundGrid()` and `gridOccupancy()`; occupancy reads GPU counts
directly, with red over-budget cells. Debug-view pixel assertions run in the GPU suite.
Node/Dawn and Chrome both reported AMD GCN-5; correctness checks cover prefix sizes through
65,537 elements, four 10,000-agent layouts, membership, overflow, species masks, stable stages,
repeated builds, empty rebuild, input refusals, debug pixels and teardown. Source buffers remain
caller-owned. See [`api/agent-grid.md`](./api/agent-grid.md) for usage and limitations.
Final validation: **36/36 GPU checks on each host, zero WebGPU errors**, 29 CPU agent checks,
17 core checks, library/TypeScript and IDE builds, and diff whitespace checks passed. Reviewed
floor/overflow PNGs are in `C:/Users/Vikas/AppData/Local/Temp/nullgraph-agent-grid-step2/`.
This completes the step-2 library/test debug views, **not** an `AgentSystem`, a moving crowd or a
new frontend GPU route. No performance claim follows from the correctness harness's readbacks.

**Demo: Spatial Grid — CPU Reference (2026-10-11).** A separate test-engine experiment at
`/dashboard/agents/spatial-grid` visualizes 256 static seeded points, 15×15 occupancy cells,
overflow, bounded nearest neighbours and brute-force misses. Herd & Wolves is unchanged.
Controls select uniform/boundary/crowded layouts, scan budget 1–64 and agent index 0–255.
Grid construction and queries run on the CPU only; rendering/culling run on the GPU. The shared
grid reference is exported through `null-graph/agents`; there is no new package subpath.
- **Purpose clarified:** occupancy partitions neighbour-search candidates. Queries visit a 3×3
  cell neighbourhood instead of the whole population; empty cells have no candidates. This is
  not rendering-tile culling. Red means entries exceed the per-cell scan budget (not lost storage);
  pink identifies neighbours missed by the bounded query.
- **Committed reference:** `d607d73` contains the library CPU grid/query reference and tests.
  The frontend experiment remains local under the commit caveat below.
- **Server cleanup:** on 2026-10-11, stopped the stale Vite Node process on port 8081 (PID 6328)
  at the user's request and verified no listening connection remained. Already-loaded browser
  pages can continue rendering until closed/refreshed; this did not stop the port-8083 server.
- **Next-work clarified:** the user selected continuing C1: finish step 2 GPU grid/query, then
  step 3. Step 2 is implemented locally with production wiring and both-host checks.
  S3 gates C1 step 3. The handoff clarification/server cleanup is committed as `747ec45`;
  the completed step-2 implementation remains uncommitted. No frontend edits were made here.
- **Validated after visual/type fixes:** frontend `tsc --noEmit`, IDE build, library build and
  29 agent checks passed. Targeted ESLint has zero errors (12 style warnings in integration/demo
  files); the five new experiment files pass Prettier. Existing integration formatting and unrelated
  frontend diff-check failures were left alone.
- **Browser:** forced Vite on port 8083; 11 driver steps, four screenshots, exit 0, no recorded
  console/WebGPU errors. Uniform/boundary: zero overflow and zero mismatches. Crowded/budget 2:
  105 excess entries and 136/256 mismatched queries. Tiles and unobscured label visually reviewed;
  wheel zoom exercised. This run did not exercise drag/orbit. Four warnings are React Router
  future flags and the Windows ignored power-preference warning (forwarded twice).
- **Evidence:** `C:/Users/Vikas/AppData/Local/Temp/nullgraph-spatial-grid-reviewed/` contains
  `report.json`, `console.log` and uniform/boundaries/crowded/zoomed screenshots. The shell wrapper
  timed out after 30 seconds, but the driver completed and its saved report records success.
- **Frontend commit caveat:** root `D:/NULLGRAPH/Minimal_TypeScript_v6.0.1` has existing staged
  unrelated work and most integration files are untracked. Do not stage entire integration files
  as an experiment-only commit without establishing their baseline. Local bundle/declaration copies
  were refreshed previously; nothing was published or pushed.

**C1 step 1 is done, headless only.** It adds `src/extras/agents/` (`agentLayout`, `agentRecords`,
`agentMath`) with the 64-byte record, the observation and action layouts derived from declared
`Sensors` / `Actions`, CPU mirrors of sensing and steering, and the fixed step. These mirrors are
the definitions later GPU stages are checked against. Run `npm run test:agents`: 24 checks, and six
injected bugs were each caught by the test meant for them. `agents.js` is 5,138 B min+gzip; core is
+0 B.

Semantics pinned and written into the design doc:
- **The local frame:** positive side = the direction a positive turn steers toward.
- **Non-finite action rows mean stop.** The GPU detects them via exponent bits, because WGSL may
  fold `v != v` away.
- **Spawn speed** is clamped to the agent's max; negative speed is refused.
- **The fixed step drops time only under real debt.** Physics P1's pseudocode counts a clamp at
  exactly `maxSteps × dt`; its spec text should be corrected to match.

Committed as `7f4cdb5`, directly after Cline's GI work (`1a70e8d`); the two were kept separate.

**Demo: Herd & Wolves, CPU preview (2026-10-11).** It lives in the test-engine site under a new nav
section, **Agent Experiments**, at `/dashboard/agents/herd-wolves-preview` (also the section's
index).
- **What it runs:** step 1's real API. `defineAgents` with `Sensors.self/nearest/bounds` gives the
  24-float observation; `AgentRecords`, `FixedStep`, `senseAll` and `stepAll` do the rest.
- **The brain:** a JS rule brain (`brain.ts`) that reads only what a WGSL rule brain will get —
  the observation row via named offsets, the species, the index and the tick. It is the prototype
  for step 3.
- **The scene:** 24 wolves and 300 prey as arrow-shaped darts, so headings show, on a 60 m field.
  Prey flock and flee; wolves chase, catch (a "caught!" marker), then rest 5 s. Caught prey respawn
  far from the pack. Rendering is `VisibilitySystem` instances written from the records each frame.
- **Interaction:** wolf names follow `vis.instanceBuffer`. Hover highlights; click opens the
  inspector label — what the agent senses (wolf / prey / wall in its own frame) and decides (turn,
  throttle, the brain's reason). A second, no-depth-test `TextSystem` holds the inspector, a stats
  sign and the catch markers.
- **Controls:** flocking, fleeing, simulation speed, wolf names.

Verified in headless Chrome (AMD iGPU), zero console errors:
- **Speed:** 60 FPS; one sim step costs 1.2–1.9 ms on the CPU (brute-force O(N²) sensing at 324
  agents). That cost is the reason for the GPU grid in step 2.
- **Inspector:** shows live values ("prey 3 of 3 · nearest side -0.15 fwd -0.05 → turn +0.02 ·
  throttle +0.25").
- **Site wiring:** the nav section and the Docs tab render.
- **Gameplay is brisk:** about 2 catches per second even after tuning (wolf 3.7 m/s, rest 5 s, prey
  sense 7 m). Retune at 20,000 agents in step 5.

Site files (the site's git tracks almost nothing; this is the record):
- **New:**
  - `src/data/engine-demos/demos/AgentsExperiments/HerdWolvesPreview/` (`index.ts`, `brain.ts`,
    `scene.ts`, `font.ts`)
  - `…/AgentsExperiments/HerdWolvesPreviewExample.ts`, `…/AgentsExperiments/HerdWolvesPreviewDocumentation.ts`
  - `src/pages/dashboard/AgentsExperiments/HerdWolvesPreview.tsx`
  - `src/layouts/components/engine-inspector/controls/herd-preview-controls.tsx`
- **Wiring:** `routes/paths.ts` (`agents` group), `routes/sections/dashboard.tsx` (`agents` routes),
  `layouts/config-nav-dashboard.tsx` (group + `agents` icon), `demoMap.ts`, `docsMap.ts`,
  `workers/engine/demoRegistry.ts` (`AGENTS_HERD_PREVIEW`), `workers/types/types.ts`
  (`HerdPreviewState`), `store/useDemoControlStore.ts`, `inspector-tab.tsx`.

The plan and design spec is [`agents-design.md`](./agents-design.md): batched agents, sense → think
→ act on the GPU, as an opt-in `null-graph/agents` module with zero core change. Phases:
- **C0:** spikes S1–S5, scripts only.
- **C1:** a population with rule brains, the shared spatial grid (physics P2's stages A–C), and
  output to visibility, labels and picking. A product on its own.
- **C2:** neural brains — a built-in WGSL kernel, `MlpBrain.fromOnnx`, an `OrtBrain` adapter, and
  JS training tools.

**Decisions settled with the user** (the doc's last section): a hybrid brain runtime — rules and a
built-in WGSL kernel ship, ORT is the optional research path, never the default; ORT goes through a
generic opt-in `null-graph/inference` adapter (ORT an optional peer dependency, never bundled); JS
training tool; Herd & Wolves demo; module name `null-graph/agents`. Still open: the engine-first
device amendment, decided by spike S5. Nothing is implemented yet.

Facts verified in ONNX Runtime 1.29 while planning, and why they matter:
- **`run()` cannot submit inside the call.** It awaits input preparation first, so an ORT brain
  lands at least one frame late in an interactive loop. WGSL brains have no such gap.
- **Graph capture binds GPU inputs and outputs once, then silently ignores new ones.** So the
  agents' observation and action buffers must be fixed for life.
- **`env.webgpu.device` can be set to an app-created device.** Cycle A's "ORT owns the device, never
  the reverse" can relax for engine-first hosts such as the site's worker, after spike S5.
- **ORT's WebGPU WASM is 25.7 MB, 6.4 MB gzipped.** A typical agent brain is 7.7 KB, which is why
  small dense networks default to the built-in kernel.

### GI colour-bleeding check — 2026-10-11 (committed `1a70e8d`)

`test/probes-gpu/{run.js,index.html}`, registered as `probes` in both GPU runners, now measures a
red wall beside a neutral diffuse floor. Six numeric checks on real rendered linear-HDR pixels:
nonblank/finite/unsaturated patches, stronger red near the wall than far away, disabled-GI and
neutral-wall controls, feedback versus direct-only captures, and convergence over 16 complete
refreshes of a 24-probe volume. One probe per frame keeps Node pipeline construction under the
execution tool timeout. Late change is measured before tone mapping or RGBA8 quantization.

NVIDIA/Node: near red excess **0.01826**, far **0.00249**, direct-only **0.00540**; AMD/Chrome:
**0.01825 / 0.00248 / 0.00540**. Both controls remove the tint; the final four refreshes have zero
measured change at rgba16float precision. All six checks pass with zero WebGPU errors on both.
The first estimate includes environment fallback, so convergence is not necessarily monotonic.
This does not yet verify the enclosed white-room energy test, leak bounds, or local reflections.

**Bug found and fixed:** `ReflProbeBuffer` used `count: u32; _pad: vec3<u32>` — WGSL aligns that
vec3 to byte 16, putting the runtime probe array at byte 32, while CPU records start at byte 16.
The empty 64-byte placeholder was rejected (minimum 80 bytes), dropping every GI frame. Three
scalar padding fields restore the intended 16-byte header; a new reflector-based headless check
pins the byte offset (old 32, fixed 16). No core or public API changes.

Known lifecycle gap found while reading: `ProbeVolume.destroy()` leaves its capture batches in
the visibility system. This harness explicitly unregisters those batches before destruction;
the library lifecycle fix and its own regression test are follow-up work.

---

## Frontend GI sequencing — 2026-10-11 (uncommitted)

Updated the external site at `D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`:
- Architectural Irradiance Probes: submit environment + shadow atlas before startup prewarm;
  use public `setIntensity`; full-volume UI rebake invalidates bounds and refreshes amortized.
- Sky Sources: UI rebake uses amortized invalidation of the complete grid (including edge probes).
- Both demos: `afterRender` copies environment SH after the bake submission. `EngineCore` calls
  this optional hook and suppresses rendering during setup and asynchronous teardown.
- Irradiance tutorial: initialize lights, camera, environment and shadows before startup prewarm;
  copy SH after submitted frame bakes; remove the claim that initial filling proves convergence.
- Updated both rebake button labels and Architectural GI documentation for progressive behavior.

Validation: IDE build reports no problems; `npm run build` passed its TypeScript phase but the
Vite production build did not finish within the tool wait limits (not claimed passing). Separate
`tsc --noEmit` emitted no diagnostics. All four embedded tutorial source strings transpile without
syntax errors (not a semantic or live tutorial test). Chrome dashboard smoke: both routes loaded,
rendered at roughly 36–38 FPS, and accepted intensity/sun/rebake/invalidation worker-state messages.
Page-console capture showed only React Router future-flag warnings. This smoke did not attach
to worker console targets, prove visual changes, or verify in-place route teardown. Live tutorial,
worker-level WebGPU error capture, and thorough interactive/visual checks remain follow-up.
No library API changes, package refresh, commits, push, or agent-kit implementation this session.

---

## Cycle B — picking + text labels

Specs first, as house style: [`picking-design.md`](./picking-design.md),
[`text-labels-design.md`](./text-labels-design.md). API docs: [`api/picking.md`](./api/picking.md),
[`api/text.md`](./api/text.md).

### What shipped

| | Picking (`null-graph/picking`) | Text (`null-graph/text`) |
|---|---|---|
| What | `picker.pick(x, y, camera, { radius })` → source, the app's instance index, depth, world position | Screen-space labels: `TextSystem` + `FontAtlas` + pure `layoutText` |
| How | One more view: a camera zoomed onto the (2R+1)² clicked pixels, culled by the visibility system's own kernel, drawn into an `rgba32uint` target, a few texels read back. Its own submission, off the frame path | Glyphs are 16-byte records; a GPU cull compacts visible glyphs into one `drawIndexedIndirect`; MSDF/MTSDF/SDF antialiasing with halos; pixel-snapped; depth test at the anchor |
| Sources / inputs | `VisibilitySystem` instances (`addInstances`), stock-material plain batches (`addBatch`), custom layouts (`shaderCode: (tag) => wgsl`) | msdf-atlas-gen JSON + image (`fromMSDF`), runtime Canvas SDF from any browser font (`fromCanvas`), raw (`fromSDF`) |
| Headline | ~0.5 ms per pick (NVIDIA, Node); free when idle (pick culls ride the chain disabled) | 600,000 glyphs, one draw; labels **follow any GPU buffer** (e.g. `vis.instanceBuffer`) with zero CPU work |
| Headless tests | `test/picking.test.mjs` — 11 | `test/text.test.mjs` — 18 |
| GPU tests | `test/picking-gpu` P1–P8, every expectation checked against a **CPU ray cast** | `test/text-gpu` X1–X9 (+X4b) — position to the pixel, constant size, cull = CPU mirror, follow, depth, halo, AA coverage, scale; X9 (real font via Canvas) browser-only |
| Bundle (min+gz) | 4,150 B, opt-in import | 8,417 B, opt-in import |

**Core: untouched by Cycle B** (`index.js` 5,972 B before and after). Engine touchpoints, both in
`null-graph/visibility` (+35 B):
- **Compacted instance records carry `sourceIndex`** — the instance cull writes each survivor's index
  in the app's buffer into the record's last field (was `_pad`; `INSTANCE_SOURCE_INDEX_OFFSET` = 23).
  Picking reports it; any custom instance-tier shader can now find its own per-instance data.
- **`VisibilitySystem.instanceBuffer` is public read-only** — so labels (or a simulation) can read the
  same GPU records with no copy.

### Decisions (Cycle B)

| Decision | Why |
|---|---|
| A pick is an immediate, separate `engine.render()`, not part of the app's frame | No core hook, no per-frame cost, no frame coupling; one GPU round trip of latency |
| Fonts: the module ships **no font data**; three atlas sources (msdf-atlas-gen, runtime Canvas, raw) | Keeps the package lean (user's standing preference); Canvas gives zero-setup in browsers |
| Tests use a **synthetic font** of boxes/bar/ring with analytic fields (`test/gpu/synthetic-font.mjs`) | Pixel-exact assertions, no licence, no download; real fonts covered by X9 in Chrome |
| Labels are screen-space only in v1; one draw batch per `TextSystem` | The use cases (nameplates, map/data labels) are pixel-sized; one cull per view keeps it simple |
| Glyph-record culling, not per-label expansion | Trivially parallel, no prefix sums; 600k glyphs cull in well under a millisecond on desktop |

### Known gaps (Cycle B)

- Picking: meshlet tier, skinned meshes, labels, rect selection. (Hover works app-side: the demo
  re-picks on mouse move and every 120 ms.)
- **`Picker.destroy()` leaves its (empty) pass in `engine.passes`.** Core has no `removePass`, so
  an app that calls `engine.render()` with **no** sequence after destroying a picker would bind
  the destroyed pick textures. Sequence-driven apps (all the demos) are unaffected. Fix: a core
  `removePass`, or keep the pick pass out of the tracked list.
- Text: decluttering, shaping (Arabic/Devanagari), rich text, world-size labels, picking labels,
  atlas generation in Node (no OffscreenCanvas — load an msdf-atlas-gen atlas instead).
- **Pointer — label decluttering (text v2).** Seen in practice: both site demos (Picking & Labels,
  Herd & Wolves) have dense spots where labels overlap and become unreadable.
  - **Planned approach** (`text-labels-design.md` §1: "a compute pass over the same records"): after
    the text cull, each visible label claims the cells of a coarse screen grid (about 32 px) under
    its rectangle with `atomicMin(priority << 16 | label)`. A second pass keeps only labels that won
    their cells. Priority: selected / hovered first, then nearer.
  - **No CPU, no readback,** one extra pass.
  - **Workarounds today:** `maxDistance`, a second `depthTest: false` system for labels that must
    read, and fading the rest during a selection.
- Text draw order inside one system is undefined (atomic compaction), and the depth test uses the
  anchor's depth across the whole quad. Both are documented with workarounds in `api/text.md` §5.
  The demo shows the pattern: a second `depthTest: false` system for overlay labels.
- Real-font quality: `fromCanvas` (Inter 600, 48 px em, range 16) reviewed by eye in the demo, crisp
  at 13–17 px with 2–3 px halos. An msdf-atlas-gen atlas has still not been looked at.
- A render-CLI text scene would need an atlas file in Node — none is committed (licence).
- Fixed while building the demo: `TextSystem.destroy()` left its batch in the pass (same class as
  Cycle A's caster bug). It now calls `engine.clearBatch`; S5 in `test/text.test.mjs` was watched
  failing first.

### Demo — Picking & Labels in the test-engine site (2026-10-11)

`D:\NULLGRAPH\Minimal_TypeScript_v6.0.1\starter-vite-ts`, route `/dashboard/engine/picking-labels`.
- **The scene:** 2,000 capsule agents in four teams wander a 180 m grid. They are GPU-culled through
  `VisibilitySystem`, one indirect draw per colour variant.
- **Labels:** every agent has a name label that follows `vis.instanceBuffer` on the GPU.
- **Interaction:** hover highlights (picks on mouse move); clicking an agent selects it (gold, camera
  follows, live info label); clicking the ground sends the selection to the exact picked world point.
- **Highlighting** works by rewriting the instance's `meshIndex` to a variant mesh. No colour channel
  is needed, and the picker registers every variant.

Verified in headless Chrome (AMD iGPU, 818×824) by a CDP script that hovers and clicks:
- **Interactions:** every one behaves, with zero console errors and zero WebGPU warnings.
- **Ground pick accuracy:** a centre click landed at (−0.6, −0.9). Hand-computing it from the orbit
  angles gives (−0.64, −0.93).
- **Cost:** GPU 1.7–3.8 ms per frame.

Site changes. The site's git tracks almost nothing, so this list is the record:
- **New:**
  - `src/data/engine-demos/demos/AxionEngineLoop/PickingLabels/` (`index.ts`, `agents.ts`, `scene.ts`, `font.ts`)
  - `…/AxionEngineLoop/PickingLabelsExample.ts`, `…/AxionEngineLoop/PickingLabelsDocumentation.ts`
  - `src/pages/dashboard/AxionEngineLoop/PickingLabels.tsx`
  - `src/layouts/components/engine-inspector/controls/picking-labels-controls.tsx`
- **Wiring:** `demoRegistry.ts`, `demoMap.ts`, `docsMap.ts`, `routes/paths.ts`,
  `routes/sections/dashboard.tsx`, `layouts/config-nav-dashboard.tsx`, `inspector-tab.tsx`,
  `store/useDemoControlStore.ts`, `workers/types/types.ts` (`PickingLabelsState`, `PointerClick`).
- **A click channel the site lacked:**
  - `EngineCanvas.tsx` posts `POINTER_CLICK` for a press released within 6 px and 500 ms (drags still orbit).
  - `MessageRouter.ts` and `StateStore.ts` keep `click {x, y, button, seq}`.
  - `EngineCore.ts` adds `click` to every demo's `getInput()`, so any demo can use it.

To refresh the site after null-graph changes, run `npm run yalc:push` here, then `npm run dev`
there (port 8081). `yalc:push` pushes to **three** local consumers (this site,
`D:\NULLGRAPH-STARTER\…v6.1.0`, `D:\orion-realms\orion-engine-test`) and leaves a **production**
`dist/` in this repo. Run `npm run build:lib` before committing.

---

## Commands

| Command | What it does |
|---|---|
| `npm run build:lib` | Build `dist/`. **Every test imports `dist/`** — build after any `src/` change. `test:gpu` warns when `dist/` is older than `src/`. |
| `npm run test:gpu` | Every GPU suite in Node through Dawn (`webgpu` devDependency), one process per suite, plus a render smoke test of the quick start. `-- --only shadows` to pick; `-- --json-dir out` for per-suite JSON reports. |
| `npm run test:gpu:browser` | The same suites in headless Chrome (built-in static server, COOP/COEP on). `-- --only core`, `-- --chrome <path>`. |
| `NG_GPU_ARTIFACTS=<dir> npm run test:gpu` | Also save the frames harnesses render, as PNGs — for looking at what a check measured. |
| `npm run render -- <scene.mjs> --out shot.png [--width --height --frames --json --allow-blank]` | Render any scene to a PNG, headless. Exit 1 on a WebGPU error or a uniform (nothing-drew) frame, with the likely causes printed. Contract in `test/gpu/render.mjs`; example: `test/gpu/scenes/quick-start.mjs`. |
| `npm run drive -- <url> [--wait-for re] [--actions steps.json \| --interactive] [--canvas sel] [--out dir]` | Drive a page in headless Chrome: move/click/key/wheel, screenshots, eval. Captures page **and worker** consoles and watches every GPU device (a hook installed before page and worker scripts run). Exit 1 on an exception, console error or WebGPU error; 2 when it cannot run. `--serve <dir> <path>` drives repository pages. See `test/gpu/drive-page.mjs` and `docs/agent/validation.md`. |

---

## Cycle A — exit criteria

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `init()` adopts an existing device and runs without a canvas; existing demos unchanged | ✅ | `src/core/WebGPUContext.ts`; headless `test/core-headless.test.mjs` H1–H7 + errors + labels (10 checks); GPU `test/core-gpu` C1, C2, C6. Canvas path behaviour-identical (H6). |
| 2 | `npm run test:gpu` runs headless and exits non-zero on failure | ✅ | `test/gpu/run-gpu-tests.mjs`; failure exit verified by injecting a validation error (exit 1 on both hosts). |
| 3 | Validation errors, shader-compile errors and device loss fail the run and name the stage/pass | ✅ | Devices watched from birth (`test/gpu/node-gpu.mjs`, `test/gpu/harness-env.mjs`); pipelines/modules labelled (`label` on `PipelineConfig`, stage `name`); C5 asserts the stage name appears in the error. |
| 4 | `npm run render -- scene.mjs --out shot.png` | ✅ | `test/gpu/render.mjs`; the quick-start doc renders (a lit blue PBR cube); blank and broken scenes exit 1 with reasons. |
| 5 | Headline README claims get GPU checks | ✅ | ✅ one draw per mesh at 13k instances; ✅ GPU cull == CPU mirror; ✅ Hi-Z; ✅ no shimmer/crawl; ✅ cascade seams; ✅ **GI colour bleeding + refresh convergence** (`test/probes-gpu` G1–G6). |
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
| ~~No `null-graph/ml` module~~ **Revised:** a small generic, opt-in `null-graph/inference` adapter; ORT stays an optional peer dependency, never bundled or on the default path | Cycle A assumed the seam was trivial (adopt a device, bind a raw `GPUBuffer`). Cycle C planning showed it is not: device direction, graph capture's fixed-buffer rule, two schedules and shape checks are each easy to get wrong, and both agents and image projects need them | Cycle A; revised by the user, 2026-10-11 (Cycle C) |
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

## Cycle B — commit record

**Cycle B — committed locally, not pushed.** Seven dependency-ordered commits:
1. `be3b6bf` — picking and text design specs.
2. `e6629eb` — visibility source indices, public instance buffer, API notes.
3. `92fab9b` — picking implementation, build entry point, package export, API docs.
4. `846fac8` — picking headless/GPU tests, test script, suite registration.
5. `e142a74` — text implementation, font atlases, build entry point, package export, API docs.
6. `102729a` — text headless/GPU tests, synthetic font, test script, render-CLI integration.
7. Final Cycle B docs commit — README, architecture tree, this handoff (see `git log`).

Final-tree verification on 2026-10-11: development build + TypeScript pass; all eight headless
test groups pass (457 checks); Chrome passes 59 GPU checks; Node passes 58 GPU checks plus the
render smoke, all with zero WebGPU errors. Node's full command exceeded the execution tool's
30-second limit, so every suite was rerun successfully via `--only` in smaller groups.
The extra Chrome check is X9 (Canvas font); it is intentionally skipped in Node.
Isolated historical commit-boundary verification was attempted but timed out; it is not claimed
as completed. The tracked core `dist/index.js` and source map rebuild without changes;
other bundles remain ignored, matching the existing repository policy.

Cycle A, for the record — **done, committed as below** (`c76fccc` indirect dispatch · `424e346` core · `c495a7e` fixes · `f43154f` GPU tests · `e623777` docs · `e8c17c6` dist). The README / roadmap / architecture-tree edits from before Cycle A went in with the docs commit; `package.json` went in with the GPU-test commit. Kept for the record:

Nothing from this cycle was committed, and the working tree also held **in-flight work from before
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

## Agent kit Stage 1 — 2026-10-11

Implemented Option 1 (README discovery, repository-side kit), not Option 2 packaging:
- Manifest bumped to 1.1.1; no tag, commit, push or npm publication performed.
- Root `llms.txt` points into curated `docs/agent/`: router, guide, validation/open-platform pointers.
- README provides branch/raw discovery links explicitly marked development-only. Anonymous
  tag-specific URLs must be verified after publishing v1.1.1; do not claim that tag already exists.
- Asset-free instancing and instance-tier frustum-culling starters were distilled from frontend
  data-oriented setup, without copying its React/worker/assets/post/meshlet dependencies.
- Registered both Node render smoke scenes. Guide snippets are exact copies, enforced by
  `npm run test:agent-kit`; link, version, registration and approximate 50k-token ceiling checks.
- Corrected historical composition guide claims that visibility/meshlets/shadows were unshipped.
- Reused the full existing render harness rather than introducing a weaker tiny offline harness.

Validation: fresh library build + declarations passed; 4 agent-kit checks passed; Node/Dawn on
NVIDIA Turing rendered all three starters without recorded GPU errors. Visibility's 5 semantic
checks passed separately (draw-count scaling, CPU-mirror frustum match, Hi-Z controls). These
are not an independent fresh-agent eval or new cross-platform coverage. Tag/release pinning,
independent before/after eval and broader platform checks remain open. Option 2 requires a
separate packaging decision; a documentation subpath resolves files but does not auto-read them.

### Local kit-guided trials — 2026-10-11

The user ran three prompts in separate project sessions using the local kit entry point.
This is useful discovery/adaptation evidence, not a controlled before/after benchmark or a
published npm-to-HTTP discovery test. No broader platform coverage was established.

1. **Plain instancing:** `D:/orion-realms/nullgraph-agent-mini/blue-grid.mjs` constructs a
   deterministic 50 x 50 grid (2,500 cubes), one plain batch/material, 14-float records,
   blue base color `[0.1, 0.3, 1, 1]`, camera `[0, 48, 42]`. The source and saved report were
   inspected here: `blue-grid.json` records a 1024 x 768, two-frame Node/Dawn render on NVIDIA
   Turing/D3D12, no recorded GPU errors, non-blank output, about 30.8% coverage. The agent
   reported exit 0 and image inspection. Record construction/upload specifies 2,500; neither
   pixel counting nor encoder draw/instance-count instrumentation independently verified it.
   Files also include `blue-grid.png` and `render.log`. The exact requested output folder was
   created; it differs from the agent workspace named `null-graph-agents-mini`.
2. **Instance-tier culling:** artifacts are in
   `D:/orion-realms/nullgraph-agent-culling-mini/` (`culling-scene.mjs`, PNG/JSON, logs, REPORT.md).
   The agent reported a deterministic 10,000-cube, spacing-4 grid, one instance-tier batch,
   no meshlets/Hi-Z/post/assets. The saved `culling.json` was inspected: 512 x 512, two frames,
   NVIDIA Turing/D3D12, no recorded GPU errors, non-blank output, about 18.1% coverage.
   The agent additionally reported visibility-suite exit 0, 5/5 checks. Those checks validate
   separate library test scenes, not this generated scene's visible count. Its scene-specific
   cull count and GPU-vs-CPU reference comparison remain unverified.
3. **Standalone Vite adaptation:** project at
   `D:/orion-realms/null-graph-agents-mini/instancing-vite`. Its source/package/config were
   inspected: public engine subpath imports replace runner injection; one render loop;
   ResizeObserver updates engine size/projection; HMR cancels the loop. The agent reported
   `vite build` passed but initially had not browser-tested it. The user subsequently confirmed
   the browser renders as expected. No independent browser-console capture was performed here.
   It uses the kit's 10,000-cube starter, NOT the original 2,500-cube scene, so this proves browser
   integration rather than faithful preservation of the first task. Dependency is
   `file:../.yalc/null-graph`; the agent reported that copy is 1.1.0 versus guide target 1.1.1.
   `gl-matrix` is direct and deduped in Vite. This tests linked development consumption, not a
   fresh packed 1.1.1 consumer install. HMR does not disconnect the observer or release owned
   GPU resources; repeated reload/teardown behavior remains open.

The sessions followed the root index into the curated guide/examples and accurately distinguished
render smoke evidence from semantic culling assertions. They disclosed using existing `dist/`
without rebuilding or establishing source/build identity. That is a reproducibility gap, not
proof that the tested build was stale. Option 1 remains the current choice; Option 2 is deferred.

**Small next improvements (identified, not yet implemented):**
- Guide: preserve requested count/layout/material/camera when converting an existing scene;
  use a starter for structure, not as a replacement for task requirements.
- Require explicit engine/build provenance and disclose version mismatches before validation.
- Make browser cleanup concrete: cancel scheduling, disconnect observers, release owned resources.
- Repeat the original 2,500-cube browser conversion with a freshly built local 1.1.1 tarball;
  check rendering, resize and repeated HMR. Keep yalc as the development workflow.
- Keep scene-specific culling instrumentation as a separate follow-up, and retain the pending
  independent before/after eval and release-pinned anonymous discovery test.

## Next, in order

1. ~~Commit Cycle A~~ — done. ~~**Cycle B: implement and commit**~~ — done (seven local commits, not pushed).
2. ~~**GI colour-bleeding GPU check**~~ — done, six checks pass on Node/NVIDIA and Chrome/AMD;
   header-layout fix and harness committed as `1a70e8d`. See the evidence above.
3. **Agent kit follow-up** — Stage 1 implemented; three local kit-guided trials completed as
   recorded above. Keep Option 1 (README-linked root index, kit not shipped in npm), the curated
   `docs/agent/` folder and approximate 50k-token guide ceiling. Implement the small guidance
   fixes, repeat a requirement-preserving browser conversion with a local 1.1.1 tarball, then
   review/release/tag and verify pinned discovery. Controlled before/after evaluation remains
   pending; do not equate the local trials with that benchmark.
4. **Cycle C** — batched agents, now planned in [`agents-design.md`](./agents-design.md).
   1. ~~Clean tree~~ — agent kit `7054b16`, GI work `1a70e8d`, C1 step 1 `feat(agents)`.
   2. ~~Get the user's answers~~ — settled 2026-10-11 (hybrid, `null-graph/inference`, JS, Herd & Wolves).
    3. ~~C1 steps 1–2~~ — done: CPU references, shared XZ point-grid GPU stages, query and debug
       draws (U2, U3, A1). **Next: spike S3**, the built-in kernel's cost, before step 3; the ORT spikes S1,
      S2, S4 and S5 run before C2's ORT step.
   4. C1 steps 3–5, then C2. Labels and picking from Cycle B are the "click a unit, see its brain"
      half.
5. Verify (then fix) the `uvMinA` suspect with an atlas light-view occlusion GPU test.
6. CI: a pinned environment (Linux + lavapipe, as vgpu does) if golden images are ever wanted.

### After Cycle B — the ONNX/AI product plan (agreed with the user, 2026-10-11)

Positioning: *AI outputs that light, shadow and populate a world, entirely on the user's GPU* —
model outputs feed engine systems (lights, IBL, shadows, post, instancing), not just point clouds.
On-device inference = no server cost per user, and private data never leaves the device.

1. **Event photo booth** (on the user's own event-booking platform) — person segmentation + depth
   put an attendee inside the event's branded 3D set, in front of / behind objects, lit and
   shadowed; share a photo or 5 s clip. Camera never leaves the device. Most directly monetizable
   (premium custom sets for organizers). Small segmentation model (~1 MB).
2. **Batched neural agents** (the showpiece) — agent state in the instance buffer; a compute stage
   builds every agent's observation into one tensor; one ORT call for all N; a compute stage
   applies actions. Path: rule-based compute shader → behaviour-cloned tiny MLP (training script
   provided, no ML background needed) → in-browser evolution. First milestone: 20k agents, one
   brain, 60 fps, inference time measured. Ideas: live-evolving ecosystem, train-your-swarm game,
   venue evacuation sim (ties to the event platform).
3. **Relight your photo** — depth model (Depth Anything V2 Small, Apache-2.0) → relightable surface
   with real lights, shadows, post. Privacy line: works in airplane mode after first load. A
   showcase/funnel more than a business; reuses the depth pipeline from (1).

Parked: room-matched lighting for e-commerce (needs market access the user does not have).

### Far horizon — ideas worth keeping (2026-10-11)

Server-side runs use the same code: NullGraph runs headless in Node (Dawn), so a rented Linux GPU
box can host it. Not yet verified on Linux — first step is `npm run test:gpu` on one.

1. **Train on server, play in browser** — evolve batched-agent brains at 1M agents on a rented GPU,
   ship the best `.onnx` brains to browsers running the same engine. (Novel combination.)
2. **Synthetic training-data factory** — randomized sky/time/materials/post, exact labels for free;
   train small models, run them back in NullGraph via ORT (e.g. the photo booth's segmentation).
3. **Server sim, browser rendering** — flat-array state streams naturally; venue digital twins.
4. **Agents that see** — each agent gets its own tiny rendered view (16×16–32×32) and a small
   vision brain; the shadow atlas is already a many-cameras-in-one-texture renderer with per-view
   culling. Lighting, night and fog then change what agents perceive. (Most extreme; see below.)
5. **Auto-cinematographer** — an aesthetic-score model + optimizer over time of day, lights, fog,
   post presets, run headless with the render CLI → best shot of a scene (event hero images).
6. **Self-narrating sim** — a small language model writes the story from aggregate sim stats.
7. **Opt-in visitor GPU time** for evolution runs — only ever clearly opt-in.

**Agents that see — phases.** P0 ray-cast "1D eyes" (16–32 rays per agent, compute shader,
10k+ agents) → P1 rendered 16×16 eyes in an atlas, multi-view instanced draws, hundreds to ~2k
agents → P2 perception depends on the physical sky/night/fog/shadows; colour genes let camouflage
evolve → P3 server-scale evolution (idea 1) with browser playback. Per-agent evolved weights fit a
hand-written WGSL tiny-net kernel better than ORT (which runs one shared set of weights); ORT fits
a shared brain trained offline.

Rendering options for many eyes: (A) multi-view instancing — eye cameras computed on the GPU,
(eye, object) pairs found through a spatial grid (the physics P2 broadphase grid), one indirect
draw into an eye atlas, out-of-tile fragments discarded; (B) compute ray-march of a coarse world
for ≤16×16 eyes (P0's 1D eye is its one-row case). The existing shadow atlas fits the "few
detailed eyes" end of the dial; it is depth-only and per-view-batch today.

Ideas to keep: asymmetric eyes per species (prey wide/low-res, predators narrow/sharp), eyes as an
energy-costing gene; day/night as the main lever — nocturnal vs diurnal niches, moon phase, long
evening shadows as cover, fog ambush, a night-vision-vs-acuity gene, agent-emitted light
(bioluminescence) via clustered lights. **Catch:** agent eyes must bypass auto-exposure/tone mapping
(fixed exposure, or exposure as a gene), or night stops being dark to them; compress a day to
minutes. Defensible pitch: "thousands of agents learning to see — live in your browser, under a
real sky"; cite Visual Hide and Seek (arXiv 1910.07882) and What if Eye…? (arXiv 2501.15001). Prior art to study: Madrona batch renderer (Stanford), Isaac Lab
tiled cameras (both server/native RL), ray-sense evolution sims like The Bibites.

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
- **2026-10-11 — Claude.** Committed Cycle A as six commits (your pre-existing indirect-dispatch work first, separately). Rebuilt `dist/` as a dev build — a production build had replaced it in the meantime.
- **2026-10-11 — Claude (Opus 5.5).** Cycle B: specs, then `null-graph/picking` and `null-graph/text`;
  the `sourceIndex` touchpoint; `vis.instanceBuffer` public. 28 headless + 18 GPU checks, every key
  check watched failing on broken code (sourceIndex revert → every pick returns #0; doubled pixel
  range → 11% coverage instead of 30%; no off-screen cull → 1,058 vs 237 glyphs). Green on Node/NVIDIA
  and Chrome/AMD. Core bundle unchanged. Not committed yet.
- **2026-10-11 — Claude (Opus 5.5).** Built the Picking & Labels demo in the test-engine site, at the
  user's request, before committing Cycle B.
  - **Site:** added a click channel to the site's worker plumbing. Drove the demo in headless Chrome
    over CDP (hover → select → command → arrive → release) and fixed what the screenshots showed: the
    far-label pile-up, the info label being unreadable, and a waypoint label hidden by nearer ground.
  - **Library fix:** `TextSystem.destroy()` now takes its batch out of the pass (S5 failed first).
  - **Docs:** `api/text.md` gained the draw-order and depth notes.
  - **Tests:** all 13 headless suites and 8 Node GPU suites green. Cycle B still not committed — the
    user wanted to see the demo first.
- **2026-10-11 — Cline.** Picked up Cycle B and committed it in seven dependency-ordered commits.
  Preserved the existing implementation and split shared build/package/registry changes by feature.
  Rebuilt development `dist/`; confirmed 457 headless checks, all seven Node GPU suites plus the
  render smoke, and all seven Chrome GPU suites, with zero WebGPU errors. Corrected the README's
  stale headless count and clarified Node versus Chrome GPU totals. Historical boundary checks
  exceeded the tool timeout; final-tree verification completed. No push, demo-site edits, or
  unrelated follow-on work. Next: the GI colour-bleeding GPU check.
- **2026-10-11 — Cline.** Added the red-wall/neutral-floor GPU harness with six non-vacuous checks
  and controls, measured linear-HDR pixels, and verified colour bleeding and feedback convergence
  on NVIDIA/Node and AMD/Chrome. The first run exposed the reflection-buffer header alignment bug;
  fixed it with scalar padding and pinned it headless. Corrected harness cube dimensions and floor
  rotation before accepting results. Node's large capture budgets exceeded the tool timeout;
  one probe per frame passes in about 21 seconds. Updated verification docs. No commits or push.
  Next: review/commit this GI work, then the agent kit; white-room energy/leak/reflection checks and
  ProbeVolume capture-batch teardown remain follow-ups.
- **2026-10-11 — Claude (Opus 5.5).** Planned Cycle C at the user's request: design only, no code.
  - **Read first:** Cline's handoff updates (Cycle B committed, GI check, agent kit), the design
    principles, the physics plugin specs, and the roadmaps.
  - **Verified against source:** the ONNX Runtime 1.29 behaviours Cycle C depends on — GPU
    inputs/outputs, pre-allocated outputs, graph capture binding once, `run()` awaiting before it
    submits, device setter/getter, input-shape metadata — and measured its WebGPU WASM size.
  - **Wrote `docs/agents-design.md`** in the §14 shape: a 64-byte agent record compatible with P1,
    P2's grid built once and shared, declared sensors, three brain backends behind one socket,
    a test plan with expected numbers, verbatim errors, touchpoints (one visibility addition,
    `setInstanceCount`), and a spike-gated build order.
  - **Changed only** that doc and this file. Five open questions await the user.
- **2026-10-11 — Claude (Opus 5.5).** Settled Cycle C's open questions with the user and updated
  `agents-design.md`: hybrid brains (ORT optional, the research path, never default); ORT behind a
  generic `null-graph/inference` adapter (revising Cycle A's "no ML module", reason recorded in
  Decisions); goals ladder (§9.7); spikes reordered agents-first (S3 before C1 step 3; ORT spikes
  before C2). Docs only. Next: C1 step 1 — record, layouts, CPU mirrors.
- **2026-10-11 — Claude (Opus 5.5).** C1 step 1 of Cycle C, headless only.
  - **Built** `src/extras/agents/`: the record layout, `defineAgents` (derived observation and action
    layouts, named WGSL offsets, slot names, species masks and table), `AgentRecords`, CPU mirrors
    (brute-force k-nearest, sensing, steering) and `FixedStep`.
  - **Tests:** `test/agents.test.mjs`, 24 checks. Six injected bugs were each caught by their test.
    A real gap was found and fixed (spawn speed above the agent's max).
  - **Docs:** design doc §5.1, §5.2 and §7 now state the exact semantics; A3's tolerance is derived
    from WGSL's trig accuracy.
  - **Committed** after the GI work (`1a70e8d`), as requested. Next: a CPU-preview demo in the
    test-engine site, then C1 step 2 — the grid.
- **2026-10-11 — Claude (Opus 5.5).** Committed the GI work (`1a70e8d`) and Cycle C plan + C1 step 1
  (`7f4cdb5`) separately.
  - **Built** the Herd & Wolves CPU-preview demo in the test-engine site under a new "Agent
    Experiments" section, and verified it in headless Chrome.
  - **Published** with `yalc:push`, which now reaches five local consumers; the dev `dist/` was
    restored afterwards.
  - **Next:** C1 step 2 — the GPU grid.
- **2026-10-11 — Claude (Opus 5.5).** Added `npm run drive` (`test/gpu/drive-page.mjs`), the
  page driver behind the demo checks, grown from `browser-run.mjs`. Documented in
  `docs/agent/validation.md`; the agent-kit test passes.
  - **Watched failing first:** an uncaught WebGPU error in the page and in a worker, an exception,
    a 404 page and a Git-Bash-mangled path each exit non-zero; a clean page, a 9-step site
    interaction and the text GPU harness page exit 0.
  - **Bugs found while building it,** each fixed:
    - A fixed debug port reached a previous run's closing browser — now port 0, read from Chrome's
      DevToolsActivePort file.
    - Git Bash rewrote `/x.html` into a Windows path — now refused by name.
    - The favicon 404 counted as a page error — now answered with 204.
    - Workers ran before the device watch existed — now a before-script instrumentation breakpoint.
  - **Plan:** the step 2 entry now includes `groundGrid()` and the grid-occupancy overlay.
  - **Pointer:** label decluttering, in the Cycle B known gaps.
- **2026-10-11 — Cline.** Committed the page driver, script wiring and associated documentation as
  `68ab721` (`feat(tooling): add interactive WebGPU page driver`). Syntax check, 4/4 agent-kit checks
  and the live text GPU harness driven through Chrome passed; artifacts are in the local temp
  directory `nullgraph-driver-check`. No push or publication.
  - **Started C1 step 2:** shared point-grid CPU reference and bounded nearest-query mirror;
    `test/agents-grid.test.mjs` joins `npm run test:agents`. New work remains uncommitted.
  - **Semantics:** closed maximum faces/outside positions clamp to edge cells; cells are X-major;
    offsets are exclusive with a terminal total; scatter retains all live entries; overflow is
    excess entries counted once per cell/build, not multiplied by query count. Exact query agreement
    is claimed only below the scan budget; overflowing GPU scatter will not have stable membership.
  - **Validated:** library build/declarations and IDE build, 24 existing + 5 new agent checks, 4 kit
    checks. Farthest-first and higher-index-tie mutations each failed the intended test; restoring
    the generated bundle returned 5/5 passing. Tracked core bundles are unchanged.
  - **Next:** implement GPU stages and hierarchical prefix scan against these definitions, add A1
    Node/Dawn + Chrome readback comparisons, then debug views and site overlay/driver screenshots.
