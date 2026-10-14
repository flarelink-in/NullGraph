# NullGraph Agents — C2 step 10: the live switch in the Herd & Wolves demo — Technical Design

**Parent:** [`agents-design.md`](./agents-design.md) §13 (C2) and [`work/plan.md`](./work/plan.md)
§3, step 10: "Demo: switch Rules / Network / ORT live with timings — the showable result".
**Builds on:**
- step 7 ([`agents-cloning-design.md`](./agents-cloning-design.md)): the two trained networks, and
  its decision 3: "the site's demo keeps its own copy [of the rules] until step 10", which "also
  moves the demo onto the scenario file";
- step 8 ([`agents-onnx-reader-design.md`](./agents-onnx-reader-design.md)): `mlpFromOnnx`, and
  "the demo loading `.onnx`: step 10";
- step 9 ([`agents-ort-design.md`](./agents-ort-design.md)): `OrtRuntime`, `OrtBrain`,
  `prepareBrain`, and its deferred item: "the site's engine-first worker and the live ORT switch".

**Asked by the user:** 2026-10-14, "lets do design for last step … meaning our herd and wolves
demo would be revisited for this one right at …/AgentsExperiments/HerdWolvesGPU".

**Where the work is:** mostly in the test-engine site
(`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`): the demo
(`src/data/engine-demos/demos/AgentsExperiments/HerdWolvesGPU/`), its controls, and the worker's
`EngineCore`. In this repository: the validation, and the `.onnx` files that
`npm run agents:export` writes. The site's repository has no commits yet; its changes stay
uncommitted unless the user says otherwise.

**Status:** decided with the user, 2026-10-14 (§12). Phases 10.0–10.5 follow (§13).

---

## 1. Purpose

**In the Herd & Wolves demo, each species switches live between three brains, and the screen
shows what each one costs, measured as it runs:**
- **Rules:** the WGSL rule, as today;
- **Network:** the trained 24→64→64→64→2 network on the built-in kernel (`MlpBrain`), from its
  `.mlp.json` file, as today;
- **ONNX Runtime:** the same network, exported as `.onnx`, run by ONNX Runtime (`OrtBrain`). The
  demo checks that the file holds exactly the `.mlp.json`'s weights.

The switch itself compiles nothing: every brain is prepared ahead (`prepareBrain`), so a switch
takes under a millisecond (A33: 0.11–0.65 ms). The numbers on screen are the think pass's GPU
time per brain, ONNX Runtime's run (GPU time, main-thread time, one frame of latency), the cost
of the last switch, and ONNX Runtime's start-up.

**Words used here:**
- **brain choice:** `rules`, `network` or `ort`, per species;
- **the ORT device:** the GPU device an `OrtRuntime` made, which the worker's engine adopts;
- **the device switch:** making the worker's engine again, on the ORT device (§4);
- **the site:** the test-engine site; **the worker:** its engine worker (`src/workers/`).

### The single most important pre-existing fact: the worker makes its own device

`EngineCore.init` makes the engine when the page's canvas starts, before any demo, on a device
NullGraph requests itself (fact 1). ONNX Runtime 1.29 runs only on a device it made (step 9's
spike S5: given another, its runs fail, and after a few it stops answering). **So the page cannot
run ONNX Runtime on the device it has.** It needs an engine on the ORT device, and when to make
one is decision D1 (§4).

### Non-goals

- **Other scenes,** and ONNX Runtime in the site's other demos.
- **Graph capture:** off, as step 9 decided (§14, item 6 there).
- **Training in the browser,** and networks for settings other than the demo's defaults
  (step 7's decision 6).
- **A watchdog for a hung run** (step 9's non-goal).
- **Browsers other than Chrome on Windows,** where the site is tested: WebGPU in a worker and
  ONNX Runtime elsewhere are not measured.
- **Committing the site's repository.** It has no commits; that is the user's call.

## 2. What the site does today (verified 2026-10-14)

| # | Fact | Source | Consequence |
|---|---|---|---|
| 1 | The worker makes one engine at `INIT`, before any demo: `new NullGraph()` and `init(canvas, { desiredFeatures: ['timestamp-query'], powerPreference: 'high-performance', samplableDepth: true })`. A `GPUProfiler` is made on its device. | site `src/workers/engine/EngineCore.ts:56-87` | ONNX Runtime cannot use this device (§1). |
| 2 | One worker per mounted canvas; demos load and unload inside it (`LOAD_DEMO` → `loadDemo` → `cleanupDemo`, then the demo's `setup`). | `src/sections/engine-view/EngineCanvas.tsx:81-146`; `EngineCore.ts:89-185` | An engine made again serves every later demo of the page. |
| 3 | The demo receives the engine once: `setupHerdWolvesGPU(engine, camera, getState, getInput)`. It builds a `Sim` per configuration (count, cadence, budget, seed, draw). | `HerdWolvesGPU/index.ts:247-252, 272-578` | The demo cannot change its engine; the worker must, and then load the demo again (§4). |
| 4 | Brains are `rules` or `network` per species. The network brain is made, and its pass compiled by `setBrain`, at the first switch. | `options.ts:21-24`; `index.ts:950-980` | A hitch at the first switch, which `prepareBrain` removes (§7). |
| 5 | The networks are `networks/<species>.mlp.json`, about 180 KB each, imported on the first switch. | `networks.ts:37-46` | Step 8 planned `.onnx` here; D2 keeps `.mlp.json` for Network (§5). |
| 6 | The overlay times the first step of a frame in four groups: grid, neighbours, "sense + think", "act + rules", with timestamp passes between them, read back asynchronously. | `index.ts:86-87, 177-183, 690-728`; `probe.ts` | Sense and think are timed together; an ORT run, between frames, is not timed (§8). |
| 7 | `update()` calls `agents.advance()` with no `try`. | `index.ts:1228` | A failed ORT run, thrown by `advance()`, would break the frame (§10). |
| 8 | The site's null-graph is a yalc copy from 2026-10-13: no `inference` entry, no `prepareBrain`. `onnxruntime-web` is not a dependency. | site `package.json:55`; `.yalc/null-graph/dist` | Both arrive in phase 10.1. |
| 9 | Vite builds the worker as an ES module; the site sends no cross-origin-isolation headers. | site `vite.config.ts` | ONNX Runtime runs with one thread, as in every step 9 test. |
| 10 | The site already vendors libraries into `public/vendor/` (`gl-matrix`, `null-graph`). | site `public/vendor` | ONNX Runtime's WASM can be served the same way (§6). |
| 11 | ONNX Runtime's device has `timestamp-query`, and the same storage limits as NullGraph's own (2 GiB, 16 storage buffers per stage). It lacks `bgra8unorm-storage` and `dual-source-blending`, which neither null-graph's sources nor the site use. | measured 2026-10-14 on Node/NVIDIA, Node/AMD and Chrome/AMD (a probe beside `test/spikes/ort/`) | An engine on the ORT device can run this demo, its overlay, and the page's other demos. |
| 12 | An adopted device refuses `powerPreference` and `autoNegotiate`. | `src/core/WebGPUContext.ts:325-333` | The device switch passes neither (§4). |
| 13 | The demo's rule is the site's own copy (`HERD_BRAIN`). The networks' teacher is `tools/agents/scenario/herd-wolves.mjs`; each `.mlp.json` carries the SHA-256 of its label WGSL (`trainedOn.teacher`); the `.onnx` files carry species and names, not that hash. | site `brain.ts:20-92`; `tools/agents/export.mjs:33-34`; `tools/agents/models/herd-wolves/*.mlp.json` | The demo cannot tell today whether its rules are the ones the networks copy (§5). |
| 14 | The page is `/dashboard/agents/herd-wolves-gpu`. | site `src/routes/paths.ts:165` | `npm run drive` can load it and drive the controls (§11). |
| 15 | The demo has a documentation page in the site. | site `HerdWolvesGPUDocumentation.ts` (111 lines) | It gains the third brain and the timings (phase 10.5). |

## 3. Architecture

```
 main thread                     worker: EngineCore                         demo: HerdWolvesGPU
 controls: Rules|Network|ORT ──► UPDATE_STATE ───────────────────────────► applyBrains (per species)
                                 first ORT pick (D1):    ◄── needs ORT ──┘
                                   import onnxruntime-web/webgpu, null-graph/inference (on demand)
                                   OrtRuntime.create(ort) → engine made again on runtime.device
                                   the demo loaded again, its settings kept ──► setup(…, { runtime })
 <species>.mlp.json (D2) ────────────────────────────────────────────────► MlpBrain (Network)
 <species>.onnx ─── mlpFromOnnx: the same weights as the .mlp.json? ───────► runtime.load → OrtBrain
```

- **Per species, three brains,** all over the same system: the rule brain (as today); an
  `MlpBrain` from the `.mlp.json`; an `OrtBrain` over `runtime.load(bytes)` of the `.onnx`, whose
  weights the demo has checked against the `.mlp.json`. Each is prepared ahead, and `setBrain`
  moves the species between them.
- **ONNX Runtime is loaded only when it is asked for:** its JavaScript (58 KB gzipped) as a dynamic
  import in the worker, its WASM (6.4 MB gzipped) by ONNX Runtime itself. Visitors who never pick
  it download nothing more than today.
- **The runtime lives as long as the worker,** and is destroyed with it: every later demo of the
  page runs on its device.

## 4. Decision D1: the ORT device

| Option | What happens | Cost |
|---|---|---|
| **A. Made on the first ORT pick** (recommended) | The first time a species is set to ORT, the worker starts ONNX Runtime, makes its engine again on the ORT device, destroys the old device, and loads the demo again with its settings kept. A loading overlay covers it. | One pause the first time, about the start-up step 9 measured (0.8–0.9 s in Chrome, more while the 6.4 MB download is not cached), and the scene starts over: catches and positions reset. A change to the site's `EngineCore`. |
| B. ORT first on this page | When the page opens on Herd & Wolves, the worker makes the ORT device before its engine (the `INIT` message says so). | Every visit to the demo pays ONNX Runtime's download and start-up, even with rules and networks only; a visitor who arrives from another demo still needs A's path. |
| C. ORT first always | The worker makes every page's engine on the ORT device. | Every demo of the site pays ONNX Runtime's start-up and download. |

With A, the device switch is:
1. `const ort = await import('onnxruntime-web/webgpu')`, with the WASM's location set (§6);
2. `runtime = await OrtRuntime.create(ort)`;
3. the active demo's `destroy()`, as `cleanupDemo` does today;
4. `engine = new NullGraph()`, `await engine.init(canvas, { device: runtime.device, samplableDepth: true, width, height })`:
   no `powerPreference` (fact 12); the same `OffscreenCanvas`, whose context is configured again
   with the new device (phase 10.0 verifies it);
5. the profiler made again on the new device; the old device destroyed;
6. the demo loaded again; it is told the runtime exists, and sets ORT where it was asked.

A failure at step 1 or 2 keeps the old engine and device: the demo reports why, and its ORT
choice stays off (§10).

## 5. The networks and the rules (D2, D3)

**D2, the networks' files. Decided: B, `.mlp.json` kept for Network.**

| Option | What | Cost |
|---|---|---|
| A. One `.onnx` per species, for both | `npm run agents:export`'s files (41 KB each) replace the `.mlp.json` copies. `mlpFromOnnx` reads them for Network; ONNX Runtime runs the same bytes for ORT. | Step 8's plan ("the demo loading `.onnx`"). 82 KB for both, against 360 KB of JSON today. |
| **B. Keep `.mlp.json` for Network, add `.onnx` for ORT** (taken) | Network loads `.mlp.json` as today; ORT loads `npm run agents:export`'s `.onnx` files | Two files per species to keep in step |

As built under B:
- **Network** loads `networks/<species>.mlp.json`, as today.
- **ORT** loads `networks/<species>.onnx`. When ONNX Runtime is ready, the demo reads each one with
  `mlpFromOnnx` (from `null-graph/inference`, loaded with ONNX Runtime) and compares it with the
  `.mlp.json`'s network, bit for bit: the reader gives the weights back exactly (step 8). A file
  that differs is refused for ORT ("`prey.onnx` is not `prey.mlp.json`'s network: export it
  again"). So the two brains are known to run one network, and the files cannot drift apart.
- Step 8's note, "the demo loading `.onnx`: step 10", holds for ORT only.

**D3, the rules' source** (step 7's decision 3: "moves the demo onto the scenario file").
**Decided: A, a checked copy.**

| Option | What | Cost |
|---|---|---|
| **A. A checked copy** (taken) | The site's rule becomes a copy of the scenario file's texts (`herdRule` with its four seams, catches, respawn). The demo hashes its own label text at load and compares it with the teacher's hash that each `.mlp.json` already carries (`trainedOn.teacher`): when they differ, the overlay says the rules have moved on from what the networks copied. | A copy still, but one that says when it drifts. No change to the exporter: under D2's B, the hash is in the `.mlp.json` files the demo loads. |
| B. The package ships the scenario | Herd & Wolves' rules become part of `null-graph/agents`, or of a new entry | Demo content in the library; its size and its API kept for good |
| C. A generator | A script here writes the site's rule module from the scenario file | Tooling that writes into another repository |

## 6. ONNX Runtime in the site (D4, D5)

- **The version (D4). Decided: pinned at 1.29.0,** the version every step 9 suite ran.
  `^1.29.0`, the package's peer range, would install 1.30.0, on which only the `inference` suite
  ran, under Node, on an NVIDIA GPU.
- **Where the WASM comes from (D5). Decided: A, vendored.**

  | Option | What | Cost |
  |---|---|---|
  | **A. Vendored, same origin** (taken) | `ort-wasm-simd-threaded.asyncify.mjs` and `.wasm` copied into `public/vendor/onnxruntime-web/`, as the site does for `gl-matrix` and null-graph; `ort.env.wasm.wasmPaths` points there | 25.7 MB in the site's files; the copy refreshed with the version |
  | B. A CDN | `wasmPaths` at jsDelivr or unpkg | A third party in the page's path; nothing vendored |

- **The JavaScript** comes through Vite: `import('onnxruntime-web/webgpu')` in the worker, a chunk
  of its own, loaded on the first ORT pick.
- **One thread:** `ort.env.wasm.numThreads = 1` (fact 9).

## 7. Switching live

- **Network brains are prepared on the first Network pick,** in the background: the two
  `.mlp.json` files load as today (360 KB, only for visitors who ask), become `MlpBrain`s, and are
  prepared with `prepareBrain`. The species keeps its brain until its pass is compiled (Chrome
  compiled it in 255 ms in A33, with frames of 20 ms at most), then switches. Every later switch
  compiles nothing.
- **ORT brains are prepared when ONNX Runtime is ready:** each `.onnx` is checked against its
  `.mlp.json` (§5), loaded with `runtime.load(bytes)`, made an `OrtBrain`, prepared, and run once
  to warm up, which compiles ONNX Runtime's pipelines (0.14–0.46 s in step 9) before the species
  needs it. Without the warm-up, a species switched to ORT acts on its last actions for that long.
- **A switch** is `setBrain(brain, { species })`: under 1 ms. The overlay shows the last switch's
  time. A switch to ORT also copies the species' actions into the brain's results, so it keeps
  moving as it did until the first run lands.
- **A rebuild** (count, cadence, budget, seed or drawing changes) makes a new system: its brains
  are prepared again, for that system's buffers.
- **The inspector** reads "decided by its network" for Network and "decided by ONNX Runtime" for
  ORT: both write reason 0.

## 8. Timings: what is measured, and how

| On screen | Measured as | Notes |
|---|---|---|
| Each group's GPU time per step: grid, neighbours, sense, think, act + rules | The overlay's timestamp passes between groups of the first step (fact 6), with sense and think in groups of their own | Think is every brain's pass, both species together; the overlay names who decides for each |
| ONNX Runtime's run, GPU | A timestamp pass after the frame's work and another when the run resolves: the run is queued between them | Includes any time the GPU waits for ONNX Runtime's submission; phase 10.0 measures that against step 9's bracket |
| ONNX Runtime's run, main thread | From `run()` to its resolution, as A6 measured | 0.6–0.8 ms in step 9 |
| Latency | Stated: "acts on the frame before" (A6: 299 of 300 frames in Chrome) | Not measured live: it needs a marker the real network cannot carry |
| The last switch | `performance.now()` around `setBrain` | Under 1 ms when prepared |
| ONNX Runtime's start-up | Its import, `OrtRuntime.create`, each model's load and warm-up run, shown once | The device switch's pause, broken down |
| Runs per second | Runs started per wall second | With `thinkEvery: 3` at 60 steps a second, 20 |

The page's performance widget keeps showing the frame's CPU and GPU time. An ORT run lands
between two frames' measurements, so the widget's GPU time does not include it; the overlay's
line does.

## 9. The stress test with ORT (D6)

The demo builds 4K to 1M agents (fact 3). ONNX Runtime's cost grows with the count: step 9
measured 6.6–6.8 ms of GPU time per run at 100,000 agents on the AMD iGPU, for a smaller network.
Memory grows too: the prey network's hidden layers are 64 wide, so each layer's activations for
1M agents are 256 MB (an estimate; ONNX Runtime's own allocations are not known here).

| Option | What |
|---|---|
| **A. A cap, measured** (recommended) | Phase 10.0 measures ONNX Runtime's run time and memory at each count. ORT is offered up to the largest count that ran within the frame budget, and the overlay says why above it. |
| B. No cap | The demo's guard watches agent steps only; an ORT run outside them is not guarded. |

## 10. Failure and teardown

- **ONNX Runtime does not load** (no WebGPU in the worker, the WASM not found, a device not made):
  the old engine stays, the ORT choice is refused, and the overlay says why.
- **A run fails:** `advance()` throws once, naming the species. The demo catches it, gives the
  species the rules, and says so.
- **A model is refused** (`OrtBrain`, `setBrain`): the same as a failed load, for that species.
- **The demo is destroyed** (another demo, a rebuild): the system first, then each brain, then each
  model. The runtime stays with the worker.
- **The worker is torn down** (the canvas unmounts): the demo, then `await runtime.destroy()`.

## 11. Validation

The site has no test suite of its own. Each phase is checked in headless Chrome with this
repository's `npm run drive` against the site's dev server, on the AMD iGPU (Chrome on Windows
takes it):
- **every run:** no WebGPU error and no uncaught error, in the page or the worker;
- **scripted switches:** `eval` clicks the controls' buttons, and the demo logs one line per
  switch (species, brain, milliseconds, prepared or not), which `waitFor` checks;
- **the device switch:** the first ORT pick logs the steps of §4 and their times, and the demo runs
  on afterwards;
- **screenshots** of the overlay at each brain, read by eye: what a check cannot judge;
- **the numbers** go into this document's results, as in every step.

In this repository nothing changes: the agents suites keep checking the committed networks and the
teacher's hash they carry (step 7). The demo's own checks, the `.onnx` against the `.mlp.json` and
its rules against the teacher, are watched failing in the site with a changed file and a changed
rule.

## 12. Decisions (the user, 2026-10-14)

1. **D1, the ORT device: A, made on the first ORT pick** (§4).
2. **D2, the networks' files: B, `.mlp.json` kept for Network;** ORT loads `.onnx`, checked
   against the `.mlp.json` bit for bit (§5). The recommendation was A.
3. **D3, the rules' source: A, a copy checked by the teacher's hash,** read from the `.mlp.json`
   files (§5).
4. **D4, ONNX Runtime's version in the site: 1.29.0, pinned** (§6).
5. **D5, the WASM: A, vendored in the site** (§6).
6. **D6, ORT at large counts: A, a measured cap** (§9).

**Nothing is open.**

## 13. Build order

| Phase | Builds | Checked by |
|---|---|---|
| 10.0 | Spikes, no demo change: ONNX Runtime in the site's worker through Vite; the device switch over one `OffscreenCanvas` (D1); ONNX Runtime's time and memory at 4K–1M agents (D6); the run bracket's bias (§8) | Measurements, recorded here |
| 10.1 | Plumbing: null-graph pushed to the site; `onnxruntime-web` 1.29.0 installed (D4) and its WASM vendored (D5); the two `.onnx` files exported and placed beside the `.mlp.json` (D2) | The site builds; the page loads as today |
| 10.2 | Network prepared on its first pick (no hitch); the rules as a copy of the scenario's texts, checked by the teacher's hash (D3); the switch line on the overlay | Drive: switches compile nothing; a changed rule is reported |
| 10.3 | ORT: the device switch (D1), the third choice in the controls, the ORT brains prepared and warmed up, failures handled | Drive: the device switch, switches to ORT and back, a refused model |
| 10.4 | Timings: sense and think timed apart, ONNX Runtime's run, start-up, the overlay's lines | Drive and screenshots; numbers recorded |
| 10.5 | The stress test's cap (D6), the site's documentation page, the guides' pointer to the demo | Drive at each count; step 10 closed, and with it C2 |

Each phase leaves the demo running. Every check is watched failing first, where a check can fail.

### Deferred

- **Other browsers,** and Chrome on an NVIDIA GPU (Chrome on Windows takes the iGPU).
- **Live latency** with a marker network (A6's method), in the demo itself.
- **ONNX Runtime in the site's other demos.**
