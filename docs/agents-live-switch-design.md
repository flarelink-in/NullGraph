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

**Status:** decided with the user, 2026-10-14 (§12). Phase 10.0 measured on 2026-10-14 (results at
the end). After it the user revised two decisions (§12): D5, Vite serves ONNX Runtime's own files;
D6, no cap on the agents ORT runs over. Phases 10.1 and 10.2 built on 2026-10-14, and 10.3 on
2026-10-15 (results at the end); phases 10.4 and 10.5 follow (§13).

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
| 10 | The site already vendors libraries into `public/vendor/` (`gl-matrix`, `null-graph`). | site `public/vendor` | D5 first served ONNX Runtime's WASM the same way; phase 10.0 found that Vite's dev server refuses that, and D5 changed (§6). |
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
1. `const ort = await import('onnxruntime-web/webgpu')`, with one thread; no WASM location is set,
   since Vite serves ONNX Runtime's files (§6);
2. `runtime = await OrtRuntime.create(ort)`;
3. the frames stopped, and the active demo's `destroy()`, as `cleanupDemo` does today;
4. `engine = new NullGraph()`, `await engine.init(canvas, { device: runtime.device, samplableDepth: true, width, height })`:
   no `powerPreference` (fact 12); the same `OffscreenCanvas`, whose context is configured again
   with the new device (phase 10.0 verified it);
5. the profiler made again on the new device; the old device destroyed once its queue is done
   (`onSubmittedWorkDone`, then one task), so that the old profiler's last read lands (phase 10.0);
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
- **Where the WASM comes from (D5). Decided: C, Vite serves ONNX Runtime's own files.** The user
  first took A; phase 10.0 found that A fails in Vite's dev server, and the user took C (§12).

  | Option | What | Cost |
  |---|---|---|
  | A. Vendored, same origin (first taken) | `ort-wasm-simd-threaded.asyncify.mjs` and `.wasm` copied into `public/vendor/onnxruntime-web/`, as the site does for `gl-matrix` and null-graph; `ort.env.wasm.wasmPaths` points there | 25.7 MB in the site's files; the copy refreshed with the version. **Fails in the dev server** (phase 10.0): Vite refuses ONNX Runtime's import of a loader from `public/` |
  | B. A CDN | `wasmPaths` at jsDelivr or unpkg | A third party in the page's path; nothing vendored |
  | **C. Vite serves ONNX Runtime's own files** (taken) | `onnxruntime-web` in `optimizeDeps.exclude`, and no `wasmPaths`. The dev server serves the package from `node_modules`, with its `.wasm` beside it; a build emits the `.wasm` once, as a hashed asset of the same origin | Nothing vendored or refreshed by hand |

- **The JavaScript** comes through Vite: `import('onnxruntime-web/webgpu')` in the worker, a chunk
  of its own, loaded on the first ORT pick. Under C, the dev server does not pre-bundle it.
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
- **ONNX Runtime takes one call at a time** (found in phase 10.3). Since 2026-10-15 `OrtRuntime`
  keeps every load, run and release of its models in line ([`agents-ort-design.md`](./agents-ort-design.md),
  "After step 9"), so the prey's and the wolves' calls never overlap; the demo's own queue of 10.3
  is gone. The two species' runs take turns.
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
| A. A cap, measured (recommended; first taken) | Phase 10.0 measures ONNX Runtime's run time and memory at each count. ORT is offered up to the largest count that ran within the frame budget, and the overlay says why above it. |
| **B. No cap** (taken) | The demo's guard watches agent steps only; an ORT run outside them is not guarded. |

**Decided: B, no cap.** The user first took A, whose number phase 10.0 put at 65,536 agents. The
user then took B: anyone can view the site, and visitors with stronger GPUs can test larger counts.
ORT is offered at every count the demo builds, 4K to 1M, and the overlay shows what each run costs.

**The timeout guard still covers ORT's run.** This is the rule for every demo with a size control,
not part of B as offered. Windows resets a GPU that one submission keeps busy for about 2 s. The
page's device is then lost, and here that is the ORT device, which the engine shares (§4).
- ONNX Runtime's run cannot be split into smaller submissions. So the guard predicts the run's time
  at a count from the runs it measured (§8).
- Where a run would exceed the demo's step limit (`STEP_LIMIT_MS`, 600 ms), the species stays on
  Network, with the reason on screen.
- None of phase 10.0's runs came near it: 306 ms at 1M on the AMD iGPU.
- Phase 10.5 builds the guard.

## 10. Failure and teardown

- **ONNX Runtime does not load** (no WebGPU in the worker, the WASM not found, a device not made):
  the old engine stays, the ORT choice is refused, and the overlay says why.
- **A run fails:** `advance()` throws once, naming the species. The demo catches it, gives the
  species the rules, and says so.
- **A model is refused** (`OrtBrain`, `setBrain`): the same as a failed load, for that species.
- **The demo is destroyed** (another demo, a rebuild): the system first, then each brain, then each
  model. The runtime stays with the worker.
- **The worker is torn down** (the canvas unmounts): the page terminates the worker, which takes
  ONNX Runtime and its device with it (phase 10.3). `EngineCore.destroy()` would destroy the demo,
  then `await runtime.destroy()`, but nothing calls it.

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

In this repository the suites do not change: the agents suites keep checking the committed networks
and the teacher's hash they carry (step 7). The demo's own checks, the `.onnx` against the
`.mlp.json` and its rules against the teacher, are watched failing in the site with a changed file
and a changed rule. Since phase 10.2, `test/spikes/live-switch/check-site.mjs` checks the site's
copies from here: its rule, networks and `.onnx` files.

## 12. Decisions (the user, 2026-10-14)

1. **D1, the ORT device: A, made on the first ORT pick** (§4).
2. **D2, the networks' files: B, `.mlp.json` kept for Network;** ORT loads `.onnx`, checked
   against the `.mlp.json` bit for bit (§5). The recommendation was A.
3. **D3, the rules' source: A, a copy checked by the teacher's hash,** read from the `.mlp.json`
   files (§5).
4. **D4, ONNX Runtime's version in the site: 1.29.0, pinned** (§6).
5. **D5, the WASM: A, vendored in the site** (§6). **Revised after phase 10.0: C.**
6. **D6, ORT at large counts: A, a measured cap** (§9). **Revised after phase 10.0: B.**

**Revised after phase 10.0** (its results, "What phase 10.0 changes in the plan"; the user,
2026-10-14):
- **D5: C, Vite serves ONNX Runtime's own files,** the recommendation. With the `.wasm` vendored
  as well, a build would hold it twice, and the vendored copy would be one more thing to refresh by
  hand. `onnxruntime-web` goes in `optimizeDeps.exclude`, with no `wasmPaths` (§6).
- **D6: B, no cap.** Anyone can view the site, and visitors with stronger GPUs can test larger
  counts. The timeout guard still covers ORT's run (§9).

## 13. Build order

| Phase | Builds | Checked by |
|---|---|---|
| 10.0 ✅ | Spikes, no demo change: ONNX Runtime in the site's worker through Vite; the device switch over one `OffscreenCanvas` (D1); ONNX Runtime's time and memory at 4K–1M agents (D6); the run bracket's bias (§8) | Measurements, recorded here |
| 10.1 ✅ | Plumbing: null-graph pushed to the site; `onnxruntime-web` 1.29.0 installed (D4), and left out of Vite's pre-bundling, so that Vite serves its files (D5); the two `.onnx` files exported and placed beside the `.mlp.json` (D2) | The site builds; the page loads as today |
| 10.2 ✅ | Network prepared on its first pick (no hitch); the rules as a copy of the scenario's texts, checked by the teacher's hash (D3); the switch line on the overlay | Drive: switches compile nothing; a changed rule is reported |
| 10.3 ✅ | ORT: the device switch (D1), the third choice in the controls, the ORT brains prepared and warmed up, failures handled | Drive: the device switch, switches to ORT and back, a refused model |
| 10.4 | Timings: sense and think timed apart, ONNX Runtime's run, start-up, the overlay's lines | Drive and screenshots; numbers recorded |
| 10.5 | The stress test with ORT at every count (D6), under the timeout guard (§9); the site's documentation page; the guides' pointer to the demo | Drive at each count; step 10 closed, and with it C2 |

Each phase leaves the demo running. Every check is watched failing first, where a check can fail.

### Deferred

- **Other browsers,** and Chrome on an NVIDIA GPU (Chrome on Windows takes the iGPU).
- **Live latency** with a marker network (A6's method), in the demo itself.
- **ONNX Runtime in the site's other demos.**

## Phase 10.0 results — 2026-10-14: ONNX Runtime in a worker through the site's Vite, the device switch, and ONNX Runtime at the demo's counts

**Where:**
- `test/spikes/live-switch/` (`node test/spikes/live-switch/run.mjs --mode dev|build --wasm …`): a
  page and a module worker, served and built by the site's own Vite (5.4.21, from its
  `node_modules`) with the settings of the site's `vite.config.ts` that touch a worker
  (`worker.format: 'es'`, `optimizeDeps.exclude: ['null-graph']`). null-graph comes from this
  repository's `dist/`, which phase 10.1 pushes to the site, and `onnxruntime-web` 1.29.0 from this
  repository (D4). The vendored files go in a scratch `public/` folder. Driven by `npm run drive`
  in headless Chrome. **Nothing in the site changed.**
- `test/spikes/ort/counts.js` (`npm run spike:ort-counts`; in Chrome, `npm run drive -- --serve .
  "test/spikes/ort/index.html?spike=counts" --wait-for "counts done" --timeout 1200`): ONNX
  Runtime's time and memory at the demo's counts, and the overlay's timing against step 9's bracket.

**Hosts:** Chrome on the AMD GCN-5 iGPU, the demo's host here (Chrome on Windows takes the iGPU).
For the counts, also Node/Dawn on NVIDIA Turing and on the AMD iGPU.
**Networks:** the committed prey and wolf networks (24→64→64→64→2, tanh), written as `.onnx` by
step 7's writer and loaded through `OrtRuntime`. Each is bound over every agent, as `OrtBrain` binds
it.

### A. ONNX Runtime in a worker, through the site's Vite

| How ONNX Runtime finds its WASM | Dev server | Production build (`vite preview`) |
|---|---|---|
| **D5 as written:** `wasmPaths: '/vendor/onnxruntime-web/'`. ONNX Runtime imports its loader (`ort-wasm-simd-threaded.asyncify.mjs`) and fetches the `.wasm` from there | **Fails.** Vite adds `?import` to ONNX Runtime's import of the loader, then refuses a file of `public/` "imported from source code". ONNX Runtime reports "no available backend found". The same with ONNX Runtime left out of `optimizeDeps` | Not run with the default bundle; ONNX Runtime's other build (last row) runs this way |
| **The `.wasm` alone vendored:** `wasmPaths: { wasm: '/vendor/onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm' }`. The loader built into ONNX Runtime's bundle is used | Runs | Runs. The build also holds Vite's copy of the `.wasm` (below), which nothing fetches |
| **Nothing set,** ONNX Runtime pre-bundled by Vite (the site's `optimizeDeps` today) | **Fails.** ONNX Runtime fetches its `.wasm` beside the pre-bundled file, in Vite's cache, and Vite answers with its HTML page ("expected magic word 00 61 73 6d, found 3c 21 44 4f") | Runs, from Vite's copy |
| **Nothing set,** `onnxruntime-web` added to `optimizeDeps.exclude` | Runs: ONNX Runtime is served from `node_modules`, with its `.wasm` beside it | Runs, from Vite's copy |
| ONNX Runtime's other build (condition `onnxruntime-web-use-extern-wasm`: `ort.webgpu.min.mjs`, no loader built in), with the prefix | Fails, as in the first row: it imports the loader | Runs. No copy emitted; the loader and the `.wasm` are fetched from `/vendor/` |

- **Every production build of the default bundle emits the `.wasm` itself.** The bundle names it
  with `new URL('ort-wasm-simd-threaded.asyncify.wasm', import.meta.url)`. Vite turns that into a
  hashed asset, `assets/ort-wasm-simd-threaded.asyncify-<hash>.wasm`, of 25,749,873 B (6,368,495 B
  gzipped), whatever `wasmPaths` says at run time. With a vendored copy as well, the build holds the
  file twice.
- **What the worker fetched,** in the build:
  - before the first ORT pick, nothing more than without ONNX Runtime;
  - at the first ORT pick: ONNX Runtime's chunk (116,633 B, 38,788 B gzipped), the chunk of
    `null-graph/inference` (4,639 B, 2,115 B gzipped) and the `.wasm` (25.7 MB).
- **In the dev server,** the pre-bundled ONNX Runtime is 163 KB, and the page did not reload when
  Vite pre-bundled it.

### B. The device switch over one OffscreenCanvas (D1)

The worker made its engine as `EngineCore.init` does: its own device, `timestamp-query`,
`powerPreference`, samplable depth. It drew a ground grid every frame, timed by a `GPUProfiler`.
Then it did §4's switch:
1. it imported ONNX Runtime and started `OrtRuntime`;
2. it stopped the frames and destroyed the scene;
3. it made the engine again with `{ device: runtime.device }` over the same `OffscreenCanvas`;
4. it made the profiler again, destroyed the old device, and made the scene again with a coarser
   grid.

| ms, a cold page from localhost | Dev server | Build |
|---|---|---|
| `import('onnxruntime-web/webgpu')` | 122–172 | 19–31 |
| `import('null-graph/inference')` | 13–22 | 8–11 |
| `OrtRuntime.create`: the WASM fetched and compiled, the device made | 543–627 | 653–896 |
| The engine made again on ONNX Runtime's device | 0.4–0.6 | 0.4 |
| The profiler made again, the old device destroyed | 0.1–0.3 | 0.1–0.2 |
| **The switch, in all** | **692–803** | **692–939** |

Afterwards, on the worker's thread, per model: load 18–24 ms, first run 141–160 ms (ONNX Runtime
compiles its pipelines), each later run 0.8–1.1 ms.

- **It works:**
  - the same `OffscreenCanvas` presents the new engine's frames: screenshots before and after the
    switch, and after a resize to 640 × 360, as `MessageRouter` resizes;
  - the profiler times frames on ONNX Runtime's device;
  - ONNX Runtime ran the prey network over a buffer the new engine made, within 1.8e-6 of
    `mlpForward`;
  - no WebGPU error, in the page or the worker.
- **One trap: the profiler's last read.** `GPUProfiler.end` reads its timestamps with
  `mapAsync(…).then(…)` and no `catch`. The first run destroyed the old device while the last
  frame's read was in flight, and the read rejected: an uncaught `AbortError` in the worker. Every
  later run first waited for the old device's queue (`onSubmittedWorkDone`, then one task: 0.5 ms),
  and the error was gone. §4's step 5 needs that wait.

### C. ONNX Runtime at the demo's counts (D6)

GPU time of one run (µs, step 9's bracket: the runs queued behind a long WGSL pass), and the
main-thread time of `run()` in Chrome:

| Agents | Node · NVIDIA | Node · AMD | Chrome · AMD | Chrome, main thread |
|---|---|---|---|---|
| 4,096 | 422 | 1,085 | 991 | 0.91 ms |
| 16,384 | 1,536 | 3,764 | 3,271 | 0.54 ms |
| 65,536 | 6,001 | 14,068 | 12,648 | 0.85 ms |
| 262,144 | 23,894 | 55,214 | 51,366 | 1.20 ms |
| 1,048,576 | 101,712 | 308,740 | 305,972 | 145.5 ms |

- **Both species on ORT** are two runs, each over every agent (`OrtBrain` binds the whole system):
  twice these figures.
- **First runs.** ONNX Runtime compiles its pipelines on a process's first run: 428–439 ms under
  Node and 157 ms in Chrome, at 4,096 agents. A first run at a larger count then took, in Chrome,
  9 / 41 / 173 / 862 ms at 16K / 64K / 256K / 1M (the first model).
- **Memory.** ONNX Runtime's GPU buffers for both models:
  - 0.2 MB of weights;
  - activations of 2.0 / 8.1 / 32.5 / 130 MB at 4K / 16K / 64K / 256K, made in the first runs and
    reused by every run after. They are two buffers of N × 64 floats, which the second model reused
    as well;
  - none destroyed when the models were released. The spike cannot tell whether ONNX Runtime keeps
    them for later or leaves them to the garbage collector;
  - **at 1M,** 1,032 MB in the first runs. After that, every run creates and destroys two 256 MiB
    buffers: 29 GB created and freed over the timed runs. That is the main-thread time of a run
    there: 20 ms (NVIDIA) to 146 ms (Chrome);
  - ONNX Runtime's WASM memory stayed at 16.0 MB.

  The system's own buffers add N × 96 B of observations and N × 8 B per results buffer.
- **Every run matched `mlpForward`** within 2.0e-6 on its first 256 rows, on all three pairs.

### D. The overlay's timing against the bracket (§8)

The overlay's way: the run started, the frame's work submitted, a mark; a second mark when `run()`
resolves. The span between the marks minus the bracket, in µs, for 0 / 2 / 8 ms of the frame's work
ahead of the run:

| Agents | Node · NVIDIA | Node · AMD | Chrome · AMD |
|---|---|---|---|
| 4,096 | +627 / +37 / +37 | +291 / −168 / −102 | +452 / −61 / −76 |
| 16,384 | +627 / −29 / +37 | +496 / −160 / −291 | +468 / −80 / −140 |
| 65,536 | +684 / −37 / +29 | −568 / −1,223 / −634 | −157 / −649 / −72 |
| 262,144 | +878 / +26 / +26 | +426 / −950 / −1,409 | +4,748 / +2,528 / +4,911 |
| 1,048,576 | +20,120 / +17,760 / +11,862 | +33,817 / +35,783 / +19,268 | +32,015 / +24,241 / +12,736 |

- **With frame work ahead of the run,** the two marks read within 0.3 ms of the bracket up to 16K.
  Up to 256K they read within 1.4 ms, at most 10%, except Chrome at 256K: 2.5 to 4.9 ms high, 5 to
  10%. The demo's case is the 2–8 ms columns. A frame that starts a run has just submitted at least
  one step (§8.2 of step 9: no step, no run), and a step at 4,096 agents costs about 2.5 ms on this
  iGPU.
- **With no frame work,** the marks read 0.3–0.6 ms high at 4K and 16K on every host. That is the
  time `run()` takes before ONNX Runtime submits, while the GPU idles.
- **At 1M,** both columns read high by the main-thread time of ONNX Runtime's buffer churn, while
  the GPU idles.
- So the overlay can show the two marks if it says what they include: "from the end of the frame's
  work to the end of the run".

### What phase 10.0 changes in the plan

1. **D5 does not work as written.** In the dev server, ONNX Runtime cannot import a loader from
   `public/`. Two ways work in both modes; the user decides (§12):
   - **Recommended: let Vite serve ONNX Runtime's own files.** `onnxruntime-web` goes in
     `optimizeDeps.exclude`, and `wasmPaths` is not set. The dev server serves it from
     `node_modules`, and the build emits one hashed `.wasm` from the same origin. Nothing is
     vendored or refreshed by hand.
   - **D5 with the `.wasm` alone:** in `public/vendor/onnxruntime-web/`, with `wasmPaths: { wasm }`.
     The build then holds two 25.7 MB copies, and one is never fetched.

   **Decided: the recommended way** (§6, §12).
2. **§4, step 5:** wait for the old device's queue before destroying it (the profiler's read). Now
   in §4.
3. **D6's number.** §9's rule, "the largest count that ran within the frame budget", taken as the
   demo's 40 ms of GPU work per frame (`SUBMIT_BUDGET_MS`), gives **65,536 agents**. There, both
   species' runs take 25 ms in Chrome on the AMD iGPU; at 262,144 one run takes 51 ms. On NVIDIA
   under Node the rule would allow 262,144 for one species. The user confirms the number.
   **Decided: no cap** (§9, §12).
4. **Phase 10.1's plumbing.** The site's install is not what its `package.json` says
   (`docs/work/gotchas.md`): push null-graph with yalc before any `npm install` there. **Done by
   the user on 2026-10-14:** `npm run yalc:push`, then the site's `npm run dev -- --force`. The site's
   `.yalc` and `node_modules` copies of `dist/` then matched this repository's build, file for file.

**Not run:**
- Chrome on NVIDIA, and any other browser;
- the default bundle with the prefix in a build;
- ONNX Runtime 1.30.0;
- the site's own worker and demo, which are phase 10.3's.

## Phase 10.1 results — 2026-10-14: the plumbing

**What changed in the site.** The site has no commits, so its changes are listed here, with what each
file held before.

| File | Now | Before |
|---|---|---|
| `package.json` | `"onnxruntime-web": "1.29.0"` among the dependencies, exact (D4) | No ONNX Runtime |
| `package-lock.json` | Rewritten by npm to match `package.json`: null-graph is `file:.yalc/null-graph`, a link. Added: `onnxruntime-web` and `onnxruntime-common` 1.29.0, `flatbuffers` 25.9.23, `guid-typescript` 1.0.9, `platform` 1.3.6 | null-graph named as the registry's 1.0.5 |
| `node_modules/null-graph` | A copy again, which yalc made (below). Between the install and that update it was a junction to `.yalc/null-graph`: npm 11 links a `file:` dependency to a folder | A copy, which yalc made |
| `.yalc/null-graph`, `yalc.lock` | This repository's build at `a8d0107`, with the profiler's fix (signature `5fe8997d…`) | The user's push of 2026-10-14 (`451a1eca…`) |
| `vite.config.ts` | `optimizeDeps.exclude: ['null-graph', 'onnxruntime-web']`, with a comment saying why (D5) | `['null-graph']` |
| `HerdWolvesGPU/networks/prey.onnx`, `wolf.onnx` | New, 41,186 B each (D2) | — |

- **How:**
  - `npm install onnxruntime-web@1.29.0 --save-exact` (npm 11.6.2). A dry run first listed those
    five packages and null-graph, and nothing else.
  - `npm run agents:export -- --students <the demo's networks/> --out <the same folder>`.
- **No source file of the site changed.**
- **null-graph in the site.** The user pushed it with `npm run yalc:push` on 2026-10-14; its library
  was that of `9320373`. To bring the profiler's fix (`a8d0107`), the site alone was then updated:
  `yalc publish` here and `yalc update null-graph` there. The four other projects that `yalc:push`
  reaches kept their copy (`docs/work/gotchas.md`).
- **The update turned the junction back into a copy:** yalc deletes a link at
  `node_modules/null-graph` (the link, not its target) and copies the package there. Both shapes
  ran the demo, and the next `npm install` makes the junction again.

**Checks:**
- **The `.onnx` files hold the `.mlp.json` networks,** as 10.3's check will read them.
  `mlpFromOnnx` reads each `.onnx` back as its `.mlp.json`'s layers, bit for bit. The files are
  also byte for byte an export of the committed networks, and the site's `.mlp.json` files equal
  those (SHA-256). Checked by `test/spikes/live-switch/check-site.mjs` (written in 10.2).
- **The site builds:** `tsc` passes (35 s), and `vite build` passes into a scratch folder, so the
  site's `dist/` was left alone. The build holds no ONNX Runtime: nothing imports it yet.
- **The dev server serves ONNX Runtime's files from `node_modules`:**
  `ort.webgpu.bundle.min.mjs` (799,159 B) and `ort-wasm-simd-threaded.asyncify.wasm`
  (25,749,873 B, `application/wasm`). That is the user's dev server, which Vite restarted when its
  config changed.
- **null-graph's path in the dev server:** after the install, it was served from
  `/.yalc/null-graph/dist/`, as source, with no `?v=` stamp (the junction). The yalc update then
  reached the running server without a restart (`profiler.js?t=…`).
- **The page loads as today.** `npm run drive` loaded `/dashboard/agents/herd-wolves-gpu` in Chrome
  on the AMD iGPU three times: on the dev server through the junction, on the build
  (`vite preview`, port 8091), and on the dev server after the yalc update. Each time:
  - 4,096 agents on rules, at 60 fps;
  - both species switched to Network and back;
  - no error, in the page or the worker.

  Screenshots, read by eye: the overlay names `network 24→64→64→64→2` for each species, then
  rules again.

**Not run:** ONNX Runtime in the site's own worker, and a build that serves it. Both arrive with
10.3, when the worker first imports it.

## Phase 10.2 results — 2026-10-14: switches that compile nothing, and the rules checked against the networks

**What changed in the site,** all in `src/data/engine-demos/demos/AgentsExperiments/HerdWolvesGPU/`:

| File | Now | Before |
|---|---|---|
| `brain.ts` | A copy of `tools/agents/scenario/herd-wolves.mjs`'s texts (D3): the rule's four functions, `herdRule(seams)`, `DRIVER` and `LABEL`. `HERD_BRAIN` is `herdRule(DRIVER)`; `HERD_LABEL` is `herdRule(LABEL)`, the text the networks copy. `HERD_CATCHES` and `herdRespawnWGSL` unchanged: they already were the scenario's | `HERD_BRAIN`, the rule as one text, which decides what `herdRule(DRIVER)` decides |
| `networks.ts` | `trainedOn.teacher` read from each file; `sha256Hex(text)`, which is null on a page that is not secure (no `crypto.subtle`) | — |
| `index.ts` | The networks load with the rule check. Each system builds both networks and compiles their passes with `prepareBrain` once the files have loaded, and keeps the rule's pass from its build. A species switches only to a compiled pass. Every switch is timed and logged. The overlay gains a line | The network's pass compiled inside `setBrain`, at the first switch |

- **The overlay:**
  - each species' line says `compiling network…` until its network's pass is ready, `network failed`
    if it does not compile, and `trained on another rule` when the check fails;
  - a new last line: `last switch: wolf → network in 0.30 ms · compiled ahead` (or `compiled in the
    switch`).
- **The console:**
  - `[HerdWolvesGPU] networks loaded; this rule 9518de56…: prey copies it, wolf copies it`;
  - `… prey network compiled ahead in 212 ms`;
  - `… switch: prey → network in 0.10 ms, compiled ahead` (or `compiled now`).
- **One addition to §7: the rule's pass is kept.** `AgentSystem` releases the pass of a brain that
  no species runs, unless the brain was prepared. So with both species on networks, a return to the
  rules compiled the rule's pass again. `prepareBrain` of the rules brain, which the system already
  runs, keeps its pass and compiles nothing.
- **"Compiled ahead" is computed as `setBrain` decides it:** the brain is kept (prepared), or still
  runs for the other species. The phrase is not a constant.

**Checks:**
- **The copy is exact.** `node test/spikes/live-switch/check-site.mjs` transpiles `brain.ts` with
  the site's esbuild and compares it with the scenario. Every text was equal, byte for byte: both
  rules, the catches, and the respawn at five flash rates. `HERD_LABEL`'s SHA-256 is `9518de56…`,
  the `trainedOn.teacher` of both committed networks. The same script checks 10.1's files: 16
  checks in all.
- **The script, watched failing** on a scratch copy of the demo's folder with three changes (a
  changed rule, one byte of `prey.onnx`, one weight of `wolf.mlp.json`): 9 checks failed, and each
  change was named.
- **The site's checks:** `tsc` passes, and ESLint reports nothing in the demo's folder. Prettier: the
  new code is formatted as the site's; two older differences in `index.ts` were left alone.
- **The drive,** in Chrome on the AMD iGPU, on the dev server and on a build (`vite preview`). The
  script: prey to Network, wolf to Network, both back to Rules, both to Network again, then 16K
  agents and back to 4K:
  - **10 switches, every one compiled ahead, in 0.0–0.3 ms.** Chrome steps `performance.now()` by
    0.1 ms on this page, which is not cross-origin isolated;
  - **the networks' passes compiled in the background:** 209–281 ms for both species at once (four
    page loads), 266–366 ms on the 16K system (three runs), and 27–36 ms back at 4K, a WGSL the
    device had compiled before;
  - **the rule check:** prey and wolf copy this rule;
  - no error, in the page or the worker.
- **Watched failing,** each change made and then reverted:
  - **a changed rule** (the prey's base throttle 0.1 → 0.12): the log said `prey copies another
    (9518de56…)`, and the overlay said `trained on another rule` for both species;
  - **the rule's pass not kept** (`prepareBrain(rules)` removed): the first species back on the
    rules logged `compiled now` (0.60 ms).

  After both reverts, the copy check and the whole drive ran clean again.

**Not measured:** the hitch that preparing removes, as Chrome shows it. Chrome compiles in its GPU
process (`docs/work/gotchas.md`), so the worker's thread showed no stall either way. Under Node,
step 9 measured it at 240–1,759 ms (A33).

## Phase 10.3 results — 2026-10-15: ONNX Runtime in the demo

**What changed in the site:**

| File | Now | Before |
|---|---|---|
| `src/workers/engine/EngineCore.ts` | `ort`, the worker's ONNX Runtime service. Its first `start()` makes the device switch (D1, §4): ONNX Runtime imported, with one thread, and `null-graph/inference`; `OrtRuntime.create`; the demo destroyed; the old device's queue and one task awaited; the engine made on `runtime.device` over the same canvas; the profiler made again; the old device destroyed; the demo loaded again, and `DEMO_LOADED` posted. A `LOAD_DEMO` during the switch waits for it. Each part is timed, and logged once. Every demo's setup gets a fifth argument, `{ ort }` | One engine, made at `INIT`; four arguments |
| `src/workers/types/types.ts` | `preyBrain` and `wolfBrain` take `'ort'`; `OrtService`, `DemoServices` | `'rules' \| 'network'` |
| `HerdWolvesGPU/options.ts` | `HERD_BRAINS`: `'rules'`, `'network'`, `'ort'` | The first two |
| `HerdWolvesGPU/networks.ts` | `loadHerdOnnx()` fetches both `.onnx` files (`new URL(…, import.meta.url)`, so a build emits them); `holdsNetwork()` compares a network that `mlpFromOnnx` read with a `.mlp.json`'s, bit for bit (D2) | — |
| `HerdWolvesGPU/index.ts` | ONNX Runtime as the third brain (below) | Rules or network |
| `…/controls/herd-gpu-controls.tsx` | A third button per species, "ONNX Runtime", and a caption: the first pick downloads it (about 6.5 MB) and moves the engine to its GPU device, so the simulation starts over | Two buttons |

**The demo:**
- **The first ORT pick** asks the worker to start ONNX Runtime. Until the demo is loaded again on
  the new device, the old engine draws, the species keep their brains, and the species line says
  `starting ONNX Runtime…`. The page's loading overlay shows only for the switch itself, from the
  demo's destroy to its load on the new device: about 60–110 ms by the logs.
- **Once ONNX Runtime runs,** the `.onnx` files load, and each is read with `mlpFromOnnx` and
  compared with its `.mlp.json` (§5). A file that differs is refused for ORT.
- **Each system prepares its ORT brains once:** `runtime.load(bytes)`, `new OrtBrain(model)`,
  `prepareBrain`, one warm-up run. A species switches when its brain is ready. Until then its line
  says `loading ONNX model…`, then `warming up ONNX Runtime…`.
- **A failed run** (`advance()` throws once, naming the species) puts that species back on the
  rules. That frame steps nothing, and the warning line says why (§10).
- **Destroying a system** destroys its `OrtBrain`s, then their models. The runtime stays with the
  worker.
- **The overlay:** a species on ORT reads `ONNX Runtime 24→64→64→64→2`. One that asked for it and
  cannot have it reads `ONNX Runtime did not start`, `.onnx files failed` or `ONNX Runtime
  refused`, and the reason goes on the warning line. The last-switch line names the brain.
- **The inspector:** `decided by ONNX Runtime`.
- **The console:**
  - `[Engine] ONNX Runtime 1.29.0 started; the engine moved to its device: import 49.6 ms, runtime 1096.8 ms, demo destroyed 2.2 ms, engine 1.3 ms, demo loaded 86.2 ms, total 1236.1 ms`;
  - `… .onnx files loaded: prey.onnx holds prey.mlp.json's network; wolf.onnx holds wolf.mlp.json's network`;
  - `… prey ONNX Runtime brain ready: loaded and prepared in 39 ms, first run 184 ms`;
  - 10.2's switch lines, with `→ ONNX Runtime`.
- **Loaded on the first ORT pick only:** in a build, the worker reaches ONNX Runtime's chunk and
  `null-graph/inference`'s only through `import()` on that path, and the demo fetches the `.onnx`
  files only there (read from the built chunks).

**Found in 10.3:**
1. **ONNX Runtime 1.29 takes one call at a time, across all its sessions.** With both species on
   ORT:
   - the wolves' model, created while the prey's was being created, failed:
     `another WebGPU EP inference session is being created.`;
   - with the creates in turn, the wolves' create during the prey's warm-up run failed inside ONNX
     Runtime's WASM (`operation does not support unaligned accesses`), and ONNX Runtime then threw
     `memory access out of bounds`, uncaught;
   - with each preparation in turn, the two species' runs in flight together failed the same way.
     The wolves went back to the rules (§10), and ONNX Runtime threw its own uncaught error.

   Step 9's suites never have two of ONNX Runtime's calls in flight at once: A33 loads two models,
   one after the other, and runs each in turn. `OrtRuntime` does not queue calls. The demo now
   queues every load, warm-up, run and release of its models on one promise chain. Each model's
   `run` is wrapped after its warm-up, so the runs that `AgentSystem` starts wait their turn. With
   the queue, both species ran on ORT in every drive, with no error. **Cause, not verified:** the
   WASM that Vite serves is ONNX Runtime's asyncify build, which can suspend one call at a time.
   Whether null-graph should queue the calls itself is the user's decision (`work/plan.md` §5,
   item 14).
2. **A warning wider than the overlay's line wrote its second line over the last-switch line.**
   The overlay wraps each line at 44 em, and 10.2 put the last-switch line just below the warning
   line. ORT's start failure showed it, since its reason is long. The timeout guard's notice and
   the darts note above 262,144 agents wrap too. A warning now comes last, below the last-switch
   line, where nothing is drawn under it.
3. **§10's last bullet did not happen as written.** The page terminates the worker when the canvas
   unmounts (`useEngineStore.cleanup`), and that takes ONNX Runtime and its device with it.
   `EngineCore.destroy()` now destroys the demo and then awaits the runtime's destroy, but nothing
   calls it, before 10.3 or since. §10 says so now.

**Checks:**
- **The site's own:**
  - `tsc` passes (30 s), and ESLint reports nothing in the changed files;
  - Prettier: the new code is formatted as the site's. Older differences were left alone:
    `EngineCore.ts`, `types.ts` and the controls hold some, and `index.ts` one of the two that 10.2
    noted (10.3 rewrote the other's code).
- **`check-site.mjs`:** 16/16. 10.3 reads the files it checks and changes none.
- **The build** (`vite build`, 101 s) emits:
  - ONNX Runtime's chunk, `ort.webgpu.bundle.min` (116,633 B), and its `.wasm` once (25,749,873 B);
  - `null-graph/inference`'s chunk (13,516 B);
  - both `.onnx` files as hashed assets (41,186 B each).
- **The drive,** in Chrome on the AMD iGPU: the whole script four times, twice on the dev server
  and twice on a build (`vite preview`). The script sets the prey to ORT, which starts it, then
  the wolves; both back to Rules, both to Network, both to ORT again; then 16K agents, and 4K:
  - **the device switch,** 13 first picks measured, in this script and in the shorter drives:
    781–1,295 ms, and 1,824 ms on the first pick after the dev server started. Its import of ONNX
    Runtime took 348 ms there, against 35–74 ms otherwise. Phase 10.0's spike measured
    692–939 ms. The parts:

    | Part | Time |
    |---|---|
    | `OrtRuntime.create`: ONNX Runtime's WASM fetched and compiled, its device made | 649–1,137 ms (1,384 ms on that first pick) |
    | The demo destroyed | 1.2–3.8 ms (39.9 ms once) |
    | The engine made on the new device | 0.4–1.3 ms |
    | The demo loaded again | 59–102 ms |

  - **the ORT brains:**
    - each system's first model loaded and prepared in 39–126 ms. Its warm-up run took
      155–485 ms, while ONNX Runtime compiled its pipelines;
    - the other species' model then took 6–10 ms and 2–6 ms, since ONNX Runtime had compiled the
      same shapes;
    - after a rebuild (16K, then 4K), each model took 6–86 ms and 2–29 ms;
  - **61 switches** in the logs kept, every one compiled ahead:

    | To | Switches | Time |
    |---|---|---|
    | ONNX Runtime | 42 | 0.1–1.1 ms; one above 1 ms (the wolves, at 16K) |
    | Network | 11 | 0.0–0.2 ms |
    | Rules | 8 | 0.0–0.5 ms |

  - **60 fps** with both species on ORT, at 4K and at 16K (the page's widget), with catches going
    on (1,620–2,134 a minute at 4K);
  - **no error,** in the page or the worker, in any run of the finished code.
- **Read by eye** (screenshots):
  - while ONNX Runtime started, the old engine ran on the rules (`starting ONNX Runtime…`);
  - each species then read `ONNX Runtime 24→64→64→64→2`, at 4K and at 16K;
  - zoomed in and paused, the inspector read `Prey 2,113 · decided by ONNX Runtime`.
- **The failures, each caused and then undone:**
  - **a refused model:** one bit of `prey.onnx`'s weights flipped. The log said `prey.onnx is not
    prey.mlp.json's network: export it again`. The prey stayed on the rules, with `ONNX Runtime
    refused` on its line and the reason on the warning line, and the wolves ran on ORT. The file was
    restored byte for byte, and `check-site.mjs` passed again;
  - **ONNX Runtime not starting:** for one run, `wasmPaths` pointed at a folder that does not exist.
    The engine stayed on its device. The prey line said `ONNX Runtime did not start`, the warning
    line gave ORT's reason (`could not start its WebGPU provider: no available backend found`), and
    the wolves still switched to Network;
  - **a failed run:** met only before the queue (finding 1). The wolves went back to the rules, as
    §10 says. No run failed once the calls were queued, and nothing here fails one on demand.
- **Watched failing:**
  - **the queue:** without it, the wolves' model did not load, and then ONNX Runtime's memory broke
    (finding 1);
  - **the warning's place:** before the change, ORT's start-failure reason overprinted the
    last-switch line. After it, the reason sits below, and so does the timeout guard's two-line
    notice, which refused 1M agents on this iGPU.

**Not run:**
- **the engine failing on ONNX Runtime's device,** the branch that keeps the old engine: nothing
  here makes it fail;
- **a second demo on ONNX Runtime's device:** each page has its own worker (`EngineCanvas` is keyed
  by the path), so ONNX Runtime goes with the page;
- **counts above 16K with ORT** (phase 10.5). At 1M, the demo's guard refused the build on this
  iGPU: about 2.0 s per step predicted, over its 600 ms limit;
- **Chrome on NVIDIA, and other browsers** (Deferred).
