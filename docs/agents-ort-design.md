# NullGraph Agents — C2 step 9: ONNX Runtime brains — Technical Design

**Parent:** [`agents-design.md`](./agents-design.md) §9.1 (the brain contract), §9.4 (`OrtBrain`,
`null-graph/inference`), §9.6, §10 (U13, A5, A6), §11, §12.3, §13 (C0: spikes S1, S2, S4, S5; C2
step 9) and decision 6 (device ownership).
**Builds on:**
- [`agents-onnx-reader-design.md`](./agents-onnx-reader-design.md): the entry `null-graph/inference`,
  `mlpFromOnnx`, and A28, which runs ONNX Runtime beside the built-in kernel;
- [`agents-species-brains-design.md`](./agents-species-brains-design.md): one think pass per brain,
  gated to its species;
- [`agents-gpu-loop-design.md`](./agents-gpu-loop-design.md): the step chain and `thinkEvery`.

**Asked by the user:** 2026-10-14, "yes commit 8.3 and step 9".

**Decided with the user, 2026-10-14** (§14; [`work/plan.md`](./work/plan.md) §5, item 9): "ORT
creates the device, graph capture on, prepare at load compiled in the background, and both paths
read separate weight files."
1. **ORT creates the device; the engine adopts it.** The parent's engine-first direction and its
   §12.3 are withdrawn (decision 6 of the parent, settled by S5).
2. ~~**Graph capture is on by default;** `capture: false` turns it off for one model.~~
   Superseded by item 6.
3. **Brains are prepared at load, compiled in the background:** `await agents.prepareBrain(brain)`,
   over a new core method that creates a compute stage asynchronously (§8.6, §12.1). The parent's
   brain-swap stall (`work/plan.md` §5, item 3) is settled by it.
4. **Both paths read weights stored outside the `.onnx` file:** ORT through its own option, and
   `mlpFromOnnx` given the same files (§6.1). Step 8's refusal becomes the message for a file not
   given (`work/plan.md` §5, item 8).

5. **Added after phase 9.2: graph capture is off for a model whose weights are in a second file,**
   unless the app asks for it (§14, item 5; the user: "ok pick A"). Superseded by item 6.
6. **Added after phase 9.4's A6: graph capture is off by default for every model;** `capture: true`
   turns it on for one model (§14, item 6; the user chose option A). Under capture ONNX Runtime can
   fail a model from its second run, and it leaks memory with every run.

**Nothing is open** (§14).

**Status:** design decided 2026-10-14.
- **Phase 9.0 ran on 2026-10-14:** spikes S1, S2, S4 and S5, on Node/NVIDIA, Node/AMD and
  Chrome/AMD. Results: "Phase 9.0 results", at the end. They changed the parent's plan in one
  place that matters: **ONNX Runtime 1.29 runs only on a device it made itself** (S5), so the engine
  adopts ONNX Runtime's device, never the reverse.
- **Phase 9.1 was built on 2026-10-14:** `OrtRuntime`, `OrtModel` and the anchor. A29 passes on
  three host/GPU pairs: the device outlives every model, and a captured model loaded after a release
  runs 100 times on AMD. The adapter is 1,925 B min+gzip, under its line of 3,000 B.
- **Phase 9.2 was built on 2026-10-14:** `mlpFromOnnx` reads weights stored in a second file, and
  both paths read PyTorch's two-file exports (A32 on three host/GPU pairs). It found two faults of
  ONNX Runtime: a rule for buffers (rounded up to 16 B) guards against one; against the other,
  capture is off by default for models whose weights are in a second file (§14, item 5).
- **Phase 9.3 was built on 2026-10-14:** `OrtBrain`, and async brains in `AgentSystem`. In
  lockstep, prey on `OrtBrain` and wolves on `MlpBrain` act within 1.1e-5 of `MlpBrain` on the same
  step, with graph capture on and off; two runs of 120 steps are byte-identical (A5, A30, A31 on
  three host/GPU pairs).
- **Phase 9.4 was built on 2026-10-14:** A6, the interactive schedule on the GPU. Across 600
  frames, no `mapAsync` outside the inspector and no GPU error; every frame acted on actions one
  frame old, except 1 of 300 in Chrome at the display's pace (three host/GPU pairs). A6 also found
  a fault of ONNX Runtime under graph capture, and a leak; capture is now off by default (item 6).
- Phases 9.5–9.6 follow (§13).

**Target:** `src/extras/inference/` gains `OrtRuntime`, `OrtModel` and `OrtBrain`, and
`mlpFromOnnx` reads weights stored outside the file; `src/extras/agents/AgentSystem.ts` learns
async brains (the run schedule, and lockstep with a run) and `prepareBrain`; core gains one method,
`createComputeStageAsync`. ONNX Runtime becomes an optional peer dependency.

---

## 1. Purpose

**Run an ONNX model with ONNX Runtime's WebGPU provider on the engine's device, over GPU buffers
bound once,** and make such a model the brain of one or more species. The generic part
(`OrtRuntime`, `OrtModel`) serves any model with GPU inputs and outputs: an image in, a mask
out, for the photo-booth and relight ideas. `OrtBrain` is the agents' wrapper over it.

**Words used here:**
- **ORT:** ONNX Runtime Web, the app's `onnxruntime-web` package, version 1.29.0 here;
- **runtime:** an `OrtRuntime`, which owns ORT's device for its life;
- **anchor session:** a session of a one-node model that the runtime holds, so that the device
  outlives every model (§5);
- **model:** an `OrtModel`, one ORT session bound to fixed GPU buffers;
- **results buffer:** what an `OrtBrain`'s runs write, `[capacity × actDim]` f32;
- **copy pass:** the think pass of an async brain: it copies the brain's results into `act`, for
  its species' agents that think this step;
- **interactive schedule / lockstep:** the parent's two schedules (§8);
- **graph capture:** ORT's mode that records a session's GPU work on the first run and replays it;
- **prepared brain:** one whose think pass `prepareBrain` compiled ahead, in the background, and
  the system keeps (§8.6);
- **external data:** weights an exporter stored in a file beside the `.onnx` file, which the
  `.onnx` file names (§6.1).

### The single most important pre-existing fact: ORT runs only on a device it made

The parent planned two directions for the device (§9.4, decision 6): *ORT first*, where the
engine adopts ORT's device (Cycle A), and *engine first*, where ORT is handed the engine's device
(parent fact 15). **S5 measured the second, and it does not work with ORT 1.29:**
- handed the engine's device, a device with WebGPU's default limits, or a device asked for exactly
  as ORT asks for its own, the WebGPU provider creates sessions, and **every run fails**:
  `No cached or pending pipeline for deferred dispatch: Tanh:1:1|:2;:`. After a few failures ORT
  can stop answering altogether (Node/AMD and Chrome).
- The older provider (the `all` bundle) ignores a device set in `ort.env.webgpu.device`, makes
  its own, and leaves the app's buffers unwritten.
- Limits are not the cause: the device asked for like ORT's own fails the same way. So the
  parent's §12.3 (`requiredLimits` in core) is withdrawn.

So **ORT makes the device and the engine adopts it**, with `engine.init(canvas, { device })`, which
core already supports and never destroys (fact 8). An engine-first host, such as the
test-engine site's worker, must create the runtime before its engine, or initialise its engine
again (§5, and Deferred for step 10).

### The second fact: ORT's device dies with its last session

With the WebGPU bundle, releasing the last session destroys ORT's device. The next read-back on
it fails with `AbortError`, and the next session gets a new device (S1, S5; `work/gotchas.md`). An
engine that adopted the device would lose it the moment an app swapped one model for another. So
the runtime holds an **anchor session** for its whole life (§5). The anchor also closes a trap in
graph capture: on AMD, a capture session created on a re-made device failed on its second run, 4
times of 4 (Phase 9.0 results, S5).

### The third fact: on the GPU, ORT costs 3× to 10× the built-in kernel

For the 24 → 32 → 32 → 2 network at 20,000 agents, one ORT run takes 515–1,301 µs of GPU time on
NVIDIA and 1,533–1,684 µs on the AMD iGPU. The built-in kernel, timed the same way in the same
runs, takes 180–422 µs and 213–225 µs: ORT is about 3× on NVIDIA and 7–8× on the iGPU, and up to
10× at 100,000 agents (S1). Starting ORT takes 0.8–1.6 s to the first actions and 6.4 MB of WASM
(S4). This confirms decision 1 of 2026-10-11: rules and the built-in kernel ship; **ORT is the
research path**, for networks the kernel cannot run and as the cross-check that keeps it honest
(A5).

### Non-goals

- **Running ORT on a device the app made.** Refused by a check, with the fix named (§11). Measured
  again with `npm run spike:s5` when ORT is upgraded.
- **The older JavaScript provider (`all` bundle) as a tested path.** It cannot load under
  Node/Dawn; the guide names `onnxruntime-web/webgpu`.
- **CPU inputs or outputs.** Every input and output is a GPU buffer; graph capture needs that, and
  nothing in the frame reads back.
- **Tensor types other than float32.** Refused by name; int64 token inputs, float16 and quantized
  models are later work.
- **Per-agent weights, and `setWeights`.** One weight set per session. New weights mean a new
  model and `setBrain`.
- **A watchdog for a run that never answers.** ORT stopped answering only on app-made devices,
  which the runtime never uses. The tests guard with time limits.
- **The demo** (step 10), and the site's engine-first worker.
- **Bundling ORT, or importing its types.** The library declares the few members it uses
  (§12.3); an app that imports only `mlpFromOnnx` needs no ORT, not even its types.

### Design principles applied

- **Verify against the tree** ([`design-principles.md`](./design-principles.md) §1): §3. The device
  question was measured (S5), not assumed from ORT's typings, which said it would work.
- **Fail loud, naming the offender** (§2): every refusal names the tensor, the axis or the species,
  and the fix (§11). A failed run is thrown by the next `advance()` or `step()`, naming the
  species.
- **Buffers are fixed for life** (parent principle 5): a model binds once; a second binding is
  refused, because a captured session would ignore it silently (S2).
- **Detect, don't polyfill** (§12): the device check refuses; nothing emulates engine-first.
- **Everything expensive happens at build time** (§11): sessions are created, checked and bound
  before the first frame, and a brain's pass can be compiled at load, in the background (§8.6);
  the frame path starts at most one run and allocates nothing.

## 2. Architecture

```ts
import * as ort from 'onnxruntime-web/webgpu';                 // the app's dependency, the app's import
import { OrtRuntime, OrtBrain } from 'null-graph/inference';

const runtime = await OrtRuntime.create(ort);                   // ORT makes the device; the anchor keeps it
await engine.init(canvas, { device: runtime.device });          // the engine adopts it (Cycle A's direction)
const agents = new AgentSystem(engine, definition);
const model = await runtime.load(bytes);                        // a session: WebGPU, GPU outputs, no capture
agents.setBrain(new OrtBrain(model), { species: 'prey' });      // binds obs and its results buffer once

// Interactive: the frame code does not change.
agents.advance(frameSeconds);                                   // also starts a run when none is in flight
engine.render([...agents.getComputeStages(), ...vis.getComputeStages(), scenePass]);

// Lockstep (tests, recording, replays):
await agents.step();                                            // sense → await the run → copy, act
```

```
  frame k: one submission                                    between frames k and k+1
  ┌─────────────────────────────────────────────────────┐    ┌──────────────────────────┐
  │ grid → sense ──► obs                                 │    │ the ORT run that         │
  │ think pass (MlpBrain):  obs ──► act (wolves' rows)   │───►│ advance() started in     │
  │ copy pass (OrtBrain): results ──► act (prey's rows)  │    │ frame k: reads obs,      │
  │ act → tick → instances                               │    │ writes results           │
  └─────────────────────────────────────────────────────┘    └────────────┬─────────────┘
  frame k+1: its copy pass reads those results ◄──────────────────────────┘

  OrtRuntime ── anchor session (one node) ── keeps ORT's device ◄── adopted by the engine
      └── OrtModel (session + bound tensors) ◄── OrtBrain { kind: 'async', wgsl: copy pass, buffers: [results] }
```

An ORT run is never a stage in the frame (parent §1, second fact). In the interactive schedule it
lands between two frames' submissions, so a frame acts on the actions of the frame before (S2:
298–299 frames of 300 at 60 fps on every host, and never a frame's own). In lockstep, `step()`
splits its submission around the run.

## 3. Integration facts (verified against this repository, 2026-10-14)

| # | Fact | Source | Consequence |
|---|---|---|---|
| 1 | The brain contract as built: `kind: 'stage' \| 'async'`, `wgsl`, `buffers` (group 1), `dims`, `assertDevice`, and `bind?(io: { obs, act, capacity, obsDim, actDim })` and `run?()`, which nothing implements yet. | `src/extras/agents/RuleBrain.ts:1-11` | Async brains fit the contract as it is (§7). |
| 2 | `setBrain` refuses async brains. | `AgentSystem.ts:209` | Step 9 replaces that refusal. |
| 3 | Each attached brain is one think pass, gated by a 16-byte species mask and by `thinksNow` (alive, newborn, or its block's turn under `thinkEvery`). | `AgentSystem.ts:179-224`; `agentThinkShaders.ts:26-54`; `agentSenseShaders.ts:5-12` | An async brain's copy pass is an ordinary think pass: species and cadence come free. |
| 4 | A think pass binds the brain's `buffers` as group 1, at most `maxStorageBuffersPerShaderStage − 5`. | `AgentSystem.ts:140-141,166-170,213-214` | The results buffer is the copy pass's one group-1 buffer. |
| 5 | `MlpBrain`'s think writes every action, then `setReason(i, 0u)`. | `mlpShaders.ts:51` | The copy pass does the same, so the inspector shows no stale reason. |
| 6 | The step chains are prebuilt for 0…`maxStepsPerFrame` steps: grid → queries → sense → thinks → act → app stages → tick, then the instance stage once. | `AgentSystem.ts:150-163` | Lockstep's two halves are prebuilt the same way. |
| 7 | `advance()` returns the steps of this frame; `step()` renders one step and awaits `onSubmittedWorkDone`. | `AgentSystem.ts:378-387` | The interactive schedule lives in `advance()`; lockstep in `step()`. |
| 8 | `engine.init(canvas, { device })` adopts a device; NullGraph never calls `destroy()` on it. | `src/core/types.ts:144-147`; `test/onnx-gpu/run.js:83-91` (O2) | ORT first needs no core change. |
| 9 | One `engine.render()` is one command encoder and one submission. | `src/core/RenderPipeline.ts:14,171` | An ORT run lands before or after a whole frame, never inside it. |
| 10 | ORT passes `enableGraphCapture` to the WebGPU provider only when the provider is given as an object; the session's own setting is passed either way. | `onnxruntime-web/lib/wasm/session-options.ts:105,118-121,263-271` | With `'webgpu'` as a string, the JavaScript side captures and the provider does not; S2's run hung the process. The runtime always passes `{ name: 'webgpu' }`. |
| 11 | The provider's `device` option registers an app's device. | `session-options.ts:108-114,156-163` | The way S5 handed the engine's device over; it fails (§1). |
| 12 | Pre-allocated GPU outputs and GPU-resident inputs are accepted. | parent fact 11 | Inputs and outputs stay in the app's buffers. |
| 13 | Graph capture binds a session's buffers once and ignores a different buffer later. | parent fact 12; S2 | `bind` happens once, and a second one is refused. |
| 14 | `run()` awaits its input preparation before dispatching. | parent fact 13; S2 | A run started before a frame's render reads that frame's observations, and lands after it. |
| 15 | A session reports its inputs' and outputs' types and shapes; symbolic axes are strings. | parent fact 14 | `OrtModel` checks every binding against them. |
| 16 | Parent fact 15 does not hold for ORT 1.29: an app-made device, handed over either way, does not run. | S5 | ORT first only (§5); the parent's §12.3 withdrawn. |
| 17 | ORT's device (WebGPU bundle) is destroyed with its last session. | S1, S5; `docs/work/gotchas.md` | The anchor session (§5). |
| 18 | `null-graph/inference` exists: `mlpFromOnnx`, 2,525 B min+gzip under a line of 2,600 B. | `build.js:29`; `src/extras/inference/index.ts:1`; `test/inference-onnx.test.mjs:348` | The ORT adapter joins it, with a size line of its own (§12.4). |
| 19 | `onnxruntime-web` 1.29.0 is a devDependency; the package has no `peerDependencies`. | `package.json` | §12.3 adds it as an optional peer dependency. |
| 20 | The default step is 1/60 s; up to `maxStepsPerFrame` steps per frame. | `agentLayout.ts:244`; `agentMath.ts:226-251` | At the defaults, one run per frame is 60 runs a second (§8.3). |

## 4. Data model

| What | Size | Owner | Notes |
|---|---|---|---|
| `obs` | `capacity × obsDim × 4` B | `AgentSystem` (exists) | ORT reads it as the model's input, `[capacity, obsDim]`. |
| results buffer | `capacity × actDim × 4` B rounded up to 16 B (§6), `STORAGE \| COPY_SRC \| COPY_DST` | `OrtBrain` | ORT writes it as the model's output; the copy pass reads it. Seeded from `act` at `bind`. 160,000 B at 20,000 agents × 2 actions. |
| `act` | exists | `AgentSystem` | Written only by think passes, as today. ORT never writes it. |
| species mask | 16 B uniform | `AgentSystem` | The think pass's, unchanged (fact 3). |
| two tensors | — | `OrtModel` | `ort.Tensor.fromGpuBuffer` over `obs` and over the results buffer, made once at `bind`. |
| anchor model | about 100 B, embedded | `OrtRuntime` | One node the WebGPU provider runs (an `Add` of a 1-float input, as `test/onnx-gpu/tiny-model.mjs` encodes). |

No new uniform and no new struct. The copy pass:

```wgsl
@group(1) @binding(0) var<storage, read> ortResults: array<f32>;
fn think(i: u32, species: u32, tick: u32) {
    for (var k = 0u; k < ACT_DIM; k++) { setAct(i, k, ortResults[i * ACT_DIM + k]); }
    setReason(i, 0u);
}
```

It is a constant string: `ACT_DIM`, `setAct` and `setReason` come from the generated think module
(`agentThinkShaders.ts:26-54`).

## 5. The device: ORT first, held by an anchor

`OrtRuntime.create(ort, { adapter? })`:
1. **Checks `ort`:** `InferenceSession.create`, `Tensor.fromGpuBuffer` and `env.webgpu` must exist,
   or it refuses, naming `onnxruntime-web/webgpu` (§11).
2. **Sets `ort.env.webgpu.adapter`** when `adapter` is given; otherwise ORT asks for its own.
3. **Creates the anchor session** from the embedded one-node model, with `{ name: 'webgpu' }`. This
   is when ORT loads its WASM and makes its device: S4's "first session", 0.49–0.69 s under Node and
   0.54–0.57 s in Chrome.
4. **Reads `await ort.env.webgpu.device`,** which is `runtime.device`.

The engine adopts it: `engine.init(canvas, { device: runtime.device })`. Every model the runtime
loads runs on it. **Destroy order:** the engine and everything on the device first, then
`await runtime.destroy()`, which releases every model still loaded, then the anchor; the device
goes with the anchor.

**Two runtimes** share ORT's one device, each with its own anchor. An app that also uses ORT
directly shares it too.

**Phase 9.1 verifies the anchor before anything relies on it** (A29): load, release and load again
a model with the runtime alive; the device stays alive and a captured session runs 100 times on AMD.
If a one-node `Add` session does not hold the device, the anchor becomes a model the provider runs
with more than one node; the design does not change.

**Engine-first hosts.** The site's worker makes its engine at start-up. With ORT 1.29 it has two
ways to an ORT brain: create the runtime before the engine when ORT may be used (paying S4's
0.8–1.6 s and 6.4 MB on every visit), or initialise its engine again on the runtime's device when
the user first picks ORT. That choice belongs to step 10's design (Deferred).

## 6. Models: load, bind, run

`await runtime.load(bytes, { capture?, externalData? })` creates one session with options the
runtime sets and the app cannot override:
- `executionProviders: [{ name: 'webgpu' }]`, **an object, never the string** (fact 10);
- `preferredOutputLocation: 'gpu-buffer'`: no output ever goes to the CPU;
- `enableGraphCapture: capture`, off unless `capture: true` (§9, §14 item 6; until phase 9.4 it
  was on unless `capture: false`, and off for a model whose weights are in a second file);
- `externalData`, passed through to ORT as it is (§6.1).

`model.inputs` and `model.outputs` list each tensor's name, type and axes (numbers, or strings for
symbolic axes).

`model.bind({ inputs: { name: { buffer, dims } }, outputs: { … } })`, once:
- **every input and every output** gets a GPU buffer. Graph capture needs every tensor on the GPU,
  and a captured session without pre-allocated outputs fails (Phase 9.0 results);
- each tensor must be float32, of the model's rank; each fixed axis must equal the given one; a
  symbolic axis takes the given size, and the same symbol must get the same size everywhere;
- each buffer must hold the tensor's values **rounded up to 16 B**, and have `STORAGE` usage. ONNX
  Runtime's kernels read and write 16 B at a time; on NVIDIA, a write that ends past the buffer is
  dropped whole, and the last values stay as they were, with no error (phase 9.2). The system's
  `obs` is therefore allocated rounded up as well (phase 9.3);
- the tensors (`ort.Tensor.fromGpuBuffer`) and the feeds and fetches objects are made here, once.
  A run allocates nothing on the app's side.

`await model.run()` runs the session over the bound buffers. It resolves when ORT has queued its
GPU work, not when the GPU has finished; the queue orders everything after it. One run at a time:
a second `run()` before the first resolves is refused.

`await model.destroy()` releases the session after any run in flight.

### 6.1 Weights outside the file: both paths read them

An exporter may store large tensors in a second file, leaving in the `.onnx` file, for each one, a
location, an offset and a length. PyTorch's default call does it for every tensor over 256 B
(step 8's phase 8.0: `torch-big` wrote `model.onnx`, 10,518 B, and `model.onnx.data`, 39,424 B).

One shape serves both paths, ORT's own: `externalData: [{ path, data }]`, where `path` is the
location as the `.onnx` file writes it and `data` is the second file's bytes.
- **ORT,** `runtime.load(bytes, { externalData })`: passed through as it is. ORT also takes a URL as
  `data`, and fetches it.
- **The reader,** `mlpFromOnnx(bytes, { externalData })`: bytes only, because the reader is
  synchronous and fetches nothing. For each initializer stored outside the file:
  - its location must be one of the given paths, compared as written. The reader never opens a
    file, so a location such as `../x` reaches nothing it was not given;
  - its offset, and its length (the rest of the file when no length is written), must lie inside
    that file, and the length must be 4 bytes per value: float32, little-endian, as `raw_data` is;
  - the values are then decoded as `raw_data` is today, and count against the reader's memory
    budget, which grows by the bytes of the files given.

  A location not given is refused with step 8's message, which now names both ways out (§11).
- **Size:** about +200 B on the reader. Its line moves from 2,600 B to 2,900 B unless the build
  shows otherwise; the number goes back to the user either way.

## 7. `OrtBrain`

`new OrtBrain(model, { input?, output? })`. The names default to the model's only input and only
output; a model with several needs them named.
- **Checks at construction:** input `[N, obsDim]` float32 and output `[N, actDim]` float32, where
  `N` is symbolic or a number. `dims` is `{ obs, act }` when both second axes are numbers.
- **A fixed first axis** must equal the system's capacity, checked at `bind`. PyTorch's export
  fixes it at the example's batch size unless told otherwise, so the refusal names
  `dynamic_shapes` (§11).
- `kind: 'async'`; `wgsl` is the copy pass (§4); `buffers` is `[results]` once bound.
- `assertDevice(device)` refuses any device but `runtime.device`, naming the fix (§11).
- `bind({ obs, act, capacity, obsDim, actDim })`, called by `setBrain` before the pass is built:
  - creates the results buffer, rounded up to 16 B;
  - **seeds it from `act`** with one copy, so a species keeps its last actions until the first
    run lands (S4: 0.14–0.46 s for the first run, which compiles ORT's pipelines);
  - binds the model: `obs` as `[capacity, obsDim]`, the results as `[capacity, actDim]`.

  A second `bind` with the same buffers is accepted (re-attaching to the same system); with other
  buffers it is refused.
- `run()` is `model.run()`.
- `destroy()` waits for a run in flight, then destroys the results buffer. The model is the app's.

**The brain contract does not change** (fact 1). Its `bind` gains a stated meaning: it is called
for async brains only, before their pass is built, and reads `act` once, to seed, never writing it.
Stage brains are untouched.

## 8. The two schedules (in `AgentSystem`)

### 8.1 Attach

For an async brain, `attach` calls `assertDevice`, checks dims, calls `bind`, then builds the think
pass from the brain's `wgsl` and `buffers`, exactly as for a stage brain (fact 3). Species, masks
and `thinkEvery` behave as they do for every brain. A stage brain for one species and an async
brain for another share one chain.

### 8.2 Interactive

`advance(frameSeconds)`:
1. throws a failure kept from an earlier run (below);
2. advances the clock, as today;
3. for each attached async brain with **no run in flight** that has seen **at least `thinkEvery`
   steps since its last run started**, starts `brain.run()`. A frame with no step starts none: the
   observations have not changed.

`getComputeStages()` is unchanged: the copy pass sits where the brain's think pass sits.

What a frame sees (S2, 20,000 agents, 300 frames, three hosts):
- **actions from the frame before** in 298–299 of 300 frames at 60 fps, and 293–299 with frames as
  fast as they come; never a frame's own; a run read the observations of the frame that started
  it every time;
- **after a stall, the latest run wins:** in Chrome, with frames as fast as they come, one run
  spanned 7 frames, which then acted on 2- to 7-frame-old actions, and the next run caught up;
- **a newborn agent** acts for one frame on its slot's last results, before a run has seen it. It
  is stated, not fixed.

A run that **fails** is kept, and the next `advance()` or `step()` throws it, naming the species
(§11). No further runs start for that brain; its species keeps its last actions until `setBrain`
gives it another brain.

### 8.3 How often ORT runs

The gate in the parent's C0 reads: "if S1 is under ~1 ms at 20,000, `OrtBrain` is comfortable
interactively at ≤ 30 Hz thinking". S1 measured 0.52 ms on NVIDIA at its fast clocks, 1.30 ms at
its slow ones, and 1.53–1.69 ms on the AMD iGPU. **So the gate is met only on NVIDIA at its fast
clocks.** Elsewhere a run every frame costs 8–10% of a 60 fps frame's 16.7 ms, and a run every
other frame half that.

The rate is the existing `thinkEvery`, not a new option: a brain runs at most once per `thinkEvery`
steps. At the default step of 1/60 s, `thinkEvery: 2` runs ORT 30 times a second. The guide gives
the numbers; the default stays 1.

### 8.4 Lockstep

`step()`, with at least one async brain attached:
1. awaits any run in flight;
2. renders the first half: grid, queries, sense;
3. awaits `run()` of every async brain;
4. renders the second half: every think pass (the copy passes among them), act, app stages, tick,
   the instance stage;
5. awaits `onSubmittedWorkDone`, as today.

Every step runs every async brain, whatever `thinkEvery` says: the copy pass still copies only the
agents that think this step, so the result is exact. S2 measured 60 of 60 lockstep steps acting on
their own observations, at 2.2–2.9 ms a step under Node and 8.3 ms in Chrome (where
`onSubmittedWorkDone` resolves about every 8 ms). Without an async brain, `step()` is one render,
as today. The parent's "mixed schedules" error is gone: `step()` waits for the run instead.

### 8.5 Swaps and destroy

- **Detaching an async brain** with a run in flight is safe: the run writes the brain's own results
  buffer, which the brain keeps until its `destroy()`.
- **`agents.destroy()`** with a run in flight keeps `obs` until that run has been queued, then
  destroys it. A buffer destroyed before ORT's submission would be a validation error.
- **The pass compile** on `setBrain` is the synchronous stall every brain has today, unless the
  brain was prepared (§8.6).

### 8.6 Preparing a brain at load, compiled in the background

`setBrain` compiles a brain's think pass synchronously: about 1.1 s under Node/Dawn on NVIDIA and
76 ms in Chrome on AMD, for `MlpBrain` (`work/plan.md` §5, item 3). Step 10's live switch would
stall that long.

`await agents.prepareBrain(brain)`:
1. checks the brain as `setBrain` does (device, dims), and binds an async brain (§7);
2. builds the think WGSL and creates the stage with **`engine.createComputeStageAsync`** (§12.1),
   whose pipeline comes from `createComputePipelineAsync`: the compile runs while frames go on;
3. makes the species mask and the bind groups, and keeps the pass, disabled and in no chain,
   until `agents.destroy()`.

`setBrain` with a prepared brain compiles nothing: it puts the kept pass in the chains and writes
the masks. Detaching keeps the pass, so switching back is free as well. A brain never prepared
behaves as today: compiled by `setBrain`, dropped when it serves no species. Preparing a brain
twice returns the first promise. `setBrain` while the brain's preparation is pending is refused
(§11); so is `prepareBrain` after `agents.destroy()`.

What "in the background" means is the host's: Chrome compiles in its GPU process; Dawn under Node
may compile on the calling thread before the promise resolves. Phase 9.5 measures the longest
frame while a brain is prepared, on three hosts (A33). A kept pass costs a pipeline, its bind
groups and a 16-byte mask.

## 9. Graph capture and the other settings

S1 measured graph capture: **the same GPU time**, and **a third or less of the main thread's time
per run**: 0.20–0.28 ms against 0.57–0.69 ms under Node, 0.06–0.09 ms against 0.74–0.83 ms in Chrome.
Phase 9.0 also found four traps in it. The design closes each by construction:

| Trap (Phase 9.0) | Closed by |
|---|---|
| `'webgpu'` as a string with capture on: the JavaScript side captures, the provider does not. A later run read a new buffer, and S2's frame loop hung. | The runtime always passes `{ name: 'webgpu' }` (fact 10). |
| A captured session ignores a different buffer on a later run (parent fact 12). | `bind` once; a second, different binding is refused. |
| A captured run without a pre-allocated output fails (`no overload matched for createBindGroup`). | Every output is bound to a buffer. |
| On AMD, a captured session made on a re-made device failed on its second run (4 of 4). | The anchor: the device is never re-made while the runtime lives. |
| Found in phase 9.2: a captured model whose weights are in a second file can fail from its second run with the same error, on every host, depending on the model and on what ran before it. | Phase 9.4 found the cause, the next row: this is very likely one of its cases. |
| Found in phase 9.4: from a captured session's second run on, ONNX Runtime looks up its output buffer through a handle it released after the first run. The run fails unless the allocator has since put another buffer's handle at that address: after another session was created and released, or created, run and kept, between the first two runs (1.29, every host, depending on the case); with nothing in between (1.30.0 on NVIDIA). | Not closed in ONNX Runtime; avoided by default: capture is off unless asked (§14, item 6). |
| Found in phase 9.4: every captured run from the second keeps an output tensor that `InferenceSession.run` drops: ONNX Runtime's memory grows with every run. | The same default. |

With the first four closed, captured sessions ran for hundreds of runs without a failure on all
three hosts (S1, S2), and capture was decided on by default (§14, item 2). **Decided again after
phase 9.4** (§14, item 6): capture is off by default, since the last two cannot be closed from
outside ONNX Runtime; `capture: true` turns it on for one model, for an app that measured its gain
and accepts the faults.

Left to the app: `ort.env.wasm.numThreads` (1 unless the page is cross-origin isolated), the WASM's
location, and ORT's log level.

## 10. Test suite

**Headless.**

| # | Asserts |
|---|---|
| U13 | (parent) The schedule, in the agents suite, against a fake async brain whose `run()` the test resolves by hand: one run in flight at a time; none started in a frame with no step; at most one per `thinkEvery` steps; lockstep submits the first half, then runs, then the second half, in that order; `step()` waits for an interactive run in flight; a failed run is thrown once by the next `advance()`, naming the species, and no run starts after it; `destroy()` with a run in flight destroys `obs` only after the run settles. |
| U32 | `OrtRuntime` and `OrtModel` against a fake `ort`: the anchor is created first and released last; every session gets `[{ name: 'webgpu' }]`, `'gpu-buffer'` and the capture setting, and the app cannot override them; `bind` checks type, rank, fixed axes, symbols, sizes and usage, each refusal naming the tensor; a second `bind` is refused; a second `run()` in flight is refused; `destroy()` releases after a run in flight. |
| U33 | `OrtBrain` against a fake model: dims from the metadata; one input or output, or the names given; the fixed first axis refused with `dynamic_shapes` named; the results buffer is seeded from `act` before the model is bound; the same buffers re-bind, others are refused; the copy pass's WGSL parses with the think module and contains no backtick (parent U9). |
| U34 | `engine.createComputeStageAsync`, in the core suite: the stage equals `createComputeStage`'s for the same config (name, dispatch, clears, a bind group set on it), the pipeline comes from `createComputePipelineAsync`, and bad WGSL rejects with the stage's name. |
| U35 | `prepareBrain`, in the agents suite (mock device): the pass is created asynchronously; `setBrain` after it, and a re-attach after a detach, create no pipeline (counted on the mock device); a brain never prepared behaves as today; preparing twice gives one pass; `setBrain` during a pending preparation, and `prepareBrain` after `destroy()`, are refused by name. |
| U36 | The reader with external data, in the inference suite: PyTorch's two-file export (`torch-outside`, and its data file, kept in `fixtures.json`) reads with the maker's weights bit for bit; without the data file it is refused, naming the location and both ways out; an offset or length past the file, a length that is not 4 bytes per value, and a location not given are refused by name; U30's damaged files include cut and damaged data files. |
| U14 | (parent) Every §11 error fires and names the offender. |
| — | Size: the ORT part of `inference.js` under its line, the reader under its new line, the agents entry under C2's ceiling, and core's growth reported (§12.4). The guide's example type-checks and runs against the stand-in engine. |

**GPU** (Node/Dawn on NVIDIA and AMD, Chrome on AMD; ORT's own device in every test).

| # | Asserts |
|---|---|
| A5 | (parent) The Herd & Wolves networks as `.onnx`, through `OrtBrain`, equal `MlpBrain` on the same observations within 1e-4, in lockstep, with capture on and off. Also with prey on `OrtBrain` and wolves on `MlpBrain`: each species' rows come from its own brain. |
| A6 | (parent) 300 interactive frames with `OrtBrain`: zero `mapAsync` outside the inspector (counted by wrapping it), zero GPU errors, and actions one frame old (S2's marker method). As built (phase 9.4): 300 frames at the display's pace and 300 as fast as they come, with the default settings. |
| A29 | The anchor: with the runtime alive, a model loaded, released and loaded again runs; the device answers a read-back afterwards; a captured session made after the release runs 100 times on AMD (the case that failed without an anchor). |
| A30 | Two lockstep runs of 120 steps from one seed, with `OrtBrain` serving prey, give byte-identical agent buffers on one device (the parent's A7, extended). |
| A31 | Swaps: rules → `OrtBrain` → `MlpBrain` → the same `OrtBrain` again, interactive, with zero GPU errors; until the first run lands, prey act on their last actions (the seed), not on zeros. |
| A32 | PyTorch's default export, two files, given to both paths: through `OrtBrain`, and through `mlpFromOnnx` into `MlpBrain`. Both equal PyTorch's own outputs within 1e-4, and each other. |
| A33 | Prepared in the background: with frames running, `prepareBrain` of an `MlpBrain` and of an `OrtBrain`; the longest frame during it is reported beside one with `setBrain` compiling; then `setBrain` of the prepared brain takes under 2 ms on every host (76 ms to 1.1 s today), with zero GPU errors. |

**Watched failing**, one mutation at a time: the provider as a string; no anchor; a run started
while one is in flight; runs started every frame ignoring `thinkEvery`; the copy pass reading row
`i + 1`; no seeding; lockstep acting before the run; `obs` destroyed with a run in flight;
`setBrain` recompiling a prepared brain; the reader ignoring the offset; the reader taking a
location it was not given.

## 11. Errors

| Condition | Message |
|---|---|
| Not ORT's web build | `OrtRuntime: that is not onnxruntime-web (no InferenceSession, Tensor.fromGpuBuffer or env.webgpu); pass the namespace of import * as ort from 'onnxruntime-web/webgpu'.` |
| The provider cannot start | `OrtRuntime: ONNX Runtime could not start its WebGPU provider: <first line of ORT's message>.`, or `…: it made no device.` |
| Not bytes (added in 9.1) | `OrtRuntime: expected the model's bytes, as a Uint8Array or an ArrayBuffer.` |
| ORT cannot load the model (added in 9.1) | `OrtRuntime: ONNX Runtime could not load the model: <first line of ORT's message>.` |
| No metadata (added in 9.1) | `OrtRuntime: this ONNX Runtime reports no input or output metadata; onnxruntime-web 1.29 or later is needed.` |
| Used after destroy | `OrtRuntime: used after destroy().` · `OrtModel: used after destroy().` · `OrtBrain: used after destroy().` |
| Not a float32 tensor | `OrtModel: <input\|output> "<n>" is <type>; only float32 tensors can be bound.` |
| Unknown name | `OrtModel: "<n>" is not an <input\|output> of this model; it has <names>.` |
| Missing binding | `OrtModel: <input\|output> "<n>" has no buffer; every input and output is bound to one.` |
| Shape | `OrtModel: <input\|output> "<n>" is [<model dims>]; it was given [<dims>].` |
| Symbol | `OrtModel: the axis "<s>" is <a> in "<n>" and <b> in "<m>".` |
| Buffer | `OrtModel: "<n>" needs <bytes> B with STORAGE usage: its values, rounded up to 16 B, as ONNX Runtime reads and writes 16 B at a time; its buffer has <size> B<, without STORAGE>.` (as built in 9.2) |
| Bound twice | `OrtModel: bound already; buffers are fixed for life, because graph capture binds them once. Load the model again for other buffers.` |
| Run before bind | `OrtModel: run() before bind().` |
| Run in flight | `OrtModel: run() while a run is in flight; await it first.` |
| Several inputs | `OrtBrain: the model has <k> <inputs\|outputs> (<names>); name the <observations\|actions> with { <input\|output>: '<name>' }.` |
| Brain shape | `OrtBrain: <input\|output> "<n>" is [<dims>] <type>; expected [N, <dim>] float32.` |
| Fixed batch axis | `OrtBrain: input "<n>" is [<dims>]: its first axis is fixed at <k>, and the system has <capacity> agents. Export it with a symbolic first axis (PyTorch: dynamic_shapes, or dynamic_axes={'<n>': {0: 'N'}}).` |
| Another device | `OrtBrain: this system's engine has another device. ONNX Runtime 1.29 runs only on a device it made: init the engine with { device: runtime.device }.` (Replaces the parent's "ORT on another device", whose first fix does not work.) |
| Bound elsewhere | `OrtBrain: bound to another system's buffers; buffers are fixed for life. Load the model again for this system.` |
| A run failed | `AgentSystem: the brain of species "<s>"<, "<t>"…> failed in a run: <first line of ORT's message, without its last period>. Give the species another brain with setBrain().` (as built in 9.3) |
| Not a brain (added in 9.3) | `AgentSystem: a brain is a stage brain, or an async brain with bind() and run().` (replaces "async brains are deferred to C2") |
| Model name (added in 9.3) | `OrtBrain: the model has no <input\|output> "<n>"; it has <names>.` |
| Preparation pending | `AgentSystem: this brain is still being prepared; await prepareBrain(brain) before setBrain(brain).` |
| A pass that does not compile | `NullGraph: compute stage "<name>" did not compile: <the device's message>.` (from `createComputeStageAsync`; `prepareBrain` passes it on) |
| Data file not given | `mlpFromOnnx: initializer "<n>" is stored outside the file, in "<where>"; pass that file, { externalData: [{ path: '<where>', data }] }, or export with the weights inside it (PyTorch: external_data=False).` (Step 8's message, with the first way out added.) |
| Outside the data file | `mlpFromOnnx: initializer "<n>" reads <length, or "the rest"> B at offset <offset> of "<where>", which holds <size> B.` (as built; the offset and length as the file writes them) |
| Wrong length | `mlpFromOnnx: initializer "<n>" is [<dims>] but holds <m> values.` (as built: step 8's message for the same fault inside the file) |
| Bad `externalData` | `mlpFromOnnx: externalData must be a list of { path, data }, with data a Uint8Array or an ArrayBuffer; a URL is for ONNX Runtime, which fetches it.` |

All but "A run failed" are thrown at construction, load, bind, `setBrain`, `prepareBrain` or
`mlpFromOnnx`. "A run failed" is thrown once, by the first `advance()` or `step()` after the
failure; nothing is checked per frame beyond one field.

## 12. Engine touchpoints

### 12.1 Core — required by §14, item 3: `createComputeStageAsync`

Device adoption ships (fact 8), so ORT first needs nothing. **The parent's §12.3
(`requiredLimits`) is withdrawn:** S5 found a device asked for with ORT's own limits fails like the
engine's, so no limit would help.

§14, item 3 needs one method, added beside `createComputeStage` (`NullGraph.ts:128`,
`BatchManager.ts:172-185`), which it mirrors:
`engine.createComputeStageAsync(config): Promise<ComputeStageNode>`. The same validation and the
same stage; the pipeline from `createComputePipelineAsync`; registered like the synchronous one.
A compile error rejects, naming the stage (§11). Nothing existing changes. Estimate: +60–100 B
min+gzip on core, measured in phase 9.5.

**Zero-change fallback:** `prepareBrain` compiles synchronously at load. The stall moves from the
switch to the load, where it freezes the page instead: 76 ms in Chrome on AMD to 1.1 s under Node
on NVIDIA, per brain.

### 12.2 Agents — required: async brains in `AgentSystem`

`attach` binds an async brain before building its pass; `advance()` starts runs; `step()` splits
around them; `destroy()` waits to destroy `obs`; `prepareBrain` keeps prepared passes. Every
rule-only and `MlpBrain` app pays their bytes, because they sit in `AgentSystem`'s methods
(estimate: 400–700 B min+gzip for async brains, 150–250 B for `prepareBrain`).

**Zero-change fallback:** the app runs ORT itself, writing a buffer and copying it into `act` with
a step stage. It works, and it costs: per-species brains and `thinkEvery` re-done by hand, no
lockstep (the app cannot split `step()`), and every trap of §9 met again by every app.

### 12.3 Package — required: an optional peer dependency, and no ORT types

- `peerDependencies: { "onnxruntime-web": "^1.29.0" }` with `peerDependenciesMeta` marking it
  optional, so npm does not install it.
- The library declares the few members of `ort` it uses as its own structural type, `OrtLike`.
  Its declarations never import `onnxruntime-web`, so an app that imports only `mlpFromOnnx` needs
  no ORT, not even its types.

### 12.4 Bundle

- **`inference.js`:** the ORT adapter and `OrtBrain` are paid only by apps that import them;
  `mlpFromOnnx` alone tree-shakes them away. Estimate 1.5–2.5 KB min+gzip. A line of its own,
  3,000 B; phase 9.1's adapter is 1,925 B. The reader's own line moves from 2,600 B to 2,900 B for
  external data (§6.1).
- **`agents.js`:** 19,323 B of C2's 21,495 B today. Async brains and `prepareBrain` must fit, with
  no new allowance.
- **Core:** `createComputeStageAsync`, +60–100 B (§12.1).
- If any line is crossed, the numbers come back to the user. Sizes are measured as phase 6.3
  measured them (`node test/spikes/mlp-think-cost/size.mjs`), with a line added for core.
- **ORT itself:** 0 B in every bundle.

## 13. Build order

| Phase | Builds | Tests | Commit |
|---|---|---|---|
| 9.0 ✅ | Spikes S1, S2, S4, S5 (`test/spikes/ort/`, `npm run spike:s1` … `spike:s5`); this design, with the user's decisions | — (measurements) | the spikes and the design |
| 9.1 ✅ | `OrtRuntime`, `OrtModel`, the anchor, ORT's `externalData` passed through; the peer dependency; `OrtLike`; the adapter's size line | U32, A29, U14 (their errors) | one |
| 9.2 ✅ | Weights outside the file in the reader: `make.py` keeps the data files in `fixtures.json`; `mlpFromOnnx(bytes, { externalData })`; the reader's new line; both paths on the GPU | U36, U30–U31 updated, A32 | one |
| 9.3 ✅ | Async brains in `AgentSystem` (attach, `advance`, `step`, `destroy`); `OrtBrain` and its copy pass | U13, U33, A5, A30, A31 | one |
| 9.4 ✅ | The interactive schedule on the GPU: A6 on three hosts, with the latency recorded. Added: capture off by default (§14, item 6), and faults 3 and 4 in `spike:ort-issues` | A6, U32 | one |
| 9.5 | Prepared brains: core's `createComputeStageAsync`, then `agents.prepareBrain`; core's size reported | U34, U35, A33 | one |
| 9.6 | The guides: `docs/api/inference.md` gains ORT and external data, with checked examples; the agents guide gains `prepareBrain`; the parent's facts 15–16, §9.4, §11 and §12.3 | the examples' tests | one |

Each phase leaves every suite green. Every new test is watched failing first.

### Deferred

- **The site's engine-first worker** and the live ORT switch: step 10's design (§5).
- **A run watchdog** (non-goals).
- **Other tensor types, CPU tensors, several outputs** for the generic model.
- **The older JavaScript provider** as a tested path.
- **Re-measuring S5 on a newer ORT;** if an app-made device runs, engine-first can return as an
  option, detected, not assumed.

## 14. Decisions

### Decided by the user, 2026-10-14

Asked after phase 9.0 with the options below, the user answered: "ORT creates the device, graph
capture on, prepare at load compiled in the background, and both paths read separate weight
files."

1. **Device ownership (the parent's decision 6): ORT creates the device; the engine adopts it.**
   S5 had settled the question: ORT 1.29 runs only on its own device. The parent's engine-first
   direction and its §12.3 are withdrawn; its fact 15 and its device error are rewritten (§11,
   phase 9.6).
2. **Graph capture: on by default** (option A, the parent's plan). *Superseded after phase 9.4 by
   item 6: off by default.*

   | Option | For | Against |
   |---|---|---|
   | **A. On** (taken) | A third or less of the main thread per run; the same GPU time; every trap found is closed by construction (§9) | More of ORT's code in use; a trap not yet found would surface as a failed run |
   | B. Off | The simplest path through ORT | 0.4 ms (Node) to 0.7 ms (Chrome) more of the main thread per run, every frame |

3. **The brain-swap stall** (deferred to step 9 on 2026-10-12): **prepare at load, compiled in
   the background.** Option B's `prepareBrain`, built on option C's asynchronous stage creation,
   which is a core addition (§8.6, §12.1).

   | Option | What | Cost |
   |---|---|---|
   | A. Accept the hitch | Nothing built | A hitch on every switch, in step 10's live demo too |
   | B. Prepare at load | `agents.prepareBrain(brain)` compiles a brain's pass at load time and keeps it while the system lives; `setBrain` then compiles nothing | Agents only, estimate +150 B; the stall moves to load time |
   | C. Asynchronous stage creation | A core method over `createComputePipelineAsync` | A core change |
   | **B over C** (taken) | `prepareBrain` compiles through the core method, so neither the load nor the switch stalls | Both costs: agents +150–250 B, core +60–100 B |

4. **Weights stored outside the `.onnx` file** (the user, earlier the same day: "since onnx is
   step 9 to test running larger models then we can rethink this"): **both paths read them**
   (option B, §6.1).

   | Option | What | Cost |
   |---|---|---|
   | A. ORT reads them; the reader keeps refusing | `runtime.load(bytes, { externalData })` passes ORT's own option through | About 20 B and A32 |
   | **B. Both read them** (taken) | A, and `mlpFromOnnx(bytes, { externalData })` with the same list | About +200 B on the reader, whose line moves to 2,900 B; a data file in the fixtures |
   | C. Neither | Both refuse | ORT users must re-export larger models too |

### Decided by the user after phase 9.2, 2026-10-14

5. **Graph capture for a model whose weights are in a second file: off unless asked** (option A;
   the user: "ok pick A"). Item 2 was decided on the ground that every trap found was closed by
   construction. Phase 9.2 found one that is not: under capture, ONNX Runtime 1.29 can fail such a
   model from its second run (`Cannot set properties of undefined (setting
   'Symbol(gpuBufferMetadata)')`), on Node/NVIDIA, Node/AMD and Chrome/AMD, depending on the model
   and on what ran before it. The failure is loud: the run rejects.

   | Option | What | Cost |
   |---|---|---|
   | **A. Off for such models** (taken) | `load` turns capture off when `externalData` is given, unless the app passes `capture: true` | 4 B as built. Those models pay 0.4–0.7 ms more of the main thread per run; every other model keeps capture |
   | B. On for all, as decided | Nothing changes; the guide names the failure and `capture: false` | An app that loads a two-file model can see its second run fail |
   | C. Off for all | Capture only on request | Every model pays the main-thread cost |

   Built before phase 9.2's commit: U32 checks the default, the request and an empty list; A32
   takes the default. *Superseded after phase 9.4 by item 6.*

### Decided by the user after phase 9.4, 2026-10-14

6. **Graph capture: off by default for every model** (option A, which the user chose from the
   options below). A6 failed inside the `inference` suite: the marker model's second run threw
   item 5's error, after a second session had been created, run and released between its first
   two runs. Phase 9.4 traced it into ONNX Runtime's JavaScript (Phase 9.4 results, fault 3):
   under capture, every run from the second looks up the output buffer through a handle released
   after the first, so any captured model can fail, depending on what ONNX Runtime did in between.
   On 1.30.0, which the peer range `^1.29.0` admits, the smallest case failed with nothing in
   between (Node/NVIDIA). Capture also leaks memory with every run (fault 4). Item 5's carve-out
   treated a symptom.

   | Option | What | Cost |
   |---|---|---|
   | **A. Off for every model** (taken) | `capture: true` stays as the app's choice, documented with the faults; item 5's special case goes | 0.3–0.5 ms more of the main thread per run under Node, about 0.7 ms in Chrome (S1); the same GPU time. −6 B as built |
   | B. On, and ONNX Runtime pinned to 1.29.x | Avoids 1.30.0's failure with nothing in between | Apps still meet the fault on 1.29 when they load or release a second model, as step 10's live switch will; the leak stays |
   | C. On, as decided | Nothing changes | A captured model can fail from its second run on 1.30.0, and on 1.29 after ordinary steps of an app |

   One statement in the question was too strong: it said 1.30.0 fails "always". Measured, the
   smallest case failed with nothing in between on Node/NVIDIA (3 of 3 processes) and not on
   Node/AMD (2 of 2), and captured models ran inside the `inference` suite on 1.30.0 (Phase 9.4
   results). The choice does not rest on it: B and C keep a fault that 1.29 shows as well.

   Built in phase 9.4: `load` captures only with `capture: true`; U32 checks the default and the
   request; A29's two captured models ask for capture, and its third takes the default; A6, A30,
   A31 and A32 take the default; A5 runs both.

**Nothing is open.**

---

## Phase 9.0 results — 2026-10-14: what ONNX Runtime costs, where its runs land, and whose device it runs on

**Where:** `test/spikes/ort/` (`common.mjs`, `s1.js`, `s2.js`, `s4.js`, `s5.js`, `index.html`);
`npm run spike:s1` … `spike:s5` under Node, with `NG_DAWN_FLAGS=adapter=AMD` for the iGPU; in
Chrome, `npm run drive -- --serve . "/test/spikes/ort/index.html?spike=s1" --wait-for "S1 done"`.
**What:** ONNX Runtime Web 1.29.0, its WebGPU bundle (`ort.webgpu.bundle.min.mjs`, the provider
compiled into the WASM) unless stated; the parent's 24 → 32 → 32 → 2 tanh network, written by the
step 7 writer. **Hosts:** Node/Dawn on NVIDIA Turing and on the AMD GCN-5 iGPU; Chrome on the AMD
iGPU (Chrome on Windows ignores `powerPreference`).

### S1 — the cost of a run

**Method:** GPU buffers in and out, the output pre-allocated; medians of 10 runs. ORT's passes
cannot hold our timestamps, so its GPU time is **bracketed**: a long WGSL pass (the blocker) is
submitted first, the timed work queues behind it, and the time runs from the blocker's end to the
work's end. The WGSL kernel is timed both ways: on it, the bracket reads 12–45 µs above direct
timestamps on NVIDIA and 56–147 µs above on AMD, the cost of starting work after the blocker. So
**the ratios compare bracket with bracket.** Every run matched `mlpForward` within 3.6e-7, and the
WGSL kernel within 4.2e-7.

| GPU µs per run, 1,000 / 20,000 / 100,000 agents | WGSL, direct | WGSL, bracketed | ORT, bracketed | ORT ÷ WGSL, bracketed, at 20,000 / 100,000 |
|---|---|---|---|---|
| Node · NVIDIA, fast clocks (one run) | 37 / 168 / 836 | 53 / 180 / 848 | 52–56 / 515–518 / 2,428 | 2.9× / 2.9× |
| Node · NVIDIA, slow clocks (three later runs) | 45–46 / 393–403 / 1,294–1,311 | 70–74 / 418–422 / 1,339–1,352 | 75–79 / 1,298–1,301 / 6,016–6,026 | 3.1× / 4.5× |
| Node · AMD iGPU | 34 / 148 / 533 | 111 / 213 / 680 | 105–193 / 1,540–1,648 / 6,580–6,816 | 7.2–7.7× / 9.7–10.0× |
| Chrome · AMD iGPU | 38 / 159 / 632 | 134 / 225 / 688 | 140–178 / 1,533–1,684 / 6,716–6,855 | 6.8–7.5× / 9.8–10.0× |

- **Graph capture changes no GPU time;** the ORT ranges cover capture on and off.
- **NVIDIA's clock state moved everything about 2.5×** between the first run and three later ones;
  only ratios inside one run compare. S3's AMD figures (281 µs at 20,000) were timed another way,
  and are about twice S1's direct figures for the same kernel.
- **At 1,000 agents** ORT is level with the bracketed kernel on NVIDIA and up to 1.7× on AMD:
  starting work dominates there.
- **Main-thread time of `run()`:** capture off 0.57–0.69 ms (Node), 0.74–0.83 ms (Chrome); capture
  on 0.20–0.28 ms (Node), 0.06–0.09 ms (Chrome). The WGSL kernel's submit: 0.01–0.14 ms.
- **From `run()` to `onSubmittedWorkDone`** at 20,000: 0.75–2.24 ms under Node; about 8 ms in Chrome
  for ORT and the WGSL kernel alike, Chrome's cadence for that promise.
- **A session's first run** compiles ORT's pipelines: 401–445 ms for the first session of a process
  under Node, 159 ms in Chrome; 2–31 ms for later sessions. Creating a session: 2–7 ms.

### S2 — where a run lands

**Method:** 20,000 agents; a frame starts a run when none is in flight, then submits its stages. A
sense stand-in writes the frame's number into every first observation; a marker network (relu,
relu, identity) carries it to the first action; a probe in act's place records what it finds. 300
frames at a 60 fps timer and 300 as fast as they come; then 60 lockstep steps; then the swap test.

| | Node · NVIDIA | Node · AMD | Chrome · AMD |
|---|---|---|---|
| Frames whose actions were 1 frame old, of 300 (capture off / on) | 299 / 299 at 60 fps; 298 / 299 fast | the same | 298 / 299 at 60 fps; 293 / 299 fast |
| Older actions (all with capture off) | fast: one frame, 2 old | fast: one frame, 2 old | 60 fps: one frame, 2 old; fast: one run spanned 7 frames (6 frames 2–7 old) |
| Actions from the same frame or a later one | 0 | 0 | 0 |
| Lockstep steps acting on their own observations | 60 / 60 (2.2–2.5 ms a step) | 60 / 60 (2.5–2.9 ms) | 60 / 60 (8.3 ms) |
| A later run given another input buffer | capture off: read it; **capture on: ignored it** | the same | the same |

A run always read the observations of the frame that started it, and its actions were first used
by the next frame. Under Node, the "60 fps" timer gave 31–32 ms frames (Windows' timer resolution).

### S4 — the cost of starting

**Method:** one cold process or page per run, three runs per host; Chrome loads from localhost.

| ms | Node · NVIDIA | Node · AMD | Chrome · AMD |
|---|---|---|---|
| Loading the bundle | 7.8–10 | 7.4–10.6 | 25–31 |
| Reading the WASM from disk (Node) | 16–25 | 15–29 | — |
| Asking for an adapter | 341–385 | 335–420 | 87–92 |
| The first session (WASM compiled and started, device made) | 525–685 | 491–631 | 543–567 |
| Its first run (ORT's pipelines compiled) | 418–451 | 417–458 | 142–209 |
| **From the first line to the first actions** | **1,393–1,577** | **1,315–1,586** | **831–917** |
| A second run | 1.7–2.2 | 2.3–3.7 | 7.3–8.6 |
| A second session; its first run | 5.8–8.7; 7–46 | 5.1–5.6; 36–41 | 3.7–4.7; 15–20 |
| Built-in path: `mlpFromOnnx`; its WGSL pipeline | 1.4–1.6; 52–54 | 1.5–1.6; 47–49 | 1.3–1.9; 163–166 |

**Sizes:** the WASM 25,749,873 B (6,368,495 B gzipped); its loader 51,407 B (18,828 B); the bundle
116,102 B (38,738 B).

### S5 — whose device

**Method:** the device made four ways (the engine's, as `NullGraph.init` makes it; the engine's
with WebGPU's default limits; one asked for exactly as ORT asks for its own; ORT's own, the
control), handed over through the provider's `device` option and `ort.env.webgpu.device`; one
session with capture off and one with it on, three runs each, each checked against `mlpForward`;
then the WGSL kernel on the same device, to see that the device survived.

| Device | Node · NVIDIA | Node · AMD | Chrome · AMD |
|---|---|---|---|
| The engine's | sessions made; **every run failed**: `No cached or pending pipeline for deferred dispatch: Tanh:1:1\|:2;:` | the same | stopped answering, 3 times of 3; the run with progress lines stopped after the capture-on session's third run began |
| The engine's, default limits | the same failure | stopped answering, 2 times of 2, at the same place | the same failure |
| Asked for like ORT's own | the same failure | the same failure | the same failure |
| ORT's own (control) | runs, both sessions | capture off runs; **the capture-on session, made after the first was released (a new device), failed its 2nd and 3rd runs**: `Cannot set properties of undefined (setting 'Symbol(gpuBufferMetadata)')` | runs, both sessions |
| The `all` bundle, the engine's device | does not load under Node/Dawn | — | device ignored: ORT made its own, the app's buffers were not written, 24 WebGPU errors |
| The `all` bundle, its own | — | — | runs; its device outlives its sessions |

- **The app's device stayed sound** in every case where ORT answered: the WGSL kernel ran on it
  afterwards and matched `mlpForward`.
- **The WebGPU bundle destroys its device with its last session** (`AbortError` on the next
  read-back, everywhere). The `all` bundle keeps its device.
- The AMD control failure happened 4 times of 4, and again with the final script. It did not
  happen when another session stayed alive across the release, nor with the marker network, nor
  on NVIDIA.

### Also found

- **The provider as a string with capture on** half-captures (fact 10): a later run read a new
  buffer, and S2's frame loop hung the process. S2 keeps that variant behind `string=1`.
- **A captured run without a pre-allocated output** fails: `no overload matched for createBindGroup`.
- **Dawn:** a submission that uses a buffer still in use by queued work takes milliseconds; with
  about 30 submissions queued, the next one waits. S1's bracket queues 16 at most.

### What phase 9.0 changed in the plan

- **The device:** ORT first only; the anchor session; §12.3 withdrawn; the device error rewritten
  (§1, §5, §11).
- **The results buffer and the copy pass,** instead of ORT writing `act`: ORT writes every row, so
  per-species brains and `thinkEvery` need a pass between it and `act` (§7, §8).
- **Graph capture stays on by default,** with four traps closed by construction (§9).
- **The gate** (under ~1 ms at 20,000) is met only on NVIDIA at its fast clocks. `thinkEvery`
  becomes ORT's run rate, and the guide gives the costs (§8.3).
- **Not run:** S1 with the network's real Herd & Wolves weights (S1 uses random weights; the cost
  does not depend on them); any host but these three; Chrome on NVIDIA (Chrome on Windows always
  took the iGPU).

## Phase 9.1 results — 2026-10-14: the runtime, the model and the anchor

**Built:**
- `src/extras/inference/OrtRuntime.ts`, exported from `null-graph/inference`:
  - `OrtLike`, the members of `ort` used (with `OrtSessionLike` and `OrtValueMetadataLike`);
  - `OrtRuntime`: `create(ort, { adapter })`, `load(bytes, { capture, externalData })`,
    `destroy()`, `device`;
  - `OrtModel`: `inputs`, `outputs`, `device`, `capture`, `bind`, `run`, `destroy`;
  - the anchor's model: y = x + x on one float32, 66 bytes, kept as base64 in the source and
    decoded only by `create`, so that an unused import leaves nothing behind.
- `package.json`: `onnxruntime-web` `^1.29.0` as an optional peer dependency; it stays a
  devDependency at 1.29.0. `npm run test:inference` runs the new file.
- `test/inference-ort.test.mjs`: U32, U14 for phase 9.1's refusals, the size line, and a type
  check. The type check compiles, strict, an app that passes the real `onnxruntime-web/webgpu`
  namespace as an `OrtLike`.
- `test/inference-gpu/` (the GPU suite `inference` in `test/gpu/suites.mjs`): A29.

**Measured:**

| | |
|---|---|
| U32, U14, size, types | 8 of 8; with the reader's 8, `npm run test:inference` passes 16 of 16 |
| A29 | 6 of 6, zero WebGPU errors, on Node/NVIDIA, Node/AMD and Chrome/AMD. On AMD, the captured model loaded after the release ran right at runs 1, 2 and 100: worst difference from `mlpForward` 3.1e-7 |
| `OrtRuntime.create` | 502 ms on Node/AMD (S4's first session: 491–631 ms) |
| Sizes, min+gzip | `OrtRuntime` with `OrtModel` 1,925 B (line 3,000 B); the whole inference entry 4,268 B; `mlpFromOnnx` alone 2,527 B (2,525 B before: the minifier's names; its bundle holds no adapter code); the agents entry unchanged at 19,323 B |

**Watched failing.** Ten mutations of the built adapter, one at a time, each caught by the test
meant for it:
- the provider given as a string; capture off by default; the app's session options passed
  through;
- the anchor released before the models; `destroy` not waiting for a run in flight;
- a second `bind` allowed; no check of named axes; no `STORAGE` check;
- two runs in flight. At first this was caught only by the test runner's 30 s limit, because the
  fake's second run never settles; the test now races it against a tick and fails in 5 ms;
- no metadata check.

On the GPU, the anchor released at once fails 6 of A29's checks on AMD. The engine's buffers then
belong to a destroyed device, and every run fails.

**Differs from the design, or adds to it:**
- Three refusals added to §11: bytes that are not bytes; a model ORT cannot load, with ORT's first
  line; an ORT that reports no metadata.
- `OrtModel` also carries `device` and `capture`, which `OrtBrain` will read (phase 9.3).
- A failed run passes on ORT's own error unchanged; `AgentSystem` adds the species in phase 9.3.
- An axis that the file leaves free without a name (ORT reports it as -1) takes any size; a tensor
  whose shape the file leaves out takes any dims.
- After `runtime.destroy()`, the device is gone on all three hosts (logged, not checked): ORT
  destroys it with the anchor.

**Also green:** headless agents (reference + 73), core, agent-kit (4); `onnx` 23/23 on
Node/NVIDIA.
**Not run:** `onnx` on AMD and in Chrome (its code did not change; the entry it imports grew); the
agents GPU suite (nothing it runs changed).

## Phase 9.2 results — 2026-10-14: weights in a second file, read both ways

**Built:**
- `mlpFromOnnx(bytes, { externalData })` (`src/extras/inference/mlpOnnx.ts`), with the types
  `OnnxReadOptions` and `OnnxExternalData`. The list is ONNX Runtime's own shape, with bytes as
  `data`. A tensor stored outside is found by its location exactly as written; its offset and
  length are read as text, the length defaulting to the rest of the file; the memory budget grows
  by the files given. Step 8's refusal now names both ways out.
- `test/onnx-fixtures/make.py` keeps the files an export writes beside the `.onnx` file, as base64
  with their size and SHA-256, and judges such a case with them in place. Two cases have one:
  - `torch-outside`, step 8's case: `Linear(13, 5)`, a fixed batch of one, its weights (260 B) in
    `model.onnx.data`;
  - **`torch-outside-batch`, new and last in the list:** 13→17 tanh, 17→11 relu, 11→3, a free batch.
    PyTorch put two weight tensors in `model.onnx.data`, at offsets 0 and 748, in the reverse of the
    layers' order, and left the third, of 33 floats, inside.

  Every other case is unchanged. PyTorch's default call also writes an empty `model.onnx.data`
  when every tensor fits inside; such a file is not kept.
- Tests: U36 (new, in `test/inference-onnx.test.mjs`); U31 reads both two-file cases bit for bit
  and refuses each without its file; U14's message updated; the reader's line moved to 2,900 B.
  On the GPU, A28 (`test/onnx-gpu`) reads both two-file cases into `MlpBrain` and runs them in ONNX
  Runtime with the same list; A32 (`test/inference-gpu`) runs them through `OrtRuntime.load(bytes,
  { externalData })`.
- `test/spikes/ort/issues.js` (`npm run spike:ort-issues`): the two faults below in their smallest
  form, for an upgrade of ONNX Runtime and for a report to its authors.
- The guide's reader section (`docs/api/inference.md`): a second file is read when given; its
  example takes `externalData`, and its test reads `torch-outside-batch` with its file.

**Measured:**

| | |
|---|---|
| Headless | `npm run test:inference` 17 of 17; U31: 15 files read with their makers' weights bit for bit (2 with their data files), `mlpForward` within 7.2e-7 of the makers' outputs |
| A28 | 25 of 25 on Node/NVIDIA, Node/AMD and Chrome/AMD; both two-file cases agree with ONNX Runtime within 1e-4 |
| A32 | 8 of 8 in the `inference` suite on the same three pairs, zero WebGPU errors: against PyTorch 4.8e-7 (`torch-outside`) and 3.7e-8 (`torch-outside-batch`); against the reader 6.0e-7 and 8.9e-8 |
| Sizes, min+gzip | `mlpFromOnnx` 2,848 B, line 2,900 B (the estimate was +200 B; it took +321 B); the adapter 1,984 B, line 3,000 B (+55 B for the 16 B rule, +4 B for item 5's default); the whole inference entry 4,635 B; agents unchanged at 19,323 B |

**Two faults of ONNX Runtime 1.29, found by A32:**
1. **Its kernels read and write 16 B at a time.** Bound to an app's buffer of exactly its values'
   size, a model whose last write ends past the buffer loses it on NVIDIA: a 13→5 `Gemm` and
   `Tanh` left the fifth value as it was, on every run, with no error, with graph capture on or
   off. AMD wrote it. Buffers rounded up to 16 B are right everywhere. **Guarded:** `OrtModel.bind`
   asks for the values rounded up to 16 B (§6). A28, S1 and S2 never met it: their buffers were
   multiples of 16 B.
2. **Under graph capture, a model whose weights are in a second file can fail from its second run**
   with `Cannot set properties of undefined (setting 'Symbol(gpuBufferMetadata)')`, the error spike
   S5 saw on AMD on a re-made device. ONNX Runtime's JavaScript recognises a pre-allocated output
   only while binding, on the first run; on later captured runs it gets back an output that is not
   one of the registered buffers (`onnxruntime-web/lib/wasm/wasm-core-impl.ts:781,800,876,954`).
   - Seen with one `Gemm` 13→5 whose weights are in the second file: in a fresh process, with a
     fixed batch on Node (both GPUs), with a free batch in Chrome. With a node after the `Gemm`, or
     with the weights inside the file, it ran everywhere.
   - In the `inference` suite, after other models, the same files ran under capture; the probe at
     its end reports which happened. **Avoided by default,** decided by the user after the phase
     (§14, item 5): `load` leaves capture off for a model given `externalData`, unless asked. A32
     takes that default; U32 checks it, and a mutation back to "always on" fails U32.
   - **Phase 9.4 found the cause** (its fault 3): the lookup above uses a handle ONNX Runtime
     released after the first run, so the second-file models were very likely one case of a fault
     that any captured model can meet. Capture is now off for every model (§14, item 6).

**Watched failing.** Seven mutations of the built reader and adapter, each caught: the offset
ignored (U36, and U31 through PyTorch's offset 748); any file taken for a location not given; no
bounds check; the budget not counting the files; a missing length read as zero; `externalData` not
checked; buffers not rounded up to 16 B (U32).

**Differs from the design:**
- The reader's "wrong length" refusal is step 8's message for the same fault; §11 updated.
- `OrtModel.bind`'s rule for sizes (§6, §11) and the results buffer's (§4, §7) are new: fault 1.
- Capture's default for two-file models (§14, item 5) is new. The guide's reader section changed
  in this phase rather than in 9.6, because the behaviour it described changed here.

**Also green:** headless agents (reference + 73), core, agent-kit (4).
**Not run:** the agents GPU suite (nothing it runs changed).

## Phase 9.3 results — 2026-10-14: OrtBrain, and async brains in AgentSystem

**Built:**
- `src/extras/inference/OrtBrain.ts`, exported from `null-graph/inference`: `new OrtBrain(model,
  { input, output })`, as §7 has it. Its think pass is the copy pass of §4. `bind` makes a results
  buffer rounded up to 16 B, seeds it from `act`, then binds the model; the same system's buffers
  again re-seed it; another system's are refused. `destroy` waits for a run in flight.
- `AgentSystem` (`src/extras/agents/AgentSystem.ts`), as §8 has it:
  - `attach` binds an async brain before it builds the brain's think pass;
  - `advance()` starts a run when the brain has none in flight and `thinkEvery` steps have passed
    since its last one started, never in a frame without a step;
  - a failed run is thrown once, by the next `advance()` or `step()`, naming the species, and its
    brain starts no run after;
  - `step()` waits for an interactive run in flight, renders grid, queries and sense, awaits every
    async brain's run, then renders the passes, act, app stages, tick and instances;
  - `destroy()` releases the buffers once every run in flight has settled;
  - a brain detached while its run is in flight and attached again keeps that run: the next
    `advance()` or `step()` waits for it rather than start a second (found at review, before the
    commit; ONNX Runtime refuses a run while one is in flight, which would have failed the species);
  - `obs` is rounded up to 16 B (§6).
- The brain contract (`RuleBrain.ts`) documents `bind` and `run` for async brains.
- Tests: U13, six tests in `test/agents-system.test.mjs`; U33, four tests in
  `test/inference-ort.test.mjs`; U17's refusal updated. A5, A30 and A31 in `test/inference-gpu`.

**Measured:**

| | |
|---|---|
| Headless | agents (reference + 79), inference 21/21 |
| A5 | 4,096 live agents, `thinkEvery: 1`, after 60 steps of the rule. Against `mlpForward`: prey 4.0e-6 to 1.1e-5, wolves 4.8e-7 to 1.9e-6. `OrtBrain` against `MlpBrain` on the same step: 4.0e-6 to 1.1e-5. Graph capture on and off alike; wolves on `MlpBrain` the same both times. On Node/NVIDIA, Node/AMD and Chrome/AMD |
| A30 | 65,536 words of agent records, 0 differ, on all three |
| A31 | seeded at both attaches; after 30 frames, 8,167 to 8,192 of 8,192 result words had moved from the seed, on all three |
| The `inference` GPU suite | 12/12 on all three, zero WebGPU errors |
| The agents GPU suite | 137/137 on all three, zero WebGPU errors: `AgentSystem` changed, and every brain, sensor and stage it ran before still runs |
| Sizes, min+gzip | the agents entry 19,894 B, of C2's 21,495 B: +571 B for async brains (estimate 400–700 B); the rule path 14,144 B (+558 B). The re-attach fix adds 38 B (19,899 → 19,937 B, measured with esbuild's ESM bundle of `dist/agents.js`), about 19,932 B by the measure above. The adapter with `OrtBrain` 2,994 B, under its line of 3,000 B, after trimming (below) |

**Watched failing.** Ten mutations of the built agents and inference entries, one at a time, each
caught by the test meant for it:
- U13: a run started while one is in flight; a re-attached brain forgetting its run in flight; runs every frame, `thinkEvery` ignored; a run in a
  frame without a step; lockstep acting before the run; `obs` released with a run in flight; a
  failed run never thrown;
- U33: the copy pass reading the next agent's row; no seeding; the results buffer not rounded up.

On the GPU (the `inference` suite, Node/AMD): the copy pass reading the next agent's row fails both
of A5's checks; no seeding fails A31.

**Differs from the design, or adds to it:**
- `thinkEvery` counts the steps since a run started, those taken while it was in flight included;
  U13 pins it.
- Two refusals added to §11; the "failed in a run" message drops the last period of ORT's line.
- As first built, the adapter with `OrtBrain` was 3,049 B. It fits its line after three changes no
  user sees: fields without a value of their own are declared only (no code is emitted for them),
  the copy pass's WGSL lost its indentation, and the seed copy's command encoder its label. No
  message was shortened. The margin is 6 B.

## Phase 9.4 results — 2026-10-14: the interactive schedule on the GPU, and two faults of graph capture

**Built:**
- **A6** in `test/inference-gpu/run.js` (the GPU suite `inference`): spike S2's marker method,
  through `AgentSystem` and `OrtBrain`.
  - 20,000 agents of one species, `thinkEvery: 1`. A custom sensor `marker` writes 1,000 + the
    step's tick into every agent's first observation. The marker network (24 → 32 → 32 → 2, relu,
    relu and identity, one weight of 1 per layer: Herd & Wolves' shape) carries a first
    observation to the first action unchanged. A step stage `probe`, after act, records for the
    first and the last agent what they act on and what sense wrote. A value under 1,000 is a seed.
  - One lockstep step first: its run compiles ONNX Runtime's pipelines. Then 300 frames at the
    display's pace (`requestAnimationFrame` in Chrome; a 60 Hz timer under Node, 31 ms frames) and
    300 as fast as they come (`setTimeout(0)`). In each half the inspector reads an agent back at
    frames 50, 150 and 250.
  - For the frames, `GPUBuffer.prototype.mapAsync` is wrapped. The inspector's staging buffer is
    counted apart; any other call is a fault. First, a direct ONNX Runtime session with its output
    on the CPU runs once under the wrapper: the control, which shows that the count sees ONNX
    Runtime's own calls. `uncapturederror` is counted on the device, and each run's starting frame
    is recorded by wrapping the brain's `run()`.
  - Three checks. No `mapAsync` outside the inspector and no GPU error, with the inspector's six
    and the control's one counted. Then, per pace: no frame acts on its own or a later frame's
    observations, nor on a seed; the first and the last agent act on the same run (no tearing);
    every run read the observations of the frame that started it; and frames act on actions one
    frame old, 90% of them at least at the display's pace, more than half as fast as they come.
- **The capture default** (§14, item 6): `OrtRuntime.load` captures only with `capture: true`.
  U32 checks the default and the request. A29's two captured models ask for capture; its third
  takes the default.
- **`npm run spike:ort-issues`** (`test/spikes/ort/issues.js`) gains faults 3 and 4: fault 3's
  four cases, run first in a fresh runtime, or one alone with `NG_SPIKE_CASE`; fault 4 with
  `NG_SPIKE_LEAK=1`.
- **`NG_ORT_DIST`** (Node): the `inference` suite reads another ONNX Runtime build's `dist`
  folder. Used for 1.30.0.

**Measured, A6, ONNX Runtime 1.29.0:**

| | Node · NVIDIA | Node · AMD | Chrome · AMD |
|---|---|---|---|
| `mapAsync` outside the inspector; GPU errors | 0; 0 | 0; 0 | 0; 0 |
| Counted: the inspector's; the control's | 6; 1 | 6; 1 | 6; 1 |
| Display's pace: a frame (median) | 31.7 ms | 31.6 ms | 16.7 ms |
| Frames by the age of their actions | 1: 300 | 1: 300 | 1: 299, 2: 1 |
| Runs in 300 frames; `run()` resolved in (median, most) | 300; 1.72, 3.26 ms | 300; 1.58, 2.46 ms | 299; 0.91, 4.65 ms |
| As fast as they come: a frame (median) | 1.4 ms | 1.3 ms | 4.6 ms |
| Frames by the age of their actions | 1: 300 | 1: 300 | 1: 295; 2, 3, 4, 5, 6: 1 each |
| Runs in 300 frames; `run()` resolved in (median, most) | 300; 1.01, 4.88 ms | 300; 0.85, 34.95 ms | 295; 0.58, 1.84 ms |

- Every run a frame acted on had read the observations of the frame that started it. No frame
  acted on its own frame's observations, and the first and the last agent always acted on the
  same run.
- In Chrome, as fast as frames come, A6 met S2's stall again: one run spanned 6 frames, which
  acted on actions 2 to 6 frames old, and the next run caught up. An earlier Chrome run showed the
  same with 7 frames.
- **ONNX Runtime 1.30.0** (Node/NVIDIA, `NG_ORT_DIST`): the `inference` suite 15/15; A6 1: 300 at
  both paces.

**Two faults of ONNX Runtime under graph capture, found by A6:**

3. **From a captured session's second run on, its output is looked up through a handle ONNX
   Runtime has released.** On the first run, ONNX Runtime's JavaScript registers the bound buffers
   in a handle table, binds them, and releases the handles at the run's end; the binding keeps
   them. On every later run it reads the output tensor's data pointer, that handle, and looks the
   buffer up with it (`webgpuGetBuffer(dataOffset)`, `lib/wasm/wasm-core-impl.ts:939-954`; the
   table in `ort-wasm-simd-threaded.asyncify.mjs`). The lookup finds a buffer only when the
   allocator has put another buffer's handle at that address since. Otherwise the run rejects
   with `Cannot set properties of undefined (setting 'Symbol(gpuBufferMetadata)')`, after its GPU
   work was queued: only the bookkeeping fails.
   - **How A6 met it:** inside the suite, with capture on (the default then), the marker model's
     second run failed. Between its first two runs the control session had been created, run and
     released. The same sequence in a fresh process ran.
   - **When it does not fail,** the lookup finds the session's input buffer: after a second run,
     `obs` carried ONNX Runtime's registration and the output buffer none. Harmless for the
     results; wrong all the same.
   - **The cases** (1.29.0; five runs, the case between the first two; each in a fresh process
     with `NG_SPIKE_CASE`):

     | Between the first two runs | Node · NVIDIA | Node · AMD | Chrome · AMD |
     |---|---|---|---|
     | 1. Nothing | ran | ran | ran |
     | 2. Another session created and released | run 2 failed | ran | runs 2–5 failed |
     | 3. Another session created, run and kept | ran | runs 2–5 failed | runs 2–5 failed |
     | 4. Case 3, capture off | ran | ran | ran |

     In one process, in that order: cases 2 and 3 failed runs 2–5 on Node/NVIDIA and in Chrome;
     none failed on Node/AMD. Fault 2's cases, run after them, all ran.
   - **On 1.30.0,** which has the same code at both places, a temporary probe of the same model
     (with an anchor session of its own) failed run 2 with nothing between the runs on Node/NVIDIA
     in 3 processes of 3, and ran on Node/AMD in 2 of 2. Inside the `inference` suite, its
     captured models ran. Capture off ran every case on both versions.
4. **Every captured run from the second keeps an output tensor that `InferenceSession.run` drops,**
   with the function that would release it (`session-handler-inference.ts`: a fetch given wins
   over the tensor returned). ONNX Runtime's WASM memory over a million runs (`NG_SPIKE_LEAK=1`,
   Node/NVIDIA): with capture, from 16.78 MB to 29.10 MB in 177 s; without, unchanged in 363 s.
   At 60 runs a second that is at least 2.7 MB an hour, never given back while the page lives.

Both went to the user, who chose to turn capture off for every model (§14, item 6).

**Watched failing.** Five mutations of the built entries, one at a time, each caught by the test
meant for it:
- A6, on Node/AMD: `advance()` starting no run (both paces fail; actions 1 to 300 frames old,
  then 301 to 600); a run every second step (both paces fail); a read-back in `OrtBrain.run`
  (365 `mapAsync` outside the inspector; the fast pace fails as well, its runs slowed); the copy
  pass writing zeros (300 frames on a seed, both paces).
- U32: the capture default back to on.

The first of them at first failed A6 only through a crash of its own report line (the median of
no runs); the line now handles it, and the checks fail as meant. The same mutation showed that
the fast pace's checks passed with no runs at all, so the fast pace gained its floor: more than
half of its frames one frame old.

**Differs from the design, or adds to it:**
- Graph capture is off by default for every model (§14, item 6): not on (item 2), nor off for
  two-file models only (item 5).
- A6 checks more than §10's row: that each run read its own frame's observations, no tearing, a
  second pace, and a control showing the count sees ONNX Runtime's own `mapAsync`.
- Fault 2 of phase 9.2 is very likely a case of fault 3.
- S2 measured the schedule with stand-in passes; A6 measured it through `AgentSystem` and
  `OrtBrain`, and found the same latency.
- The parent's §9.4 still says `enableGraphCapture: true`; phase 9.6 rewrites it.

**Sizes, min+gzip:** the adapter with `OrtBrain` 2,988 B, line 3,000 B (−6 B: the default lost its
condition); the reader 2,846 B, line 2,900 B; the agents entry 19,929 B, unchanged since phase
9.3's commit (its "about 19,932 B" was an estimate; this is the size test's measure).

**Also green:** headless agents (reference + 79), inference 21/21, core, agent-kit (4); `inference`
15/15 on Node/NVIDIA, Node/AMD and Chrome/AMD.
**Not run:** A6 with graph capture on (fault 3); the agents GPU suite and `onnx` (nothing they run
changed); 1.30.0 in Chrome, on AMD, or through any suite but `inference`; Chrome on NVIDIA.
