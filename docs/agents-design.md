# NullGraph Agents — Cycle C: batched agents (sense → think → act) — Technical Design & Plan

**Target:** `src/extras/agents/` (`agentLayout`, `agentMath`, `agentShaders`, `AgentSystem`, brains) and
`src/extras/spatial/` (the uniform grid physics P2 shares)
**Package:** `null-graph/agents` — opt-in subpath; core untouched. The optional ONNX Runtime path
goes through a separate, generic `null-graph/inference` subpath (§9.4); ORT is never bundled.
**Status:** 🔨 **C1 steps 1–2 done** — 29 CPU checks including U2/U3; GPU grid/query production
wrappers and debug draws validated on Node/Dawn and Chrome (AMD GCN-5). See
[`api/agent-grid.md`](./api/agent-grid.md). Steps 3–4 are designed and phased in
[`agents-gpu-loop-design.md`](./agents-gpu-loop-design.md), accepted 2026-10-11. Its phase 3.0,
spike S3, is measured (§13, C0); phase 3.1's GPU system skeleton is implemented and validated
(phase 3.2 sensing also complete: 36 headless agent checks; 63/63 GPU checks on both hosts).
Phase 3.3, the act generator, is next.
No live GPU agent simulation ships yet. Phase 1 = Cycle C (C0 spikes → C1 rules → C2 neural brains); the C0 spikes gate
§9.4's ONNX Runtime defaults, and their numbers go back into this document.
**Depends on:** Cycle A (headless runs, device adoption, the ORT seam), Cycle B (`sourceIndex`, public
`vis.instanceBuffer`, labels that follow GPU buffers, picking), physics P2's grid spec (built here).
**Terminology:** *agents* here are simulated entities — units, creatures, pedestrians. The kit for
coding assistants lives in [`docs/agent/`](./agent/) and is unrelated.

---

## 1. Purpose

`null-graph/agents` runs a population on the GPU. Every simulation step, each live agent **senses**
its neighbourhood, a **brain** turns those observations into actions, and the actions **move** it.
The result is written straight into the instance buffer that `VisibilitySystem` culls, `Picker`
resolves and `TextSystem` labels follow — so 20,000 agents cost the CPU a few dispatch calls per
frame, and nothing is read back. The brain is a socket: hand-written WGSL rules, a built-in kernel
for small neural networks, or any ONNX model through ONNX Runtime, all reading and writing the same
two buffers.

### The single most important pre-existing fact

**Sense → think → act is three compute stages around buffers the renderer already reads.** Every
piece the loop needs ships today, and none of it needs a core change:

| Need                                                         | Already ships                                              | Where                                           |
|--------------------------------------------------------------|------------------------------------------------------------|-------------------------------------------------|
| Compute over storage buffers, in an order the caller chooses | `createComputeStage`, the node array                       | `NullGraph.ts:208`, `RenderPipeline.ts:24-74`   |
| Transforms produced on the GPU, culled and drawn             | `vis.instanceBuffer` (public since Cycle B)                | `VisibilitySystem.ts:71`                        |
| "Which agent is under the cursor?"                           | picks report the app's index through the cull's compaction | `visibilityShaders.ts:298`, `pickShaders.ts:52` |
| Names over 20,000 moving agents, zero CPU                    | labels follow any storage buffer                           | `TextSystem.ts:117-125`                         |
| One ONNX inference for all N, on the GPU                     | a shared device; GPU buffers bound as-is                   | `test/onnx-gpu/run.js` O1–O3                    |

So Cycle C's new code is a 64-byte agent record, the uniform grid physics P2 already specified, a
declarative observation layout, and a schedule for the one stage that cannot live in the frame.

### The second fact: ONNX Runtime cannot be a stage in the frame

`InferenceSession.run()` awaits its input preparation before it dispatches anything
(`wasm-core-impl.ts:747,760` → `:841`). The demo host calls `update()`, the camera update and
`engine.render()` in one synchronous task (site `EngineCore.ts:223-244`), so a run started in
`update()` reaches the GPU queue **after** that frame's submission. Interactive play therefore
applies ORT's actions **at least one frame late** — never torn, because WebGPU executes submissions
in order — and exactness needs a lockstep mode that awaits each run. A WGSL brain has no such gap:
it is a compute stage between sense and act in the same submission.

ORT's fast path, graph capture, binds a session's GPU inputs and outputs **once**, on the first run,
and skips binding afterwards (`:781`, `:1061`). A different buffer passed later is ignored without
an error. This design adopts that as a rule everywhere: observation and action buffers are fixed for
the system's lifetime.

### The third fact: the runtime is ~830× the brain

ONNX Runtime's WebGPU build (`ort-wasm-simd-threaded.asyncify.wasm`, the one the `webgpu` bundle
loads — `wasm-utils-import.ts:214-221`) is **25,749,873 bytes, 6,365,367 gzipped**, plus a 116,102-byte
loader — measured in this repo's devDependency, 1.29.0. The brains this cycle needs are small dense
networks: 24 → 32 → 32 → 2 is **1,922 parameters, 7,688 bytes**. A product that ships a crowd should
not download 6.4 MB to run 8 KB of weights. So small dense networks default to a built-in WGSL kernel
(§9.3), ORT is the socket for everything else (§9.4), and **the same `.onnx` file runs in either**.

### Non-goals for Cycle C

- **Gradient training in the browser** (reinforcement learning, backprop on the GPU). Brains are
  trained offline (§9.5) or, in Phase 2, evolved.
- **Evolution** — per-agent genomes, energy, reproduction. Phase 2; this cycle reserves the record
  field and the weight-table indexing (§9.3).
- **Agents that see** — ray or rendered eyes. Phase 2 and the far horizon; the sensor list (§7.1) is
  the seam they plug into.
- **Pathfinding, navmeshes, flow fields, walls** beyond the world bounds. The evacuation demo needs
  them; deferred.
- **Collision resolution.** Agents steer apart (separation is a behaviour); bodies that must not
  overlap belong to physics P3.
- **Flying / full-3D agents.** Phase 1 agents move on the XZ plane; `y` is carried, not integrated.
- **ORT on the default path.** Nothing in `null-graph/agents` imports, bundles or downloads ONNX
  Runtime. ORT lives behind the generic `null-graph/inference` adapter (§9.4), which apps opt into
  with one import and their own install. **This revises Cycle A's "no ML module" decision**: that
  decision assumed the seam was trivial (bind one buffer). Planning showed it is not — device
  direction, the fixed-buffer rule of graph capture, two schedules, and shape checks are each easy to
  get wrong — so the seam earns a small, generic, optional module.
- **Determinism across GPUs.** Lockstep is bit-exact on one device while no cell overflows (A7,
  §6.3); across vendors it is not,
  and nothing depends on it.
- **Shipping ORT, models or training tools in the npm package.** `files` stays `["dist"]`; ORT stays
  the app's dependency, imported by the app, passed in.
- **f16 inference.** `shader-f16` is detected, not used; f32 is the universal path.

### Design principles

1. **A producer of transforms.** The physics README's framing: the module emits compute stages the
   app splices into its own sequence, core gains nothing, and abandoning the module leaves nothing
   behind in `src/core/`.
2. **The brain is a socket, not a framework.** Observations and actions are plain `f32` buffers in
   ONNX's own layout — `[N, D]`, row-major, tightly packed — so any producer can fill them: a WGSL
   rule, the built-in kernel, ORT, a future WebNN path.
3. **Dimensions are derived, never typed twice.** `obsDim` and `actDim` come from the declared sensor
   and action lists; brains and models are validated against them at construction.
4. **In the frame by default; async only where a runtime forces it** — and the latency is stated
   here, not discovered in a demo.
5. **Buffers are fixed for life.** Graph capture's rule, adopted for every brain.
6. **Sensing is order-independent.** Built-in sensors use only order-independent reductions
   (§6.3), so determinism comes from the kernels. Since step 5 the grid's order is fixed as well,
   and determinism holds past the scan budget too: a crowded cell yields a window that depends
   only on the cell and the tick. (Before step 5, the scatter was atomic, and the scanned subset
   followed its timing.)
7. **A missed neighbour is a wrong decision.** The conservative direction for queries is to look at
   more, never fewer (§6.2).
8. **CPU mirrors are pinned by definitions.** Brute-force k-nearest is the definition the grid query
   is tested against ([`design-principles.md`](./design-principles.md) §13.1).

---

## 2. Architecture

```ts
// A WGSL brain (rules or the built-in network): one submission, no latency
agents.advance(frameSeconds);                // fixed-step accumulator → steps this frame (§5.2)
agents.upload();                             // spawn/kill dirty ranges only
return [
    ...agents.getComputeStages(),            // per step: grid → sense → think → act; then → instances
    ...vis.getComputeStages(),               // the cull reads the transforms the act stage just wrote
    ...names.getComputeStages(),             // labels follow agents.buffer
    scenePass,
];

// An ONNX Runtime brain, interactive: identical frame code. advance() also starts a run on the
// fixed buffers when none is in flight; its result lands between frames and act applies the latest.
// getComputeStages() is then grid → sense → act → instances, with no think stage.

// Lockstep (tests, dataset recording, exact replays): every step awaits its brain
await agents.step();                         // render(grid, sense) → await run → render(act, instances)
```

```
   spawn / setRawData (CPU, zero-copy) ──upload──▶ agents.buffer  (64 B/agent, GPU-owned after upload)
                                                     │ floats 0–3 = position + radius (P1's first vec4)
              ┌──────────────────────────────────────┤
              ▼                                      │
   ┌──────────────────────┐                          │
   │ Grid (P2 stages A–C) │ bin → prefix sum → scatter (one cell per agent)
   └──────────┬───────────┘                          │
              ▼                                      │
   ┌──────────────────────┐   obs [N × obsDim] f32   │
   │ Sense (declared      │ ───────────────────────┐ │
   │ sensors, generated)  │                        ▼ │
   └──────────────────────┘        ┌──────────────────────────────┐
                                   │ Think: RuleBrain │ MlpBrain   │  stages, same submission
                                   │        OrtBrain (async)      │  fixed buffers, ≥ 1 frame late
                                   └──────────────┬───────────────┘
                                                  ▼ act [N × actDim] f32
   ┌──────────────────────┐                       │
   │ Act (steering,       │ ◀─────────────────────┘
   │ fixed dt, sanitize)  │ ───────────────────────────▶ agents.buffer
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐   vis.instanceBuffer (24 floats)   ──▶ VisibilitySystem culls ──▶ draws
   │ To instances         │   meshIndex per species / highlight ──▶ Picker: hit.index = agent index
   └──────────────────────┘
   TextSystem follows agents.buffer (stride 16, offset 0) — no CPU writes per frame
```

---

## 3. Integration facts (verified against this repository)

| #  | Fact                                                                                                                                  | Source                                                                                          | Consequence                                                                                                                                                 |
|----|---------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | The engine executes a caller-built array of compute stages and render passes, in order.                                               | `NullGraph.ts:208`, `RenderPipeline.ts:24-74`                                                   | Agents are stages in the app's array; no new node type.                                                                                                     |
| 2  | Consecutive compute stages share a pass; `preClearBuffers` closes it to write; a disabled stage skips its clear **and** its dispatch. | `RenderPipeline.ts:28-47`, `types.ts:257`                                                       | Grid counter clears sit on the first grid stage (P2 §6.3). `thinkEvery` disables sense and think together ([design-principles](./design-principles.md) §9). |
| 3  | `dispatchCount` may be a per-frame callback over CPU-known values.                                                                    | `types.ts:207`                                                                                  | Every agent stage dispatches `ceil(capacity / 64)`; no indirect dispatch in Phase 1.                                                                        |
| 4  | `vis.instanceBuffer` is public, `STORAGE \| COPY_DST`.                                                                                | `VisibilitySystem.ts:71,127-130`                                                                | The instance stage writes it as `read_write` storage.                                                                                                       |
| 5  | `vis.upload()` rewrites the instance buffer from the adopted CPU array every frame once one is set.                                   | `VisibilitySystem.ts:187-208`                                                                   | GPU-written transforms would be overwritten → the `setInstanceCount` touchpoint (§12.2).                                                                    |
| 6  | Instance records are 24 floats: model 0–15, meshIndex 20; the cull stamps the source index into 23.                                   | `cullMath.ts:20,26,37`, `visibilityShaders.ts:298`                                              | Agent `i` ↔ instance `i`. A dead agent gets a meshIndex no batch draws (the Cycle B demo's `HIDDEN_MESH`).                                                  |
| 7  | Picks report `inst.sourceIndex`.                                                                                                      | `pickShaders.ts:52`                                                                             | `hit.index` is the agent index; no lookup table.                                                                                                            |
| 8  | Labels follow any storage buffer, given a stride and a position offset.                                                               | `TextSystem.ts:117-125`                                                                         | Names follow `agents.buffer` (stride 16, offset 0).                                                                                                         |
| 9  | P1's body record keeps world position + radius in floats 0–3, so P2 bins with one `vec4` load.                                        | `physics/physics-p1-transform-bridge-design.md` §4.2                                            | The agent record keeps the same first `vec4`; one binning kernel serves agents and bodies.                                                                  |
| 10 | P2's grid — bin, scan, scatter, emit pairs — is specified, not implemented.                                                           | `physics/physics-p2-broadphase-design.md` (status, §2)                                          | Cycle C builds stages A–C in `src/extras/spatial/`; P2 later adds only stage D.                                                                             |
| 11 | ORT accepts pre-allocated outputs (a fetch with a value) and GPU-resident tensors.                                                    | `onnxruntime-common/dist/esm/inference-session.d.ts:20-28`, `tensor-factory.d.ts:266`           | Obs and act stay NullGraph-owned; ORT reads and writes them in place.                                                                                       |
| 12 | Graph capture is WebGPU-only, needs every input and output to be an external GPU buffer, and binds them once.                         | `inference-session.d.ts:135-139`; `onnxruntime-web/lib/wasm/wasm-core-impl.ts:599-603,781,1061` | Fixed buffers for life (§9.4). `OrtBrain` refuses a rebind, because ORT would ignore it silently.                                                           |
| 13 | `run()` awaits input preparation before dispatching.                                                                                  | `wasm-core-impl.ts:747,760,841`                                                                 | An ORT run lands after the frame that started it (§1, §9.4).                                                                                                |
| 14 | A session reports input/output shapes; symbolic dims are strings.                                                                     | `inference-session.d.ts:452-468,520-524`                                                        | `OrtBrain` checks `[N or symbol, dim]` float32 at construction.                                                                                             |
| 15 | `env.webgpu.device` can be **set** to an app-created device before the first session.                                                 | `onnxruntime-common/dist/esm/env.d.ts:211-222`                                                  | An engine created first (the site's worker) can hand its device to ORT — pending S5.                                                                        |
| 16 | ORT, creating its own device, requests the adapter's maximum compute/storage limits; NullGraph negotiates features and, since 2026-10-12, three storage limits (`PROGRESSION_LIMITS`), not compute limits. | `onnxruntime-web/lib/wasm/jsep/backend-webgpu.ts:231-257`; `WebGPUContext.requestDevice` (`PROGRESSION_LIMITS`)           | An engine-created device has raised storage limits but default compute limits; S5 checks ORT's dense kernels run on it.                                                                     |
| 17 | The test-engine host runs `update()`, the camera update and `render()` in one synchronous task.                                       | site `src/workers/engine/EngineCore.ts:223-244`                                                 | With fact 13: an ORT run started in `update()` lands between frames.                                                                                        |
| 18 | Readback is forbidden in the frame loop, legal in tests.                                                                              | [`design-principles.md`](./design-principles.md) §13                                            | The inspector and dataset recording read back off the frame path, like `Picker`.                                                                            |
| 19 | Compute stages do not spend the fragment bind-group budget.                                                                           | [`roadmap.md`](./roadmap.md) §6.1                                                               | Agents add no fragment bindings; the instance-tier draw is unchanged.                                                                                       |
| 20 | The ONNX suite already hand-encodes a model as protobuf.                                                                              | `test/onnx-gpu/tiny-model.mjs`                                                                  | The network exporter (§9.5) extends it: no Python, no converter.                                                                                            |

---

## 4. Data model

### 4.1 The agent record — 64 bytes, four `vec4`

```
float  0..2   position       world; y carried, not integrated in Phase 1
float  3      radius         world body radius — P1's floats 0–3, binned with one vec4 load
float  4..6   velocity       world
float  7      heading        yaw, radians, 0 = +Z
float  8      energy         app / Phase 2
float  9      age            seconds alive
float  10     speedScale     per-agent multiplier on the species' max speed
float  11     user0
u32    12     species        index into the species table
u32    13     flags          bit 0 alive · bit 1 frozen · bits 8+ app
u32    14     genome         Phase 2: weight-table row; 0 in Phase 1
u32    15     user1
```

```wgsl
struct Agent {
    posRadius: vec4<f32>,
    velHeading: vec4<f32>,
    state: vec4<f32>,      // energy, age, speedScale, user0
    ids: vec4<u32>,        // species, flags, genome, user1
};
```

CPU constants mirror `cullMath`'s: `AGENT_FLOATS = 16`, `AGENT_SPECIES_OFFSET = 12`,
`AGENT_FLAGS_OFFSET = 13`, `AGENT_GENOME_OFFSET = 14`. The `Uint32Array` view inherits `byteOffset`
and `length` ([design-principles](./design-principles.md) §6).

### 4.2 Species table — 32 bytes each, ≤ 16 species, one uniform

```wgsl
struct Species {
    maxSpeed: f32, maxTurn: f32, accel: f32, senseRadius: f32,   // m/s, rad/s, m/s², m
    meshIndex: u32, hoverMesh: u32, selectedMesh: u32, boundsRadius: f32,   // object-space radius
};
```

### 4.3 Observations and actions — ONNX's layout

| Buffer | Shape | At 20,000 agents | Usage |
|---|---|---|---|
| `obs` | `[capacity, obsDim]` f32 | 20,000 × 24 × 4 = 1.92 MB | `STORAGE \| COPY_SRC \| COPY_DST` |
| `act` | `[capacity, actDim]` f32 | 20,000 × 2 × 4 = 160 KB | `STORAGE \| COPY_SRC \| COPY_DST` |

Rows of dead agents are don't-care. Shapes are static — graph capture requires it. The usage mask is
the one ORT gives its own tensors (`gpu-data-manager.ts:316`, JSEP), and `COPY_SRC` lets the
inspector copy rows.

### 4.4 Grid — P2 §4.1–4.2, 2D case (`dims.y = 1`)

`GridParams` (32 B) · `cellCounts`, `cellOffsets` (cells × 4 B) · `cellAgents` (capacity × 4 B, one
entry per agent — §6.1) · `overflow` (4 B). A 180 m × 180 m world at a 4 m cell is 45 × 45 = 2,025
cells: 16 KB of counts and offsets, 80 KB of entries at 20,000 agents.

### 4.5 Step parameters — 32-byte uniform, plus one GPU tick word

`dt`, world min/max (XZ), `capacity`, species count. The tick counter is a storage word the act
stage increments, so repeated steps inside one submission see distinct ticks (uniform writes all
land before a submission runs).

### 4.6 Memory at 20,000 agents

agents 1.28 MB + obs 1.92 MB + act 0.16 MB + grid ~0.1 MB + instances (`vis`, 96 B each) 1.92 MB
≈ **5.4 MB**, allocated once.

---

## 5. Population

### 5.1 Spawn, kill, ownership

- `agents.setRawData(f32, count)` adopts a filled array by reference (a Worker can build the initial
  population) — the `LightSystem.setRawData` / `InstanceBuffer` contract.
- `spawn(i, init)` / `kill(i)` write the CPU staging record and mark a dirty range; `upload()` writes
  dirty ranges only — `TextSystem`'s pattern. Spawn takes a heading and a **speed**, never a raw
  velocity, so velocity always lies along the heading. Speed is clamped to `maxSpeed × speedScale`
  and a negative one is refused: no agent ever starts faster than steering allows (U1c, U5f).
  `kill` clears the alive bit only; app flag bits survive.
- **After upload the GPU owns the record.** The CPU array is a spawn staging area, not a mirror;
  state comes back only through `inspect()` (§8.4), off the frame path.
- Capacity is fixed at construction (GPU memory against the cap, [design-principles](./design-principles.md)
  §10). Stages dispatch over `capacity` and dead agents return early — 313 workgroups at 20,000.

### 5.2 Fixed step

`advance(frameSeconds)` is P1 §5.1's accumulator: fixed `dt` (default 1/60 s), clamped by
`maxStepsPerFrame` (default 3; P1 §5.2 — the spiral of death is a clamp, not an error), with the
same 0.25 s tab-switch guard. **One deliberate difference from P1's pseudocode:** time is dropped,
and the frame counted in `clampedFrames`, only when **debt remains** after the last allowed step.
P1 drops the remainder whenever it reaches the cap, which discards up to a step of real time and
counts a clamp that never happened (a frame of exactly `maxSteps × dt`). U7a pins this. A NaN,
negative or zero frame time runs no steps and never throws.
`getComputeStages()` returns the step chain repeated `steps` times, from arrays prebuilt for every
count (no per-frame allocation, [design-principles](./design-principles.md) §5). No render
interpolation in Phase 1: a 60 Hz display over 60 Hz steps has nothing to interpolate.

---

## 6. The grid and neighbour queries

### 6.1 Points, not spheres

Agents bin by position into exactly one cell. P2 bins spheres into every cell they touch; that
generality stays in P2. The stages: **A** bin (`atomicAdd` per cell; one `preClearBuffers` zeroes
counts and `overflow`) → **B** P2 §5.4's three-pass prefix sum → **C** scatter. They live in
`src/extras/spatial/` under P2's names and data model, and P2's `GpuBroadphase` imports them.
Since step 5, **C** is a stable radix sort by cell, so a cell lists its agents by index
([`agents-ordered-grid-design.md`](./agents-ordered-grid-design.md)).

### 6.2 The query, and the conservative direction

`senseRadius ≤ cellSize` is enforced at construction (§11), so the 3×3 cells around an agent always
contain every neighbour in range. **A missed neighbour is a wrong decision** — a wolf the prey never
saw — so the query is exact by construction.

Crowded cells bound the scan at `maxScanPerCell` (default 64). Past it, neighbours are dropped and
`overflow` counts them. As in P2 §6.1, there is no safe direction inside the kernel, so the failure
is surfaced, never silent: the inspector and debug readout show the counter. The default cell keeps
it rare: at 0.06 agents/m² a 4 m cell averages about one agent. **That holds for evenly spread
agents, not for flocks** (measured in phase 4.4): Herd & Wolves exceeded the budget in 85% of steps
once its flocks formed. Size the budget, or the cell, for the crowd's peak density, and read
`overflowSteps`.

### 6.3 Determinism

Built-in sensors use only order-independent reductions: k-nearest with ties broken by agent
index, counts, minima. Since step 5 the grid's order is fixed too: a stable sort lists each cell's
agents by index ([`agents-ordered-grid-design.md`](./agents-ordered-grid-design.md)). **Same
device + lockstep + built-in sensors ⇒ bit-exact reruns, with or without overflow:**
- **without overflow,** A7 ([`agents-gpu-loop-design.md`](./agents-gpu-loop-design.md) §8.3);
- **with overflow in every step,** A23, on three host/GPU pairs.

**Past the scan budget,** a crowded cell yields a window of `maxScanPerCell` consecutive entries
that rotates per cell and tick. It depends only on the cell, the tick and the fixed order. The
user decided it on 2026-10-13, and revised it the same day from a window per querying agent, which
cost +15% to +26% on a step.

**Before step 5,** the scatter's order within a cell followed atomic timing, so reruns diverged
past the budget. Measured 2026-10-12: 19,990 of 64,000 record words differed between two identical
200-step runs at budget 64; none differed with no overflow.

**A custom WGSL sensor that sums floats over neighbours** was said here to break determinism, "and
the API docs say so". They did not say so, and it no longer holds: the grid's order and the
neighbour lists are fixed, so such a sum is taken in the same order on every rerun. A custom sensor
still has no contract for reading the lists ([`api/agents.md`](./api/agents.md), "Definition and
observations").

---

## 7. Sensors and actions

### 7.1 A declared observation layout

```ts
const agents = new AgentSystem(engine, {
    capacity: 20_000,
    world: { min: [-90, -90], max: [90, 90] },     // XZ
    cellSize: 4,                                   // ≥ every species' senseRadius
    species: [
        { name: 'prey', maxSpeed: 3.0, maxTurn: 4, accel: 6, senseRadius: 4, radius: 0.4, mesh: PREY },
        { name: 'wolf', maxSpeed: 4.5, maxTurn: 3, accel: 5, senseRadius: 4, radius: 0.6, mesh: WOLF },
    ],
    sensors: [
        Sensors.self(),                            //  2: speed, energy
        Sensors.nearest({ species: 'wolf', k: 1 }),//  5 per neighbour: side, forward, velSide, velForward, present
        Sensors.nearest({ species: 'prey', k: 3 }),// 15
        Sensors.bounds(),                          //  2: side, forward toward the nearest wall
    ],                                             // obsDim = 24, derived
    actions: Actions.steering(),                   // actDim = 2: turn, throttle
});
```

Declarations live in the `Sensors` and `Actions` namespaces rather than as bare exports: a function
exported as `self` would shadow the worker global of the same name in any module that imported it.

From the list, `defineAgents()` derives every number a stage agrees on (implemented, C1 step 1):
- **Named offsets** for rule brains — `OBS_SELF 0`, `OBS_WOLF_0 2`, `OBS_PREY_0 7`, `OBS_PREY_1 12`,
  `OBS_PREY_2 17`, `OBS_BOUNDS 22`, `OBS_DIM 24`, `ACT_DIM 2` — plus the field offsets inside each
  block (`NEAR_SIDE 0` … `NEAR_PRESENT 4`, `SELF_SPEED`, `BOUNDS_FORWARD`, `ACT_TURN`, `ACT_THROTTLE`).
  `constantsWGSL()` emits them as WGSL `const`s.
- **Slot names** (`obsNames`: `wolf[0].side`, …), for the inspector's "what does it see" panel.
- **A species bitmask** per `nearest` sensor, and the grid's cell counts.

**The local frame**, shared by every sensor and by steering. With heading `h` (0 faces +Z):
forward `f = (sin h, cos h)`, side `s = (cos h, −sin h)`. `s` is the direction `f` turns toward
as `h` grows, so **a positive `side` reading means "a positive turn steers toward it"**; with Y up,
that side is the agent's left. If either the sensors or steering flipped it, an agent told to turn
toward a target would settle facing away. U5a closes that loop and fails under either flip.

**The built-in sensors**, every value clamped to [−1, 1]; a dead agent's row is all zeros:

| Sensor | Floats | Definition |
|---|---|---|
| `self()` | 2 | speed / (`maxSpeed` × `speedScale`); energy |
| `nearest({ species, k })` | 5 per rank | The `k` nearest **live** agents of the listed species (all species if omitted), excluding itself, at XZ distance ≤ the observer's `senseRadius`, nearest first, an exact tie going to the **lower index**. Per rank: Δpos in the local frame ÷ `senseRadius` (side, forward); Δvelocity (theirs − mine) in the local frame ÷ (2 × the observer's `maxSpeed`) (velSide, velForward); present = 1. Empty ranks are all zeros |
| `bounds()` | 2 | The nearest world wall's outward normal, in the local frame, × max(0, 1 − distance ÷ `senseRadius`): 1 at the wall, 0 from `senseRadius` inward. Equidistant walls resolve in the order −X, +X, −Z, +Z |
| `custom({ name, width, wgsl, mirror? })` | `width` | App WGSL. The optional `mirror` lets it run in CPU tests and dataset recording; without one, CPU sensing refuses it by name |

Ego-centric and normalized is what a small network learns from, and what lets one brain work
anywhere on the map.

### 7.2 Actions

`Actions.steering()`, per agent per fixed step (implemented as the CPU mirror, C1 step 1):

1. **Sanitize.** A row with **any non-finite value** is discarded: no turn, target speed 0. A
   broken brain makes agents coast to a stop; it never moves them or corrupts the record. On the
   GPU, "non-finite" is tested on the **exponent bits** (`bitcast<u32>(v) & 0x7f800000u ==
   0x7f800000u`), because WGSL lets compilers assume values are never NaN or infinite, so `v != v`
   may be optimized away.
2. **Turn:** `h += clamp(turn, −1, 1) × maxTurn × dt`, wrapped to [−π, π).
3. **Speed:** target = (clamp(throttle, −1, 1) + 1) ÷ 2 × `maxSpeed` × `speedScale`, so −1 stops,
   0 is half speed and +1 is full. Current speed moves toward the target by at most `accel × dt`.
4. **Move:** velocity = `f(h)` × speed (velocity always lies along the heading); position += velocity × dt.
5. **Walls:** the centre is clamped inside the world, inset by the agent's radius. The heading is
   reflected (`h → −h` at an X wall, `h → π − h` at a Z wall) only if the agent was moving into that
   wall, so an agent spawned outside turns back instead of oscillating.

A frozen agent only ages (velocity zero); a dead one is untouched. `Actions.custom({ name, width,
wgsl, mirror? })` replaces the integrator.

---

## 8. Output to the renderer

### 8.1 Instances

`agents.toInstances(vis)` adds a stage that writes all 24 floats per agent: model = translate · yaw,
bounds `[0, 0, 0, species.boundsRadius]` (object space, as `InstanceBuffer.write` expects), meshIndex
from the species — or a hidden index for dead agents — `LOD_AUTO`, flags 0. It requires
`vis.setInstanceCount(capacity)` (§12.2) and refuses a `VisibilitySystem` holding a CPU array.

### 8.2 Highlight

`agents.setHighlight({ selected, hovered })` writes an 8-byte uniform. The instance stage swaps the
meshIndex to the species' `hoverMesh` / `selectedMesh`. This is the Cycle B demo's trick moved to the
GPU, because the CPU no longer owns the records.

### 8.3 Labels and picking

Labels: `follow: { buffer: agents.buffer, strideFloats: 16, positionOffset: 0 }`. Picking:
`picker.addInstances(vis, …)` per species mesh; `hit.index` is the agent index. Nothing new.

### 8.4 Inspector

`agents.inspect(i)` → `Promise<{ record, obs, act, obsNames }>`. It copies three ranges into a
`MAP_READ` staging buffer, one request at a time, off the frame path (`Picker`'s pattern). This is
the "click a unit, see its brain" view: what the agent senses, what it chose, and why.

---

## 9. Brains

### 9.1 The contract

```ts
interface Brain {
    readonly kind: 'stage' | 'async';
    bind(io: { obs: GPUBuffer; act: GPUBuffer; capacity: number; obsDim: number; actDim: number }): void;
    getComputeStages(): ComputeStageNode[];   // 'stage': think, spliced between sense and act
    run?(): Promise<void>;                    // 'async': one run over the bound buffers
    destroy(): void;
}
```

`bind` is called once, by `agents.setBrain(brain)`. The system owns `obs` and `act`; a brain only
reads one and writes the other, and one built for other dimensions is refused at `setBrain`. The
app's frame code does not change when the brain does.

### 9.2 `RuleBrain` — WGSL, no ML

`new RuleBrain(wgsl)`, where `wgsl` defines `fn think(i: u32)`, reading `obs` through the named
offsets and writing `act`. It is the shipping brain for many games, and the teacher for cloning.

### 9.3 `MlpBrain` — the built-in kernel

- Dense layers only: ≤ 4 layers, hidden width ≤ 64, `obsDim` ≤ 128, `actDim` ≤ 16; activations
  `relu`, `tanh`, `sigmoid`, `identity`.
- The network's shape is compiled into the kernel as constants; the weights live in one storage
  buffer. One thread per agent. Every thread reads the same weights, so they stay in cache.
  New weights cost one `writeBuffer`; a new shape costs a pipeline (build time,
  [design-principles](./design-principles.md) §11).
- **The kernel S3 chose:**
  - weights read from storage, in loops;
  - no barriers, so the think entry can return early;
  - `tanh` inputs clamped to ±15, because FXC's `tanh` returns NaN from |x| ≥ 89 (§13, S3).
- `MlpBrain.fromOnnx(bytes)` reads the dense subset: `Gemm`, or `MatMul` + `Add`, then `Relu` /
  `Tanh` / `Sigmoid`, with float32 initializers. That is what `torch.onnx.export` of a stack of
  `Linear` + activation layers emits. Any other op is refused by name, pointing at `OrtBrain`.
  **As built in step 8** ([`agents-onnx-reader-design.md`](./agents-onnx-reader-design.md),
  decided with the user 2026-10-14): a function, `mlpFromOnnx(bytes)`, in `null-graph/inference`,
  which returns an `MlpNetwork`. It also passes over a `Cast` to float32 or a `Flatten` before
  the first layer, a `Reshape` after the last, and an `Identity` of an initializer, which
  scikit-learn and PyTorch's older exporter write. Weights stored outside the file, which is
  what PyTorch's default call writes, are refused with the fix named; step 9 looks again. No
  refusal points at `OrtBrain` until step 9 builds it.
- **Phase 2 seam:** the record's `genome` word will select a row of a weight table; Phase 1 has one
  row.

### 9.4 `OrtBrain` — ONNX Runtime as an optional socket, via `null-graph/inference`

**Where it lives.** Everything below except "obs in, act out" is generic — device sharing, wrapping
NullGraph buffers as tensors, the fixed-buffer rule, the two schedules, shape checks — and other
projects need exactly that (relight-your-photo, the photo booth: an image buffer in, a mask or depth
buffer out). So the generic part is its own opt-in subpath, **`null-graph/inference`** — named for
the job (running a trained network), not for "models", which in a 3D engine means meshes. Agents
are its first consumer; `OrtBrain` is a ~20-line wrapper over it. ORT is an **optional peer
dependency**: listed, never auto-installed, never bundled. An app that imports only
`null-graph/agents` gets zero bytes of it and no ORT download.

- `await OrtBrain.create({ ort, session, input: 'obs', output: 'act' })`. The app passes its `ort`
  namespace; null-graph imports ORT's **types only** and never bundles it.
- **Construction checks:** the same device (`await ort.env.webgpu.device === engine.device`), and
  input/output metadata of `[N or a symbol, dim]` float32 (fact 14).
- **Binding, once:** `ort.Tensor.fromGpuBuffer(obs, { dataType: 'float32', dims: [capacity, obsDim] })`
  as the feed, and the same over `act` as a pre-allocated fetch (fact 11).
- **Session options the docs require:** `executionProviders: ['webgpu']`,
  `preferredOutputLocation: 'gpu-buffer'`, and `enableGraphCapture: true` (S1 measures its effect).
- **Interactive schedule — latest wins, one in flight.** `advance()` starts a run only when none is
  pending. Sense writes `obs` every step. ORT reads whatever `obs` holds at its position in the
  queue and writes `act`. The act stage applies the latest `act`. Latency is ≥ 1 frame, typically 1
  (S2 measures the distribution). There is no tearing, because the queue orders submissions.
- **Lockstep schedule:** `await agents.step()` — sense, await the run, act. Exact; used by tests,
  dataset recording and replays.
- **Lifetime:** the tensors wrap the system's buffers. `destroy()` drops the wrappers; the session
  belongs to the app, the buffers to the system.
- **Device ownership — two supported directions.** *ORT first* (Cycle A's rule; ORT asks for
  adapter-maximum limits, so use it for large models). *Engine first*, when the engine must exist
  before any model (the site's worker): set `ort.env.webgpu.device = engine.device` before the first
  session (fact 15). S5 decides whether default limits suffice for the dense subset; if not, §12.3.
  This amends Cycle A's "never the reverse" only after S5, and only for this case.

### 9.5 Training without ML experience — `tools/agents/` (in the repo, not in the package)

1. **`record.mjs`** runs the population headless (Node + Dawn) in lockstep with the `RuleBrain`,
   samples (obs, act) every *k* steps → `dataset.bin` plus a JSON header holding the observation
   layout.
2. **`train.mjs`** — a ~150-line dense-network trainer in plain JS: Adam, mean-squared error,
   seeded. No Python, nothing to install.
3. **`export.mjs`** writes `brain.onnx` (`Gemm` + activation), extending
   `test/onnx-gpu/tiny-model.mjs`'s encoder.
4. The same file runs in `MlpBrain.fromOnnx` and in `OrtBrain`; A5 checks they agree. (Step 8
   built the reader as `mlpFromOnnx`, and its A28 makes this comparison without `OrtBrain`: the
   bytes read into an `MlpBrain` against ONNX Runtime running the same bytes.)

This is behaviour cloning — "make the network copy the rules". It is the honest first step: it
proves the loop end to end, and its quality is measured (A8), not claimed. A brain trained anywhere
else (PyTorch → ONNX) plugs into the same socket.

**Detailed in [`agents-cloning-design.md`](./agents-cloning-design.md)** (decided with the user,
2026-10-13). It changes two things in the list above: labels come from a separate pass of the
teacher, not from the driver's own actions, and a `.mlp.json` file is the working format, with
`brain.onnx` last.

### 9.6 Why the built-in kernel is the default

| | `MlpBrain` (WGSL) | `OrtBrain` |
|---|---|---|
| Extra download | none — the kernel is generated in `agents.js` | 6.4 MB gzip WASM + 116 KB JS |
| Place in the frame | a stage in the same submission; deterministic | async; ≥ 1 frame late interactively, exact in lockstep |
| Ops | the dense subset | anything the WebGPU EP runs: convolutions, attention, … |
| Per-agent weights (evolution) | yes, by genome row (Phase 2) | no — one weight set per session |
| Dependency | none | the app's `onnxruntime-web` |
| Per-run overhead | one dispatch | measured by S1 |

**Decision (user, 2026-10-11): hybrid.** `RuleBrain` and `MlpBrain` are the shipped path. ORT is
never the default; it is the **research path** — the cross-check that keeps the kernel honest (A5),
and the way in for networks the kernel cannot run. Its natural home is "agents that see": one
shared vision network reading every agent's eye image in one batch, which is exactly what ORT is
good at.

### 9.7 Goals — what can be demonstrated, phase by phase

| When | What exists | Demos / use cases | For |
|---|---|---|---|
| **Today** | Picking, labels, GPU culling; ORT proven to share NullGraph's device (`test/onnx-gpu`) | The Picking & Labels demo (2,000 agents moved by CPU code); standalone ORT models in a browser, not yet feeding agents | Showing the parts; nothing learns yet |
| **After C1** | 20,000 agents sensing, deciding, acting on the GPU | **Herd & Wolves**; click an agent to see what it senses and decides; crowds and RTS-style units | Games, crowd visuals, the engine showcase |
| **After C2** | Trained brains in the built-in kernel; exact lockstep replays; headless runs | Rules vs. learned brain side by side, live; "a network learned to herd from the rules"; reproducible headless experiments | First research-style results |
| **Phase 2** | Per-agent brains; birth, death, mutation on the GPU | Predator–prey co-evolution in the browser with generation charts; long headless runs on a rented GPU, best brains replayed in the browser | Artificial-life research, "watch them learn" |
| **Research horizon** | Eyes: rays first (built-in kernel), then rendered views read by a shared vision network (ORT via `null-graph/inference`) | Camouflage, day/night and fog changing what agents see, ray eyes vs. camera eyes | Vision-based agents in the browser |

Every row stands alone as a demo; work can stop after any of them. Lockstep, repeatable runs (C2)
are what make the later rows research rather than animation.

---

## 10. Test suite

**Headless** (`npm run test:agents`):

| # | Asserts |
|---|---|
| U1 | Record layout: every field's offset; floats 0–3 identical to P1's body record; the `Uint32Array` view respects `byteOffset` and `length` |
| U2 | Grid mirror: cell index ↔ position round-trips for 4,096 positions including every boundary face; the prefix sum equals a CPU scan over 64 random arrays, including all-zero and all-max (P2 U1, U4) |
| U3 | **k-nearest equals brute force** over 512 random scenes — exact sets, index tie-break. The inverted version (farthest-first) must fail it |
| U4 | Each built-in sensor's mirror on hand-built scenes with known answers (a wolf 2 m dead ahead → local Δpos (0, 0.5)) |
| U5 | Steering over 600 steps stays in bounds and within speed and turn limits; a NaN action stops the agent and keeps the record finite |
| U6 | `obsDim` / `actDim` and the named offsets are derived from the declarations; a brain or model with other dimensions is refused |
| U7 | The fixed-step accumulator: steps per frame for a sequence of frame times, the clamp, and zero per-frame allocation (counted constructors, [design-principles](./design-principles.md) §5) |
| U8 | Stage arrays are prebuilt per step count; `thinkEvery` disables sense and think in the same call |
| U9 | Every generated WGSL module parses, declares its entry point, uses each binding, and contains no backtick |
| U10 | `MlpBrain` CPU forward equals a reference on known weights; the kernel's constants match the network |
| U11 | The trainer fits y = Ax + b (10 dims) to MSE < 1e-6 and XOR with one hidden layer; the same seed gives byte-identical weights |
| U12 | Exporter ↔ `fromOnnx` round trip is byte-exact on weights; an unsupported op is refused by name. Built in two halves: the writer's in step 7 (phase 7.10), the reader's in step 8 (`mlpFromOnnx`), with U29–U31 and A28 beside it |
| U13 | The `OrtBrain` scheduler against a fake async session: one run in flight, latest wins, lockstep order sense → run → act, a rebind refused |
| U14 | Every §11 error fires and names the offender |
| U15 | The suite exits non-zero on failure — verified once by injection ([design-principles](./design-principles.md) §13.2) |

**GPU** (`test/agents-gpu`, Node + Dawn and Chrome):

| # | Asserts |
|---|---|
| A1 | The GPU grid equals the CPU mirror for 10,000 agents (cell membership as sets) |
| A2 | `obs` equals the CPU mirror for 10,000 agents (≤ 1e-5 absolute; counts exact) |
| A3 | One step of act equals the CPU mirror. WGSL only requires `sin`/`cos` to within 2⁻¹¹ absolute on [−π, π], so the bound is derived, not guessed: the heading to f32 rounding, velocity and position to `speed × 2⁻¹¹` per step. The measured maximum is reported per GPU |
| A4 | `MlpBrain` on the GPU equals the CPU forward pass (≤ 1e-5 relative) |
| A5 | The same `.onnx` through `OrtBrain` (graph capture on) equals `MlpBrain` (≤ 1e-4 absolute) |
| A6 | Across 300 interactive frames with ORT, **zero** `mapAsync` outside the inspector (instrumented) |
| A7 | Two lockstep runs of 600 steps from one seed produce byte-identical agent buffers (same device, no overflow) |
| A8 | Cloning quality: held-out action MAE ≤ 0.05, and prey survival after 30 s within 10% of the teacher |
| A9 | Step 5 characterizes full rule-step scaling, density/query quality and proven bottlenecks on available GPU targets; report conditional limits, not a population/timing pass target. Original 20K / ≤2 ms NVIDIA goal is historical (scope amended 2026-10-12); network measurements follow in C2 |
| A10 | Picking an agent after 120 moving steps returns its agent index |
| A11 | A label tracks its agent for 120 steps with zero CPU writes to label data |
| A12 | Dead agents draw nothing and are never picked |

---

## 11. Errors

| Condition | Message |
|---|---|
| Bad capacity | `AgentSystem: capacity <n> must be a positive integer.` |
| Sense radius past the cell | `AgentSystem: species "<s>" senses <r> m but cells are <c> m — a neighbour two cells away would be missed. Raise cellSize to ≥ <r>.` |
| Grid too large | `AgentSystem: <x>×<z> cells exceeds maxCells <m>; raise cellSize or maxCells.` |
| Unknown species | `AgentSystem: species "<s>" is not declared.` |
| Brain dimensions | `AgentSystem: the brain maps <bo>→<ba>; the layout is <o>→<a>.` |
| Instances owned by the CPU | `AgentSystem: toInstances() writes vis.instanceBuffer on the GPU; use vis.setInstanceCount(n), not setInstances() (it re-uploads every frame).` |
| Foreign `VisibilitySystem` | `AgentSystem: that VisibilitySystem belongs to a different engine.` |
| Index out of range | `AgentSystem: index <i> is outside capacity <n>.` |
| Used after destroy | `AgentSystem: used after destroy().` |
| Network too wide | `MlpBrain: layer <k> is <w> wide; the built-in kernel takes ≤ 64. Use OrtBrain.` |
| Unsupported ONNX op | `MlpBrain.fromOnnx: node "<n>" is <Op>; supported: Gemm, MatMul, Add, Relu, Tanh, Sigmoid. Use OrtBrain.` As built in step 8: `mlpFromOnnx: node "<n>" is <Op>; supported: Gemm, MatMul, Add, Relu, Tanh, Sigmoid.`, with the reader's other refusals in [`agents-onnx-reader-design.md`](./agents-onnx-reader-design.md) §11 |
| ORT on another device | `OrtBrain: ONNX Runtime uses a different device. Set ort.env.webgpu.device = engine.device before creating the session, or init the engine with { device: await ort.env.webgpu.device }.` |
| Model shape | `OrtBrain: input "<n>" is [<dims>] <type>; expected [N, <obsDim>] float32.` |
| Mixed schedules | `OrtBrain: step() while an interactive run is in flight — await it first.` |

All are thrown at construction or registration, never per frame ([design-principles](./design-principles.md) §2).

---

## 12. Engine touchpoints

### 12.1 Core — required: **none**

Compute stages, storage buffers, `atomic<u32>`, `preClearBuffers`, `enabled` and adopted devices all
ship. Core bundle: **+0 B**.

### 12.2 Visibility — required for the 20,000-agent target: `vis.setInstanceCount(n)`

It sets a count without adopting a CPU array, so `upload()` writes only the cull parameters. About
+40 B min+gz, with a headless test that `upload()` issues no instance write afterwards.

**Zero-change fallback:** keep a dummy CPU instance array, call `setInstances(dummy, n)`, and splice
the instance stage after `vis.upload()` so the GPU overwrites the uploaded garbage every frame. It
works, and it costs 1.92 MB of pointless upload per frame at 20,000 agents (≈115 MB/s at 60 Hz) plus
a full CPU array that mirrors nothing.

### 12.3 Core — optional: `EngineInitOptions.requiredLimits`

Only if S5 shows ORT's dense kernels fail or crawl on an engine-created device with default limits.
**Since 2026-10-12** core raises three storage limits (`PROGRESSION_LIMITS`) on every device it
creates. If S5 finds ORT needs compute limits too, extending that curated list is the first
option, ahead of a new `requiredLimits` option.
**Fallback:** the ORT-first device (Cycle A's direction) — which an engine-first host, like the
site's worker, can only reach by re-initialising its engine.

### 12.4 Physics docs — amended, not changed

P2's spec gains one line: stages A–C ship in `src/extras/spatial/` (Cycle C); `GpuBroadphase`
imports them and adds stage D.

### 12.5 Bundle budget

`agents.js` ≤ 16 KB min+gz for C1 (grid, layout, stages, inspector; raised from 12 KB on
2026-10-11, decision 7 below). The frame path's tree-shaken cost is reported beside it. At most +4 KB for C2
(`MlpBrain`, `fromOnnx`); `inference.js` (the generic ORT adapter plus `OrtBrain`'s wrapper) a few
KB, paid only by apps that import it; `visibility.js` about +40 B; core +0 B. ONNX Runtime itself:
0 B in every bundle.
**Since step 8** the ONNX reader is in `inference.js`, not in `agents.js`: 2,525 B min+gzip under a
line of its own of 2,600 B, paid only by apps that import it. `agents.js` is 19,323 B of C2's
21,495 B, with nothing spent on the reader.

---

## 13. Build order

### C0 — spikes: measure before building (scripts only, no library code)

Agents-first reordering: **S3 runs before C1's step 3** (it sizes the built-in kernel, the shipped
path). ✅ **S3 measured 2026-10-11**; the results are below the table. S1, S2, S4 and S5 measure only the optional ORT path and run before C2's step 9.

| # | Question | Measured how |
|---|---|---|
| S1 | What does one ORT run of the 24→32→32→2 network cost at N = 1,000 / 20,000 / 100,000? | GPU-resident in/out, graph capture on and off; Node/NVIDIA and Chrome/AMD |
| S2 | Where does a run started in `update()` land? | Start runs for 300 frames; record which frame's actions each act stage consumed → the latency distribution. Also confirm a swapped buffer is ignored under capture (fact 12) |
| S3 | What does the same network cost as a WGSL kernel? | The baseline S1 is judged against |
| S4 | What does ORT cost to start? | WASM fetch + compile, session creation, first (captured) run, in ms |
| S5 | Does ORT run on an engine-created device with default limits? | The network runs and its output equals S3's |

**Gate:** the numbers decide §9.4's defaults (capture on, the think rate) and whether §12.3 is
needed. If S1 is under ~1 ms at 20,000, `OrtBrain` is comfortable interactively at ≤ 30 Hz thinking.

**S3 — measured 2026-10-11.**
- **Where:** `test/spikes/s3-think-kernel/`, run with `npm run spike:s3` (Node, every adapter
  Dawn offers) and in Chrome with `npm run drive`.
- **What:** the 24→32→32→2 tanh network, one invocation per agent, workgroup 64, weights read from
  a storage buffer in loops. That is the shape C2's kernel takes
  ([`agents-gpu-loop-design.md`](./agents-gpu-loop-design.md) §7.1).
- **Correctness:** every variant matched an f64 CPU forward pass to within 3.7e-7.
- **Method:** GPU timestamps; the lower of two medians of five, after a 1 s warm-up.

| µs per think dispatch | 1,000 | 20,000 | 100,000 |
|---|---|---|---|
| NVIDIA Turing — Node/Dawn | 38 | 176 | 909 |
| AMD GCN-5 iGPU — Node/Dawn | 33 | 281 | 1,229 |
| AMD GCN-5 iGPU — Chrome | 41 | 337 | 1,286 |

That is 9–13 ns per agent at 100,000. At 20,000 the network takes about 9% of A9's 2 ms step budget
on NVIDIA. What the spike found:
- **Weights staged in workgroup memory are 2.1–2.3× faster on NVIDIA and 2.1–2.3× slower on AMD.**
  There is no portable winner. Staging also needs barriers, fits only small networks, and cannot
  serve Phase 2's per-agent weights. The think entry therefore keeps early returns.
- **A uniform buffer** is 10–19% faster on NVIDIA and level on AMD. **A fixed-size storage array**
  (no bounds checks) changes nothing.
- **Unrolled code is at most 10% faster than loops**, but takes 0.2–0.4 s to compile, and up to
  4.4 s under FXC, against 0.03–0.26 s for loops. `MlpBrain` uses loops.
- **`tanh` returns NaN for |x| ≥ 89 under Node/Dawn (D3D12 through FXC) on both GPUs.** Chrome,
  which compiles with DXC, is finite. A NaN in a hidden layer would stop agents without a sound, so
  `MlpBrain` clamps `tanh` inputs to ±15, which is exact in f32 (§9.3). No engine shader calls
  `tanh` today.
- **With `thinkEvery = 3`, staggering by agent index saves nothing:** it costs 88–111% of all
  agents thinking, because the lanes of one wavefront run together. Staggering by 64-agent block
  costs 32–42%. `thinkEvery` staggers by block.

### C1 — a population with rules, no ML (a stopping point)

1. ✅ **Record, layouts, CPU mirrors — U1, U4–U7, U14 (step 1's errors). No GPU.** Done
   2026-10-11.
   - **Files:** `agentLayout.ts` (record constants, WGSL structs, `Sensors` / `Actions`,
     `defineAgents`, `packSpecies`, `constantsWGSL`, `assertBrainDims`), `agentRecords.ts` (spawn,
     kill, dirty range, zero-copy adoption), `agentMath.ts` (the mirrors, brute-force
     `nearestNeighbours`, `FixedStep`).
   - **Wiring:** build entry `agents`, export `./agents`, `npm run test:agents`. 24 checks.
   - **Watched failing.** Six bugs were injected one at a time, and each was caught by the test meant
     for it: a flipped sensed side axis (U4a, U4d, U5a), a flipped turn sign (U5a, U5b), P1's clamp
     rule (U7a), ties to the higher index (U4d), no non-finite sanitize (U5c, U5f), and a `u32` view
     without `byteOffset` (U1b). The suite's exit 1 was seen on a real failure: unclamped spawn
     speed, now fixed (§5.1).
   - **Bundle:** `agents.js` 5,138 B min + gzip, much of it error text; core +0 B.
2. ✅ **Grid, query and GPU debug views implemented.** `spatial/gridMath.ts` and `agents/agentGridMath.ts` pin U2/U3
   with five checks. `spatial/gridShaders.ts` and `agents/agentGridShaders.ts` add shared XZ bin,
   recursive block scan/add-back, full scatter and bounded query generators. The registered
   `agents` GPU harness passes 36/36 checks, zero GPU errors, on both Node/Dawn and Chrome
   (AMD GCN-5). `PointGrid` / `AgentGrid`
   own prebuilt stages/buffers; GPU resets support repeated steps in one submission without
   queue-clear hazards. `debug-ui/gridDrawing.ts` exports the two draws below, with real pixel
   assertions (nonuniform floor, red overflow, empty rebuild removes red). Overflow preserves every index; bounded
   query subsets depend on atomic scatter order, so the crowded oracle uses that actual order.
   The grid (P2 stages A–C) and the k-nearest
   — U2, U3, A1. With it, the first **GPU debug
   views** in `null-graph/debug-ui` (today two DOM performance widgets). The physics plan already
   names `debug-ui` as the home of debug drawing (P8), so these are its first GPU members:
   - **`groundGrid()`** — the anti-aliased minor/major floor grid with distance fade and a field edge,
     extracted from the two site demos that each carry a copy.
   - **A grid-occupancy overlay** — the agent grid's cells drawn at `cellSize`, tinted by the
     `cellCounts` the GPU just wrote, with `overflow` cells marked. A binning bug is visible at a
     glance, not only in A1's numbers.

   Verified by A1 (GPU grid = CPU mirror) and by `npm run drive` screenshots of the overlay in the
   site demo.
3. Sensors → `obs`, `RuleBrain`, steering, fixed step — U8, U9, A2, A3, A7.
   Phases 3.1–3.3 are implemented locally: system, sense and act. Current chain is
   grid → queries → sense → act → tick; RuleBrain/app stages follow in phase 3.4.
   Phase 3.3 validates 38 headless agent checks, 17 core checks and 72/72 GPU checks on each
   of Node/Dawn and Chrome (both AMD GCN-5), with zero WebGPU errors. A3 checks 10,000 slots,
   A14 checks GPU-written non-finite actions and A7 verifies two byte-identical 600-step runs.
   Whole entry min+gzip is 11,791 B; tree-shaken system/declarations is 9,075 B. Details and
   measured error bounds are in the child design's phase-3.3 result. No new frontend/visual review.
4. Instance output, `vis.setInstanceCount`, highlight, inspector — A10–A12.

   Phase 3.4 is also implemented locally: RuleBrain, think helpers/shared cadence and app stages.
   The production chain is grid → queries → sense → think → act → app stages → tick; a brain is
   now required. A15/A16/A18 and production-brain A13 pass: 43 headless agent checks, 17 core checks,
   90/90 GPU checks on each host, zero WebGPU errors (both AMD GCN-5). Two 600-step catches/respawn
   runs are byte-identical and overflow-free. Whole entry min+gzip is 13,428 B; tree-shaken
   system/declarations/RuleBrain is 10,593 B. App race safety is a caller contract; auxiliary
   buffers/brains remain caller-owned. Phase 3.5 is next; no new frontend/interactive review yet.

   **Steps 3–4 are designed in [`agents-gpu-loop-design.md`](./agents-gpu-loop-design.md),**
   phased 3.0 (S3) → 3.5 and 4.1 → 4.5, with tests U16–U20 and A13–A20 added. Its §1.5 lists where
   it refines this plan: the think stage is built by the system, `think` gains `species` and
   `tick`, a GPU tick stage, staggered `thinkEvery`, and queries at `cellSize`. All of it was
   accepted on 2026-10-11 (decision 7).
5. **Scaling limits and proven bottlenecks** (scope amended with the user, 2026-10-12): reproduce
   the user-reported good frame rates at 65,536, measure population/density/cadence/render scaling,
   simulation throughput and query quality, then prioritize evidence-backed finer improvements.
   No fixed count/timing pass target or core rewrite; no optimization is a valid outcome.
   [`agents-demo-design.md`](./agents-demo-design.md) defines the investigation. Detailed experiment
   choices remain draft; no new measurements are claimed. The original 20K/A9 target is historical.
   API guide complete; remaining README/starter/showcase packaging is separate product follow-up.
   **Resumed 2026-10-13 by the user, ahead of step 7, part 2,** for deterministic neighbour lookups
   (a stable grid order): [`agents-ordered-grid-design.md`](./agents-ordered-grid-design.md),
   phases 5.5–5.8, **complete 2026-10-13**. The rest of the study stays deferred.

C1 alone is a product: crowds, herds and RTS-style units, with no ML anywhere.

### C2 — neural brains

6. `MlpBrain` and its CPU forward pass — U10, A4. Draft focused design:
   [`agents-mlp-design.md`](./agents-mlp-design.md), phases 6.0–6.3. Same sense → think → act
   loop; neural inference replaces only think. No training, ONNX or async scheduling in this step.
   Step 5 remains deferred, not complete; it does not block these correctness/integration checks.
   Phase 6.0 implemented: model validation/layout/packing and CPU forward with U10 tests.
   Phase 6.1 implemented: generated think WGSL, GPU-owned MlpBrain weights/updates and mock
   integration/lifetime checks.
   Phase 6.2 implemented: A4 passes on Node/Dawn and Chrome (120/120 agents GPU checks): five
   fixtures up to the maximum shape agree within 1e-5 + 1e-5·|cpu| (max abs 4.2e-7), plus extreme
   inputs, gating, updates, replacement and lifecycle.
   ✅ Phase 6.3: an executable known-weight guide example, the production think stage's cost on
   three host/GPU pairs (0.83–0.88 ms per dispatch at 100,000 agents for 24→32→32→2), and
   +1,424 B for an app that adds `MlpBrain`. **Step 6 is complete.**
7. `tools/agents`: record → train → export — U11, U12.
   **Part 1, decided with the user 2026-10-12: per-species brains.** One network per species,
   designed in [`agents-species-brains-design.md`](./agents-species-brains-design.md), phases
   7.1–7.5:
   - weight-table rows of one `MlpBrain` (U21, A21);
   - then, after 7.2 measured per-lane routing at up to 5.6× in Chrome, one think pass per brain,
     gated to its species: `setBrain(brain, { species })` (U22, A22).

   The rule: passes below 8 species, rows from 8. ✅ **Part 1 is complete (2026-10-13):** both
   mechanisms are tested on three host/GPU pairs, and the guide has an executable per-species
   example.
   **Part 2:** record → train → export, phases 7.6–7.10:
   [`agents-cloning-design.md`](./agents-cloning-design.md), decided with the user 2026-10-13.
   Nothing is built yet.
8. `MlpBrain.fromOnnx` — U12. **Built as `mlpFromOnnx` in `null-graph/inference`,** phases
   8.0–8.3: [`agents-onnx-reader-design.md`](./agents-onnx-reader-design.md), decided with the user
   2026-10-14. The entry of step 9 is created here.
9. `null-graph/inference` (generic: device checks, buffer wrapping, both schedules) and `OrtBrain`
   over it — U13, A5, A6. ORT as an optional peer dependency.
10. **Demo:** switch Rules / Network (WGSL) / Network (ORT) live, with per-step timings on screen;
    A8 cloning quality.

Every regression test is watched failing on broken code before it passes
([design-principles](./design-principles.md) §13).

### Deferred — Phase 2, its own document after C2

- **Evolution:** per-agent genomes as weight-table rows, energy and food, reproduction and mutation
  on the GPU — the live ecosystem. Per-agent weights cost bandwidth (2k floats × N per think), so
  the think rate and genome sharing are Phase 2's design problem.
- **Eyes:** 1D ray sensors (P0 of "agents that see") as a declared sensor; rendered eyes after that.
- **Walls, obstacles, flow fields** → the venue-evacuation demo for the event platform.
- Render interpolation, amortized think slices, 3D/flying agents, f16 (detect-only), server-scale
  runs on a Linux GPU.

### Decisions — settled with the user, 2026-10-11

1. **Brain runtime: hybrid.** Rules and the built-in kernel ship; ORT is the optional research path,
   never the default (§9.6).
2. **ORT's home: `null-graph/inference`**, a generic opt-in adapter that agents use first and
   image projects can reuse; ORT an optional peer dependency (§9.4). Revises Cycle A's "no ML
   module" (§1 non-goals).
3. **Training tool:** plain JS, zero install (§9.5).
4. **Cycle C demo:** Herd & Wolves.
5. **Module name:** `null-graph/agents`.
6. **Still open:** the device-ownership amendment (engine first allowed, §9.4) — decided by S5.
7. **C1 steps 3–4 design accepted** (2026-10-11): every refinement in
   [`agents-gpu-loop-design.md`](./agents-gpu-loop-design.md) §1.5, including the brain seam (the
   system builds the think stage). The C1 bundle line is raised to 16 KB (§12.5), and the arrow
   preview phase is kept. Live demos go in the test-engine site's Agent Experiments section.
