# NullGraph Agents — C1 steps 3–4: the GPU loop and renderer output — Technical Design

**Parent:** [`agents-design.md`](./agents-design.md) §13, C1 steps 3 and 4. It is kept in its own
file so the parent stays a plan.
**Builds on:**
- step 1: definitions, records, CPU mirrors (`7f4cdb5`);
- step 2: `PointGrid` / `AgentGrid`, the bounded query, the debug draws ([`api/agent-grid.md`](./api/agent-grid.md)).

**Target:**
- `src/extras/agents/`: `AgentSystem`, the generated sense / think / act / tick / instance stages,
  `RuleBrain`, `inspect`.
- One additive method in `null-graph/visibility`.
- One debug draw in `null-graph/debug-ui`.

**Status:** ✅ **designed and accepted (2026-10-11)**. ✅ Phase 3.0 (spike S3) is measured: §7.1
and §7.3 record what it decided. Next is phase 3.1.
It is built in phases 3.0–3.5 and 4.1–4.5 (§13), each with its own exit criteria. §1.5 lists where
this document refines the parent, and §13.3 records the user's decisions.

---

## 1. Purpose

**Step 3** moves the whole sense → think → act loop onto the GPU, as compute stages in the app's
sequence:
- the grid build (step 2);
- one neighbour query per `nearest` sensor;
- a sense kernel generated from the same declarations the CPU mirror reads;
- a WGSL rule brain;
- the steering integrator;
- the app's own game-logic stages;
- a tick.

The stages repeat once for each fixed step the frame runs. The CPU's work per frame is then small:
count the steps, upload spawns if there are any, and return a prebuilt array.

**Step 4** connects the loop to the renderer with no CPU round trip:
- agents write their own instance rows into `vis.instanceBuffer` and highlight themselves from a
  16-byte uniform;
- labels follow the agent buffer;
- a pick resolves to an agent index;
- an off-frame `inspect(i)` reads one agent back for a "see its brain" panel.

### 1.1 The single most important pre-existing fact

**Every input the loop needs is already a GPU buffer, and every output it feeds is already read
from one.**
- Step 2 left the grid and the neighbour query entirely on the device (`PointGrid.ts`,
  `agentGridShaders.ts`).
- Cycle B left culling, labels and picking reading GPU buffers: `vis.instanceBuffer`, the
  `TextSystem` `follow` option, and the cull's `sourceIndex`.

Steps 3–4 are only the stages in between. The frame path has no readback anywhere and no core
change.

### 1.2 The second fact: per-step state must be GPU state

`preClearBuffers` and every `queue.writeBuffer` are issued while the frame is being encoded. They
therefore land **before the whole submission**, not at their stage's position (fact 1). A frame
that runs three fixed steps in one submission cannot give each step a different uniform.

Step 2 already met this, and resets the grid with a dispatch. Step 3 follows the same rule
everywhere:
- the tick counter is advanced by a one-thread stage;
- whether an agent thinks this step is decided on the GPU, from that tick;
- nothing that varies between steps is written from the CPU.

### 1.3 The third fact: one query is one mask, one radius, one k

`buildAgentGridQueryWGSL(def, mask, radius, k)` scans the 3×3 cells once. It keeps the k nearest
agents of one species mask within one radius: nearest first, ties to the lower index (fact 6).

A layout may declare several `nearest` sensors (Herd & Wolves has wolf k=1 and prey k=3), and each
species senses at its own radius. Both are cheap to handle:
- **One query stage per `nearest` sensor, all sharing one grid build.** Using one `AgentGrid` per
  sensor would rebuild the grid for each one (fact 7).
- **Query at `cellSize` and filter per observer in the sense kernel.** Results are nearest-first,
  so the neighbours inside a smaller radius are a *prefix* of the list. Filtering loses nothing,
  and `senseRadius` can be tuned live up to `cellSize` without a rebuild.

### 1.4 Non-goals (steps 3–4)

- **C2 work:** neural brains, ONNX Runtime and async brains. The think stage is shaped so they slot
  in (§7.1), but none of their parts are built here.
- **Phase 2 work:**
  - evolution and per-agent weights;
  - **killing or creating** agents on the GPU (free lists).
  
  An app stage may *respawn* an agent in its own slot (§8.2).
- Render interpolation, variable dt and 3D motion.
- A fused multi-sensor query, where one cell scan fills several top-k lists. This is an
  optimization to measure first (§13.2).
- Indirect dispatch over live agents only. Stages dispatch over `capacity`, and dead agents return
  early (parent §5.1).
- Label decluttering (handoff, Cycle B known gaps).
- Several `AgentSystem`s sharing one grid.

### 1.5 Where this refines the parent plan

Each row is a deliberate change, with its reason. None of them touches core.

| Parent says | This document says | Why |
|---|---|---|
| §9.1: a `'stage'` brain builds its own stages (`bind(io)`, `getComputeStages()`) | **The system builds the think stage.** A `'stage'` brain contributes WGSL (`fn think`) and, from C2, its own read-only buffers at group 1. `'async'` brains are unchanged | The think-rate gate, the dead and newborn checks and the binding plumbing live in one place, so a brain cannot forget the gate (design-principles §9). S3 decides one detail (§7.1) |
| §9.2: `fn think(i: u32)` | `fn think(i: u32, species: u32, tick: u32)` | One rule set usually serves several species (Herd & Wolves branches on species). WGSL has no RNG, so randomness hashes `(i, tick)` |
| §4.5: step parameters in a 32-byte uniform; the act stage increments the tick | dt, the world, capacity and the cell layout are WGSL constants. A one-thread **tick stage** ends each step and also accumulates grid overflow | These values never change for a system, and step 2's generators already bake them (fact 8). App stages run after act and must see the same tick as think. The grid zeroes `overflow` every step (fact 5) |
| fact 2 and U8: `thinkEvery` disables the sense and think stages | One GPU predicate, `thinksNow(i, agent)`. It is staggered by **64-agent block**, and is true on an agent's first step | Repeated chain copies share node objects, so `enabled` cannot differ between the steps of one frame. Staggering spreads the cost (design-principles §11, corollary); it is the simplest form of the parent's deferred "amortized think slices", and costs one term in the predicate. The stagger is by block because S3 measured staggering by index at 88–111% of the full cost and by block at 32–42% (§7.3). A newborn never acts on its slot's previous row |
| §6.2: the query radius is the sense radius | Query at `cellSize`; each observer filters by its own `senseRadius` | §1.3 |
| A2: obs within 1e-5 absolute | A2 uses a bound derived from WGSL's `sin`/`cos` accuracy, like A3, and reports the measured maximum | Sense rotates into the local frame with `sin`/`cos`, and WGSL promises only 2⁻¹¹ absolute for them (§6.3). 1e-5 is not guaranteed |
| §8.1: bounds `[0, 0, 0, boundsRadius]` | Unchanged; **plus** a refusal for a drawn species whose `boundsRadius` is 0 | `defineAgents` defaults `boundsRadius` to 0, and a zero-radius sphere is culled while the mesh is still on screen |
| §8.4: `inspect` returns `{ record, obs, act, obsNames }` | It also returns `reason`, the neighbour indices, `tick` and the overflow counters | "Why did it do that?" and "did the grid drop anyone?" are the inspector's questions |
| — | New: `setSpecies(name, partial)`, app step stages, and the `agentArrows()` debug draw | Live tuning and late mesh assignment; game logic such as catches on the GPU without readback; seeing step 3 before step 4 exists |

### 1.6 Design principles

1. **The CPU mirror is the definition.** Every generated kernel has a step-1 twin (`senseAgent`,
   `actAgent`). The GPU tests compare against it, with tolerances derived from WGSL's stated
   accuracy, never guessed (§10).
2. **Nothing per step from the CPU.** Anything that varies between steps is GPU state (§1.2).
3. **Build once, return arrays.** Pipelines, bind groups and the chain for every step count are
   built when the system changes shape. `getComputeStages()` returns a cached array (design-principles
   §5, §11).
4. **An invocation writes only its own record, and never a field another invocation of the same
   stage reads.** This rule keeps game logic race-free and reruns bit-exact without atomics (§8.2).
5. **Suppress compute with its consumer.** One predicate gates both sense and think, emitted once
   into both kernels (design-principles §9).

---

## 2. Architecture

```ts
// setup — once
const def = defineAgents({ capacity: 20_000, world, cellSize: 4, species, sensors, actions }); // step 1
const agents = new AgentSystem(engine, def, { thinkEvery: 1 });
agents.setBrain(new RuleBrain(herdWGSL));                       // §7 — the system builds the think stage
agents.addStepStage({ name: 'catches', wgsl: catchesWGSL });    // §8.2 — game logic, per agent per step
agents.addStepStage({ name: 'respawn', wgsl: respawnWGSL });
vis.setInstanceCount(agents.capacity);                          // §9.1 — GPU-owned instance rows (step 4)
agents.toInstances(vis);                                        // §9.2
const names = new TextSystem(engine, { follow: { buffer: agents.buffer, strideFloats: 16, positionOffset: 0 }, /* … */ });

// frame
agents.advance(frameSeconds);   // FixedStep → 0..maxStepsPerFrame steps
agents.upload();                // dirty spawns and species only — usually nothing
vis.upload();                   // cull parameters only, after setInstanceCount
return [
    ...agents.getComputeStages(),   // steps × [grid, queries, sense, think, act, app stages, tick], then instances
    ...vis.getComputeStages(),      // the culls read the rows the agents just wrote
    ...names.getComputeStages(),    // labels follow agents.buffer
    scenePass,
];

// lockstep — tests, dataset recording, replays; never in the frame loop
await agents.step();            // one step, submitted and awaited
```

```
 agents.buffer (64 B per agent, GPU-owned after upload)
   │
   │  ┌──────────────── one fixed step — repeated N times per frame (N = 0..maxStepsPerFrame) ────────────────┐
   └─▶│ grid      reset → bin → scan → scatter                           PointGrid (step 2)                   │
      │ query ×S  one per nearest sensor → neighbours[i·K + base_s ..]   one shared buffer (§6.1)             │
      │ sense     generated from the sensor list → obs[i·obsDim ..]      gated by thinksNow(i, agent)         │
      │ think     brain WGSL → act[i·actDim ..], reason[i]               gated by the same predicate          │
      │ act       steering integrator → agents.buffer, in place                                               │
      │ app       fn step(i, tick) — catches, timers, respawn             own record only (§8.2)               │
      │ tick      one thread: state.tick += 1, overflow accumulated                                            │
      └────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                          │ once per frame, after the last step
                                          ▼
      instances   agents.buffer → vis.instanceBuffer rows (model, bounds, meshIndex + highlight)    step 4
                                          │
          VisibilitySystem culls ─────────┼──────── TextSystem follows agents.buffer ──── Picker: hit.index → agent
                                          ▼
          inspect(i), off-frame: record + obs row + act row + reason + neighbours + counters
```

---

## 3. Integration facts (verified against this repository)

| # | Fact | Source | Consequence |
|---|---|---|---|
| 1 | `preClearBuffers` and `queue.writeBuffer` are issued while the sequence is encoded. Every one of them lands before the submission runs. | `src/core/RenderPipeline.ts:30-44` | Nothing that varies between steps comes from the CPU. The tick is advanced by a stage (§5.3). |
| 2 | A disabled stage skips its clear and its dispatch. The sequence is walked node by node with no de-duplication, so one node may appear many times. | `RenderPipeline.ts:17-28`, `src/core/types.ts:257` | `chains[n]` repeats the same nodes. Gating that differs between repeats is decided on the GPU (§7.3). |
| 3 | Consecutive compute stages share one pass. Dispatches in a pass are ordered, and a storage write is visible to the next dispatch. | `RenderPipeline.ts:45-68` (comment at 58-62) | A whole multi-step chain is one compute pass with no explicit barriers. |
| 4 | Compute pipelines use `layout: 'auto'` and the entry point `cs_main`. An unreferenced binding is dropped from the layout, and bind-group creation then fails. | `src/core/BatchManager.ts:178-182`; handoff, Gotchas ("Group 0 is engine-owned…") | Generated entry points reference every binding, whether or not the app's WGSL uses it (§7.2). |
| 5 | `PointGrid` builds reset → bin → scan → scatter once. Reset is a dispatch that also zeroes `overflow`, so repeated builds in one submission are safe, but `overflow` describes only the last build. `overflow` counts agents past a cell's scan budget. | `src/extras/spatial/PointGrid.ts:51-67` (reset 55-62); `gridShaders.ts:31` | One `PointGrid` per system, repeated every step. The tick stage folds `overflow` into run-long counters (§5.3). |
| 6 | The query generator takes one species mask, a radius ≤ `cellSize` and k ≤ 8. It writes `result[i * k + n]` nearest first, ties to the lower index, padded with `0xffffffff`, scanning at most `maxScanPerCell` agents per cell. | `src/extras/agents/agentGridShaders.ts:7-57` | One query stage per `nearest` sensor. The generator gains `stride` and `base` (§6.1). |
| 7 | `AgentGrid extends PointGrid`, so each `AgentGrid` builds its own grid for one query. | `AgentGrid.ts:8-28` | `AgentSystem` composes one `PointGrid` with raw query stages instead. |
| 8 | Step 2's generators bake the world, the cell size and the capacity into WGSL as literals. | `agentGridShaders.ts:19-34` (`pointCellWGSL`) | Step parameters are constants here too, not a uniform (§1.5). |
| 9 | `defineAgents` refuses `senseRadius > cellSize`, k > 8, obsDim > 256 and actDim > 16. `spawn` refuses a non-positive `speedScale` and sets `age` to 0 unless one is given. | `agentLayout.ts:266,303,328,339`; `agentRecords.ts:81,102` | The 3×3 scan is exact, Σk ≤ 51, `self()` never divides by zero, and age 0 marks a newborn (§7.3). |
| 10 | The CPU definitions are `senseAgent`, `actAgent` and `FixedStep`. Named offsets come from `constantsWGSL` and the structs from `AGENT_STRUCT_WGSL`. | `agentMath.ts:85,162,226`; `agentLayout.ts:46,383` | Kernels are generated from the same declarations. A2 and A3 compare row by row. |
| 11 | `vis.instanceBuffer` is public, with `STORAGE \| COPY_DST`. `upload()` rewrites it from the adopted CPU array every frame. The culls dispatch over the private `instanceCount` and receive it as a parameter. | `src/extras/visibility/VisibilitySystem.ts:71,73,127-130,187-208,270,557` | Step 4 needs `setInstanceCount(n)` (§9.1). |
| 12 | The cull stamps each survivor's source index, and picks report it as `hit.index`. | `visibilityShaders.ts:298`; `pickShaders.ts:52`; `Picker.ts:38-39,128` | Instance `firstInstance + i` is agent `i`. |
| 13 | Labels follow any `STORAGE` buffer, given a stride and a position offset. `setLabelHidden` hides a label from the CPU. | `src/extras/text/TextSystem.ts:114-127,200` | Names follow `agents.buffer` unchanged. A label is hidden by whoever kills its agent (§9.4). |
| 14 | Debug draws are ordinary batches whose generated shaders read GPU buffers. | `src/extras/debug-ui/gridDrawing.ts:68-122` | `agentArrows()` follows the same pattern (§8.4). |
| 15 | Headless suites parse generated WGSL with `wgsl_reflect`. | `package.json:124`; `test/environment-aerial.test.mjs:4,482` | U9 parses every generated module. |
| 16 | `agents.js` is 8,631 B min + gzip after steps 1–2. | esbuild of `src/extras/agents/index.ts`, measured with `gzip -9` | The 12 KB C1 line leaves ~3.4 KB for steps 3–5 (§12.5). |

---

## 4. Data model

### 4.1 Buffers

Every buffer is allocated once, against capacity. The byte counts below are for the Herd & Wolves
layout (obsDim 24, actDim 2, K = 1 + 3 neighbour slots) at **20,000 agents** on a 45 × 45 grid:

| Buffer | Layout | Bytes | Usage | Written by |
|---|---|---|---|---|
| `agents.buffer` | capacity × `Agent` (parent §4.1) | 1.28 MB | `STORAGE \| COPY_DST \| COPY_SRC` | `upload()` (spawns), act, app stages |
| `neighbours` | capacity × K u32; sensor s owns words `[base_s, base_s + k_s)` of each row | 320 KB | `STORAGE \| COPY_SRC` | query stages |
| `obs` | capacity × obsDim f32, ONNX layout (parent §4.3) | 1.92 MB | `STORAGE \| COPY_SRC \| COPY_DST` | sense |
| `act` | capacity × actDim f32 | 160 KB | `STORAGE \| COPY_SRC \| COPY_DST` | think |
| `reason` | capacity × u32: a brain's optional "why", for the inspector | 80 KB | `STORAGE \| COPY_SRC` | think (`setReason`) |
| `state` | `AgentState` | 16 B | `STORAGE \| COPY_SRC \| COPY_DST` | tick stage |
| `species` | uniform `array<Species, 16>` (`packSpecies`) | 512 B | `UNIFORM \| COPY_DST` | `upload()` after `setSpecies` |
| grid | step 2: counts, cursors, offsets (+1), indices, overflow, block totals | ~105 KB | step 2 | grid stages |
| **step 4** `highlight` | uniform `Highlight` | 16 B | `UNIFORM \| COPY_DST` | `setHighlight()` |
| **step 4** instance rows | `vis.instanceBuffer` rows `firstInstance + i`, 24 floats each | 1.92 MB | owned by visibility | instance stage |
| **step 4** inspector staging | record + obs row + act row + reason + K neighbours + `AgentState` | 204 B | `MAP_READ \| COPY_DST` | `inspect(i)` |

**Total: ≈ 5.8 MB, allocated once.** That is the parent's 5.4 MB plus the neighbour and reason
buffers. `COPY_SRC` on obs, act, reason and neighbours exists for the inspector and the tests only;
nothing reads back on the frame path.

### 4.2 New structs

```wgsl
struct AgentState {        // 16 B, storage — written only by the tick stage
    tick: u32,             // steps run since construction
    overflowSteps: u32,    // steps whose grid left someone out of the queries
    overflowTotal: u32,    // the sum of PointGrid.overflow over those steps
    _pad: u32,
};
struct Highlight {         // 16 B, uniform — written by setHighlight()
    selected: u32,         // agent index; 0xffffffff = none
    hovered: u32,
    _pad: vec2<u32>,
};
```

The world bounds, capacity, `dt`, the cell layout, `thinkEvery`, `firstInstance` and the obs and act
dimensions are **WGSL constants**, because `defineAgents` and construction fix them for the
system's life (fact 8). Species parameters stay in a **uniform**, so an app can tune them live.

### 4.3 Bindings per stage — the storage budget

Core WebGPU guarantees 8 storage buffers per stage, and NullGraph requests default limits.

| Stage | Group 0 storage | Uniforms | Storage used |
|---|---|---|---|
| grid (4 kinds) | step 2 | — | ≤ 4 |
| query, per sensor | agents, counts, offsets, indices (read); neighbours (rw) | — | 5 |
| sense | agents, neighbours, state (read); obs (rw) | species | 4 |
| think | agents, obs, state (read); act, reason (rw) | species | 5, + the brain's group 1 (C2) |
| act | act (read); agents (rw) | species | 2 |
| app step stage | neighbours, obs, act, state (read); agents (rw) | species | 5, + the app's group 1 (≤ 3) |
| tick | overflow (read); state (rw) | — | 2 |
| instances | agents (read); instance rows (rw) | species, highlight | 2 |

The largest system-owned stage uses 5, which leaves 3 for an app stage's or a brain's own buffers.
`addStepStage` checks against `device.limits.maxStorageBuffersPerShaderStage`. Compute stages do
not spend the fragment budget (parent fact 19). `agentArrows()` binds `agents.buffer` in its own
pipeline, as `gridOccupancy()` binds `cellCounts`.

---

## 5. The system and the step chain

### 5.1 API

```ts
const agents = new AgentSystem(engine, def, { thinkEvery?: number });   // def from defineAgents()
agents.buffer                                   // GPUBuffer, 64 B × capacity — follow it, bind it
agents.records                                  // step 1's AgentRecords: spawn, kill, adopt, dirty range
agents.spawn(i, init) · agents.kill(i)          // shorthands for agents.records
agents.setSpecies(name, partial)                // live tuning or late mesh assignment (§5.2)
agents.setBrain(brain)                          // §7 — builds the think stage, rebuilds the chains
agents.addStepStage({ name, wgsl, buffers? })   // §8.2 — builds a stage, rebuilds the chains
agents.toInstances(vis, { firstInstance? })     // §9.2 — step 4
agents.setHighlight({ selected, hovered })      // §9.3 — step 4
agents.advance(frameSeconds): number            // FixedStep → steps this frame
agents.upload(): void                           // dirty records and species; nothing else
agents.getComputeStages(): ComputeStageNode[]   // the prebuilt chain for this frame's step count
agents.step(): Promise<void>                    // lockstep: upload, one step, await the queue
agents.inspect(i): Promise<AgentSnapshot>       // §9.6 — step 4
agents.destroy(): void
```

### 5.2 Ownership and shape changes

- **Records.** Spawning, killing and adopting a population are step 1's `AgentRecords` (parent
  §5.1). After `upload()` the GPU owns a record; the CPU array is a staging area, not a mirror.
- **Shape changes happen at call time, never per frame.** `setBrain`, `addStepStage`, `toInstances`
  and construction build pipelines and bind groups and rebuild the chains (design-principles §11).
  Stages are built from what is registered: one query per declared `nearest` sensor, and an
  instance stage only after `toInstances` (design-principles §10).
- **`setSpecies(name, partial)`** re-validates with `defineAgents`' own rules and messages, repacks
  the table, and marks it dirty; `upload()` writes 512 bytes. The new values apply to every step
  of the next frame. A `senseRadius` up to `cellSize` is safe, because the queries run at
  `cellSize` (§1.3). The same call assigns `mesh`, `hoverMesh`, `selectedMesh` and `boundsRadius`
  once meshes are registered.
- **`destroy()`** disables every stage and destroys the buffers the system owns, including
  `agents.buffer`. Dependents go first: a `TextSystem` following `agents.buffer` still binds it,
  so the app destroys it, or stops following, before `agents.destroy()` (A20).

### 5.3 The chain and the tick

One step is the stage list `[...grid, ...queries, sense, think, act, ...appStages, tick]`.
`chains[n]`, for `n = 0..maxStepsPerFrame`, is that list repeated `n` times, followed by the
instance stage once `toInstances` is wired. `chains[0]` still holds the instance stage, so a
highlight change shows on a frame that runs no step.

`advance()` runs step 1's `FixedStep` and selects the chain. `getComputeStages()` returns it. No
array is built per frame (U8).

The **tick stage** is one invocation, last in every step:

```wgsl
state.tick += 1u;
if (overflow > 0u) { state.overflowSteps += 1u; state.overflowTotal += overflow; }
```

It is the only per-step variation, and it lives on the GPU (§1.2). It also turns `overflow`, which
the grid zeroes every step (fact 5), into counters that cover the whole run, for the inspector and
the tests.

### 5.4 Lockstep

`step()` calls `upload()`, renders `chains[1]`, and awaits `device.queue.onSubmittedWorkDone()`.
It is used by tests, dataset recording and replays, never by the frame loop. C2's async brains
extend it to sense → await the run → act (parent §9.4).

---

## 6. Neighbours and sensing

### 6.1 Queries: one grid build, one shared neighbour buffer

`AgentSystem` builds one `PointGrid` over `agents.buffer` (stride 16, flags word 13, alive mask 1)
and one query stage per `nearest` sensor s:

```
query_s = buildAgentGridQueryWGSL(def, nearestMasks[s], def.cellSize, k_s, { stride: K, base: base_s })
```

The generator gains two parameters:
- `stride`, default `k`;
- `base`, default 0.

With them, every query writes its own slice of one `neighbours` buffer. The sense stage then binds
**one** buffer whatever the sensor count. Defaults reproduce step 2's text exactly, so `AgentGrid`
and its tests are untouched (U16).

Over the scan budget, the bounded query can miss neighbours. The miss is counted in `overflow`,
folded into `state` (§5.3), shown red by `gridOccupancy()`, and reported by `inspect()`. A2 and A7
run overflow-free scenes and assert `overflowSteps == 0`.

### 6.2 The sense kernel, generated from the declarations

There is one kernel per system and one invocation per agent. It is emitted from `def.sensors` in
declaration order, mirroring `senseAgent` (`agentMath.ts:85`):

| Sensor | WGSL |
|---|---|
| gate | `if (!thinksNow(i, a)) { return; }` — the row keeps its last values (§7.3) |
| `self()` | `clamp(length(v.xz) / (maxSpeed × speedScale), −1, 1)`, `clamp(energy, −1, 1)` |
| `nearest()` | per rank r: `j = neighbours[i·K + base + r]`. If `j ≠ 0xffffffff` and its squared XZ distance ≤ the observer's `senseRadius²` (inclusive, as the mirror), write side and forward of Δpos ÷ `senseRadius`, side and forward of Δvel ÷ (2 × `maxSpeed`), and present = 1. Otherwise write five zeros |
| `bounds()` | the nearest wall (ties −X, +X, −Z, +Z), its outward normal in the local frame × max(0, 1 − max(d, 0) ÷ `senseRadius`) |
| `custom()` | the app's `fn <name>(i: u32, a: Agent) -> array<f32, width>`, written at its offset |

**Dead rows are don't-care on the GPU** (parent §4.3). The mirror writes zeros, but writing
`obsDim` zeros for every dead slot every step would be pure bandwidth. A2 compares live rows only.

### 6.3 Tolerance, derived

The local frame uses `sin` and `cos` of the heading, once per agent. WGSL promises them only to
within 2⁻¹¹ absolute on [−π, π], and headings are wrapped into that range.

A rotated feature is `(dx·cos − dz·sin) / r`, where `|dx| + |dz| ≤ √2 · r` inside the sense
radius. Its error is therefore at most `√2 × 2⁻¹¹ ≈ 6.9e-4`, plus f32 rounding. The velocity
features obey the same bound, because `|Δv| ≤ 2 × maxSpeed`.

**A2 asserts 7e-4 and reports the measured maximum per GPU.** About 1e-6 is expected on real
hardware. Presence is a comparison, so the radius boundary is tested twice:
- a hand-built neighbour at exactly `senseRadius` must be present on both sides;
- random scenes skip, and count, observers with a candidate within 1e-4 m of their boundary (about one in
  10,000 at the demo's density).

---

## 7. Thinking

### 7.1 The seam — amended from parent §9.1

```ts
interface Brain {
    readonly kind: 'stage' | 'async';
    // 'stage' — RuleBrain now, MlpBrain in C2. The system builds the think stage around this WGSL,
    // which defines fn think(i: u32, species: u32, tick: u32).
    readonly wgsl?: string;
    readonly buffers?: readonly GPUBuffer[];        // C2: read-only, bound at group 1 in order
    readonly dims?: { obs: number; act: number };   // checked by setBrain (parent §11, "Brain dimensions")
    // 'async' — OrtBrain in C2: parent §9.1–9.4, unchanged
    bind?(io: { obs: GPUBuffer; act: GPUBuffer; capacity: number; obsDim: number; actDim: number }): void;
    run?(): Promise<void>;
    destroy(): void;
}
```

The generated think stage includes the structs, the constants, the bindings, the helpers (§7.2) and
the brain's WGSL, then:

```wgsl
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    // every binding referenced here, whatever think() uses (fact 4)
    let i = id.x;
    if (i >= CAPACITY) { return; }
    let a = agents[i];
    if (!thinksNow(i, a)) { return; }
    think(i, a.ids.x, state.tick);
}
```

**S3's answer (phase 3.0, 2026-10-11): the entry returns early.** The open detail was whether a
network kernel would stage its weights in workgroup memory. Staging needs `workgroupBarrier()`,
which WGSL allows only in uniform control flow, so the entry could not return early.

S3 measured the 24→32→32→2 network on NVIDIA Turing and on an AMD GCN-5 iGPU, under Node/Dawn and
Chrome (parent §13, C0):
- **Staging is 2.1–2.3× faster on NVIDIA and 2.1–2.3× slower on AMD.** The iGPU pays for the
  staging traffic in every workgroup.
- **Staging only fits small networks.** The workgroup-memory limit is 16 KiB by default.
- **Staging cannot serve Phase 2's per-agent weights.**

Weights read from storage are within about 2.3× of the fastest variant everywhere. A network at
20,000 agents costs 0.18 ms on NVIDIA and 0.28–0.34 ms on AMD. So `think` is called after the
early returns, as shown above. Staging stays a measured per-GPU option for later (§13.2).

### 7.2 `RuleBrain` and the helpers

```ts
agents.setBrain(new RuleBrain(/* wgsl */ `
fn think(i: u32, species: u32, tick: u32) {
    if (species == SPECIES_PREY) {
        let w = OBS_WOLF_0;
        if (obs(i, w + NEAR_PRESENT) > 0.0) {                  // flee: turn away from the wolf
            setAct(i, ACT_TURN, clamp(-4.0 * obs(i, w + NEAR_SIDE), -1.0, 1.0));
            setAct(i, ACT_THROTTLE, 1.0);
            setReason(i, 1u);
            return;
        }
    }
    setAct(i, ACT_TURN, rand01(i, tick / 32u, 0u) * 2.0 - 1.0); // wander: a new turn every 32 steps
    setAct(i, ACT_THROTTLE, -0.4);
    setReason(i, 0u);
}`));
```

The system provides:
- `obs(i, k) -> f32`;
- `setAct(i, k, v)`;
- `setReason(i, r)`;
- `rand01(i, tick, stream) -> f32`, a PCG-style integer hash: exact and identical on every GPU and
  in its JS twin;
- the constants from `constantsWGSL` (`OBS_*`, `NEAR_*`, `SELF_*`, `BOUNDS_*`, `ACT_*`);
- one `SPECIES_<NAME>` constant per species.

`obs` is bound read-only, so a brain cannot write its own inputs. Dead agents never reach `think`.
`RuleBrain` refuses WGSL that lacks the `think` signature, by name.

### 7.3 The think rate

```wgsl
fn thinksNow(i: u32, a: Agent) -> bool {
    return (a.ids.y & 1u) != 0u && (a.state.y == 0.0 || (state.tick + i / 64u) % THINK_EVERY == 0u);
}
```

- **Staggered by 64-agent block.** With `thinkEvery = n`, the agents of block `b = i / 64` sense
  and think on the steps where `(tick + b) % n == 0`. Each step does 1/n of the work, instead of
  all of it every nth step. Between thinks, an agent keeps acting on its last action row.
- **Why blocks, not single agents.** The lanes of one wavefront execute together, so a wavefront
  with any agent thinking runs the whole brain. S3 measured staggering by index at 88–111% of the
  full cost, on all three hosts, and staggering by block at 32–42%. 64 is the workgroup size, so a
  block is whole wavefronts on any subgroup size up to 64.
- **Newborns think at once.** `age == 0` marks an agent that has never acted (spawn sets it, the
  `respawn` helper resets it, and act advances it by `dt`). It never steers by the previous
  occupant's last row. Its wavefront runs the brain that step even if its block is not due, which
  is rare and costs one wavefront. An app that spawns with an explicit nonzero `age` is restoring a
  population and opts out of this.
- **One predicate, two kernels.** It is emitted once and called by both sense and think
  (principle 5). It lives on the GPU because repeated chain copies share node objects (fact 2).
- **Queries stay ungated.** App stages read `neighbours` every step. Gating queries for agents
  that do not think is a measured optimization for later (§13.2).
- `thinkEvery = 1`, the default, makes the modulo a constant the compiler folds away.

---

## 8. Acting and game logic

### 8.1 The act kernel

This is the WGSL twin of `actAgent` (`agentMath.ts:162`), in the mirror's order:
1. age += dt;
2. a frozen agent zeroes its velocity and stops there;
3. sanitize;
4. turn;
5. approach the target speed;
6. move;
7. X walls, then Z walls, each clamped and reflected only when moving into the wall;
8. velocity along the final heading.

The non-finite test is the **exponent-bit test** `(bitcast<u32>(v) & 0x7f800000u) == 0x7f800000u`,
because WGSL lets a compiler assume values are never NaN or infinite, and `v != v` may be optimized
away (parent §7.2). Dead agents are untouched.

`Actions.custom({ name, width, wgsl })` replaces steps 3–8 with the app's `fn <name>(i: u32)` over the same
bindings.

**A3's tolerance** is the parent's derived bound:
- the heading to f32 rounding;
- velocity to `speed × 2⁻¹¹`;
- position to `speed × dt × 2⁻¹¹` per step.

Headings are compared as **wrapped angular differences**. A heading within an f32 ulp of ±π may
wrap on one side and not the other, which would be a 2π "error" between identical directions.

### 8.2 App step stages — game logic on the GPU

```ts
agents.addStepStage({ name: 'catches', wgsl: `
const CAUGHT = 1u << 8u;                                    // an app flag bit (bits 8+)
fn step(i: u32, tick: u32) {
    let me = agents[i];
    if (me.ids.x == SPECIES_PREY) {
        let w = neighbours[i * NEIGHBOURS_PER_AGENT + NEIGHBOURS_WOLF];
        if (w != NO_AGENT && distance(agents[w].posRadius.xz, me.posRadius.xz) < 0.9) {
            agents[i].ids.y = me.ids.y | CAUGHT;            // my flags — nobody reads them in this stage
        }
    } else if (me.ids.x == SPECIES_WOLF) {
        let p = neighbours[i * NEIGHBOURS_PER_AGENT + NEIGHBOURS_PREY];
        if (p != NO_AGENT && distance(agents[p].posRadius.xz, me.posRadius.xz) < 0.9) {
            agents[i].state.w = 5.0;                        // my rest timer (user0)
        }
    }
}` });
agents.addStepStage({ name: 'respawn', wgsl: `
const CAUGHT = 1u << 8u;
fn step(i: u32, tick: u32) {
    if ((agents[i].ids.y & CAUGHT) == 0u) { return; }
    respawn(i, randomPointInWorld(i, tick), rand01(i, tick, 1u) * 6.2831853 - 3.1415927);
    agents[i].ids.y &= ~CAUGHT;
}` });
```

Each app stage runs once per live agent per step, after act and before the tick. It gets the system's
bindings (§4.3) and, through `buffers`, up to three of its own at group 1, such as a score counter
the app increments atomically.

The generated helpers are:
- `NEIGHBOURS_PER_AGENT` and `NEIGHBOURS_<SENSOR>`;
- `NO_AGENT`;
- `rand01`;
- `randomPointInWorld(i, tick)`;
- `respawn(i, xz, heading)`. It writes the agent's own position, heading and zero velocity, and
  resets `age` to 0 so the agent thinks on its next step (§7.3).

**The rule that keeps it correct.** An invocation writes only its own record, and never a field
another invocation of the same stage reads. That is why marking a catch and moving an agent are two
stages:
- In `catches`, nobody writes a position. A prey marks itself, and a wolf that sees the same prey
  inside the same distance starts its own timer.
- In `respawn`, a caught prey moves itself, and nobody else reads positions.

Put both in one stage and a wolf could read a prey's position mid-respawn. That is a data race: a
rerun can differ, and A7's bit-exact guarantee is gone. No two invocations ever write the same
word, so the stages need no atomics on agent records, and lockstep runs stay deterministic (A18).

### 8.3 Determinism, and the safe directions

- **Bit-exact reruns on one device** require four things:
  - only built-in sensors;
  - a brain that reads only obs, `i`, `tick` and species;
  - app stages that follow §8.2's rule;
  - **zero overflow**.
  
  A7 checks it.
- **Overflow is the stated exception, not a hidden one.** Past the scan budget, which candidates
  the query sees depends on atomic scatter order (step 2's API doc), so reruns can differ. It is
  counted (§5.3), drawn red by `gridOccupancy()`, and removed by a larger `maxScanPerCell` or a
  smaller cell.
- **Safe directions:**
  - a broken brain stops its agents (§8.1);
  - a missed neighbour is surfaced, never silent;
  - an agent spawned outside the world turns back in (parent §7.2).

### 8.4 Debug view — `agentArrows()` (step 3)

`debug-ui` gains `agentArrows(engine, pass, agents, { colors? })`. It is one instanced draw: an
arrow per agent, with the vertex shader reading position and heading straight from
`agents.buffer`. Arrows are coloured by species, and dead agents are clipped.

It makes step 3 visible before step 4 exists, and stays useful afterwards because it needs no
meshes, no visibility system and no picking. It is the agents' counterpart of `gridOccupancy()`,
built the same way (fact 14), and returns `{ batch, destroy() }`.

---

## 9. Output to the renderer — step 4

### 9.1 `vis.setInstanceCount(n)` — GPU-owned instance rows

This is additive in `VisibilitySystem`, as parent §12.2 specifies:
- it sets `instanceCount` and **drops the adopted CPU array**, so `upload()` writes only the cull
  parameters;
- `setInstances(data, count)` switches back;
- a read-only `instanceSource: 'cpu' | 'gpu'` lets `toInstances` refuse a system that would
  overwrite the rows every frame.

Static instances, such as props sharing the system with the agents, are written once with
`device.queue.writeBuffer(vis.instanceBuffer, …)`. The buffer is public and `COPY_DST` (fact 11),
and nothing overwrites it any more.

### 9.2 The instance stage

`agents.toInstances(vis, { firstInstance = 0 })` adds one stage. It runs **once per frame, after
the last step**, and writes row `firstInstance + i`:

| Floats | Value |
|---|---|
| 0–15 | `translate(position) · rotateY(heading)`. Local +Z maps to the forward `(sin h, 0, cos h)`, and local +X to the side `(cos h, 0, −sin h)` (parent §7.1's frame) |
| 16–19 | `[0, 0, 0, boundsRadius]`, in object space as `InstanceBuffer.write` expects |
| 20 | meshIndex: the species' `selectedMesh` or `hoverMesh` by the highlight uniform, else `mesh` |
| 21 | `LOD_AUTO` |
| 22 | flags, 0 |
| 23 | 0 (the cull writes `sourceIndex` into its compacted copies) |

A dead agent's row gets only float 20, set to `HIDDEN_MESH`, which no batch draws (parent §3,
fact 6). Mesh indices and bounds come from the species table, set in the definition or later with
`setSpecies`. `toInstances` refuses four cases:
- a `VisibilitySystem` from another engine;
- `maxInstances < firstInstance + capacity`;
- `instanceSource === 'cpu'`;
- a species with a drawn mesh and `boundsRadius` 0. A covering radius about the mesh origin is
  always correct, which is the conservative direction.

### 9.3 Highlight

`agents.setHighlight({ selected, hovered })` writes the 16-byte `Highlight` uniform, and only when
it changes. This is the Cycle B and CPU-preview mesh-variant trick moved to the GPU, because the
CPU no longer owns the records. `-1` means none and is stored as `0xffffffff`.

### 9.4 Labels

The API is unchanged: `follow: { buffer: agents.buffer, strideFloats: 16, positionOffset: 0 }`, and
the label for agent `i` follows record `i`.

In C1 only the CPU kills agents (`agents.kill(i)`), so the app hides that label with
`names.setLabelHidden(i, true)` (fact 13). Prey in Herd & Wolves respawn rather than die. A kill on
the GPU arrives with Phase 2's birth and death. Hiding labels then needs a `follow.visibleIf:
{ wordOffset, mask }` test in the text glyph cull (`textShaders.ts:88-97`), designed with Phase 2,
not built now.

### 9.5 Picking

The app registers each species mesh and its hover and selected variants with
`picker.addInstances(vis, meshName, meshIndex)`. Then `hit.index - firstInstance` is the agent
(fact 12). Dead agents are `HIDDEN_MESH`, so no batch draws or picks them (A12).

### 9.6 Inspector

`agents.inspect(i): Promise<AgentSnapshot>` works like this:
1. One encoder copies the record, the obs row, the act row, the reason word, the K neighbour
   indices and `AgentState` into one 204-byte `MAP_READ` staging buffer (§4.1).
2. It maps the buffer.
3. It decodes into the snapshot:

```ts
interface AgentSnapshot {
    index: number; alive: boolean; frozen: boolean; species: string;
    position: [number, number, number]; heading: number; speed: number; energy: number; age: number;
    obs: { name: string; value: number }[];      // def.obsNames
    act: { name: string; value: number }[];      // def.actNames
    reason: number;
    neighbours: { sensor: string; indices: number[] }[];   // NO_AGENT slots dropped
    tick: number; overflowSteps: number; overflowTotal: number;
}
```

Requests are queued one at a time, as `Picker` does, and stay off the frame path. This is the "click
a unit, see its brain" view: what the agent senses, what it chose, why, and whether the grid hid
anyone from it.

### 9.7 The CPU's per-frame budget

Per frame, the CPU calls:
- `advance()`;
- `upload()`, which does nothing unless spawns or species changed;
- `setHighlight()`, only on a change;
- `getComputeStages()`, which returns a cached array.

A19 measures it: zero typed arrays constructed per frame, and at most two `writeBuffer` calls from
the agent system in a frame without spawns.

---

## 10. Test suite

IDs continue the parent's §10: U8, U9, A2, A3, A7 and A10–A12 keep their meanings.

**Headless** (`npm run test:agents`):

| # | Asserts |
|---|---|
| U8 | `chains[n]` is prebuilt for n = 0..maxStepsPerFrame. `advance()` and `getComputeStages()` construct no typed arrays (counted). `thinksNow` is emitted once and called by both sense and think *(amended from the parent's `enabled` gating, §1.5)* |
| U9 | Every generated module parses with `wgsl_reflect`: the query with stride and base, sense, think, act, each app stage, tick, instances and the arrows draw. Each declares its entry point, references every binding, emits each constant once, and contains no backtick |
| U14 | Every new §11 error fires and names the offender (extends the parent's U14) |
| U16 | The query generator's `stride` and `base` defaults reproduce step 2's text byte for byte. Slices for Σk never overlap |
| U17 | `RuleBrain`, custom sensors, custom actions and app stages refuse a missing signature by name. `setBrain` refuses other dimensions |
| U18 | `setInstanceCount(n)`, on a mock device: `upload()` issues no instance write, and the params carry count n. **Watched failing** before the method exists |
| U19 | The instance row's CPU twin: local +Z maps to the forward and local +X to the side; `HIDDEN_MESH` for the dead; mesh selection by the highlight; bounds `[0, 0, 0, r]` |
| U20 | `inspect()` decoding against a hand-built staging buffer: names, flags, neighbours, counters |

**GPU** (`test/agents-gpu`, Node + Dawn and Chrome):

| # | Asserts |
|---|---|
| A2 | obs equals `senseAll` on live rows for 10,000 agents in overflow-free scenes (`overflowSteps == 0` asserted), within the §6.3 bound. Measured maximum reported |
| A3 | One step equals `stepAll` for 10,000 agents, within §8.1's bound, with headings compared as wrapped differences. Measured maximum reported |
| A7 | Two lockstep runs of 600 steps from one seed give byte-identical `agents.buffer` contents (overflow-free, asserted) |
| A10 | Picking an agent after 120 moving steps returns its agent index |
| A11 | A label tracks its agent for 120 steps with zero CPU writes to label data |
| A12 | Dead agents draw nothing and are never picked |
| A13 | `chains[3]` in one submission equals three single-step submissions, byte for byte, with a brain that reads `tick` |
| A14 | Act rows poisoned with NaN and ±Inf on the GPU: those agents slow to a stop, and every record stays finite (the GPU twin of U5c) |
| A15 | A fixture `RuleBrain` that uses `rand01` matches its JS twin on 10,000 CPU-built obs rows. Equality is exact, since the fixture uses no trig |
| A16 | With `thinkEvery = 3`, a fixture brain that writes `tick` into its act row: each row changes only on its block's steps, and a newly spawned agent thinks on its first step |
| A17 | Instance rows written by the GPU equal U19's twin applied to read-back records |
| A18 | `catches` + `respawn`: caught prey reappear inside the world, no record is non-finite, and two lockstep runs agree byte for byte |
| A19 | Per frame on the CPU: zero typed arrays constructed, and at most two agent-system `writeBuffer` calls without spawns |
| A20 | `destroy()` mid-run, dependents first and the sequence updated: later frames record zero WebGPU errors |

**Watched failing** (design-principles §13): each of these bugs is injected into the generated
WGSL or the code, and must be caught.
- A2: a flipped side axis.
- A3: a skipped wall reflection.
- A13: the tick taken from a CPU-written uniform.
- A16: think left ungated.
- U18: the method before it exists.

The suite's non-zero exit is already verified (parent U15).

---

## 11. Errors

The parent's messages are reused verbatim where they exist, marked †.

| Condition | Message |
|---|---|
| No brain | `AgentSystem: no brain — call setBrain() before getComputeStages() or step().` |
| Brain dimensions † | `AgentSystem: the brain maps <bo>→<ba>; the layout is <o>→<a>.` |
| Rule brain signature | `RuleBrain: the WGSL must define fn think(i: u32, species: u32, tick: u32).` |
| Custom sensor signature | `AgentSystem: custom sensor "<name>" must define fn <name>(i: u32, a: Agent) -> array<f32, <w>>.` |
| Custom actions signature | `AgentSystem: custom actions "<name>" must define fn <name>(i: u32).` |
| App stage signature | `AgentSystem: step stage "<name>" must define fn step(i: u32, tick: u32).` |
| App stage twice | `AgentSystem: step stage "<name>" is added twice.` |
| App stage bindings | `AgentSystem: step stage "<name>" binds <n> buffers; <m> are left after the system's <s>.` |
| Think rate | `AgentSystem: thinkEvery <n> must be a positive integer.` |
| Buffer past the device limit | `AgentSystem: <buffer> needs <n> bytes; this device binds at most <m>. Lower capacity or the sensor count.` |
| `setSpecies` values | `defineAgents`' messages, e.g. `AgentSystem: species "<s>" senses <r> m but cells are <c> m — …` |
| Unknown species † | `AgentSystem: species "<s>" is not declared.` |
| Instances owned by the CPU † | `AgentSystem: toInstances() writes vis.instanceBuffer on the GPU; use vis.setInstanceCount(n), not setInstances() (it re-uploads every frame).` |
| Instance room | `AgentSystem: toInstances() needs maxInstances ≥ <first + capacity>; this VisibilitySystem has <max>.` |
| Foreign system † | `AgentSystem: that VisibilitySystem belongs to a different engine.` |
| Zero bounds | `AgentSystem: species "<s>" draws mesh <m> with boundsRadius 0 — the cull would drop it while on screen. Set boundsRadius to the mesh's radius about its origin.` |
| Index out of range † | `AgentSystem: index <i> is outside capacity <n>.` (spawn, kill, inspect, setHighlight) |
| After destroy † | `AgentSystem: used after destroy().` |
| Visibility count | `VisibilitySystem: setInstanceCount received <n> instances, exceeding maxInstances <m>.` (the wording of `setInstances`, `VisibilitySystem.ts:189`) |

All of these are thrown at construction or at the call, never per frame (design-principles §2).
The capacity's dispatch limit is already enforced by `PointGrid` (`PointGrid.ts:41-46`).

---

## 12. Engine touchpoints

### 12.1 Core — required: **none**

Compute stages, storage buffers, repeated nodes, `enabled`, and `queue.writeBuffer` for uploads all
ship. The core bundle grows by **0 B**.

### 12.2 Visibility — required for step 4: `vis.setInstanceCount(n)` and `instanceSource`

This is the parent's §12.2, unchanged: about +40–60 B min + gzip, with U18 watched failing first.

**Zero-change fallback:** keep a dummy CPU array, call `setInstances(dummy, n)`, and let the
instance stage overwrite the uploaded garbage every frame. It works, at the cost of 1.92 MB of
pointless upload per frame at 20,000 agents (≈ 115 MB/s at 60 Hz) plus a CPU array that mirrors
nothing.

### 12.3 Inside `null-graph/agents` — step-2 code, not an engine change

The query generator gains `stride` and `base`, with defaults that keep step 2's output identical
(U16).

**Without it:** one neighbour buffer per `nearest` sensor. That works, but the sense stage's
storage count grows with the sensor list toward the limit of 8 (§4.3).

### 12.4 Text — none in C1

Hiding labels on GPU-side death needs `follow.visibleIf`, which is deferred to Phase 2 (§9.4).

### 12.5 Bundle

`agents.js` is 8,631 B after steps 1–2 (fact 16). Steps 3–4 add:
- `AgentSystem`;
- the sense, think, act, tick, app-stage and instance generators;
- `RuleBrain` and its helpers;
- the inspector decoder;
- error strings.

That is an estimated **+4–5 KB**, mostly WGSL text, which compresses well. It lands at about 13 KB.
**The C1 line is 16 KB for the whole entry** (raised from 12 KB, decision 2 in §13.3). The frame
path's cost after tree-shaking is reported beside it, because part of today's 8.6 KB is CPU mirrors
that apps tree-shake away (`senseAll`, `nearestNeighbours`, `buildPointGrid`) and that the C2 tools
need. Visibility grows by about 60 B, `debug-ui` by about 1 KB (`agentArrows`), core by 0 B.

---

## 13. Build order — phases

### 13.1 Phases

Every phase ends the same way:
- its tests are green on both GPU hosts;
- the bundle is measured;
- the parent's step entry and the handoff are updated;
- the change is committed separately when the user asks.

| Phase | Builds | Exit criteria | What becomes visible |
|---|---|---|---|
| **3.0** ✅ | Spike **S3** (parent §13, C0): `test/spikes/s3-think-kernel/`, `npm run spike:s3`. A script with no library code: the 24→32→32→2 network as one kernel, one invocation per agent, at 1k / 20k / 100k agents. It compares weights in storage (runtime-sized or fixed-size), uniform and workgroup memory, loops against unrolled code, index against block staggering, and probes `tanh` at large inputs | Done 2026-10-11 on NVIDIA Turing and AMD GCN-5 under Node/Dawn, and AMD under Chrome; every variant matched the CPU forward pass. Numbers are in the parent. The entry returns early (§7.1), and the stagger is by block (§7.3) | — |
| **3.1** | `AgentSystem` skeleton: buffers, `upload()`, the species uniform, `setSpecies`, `PointGrid` + query stages into the shared buffer, chains, the tick stage, `step()`, `destroy()` | U8, U16; A13 over grid, queries and tick, with a stub brain that reads `tick` (re-run as each stage lands) | — |
| **3.2** | The sense generator | U9 (sense); A2 | — |
| **3.3** | The act generator | U9 (act); A3, A14, A7 | — |
| **3.4** | The think stage, `RuleBrain`, helpers, `thinkEvery`, app step stages | U17; A15, A16, A18 | — |
| **3.5** | `agentArrows()`; a live demo in the test-engine site (`D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`), route `/dashboard/agents/herd-wolves-gpu` under Agent Experiments, beside the CPU preview (`/dashboard/agents/herd-wolves-preview`): about 4,000 agents, the preview's brain ported to WGSL, catches and respawn as app stages, the grid overlay as a toggle | `npm run drive` exits 0 with zero WebGPU errors; screenshots reviewed | The herd running entirely on the GPU, drawn as arrows |
| **4.1** | `vis.setInstanceCount` + `instanceSource` | U18, watched failing first | — |
| **4.2** | The instance stage, the `Highlight` uniform, `setHighlight`, the zero-bounds refusal | U19; A17 | Darts instead of arrows |
| **4.3** | Picking and labels on GPU agents | A10, A11, A12 | Hover, click, wolf names |
| **4.4** | `inspect(i)` and the site inspector panel | U20; A19 | "See its brain" from GPU values, with overflow shown |
| **4.5** | The site route switched to instances, picking, labels and the inspector; CPU per-frame time measured; `docs/api/agents.md` started | A20; `npm run drive` clean; numbers in the handoff | The step-3 demo with the CPU nearly idle |

Step 5 follows in the parent plan: the 20,000-agent Herd & Wolves demo, with A9's step time
measured.

### 13.2 Deferred — with the reason

| Item | Reason |
|---|---|
| A fused multi-sensor query (one scan, several top-k lists) | Measure the S separate scans first |
| Gating queries for agents that do not think this step | App stages read neighbours every step; measure first |
| Indirect dispatch over live agents only | Dispatching over capacity costs early returns, not work |
| Killing and creating agents on the GPU (free lists), and the `visibleIf` text option it needs | Phase 2's birth and death |
| Time-budgeted think slicing beyond the index stagger, and live `thinkEvery` changes | No measured need yet |
| `addStepStage({ when: 'frame' })` for once-per-frame logic | No consumer yet |
| Per-species brains (`setBrain(brain, { species })`) | C2 decides it with behaviour cloning; one network for two policies needs a species input or two networks |
| Network weights staged in workgroup memory, or read from a uniform buffer, where measured faster | S3: staging is 2.1–2.3× faster on NVIDIA and as much slower on AMD; a uniform buffer is 10–19% faster on NVIDIA and level on AMD. Staging needs an additive entry variant that keeps uniform control flow |

### 13.3 Decisions — settled with the user, 2026-10-11

1. **The §1.5 refinements are accepted as a set**, including the brain seam: the system builds the
   think stage, and a brain contributes WGSL.
2. **The C1 bundle line for `agents.js` is raised from 12 KB to 16 KB** for the whole entry. The
   frame path's tree-shaken cost is reported beside it (§12.5).
3. **Phase 3.5's arrow preview is kept.** Its live demo goes in the test-engine site, linked from
   here and from the handoff (§13.1).
4. **Implementation starts with phase 3.0**, spike S3.
