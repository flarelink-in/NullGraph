# NullGraph Agents — Phase 2 step 12: neighbour lists that feed no observation — Technical Design

**Parent:** [`work/plan-2.md`](./work/plan-2.md) §3 (the library rule) and §5 (milestone 2A spike S8;
milestone 2B step 12); [`agents-gpu-loop-design.md`](./agents-gpu-loop-design.md) §6.1 (the shared
`neighbours` buffer) and §8.2 (app step stages).
**Status:** decisions accepted 2026-10-17 (§13); corrected S8 validation resumed 2026-10-18.
The user authorized proceeding with step 12 on 2026-10-18. Phase 12.1 is headless-validated
at **21,468 B** min+gzip, below the unchanged **21,495 B** ceiling. A36 and the API guide follow.
**Builds on:**
- `defineAgents`, `Sensors.nearest` and the CPU mirrors ([`agents-design.md`](./agents-design.md) §6–§7);
- `AgentSystem`'s shared `PointGrid`, per-sensor query stages, `addStepStage` and `inspect`
  ([`agents-gpu-loop-design.md`](./agents-gpu-loop-design.md) §6, §8.2, §9.6);
- the deterministic cell-windowed grid query ([`agents-ordered-grid-design.md`](./agents-ordered-grid-design.md)).

**Target:** `src/extras/agents/` (`agentLayout.ts`, `agentMath.ts`, `agentSenseShaders.ts`,
`agentThinkShaders.ts`, `agentInspect.ts`, `AgentSystem.ts`), headless test U45, GPU test A36, and
`docs/api/agents.md`. Core unchanged.

---

## 1. Purpose

Allow an `AgentDefinition` to declare a bounded nearest-neighbour query that populates a slice of
`AgentSystem.neighbours`, exposes its offset constant `NEIGHBOURS_<NAME>` to `addStepStage` (and
`Sensors.custom`), and reports its indices in `inspect(i).neighbours` — **without** writing any
floats into `obs`, changing `obsDim`, or altering `obsNames`.

Who uses it (plan-2 §3's second consumers):
- **contacts and crowd separation in step stages:** crowds in the venue planner (I13), horde games
  (G1, G2), and Herd & Wolves arenas when a brain's input vector is fixed (such as a trained
  `.mlp.json` or `.onnx` model) or observes fewer neighbours than physical separation needs;
- **game rules in `addStepStage`:** catches, melee hits, infection or aura transfer, resource pickup,
  and turret targeting that act on nearby agents of a species or within a radius the brain itself
  does not observe;
- **custom sensors (`Sensors.custom`) over neighbours:** a custom sensor that summarizes a neighbour
  list into one or two features (local crowd density, centre of mass, or line-of-sight) instead of
  5 floats per neighbour.

### Pre-existing facts this design rests on

1. **Every neighbour list today comes from a declared `Sensors.nearest`, and every `Sensors.nearest`
   widens `obs`.** In `defineAgents` (`agentLayout.ts:302-312`), a `nearest` sensor of size `k`
   always appends `k` slots (`5 * k` floats: `side, forward, velSide, velForward, present`) to
   `obsSlots` and `obsNames`, increments `obsDim` by `5 * k`, and emits a projection loop into the
   `sense` shader (`agentSenseShaders.ts:26-43`).
2. **Widening `obs` breaks trained brains and wastes bandwidth.** `assertBrainDims`
   (`agentLayout.ts:393`) refuses any `MlpBrain` or `OrtBrain` whose input width differs from
   `def.obsDim`, and `networkFromModel` (`tools/agents/mlp.mjs`) checks `def.obsNames` on load.
   Adding an 8-neighbour list for step stages would widen `obs` by 40 floats (`160 B` per agent),
   add `40 × H₁` multiply-adds per agent to every think pass, and invalidate existing checkpoints.
3. **`addStepStage` already binds `neighbours` at `@group(0) @binding(0)`, whereas binding
   `PointGrid` directly in `addStepStage` hits the 8-buffer default device limit.** An app step
   stage binds 5 system storage buffers in group 0 (`neighbours`, `obs`, `act`, `state`, `agents`;
   `AgentSystem.ts:208-213`, `agentThinkShaders.ts:107-112`), leaving `8 - 5 = 3` storage buffers in
   group 1 on a default WebGPU device. Reading `neighbours` in a Jacobi contact stage needs only
   **1** caller buffer (`deltas`), whereas binding `grid.cellCounts`, `grid.cellOffsets`,
   `grid.cellBodies` plus `deltas` needs **4** and fails `checkExtraBuffers` on a default device.
4. **Queries run every step; `sense` runs on `thinkEvery`.** `AgentSystem` runs all `query` stages
   before `sense` on every fixed step without gating (`AgentSystem.ts:124-130, 191-195`), because
   `addStepStage` runs every step. A query-only neighbour list is therefore fresh on every step at
   any `thinkEvery`.
5. **Spike S7 proved that body overlap changes what evolves.** In S7 (`agents-coevolution-design.md`
   §16), without contacts, herds stack onto one spot (mean nearest-prey distance `0.66–0.72 m`,
   below the `2 × radius = 0.80 m` body diameter, against `1.28 m` under the rule), because
   overlapping lets many prey shelter inside one wolf's 5 s catch rest timer.

### Non-goals

- **The standalone contact solver over `PointGrid`:** that is **step 13** (`null-graph/physics`,
  `docs/physics/physics-contacts-design.md`), which builds on `PointGrid` directly as its own compute
  stages (not inside `addStepStage`'s 5-buffer group 0) after S8 measures both.
- **Changing `Sensors.nearest`'s observed features:** observed `nearest` sensors still emit the same
  5 floats per slot in the agent's local frame, clamped to `[−1, 1]`.
- **More than `MAX_NEAREST_K = 8` neighbours per single query:** the register-sorted insertion loop
  in `buildAgentGridQueryWGSL` stays bounded at `k ≤ 8`; multiple queries can be declared if needed.
- **CPU readback of query results in the frame loop:** `inspect(i)` stays the only off-frame reader.

### Design principles applied

- **The library rule** (plan-2 §3): no scenario or research words in the API, docs, or WGSL.
- **One declaration, one source of truth:** `defineAgents` derives `obsDim`, `neighbourStride`,
  `NEIGHBOURS_<NAME>`, the query stages, the step-stage header, and the inspector layout from the
  same list.
- **Measured before building:** spike S8 (phase 12.0, §14) measures whether Herd & Wolves contacts
  need a query-only neighbour list or whether step 13's direct `PointGrid` solver suffices, and
  measures the cost of both across all three host/GPU pairs.

---

## 2. The formal definition

**Symbols.**
- A definition declares sensors $s_0, \dots, s_{M-1}$.
- Each `nearest` declaration $s_m$ has:
  - `name`: identifier, unique across all declarations;
  - `species`: species filter (a subset of declared species, or `null` for all species), resolved to
    a bitmask $\text{mask}_m \in [1, 2^S - 1]$;
  - $k_m \in \{1, \dots, 8\}$: number of neighbour indices stored per agent;
  - $r_m \in (0, \text{cellSize}]$: query distance cutoff (default $\text{cellSize}$);
  - $\text{observe}_m \in \{\text{true}, \text{false}\}$: whether $s_m$ writes to `obs` (default
    $\text{true}$).
- **Neighbour buffer layout:**
  $$\text{neighbourStride} = \sum_{m:\,\text{nearest}} k_m, \qquad \text{base}_m = \sum_{\substack{j < m \\ j:\,\text{nearest}}} k_j.$$
  For agent $i$, query $m$ writes `neighbours`$\big[i \cdot \text{neighbourStride} + \text{base}_m + 0 \dots k_m - 1\big]$
  with the up to $k_m$ nearest live agents $j \ne i$ matching $\text{mask}_m$ within XZ distance
  $r_m$, ordered by ascending distance and then ascending index $j$, padded with
  $\text{NO\_AGENT} = \text{0xffffffff}$.
- **Observation row layout:**
  $$\text{obsDim} = \sum_{m:\,\text{observed}} \text{width}(s_m) \ge 1,$$
  where $\text{width}(s_m) = 5 k_m$ when $s_m$ is a `nearest` sensor with $\text{observe}_m = \text{true}$,
  and $0$ when $\text{observe}_m = \text{false}$.

**Properties, each one a test (§8):**

| | Property | Means |
|---|---|---|
| **Q1** | Zero observation footprint | Adding a `nearest` declaration with `observe: false` leaves `obsDim`, `obsSlots`, `obsNames`, `OBS_*` constants, and every agent's `obs` row byte-for-byte unchanged; a brain compiled for the original `obsDim` attaches and produces bit-identical actions |
| **Q2** | Exact neighbour slice | For every agent $i$ and every `nearest` declaration (observed or not), its slice in `neighbours` equals `gridNearestNeighbours(def, rec, grid, i, mask, radius, k, out, tick)` on every step |
| **Q3** | Step-stage, custom-sensor & inspector access | `NEIGHBOURS_PER_AGENT` and `NEIGHBOURS_<NAME>` are in scope in `addStepStage` and `Sensors.custom`; `inspect(i).neighbours` decodes the slice under `{ sensor: name, indices }` |
| **Q4** | Query radius cutoff | A `nearest` declaration with `radius < cellSize` excludes peers with $\lVert \Delta p_{xz} \rVert > \text{radius}$ in the query stage (and in `senseAgent` when observed) |

---

## 3. Integration facts (verified against this repository, 2026-10-17)

| # | Fact | Where | Consequence |
|---|---|---|---|
| 1 | `Sensors.nearest` returns `{ kind: 'nearest', name, species, k }`. | `src/extras/agents/agentLayout.ts:82, 100-103` | Adding optional `radius?: number` and `observe?: boolean` (plus `Sensors.query` as a shorthand) is backwards-compatible |
| 2 | `defineAgents` builds `nearestMasks` in `nearest` sensor order, and `AgentSystem` consumes `def.nearestMasks[sensorIndex++]` in the same loop. | `agentLayout.ts:302-311`, `AgentSystem.ts:124-130` | Keeping query-only items as `kind: 'nearest'` with `observe: false` preserves `nearestMasks` indexing in one loop |
| 3 | `AgentSystem` computes `neighbourStride` by summing `s.k` over `s.kind === 'nearest'`, and `agentSnapshotLayout` does the same. | `AgentSystem.ts:100`, `agentInspect.ts:67` | Both automatically include `observe: false` slices with no change to their reduction |
| 4 | `AgentSystem` passes `def.cellSize` as the radius to `buildAgentGridQueryWGSL(def, mask, radius, k, { stride, base })`. | `AgentSystem.ts:126-127`, `agentGridShaders.ts:9-19` | `buildAgentGridQueryWGSL` already accepts any `radius` in `[0, def.cellSize]` and bakes `${radius * radius}` at line 56; `AgentSystem` only needs to pass `sensor.radius` |
| 5 | `buildAgentSenseWGSL` walks `def.sensors`; for `nearest` it emits a loop reading `neighbours[i * stride + base + rank]` and advances `offset += sensor.k * 5` and `base += sensor.k`. | `agentSenseShaders.ts:18-43` | When `sensor.observe === false`, advancing `base += sensor.k` while skipping the `obs` loop and leaving `offset` untouched keeps every subsequent observed `nearest` sensor's `base` and `offset` exact |
| 6 | `senseAgent` (CPU mirror) walks `def.sensors`, increments `nearest` on each `nearest` sensor, and calls `nearestNeighbours(def, rec, i, def.nearestMasks[nearest++], r, s.k, _nearIdx)`. | `agentMath.ts:100-124` | When `s.observe === false`, `senseAgent` increments `nearest++` and skips the query and `out` writes; when `s.observe === true`, it queries with `Math.min(r, s.radius)` |
| 7 | `buildAgentStepWGSL` emits `const NO_AGENT = 0xffffffffu;`, `const NEIGHBOURS_PER_AGENT = ${stride}u;`, and `const NEIGHBOURS_${sensor.name.toUpperCase()} = ${stride}u;`. | `agentThinkShaders.ts:93-106` | `buildAgentSenseWGSL` already binds `neighbours` at `@group(0) @binding(1)` (`agentSenseShaders.ts:75`); emitting the same constants into `buildAgentSenseWGSL` gives `Sensors.custom` the same neighbour-list contract as `addStepStage` |
| 8 | `decodeAgentSnapshot` decodes every `s.kind === 'nearest'` slice into `snap.neighbours`. | `agentInspect.ts:84-92` | `inspect(i)` automatically reports query-only slices under their declared `name` |
| 9 | The whole `agents.js` entry is `21,362 B` min+gzip after step 11 (`133 B` under C2's `21,495 B` line); the rule path is `14,768 B`. | `docs/work/now.md:78` | Step 12 touches shared layout/shader code; its bundle delta must aim to fit inside the `133 B` remaining under `21,495 B` (§7, §13 Q4) |

---

## 4. Data model

No GPU buffer layout changes:
- **`AgentSystem.neighbours`:** still `capacity × neighbourStride × 4 B` (`u32` indices, `0xffffffff`
  for empty slots), where `neighbourStride` is the sum of `k` across all `nearest` declarations
  (`observe: true` and `observe: false` alike).
- **`AgentSystem.obs`:** still `ceil(capacity × obsDim / 4) × 16 B`, where `obsDim` sums only
  observed sensors (`observe !== false`).
- **`SensorSpec` (`kind: 'nearest'`):**
  ```ts
  {
      readonly kind: 'nearest';
      readonly name: string;
      readonly species: readonly string[] | null;
      readonly k: number;
      /** Query cutoff distance, m, in (0, cellSize]. Default: cellSize. */
      readonly radius: number;
      /** Whether this sensor writes 5·k floats into obs. Default: true. */
      readonly observe: boolean;
  }
  ```

---

## 5. API

### Recommended (Option A): `observe: false` and `radius` on `Sensors.nearest`, plus `Sensors.query`

```ts
import { defineAgents, Sensors } from 'null-graph/agents';

const def = defineAgents({
    capacity: 16_384,
    world: { min: [-180, -180], max: [180, 180] },
    cellSize: 8,
    species: [
        { name: 'prey', maxSpeed: 6, maxTurn: 5, accel: 15, senseRadius: 8, radius: 0.4 },
        { name: 'wolf', maxSpeed: 6.8, maxTurn: 4.2, accel: 13, senseRadius: 8, radius: 0.6 },
    ],
    sensors: [
        Sensors.self(),
        Sensors.nearest({ name: 'wolf', species: 'wolf', k: 1 }),
        Sensors.nearest({ name: 'prey', species: 'prey', k: 3 }),
        Sensors.bounds(),
        // Populates neighbours[i * 12 + NEIGHBOURS_CONTACTS + 0..7] within 1.6 m; adds 0 floats to obs (obsDim stays 24).
        Sensors.query({ name: 'contacts', k: 8, radius: 1.6 }),
        // Equivalent form:
        // Sensors.nearest({ name: 'contacts', k: 8, radius: 1.6, observe: false }),
    ],
});
```

- **`Sensors.nearest(o)`** gains two optional fields:
  - **`radius?: number`** — maximum XZ distance in metres searched by the grid query stage, in
    `(0, cellSize]`. Defaults to `cellSize` (checked in `defineAgents`). When `observe` is `true`
    (the default), the sense stage still normalizes by `sp.senseRadius` and filters by
    `min(sp.senseRadius, radius)`.
  - **`observe?: boolean`** — defaults to `true`. When `false`, the query stage runs every step and
    fills `k` slots in `AgentSystem.neighbours`, defines `NEIGHBOURS_<NAME>` in step stages and
    custom sensors, and appears in `inspect(i).neighbours`, but writes nothing to `obs` and adds `0`
    to `obsDim`.
- **`Sensors.query(o)`** is a 1-line convenience helper returning
  `Sensors.nearest({ ...o, observe: false })`.
- **At least one observed sensor is still required:** if `obsDim === 0` (every declared sensor has
  `observe: false`), `defineAgents` throws a named refusal (§9).

### Alternative (Option B): a separate `queries` array on `AgentOptions`

```ts
const def = defineAgents({
    /* ... */
    sensors: [Sensors.self(), Sensors.nearest({ name: 'wolf', species: 'wolf', k: 1 }), /* ... */],
    queries: [Queries.nearest({ name: 'contacts', k: 8, radius: 1.6 })],
});
```

- **Why Option A is recommended over Option B:** every neighbour list already lives in one buffer
  (`AgentSystem.neighbours`), is built by the same `buildAgentGridQueryWGSL` generator, is indexed
  by `NEIGHBOURS_<NAME>`, and is decoded into `inspect(i).neighbours`. Keeping it in `Sensors`
  avoids a second declaration list and a second namespace while sharing all validation and layout
  code in ~120 B of bundle size.

---

## 6. Custom sensors reading neighbour lists

In phase 5.8 (`docs/work/log/2026-10.md`, 2026-10-13), the user asked whether a `Sensors.custom`
can read the neighbour lists (`neighbours` and `agents` are already bound at `@group(0) @binding(1)`
and `@binding(0)` in `buildAgentSenseWGSL`).

Step 12 completes that contract:
1. **In `buildAgentSenseWGSL` (`agentSenseShaders.ts`):** emit the same three constants that
   `buildAgentStepWGSL` emits:
   ```wgsl
   const NO_AGENT = 0xffffffffu;
   const NEIGHBOURS_PER_AGENT = <stride>u;
   const NEIGHBOURS_<NAME> = <base>u;
   ```
   A custom sensor's WGSL can then iterate `for (var r = 0u; r < k; r++)` over
   `neighbours[i * NEIGHBOURS_PER_AGENT + NEIGHBOURS_<NAME> + r]` and read `agents[j]`, including
   slices from `Sensors.query(...)` declared in the same `sensors` array.
2. **In `CustomSensorMirror` (`agentLayout.ts`, `agentMath.ts`):** pass `def` (or let the mirror
   call `nearestNeighbours(def, { data: records, u32 }, i, ...)` as today) so CPU tests and dataset
   recorders can mirror a neighbour-reading custom sensor without duplicating the search.

---

## 7. Bundle

`agents.js` is `21,362 B` min+gzip after step 11 (`133 B` under C2's `21,495 B` line); the rule
path is `14,768 B`. Option A adds ~80–130 B to both paths (`observe` and `radius` checks in
`defineAgents`, `Sensors.query`, the `if (sensor.observe)` guard in `buildAgentSenseWGSL` and
`senseAgent`, and the `NEIGHBOURS_*` constants in `buildAgentSenseWGSL`). Per §13 Q4, the ceiling
stays at `21,495 B` and the implementation must try to fit inside the remaining `133 B` first; only
if it cannot fit is the exact byte count brought back to the user.

---

## 8. Test suite

IDs continue from step 11 (`U43`, `U44`, `A35`).

**Headless (U45, `test/agents.test.mjs` and `test/agents-system.test.mjs`):**
- **Q1 (Zero observation footprint):** `defineAgents` with `Sensors.query({ name: 'contacts', k: 8, radius: 1.6 })`
  added to Herd & Wolves keeps `obsDim === 24`, `obsNames.length === 24`, and identical `OBS_*`
  constants; `senseAgent` and `senseAll` produce byte-for-byte identical `obs` rows; `assertBrainDims(def, 24, 2)`
  succeeds;
- **Q2 & Q3 (Layout, shaders, inspector):** `neighbourStride` increases from `4` to `12`; two
  `AgentSystem query` stages and one `AgentSystem query contacts` stage are created with non-overlapping
  `base` offsets (`0`, `1`, `4`) and radius `1.6² = 2.56` baked into the third; both
  `buildAgentStepWGSL` and `buildAgentSenseWGSL` emit `const NEIGHBOURS_PER_AGENT = 12u;` and
  `const NEIGHBOURS_CONTACTS = 4u;` and parse with `WgslReflect`; `decodeAgentSnapshot` returns
  `{ sensor: 'contacts', indices }` as the third entry of `snap.neighbours`;
- **Q4 (Radius cutoff & refusals):** every refusal in §9 fires and names the offender (`radius <= 0`,
  `radius > cellSize`, all sensors `observe: false`, duplicate names between `nearest` and `query`).

**GPU (A36, `test/agents-gpu/run.js`, on Node/NVIDIA, Node/AMD, and Chrome/AMD):**
- **Q1 + Q2 + Q3 + Q4 on the GPU:** a population with an observed `nearest` sensor (`k = 2`), a
  `Sensors.query` (`k = 6, radius = 2.5`), a `Sensors.custom` reading `NEIGHBOURS_...`, and an
  `addStepStage` reading `NEIGHBOURS_...` runs at `thinkEvery` 1 and 3:
  - every agent's `neighbours` row matches `gridNearestNeighbours` for both slices on every step,
    and peers at distance `2.5 < d <= cellSize` are excluded from the `radius = 2.5` slice;
  - `obs` rows match `senseAgent`, and an `MlpBrain` compiled for the observed `obsDim` runs without
    change;
  - `inspect(i)` returns both neighbour slices accurately;
- **Watched failing:** removing the `if (sensor.observe)` guard in `defineAgents` fails `assertBrainDims`
  and `obsDim`; passing `def.cellSize` instead of `sensor.radius` in `AgentSystem` fails the radius
  cutoff check in A36.

---

## 9. Errors (verbatim, confirmed when built)

| Where | Condition | Message |
|---|---|---|
| `defineAgents` | `s.radius` not finite, `≤ 0`, or `> cellSize` | `AgentSystem: sensor "<name>" radius <r> must be in (0, cellSize <cellSize>].` |
| `defineAgents` | Every sensor has `observe: false` (`obsDim === 0`) | `AgentSystem: declare at least one observed sensor — a brain needs something to read.` |
| `defineAgents` | Duplicate name between any two sensors/queries | `AgentSystem: two sensors are both named "<name>"; pass a distinct name.` (today's message) |
| `defineAgents` | `k` outside `1..8` on `Sensors.query` | `AgentSystem: sensor "<name>" k <k> must be 1–8.` (today's message) |

---

## 10. Engine touchpoints

- **Core:** none.
- **Agents (`src/extras/agents/`):**
  - `agentLayout.ts`: `SensorSpec` (`nearest` gains `radius: number` and `observe: boolean`),
    `Sensors.nearest` options, `Sensors.query` helper, `defineAgents` validation and `obsDim === 0`
    refusal;
  - `agentMath.ts`: `senseAgent` skips `observe === false` while advancing `nearest++`, and uses
    `Math.min(r, s.radius)` when `observe === true`;
  - `agentSenseShaders.ts`: `buildAgentSenseWGSL` skips the `obs` write block when `!sensor.observe`
    (still advancing `base += sensor.k`) and emits `NO_AGENT`, `NEIGHBOURS_PER_AGENT`, and
    `NEIGHBOURS_<NAME>` constants;
  - `AgentSystem.ts`: passes `sensor.radius` (instead of `def.cellSize`) to `buildAgentGridQueryWGSL`.

---

## 11. Build order

| Phase | Delivers | Exit |
|---|---|---|
| **12.0** ✅ `f777bc2` | Spike S8 (`npm run spike:s8`, `test/spikes/contacts/`): contacts over today's lists, over a query-only list, and over `PointGrid` (§14) | Complete 2026-10-18 on Node/NVIDIA, Node/AMD and Chrome/AMD; exact reruns, exhaustive dense-count oracle and 600-step histories pass, zero GPU errors; corrected findings and solver limits below |
| **12.1** ✅ | `Sensors.query` / `Sensors.nearest({ observe, radius })`, custom-sensor neighbour constants, CPU mirror, U45 headless, bundle check | Complete 2026-10-18: U45 green and four mutations caught; `test:agents` 94/94 plus 24 mirror checks; whole entry 21,468 B min+gzip, 27 B under the unchanged ceiling |
| **12.2** | A36 in the agents GPU suite on Node/NVIDIA, Node/AMD, and Chrome/AMD | Green on all three pairs, watched failing, zero WebGPU errors |
| **12.3** | API guide (`docs/api/agents.md`) and checked example in `test/agents-api-example.test.mjs` | Example runs and is watched failing |

Add ✅ to each subsequent phase only when its own exit is verified; no check mark means pending.

### Phase 12.1 implementation notes — 2026-10-18

- `radius` and `observe` remain optional on `SensorSpec` to preserve existing hand-written declarations;
  omission means `cellSize` and `true`. Definitions do not mutate the caller's declarations.
- Neighbour offsets, stride and `NO_AGENT` are derived once into `def.constants`, so sense, step and
  think wrappers share them. `OBS_*`, observation names and slots remain unchanged. U44 strips only
  these newly added constants before comparing the original four shader fingerprints.
- Custom CPU mirrors receive the definition as their sixth argument; existing five-argument mirrors
  remain valid. Radius filtering changes the query cutoff, not the observation normalization.
- U45 covers query slices before, between and after observed lists, CPU bytes, shader loops and
  offsets, inspector decoding, invalid radii/k, duplicate and case-folded generated names, custom
  neighbour-reading WGSL, and mirror access to the definition. Removing each of the layout/sense/CPU
  observation guards or replacing the query radius with cellSize was observed failing.
- Size: whole agents entry **21,468 B** (+106 B); rule path **14,856 B** (+88 B). No ceiling increase.

### Fresh review before the phase 12.1 commit — 2026-10-18

- No production defect found in the declaration, shared constants, query dispatch, observation
  projection or snapshot paths. Query-only lists must advance both the neighbour base and the
  species-mask index; observed radius limits search, not senseRadius normalization.
- Strengthened U45: the custom CPU mirror now actually searches using its definition argument;
  a separate two-species regression checks the mask after a hidden list, inclusive radius,
  exclusion beyond radius, and unchanged normalization (distance 1 / senseRadius 4 = 0.25).
- Development, production and IDE builds pass. Full headless rerun: **95/95 plus 24 mirror
  checks**. U45 still enforces the unchanged bundle ceiling. A36 remains pending; headless
  shader parsing alone is not evidence that custom neighbour consumers execute correctly.

---

## 12. Where this refines `plan-2.md`

| `plan-2.md` | This design |
|---|---|
| §5, step 12: "Within step 13's design, or its own" | In its own design (`docs/agents-queries-design.md`), because step 12 is an `agents` declaration change (`src/extras/agents/`), whereas step 13 is a new `null-graph/physics` entry (`docs/physics/physics-contacts-design.md`) |
| §5, step 12: "neighbour lists that feed no observation (a query-only declaration), readable by step stages as today's lists are" | Also adds optional `radius <= cellSize` per query and emits `NEIGHBOURS_*` into `buildAgentSenseWGSL` so `Sensors.custom` can read neighbour lists under the same contract as `addStepStage` |
| §5, S8: "two step stages over the lists the definition already declares (1 nearest wolf, 3 nearest prey)" | S8 measures three contact sources side by side — (a) today's `1 wolf + 3 prey` lists, (b) an unobserved `k = 8` any-species query list (step 12), and (c) a direct 3×3 `PointGrid` scan (step 13) — so S8 decides both step 12 and step 13 in one run |

---

## 13. Questions for the user

**Decided 2026-10-17.** The user: "ok thats cool pick all recommended and dont raise the cieling
now ,try fitting if failed only then increase like but dont depend on the high cieling until
absolutely needs it".
- **Q1:** (A), run S8 first (phase 12.0), then decide whether to build phases 12.1–12.3.
- **Q2:** (A), `Sensors.query({ name, species, k, radius })` and `Sensors.nearest({ ..., radius, observe: false })`.
- **Q3:** Yes, emit `NEIGHBOURS_PER_AGENT`, `NEIGHBOURS_<NAME>`, and `NO_AGENT` into `buildAgentSenseWGSL` as well as `buildAgentStepWGSL`.
- **Q4:** Keep the `21,495 B` ceiling (`133 B` remaining) and fit inside it if possible; only ask to raise it if the build cannot fit.

The questions as they were asked:

1. **Q1. Should we build Step 12 even if S8 shows Step 13 (`PointGrid` contacts) handles Herd &
   Wolves contacts on its own?**
   - **(A) Run S8 first, then decide** (recommended, following `plan-2.md` §5 and §8): S8 (phase
     12.0) measures (a) today's lists, (b) Step 12's query-only list, and (c) Step 13's `PointGrid`
     scan first; we then build Step 12 if S8 shows the need or if you want it for step-stage /
     custom-sensor consumers.
   - **(B) Build Step 12 unconditionally after S8:** it is ~120 B and also completes the
     custom-sensor neighbour contract from phase 5.8.
2. **Q2. The API (§5):**
   - **(A) `Sensors.query({ name, species, k, radius })` and `Sensors.nearest({ ..., radius, observe: false })`**
     (recommended): reuses the existing `nearest` pipeline and inspector decoding with one boolean
     and optional `radius <= cellSize`.
   - **(B) A separate `queries: [Queries.nearest(...)]` option on `defineAgents`.**
3. **Q3. Expose `NEIGHBOURS_PER_AGENT`, `NEIGHBOURS_<NAME>`, and `NO_AGENT` in `Sensors.custom` as
   well as `addStepStage` (§6)?**
   - **Recommended: yes.** `neighbours` and `agents` are already bound in the sense shader; emitting
     those three constants makes custom sensors over neighbour lists work out of the box.
4. **Q4. Bundle ceiling (§7):** `agents.js` is `21,362 B` (`133 B` under C2's `21,495 B` line).
   Recommended: keep `21,495 B` if Step 12 fits within the remaining `133 B`; if it lands ~20–60 B
   above, raise the Phase 2 `agents.js` line to `21,600 B` and report the exact byte count before
   committing.

---

## 14. Phase 12.0 — spike S8: contacts, cost, and whether today's neighbour lists suffice

**Scripts only** (`test/spikes/contacts/{run.js,index.html}`, `npm run spike:s8`), run on
Node/NVIDIA, Node/AMD, and Chrome/AMD. No library code changed.

### What S8 builds in spike code

An XPBD/Jacobi sphere-separation pass on the XZ plane (`posRadius.xz`, radius `r_i = posRadius.w`,
equal mass per unit area or equal weight per agent, compliance $\alpha \ge 0$):
- **Stage 1 (`accumulate`):** for each live agent $i$, iterates its candidate peers $j$, computes
  penetration $\delta = (r_i + r_j) - \lVert p_i - p_j \rVert$ for $0 < \lVert p_i - p_j \rVert < r_i + r_j$
  (with a deterministic tie-break normal from $(i, j)$ when $p_i = p_j$), accumulates the position
  correction into `deltas[i].xy` and increments `deltas[i].z += 1.0`.
- **Stage 2 (`apply`):** if `deltas[i].z > 0.0`, adds `deltas[i].xy / deltas[i].z` (clamped to
  `maxPush` per substep and kept inside the world/arena walls inset by $r_i$), and damps the
  inward velocity component when moving into the mean contact normal so agents do not oscillate
  against a crowd.

Three candidate peer sources compared on the **same** Herd & Wolves simulation (C2's clones at
4,096 / 16,384 / 65,536 agents, plus S7's 64-arena layout of 17,408 agents):
1. **`(a) today-lists`:** `addStepStage` reading Herd & Wolves' existing `neighbours` (`1` wolf +
   `3` prey = `4` slots per agent, no query stage added);
2. **`(b) query-k8`:** `addStepStage` reading a prototype `k = 8` any-species query slice within
   contact radius `1.6 m` (`2 × maxRadius = 1.2 m` plus margin), while `obs` stays 24 floats;
3. **`(c) grid-3x3`:** standalone stages reading `agents.grid` (`cellCounts`, `cellOffsets`,
   `cellBodies`) directly over the 3×3 cell neighbourhood (step 13's prototype), tested at 1 and 2
   Jacobi iterations per step.

### What S8 measures

| # | Question | Metric |
|---|---|---|
| 1 | **Step cost** | GPU ms per step (and % added to `none`) at 4,096, 16,384, and 65,536 agents on Node/NVIDIA, Node/AMD, and Chrome/AMD |
| 2 | **Worst overlap & stacking** | Over 600 steps: worst penetration $\max_{i \ne j} \max(0, r_i + r_j - d_{ij})$ (in m and % of $r_i + r_j$), % of agents with any overlap $> 5\%$, mean nearest-prey distance (against S7's `0.66–0.72 m` stacking), and peak cell count |
| 3 | **Pile-up stress test** | 256 prey spawned inside a `4 m × 4 m` box (guaranteed >6-way overlaps): how many steps `(a)`, `(b)`, and `(c)` take to separate below 5% overlap, and whether `(a)`'s `k = 3` prey limit leaves permanent overlaps in the interior |
| 4 | **Jitter** | Mean high-frequency position reversal ($\cos \theta < -0.5$ between consecutive step displacements) in dense herds, with and without normal-velocity damping |
| 5 | **Exact reruns** | Two 300-step runs from the same seed compared byte for byte across all `capacity × 16` record words (`0` differing words required on every pair) |

**Measurement clarification (2026-10-18):** cost uses queue-completed wall-clock time, including
host submission overhead, not isolated GPU timestamp milliseconds. The sequential single samples
show timing noise and mode-specific trajectories; negative deltas are not proof of a speedup.
Overlap now records every step 1–600, with full histories and run maxima separate from the final
state. Peak cell counts are reconstructed from final positions, after contacts and respawns.
The exhaustive grid path rebins after motion and before every iteration and does not apply the
sensor scan budget. A dense GPU brute-force oracle (maximum 42 contacts) catches reinstating the
64-entry limit: **147 agents' contact counts differ on each Node adapter**.

**Solver limitation:** all sources use the same contact-count average, clamped push and normalized
damping normal. This spike uses zero compliance and a positional Jacobi correction, not a full
XPBD accumulated-multiplier implementation. None of the measured 1/2-iteration configurations
clears the 256-agent pile-up below 5% within 180 steps; production convergence and compliance
remain step 13 design questions. Do not claim the query-only list solves dense contacts.

### What S8 decides

- **Step 12:** whether `(a)`'s `1 wolf + 3 prey` lists leave unresolved overlaps in pile-ups or wolf
  packs that `(b)`'s `k = 8` query-only list fixes, or whether `(c)` (`PointGrid` direct scan) is
  both faster and more complete than `(b)`.
- **Step 13 (`docs/physics/physics-contacts-design.md`):** iteration count (1 vs 2), compliance
  $\alpha$, normal-velocity damping, and how `null-graph/physics` attaches to an `AgentSystem` (or
  any `PointGrid` + position/radius buffer) so its stages run inside every fixed step (`step()` and
  `getComputeStages()`).

### Corrected S8 findings — 2026-10-18

The initial final-state-only, capped-grid measurements are superseded. The corrected Node runs
visit every step and every direct-grid candidate; the dense count oracle is exact on both adapters,
and replacing the exhaustive loop with 64 entries is watched failing on both. All four contact
variants repeat byte for byte over two 300-step runs (zero differing record words).
Compact numeric evidence for all three pairs is checked in at `test/spikes/contacts/results.json`.
Node checks are 12/12. Chrome has 16 distinct checks, all passing (the original reporting loop
repeated overlap assertions, displaying 56/56; corrected without changing any measurement).
The final Node overlap run likewise displayed 100/100 for 20 distinct assertions. Raw histories
and error-watched reports remain in the ignored `test/spikes/contacts/out/` directory.

| Layout / metric | none | today-lists | query-k8 | grid 1 iteration | grid 2 iterations |
|---|---:|---:|---:|---:|---:|
| 4,096, final worst penetration, NVIDIA (m) | 0.784 | 0.054 | 0.082 | 0.068 | 0.028 |
| 4,096, maximum penetration over 600 steps, NVIDIA (m) | 1.176 | 0.654 | 0.654 | 0.654 | 0.255 |
| 4,096, maximum agents >5% overlap over run (%) | 62.43 | 9.03 | 9.03 | 9.03 | 4.10 |
| 64 arenas, final worst penetration, NVIDIA (m) | 1.068 | 0.079 | 0.083 | 0.083 | 0.044 |
| 64 arenas, maximum penetration over run, NVIDIA (m) | 1.189 | 0.686 | 0.686 | 0.686 | 0.667 |
| Static 256-agent pile, agents >5% after 180 steps, NVIDIA (%) | 100 | 75.78 | 86.33 | 87.50 | 62.89 |

AMD's final worst penetration at 4,096 is 0.819 / 0.049 / 0.057 / 0.060 / 0.022 m respectively.
Every configuration leaves unresolved stress-pile overlaps; adding neighbours is not a convergence
guarantee. Damping reduces query-k8 displacement reversal from 11.92% to 6.23% on NVIDIA and
11.98% to 8.08% on AMD; exhaustive-grid variants go below the movement threshold (reported 0%),
which is not proof that the dense pile has separated.

**Decision:** proceed with phase 12.1 at the user's explicit 2026-10-18 request for the general
query-only step-stage/custom-sensor API. S8 does **not** establish that contacts require it or that
it outperforms the exhaustive grid. For step 13, use exhaustive candidates and rebin between
iterations; two iterations are a useful ordinary-herd starting point, not a dense-pile acceptance
result. Keep compliance/iteration convergence open for that design. Do not promote this scripts-only
prototype into a production physics solver.
