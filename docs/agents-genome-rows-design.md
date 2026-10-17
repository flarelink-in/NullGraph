# NullGraph Agents — Phase 2 step 11: a network per agent (`MlpBrain.byGenome`) — Technical Design

**Parent:** [`work/plan-2.md`](./work/plan-2.md) §5 (milestone 2B, step 11);
[`agents-design.md`](./agents-design.md) §4.1 (the record's `genome` word), §9.3 (the Phase 2 seam)
and §13 ("Deferred — Phase 2").
**Status:** decided 2026-10-17 (§14): the user took every recommendation, Q5 parked. Spike S6 came
first (phase 11.0). Phases 11.0–11.2 committed (`6042e41`, `2e3c8c2`, `2d51159`; results at the
end). Next: phase 11.3, A35 on three host/GPU pairs and a 16,384-agent step timed.
**Builds on:** per-species rows ([`agents-species-brains-design.md`](./agents-species-brains-design.md)
§5, `MlpBrain.bySpecies`) and `MlpBrain` ([`agents-mlp-design.md`](./agents-mlp-design.md)).
**Target:** `src/extras/agents/` (`MlpBrain`, `mlpShaders`; with the recommended kernel, a
cooperative entry in the think wrapper, §11), tests U43, U44 and A35, the API guide. Core unchanged.

---

## 1. Purpose

One `MlpBrain` holds N networks of one shape as the rows of one weight table, and each agent reads the
row its record's `genome` word names. It is the one library piece Phase 2's first question needs for
certain (plan-2 §3), said without that question's words: **a network per agent**.

Who uses it (plan-2 §3's second consumers):
- **evolution:** the candidates of a generation in their arenas (plan-2 §4); later an ecosystem whose
  births write rows (the backlog);
- **crowds with varied behaviour:** a few dozen personalities over thousands of agents;
- **A/B tests:** two policies in one population, under the same conditions.

### Pre-existing facts this design rests on

1. **The record reserves the word.** `genome`, a u32 at word 14, "Phase 2: weight-table row; 0 in
   Phase 1" ([`agents-design.md`](./agents-design.md) §4.1). `spawn()` takes it
   (`AgentInit.genome`), and `inspect()` reports it.
2. **The think wrapper hands a brain its agent's record.** The think stage binds `agents` at group 0,
   binding 0 (`agentThinkShaders.ts`), so a brain's WGSL reads `agents[i].ids.z` with no change to
   `AgentSystem`.
3. **Rows already exist.** `MlpBrain.bySpecies` packs same-shape networks into one table, row r at
   r × `mlpLayout(network).floats`, and `setWeights(network, { species })` uploads one row (phase
   7.1). A row per agent is another way of choosing the row, over the same table.
4. **The cost question.** Phase 7.2 measured a row chosen per lane at 4.6× a single network in Chrome,
   even with every agent reading the same row: "the rows penalty is the compiler's, not the data's".
   Spike S6 measured what a row per agent costs, in the kernel and in a whole step (results at the
   end). Reading the row in each lane costs 4–7% of a step on NVIDIA and up to 36% on the AMD iGPU.
   Copying a block's row into workgroup memory first, the kernel §6 recommends, is faster than
   today's single network on NVIDIA and in Chrome, and bit-identical.

### Non-goals

- **Training or evolving.** The brain runs whatever weights it holds. Evolution strategies live in
  `tools/agents/` (plan-2 step 14).
- **Rows written on the GPU** (mutation, births): the backlog's ecosystem.
- **Networks of different shapes in one brain.** One shape per brain; another shape is another brain,
  gated by species as today.
- **Rows chosen by anything but the genome word.** By species, `bySpecies` stays.
- **Rows for `OrtBrain`.** ONNX Runtime runs one model per session.

### Design principles applied

- **The library rule** (plan-2 §3): no evolution words in the API, the docs or the WGSL.
- **Buffers fixed for life:** the table's size is set at construction. New weights cost one
  `writeBuffer`, never a pipeline.
- **Determinism:** an agent's actions depend on its row and its observations only.
- **Measured, not assumed:** the kernel is chosen from S6's numbers on three host/GPU pairs, and the
  forms it does not build are filed with their measured cost (as P3 files graph colouring beside its
  Jacobi baseline).

---

## 2. The formal definition

**Symbols.**
- **σ**, a network shape: the layer widths and activations `MlpBrain` accepts (≤ 4 layers, ≤ 64 wide).
- **W** = (w₀ … w_{N−1}), N ≥ 1 networks of shape σ: the rows.
- For agent i: **gᵢ**, its record's `genome` word (u32); **oᵢ**, its observation row; **aᵢ**, the
  actions its think writes.
- **f(w, o)**, `MlpBrain`'s forward pass; `mlpForward` is its CPU mirror.

**The brain:** aᵢ = f(w_{gᵢ}, oᵢ) for gᵢ < N. For gᵢ ≥ N, see §14, Q1.

**Properties, each one a test (§9):**

| | Property | Means |
|---|---|---|
| P1 | Exact rows | Every agent's actions equal `mlpForward` of its own row, within A4's bound, with rows aligned to 64-agent blocks, in runs of 8, and shuffled |
| P2 | Row isolation | Writing row r changes the actions of exactly the agents whose genome is r |
| P3 | Equivalence | N identical rows of network w run the same simulation as `new MlpBrain(engine, w)`, word for word, except the genome words (S6, part 2) |
| P4 | Determinism | Two runs from the same spawn give the same records, word for word |
| P5 | No compile on new weights | Writing rows compiles nothing |
| P6 | Out-of-range genome | The behaviour chosen in Q1, and never a read outside the table |

---

## 3. Integration facts (verified against this repository, 2026-10-17)

| Fact | Where | Consequence |
|---|---|---|
| `genome` is word 14 of the record, `AGENT_GENOME_OFFSET`; `ids.z` in WGSL | `agentLayout.ts:28`; `AGENT_STRUCT_WGSL` | The row's index needs no new field |
| `AgentInit.genome` defaults to 0 | `agentRecords.ts:33, 107` | With no genome given, every agent reads row 0 |
| `inspect()` decodes `genome` | `agentInspect.ts:36, 108` | The inspector shows each agent's row as it is |
| The think module binds `agents` (read) at group 0, binding 0 | `agentThinkShaders.ts` | A brain's WGSL can read `agents[i].ids.z` |
| Rows are packed at r × `mlpLayout(network).floats`, a multiple of 4 | `MlpBrain.ts`, `mlpMath.ts:61` | A row's offset in vec4s is r × floats / 4 |
| A think pass serves the species its mask names | `AgentSystem.setBrain(brain, { species })` (phase 7.3) | Prey and wolves can each have a rows brain, or share one |
| The whole agents entry is 20,209 B min+gzip; `MlpBrain` adds 1,927 B to a rule-path app | `test/spikes/mlp-think-cost/size.mjs`, 2026-10-17 | C2's allowance (21,495 B) leaves 1,286 B (§8) |

---

## 4. Data model

- **The table:** N × floats f32 in one storage buffer, floats = `mlpLayout(σ).floats`. Row r starts
  at byte r × floats × 4.
  - C2's shape (24 → 64 → 64 → 64 → 2): 10,180 floats, **40,720 B a row**; 64 rows 2.6 MB, 256 rows
    10.4 MB.
  - Phase 7.2's test shape (24 → 32 → 32 → 2): 1,988 floats, 7,952 B a row.
  - The limit is min(`maxBufferSize`, `maxStorageBufferBindingSize`): 128 MiB on a default device,
    3,296 rows of C2's shape.
- **The record:** unchanged. `genome` names the row.

---

## 5. API

```ts
import { AgentSystem, MlpBrain, packMlpWeights } from 'null-graph/agents';

// N networks of one shape, as rows; each agent reads row `genome`.
const brain = MlpBrain.byGenome(engine, networks);       // networks.length = N ≥ 1
brain.rowCount;                                          // N
agents.setBrain(brain, { species: 'prey' });             // as any brain
agents.spawn(i, { position, species: 'prey', genome: 7 });

brain.setWeights(network, { row: 7 });                   // one row
brain.setRows(packed, { first: 0 });                     // rows packed back to back, one upload
```

- **`setRows`** takes rows packed back to back, each `mlpLayout(network).floats` long: what
  `packMlpWeights` writes. A trainer packs a generation's candidates into one array it keeps, and
  uploads them in one call.
- **`setWeights(network)` with no `row`** on a `byGenome` brain is refused, naming `{ row }`, as
  `bySpecies` names `{ species }`.
- **`byGenome` and `bySpecies` are separate constructors:** one chooses rows by species, one by genome.
- **One brain may serve several species** when they share a shape: their genomes then index one table
  (prey rows 0–63, wolves 64–127, for instance), in one pass.
- **Fast when each block of 64 agents shares one row** (agents 0–63, 64–127, …): the block's row is
  read once into workgroup memory (§6). Any other layout is correct, and costs up to about twice what
  reading the row per agent costs (S6). The guide says so; arenas are laid out that way (plan-2
  step 14).

---

## 6. The kernel

**The rule** (plan-2 §5): if a row read from the genome word in each lane adds at most 10% to a step
at the counts evolution uses, on every host/GPU pair, build that; otherwise the cheapest variant
that keeps one row per agent, or the numbers go to the user. **It failed** (S6, "What it shows",
item 1). The variants measured:

| | Variant | What the kernel does | What it asks |
|---|---|---|---|
| (a) | Genome per lane | `row = agents[i].ids.z`, then every weight read offset by the row | Nothing |
| (b) | Workgroup index | The row derived from `@builtin(workgroup_id)` | Each block one row, by its position; the genome word unused |
| (c) | Workgroup broadcast | Lane 0 reads its genome; `workgroupUniformLoad` shares it | Each block one row; a barrier before any early return |
| (d) | One pass per row | A dispatch per row over that row's agents, its offset in a uniform | Each row's agents in one range; N dispatches per think |
| (e) | Staged | All 64 lanes copy the block's row into workgroup memory, a tile at a time, then read it there | Each block one row; barriers |
| (e′) | Staged + per-lane | (e), and a lane whose genome names another row reads its own from storage, as (a) | Barriers |

**Recommended: (e′), staged with a per-lane fallback** (§14, Q4).
- **The cheapest row kernel** for rows aligned to blocks in five of the six GPU-and-shape cells S6
  measured. With C2's networks and 64 rows it runs **faster than today's single network** on NVIDIA
  (0.58–0.86×) and in Chrome (0.31–0.59×), and at 1.8–2.8× under Node on the AMD iGPU, against (a)'s
  2.7–3.5×.
- **Correct for any layout,** and bit-identical to (a) and to today's kernel in every cell measured
  (37 of 37 on each host). A block whose agents read different rows costs 0.6–2.1× what (a) costs.
- **Its costs:**
  - a think entry where every lane reaches the barriers (below);
  - more WGSL;
  - 2–3 s to compile under FXC (Node's tools and tests; 0.4–0.6 s in Chrome);
  - phase 7.2's small network in Chrome: 3.0–4.0×, where (b) and (c) cost 1.7–2.1× and (a)
    3.6–4.8×.
- **Estimated effect on a 16,384-agent step** with C2's networks, from part 1's numbers (part 2
  could time only (a)): about −9% on NVIDIA, +13% under Node on AMD, −44% in Chrome. Phase 11.3
  measures it.

**The alternative: (a).** No change outside `MlpBrain`, and the simplest code. It adds 4–7% to a step
on NVIDIA, 9–17% in Chrome and 33–36% under Node on the AMD iGPU (C2's networks, 4,096–16,384 agents,
thinkEvery 1). A later kernel could replace it behind the same API.

**How (e′) runs.**
- **The block's row** is its first agent's genome, shared with `workgroupUniformLoad` (clamped, or as
  Q1 decides).
- **A block where no agent thinks this step stages nothing.** With `thinkEvery`, whole blocks rest
  together (the cadence is per block; only newborns think out of turn): at `thinkEvery` 3, two blocks
  in three skip. One workgroup flag, read uniformly, decides.
- **Each layer** is copied in tiles of at most 512 vec4 (8 KB of workgroup memory, half the default
  limit) by all 64 lanes, coalesced, between two barriers. The lanes that think then compute from
  the tile, or from their own row in storage when it differs. The arithmetic is today's, in the same
  order, so the bits are today's.
- **The think entry** changes for brains that ask for it: instead of returning early, every lane
  computes whether it thinks and calls the brain's block function, which may use barriers. A rule
  brain and a single-network `MlpBrain` keep today's entry (§11).

---

## 7. Lifecycle and updates

- **Construction** validates every network's shape against row 0, packs all rows and uploads once.
- **`setWeights({ row })` and `setRows`** validate, pack and write with one `writeBuffer` each. The
  queue orders them with the steps: a write lands before the next step submitted after it.
- **The table's size is fixed.** More rows means a new brain; `prepareBrain` compiles it in the
  background.
- **Destroy** as today: detach or destroy its systems first.

---

## 8. Bundle

The agents entry stands at 20,209 B min+gzip; C2's allowance is 21,495 B, so 1,286 B remain.
Expected on the `MlpBrain` path: about 300–500 B with (a) (the constructor, `setRows`, the row guard,
their errors), and about 800–1,200 B with (e′) (the staged generator's WGSL text and the cooperative
entry as well). The size line is set from the build and reported (§14, Q3); rule-only apps must still
carry no MLP code (`size.mjs`).

---

## 9. Test suite

**Headless (U43, `test/agents-mlp.test.mjs`; with (e′), U44 in `test/agents-system.test.mjs`):**
- `byGenome` packs row r at r × floats; `rowCount` is N;
- `setWeights(network, { row })` writes exactly row r's bytes, and `setRows(packed, { first })`
  exactly its range (mock device);
- every refusal of §10 fires and names the offender;
- the generated module parses, reads `ids.z`, and carries the out-of-range guard (U9's checks);
- with (e′): a brain that asks for the cooperative entry gets one with no early return before its
  block function, and every other brain keeps today's entry, byte for byte.

**GPU (A35, agents suite; Node/Dawn on NVIDIA and AMD, and Chrome on AMD):**
- **P1:** 64 rows of distinct networks; every live agent matches `mlpForward` of its own row, rows
  aligned, in runs of 8 and shuffled (the fallback path), at `thinkEvery` 1 and 3 with newborns;
- **P2:** rewriting one row changes exactly that row's agents; the others are bit-exact;
- **P3:** Herd & Wolves for 300 steps on 64 identical rows equals today's brain, word for word, except
  the genome words;
- **P6:** an agent whose genome is out of range behaves as Q1 decides;
- **Watched failing:** with the row term removed (every agent on row 0), P1 fails on every agent not
  on row 0; with (e′), with the fallback removed, P1 fails on runs of 8 and shuffled rows.

---

## 10. Errors (verbatim, confirmed when built)

| Where | Message |
|---|---|
| `MlpBrain.byGenome` | `MlpBrain: byGenome needs at least one network.` |
| `MlpBrain.byGenome` | `MlpBrain: the network for row <r> differs in shape from row 0 at layer <k>; give it its own brain.` |
| `MlpBrain.byGenome` | `MlpBrain: weights exceed device buffer limits.` (today's message) |
| `setWeights` | `MlpBrain: this brain holds one network per row; call setWeights(network, { row }).` |
| `setWeights(…, { row })` | `MlpBrain: setWeights({ row }) needs a brain from MlpBrain.byGenome.` |
| `setWeights(…, { row })`, `setRows` | `MlpBrain: row <r> is outside this brain's <N> rows.` |
| `setRows` | `MlpBrain: setRows needs whole rows of <floats> floats; got <length>.` |
| `setRows` | `MlpBrain: packed weights[<k>] must be finite.` (as the constructor and `setWeights` refuse non-finite weights) |
| `setRows` | `MlpBrain: setRows needs a brain from MlpBrain.byGenome.` |

---

## 11. Engine touchpoints

- **Core — required: none.**
- **Agents — required:** `MlpBrain.byGenome`, `setWeights({ row })`, `setRows`, and the genome
  kernel in `mlpShaders.ts`.
- **Agents — required with (e′): a cooperative think entry** (built in phase 11.1). A stage brain
  declares `cooperative: true`, and its WGSL defines
  `fn thinkBlock(i: u32, species: u32, tick: u32, thinking: bool, lane: u32)` instead of `think`.
  `buildAgentThinkWGSL(def, wgsl, thinkEvery, { cooperative: true })` then generates an entry in which
  every lane computes its gate (in range, alive, cadence, species mask) and calls it, with no early
  return; a block in which no lane thinks stops before the call, all its lanes together. The block's
  first agent is `i - lane`. S3 chose an entry without barriers so that think could return early;
  that stays the entry of every other brain, byte for byte. **Without it,** rows run on (a), at the
  costs S6 measured (§6).

---

## 12. Build order

| Phase | Delivers | Exit |
|---|---|---|
| **11.0** ✅ `6042e41` | Spike S6 (`npm run spike:s6`) | Results below; the kernel to the user (Q4) |
| **11.1** ✅ `2e3c8c2` | The cooperative think entry; U44, and A35's first part on the GPU | Green and watched failing; every other brain's entry unchanged, byte for byte (results below) |
| **11.2** ✅ `2d51159` | `byGenome`, `setWeights({ row })`, `setRows`, the kernel and its guard; U43; the size line | U43 green and watched failing; size reported (results below: 21,362 B, 133 B under C2's limit) |
| **11.3** | A35 on three host/GPU pairs; a 16,384-agent step timed against today's brains | Green, watched failing; zero WebGPU errors; the step's cost reported |
| **11.4** | The guide: "A network per agent" in `docs/api/agents.md`, with a checked example and the block layout | The example runs in `agents-api-example.test.mjs` |

Phase 11.1 also runs A35's first part, the cooperative entry on the GPU, so the entry is proven on
three host/GPU pairs before any network uses it.

### Deferred

- Rows written on the GPU (mutation, births): the backlog's ecosystem.
- The staged kernel for single networks (§14, Q5).
- A tile sized to the network's largest layer instead of 512 vec4 (8 KB): less workgroup memory for
  small networks, so possibly more blocks at once; to be measured, not assumed.
- (b), (c) and (d), filed with their cost (S6): (b) and (c) are the cheapest rows for phase 7.2's
  small network in Chrome, and nowhere else.

---

## 13. Where this refines the parent

| Parent | This design |
|---|---|
| `agents-design.md` §9.3: "the record's `genome` word will select a row of a weight table" | Built as its own constructor, `MlpBrain.byGenome`, beside `bySpecies` |
| §13, Deferred: "Per-agent weights cost bandwidth (2k floats × N per think), so the think rate and genome sharing are Phase 2's design problem" | Measured by S6 at C2's shape (10,180 floats a row) and 7.2's, 16–256 rows. The layout decides more than the bandwidth: rows shared by 64-agent blocks are cheap. `thinkEvery` stays the app's lever |
| §9.3 and S3: the think kernel reads weights from storage in loops, with no barriers, "so the think entry can return early" | With (e′), a rows brain reads a block's row from workgroup memory, behind barriers, in a cooperative entry; every other brain keeps S3's kernel and entry |

---

## 14. Questions for the user

**Decided 2026-10-17.** The user: "for q1-q3 follow default and for q5 park its for later,and q4 fine
lets accept the costs and do recommended and you can start first step".
- **Q1:** (A), no action: an agent whose genome is outside the table stops (NaN actions).
- **Q2:** the API of §5, as written.
- **Q3:** C2's allowance kept; the number comes back to the user if the build needs more.
- **Q4:** (e′), staged with a per-lane fallback, its costs accepted: the cooperative think entry
  (§11), more WGSL, 2–3 s to compile under FXC.
- **Q5:** parked for later (plan-2's backlog).

The questions as they were asked:

1. **Q1. A genome word outside the table** (gᵢ ≥ N). Options:
   - **(A) No action** (recommended): the think writes NaN actions, so the built-in steering stops the
     agent, as it does for a NaN from any brain, and the inspector shows the genome. Visible, cheap
     (one compare), never a read outside the table. Custom actions receive the NaN and must handle
     it, as they must for any brain.
   - (B) Clamp to the last row: no branch, but a wrong row is silent.
   - (C) Row 0: silent too.
2. **Q2. The API of §5:** `byGenome(engine, networks)`, `setWeights(network, { row })`,
   `setRows(packed, { first })`, `rowCount`. Recommended as written.
3. **Q3. The size line:** inside C2's allowance (21,495 B for the whole entry) if the build adds at
   most 1,286 B; otherwise a new allowance for Phase 2. Recommended: keep C2's allowance and report;
   with (e′) the build may need more, and the number comes back to the user before 11.2 is committed.
4. **Q4. The kernel** (§6; the rule failed, so it comes to the user):
   - **(e′) Staged with a per-lane fallback** (recommended): the fastest rows where blocks share a
     row, faster than today's single network on NVIDIA and in Chrome with C2's networks; correct for
     any layout; bit-identical. Needs the cooperative think entry (§11), more WGSL, and 2–3 s to
     compile under FXC.
   - **(a) Genome per lane:** no change outside `MlpBrain`; +4–7% of a step on NVIDIA, +9–17% in
     Chrome, +33–36% under Node on the AMD iGPU, with C2's networks at 4,096–16,384 agents.
5. **Q5. Today's single-network kernel in Chrome** (S6, "What it shows", item 6). C2's networks
   think 7–12× slower in Chrome than under Node on the same AMD GPU: about three quarters of a
   16,384-agent Herd & Wolves step at thinkEvery 1, and half at 3, the demo's default. Staging runs
   them 2.5–4× faster, bit for bit. But staging slows other cases: a single network under Node on AMD
   (1.3–2×), and phase 7.2's small network in Chrome (2.7–3.9×). Recommended: **a follow-up after step 11,** added to plan-2's backlog: a spike over
   more shapes (widths 16–64, 2–4 layers) to find when staging pays, then a design that moves single
   networks onto step 11's staged kernel where it does. Step 11 itself changes no single-network
   brain.

---

## Spike S6 results — 2026-10-17

`npm run spike:s6` under Node/Dawn on both GPUs (the GTX 1650, "nvidia turing", and the AMD GCN-5
iGPU), and its page in Chrome on the AMD iGPU (`test/spikes/genome-rows/`). Every kernel was checked
against `mlpForward` of every agent's own row before it was timed, and every rows brain ran the same
Herd & Wolves simulation as today's brains, word for word. GPU timestamps; nothing else ran on the
GPU. Four runs:
- the whole matrix under Node (304 checks) and in Chrome, all green;
- (e′) with (a) and (e) beside it, kernels only, under Node (264 checks) and in Chrome, all green;
- a first Node run, before (e) existed and with part 2 timed one configuration after another, is
  superseded (its numbers agree within a few percent; part 2's thinkEvery 3 cell at 4,096 drifted).

Chrome measured the same kernel 13–16% apart between two runs (today's C2 kernel: 1,528 against
1,286 µs at 4,096 agents); compare ratios within a run.

### Part 1 — the think kernel alone

Microseconds per think dispatch for today's kernel, then each variant against it (×), at 4,096 /
16,384 / 65,536 agents, 64 rows, every agent thinking.

**C2's shape** (24 → 64 → 64 → 64 → 2, 40,720 B a row):

| Kernel | NVIDIA, Node | AMD, Node | AMD, Chrome |
|---|---|---|---|
| Today's kernel, one network (µs) | 246 / 830 / 3,146 | 206 / 721 / 2,589 | 1,528 / 7,665 / 31,160 |
| (e) staged, one network | 0.45 / 0.38 / 0.40 | 1.71 / 1.75 / 1.75 | 0.40 / 0.27 / 0.25 |
| (a) genome per lane, aligned | 1.15 / 1.14 / 1.05 | 3.49 / 2.94 / 2.68 | 1.43 / 1.09 / 1.02 |
| (a) genome per lane, runs of 8 | 1.37 / 1.60 / 1.69 | 5.71 / 5.41 / 5.75 | 1.54 / 1.29 / 1.42 |
| (a) genome per lane, shuffled | 4.85 / 5.33 / 5.87 | 6.88 / 6.64 / 6.91 | 1.55 / 1.72 / 1.81 |
| (b) workgroup index, aligned | 1.20 / 1.28 / 1.16 | 3.41 / 2.94 / 2.46 | 0.95 / 0.94 / 1.04 |
| (c) workgroup broadcast, aligned | 1.22 / 1.21 / 1.09 | 3.27 / 2.79 / 2.46 | 0.97 / 1.08 / 1.01 |
| (d) one pass per row, aligned | 54 / 16 / 4.3 | 63 / 18 / 5.1 | 10.9 / 2.2 / 0.63 |
| (e) staged, aligned | 0.85 / 0.66 / 0.55 | 2.86 / 1.91 / 1.75 | 0.43 / 0.28 / 0.24 |

**Phase 7.2's shape** (24 → 32 → 32 → 2, 7,952 B a row):

| Kernel | NVIDIA, Node | AMD, Node | AMD, Chrome |
|---|---|---|---|
| Today's kernel, one network (µs) | 44 / 156 / 597 | 28 / 166 / 533 | 55 / 200 / 628 |
| (e) staged, one network | 0.46 / 0.47 / 0.47 | 2.03 / 1.30 / 1.35 | 2.71 / 3.18 / 3.93 |
| (a) genome per lane, aligned | 2.13 / 1.24 / 1.00 | 2.75 / 1.91 / 1.94 | 5.03 / 4.08 / 4.36 |
| (b) workgroup index, aligned | 1.78 / 1.13 / 0.98 | 2.69 / 1.58 / 1.50 | 1.95 / 1.74 / 1.81 |
| (c) workgroup broadcast, aligned | 1.87 / 1.13 / 1.00 | 2.75 / 1.50 / 1.53 | 2.13 / 1.79 / 1.84 |
| (e) staged, aligned | 0.54 / 0.57 / 0.48 | 2.75 / 1.36 / 1.35 | 3.53 / 3.35 / 3.81 |

- **More rows:** at 65,536 agents, aligned, with 16 / 64 / 256 rows of C2's shape, (a) costs 0.98 /
  1.05 / 1.09× on NVIDIA, 2.23 / 2.68 / 2.91× on AMD under Node, 1.03 / 1.02 / 1.06× in Chrome; (e)
  0.43 / 0.55 / 0.68×, 1.75 / 1.75 / 2.03× and 0.23 / 0.24 / 0.26×.
- **Compile time** (16,384 agents, C2's shape): about 0.1 s for every kernel but (e) under FXC, which
  takes 2.9–3.2 s on NVIDIA and 1.8–2.0 s on AMD. In Chrome, today's kernel compiles in 0.30 s and
  (e) in 0.43–0.65 s.
- **Bit for bit:** (e)'s and (e′)'s actions equal today's kernel's (one row) and (a)'s (rows) in every
  cell compared, both shapes: 37 of 37 per host in the (e′) run, at every count and layout.

**(e′), staged with a per-lane fallback** (the (e′) run; against today's kernel, 64 rows):

| Shape, layout | NVIDIA, Node | AMD, Node | AMD, Chrome |
|---|---|---|---|
| C2, aligned | 0.86 / 0.68 / 0.58 | 2.84 / 1.93 / 1.79 | 0.59 / 0.35 / 0.31 |
| C2, runs of 8 | 2.86 / 2.21 / 2.25 | 8.10 / 7.91 / 8.38 | 2.06 / 1.34 / 1.22 |
| C2, shuffled | 6.18 / 6.79 / 7.54 | 9.13 / 8.91 / 9.36 | 2.37 / 1.53 / 1.39 |
| 7.2, aligned | 0.53 / 0.59 / 0.48 | 2.81 / 1.50 / 1.36 | 3.45 / 2.96 / 3.95 |
| 7.2, runs of 8 | 1.68 / 1.70 / 1.61 | 4.81 / 3.13 / 3.33 | 6.48 / 6.18 / 7.92 |
| 7.2, shuffled | 6.10 / 6.54 / 6.64 | 6.39 / 3.56 / 3.78 | 7.57 / 7.23 / 9.32 |

- **The fast path pays little for the fallback:** aligned, (e′) costs 1–9% more than (e) under Node
  (mostly under 5%), and up to 16% more in Chrome.
- **Mixed blocks** cost 0.6–2.1× what (a) costs in the same run: worse with C2's networks under Node,
  sometimes better in Chrome and with the small network on NVIDIA.

### Part 2 — a whole Herd & Wolves step

Today's brains against (a) with 64 rows per species, every row an identical copy of its species'
network: the same simulation, checked word for word. GPU milliseconds per step for today's brains
(median of 9 submissions, taken in turn with (a)'s after the settle), then what (a) adds, in % of
that step, at thinkEvery 1 / 3.

**C2's networks, NVIDIA (Node):**

| Agents | Today (ms) | (a) aligned | (a) runs of 8 | (a) shuffled |
|---|---|---|---|---|
| 4,096 | 1.12 / 1.03 | +6.6 / +14.5 | +14.4 / +19.4 | +105 / +55 |
| 16,384 | 2.84 / 2.36 | +6.5 / +5.2 | +19.4 / +10.4 | +128 / +65 |
| 65,536 | 15.7 / 13.5 | +3.7 / +2.7 | +13.5 / +5.3 | +100 / +45 |

**C2's networks, AMD (Node):**

| Agents | Today (ms) | (a) aligned | (a) runs of 8 | (a) shuffled |
|---|---|---|---|---|
| 4,096 | 1.82 / 1.67 | +32.8 / +17.7 | +67.1 / +37.4 | +89.2 / +48.3 |
| 16,384 | 5.15 / 4.23 | +35.6 / +18.3 | +65.4 / +28.7 | +79.7 / +36.7 |
| 65,536 | 61.7 / 64.0 | +11.0 / −1.0 | +19.7 / +4.3 | +25.7 / +6.1 |

**C2's networks, AMD (Chrome):**

| Agents | Today (ms) | (a) aligned | (a) runs of 8 | (a) shuffled |
|---|---|---|---|---|
| 4,096 | 2.82 / 1.86 | +17.4 / +14.3 | +34.8 / +41.9 | +56.9 / +61.3 |
| 16,384 | 11.36 / 5.69 | +8.9 / +8.8 | +17.6 / +15.7 | +41.1 / +30.1 |
| 65,536 | 82.5 / 62.6 | +5.3 / +3.2 | +18.0 / +6.3 | +31.7 / +8.7 |

**Phase 7.2's shape** (random networks, so the herd behaves differently), (a) aligned; today's step
in ms, then what (a) adds:

| Agents | NVIDIA, Node | AMD, Node | AMD, Chrome |
|---|---|---|---|
| 4,096 | 0.54 / 0.50: +13.2 / +2.9 | 0.86 / 0.77: +10.2 / +8.5 | 0.78 / 0.69: +26.7 / +12.3 |
| 16,384 | 1.83 / 1.67: +4.3 / −1.2 | 3.86 / 3.55: +3.3 / −0.7 | 3.60 / 3.19: +19.2 / +4.7 |
| 65,536 | 16.1 / 15.1: +0.5 / +0.8 | 69.9 / 64.0: −0.7 / −0.4 | 60.7 / 59.2: +5.0 / +1.0 |

### What it shows

1. **The decision rule fails.** (a), with rows aligned to blocks, adds to a step at 4,096–16,384
   agents (thinkEvery 1): 4–7% on NVIDIA, 9–17% in Chrome and 33–36% under Node on the AMD iGPU with
   C2's networks; 4–13%, 19–27% and 3–10% with phase 7.2's. On NVIDIA at 4,096 with thinkEvery 3 it
   adds 15%: at that size the think pass is bound by one block's latency, so a third of the blocks
   thinking take about as long as all of them.
2. **The layout matters more than the kernel.** Shuffled rows cost up to 15× a single network, runs of
   8 up to 12×, aligned rows 1.0–3.5× (5× for the small network in Chrome). Arenas should give each
   64-agent block one row (for step 14).
3. **A row the compiler can prove uniform helps in Chrome only.** Under Node (FXC), (b) and (c) cost
   what (a) costs, or more. In Chrome (DXC) they cost 1.7–2.1× for the small network against (a)'s
   4.1–5.0×: phase 7.2's explanation holds for DXC, not for FXC.
4. **One pass per row is the worst** almost everywhere: its small dispatches run one after another
   (each writes `act`), 2–100×.
5. **Staging the block's row in workgroup memory (e) is the fastest row kernel in five of the six
   GPU-and-shape cells, and bit-identical everywhere.** For C2's shape it is faster than today's
   single network on NVIDIA (0.55–0.85× with 64 rows) and in Chrome (0.24–0.43×), and the cheapest row
   kernel under Node on AMD (1.75–2.86×). The exception is the small network in Chrome: 3.4–3.8×,
   where (b) and (c) cost 1.7–2.1× (and (a) 4.1–5.0×). It compiles in 2–3 s under FXC (0.4–0.6 s in
   Chrome), and every lane must reach its barriers: a think entry without early returns.
6. **Today's kernel is slow in Chrome for C2's networks, and staging fixes it.** On the same AMD GPU,
   C2's network thinks 7–12× slower in Chrome than under Node (1.5 / 7.7 / 31 ms against 0.21 / 0.72 /
   2.6 ms per pass). At 16,384 agents the think passes take about three quarters of a step at
   thinkEvery 1 (11.4 ms) and half at 3 (5.7 ms), the live demo's default. Staging runs them 2.5–4×
   faster, bit for bit. This concerns every `MlpBrain` in Chrome, step 10's live demo first: a finding
   outside step 11 (§14, Q5).
7. **The network's size is the other lever.** Phase 7.2's network (1,988 parameters) thinks 4–7×
   faster than C2's (10,180) under Node and 28–50× faster in Chrome. With it, (a) adds at most 10–13%
   under Node at 4,096 agents and about nothing from 16,384 on.
8. **A per-lane fallback makes staging safe for any layout** at little cost to its fast path (the
   (e′) table): correct and bit-identical in every layout, fast where blocks share a row.

### What changes in the plan

- **Step 11's kernel goes to the user** (§14, Q4), recommended (e′).
- **Step 14 gains two inputs:** lay arenas out so that each 64-agent block shares one row; and weigh
  the network's size, which S6 shows costs as much as the kernel (plan-2 §4.4 already asks it).
- **A follow-up for every `MlpBrain`** (§14, Q5): C2's networks are slow in Chrome today, and
  staging fixes them, bit for bit; it waits on the user, after step 11.

### Not run

- Chrome on the NVIDIA GPU: Chrome on Windows ignores `powerPreference` and took the AMD iGPU.
- Part 2 with (b)–(e′): their whole-step effect needs the cooperative entry; §6's figures are
  estimates from part 1.
- Other browsers, and other shapes than the two measured.

---

## Phase 11.1 results — 2026-10-17: the cooperative think entry

**Built** (uncommitted):
- `Brain.cooperative` (`RuleBrain.ts`) and `assertThinkBlockSignature`;
- `buildAgentThinkWGSL(def, wgsl, thinkEvery, { cooperative })` (`agentThinkShaders.ts`);
- `AgentSystem` passes the flag through, for `setBrain` and `prepareBrain` alike, and refuses it on an
  async brain.

**How the entry decides that a block has work.** Each thinking lane sets a workgroup atomic; after a
barrier, lane 0 copies it into a plain workgroup word, which `workgroupUniformLoad` reads uniformly.
Dawn and Chrome also accept `workgroupUniformLoad` on the atomic itself, which saves a barrier. But
that overload came to WGSL later, so the entry keeps to the older, portable form.

**Tests:**
- **U44** (`test/agents-system.test.mjs`, 4 tests):
  - every non-cooperative entry is today's, byte for byte: fingerprints of a rule brain at
    thinkEvery 1 and 3, an `MlpBrain`, and `bySpecies` rows, taken before the change;
  - a cooperative entry reaches `thinkBlock` on every lane, behind today's gate, with one return,
    after the block's uniform flag;
  - the contract is checked (`thinkBlock` for cooperative brains, `think` for the rest, stage brains
    only);
  - `setBrain` builds the pass with its mask and group-1 buffers.

  **Watched failing:** one changed byte in today's entry, and an early return in the cooperative
  one, fail the first two tests.
- **A35's first part** (agents GPU suite): a cooperative brain sums its block's thinking agents in
  workgroup memory behind a barrier, with 130 agents (a partial last block), thinkEvery 3, wolves on
  another brain, two dead agents and a newborn. Exactly the agents today's gate lets think wrote
  their block's sum: 207 rows over 6 steps, every other row untouched.

  **Watched failing:** with the cadence taken out of the gate, all 52 off-cadence prey rows are
  written.
- **Runs:**
  - the agents GPU suite: 139/139 under Node on NVIDIA and on AMD, and green in Chrome on AMD;
  - `npm run test:agents`: 87/87;
  - `npm run test:inference`: 24/24;
  - the library and its types build.

**Size** (`size.mjs`):
- the whole agents entry: 20,209 → 20,532 B min+gzip (+323 B), within C2's 21,495 B. 963 B remain
  for phase 11.2.
- the rule path: 14,464 → 14,768 B (+304 B), within C1's 16 KB. The entry is generated by
  `AgentSystem`, so apps that use only rule brains carry it too.

---

## Phase 11.2 results — 2026-10-17: `MlpBrain.byGenome` and the staged kernel

**Built** (uncommitted):
- `MlpBrain.byGenome(engine, networks)`, `brain.rowCount`, `brain.cooperative` (`true` for
  `byGenome`, `undefined` for single and `bySpecies`), `brain.setWeights(network, { row })` and
  `brain.setRows(packed, { first })` (`MlpBrain.ts`), with all §10 refusal messages verbatim;
- `buildMlpThinkWGSL(network, rows?: readonly string[] | number)` (`mlpShaders.ts`): when `rows` is
  a count, emits `var<workgroup> mlpTile: array<vec4<f32>, 512>` (8 KB whatever the network: the size
  S6 measured; a tile sized to the network's largest layer is filed in "Deferred") and
  `var<workgroup> stagedRow: u32`, and generates `fn thinkBlock(i: u32, species: u32, tick: u32, thinking: bool, lane: u32)`
  implementing kernel (e′):
  - lane 0 reads the block's first agent's genome (`agents[i].ids.z`, where `i == blockFirst` at
    `lane == 0`), clamps it to `rows - 1` so cooperative tile loads never read outside `mlpWeights`,
    and stores `stagedRow = min(agents[i].ids.z, rows - 1u) * rowQuads`;
  - each layer runs in tiles of up to 512 `vec4<f32>` between `workgroupBarrier()` pairs; thinking
    lanes inside the table (`g < rows`) read `mlpTile` when `myBase == stagedRow` and fall back to
    `mlpWeights[myBase + ...]` otherwise;
  - thinking lanes with `g >= rows` (Q1 Option A) skip layer math and write `bitcast<f32>(nan)`
    (`0x7fc00000u`) to `act[0..A-1]`, never reading outside `mlpWeights`;
  - single-network and `bySpecies` paths share the inner row-loop helper and produce byte-for-byte
    identical WGSL to Phase 11.1 (verified by U21 golden and U44 SHA-256 fingerprints).

**Tests:**
- **U43** (`test/agents-mlp.test.mjs`, 4 tests):
  - `byGenome` packs rows in order into one `STORAGE | COPY_DST` buffer of `N * packMlpWeights.byteLength`
    bytes, sets `rowCount === N` and `cooperative === true` (`rowCount === 1` on single, species
    count on `bySpecies`);
  - `setWeights(net, { row })` writes `packMlpWeights(net).byteLength` bytes at offset `row * rowBytes`;
    `setRows(packed, { first })` (and omitted `{ first }`, or a subarray view) writes `packed.byteLength`
    bytes at `first * rowBytes`;
  - every refusal in §10 throws before any buffer creation or upload, with the verbatim message;
  - `WgslReflect` parses a complete `buildAgentThinkWGSL` module around a `byGenome` brain, including
    a wide layer (`128 -> 64`, `2,064 vec4<f32>` > 512, 4 tiles) with `mlpTile` sized at 512 quads
    and the `ids.z < N` guard writing `NaN` on out-of-range genomes.
- **Watched failing:**
  - perturbing the constructor's row offset, `setRows`'s byte offset, or `setRows`'s upper bound in
    `MlpBrain.ts` failed U43's packing, upload, and refusal tests;
  - dropping the `g < rows` guard in `mlpShaders.ts` failed U43's WGSL / guard test;
  - restored byte for byte.
- **Runs:**
  - `npm run test:agents`: 91/91 node:test + 24 reference checks;
  - `npm run test:inference`: 24/24 (including its bundle size assertions);
  - `npm run build:lib`: clean.

**Size** (`node test/spikes/mlp-think-cost/size.mjs`):
- the whole agents entry: 20,532 → **21,362 B** min+gzip (+830 B in 11.2; +1,153 B across 11.1–11.2
  from 20,209 B), **133 B within C2's 21,495 B allowance**;
- the rule path: **14,768 B** min+gzip (+0 B in 11.2; `MLP code: none`), within C1's 16 KB;
- rule path + `MlpBrain`: 16,711 → **17,547 B** min+gzip (+836 B; `MlpBrain` adds 2,779 B min+gzip
  to a rule-path app).

**Review** (Claude, 2026-10-17, before the commit; the user asked for a minor review):
- **Fixed, blocking:** the staged kernel declared `var active`. `active` is a WGSL reserved word, so
  no device could compile a `byGenome` brain ("'active' is a reserved keyword", Dawn). U43 parsed it
  because `wgsl_reflect` does not know reserved words, and 11.2 had no GPU run. Renamed `inTable`;
  the module now compiles under Node/Dawn. Recorded in `gotchas.md`.
- **Fixed in this document:** the tile is always 512 vec4 (8 KB), the size S6 measured, not
  `min(maxLayerQuads, 512)`; a tile sized to the network is filed under "Deferred".
- **Added to §10:** `setRows`' refusal of non-finite weights (as the constructor and `setWeights`
  refuse them) and of a brain not made by `byGenome`.
- **Checked, no change:**
  - single-network and `bySpecies` WGSL is byte for byte phase 11.1's (U44's fingerprints);
  - the gate, the block's staged row (its first agent's genome, clamped), the fallback, and Q1's NaN
    actions follow §6;
  - workgroup memory is 8,204 B, within the default 16 KB.

**Committed** as `2d51159`, after `6042e41` (phase 11.0, spike S6) and `2e3c8c2` (phase 11.1). Each
commit was validated alone in a temporary worktree:
- `npm run build:lib`;
- `test:agents`: 87/87 at 11.1, 91/91 at 11.2;
- `test:inference`: 24/24;
- the agents GPU suite: 139/139 on NVIDIA;
- at 11.2, a `byGenome` think module compiled on a device.

