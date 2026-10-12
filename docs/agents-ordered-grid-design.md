# NullGraph Agents — C1 step 5: deterministic neighbour lookups (a stable grid order) — Technical Design

**Parent:** [`agents-demo-design.md`](./agents-demo-design.md) (step 5) §5–§6, where "grid-ordered
queries" are listed as a possible improvement; [`agents-design.md`](./agents-design.md) §6, the
determinism note.

**Asked by the user, 2026-10-13:** "build step 5 first", before step 7, part 2, so that recorded
teacher data is reproducible even when crowded cells overflow the scan budget.

**Decided with the user, 2026-10-13** (§10; [`work/plan.md`](./work/plan.md) §5, item 6):
1. **Scope:** the fixed-order lookups only. The scaling study (the parent's phases 5.0–5.4) comes
   later.
2. **Crowded cells:** a rotating window, offset per querying agent and tick, and deterministic.
   This is decided now, before any teacher data is recorded.
3. **Always on, as a single path.** The numbers go back to the user only if the sort adds more than
   about 10% to a step at 4,096 or 65,536 agents, on any host.
4. **Custom sensors reading the neighbour lists:** contract (a), later, when a demo needs it; not in
   step 5. Phase 5.8 documents today's limits and fixes the parent's claim (§8).

**Status:** ✅ design decided 2026-10-13. Next: phase 5.5.

**Target:**
- `src/extras/spatial/`: the grid's scatter and the CPU mirror;
- `src/extras/agents/`: the query's window and its tick source;
- tests, and the determinism paragraphs of the guides.

Core unchanged.

---

## 1. Purpose

**Same device, lockstep, built-in sensors ⇒ bit-exact reruns, including when crowded cells
overflow.** Today that holds only without overflow:
- **Measured 2026-10-12:** between two identical 200-step runs at budget 64, 19,990 of 64,000
  record words differed; none differed without overflow.
- **Overflow is the norm for flocks:** Herd & Wolves overflowed in 85% of steps at the default
  budget ([`api/agents.md`](./api/agents.md)).

Teacher data recorded from it would therefore not be reproducible:
- re-recording the same scenario gives different (observation, action) pairs;
- a trained network cannot be compared with the rule on the same run twice.

**A second gain:** the GPU grid becomes identical to its CPU mirror, order included. Tests can then
check bounded queries against the CPU reference directly, instead of against an oracle built from
the GPU's own scatter order.

**A third gain, from decision 2:** in a crowded cell, each querying agent sees a different slice of
the cell, and the slice moves every tick. Today every query in a cell sees the same first entries
of the cell every step.

### Non-goals

- **Removing the budget:** overflow still drops candidates. The approximation stays, but becomes
  deterministic.
- **Cross-device bit-exactness:** float rounding differs between GPUs; not claimed.
- **The scaling study** (the parent's phases 5.0–5.4): later.
- **Faster queries or memory locality:** separate investigations.
- **A contract for custom sensors reading the neighbour lists:** later (decision 4, §8).

## 2. Where the order comes from today (verified against this repository, 2026-10-13)

| # | Fact | Where | Consequence |
|---|---|---|---|
| 1 | The grid builds as reset → bin (atomic count, overflow) → exclusive scan → scatter (an atomic cursor per cell) | `PointGrid.ts` constructor; `gridShaders.ts` | Counts, offsets and the overflow total are order-independent. **Only the scatter's slot within a cell follows GPU timing** |
| 2 | A query scans the first `min(count, maxScanPerCell)` entries of each of its 3×3 cells in `cellBodies` order, keeping k-nearest with ties broken by index | `agentGridShaders.ts` | Within budget the result is order-independent. Past it, **which** entries are scanned follows the scatter order |
| 3 | The CPU mirror scatters in index order, so each of its cells lists ascending agent indices | `buildPointGrid`, `gridMath.ts` | A stable GPU scatter reproduces the mirror exactly, order included |
| 4 | The grid query's CPU mirror is `gridNearestNeighbours`; the CPU step reference uses brute force (`nearestNeighbours`) | `agentGridMath.ts`; `agentMath.ts` | The window changes the grid mirror only. A trailing `tick` parameter breaks no caller |
| 5 | A1 compares cells only as sets, and bounded queries against a mirror rebuilt from the GPU's own order | `test/agents-gpu/run.js`, A1 | Both checks can become exact against the CPU mirror |
| 6 | A7 asserts zero overflow (budget = capacity) | A7 | No test covers determinism past the budget |
| 7 | `AgentSystem`'s tick is word 0 of its state buffer, written only by the GPU tick stage at the end of each step. `AgentGrid` has no tick | `agentSystemShaders.ts`; `AgentSystem.ts` | A query can read the step's tick (the one the brain sees). `AgentGrid` needs a build counter of its own (§4) |
| 8 | `PointGrid` serves `AgentSystem`, `AgentGrid` and debug-ui's `gridOccupancy`, which reads counts only | grep of `src/` | The change is local to the spatial grid and the agent query |
| 9 | The scan runs 256-lane workgroups, and `PointGrid` already requires them | `GRID_SCAN_BLOCK`, `PointGrid.ts` | New passes can reuse the 256-lane block and the recursive scan |
| 10 | `rand01`'s PCG hash has a bit-exact CPU twin | `AGENT_RANDOM_WGSL`; `RuleBrain.ts` | The window offset can use the same hash core with a reserved stream, without touching `rand01` |

## 3. The design

### A stable scatter: radix-sort the grid entries by cell

Replace the atomic scatter with a **stable least-significant-digit radix sort** keyed on the cell:
- **Keys and values:** the key is the agent's cell; a dead agent gets the key `cellTotal`, so it
  sorts last and is never read. The values are agent indices, entering in index order.
- **Passes:** 8-bit digits; the number of passes is ⌈bits(`cellTotal` + 1) ÷ 8⌉.
  - 1 pass up to 255 cells;
  - 2 passes up to 65,535 cells: Herd & Wolves has 400 cells at 4,096 agents and 6,400 at 65,536;
  - 3 passes beyond: 102,400 cells at the demo's 1,048,576 agents.
- **Each pass:**
  1. **Histogram:** each 256-agent block counts its digits in workgroup memory. Counts do not
     depend on order.
  2. **Scan:** an exclusive scan over the digit-major table `[digit][block]`, with the existing
     scan kernels.
  3. **Stable scatter:** an agent's new position is the table offset for its (digit, block), plus
     the number of earlier lanes in its block with the same digit. That count is a loop over
     workgroup memory, with no atomics.
- **The result:** `cellBodies` in (cell, index) order, identical to the CPU mirror. The bin pass
  still produces `cellCounts` and the overflow total, and the scan still produces `cellOffsets`.
- **Removed:** the cursors buffer, its reset and the atomic scatter. This is a single path: the
  atomic scatter is not kept as an option (decision 3).
- **Added:**
  - ping-pong keys and values (4 × capacity words; 1 MB at 65,536 agents);
  - the digit table (256 × blocks words; 256 KB at 65,536 agents);
  - about 5 dispatches per pass.

**Rejected:**
- **Sorting only the cells that overflow:** one crowded cell of 10,000 agents becomes one long
  sort, so the work per step has no bound. That risks the GPU reset on Windows past about 2 s per
  submission.
- **Recording with overflow off:** that stays possible for any recording that can afford it.

### The window: which entries a crowded cell yields (decision 2)

**In each of its 3×3 cells whose count exceeds the budget B,** a query scans B consecutive entries
of the cell, starting at `w mod count` and wrapping within the cell:

```wgsl
let w = gridWindow(i, tick);            // per querying agent and tick
for (each of the 3×3 cells c) {
    let count = counts[c];
    let n = min(count, B);
    var start = 0u;
    if (count > B) { start = w % count; }
    for (var s = 0u; s < n; s++) {
        var at = start + s;
        if (at >= count) { at -= count; }
        let j = bodies[offsets[c] + at];
        // … unchanged: skip self, dead and masked-out agents; keep k-nearest, ties by index
    }
}
```

- **The hash:** `gridWindow(i, tick)` is the PCG hash that `rand01` uses, before its float
  conversion, with a stream reserved for the grid. Its CPU twin is bit-exact (fact 10).
- **A cell within budget** is scanned whole, so its start does not matter; the code uses 0.
- **Deterministic:** the window depends only on `(i, tick)` and on the cell's fixed order. Both
  parts are needed: the stable sort fixes what each position holds, and the window picks which
  positions a query reads.
- **The tick source:** the query gains binding 5, a storage word holding the tick.
  - **`AgentSystem`** binds its state buffer, so the query sees the step's tick, the same tick its
    brain and `rand01` see.
  - **`AgentGrid`**, which has no tick, owns a counter word. A small stage after its query
    increments it, so build b reads tick b.
- **The CPU mirror:** `gridNearestNeighbours(def, rec, grid, i, mask, radius, k, out, tick = 0)`
  applies the same window. A1 and A23 pass the step's tick to it.

## 4. Stages

Per step, today:

```
reset → bin → scan → scatter → queries → … → tick
```

After this change:

```
reset → bin → scan → [histogram → scan → stable scatter] × passes → queries (window) → … → tick
```

The last pass writes `cellBodies` directly. `AgentGrid` appends its counter stage after its query.

## 5. The determinism claim afterwards

Same device, lockstep, built-in sensors, and custom sensors that read only their own record ⇒
bit-exact reruns, with or without overflow. The guides' overflow caveats are rewritten to match in
phase 5.8:
- [`api/agents.md`](./api/agents.md): "Definition and observations" and the lockstep paragraph;
- [`api/agent-grid.md`](./api/agent-grid.md): "Atomic scatter order is unspecified".

## 6. Costs, and the rule for bringing them back (decision 3)

**Measured in phase 5.5 (before) and 5.7 (after)**, on all three host/GPU pairs: Node/Dawn on the
AMD iGPU and on the NVIDIA GPU (`NG_DAWN_FLAGS=adapter=NVIDIA`), and Chrome on the AMD iGPU.
- **What:**
  - the time of a whole step;
  - the grid build's share of that step.
- **Workload:** Herd & Wolves' setup, with its density, sensors, species and scan budget 64, at
  4,096 and 65,536 agents.
- **The step:** the library's step alone, without the demo's own app stages. That is the
  conservative choice, because a smaller step makes the sort's share look larger.

**The rule:** if the sort adds more than about 10% to a step at either size, on any host, the
numbers go back to the user before anything else. Otherwise the stable scatter stays as the single
path.

**Bundle:** estimated +0.6–1.0 KB min+gzip on the rule path, which every agents app carries:
- the radix kernels;
- the window;
- `AgentGrid`'s counter.

C2's allowance has 3,317 B left (18,178 of 21,495 B), with `fromOnnx` (about 2 KB) still to come.

## 7. Tests

**Headless (U23):**
- the generated radix kernels and the windowed query parse (`WgslReflect`), for 1, 2 and 3
  passes;
- a CPU model of the passes equals `buildPointGrid`'s `cellBodies` on random, crowded and tie
  layouts, which checks the algorithm without a GPU;
- the window's CPU twin:
  - starts at `w mod count` and wraps;
  - leaves cells within budget unchanged;
  - with tick 0 and every cell within budget, gives exactly today's results;
- `PointGrid` builds the expected stages for each pass count, and `AgentGrid` its counter stage
  (mock engine);
- the new buffers are refused by name when they exceed device limits.

**GPU:**
- **A1, tightened:**
  - `cellBodies` equal the CPU mirror exactly, on all four layouts, including crowded at
    budget 2;
  - bounded queries equal `gridNearestNeighbours` with the same tick directly, over several
    builds of an `AgentGrid`;
  - the oracle built from the GPU's order is retired.
- **A23 (new):** two identical lockstep runs with heavy overflow, about 4,096 agents at Herd &
  Wolves' density with a budget of 8, for 200 steps. They must give byte-identical records,
  observations, actions and reasons, and the overflow counter is asserted > 0.
  - **Watched failing:** phase 5.5 measures today's divergence in the spike. Once the stable
    scatter lands, a mutation that brings back an atomic slot must make A23 fail.
  - If a host happens not to diverge, that is recorded as "watch not observed there", never
    hidden.
- **Unchanged:** A7 and every other check, on all three host/GPU pairs.

## 8. Custom sensors and the neighbour lists (decision 4)

**The question (the user, 2026-10-13):** can a custom sensor read the neighbour lists, as a "can I
see the nearest seeker" check would need?

**Today's limits:**
- **Technically yes.** A custom sensor's WGSL is compiled into the sense shader. There the agent
  records (`agents`) and every nearest sensor's index list (`neighbours`) are in scope, and the
  query stages fill the lists earlier in the same step.
- **But there is no contract:**
  - no generated constants for the list layout. The stride is the sum of all nearest sensors' `k`,
    and a sensor's base is the sum of the earlier sensors' `k`, so the app would hard-code both;
  - the binding names are not documented;
  - the CPU mirror callback (`records, u32, i, out, o`) does not receive the lists;
  - sensors cannot bind app buffers, so walls and occluders are out of reach.
- **For a view-cone and range check, no custom sensor is needed.**
  `Sensors.nearest({ name: 'seeker', species: ['seeker'], k: 1 })` already gives the nearest
  seeker's side, forward and presence, and the brain can test the cone from those.
- **True line of sight** (walls blocking the view) needs app data in the sense stage, which the
  API does not offer.

**Decided:** contract (a), later, when a demo needs it; not in step 5. Contract (a) means:
- generated constants: `NEIGHBOUR_STRIDE`, and one `NEIGHBOURS_<SENSOR>` base per nearest sensor;
- documented `agents` and `neighbours` access;
- the lists passed to the CPU mirror.

**Phase 5.8 documents today's limits above in the guide.** It also fixes
[`agents-design.md`](./agents-design.md) §6, which says the API docs warn against order-dependent
float sums over neighbours. They do not. After this step, such sums are deterministic in any case,
because the grid's order and the lists are fixed.

## 9. Build order

Numbered after the study's later phases 5.0–5.4.

| Phase | Builds | Exit |
|---|---|---|
| **5.5** | A measurement spike on today's code, on three host/GPU pairs: (1) A23's scenario, two identical runs with heavy overflow, counting the differing words; (2) the baseline step time and grid share at 4,096 and 65,536 agents (§6) | Numbers recorded; the watch observed, or its absence recorded |
| **5.6** | The stable scatter, the window and its tick source, and the CPU mirror; U23; A1 exact; A23 green, and watched failing by mutation | Headless green; GPU green on three host/GPU pairs |
| **5.7** | The same spike after the change; the bundle | Within the 10% rule, or the numbers taken to the user |
| **5.8** | Docs: the determinism paragraphs (guide, `agent-grid.md`, parent §6); today's custom-sensor limits; closure | Docs match what was measured; `docs/work` updated |

## 10. Decisions (the user, 2026-10-13)

1. **Scope:** the fixed-order lookups only; the scaling study later.
2. **Crowded cells:** a rotating window, offset per querying agent and tick, and deterministic.
   Decided now, before any teacher data is recorded.
3. **Always on, as a single path.** The numbers go back to the user only if the sort adds more than
   about 10% to a step at 4,096 or 65,536 agents, on any host.
4. **Custom sensors:** contract (a), later, when a demo needs it; not in step 5. Phase 5.8
   documents today's limits and fixes the parent's claim.
