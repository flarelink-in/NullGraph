# NullGraph Agents — C1 step 5: deterministic neighbour lookups (a stable grid order) — Technical Design

**Parent:** [`agents-demo-design.md`](./agents-demo-design.md) (step 5) §5–§6, where "grid-ordered
queries" are listed as a possible improvement; [`agents-design.md`](./agents-design.md) §6, the
determinism note.

**Asked by the user, 2026-10-13:** "build step 5 first", before step 7, part 2, so that recorded
teacher data is reproducible even when crowded cells overflow the scan budget.

**Decided with the user, 2026-10-13** (§10; [`work/plan.md`](./work/plan.md) §5, item 6):
1. **Scope:** the fixed-order lookups only. The scaling study (the parent's phases 5.0–5.4) comes
   later.
2. **Crowded cells:** a rotating window, offset per cell and tick, and deterministic. This is
   decided before any teacher data is recorded.
   - **Revised by the user, 2026-10-13.** First decided as a window per querying agent. Phase 5.7
     measured that at +15% to +26% on a step, so the window is now per cell (§10; "Phase 5.7
     follow-up").
3. **Always on, as a single path.** The numbers go back to the user only if the sort adds more than
   about 10% to a step at 4,096 or 65,536 agents, on any host.
4. **Custom sensors reading the neighbour lists:** contract (a), later, when a demo needs it; not in
   step 5. Phase 5.8 documents today's limits and fixes the parent's claim (§8).

**Status:** ✅ design decided 2026-10-13 (`b7050b7`); ✅ 5.5 measurements and initial 5.6
implementation committed (`e4eaad2`); ✅ 5.6 correctness verified 2026-10-13 (`caaf930`).
5.7 first measured 2026-10-13 (`da262ad`): whole-step regressions over the ~10% rule, so the
numbers went back to the user. Its follow-up found the sort within the rule and the cost in the
window per querying agent. **The user revised decision 2 on 2026-10-13: a window per cell and
tick.** ✅ **5.7 is within the rule with that change:** the worst case is +9.6% (results at the
end). 5.8 is not started.

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

**A third gain, from decision 2:** the slice of a crowded cell that queries see moves every tick,
so no agent is hidden for good by its index. Before step 5, every query in a cell saw the same
first entries of the cell every step.

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
of the cell, starting at `gridWindow(c, tick) mod count` and wrapping within the cell:

```wgsl
for (each of the 3×3 cells c) {
    let count = counts[c];
    let first = offsets[c];
    let stop = first + count;
    var at = first;
    if (count > B) { at += gridWindow(c, tick) % count; }    // per cell and tick
    for (var left = min(count, B); left > 0u; left--) {
        let j = bodies[at];
        at++;
        if (at == stop) { at = first; }
        // … unchanged: skip self, dead and masked-out agents; keep k-nearest, ties by index
    }
}
```

- **Per cell, not per querying agent** (revised 2026-10-13). Every query of a crowded cell reads
  the same B entries in a tick, so the GPU fetches them once and serves the rest from its cache. A
  window per querying agent made every query read different entries, and cost +15% to +26% on a
  step ("Phase 5.7 follow-up").
- **The loop** reads the cell's offset once and walks one index. Phase 5.6's first form recomputed
  the position for every candidate, and cost the AMD iGPU 10 to 18 points at 4,096 agents.
- **The hash:** `gridWindow(c, tick)` is the PCG hash that `rand01` uses, before its float
  conversion, with a stream reserved for the grid. Its CPU twin is bit-exact (fact 10).
- **A cell within budget** is scanned whole, so its start does not matter; the code uses 0.
- **Deterministic:** the window depends only on `(c, tick)` and on the cell's fixed order. Both
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
| **5.5** ✅ | A measurement spike on today's code, on three host/GPU pairs: (1) A23's scenario, two identical runs with heavy overflow, counting the differing words; (2) the baseline step time and grid share at 4,096 and 65,536 agents (§6) | Measured 2026-10-13: the watch observed on all three; results below |
| **5.6** | The stable scatter, the window and its tick source, and the CPU mirror; U23; A1 exact; A23 green, and watched failing by mutation | Headless green; GPU green on three host/GPU pairs |
| **5.7** ✅ | The same spike after the change; the bundle | The numbers went to the user, who revised decision 2. Within the 10% rule with a window per cell, measured like for like 2026-10-13 |
| **5.8** | Docs: the determinism paragraphs (guide, `agent-grid.md`, parent §6); today's custom-sensor limits; closure | Docs match what was measured; `docs/work` updated |

## 10. Decisions (the user, 2026-10-13)

1. **Scope:** the fixed-order lookups only; the scaling study later.
2. **Crowded cells:** a rotating window, offset per cell and tick, and deterministic. Decided
   before any teacher data is recorded.
   - **Revised 2026-10-13:** per cell, with the tighter scan loop (option B of "Phase 5.7
     follow-up"). First decided per querying agent, which cost +15% to +26% on a step at 65,536
     agents flocked and 8 to 13 times the step in a pile-up.
3. **Always on, as a single path.** The numbers go back to the user only if the sort adds more than
   about 10% to a step at 4,096 or 65,536 agents, on any host.
4. **Custom sensors:** contract (a), later, when a demo needs it; not in step 5. Phase 5.8
   documents today's limits and fixes the parent's claim.

## Phase 5.5 results — 2026-10-13: today's grid

**The spike:** `test/spikes/grid-order/` (`run.js`, `index.html`; commands in its header). It is
not registered as a suite. It runs on every adapter, with the library's step and Herd & Wolves'
setup: density, sensors, species, speeds, `thinkEvery` 3. The test rule makes prey flee the nearest
wolf and wolves chase the nearest prey, and both turn away from near walls.

**Divergence (A23's scenario).** Two identical lockstep runs at 4,096 agents with budget 8; every
one of the 200 steps overflowed. The words that differed afterwards:

| Host and GPU | Records (of 65,536) | Observations (of 98,304) | Actions (of 8,192) |
|---|---|---|---|
| Node/Dawn, NVIDIA | 19,257 | 70,086 | 3,901 |
| Node/Dawn, AMD iGPU | 16,302 | 63,135 | 2,883 |
| Chrome, AMD iGPU | 16,529 | 63,691 | 2,984 |

The watch is observed on all three host/GPU pairs. Reasons never differ, because the test rule's
reason depends only on the species.

**Cost before the change.** GPU timestamps, budget 64, the median of 5 trials, in milliseconds;
"grid" is the grid build alone:

| Agents, start | NVIDIA, Node: step / grid (share) | AMD, Node: step / grid (share) | AMD, Chrome: step / grid (share) |
|---|---|---|---|
| 4,096, spread | 0.508 / 0.017 (3.4%) | 0.894 / 0.016 (1.7%) | 0.740 / 0.015 (2.0%) |
| 4,096, flocked | 0.728 / 0.014 (1.9%) | 1.897 / 0.015 (0.8%) | 1.764 / 0.015 (0.9%) |
| 65,536, spread | 13.959 / 0.080 (0.6%) | 65.536 / 0.568 (0.9%) | 62.470 / 0.533 (0.9%) |
| 65,536, flocked | 23.986 / 0.078 (0.3%) | 124.125 / 0.526 (0.4%) | 115.275 / 0.515 (0.4%) |

- **What the 10% rule allows in 5.7:** at 4,096 agents spread, the sort may add at most about
  51 µs on NVIDIA, 89 µs on AMD under Node and 74 µs in Chrome. At 65,536 agents the allowance is
  1.4–12 ms.
- **What matters at 4,096 agents** is the number of dispatches, not bandwidth. Today a step has 12
  dispatches, 6 of them the grid.
- **A refinement for 5.6:** choose the digit width that needs the fewest passes, up to 10 bits
  (1,024 buckets), instead of a fixed 8. Herd & Wolves' 400 cells then need one pass, and its 6,400
  and 102,400 cells two. 5.6 confirms the choice by measurement.
- **Overflow during the measured window:**
  - **flocked:** in every step;
  - **spread:** in 0–17% of steps. NVIDIA, being faster, runs more steps in the window, and prey
    gather as they flee.
- **The first run was discarded.** Without wall avoidance, prey piled into the corners, and a
  warm-up that ran steps for a fixed time left each host in a different state. The spike now warms
  up on the grid build alone, which moves nobody.

## Phase 5.6 results — 2026-10-13: correctness

- Stable LSD radix sorting is the sole production scatter path. Digit width minimizes passes
  (up to 10 bits), then table size; dead entries sort behind the live cells.
- Queries read a deterministic PCG window from agent index and GPU tick. `AgentGrid` increments
  its own counter after querying; `AgentSystem` binds its existing state word.
- A1 compares ordered live entries and bounded queries directly with the CPU grid. The production
  wrapper also matches the CPU window at ticks 1–5, including two builds in one submission.
- A23 uses 4,096 agents in a crowded 24 m square within the 160 m world, budget 8, 200 steps,
  and the Herd & Wolves sensors/rule. Every step overflows; both production runs have identical
  records, observations, actions and reasons. This is a deliberately crowded correctness fixture,
  not a replacement for phase 5.7's unchanged measurement spike.
- **GPU:** 137/137 checks, zero GPU errors, on Node/Dawn AMD, Node/Dawn NVIDIA and Chrome AMD.
- **Mutation watch:** a test-only stage overwrites the sorted grid with the old atomic-slot
  scatter before querying. Two mutated runs diverge on all three hosts:

  | Host | Record / observation / action words differing |
  |---|---|
  | Node/Dawn AMD | 20,372 / 72,137 / 4,270 |
  | Node/Dawn NVIDIA | 20,374 / 72,895 / 3,817 |
  | Chrome AMD | 20,349 / 72,347 / 4,162 |

- U23 covers generated kernels and a stable block model for 1/2/3 passes, random/crowded/tie
  scenes against `buildPointGrid`, wrapped windows, unchanged within-budget queries, stage/counter
  layout, named radix limit refusals, and allocation/binding failure cleanup. Caller records survive.
- **Next:** 5.7 measures the unchanged spike in isolation and checks bundle size. The 10% rule
  still applies before guide closure or teacher-data work.

## Phase 5.7 results — 2026-10-13: acceptance paused

The unchanged phase 5.5 spike was run after a fresh development build, Node adapters sequentially,
then Chrome separately, without concurrent GPU validation. GPU timestamps, median of 5 trials,
same workload and measurement protocol as 5.5. Both Node adapters and Chrome completed with
zero GPU errors. The two 200-step recordings now differ in **zero** record, observation, action
and reason words on every host; every step overflowed.

**Whole-step time**, milliseconds; parentheses show change from the recorded 5.5 baseline:

| Agents / layout | Node NVIDIA | Node AMD | Chrome AMD |
|---|---|---|---|
| 4,096 / spread | 0.523 (+2.9%) | 0.960 (+7.4%) | 0.891 (+20.5%) |
| 4,096 / flocked | 1.024 (+40.7%) | 2.119 (+11.7%) | 1.833 (+3.9%) |
| 65,536 / spread | 14.123 (+1.2%) | 99.353 (+51.6%) | 65.358 (+4.6%) |
| 65,536 / flocked | 30.802 (+28.4%) | 159.252 (+28.3%) | 140.981 (+22.3%) |

**Grid build alone**, milliseconds:

| Agents / layout | Node NVIDIA | Node AMD | Chrome AMD |
|---|---|---|---|
| 4,096 / spread | 0.064 | 0.068 | 0.076 |
| 4,096 / flocked | 0.046 | 0.070 | 0.075 |
| 65,536 / spread | 0.266 | 1.763 | 1.428 |
| 65,536 / flocked | 0.266 | 1.298 | 1.453 |

The build increase alone is less than 10% of the old step in every measured case (largest about
9.3%, NVIDIA 4,096 spread). **The whole ordered-query change is not:** multiple whole-step cases
exceed 10%. These measurements do not isolate causation. Rotating bounded-query access and changed
simulation trajectories are possible contributors; clock variation and the time-budgeted evolving
step window also warrant an interleaved baseline rerun before attributing the entire difference.
No optimization, alternate scatter path or policy change was made in response.

**Bundle**, same source-based esbuild browser ESM/minify + default gzip settings as 6.3. Baseline
source at `b7050b7` was bundled with the same installed toolchain, reproducing 18,178 B:

| Import set | Before | After | Increase |
|---|---:|---:|---:|
| Whole agents entry | 18,178 B | 19,305 B | 1,127 B |
| Rule path | 12,741 B | 13,566 B | 825 B |
| Rule path + MlpBrain | 14,657 B | 15,495 B | 838 B |

The whole entry is below the 21,495 B ceiling by **2,190 B**. Rule-only imports still exclude all
MLP markers. This leaves little margin for the estimated ~2 KB `fromOnnx` work.

**Decision needed:** review the regressions before 5.8 or teacher-data work. Suggested next
investigation: interleaved before/after measurements from identical fixed snapshots, separating
stable-sort cost from windowed-query cost. The agreed stable ordering/window semantics remain
unchanged pending that discussion.

## Phase 5.7 follow-up — 2026-10-13: where the cost comes from

**In short:**
- **The sort is cheap.** A sorted grid with the old query costs at most +8% on a step (4,096
  agents on the AMD iGPU, where the build's extra 0.05 ms shows) and stays within noise at 65,536.
- **The window per querying agent is the regression:** +15% to +26% at 65,536 agents flocked on
  all three host/GPU pairs, and 8 to 13 times the step in a pile-up.
- **A window per cell and tick** keeps the determinism and the rotation for at most +8% on that
  case.
- **Two of 5.7's worst figures did not reproduce** (+40.7% and +51.6%). They came from how the
  acceptance spike times a run that keeps evolving.

**Decision 2 went back to the user, who chose option B** (at the end of this section). This
section was measured before that change: its "today" and "production" mean the window per
querying agent.

### How it was measured

**The spike:** `test/spikes/grid-policy/` (`run.js`, `workload.mjs`, `workload.test.mjs`,
`prepare.mjs`, `index.html`; commands in `run.js`). Test-only and not registered as a suite.

- **The workload** is the acceptance spike's, copied character for character (a headless test
  fails if the two drift apart): Herd & Wolves' setup, budget 64, `thinkEvery` 3, at 4,096 and
  65,536 agents, spread and flocked.
- **One snapshot per case,** three steps after spawn, where the acceptance spike starts. Every
  sample restores it (records, observations, actions, tick), then runs a fixed number of steps.
- **The chains are sampled in turn** in one process; the order rotates and reverses, so drift
  favours none. Only the grid build and the two query stages differ between chains: sense, think,
  act and tick are the system's own stages in all of them.

| Chain | Grid | Which entries a crowded cell yields |
|---|---|---|
| `old` | atomic scatter, bundled from `b7050b7` | its first entries, in the order GPU timing left them |
| `first` | sorted (today's) | its first entries: the lowest indices |
| `cell` | sorted | a window that rotates per cell and tick |
| `agent` | sorted | a window that rotates per querying agent and tick: **today's production step, untouched** |
| `cell+`, `agent+` | sorted | the same entries in the same order as `cell` and `agent`, scanned by a tighter loop: the cell's offset is read once, and one index walks the cell and wraps |
| `agent2` | sorted | `agent` again, from stages the spike builds: the noise between two identical chains |

**Two timings,** both in GPU timestamps:
- **a few steps** from the snapshot: 16 at 4,096 agents and 2 at 65,536; the median of 15 and 9
  samples;
- **a longer run** from the snapshot: 512 steps at 4,096 agents and 64 at 65,536; the mean of 3
  and 2 runs. Neighbours differ by policy, so herds move differently, and a step costs what the
  crowding it meets costs.

**Checked before anything is timed,** in every case on every host:
- each chain's query output equals a CPU scan of the grid it read, at two ticks, for every agent
  and both sensors;
- emptying the other grid changes nothing, so each chain reads the grid it builds;
- `cell+` and `agent+` equal `cell` and `agent` word for word, at both ticks and after the longer
  run;
- every sample ends at the expected tick;
- the `first`, `cell` and `agent` runs repeat word for word.

**Results:** 352/352 checks under Node (NVIDIA and AMD), 176/176 in Chrome, zero GPU errors. The
headless test's four checks were watched failing under six mutations.

### Whole step, change against the code before step 5

Each cell reads **a few steps / the longer run**. "Before" is `old`, in milliseconds.

**Node/Dawn, NVIDIA**

| Agents, layout | Before, ms | `first` | `cell` | `cell+` | `agent` (today) | `agent+` |
|---|---:|---:|---:|---:|---:|---:|
| 4,096, spread | 0.279 / 0.418 | +4.4% / +3.3% | +5.9% / +6.1% | +1.5% / +1.1% | +5.9% / +6.2% | +2.9% / +1.0% |
| 4,096, flocked | 0.745 / 0.694 | −0.5% / 0.0% | +4.4% / +3.1% | −1.1% / −2.2% | +11.5% / +5.3% | +6.0% / 0.0% |
| 65,536, spread | 13.959 / 13.775 | +0.5% / +0.4% | −0.9% / −0.7% | +0.2% / +0.3% | −0.7% / −0.8% | +0.2% / +0.3% |
| 65,536, flocked | 24.052 / 25.023 | +0.8% / +0.6% | +7.4% / +3.0% | +7.1% / +2.9% | **+26.3% / +20.5%** | +26.3% / +20.8% |

**Node/Dawn, AMD iGPU**

| Agents, layout | Before, ms | `first` | `cell` | `cell+` | `agent` (today) | `agent+` |
|---|---:|---:|---:|---:|---:|---:|
| 4,096, spread | 0.614 / 1.028 | +8.0% / +4.4% | +13.3% / +11.8% | +2.0% / −3.4% | +13.3% / +11.9% | +2.0% / −3.6% |
| 4,096, flocked | 1.892 / 1.714 | −0.2% / +1.5% | +9.7% / +9.1% | −5.2% / −6.2% | +11.5% / +10.2% | −3.5% / −5.1% |
| 65,536, spread | 59.671 / 64.905 | +5.0% / −2.6% | −1.3% / −6.3% | +2.1% / −3.1% | −0.7% / −6.7% | +8.2% / −5.2% |
| 65,536, flocked | 121.373 / 126.441 | +1.8% / +1.6% | +3.5% / −0.7% | +6.4% / +2.7% | **+21.2% / +19.4%** | +21.9% / +18.3% |

**Chrome, AMD iGPU**

| Agents, layout | Before, ms | `first` | `cell` | `cell+` | `agent` (today) | `agent+` |
|---|---:|---:|---:|---:|---:|---:|
| 4,096, spread | 0.607 / 0.970 | +8.0% / +5.2% | +8.8% / +6.7% | −1.0% / −10.2% | +8.8% / +6.6% | +0.1% / −10.0% |
| 4,096, flocked | 1.785 / 1.603 | +0.2% / +2.2% | +4.2% / +4.0% | −12.4% / −13.7% | +6.5% / +5.1% | −9.7% / −12.5% |
| 65,536, spread | 57.407 / 58.592 | +1.8% / +1.5% | −0.1% / +0.7% | +2.3% / +1.2% | +0.8% / −0.6% | +1.9% / +2.7% |
| 65,536, flocked | 108.363 / 113.631 | −0.1% / −0.6% | +3.7% / −1.5% | +8.0% / +5.8% | **+20.3% / +15.2%** | +26.6% / +20.0% |

**What the numbers say:**
- **The sort alone (`first`):** between −3% and +8%. The build grows from 0.02 ms to 0.05–0.07 ms
  at 4,096 agents, and at 65,536 from 0.08 to 0.27 ms on NVIDIA and from 0.5 to 1.2–1.4 ms on AMD,
  as 5.7 measured.
- **The window per agent (`agent`)** costs nothing while no cell is crowded, and +15% to +26% at
  65,536 agents flocked on every host/GPU pair, whichever loop scans it.
  - **Why:** with the old query, every query of a crowded cell reads the same 64 entries, so the
    GPU fetches them once and serves the rest from its cache. With a window per agent, every query
    reads a different 64.
  - **The two queries alone,** 65,536 flocked, in ms: NVIDIA 23.9 (`first`), 25.5 (`cell`), 30.1
    (`agent`); Chrome 105.5, 109.4, 128.9.
- **The window per cell (`cell`, `cell+`)** gives every query of a cell the same 64 again, and a
  different 64 on the next tick: between −2% and +8% at 65,536 flocked.
- **Both windows change what agents see to the same degree.** At the flocked snapshots, 1,928
  (`cell`) and 1,780 (`agent`) of 4,096 neighbour lists differ from `first`; at 65,536 agents,
  34,323 and 33,891.
- **The loop that 5.6 wrote is slow at 4,096 agents,** crowded or not: `cell` is 10 to 18 points
  above `cell+` on the AMD iGPU and 4 to 6 on NVIDIA. With the tighter loop, a step on AMD at
  4,096 agents lands between +2% and −14% of the code before step 5. At 65,536 flocked the tighter
  loop is equal on NVIDIA and 3 to 7 points slower on AMD; it stays within +8%.
- **Every sorted chain repeats word for word:** records after 512 and 64 steps, on all three
  host/GPU pairs. The atomic `old` run did not repeat in either flocked case on any of them.
- **Noise:** the two identical chains differ by at most 0.3 points on NVIDIA, 1.5 in Chrome and
  3.7 under Node on AMD. Under Node on AMD at 65,536 spread, samples range from 59 to 81 ms, so
  nothing under about 8% is resolved there.

### Phase 5.7's table, rerun

The unchanged acceptance spike, on the code before step 5 and on today's, alternating twice in
one sitting. Node only. The copies differ from the spike in their three import paths only.

| Agents, layout | NVIDIA: 5.7 said | NVIDIA: today | AMD, Node: 5.7 said | AMD, Node: today |
|---|---:|---:|---:|---:|
| 4,096, spread | +2.9% | −9% to +14% | +7.4% | +3.8% to +7.0% |
| 4,096, flocked | **+40.7%** | +9.0% to +9.5% | +11.7% | +11.1% to +11.8% |
| 65,536, spread | +1.2% | +0.2% to +0.9% | **+51.6%** | −4.1% to −3.6% |
| 65,536, flocked | +28.4% | +25.7% to +27.4% | +28.3% | +23.3% to +37.3% |

- **+40.7% and +51.6% did not reproduce.** NVIDIA at 4,096 spread cannot resolve 10% at all: its
  two baseline runs gave 0.450 and 0.552 ms.
- **Why the spike misleads as a before/after:** it times as many steps as fit in 40 ms. A slower
  build fits fewer, so it is timed on an earlier stretch of a run whose step cost drifts. Spread
  rises from 0.30 to 0.50 ms over 512 steps on NVIDIA as prey gather; flocked falls as herds
  loosen. The spike flatters a slower build where the cost rises and penalises it where it falls.
- **Its divergence half stands:** 0 differing words on today's code in both rounds, and
  15,447–19,308 differing record words on the code before step 5.
- **Not rerun in Chrome.**

### A pile-up

Not an acceptance layout: 65,536 agents in four knots 12 m across, which is where a rule without
wall avoidance ends up. 16 cells are over budget and the largest holds 4,770 agents. Change
against `old`, a few steps / 64 steps:

| Chain | NVIDIA (before: 3.9 ms) | AMD, Node (before: 11.9 ms) | AMD, Chrome (before: 10.3 ms) |
|---|---:|---:|---:|
| `first` | −6.7% / −4.7% | −11.6% / −4.8% | −0.3% / −0.6% |
| `cell` | +19.2% / +10.9% | +40.5% / +29.7% | +42.6% / +27.5% |
| `cell+` | +13.3% / +4.8% | +24.8% / +14.8% | +27.4% / +15.2% |
| `agent` (today) | **8.5× / 7.6×** | **12.3× / 11.0×** | **13.0× / 11.5×** |

- **The window per agent** turns a 4–12 ms step into 33–147 ms. A window per cell with the tighter
  loop adds 0.5–3 ms.
- **Why `cell` costs more than `first` here,** although both share their entries: the cheap answer
  is a wrong one. Wolves hold the lowest indices, so they are 6% of these cells and all of each
  cell's first 64 entries. Under `first` no agent sees a prey at all (a CPU scan of the spawn), and
  the prey query has nothing to rank. A window lands among the prey and does the work.
- **Checks:** 88/88 under Node, 44/44 in Chrome, zero GPU errors.

### The earlier attribution spike

A spike by Cline, `test/spikes/grid-query/`, asked whether a change that keeps today's results
exactly could recover the cost. Its corrected run completed on NVIDIA. Under Node on AMD it
stopped after 6 of its 10 layouts, and it never ran in Chrome. **It was not completed:** the cause
above does not depend on its missing rows. It was never committed, and the user had it deleted on
2026-10-13: it transformed the per-agent query, which production no longer has. What it found:
- **Hoisting the window's hash; 128-lane workgroups:** nothing beyond sample variation.
- **Two loops, one for ordinary cells and one for crowded cells:** from −45% to +39% by case.
- **Rejecting a candidate that cannot beat the kth result:** −3% to −28% on a query with k = 3 at
  4,096 agents on both GPUs, and nothing at 65,536. A lead for faster queries, which is a separate
  investigation (§1, non-goals).
- **None of them changes the crowded-cell cost.** On NVIDIA, 65,536 agents in four clusters cost
  33.3 ms a step with today's window, against 4.5 ms before.

### Also found

- **The CPU `pointCell` is not the GPU's.** It divides in f64 and the shader in f32, so the two
  bin one agent of 65,536 (at x = −24.0000076) in different cells. The GPU's sorted grid equals an
  f32 mirror exactly. Only tests that compare a GPU grid with `buildPointGrid` are affected; reruns
  on a device are not. Not fixed; recorded in [`work/gotchas.md`](./work/gotchas.md).
- **A window is a run of consecutive indices, and species sit in index ranges.** In a very crowded
  cell a window therefore holds almost one species. In the pile-up's spawn, scanned on the CPU at
  tick 3:
  - under `agent`, 26% of agents see a wolf;
  - under `cell`, none does: a cell's queries share one window, and it fell among the prey. By the
    arithmetic it falls among the wolves about one tick in thirteen;
  - under `first`, every agent sees wolves and none sees prey.

  In the acceptance layouts it does not show: 99–100% of agents see a wolf under every policy. A
  comb (every ⌊count ÷ 64⌋-th entry, shifted each tick) would sample a crowded cell evenly. **Not
  built and not measured.**

### Options for decision 2

Which entries a crowded cell yields. All three keep the sorted grid and bit-exact reruns.

| | Option | Worst acceptance case, against the code before step 5 | The pile-up | What agents see in a crowded cell |
|---|---|---|---|---|
| **A** | Keep today's window per querying agent | **+26%** at 65,536 flocked; +15% to +26% on every host/GPU pair (+27% with the tighter loop) | 8× to 13× | each agent its own 64, new every tick |
| **B** | A window per cell and tick, with the tighter loop | **+8%** at 65,536 flocked in Chrome; on AMD at 4,096 agents, +2% to −14% | +5% to +27% | every agent the same 64 of that cell, new every tick |
| **C** | No window: the cell's first entries | **+8%** at 4,096 spread on AMD, which is the sort itself | −12% to 0% | always the 64 lowest indices: the same agents every tick, one species first |

**Decided by the user, 2026-10-13: B.** It was the recommendation:
- **It is the smallest change that meets the 10% rule and keeps what decision 2 was for:** no
  agent is hidden for good by its index.
- **What it gives up against A:** within one tick, the agents that scan a crowded cell share its
  window. In the acceptance layouts that does not change who is seen.
- **What it changes:** the window's key in the query and in its CPU mirror (`gridWindow(c, tick)`
  for `gridWindow(i, tick)`), the scan loop, U23's window checks, and this document's §3 and
  decision 2. A1 and A23 follow the mirror. Then the three host/GPU pairs, the bundle and this
  spike again.
- **C instead removes** the tick binding, `AgentGrid`'s counter and the mirror's `tick` parameter.

**Not measured:** the acceptance spike in Chrome today; the earlier attribution spike's missing
rows; more than one seed, or more than one snapshot per case; a run longer than 64 steps at
65,536 agents; the comb.

## Phase 5.7 results, after decision 2's revision — 2026-10-13: within the rule

**What changed in `src/`:**
- `agentGridShaders.ts`: the window's key is the cell, and the scan is the tighter loop (§3);
- `agentGridMath.ts`: the CPU mirror takes the same key;
- `agentGridWindow.ts`: the hash's parameter is renamed to `cell`. The hash itself is unchanged.

**Tests:**
- **Headless:** U23's window check now requires a cell's observers to share its window, and two
  cells to move apart; U2/U3's crowded check and U16/U23's shader check follow. All three were
  watched failing on the per-agent code first. Agents: reference + 63 pass; TypeScript is clean.
- **GPU:** 137/137 checks, zero GPU errors, on Node/Dawn AMD, Node/Dawn NVIDIA and Chrome AMD.
  - **Watched failing:** with only the shader put back to a per-agent key, 8 checks fail (A1
    crowded on its three masks, the wrapper's oracle, A1 `AgentGrid` at ticks 2 to 5).
  - **A23** is byte-identical, and its atomic-slot mutation still diverges (20,307 / 72,752 /
    3,170 record / observation / action words on NVIDIA).
- **The window's hash keyed by cell** is even over ticks and over cells, and neither two cells nor
  one cell on consecutive ticks start together more often than chance (CPU check, 20,000 ticks).

**Cost, like for like.** The same spike, rebased: its chains now come from bundles of `b7050b7`
and `da262ad`, and `now` is today's production step, untouched. Change against the code before
step 5, a few steps / the longer run:

| Agents, layout | Node/Dawn, NVIDIA | Node/Dawn, AMD iGPU | Chrome, AMD iGPU |
|---|---:|---:|---:|
| 4,096, spread | +2.9% / +1.3% | +2.0% / −3.2% | −3.8% / −10.1% |
| 4,096, flocked | −2.2% / −3.6% | −5.8% / −7.6% | −12.1% / −15.4% |
| 65,536, spread | +0.5% / +0.4% | +5.0% / −0.2% | +2.1% / −0.4% |
| 65,536, flocked | +6.4% / +3.4% | +6.1% / +1.2% | **+9.6%** / +1.7% |

- **Within the ~10% rule in all twelve cases, by both timings.** The worst is +9.6%: Chrome at
  65,536 flocked, a few steps. The same kernel read +8.0% there in the follow-up's run.
- **Checks:** 352/352 under Node and 176/176 in Chrome, zero GPU errors. They include that the
  production query equals `cell+` word for word, and that the production run repeats word for word.
  A headless check holds the two shader texts equal but for the name of the hash's parameter, so
  the follow-up's `cell+` figures are measurements of the query that ships.
- **Noise:** under Node on AMD at 65,536 spread, samples range from 60 to 89 ms. `now` reads +5.0%
  there and `cell+`, the same kernel, −0.4%.

**The unchanged acceptance spike** on the new code:
- **Divergence:** 0 differing words (records, observations, actions, reasons) on all three host/GPU
  pairs, with overflow in every one of the 200 steps.
- **Cost, for the record only:** this half cannot judge a before/after (see the follow-up).

| Agents, layout | Node NVIDIA: step / grid, ms | Node AMD: step / grid, ms | Chrome AMD: step / grid, ms |
|---|---:|---:|---:|
| 4,096, spread | 0.452 / 0.043 | 0.685 / 0.068 | 0.778 / 0.075 |
| 4,096, flocked | 0.708 / 0.035 | 1.763 / 0.069 | 1.535 / 0.075 |
| 65,536, spread | 14.254 / 0.266 | 74.580 / 1.340 | 69.266 / 1.436 |
| 65,536, flocked | 25.887 / 0.266 | 135.332 / 1.363 | 123.322 / 1.387 |

**Bundle** (`test/spikes/mlp-think-cost/size.mjs`, the settings of 6.3):

| Import set | After 5.6 | Now | Against the code before step 5 |
|---|---:|---:|---:|
| Whole agents entry | 19,305 B | 19,323 B | +1,145 B |
| Rule path | 13,566 B | 13,586 B | +845 B |
| Rule path + MlpBrain | 15,495 B | 15,517 B | +860 B |

The whole entry is **2,172 B** below the 21,495 B ceiling. Rule-only imports still exclude every
MLP marker.

**Left open:**
- **The loop at 65,536 agents on AMD.** The tighter loop gains 10 to 18 points at 4,096 agents and
  costs 3 to 7 at 65,536 flocked against 5.6's loop. A third form, with the offset read once and
  the position still computed from the loop counter, was not tried.
- **The pile-up** was not rerun on the final code. `cell+`, the same kernel, measured +5% to +27%
  there.
- **5.8** is not started.
