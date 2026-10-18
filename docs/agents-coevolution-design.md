# NullGraph Agents — Phase 2 step 14: co-evolution in Herd & Wolves (`agents:coevolve`) — Technical Design

**Parent:** [`work/plan-2.md`](./work/plan-2.md) §4 (the first question) and §5 (milestone 2C, step 14).
**Status:** decided 2026-10-17 (§15): the user took every recommendation. Phase 14.0 (spike S7,
`npm run spike:s7`) measured on all three host/GPU pairs (results in §16): A = 64, P = 256, W = 16,
K = 32, E = 1, T = 600 set from the data. Next: phase 14.1, the arena scenario.
**Builds on:**
- `MlpBrain.byGenome` (step 11, [`agents-genome-rows-design.md`](./agents-genome-rows-design.md));
- the training tools ([`agents-cloning-design.md`](./agents-cloning-design.md)): the scenario, the
  evaluator, the model files, the ONNX writer;
- the films' step guard (`tools/film/guard.mjs`).

**Target:** `tools/agents/` only, never shipped in the package. **No library change** (plan-2 §3).

---

## 1. Purpose

Train the prey and the wolves of Herd & Wolves by competition, with evolution strategies (ES), in
many independent arenas at once, on the user's GPU, headless under Node. What comes out:
- **checkpoints** (`.mlp.json`, `.onnx` on request) that any demo, film (step 15) or evaluation
  loads;
- **run logs** that answer plan-2 §4.2's hypotheses H1–H3.

### Pre-existing facts this design rests on

1. **A network per agent exists** (step 11). One `MlpBrain.byGenome` brain per species holds every
   candidate as a row. When a 64-agent block shares a row, the row is staged in workgroup memory: at
   16,384 agents a step costs −10% on NVIDIA, −38% in Chrome and +31% under Node on AMD, against
   today's brains (phase 11.3).
2. **Every piece of an arena is a hook** (plan-2 §3), so the library does not change:
   - custom actions: the steering integrator, with the arena's walls;
   - a custom sensor: the arena's walls, in `Sensors.bounds()`'s format;
   - step stages: today's catches; respawn inside the arena; one catch counter per arena;
   - the record's free words: `genome` names the row, `user1` the arena.
3. **The tools already have most of what a trainer needs:**
   - the scenario (`createHerd`, the rule's texts, catches and respawns);
   - the evaluator (held-out seeds 21–25, the matchups T, C, S, Sp and Sw);
   - model files (`modelText`, `networkFromModel`) and the ONNX writer (`onnxFromNetwork`);
   - parameter vectors and a seeded start (`flatParams`, `networkOf`, `initNetwork`), and Adam's
     constants (`TRAIN_DEFAULTS`);
   - the films' step guard (`predictStep`, `calibrateSteps`, a 1 s limit per submission).
4. **Reruns are exact on one device** (A7, A23). With the noise seeded, a whole run is reproducible
   (I5).

### Non-goals

- **Library code.** Everything lives in `tools/agents/`.
- **Gradient reinforcement learning** (PPO and the like): ES only, as plan-2 decided.
- **Evolution in the browser.** Step 16 decides from S7's generation rate.
- **Contacts, obstacles, stamina.** S8 and steps 12–13 come later; H4 waits for them.
- **The ecosystem** (births, deaths, mutation on the GPU): plan-2's backlog.
- **Films of the results:** step 15.

### Principles

- **The library rule** (plan-2 §3): the optimizer sees weight vectors and scores, nothing of Herd &
  Wolves; the scenario's code lives in its own file.
- **Research hygiene:**
  - fitness and metrics are fixed before a run;
  - held-out seeds never train;
  - every run records its config, seeds, git commit, adapter and driver.
- **Exact and reproducible:** integer counters, seeded noise, a fixed order everywhere.
- **Measured, not assumed:** S7 sets the arena layout, the population and the episode length.
- **Within the GPU guard:** no submission near 2 s; steps per submission come from a measured step.

---

## 2. The formal definition

**Symbols.**
- **Arenas** a = 0 … A−1. Arena a is a square tile of side L at (xₐ, zₐ), on a grid of C columns.
  Tiles are g apart, and g is more than the sense radius r (8 m), so no query reaches across.
- **Bodies.** W wolves and P prey per arena; P is a multiple of 64, so prey blocks share a row. The
  wolves come first, as in today's scenario: arena a's wolves are agents [aW, (a+1)W), and its prey
  are [AW + aP, AW + (a+1)P).
- **Policies.** θ for prey and φ for wolves, as unpadded parameter vectors (`flatParams`).
- **A generation t** evolves one species X (§7). Its candidates are xₜ ± σ εₜ,ₖ for k = 0 … K−1, where
  εₜ,ₖ is a standard Gaussian vector drawn from (seed, t, k).
- **Arenas of a pair.** Candidate (k, +) plays in arena 2k and (k, −) in arena 2k+1. With E = 1
  arena per candidate by default (§16; configurable), the pair takes 2E arenas (A = 64, K = 32,
  P = 256, W = 16, T = 600).
- **The opponent.** The frozen species Y plays one policy oₖ per pair, the same in both of its arenas
  (I3): its current weights, or a member of its archive (§7).
- **An episode.** Spawn sₜ,ₖ: positions and headings relative to the tile, drawn from (seed, t, k),
  and the same in both arenas of a pair. Then T steps. Cₐ counts the catches in arena a; a caught
  prey respawns inside its tile.
- **Fitness.** A candidate scores −Cₐ when prey evolve and +Cₐ when wolves evolve: zero-sum, exact
  integers.
- **The update** (Salimans et al. 2017):
  - u are the centered ranks of the 2K scores, in [−0.5, 0.5], ties broken by arena index;
  - g = Σₖ (u⁺ₖ − u⁻ₖ) εₜ,ₖ / (2Kσ);
  - xₜ₊₁ = Adam(xₜ, g − λxₜ), which climbs.
- **The archive.** Every H generations, each species' current weights join its archive.

**Properties, each one a test (§10):**

| | Property | Means |
|---|---|---|
| I1 | Independence | No agent sees, touches or is counted with an agent of another arena (plan-2 §4.1) |
| I2 | Equivalence | With one arena covering the world, the arena scenario's records equal `createHerd`'s, bit for bit |
| I3 | Fair pairs | Both arenas of a pair have the same spawn and the same opponent |
| I5 | Reproducible runs | The same seed, config, build and device give the same checkpoints, byte for byte |
| I6 | No readback in the step loop | Scores come back once per episode, A × 4 B |
| E1 | Ascent | On a known objective (a quadratic in parameter space), the optimizer's loss falls |
| E2 | Neutral pairs | When every candidate scores the same, g is exactly 0 |
| E3 | Rank invariance | A monotone change of the scores leaves the update unchanged, bit for bit |
| E4 | Seeded noise | (seed, t, k) gives the same ε on every run and every host |
| E5 | Counted once | The per-arena counters add up to the scenario's own catch count |

(I4, exact rows, is step 11's A35.)

---

## 3. Integration facts (verified against this repository, 2026-10-17)

| Fact | Where | Consequence |
|---|---|---|
| `AgentInit` takes `genome` and `user1` | `agentRecords.ts:15–40` | The row and the arena go in the record; no new field |
| A custom sensor is `fn name(i: u32, a: Agent) -> array<f32, width>` in the sense module | `agentSenseShaders.ts:58–70` | The arena's wall sensor reads `a.ids.w` (the arena) and the heading |
| The built-in wall sensor: nearest of four walls, `max(0, 1 − distance / senseRadius)`, projected on side and forward | `agentSenseShaders.ts:44–56` | Copied with the tile's bounds; I2 checks the copy |
| Custom actions replace the integrator: `fn name(i: u32)` over `agents` (read-write) | `agentActShaders.ts` | The steering integrator copied with the tile's bounds; I2 checks it |
| A step stage binds at most 3 buffers of its own on a default device; today's respawn binds 3 (settings, stats, the catch ring) | `AgentSystem.checkExtraBuffers`, `herd-wolves.mjs` | The arena's respawn binds settings, stats and the arena counters; it drops the catch ring, which only the demo's flashes read |
| Today's respawn picks a point anywhere in the world | `herdRespawnWGSL` (`randomPointInWorld`) | The arena's respawn picks a point in the tile |
| Wolves are the first 1 in 16 agents; thinkEvery 3; a caught wolf rests 5 s | `herd-wolves.mjs` | Kept, per arena |
| The committed networks check observation names on load | `networkFromModel` | They are loaded against `herdDefinition()`, then used in the arena system, whose shapes match |
| The evaluator runs held-out seeds 21–25 and the matchups T, C, S, Sp, Sw | `evaluator.mjs`, `evaluate.mjs` | A checkpoint folder of `<species>.mlp.json` evaluates as students, unchanged |
| Parameter vectors, seeded starts, Adam's constants | `mlp-train.mjs` | Reused by the optimizer |
| A step guard: 1 s per submission, predicted from measured steps | `tools/film/guard.mjs` | Reused for the episode loop |

---

## 4. Data model

- **Constants compiled into the arena's WGSL:** A, L, g, C, W, P.
- **Record words:** `genome` is the row; `user1` is the arena.
- **Caller buffers:**
  - the arena catch counters (A × u32, atomics), zeroed at each episode's start;
  - the scenario's settings and stats, as today.
- **Brains:** one `MlpBrain.byGenome` per species, with a fixed table of 2K + M rows (M, the archive's
  cap, §7), set at the run's start and never rebuilt.
  - When a species evolves, rows 0 … 2K−1 hold its candidates.
  - When it is frozen, row 2K holds its current weights and rows 2K+1 … its archive.
  - Rows are rewritten with one `setRows` per species per generation.
- **Block alignment:**
  - Prey: P is a multiple of 64, so every prey block reads one row (the staged path).
  - Wolves: W per arena (e.g. 16) puts 64/W arenas in a block. When wolves are frozen, those
    arenas meet one opponent, so the block shares a row. When wolves evolve, a block mixes 64/W rows
    and the fallback runs; the wolf pass is small (1 agent in 17).
- **Files** (`tools/agents/out/coevo/<name>/`, ignored by git):
  - `run.json`: the config, the seeds, the git commit, the adapter and driver, the starting
    networks' hashes;
  - `log.jsonl`: one line per generation (§8);
  - `gen-NNNN/prey.mlp.json` and `wolf.mlp.json` (and `.onnx` with `--onnx`), written with
    `modelText`. Their `trainedOn` names the tool, the run, the generation and the config's hash, so
    `agents:evaluate -- --students gen-NNNN` reads them as it reads the clones;
  - `archive/prey-NNNN.mlp.json` and `wolf-NNNN.mlp.json`.

---

## 5. The arena scenario (`tools/agents/scenario/herd-wolves-arenas.mjs`)

- **The definition:** today's species, `thinkEvery` and sensors, with one change: `Sensors.custom`
  (2 floats, the arena's walls) in place of `Sensors.bounds()`. The networks' inputs keep their
  order and meaning.
- **The actions:** `Actions.custom` (2 floats): today's steering integrator, its walls the tile's.
- **The step stages:**
  - **catches:** `HERD_CATCHES` unchanged; neighbours never cross tiles (I1);
  - **respawn:** today's, with a point inside the tile and `atomicAdd(&arenaCatches[a.ids.w], 1u)`
    beside the scenario's own count (E5).
- **An episode's start:**
  - every agent is spawned on the CPU: its arena's tile, the pair's seed, its row and its arena;
  - the counters are zeroed;
  - one upload.
- **Equivalence (I2):** with A = 1 and the tile equal to the world, the arena scenario must give
  `createHerd`'s records, bit for bit. It also proves the copied sensor and integrator match.

---

## 6. The optimizer (`tools/agents/es.mjs`)

Pure functions over vectors; it knows nothing of the scenario.
- **`noise(seed, t, k, n)`:** a standard Gaussian vector (Box–Muller over `rand01`'s hash). Every host
  computes the same numbers (E4).
- **`candidates(x, σ, K, t, seed)`:** the 2K perturbed vectors, in pair order.
- **`centeredRanks(scores)`:** ranks scaled to [−0.5, 0.5], ties broken by index (E3).
- **`gradient(ranks, ε, σ)`:** Σₖ (u⁺ₖ − u⁻ₖ) εₖ / (2Kσ), exactly 0 when every score is equal (E2).
- **`adam(state, g, rate, λ)`:** `TRAIN_DEFAULTS`' β₁, β₂ and ε, with weight decay λ.

Its tests (E1–E4) run headless, on known objectives.

---

## 7. The schedule

- **Alternating** (recommended, §15 Q1): generation t evolves the prey when t is even and the wolves
  when it is odd. The frozen species keeps its weights for that generation. This is textbook ES:
  each score depends on one perturbation.
- **The opponents of the frozen species:**
  - each pair meets its current weights with probability 1 − h, or else a member of its archive,
    drawn uniformly;
  - the draw is seeded by (seed, t, k), and both arenas of a pair meet the same opponent (I3);
  - arenas that share a wolf block share it too.
- **The archive** (recommended, Q3): each species' current weights join every H = 5 generations, up
  to M = 32 members; past the cap, the oldest after the first goes. The first member, the starting
  network, stays. h = 0.5 once the archive holds anyone.
- **The start:** the clones (C2's shape, for H1) or `initNetwork` of a smaller shape (for H2).

---

## 8. Runs, logs and the guard (`tools/agents/coevolve.mjs`, `npm run agents:coevolve`)

```
npm run agents:coevolve -- --name h1-clones --start clones --generations 200 --arenas 64 --pairs 32 \
    --episode 600 --sigma 0.02 --rate 0.01 --decay 0.005 --archive-every 5 --archive-share 0.5 --seed 1
npm run agents:coevolve -- --name h2-small --start random --hidden 16x16 …
```

- **One generation:**
  1. write both tables' rows;
  2. spawn every arena and zero the counters;
  3. run T steps in submissions of s steps, s from the guard;
  4. read the counters (A × 4 B);
  5. compute the scores, the ranks and the update;
  6. log, and write a checkpoint when one is due.
- **The guard:** `calibrateSteps` measures a step at the run's population, and s keeps a submission
  under 1 s. A step over 1.5 s stops the run (`checkStep`). Costs are cached per adapter, driver and
  workload, as the films cache them.
- **One `log.jsonl` line per generation:** t, the species evolved, the scores' mean, median, best and
  worst, the catches in all arenas, the seconds per generation, the steps a second.
- **Checkpoints:** every N = 10 generations and at the end (Q4); the archive as it fills.
- **Reproducible (I5):** the GPU's steps are exact, the noise is seeded, and the optimizer runs in
  float64 in a fixed order. Two runs give the same files, byte for byte.

---

## 9. Evaluation (plan-2 §4.2–4.3)

- **H1 and H2: evolved brains against the rule and the clones.** `agents:evaluate -- --students
  gen-NNNN --episodes 20 --seed 21` runs the matchups T, S, Sp and Sw on held-out seeds. Plan-2's
  protocol asks for at least 20 episodes; the seeds stay none of training's. It reports means with
  95% intervals.
- **H3: cycling.** A new `agents:crossplay -- --run <dir> --every 10` plays the prey of generation i
  against the wolves of generation j, on held-out seeds, for chosen generations. It writes the
  catches as a matrix (the data of Cliff & Miller's CIAO plots), for a run with the archive and one
  without.
- **Where results go:** each research run's numbers are recorded in this design's results sections.
  Charts come later, through the films (step 15) or the demo (step 16).

---

## 10. Test suite

IDs continue from U45 and A36; U43, U44 and A35 are step 11's.

**Headless (`test/agents-coevo.test.mjs`):**
- **E1–E4:** the optimizer on known objectives (a quadratic falls; neutral scores give g = 0; a
  monotone change of scores leaves the update unchanged; the same seeds give the same noise);
- **I3:** the schedule gives both arenas of a pair the same spawn and opponent, and alternates the
  species;
- **the layout:** tiles never closer than the sense radius; prey blocks aligned; the CLI refuses a
  layout that breaks either;
- **the files:** a checkpoint written with `modelText` reads back with `networkFromModel`, bit for
  bit.

**GPU (agent-tools suite, three host/GPU pairs):**
- **I1:** over 600 steps, in every layout used, no neighbour index points into another arena;
- **I2:** one arena covering the world equals `createHerd`, bit for bit;
- **E5:** the arena counters add up to the scenario's count;
- **I5:** a 3-generation run twice gives the same checkpoints, byte for byte;
- **I6:** one readback per episode (instrumented).

**Watched failing** for each: the tiles closer than the sense radius (I1); the original
`randomPointInWorld` in respawn (I1, E5); the copied integrator with one wall dropped (I2); a pair
given two seeds (I3).

---

## 11. Errors (the CLI's refusals, verbatim when built)

- `coevolve: prey per arena (<P>) must be a multiple of 64.`
- `coevolve: the gap between arenas (<g> m) must be more than the sense radius (<r> m).`
- `coevolve: <A> arenas cannot hold <K> pairs of <E> arenas each.`
- `coevolve: --start clones needs the committed networks' shape; got --hidden <shape>.`

---

## 12. Engine touchpoints

- **Core:** none.
- **Library:** none. The trainer uses step 11's `MlpBrain.byGenome`, custom actions, custom sensors,
  step stages and caller buffers, all shipped.

---

## 13. Build order

| Phase | Delivers | Exit |
|---|---|---|
| **14.0** ✅ | Spike S7 (§16): one generation end to end, its rate, its noise, a 20-generation run, exploits | Results recorded (§16); the *S7* numbers set; questions answered (uncommitted) |
| **14.1** | The arena scenario (§5); I1, I2, E5 on the GPU | Green on three host/GPU pairs, watched failing |
| **14.2** | The optimizer (§6); E1–E4 headless | Green, watched failing |
| **14.3** | The trainer (§7, §8): the schedule, the archive, the guard, logs, checkpoints; I3, I5, I6 | Green; a 10-generation run on both Node GPUs |
| **14.4** | Evaluation (§9): checkpoints in `agents:evaluate`, `agents:crossplay`; the tools' README | A run evaluated on 20 held-out seeds |
| **14.5** | The first research runs: H1 from the clones, H2 from random weights, H3 with the archive on and off | Results and verdicts recorded here |

### Deferred

- Simultaneous co-evolution (a 2 × 2 design for mirrored pairs on both sides).
- Resuming a run from a checkpoint.
- Shaping terms (distance, closing speed), as experiments.
- More than one arena per candidate, if S7 finds the scores too noisy at E = 1.
- Evolution in the browser (step 16, from S7's rate).

---

## 14. Where this refines plan-2

| Plan-2 | This design |
|---|---|
| §4.1: prey fitness is "the share of the episode each prey survives"; wolves, "catches per wolf" | Both from one exact count per arena, zero-sum: prey −Cₐ, wolves +Cₐ. With respawns the two orderings agree, and the counters stay integers |
| §4.1: arenas are tiles further apart than the sense radius | The arena is stored in `user1`; respawn stays in the tile; the walls are the tile's |
| §5: step 14 writes `.mlp.json` and `.onnx` | `.onnx` on request (`--onnx`); `.mlp.json` always, in the clones' format |

---

## 15. Questions for the user

**Decided 2026-10-17.** The user: "i choose all recommended option for this one and u can run s7".
Q1 alternating every generation; Q2 catches per arena, zero-sum, no shaping; Q3 the archive every 5
generations, half the frozen side's pairs, 32 members, the start kept; Q4 checkpoints every 10
generations and the last, `.onnx` on request; Q5 24 → 16 → 16 → 2 for H2.

The questions as they were asked:

1. **Q1. The schedule:** alternate the evolving species every generation (recommended); in blocks of
   several generations; or both at once (deferred: it needs a 2 × 2 design).
2. **Q2. Fitness:** catches per arena, zero-sum (recommended), with no shaping by default.
3. **Q3. The archive:** every 5 generations, half of the frozen side's pairs meet archived opponents,
   capped at 32 members, the start always kept (recommended). Revisited after the first runs.
4. **Q4. Checkpoints:** every 10 generations and the last; `.onnx` only on request (recommended).
5. **Q5. H2's network:** 24 → 16 → 16 → 2 tanh, about 700 parameters (recommended), or phase 7.2's
   24 → 32 → 32 → 2 (about 2,000). S6 found small networks cost little per agent.

The population, the arena size and the episode length come from S7 (§16).

---

## 16. Phase 14.0 — spike S7: one generation, end to end

**Scripts only** (`test/spikes/coevo/`, `npm run spike:s7`), Node on both GPUs, and Chrome on the
AMD iGPU for the rate.

**What it builds, as spike code:** the arena scenario (§5), episodes, the per-arena counters, the
optimizer (§6) and the alternating schedule (§7), with step 11's `MlpBrain.byGenome` as the brains.

**What it measures:**

| | Question | How |
|---|---|---|
| 1 | Generations a minute | A = 16, 32, 64 arenas of 256 prey + 16 wolves; T = 300, 600, 1,200 steps; on the three host/GPU pairs |
| 2 | How noisy a score is | The same candidate, 20 seeds, per T: the spread of Cₐ against the gap between candidates. Decides T and E |
| 3 | Does it learn | 20 generations from the clones, prey and wolves alternating: the frozen side's catches against the rule's, on 5 held-out seeds, before and after |
| 4 | Which exploits appear without physics | Prey packed against walls or corners (the share within 1 m of a wall), herds piled onto one point (the nearest-neighbour distance), wolves parked; with images of the arenas |
| 5 | I1 and I2 hold | As in §10, as spike checks |

Before any timing, the spike prints its estimate from phase 11.3's steps, with C2's networks at
17,408 agents (64 arenas) and thinkEvery 3: about 40 generations a minute on NVIDIA, 20 under Node
on AMD, and 22 in Chrome, at T = 600.

**What it decides:**
- A, P, W, K, E and T (§2);
- whether evolution in the browser is worth building (step 16);
- whether contacts and obstacles matter to the research, or only to the look (S8, steps 12–13).

---

## Spike S7 results — 2026-10-17

**Adapters measured** (nothing else on the GPU):
- **Node/Dawn, NVIDIA Turing** (`D3D12 driver version 32.0.15.8183`);
- **Node/Dawn, AMD GCN-5 iGPU** (`D3D12 driver version 31.0.12002.1002`);
- **Chrome, AMD GCN-5 iGPU** (`npm run drive -- --serve . "test/spikes/coevo/index.html" --wait-for "S7 done" --timeout 2400`).

### 1. Invariant checks (`I1`, `I2`, `E5`)

All 4 checks passed on every adapter (8/8 under Node, 4/4 in Chrome, zero WebGPU errors):
- **I2 (equivalence):** 1 arena of 4,096 agents (256 wolves, 3,840 prey) covering the world on the
  rule, stepped 300 times in lockstep with `createHerd`: **0 of 4,096 records differ**, and catches
  agree (`272` and `272`) on all three pairs.
- **I1 (independence):** 64 arenas (`17,408` agents, tile side `L = 41.9 m`, gap `10 m > 8 m` sense
  radius) on the clones over 600 steps (neighbour buffer read every 50 steps): **0 of 761,991**
  sampled neighbour entries on NVIDIA and **0 of 761,498** on AMD (Node and Chrome) point to an agent
  in another arena.
- **E5 (counted once):** the 64 per-arena counters sum to `stats[0]` (`1,980 = 1,980` on NVIDIA;
  `1,977 = 1,977` on AMD Node and Chrome).
- **Cross-host reproducibility on one GPU:** on the AMD GCN-5 iGPU, Node/Dawn and Chrome produced
  **bit-identical** catch counts, noise statistics, 20-generation trajectories, and held-out
  evaluations.

### 2. Generations a minute (`rate`)

Arenas of 256 prey + 16 wolves (`272` agents/arena), `thinkEvery = 3`, C2's `24 → 64 → 64 → 64 → 2`
networks on `MlpBrain.byGenome` (one full generation: Box–Muller noise, `setRows` on both species,
CPU spawn + `upload`, `T` GPU steps, counter readback, and CPU Adam update):

| Host / GPU | Arenas (`agents`) | `T = 300` | `T = 600` | `T = 1,200` |
|---|---|---|---|---|
| **NVIDIA, Node** | 16 (`4,352`) | 0.33 s (**180.8/min**) | 0.66 s (**91.6/min**) | 1.36 s (**44.1/min**) |
| **NVIDIA, Node** | 32 (`8,704`) | 0.43 s (**140.9/min**) | 0.80 s (**74.8/min**) | 1.67 s (**35.9/min**) |
| **NVIDIA, Node** | 64 (`17,408`) | 0.51 s (**116.6/min**) | **1.06 s (56.4/min)** | 2.10 s (**28.5/min**) |
| **AMD, Node** | 16 (`4,352`) | 0.47 s (**128.4/min**) | 1.04 s (**57.5/min**) | 2.12 s (**28.3/min**) |
| **AMD, Node** | 32 (`8,704`) | 0.64 s (**93.8/min**) | 1.45 s (**41.5/min**) | 3.03 s (**19.8/min**) |
| **AMD, Node** | 64 (`17,408`) | 1.31 s (**45.9/min**) | **2.72 s (22.1/min)** | 5.58 s (**10.7/min**) |
| **AMD, Chrome** | 16 (`4,352`) | 0.57 s (**105.0/min**) | 1.21 s (**49.7/min**) | 2.45 s (**24.5/min**) |
| **AMD, Chrome** | 32 (`8,704`) | 0.78 s (**77.0/min**) | 1.61 s (**37.2/min**) | 3.34 s (**18.0/min**) |
| **AMD, Chrome** | 64 (`17,408`) | 1.31 s (**45.9/min**) | **2.70 s (22.2/min)** | 5.51 s (**10.9/min**) |

- **Against the pre-run estimate** at 64 arenas and `T = 600` (`~40/min` on NVIDIA, `~20/min` under
  Node on AMD, `~22/min` in Chrome):
  - **NVIDIA beat the estimate by 41%** (`56.4/min`, and `0.94–1.09 s` per generation during the
    20-generation run): partitioning `17,408` agents across 64 disjoint tiles keeps spatial grid cells
    lighter than a single crowded world.
  - **AMD matched the estimate** on both Node (`22.1/min`, `2.52–2.70 s`) and Chrome (`22.2/min`,
    `2.36–2.71 s`). With step 11's staged cooperative kernel, Chrome is as fast as Node at 64 arenas.

### 3. Score noise (`noise`)

64 arenas, all running the unperturbed clones on both sides, each arena on its own independent spawn:

| Episode length `T` | Mean catches / arena | Spawn SD (`sd`) | Coefficient of variation (`cv`) | Candidate `scoreSd` in `learn` |
|---|---|---|---|---|
| **300** (5 s) | 17.0 | 1.34–1.35 | 7.9–8.0% | — |
| **600** (10 s) | 30.5–30.6 | 2.25–2.27 | **7.4%** | **2.11–2.82** |
| **1,200** (20 s) | 50.5–50.6 | 3.19–3.56 | 6.3–7.0% | — |

- Across 64 arenas at `T = 600`, the standard error of the 64-arena mean is `2.26 / √64 = 0.28`
  catches/arena (`95% CI ± 0.6`).
- Within each mirrored pair `(2k, 2k+1)`, both candidates share the same spawn (`mix(seed, t, k)`)
  and the same opponent (`I3`), so spawn-to-spawn variance cancels inside `(u⁺ₖ − u⁻ₖ)`. `T = 600`
  and `E = 1` (32 pairs in 64 arenas) give ~30 catches per arena in 1.0 s on NVIDIA (2.7 s on AMD).

### 4. 20-generation run from the clones and exploit check (`learn`)

20 generations from the clones (`A = 64, K = 32, T = 600, σ = 0.02, rate = 0.01, decay = 0.005`,
prey on even generations, wolves on odd), then evaluated on **64 held-out spawns** (`mix(900001, a)`):

| Matchup (64 held-out spawns, `T = 600`) | Catches / arena ± 95% CI (NVIDIA) | Catches / arena ± 95% CI (AMD Node & Chrome) | Prey `< 1 m` from wall | Nearest prey (m) | Prey speed (m/s) | Wolf speed (m/s) |
|---|---|---|---|---|---|---|
| **clone prey · clone wolves** | 30.3 ± 0.6 | 30.2 ± 0.6 | 0.0% | 0.71–0.72 | 2.73 | 0.68–0.70 |
| **evolved prey · clone wolves** | **29.8 ± 0.7** | **29.6 ± 0.7** | 0.0% | **0.66** / 0.72 | 2.69 / 2.45 | **0.87** |
| **clone prey · evolved wolves** | 30.3 ± 0.6 | **30.8 ± 0.6** | 0.0–0.1% | 0.72–0.73 | 2.74 / 2.73 | 0.71 / 0.64 |
| **evolved prey · evolved wolves** | 30.3 ± 0.6 | 30.1 ± 0.6 | 0.1% | 0.67 / 0.72 | 2.68 / 2.51 | 0.79 |
| **rule · rule** | 33.0 ± 0.5 | 32.8 ± 0.6 | **4.4–4.6%** | **1.28** | 2.69 / 2.51 | 0.14–0.15 |

### What it shows and decides

1. **The *S7* parameters are set (§2, §8):**
   - `A = 64` arenas, `P = 256` prey + `W = 16` wolves per arena (`17,408` agents, tile side
     `L = 41.9 m`, gap `g = 10 m`), `K = 32` mirrored pairs (`E = 1` arena per candidate),
     `T = 600` steps.
   - At those settings, a 200-generation research run (H1–H3) takes **~3.5 minutes on NVIDIA** and
     **~9 minutes on AMD**.
2. **No measurable learning yet in 20 generations; H1 needs the longer runs** (Claude's review,
   2026-10-18, corrected an earlier "directional movement appears"):
   - after 10 prey and 10 wolf updates, every held-out change lies inside the noise: evolved prey
     against clone wolves `29.8` vs `30.3` (NVIDIA) and `29.6` vs `30.2` (AMD), evolved wolves against
     clone prey `30.3` vs `30.3` (NVIDIA) and `30.8` vs `30.2` (AMD), each `± 0.6–0.7` (95%);
   - one candidate explanation, not measured: σ = 0.02 is small beside the clones' weights (up to
     about 178, A26), so 20 updates move the policy little. Phase 14.5 runs 200 generations and may
     try a larger σ; H1 is judged there.
3. **Exploits without physics — contacts (S8, step 13) matter to both the research and the look:**
   - **No wall-hugging exploit:** `0.0–0.1%` of clone or evolved prey sit within `1 m` of a tile
     wall (less than the rule's `4.4–4.6%`).
   - **Herd-stacking / body-overlap exploit:** under `rule · rule` (which includes random wander),
     the mean distance to the nearest fellow prey is **`1.28 m`**, well above `2 × radius = 0.80 m`.
     Under the clones (which have no random wander), prey pack to **`0.71–0.72 m`** — already
     overlapping inside `0.80 m` — and on NVIDIA `evolved prey · clone wolves` packs even tighter to
     **`0.66 m`**. Without collision resolution, stacking multiple prey on top of one another lets a
     flock share a single evasion trajectory while a catching wolf's 5 s rest timer freezes it after
     catching only one body from the stack. (A hypothesis, not measured: S7 measured only the distances.)
4. **Live evolution in the browser (step 16):**
   - Chrome on the AMD iGPU runs at **`22.2 generations/min`** at full scale (`A = 64, T = 600`) and
     **`105.0 generations/min`** (`0.57 s/gen`) at `A = 16, T = 300`, bit-identical to Node/Dawn on
     the same GPU. Step 16's checkpoint scrubber and arena viewer remain first; a live worker toggle
     at `A = 16` or `64` is fast enough to offer if desired.

