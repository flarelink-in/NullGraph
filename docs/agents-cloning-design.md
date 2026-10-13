# NullGraph Agents — C2 step 7, part 2: record → train → export — Technical Design

**Parent:** [`agents-design.md`](./agents-design.md) §9.5 (the training tools) and §13, C2 step 7.
**Builds on:**
- [`agents-species-brains-design.md`](./agents-species-brains-design.md): one brain per species;
- [`agents-mlp-design.md`](./agents-mlp-design.md): `MlpBrain`, its network format and limits;
- [`agents-ordered-grid-design.md`](./agents-ordered-grid-design.md): bit-exact lockstep reruns.

**Asked by the user, 2026-10-13:** "start step 7 part 2 design", after step 5's deterministic
neighbour lookups were completed for it.

**Decided with the user, 2026-10-13** (§14; [`work/plan.md`](./work/plan.md) §5, item 7):
1. **The network copies the rule's decision without its random wander and its index coin-flips.**
2. **No observation for the wolf's rest timer:** a resting wolf is frozen, and its rows are dropped.
3. **The teacher lives in this repository,** under `tools/agents/`.
4. **`.mlp.json` is the working format,** with the two trained Herd & Wolves files committed;
   `.onnx` is written in the last phase, 7.10, and stays in this part.
5. **The bars are the parent's A8,** read as in §8; if they are missed, the numbers go back to the
   user.
6. **The scene is the demo's defaults only.**

**Status:** ✅ design decided 2026-10-13 (`f1244cb`). ✅ 7.6 committed (`c2608d7`). ✅ 7.7 committed
(`0d3fb9f`). ✅ 7.8 committed (`8afc6fa`). ✅ 7.9 committed (`cd76a25`). ✅ 7.10 committed
(the commit after `cd76a25`; results at the end):
- **7.7:** both species meet the 0.05 bar at round 0;
- **7.8:** the students' catches are within 1% of the rule without wander (the bar is 10%), and
  they do not drift, so no round of correction was needed;
- **7.9:** the two networks are in the repository with their tests, and the guide shows how to load
  one;
- **7.10:** each network is written as an `.onnx` file, and ONNX Runtime gives `mlpForward`'s
  actions within 4.5e-6.

**Part 2 is built.** Next is step 8, `MlpBrain.fromOnnx`.

**Target:** `tools/agents/` (new, in the repository, not in the package), tests, one guide
section. **`src/` unchanged; no bundle bytes.**

---

## 1. Purpose

**Make the first network that has learned Herd & Wolves.** Three tools:
1. **record** what the rule brain decides, in a headless lockstep run;
2. **train** one small network per species to give the same decisions;
3. **export** each network as a file that today's `MlpBrain` runs.

Then **measure the copy**, twice: decision by decision, and in the running simulation.

**Words used here:**
- **teacher:** the rule brain that is copied;
- **student:** a network trained to copy it;
- **label:** the action the teacher gives for one observation;
- **driver:** whichever brain is moving the agents during a run.

### The single most important pre-existing fact

**The teacher is a GPU rule, so it can label any observation, for free.** A human demonstrator
can only show what they did. This teacher can also be asked what it *would* do in a situation the
student got itself into. The recorder is built around that: labels come from a separate pass, so
they do not depend on who is driving (§6).

### The second fact: the teacher's action is not a function of what the student sees

Herd & Wolves' rule reads three things beside the observations (fact 1):
- **the wolf's rest timer,** from its record;
- **a wander value,** a hash of the agent's index and the tick, constant for 32 steps;
- **two coin-flips by index parity,** for exact ties and for a wall dead ahead.

A network sees observations only. Trained on the teacher's raw actions, it would learn the average
over what it cannot see, and its error could not go below the size of the hidden part. For prey
that is not fleeing, the wander term alone is 0.3 × a uniform value in [−1, 1). Before any clamp,
that is a mean absolute error of 0.15 on the turn: three times the parent's target of 0.05 (A8).

### Non-goals

- **Learning by trial and reward** (reinforcement learning), and any training in the browser or on
  the GPU. The parent's non-goals stand.
- **A student that remembers.** The networks are feed-forward: observation in, action out.
- **Reading ONNX** (`MlpBrain.fromOnnx`, step 8), **ONNX Runtime brains** (step 9), and **the live
  Rules / Network switch** (step 10).
- **Other scenes.** The tools take a scenario module, but only Herd & Wolves is delivered and
  measured.
- **One network for every setting.** The students are trained at the demo's defaults (§5). How
  they behave when the speed or rest sliders move is not promised.
- **A new library API.** `src/` is not touched (§12).

### Design principles applied

- **Verify against the tree** ([`design-principles.md`](./design-principles.md) §1): §3's table.
- **A CPU mirror is a second implementation, and it drifts** (§13.1): labels are computed by the
  teacher's own WGSL on the GPU, never by a JavaScript port of the rule (§5, rejected options).
- **Tests pin numbers, and must be seen to fail** (§13): §10.
- **Measure, then claim:** the copy's quality is a measured number with a stated bar (§8), and the
  numbers go back to the user when the bar is missed (§13, phase 7.8).

## 2. Architecture

```
            ┌───────────────────── one lockstep step (Node + Dawn) ─────────────────────┐
 scenario   │ grid → queries → sense → think (driver) → LABEL PASS → act → app → tick   │
 (teacher)  └───────────────│──────────────────────────────│──────────────────────────────┘
                            ▼ obs                          ▼ labels (the rule, minus what
                                                             the student cannot see)
 record.mjs ──────────► dataset: rows of (obs, label, driver's act) + header ──┐
                                                                               ▼
 train.mjs  ◄──────────────────────────────────────────────────────────── per species
     │  plain JS: Adam, mean-squared error, seeded
     ▼
 prey.mlp.json, wolf.mlp.json  ──►  evaluate.mjs: the students drive; the label pass still
     │                               labels every observation they meet
     ▼                                        │
 MlpBrain per species (today's API)           └──► more rows for the next round, if needed
```

- **The driver and the label are separate.** In the first run the teacher drives. In later runs
  the students drive. The label pass is the same in both.
- **Everything runs headless** under Node with Dawn, in lockstep, as the parent planned (§9.5).

## 3. Integration facts (verified against this repository, 2026-10-13)

| # | Fact | Where | Consequence |
|---|---|---|---|
| 1 | The demo's rule reads the wolf's rest timer (`agents[i].state.w`), a wander hash of `(i, tick >> 5)` and index parity (`i % 2`), beside `obs`, a settings buffer and prey's `maxSpeed` | the test-engine site, `HerdWolvesGPU/brain.ts`, `HERD_BRAIN` | The teacher's action is not a function of the observation. The label must leave those inputs out (§5) |
| 2 | A rule's `think` may read every agent record, the tick and its own group-1 buffers | `agentThinkShaders.ts:26–55`; `AgentSystem.ts:208–225` | Hidden inputs are legal for a rule. Every teacher needs fact 1's audit |
| 3 | A frozen agent's action is never applied: act zeroes its velocity and returns | `agentActShaders.ts:69–74` | The demo freezes a resting wolf. Its action needs no copying (§5) |
| 4 | Sense and think share one gate, `thinksNow`: alive, and newborn or `(tick + i / 64) % thinkEvery == 0` | `agentSenseShaders.ts:10–12`; `agentThinkShaders.ts:47` | An agent's `obs` row and its action row always come from the same step |
| 5 | `obs`, `act`, `reason`, `state` and the records are `COPY_SRC` | `AgentSystem.ts:96–101` | A tool reads them back with its own copies |
| 6 | The think wrapper is exported: `buildAgentThinkWGSL(def, wgsl, thinkEvery)`. A think pass writes only its own agents' rows, behind a species mask | `agentThinkShaders.ts:26, 36, 48`; `agents/index.ts` | A tool can run a second think pass that writes into its own buffers: the label pass (§6) |
| 7 | A step is grid → queries → sense → think passes → act → app stages → tick | `AgentSystem.ts:150–155` | The label pass belongs after the last think and before act |
| 8 | `await agents.step()` uploads, submits one step and waits | `AgentSystem.ts:381–387` | Lockstep recording, as the parent's §9.5 asks |
| 9 | Lockstep reruns on one device are bit-exact, with or without overflow | ordered-grid design, closure; A7, A23 | A recording is reproducible on its device, byte for byte |
| 10 | `MlpBrain` takes 1–4 dense layers: input ≤ 128, hidden ≤ 64, output ≤ 16. `mlpForward` is its f32 CPU reference, and the GPU agrees with it within 1e-5 + 1e-5·\|cpu\| (A4) | `mlpMath.ts:29–39, 110`; mlp design | The trainer's result is judged by `mlpForward`: what is measured is what runs |
| 11 | Each species can have its own brain, of its own shape: one gated think pass per brain | `AgentSystem.ts:179–205`; A22 | One network per species needs nothing new |
| 12 | `rand01(i, tick, stream)` has a bit-exact CPU twin | `RuleBrain.ts:32` | Seeds, sampling and initial weights need no new generator |
| 13 | The repository already encodes ONNX by hand, and runs ONNX Runtime under Node | `test/onnx-gpu/tiny-model.mjs:42`; `test/onnx-gpu/run.js` | An exported `.onnx` can be checked against `mlpForward` before step 8's reader exists |
| 14 | The package ships `dist` only | `package.json:11–13` | Tools and trained files live in the repository. No bundle bytes |
| 15 | Herd & Wolves observes 24 floats (self 2, nearest wolf 5, three nearest prey 15, bounds 2) and acts with 2 (turn, throttle) | the demo's `defineAgents`; `agentLayout.ts:70–73` | 24 → … → 2 networks: the shape S3 and phase 6.3 timed |
| 16 | The demo's rules exist only in the test-engine site | search of this repository | The teacher must come into this repository to run headless (§5) |

## 4. Data model

### 4.1 A dataset

One recording is a folder: `header.json` and, per species, two flat little-endian files.

| File | Row | Contents |
|---|---|---|
| `<species>.f32` | `obsDim + 2 × actDim` floats (28 for Herd & Wolves) | the observation, the label, the driver's action |
| `<species>.u32` | 4 words | episode, tick, agent index, flags: bit 0 frozen, bits 4–7 the teacher's reason code |

**`header.json` holds what makes the files reproducible:**
- the format version; the scenario's name, and a hash of each WGSL text (driver, label, app stages);
- the definition: capacity, world, cell size, scan budget, `dt`, `thinkEvery`, species, and
  `obsNames` and `actNames` in order;
- the scenario's settings (speeds, rest seconds, flocking and fleeing switches);
- the episodes: seed, steps, and who drove each;
- the sampling: read-back stride and keep rate per species;
- the row count and SHA-256 of every file;
- where it was made: the library's commit (and whether the tree was dirty), Node's version, the
  GPU adapter.

**Not committed:** datasets are tens of megabytes. They go to `tools/agents/out/`, which is ignored.

### 4.2 A trained network

`<species>.mlp.json`, one per species:

```jsonc
{
  "format": "null-graph/mlp", "version": 1,
  "species": "prey",
  "obsNames": ["self.speed", "self.energy", "wolf[0].side", "…"],   // the definition's, in order
  "actNames": ["turn", "throttle"],
  "layers": [{ "input": 24, "output": 32, "activation": "tanh", "weights": [/* row-major */], "biases": [] }],
  "trainedOn": { "dataset": "<sha-256 of header.json>", "teacher": "<hash of the label WGSL>",
                 "settings": {}, "seed": 1, "rounds": 1 },
  "measured": { "heldOutMae": { "turn": 0, "throttle": 0 }, "rows": 0 }
}
```

- **`layers` is `MlpNetwork`** with arrays for the two `Float32Array`s. Numbers are written with 9
  significant digits, which is enough to read every f32 back exactly.
- **Loading it needs no new API:** turn the arrays into `Float32Array`s, check `obsNames` and
  `actNames` against the definition, then `new MlpBrain(engine, layers)` and
  `agents.setBrain(brain, { species })`.
- **Committed:** the two Herd & Wolves files and their reports, in
  `tools/agents/models/herd-wolves/` (§14, decision 4). The draft estimated about 25 KB each, for
  24→32→32→2. At the shape 7.7 needed they are about 180 KB each, and 476 KB with their reports and
  the evaluation.

## 5. The teacher and the label

### The teacher comes into the repository

`tools/agents/scenario/herd-wolves.mjs` holds what the demo holds today: the definition, the
rule, the two app stages (catches, respawn), the spawn and the settings. It is the demo's code,
with one change of shape: **the rule's three hidden inputs and its reason word go through four
small named functions.**

```wgsl
fn wander(i: u32, tick: u32) -> f32 { … }    // the hash, as today
fn coin(i: u32) -> bool { return (i % 2u) == 0u; }
fn resting(i: u32) -> bool { return agents[i].state.w > 0.0; }
fn tag(i: u32, tick: u32, why: u32) -> u32 { return why; }
```

**The driver's text** uses them as written, and decides exactly what `HERD_BRAIN` decides. Phase
7.6 checks that once against the site's file: the same actions, word for word, over 600 steps.

### The label is the teacher without what the student cannot see

**The label's text** is the same rule with the four functions replaced:

| Function | In the label's text | Why |
|---|---|---|
| `wander` | returns 0 | The student has no source of chance. The label is the rule's decision without its wander |
| `coin` | returns `true` | One fixed side for exact ties and for a wall dead ahead, instead of a side by index |
| `resting` | returns `false` | A resting wolf is frozen, so its action is never applied (fact 3). The label says what it would do otherwise, and the trainer drops frozen rows |
| `tag` | returns `(tick << 8) \| (frozen << 4) \| why` | The row carries the step it was written in, so the recorder knows which rows are fresh without re-deriving the think gate on the CPU |

**The label is then a function of the species and the observation only,** at fixed settings. That
is something a network can learn, and an error against it measures the network and nothing else.

**What this gives up:** the student does not wander, and it always picks the same side on a tie.
Phase 7.8 measures what that changes in the running simulation (§8).

**The settings are fixed at the demo's defaults:** 4,096 agents, one wolf in 16, `thinkEvery` 3,
scan budget 64, prey speed 3.2, wolf speed 4.2, rest 5 s, flocking and fleeing on.

### Rejected

- **Labels from the driver's own `act`,** as the parent's §9.5 sketched. The hidden inputs put a
  floor under the error that no network can pass (§1). The driver's action is still recorded
  beside the label, so phase 7.6 can state that floor as a measured number.
- **Labels from a JavaScript port of the rule.** It is a second implementation of 60 lines of
  branching WGSL, and nothing would hold the two together (design-principles §13.1).
- **The rest timer as an observation** ([`agents-mlp-design.md`](./agents-mlp-design.md) §5.1
  noted it for this part). It would change the demo's definition to teach the network an action
  that is never applied. Dropping frozen rows costs nothing (§14, decision 2).
- **The wander value as an observation.** It would let the student wander like the teacher. It
  needs a sensor that reads the tick, which a custom sensor has no contract for today, or a rule
  that wanders by the agent's age instead. Kept as the fallback if 7.8 shows that wander matters
  (§14, decision 1).

## 6. Recording

### The label pass

The recorder builds one more think stage with the library's own wrapper (fact 6):

- **Its code** is `buildAgentThinkWGSL(definition, labelText, thinkEvery)`: the same gate, the same
  accessors, the same tick as the system's think.
- **It reads** the system's records, observations, state and species table, and the scenario's
  settings.
- **It writes** into two buffers the recorder owns: the labels, and their tag words.
- **It runs** after the system's last think pass and before act (fact 7). The recorder splices it
  into the step's stage array at the stage named `AgentSystem act`, and refuses by name if that
  stage is not found (§11).

Because the gate is shared (fact 4), **an agent's observation row and its label row are written in
the same step, for the same agents.** The tag's tick says which step.

**Tried before this draft,** with a scratch script outside the repository: 4,096 agents under
Node/Dawn on the AMD iGPU, 300 lockstep steps with respawns and resting wolves, and a stand-in
rule that has the demo's three kinds of hidden input behind §5's four functions.
- **Given the driver's text, the label pass wrote the system's own actions, word for word,** in
  all 300 steps; no validation error.
- **Spliced after act instead, it differed** in 223 of 300 steps (4,531 rows): the check A25 needs.
- **Given the label's text,** every row the system's think wrote in a step carried that step's
  tag, and the driver's turn differed from the label's by 0.11 on average. That is the stand-in
  rule's hidden part, more than twice the 0.05 bar; 7.6 measures the real rule's.

### What a recording run does

For each episode (a seed):
1. spawn as the demo spawns, with that seed;
2. step in lockstep (fact 8). Every `stride` steps, copy out the observations, the labels, the
   tags and the driver's actions;
3. keep the rows whose tag was written since the last copy, so each agent's decision is kept once;
4. keep a share of them by a hash of the agent, the tick and the species (fact 12), so prey do
   not outnumber wolves fifteen to one in the files.

**Defaults:** 5 episodes of 1,800 steps (30 s); a copy every 6 steps; every wolf row and one prey
row in 16. That is about 75,000 rows per species and episode, and about 45 MB per species in all.
**The last episode is held out:** training never sees it.

**Reproducible:** on one device, the same library build, scenario and seeds give the same bytes
(fact 9). The header's hashes say so, and A24 checks it.

**Who drives is a parameter.** In round 0 the teacher drives both species. Later rounds can put a
student on either species (§8). The label pass does not change.

## 7. Training

`tools/agents/mlp-train.mjs` is the parent's "150-line dense-network trainer in plain JS": no
install, no native code.

- **The network:** `MlpNetwork`'s shapes. Start at 24 → 32 → 32 → 2 with `tanh` throughout: the
  shape phases S3 and 6.3 timed, and actions already lie in [−1, 1]. Each species has its own
  shape; 7.7 tries smaller and larger ones within `MlpBrain`'s limits (fact 10).
- **The loss:** mean-squared error against the label. **The optimizer:** Adam, mini-batches of
  256, a fixed number of passes over the rows, with the step size decaying.
- **Seeded:** initial weights and the shuffle come from `rand01` (fact 12). One thread, in order:
  the same rows and seed give byte-identical weights on one Node version (U11).
- **Rows used:** one species' rows, minus frozen ones, minus the held-out episode.
- **The result is stored as f32 and judged as f32.** After training, every reported error is
  computed with the library's `mlpForward` on the rounded weights (fact 10), including its clamp of
  `tanh`'s input to ±15.
- **The best pass wins:** the trainer keeps the weights with the lowest held-out error, not the
  last ones.

**Cost, estimated, to be measured in 7.7:** about 1,900 weights; 300,000 rows × 40 passes is a
few minutes of plain JS on one core.

**As built in 7.7:**
- **Training is in f64; the result is f32.** The forward and backward passes run in f64, with
  `MlpBrain`'s activations exactly: `tanh` with its input clamped to ±15 (and a zero slope
  outside), and the stable sigmoid. The weights are returned as `Float32Array`s and judged with
  `mlpForward`.
- **Initial weights** are uniform in ±√(6 / (in + out)), or ±√(6 / in) before a `relu`, from
  `rand01`; biases start at zero.
- **The defaults are what 7.7 measured to meet the bar:**
  - 24→64→64→64→2 with `tanh` throughout, the largest shape `MlpBrain` runs at 64-wide layers;
  - 200 passes;
  - a step size that decays exponentially from 1e-2 at the first pass to 1e-4 at the last.

  The draft's start, 24→32→32→2 at 40 passes from 3e-3, fell far short for prey (results below).
  **Adam's other constants** are β1 0.9, β2 0.999 and ε 1e-8.
- **The best pass is chosen on the held-out episode,** the same episode the error is reported on,
  so the report is slightly optimistic. That was measured: the best and last passes differ by at
  most 0.0003.
- **`train.mjs` writes `<species>.mlp.json` (§4.2) itself.** It reads each file back and refuses to
  finish if a weight changed bit for bit. Nothing that varies between runs (such as the time taken)
  goes in the file, so the same dataset and seed write the same bytes. Timing goes to
  `<species>.report.json`, one per species, so species trained in separate runs keep both.
- **`tools/agents/cli.mjs`** holds what the command lines share: the check that `dist/` is not
  older than `src/`, and the option readers. `record.mjs` now uses it too.

## 8. Judging the copy

### Decision by decision

**Held-out error:** the mean absolute difference between the student's action and the label, on
the episode training never saw. Reported per species, per action (turn, throttle) and per reason
code, so a weak branch of the rule shows.

**The bar is the parent's A8: 0.05 or less, per species and action.**

### In the running simulation

`tools/agents/evaluate.mjs` runs held-out seeds for 30 s each, in lockstep, under four drivers:

| Run | Prey | Wolves | Answers |
|---|---|---|---|
| **T** | teacher | teacher | the reference, with its spread across seeds |
| **C** | the rule without wander and coin-flips | the same | what the label, followed exactly, does |
| **S** | student | student | the copy |
| **Sp** | student | teacher | how good the prey's copy is alone |
| **Sw** | teacher | student | how good the wolves' copy is alone |

It reports:
- **catches in 30 s.** The parent's A8 says "prey survival"; in this demo a caught prey respawns,
  so the count of catches is that measure;
- **mean speed** per species, and **how tight the herds are** (the mean distance to the nearest
  prey, from the observations);
- **the student's error on its own run:** the label pass is still running, so every observation
  the student meets has a label. This is the error that counts, because a student drifts into
  situations the teacher never visited.

**The bars:**
- **S against C: mean catches over 5 held-out seeds within 10%** (A8's figure). T's own spread
  across seeds is reported beside it, because a 10% bar means little if seeds alone differ by more.
- **C against T is reported, not gated.** If the rule without wander behaves differently from the
  rule, that is a property of the label, not of the training, and it goes back to the user (§14,
  decision 1).

### Rounds of correction

If the student's error on its own run is well above its held-out error, the cause is the one
named above: it visits situations that are not in its data. The fix is known as dataset
aggregation:
1. let the students drive, and record the observations they meet, with the teacher's labels;
2. add those rows to the dataset;
3. train again from scratch, with the same seed.

The recorder needs nothing new for this: only the driver changes. **Whether any round is needed
is not known;** 7.8 measures it, and stops at 3 rounds.

**As built in 7.8:**
- **Two files, like the recorder.** `evaluator.mjs` has no Node import; `evaluate.mjs` is its
  command line, `npm run agents:evaluate`.
- **The seeds are 21–25.** No recording used them: training saw 11–14, and 15 chose the best pass.
- **Every decision is counted,** not one prey row in 16 as in a recording.
- **The measures come from the scenario** (`herdWolves.measures`): mean speed, the distance to the
  nearest prey, and the share of prey with no prey in sight, each over live agents.
- **Run C checks the tool itself.** It drives with the label's decisions, so a live agent's action
  must equal its label exactly. The command line exits 1 if it does not.
- **A missed bar is reported, and is not an error exit.**
- **`model.mjs` reads a `.mlp.json`** and checks its names (§9). It came forward from 7.9, because
  the evaluation loads the students.
- **No round of correction was built:** the students did not drift (results below), so the
  recorder still takes the teacher as its only driver.

## 9. Export and loading

- **`<species>.mlp.json`** (§4.2) is the working format, written by `train.mjs`. It loads into
  today's `MlpBrain` with six lines of app code.
- **The names are checked on load.** `obsNames` and `actNames` must equal the definition's, in
  order. Equal widths do not show that a feature means what the network expects (the guide says
  so already); equal names do.
- **`<species>.onnx`** is written last (phase 7.10): `Gemm` + activation per layer, extending
  `test/onnx-gpu/tiny-model.mjs`'s encoder, as the parent's §9.5 says. Step 8's reader does not
  exist yet, so the file is checked the other way (fact 13): ONNX Runtime runs it under Node, and
  its output must equal `mlpForward`'s.
- **A guide section,** "A trained network", shows the loading code as an executable example.

## 10. Test suite

**Headless** (`npm run test:agents`, new file `test/agents-tools.test.mjs`):

| # | Asserts |
|---|---|
| U11 | The trainer fits y = Ax + b (10 inputs) to a mean-squared error under 1e-6, and XOR with one hidden layer. The same seed gives byte-identical weights. Its gradients equal finite differences within 1e-4 relative; **watched failing** with one term of the backward pass dropped |
| U12 | The ONNX writer's bytes decode to the layers that were written: shapes, activations, weights bit for bit. (The reader half and "an unsupported op is refused by name" stay with step 8) |
| U24 | A dataset written and read back is byte-identical. One flipped byte fails its hash, and a short file is refused with both sizes named |
| U25 | The scenario's two rule texts parse inside the think wrapper. They differ in the four functions and nowhere else. A tag packs and unpacks tick, frozen and reason |
| U26 | A `.mlp.json` written and read back gives bit-identical f32 weights. A file whose `obsNames` differ from the definition is refused, naming the first difference |
| U27 | The row sampler is the same on every run, keeps the asked share within 1% over 100,000 rows, and never puts a held-out episode's row into training |
| U28 | An evaluation's counts (added in 7.8): a frozen agent is counted and compared with nothing; errors split by reason and pool over seeds by decision; the scenario's measures read one observation row; an unknown driver, or a student without its network, is refused by name |

**GPU** (a suite of its own, `agent-tools`, in `test/agent-tools-gpu/`; Node + Dawn and Chrome):

| # | Asserts |
|---|---|
| A24 | Two recordings of the same seeds are byte-identical, every file. A recording with another seed is not |
| A25 | The label pass, given the driver's text, writes the system's own actions, word for word, for exactly the agents that thought in that step, over 300 steps with respawns. **Watched failing** with the pass spliced after act, where a newborn's gate has already closed |
| A26 | The committed Herd & Wolves files load into one `MlpBrain` per species, and over 60 steps every action written in a step equals `mlpForward` of that step's observation within 1e-4. A4's tolerance is too tight for trained weights (7.9's results). **Watched failing** against the other species' network |
| A8 | Cloning quality, the parent's check, with its two bars stated in §8: error at most 0.05 per species and action; catches of S within 10% of C over 30 s on 5 seeds. As built, the error is the networks' on their own runs, which needs no dataset; the held-out error stored in each file is checked by U26. Run C must differ from its label by nothing |
| A27 | ONNX Runtime runs each exported `.onnx` on 256 observations, and its actions equal `mlpForward`'s within 1e-4 (the bound of the parent's A5). In the `onnx` suite. As built: on both hosts, not Node only, and on observations the suite takes from 120 steps of the simulation, because no dataset is committed |

## 11. Errors

| Condition | Message |
|---|---|
| The step has no stage named `AgentSystem act` | `agents-tools: the step has no "AgentSystem act" stage. These tools are tied to the library build they were written for; rebuild, or update tools/agents/label-pass.mjs.` |
| `dist/` is older than `src/` | `agents-tools: dist/ is older than src/. Run npm run build:lib first.` |
| A dataset file's size disagrees with its header | `dataset: <file> is <n> bytes; header.json promises <m> (<rows> rows × <stride> bytes).` |
| A file's hash disagrees | `dataset: <file> does not match header.json's SHA-256. It was changed or cut short.` |
| The dataset's layout is not the definition's | `dataset: obsNames[<k>] is "<a>"; the scenario's definition has "<b>".` |
| A species has no usable rows | `train: no rows for species "<name>" after dropping frozen rows and the held-out episode.` |
| A network file does not fit the definition | `model: <file> obsNames[<k>] is "<a>"; the definition has "<b>".` |
| A shape `MlpBrain` cannot run | `MlpBrain`'s own messages, unchanged |

## 12. Engine touchpoints

### Required: none

No change to core or to `src/extras/agents/`. Nothing is added to the package (fact 14), so the
C2 allowance keeps its 2,172 B for step 8.

### Optional, not proposed now: a public seam for a pass that only watches

The label pass relies on a stage's name to find its place in the step (§6). That is the
zero-change path, and its cost is honest: a library refactor that renames the stage breaks the
tools, loudly (§11), and A25 fails. A public seam, such as a way to add a think pass with its own
output buffers, would remove that coupling for an estimated 300 B. It is worth proposing only if a
second tool needs it.

## 13. Build order

Phases continue from part 1's 7.5.

| Phase | Builds | Exit |
|---|---|---|
| **7.6** ✅ | The scenario, the label pass, the dataset format, `record.mjs`; U24, U25, U27; A24, A25 | A round-0 recording that reproduces byte for byte. **Measured:** the difference between the driver's action and the label (what no network could learn), the teacher's catches in 30 s and their spread across seeds, the share of frozen rows, the mix of reason codes |
| **7.7** ✅ | `mlp-train.mjs`, `train.mjs`; U11 | Round-0 students. **Measured:** held-out error per species, action and reason code, for three shapes; training time. Built 2026-10-13 (`0d3fb9f`): both species within 0.05 at round 0; results below |
| **7.8** ✅ | `evaluator.mjs`, `evaluate.mjs`, `model.mjs`; U28, and U26's reader half | **Measured:** T, C, S and the mixed runs; the students' error on their own runs. Built 2026-10-13 (`8afc6fa`): within §8's bars at round 0, so no round of correction; results below |
| **7.9** ✅ | `.mlp.json` export and the committed files; U26; A26, A8; the guide's "A trained network"; `tools/agents/README.md`; closure | The tests pass on three host/GPU pairs; docs match what was measured; `docs/work` updated. Built 2026-10-13 (`cd76a25`); results below |
| **7.10** ✅ | `<species>.onnx` writer; U12; A27 | ONNX Runtime's output equals `mlpForward`'s. Stays in this part (§14, decision 4). Built 2026-10-13; results below |

Every test is watched failing before it passes.

### Deferred

- **Reading ONNX:** step 8. **ONNX Runtime brains:** step 9. **The live switch and its timings:**
  step 10, which also moves the demo onto the scenario file (§14, decision 3).
- **Students that hold across settings** (speed, rest, cadence): train over a range, later.
- **A student that wanders:** only if 7.8 shows the need.
- **Scenes other than Herd & Wolves,** and any learning beyond copying a rule.

## 14. Decisions (the user, 2026-10-13)

The user accepted the six recommendations of the draft, and kept the `.onnx` phase in this part.

1. **What the network copies:** the rule's decision without its random wander and its index
   coin-flips. A network can then match it, and the error measures the network. The copy does not
   wander; 7.8 measures what that changes.
   - **Not chosen:** the wander value as an extra observation. It stays the fallback if 7.8 shows
     that wander matters.
2. **The wolf's rest timer:** no new observation. A resting wolf is frozen, so its action is never
   applied; the trainer drops those rows, and the demo's definition stays as it is. This replaces
   the note in [`agents-mlp-design.md`](./agents-mlp-design.md) §5.1.
3. **Where the teacher lives:** `tools/agents/scenario/herd-wolves.mjs` in this repository holds
   the rules that are trained on. The site's demo keeps its own copy until step 10. Each trained
   file carries a hash of the teacher's text, so the demo can tell when its rules have moved on.
4. **Which files:**
   - `.mlp.json` now, which today's `MlpBrain` loads;
   - the two Herd & Wolves files and their report are committed (about 50 KB in the draft; 476 KB as
     built, see 7.9's results); datasets are not;
   - `.onnx` is the last phase, 7.10, checked by ONNX Runtime. It stays in this part and does not
     move to step 8.
5. **The bars:** the parent's A8, read as in §8: held-out error at most 0.05 per species and action
   against the label; catches within 10% of the rule without wander, over 30 s on 5 seeds. If 7.8
   misses them after 3 rounds and the largest allowed shape, the numbers go back to the user.
6. **The scene:** the demo's defaults only (4,096 agents, `thinkEvery` 3, budget 64, speeds 3.2 and
   4.2, rest 5 s, flocking and fleeing on). Other settings are deferred.

## Phase 7.6 results — 2026-10-13: the scenario, the label pass, the recorder

**Built.** No `src/` change; committed as `c2608d7`.
- `tools/agents/scenario/herd-wolves.mjs`: the demo's definition, rule, app stages, spawn and
  settings. The rule's hidden inputs go through `wander`, `coin`, `resting` and `tag` (§5).
- `tools/agents/label-pass.mjs`: the label pass, `lockstep()`, and a read-back of several buffers
  in one submission.
- `tools/agents/dataset.mjs`: rows, the sampler, the split, and the files with their sizes and
  hashes.
- `tools/agents/recorder.mjs`: what a recording run does. `tools/agents/record.mjs`: its command
  line, `npm run agents:record`.
- Tests: `test/agents-tools.test.mjs`, in `npm run test:agents`; a new GPU suite, `agent-tools`.

**Three things differ from the draft:**
- **The GPU checks have a suite of their own,** `agent-tools`. The library's suite keeps its 137
  checks, and the guides' counts stay true.
- **A tag stores its tick plus one.** A row no pass has written is the word 0; it now reads as
  tick −1 and is never taken for a decision made at tick 0.
- **The recorder is two files.** `recorder.mjs` has no Node import, so A24 runs it in Chrome too.

**The scenario is the demo's.** Checked once against the site's `HerdWolvesGPU/brain.ts` (SHA-256
`d294809f7a364d27…`), under Node/Dawn on the AMD iGPU:
- the two app stages are equal as text;
- with the site's `HERD_BRAIN` run as a label pass beside this repository's driver, **0 rows
  differed** in actions or reason words over 600 steps, for seeds 11 and 12 (440 and 427 catches).

The site is another repository, so this is a scratch script and not a suite test. It must be run
again when either side's rule changes.

**Tests:**
- **Headless:** U24, U25 and U27 pass; the agents suite is reference + 66.
- **GPU:** 10/10 checks, zero GPU errors, on Node/Dawn AMD, Node/Dawn NVIDIA and Chrome AMD.
  - **A25:** given the driver's text, the label pass differs in 0 rows over 300 steps, with 272
    catches and 62,187 frozen agent-steps on the way. Spliced after act, 4,132 rows differ in 30
    of 60 steps.
  - **A24:** two recordings of the same seeds are byte-identical; a recording of other seeds is not.
- **Watched failing:** 27 mutations of the tools, each seen failing a named check.
  - **One was not seen at first:** with a row's label stored in its action's columns, the dataset
    still read back as written. U24 now pins a row's layout, not only its round trip.

**The first recording,** with §6's defaults: 5 episodes of 1,800 steps, a copy every 6 steps, every
wolf row and one prey row in 16.

| | Node/Dawn, AMD iGPU | Node/Dawn, NVIDIA |
|---|---:|---:|
| Time to record | 21.7 s | 12.7 s |
| Prey rows | 360,453 | 360,465 |
| Wolf rows (of which frozen) | 384,000 (61.8%) | 384,000 (61.4%) |
| Driver against label, prey's turn | 0.082 | 0.082 |
| Driver against label, wolf's turn | 0.008 | 0.010 |
| Driver against label, throttle | 0 | 0 |
| Catches in 30 s, by seed | 1033, 1013, 1044, 1026, 1033 | 1011, 1023, 1018, 1004, 1045 |
| Catches, spread around the mean | −1.6% to +1.4% | −1.6% to +2.4% |

- **Reproducible:** two recordings on the AMD iGPU gave five identical files, `header.json`
  included. The AMD and NVIDIA files differ, as expected: no equality across devices is claimed.
- **What no network could learn:** 0.082 of the prey's turn is hidden input. A copy of the driver's
  raw actions could not reach the 0.05 bar. A copy of the label can, and the wolves' hidden part is
  small either way.
- **Why the label acted:** prey flee in 54% of their rows, flock in 41%, and turn from a wall in
  5%. Wolves that are not frozen chase in 92%, turn from a wall in 6%, and have nothing in sight
  in 2%.
- **Wolves rest most of the time:** 62% of the wolf rows are frozen, which leaves about 147,000
  usable wolf rows beside 360,000 prey rows. 7.7 shows whether that is enough.
- **The 10% bar means something:** seeds alone move the catches by about 2%.
- **Size:** 95 MB in all.

**Left for later phases:** `tools/agents/README.md` (7.9).

## Phase 7.7 results — 2026-10-13: the trainer, and round-0 students

**Built** (`0d3fb9f`; no `src/` change):
- `tools/agents/mlp-train.mjs`: the trainer (§7, "As built").
- `tools/agents/train.mjs`: its command line, `npm run agents:train`.
- `tools/agents/cli.mjs`: the checks and option readers both command lines share.
- `trainingSet` in `dataset.mjs`: one species' rows, read in place.

**U11** (two tests; the agents suite is now reference + 68):
- **Linear fit:** 10 → 3, fitted to a mean-squared error of 1.1e-12 (bar 1e-6), down from 0.76 at
  the first pass.
- **XOR**, with 4 hidden units: 0.009, 0.988, 0.991, 0.009.
- **Determinism:** the same rows and seed give the same weights, bit for bit; another seed does not.
  The command line's files repeat byte for byte too.
- **The gradient against central differences,** through all four activations at the deepest shape
  (5→6→4→3→2): worst relative error 3.9e-6 (bar 1e-4), all 87 gradients large enough to compare.
- **Best-pass selection:** the best pass is the one returned, pass 5 of 12 in the test, so not the
  last.
- **Errors by reason** add up to the whole.
- **Refusals by name:** a species with nothing to train on (§11); a shape the rows do not fit; a
  shape `MlpBrain` cannot run, with its own message.

**Watched failing:** 7 mutations, each seen failing U11 and restored byte for byte:
- the activation's slope dropped from the backward pass;
- the bias gradient dropped;
- an unseeded shuffle;
- the last pass returned instead of the best;
- training on the driver's action columns instead of the label;
- frozen rows kept;
- kept errors shifted by a row.

**One was missed at first:** with frozen rows kept, every check still passed. The test now asserts
that no frozen or held-out row trains.

**The dataset:** a fresh round-0 recording on the AMD iGPU at `cf3f972`, a clean tree, with §6's
defaults. It reproduced 7.6's numbers exactly: the same catches per seed, and 360,453 prey rows.
- **Prey:** 288,362 rows train, and 72,091 are held out (seed 15).
- **Wolves:** 117,652 live rows train, and 28,984 are held out.

**Round 0: three shapes, 40 passes each** (the default schedule, 3e-3 to 1e-4; tanh throughout).
Held-out mean absolute error against the label:

| Shape (weights) | Prey turn | Prey throttle | Wolf turn | Wolf throttle | Time, prey + wolves |
|---|---|---|---|---|---|
| 24→16→16→2 (706) | 0.214 | 0.012 | 0.094 | 0.002 | 47 + 19 s |
| 24→32→32→2 (1,922) | 0.177 | 0.006 | 0.060 | 0.002 | 107 + 41 s |
| 24→64→64→2 (5,890) | 0.135 | 0.005 | **0.048** | 0.001 | 283 + 111 s |

The three ran side by side on a Ryzen 5 5600H (6 cores, 12 threads), so the times are under load.

**By reason, at 24→64→64→2** (turn):
- **prey:** flee 0.121, flock 0.138, wall 0.251;
- **wolves:** chase 0.041, wander 0.076, wall 0.149.

**What the three shapes show:**
- **The wolves meet the 0.05 bar** at 64×64, and every throttle meets it at every shape.
- **The prey's turn does not:** 0.135 at the best shape, 2.7 times the bar.
- **It is underfitting, not overfitting.** Training error equals held-out error within 0.003, and
  the best pass was the last, or the one before it, in every run. Capacity and training help. More
  data, or 7.8's rounds of correction, which fix drift rather than underfitting, would not.
- **The error is spread, not only at the label's jumps.** This comes from a scratch diagnostic at
  64×64:
  - the median turn error is 0.091, and the worst 10% of rows hold 35% of the error;
  - rows near no jump still average 0.113;
  - one jump is everywhere: 67% of held-out prey rows have a neighbour within 2.4 m and within
    0.4 m of the centreline, where the separation term flips sign by up to 2.4.
- **Best against last:** at most 0.0003 apart, so choosing the pass on the held-out episode flatters
  the report by about that much.

**Longer and larger: prey only, run side by side** (64×64×64 is the largest shape `MlpBrain` runs
with 64-wide layers):

| Run | Shape | Passes | Step size | Prey turn | Prey throttle | On training rows (turn) | Time |
|---|---|---|---|---|---|---|---|
| E1 | 24→64→64→2 | 150 | 3e-3 → 1e-4 | 0.084 | 0.0009 | 0.082 | 19.6 min |
| E3 | 24→64→64→2 | 150 | 1e-2 → 1e-4 | 0.067 | 0.0004 | 0.065 | 19.8 min |
| E2 | 24→64→64→64→2 | 150 | 3e-3 → 1e-4 | 0.059 | 0.0007 | 0.055 | 33.1 min |
| **E4** | 24→64→64→64→2 | 200 | 1e-2 → 1e-4 | **0.033** | 0.0002 | 0.031 | 43.5 min |

- **Longer training roughly halves the prey's error:** at 64×64, 0.135 after 40 passes becomes
  0.067 after 150 with the larger first step.
- **Depth and the larger first step both help, and they add up:** E4 combines them and reaches
  0.033. It crossed 0.05 at pass 75, and its best pass was 199 of 200.
- **The training rows stay close:** 0.031 against 0.033 held out.

**The round-0 students** (E4's settings for both species; they are now the trainer's defaults,
§7):

| Species | Turn | Throttle | Turn median / 90th percentile | On training rows (turn) | Time |
|---|---|---|---|---|---|
| Prey | **0.033** | 0.0002 | 0.018 / 0.075 | 0.031 | 43.5 min |
| Wolves | **0.011** | 0.0002 | 0.003 / 0.024 | 0.007 | 17.9 min |

- **The bar, 0.05 per species and action, is met by both species,** for the turn and the throttle.
- **By reason, it is met everywhere except the wall:**
  - prey: flock 0.030, flee 0.030, wall 0.101;
  - wolves: chase 0.008, wander 0.013, wall 0.058.

  The wall branch is about 5% of rows for each species. The bar is per species and action, not per
  reason, so this is reported, not gated. 7.8 shows whether it matters in the running simulation.
- **Wolves now generalise a little less well** than they fit (0.007 on training rows, 0.011 held
  out). They are still four times inside the bar.
- **The students live in `tools/agents/out/e4-64x64x64-200-r01/`** (ignored). That folder has
  `prey.mlp.json` and `wolf.mlp.json`, with the prey's `report.json` from before reports went per
  species and `wolf.report.json`. `npm run agents:train` with the new defaults reproduces them from
  the same recording.

**What changed against the draft (§7):**
- **The defaults:** 24→64→64→64→2 at 200 passes from 1e-2, not 24→32→32→2 at 40 passes from
  3e-3.
- **The cost:** "a few minutes" was the draft's estimate. Meeting the bar takes about an hour of
  CPU for the two species.
- **One report per species** (`<species>.report.json`), so species trained in separate runs keep
  both.

**Next: 7.8.** `evaluate.mjs` will run T, C, S and the mixed runs, comparing catches and the
students' error on their own runs. Rounds of correction come only if the students drift.

## Phase 7.8 results — 2026-10-13: the copy in the running simulation

**Built** (`8afc6fa`; no `src/` change):
- `tools/agents/evaluator.mjs`: one run under a driver per species, and its counts.
- `tools/agents/evaluate.mjs`: its command line, `npm run agents:evaluate`.
- `tools/agents/model.mjs`: a `.mlp.json` read back, with its names checked.
- In the scenario: `rule(kind)` on a built simulation, and `measures`.

**Tests** (the agents suite is now reference + 70):
- **U28** and **U26's reader half** pass. U26's other half, a written file read back bit for bit,
  stays with 7.9.
- **Watched failing:** 14 mutations, each seen failing U28 or U26 and restored byte for byte.
- **The GPU suite is unchanged:** `agent-tools` 10/10, zero GPU errors, on Node/Dawn AMD, Node/Dawn
  NVIDIA and Chrome AMD. A8, the GPU check of these bars, comes with the committed files in 7.9.

**The runs:** the round-0 students of 7.7 (24→64→64→64→2), seeds 21–25, 1,800 steps each, at the
demo's defaults.

**Catches in 30 s:**

| Run | Prey / wolves | AMD iGPU: by seed | AMD: mean | Against C | NVIDIA: mean | Against C |
|---|---|---|---:|---:|---:|---:|
| T | teacher / teacher | 1013, 1013, 998, 1040, 1015 | 1015.8 | −0.6% | 1018.8 | −0.2% |
| C | the rule without wander | 1022, 1019, 1024, 1012, 1031 | 1021.6 | — | 1021.0 | — |
| **S** | student / student | 1061, 1015, 1016, 1016, 1010 | **1023.6** | **+0.2%** | **1015.0** | **−0.6%** |
| Sp | student / teacher | 1043, 1017, 1009, 982, 1023 | 1014.8 | −0.7% | 1011.4 | −0.9% |
| Sw | teacher / student | 1035, 1013, 1005, 1011, 1050 | 1022.8 | +0.1% | 1019.0 | −0.2% |

- **The bar is met: S is within 10% of C,** at +0.2% on the AMD iGPU and −0.6% on the NVIDIA GPU.
- **Every run is inside the spread of seeds.** One run's seeds differ by −3.2% to +3.7% around
  their mean, and the five runs' means differ by under 1%.
- **C against T (reported, not gated): the teacher is 0.6% below C on the AMD iGPU and 0.2% below
  on the NVIDIA GPU, so no difference is seen.** The rule without wander and coin-flips catches as
  many prey as the rule.

**The driver against the label, on the driver's own run** (mean absolute difference; AMD iGPU,
with the NVIDIA GPU within 0.001 of each):

| Run | Prey turn | Prey throttle | Wolf turn | Wolf throttle |
|---|---:|---:|---:|---:|
| T (what no network could learn) | 0.084 | 0 | 0.009 | 0 |
| C (must be zero) | 0 | 0 | 0 | 0 |
| **S** | **0.034** | 0.0003 | **0.011** | 0.0002 |
| Held out, from 7.7 | 0.033 | 0.0002 | 0.011 | 0.0002 |

- **The students do not drift.** Their error on their own runs equals their held-out error, so
  they meet no situations their data lacks. **No round of correction is needed,** and none was
  built.
- **By reason, in run S** (turn): prey flock 0.030, flee 0.030, wall 0.105; wolves chase 0.007,
  wander 0.021, wall 0.070. The wall branch is still the weak one, at 5% of decisions, and it does
  not show in the catches.
- **The student's turn is closer to the label than the teacher's own turn is,** for prey: 0.034
  against 0.084.

**The other measures, on the AMD iGPU:**

| Run | Prey speed | Wolf speed (live) | Nearest prey | Prey with none in sight | Wolves frozen |
|---|---:|---:|---:|---:|---:|
| T | 2.65 m/s | 3.53 m/s | 0.78 m | 0.1% | 61.2% |
| C | 2.67 m/s | 3.52 m/s | 0.77 m | 0.0% | 61.4% |
| S | 2.65 m/s | 3.54 m/s | 0.77 m | 0.1% | 61.4% |

Speeds and herd tightness are the same in all three, within 1%.

**Repeatable:** runs S and C, run again on the AMD iGPU, gave the same report, number for number.

**Time:** 116 s for the 25 episodes on the AMD iGPU (22 s for a run of rules, 25 s with both
students). On the NVIDIA GPU a run takes 13 to 18 s.

**Decision 1 holds.** A copy that does not wander was the cost of copying the label (§14). In
catches, speed and herd tightness the cost is not measurable here. The fallback, the wander value
as an observation, is not needed. What the eye sees in the demo is for step 10's live switch.

**Next: 7.9.** Commit the two `.mlp.json` files and their report; U26's writer half, A26 and A8;
the guide's "A trained network"; `tools/agents/README.md`; closure.

## Phase 7.9 results — 2026-10-13: the committed networks, their tests, the guide

**Built** (`cd76a25`; no `src/` change; no bundle bytes):
- `tools/agents/models/herd-wolves/`: `prey.mlp.json` and `wolf.mlp.json` (7.7's round-0 students),
  their `.report.json` files, and `evaluation.json` (7.8's five runs, on the AMD iGPU).
- `model.mjs`: `modelText`, the writer, moved out of `train.mjs`. It refuses a network that would
  not read back bit for bit.
- `train.mjs` and `evaluate.mjs` write paths relative to the repository, so a report does not name
  the machine it was made on.
- The guide: "A trained network" in `docs/api/agents.md`, with a fourth checked example.
- `tools/agents/README.md`.
- `test/gpu/suites.mjs`: a suite can state its own time limit for the browser runner.

**The committed files are 476 KB, not the draft's 50 KB.** The draft sized them for 24→32→32→2;
7.7 needed 24→64→64→64→2, which is 9,986 weights per network.
- `prey.mlp.json`: 176 KB;
- `wolf.mlp.json`: 176 KB;
- `prey.report.json`: 34 KB;
- `wolf.report.json`: 35 KB;
- `evaluation.json`: 56 KB.

They are in the repository only; the package still ships `dist`.

**The files reproduce.** A second training run of each species, from the same recording and seed with the defaults, wrote both `.mlp.json` files byte for byte (45 minutes for prey, 19 for wolves, side by side). The committed reports are that run's.

**Tests** (the agents suite is now reference + 72):
- **U26, the writer:** a network with a denormal, the largest f32 and two neighbours one bit apart
  reads back bit for bit; the same input gives the same bytes; a −0, a NaN and an infinity are
  refused.
- **U26, the committed files:**
  - each fits the scenario's names and is exactly what the writer writes;
  - each matches the SHA-256 in its report;
  - each stores a held-out error within 0.05;
  - **each was trained on the scenario's current label.** The file's `trainedOn.teacher` must equal
    the hash of the label's WGSL, so a change to the rule fails the test until the networks are
    trained again.
- **The guide's fourth example** loads the two committed files on a stand-in engine: two think
  passes, and a file with other names is refused before anything is built.
- **A26 and A8** pass on Node/Dawn AMD, Node/Dawn NVIDIA and Chrome AMD: `agent-tools` is 15/15 with
  zero GPU errors.
- **Watched failing:** 10 mutations, each seen failing a named check and restored byte for byte. Two
  of them ran the GPU suite: with the students never attached, and with a file's biases read as
  zero, A8's error check fails.

**A26 found that A4's tolerance does not hold for trained weights.**

| | AMD iGPU (Node and Chrome) | NVIDIA |
|---|---:|---:|
| Actions compared, over 60 steps | 169,438 | 169,438 |
| Beyond A4's 1e-5 + 1e-5·\|cpu\| | 4 | 7 |
| Worst difference from `mlpForward` | 2.9e-5 | 4.4e-5 |

- **Why:** A4's networks have weights of unit scale. These have weights up to 178 in the prey's
  first layer, and rows whose absolute values sum to about 25; f32 rounding grows with them.
- **What A26 asserts:** every action within 1e-4, the bound A27 uses for ONNX Runtime. The count
  beyond A4's bound is printed.
- **It is not a wrong network:** judged by the other species' network, 99.8% of turns fail the same
  bound.
- **The guide now says so** under `MlpBrain`'s numerics. The library's A4 is unchanged.

**A8, as the suite runs it** (seeds 21–25, 1,800 steps):

| | Node/Dawn AMD | Node/Dawn NVIDIA | Chrome AMD |
|---|---:|---:|---:|
| Catches, networks against the label | +0.2% | −0.6% | +0.3% |
| Prey turn error, own runs | 0.034 | 0.034 | — |
| Wolf turn error, own runs | 0.011 | 0.011 | — |
| Time for the whole suite | 67 s | 51 s | about 80 s |

- **Chrome's label run differs from Node's on the same GPU** (1020.8 against 1021.6 mean catches).
  No equality across hosts is claimed; the bar is met on each.
- **The suite went from 10 s to about a minute.** In Chrome it once passed the runner's 120 s limit
  while a training run was using the CPU, so `agent-tools` now states a limit of 360 s.

**Differs from the draft:**
- **A26's bound** is 1e-4, not A4's.
- **A8's error** is the networks' on their own runs, not the held-out error, which would need the
  dataset in the repository.
- **The size** of the committed files.
- **Three files hold `.mlp.json` code,** not one: `model.mjs` writes and reads, `train.mjs` calls
  it.

**Not done here:** the site's demo does not load these files yet. That is step 10.

**Next: 7.10.** The `.onnx` writer; U12; A27.

## Phase 7.10 results — 2026-10-13: the `.onnx` writer

**Built** (no `src/` change; no bundle bytes):
- `tools/agents/onnx.mjs`: `onnxFromNetwork(network, { name, metadata })`, hand-encoded protobuf with
  no dependency, as `test/onnx-gpu/tiny-model.mjs` is.
- `tools/agents/export.mjs`: `npm run agents:export`. It reads every `<species>.mlp.json` in a
  folder (by default the committed networks) and writes `<species>.onnx` to `tools/agents/out/onnx`.

**What a file holds:**
- **One `Gemm` per layer, with `transB = 1`,** so the weights are the layer's row-major
  `[output][input]` array unchanged; then `Tanh`, `Relu` or `Sigmoid`. An identity layer has no
  second node.
- **Opset 13, IR version 8.** The input is `obs`, float32 `[batch, 24]`; the output is `act`,
  float32 `[batch, 2]`. The batch dimension is free.
- **Weights as raw little-endian f32,** so they are the network's, bit for bit.
- **Metadata:** the species, and the observation and action names as JSON, so step 8's reader can
  check them as the `.mlp.json` reader does.
- **No clamp before `Tanh`.** `MlpBrain` clamps tanh's input to ±15; `tanh(15)` is 1 in f32, so
  the results are the same.
- **Size:** 41,186 B per Herd & Wolves network, against 180 KB as `.mlp.json`.

**The `.onnx` files are not committed.** They are a pure function of the committed `.mlp.json`
files: `npm run agents:export` writes them in under a second, and the same network gives the same
bytes.

**Tests** (the agents suite is now reference + 73):
- **U12:** a five-layer network with all four activations decodes to what was written: the node
  order and names, `transB`, each tensor's dimensions, the weights bit for bit (a −0 included), the
  free batch dimension, the opset and the metadata. A network with an unknown activation, a wrong
  weight count or layers that do not chain is refused by name. The test carries its own small
  protobuf decoder; the reader proper is step 8's.
- **A27,** in the `onnx` suite, on 256 observations per species:

| | Node/Dawn NVIDIA | Node/Dawn AMD | Chrome AMD |
|---|---:|---:|---:|
| Prey: worst difference from `mlpForward` | 4.5e-6 | 9.5e-7 | 9.2e-7 |
| Wolves: worst difference | 4.0e-7 | 1.1e-6 | 5.4e-7 |
| Actions beyond 1e-4, of 512 per species | 0 | 0 | 0 |

  The suite is 5/5 with zero GPU errors on all three. It takes the default high-performance
  adapter, which is the NVIDIA GPU under Node here.
- **Watched failing:** 10 mutations of the writer. Eight fail U12. Two ran the GPU suite: with the
  biases written as zero, and with `Relu` in place of `Tanh`, A27 fails for both species.

**ONNX Runtime is closer to `mlpForward` than `MlpBrain` is:** 4.5e-6 at worst, against 4.4e-5
(A26). Both are rounding; the kernels sum in different orders.

**Differs from the draft:**
- **A27 runs in Chrome as well as under Node.**
- **Its observations come from the simulation,** not from a held-out episode: 120 steps under the
  rule on the suite's own device, 256 agents of each species.
- **A27 must run before the suite releases its first ONNX Runtime session.** Run after it, the
  read-back of the observations aborted: the device ONNX Runtime made does not outlive its
  sessions. It is in `gotchas.md`.
- **A command line,** `export.mjs`, beside the writer.

## Closure of part 2 — 2026-10-13

**Every phase, 7.6 to 7.10, is built and measured.** The first network that has learned Herd &
Wolves exists, is in the repository, and is checked on every run of the suites.

| Asked for | Result |
|---|---|
| Record the rule | `npm run agents:record`: 22 s for 5 episodes; byte-identical on a rerun |
| Train a copy per species | `npm run agents:train`: about an hour of CPU; byte-identical on a rerun |
| Error at most 0.05 per species and action | prey turn 0.033 held out and 0.034 on its own runs; wolves 0.011; throttles under 0.001 |
| Catches within 10% of the rule without wander | +0.2% (AMD), −0.6% (NVIDIA) |
| A file today's `MlpBrain` runs | `.mlp.json`, two committed; loading is in the guide |
| An `.onnx` file | written by `npm run agents:export`; ONNX Runtime agrees within 4.5e-6 |
| No library change | `src/` untouched; bundle unchanged |

**What the measurements changed against the draft:**
- **The network is larger and trains longer:** 24→64→64→64→2 at 200 passes, not 24→32→32→2 at 40.
  An hour of CPU, not minutes; 476 KB committed, not 50 KB.
- **No round of correction was needed.** The students do not drift.
- **The copy does not wander, and nothing measured shows it.** Catches, speeds and herd tightness
  are the same with and without the rule's wander.
- **Trained weights are large,** so `MlpBrain` agrees with `mlpForward` within 4.4e-5, not within
  A4's bound.

**Left open:**
- **The wall branch** is above 0.05 for both species (0.105 and 0.070), on 5% of decisions.
- **Other settings** (speeds, rest, cadence) are not measured.
- **The site's demo** still has its own copy of the rule, and does not load the networks: step 10.
- **A public seam for a watching pass** (§12) is still not proposed; one tool uses the label pass.
