# NullGraph — Phase 2 program map (`plan-2.md`)

**Status: accepted by the user on 2026-10-17 (revision 2), with the answers recorded in §9.**
Revision 1 was drafted with Gemini. Revision 2 (Claude, at the user's request) checked it against
the repository and fixed what did not hold; appendix A lists every change and why. Only the user
changes scope.

**What this file is:** the map for Phase 2, the milestone of Cycle C after C2
([`plan.md`](./plan.md) §2): the questions it asks, the pieces it builds, where each piece lives,
and in what order. Detailed designs go in their own `docs/*-design.md` files, written before any
code.

**Relationship to [`plan.md`](./plan.md):** `plan.md` stays the record of Cycles C, D and E and of
every decision. When the user accepts this plan, `plan.md` gets one dated line under "Scope changes"
pointing here.

---

## 1. The goal

**Phase 2 is research.** It asks questions the user plans to answer by experiment, in demos and in
`tools/`. Each question ends in a measured result: numbers with their spread, recorded in a design
doc's results section, whatever they show.

**The library stays general purpose.** The user, 2026-10-17: "i want my null-graph to stay general
purpose all this are researches i m planning to conduct". The research uses the library; the
library never learns the research's words. §3 is the rule that keeps it so.

**The first question (decided 2026-10-17; §9):** can brains evolved by competition, on the user's own GPU,
beat Herd & Wolves' hand-written rule and its clone? And how do the evolved strategies change once
bodies cannot overlap and the field has obstacles?

Where C2 left things:
- **Clones, not learners.** C2's two networks (24 → 64 → 64 → 64 → 2, tanh; 10,050 parameters per
  species) were trained offline to copy the rule (`HERD_LABEL`). Nothing adapts in the loop.
- **Bodies pass through each other.** Agents steer apart, but nothing stops an overlap. Collision
  resolution was a non-goal of Cycle C ([`../agents-design.md`](../agents-design.md) §1), and the
  films showed it (`plan.md`, scope changes, 2026-10-16).
- **No obstacles.** The world's bounds are the only walls.

---

## 2. How the work is named

| Level | Example | Means |
|---|---|---|
| **Milestone** | 2A, 2B, 2C | Stopping points inside Phase 2. **2A:** spikes. **2B:** general library pieces. **2C:** the research itself, in `tools/` and the site. The backlog (§5) is not scheduled. |
| **Step** | step 11 | A numbered deliverable, continuing Cycle C's steps 1–10. |
| **Phase** | phase 11.1 | One commit-sized piece of a step, defined in that step's design doc. |
| **Spike** | S6 | A measurement script in `test/spikes/` that decides a design before library code. Continues S1–S5. |
| **Test IDs** | U43, A35 | Continue the agents suites (`U` headless, `A` real GPU; U42 and A34 are the last taken). A new package entry numbers its tests in its own design doc, as the physics docs do. |

Two names are avoided on purpose. **P1–P8** are the physics plugin's phases
([`../physics/README.md`](../physics/README.md)), so no milestone or test here is called P-anything.
**"Phase 2.1"** would read as step 2's first phase.

---

## 3. The library rule

A research need becomes library code only when **all four** hold:

1. **It can be said without the research's words:** no prey, wolves, stamina, catches, fitness or
   generations in its API, its docs or its WGSL.
2. **A second, unrelated consumer is named:** a use case from
   [`../use-cases-after-phase-2.md`](../use-cases-after-phase-2.md), another demo, or the product.
3. **Today's hooks cannot do it,** or do it only at a cost that was measured.
4. **It meets the library's bar:** a design doc, a CPU mirror, tests watched failing, a size line,
   and nothing added to apps that do not import it.

Anything else stays in the demo or in `tools/`. A demo-side piece that proves itself can be promoted
later, by its own design, once its second consumer is being built.

**Today's hooks** (shipped; [`../api/agents.md`](../api/agents.md)):
- **`Actions.custom`:** app WGSL replaces the integrator (turn limits, inertia, per-arena bounds).
- **`Sensors.custom`:** app WGSL observes its own agent's record. Shapes known when the demo is
  built, such as a field's rocks, can be compiled into it as WGSL constants.
- **`addStepStage`:** app WGSL after act, with caller-owned buffers and the neighbour lists (named
  by generated constants). A stage writes only its own record: reads of peers go in one dispatch,
  and the writes they cause in the next. Herd & Wolves' catches and respawns already work this way.
- **Free record words:** `energy` (`a.state.x`), `genome` (`a.ids.z`) and `user1` (`a.ids.w`). The
  demo uses `user0` (`a.state.w`) as the wolves' rest timer.
- **A brain is any object of the `Brain` shape** (`kind`, `wgsl`, `buffers`), so a spike can
  prototype a brain without library code.
- **Reruns are exact on one device:** the same definition, seeds and steps give the same records
  (A7, A23).

**What the first question needs, and where each piece lives:**

| Need | Lives in | Why |
|---|---|---|
| A network per agent: weight rows chosen by the record's `genome` word | **Library, step 11** | The seam Cycle C reserved for it ([`../agents-design.md`](../agents-design.md) §4.1, §9.3). A copy in the demo would duplicate the kernel, its CPU mirror and its export. Other consumers: evolution of any kind, crowds with varied behaviour, A/B tests of two policies in one population |
| Neighbour lists that feed no observation | **Library, step 12, only if S8 needs it** | Every neighbour list today comes from a declared sensor, and every sensor adds observations, which would change the networks' 24 inputs. Other consumers: any app stage that needs neighbours (contacts, catches, effects) |
| Isolated arenas, one per candidate | Demo | Custom actions keep each agent in its tile, a custom sensor reports the tile's walls in `Sensors.bounds()`'s format, and respawns stay in the tile. A library option is in the backlog |
| Episodes, resets and fitness counters | Demo | Step stages, and one counter per agent in a caller buffer, read once per episode: the pattern of `HERD_CATCHES` |
| Stamina, metabolism, inertia | Demo | Custom actions and a step stage, on `energy`: a modelling choice of this research |
| Static obstacles: sensing and push-out | Demo | Rocks compiled into a custom sensor and the custom actions as constants. A sensor that reads app buffers (`Sensors.field`) waits for NAV, in the backlog |
| Bodies that cannot overlap | Demo first (S8); then a new `null-graph/physics` entry (step 13) | Two step stages: corrections into a caller buffer, then applied in the next dispatch. Other consumers: crowds (the venue planner, I13), horde games (G1, G2), the films' look |
| Evolution strategies, the Hall of Fame, runs and evaluation | `tools/agents/`, step 14 | Never shipped in the package. The optimizer sees weight vectors and scores, nothing of the scenario |
| Films of the results | `tools/film`, step 15 | D2's offline films: exact, smooth video of any checkpoint, however slowly it computes |
| The research demo | The site, step 16 | Like every demo |

So the first question costs the library **one piece for certain** (step 11), one small piece if a
spike shows the need (step 12), and one piece the user chose (step 13, a new `null-graph/physics`
entry). None of them mentions Herd & Wolves.

---

## 4. The first question: co-evolution in Herd & Wolves

### 4.1 Definition

- **Policies.** θ for prey and φ for wolves, each an `MlpNetwork`: either C2's shape, so that the
  clones can be the start, or a smaller one (§4.4).
- **Arena.** An independent copy of the scenario: a tile of the world holding a fixed number of prey
  and wolves, spawned from a seed and run for T steps. Tiles are further apart than the sense
  radius, so no query reaches across.
- **Candidate.** One policy's weights, perturbed: θ + σε or θ − σε, with ε drawn from a seeded
  Gaussian. K mirrored pairs make 2K candidates per generation. Every body of a species in an arena
  runs that arena's candidate, so a candidate's score is averaged over many bodies.
- **Opponent.** The other species' policy in the arena: its current weights, or a member of its Hall
  of Fame (an archive of earlier champions).
- **Fitness.** Written in the demo's code and fixed before a run starts. Start simple: for prey, the
  share of the episode each prey survives, averaged; for wolves, catches per wolf.
- **Update** (Salimans et al. 2017). Centered ranks over the 2K scores (u in [−0.5, 0.5]); the
  estimate g = Σₖ (u⁺ₖ − u⁻ₖ) εₖ / (2Kσ); weight decay; an Adam step on θ or φ.

**Invariants**, each a test in the step that builds it:

| # | Property | Checked by |
|---|---|---|
| I1 | **Independence:** no agent sees, touches or is counted with an agent of another arena | Over 600 steps and every arena layout, no neighbour index points into another tile |
| I2 | **Equivalence:** one arena covering the whole world behaves exactly like today's demo | The arena's custom actions and wall sensor give bit-identical records to `Actions.steering()` and `Sensors.bounds()` |
| I3 | **Fair pairs:** both members of a mirrored pair meet the same seed and the same opponent | The trainer's schedule, unit-tested |
| I4 | **Exact rows:** each agent's actions equal `mlpForward` of its own row, and follow a change of that row | A GPU test in step 11, on A21's pattern |
| I5 | **Reproducible runs:** the same seed, config, build and device give the same checkpoints, byte for byte | Two short runs compared |
| I6 | **No readback in the frame loop:** scores come back once per episode, off the frame path | Instrumented, on A6's pattern |

### 4.2 Hypotheses

| # | Hypothesis | Judged by |
|---|---|---|
| H1 | Started from the clones, evolution beats the rule: evolved wolves catch more rule prey, and evolved prey survive rule wolves longer | Held-out seeds; the 95% confidence intervals do not overlap |
| H2 | Started from random weights, evolution reaches at least the clones' level | The same measure, within a budget of generations set in the design |
| H3 | The Hall of Fame reduces cycling: strategies that beat the latest opponent but lose to older ones | Cross-play between generations ("CIAO" plots, Cliff & Miller 1995), with the archive on and off |
| H4 | Contacts and obstacles change what evolves | Where catches happen relative to obstacles, herd density and survival, with and without; only once S8 and the obstacles exist |

A hypothesis that fails is a result too, and is recorded the same way.

### 4.3 Protocol

- **Seeds:** training seeds and held-out seeds never overlap, as `agents:evaluate` keeps them today
  (seeds 21–25).
- **Baselines:** the rule with and without wander, and the clones, in `agents:evaluate`'s matchups
  (T, S, Sp, Sw), extended with evolved brains.
- **Ablations:** Hall of Fame on and off; start from the clones or from random weights; contacts and
  obstacles on and off.
- **Spread:** at least 20 episodes per evaluated pairing; means with 95% intervals.
- **Every run writes:** its config, seeds, git commit, adapter and driver, one line of statistics
  per generation, and checkpoints (`.mlp.json`, `.onnx`) at chosen generations.

### 4.4 Questions the designs must answer

- **Alternating or simultaneous.** Evolve one species against a frozen opponent, then swap
  (textbook ES, scores taken independently), or both at once (twice the throughput, but mirrored
  pairs then need a 2 × 2 design, so that one species' perturbation does not bias the other's
  score).
- **Network size.** C2's 10,050 parameters allow a start from the clones. A smaller network makes
  each generation cheaper and the search space smaller, and suits a start from random weights.
- **The numbers:** arena size, bodies per arena, K, σ, learning rate, episode length.
- **The archive:** how often a champion enters it, and what share of arenas meets archived
  opponents. Bansal et al. (2018) sample opponents from a policy's history.
- **Shaping.** Rank shaping removes the effect of a score's scale, not of a wrong incentive. Added
  terms (distance to wolves, closing speed) are experiments, not defaults.

---

## 5. The steps

### 2A — spikes: measure before building (scripts only, no library code)

| Spike | Question | How | Decides | Status |
|---|---|---|---|---|
| **S6** | What does a weight row per agent cost? | `npm run spike:s6` (`test/spikes/genome-rows/`, on phase 7.2's pattern), on Node/NVIDIA, Node/AMD and Chrome/AMD. **Part 1, the think kernel alone:** C2's shape and 7.2's smaller one; 16, 64 and 256 rows; 4K / 16K / 64K agents. Variants: (a) the row read from `genome` in each lane; (b) the row derived from the workgroup's index, so the compiler can prove it uniform; (c) the row broadcast through workgroup memory; (d) one pass per row. Rows aligned to 64-agent blocks, in runs of 8, and shuffled. **Part 2, a whole step:** Herd & Wolves on today's brains against (a), whose rows are identical copies of the committed networks, so the simulation is the same and only the addressing differs | Step 11's kernel. **If (a) adds at most 10% to a step at the counts evolution uses, on every pair, build (a).** Otherwise the cheapest variant that keeps one row per agent, or the numbers go to the user | ✅ Measured 2026-10-17 ([results](../agents-genome-rows-design.md#spike-s6-results--2026-10-17)). The rule failed: (a) adds 4–7% to a step on NVIDIA, 9–17% in Chrome and 33–36% under Node on the AMD iGPU (C2's networks). A staged kernel (e′) added to the spike is the cheapest row kernel where 64-agent blocks share a row, bit-identical; the kernel goes to the user in step 11's design (Q4) |
| **S7** | Does one generation work end to end, and how fast? | Spike code: arenas as tiles (custom actions, the tile's wall sensor), episodes and resets, per-agent score counters, one readback per episode, the ES update on the CPU, S6's kernel as a prototype brain object. Generations a minute on the three pairs; a 20-generation run from the clones; which exploits appear with no physics | Whether live evolution in the browser is worth building (step 16); the arena layout; whether contacts and obstacles matter to the research, or only to the look | Not started |
| **S8** | What do contacts cost, and do today's neighbour lists suffice? | Spike code: two step stages over the lists the definition already declares (1 nearest wolf, 3 nearest prey); corrections written to a caller buffer and applied in the next dispatch (Jacobi). 4K–64K agents on the three pairs: cost, the worst overlap and peak density with and without, jitter, and two runs compared byte for byte | Step 12 (needed or not); step 13's parameters and home | Not started |

Revision 1 assumed aligning rows to workgroups removes the rows penalty; phase 7.2 had already found
that penalty does not come from the data (appendix A, item 1). S6 confirmed it under Node, and found
two levers that do matter: the layout (rows shared by 64-agent blocks) and staging a block's row in
workgroup memory. It also found the network's size matters as much as the kernel (step 11's design,
"What it shows").

### 2B — general library pieces (each: design doc first, size line, tests watched failing)

| Step | Delivers | Second consumers | Design | Status |
|---|---|---|---|---|
| **11** | **`MlpBrain.byGenome`:** a weight table of N rows of one shape; each agent reads the row its `genome` word names. All rows written in one upload, or one row alone; `inspect()` reports the row; the CPU mirror reads it too. The kernel is S6's choice | Evolution of any kind (the backlog's ecosystem included), varied crowds, A/B tests in one population | [`docs/agents-genome-rows-design.md`](../agents-genome-rows-design.md) | Phase 11.0 (S6) ✅. Design decided 2026-10-17 (§14 there: every recommendation; the staged kernel with a per-lane fallback; Q5 parked). Phase 11.1 ✅ `2e3c8c2` (the cooperative think entry; U44, A35's first part on three pairs) and Phase 11.2 ✅ `2d51159` (`byGenome`, `setWeights({ row })`, `setRows`, staged kernel + guard, U43, size 21,362 B / 133 B under C2 limit; a reserved word fixed in review), committed 2026-10-17. Next: 11.3 |
| **12** | **Only if S8 needs it: neighbour lists that feed no observation** (a query-only declaration), readable by step stages as today's lists are | Contacts, catches, any app stage that needs neighbours | Within step 13's design, or its own | Not started |
| **13** | **Contacts in a new `null-graph/physics` entry** (decided, §9), after S8. A stage that separates overlapping spheres in any buffer whose floats 0–3 are position and radius (the agent record matches the physics body record there), over the point grid: Jacobi with a correction buffer, XPBD compliance, no float atomics, exact reruns | Crowds (I13), horde games (G1, G2), herds in films; later, P3's particles | `docs/physics/physics-contacts-design.md` | Not started |

None of the three names a species, a score or a generation.

### 2C — the research (tools and site; no library code)

| Step | Delivers | Design | Status |
|---|---|---|---|
| **14** | **`npm run agents:coevolve`** in `tools/agents/`: the ES optimizer (weight vectors and scores in, weight vectors out, nothing of the scenario); the Hall of Fame; the Herd & Wolves arena scenario (custom actions, the wall sensor, resets, score counters; stamina and obstacles when used); run logs and checkpoints; `agents:evaluate` extended with held-out cross-play. Each submission sized from measured step cost, using the films' step guard (D2.1, calibrated in D2.3) | `docs/agents-coevolution-design.md`, formal definition first, starting from §4 | Not started |
| **15** | **Films of the results** through `tools/film` (D2), at the user's word (§9): a scene that loads checkpoints into the arenas, or into one champion arena, and records them frame by frame, exact and smooth however slowly they compute. Generation 0 against generation N, evolved brains against the rule. Built on D2's general scene contract, alongside D2.3 | `docs/agents-coevo-films-design.md`, or a section of step 14's | Not started |
| **16** | **`HerdWolvesCoevo`,** a copy of `HerdWolvesGPU` in the site's `AgentsExperiments/` (`HerdWolvesGPU` stays as it is). A checkpoint scrubber (random start, chosen generations, the clone, the rule) and matchups between any two; an arenas view, each tile a candidate, and a champion view; a HUD with the generation, catch and survival rates with their intervals, and the cross-play chart. Live evolution in the worker only if the user accepts S7's rate. Validated with `check-site.mjs` and `npm run drive` | `docs/agents-coevo-demo-design.md` | Not started |

Outlets, once results exist: the films (step 15), the demo (step 16), and a write-up of §4.2's
answers, for the use-case study's videos and classroom lab (C3, E1, E8).

### Backlog — not designed; each promoted only by the user, with its own design

| Piece | What it is | Second consumers named |
|---|---|---|
| Worlds | N independent copies of a scenario in one system: the arenas as a library option | Side-by-side venue layouts (I13), classroom runs (E8) |
| Field sensor (`Sensors.field`) | Observes an app's 2D grid over the world: a value and its gradient in the agent's frame | Walls and flow fields (NAV, I13), food and scent (the ecosystem), obstacles beyond a few dozen |
| Flow fields (NAV) | Paths to goals around walls, computed on the GPU and read through the field sensor | The venue planner (I13), horde games (G2) |
| Ecosystem evolution | The life cycle on the GPU: energy and food, birth, death, and mutation that writes rows on the GPU | The live ecosystem of [`../agents-design.md`](../agents-design.md) §13; E8, C3, G10 |
| Ray eyes (EYES) | Rays through the grid and the obstacles, inside the sense stage. Not physics P4, which answers the CPU's queries a frame late | Camouflage with evolution (E8), stealth games (G12) |
| Wider brains (BRAIN+) | Memory per agent, a tiled kernel wider than 64, discrete output heads (`plan.md`, 2026-10-15) | Colony games (G11), NPC crowds (D4) |
| Staged kernel for single networks (parked by the user, 2026-10-17) | Found by S6: C2's networks think 7–12× slower in Chrome than under Node on the same AMD GPU, and step 11's staged kernel runs them 2.5–4× faster, bit for bit; it slows other cases, so a spike over more shapes comes first (step 11's design, Q5) | Every `MlpBrain` in Chrome, step 10's live demo first |
| 3D agents; physics P1–P3; server runs (SRV); terrain | As in their own docs. For terrain, Cycle E's question of how agents read the ground | [`../use-cases-after-phase-2.md`](../use-cases-after-phase-2.md) §6 |

---

## 6. Order against other work

Phase 2 is not scheduled yet. It competes with:
- **D2.3** (offline films, started) and **D1's phase 5.4** (waiting for the skinned animals);
- **Cycle E**, terrain for games, proposed as the next cycle after D (`plan.md` §5, item 15);
- **the portfolio** (plain layer by 2026-10-25, v1 by 2027-01-10), which runs on Axion.

**Decided 2026-10-17: Phase 2 runs alongside D2.3** (§9). Its spikes are scripts only; GPU
measurements take turns with D2.3's GPU runs (§7).

---

## 7. Constraints for every step

- **Design before code:** a formal definition first, options with a recommendation, open questions
  at the end. Spikes before library code; phases with exit criteria.
- **§3's rule** for anything that enters `src/`.
- **Tests watched failing** before they are trusted; IDs from U43 and A35.
- **No readback in the frame loop.** Scores come back once per episode, mapped off the frame path.
- **Reruns stay exact on one device:** no float atomics in new stages, each invocation writes only
  its own rows, and the grid's fixed order holds.
- **The GPU timeout guard:** no submission near 2 s, predicted from measured cost, in the trainer,
  the demo and the spikes. **No fixed caps** in the site tuned to this machine's GPU.
- **Take turns on the GPU:** a timing measured while another GPU job runs (D2.3's films, another
  session's suites) is discarded and taken again.
- **Bundle:** each library piece gets a size line set from its build; apps that do not use it pay
  nothing (`size.mjs`).
- **`HerdWolvesGPU` untouched:** `check-site.mjs` and `film:check` stay green.
- **No 3D models in the repository** (glb, gltf, fbx, obj, blend).
- **Commit only when the user asks;** never push or publish unasked.

---

## 8. First moves

1. S6, then S7 (which uses S6's prototype), then S8. Their results go in the designs they decide.
2. Step 11's design, then its phases.
3. Step 14's design (§4 as its start), then the trainer: first runs from the clones, then from
   random weights.
4. Step 15 (films) and step 16 (the demo), each design first.
5. Steps 12 and 13 after S8; step 12 only if S8 shows the need.

---

## 9. Decisions

**Decided by the user:**

1. **2026-10-17: the library stays general purpose, and Phase 2's work is research.** The user: "i
   want my null-graph to stay general purpose all this are researches i m planning to conduct".
   §3 applies it.

2. **2026-10-17: the answers to revision 2's five questions.** The user: "q1 yes co evoluution
   first and q2 yes train headless,offline rendering/filming would be useful here we have that also
   now under tools/films and q3 yes q4 yes a new entry yes against d2.3 idea is to run parrelly ok".
   - **Q1, which question first:** co-evolution, ahead of NAV (the use-case study's proposal).
     NAV's field sensor waits in the backlog until NAV starts.
   - **Q2, training and playback:** headless training first (step 14), and offline films of the
     results through `tools/film` (step 15, added for this answer). The demo scrubs the checkpoints
     (step 16); live evolution in the browser follows only if S7's generation rate makes it
     watchable.
   - **Q3, grounding:** contacts, rocks and stamina as demo code first (S8, step 14).
   - **Q4, contacts' home:** a new `null-graph/physics` entry (step 13, after S8): P3's contact
     piece brought forward, generic over any buffer of positions and radii. That changes the
     physics series, designed P1 → P2 → P3 with P3's contacts reading P2's pair list; the physics
     README gains a line for it when step 13's design is accepted.
   - **Q5, when:** alongside D2.3.
3. **2026-10-17: spikes first, then designs** (the user: "i think first spike then design that is
   format"), as §8 orders them. S6 starts.

**Waiting on the user:** nothing at this level. Each step's design brings its own questions.

---

## 10. References

1. Salimans, T., Ho, J., Chen, X., Sidor, S. & Sutskever, I. (2017). *Evolution Strategies as a
   Scalable Alternative to Reinforcement Learning.* arXiv:1703.03864. **Used for:** mirrored
   sampling, centered-rank shaping, weight decay, and §4.1's update.
2. Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J. & Schmidhuber, J. (2014).
   *Natural Evolution Strategies.* JMLR 15. **Used for:** rank-based utilities and the background
   of the gradient estimate.
3. Rosin, C. D. & Belew, R. K. (1997). *New Methods for Competitive Coevolution.* Evolutionary
   Computation 5(1). **Used for:** the Hall of Fame.
4. Cliff, D. & Miller, G. F. (1995). *Tracking the Red Queen: Measurements of Adaptive Progress in
   Co-Evolutionary Simulations.* ECAL 1995. **Used for:** cross-play between generations (CIAO
   plots), H3's measure.
5. Bansal, T., Pachocki, J., Sidor, S., Sutskever, I. & Mordatch, I. (2018). *Emergent Complexity
   via Multi-Agent Competition.* ICLR 2018. **Used for:** sampling opponents from a policy's
   history. Its agents were trained with PPO, not evolution strategies; it is cited for the
   opponent sampling only.
6. Sims, K. (1994). *Evolving 3D Morphology and Behavior by Competition.* Artificial Life IV.
   **Used for:** the precedent of behaviour evolved by competition.
7. Macklin, M., Müller, M. & Chentanez, N. (2016). *XPBD: Position-Based Simulation of Compliant
   Constrained Dynamics.* MIG 2016. **Used for:** compliance in step 13's contacts, as in
   [`../physics/physics-p3-solver-design.md`](../physics/physics-p3-solver-design.md).

---

## Appendix A — what revision 2 changed, and why

| # | Revision 1 said | What the repository shows | Revision 2 |
|---|---|---|---|
| 1 | Aligning each candidate to a 64-agent workgroup makes every lane read the same row, so rows keep a single network's speed in Chrome | Phase 7.2 measured rows at **4.6× in Chrome with every agent the same species**: "the rows penalty is the compiler's, not the data's" ([`../agents-species-brains-design.md`](../agents-species-brains-design.md), "Phase 7.2 results"). It also measured a think pass at 4,096 agents in tens of microseconds | S6 measures four variants, rows the compiler can prove uniform among them, and judges each by its share of a whole step |
| 2 | 64 candidates evaluated together in one arena | A candidate's score then depends on the other candidates around it, while ES needs each score taken independently | One arena per candidate (§4.1, I1) |
| 3 | `InertialDynamics` with stamina, a fitness stage with predator–prey rewards, `ObstacleSet` and `Sensors.obstacles` as library code | Custom actions, custom sensors and step stages already do each of them (§3) | Demo code. The library gets only pieces that pass §3 |
| 4 | A new `null-graph/physics` entry in the first milestone | The physics series is designed P1 → P2 → P3, with contacts reading P2's pair list; agents can be kept apart with step stages today | Demo first (S8); the entry by decision (Q4) |
| 5 | "Decisions recorded with the user (2026-10-17)": co-evolution ahead of NAV and terrain, hybrid training, the physics scope | `now.md` records the order (NAV first, then EVO) as awaiting the user; nothing in the repository records the other choices | Open questions Q1–Q5 |
| 6 | Test IDs P1–P3 for physics checks; milestones P2.0–P2.3 | P1–P8 are the physics phases, and P2 is the broadphase | Milestones 2A–2C; tests continue from U43 and A35 |
| 7 | Ray eyes through physics P4's queries | P4 is the CPU's query API, answered a frame late; a ray sensor runs inside the sense stage | In the backlog, described as such |
| 8 | The Hall of Fame from Sims (1994); "genuine adversarial reinforcement learning" | The Hall of Fame is Rosin & Belew (1997); Bansal et al. used PPO; evolution strategies are not reinforcement learning | References corrected; plainer claims |
| 9 | Live evolution of 64 cohorts in the browser, promised | No generation rate has been measured | Depends on S7's rate and the user (Q2) |
| 10 | Steps 15–18 specified in detail (eyes, field sensors, memory, full physics, 3D, server runs) | Far work, specified before the first question has a result | A backlog, with each piece's second consumers named |
