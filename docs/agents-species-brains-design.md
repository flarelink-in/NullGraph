# NullGraph Agents — C2 step 7, part 1: per-species brains — Technical Design

**Parent:** [`agents-design.md`](./agents-design.md) §13, C2 step 7.
**Decided with the user, 2026-10-12** ([`work/plan.md`](./work/plan.md) §5):
- one network per species (option B);
- after phase 7.2's measurements, **one think pass per brain, gated to its species** (§6), with
  weight-table rows kept for many same-shape species (§5) — the rule: passes below 8 species, rows
  from 8;
- no fallback brain;
- core limit negotiation (landed, §12).

The first draft's single-stage strategy and its composite brain were withdrawn when 7.2 measured
per-lane routing at 4.6–5.6× a single network in Chrome; see "Phase 7.2 results" below.

**Builds on:** the brain seam ([`agents-gpu-loop-design.md`](./agents-gpu-loop-design.md) §9.1);
`MlpBrain` and its measurements ([`agents-mlp-design.md`](./agents-mlp-design.md), A4, phase 6.3).
**Status:** ✅ 7.1 committed (`1dc7e74`); ✅ 7.2 committed (`94069ea`; option B docs `30336d1`);
✅ 7.3 committed (`681ccf4`), with one change from the reviewed §4: a species mask per pass instead
of a slot table (see "Phase 7.3 implementation" below). Next: 7.4 (A22).
**Target:** `src/extras/agents/` (`AgentSystem.setBrain(brain, { species })`, a per-stage species
gate in the think wrapper), headless and GPU tests, the API guide. Core unchanged.

---

## 1. Purpose

Each species runs its own brain. Two mechanisms, chosen by measurement (§7):

- **One think pass per brain, gated to its species** (`setBrain(brain, { species })`, §6). Mixed
  kinds — prey on rules, wolves on a network — and networks of different shapes need nothing more.
  The default for a few species.
- **Weight-table rows** (`MlpBrain.bySpecies`, §5): many same-shape networks as rows of one table,
  read in one pass. Cheaper from about 8 species on, because one pass reads each record once.

Both may be combined: a rows brain is just a brain, serving several species in one pass.

**The single most important pre-existing fact:** every think invocation writes only its own agent's
action and reason rows (fact 5). Several think passes over disjoint species sets therefore never
race, whatever their order, and need no synchronization beyond the dispatch order they already have.

**The second fact:** 7.2 measured what one pass with per-lane routing costs: 4.6–5.6× a single
network in Chrome on the AMD iGPU, whatever the species layout, because a row chosen per lane stops
the compiler from loading weights as wave-uniform data. Separate passes keep each brain's code
exactly as fast as when it serves every species (fact 8).

### Non-goals

- **Per-agent weights or genomes:** Phase 2 (rows selected per agent).
- **A fallback brain for unlisted species:** decided against; every species is named.
- **Species added or removed at run time:** the definition is fixed for the system's life.
- **Compiling a new brain's stage asynchronously:** deferred to step 9 (plan §5, item 3).
- **The composite `SpeciesBrain`:** withdrawn; per-species passes mix kinds without WGSL
  composition.
- **Recording, training and export:** step 7, part 2, with its own design.

### Design principles applied

- **Measure, then decide:** the mechanism follows 7.2's numbers on three host/GPU pairs, not the
  first draft's analysis, which missed the compiler effect.
- **Everything expensive at build time** ([`design-principles.md`](./design-principles.md) §11): a
  stage compiles once per brain at `setBrain`; reassigning species between brains that already have
  stages uploads one 16-byte mask per changed brain and compiles nothing.
- **Fail loud at build time** (§2): unknown species are refused at `setBrain`; a species left
  without a brain is refused before the first step, by name.
- **Size by demand** (§10): one stage per distinct brain in use, never per declared species.
- **Declared budgets** (§7): each pass's species mask is a uniform, outside the storage budget (§4).

## 2. Architecture

The frame is unchanged: `engine.render([...agents.getComputeStages(), ...visibility, scenePass])`.
Each step's chain gains one think pass per distinct brain, in the order the brains were first
attached, between sense and act:

```
grid → queries → sense → think[brain A] → think[brain B] → … → act → app stages → tick → (instances)
                           │                 │
                           e.g. rules        e.g. wolfNet     each skips agents whose species bit
                                                              is clear in its own brainSpecies mask
```

## 3. Integration facts (verified against this repository, 2026-10-12)

| # | Claim | Where | Consequence |
|---|---|---|---|
| 1 | `AgentSystem` holds one think stage; `setBrain` replaces it | `thinkStage` (`AgentSystem.ts:58`), `setBrain` (`:163`), chains (`rebuildChains`, `:140`) | Per-species passes need an `AgentSystem` change: a stage per distinct brain |
| 2 | The wrapper gates on `thinksNow` and passes the species index to `think` | `agentThinkShaders.ts:46`, `:51` | A species gate sits beside the cadence gate, in generated code only |
| 3 | The think stage binds 5 storage buffers and the species uniform at group 0 | `agentThinkShaders.ts:30–35` | A 7th group-0 binding, a uniform, costs no storage budget |
| 4 | The species table is full: 8 words, 32 bytes, in a 16-byte-aligned uniform array | `struct Species` (`agentLayout.ts:53`), `SPECIES_FLOATS` (`:37`) | Each pass's species set is its own 16-byte uniform, not a species field |
| 5 | A think invocation writes only its own agent's `act` and `reason` rows | `setAct`, `setReason` (`agentThinkShaders.ts:39–40`) | Passes over disjoint species never race, in any order |
| 6 | Compute pipelines are created synchronously | `BatchManager.createComputeStage` (`BatchManager.ts:178`) | A brain's first stage compiles at `setBrain`; reassignment through the masks compiles nothing |
| 7 | Rows are chosen by the system's `SPECIES_<NAME>` constants | `buildMlpThinkWGSL` (`mlpShaders.ts`), `speciesConstantName` (`agentLayout.ts`) | A rows brain serves several species in one pass, alongside other brains |
| 8 | One pass per species keeps each brain's code unchanged; per-lane routing does not | Phase 7.2 results (below) | Passes cost 1.06–1.5× per 2 species; rows are flat in species count |
| 9 | The group-1 budget is checked per stage | `AgentSystem.checkExtraBuffers` (`AgentSystem.ts:156`) | Each brain's buffers are budgeted on their own stage |
| 10 | `setBrain` asks `brain.assertDevice?.(device)` first | `AgentSystem.ts:166` | Unchanged per brain |

## 4. Data model

**The species mask.** Each think pass binds its own `UNIFORM | COPY_DST` buffer of 16 bytes at
group 0, binding 6: `brainSpecies: vec4<u32>`. Its `.x` holds one bit per species the brain serves
(bit `s` for species index `s`, 16 species at most); the other three words pad it to a uniform's
16-byte minimum. A system with one brain has one pass whose mask holds every species' bit, so its
gate always passes.

**Masks are written** when an assignment changes: one 16-byte `writeBuffer` per brain whose species
set changed, never per frame. A mask is created with its brain's pass and destroyed when that brain
serves no species.

**Changed in 7.3 from the reviewed design** (2026-10-13). The reviewed §4 had one 64-byte slot table
per system and a `BRAIN_SLOT` constant compiled into each stage. A mask per pass keeps a brain's
generated shader free of anything positional:
- A brain retired and attached again generates identical shader text (U22 pins it), so the
  pipeline cache can serve it; a new slot constant would have forced a new compile.
- Moving species between attached brains still compiles nothing.
- The gate is one bit test instead of a two-level table lookup.
- The cost is 16 bytes per pass instead of one 64-byte table per system.

## 5. Weight-table rows (`MlpBrain.bySpecies`) — built in 7.1, kept for many species

As built and tested in 7.1 and 7.2: one table, one row per species, chosen by species name in one
pass; `setWeights(network, { species })` uploads one row. Use it for **8 or more same-shape
networks**; below that, give each species its own brain (§6). Measured (rows ÷ passes; below 1,
rows faster; 1,000 / 20,000 / 100,000 agents):

| Species | NVIDIA, Node | AMD, Node | AMD, Chrome |
|---|---|---|---|
| 2 | 0.72 / 0.88 / 0.85 | 1.32 / 1.37 / 1.25 | 0.61 / 3.11 / 2.86 |
| 8 | 0.13 / 0.37 / 0.77 | 0.21 / 0.48 / 0.40 | 0.24 / 1.07 / 1.25 |
| 16 | 0.06 / 0.19 / 0.58 | 0.11 / 0.25 / 0.23 | 0.10 / 0.55 / 0.68 |

The threshold is a round number near the measured crossovers: in Chrome at 20,000–100,000 agents
8 passes are 7–34% faster than rows (7.2's and 7.3's runs) and 16 are clearly slower; on Node, and
at 1,000 agents everywhere, rows already win at 8.

## 6. Per-species think passes (`setBrain(brain, { species })`)

```ts
agents.setBrain(rules);                                    // every species, as today
agents.setBrain(wolfNet, { species: 'wolf' });             // wolves only; others keep theirs
agents.setBrain(herdRows, { species: ['s2', 's3', 's4'] }); // one rows brain, several species
agents.setBrain(rules, { species: 'wolf' });               // back: rules has a stage, so 0 compiles
```

- **`setBrain(brain)`** assigns every species to `brain`, as today. **`setBrain(brain, { species })`**
  assigns the named species (one name or a list) and leaves the others.
- **Stages follow distinct brains.** A brain new to the system gets a stage and its mask: one
  compile. A brain that already serves a species keeps its stage: only the changed masks are
  uploaded, no compile. A brain left serving no species loses its stage (disabled and removed from
  every chain) and its mask; it may then be destroyed, as after a replacement today.
- **The gate** is generated in the wrapper beside the cadence gate:
  `if ((brainSpecies.x & (1u << a.ids.x)) == 0u) { return; }`, so a brain's own WGSL is unchanged
  and runs exactly as fast as when it serves everyone.
- **Completeness:** before the first `getComputeStages()` or `step()`, every species must have a
  brain; the error names the first missing one. There is no fallback.
- **Checks per brain**, before anything is built: `assertDevice`, dims, the think signature and its
  group-1 budget, exactly as today.
- **Order:** think passes run in the order their brains were first attached. Species sets are
  disjoint (fact 5), so order cannot change results.
- **Cost:** each pass reads every agent's record, then returns early for other species' agents
  (7.2's "passes"); a system with one brain pays exactly what it pays today plus one uniform read.

## 7. Cost — measured in 7.2

Per pass, the fixed cost of reading records and gating is paid once per distinct brain; a brain's
own code is never slowed. Two species cost 1.06× (NVIDIA), 1.48× (AMD, Node) and 1.5× (AMD, Chrome)
a single network at 100,000 agents with species in contiguous blocks, and about 2× when species
alternate. Rows are flat in species count but pay a compiler-dependent per-lane penalty (up to 5.6×
in Chrome). Hence §5's rule. Full tables: "Phase 7.2 results" below. 7.3 re-measured passes with
the production gate: 1.04×, 1.42× and 1.52× contiguous, 1.98×, 1.67× and 1.96× alternating ("Phase
7.3 implementation").

## 8. Lifecycle and updates

- Brains stay caller-owned and belong to their engine's device.
- `setWeights` on any `MlpBrain` (single or rows) uploads without touching stages or the table.
- Destroy order is as today: reassign the species or destroy the system first, then the brain.

## 9. Bundle

The per-species passes live in `AgentSystem` and the wrapper, so every agents app carries them:
estimated +0.3–0.5 KB min+gzip on the rule path, **measured +366 B** in 7.3 (12,375 → 12,741 B).
Rows stay on the MLP path (7.1: +521 B). The withdrawn composite saves its planned 0.5–0.9 KB. The
whole entry must stay within C2's allowance of at most 21,495 B: 17,827 B after 7.1, **18,174 B**
after 7.3.

## 10. Test suite

**Headless (U22, `test/agents-system.test.mjs`; as built in 7.3, five tests):**
- one pass per distinct brain, each with its own mask: `0b111` for a one-brain system, then
  `0b001`, `0b010` and `0b100` for three brains; seven group-0 bindings; sense → passes → act;
- moving a species to an attached brain builds no stage and rewrites the masks; a brain left with
  no species is disabled and leaves every cached chain; attaching it again builds one stage with
  identical shader text;
- an unknown species is refused at `setBrain` with no stage built; a missing species fails
  `getComputeStages()` and `step()` by name;
- an attached brain destroyed since is refused when re-assigned, and no mask is rewritten;
- the gate sits beside the cadence gate, and the wrapper parses (`WgslReflect`: `brainSpecies` at
  group 0, binding 6).

The existing bind-failure test also asserts that a failed attach destroys its mask.

**GPU (A22, agents suite, Node/Dawn and Chrome):**
- prey on a `RuleBrain` (exact constants, reason 7), wolves on an `MlpBrain` (A4's tolerance,
  reason 0), deer on a second network: every agent matches its own brain, contiguous and
  interleaved;
- moving wolves back to the rule brain changes the next step's wolf actions to the constants with
  no new stage;
- zero validation errors.
- **Watched failing:** with the species gate removed, every pass writes every agent, and the
  species no longer match their brains.

## 11. Errors (verbatim)

| Where | Message |
|---|---|
| `setBrain(…, { species })` | `AgentSystem: "<name>" is not a declared species.` |
| `getComputeStages`, `step` | `AgentSystem: no brain for species "<name>"; call setBrain(brain, { species: '<name>' }).` |
| `MlpBrain.bySpecies`, `setWeights` | as listed in 7.1 (unchanged) |

`AgentSystem`'s existing errors stay: dims, the group-1 budget, a destroyed or foreign brain.

## 12. Engine touchpoints

- **Core — required: none.** Optional, landed 2026-10-12: `PROGRESSION_LIMITS` (`7f1acb0`), which
  raised each stage's group-1 budget from 3 to `maxStorageBuffersPerShaderStage − 5`.
- **Agents — required:** `AgentSystem` holds a stage and a species mask per distinct brain, and the
  `{ species }` option; `buildAgentThinkWGSL` gains the mask binding (group 0, binding 6) and the
  gate. **Without them** a
  multi-species network brain would route per lane in one pass — the 4.6–5.6× Chrome cost 7.2
  measured.

## 13. Build order

| Phase | Builds | Exit |
|---|---|---|
| **7.1** ✅ | Rows: `MlpBrain.bySpecies`, per-row `setWeights`; U21 | Committed `1dc7e74` |
| **7.2** ✅ | Rows on the GPU (A21); the cost of rows, switches and per-species passes at 2, 8 and 16 species | Both hosts green; numbers below |
| **7.3** ✅ | Per-species passes: `setBrain(brain, { species })`, a species mask per pass and the gate; U22; a size line | Committed `681ccf4`: headless and both GPU hosts green, U22 watched failing, +366 B; results below |
| **7.4** | A22 on both hosts | Both hosts green, watched failing |
| **7.5** | The guide's per-species section (the passes-or-rows rule) with an executable example; closure | Example test green; `docs/work` updated |

Step 7, part 2 (record → train → export per species, the rest timer as an observation) follows
with its own design doc, from phase 7.6.

### Deferred

- **Rows selected per agent by `genome`:** Phase 2, which will face 7.2's per-lane penalty and needs
  its own answer (for example subgroup-uniform loads behind the `subgroups` feature).
- **A fallback brain:** decided against; an `others:` option can be added later without breaking
  anything.
- **Asynchronous stage creation:** step 9.

### Phase 7.1 implementation — 2026-10-12

**Built:**
- `MlpBrain.bySpecies(engine, definition, networks)` packs one row per species into one buffer,
  in the definition's species order whatever the map's order.
- `buildMlpThinkWGSL(network, rows?)` adds a name-based row switch (`case SPECIES_WOLF: { row =
  <offset>u; }`) and a `row + ` term on every weight index.
- `setWeights(network, { species })` uploads one row at `r × F × 4`.
- `speciesConstantName` moved into `agentLayout.ts` and is now shared by the think wrapper and the
  generator, so the two cannot drift apart. It is exported.
- A single-network `MlpBrain` keeps phase 6's WGSL byte for byte, pinned by a golden string.
  `AgentSystem` and core are unchanged.

**U21** (`test/agents-mlp.test.mjs`, 6 tests):
- the golden single-network WGSL;
- the table layout and row order;
- the row switch, which parses inside the full wrapper;
- the per-row upload offset;
- every refusal of §11, verbatim, before any upload;
- attachment.

Watched failing three ways: before the feature existed (5 of 6 failed; the golden passed, as it
should); with rows packed in the map's order (the order test failed); and with a row update
written at offset 0 (the upload test failed). Each mutation was restored byte for byte.

**Regression:**
- library build;
- headless agents 24 reference + 54 node:test, core, agent-kit 4/4;
- the agents GPU suite at 120/120 on Node/Dawn and in Chrome, with zero WebGPU errors.

That suite does not yet run per-species rows; that is A21, phase 7.2.

**Size** (`size.mjs`): the MLP path grew from 13,793 to **14,314 B** (+521 B, a little over the
0.3–0.5 KB estimate, mostly the verbatim errors). The whole entry is **17,827 B**, so 3,668 B of
C2's allowance remains. The rule path is 12,375 B (+6 B, the shared helper) and still carries no
MLP code.

### Phase 7.2 results — 2026-10-12: rows are correct, but the strategy needs revisiting

**A21** (agents GPU suite) passes on Node/Dawn and in Chrome, 122/122, with zero WebGPU errors:
- three species, one row each; every live agent matches its own species' network (max abs
  1.19e-7) and fails the other two, on contiguous and interleaved layouts;
- `setWeights({ species: 'wolf' })` rewrites exactly the wolf row of the GPU buffer, the other rows
  bit-exact;
- watched failing in the suite: with every row forced to 0, all 288 wolf outputs leave their own
  network and match prey's exactly.

**Cost** (`npm run spike:mlp-think`; every kernel checked against the CPU first, on all three
host/GPU pairs; 24→32→32→2, two species). Microseconds per think pass at 100,000 agents, then the
ratio to one shared network:

| Variant | NVIDIA, Node (FXC) | AMD, Node (FXC) | AMD, Chrome (DXC) |
|---|---|---|---|
| One network, every agent (reference) | 885 | 811 | 842 |
| **Rows**, species contiguous | 795 (0.90×) | 1,335 (1.65×) | 4,149 (**4.9×**) |
| Rows, species alternating | 950 (1.07×) | 2,007 (2.47×) | 4,719 (**5.6×**) |
| Rows, every agent one species | 819 (0.93×) | 1,303 (1.61×) | 3,894 (**4.6×**) |
| Switch with constant row per branch, contiguous | 778 (0.88×) | 852 (1.05×) | 4,412 (**5.2×**) |
| Switch with constant row per branch, alternating | 1,810 (2.05×) | 950 (1.17×) | 9,694 (**11.5×**) |
| **One dispatch per species** (strategy 2), contiguous | 934 (1.06×) | 1,204 (1.48×) | 1,262 (**1.5×**) |
| One dispatch per species, alternating | 1,753 (1.98×) | 1,458 (1.80×) | 1,625 (**1.9×**) |

What it shows:
- **The rows penalty is the compiler's, not the data's.** With every agent the same species, rows
  in Chrome still cost 4.6×: a row chosen per lane stops the compiler from loading weights as
  wave-uniform data, whatever the layout. On NVIDIA (FXC) rows are free; on the AMD iGPU they cost
  1.6–2.5× under FXC and 4.6–5.6× under Chrome's DXC. NVIDIA's rows ratio also varied between runs
  (1.18×, 0.92×, 0.90×).
- **Choosing the row in a switch** with a constant offset per branch recovers FXC on AMD (1.05×), but
  not Chrome (5.2×). It also inlines one network per species, so its first compile took 3.6 s on
  NVIDIA, and it doubles cost wherever wavefronts mix species.
- **One dispatch per species** pays the stage's fixed cost per species (§7's concern), yet it is the
  only variant that is never catastrophic: 1.06–1.5× with contiguous species and about 2× when
  they alternate. **In Chrome it is about 3.3× faster than rows.**
- **This reverses the premise of strategy 1 for networks.** That premise was that one stage with
  per-agent routing costs about one network. §4 and §7's analysis held for control-flow
  divergence; it missed this compiler effect. The composite (§6) routes networks the same way, so
  it inherits the same Chrome cost.
- **The penalty is real, but small in absolute terms at Herd & Wolves' scale.** At 4,096 agents, a
  think pass is tens of microseconds either way, while the 64K step is dominated by neighbour
  queries (about 60 ms). It matters for large populations and for Phase 2's per-agent rows, which
  will face the same per-lane penalty.
- **Compile times in this table depend on Dawn's and Chrome's pipeline caches.** A shader's first
  compile is the cold figure (switch 3.6 s, per-species 2.2 s on NVIDIA under FXC); repeats hit the
  cache.

**The decision this reopens** (for the user; [`work/plan.md`](./work/plan.md) §5, item 4):
- **A. Keep strategy 1:** rows, then the composite. Simplest to keep. Costs up to 5× think time in
  Chrome on integrated GPUs when networks differ by species.
- **B. Strategy 2 for brains:** one think stage per distinct brain, gated to its species
  (`setBrain(brain, { species })`, the GPU-loop design's original sketch).
  - Never catastrophic in the measurements.
  - Mixing rules and networks, and networks of different shapes, needs no WGSL composition: the
    composite and its renaming rules disappear.
  - Swapping one species' brain recompiles only that stage, a smaller stall for step 10.
  - It needs an `AgentSystem` change (several think stages in the chain).
  - Rows (`MlpBrain.bySpecies`) become redundant. They can be removed while unreleased, or kept as
    measured groundwork for Phase 2.
- **C. Strategy 1 with a subgroup-uniform fast path** (`subgroupBroadcastFirst`) behind the
  `subgroups` feature. Unmeasured, a dual code path, and unavailable under FXC.

**Rows or passes at 8 and 16 species** (the user's condition for keeping rows; contiguous blocks, a
different network per species, every agent checked against its own network first). Microseconds per
think, rows / passes:

| Species, agents | NVIDIA, Node | AMD, Node     | AMD, Chrome   |
|-----------------|--------------|---------------|---------------|
| 8, 20,000       | 212 / 568    | 360 / 754     | 1,127 / 1,050 |
| 8, 100,000      | 909 / 1,188  | 1,401 / 3,498 | 4,055 / 3,250 |
| 16, 20,000      | 219 / 1,122  | 354 / 1,425   | 975 / 1,788   |
| 16, 100,000     | 950 / 1,638  | 1,417 / 6,054 | 4,417 / 6,463 |

Rows are flat in species count (one pass); passes grow with it (one record read per species). Rows
win at 16 species everywhere, and at 8 everywhere except Chrome at 20,000–100,000 agents, where
passes are 7–25% faster. **By the
user's rule both stay**, with §5's rule in the docs: passes below 8 species, rows from 8. The user
chose option B the same day; §1–13 above were rewritten for it.

### Phase 7.3 implementation — 2026-10-13

**Built** (`AgentSystem.ts`, `agentThinkShaders.ts`; core unchanged):
- `setBrain(brain, { species })` takes one name or a list; without it, every species. Names are
  resolved first, so an unknown species is refused before anything changes. A brain new to the
  system is validated and gets its pass before any species is assigned to it.
- One think pass per attached brain, in attachment order, each with its own 16-byte species mask
  (§4, changed from the reviewed slot table). Only the masks of brains whose species set changed
  are rewritten. A brain left with no species is disabled, leaves every chain, and its mask is
  destroyed.
- The wrapper's gate, `if ((brainSpecies.x & (1u << a.ids.x)) == 0u) { return; }`, follows the
  cadence gate; a brain's own WGSL is unchanged.
- The missing-brain error is now per species (§11). It replaces the GPU-loop design's
  `AgentSystem: no brain — call setBrain() …` ([`agents-gpu-loop-design.md`](./agents-gpu-loop-design.md)
  §11), including for a system with no brain at all.
- A failed attach (a binding error) destroys its mask instead of keeping it until `destroy()`.

**U22** (5 tests, §10). Watched failing four ways, each mutation restored byte for byte:
- before the feature existed: 5 of 5 failed;
- masks never written: 4 failed (3 U22 tests and U8's setup-write check);
- a brain with no species never retired: 2 failed (1 U22 test and U9/U8's replacement check);
- a failed attach keeping its mask: the bind-failure test failed.

**Existing tests updated:**
- group-0 bindings: 6 → 7;
- U8 and the dirty-upload test count `setBrain`'s one mask write;
- the instances tests find the highlight buffer by label;
- A16's ungated dispatch binds a mask with every species.

**Regression:**
- library build with `tsc`;
- headless agents: the reference test plus 59 node:test, core, agent-kit 4/4;
- the agents GPU suite at 122/122 on Node/Dawn and in Chrome, with zero WebGPU errors.

**Cost with the production gate.** The spike's per-species passes are now built as `AgentSystem`
builds them: one single-network `MlpBrain` per species, each pass gated by its own mask. Every
kernel was checked against the CPU first (30/30 on Node across both adapters, 15/15 in Chrome).
Microseconds per think at 100,000 agents, then the ratio to one network:

| Variant | NVIDIA, Node | AMD, Node | AMD, Chrome |
|---|---|---|---|
| One network, every agent (reference) | 926 | 827 | 837 |
| Passes, 2 species contiguous | 967 (1.04×) | 1,171 (1.42×) | 1,271 (1.52×) |
| Passes, 2 species alternating | 1,835 (1.98×) | 1,384 (1.67×) | 1,639 (1.96×) |

Rows ÷ passes (below 1, rows faster; 1,000 / 20,000 / 100,000 agents):

| Species | NVIDIA, Node | AMD, Node | AMD, Chrome |
|---|---|---|---|
| 2 | 0.70 / 0.79 / 0.82 | 1.23 / 1.37 / 1.13 | 0.83 / 2.91 / 3.47 |
| 8 | 0.21 / 0.37 / 0.75 | 0.27 / 0.54 / 0.44 | 0.19 / 1.07 / 1.34 |
| 16 | 0.10 / 0.19 / 0.57 | 0.13 / 0.31 / 0.25 | 0.10 / 0.57 / 0.71 |

These match 7.2's within run-to-run noise. The rule stands: passes below 8 species, rows from 8.

**Size** (`size.mjs`, min+gzip):

| Import set | After 7.1 | After 7.3 | Change |
|---|---|---|---|
| Rule path (no MLP code) | 12,375 B | 12,741 B | +366 B (estimate 0.3–0.5 KB) |
| MLP path | 14,314 B | 14,655 B | +341 B |
| Whole entry | 17,827 B | 18,174 B | +347 B |

The whole entry is 3,321 B within C2's 21,495 B allowance.
