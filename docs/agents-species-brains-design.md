# NullGraph Agents — C2 step 7, part 1: per-species brains — Technical Design

**Parent:** [`agents-design.md`](./agents-design.md) §13, C2 step 7.
**Decided with the user, 2026-10-12** ([`work/plan.md`](./work/plan.md) §5):
- one network per species (option B), chosen in one think stage (strategy 1);
- **weight-table rows first, then a composite brain** for mixing;
- no fallback brain;
- core limit negotiation (landed, §12).

**Builds on:** the brain seam ([`agents-gpu-loop-design.md`](./agents-gpu-loop-design.md) §9.1);
`MlpBrain` and its measurements ([`agents-mlp-design.md`](./agents-mlp-design.md), A4, phase 6.3).
**Status:** ✍ designed 2026-10-12, ready for phase 7.1. No agents code yet.
**Target:** `src/extras/agents/` (`MlpBrain.bySpecies`, a new `speciesBrain.ts`), headless and GPU
tests, the API guide. `AgentSystem`'s brain seam and core are unchanged by this part.

---

## 1. Purpose

Each species runs its own brain inside the existing single think dispatch. There are two
mechanisms, for two needs:

- **Weight-table rows** (`MlpBrain.bySpecies`): one network per species, all the same shape, as
  rows of one weight buffer. Each agent reads its species' row. This is what step 7 needs: a
  trained prey network and a trained wolf network running together.
- **A composite brain** (`SpeciesBrain`): different kinds of brain per species, such as prey on
  rules and wolves on a network, or networks of different shapes. This is what evaluation and step
  10's live Rules/Network switch need.

**The single most important pre-existing fact:** the think wrapper already hands every brain the
agent's species, in `think(i, a.ids.x, state.tick)` (fact 2). It also generates a `SPECIES_<NAME>`
constant per species (fact 3). Routing by species needs no new plumbing; both mechanisms only
change the WGSL a brain contributes.

**The second fact:** the parent design reserved this seam. *"The record's `genome` word will select
a row of a weight table; Phase 1 has one row"* (fact 13). Rows by species are that table with one
row per species; Phase 2 selects rows per agent.

### Non-goals

- **Per-agent weights or genomes:** Phase 2. Rows are selected by species only here.
- **A fallback brain for unlisted species:** decided against. Every species is named.
- **Species added or removed at run time:** the definition is fixed for the system's life.
- **Compiling a swapped brain asynchronously:** deferred to step 9 ([`work/plan.md`](./work/plan.md)
  §5, item 3).
- **Recording, training and export:** step 7, part 2, with its own design.

### Design principles applied

- **Data-oriented first** (roadmap constraint 1): same-shape networks differ only in numbers, so
  they are rows of one buffer, not branches of code.
- **Decorate the contract; don't fork it** ([`design-principles.md`](./design-principles.md) §4):
  the composite renames each member's `think` and wraps them, guarded by the existing signature
  check, as `ShadowSystem.wrapLighting` wraps `getIncidentLight`.
- **Fail loud at build time** (§2): every refusal happens at construction or `setBrain`, naming
  the species.
- **Everything expensive at build time** (§11): composition at construction, the compile at
  `setBrain`, nothing per frame.
- **Size by demand** (§10): rows per declared species; the composite includes only distinct
  members.

## 2. Architecture

The frame is unchanged:
`engine.render([...agents.getComputeStages(), ...visibility, scenePass])`, with the step chain
grid → queries → sense → **think** → act → app stages → tick, then instances. Only the WGSL
inside think differs:

```
think stage (one dispatch, one thread per agent)
├─ rows:       MlpBrain.bySpecies ─ row = rowOf(species) ─ the same loops, that row's weights
│              mlpWeights: [ row 0 (prey) │ row 1 (wolf) │ … ]   one storage buffer
└─ composite:  SpeciesBrain ─ switch species ─┬─ SPECIES_PREY → brain0_think (a RuleBrain)
                                              └─ SPECIES_WOLF → brain1_think (an MlpBrain, rows or one network)
```

## 3. Integration facts (verified against this repository, 2026-10-12)

| # | Claim | Where | Consequence |
|---|---|---|---|
| 1 | `setBrain` builds the think stage from `brain.wgsl` and binds `brain.buffers` as group 1 | `AgentSystem.setBrain` (`AgentSystem.ts:163`), `AgentSystem.stage` (`:126`) | Both mechanisms are ordinary brains; `AgentSystem` needs no change |
| 2 | The wrapper calls `think(i, a.ids.x, state.tick)`: the species index is an argument | `buildAgentThinkWGSL` (`agentThinkShaders.ts:51`) | A brain can route by species with no new binding |
| 3 | The wrapper emits `const SPECIES_<NAME> = <index>u;` per species | `speciesConstants` (`agentThinkShaders.ts:14`) | Routing can follow species *names* as the system defines them |
| 4 | Species names are identifiers; at most 16 species | `IDENT` (`agentLayout.ts:215`), `MAX_SPECIES` (`:38`) | Every constant is valid WGSL; a table has ≤ 16 rows |
| 5 | `MlpBrain` packs one network into vec4-aligned floats with compile-time offsets | `packMlpWeights`, `mlpLayout` (`mlpMath.ts`), `buildMlpThinkWGSL` (`mlpShaders.ts:4`, indices at `:18`, `:20`) | A row is `mlpLayout(net).floats` floats; selecting one adds a `row` term to each index |
| 6 | `setWeights` validates, then makes one `writeBuffer` at offset 0 | `MlpBrain.setWeights` (`MlpBrain.ts:38`, `:46`) | A row update is one `writeBuffer` at that row's offset |
| 7 | The weights must fit `min(maxBufferSize, maxStorageBufferBindingSize)` | `MlpBrain` constructor (`MlpBrain.ts:18`) | 16 rows of the largest network (70,464 B) is 1.1 MB: far inside |
| 8 | A brain's group-1 budget is `maxStorageBuffersPerShaderStage − 5` | `AgentSystem.checkExtraBuffers` (`AgentSystem.ts:156`) | Rows use 1 buffer whatever the species count; a composite sums its members' |
| 9 | Core raises `maxStorageBuffersPerShaderStage` to the adapter's value when it creates the device | `PROGRESSION_LIMITS` (`WebGPUContext.ts:27`, applied at `:240`); C7 | The budget is 11 on the measured GPUs (16 − 5), 3 on a default device |
| 10 | The think entry's signature is checked by one regex, comments stripped | `assertThinkSignature` (`RuleBrain.ts:18`) | The composite finds and renames the entry with the same check: the marker guard of principle §4 |
| 11 | `setBrain` asks `brain.assertDevice?.(device)` before building a stage | `AgentSystem.setBrain` (`AgentSystem.ts:166`) | The composite delegates to members, refusing destroyed or foreign brains early |
| 12 | The think stage's fixed cost (gate, record read) is 6.3–6.6 ns per agent on the AMD iGPU, 0.8 ns on the NVIDIA GPU | [`agents-mlp-design.md`](./agents-mlp-design.md) phase 6.3 table | One dispatch beats one per species, especially on bandwidth-bound GPUs |
| 13 | The parent reserved per-agent weight-table rows for Phase 2 | [`agents-design.md`](./agents-design.md) §9.3, "Phase 2 seam" | Rows by species are the same seam, one row per species |
| 14 | The site demo keeps each species in one index range (wolves first) | `agentName` (site `HerdWolvesGPU/index.ts:740`) | The composite's lockstep cost (§7) stays low in the demo |

## 4. Data model

**The weight table.** For a network whose packed size is `F` floats (`mlpLayout(net).floats`,
always a multiple of 4), a table for `S` species is `S × F` floats in one `STORAGE | COPY_DST`
buffer. Row `r` starts at float `r × F`, in the definition's species order. In WGSL, the row's
offset in vec4 units is `row × F/4`, added to every index of fact 5.

| Network | Row (`F` × 4 B) | 2 species | 16 species (the maximum) |
|---|---|---|---|
| 24→32→32→2 (Herd & Wolves) | 7,952 B | 15,904 B | 127,232 B |
| 128→64→64→64→16 (the largest) | 70,464 B | 140,928 B | 1,127,424 B |

**Row selection by name.** The generated code maps the system's species constants to rows:
`switch species { case SPECIES_PREY: { row = 0u; } case SPECIES_WOLF: { row = 1u; } default: {} }`,
then multiplies by `F/4`. That keeps rows correct even if the system's definition orders species
differently. A name the system lacks fails to compile, not silently.

**The composite** holds no data. Its buffers are its members' buffers, concatenated in member order.

## 5. Mechanism A: weight-table rows (`MlpBrain.bySpecies`)

```ts
const brain = MlpBrain.bySpecies(engine, definition, { prey: preyNet, wolf: wolfNet });
agents.setBrain(brain);                              // one compile
brain.setWeights(retrainedWolf, { species: 'wolf' }); // one row, one writeBuffer, no compile
```

- **Every declared species gets a network**, and every network has the same shape and
  activations: one kernel serves all rows. Different shapes need the composite (§6).
- **Generated code:** the single-network loops with a `row` term (§4). Every lane runs the same
  instructions; only addresses differ by species. There is no branch divergence, whatever the
  agent order.
- **`setWeights(network, { species })`** validates the whole network, then uploads one row. On a
  per-species brain, `setWeights(network)` without a species is refused; on a single-network
  brain, a species is refused. Everything else about `MlpBrain` holds: a copy is uploaded, the
  buffer and stage stay the same, `destroy()` is idempotent, reason 0.
- **A single-network `MlpBrain` is unchanged,** and its WGSL stays byte for byte as today, so A4 and
  the guide's example keep testing the same code.

## 6. Mechanism B: the composite (`SpeciesBrain`)

```ts
agents.setBrain(new SpeciesBrain(definition, { prey: rules, wolf: wolfNet }));
```

- **Every declared species gets exactly one brain.** One brain can serve several species and is
  included once. Members are stage brains (`RuleBrain`, `MlpBrain` single or per-species, custom);
  a declared `dims` must equal the definition's. A `SpeciesBrain` member is refused.
- **The composite owns nothing.** `destroy()` releases nothing, and `assertDevice` asks every
  member.
- **Composition, the minimum that keeps members independent:**
  1. Each member's entry is renamed `brainK_think`, located by the signature check (fact 10).
  2. Each member's `@group(1) @binding(j)` moves up by the buffer count of the members before it.
     An index outside the member's own buffers is refused.
  3. Each member's group-1 variables are renamed `brainK_<name>`: whole words, not after `.`,
     within that member's text. So two `MlpBrain`s, which both declare `mlpWeights`, compose.
  4. Any other top-level name repeated across members is refused, naming both species: a helper
     `fn`, `const`, `struct`, `alias` or `override`, found by a brace-depth-0 scan with comments
     removed.
  5. `fn think` switches on the system's constants:
     `case SPECIES_PREY: { brain0_think(i, species, tick); }`, with species sharing a member
     listed in one case. `default: {}` is required by WGSL and unreachable for declared species.
- **One distinct member costs nothing:** its WGSL is returned unchanged.

## 7. Cost: lockstep execution, and what phase 7.2 / 7.4 measure

A GPU runs 32–64 invocations in lockstep. Rows never diverge: every lane runs the same loops. The
composite does diverge: a wavefront holding several species runs each member's branch in turn, so
it pays for each. Keeping each species in one contiguous index range limits that to wavefronts at
range boundaries; the Herd demo already does this (fact 14). Interleaved species are correct, at
about the sum of the members' costs.

The rejected alternative, one dispatch per species, never diverges. But it pays the stage's fixed
cost per species, which is most of the cost on the AMD iGPU (fact 12).

`npm run spike:mlp-think` gains these cases, each checked against the CPU first, on the three
host/GPU pairs of phase 6.3:
- rows, contiguous and interleaved;
- the composite, contiguous and interleaved;
- one shared network;
- a per-species-dispatch reference.

## 8. Lifecycle and updates

- A brain belongs to its engine's device; members are checked when the composite is attached.
- Replacing a composite or a rows brain builds one new stage (one compile) and disables the old.
- `setWeights` on a rows brain, or on any `MlpBrain` member of a composite, uploads without a
  rebuild.
- Destroy order is as today: the system, or the brain's replacement, first; then the brain and its
  members.

## 9. Bundle

`MlpBrain.bySpecies` lives on `MlpBrain`, so it costs the MLP path only (estimate 0.3–0.5 KB).
`speciesBrain.ts` is a separate module (estimate 0.5–0.9 KB). Rule-only apps pay 0 B for both;
`size.mjs` gains lines for each and keeps its rule-path check. The whole entry must stay within
C2's allowance: at most 21,495 B min+gzip ([`work/plan.md`](./work/plan.md) §5, item 2).

## 10. Test suite

**Headless (in `npm run test:agents`):**
- **U21, rows** (`test/agents-mlp.test.mjs`):
  - the table layout (`S × F` floats, rows in species order, zero padding per row);
  - the generated WGSL parses inside the full wrapper, with the name-based row switch;
  - a row update is one `writeBuffer` at `r × F × 4`;
  - a single-network brain's WGSL is unchanged byte for byte;
  - every refusal in §11.
- **U22, composite** (`test/agents-species.test.mjs`):
  - the composed think parses inside the full wrapper;
  - case lists, entry renames, binding shifts and variable renames as specified, including two
    `MlpBrain`s at bindings 0 and 1;
  - a single distinct member is returned byte for byte;
  - every refusal in §11 happens before any GPU work;
  - the rewriter's own hazards (principle §15): `fn think(` inside a comment, attribute order
    `@binding(0) @group(1)`, nested block comments, a `.name` member access sharing a variable's
    name.

**GPU (agents suite, Node/Dawn and Chrome):**
- **A21, rows:** prey and wolf networks in one brain. Each species' actions equal `mlpForward` of
  its own network on its GPU observations (A4's tolerance), on contiguous and interleaved layouts.
  `setWeights({ species: 'wolf' })` changes only wolf rows; prey rows stay bit-exact. Zero
  validation errors. Watched failing: with the row term multiplied by 0, the wolves match the prey
  network.
- **A22, composite:**
  - prey on a `RuleBrain` (exact constants, reason 7) and wolves on an `MlpBrain` (A4's tolerance,
    reason 0);
  - two single-network `MlpBrain`s: each species matches its own network;
  - a per-species `MlpBrain` as a member;
  - a swap builds one stage; a member's `setWeights` applies with no rebuild.

  Watched failing: with the binding shift disabled, both members read binding 0; with one case
  dropped, that species keeps a sentinel.

## 11. Errors (verbatim)

| Where | Message |
|---|---|
| `MlpBrain.bySpecies` | `MlpBrain: no network for species "<name>"; give one per species.` |
| `MlpBrain.bySpecies` | `MlpBrain: "<name>" is not a declared species.` |
| `MlpBrain.bySpecies` | `MlpBrain: the network for "<name>" differs in shape from "<first>" at layer <k>; use SpeciesBrain for different shapes.` |
| `setWeights` | `MlpBrain: this brain holds one network per species; call setWeights(network, { species }).` |
| `setWeights` | `MlpBrain: setWeights({ species }) needs a brain from MlpBrain.bySpecies.` |
| `SpeciesBrain` | `SpeciesBrain: species "<name>" has no brain; list every species.` |
| `SpeciesBrain` | `SpeciesBrain: "<name>" is not a declared species.` |
| `SpeciesBrain` | `SpeciesBrain: the brain for "<name>" is not a stage brain.` |
| `SpeciesBrain` | `SpeciesBrain: the brain for "<name>" maps <o> → <a>; the definition needs <obsDim> → <actDim>.` |
| `SpeciesBrain` | `SpeciesBrain: brains cannot nest; list the inner brain's species directly.` |
| `SpeciesBrain` | `SpeciesBrain: the brain for "<name>" declares @binding(<j>) but supplies <n> buffer(s).` |
| `SpeciesBrain` | `SpeciesBrain: the brains for "<a>" and "<b>" both declare <name>; rename one, or combine them into one RuleBrain.` |
| `SpeciesBrain` | `SpeciesBrain: the brain for "<name>" must define fn think(i: u32, species: u32, tick: u32).` |

`AgentSystem`'s own errors are unchanged: dims, the group-1 budget, and a destroyed or foreign
brain.

## 12. Engine touchpoints

- **Core — required: none.**
- **Core — optional, landed 2026-10-12 at the user's request:** `PROGRESSION_LIMITS` raises
  `maxStorageBuffersPerShaderStage`, `maxStorageBufferBindingSize` and `maxBufferSize` to the
  adapter's values on devices NullGraph creates. That removed `AgentSystem`'s old cap of 3, so a
  composite may bind 11 members' buffers on the measured GPUs. Core grew 89 B min+gzip.
  - Evidence: headless H8 (watched failing); GPU C7 (watched failing: 8 against the adapter's 16);
    every headless suite and every GPU suite green on both hosts; the site demo driven at 4K and 64K
    agents with no errors.
  - **Without it:** 3 group-1 buffers on every device, so at most 3 network members in a composite.
    Rows are unaffected (1 buffer).
- **Agents — required:** `MlpBrain.bySpecies` and a row-aware generator in `mlpShaders.ts`; the new
  `speciesBrain.ts`. `AgentSystem` unchanged.

## 13. Build order

| Phase | Builds | Exit |
|---|---|---|
| **7.1** | Rows: `MlpBrain.bySpecies`, the row-aware generator, per-row `setWeights`; U21; a size line | Headless green, U21 watched failing, size within the allowance |
| **7.2** | Rows on the GPU: A21 on both hosts; the rows cases of §7 | Both hosts green; numbers recorded here |
| **7.3** | Composite: `SpeciesBrain`; U22; a size line | Headless green, U22 watched failing |
| **7.4** | Composite on the GPU: A22 on both hosts; the composite and reference cases of §7 | Both hosts green; numbers recorded here |
| **7.5** | The guide's per-species section with an executable example of each; closure | Example test green; `docs/work` updated |

Step 7, part 2 (record → train → export per species, the rest timer as an observation) follows
with its own design doc, from phase 7.6.

### Deferred

- **Rows selected per agent by `genome`:** Phase 2. Same table, a different row index.
- **Packing several composite members' weights into one buffer:** only if a composite ever needs
  more network members than the device's group-1 budget (11 on the measured GPUs, 3 on a default
  device). Same-shape networks avoid the question entirely as rows.
- **A fallback brain:** decided against; an `others:` option can be added later without breaking
  anything.
- **Asynchronous stage creation for swaps:** step 9.
