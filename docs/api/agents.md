# GPU agents — rule brains and network brains

Import from `null-graph/agents`. Debug draws, visibility, picking and text are separate opt-in
entries. This guide covers the GPU loop (C1 steps 3–4) with rule brains, `MlpBrain` (C2 step 6),
which runs a small dense network you supply, per-species brains, [a network per agent](#a-network-per-agent)
(Phase 2 step 11), [neighbour lists for step stages and custom sensors](#neighbour-lists-for-stages-and-custom-sensors)
(Phase 2 step 12), switching brains without a compile in the frame, and loading a trained network
(C2 steps 7 and 9). Training, camera eyes and a replay/rewind UI are not implemented by this API.
Training is done by tools in the repository, outside the package ([a trained network](#a-trained-network)).
An `.onnx` file is read, and ONNX Runtime brains (`OrtBrain`) run, from another entry, `null-graph/inference`
([inference.md](./inference.md)).

**Disc contacts:** [the physics guide](./physics.md) shows `DiscContacts` from the separate
`null-graph/physics` entry and `addStepStage({ name, stages })` to run its stages after movement
on every fixed step. Contacts and other app-stage groups run in registration order; register
contacts after respawn/movement stages and before stages that inspect separated positions.
The caller owns external stages and destroys contacts after the agent system stops using them.

**See it running:** the test site's Herd & Wolves demo ([nullgraph.dev](https://nullgraph.dev/),
`/dashboard/agents/herd-wolves-gpu`) runs on this API, from 4,096 to 1M agents as far as the GPU
allows. Each species switches live between the rule, its trained network on `MlpBrain` and the
same network on ONNX Runtime, and the overlay shows what each brain costs on the GPU.

## Minimal rule simulation

The host supplies an initialized `NullGraph`, camera updates and elapsed time in **seconds**.
The TypeScript example targets a browser bundler (`moduleResolution: "Bundler"`), like the demo.
This example includes the complete agent setup, frame function and teardown for an arrow view:

```ts
import type { NullGraph } from 'null-graph';
import { AgentSystem, defineAgents, Sensors, RuleBrain } from 'null-graph/agents';
import { agentArrows } from 'null-graph/debug-ui';

export function createAgentDemo(engine: NullGraph) {
    const definition = defineAgents({
        capacity: 256,
        world: { min: [-40, -40], max: [40, 40] },
        cellSize: 8,
        species: [{ name: 'scout', maxSpeed: 3, maxTurn: 2, accel: 4,
            senseRadius: 8, radius: 0.4 }],
        sensors: [Sensors.self(), Sensors.bounds()],
    });
    const agents = new AgentSystem(engine, definition, { thinkEvery: 2 });
    agents.setBrain(new RuleBrain(`
fn think(i: u32, species: u32, tick: u32) {
    setAct(i, ACT_TURN, rand01(i, tick, 0u) * 2.0 - 1.0);
    setAct(i, ACT_THROTTLE, 0.25);
    setReason(i, 1u);
}`));
    for (let i = 0; i < definition.capacity; i++) {
        agents.spawn(i, { species: 'scout',
            position: [(i % 16) * 4 - 30, 0, Math.floor(i / 16) * 4 - 30] });
    }
    agents.upload();
    const pass = engine.createPass({ name: 'Agents', isMainScreenPass: true });
    const arrows = agentArrows(engine, pass, agents);
    let destroyed = false;
    return {
        agents,
        frame(deltaSeconds: number) {
            if (destroyed) return;
            agents.advance(deltaSeconds);
            agents.upload();
            engine.render([...agents.getComputeStages(), pass]);
        },
        destroy() {
            if (destroyed) return;
            destroyed = true;
            arrows.destroy();
            agents.destroy();
        },
    };
}
```

The example constructs a sequence array per frame for clarity. A production host can cache its
render suffix/sequence for each step count instead; the agent system's chains are cached already.
The host must stop scheduling this demo before teardown or replace its active sequence afterward.

## Definition and observations

`defineAgents` validates declarations and derives the grid, species indices, observation/action
dimensions, names and WGSL constants. Capacity, world bounds, cell size and fixed-step settings
are fixed for the system's lifetime. Species movement parameters can be tuned with
`agents.setSpecies(name, partial)`; the next `upload()` uploads the dirty species table.

- `Sensors.self()`: normalized speed and energy.
- `Sensors.nearest({ name, species, k, radius?, observe? })`: up to `k` neighbors, each with local
  side/forward position, relative side/forward velocity and presence. Species filters use declared
  names. `radius` narrows the search; `observe: false` keeps the list and drops its floats.
- `Sensors.query({ name, species, k, radius? })`: a neighbour list that adds no observation floats,
  for step stages and custom sensors ([neighbour lists](#neighbour-lists-for-stages-and-custom-sensors)).
- `Sensors.bounds()`: proximity-weighted nearest-wall direction in the agent's local frame.
- Custom sensors/actions require their declared WGSL signatures; CPU mirrors are needed when
  using the CPU reference routines.

**What a custom sensor can read.** `Sensors.custom({ name, width, wgsl, mirror? })` compiles
`fn <name>(i: u32, a: Agent) -> array<f32, width>` into the sense stage.

- **Its agent and its neighbours:** the agent's own record, `a`; every `nearest` and `query`
  list, and the records the lists name ([neighbour lists](#neighbour-lists-for-stages-and-custom-sensors)).
  Its CPU `mirror(records, u32, i, out, at, definition)` is given the definition, so it can repeat
  the search.
- **No app data.** A sensor cannot bind an app buffer, so walls and other occluders are out of its
  reach, and true line of sight is not available.
- **"Can I see the nearest seeker?" needs no custom sensor.**
  `Sensors.nearest({ name: 'seeker', species: ['seeker'], k: 1 })` gives its side, forward and
  presence, and the brain can test a view cone and a range from those.

The default actions are steering: turn and throttle in `[-1, 1]`. Movement enforces speed,
acceleration and turn limits, brakes on non-finite actions, and clamps/reflects at world bounds.
These are steering agents, not a rigid-body collision solver.

Nearest queries use the bounded grid search described in [agent-grid.md](./agent-grid.md).
Use `senseRadius <= cellSize`. Dense cells can exceed `maxScanPerCell`: all indices remain
stored, but a query scans only its budget, so an agent can miss a neighbour that is in range.
Overflow counters expose that loss of perception. It does not cost determinism: reruns stay
bit-exact (see the next section).

- **Which entries a crowded cell yields:** a window of `maxScanPerCell` consecutive entries of
  the cell, which lists its agents by index. The window's start changes with the cell and the
  tick, and it wraps. Every query that scans the cell in a tick reads the same entries, and over
  many ticks every agent of the cell gets the same share of scans.
- **A window is a run of consecutive indices.** Agents spawned as one index block per species
  sit together in a cell's list. In a cell many times over its budget, a window can therefore
  hold a single species on a given tick.
- **Flocks pack far above their mean density** (Herd & Wolves overflowed in 85% of steps at the
  default 64), so size the budget or the cell for the peak, and watch `overflowSteps`.

## The frame and fixed-step paths

```text
Each step: grid → queries → sense → think (one pass per brain) → act → app stages → tick
Each frame: instance output (when enabled) → visibility/text → draw
```

`advance(deltaSeconds)` chooses zero or more fixed steps using the definition's `dt`,
`maxStepsPerFrame` and `maxFrameDelta`. Call `upload()` for dirty CPU staging records and
parameters, then render `getComputeStages()` before dependent draws/culls.

`thinkEvery` is fixed at construction. Sense and think share a cadence staggered by 64-agent
blocks; newborns get an immediate decision. Act/app/tick still run each simulation step.
Observations/actions can therefore represent an earlier decision than the current position.

`await agents.step()` uploads and submits one step, waiting for GPU completion. It is the
lockstep experiment/test path, not a replacement for the regular frame loop.

**Lockstep reruns on one device are bit-exact, with or without grid overflow.** The same
definition, spawns and brains, stepped the same number of times, give the same records,
observations, actions and reasons, word for word.

- **It holds for** the built-in sensors, and for custom sensors that read records and neighbour
  lists: the sense stage writes neither. The grid lists each cell's agents in a fixed order, and a
  crowded cell's window depends only on the cell and the tick.
- **It needs** brains and app stages that write only their own agent's rows, as the next section
  asks.
- **No cross-device guarantee is made:** float rounding differs between GPUs.

**CPU records are a staging area, not a live mirror.** After GPU movement, `agents.records`
does not contain current positions. `spawn(i, init)` and `kill(i)` dirty individual records;
`upload()` sends those edits. Use inspection for current GPU state, not a full staging re-upload.

## Rule brains and app logic

`setBrain(new RuleBrain(wgsl))` installs WGSL with the exact entry:
`fn think(i: u32, species: u32, tick: u32)`.
Use `obs(i, offset)`, `setAct(i, offset, value)` and `setReason(i, word)` with the definition's
generated constants. Every species needs a brain before `getComputeStages()` or `step()`:
`setBrain(brain)` serves all of them, and [per-species brains](#per-species-brains) split them.

`addStepStage({ name, wgsl, buffers? })` adds an app stage after act and before tick, with entry
`fn step(i: u32, tick: u32)`. App stages see the same tick
as that step's brain, and read its [neighbour lists](#neighbour-lists-for-stages-and-custom-sensors).
Auxiliary group-1 buffers and brain objects remain caller-owned.

For peer-dependent mechanics, separate detection and mutation dispatches: the Herd & Wolves
demo first reads peer positions and marks its own flags, then consumes them for respawn/rest.
Do not write peer records concurrently and assume another invocation sees a consistent result.

## Neighbour lists for stages and custom sensors

Every `nearest` declaration fills, each step, a list of up to `k` neighbour indices per agent in
`agents.neighbours`. `Sensors.query` declares a list that no brain observes: it adds no floats to
the observation row, so `obsDim`, `obsNames` and any network trained on them stay as they are.
Step stages and custom sensors read the lists.

```ts
import type { NullGraph } from 'null-graph';
import { AgentSystem, RuleBrain, defineAgents, nearestNeighbours, Sensors } from 'null-graph/agents';

export function createCrowd(engine: NullGraph) {
    const found = new Uint32Array(8);   // the mirror's scratch, reused for every agent
    const definition = defineAgents({
        capacity: 1024,
        world: { min: [-40, -40], max: [40, 40] },
        cellSize: 8,
        species: [{ name: 'walker', maxSpeed: 1.5, maxTurn: 3, accel: 3,
            senseRadius: 8, radius: 0.3 }],
        sensors: [
            Sensors.self(),
            Sensors.bounds(),
            // Up to 8 agents within 1.2 m, nearest first: indices only, no observation floats.
            Sensors.query({ name: 'contacts', k: 8, radius: 1.2 }),
            // One float from that list: the share of its 8 slots that are filled.
            Sensors.custom({ name: 'crowd', width: 1, wgsl: `
fn crowd(i: u32, a: Agent) -> array<f32, 1> {
    var filled = 0u;
    for (var r = 0u; r < 8u; r++) {
        if (neighbours[i * NEIGHBOURS_PER_AGENT + NEIGHBOURS_CONTACTS + r] != NO_AGENT) { filled++; }
    }
    return array<f32, 1>(f32(filled) / 8.0);
}`,
                // The CPU twin, for tests and recordings: the same search, through the definition.
                // nearestMasks holds a species mask per nearest or query declaration: [0] is contacts.
                mirror(records, u32, i, out, at, def) {
                    out[at] = nearestNeighbours(def, { data: records, u32 }, i,
                        def.nearestMasks[0], 1.2, 8, found) / 8;
                } }),
        ],
    });
    // definition.obsDim is 5: self (2), bounds (2) and crowd (1). The list adds none.
    const agents = new AgentSystem(engine, definition, { thinkEvery: 2 });
    // Away from walls, and slower the more crowded: crowd is read like any observation.
    agents.setBrain(new RuleBrain(`
fn think(i: u32, species: u32, tick: u32) {
    setAct(i, ACT_TURN, -2.0 * obs(i, OBS_BOUNDS + BOUNDS_SIDE));
    setAct(i, ACT_THROTTLE, 1.0 - obs(i, OBS_CROWD));
    setReason(i, 1u);
}`));
    // Per agent, how many agents on its list its body overlaps. The app owns this buffer.
    const touching = engine.device.createBuffer({ size: definition.capacity * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    agents.addStepStage({ name: 'touching', buffers: [touching], wgsl: `
@group(1) @binding(0) var<storage, read_write> touching: array<u32>;
fn step(i: u32, tick: u32) {
    let me = agents[i].posRadius;
    var overlaps = 0u;
    for (var r = 0u; r < 8u; r++) {
        let j = neighbours[i * NEIGHBOURS_PER_AGENT + NEIGHBOURS_CONTACTS + r];
        if (j == NO_AGENT) { break; }
        let other = agents[j].posRadius;
        if (distance(other.xz, me.xz) < other.w + me.w) { overlaps++; }
    }
    touching[i] = overlaps;
}` });
    for (let i = 0; i < definition.capacity; i++) {
        // A 32 × 32 block, 1 m apart: within 1.2 m, 4 agents inside it, 3 on an edge, 2 at a corner.
        agents.spawn(i, { species: 'walker', heading: i * 2.4,
            position: [(i % 32) - 15.5, 0, Math.floor(i / 32) - 15.5] });
    }
    agents.upload();
    return {
        agents,
        touching,
        destroy() {
            agents.destroy();   // first: its step stage binds touching
            touching.destroy();
        },
    };
}
```

Render it with the same frame function as the rule example. `touching` stays on the GPU, for a
later stage or a draw to read.

- **Declaring a list:**
  - `Sensors.query({ name, species, k, radius })` is `Sensors.nearest({ ..., observe: false })`.
    `species` takes declared names (every species when omitted), `k` is 1–8 (default 1), and
    `radius` is in `(0, cellSize]` (default `cellSize`).
  - An observed `Sensors.nearest` takes `radius` too. It limits the search, and the 5 floats still
    use `senseRadius`: a neighbour is seen within `min(senseRadius, radius)`, and its position is
    still divided by `senseRadius`.
  - A brain still needs something to read: a definition whose every sensor has `observe: false` is
    refused with `AgentSystem: declare at least one observed sensor — a brain needs something to
    read.` A radius out of range is refused with `AgentSystem: sensor "<name>" radius <r> must be
    in (0, cellSize <cellSize>].`
- **Reading a list in WGSL:** for a declaration named `contacts`, agent `i`'s list is
  `neighbours[i * NEIGHBOURS_PER_AGENT + NEIGHBOURS_CONTACTS + r]`, for `r` from 0 to `k - 1`.
  - It holds live agents other than `i`, of the declared species, within the radius (inclusive),
    nearest first with ties to the lower index, then `NO_AGENT` in every unused slot.
    `agents[j]` is agent `j`'s record.
  - Names fold to upper case: `contacts` gives `NEIGHBOURS_CONTACTS`, and two names that fold to
    the same constant are refused.
  - Observed lists are read the same way, through their own `NEIGHBOURS_<NAME>`.
- **When a list is read:** each step builds its lists first, from the positions the step starts
  with (grid → queries → sense → think → act → app stages).
  - A step stage reads them on every step, whatever `thinkEvery`, after act has moved the agents
    by one step (here at most 1.5 m/s × 1/60 s = 2.5 cm). An agent that a stage kills stays on
    other agents' lists until the next step.
  - A custom sensor runs in the sense stage, on the brain's cadence: here, every other step.
  - A brain reads observations only. To give it something from a list, reduce the list to a few
    floats in a custom sensor, as `crowd` does.
- **The CPU mirror:** `nearestNeighbours(def, { data: records, u32 }, i, mask, radius, k, out)`
  is the reference search, and `def.nearestMasks` holds the species mask of each `nearest` and
  `query` declaration, in declaration order. That search equals the GPU's list while no cell holds
  more than `maxScanPerCell` agents. Past that the GPU scans a window of the cell (see
  [definition and observations](#definition-and-observations)), and `gridNearestNeighbours`
  ([agent-grid.md](./agent-grid.md)) mirrors that.
- **Inspection:** `inspect(i).neighbours` holds `{ sensor: 'contacts', indices }` beside the
  observed lists, empty slots dropped.
- **Cost:** each list is one more query dispatch per step (the search an observed list runs) and
  `4·k` bytes per agent in `agents.neighbours`. A query-only list adds nothing to sense or think.
- **Not a contact solver:** a list finds overlaps; it does not separate them. In a pile of 256
  agents, position corrections over an 8-neighbour list still left 86% of them overlapping a
  neighbour by more than 5% of their summed radii after 180 steps, on every measured host; two
  correction passes over the whole grid neighbourhood left 63% (`npm run spike:s8`).

## Network brains: MlpBrain

`new MlpBrain(engine, network)` runs a small dense network as the think stage. The loop, the
cadence and the act stage are the same as with a rule brain; only the observation-to-action
calculation changes. It runs the weights you give it and does not learn. The example below uses
hand-set weights; [a trained network](#a-trained-network) comes from a file.

```ts
import type { NullGraph } from 'null-graph';
import { AgentSystem, MlpBrain, defineAgents, mlpForward, Sensors, type MlpNetwork } from 'null-graph/agents';

// Inputs in the definition's observation order: self.speed, self.energy, bounds.side,
// bounds.forward. Outputs in its action order: turn, throttle. Hand-set, not trained.
export const wallAvoider: MlpNetwork = [{
    input: 4, output: 2, activation: 'tanh',
    weights: new Float32Array([
        0, 0, -3, 0,   // turn     = tanh(-3 · bounds.side): away from the nearest wall
        0, 0, 0, -2,   // throttle = tanh(1 - 2 · bounds.forward): slower when it is ahead
    ]),
    biases: new Float32Array([0, 1]),
}];

// The CPU reference, for a wall to one side and none ahead: turn ≈ -0.905, throttle ≈ 0.762.
export const sample = mlpForward(wallAvoider, new Float32Array([0, 1, 0.5, 0]));

export function createWallAvoiders(engine: NullGraph) {
    const definition = defineAgents({
        capacity: 1024,
        world: { min: [-40, -40], max: [40, 40] },
        cellSize: 8,
        species: [{ name: 'scout', maxSpeed: 3, maxTurn: 2, accel: 4,
            senseRadius: 8, radius: 0.4 }],
        sensors: [Sensors.self(), Sensors.bounds()],
    });
    const agents = new AgentSystem(engine, definition);
    const brain = new MlpBrain(engine, wallAvoider);   // validates, copies and uploads once
    agents.setBrain(brain);                            // compiles the think stage
    for (let i = 0; i < definition.capacity; i++) {
        agents.spawn(i, { species: 'scout', heading: i * 2.4,
            position: [(i % 32) * 2.4 - 37, 0, Math.floor(i / 32) * 2.4 - 37] });
    }
    agents.upload();
    return {
        agents,
        // Same shape, new numbers: one upload and no recompile; later steps use them.
        setTurnGain(gain: number) {
            brain.setWeights([{ ...wallAvoider[0],
                weights: new Float32Array([0, 0, -gain, 0, 0, 0, 0, -2]) }]);
        },
        destroy() {
            agents.destroy();   // first: its think stage binds the brain's weights
            brain.destroy();
        },
    };
}
```

Render it with the same frame function as the rule example: `advance`, `upload`, then
`getComputeStages()` before the draws.

- **The network:** 1–4 layers of `{ input, output, activation, weights, biases }`. Weights are
  row-major `[output][input]`; activations are `'identity'`, `'relu'`, `'tanh'` or `'sigmoid'`;
  every value must be finite. Input width ≤ 128, hidden ≤ 64, output ≤ 16. The first input must
  equal `definition.obsDim` and the last output `definition.actDim`; `setBrain` refuses a mismatch.
- **Order and meaning:** inputs follow `definition.obsNames`, outputs `definition.actNames`. No
  normalization is added, and matching widths cannot check that a feature means what the network
  expects.
- **One network for every species it serves:** identical observations give identical actions,
  whatever the species. When species need different policies, give them their own brains
  ([per-species brains](#per-species-brains)); when individual agents need their own networks of
  one shape, use [a network per agent](#a-network-per-agent).
- **Numerics:** tanh's input is clamped to ±15 (the FXC compiler behind Dawn's D3D12 backend
  returns NaN beyond about 89);
  sigmoid is evaluated stably; identity and relu outputs are not clipped. The act stage still
  clamps steering and brakes on non-finite actions. `mlpForward(network, input, out?, scratch?)` is
  the CPU reference, with the same definitions and f32 rounding, allocation-free when given its
  output and scratch arrays. GPU results agree within `1e-5 + 1e-5·|cpu|`, not bit for bit.
  That bound was measured with weights of unit scale. Rounding grows with the weights: the trained
  Herd & Wolves networks, with weights up to 178, differ by up to 4.4e-5.
- **Updates:** `brain.setWeights(network)` needs the same shape and activations. It validates the
  whole network, then uploads once into the same buffer: no new stage, no recompile, and work
  submitted afterwards uses it. A different shape needs a new brain and `setBrain`.
- **A new brain costs a compile:** `setBrain` builds a think pipeline for a brain the system does
  not run yet. For a 24→32→32→2 network that took 76 ms in Chrome on an AMD iGPU and about 1.1 s
  under Node/Dawn on an NVIDIA GPU. Prepare the brain at load instead
  ([switching brains without a hitch](#switching-brains-without-a-hitch)), or swap at a moment
  that can absorb a hitch. Moving species to a brain the system already runs compiles nothing.
- **Reasons:** an `MlpBrain` writes reason 0. Inspection shows its observations and actions, not
  a rationale.
- **Ownership:** the caller owns the brain, and it belongs to its engine's device: `setBrain`
  refuses a brain from another device or one already destroyed. Give its species another brain,
  or destroy the system, before `brain.destroy()`: its think pass binds the weights. `destroy()` is
  idempotent.
- **Cost:** one think dispatch of a 24→32→32→2 network over 100,000 agents took 0.83–0.88 ms
  (8–9 ns per agent) on the measured GPUs, and `thinkEvery: 3` cuts it to 43–57% of that
  (`npm run spike:mlp-think`). `MlpBrain` adds 1.4 KB min+gzip to an app; rule-only imports
  carry none of it.

## Per-species brains

`setBrain(brain, { species })` gives the named species their own brain, and the other species
keep theirs. Prey can keep their rules while wolves run a network, and each species can run a
network of its own shape. Each distinct brain is one think pass that skips other species' agents,
and the brain's own WGSL runs unchanged.

```ts
import type { NullGraph } from 'null-graph';
import { AgentSystem, MlpBrain, RuleBrain, defineAgents, Sensors, type MlpNetwork } from 'null-graph/agents';

export function createHerd(engine: NullGraph) {
    const definition = defineAgents({
        capacity: 1024,
        world: { min: [-40, -40], max: [40, 40] },
        cellSize: 8,
        species: [
            { name: 'prey', maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 8, radius: 0.4 },
            { name: 'wolf', maxSpeed: 4, maxTurn: 3, accel: 5, senseRadius: 8, radius: 0.6 },
        ],
        sensors: [Sensors.self(),
            Sensors.nearest({ name: 'wolf', species: ['wolf'], k: 1 }),
            Sensors.nearest({ name: 'prey', species: ['prey'], k: 1 })],
    });
    // One rule brain for both species: prey flee the nearest wolf; wolves chase the nearest prey.
    const rules = new RuleBrain(`
fn think(i: u32, species: u32, tick: u32) {
    if (species == SPECIES_WOLF) {
        setAct(i, ACT_TURN, sign(obs(i, OBS_PREY_0 + NEAR_SIDE)));
        setAct(i, ACT_THROTTLE, 1.0);
        setReason(i, 2u);
    } else {
        let near = obs(i, OBS_WOLF_0 + NEAR_PRESENT);
        setAct(i, ACT_TURN, -sign(obs(i, OBS_WOLF_0 + NEAR_SIDE)) * near);
        setAct(i, ACT_THROTTLE, 0.4 + 0.6 * near);
        setReason(i, 1u);
    }
}`);
    // A smooth chase for wolves, hand-set, not trained:
    // turn = tanh(3 · prey[0].side), throttle = tanh(1).
    const weights = new Float32Array(2 * definition.obsDim);
    weights[definition.obsNames.indexOf('prey[0].side')] = 3;
    const chase: MlpNetwork = [{ input: definition.obsDim, output: 2, activation: 'tanh',
        weights, biases: new Float32Array([0, 1]) }];
    const chaser = new MlpBrain(engine, chase);

    const agents = new AgentSystem(engine, definition);
    agents.setBrain(rules);                          // every species: one think pass
    agents.setBrain(chaser, { species: 'wolf' });    // wolves only: a second pass, one compile
    for (let i = 0; i < definition.capacity; i++) {
        agents.spawn(i, { species: i < 64 ? 'wolf' : 'prey', heading: i * 2.4,
            position: [(i % 32) * 2.4 - 37, 0, Math.floor(i / 32) * 2.4 - 37] });
    }
    agents.upload();
    return {
        agents,
        // The rules already have a pass: this uploads 16 bytes and compiles nothing. The chaser
        // then serves nobody and leaves the chain.
        wolvesUseRules() { agents.setBrain(rules, { species: 'wolf' }); },
        // Builds the chaser's pass again: one compile, from the same shader text as before.
        wolvesUseNetwork() { agents.setBrain(chaser, { species: 'wolf' }); },
        destroy() {
            agents.destroy();   // first: a think pass may bind the chaser's weights
            chaser.destroy();
        },
    };
}
```

- **Assignment:**
  - `setBrain(brain)` gives every species to `brain`, as before.
  - `{ species: 'wolf' }` or `{ species: ['wolf', 'bear'] }` gives only those species.
  - An unknown name is refused before anything changes: `AgentSystem: "<name>" is not a
    declared species.`
- **Every species needs a brain** before `getComputeStages()` or `step()`. Otherwise both throw
  `AgentSystem: no brain for species "<name>"; call setBrain(brain, { species: '<name>' }).`
  There is no fallback brain.
- **Compiles:**
  - A brain the system does not run yet compiles its pass, at the cost given under "A new brain
    costs a compile" above.
  - Moving species to a brain that already has a pass uploads one 16-byte mask per changed brain
    and compiles nothing.
  - A brain left with no species leaves the chain, and may then be destroyed. Giving it species
    again compiles its pass again, from identical shader text, which a pipeline cache can reuse,
    unless the brain was [prepared](#switching-brains-without-a-hitch): a prepared brain keeps its
    pass.
- **Mixing kinds and shapes:** each pass runs its own brain's code, so rule brains, networks and
  networks of different shapes mix freely. Every network still maps `obsDim` to `actDim`.
  Passes write only their own agents' actions, so their order cannot change results.
- **Many same-shape networks as rows:**
  - `MlpBrain.bySpecies(engine, definition, { prey: preyNet, wolf: wolfNet })` holds one network
    per species as rows of one weight table, read in one pass.
  - `brain.setWeights(network, { species: 'wolf' })` uploads one row.
  - It needs a network for every declared species, even one that another brain serves; any
    network of the same shape will do there.
  - A network of another shape needs its own brain.
- **Passes or rows: use passes below 8 species and rows from 8.** Measured with a 24→32→32→2
  network (`npm run spike:mlp-think`):
  - Each pass reads every agent's record, so passes grow with the number of brains. At 100,000
    agents, two species as passes cost 1.04–1.52× one shared network when each species fills
    one contiguous index block, and 1.7–2.0× when species alternate.
  - Rows cost one pass whatever the species count. But choosing a row per agent stops the
    compiler from loading weights uniformly: up to 5.6× one network in Chrome on an AMD iGPU,
    against 0.9–1.1× on an NVIDIA GPU under Node/Dawn.
  - At 16 species, rows were faster on every measured host. At 8, they were faster everywhere
    except in Chrome at 20,000–100,000 agents.
  - Spawn each species in one contiguous index block when you can.
- **Cost for one brain:** a system with one brain reads one more uniform per thinking agent than
  before.

## A network per agent

`MlpBrain.byGenome(engine, networks)` holds `N` (`networks.length >= 1`) networks of one shape as
the rows of one weight table (`brain.rowCount === N`), and each agent reads the row its record's
`genome` word names (`spawn(i, { ..., genome })`). With no `genome` given, `spawn` sets it to `0`,
and `inspect(i)` reports it.

```ts
import type { NullGraph } from 'null-graph';
import { AgentSystem, MlpBrain, defineAgents, mlpLayout, packMlpWeights, Sensors, type MlpNetwork } from 'null-graph/agents';

export function rowForGain(gain: number): MlpNetwork {
    return [{
        input: 4, output: 2, activation: 'tanh',
        weights: new Float32Array([
            0, 0, -gain, 0,   // turn     = tanh(-gain · bounds.side)
            0, 0, 0, -2,      // throttle = tanh(1 - 2 · bounds.forward)
        ]),
        biases: new Float32Array([0, 1]),
    }];
}

export function createByGenomeDemo(engine: NullGraph) {
    const definition = defineAgents({
        capacity: 256,
        world: { min: [-40, -40], max: [40, 40] },
        cellSize: 8,
        species: [{ name: 'scout', maxSpeed: 3, maxTurn: 2, accel: 4,
            senseRadius: 8, radius: 0.4 }],
        sensors: [Sensors.self(), Sensors.bounds()],
    });
    const rows = [1, 2, 3, 4].map(rowForGain);
    const floats = mlpLayout(rows[0]).floats;
    const brain = MlpBrain.byGenome(engine, rows);   // 4 rows in one weight table
    const agents = new AgentSystem(engine, definition);
    agents.setBrain(brain);                          // one cooperative think pass
    for (let i = 0; i < definition.capacity; i++) {
        // Lay rows out in 64-agent blocks: agents 0–63 share row 0, 64–127 row 1, …
        agents.spawn(i, { species: 'scout', genome: Math.floor(i / 64), heading: i * 2.4,
            position: [(i % 16) * 4 - 30, 0, Math.floor(i / 16) * 4 - 30] });
    }
    agents.upload();
    return {
        agents,
        brain,
        // One row in place: one upload, no recompile.
        setRowGain(row: number, gain: number) {
            brain.setWeights(rowForGain(gain), { row });
        },
        // Contiguous rows packed back to back: one upload for the range, no recompile.
        setPairGains(first: number, gainA: number, gainB: number) {
            const packed = new Float32Array(2 * floats);
            packMlpWeights(rowForGain(gainA), packed.subarray(0, floats));
            packMlpWeights(rowForGain(gainB), packed.subarray(floats));
            brain.setRows(packed, { first });
        },
        destroy() {
            agents.destroy();
            brain.destroy();
        },
    };
}
```

- **Construction and updates:**
  - `MlpBrain.byGenome(engine, networks)` checks that every network matches row `0`'s layer widths
    and activations, packs row `r` at `r * mlpLayout(network).floats`, and uploads the table once.
    `brain.rowCount` is `networks.length` (`1` on a single-network `MlpBrain`, the species count on
    `MlpBrain.bySpecies`).
  - `brain.setWeights(network, { row })` validates and uploads one row at `row * floats * 4` bytes.
    Calling `setWeights(network)` without `{ row }` on a `byGenome` brain is refused.
  - `brain.setRows(packed, { first })` uploads one or more whole rows packed back to back with
    `packMlpWeights`, starting at row `first` (`0` when omitted), in one `writeBuffer` call. Every
    weight must be finite.
  - Neither update creates a buffer or recompiles a pass. The row count is fixed for the brain's
    lifetime; changing `rowCount` or the network shape requires a new brain (`prepareBrain`
    compiles it ahead).
- **The 64-agent block layout and the staged kernel:**
  - The think pass runs in workgroups of 64 agents (`0–63`, `64–127`, …). When a block has at least
    one thinking agent, all 64 lanes cooperatively copy the block's first agent's row
    (`agents[block * 64].genome`) into 8 KB of workgroup memory (`512 vec4<f32>`) a tile at a time
    and evaluate from there. Any lane in the block whose `genome` names another row falls back to
    reading its own row from storage. With `thinkEvery > 1`, a resting block with no newborn exits
    before staging anything.
  - **Spawn agents that share a row in contiguous 64-agent blocks when you can.** Every layout gives
    the same actions as a single-network `MlpBrain` holding that row, bit for bit on the same GPU,
    and agrees with `mlpForward` within the tolerance of [network brains](#network-brains-mlpbrain);
    but mixed blocks pay for both the cooperative stage and per-lane storage reads.
  - **Measured cost** (`npm run spike:s6`, 24→64→64→64→2 networks, 64 rows, 16,384 agents,
    `thinkEvery: 1`): with rows aligned to 64-agent blocks, a full simulation step ran **9.8%
    faster** than a single-network `MlpBrain` on an NVIDIA GPU under Node/Dawn (`−8.1%` at
    `thinkEvery: 3`), **38.0% faster** in Chrome on an AMD iGPU (`−22.6%` at `thinkEvery: 3`), and
    **30.6% slower** under Node/Dawn on the AMD iGPU (`+12.4%` at `thinkEvery: 3`). When agents in
    a block were shuffled across rows, the step cost `+164%`, `+34.4%` and `+114%`. Smaller
    networks gain less: with 24→32→32→2 networks and aligned rows, the step cost `−4.6%`, `+14.2%`
    and `+4.9%` on the same three hosts. Under FXC
    (Node/Dawn on Windows), compiling a `byGenome` pass takes 2–3 s (0.4–0.6 s in Chrome); use
    [prepareBrain](#switching-brains-without-a-hitch) when building one after load.
- **Out-of-range genomes:**
  - An agent whose `genome` word is `>= brain.rowCount` never reads outside the weight table. Its
    think writes `NaN` to every action (and `0` to `reason`), so the built-in steering brakes the
    agent to a stop while `inspect(i)` still reports its `genome`. Custom actions receive the `NaN`
    values and must guard them as for any brain.
- **`byGenome` and `bySpecies`:**
  - `MlpBrain.bySpecies` chooses a row from the agent's species; `MlpBrain.byGenome` chooses it
    from `genome`. They are separate constructors. One `byGenome` brain can serve several species
    when they share a shape (for example, rows `0–31` for one species and `32–63` for another) in
    one pass.

## Switching brains without a hitch

`await agents.prepareBrain(brain)` compiles a brain's think pass ahead, while frames go on, and
the system keeps it. `setBrain(brain)` then compiles nothing, now or after the brain served no
species for a while.

```ts
import type { AgentSystem, Brain } from 'null-graph/agents';

/**
 * Compiles each brain's pass at load, off the frame. Afterwards `use(brain)` compiles nothing,
 * however often `species` switches between them.
 */
export async function preparedSwitch(agents: AgentSystem, species: string, brains: readonly Brain[]) {
    await Promise.all(brains.map(brain => agents.prepareBrain(brain)));
    return {
        use(brain: Brain) { agents.setBrain(brain, { species }); },
    };
}
```

- **What it does:** it checks the brain as `setBrain` does, binds an ONNX Runtime brain to the
  system's buffers, compiles the pass with `engine.createComputeStageAsync`, and keeps the pass,
  out of every chain, until `agents.destroy()`. A kept pass costs a pipeline, its bind groups and
  a 16-byte buffer.
- **What it saves,** measured with Herd & Wolves' 24→64→64→64→2 networks and frames running:
  - `setBrain` of a prepared brain took 0.11–0.65 ms on every host.
  - Compiled inside `setBrain`, the same network's pass held its frame for 240 ms under Node/Dawn
    on an NVIDIA GPU and 1,759 ms on an AMD iGPU. Prepared, it compiled while frames went on; the
    longest frame meanwhile took 39 ms and 200 ms.
  - In Chrome, `setBrain`'s compile did not hold the main thread (frames of 17 ms): Chrome
    compiles in its GPU process. What it costs the GPU is not measured here.
- **Preparing again** returns the first promise. Preparing a brain the system already runs keeps
  its pass, and compiles nothing.
- **`setBrain` before the preparation settles is refused:** `AgentSystem: this brain is still
  being prepared; await prepareBrain(brain) before setBrain(brain).`
- **A pass that does not compile** rejects the promise with `NullGraph: compute stage "AgentSystem
  think" did not compile: <what the device said about the WGSL>`. The failed preparation is
  forgotten: `setBrain` then compiles as for any brain.
- **Ownership:** destroy a prepared brain only after the system, or once no species uses it; its
  kept pass then never runs. An ONNX Runtime brain stays bound to this system's buffers.

## A trained network

A network trained outside the app arrives as a `<species>.mlp.json` file: the layers as plain
arrays, with the observation and action names it was trained with. Loading one needs no new API.

```ts
import type { NullGraph } from 'null-graph';
import { MlpBrain, type AgentSystem, type MlpActivation } from 'null-graph/agents';

/** A parsed <species>.mlp.json. */
export interface TrainedNetwork {
    species: string;
    obsNames: string[];
    actNames: string[];
    layers: { input: number; output: number; activation: MlpActivation;
        weights: number[]; biases: number[] }[];
}

/** Puts a trained network on its species. The caller owns the returned brain. */
export function useTrainedNetwork(engine: NullGraph, agents: AgentSystem, file: TrainedNetwork) {
    const { obsNames, actNames } = agents.definition;
    // Equal widths cannot show that an input means what the network expects; equal names do.
    if (file.obsNames.join() !== obsNames.join() || file.actNames.join() !== actNames.join()) {
        throw new Error(`${file.species}.mlp.json was trained on other observations or actions.`);
    }
    const brain = new MlpBrain(engine, file.layers.map(layer => ({ ...layer,
        weights: new Float32Array(layer.weights), biases: new Float32Array(layer.biases) })));
    agents.setBrain(brain, { species: file.species });   // this species only: one pass, one compile
    return brain;
}
```

Call it once per species, for example with `await (await fetch('prey.mlp.json')).json()`. Destroy
the system, or give the species another brain, before destroying the returned brains.

- **Two trained files are in the repository,** not in the package:
  `tools/agents/models/herd-wolves/prey.mlp.json` and `wolf.mlp.json`, about 180 KB each. Each is
  a 24→64→64→64→2 `tanh` network that copies the Herd & Wolves rule for its species.
- **What they copy:** the rule's decision without its random wander and its coin-flips by agent
  index, because a network sees observations only. The copies do not wander.
- **How close they are,** on 5 runs of 30 s that training never saw:
  - against the rule's decision, the mean absolute difference is 0.034 for the prey's turn, 0.011
    for the wolves' turn and under 0.001 for each throttle;
  - wolves driven by the networks catch as many prey as wolves driven by the rule: within 1%,
    which is inside the spread between runs.
- **The weak spot is turning away from a wall:** 0.105 for prey and 0.070 for wolves, on about 5%
  of decisions.
- **They are valid for one definition and one set of settings:** the demo's defaults (speeds 3.2
  and 4.2, `thinkEvery` 3, flocking and fleeing on). The names are checked on load; the settings
  are in the file's `trainedOn`, and nothing checks them. Behaviour at other speeds is not
  measured.
- **Game rules stay in the app.** A resting wolf is frozen by the demo's app stage, and a frozen
  agent's action is not applied, so the network was never trained to rest.
- **As ONNX:** `npm run agents:export` writes each network as a 41 KB `.onnx` file.
  `mlpFromOnnx`, from `null-graph/inference`, reads such a file back to the same network bit for
  bit ([inference.md](./inference.md)). ONNX Runtime runs the files and agrees with `mlpForward`
  within 4.5e-6.
- **To train your own:** `npm run agents:record`, `agents:train` and `agents:evaluate`, described
  in [`tools/agents/README.md`](../../tools/agents/README.md). They run under Node, and training
  the two networks takes about an hour of CPU.

## Instances, highlights, labels and picking

Set up visibility geometry and species meshes first. Select GPU-owned rows using
`vis.setInstanceCount(firstInstance + agents.capacity)` (not `setInstances()`), then call
`agents.toInstances(vis, { firstInstance })`.

The system writes transforms, object-space bounds, species/highlight mesh and LOD into visibility
rows once per frame, including frames with zero simulation steps. Dead agents receive the hidden
mesh. Visibility must belong to the same engine, have sufficient capacity and use GPU rows;
drawn species need a positive conservative `boundsRadius`.

- `setHighlight({ selected, hovered })`: agent indices, `-1` to clear; omitted fields retain
  their value. Selected wins over hovered. Species specify `mesh`, `hoverMesh`, `selectedMesh`.
- Labels follow `{ buffer: agents.buffer, strideFloats: 16, positionOffset: 0 }`. The app hides
  labels for CPU-killed agents; automatic alive-bit label filtering is deferred.
- Register each drawn mesh/variant with `picker.addInstances(vis, meshName, meshIndex)`.
  A hit's agent index is `hit.index - firstInstance`. Picking and inspection are asynchronous,
  off the simulation frame path.

## Inspect one agent

`await agents.inspect(i)` returns `AgentSnapshot | null`:

- Position, heading, speed, radius, species, alive/frozen flags, energy and age.
- Record app words (`user0`, `user1`, `genome`, `flags`) and speed scale.
- Named observations/actions, reason word, and neighbor indices grouped by sensor.
- Tick and run-long grid overflow counters.

It copies one agent's rows and system counters in one submission into a lazily allocated,
reused staging buffer (204 bytes for Herd & Wolves). Requests are serialized. A queued/in-flight
request resolves `null` if destroyed; invalid indices and calls after destroy throw. A dead
agent still yields a snapshot. Reading a snapshot does not force a fresh think step.

`agentSnapshotLayout(def)` and `decodeAgentSnapshot(def, index, data)` are the pure layout/decoder
helpers. Throttle UI inspection rather than queuing one request per frame.

## Ownership and teardown

Stop the old frame sequence first. Destroy objects reading agent/visibility buffers before
releasing those buffers: picker, following text/debug draws, then agents and visibility as
appropriate to their bindings. Remove visibility draw batches before destroying visibility.
Update the active sequence so later frames do not reference released draws or buffers.

`AgentSystem.destroy()` is idempotent, disables its stages and releases its owned buffers/grid.
It does not destroy the caller's engine/device, visibility rows, brains or auxiliary app buffers.
Dispose those caller-owned resources separately once no dependent work can use them.

## Current validation and limits

Headless tests cover layouts, CPU mirrors, generated WGSL, the MLP's mathematics and validation,
request queues and refusals. The agents GPU suite covers the production loop, instance output,
picking, label following, inspection, the allocation/upload budget, teardown (A20), `MlpBrain`
on the production chain (A4), weight-table rows (A21), per-species passes (A22), the cooperative
think entry and per-agent weight rows (A35), query-only lists read by a custom sensor, a step
stage and the inspector at `thinkEvery` 1 and 3 (A36), the grid's exact order and crowded-cell
windows (A1), and bit-exact reruns without overflow (A7) and with overflow in every step (A23). It
passes 197/197 checks with zero WebGPU errors on Node/Dawn, on an AMD iGPU and on an NVIDIA GPU,
and in Chrome. A4's networks, from one affine layer to the largest allowed shape, agree with
`mlpForward` within 4.2e-7 on an AMD iGPU; `npm run spike:mlp-think` found 3.8e-7 on an NVIDIA GPU.

`test/agents-api-example.test.mjs`, in `npm run test:agents`, checks this guide's seven
TypeScript examples:
- each one type-checks and bundles, and its generated think WGSL parses;
- the neighbour-list example runs on a stand-in engine: its list adds no observation, every
  generated constant its WGSL names is declared, and its custom sensor's CPU mirror agrees with
  the grid's reference search for every agent;
- the network example's numbers match its comment;
- the per-species example runs on a stand-in engine that checks its passes and compiles;
- the per-agent network example packs 4 rows on a stand-in engine, spawns 64-agent blocks on rows
  `0–3`, and updates one row and a pair of rows without compiling a new stage;
- the switching example prepares a network and a rule brain on a stand-in engine, each compiled
  once, asynchronously, and its switches compile nothing;
- the trained-network example loads the two committed files on a stand-in engine, and refuses a
  file with other names.

Prepared brains are measured on the GPU by the `inference` suite (A33), on the same three host
and GPU pairs, with an `MlpBrain` and an ONNX Runtime brain.

The `agent-tools` GPU suite covers the training tools and the committed networks: the label pass
(A25), repeatable recordings (A24), the networks as `MlpBrain`s against `mlpForward` (A26), their
error and catches in the running simulation (A8), and Herd & Wolves on 64 identical per-agent rows
matching today's single-network brains word for word (A35 P3). It passes 17/17 checks with zero
WebGPU errors on the same three host and GPU pairs.

Integrated-demo CPU frame work averaged 0.57 ms at 4,096 agents and 0.60 ms at 16,384 agents on
an AMD GCN-5/Chrome host: means of the host's roughly 250 ms reporting windows, not GPU time or a
portable guarantee. Performance depends on hardware and workload. Dense flocks can exceed the
query's scan budget, and the scaling study (step 5) is deferred, so do not infer capacity from
the API.