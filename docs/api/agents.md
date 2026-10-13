# GPU agents — rule brains and network brains

Import from `null-graph/agents`. Debug draws, visibility, picking and text are separate opt-in
entries. This guide covers the GPU loop (C1 steps 3–4) with rule brains, `MlpBrain` (C2 step 6),
which runs a small dense network you supply, per-species brains, and loading a trained network
(C2 step 7). Training, ONNX import, ONNX Runtime brains, camera eyes and a replay/rewind UI are not
implemented by this API. Training is done by tools in the repository, outside the package
([a trained network](#a-trained-network)).

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
- `Sensors.nearest({ name, species, k })`: up to `k` neighbors, each with local side/forward
  position, relative side/forward velocity and presence. Species filters use declared names.
- `Sensors.bounds()`: proximity-weighted nearest-wall direction in the agent's local frame.
- Custom sensors/actions require their declared WGSL signatures; CPU mirrors are needed when
  using the CPU reference routines.

**What a custom sensor can read today.** `Sensors.custom({ name, width, wgsl, mirror? })` compiles
`fn <name>(i: u32, a: Agent) -> array<f32, width>` into the sense stage. Its contract is the
agent's own record, `a`.

- **No contract for other agents or the neighbour lists.** Both are in scope in that shader, but
  nothing is generated or promised for them: no constants for the lists' layout, no documented
  binding names, and the CPU `mirror` is not given the lists. Do not build on them yet.
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

- **It holds for** the built-in sensors, and for a custom sensor that reads only its own agent's
  record. The grid lists each cell's agents in a fixed order, and a crowded cell's window depends
  only on the cell and the tick.
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
as that step's brain. Auxiliary group-1 buffers and brain objects remain caller-owned.

For peer-dependent mechanics, separate detection and mutation dispatches: the Herd & Wolves
demo first reads peer positions and marks its own flags, then consumes them for respawn/rest.
Do not write peer records concurrently and assume another invocation sees a consistent result.

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
  ([per-species brains](#per-species-brains)).
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
  under Node/Dawn on an NVIDIA GPU. Swap brains at a moment that can absorb a hitch, not every
  frame. Moving species to a brain the system already runs compiles nothing.
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
    again compiles its pass again, from identical shader text, which a pipeline cache can reuse.
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
- **As ONNX:** `npm run agents:export` writes each network as a 41 KB `.onnx` file. This API
  does not read ONNX yet; ONNX Runtime runs the files and agrees with `mlpForward` within 4.5e-6.
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
on the production chain (A4), weight-table rows (A21), per-species passes (A22), the grid's exact
order and crowded-cell windows (A1), and bit-exact reruns without overflow (A7) and with overflow
in every step (A23). It passes 137/137 checks with zero WebGPU errors on Node/Dawn, on an AMD iGPU
and on an NVIDIA GPU, and in Chrome. A4's networks, from one affine layer to the largest allowed
shape, agree with `mlpForward` within 4.2e-7 on an AMD iGPU; `npm run spike:mlp-think` found
3.8e-7 on an NVIDIA GPU.

`test/agents-api-example.test.mjs`, in `npm run test:agents`, checks this guide's four TypeScript
examples:
- each one type-checks and bundles, and its generated think WGSL parses;
- the network example's numbers match its comment;
- the per-species example runs on a stand-in engine that checks its passes and compiles;
- the trained-network example loads the two committed files on a stand-in engine, and refuses a
  file with other names.

The `agent-tools` GPU suite covers the training tools and the committed networks: the label pass
(A25), repeatable recordings (A24), the networks as `MlpBrain`s against `mlpForward` (A26), and
their error and catches in the running simulation (A8). It passes 15/15 checks with zero WebGPU
errors on the same three host and GPU pairs.

Integrated-demo CPU frame work averaged 0.57 ms at 4,096 agents and 0.60 ms at 16,384 agents on
an AMD GCN-5/Chrome host: means of the host's roughly 250 ms reporting windows, not GPU time or a
portable guarantee. Performance depends on hardware and workload. Dense flocks can exceed the
query's scan budget, and the scaling study (step 5) is deferred, so do not infer capacity from
the API.