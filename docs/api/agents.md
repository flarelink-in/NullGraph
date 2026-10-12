# GPU agents — rule brains and network brains

Import from `null-graph/agents`. Debug draws, visibility, picking and text are separate opt-in
entries. This guide covers the GPU loop (C1 steps 3–4) with rule brains, and `MlpBrain` (C2 step
6), which runs a small dense network you supply. Training, ONNX import, ONNX Runtime brains,
camera eyes and a replay/rewind UI are not implemented by this API.

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

The default actions are steering: turn and throttle in `[-1, 1]`. Movement enforces speed,
acceleration and turn limits, brakes on non-finite actions, and clamps/reflects at world bounds.
These are steering agents, not a rigid-body collision solver.

Nearest queries use the bounded grid search described in [agent-grid.md](./agent-grid.md).
Use `senseRadius <= cellSize`. Dense cells can exceed `maxScanPerCell`: all indices remain
stored, but a query scans only its budget. Overflow counters expose that loss of perception. It
also costs determinism: which entries a crowded cell yields follows GPU timing. Flocks pack far
above their mean density (Herd & Wolves overflowed in 85% of steps at the default 64), so size the
budget or the cell for the peak, and watch `overflowSteps`.

## The frame and fixed-step paths

```text
Each step: grid → queries → sense → think → act → app stages → tick
Each frame: instance output (when enabled) → visibility/text → draw
```

`advance(deltaSeconds)` chooses zero or more fixed steps using the definition's `dt`,
`maxStepsPerFrame` and `maxFrameDelta`. Call `upload()` for dirty CPU staging records and
parameters, then render `getComputeStages()` before dependent draws/culls.

`thinkEvery` is fixed at construction. Sense and think share a cadence staggered by 64-agent
blocks; newborns get an immediate decision. Act/app/tick still run each simulation step.
Observations/actions can therefore represent an earlier decision than the current position.

`await agents.step()` uploads and submits one step, waiting for GPU completion. It is the
lockstep experiment/test path, not a replacement for the regular frame loop. Reruns on one device
are bit-exact only with built-in sensors and no grid overflow; no cross-device guarantee is made.

**CPU records are a staging area, not a live mirror.** After GPU movement, `agents.records`
does not contain current positions. `spawn(i, init)` and `kill(i)` dirty individual records;
`upload()` sends those edits. Use inspection for current GPU state, not a full staging re-upload.

## Rule brains and app logic

`setBrain(new RuleBrain(wgsl))` installs WGSL with the exact entry:
`fn think(i: u32, species: u32, tick: u32)`.
Use `obs(i, offset)`, `setAct(i, offset, value)` and `setReason(i, word)` with the definition's
generated constants. A brain is required before `getComputeStages()` or `step()`.

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
hand-set weights; a network trained to imitate the rules is a later step.

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
- **One network for every species:** identical observations give identical actions, whatever the
  species. Policies that differ by species need a feature that tells them apart.
- **Numerics:** tanh's input is clamped to ±15 (the FXC compiler behind Dawn's D3D12 backend
  returns NaN beyond about 89);
  sigmoid is evaluated stably; identity and relu outputs are not clipped. The act stage still
  clamps steering and brakes on non-finite actions. `mlpForward(network, input, out?, scratch?)` is
  the CPU reference, with the same definitions and f32 rounding, allocation-free when given its
  output and scratch arrays. GPU results agree within `1e-5 + 1e-5·|cpu|`, not bit for bit.
- **Updates:** `brain.setWeights(network)` needs the same shape and activations. It validates the
  whole network, then uploads once into the same buffer: no new stage, no recompile, and work
  submitted afterwards uses it. A different shape needs a new brain and `setBrain`.
- **Swapping costs a compile:** `setBrain` builds a new think pipeline. For a 24→32→32→2 network
  that took 76 ms in Chrome on an AMD iGPU and about 1.1 s under Node/Dawn on an NVIDIA GPU. Swap
  brains at a moment that can absorb a hitch, not every frame.
- **Reasons:** an `MlpBrain` writes reason 0. Inspection shows its observations and actions, not
  a rationale.
- **Ownership:** the caller owns the brain, and it belongs to its engine's device: `setBrain`
  refuses a brain from another device or one already destroyed. Replace the brain or destroy the
  system before `brain.destroy()`, because the think stage binds the weights. `destroy()` is
  idempotent.
- **Cost:** one think dispatch of a 24→32→32→2 network over 100,000 agents took 0.83–0.88 ms
  (8–9 ns per agent) on the measured GPUs, and `thinkEvery: 3` cuts it to 43–57% of that
  (`npm run spike:mlp-think`). `MlpBrain` adds 1.4 KB min+gzip to an app; rule-only imports
  carry none of it.

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
It does not destroy the caller's engine/device, visibility rows, brain or auxiliary app buffers.
Dispose those caller-owned resources separately once no dependent work can use them.

## Current validation and limits

Headless tests cover layouts, CPU mirrors, generated WGSL, the MLP's mathematics and validation,
request queues and refusals. The agents GPU suite covers the production loop, instance output,
picking, label following, inspection, the allocation/upload budget, teardown (A20) and
`MlpBrain` on the production chain (A4): 120/120 checks on Node/Dawn and in Chrome, zero WebGPU
errors. A4's networks, from one affine layer to the largest allowed shape, agree with
`mlpForward` within 4.2e-7 on an AMD iGPU; `npm run spike:mlp-think` found 3.8e-7 on an NVIDIA
GPU. Both TypeScript examples in this guide are type-checked and bundled, and their generated
think WGSL parsed, by `test/agents-api-example.test.mjs` in `npm run test:agents`.

Integrated-demo CPU frame work averaged 0.57 ms at 4,096 agents and 0.60 ms at 16,384 agents on
an AMD GCN-5/Chrome host: means of the host's roughly 250 ms reporting windows, not GPU time or a
portable guarantee. Performance depends on hardware and workload. Dense flocks can exceed the
query's scan budget, and the scaling study (step 5) is deferred, so do not infer capacity from
the API.