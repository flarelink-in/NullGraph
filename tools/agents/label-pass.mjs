// Step 7, part 2 (docs/agents-cloning-design.md §6): a second think pass that only watches.
//
// The label pass runs a rule through the library's own think wrapper, so it has the system's gate,
// accessors and tick, and reads the system's records, observations and species table. It writes
// into two buffers of its own, so it changes nothing in the simulation. Spliced in after the last
// think pass and before act, it writes rows for exactly the agents whose observations were written
// in that step.
//
// The library has no public seam for such a pass, so lockstep() finds its place by the act stage's
// name. If a library change renames that stage, it refuses by name and A25 fails.
//
// No node: imports here: the GPU suite loads this file in a browser as well.
import * as A from '../../dist/agents.js';

const ACT_STAGE = 'AgentSystem act';

/**
 * A think pass over every species that writes actions into `act` and reason words into `tags`.
 * `buffers` are the rule's own group-1 buffers, as a brain's are.
 */
export function createLabelPass(engine, agents, { wgsl, buffers = [] }) {
    const device = engine.device, { capacity, definition, thinkEvery } = agents;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    const act = device.createBuffer({ label: 'label pass actions', size: capacity * definition.actDim * 4, usage });
    const tags = device.createBuffer({ label: 'label pass tags', size: capacity * 4, usage });
    const everySpecies = device.createBuffer({ label: 'label pass species', size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const release = () => { for (const b of [act, tags, everySpecies]) b.destroy(); };
    let stage;
    try {
        device.queue.writeBuffer(everySpecies, 0, new Uint32Array([0xffffffff, 0, 0, 0]));
        stage = engine.createComputeStage({ name: 'agents-tools label pass',
            shaderCode: A.buildAgentThinkWGSL(definition, wgsl, thinkEvery), dispatchCount: [Math.ceil(capacity / 64), 1, 1] });
        const group = list => list.map((buffer, binding) => ({ binding, resource: { buffer } }));
        engine.setComputePassBindGroup(stage, group([agents.buffer, agents.obs, agents.state, act, tags,
            agents.species, everySpecies]), 0);
        if (buffers.length) engine.setComputePassBindGroup(stage, group(buffers), 1);
    } catch (error) {
        if (stage) stage.enabled = false;
        release();
        throw error;
    }
    return {
        stage, act, tags,
        destroy() {
            stage.enabled = false;
            release();
        },
    };
}

/**
 * One lockstep step of `agents`, with `watchers` (label passes' stages) run after the last think
 * pass and before act. Returns `step()`, which uploads, submits and waits, as agents.step() does.
 * Call it again after setBrain(): the system rebuilds its chains.
 * `before` names the stage the watchers go in front of; only A25's failing case passes another.
 */
export function lockstep(engine, agents, watchers = [], before = ACT_STAGE) {
    if (agents.advance(agents.definition.dt) !== 1) {
        throw new Error('agents-tools: one advance(dt) did not select one step; lockstep() needs a fresh clock.');
    }
    const chain = agents.getComputeStages(), at = chain.findIndex(node => node.name === before);
    if (at < 0) {
        throw new Error(`agents-tools: the step has no "${before}" stage. These tools are tied to the library ` +
            'build they were written for; rebuild, or update tools/agents/label-pass.mjs.');
    }
    const sequence = [...chain.slice(0, at), ...watchers, ...chain.slice(at)];
    return async function step() {
        agents.upload();
        engine.render(sequence);
        await engine.device.queue.onSubmittedWorkDone();
    };
}

/** Copies several buffers out in one submission; resolves to one Uint32Array per buffer. */
export async function readBuffers(device, buffers) {
    const total = buffers.reduce((n, b) => n + b.size, 0);
    const staging = device.createBuffer({ size: total, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    let offset = 0;
    for (const b of buffers) {
        encoder.copyBufferToBuffer(b, 0, staging, offset, b.size);
        offset += b.size;
    }
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const bytes = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    offset = 0;
    return buffers.map(b => {
        const words = new Uint32Array(bytes, offset, b.size / 4);
        offset += b.size;
        return words;
    });
}
