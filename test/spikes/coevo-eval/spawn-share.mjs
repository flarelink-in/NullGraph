// Measurement, not a test (phase 14.4, design §10): how much of an episode's catches no policy can
// influence. Replays an evaluation's layout and held-out seeds for several pairings and reports mean
// catches per tile after 1, 20, 300, 600 and 1,200 steps. Reads counters mid-run, so it is a diagnostic
// outside the I6 production contract and not timing evidence.
//
//   node test/spikes/coevo-eval/spawn-share.mjs <evaluation.json> [--L <m>] [--separation <m>] [--P <prey>]
//
// --L replaces the tile side (density); --separation re-draws any prey spawned closer than that to
// a wolf of its tile (measurement only, before contact-v3 existed; a contact-v3 evaluation applies
// its own spawnSeparation); --P replaces the prey per tile (a multiple of 64). Per-tile catches are
// kept for paired statistics. NG_DAWN_FLAGS picks the GPU.
import { readFileSync } from 'node:fs';
import { installNodeGPU } from '../../gpu/node-gpu.mjs';
import { NullGraph } from '../../../dist/index.js';
import { MlpBrain } from '../../../dist/agents.js';
import { buildLane, readBuffers } from '../../../tools/agents/scenario/herd-wolves-arenas.mjs';
import { loadClones } from '../../../tools/agents/coevolve-eval.mjs';
import { tupleHash } from '../../../tools/agents/es.mjs';

const [file, ...rest] = process.argv.slice(2), option = name => {
    const at = rest.indexOf(`--${name}`);
    return at < 0 ? undefined : Number(rest[at + 1]);
};
const evaluation = JSON.parse(readFileSync(file, 'utf8'));
const x = evaluation.experiment, seeds = evaluation.heldOut.seeds;
const geometry = { ...evaluation.evaluation.geometry, ...(option('L') ? { L: option('L') } : {}),
    ...(option('P') ? { prey: option('P') } : {}) };
const separation = option('separation') ?? 0;

/** Re-draws a prey spawned within `separation` of a wolf of its tile, deterministically. */
function separated() {
    const wolves = new Map();
    return (init, { a, slot, lo, lay, wolf }) => {
        if (wolf) { (wolves.get(a) ?? wolves.set(a, []).get(a)).push(init.position); return; }
        const near = p => (wolves.get(a) ?? []).some(w => Math.hypot(p[0] - w[0], p[2] - w[2]) < separation);
        for (let retry = 0; near(init.position); retry++) {
            if (retry > 1000) throw new Error('no clear spawn');
            const u = k => tupleHash(a, slot, retry, k) / 4294967296;
            init.position = [lo[0] + 1 + u(0) * (lay.L - 2), init.position[1], lo[1] + 1 + u(1) * (lay.L - 2)];
        }
    };
}

const host = installNodeGPU();
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
const device = await adapter.requestDevice(), engine = new NullGraph();
await engine.init(null, { device, width: 8, height: 8 });
const clones = loadClones();
const tables = { prey: MlpBrain.byGenome(engine, [clones.prey.network]), wolf: MlpBrain.byGenome(engine, [clones.wolf.network]) };
const pairings = { 'rule/rule': { prey: 'rule', wolf: 'rule' }, 'label/label': { prey: 'label', wolf: 'label' },
    'clone/clone': tables, 'rule/clone': { prey: 'rule', wolf: tables.wolf }, 'clone/rule': { prey: tables.prey, wolf: 'rule' },
    // Bounds on what any policy can change: one species standing still (throttle -1, no turn).
    'rule/still': { prey: 'rule', wolf: 'rule', still: 'wolf' }, 'still/rule': { prey: 'rule', wolf: 'rule', still: 'prey' } };
const stationary = { kind: 'stage', wgsl: 'fn think(i: u32, species: u32, tick: u32) { setAct(i, 0u, 0.0); setAct(i, 1u, -1.0); }' };
const marks = [1, 20, 300, 600, 1200];
const out = { adapter: adapter.info.description, dt: x.dt, L: geometry.L, P: geometry.prey, W: geometry.wolves,
    separation, marks, pairings: {}, tiles: {} };
try {
    for (const [name, { still, ...brains }] of Object.entries(pairings)) {
        const lane = buildLane(engine, geometry, { environment: x.environment, contacts: x.contacts, contactOptions: x.contactOptions,
            catchMargin: x.catchMargin, spawnSeparation: x.spawnSeparation, thinkEvery: x.thinkEvery, dt: x.dt, monitorFinite: true,
            brains, seedOf: a => seeds[a], rowOf: () => 0, fixture: separation ? separated() : undefined });
        try {
            if (still) lane.sys.setBrain(stationary, { species: still });
            const row = {}, tiles = {};
            for (let step = 0, k = 0; k < marks.length; k++) {
                for (; step < marks[k]; step++) engine.render(lane.chain);
                const [counters] = await readBuffers(device, [{ buffer: lane.counters, words: geometry.arenas }]);
                row[marks[k]] = counters.reduce((a, b) => a + b, 0) / geometry.arenas;
                tiles[marks[k]] = Array.from(counters);
            }
            out.pairings[name] = row;
            out.tiles[name] = tiles;
        } finally { lane.destroy(); }
    }
    await host.settle();
    out.gpuErrors = host.events;
    console.log(JSON.stringify(out));
} finally {
    tables.prey.destroy(); tables.wolf.destroy(); device.destroy();
}
process.exit(out.gpuErrors?.length === 0 ? 0 : 1);
