// D2.1 (docs/offline-films-design.md §5, §10): Herd & Wolves as a film scene, in record mode.
//
// The simulation is tools/agents/scenario/herd-wolves.mjs, the demo's rules checked against the site
// (step 10's D3), with both species on the committed networks (or the rules). The look is the demo's
// Models mode, composed from byte-for-byte copies of the site's files (herd-wolves/, checked by
// `npm run film:check`), with assets read from the site's public/ folder.
//
// Settings (the film file's "settings"): count (4096; a multiple of 16: one agent in 16 is a wolf),
// seed (11), thinkEvery (3), brains ('network' | 'rules'), look ('anime' | 'faithful'), flashes (true).
//
// The guard (design §12): a count above 65,536 is first calibrated on this GPU at a sixteenth and a
// quarter of itself (with progressively smaller probes first), and refused when its step is predicted
// above 1 s. D2.3 keeps maximum queue-completion wall measurements per adapter and workload.
import { readFileSync } from 'node:fs';
import { OrbitalCameraState } from '../../../dist/cameras.js';
import { Camera } from '../../../dist/index.js';
import { MlpBrain } from '../../../dist/agents.js';
import { herdWolves } from '../../agents/scenario/herd-wolves.mjs';
import { networkFromModel } from '../../agents/model.mjs';
import { hashText } from '../../agents/dataset.mjs';
import { hookCopies } from '../resolve.mjs';
import { loadHerdAssets } from './herd-wolves/assets.mjs';
import { buildLook } from './herd-wolves/look.mjs';
import { track } from '../records.mjs';
import { OrtBrain } from '../../../dist/inference.js';
import { onnxFromNetwork } from '../../agents/onnx.mjs';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { release } from 'node:os';
import { fileURLToPath } from 'node:url';
import { calibrateSteps, checkStep, readCosts, saveCosts } from '../guard.mjs';

export const SCENE_DEFAULTS = Object.freeze({ count: 4096, seed: 11, thinkEvery: 3, brains: 'network', look: 'anime', flashes: true });
const BASE_HALF = 80, CALIBRATE_ABOVE = 65_536;
const MODELS = new URL('../../agents/models/herd-wolves/', import.meta.url);
const COPIES = new URL('./herd-wolves/', import.meta.url);

/** The settings, checked: a refusal names the setting. */
export function readSettings(settings) {
    const s = { ...SCENE_DEFAULTS, ...settings };
    const unknown = Object.keys(settings).filter(k => !(k in SCENE_DEFAULTS));
    if (unknown.length) throw new Error(`film: herd-wolves has no setting "${unknown[0]}"; it takes ${Object.keys(SCENE_DEFAULTS).join(', ')}.`);
    if (!(Number.isInteger(s.count) && s.count >= 64 && s.count % 16 === 0)) throw new Error(`film: herd-wolves count must be a whole number of at least 64, a multiple of 16; got ${s.count}.`);
    if (!(Number.isInteger(s.seed) && s.seed >= 0)) throw new Error(`film: herd-wolves seed must be a whole number; got ${s.seed}.`);
    if (!(Number.isInteger(s.thinkEvery) && s.thinkEvery > 0)) throw new Error(`film: herd-wolves thinkEvery must be a whole number above zero; got ${s.thinkEvery}.`);
    if (!['network', 'rules', 'ort'].includes(s.brains)) throw new Error(`film: herd-wolves brains must be "network", "rules" or "ort"; got ${JSON.stringify(s.brains)}.`);
    if (!['anime', 'faithful'].includes(s.look)) throw new Error(`film: herd-wolves look must be "anime" or "faithful"; got ${JSON.stringify(s.look)}.`);
    if (typeof s.flashes !== 'boolean') throw new Error('film: herd-wolves flashes must be true or false.');
    return s;
}

/** The simulation with its brains: the scenario, and both species on the committed networks unless 'rules'. */
function buildSimulation(engine, s) {
    const sim = herdWolves.create(engine, { count: s.count, seed: s.seed, thinkEvery: s.thinkEvery });
    const brains = s.brains === 'network' ? ['prey', 'wolf'].map(species => {
        const file = JSON.parse(readFileSync(new URL(`${species}.mlp.json`, MODELS), 'utf8'));
        const brain = new MlpBrain(engine, networkFromModel(file, sim.definition, `${species}.mlp.json`));
        sim.agents.setBrain(brain, { species });
        return brain;
    }) : [];
    return { ...sim, destroy() { sim.destroy(); brains.forEach(b => b.destroy()); } };
}

/** The one-step chain, taken after every stage is attached (as tools/agents/label-pass.mjs's lockstep). */
function stepChain(agents) {
    if (agents.advance(agents.definition.dt) !== 1) throw new Error('film: the simulation\'s clock did not select one step.');
    agents.upload();
    return agents.getComputeStages();
}

/** Maximum steady-state wall ms; warm-up is checked for stops but excludes pipeline compilation. */
async function timeSteps(engine, chain, warm, n, stopMs) {
    let total = 0;
    for (let k = 0; k < warm + n; k++) {
        const t = performance.now();
        engine.render(chain);
        await engine.device.queue.onSubmittedWorkDone();
        const took = checkStep(performance.now() - t, stopMs);
        if (k >= warm) total = Math.max(total, took);
    }
    return total;
}

/** Cache only identical adapters, runtime/build, scenario, networks and simulation settings. */
function costKey(engine, s) {
    const hash = createHash('sha256');
    const info = engine.device.adapterInfo ?? {};
    hash.update(JSON.stringify({ adapter: [info.vendor, info.architecture, info.device, info.description],
        platform: process.platform, os: release(), node: process.version,
        dawn: createRequire(import.meta.url)('webgpu/package.json').version,
        flags: process.env.NG_DAWN_FLAGS ?? '', brains: s.brains, thinkEvery: s.thinkEvery, seed: s.seed }));
    for (const path of ['../../../dist/index.js', '../../../dist/agents.js', '../../agents/scenario/herd-wolves.mjs',
        './herd-wolves.mjs', '../guard.mjs', '../../agents/models/herd-wolves/prey.mlp.json', '../../agents/models/herd-wolves/wolf.mlp.json']) {
        hash.update(readFileSync(new URL(path, import.meta.url)));
    }
    return hash.digest('hex');
}

/** Persistent step guard. ORT remains refused separately until calibrated through its own path. */
async function guard(engine, s, log) {
    if (s.count <= CALIBRATE_ABOVE) return null;
    const file = fileURLToPath(new URL('../out/costs.json', import.meta.url)), key = costKey(engine, s);
    return calibrateSteps({ count: s.count, samples: readCosts(file, key), log,
        save(samples) {
            try { saveCosts(file, key, samples); }
            catch (error) { log(`guard: could not save calibration: ${error.message}`); }
        },
        async measure(count, stopMs) {
            const sim = buildSimulation(engine, { ...s, count });
            try { return await timeSteps(engine, stepChain(sim.agents), 3, 10, stopMs); }
            finally { sim.destroy(); await engine.device.queue.onSubmittedWorkDone(); }
        },
    });
}

export default async function scene({ engine, width, height, scale = 1, settings = {}, mode = 'record', trackEvery = 6, runtime, log = () => {} }) {
    const s = readSettings(settings);
    if (s.brains === 'ort' && s.count > CALIBRATE_ABOVE && mode === 'record') throw new Error('film: ORT films above 65,536 need ORT-specific guard calibration (D2.3).');
    const guarded = mode === 'record' ? await guard(engine, s, log) : null;
    hookCopies();
    const load = name => import(new URL(`${name}.ts`, COPIES).href);
    const [models, field, flashes] = await Promise.all(['models', 'field', 'flashes'].map(load));
    const assets = loadHerdAssets(engine);
    const sim = buildSimulation(engine, s);
    const ortModels = [], ortBrains = [];
    if (s.brains === 'ort' && mode === 'record') {
        if (!runtime || runtime.device !== engine.device) throw new Error('film: ORT recording needs the engine on the film runtime device.');
        for (const species of ['prey', 'wolf']) {
            const modelFile = JSON.parse(readFileSync(new URL(`${species}.mlp.json`, MODELS), 'utf8'));
            const network = networkFromModel(modelFile, sim.definition, `${species}.mlp.json`);
            const model = await runtime.load(onnxFromNetwork(network));
            const brain = new OrtBrain(model);
            ortModels.push(model); ortBrains.push(brain);
            await sim.agents.prepareBrain(brain); sim.agents.setBrain(brain, { species });
        }
    }
    // The demo's catch ring starts as "caught long ago" (site index.ts:647–649), so no flash shows at the origin.
    const never = new Float32Array(1024 * 4);
    for (let k = 0; k < 1024; k++) never[k * 4 + 2] = -1e9;
    engine.device.queue.writeBuffer(sim.ring, 0, never);
    const look = buildLook(engine, sim, { models, field, flashes }, assets, { width, height, look: s.look, scale, flashes: s.flashes });
    const chain = stepChain(sim.agents);
    sim.agents.advance(0);
    const instances = sim.agents.getComputeStages();
    const dt = sim.definition.dt;
    const tracks = [
        track('agents', sim.agents.buffer, { every: trackEvery, dt, preset: 'agents',
            fields: undefined, add: [{ word: 11, store: 'f16', interp: 'step' }] }),
        track('phase', look.phase, { stride: 4, every: trackEvery, dt, fields: [{ word: 0, store: 'unorm16', interp: 'wrap', maxRate: 4 }] }),
        track('ring', sim.ring, { every: 1, dt }),
        track('stats', sim.stats, { every: 1, dt }),
        track('state', sim.agents.state, { every: 1, dt }),
    ];
    // Respawns are teleports. Do not interpolate them across the field.
    for (const f of tracks[0].layout.fields) if (f.interp === 'hermite') f.jump = 4.2 * trackEvery * dt * 1.05;

    const half = sim.definition.world.maxX, fieldScale = half / BASE_HALF;
    const camera = new Camera(60, width / height, 0.5, 1000);
    /** The demo's opening view: its orbit at 190 m, scaled with the field, azimuth 0.3, polar 0.4. */
    const defaultView = { target: [0, 0, 0], distance: 190 * fieldScale, azimuth: 0.3, polar: 0.4 };
    const texts = herdWolves.texts({ count: s.count }), wgsl = {};
    for (const [role, text] of Object.entries(texts)) wgsl[role] = await hashText(text);
    const networks = {};
    if (s.brains !== 'rules') for (const species of ['prey', 'wolf']) networks[species] = await hashText(readFileSync(new URL(`${species}.mlp.json`, MODELS), 'utf8'));

    return {
        dt: sim.definition.dt,
        tracks,
        checkpoint: [
            ['agents', sim.agents.buffer], ['actions', sim.agents.act], ['state', sim.agents.state],
            ['phase', look.phase], ['ring', sim.ring], ['stats', sim.stats],
        ].map(([name, buffer]) => ({ name, buffer })),
        settings: s,
        defaultView,
        info: { agents: s.count, field: `${(2 * half).toFixed(0)} m`, animals: look.usePalette ? 'palette' : 'hook', guard: guarded, wgsl, networks },
        /** One exact step, its own submission, waited for. */
        async step() {
            if (s.brains === 'ort') return sim.agents.step();
            sim.agents.upload();
            engine.render(chain);
            await engine.device.queue.onSubmittedWorkDone();
        },
        /** Points the camera: an orbit `{ target, distance, azimuth, polar }`, near and far as the demo sets them. */
        view(state, seconds) {
            const v = state ?? defaultView;
            const orbit = new OrbitalCameraState({ target: v.target, distance: v.distance, azimuthAngle: v.azimuth,
                polarAngle: Math.max(1e-3, v.polar), minDistance: 0.01, maxDistance: 1e9, maxPolarAngle: Math.PI - 1e-3 });
            camera.near = Math.max(0.05, v.distance * 0.004);
            camera.far = v.distance * 2 + half * 3;
            camera.updateProjection(width / height);
            camera.updateView(orbit.getPosition(), orbit.target);
            camera.updateTime(seconds);
            engine.updateCamera(camera);
        },
        draw() { return [...instances, ...look.sequence()]; },
        /** The agents buffer, for a determinism check (F1). */
        agentsBuffer: sim.agents.buffer,
        destroy() {
            sim.destroy();
            ortBrains.forEach(b => b.destroy());
            ortModels.forEach(m => m.destroy());
            look.destroy();
            assets.destroy();
            engine.clearPasses();
        },
    };
}
