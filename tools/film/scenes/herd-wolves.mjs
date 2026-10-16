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
// quarter of itself, and refused when its step is predicted above 1 s of GPU work, half the time after
// which Windows resets the GPU. D2.3 keeps the measurements per adapter.
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

export const SCENE_DEFAULTS = Object.freeze({ count: 4096, seed: 11, thinkEvery: 3, brains: 'network', look: 'anime', flashes: true });
const BASE_HALF = 80, LIMIT_MS = 1000, CALIBRATE_ABOVE = 65_536;
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
    if (s.brains === 'ort') throw new Error('film: herd-wolves brains "ort" come later (ONNX Runtime makes its own device); use "network" or "rules".');
    if (!['network', 'rules'].includes(s.brains)) throw new Error(`film: herd-wolves brains must be "network" or "rules"; got ${JSON.stringify(s.brains)}.`);
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

/** Mean wall ms of `n` steps after `warm`, each its own submission, waited for. */
async function timeSteps(engine, chain, warm, n) {
    let total = 0;
    for (let k = 0; k < warm + n; k++) {
        const t = performance.now();
        engine.render(chain);
        await engine.device.queue.onSubmittedWorkDone();
        if (k >= warm) total += performance.now() - t;
    }
    return total / n;
}

/** The guard: calibrates at N/16 and N/4 and predicts N by the demo's rule; refuses above LIMIT_MS. */
async function guard(engine, s, log) {
    if (s.count <= CALIBRATE_ABOVE) return null;
    let measured = null;
    for (const count of [s.count / 16, s.count / 4]) {
        if (count % 16) continue;
        if (measured) {
            const ms = (measured.ms / measured.count) * count * (count / measured.count) ** 0.25;
            if (ms > LIMIT_MS) break;
        }
        const sim = buildSimulation(engine, { ...s, count });
        try {
            measured = { count, ms: await timeSteps(engine, stepChain(sim.agents), 3, 10) };
        } finally {
            sim.destroy();
            await engine.device.queue.onSubmittedWorkDone();
        }
        log(`guard: ${count.toLocaleString('en-US')} agents step in ${measured.ms.toFixed(1)} ms on this GPU`);
    }
    const ms = measured ? (measured.ms / measured.count) * s.count * (s.count / measured.count) ** 0.25 : Infinity;
    if (ms > LIMIT_MS) {
        throw new Error(`film: ${s.count.toLocaleString('en-US')} agents: ~${ms.toFixed(0)} ms per step predicted on this GPU ` +
            `(${((measured?.ms ?? NaN) / (measured?.count ?? NaN) * 1e6).toFixed(0)} ns per agent at ${(measured?.count ?? 0).toLocaleString('en-US')}), ` +
            `over the ${LIMIT_MS} ms limit per submission (Windows resets a GPU busy for about 2 s).`);
    }
    return { predictedMs: ms, from: measured };
}

export default async function scene({ engine, width, height, scale = 1, settings = {}, mode = 'record', log = () => {} }) {
    if (mode !== 'record') throw new Error(`film: herd-wolves plays "${mode}" from D2.2; this build records only.`);
    const s = readSettings(settings);
    const guarded = await guard(engine, s, log);
    hookCopies();
    const load = name => import(new URL(`${name}.ts`, COPIES).href);
    const [models, field, flashes] = await Promise.all(['models', 'field', 'flashes'].map(load));
    const assets = loadHerdAssets(engine);
    const sim = buildSimulation(engine, s);
    // The demo's catch ring starts as "caught long ago" (site index.ts:647–649), so no flash shows at the origin.
    const never = new Float32Array(1024 * 4);
    for (let k = 0; k < 1024; k++) never[k * 4 + 2] = -1e9;
    engine.device.queue.writeBuffer(sim.ring, 0, never);
    const look = buildLook(engine, sim, { models, field, flashes }, assets, { width, height, look: s.look, scale, flashes: s.flashes });
    const chain = stepChain(sim.agents);

    const half = sim.definition.world.maxX, fieldScale = half / BASE_HALF;
    const camera = new Camera(60, width / height, 0.5, 1000);
    /** The demo's opening view: its orbit at 190 m, scaled with the field, azimuth 0.3, polar 0.4. */
    const defaultView = { target: [0, 0, 0], distance: 190 * fieldScale, azimuth: 0.3, polar: 0.4 };
    const texts = herdWolves.texts({ count: s.count }), wgsl = {};
    for (const [role, text] of Object.entries(texts)) wgsl[role] = await hashText(text);
    const networks = {};
    if (s.brains === 'network') for (const species of ['prey', 'wolf']) networks[species] = await hashText(readFileSync(new URL(`${species}.mlp.json`, MODELS), 'utf8'));

    return {
        dt: sim.definition.dt,
        settings: s,
        defaultView,
        info: { agents: s.count, field: `${(2 * half).toFixed(0)} m`, animals: look.usePalette ? 'palette' : 'hook', guard: guarded, wgsl, networks },
        /** One exact step, its own submission, waited for. */
        async step() {
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
        draw() { return look.sequence(); },
        /** The agents buffer, for a determinism check (F1). */
        agentsBuffer: sim.agents.buffer,
        destroy() {
            sim.destroy();
            look.destroy();
            assets.destroy();
            engine.clearPasses();
        },
    };
}
