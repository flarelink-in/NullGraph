// Step 15, first slice (docs/agents-coevo-films-design.md): a co-evolution run's checkpoints filmed in
// the arenas they were trained in. Tile a plays prey[a] against wolf[a]: a generation of the run, the
// committed clones ('clone') or, in every tile of a species, the rule ('rule'). Every tile starts from
// the same episode seed, so what differs between tiles is the brains. The simulation is buildLane, the
// lane the trainer and the evaluator step, in the run's own experiment; the look is Herd & Wolves'
// Models look (herd-wolves/look.mjs) over that lane, with a fence around every tile.
//
// Settings (the film file's "settings"): run (a run directory, relative to the repository), prey and
// wolf (one entry per tile, the tile count a multiple of 4), seed (77001), cols (2), look ('anime' |
// 'faithful'). A film longer than the run's episode (T steps) shows play past the training horizon.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { OrbitalCameraState } from '../../../dist/cameras.js';
import { Camera } from '../../../dist/index.js';
import { MlpBrain } from '../../../dist/agents.js';
import { StandardLayout, MegabufferBuilder } from '../../../dist/geometry.js';
import { InstanceBuffer, VisibilitySystem } from '../../../dist/visibility.js';
import { ToonMaterial } from '../../../dist/materials.js';
import { buildLane } from '../../agents/scenario/herd-wolves-arenas.mjs';
import { loadCheckpoint, loadClones, loadRun } from '../../agents/coevolve-eval.mjs';
import { hookCopies } from '../resolve.mjs';
import { loadHerdAssets } from './herd-wolves/assets.mjs';
import { buildLook } from './herd-wolves/look.mjs';
import { track } from '../records.mjs';

export const SCENE_DEFAULTS = Object.freeze({ run: null, prey: null, wolf: null, seed: 77001, cols: 2, look: 'anime' });
const ROOT = fileURLToPath(new URL('../../../', import.meta.url));
const COPIES = new URL('./herd-wolves/', import.meta.url);
const MARGIN = 6;   // m of field between the outer tiles' fences and the field's own
const sha256 = text => createHash('sha256').update(text).digest('hex');

/** The settings, checked: a refusal names the setting. */
export function readSettings(settings) {
    const s = { ...SCENE_DEFAULTS, ...settings };
    const unknown = Object.keys(settings).filter(k => !(k in SCENE_DEFAULTS));
    if (unknown.length) throw new Error(`film: coevo-arena has no setting "${unknown[0]}"; it takes ${Object.keys(SCENE_DEFAULTS).join(', ')}.`);
    if (typeof s.run !== 'string' || !s.run) throw new Error('film: coevo-arena needs run, a run directory relative to the repository.');
    for (const species of ['prey', 'wolf']) {
        const entries = s[species];
        if (!Array.isArray(entries) || !entries.length) throw new Error(`film: coevo-arena ${species} must list one driver per tile.`);
        for (const e of entries) if (!(e === 'clone' || e === 'rule' || (Number.isInteger(e) && e >= 0)))
            throw new Error(`film: coevo-arena ${species} entries are generations, "clone" or "rule"; got ${JSON.stringify(e)}.`);
        if (entries.includes('rule') && !entries.every(e => e === 'rule'))
            throw new Error(`film: coevo-arena ${species} runs the rule in every tile or in none (one brain per species).`);
    }
    if (s.prey.length !== s.wolf.length) throw new Error('film: coevo-arena prey and wolf must list the same number of tiles.');
    if (s.prey.length % 4 || s.prey.length > 16) throw new Error(`film: coevo-arena needs 4, 8, 12 or 16 tiles (16 wolves a tile, 64-agent blocks); got ${s.prey.length}.`);
    if (!(Number.isInteger(s.seed) && s.seed >= 0 && s.seed <= 0xffffffff)) throw new Error(`film: coevo-arena seed must be a u32; got ${s.seed}.`);
    if (!(Number.isInteger(s.cols) && s.cols >= 1 && s.cols <= s.prey.length)) throw new Error(`film: coevo-arena cols must be from 1 to the tile count; got ${s.cols}.`);
    if (!['anime', 'faithful'].includes(s.look)) throw new Error(`film: coevo-arena look must be "anime" or "faithful"; got ${JSON.stringify(s.look)}.`);
    return s;
}

const label = entry => (typeof entry === 'number' ? `gen-${String(entry).padStart(4, '0')}` : entry);
const shape = network => network.map(l => `${l.input}→${l.output} ${l.activation}`).join(', ');

/**
 * The tiles' drivers, read as the evaluator reads them: the run's experiment, and per species either
 * the rule or one network table, a row per distinct entry, every model checked against its manifest.
 */
export function planTiles(s) {
    const loaded = loadRun(resolve(ROOT, s.run)), config = loaded.run.config;
    const pairs = new Map();
    const pairOf = entry => {
        if (!pairs.has(entry)) pairs.set(entry, entry === 'clone' ? loadClones() : loadCheckpoint(loaded, entry));
        return pairs.get(entry);
    };
    const drivers = {}, rowOf = {}, networks = {};
    for (const species of ['prey', 'wolf']) {
        if (s[species][0] === 'rule') { drivers[species] = 'rule'; rowOf[species] = s[species].map(() => 0); continue; }
        const distinct = [...new Set(s[species])], rows = distinct.map(entry => pairOf(entry)[species]);
        for (const row of rows) if (shape(row.network) !== shape(rows[0].network))
            throw new Error(`film: coevo-arena ${species} networks must share one shape: ${label(distinct[0])} is ${shape(rows[0].network)}, ${row.id} is ${shape(row.network)}.`);
        drivers[species] = rows.map(row => row.network);
        rowOf[species] = s[species].map(entry => distinct.indexOf(entry));
        distinct.forEach((entry, r) => { networks[`${species}:${label(entry)}`] = rows[r].hash; });
    }
    const geometry = { arenas: s.prey.length, wolves: config.W, prey: config.P, cols: s.cols, L: config.L, gap: config.gap };
    const lane = { environment: config.environment, contacts: config.contacts, contactOptions: config.contactOptions,
        catchMargin: config.catchMargin, spawnSeparation: config.spawnSeparation, thinkEvery: config.thinkEvery, dt: config.dt };
    return { run: loaded.run, geometry, lane, drivers, rowOf, networks,
        tiles: s.prey.map((prey, a) => `${label(prey)}/${label(s.wolf[a])}`) };
}

/** The default camera: straight down, every tile in view with a tenth to spare; tile 0 top left, then reading order. */
export function topDownView(lay, aspect) {
    return { target: [0, 0, 0], azimuth: 0, polar: 1e-3,
        distance: 1.1 * Math.max(lay.spanZ / 2, lay.spanX / 2 / aspect) / Math.tan(Math.PI / 6) };
}

/** A fence around every tile: the field's posts and rails (fence.ts) in its toon wood, one ring a tile. */
function tileFences(engine, pass, targetFormat, fence, lay) {
    const rows = [];
    for (let a = 0; a < lay.arenas; a++) {
        const cx = lay.origin[0] + a % lay.cols * lay.pitch + lay.L / 2;
        const cz = lay.origin[1] + Math.floor(a / lay.cols) * lay.pitch + lay.L / 2;
        for (const row of fence.fenceRows(lay.L / 2)) {
            const model = [...row.model];
            model[12] += cx; model[14] += cz;
            rows.push({ ...row, model });
        }
    }
    const vis = new VisibilitySystem(engine, { maxInstances: rows.length });
    const mega = new MegabufferBuilder(StandardLayout.strideFloats);
    const { post, rail } = fence.fenceMeshes();
    mega.addMesh('post', post.vertices, post.indices);
    mega.addMesh('rail', rail.vertices, rail.indices);
    vis.setGeometry(mega);
    const data = new InstanceBuffer(rows.length);
    rows.forEach((row, k) => data.write(k, row.model, row.bounds, row.mesh === 'post' ? 0 : 1));
    vis.setInstances(data.data, data.count);
    const wood = rgb => new ToonMaterial(engine, { baseColor: [...rgb, 1], rampSteps: 3,
        rimLightColor: [0.1, 0.08, 0.06], rimLightPower: 3, specularThreshold: 0.98 });
    const posts = wood([0.2, 0.13, 0.08]), rails = wood([0.25, 0.18, 0.11]);
    const batches = [
        vis.createInstanceDrawBatch(pass, 'post', 0, { targetFormat, material: { kind: 'toon', instance: posts } }),
        vis.createInstanceDrawBatch(pass, 'rail', 1, { targetFormat, material: { kind: 'toon', instance: rails } }),
    ];
    vis.upload();
    vis.setInstanceCount(rows.length);
    return {
        upload: () => vis.upload(),
        getComputeStages: () => vis.getComputeStages(),
        destroy() { batches.forEach(b => vis.removeDrawBatch(b)); vis.destroy(); posts.destroy(); rails.destroy(); },
    };
}

export default async function scene({ engine, width, height, scale = 1, settings = {}, trackEvery = 6 }) {
    const s = readSettings(settings);
    const plan = planTiles(s);
    hookCopies();
    const load = name => import(new URL(`${name}.ts`, COPIES).href);
    const [models, field, flashes, fence] = await Promise.all(['models', 'field', 'flashes', 'fence'].map(load));
    const assets = loadHerdAssets(engine);
    const tables = {};
    for (const species of ['prey', 'wolf'])
        if (plan.drivers[species] !== 'rule') tables[species] = MlpBrain.byGenome(engine, plan.drivers[species]);
    const lane = buildLane(engine, plan.geometry, { ...plan.lane, monitorFinite: false,
        brains: { prey: tables.prey ?? 'rule', wolf: tables.wolf ?? 'rule' },
        seedOf: () => s.seed, rowOf: (a, species) => plan.rowOf[species][a] });
    const { sys, lay } = lane, def = sys.definition;
    // The look sizes its field by the world's half-width; a margin keeps its fence off the tiles'.
    const half = Math.max(lay.spanX, lay.spanZ) / 2 + MARGIN;
    const pens = { createHerdField(e, pass, h, grass, format) {
        const ground = field.createHerdField(e, pass, h, grass, format), fences = tileFences(e, pass, format, fence, lay);
        return { upload() { ground.upload(); fences.upload(); },
            getComputeStages: () => [...ground.getComputeStages(), ...fences.getComputeStages()],
            destroy() { fences.destroy(); ground.destroy(); } };
    } };
    const look = buildLook(engine, { agents: sys, definition: { world: { maxX: half }, species: def.species, dt: def.dt } },
        { models, field: pens, flashes }, assets, { width, height, look: s.look, scale, flashes: false });
    // Taken after the look adds its stages (as herd-wolves.mjs does): one step, then the draw-only chain.
    if (sys.advance(def.dt) !== 1) throw new Error('film: coevo-arena\'s clock did not select one step.');
    sys.upload();
    const chain = sys.getComputeStages();
    sys.advance(0);
    const instances = sys.getComputeStages();
    const dt = def.dt, fastest = Math.max(...def.species.map(sp => sp.maxSpeed));
    const tracks = [
        track('agents', sys.buffer, { every: trackEvery, dt, preset: 'agents', fields: undefined, add: [{ word: 11, store: 'f16', interp: 'step' }] }),
        track('phase', look.phase, { stride: 4, every: trackEvery, dt, fields: [{ word: 0, store: 'unorm16', interp: 'wrap', maxRate: 4 }] }),
        track('state', sys.state, { every: 1, dt }),
    ];
    // Respawns are teleports. Do not interpolate them across the tile.
    for (const f of tracks[0].layout.fields) if (f.interp === 'hermite') f.jump = fastest * trackEvery * dt * 1.05;

    const camera = new Camera(60, width / height, 0.5, 1000);
    const defaultView = topDownView(lay, width / height);
    const sources = {};
    for (const path of ['tools/agents/coevolve-eval.mjs', 'tools/agents/model.mjs'])
        sources[path] = sha256(readFileSync(resolve(ROOT, path)));

    return {
        dt,
        tracks,
        // Complete persistent state: records, held actions, the tick, the animals' phases, catch
        // counters and the contact solver's own tick (its window phase). The rest is rebuilt each step.
        checkpoint: [
            ['agents', sys.buffer], ['actions', sys.act], ['state', sys.state], ['phase', look.phase],
            ['counters', lane.counters], ['stats', lane.stats], ...(lane.solver ? [['contacts', lane.solver.counters]] : []),
        ].map(([name, buffer]) => ({ name, buffer })),
        settings: s,
        defaultView,
        info: { tiles: plan.tiles, run: s.run, experiment: plan.run.configHash, environment: plan.lane.environment,
            layout: `${lay.cols}×${lay.rows} tiles of ${lay.L} m, ${lay.prey} prey and ${lay.wolves} wolves each`,
            grid: { arenas: lay.arenas, cols: lay.cols, wolves: lay.wolves, prey: lay.prey, L: lay.L, gap: lay.gap },
            episodeSteps: plan.run.config.T, networks: plan.networks, wgsl: sources },
        /** One exact step, its own submission, waited for. */
        async step() {
            sys.upload();
            engine.render(chain);
            await engine.device.queue.onSubmittedWorkDone();
        },
        /** Points the camera: an orbit `{ target, distance, azimuth, polar }`, near and far as herd-wolves.mjs sets them. */
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
        /** The agents buffer, for a determinism check. */
        agentsBuffer: sys.buffer,
        destroy() {
            lane.destroy();
            for (const table of Object.values(tables)) table.destroy();
            look.destroy();
            assets.destroy();
            engine.clearPasses();
        },
    };
}
