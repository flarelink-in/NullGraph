// Phase 2, step 14, phase 14.0: spike S7 (docs/agents-coevolution-design.md §16). One generation of
// co-evolution, end to end: Herd & Wolves in arenas (tiles of one world), MlpBrain.byGenome brains,
// episodes, per-arena catch counters, the ES update on the CPU, prey and wolves alternating.
// A measurement, not a test suite: its numbers go in the design.
//
//   Node + Dawn, on every adapter Dawn offers:  npm run spike:s7
//   Chrome:  npm run drive -- --serve . "test/spikes/coevo/index.html" --wait-for "S7 done" --timeout 2400
//   One part:  NG_SPIKE_PART=checks|rate|noise|learn under Node, ?part=… in Chrome (default all)
//
// Parts:
//   checks  I1: no neighbour list reaches another arena (600 steps, 64 arenas). I2: one arena covering
//           the world, on the rule, gives createHerd's records word for word (300 steps). E5: the arena
//           counters add up to the scenario's own count.
//   rate    seconds per generation (spawn, rows, T steps, readback, update) at 16, 32 and 64 arenas
//           of 256 prey + 16 wolves, and T = 300, 600, 1,200 steps.
//   noise   64 arenas, clone against clone, each arena its own spawn: the spread of catches per arena
//           at each T, against the spread between candidates in a real generation.
//   learn   20 generations from the clones (64 arenas, 32 pairs, T = 600), then 64 held-out spawns:
//           catches per arena for clone/clone, evolved prey/clone wolves, clone prey/evolved wolves,
//           evolved/evolved and the rule/the rule; and what the herds look like at the end (exploits).
import { harnessEnv } from '../../gpu/harness-env.mjs';
import * as A from '../../../dist/agents.js';
import { NullGraph } from '../../../dist/index.js';
import { DRIVER, HERD_CATCHES, HERD_DEFAULTS, SETTINGS, createHerd, herdDefinition, herdRule } from '../../../tools/agents/scenario/herd-wolves.mjs';
import { networkFromModel } from '../../../tools/agents/model.mjs';
import { TRAIN_DEFAULTS, flatParams, networkOf, shapeOf } from '../../../tools/agents/mlp-train.mjs';

const { log, check, fatal, finish, inBrowser } = harnessEnv('__S7_RESULTS__');
const param = name => (inBrowser ? new URLSearchParams(location.search).get(name) : globalThis.process?.env[`NG_SPIKE_${name.toUpperCase()}`]);
const PART = param('part') || 'all';
const want = part => PART === 'all' || PART === part;
const WOLVES = 16, PREY = 256, GAP = 10;               // per arena; the gap is more than the 8 m sense radius
const SIGMA = 0.02, RATE = 0.01, DECAY = 0.005;        // Salimans et al. 2017's defaults
const ARCHIVE_EVERY = 5, ARCHIVE_SHARE = 0.5, ARCHIVE_CAP = 32;
const SUBMIT_MS = 100;                                  // steps per submission are chosen to take about this long
const STEP_STOP_MS = 1500;
const fmt = n => n.toLocaleString('en-US');
const mean = v => v.reduce((s, x) => s + x, 0) / v.length;
const sd = v => { const m = mean(v); return Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / Math.max(1, v.length - 1)); };
const ci = v => 1.96 * sd(v) / Math.sqrt(v.length);
const f32 = v => (Number.isInteger(v) ? v.toFixed(1) : String(v));
const mix = (...xs) => xs.reduce((h, x) => (Math.imul(h ^ (x >>> 0), 0x9e3779b1) + 0x7f4a7c15) >>> 0, 0x85ebca6b);

// ---------------------------------------------------------------------------------------------
// The arena layout and scenario (design §2, §5), as spike code
// ---------------------------------------------------------------------------------------------
/** Tiles of side L (Herd & Wolves' density at its own count), C columns, `gap` apart, centred. */
function layout(arenas, wolves = WOLVES, prey = PREY, gap = GAP) {
    const perArena = wolves + prey, L = 2 * 80 * Math.sqrt(perArena / HERD_DEFAULTS.count);
    const cols = Math.ceil(Math.sqrt(arenas)), rows = Math.ceil(arenas / cols), pitch = L + gap;
    const spanX = cols * pitch - gap, spanZ = rows * pitch - gap;
    return { arenas, wolves, prey, perArena, L, gap, cols, pitch, origin: [-spanX / 2, -spanZ / 2], spanX, spanZ,
        count: arenas * perArena, wolfTotal: arenas * wolves };
}
const arenaWGSL = lay => `
const ARENA_ORIGIN_X: f32 = ${f32(lay.origin[0])};
const ARENA_ORIGIN_Z: f32 = ${f32(lay.origin[1])};
const ARENA_PITCH: f32 = ${f32(lay.pitch)};
const ARENA_L: f32 = ${f32(lay.L)};
const ARENA_COLS = ${lay.cols}u;
fn arenaLo(arena: u32) -> vec2<f32> {
    return vec2<f32>(ARENA_ORIGIN_X + f32(arena % ARENA_COLS) * ARENA_PITCH, ARENA_ORIGIN_Z + f32(arena / ARENA_COLS) * ARENA_PITCH);
}`;
// Sensors.bounds() with the arena's tile for walls (agentSenseShaders.ts), arena in user1 (ids.w).
const wallSensor = lay => A.Sensors.custom({ name: 'bounds', width: 2, wgsl: `${arenaWGSL(lay)}
fn bounds(i: u32, a: Agent) -> array<f32, 2> {
    let sp = species[a.ids.x];
    let sn = sin(a.velHeading.w);
    let cs = cos(a.velHeading.w);
    let side = vec2<f32>(cs, -sn);
    let forward = vec2<f32>(sn, cs);
    let lo = arenaLo(a.ids.w);
    let hi = lo + vec2<f32>(ARENA_L, ARENA_L);
    let p = a.posRadius.xz;
    var distance = p.x - lo.x;
    var normal = vec2<f32>(-1.0, 0.0);
    if (hi.x - p.x < distance) { distance = hi.x - p.x; normal = vec2<f32>(1.0, 0.0); }
    if (p.y - lo.y < distance) { distance = p.y - lo.y; normal = vec2<f32>(0.0, -1.0); }
    if (hi.y - p.y < distance) { distance = hi.y - p.y; normal = vec2<f32>(0.0, 1.0); }
    let magnitude = max(0.0, 1.0 - max(distance, 0.0) / sp.senseRadius);
    return array<f32, 2>(dot(normal, side) * magnitude, dot(normal, forward) * magnitude);
}` });
// Actions.steering() with the tile's walls (agentActShaders.ts).
const steering = lay => A.Actions.custom({ name: 'steer', width: 2, wgsl: `${arenaWGSL(lay)}
fn steer(i: u32) {
    var a = agents[i];
    let sp = species[a.ids.x];
    let a0 = act[i * ACT_DIM];
    let a1 = act[i * ACT_DIM + 1u];
    var turn = 0.0;
    var targetSpeed = 0.0;
    if ((bitcast<u32>(a0) & 0x7f800000u) != 0x7f800000u &&
        (bitcast<u32>(a1) & 0x7f800000u) != 0x7f800000u) {
        turn = clamp(a0, -1.0, 1.0);
        targetSpeed = (clamp(a1, -1.0, 1.0) + 1.0) * 0.5 * sp.maxSpeed * a.state.z;
    }
    var h = wrapAngle(a.velHeading.w + turn * sp.maxTurn * DT);
    let dv = sp.accel * DT;
    var speed = length(a.velHeading.xz);
    speed += clamp(targetSpeed - speed, -dv, dv);
    var fx = sin(h);
    var fz = cos(h);
    var x = a.posRadius.x + fx * speed * DT;
    var z = a.posRadius.z + fz * speed * DT;
    let radius = a.posRadius.w;
    let lo = arenaLo(a.ids.w);
    let hi = lo + vec2<f32>(ARENA_L, ARENA_L);
    var loX = lo.x + radius;
    var hiX = hi.x - radius;
    var loZ = lo.y + radius;
    var hiZ = hi.y - radius;
    if (loX > hiX) { loX = (lo.x + hi.x) * 0.5; hiX = loX; }
    if (loZ > hiZ) { loZ = (lo.y + hi.y) * 0.5; hiZ = loZ; }
    if (x < loX) { x = loX; if (fx < 0.0) { h = wrapAngle(-h); } }
    else if (x > hiX) { x = hiX; if (fx > 0.0) { h = wrapAngle(-h); } }
    fz = cos(h);
    if (z < loZ) { z = loZ; if (fz < 0.0) { h = wrapAngle(PI - h); } }
    else if (z > hiZ) { z = hiZ; if (fz > 0.0) { h = wrapAngle(PI - h); } }
    fx = sin(h); fz = cos(h);
    a.posRadius.x = x;
    a.posRadius.z = z;
    a.velHeading = vec4<f32>(fx * speed, 0.0, fz * speed, h);
    agents[i] = a;
}` });
// herdRespawnWGSL with the respawn point inside the tile, and a catch counter per arena. The catch
// ring, which only the demo's flashes read, is dropped: a step stage binds at most three buffers.
const respawnWGSL = lay => `${arenaWGSL(lay)}
@group(1) @binding(0) var<storage, read> settings: array<f32>;
@group(1) @binding(1) var<storage, read_write> stats: array<atomic<u32>, 4>;
@group(1) @binding(2) var<storage, read_write> arenaCatches: array<atomic<u32>>;
fn randomPointInArena(i: u32, tick: u32) -> vec2<f32> {
    let radius = agents[i].posRadius.w;
    let lo0 = arenaLo(agents[i].ids.w);
    let hi0 = lo0 + vec2<f32>(ARENA_L, ARENA_L);
    let mid = (lo0 + hi0) * 0.5;
    let lo = min(lo0 + radius, mid);
    let hi = max(hi0 - radius, mid);
    return lo + (hi - lo) * vec2<f32>(rand01(i, tick, 0u), rand01(i, tick, 1u));
}
fn step(i: u32, tick: u32) {
    if (agents[i].ids.x == SPECIES_WOLF) {
        agents[i].state.w = max(0.0, agents[i].state.w - DT);
        if ((agents[i].ids.y & 512u) != 0u) {
            agents[i].state.w = settings[${SETTINGS.REST_SECONDS}];
            agents[i].ids.y &= ~512u;
        }
        if (agents[i].state.w > 0.0) {
            agents[i].ids.y |= 2u;
            agents[i].velHeading.x = 0.0; agents[i].velHeading.y = 0.0; agents[i].velHeading.z = 0.0;
        } else { agents[i].ids.y &= ~2u; }
    } else if ((agents[i].ids.y & 256u) != 0u) {
        atomicAdd(&stats[0], 1u);
        atomicAdd(&arenaCatches[agents[i].ids.w], 1u);
        respawn(i, randomPointInArena(i, tick), rand01(i, tick, 2u) * 6.2831853 - 3.1415927);
        agents[i].ids.y &= ~256u;
    }
}`;
function arenaDefinition(lay) {
    const o = HERD_DEFAULTS;
    return A.defineAgents({
        capacity: lay.count,
        world: { min: lay.origin, max: [lay.origin[0] + lay.spanX, lay.origin[1] + lay.spanZ] },
        cellSize: 8,
        maxScanPerCell: o.scanBudget,
        species: [
            { name: 'prey', maxSpeed: o.preySpeed, maxTurn: 4, accel: 6, senseRadius: 8, radius: 0.4 },
            { name: 'wolf', maxSpeed: o.wolfSpeed, maxTurn: 3, accel: 5, senseRadius: 8, radius: 0.6 },
        ],
        sensors: [A.Sensors.self(), A.Sensors.nearest({ name: 'wolf', species: ['wolf'], k: 1 }),
            A.Sensors.nearest({ name: 'prey', species: ['prey'], k: 3 }), wallSensor(lay)],
        actions: steering(lay),
    });
}
const ruleWGSL = `const ACT_TURN = 0u;\nconst ACT_THROTTLE = 1u;\n${herdRule(DRIVER)}`;

function buildArenas(engine, lay, thinkEvery = HERD_DEFAULTS.thinkEvery) {
    const device = engine.device, def = arenaDefinition(lay), owned = [];
    const buffer = (size, usage, data) => { const b = device.createBuffer({ size, usage }); owned.push(b); if (data) device.queue.writeBuffer(b, 0, data); return b; };
    const settingsData = new Float32Array(4);
    settingsData[SETTINGS.FLOCKING] = 1; settingsData[SETTINGS.FLEEING] = 1; settingsData[SETTINGS.REST_SECONDS] = HERD_DEFAULTS.restSeconds;
    const settings = buffer(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, settingsData);
    const stats = buffer(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const arenaCatches = buffer(Math.max(16, lay.arenas * 4), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const sys = new A.AgentSystem(engine, def, { thinkEvery });
    sys.addStepStage({ name: 'catches', wgsl: HERD_CATCHES });
    sys.addStepStage({ name: 'respawn', wgsl: respawnWGSL(lay), buffers: [settings, stats, arenaCatches] });
    const brains = [];
    return {
        lay, def, sys, settings, stats, arenaCatches,
        rule: () => ({ kind: 'stage', wgsl: ruleWGSL, buffers: [settings], destroy() {} }),
        setBrains(prey, wolf) { sys.setBrain(prey, { species: 'prey' }); sys.setBrain(wolf, { species: 'wolf' }); brains.push(prey, wolf); },
        /** Every agent spawned in its arena from `seedOf(arena)`, on the rows `rowOf(arena, 'prey' | 'wolf')`. */
        spawn(seedOf, rowOf) {
            for (let a = 0; a < lay.arenas; a++) {
                const lo = [lay.origin[0] + (a % lay.cols) * lay.pitch, lay.origin[1] + Math.floor(a / lay.cols) * lay.pitch];
                const cx = lo[0] + lay.L / 2, cz = lo[1] + lay.L / 2, inset = lay.L / 2 - 1, s = seedOf(a);
                for (let slot = 0; slot < lay.perArena; slot++) {
                    const wolf = slot < lay.wolves;
                    const i = wolf ? a * lay.wolves + slot : lay.wolfTotal + a * lay.prey + (slot - lay.wolves);
                    sys.spawn(i, { position: [cx + (A.rand01(slot, s, 0) * 2 - 1) * inset, 0.08, cz + (A.rand01(slot, s, 1) * 2 - 1) * inset],
                        species: wolf ? 'wolf' : 'prey', heading: A.rand01(slot, s, 2) * Math.PI * 2 - Math.PI,
                        genome: rowOf(a, wolf ? 'wolf' : 'prey'), user1: a });
                }
            }
            device.queue.writeBuffer(stats, 0, new Uint32Array(4));
            device.queue.writeBuffer(arenaCatches, 0, new Uint32Array(Math.max(4, lay.arenas)));
            sys.upload();
        },
        destroy() { sys.destroy(); for (const b of owned) b.destroy(); },
    };
}

async function readU32(device, src, words) {
    const dst = device.createBuffer({ size: words * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, dst, 0, words * 4);
    device.queue.submit([enc.finish()]);
    await dst.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(dst.getMappedRange().slice(0));
    dst.unmap();
    dst.destroy();
    return out;
}

/** Runs `steps` steps in submissions of about SUBMIT_MS each (probed once per arena set). */
async function runSteps(engine, arenas, steps) {
    const { sys } = arenas;
    if (!arenas.chain) {
        if (sys.advance(sys.definition.dt) !== 1) throw new Error('S7: advance(dt) did not select one step');
        arenas.chain = sys.getComputeStages();
        const t0 = performance.now();
        engine.render([...arenas.chain, ...arenas.chain]);       // pipelines compile on first use
        await engine.device.queue.onSubmittedWorkDone();
        const t1 = performance.now();
        engine.render(arenas.chain);
        await engine.device.queue.onSubmittedWorkDone();
        const probe = performance.now() - t1;
        if (probe > STEP_STOP_MS) throw new Error(`S7: one step took ${probe.toFixed(0)} ms`);
        arenas.repeat = Math.max(1, Math.min(64, Math.round(SUBMIT_MS / Math.max(probe, 0.05))));
        arenas.compileMs = t1 - t0;
        steps -= 3;                                              // the three steps just run count
    }
    for (let done = 0; done < steps; ) {
        const n = Math.min(arenas.repeat, steps - done);
        engine.render(Array.from({ length: n }, () => arenas.chain).flat());
        await engine.device.queue.onSubmittedWorkDone();
        done += n;
    }
}

// ---------------------------------------------------------------------------------------------
// The optimizer (design §6), as spike code
// ---------------------------------------------------------------------------------------------
function gaussian(n, seed, t, k) {
    const out = new Float64Array(n), tick = (t * 4096 + k) >>> 0, stream = (seed * 2 + 101) >>> 0;
    for (let j = 0; j < n; j += 2) {
        const u1 = 1 - A.rand01(j, tick, stream), u2 = A.rand01(j + 1, tick, stream);
        const r = Math.sqrt(-2 * Math.log(u1)), angle = 2 * Math.PI * u2;
        out[j] = r * Math.cos(angle);
        if (j + 1 < n) out[j + 1] = r * Math.sin(angle);
    }
    return out;
}
function centeredRanks(scores) {
    const order = scores.map((s, i) => i).sort((a, b) => scores[a] - scores[b] || a - b), u = new Float64Array(scores.length);
    order.forEach((i, r) => { u[i] = r / (scores.length - 1) - 0.5; });
    return u;
}
class Species {
    constructor(name, network, rows) {
        this.name = name; this.shape = shapeOf(network); this.x = flatParams(network);
        this.m = new Float64Array(this.x.length); this.v = new Float64Array(this.x.length); this.steps = 0;
        this.archive = [this.x.slice()]; this.rows = rows; this.start = this.x.slice();
        this.floats = A.mlpLayout(network).floats;
    }
    net(params = this.x) { return networkOf(this.shape, params); }
    pack(paramsList) {
        const out = new Float32Array(paramsList.length * this.floats);
        paramsList.forEach((p, r) => A.packMlpWeights(this.net(p), out.subarray(r * this.floats, (r + 1) * this.floats)));
        return out;
    }
    update(eps, u, K) {
        const n = this.x.length, g = new Float64Array(n);
        for (let k = 0; k < K; k++) {
            const w = (u[2 * k] - u[2 * k + 1]) / (2 * K * SIGMA), e = eps[k];
            for (let j = 0; j < n; j++) g[j] += w * e[j];
        }
        const { beta1, beta2, epsilon } = TRAIN_DEFAULTS;
        this.steps++;
        for (let j = 0; j < n; j++) {
            const gj = g[j] - DECAY * this.x[j];
            this.m[j] = beta1 * this.m[j] + (1 - beta1) * gj;
            this.v[j] = beta2 * this.v[j] + (1 - beta2) * gj * gj;
            const mh = this.m[j] / (1 - beta1 ** this.steps), vh = this.v[j] / (1 - beta2 ** this.steps);
            this.x[j] += RATE * mh / (Math.sqrt(vh) + epsilon);
        }
    }
}

/**
 * One generation: species `evolving` perturbed in K = arenas / 2 mirrored pairs (pair k in arenas 2k,
 * 2k+1, the same spawn and opponent), the other species frozen on its current weights or an archived
 * champion. Returns the scores and the timing.
 */
async function generation(engine, arenas, sp, t, seed, steps, { update = true } = {}) {
    const K = arenas.lay.arenas / 2, evolving = t % 2 === 0 ? sp.prey : sp.wolf, frozen = evolving === sp.prey ? sp.wolf : sp.prey;
    const t0 = performance.now();
    const eps = Array.from({ length: K }, (_, k) => gaussian(evolving.x.length, seed, t, k));
    const candidates = [];
    for (let k = 0; k < K; k++) {
        const plus = evolving.x.slice(), minus = evolving.x.slice();
        for (let j = 0; j < plus.length; j++) { plus[j] += SIGMA * eps[k][j]; minus[j] -= SIGMA * eps[k][j]; }
        candidates.push(plus, minus);
    }
    evolving.brain.setRows(evolving.pack(candidates), { first: 0 });
    frozen.brain.setRows(frozen.pack([frozen.x, ...frozen.archive]), { first: 2 * K });
    // The frozen side's opponent per group of two pairs (four arenas: a wolf block), seeded.
    const opponentRow = a => {
        const group = Math.floor(a / 4);
        if (frozen.archive.length < 2 || A.rand01(group, t, seed + 7) >= ARCHIVE_SHARE) return 2 * K;
        return 2 * K + 1 + Math.floor(A.rand01(group, t, seed + 11) * frozen.archive.length);
    };
    arenas.spawn(a => mix(seed, t, Math.floor(a / 2)), (a, name) => (name === evolving.name ? a : opponentRow(a)));
    await runSteps(engine, arenas, steps);
    const catches = await readU32(engine.device, arenas.arenaCatches, arenas.lay.arenas);
    const stats = await readU32(engine.device, arenas.stats, 4);
    const scores = Array.from(catches, c => (evolving === sp.prey ? -c : c));
    if (update) evolving.update(eps, centeredRanks(scores), K);
    if (update && t % ARCHIVE_EVERY === ARCHIVE_EVERY - 1) {
        for (const s of [sp.prey, sp.wolf]) {
            s.archive.push(s.x.slice());
            if (s.archive.length > ARCHIVE_CAP) s.archive.splice(1, 1);
        }
    }
    return { evolving: evolving.name, catches: Array.from(catches), total: stats[0], scores, seconds: (performance.now() - t0) / 1000 };
}

// ---------------------------------------------------------------------------------------------
async function loadJSON(url) {
    return inBrowser ? (await fetch(url)).json() : JSON.parse((await import('node:fs')).readFileSync(url, 'utf8'));
}
const MODELS = new URL('../../../tools/agents/models/herd-wolves/', import.meta.url);

async function runAdapter(device, name) {
    const engine = new NullGraph();
    const init = await engine.init(null, { device, width: 64, height: 64 });
    if (!init.success) throw new Error(`S7: engine: ${init.error}`);
    const clones = {};
    for (const s of ['prey', 'wolf']) clones[s] = networkFromModel(await loadJSON(new URL(`${s}.mlp.json`, MODELS)), herdDefinition(), `${s}.mlp.json`);
    const result = { adapter: name, host: inBrowser ? 'chrome' : 'node' };
    const rowsFor = K => 2 * K + 1 + ARCHIVE_CAP;
    const speciesFor = (arenas, K) => {
        const sp = { prey: new Species('prey', clones.prey, rowsFor(K)), wolf: new Species('wolf', clones.wolf, rowsFor(K)) };
        for (const s of [sp.prey, sp.wolf]) s.brain = A.MlpBrain.byGenome(engine, Array.from({ length: s.rows }, () => s.net()));
        arenas.setBrains(sp.prey.brain, sp.wolf.brain);
        return sp;
    };
    const release = (arenas, sp) => { arenas.destroy(); if (sp) for (const s of [sp.prey, sp.wolf]) s.brain.destroy(); };

    if (want('checks')) {
        device.pushErrorScope('validation');
        // I2: one arena covering the world, on the rule, against createHerd, 300 steps.
        const herd = createHerd(engine, { count: HERD_DEFAULTS.count, thinkEvery: HERD_DEFAULTS.thinkEvery });
        const one = buildArenas(engine, layout(1, HERD_DEFAULTS.count / 16, HERD_DEFAULTS.count * 15 / 16));
        const rule = one.rule();
        one.sys.setBrain(rule);
        one.spawn(() => HERD_DEFAULTS.seed, () => 0);
        for (let s = 0; s < 300; s++) { await herd.agents.step(); await one.sys.step(); }
        const a = await readU32(device, herd.agents.buffer, HERD_DEFAULTS.count * 16), b = await readU32(device, one.sys.buffer, HERD_DEFAULTS.count * 16);
        let differ = 0;
        for (let i = 0; i < a.length; i += 16) for (let w = 0; w < 16; w++) if (a[i + w] !== b[i + w]) { differ++; break; }
        const herdCatches = (await readU32(device, herd.stats, 4))[0], oneCatches = (await readU32(device, one.stats, 4))[0];
        check(`${name}: I2: one arena covering the world, on the rule, gives createHerd's records word for word over 300 steps`,
            differ === 0 && herdCatches === oneCatches && herdCatches > 0, `${fmt(differ)} of ${fmt(HERD_DEFAULTS.count)} records differ; catches ${herdCatches} and ${oneCatches}`);
        herd.destroy(); one.destroy();
        // I1 and E5: 64 arenas on the clones, 600 steps; every neighbour list stays in its arena.
        const lay = layout(64), arenas = buildArenas(engine, lay), sp = speciesFor(arenas, 32);
        arenas.spawn(a => mix(1, a), () => 0);
        let crossing = 0, sampled = 0;
        for (let s = 0; s < 12; s++) {
            await runSteps(engine, arenas, 50);
            const nb = await readU32(device, arenas.sys.neighbours, lay.count * arenas.sys.neighbourStride);
            const rec = await readU32(device, arenas.sys.buffer, lay.count * 16);
            for (let i = 0; i < lay.count; i++) {
                for (let k = 0; k < arenas.sys.neighbourStride; k++) {
                    const j = nb[i * arenas.sys.neighbourStride + k];
                    if (j === 0xffffffff) continue;
                    sampled++;
                    if (rec[j * 16 + 15] !== rec[i * 16 + 15]) crossing++;
                }
            }
        }
        const catches = await readU32(device, arenas.arenaCatches, lay.arenas), total = (await readU32(device, arenas.stats, 4))[0];
        const sum = catches.reduce((s, c) => s + c, 0);
        check(`${name}: I1: over 600 steps in 64 arenas, no neighbour list reaches another arena`, crossing === 0 && sampled > 0,
            `${fmt(crossing)} of ${fmt(sampled)} sampled neighbour entries cross`);
        check(`${name}: E5: the arena counters add up to the scenario's count`, sum === total && total > 0, `${fmt(sum)} and ${fmt(total)}`);
        release(arenas, sp);
        const error = await device.popErrorScope();
        check(`${name}: checks ran without a WebGPU error`, error === null, error?.message ?? '');
    }

    if (want('rate')) {
        log(`   estimate from phase 11.3 (64 arenas, 17,408 agents, T = 600, thinkEvery ${HERD_DEFAULTS.thinkEvery}): ~40/min on NVIDIA, ~20/min under Node on AMD, ~22/min in Chrome`);
        result.rate = [];
        for (const count of [16, 32, 64]) {
            const lay = layout(count), arenas = buildArenas(engine, lay), sp = speciesFor(arenas, count / 2);
            for (const steps of [300, 600, 1200]) {
                await generation(engine, arenas, sp, 0, 1, steps);       // warm (and the first compiles)
                const g = await generation(engine, arenas, sp, 1, 1, steps);
                result.rate.push({ arenas: count, agents: lay.count, steps, seconds: g.seconds, perMinute: 60 / g.seconds, repeat: arenas.repeat });
            }
            release(arenas, sp);
        }
        log(`\n== S7 rate · ${result.host} · ${name} — seconds per generation (and generations a minute), arenas of ${PREY} prey + ${WOLVES} wolves, thinkEvery ${HERD_DEFAULTS.thinkEvery}`);
        log(`   ${'arenas'.padEnd(8)}${'agents'.padStart(8)}${[300, 600, 1200].map(s => `T = ${s}`.padStart(20)).join('')}`);
        for (const count of [16, 32, 64]) {
            const rows = result.rate.filter(r => r.arenas === count);
            log(`   ${String(count).padEnd(8)}${fmt(rows[0].agents).padStart(8)}${rows.map(r => `${r.seconds.toFixed(2)} s (${r.perMinute.toFixed(1)}/min)`.padStart(20)).join('')}`);
        }
    }

    if (want('noise')) {
        result.noise = [];
        const lay = layout(64), arenas = buildArenas(engine, lay), sp = speciesFor(arenas, 32);
        for (const steps of [300, 600, 1200]) {
            arenas.spawn(a => mix(5000, steps, a), () => 0);           // row 0 holds the clone on both sides
            await runSteps(engine, arenas, steps);
            const c = Array.from(await readU32(device, arenas.arenaCatches, lay.arenas));
            result.noise.push({ steps, mean: mean(c), sd: sd(c), cv: sd(c) / mean(c) });
        }
        release(arenas, sp);
        log(`\n== S7 noise · ${result.host} · ${name} — catches per arena, 64 arenas of the clones, each its own spawn`);
        for (const r of result.noise) log(`   T = ${String(r.steps).padEnd(6)} mean ${r.mean.toFixed(1)}, sd ${r.sd.toFixed(2)}, cv ${(100 * r.cv).toFixed(1)}%`);
    }

    if (want('learn')) {
        const lay = layout(64), K = 32, arenas = buildArenas(engine, lay), sp = speciesFor(arenas, K), T = 600, gens = [];
        for (let t = 0; t < 20; t++) {
            const g = await generation(engine, arenas, sp, t, 1, T);
            gens.push({ t, evolving: g.evolving, meanCatches: mean(g.catches), scoreSd: sd(g.scores), seconds: g.seconds });
        }
        // Held-out: 64 spawns no generation used; row 0 the clone, row 1 the evolved weights.
        const evaluate = async (preyRow, wolfRow, useRule = false) => {
            if (useRule) { const r = arenas.rule(); arenas.sys.setBrain(r); }
            arenas.spawn(a => mix(900001, a), (a, nm) => (nm === 'prey' ? preyRow : wolfRow));
            await runSteps(engine, arenas, T);
            const c = Array.from(await readU32(device, arenas.arenaCatches, lay.arenas));
            const rec = await readU32(device, arenas.sys.buffer, lay.count * 16), recF = new Float32Array(rec.buffer);
            // Exploits: prey within 1 m of their tile's wall; mean distance to the nearest prey of the arena; mean speeds.
            let nearWall = 0, nnSum = 0, preySpeed = 0, wolfSpeed = 0;
            for (let a = 0; a < lay.arenas; a++) {
                const lo = [lay.origin[0] + (a % lay.cols) * lay.pitch, lay.origin[1] + Math.floor(a / lay.cols) * lay.pitch];
                const first = lay.wolfTotal + a * lay.prey;
                for (let p = 0; p < lay.prey; p++) {
                    const i = first + p, x = recF[i * 16], z = recF[i * 16 + 2];
                    if (Math.min(x - lo[0], lo[0] + lay.L - x, z - lo[1], lo[1] + lay.L - z) < 1) nearWall++;
                    let best = Infinity;
                    for (let q = 0; q < lay.prey; q++) {
                        if (q === p) continue;
                        const dx = recF[(first + q) * 16] - x, dz = recF[(first + q) * 16 + 2] - z;
                        best = Math.min(best, dx * dx + dz * dz);
                    }
                    nnSum += Math.sqrt(best);
                    preySpeed += Math.hypot(recF[i * 16 + 4], recF[i * 16 + 6]);
                }
                for (let w = 0; w < lay.wolves; w++) { const i = a * lay.wolves + w; wolfSpeed += Math.hypot(recF[i * 16 + 4], recF[i * 16 + 6]); }
            }
            const preyN = lay.arenas * lay.prey;
            if (useRule) arenas.setBrains(sp.prey.brain, sp.wolf.brain);
            return { mean: mean(c), ci: ci(c), nearWall: nearWall / preyN, nearestPrey: nnSum / preyN, preySpeed: preySpeed / preyN, wolfSpeed: wolfSpeed / lay.wolfTotal };
        };
        for (const s of [sp.prey, sp.wolf]) s.brain.setRows(s.pack([s.start, s.x]), { first: 0 });
        result.learn = { generations: gens, held: {
            'clone prey · clone wolves': await evaluate(0, 0),
            'evolved prey · clone wolves': await evaluate(1, 0),
            'clone prey · evolved wolves': await evaluate(0, 1),
            'evolved prey · evolved wolves': await evaluate(1, 1),
            'rule · rule': await evaluate(0, 0, true),
        } };
        release(arenas, sp);
        log(`\n== S7 learn · ${result.host} · ${name} — 20 generations from the clones (64 arenas, 32 pairs, T = ${T}; prey on even generations)`);
        for (const g of gens) log(`   gen ${String(g.t).padStart(2)} ${g.evolving.padEnd(5)} mean catches/arena ${g.meanCatches.toFixed(1).padStart(6)}, score sd ${g.scoreSd.toFixed(2).padStart(6)}, ${g.seconds.toFixed(2)} s`);
        log('   held-out (64 spawns): catches per arena ± 95% CI · prey within 1 m of a wall · nearest prey (m) · prey speed · wolf speed (m/s)');
        for (const [k, r] of Object.entries(result.learn.held)) {
            log(`   ${k.padEnd(32)}${`${r.mean.toFixed(1)} ± ${r.ci.toFixed(1)}`.padStart(14)}${`${(100 * r.nearWall).toFixed(1)}%`.padStart(9)}${r.nearestPrey.toFixed(2).padStart(8)}${r.preySpeed.toFixed(2).padStart(8)}${r.wolfSpeed.toFixed(2).padStart(8)}`);
        }
    }
    return result;
}

const results = [];
const seen = new Set();
for (const powerPreference of ['high-performance', 'low-power']) {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference });
    if (!adapter) continue;
    const i = adapter.info ?? {};
    const key = [i.vendor, i.architecture, i.device, i.description].join('|');
    if (seen.has(key)) continue;
    seen.add(key);
    const name = [i.vendor, i.architecture].filter(Boolean).join(' ') || powerPreference;
    log(`S7: measuring on ${name}${i.description ? ` (${i.description})` : ''} (${inBrowser ? 'Chrome' : 'Node/Dawn'}), part ${PART}…`);
    const device = await adapter.requestDevice();
    results.push(await runAdapter(device, name));
    device.destroy();
}
if (!results.length) { fatal('No WebGPU adapter'); throw new Error('No WebGPU adapter'); }
log(`\nS7 JSON ${JSON.stringify(results)}`);
await finish(`S7: checks passed on ${results.length} adapter(s).`);
log('S7 done');
