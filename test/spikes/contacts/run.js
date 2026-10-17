// Phase 2, step 12, phase 12.0: spike S8 (docs/agents-queries-design.md §14).
// Compares Jacobi XZ sphere-separation contacts across three peer sources:
//   (a) today-lists   addStepStage reading Herd & Wolves' existing 1 wolf + 3 prey neighbour slots
//   (b) query-k8      addStepStage reading an unobserved k = 8 any-species query slice (radius 1.6 m)
//   (c) grid-3x3-1/2  standalone stages scanning PointGrid's 3x3 cell stencil directly (1 or 2 Jacobi iters)
// against `none` (no contacts).
//
//   Node + Dawn, on every adapter Dawn offers:  npm run spike:s8
//   Chrome:  npm run drive -- --serve . "test/spikes/contacts/index.html" --wait-for "S8 done" --timeout 2400
//   One part:  NG_SPIKE_PART=checks|cost|overlap|pileup|jitter under Node, ?part=… in Chrome (default all)
import { harnessEnv } from '../../gpu/harness-env.mjs';
import * as A from '../../../dist/agents.js';
import { NullGraph } from '../../../dist/index.js';
import { DRIVER, HERD_CATCHES, HERD_DEFAULTS, SETTINGS, herdDefinition, herdRespawnWGSL, herdRule } from '../../../tools/agents/scenario/herd-wolves.mjs';
import { networkFromModel } from '../../../tools/agents/model.mjs';

const { log, check, fatal, finish, inBrowser } = harnessEnv('__S8_RESULTS__');
const param = name => (inBrowser ? new URLSearchParams(location.search).get(name) : globalThis.process?.env[`NG_SPIKE_${name.toUpperCase()}`]);
const PART = param('part') || 'all';
if (!['all', 'checks', 'cost', 'overlap', 'pileup', 'jitter'].includes(PART)) throw new Error(`S8: unknown part "${PART}"`);
const want = part => PART === 'all' || PART === part;

const MODES = ['none', 'today-lists', 'query-k8', 'grid-3x3-1', 'grid-3x3-2'];
const CONTACT_MODES = MODES.slice(1);
const WOLVES = 16, PREY = 256, GAP = 10;
const SUBMIT_MS = 80, STEP_STOP_MS = 1500;
const MAX_PUSH = 0.25;
const fmt = n => n.toLocaleString('en-US');
const f32 = v => (Number.isInteger(v) ? v.toFixed(1) : String(v));
const mix = (...xs) => xs.reduce((h, x) => (Math.imul(h ^ (x >>> 0), 0x9e3779b1) + 0x7f4a7c15) >>> 0, 0x85ebca6b);

// ---------------------------------------------------------------------------------------------
// Contact WGSL builders (Jacobi accumulate + apply) for single-world and multi-arena layouts
// ---------------------------------------------------------------------------------------------
const pairPushWGSL = (compliance = 0.0) => `
fn pairPush(i: u32, pi: vec2<f32>, ri: f32, j: u32, pj: vec2<f32>, rj: f32) -> vec3<f32> {
    let rij = ri + rj;
    let dvec = pi - pj;
    let d2 = dot(dvec, dvec);
    if (d2 >= rij * rij) { return vec3<f32>(0.0); }
    let d = sqrt(d2);
    let pen = rij - d;
    var n: vec2<f32>;
    if (d > 1e-5) {
        n = dvec / d;
    } else {
        let lo = min(i, j);
        let hi = max(i, j);
        let ang = rand01(lo, hi, 0u) * 6.283185307;
        let sgn = select(-1.0, 1.0, i < j);
        n = sgn * vec2<f32>(cos(ang), sin(ang));
    }
    let wi = 1.0 / max(ri * ri, 1e-4);
    let wj = 1.0 / max(rj * rj, 1e-4);
    let share = wi / (wi + wj + ${f32(compliance)});
    return vec3<f32>(n * (pen * share), 1.0);
}`;

function buildTodayAccWGSL(compliance = 0.0) {
    return `
@group(1) @binding(0) var<storage, read_write> deltas: array<vec4<f32>>;
${pairPushWGSL(compliance)}
fn step(i: u32, tick: u32) {
    let me = agents[i];
    let pi = me.posRadius.xz;
    let ri = me.posRadius.w;
    var push = vec2<f32>(0.0);
    var count = 0.0;
    for (var s = 0u; s < NEIGHBOURS_PER_AGENT; s++) {
        let j = neighbours[i * NEIGHBOURS_PER_AGENT + s];
        if (j == NO_AGENT || j == i) { continue; }
        let other = agents[j];
        if ((other.ids.y & 1u) == 0u) { continue; }
        let c = pairPush(i, pi, ri, j, other.posRadius.xz, other.posRadius.w);
        push += c.xy;
        count += c.z;
    }
    deltas[i] = vec4<f32>(push, count, 0.0);
}`;
}

function buildQueryK8AccWGSL(k = 8, compliance = 0.0) {
    return `
@group(1) @binding(0) var<storage, read> crowd: array<u32>;
@group(1) @binding(1) var<storage, read_write> deltas: array<vec4<f32>>;
${pairPushWGSL(compliance)}
fn step(i: u32, tick: u32) {
    let me = agents[i];
    let pi = me.posRadius.xz;
    let ri = me.posRadius.w;
    var push = vec2<f32>(0.0);
    var count = 0.0;
    for (var s = 0u; s < ${k}u; s++) {
        let j = crowd[i * ${k}u + s];
        if (j == NO_AGENT || j == i) { continue; }
        let other = agents[j];
        if ((other.ids.y & 1u) == 0u) { continue; }
        let c = pairPush(i, pi, ri, j, other.posRadius.xz, other.posRadius.w);
        push += c.xy;
        count += c.z;
    }
    deltas[i] = vec4<f32>(push, count, 0.0);
}`;
}

function buildGrid3x3AccWGSL(def, compliance = 0.0) {
    return `${A.AGENT_STRUCT_WGSL}
@group(0) @binding(0) var<storage, read> agents: array<Agent>;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read> bodies: array<u32>;
@group(0) @binding(4) var<storage, read_write> deltas: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> tick: u32;
${A.AGENT_RANDOM_WGSL}
${A.GRID_WINDOW_WGSL}
${A.pointCellWGSL(def)}
${pairPushWGSL(compliance)}
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= ${def.capacity}u) { return; }
    let me = agents[i];
    if ((me.ids.y & 1u) == 0u) {
        deltas[i] = vec4<f32>(0.0);
        return;
    }
    let pos = me.posRadius.xyz;
    let pi = pos.xz;
    let ri = me.posRadius.w;
    let cell = pointCell(pos);
    let cx = i32(cell % ${def.grid.cellsX}u);
    let cz = i32(cell / ${def.grid.cellsX}u);
    var push = vec2<f32>(0.0);
    var nContacts = 0.0;
    for (var z = max(0, cz - 1); z <= min(${def.grid.cellsZ - 1}, cz + 1); z++) {
        for (var x = max(0, cx - 1); x <= min(${def.grid.cellsX - 1}, cx + 1); x++) {
            let c = u32(z) * ${def.grid.cellsX}u + u32(x);
            let count = counts[c];
            let first = offsets[c];
            let stop = first + count;
            var at = first;
            // Contacts must visit every body, not the sensing query's bounded window.
            for (var left = count; left > 0u; left--) {
                let j = bodies[at];
                at++;
                if (at == stop) { at = first; }
                if (j == i) { continue; }
                let other = agents[j];
                if ((other.ids.y & 1u) == 0u) { continue; }
                let cp = pairPush(i, pi, ri, j, other.posRadius.xz, other.posRadius.w);
                push += cp.xy;
                nContacts += cp.z;
            }
        }
    }
    _ = tick;
    deltas[i] = vec4<f32>(push, nContacts, 0.0);
}`;
}

function buildApplyWGSL(def, { lay = null, damp = 1.0, maxPush = MAX_PUSH } = {}) {
    const boundsWGSL = lay ? `
    let lo0 = vec2<f32>(${f32(lay.origin[0])} + f32(a.ids.w % ${lay.cols}u) * ${f32(lay.pitch)},
                        ${f32(lay.origin[1])} + f32(a.ids.w / ${lay.cols}u) * ${f32(lay.pitch)});
    let hi0 = lo0 + vec2<f32>(${f32(lay.L)}, ${f32(lay.L)});` : `
    let lo0 = vec2<f32>(${f32(def.world.minX)}, ${f32(def.world.minZ)});
    let hi0 = vec2<f32>(${f32(def.world.maxX)}, ${f32(def.world.maxZ)});`;
    return `
@group(1) @binding(0) var<storage, read_write> deltas: array<vec4<f32>>;
fn step(i: u32, tick: u32) {
    let d = deltas[i];
    deltas[i] = vec4<f32>(0.0);
    if (d.z <= 0.0) { return; }
    var a = agents[i];
    var dp = d.xy / d.z;
    let m = length(dp);
    if (m > ${f32(maxPush)}) { dp *= ${f32(maxPush)} / m; }
    let r = a.posRadius.w;${boundsWGSL}
    let mid = (lo0 + hi0) * 0.5;
    let lo = min(lo0 + r, mid);
    let hi = max(hi0 - r, mid);
    let p = clamp(a.posRadius.xz + dp, lo, hi);
    a.posRadius.x = p.x;
    a.posRadius.z = p.y;
    if (${f32(damp)} > 0.0 && m > 1e-6) {
        let n = normalize(dp);
        var v = a.velHeading.xz;
        let vn = dot(v, n);
        if (vn < 0.0) {
            v -= (${f32(damp)} * vn) * n;
            a.velHeading.x = v.x;
            a.velHeading.z = v.y;
        }
    }
    agents[i] = a;
}`;
}

// ---------------------------------------------------------------------------------------------
// Arena layout helpers (matching S7's 64-arena geometry)
// ---------------------------------------------------------------------------------------------
function arenaLayout(arenas, wolves = WOLVES, prey = PREY, gap = GAP) {
    const perArena = wolves + prey, L = 2 * 80 * Math.sqrt(perArena / HERD_DEFAULTS.count);
    const cols = Math.ceil(Math.sqrt(arenas)), rows = Math.ceil(arenas / cols), pitch = L + gap;
    const spanX = cols * pitch - gap, spanZ = rows * pitch - gap;
    return { arenas, wolves, prey, perArena, L, gap, cols, pitch, origin: [-spanX / 2, -spanZ / 2], spanX, spanZ,
        count: arenas * perArena, wolfTotal: arenas * wolves };
}

const arenaConstsWGSL = lay => `
const ARENA_ORIGIN_X: f32 = ${f32(lay.origin[0])};
const ARENA_ORIGIN_Z: f32 = ${f32(lay.origin[1])};
const ARENA_PITCH: f32 = ${f32(lay.pitch)};
const ARENA_L: f32 = ${f32(lay.L)};
const ARENA_COLS = ${lay.cols}u;
fn arenaLo(arena: u32) -> vec2<f32> {
    return vec2<f32>(ARENA_ORIGIN_X + f32(arena % ARENA_COLS) * ARENA_PITCH, ARENA_ORIGIN_Z + f32(arena / ARENA_COLS) * ARENA_PITCH);
}`;

const wallSensor = lay => A.Sensors.custom({ name: 'bounds', width: 2, wgsl: `${arenaConstsWGSL(lay)}
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

const arenaSteering = lay => A.Actions.custom({ name: 'steer', width: 2, wgsl: `${arenaConstsWGSL(lay)}
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

const arenaRespawnWGSL = lay => `${arenaConstsWGSL(lay)}
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
        actions: arenaSteering(lay),
    });
}

// ---------------------------------------------------------------------------------------------
// Simulation builder for any mode ('none' | 'today-lists' | 'query-k8' | 'grid-3x3-1' | 'grid-3x3-2')
// ---------------------------------------------------------------------------------------------
function attachContacts(engine, sys, def, mode, { lay = null, damp = 1.0, compliance = 0.0 } = {}) {
    const device = engine.device, owned = [], stages = [];
    const groups = Math.ceil(def.capacity / 64);
    const alloc = (label, size) => {
        const b = device.createBuffer({ label, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        owned.push(b);
        return b;
    };
    const makeStage = (name, shaderCode, buffers) => {
        const s = engine.createComputeStage({ name, shaderCode, dispatchCount: [groups, 1, 1] });
        engine.setComputePassBindGroup(s, buffers.map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
        stages.push(s);
        return s;
    };
    let queryStage = null, gridContactStages = null, deltas = null;
    if (mode !== 'none') {
        deltas = alloc('S8 deltas', def.capacity * 16);
        if (mode === 'today-lists') {
            sys.addStepStage({ name: 'contact_acc', wgsl: buildTodayAccWGSL(compliance), buffers: [deltas] });
            sys.addStepStage({ name: 'contact_app', wgsl: buildApplyWGSL(def, { lay, damp }), buffers: [deltas] });
        } else if (mode === 'query-k8') {
            const crowd = alloc('S8 crowd k8', def.capacity * 8 * 4);
            queryStage = makeStage('S8 query crowd',
                A.buildAgentGridQueryWGSL(def, (1 << def.species.length) - 1, 1.6, 8, { stride: 8, base: 0 }),
                [sys.buffer, sys.grid.cellCounts, sys.grid.cellOffsets, sys.grid.cellBodies, crowd, sys.state]);
            sys.addStepStage({ name: 'contact_acc', wgsl: buildQueryK8AccWGSL(8, compliance), buffers: [crowd, deltas] });
            sys.addStepStage({ name: 'contact_app', wgsl: buildApplyWGSL(def, { lay, damp }), buffers: [deltas] });
        } else if (mode === 'grid-3x3-1' || mode === 'grid-3x3-2') {
            const iters = mode === 'grid-3x3-2' ? 2 : 1;
            const acc = makeStage('S8 grid3x3 acc', buildGrid3x3AccWGSL(def, compliance),
                [sys.buffer, sys.grid.cellCounts, sys.grid.cellOffsets, sys.grid.cellBodies, deltas, sys.state]);
            sys.addStepStage({ name: 'contact_app', wgsl: buildApplyWGSL(def, { lay, damp }), buffers: [deltas] });
            gridContactStages = { acc, iters };
        } else {
            throw new Error(`S8: unknown mode "${mode}"`);
        }
    }
    return {
        deltas,
        buildChain() {
            if (sys.advance(def.dt) !== 1) throw new Error('S8: advance(dt) did not select 1 step');
            const raw = [...sys.getComputeStages()];
            if (queryStage) {
                const senseIdx = raw.findIndex(s => s.name === 'AgentSystem sense');
                raw.splice(senseIdx, 0, queryStage);
            }
            if (gridContactStages) {
                const appIdx = raw.findIndex(s => s.name === 'AgentSystem step contact_app');
                const appStage = raw[appIdx];
                const pair = [];
                // act and the preceding Jacobi iteration can cross cell boundaries.
                // Rebin current positions before each exhaustive contact scan.
                for (let it = 0; it < gridContactStages.iters; it++) pair.push(...sys.grid.getComputeStages(), gridContactStages.acc, appStage);
                raw.splice(appIdx, 1, ...pair);
            }
            return raw;
        },
        destroy() {
            for (const s of stages) s.enabled = false;
            for (const b of owned) b.destroy();
        },
    };
}

function buildWorldSim(engine, count, mode, clones, { thinkEvery = HERD_DEFAULTS.thinkEvery, damp = 1.0, useRule = false } = {}) {
    const o = { ...HERD_DEFAULTS, count, thinkEvery };
    const def = herdDefinition(o), device = engine.device, wolves = Math.floor(count / 16), owned = [];
    const alloc = (size, usage, data) => {
        const b = device.createBuffer({ size, usage });
        owned.push(b);
        if (data) device.queue.writeBuffer(b, 0, data);
        return b;
    };
    const settingsData = new Float32Array(4);
    settingsData[SETTINGS.FLOCKING] = 1; settingsData[SETTINGS.FLEEING] = 1; settingsData[SETTINGS.REST_SECONDS] = o.restSeconds;
    const settings = alloc(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, settingsData);
    const stats = alloc(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const ring = alloc(1024 * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const sys = new A.AgentSystem(engine, def, { thinkEvery });
    const brains = [];
    if (useRule) {
        sys.setBrain({ kind: 'stage', wgsl: herdRule(DRIVER), buffers: [settings], destroy() {} });
    } else {
        const preyB = new A.MlpBrain(engine, clones.prey), wolfB = new A.MlpBrain(engine, clones.wolf);
        sys.setBrain(preyB, { species: 'prey' }); sys.setBrain(wolfB, { species: 'wolf' });
        brains.push(preyB, wolfB);
    }
    const contacts = attachContacts(engine, sys, def, mode, { damp });
    sys.addStepStage({ name: 'catches', wgsl: HERD_CATCHES });
    sys.addStepStage({ name: 'respawn', wgsl: herdRespawnWGSL(Math.max(1, count / HERD_DEFAULTS.count)), buffers: [settings, stats, ring] });
    const spawn = (seed = o.seed) => {
        const inset = def.world.maxX - 1;
        for (let i = 0; i < count; i++) {
            sys.spawn(i, {
                position: [(A.rand01(i, seed, 0) * 2 - 1) * inset, 0.08, (A.rand01(i, seed, 1) * 2 - 1) * inset],
                species: i < wolves ? 'wolf' : 'prey',
                heading: A.rand01(i, seed, 2) * Math.PI * 2 - Math.PI,
            });
        }
        device.queue.writeBuffer(stats, 0, new Uint32Array(4));
        sys.upload();
    };
    spawn(o.seed);
    return {
        sys, def, stats, wolves, contacts, chain: null, repeat: 1,
        spawn,
        destroy() {
            contacts.destroy();
            sys.destroy();
            for (const b of brains) b.destroy();
            for (const b of owned) b.destroy();
        },
    };
}

function buildArenaSim(engine, arenasCount, mode, clones, { thinkEvery = HERD_DEFAULTS.thinkEvery, damp = 1.0 } = {}) {
    const lay = arenaLayout(arenasCount), def = arenaDefinition(lay), device = engine.device, owned = [];
    const alloc = (size, usage, data) => {
        const b = device.createBuffer({ size, usage });
        owned.push(b);
        if (data) device.queue.writeBuffer(b, 0, data);
        return b;
    };
    const settingsData = new Float32Array(4);
    settingsData[SETTINGS.FLOCKING] = 1; settingsData[SETTINGS.FLEEING] = 1; settingsData[SETTINGS.REST_SECONDS] = HERD_DEFAULTS.restSeconds;
    const settings = alloc(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, settingsData);
    const stats = alloc(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const arenaCatches = alloc(Math.max(16, lay.arenas * 4), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const sys = new A.AgentSystem(engine, def, { thinkEvery });
    const preyB = A.MlpBrain.byGenome(engine, [clones.prey]), wolfB = A.MlpBrain.byGenome(engine, [clones.wolf]);
    sys.setBrain(preyB, { species: 'prey' }); sys.setBrain(wolfB, { species: 'wolf' });
    const contacts = attachContacts(engine, sys, def, mode, { lay, damp });
    sys.addStepStage({ name: 'catches', wgsl: HERD_CATCHES });
    sys.addStepStage({ name: 'respawn', wgsl: arenaRespawnWGSL(lay), buffers: [settings, stats, arenaCatches] });
    const spawn = (seedOf = a => mix(900001, a)) => {
        for (let a = 0; a < lay.arenas; a++) {
            const lo = [lay.origin[0] + (a % lay.cols) * lay.pitch, lay.origin[1] + Math.floor(a / lay.cols) * lay.pitch];
            const cx = lo[0] + lay.L / 2, cz = lo[1] + lay.L / 2, inset = lay.L / 2 - 1, s = seedOf(a);
            for (let slot = 0; slot < lay.perArena; slot++) {
                const wolf = slot < lay.wolves;
                const i = wolf ? a * lay.wolves + slot : lay.wolfTotal + a * lay.prey + (slot - lay.wolves);
                sys.spawn(i, {
                    position: [cx + (A.rand01(slot, s, 0) * 2 - 1) * inset, 0.08, cz + (A.rand01(slot, s, 1) * 2 - 1) * inset],
                    species: wolf ? 'wolf' : 'prey',
                    heading: A.rand01(slot, s, 2) * Math.PI * 2 - Math.PI,
                    genome: 0, user1: a,
                });
            }
        }
        device.queue.writeBuffer(stats, 0, new Uint32Array(4));
        device.queue.writeBuffer(arenaCatches, 0, new Uint32Array(Math.max(4, lay.arenas)));
        sys.upload();
    };
    spawn();
    return {
        lay, sys, def, stats, arenaCatches, contacts, chain: null, repeat: 1,
        spawn,
        destroy() {
            contacts.destroy();
            sys.destroy();
            preyB.destroy(); wolfB.destroy();
            for (const b of owned) b.destroy();
        },
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

async function runSteps(engine, sim, steps) {
    if (!sim.chain) {
        sim.chain = sim.contacts.buildChain();
    }
    if (sim.repeat === 1 && steps >= 2) {
        engine.render(sim.chain);
        await engine.device.queue.onSubmittedWorkDone();
        const t1 = performance.now();
        engine.render(sim.chain);
        await engine.device.queue.onSubmittedWorkDone();
        const probe = performance.now() - t1;
        if (probe > STEP_STOP_MS) throw new Error(`S8: one step took ${probe.toFixed(0)} ms`);
        sim.repeat = Math.max(1, Math.min(32, Math.round(SUBMIT_MS / Math.max(probe, 0.05))));
        steps -= 2;
    }
    const t0 = performance.now();
    for (let done = 0; done < steps; ) {
        const n = Math.min(sim.repeat, steps - done);
        engine.render(Array.from({ length: n }, () => sim.chain).flat());
        await engine.device.queue.onSubmittedWorkDone();
        done += n;
    }
    return performance.now() - t0;
}

// ---------------------------------------------------------------------------------------------
// Overlap & nearest-prey metrics on read-back agent records
// ---------------------------------------------------------------------------------------------
function measureOverlap(recU32, { lay = null, wolves = 0 } = {}) {
    const recF = new Float32Array(recU32.buffer), count = recU32.length / 16;
    let worstM = 0, worstPct = 0, over5Count = 0, nnSum = 0, preyN = 0;
    const checkRange = (indices, preyStartIdx) => {
        const prey = new Set(indices.slice(preyStartIdx));
        indices.sort((i, j) => recF[i * 16] - recF[j * 16] || i - j);
        const n = indices.length, hasOver5 = new Uint8Array(n);
        let maxRadius = 0;
        for (const i of indices) maxRadius = Math.max(maxRadius, recF[i * 16 + 3]);
        for (let a = 0; a < n; a++) {
            const i = indices[a], xi = recF[i * 16], zi = recF[i * 16 + 2], ri = recF[i * 16 + 3];
            let bestPreyD2 = Infinity;
            const iIsPrey = prey.has(i);
            for (let b = a + 1; b < n; b++) {
                const j = indices[b], dx = xi - recF[j * 16], dz = zi - recF[j * 16 + 2];
                if (-dx > ri + maxRadius) break;
                const d2 = dx * dx + dz * dz, rij = ri + recF[j * 16 + 3];
                if (d2 < rij * rij) {
                    const pen = rij - Math.sqrt(d2), pct = pen / rij;
                    if (pen > worstM) worstM = pen;
                    if (pct > worstPct) worstPct = pct;
                    if (pct > 0.05) { hasOver5[a] = 1; hasOver5[b] = 1; }
                }
            }
            if (iIsPrey) {
                for (const direction of [-1, 1]) {
                    for (let b = a + direction; b >= 0 && b < n; b += direction) {
                        const j = indices[b], dx = xi - recF[j * 16];
                        if (dx * dx > bestPreyD2) break;
                        if (!prey.has(j)) continue;
                        const dz = zi - recF[j * 16 + 2];
                        bestPreyD2 = Math.min(bestPreyD2, dx * dx + dz * dz);
                    }
                }
                if (bestPreyD2 < Infinity) { nnSum += Math.sqrt(bestPreyD2); preyN++; }
            }
        }
        for (let a = 0; a < n; a++) over5Count += hasOver5[a];
    };
    if (lay) {
        for (let a = 0; a < lay.arenas; a++) {
            const indices = [];
            for (let w = 0; w < lay.wolves; w++) indices.push(a * lay.wolves + w);
            for (let p = 0; p < lay.prey; p++) indices.push(lay.wolfTotal + a * lay.prey + p);
            checkRange(indices, lay.wolves);
        }
    } else {
        const indices = Array.from({ length: count }, (_, i) => i);
        checkRange(indices, wolves);
    }
    return {
        worstM,
        worstPct: 100 * worstPct,
        over5Pct: 100 * (over5Count / count),
        nearestPreyM: preyN > 0 ? nnSum / preyN : 0,
    };
}

async function overlapHistory(engine, sim, options) {
    const history = [];
    for (let step = 1; step <= 600; step++) {
        await runSteps(engine, sim, 1);
        const rec = await readU32(engine.device, sim.sys.buffer, sim.def.capacity * 16);
        // Count final positions, not the grid from before the last apply/respawn stage.
        const f = new Float32Array(rec.buffer), counts = new Uint32Array(sim.def.grid.cellTotal);
        let peakCell = 0;
        for (let i = 0; i < sim.def.capacity; i++) {
            if (!(rec[i * 16 + 13] & 1)) continue;
            const x = Math.max(0, Math.min(sim.def.grid.cellsX - 1, Math.floor((f[i * 16] - sim.def.world.minX) / sim.def.cellSize)));
            const z = Math.max(0, Math.min(sim.def.grid.cellsZ - 1, Math.floor((f[i * 16 + 2] - sim.def.world.minZ) / sim.def.cellSize)));
            peakCell = Math.max(peakCell, ++counts[z * sim.def.grid.cellsX + x]);
        }
        history.push({ step, ...measureOverlap(rec, options), peakCell });
    }
    return { ...history.at(-1), history,
        worstOverRunM: Math.max(...history.map(r => r.worstM)),
        worstOverRunPct: Math.max(...history.map(r => r.worstPct)),
        maxOver5Pct: Math.max(...history.map(r => r.over5Pct)),
        peakCellOverRun: Math.max(...history.map(r => r.peakCell)) };
}

// ---------------------------------------------------------------------------------------------
// Pile-up stress test & jitter harness (256 prey in a single arena/world)
// ---------------------------------------------------------------------------------------------
function buildPileupSim(engine, mode, { attractor = false, damp = 1.0 } = {}) {
    const count = 256, def = A.defineAgents({
        capacity: count,
        world: { min: [-24, -24], max: [24, 24] },
        cellSize: 8,
        maxScanPerCell: 64,
        species: [{ name: 'prey', maxSpeed: 3.2, maxTurn: 4, accel: 6, senseRadius: 8, radius: 0.4 }],
        sensors: [
            A.Sensors.self(),
            A.Sensors.nearest({ name: 'wolf', species: ['prey'], k: 1 }),
            A.Sensors.nearest({ name: 'prey', species: ['prey'], k: 3 }),
            A.Sensors.bounds(),
        ],
    });
    const sys = new A.AgentSystem(engine, def, { thinkEvery: 1 });
    // Either stationary (throttle = -1 -> targetSpeed = 0) so only contacts separate the pile-up,
    // or active inward attractor steering every agent toward (0, 0) at full speed (stacking stress).
    const wgsl = attractor ? `
fn think(i: u32, species: u32, tick: u32) {
    let a = agents[i];
    let toOrigin = -a.posRadius.xz;
    let sn = sin(a.velHeading.w);
    let cs = cos(a.velHeading.w);
    let side = dot(toOrigin, vec2<f32>(cs, -sn));
    let fwd = dot(toOrigin, vec2<f32>(sn, cs));
    setAct(i, 0u, clamp(2.5 * atan2(side, fwd), -1.0, 1.0));
    setAct(i, 1u, 1.0);
}` : `
fn think(i: u32, species: u32, tick: u32) {
    setAct(i, 0u, 0.0);
    setAct(i, 1u, -1.0);
}`;
    sys.setBrain({ kind: 'stage', wgsl, buffers: [], destroy() {} });
    // Note: in `today-lists` for Herd & Wolves, prey only have `k = 3` prey neighbours (the 1 wolf slot
    // is empty when no wolf is within 8 m). For a true comparison of Herd & Wolves' 3-prey list vs k=8 vs grid-3x3:
    let contacts;
    if (mode === 'today-lists') {
        const deltas = engine.device.createBuffer({ size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        sys.addStepStage({ name: 'contact_acc', wgsl: `
@group(1) @binding(0) var<storage, read_write> deltas: array<vec4<f32>>;
${pairPushWGSL(0.0)}
fn step(i: u32, tick: u32) {
    let me = agents[i];
    var push = vec2<f32>(0.0);
    var cnt = 0.0;
    for (var s = 0u; s < 3u; s++) {
        let j = neighbours[i * NEIGHBOURS_PER_AGENT + NEIGHBOURS_PREY + s];
        if (j == NO_AGENT || j == i) { continue; }
        let other = agents[j];
        let c = pairPush(i, me.posRadius.xz, me.posRadius.w, j, other.posRadius.xz, other.posRadius.w);
        push += c.xy; cnt += c.z;
    }
    deltas[i] = vec4<f32>(push, cnt, 0.0);
}`, buffers: [deltas] });
        sys.addStepStage({ name: 'contact_app', wgsl: buildApplyWGSL(def, { damp }), buffers: [deltas] });
        contacts = {
            buildChain() {
                if (sys.advance(def.dt) !== 1) throw new Error('S8: advance(dt) !== 1');
                return [...sys.getComputeStages()];
            },
            destroy() { deltas.destroy(); },
        };
    } else {
        contacts = attachContacts(engine, sys, def, mode, { damp });
    }
    // Spawn all 256 prey inside [-2, 2] x [-2, 2] (a 4 m x 4 m box).
    for (let i = 0; i < count; i++) {
        sys.spawn(i, {
            position: [(A.rand01(i, 77, 0) * 2 - 1) * 2.0, 0.08, (A.rand01(i, 77, 1) * 2 - 1) * 2.0],
            species: 'prey',
            heading: A.rand01(i, 77, 2) * Math.PI * 2 - Math.PI,
        });
    }
    sys.upload();
    const chain = contacts.buildChain();
    return {
        sys, def, chain, contacts,
        destroy() { contacts.destroy(); sys.destroy(); },
    };
}

// ---------------------------------------------------------------------------------------------
async function loadJSON(url) {
    return inBrowser ? (await fetch(url)).json() : JSON.parse((await import('node:fs')).readFileSync(url, 'utf8'));
}
const MODELS = new URL('../../../tools/agents/models/herd-wolves/', import.meta.url);

async function runAdapter(device, name) {
    const engine = new NullGraph();
    const init = await engine.init(null, { device, width: 64, height: 64 });
    if (!init.success) throw new Error(`S8: engine: ${init.error}`);
    const clones = {};
    for (const s of ['prey', 'wolf']) clones[s] = networkFromModel(await loadJSON(new URL(`${s}.mlp.json`, MODELS)), herdDefinition(), `${s}.mlp.json`);
    const result = { adapter: name, host: inBrowser ? 'chrome' : 'node' };

    if (want('checks')) {
        device.pushErrorScope('validation');
        result.reruns = {};
        const dense = buildPileupSim(engine, 'grid-3x3-1');
        const accAt = dense.chain.findIndex(s => s.name === 'S8 grid3x3 acc');
        if (accAt < 0) throw new Error('S8: missing exhaustive grid stage');
        engine.render(dense.chain.slice(0, accAt + 1));
        await device.queue.onSubmittedWorkDone();
        const denseRecords = new Float32Array((await readU32(device, dense.sys.buffer, 256 * 16)).buffer);
        const deltas = dense.contacts.deltas;
        const actual = new Float32Array((await readU32(device, deltas, 256 * 4)).buffer);
        let denseMismatches = 0, maxContacts = 0;
        for (let i = 0; i < 256; i++) {
            let contacts = 0;
            for (let j = 0; j < 256; j++) {
                if (i === j) continue;
                const dx = denseRecords[i * 16] - denseRecords[j * 16];
                const dz = denseRecords[i * 16 + 2] - denseRecords[j * 16 + 2];
                const r = denseRecords[i * 16 + 3] + denseRecords[j * 16 + 3];
                if (dx * dx + dz * dz < r * r) contacts++;
            }
            if (actual[i * 4 + 2] !== contacts) denseMismatches++;
            maxContacts = Math.max(maxContacts, contacts);
        }
        check(`${name}: dense exhaustive grid contact counts match brute force`, denseMismatches === 0 && maxContacts > 8,
            `mismatches=${denseMismatches}, maxContacts=${maxContacts}`);
        dense.destroy();
        for (const mode of CONTACT_MODES) {
            const simA = buildWorldSim(engine, 4096, mode, clones);
            const simB = buildWorldSim(engine, 4096, mode, clones);
            await runSteps(engine, simA, 300);
            await runSteps(engine, simB, 300);
            const recA = await readU32(device, simA.sys.buffer, 4096 * 16);
            const recB = await readU32(device, simB.sys.buffer, 4096 * 16);
            let diffWords = 0, badFloats = 0;
            const recAF = new Float32Array(recA.buffer);
            for (let w = 0; w < recA.length; w++) if (recA[w] !== recB[w]) diffWords++;
            for (let i = 0; i < 4096; i++) {
                for (const off of [0, 1, 2, 3, 4, 5, 6, 7]) {
                    if (!Number.isFinite(recAF[i * 16 + off])) badFloats++;
                }
            }
            result.reruns[mode] = { diffWords, badFloats };
            check(`${name}: rerun (${mode}): two 300-step runs from seed ${HERD_DEFAULTS.seed} match word for word`,
                diffWords === 0 && badFloats === 0, `diffWords=${diffWords}, badFloats=${badFloats}`);
            simA.destroy(); simB.destroy();
        }
        const error = await device.popErrorScope();
        check(`${name}: checks ran without a WebGPU validation error`, error === null, error?.message ?? '');
    }

    if (want('cost')) {
        result.cost = [];
        for (const spec of [
            { label: '4,096', kind: 'world', count: 4096, steps: 120 },
            { label: '16,384', kind: 'world', count: 16384, steps: 120 },
            { label: '65,536', kind: 'world', count: 65536, steps: 60 },
            { label: '17,408 (64 arenas)', kind: 'arenas', arenas: 64, count: 17408, steps: 120 },
        ]) {
            let baseMs = 0;
            for (const mode of MODES) {
                const sim = spec.kind === 'world'
                    ? buildWorldSim(engine, spec.count, mode, clones)
                    : buildArenaSim(engine, spec.arenas, mode, clones);
                await runSteps(engine, sim, Math.min(20, spec.steps)); // warm up + probe
                const elapsed = await runSteps(engine, sim, spec.steps);
                const msPerStep = elapsed / spec.steps;
                if (mode === 'none') baseMs = msPerStep;
                const addedPct = baseMs > 0 ? 100 * (msPerStep - baseMs) / baseMs : 0;
                result.cost.push({ layout: spec.label, count: spec.count, mode, msPerStep, addedPct });
                sim.destroy();
            }
        }
        log(`\n== S8 cost · ${result.host} · ${name} — ms per step (and % vs none), Herd & Wolves clones, thinkEvery ${HERD_DEFAULTS.thinkEvery}`);
        log(`   ${'layout'.padEnd(20)}${MODES.map(m => m.padStart(18)).join('')}`);
        for (const label of ['4,096', '16,384', '65,536', '17,408 (64 arenas)']) {
            const rows = result.cost.filter(r => r.layout === label);
            log(`   ${label.padEnd(20)}${rows.map(r => (r.mode === 'none'
                ? `${r.msPerStep.toFixed(3)} ms`
                : `${r.msPerStep.toFixed(3)} (${r.addedPct >= 0 ? '+' : ''}${r.addedPct.toFixed(1)}%)`).padStart(18)).join('')}`);
        }
    }

    if (want('overlap')) {
        result.overlap = { world4096: {}, arenas64: {} };
        for (const mode of MODES) {
            const sim = buildWorldSim(engine, 4096, mode, clones);
            const metrics = await overlapHistory(engine, sim, { wolves: sim.wolves });
            const catches = (await readU32(device, sim.stats, 4))[0];
            result.overlap.world4096[mode] = { ...metrics, catches };
            log(`   sampled all 600 steps: world4096 ${mode}`);
            sim.destroy();
        }
        for (const mode of MODES) {
            const sim = buildArenaSim(engine, 64, mode, clones);
            const metrics = await overlapHistory(engine, sim, { lay: sim.lay });
            const catches = (await readU32(device, sim.stats, 4))[0];
            result.overlap.arenas64[mode] = { ...metrics, catchesPerArena: catches / 64 };
            log(`   sampled all 600 steps: arenas64 ${mode}`);
            sim.destroy();
        }
        log(`\n== S8 overlap · ${result.host} · ${name} — every step 1..600 on C2 clones (table: final state)`);
        log('   single world (4,096 agents): mode · worst overlap (m / %) · agents >5% overlap · nearest prey (m) · peak cell · catches');
        for (const [m, r] of Object.entries(result.overlap.world4096)) {
            log(`   ${m.padEnd(14)} ${r.worstM.toFixed(3)} m (${r.worstPct.toFixed(1)}%)`.padEnd(34) +
                `${`${r.over5Pct.toFixed(2)}%`.padStart(10)}${`${r.nearestPreyM.toFixed(2)} m`.padStart(12)}${String(r.peakCell).padStart(10)}${String(r.catches).padStart(10)}`);
        }
        for (const [layout, modes] of Object.entries(result.overlap)) {
            for (const [mode, r] of Object.entries(modes)) {
                check(`${name}: ${layout} ${mode}: all 600 overlap samples are finite`,
                    r.history.length === 600 && r.history.every(s => Object.values(s).every(Number.isFinite)));
                log(`   over-run ${layout} ${mode}: worst ${r.worstOverRunM.toFixed(3)} m (${r.worstOverRunPct.toFixed(1)}%), max >5% ${r.maxOver5Pct.toFixed(2)}%, peak cell ${r.peakCellOverRun}`);
            }
        }
        log('   64 arenas (17,408 agents): mode · worst overlap (m / %) · agents >5% overlap · nearest prey (m) · peak cell · catches/arena');
        for (const [m, r] of Object.entries(result.overlap.arenas64)) {
            log(`   ${m.padEnd(14)} ${r.worstM.toFixed(3)} m (${r.worstPct.toFixed(1)}%)`.padEnd(34) +
                `${`${r.over5Pct.toFixed(2)}%`.padStart(10)}${`${r.nearestPreyM.toFixed(2)} m`.padStart(12)}${String(r.peakCell).padStart(10)}${r.catchesPerArena.toFixed(1).padStart(10)}`);
        }
    }

    if (want('pileup')) {
        result.pileup = { staticBox: {}, activeAttractor: {} };
        // 1. Static 256-prey 4 m x 4 m box: how many steps until max overlap < 5% (and final state at 180 steps).
        for (const mode of MODES) {
            const sim = buildPileupSim(engine, mode, { attractor: false, damp: 1.0 });
            let stepsToSub5Pct = null, finalMetrics = null;
            for (let step = 1; step <= 180; step++) {
                engine.render(sim.chain);
                if (step % 3 === 0 || step === 180) {
                    await engine.device.queue.onSubmittedWorkDone();
                    const rec = await readU32(device, sim.sys.buffer, 256 * 16);
                    const m = measureOverlap(rec, { wolves: 0 });
                    if (stepsToSub5Pct === null && m.over5Pct === 0) stepsToSub5Pct = step;
                    if (step === 180) finalMetrics = m;
                }
            }
            result.pileup.staticBox[mode] = { stepsToSub5Pct, ...finalMetrics };
            sim.destroy();
        }
        // 2. Active central attractor: all 256 prey steer into (0, 0) at 3.2 m/s for 300 steps.
        for (const mode of MODES) {
            const sim = buildPileupSim(engine, mode, { attractor: true, damp: 1.0 });
            for (let s = 0; s < 300; s += 30) {
                engine.render(Array.from({ length: 30 }, () => sim.chain).flat());
                await engine.device.queue.onSubmittedWorkDone();
            }
            const rec = await readU32(device, sim.sys.buffer, 256 * 16);
            result.pileup.activeAttractor[mode] = measureOverlap(rec, { wolves: 0 });
            sim.destroy();
        }
        log(`\n== S8 pileup · ${result.host} · ${name} — 256 prey (r = 0.4 m) in a 4 m × 4 m box`);
        log('   static relaxation (180 steps): mode · steps to 0% agents >5% overlap · final worst overlap · final nearest prey');
        for (const [m, r] of Object.entries(result.pileup.staticBox)) {
            log(`   ${m.padEnd(14)} ${String(r.stepsToSub5Pct ?? '>180').padStart(8)} steps   worst ${r.worstM.toFixed(3)} m (${r.worstPct.toFixed(1)}%), >5%: ${r.over5Pct.toFixed(2)}%, nearest ${r.nearestPreyM.toFixed(2)} m`);
        }
        log('   active central attractor (all 256 prey driving into (0,0) for 300 steps):');
        for (const [m, r] of Object.entries(result.pileup.activeAttractor)) {
            log(`   ${m.padEnd(14)} worst ${r.worstM.toFixed(3)} m (${r.worstPct.toFixed(1)}%), >5%: ${r.over5Pct.toFixed(2)}%, nearest ${r.nearestPreyM.toFixed(2)} m`);
        }
    }

    if (want('jitter')) {
        result.jitter = {};
        // Measure consecutive-step displacement reversals (cos theta < -0.5) in the active 256-prey crowd
        // over steps 100..160, comparing damp = 0.0 vs damp = 1.0.
        for (const mode of MODES) {
            result.jitter[mode] = {};
            for (const damp of [0.0, 1.0]) {
                const sim = buildPileupSim(engine, mode, { attractor: true, damp });
                engine.render(Array.from({ length: 100 }, () => sim.chain).flat());
                await engine.device.queue.onSubmittedWorkDone();
                let prev = new Float32Array((await readU32(device, sim.sys.buffer, 256 * 16)).buffer);
                let prevDx = new Float32Array(256), prevDz = new Float32Array(256), hasPrev = false;
                let reversals = 0, total = 0;
                for (let step = 0; step < 60; step++) {
                    engine.render(sim.chain);
                    await engine.device.queue.onSubmittedWorkDone();
                    const cur = new Float32Array((await readU32(device, sim.sys.buffer, 256 * 16)).buffer);
                    for (let i = 0; i < 256; i++) {
                        const dx = cur[i * 16] - prev[i * 16], dz = cur[i * 16 + 2] - prev[i * 16 + 2];
                        const len = Math.hypot(dx, dz);
                        if (hasPrev) {
                            const plen = Math.hypot(prevDx[i], prevDz[i]);
                            if (len > 1e-4 && plen > 1e-4) {
                                total++;
                                const cos = (dx * prevDx[i] + dz * prevDz[i]) / (len * plen);
                                if (cos < -0.5) reversals++;
                            }
                        }
                        prevDx[i] = dx; prevDz[i] = dz;
                    }
                    prev = cur;
                    hasPrev = true;
                }
                result.jitter[mode][damp === 1.0 ? 'damped' : 'undamped'] = total > 0 ? 100 * (reversals / total) : 0;
                sim.destroy();
            }
        }
        log(`\n== S8 jitter · ${result.host} · ${name} — % of step displacements reversing (cos θ < -0.5) in dense crowd`);
        for (const [m, r] of Object.entries(result.jitter)) {
            log(`   ${m.padEnd(14)} undamped: ${r.undamped.toFixed(2)}%   damped: ${r.damped.toFixed(2)}%`);
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
    log(`S8: measuring on ${name}${i.description ? ` (${i.description})` : ''} (${inBrowser ? 'Chrome' : 'Node/Dawn'}), part ${PART}…`);
    const device = await adapter.requestDevice();
    results.push(await runAdapter(device, name));
    device.destroy();
}
if (!results.length) { fatal('No WebGPU adapter'); throw new Error('No WebGPU adapter'); }
log(`\nS8 JSON ${JSON.stringify(results)}`);
globalThis.__S8_MEASUREMENTS__ = results;
if (!inBrowser) {
    const { mkdirSync, writeFileSync } = await import('node:fs');
    const out = new URL('./out/', import.meta.url);
    mkdirSync(out, { recursive: true });
    writeFileSync(new URL(`node-${PART}-measurements.json`, out), JSON.stringify(results, null, 2));
}
await finish(`S8: checks passed on ${results.length} adapter(s).`);
log('S8 done');
