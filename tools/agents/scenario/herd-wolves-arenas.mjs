// Step 14.1: fresh lane systems; reviewed tile wall/integrator/respawn formulas from S7.
// Wall/integrator formulas transcribed from the existing agents generators; I2 guards drift.
import * as A from '../../../dist/agents.js';
import { DiscContacts } from '../../../dist/physics.js';
import { arenaLayout, exhaustiveCatchWGSL } from './arena-layout.mjs';
import { DRIVER, HERD_CATCHES, HERD_DEFAULTS, LABEL_DRIVING, SETTINGS, herdDefinition, herdRule } from './herd-wolves.mjs';
import { diagnosticLayout, diagnosticCatchWGSL, diagnosticStateWGSL } from '../coevolve-diagnostics.mjs';
const f32 = v => Number.isInteger(v) ? v.toFixed(1) : String(v);
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
export const respawnWGSL = lay => `${arenaWGSL(lay)}
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
    if ((agents[i].ids.y & 1u) == 0u) { return; }
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
export function arenaDefinition(lay, { dt = 1 / 60 } = {}) {
    if (!Number.isFinite(dt) || dt <= 0) throw new Error('coevolve: invalid dt');
    const o = HERD_DEFAULTS;
    return A.defineAgents({
        dt,
        maxFrameDelta: Math.max(0.25, dt),
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
// The library names custom fields by index. The tile's wall sensor and steering mean
// Sensors.bounds() and Actions.steering() (I2 checks it bit for bit), so models carry the demo's
// names: clones load, and checkpoints load wherever the demo's networks do. Only these four rename.
const CANONICAL_FIELDS = Object.freeze({
    'bounds.0': 'bounds.side', 'bounds.1': 'bounds.forward', 'steer.0': 'turn', 'steer.1': 'throttle' });
export function modelNames(def) {
    const demo = herdDefinition(), names = {};
    for (const key of ['obsNames', 'actNames']) {
        names[key] = Object.freeze(def[key].map(name => CANONICAL_FIELDS[name] ?? name));
        const k = Math.max(names[key].length, demo[key].length);
        for (let j = 0; j < k; j++) if (names[key][j] !== demo[key][j])
            throw new Error(`coevolve: arena ${key}[${j}] "${def[key][j]}" does not map onto the demo's "${demo[key][j]}"`);
    }
    return Object.freeze(names);
}
export const containmentWGSL = lay => `${arenaWGSL(lay)}
fn step(i: u32, tick: u32) {
    let a = agents[i];
    let lo = arenaLo(a.ids.w) + a.posRadius.w;
    let hi = arenaLo(a.ids.w) + vec2<f32>(ARENA_L) - a.posRadius.w;
    let p = clamp(a.posRadius.xz, lo, hi);
    if ((a.posRadius.x < lo.x && a.velHeading.x < 0.0) || (a.posRadius.x > hi.x && a.velHeading.x > 0.0)) { agents[i].velHeading.x = 0.0; }
    if ((a.posRadius.z < lo.y && a.velHeading.z < 0.0) || (a.posRadius.z > hi.y && a.velHeading.z > 0.0)) { agents[i].velHeading.z = 0.0; }
    agents[i].posRadius.x = p.x; agents[i].posRadius.z = p.y;
}`;

// A species `brains` names by string runs the scenario's rule instead of a policy table: the demo's
// teacher, or the rule without wander and coin flips (the label's decisions, plan-2's baseline).
export const RULE_DRIVERS = Object.freeze({ rule: DRIVER, label: LABEL_DRIVING });
// contact-v3 is contact-v2 with separated initial spawns (design §5); both catch exhaustively.
export const ENVIRONMENTS = Object.freeze(['legacy-v1', 'contact-v2', 'contact-v3']);
const EXHAUSTIVE = new Set(['contact-v2', 'contact-v3']);

/**
 * One tile's spawn, slot by slot (wolves first): the demo's draws from the episode seed, in its
 * expression order, so legacy-v1 and contact-v2 spawn bit for bit as before. With `separation`
 * (contact-v3) a prey drawn closer than that to a wolf of its tile is drawn again from the seed's
 * next keys, the same in both mirrored lanes; headings never change.
 */
export function tileSpawns(lay, lo, seed, separation = 0) {
    const inset = lay.L / 2 - 1, spawns = [];
    const at = (slot, kx, kz) => [lo[0] + lay.L / 2 + (A.rand01(slot, seed, kx) * 2 - 1) * inset,
        lo[1] + lay.L / 2 + (A.rand01(slot, seed, kz) * 2 - 1) * inset];
    for (let slot = 0; slot < lay.perArena; slot++) {
        let [x, z] = at(slot, 0, 1);
        if (separation > 0 && slot >= lay.wolves) {
            const near = () => spawns.slice(0, lay.wolves).some(w => Math.hypot(x - w.x, z - w.z) < separation);
            for (let retry = 0; near(); retry++) {
                if (retry >= 1000) throw new Error(`coevolve: no spawn for prey slot ${slot} keeps ${separation} m from its tile's wolves`);
                [x, z] = at(slot, 3 + 2 * retry, 4 + 2 * retry);
            }
        }
        spawns.push({ x, z, heading: A.rand01(slot, seed, 2) * Math.PI * 2 - Math.PI });
    }
    return spawns;
}

export function buildLane(engine, geometry, { environment, contacts, thinkEvery, seedOf, rowOf, brains, fixture,
    catchMargin, spawnSeparation, contactOptions = {}, dt = 1 / 60, monitorFinite = false, diagnostics } = {}) {
    if (!ENVIRONMENTS.includes(environment)) throw new Error(`coevolve: unknown environment ${environment}`);
    if (environment === 'contact-v3' ? !(Number.isFinite(spawnSeparation) && spawnSeparation > 0 && spawnSeparation < geometry.L / 2)
        : spawnSeparation !== undefined)
        throw new Error('coevolve: contact-v3 needs an explicit spawnSeparation in (0, L/2) m; other environments take none');
    if (brains) for (const name of ['prey', 'wolf']) if (typeof brains[name] === 'string' ? !RULE_DRIVERS[brains[name]] : !brains[name])
        throw new Error(`coevolve: ${name} driver must be a brain, "rule" or "label"`);
    const table = name => brains && typeof brains[name] !== 'string';
    if (typeof contacts !== 'boolean') throw new Error('coevolve: contacts must be explicit');
    if (environment === 'legacy-v1' && contacts) throw new Error('coevolve: legacy-v1 requires contacts off');
    if (!Number.isInteger(thinkEvery) || thinkEvery < 1) throw new Error('coevolve: invalid thinkEvery');
    if (typeof seedOf !== 'function' || typeof rowOf !== 'function') throw new Error('coevolve: seedOf and rowOf are required');
    if (EXHAUSTIVE.has(environment) && (!Number.isFinite(catchMargin) || catchMargin < 0)) throw new Error('coevolve: explicit finite nonnegative catchMargin required');
    const allowed = ['iterations', 'maxPush', 'relaxation', 'compliance', 'damping', 'mass', 'maxScanPerCell', 'cellSize'];
    for (const key of Object.keys(contactOptions)) if (!allowed.includes(key)) throw new Error(`coevolve: forbidden contact override ${key}`);
    const lay = arenaLayout({ ...geometry, contacts: contacts ? contactOptions : null });
    const diagnostic = diagnostics ? diagnosticLayout(lay, diagnostics) : null;
    const device = engine.device, owned = [], external = [], def = arenaDefinition(lay, { dt });
    const bytes = lay.count * A.AGENT_FLOATS * 4;
    const largest = Math.max(bytes, def.grid.cellTotal * 4 + 4);
    if (largest > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize) ||
        Math.ceil(lay.count / 64) > device.limits.maxComputeWorkgroupsPerDimension)
        throw new Error('coevolve: arena exceeds device buffer/dispatch limits');
    const assignments = Array.from({ length: lay.arenas }, (_, a) => {
        const seed = seedOf(a), rows = Object.fromEntries(['prey', 'wolf'].map(name => [name, rowOf(a, name)]));
        if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error(`coevolve: seed ${seed} outside u32`);
        for (const name of ['prey', 'wolf']) if (!Number.isInteger(rows[name]) || rows[name] < 0 ||
            rows[name] > 0xffffffff || (table(name) && (!Number.isInteger(brains[name].rowCount) || rows[name] >= brains[name].rowCount)))
            throw new Error(`coevolve: ${name} row ${rows[name]} outside policy table`);
        return { seed, rows };
    });
    const sys = new A.AgentSystem(engine, def, { thinkEvery });
    let solver;
    const alloc = (size, data) => {
        const b = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        owned.push(b); if (data) device.queue.writeBuffer(b, 0, data); return b;
    };
    try {
        if (diagnostic && diagnostic.words * 4 > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize))
            throw new Error('coevolve: diagnostics exceed device buffer limits');
        const diagnosticBuffers = diagnostic ? { samples: alloc(diagnostic.words * 4), catches: alloc(diagnostic.words * 4) } : null;
        const settings = alloc(16, new Float32Array([1, 1, 5, 0])), stats = alloc(16);
        const counters = alloc(Math.max(16, lay.arenas * 4));
        const rule = seams => ({ kind: 'stage', wgsl: `const ACT_TURN = 0u;\nconst ACT_THROTTLE = 1u;\n${herdRule(seams)}`, buffers: [settings] });
        const rules = {};   // one brain per rule kind, shared by the species it drives
        if (brains) for (const name of ['prey', 'wolf']) sys.setBrain(table(name) ? brains[name]
            : (rules[brains[name]] ??= rule(RULE_DRIVERS[brains[name]])), { species: name });
        else sys.setBrain(rule(DRIVER));
        if (EXHAUSTIVE.has(environment)) {
            const snapshot = alloc(bytes);
            const stage = engine.createComputeStage({ name: 'coevolve catch snapshot', dispatchCount: [Math.ceil(lay.count / 64), 1, 1],
                shaderCode: `@group(0) @binding(0) var<storage, read> src: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<u32>>;
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= ${lay.count}u) { return; }
    for (var k = 0u; k < 4u; k++) { dst[id.x * 4u + k] = src[id.x * 4u + k]; }
}` });
            external.push(stage);
            engine.setComputePassBindGroup(stage, [sys.buffer, snapshot].map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
            sys.addStepStage({ name: 'catch snapshot', stages: [stage] });
            sys.addStepStage({ name: 'catches', wgsl: exhaustiveCatchWGSL(lay, catchMargin), buffers: [snapshot] });
        } else sys.addStepStage({ name: 'catches', wgsl: HERD_CATCHES });
        if (diagnostic) sys.addStepStage({ name: 'catch locations', wgsl: diagnosticCatchWGSL(lay, diagnostic), buffers: [diagnosticBuffers.catches] });
        sys.addStepStage({ name: 'respawn', wgsl: respawnWGSL(lay), buffers: [settings, stats, counters] });
        if (contacts) {
            solver = new DiscContacts(engine, sys.buffer, { capacity: lay.count, stride: A.AGENT_FLOATS,
                dt: def.dt,
                world: { min: lay.origin, max: [lay.origin[0] + lay.spanX, lay.origin[1] + lay.spanZ] },
                maxRadius: 0.6, alive: { offset: A.AGENT_FLAGS_OFFSET, mask: A.AGENT_FLAG_ALIVE },
                fixed: { offset: A.AGENT_FLAGS_OFFSET, mask: A.AGENT_FLAG_FROZEN },
                velocity: { offset: A.AGENT_VELOCITY_OFFSET }, clamp: true, ...contactOptions });
            sys.addStepStage({ name: 'contacts', stages: solver.getComputeStages() });
        }
        if (environment !== 'legacy-v1')
            sys.addStepStage({ name: 'containment', wgsl: containmentWGSL(lay) });
        if (diagnostic) sys.addStepStage({ name: 'diagnostics', wgsl: diagnosticStateWGSL(lay, diagnostic), buffers: [diagnosticBuffers.samples] });
        // Sticky flag shares the score readback; no per-step CPU readbacks are introduced.
        if (monitorFinite) sys.addStepStage({ name: 'finite state', buffers: [stats], wgsl: `
@group(1) @binding(0) var<storage, read_write> finiteStats: array<atomic<u32>, 4>;
fn step(i: u32, tick: u32) {
    let a = agents[i];
    let values = array<vec4<f32>, 3>(a.posRadius, a.velHeading, a.state);
    for (var v = 0u; v < 3u; v++) { for (var c = 0u; c < 4u; c++) {
        if ((bitcast<u32>(values[v][c]) & 0x7f800000u) == 0x7f800000u) { atomicStore(&finiteStats[3], 1u); }
    } }
}` });
        for (let a = 0; a < lay.arenas; a++) {
            const lo = [lay.origin[0] + a % lay.cols * lay.pitch, lay.origin[1] + Math.floor(a / lay.cols) * lay.pitch];
            const spawns = tileSpawns(lay, lo, assignments[a].seed, spawnSeparation ?? 0);
            for (let slot = 0; slot < lay.perArena; slot++) {
                const wolf = slot < lay.wolves, name = wolf ? 'wolf' : 'prey';
                const i = wolf ? a * lay.wolves + slot : lay.wolfTotal + a * lay.prey + slot - lay.wolves;
                const init = { position: [spawns[slot].x, 0.08, spawns[slot].z], species: name, heading: spawns[slot].heading,
                    genome: assignments[a].rows[name], user1: a };
                fixture?.(init, { i, slot, a, lo, lay, wolf });
                sys.spawn(i, init);
            }
        }
        sys.upload();
        if (sys.advance(def.dt) !== 1) throw new Error('coevolve: invalid one-step chain');
        let destroyed = false;
        return { sys, lay, stats, counters, settings, solver, environment, diagnostic, diagnosticBuffers,
            get chain() { if (destroyed) throw new Error('coevolve: lane destroyed'); return sys.getComputeStages(); },
            destroy() { if (destroyed) return; destroyed = true; sys.destroy(); solver?.destroy(); for (const s of external) s.enabled = false; for (const b of owned) b.destroy(); } };
    } catch (e) { sys.destroy(); solver?.destroy(); for (const s of external) s.enabled = false; for (const b of owned) b.destroy(); throw e; }
}

export async function readBuffers(device, entries) {
    const size = entries.reduce((n, e) => n + e.words * 4, 0);
    const dst = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
        const enc = device.createCommandEncoder(); let offset = 0;
        for (const e of entries) { enc.copyBufferToBuffer(e.buffer, 0, dst, offset, e.words * 4); offset += e.words * 4; }
        device.queue.submit([enc.finish()]); await dst.mapAsync(GPUMapMode.READ);
        const data = new Uint32Array(dst.getMappedRange().slice(0)); let at = 0;
        return entries.map(e => { const v = data.slice(at, at + e.words); at += e.words; return v; });
    } finally { dst.destroy(); }
}

// Caller retains shared brain tables. Destroy/recreate BOTH lanes for every episode; no reset of
// private contact phase is assumed. Row callbacks may differ by sign; seeds/geometry never do.
export function buildMirroredLanes(engine, geometry, options) {
    const lanes = [], seeds = Array.from({ length: geometry.arenas }, (_, a) => options.seedOf(a));
    try {
        for (let sign = 0; sign < 2; sign++) lanes.push(buildLane(engine, geometry, { ...options,
            seedOf: a => seeds[a], rowOf: (a, species) => options.rowOf(sign, a, species) }));
        return { lanes, destroy() { for (const lane of lanes) lane.destroy(); } };
    } catch (error) { for (const lane of lanes) lane.destroy(); throw error; }
}

