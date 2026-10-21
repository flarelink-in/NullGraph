// Replacement S7 protocol 2: spike-only arena geometry, not production code.
// Wall/integrator formulas transcribed from the existing agents generators; I2 guards drift.
import * as A from '../../../dist/agents.js';
import { DiscContacts } from '../../../dist/physics.js';
import { record } from './evidence.mjs';
import { DRIVER, HERD_CATCHES, HERD_DEFAULTS, SETTINGS, herdRule } from '../../../tools/agents/scenario/herd-wolves.mjs';
const f32 = v => Number.isInteger(v) ? v.toFixed(1) : String(v);
export function layout(arenas, wolves = 16, prey = 256, gap = 10) {
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
export function arenaDefinition(lay) {
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
export const containmentWGSL = lay => `${arenaWGSL(lay)}
fn step(i: u32, tick: u32) {
    let a = agents[i];
    if ((a.ids.y & 2u) != 0u) { return; }
    let lo = arenaLo(a.ids.w) + a.posRadius.w;
    let hi = arenaLo(a.ids.w) + vec2<f32>(ARENA_L) - a.posRadius.w;
    let p = clamp(a.posRadius.xz, lo, hi);
    if ((a.posRadius.x < lo.x && a.velHeading.x < 0.0) || (a.posRadius.x > hi.x && a.velHeading.x > 0.0)) { agents[i].velHeading.x = 0.0; }
    if ((a.posRadius.z < lo.y && a.velHeading.z < 0.0) || (a.posRadius.z > hi.y && a.velHeading.z > 0.0)) { agents[i].velHeading.z = 0.0; }
    agents[i].posRadius.x = p.x; agents[i].posRadius.z = p.y;
}`;

export function buildLane(engine, lay, { environment = 'legacy-v1', contacts = false, thinkEvery = 3,
    seedOf = () => 11, rowOf = () => 0, brains, fixture, mutation, contactOptions = {} } = {}) {
    if (!(lay.gap > 8)) throw new Error('S7: unsafe tile gap');
    const device = engine.device, owned = [], def = arenaDefinition(lay);
    const sys = new A.AgentSystem(engine, def, { thinkEvery });
    let solver;
    const alloc = (size, data) => {
        const b = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        owned.push(b); if (data) device.queue.writeBuffer(b, 0, data); return b;
    };
    try {
        const settings = alloc(16, new Float32Array([1, 1, 5, 0])), stats = alloc(16);
        const counters = alloc(Math.max(16, lay.arenas * 4));
        if (brains) for (const name of ['prey', 'wolf']) sys.setBrain(brains[name], { species: name });
        else sys.setBrain({ kind: 'stage', wgsl: `const ACT_TURN = 0u;\nconst ACT_THROTTLE = 1u;\n${herdRule(DRIVER)}`, buffers: [settings] });
        const catches = environment === 'legacy-v1' ? HERD_CATCHES : HERD_CATCHES
            .replace(' < 0.9', ' < agents[w].posRadius.w + me.posRadius.w + 0.05')
            .replace(' < 0.9', ' < agents[p].posRadius.w + me.posRadius.w + 0.05');
        sys.addStepStage({ name: 'catches', wgsl: catches });
        let respawn = respawnWGSL(lay);
        if (mutation === 'respawn-id') respawn = respawn.replaceAll('rand01(i, tick,', 'rand01(i + 1u, tick,');
        if (mutation === 'double-count') respawn = respawn.replace('atomicAdd(&arenaCatches[agents[i].ids.w], 1u)', 'atomicAdd(&arenaCatches[agents[i].ids.w], 2u)');
        sys.addStepStage({ name: 'respawn', wgsl: respawn, buffers: [settings, stats, counters] });
        if (contacts) {
            solver = new DiscContacts(engine, sys.buffer, { capacity: lay.count, stride: A.AGENT_FLOATS,
                world: { min: lay.origin, max: [lay.origin[0] + lay.spanX, lay.origin[1] + lay.spanZ] },
                maxRadius: 0.6, alive: { offset: A.AGENT_FLAGS_OFFSET, mask: A.AGENT_FLAG_ALIVE },
                fixed: { offset: A.AGENT_FLAGS_OFFSET, mask: A.AGENT_FLAG_FROZEN },
                velocity: { offset: A.AGENT_VELOCITY_OFFSET }, clamp: true, ...contactOptions });
            sys.addStepStage({ name: 'contacts', stages: solver.getComputeStages() });
        }
        if (environment !== 'legacy-v1' && mutation !== 'omit-containment')
            sys.addStepStage({ name: 'containment', wgsl: containmentWGSL(lay) });
        for (let a = 0; a < lay.arenas; a++) {
            const lo = [lay.origin[0] + a % lay.cols * lay.pitch, lay.origin[1] + Math.floor(a / lay.cols) * lay.pitch];
            const inset = lay.L / 2 - 1, seed = seedOf(a);
            for (let slot = 0; slot < lay.perArena; slot++) {
                const wolf = slot < lay.wolves, name = wolf ? 'wolf' : 'prey';
                const i = wolf ? a * lay.wolves + slot : lay.wolfTotal + a * lay.prey + slot - lay.wolves;
                const init = { position: [lo[0] + lay.L / 2 + (A.rand01(slot, seed, 0) * 2 - 1) * inset, 0.08,
                    lo[1] + lay.L / 2 + (A.rand01(slot, seed, 1) * 2 - 1) * inset],
                    species: name, heading: A.rand01(slot, seed, 2) * Math.PI * 2 - Math.PI,
                    genome: rowOf(a, name), user1: a };
                fixture?.(init, { i, slot, a, lo, lay, wolf });
                sys.spawn(i, init);
            }
        }
        sys.upload();
        if (sys.advance(def.dt) !== 1) throw new Error('S7: invalid one-step chain');
        const chain = sys.getComputeStages();
        let destroyed = false;
        return { sys, lay, stats, counters, settings, solver, chain,
            destroy() { if (destroyed) return; destroyed = true; sys.destroy(); solver?.destroy(); for (const b of owned) b.destroy(); } };
    } catch (e) { sys.destroy(); solver?.destroy(); for (const b of owned) b.destroy(); throw e; }
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

// No hidden warm-up: every submitted copy counts toward exactly steps scored steps.
export async function runSteps(engine, lane, steps) {
    // Never jump from a single fast probe to ten copies. Slowest observed cost controls growth;
    // the first four scored submissions are singles and a later outlier cannot be forgotten.
    let done = 0, repeat = 1, worst = 0, perStepHigh = 10; const batches = [];
    const preparationStart = performance.now();
    await engine.device.__s7Prepare?.();
    const preparationMs = performance.now() - preparationStart;
    while (done < steps) {
        const n = Math.min(repeat, steps - done), start = performance.now();
        engine.render(Array.from({ length: n }, () => lane.chain).flat());
        await engine.device.queue.onSubmittedWorkDone();
        const ms = performance.now() - start; worst = Math.max(worst, ms); batches.push({ n, ms });
        if (ms > 1500) {
            record('submission-failed', { steps, done, n, ms, preparationMs, batches });
            throw new Error(`S7: unsafe submission ${ms.toFixed(1)} ms`);
        }
        done += n;
        // Yield the host event loop between fenced submissions (especially Chrome), so driver
        // completion/error delivery and UI scheduling are not starved by promise-only loops.
        await new Promise(resolve => setTimeout(resolve, 0));
        perStepHigh = Math.max(perStepHigh, ms / n);
        repeat = done < 4 ? 1 : Math.max(1, Math.min(2, repeat + 1, Math.floor(25 / perStepHigh)));
    }
    return { done, worst, preparationMs, batches };
}
