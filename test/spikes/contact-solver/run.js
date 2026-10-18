// Phase 2, step 13, phase 13.0: spike S9 (docs/physics/physics-contacts-design.md §16).
// Disc contacts in the XZ plane, as the design's §2 defines them: a solver-owned PointGrid (G2) or
// the agents' grid (G1), K averaged Jacobi passes (gather, apply), relaxation, compliance, a push
// limit, a per-cell scan budget, velocity damping and two mass rules. A measurement, not a test
// suite: its numbers go in the design.
//
//   Node + Dawn, every adapter Dawn offers:  npm run spike:s9
//   One adapter:  NG_DAWN_FLAGS=adapter=NVIDIA npm run spike:s9   (or adapter=AMD)
//   Chrome:  npm run drive -- --serve . "test/spikes/contact-solver/index.html?part=grid" --wait-for "S9 done" --timeout 3000
//   One part:  NG_SPIKE_PART=checks|grid|search|frontier|finish|extreme|confirm under Node, ?part=… in Chrome
//   The point the later parts use:  NG_SPIKE_K, NG_SPIKE_OMEGA, NG_SPIKE_BUDGET, NG_SPIKE_BETA,
//   NG_SPIKE_MASS (area|equal), NG_SPIKE_ALPHA, NG_SPIKE_AVG (1: average, 0.5: square root)
//   (?k=… etc. in Chrome). Defaults are S8's. The 2026-10-18 results ran checks, finish, extreme
//   and confirm at NG_SPIKE_K=2 NG_SPIKE_OMEGA=1.75 NG_SPIKE_AVG=0.5; grid at either (relaxation and
//   the divisor do not change the work); search and frontier at the defaults (they set their own).
//
// Parts:
//   checks   completeness against brute force at every pass, at the cell rule and below it; one pair
//            ends exactly touching; two 300-step runs of three worlds word for word.
//   grid     contact stages' GPU time from one flocked state: G1 (agents' 8 m grid, exhaustive and
//            B = 64) against G2 (own grid, B = 16, 32, 64), and the step without contacts.
//   search   K ∈ {1, 2, 4} × ω ∈ {1, 1.25, 1.5, 1.75}: herds, piles and the jam; contact cost per K.
//   frontier past that grid: a 0.5 m push, 8 passes, the square-root scaling, and plain sums.
//   finish   at the chosen K, ω: damping β ∈ {0, 1} × mass {area, equal} × compliance {0, 1e-4, 1e-3}.
//   extreme  bodies spawned on one point: the step's contact time and skipped entries per budget,
//            exhaustive scans at small counts only, and how the pile opens.
//   confirm  the chosen point: herds of 4,096 and 16,384 and 64 arenas, with and without contacts,
//            catches included; whole-step costs at 4K–64K (to 1M on NVIDIA under Node).
import { harnessEnv } from '../../gpu/harness-env.mjs';
import * as A from '../../../dist/agents.js';
import { NullGraph } from '../../../dist/index.js';
import { DRIVER, HERD_DEFAULTS, SETTINGS, createHerd, ruleBrain } from '../../../tools/agents/scenario/herd-wolves.mjs';

const { log, check, fatal, finish, inBrowser } = harnessEnv('__S9_RESULTS__');
const param = name => (inBrowser ? new URLSearchParams(location.search).get(name) : globalThis.process?.env[`NG_SPIKE_${name.toUpperCase()}`]);
const PART = param('part') || 'all';
const PARTS = ['checks', 'grid', 'search', 'frontier', 'finish', 'extreme', 'confirm'];
if (PART !== 'all' && !PARTS.includes(PART)) throw new Error(`S9: unknown part "${PART}"`);
const want = part => PART === 'all' || PART === part;
const num = (name, fallback) => { const v = param(name); return v === null || v === undefined || v === '' ? fallback : Number(v); };
// The chosen point. S8's settings until the search sets them.
const POINT = {
    iterations: num('k', 2), relaxation: num('omega', 1), budget: num('budget', 32),
    damping: num('beta', 1), mass: param('mass') || 'area', compliance: num('alpha', 0), maxPush: 0.25,
    // 1: divide the summed correction by the contact count (the design's §2); 0.5: by its square root.
    averaging: num('avg', 1),
};

const STORAGE = GPUBufferUsage.STORAGE, COPY_SRC = GPUBufferUsage.COPY_SRC, COPY_DST = GPUBufferUsage.COPY_DST;
const RMAX = 0.6, RSMALL = 0.4;
const MAX_CELLS = 1_048_575;           // two radix passes
const STEP_STOP_MS = 1500;             // the timeout guard: no single step near Windows' 2 s reset
const fmt = n => n.toLocaleString('en-US');
const flt = v => (Number.isInteger(v) ? `${v}.0` : String(v));
const median = v => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)];

// ---------------------------------------------------------------------------------------------
// The §2 pass in WGSL: gather (one invocation per body, its 3×3 cells, at most B entries of each)
// and apply (average, relax, limit, clamp, damp). Pass k is baked into its gather's window.
// ---------------------------------------------------------------------------------------------
const TIE_WGSL = `
fn tieBreak(i: u32, j: u32) -> vec2<f32> {
    let lo = min(i, j);
    let hi = max(i, j);
    var h = (lo * 0x9e3779b9u) ^ ((hi + 0x7f4a7c15u) * 0x85ebca6bu);
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    h = h ^ (h >> 16u);
    let a = f32(h >> 8u) * (6.283185307179586 / 16777216.0);
    let n = vec2<f32>(cos(a), sin(a));
    return select(-n, n, i == lo);
}`;

function gatherWGSL(o, layout, pass) {
    const S4 = o.stride / 4, fv = Math.floor(o.flagsOffset / 4), fc = 'xyzw'[o.flagsOffset % 4];
    const budgeted = Number.isFinite(o.budget);
    return `
@group(0) @binding(0) var<storage, read> records: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read> bodies: array<u32>;
@group(0) @binding(3) var<storage, read_write> deltas: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: vec4<f32>;
@group(0) @binding(5) var<storage, read> counters: array<u32>;
${A.GRID_WINDOW_WGSL}
${A.pointCellWGSL(layout)}
${TIE_WGSL}
fn flagsOf(i: u32) -> u32 { return bitcast<u32>(records[i * ${S4}u + ${fv}u].${fc}); }
fn inverseMass(flags: u32, radius: f32) -> f32 {
    if ((flags & ${o.fixedMask >>> 0}u) != 0u) { return 0.0; }
    ${o.mass === 'area' ? `let r = max(radius, 1e-4);
    return ${flt(o.maxRadius * o.maxRadius)} / (r * r);` : 'return 1.0;'}
}
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= ${o.capacity}u) { return; }
    let me = records[i * ${S4}u];
    let flags = flagsOf(i);
    let wi = inverseMass(flags, me.w);
    if ((flags & ${o.aliveMask}u) == 0u || wi == 0.0) {
        deltas[i] = vec4<f32>(0.0);
        return;
    }
    let cell = pointCell(me.xyz);
    let cx = i32(cell % ${layout.grid.cellsX}u);
    let cz = i32(cell / ${layout.grid.cellsX}u);
    let window = counters[0] * ${o.iterations}u + ${pass}u;
    var sum = vec2<f32>(0.0);
    var count = 0u;
    for (var z = max(0, cz - 1); z <= min(${layout.grid.cellsZ - 1}, cz + 1); z++) {
        for (var x = max(0, cx - 1); x <= min(${layout.grid.cellsX - 1}, cx + 1); x++) {
            let c = u32(z) * ${layout.grid.cellsX}u + u32(x);
            let first = offsets[c];
            let n = offsets[c + 1u] - first;
            var at = first;
            ${budgeted ? `if (n > ${o.budget}u) { at += gridWindow(c, window) % n; }
            for (var left = min(n, ${o.budget}u); left > 0u; left--) {` : `_ = window;
            for (var left = n; left > 0u; left--) {`}
                let j = bodies[at];
                at++;
                if (at == first + n) { at = first; }
                if (j == i) { continue; }
                let other = records[j * ${S4}u];
                let d = me.xz - other.xz;
                let reach = me.w + other.w;
                let d2 = dot(d, d);
                if (d2 >= reach * reach) { continue; }
                let otherFlags = flagsOf(j);
                if ((otherFlags & ${o.aliveMask}u) == 0u) { continue; }
                let wj = inverseMass(otherFlags, other.w);
                let dist = sqrt(d2);
                let normal = select(tieBreak(i, j), d / dist, dist > 1e-6);
                sum += normal * ((reach - dist) * wi / (wi + wj + params.y));
                count++;
            }
        }
    }
    deltas[i] = vec4<f32>(sum, f32(count), 0.0);
}`;
}

function applyWGSL(o) {
    const S4 = o.stride / 4, w = o.world, vel = o.velocityOffset >= 0 ? o.velocityOffset / 4 : -1;
    return `
@group(0) @binding(0) var<storage, read_write> records: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> deltas: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: vec4<f32>;
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= ${o.capacity}u) { return; }
    let d = deltas[i];
    if (d.z <= 0.0) { return; }
    var u = d.xy * (params.x / ${o.averaging === 1 ? 'd.z' : o.averaging === 0 ? '1.0' : `pow(d.z, ${flt(o.averaging)})`});
    let len = length(u);
    if (len > params.z) { u *= params.z / len; }
    var p = records[i * ${S4}u];
    var xz = p.xz + u;
    ${o.clamp ? `let mid = vec2<f32>(${flt((w.minX + w.maxX) / 2)}, ${flt((w.minZ + w.maxZ) / 2)});
    let lo = min(vec2<f32>(${flt(w.minX)}, ${flt(w.minZ)}) + p.w, mid);
    let hi = max(vec2<f32>(${flt(w.maxX)}, ${flt(w.maxZ)}) - p.w, mid);
    xz = clamp(xz, lo, hi);` : ''}
    p.x = xz.x;
    p.z = xz.y;
    records[i * ${S4}u] = p;
    ${vel >= 0 ? `if (params.w > 0.0 && len > 1e-9) {
        let n = u / length(u);
        var v = records[i * ${S4}u + ${vel}u];
        let vn = dot(v.xz, n);
        if (vn < 0.0) {
            let damped = v.xz - (params.w * vn) * n;
            v.x = damped.x;
            v.z = damped.y;
            records[i * ${S4}u + ${vel}u] = v;
        }
    }` : ''}
}`;
}

const TICK_WGSL = `
@group(0) @binding(0) var<storage, read_write> counters: array<u32>;
@compute @workgroup_size(1) fn cs_main() { counters[0] = counters[0] + 1u; }`;

/** The spike's DiscContacts: stages for one step, over `records`. */
function buildContacts(engine, records, options) {
    const o = { stride: 16, flagsOffset: 13, aliveMask: 1, fixedMask: 2, velocityOffset: 4, clamp: true,
        maxRadius: RMAX, dt: 1 / 60, averaging: 1, ...POINT, ...options };
    const device = engine.device, owned = [], made = [];
    const groups = Math.ceil(o.capacity / 64);
    const buffer = (label, size, usage) => {
        const b = device.createBuffer({ label: `S9 ${label}`, size, usage });
        owned.push(b);
        return b;
    };
    let grid, layout, gridStages = [];
    if (o.shared) {
        ({ grid, layout } = o.shared);
    } else {
        let cellSize = o.cellSize ?? 2 * o.maxRadius + (o.iterations - 1) * o.maxPush;
        const w = o.world, cells = h => [Math.ceil((w.maxX - w.minX) / h), Math.ceil((w.maxZ - w.minZ) / h)];
        // PointGrid's radix scan dispatches (buckets × blocks + 1) / 64 workgroups: at a million
        // bodies that caps the cell count, so the cell grows (completeness only gains).
        const limit = device.limits.maxComputeWorkgroupsPerDimension, blocks = Math.ceil(o.capacity / A.GRID_SCAN_BLOCK);
        const fits = h => {
            const total = cells(h)[0] * cells(h)[1];
            return total <= MAX_CELLS && Math.ceil((A.gridRadixLayout(total).buckets * blocks + 1) / 64) <= limit;
        };
        while (!fits(cellSize)) cellSize *= 1.25;
        const [cellsX, cellsZ] = cells(cellSize);
        layout = { world: { minX: w.minX, minZ: w.minZ }, cellSize, grid: { cellsX, cellsZ, cellTotal: cellsX * cellsZ } };
        grid = new A.PointGrid(engine, records, { ...layout, capacity: o.capacity, stride: o.stride,
            flagsOffset: o.flagsOffset, aliveMask: o.aliveMask,
            maxScanPerCell: Number.isFinite(o.budget) ? o.budget : o.capacity });
        gridStages = grid.getComputeStages();
    }
    const deltas = buffer('deltas', o.capacity * 16, STORAGE | COPY_SRC);
    const params = buffer('params', 16, GPUBufferUsage.UNIFORM | COPY_DST);
    const counters = buffer('counters', 16, STORAGE | COPY_SRC | COPY_DST);
    const setParams = p => device.queue.writeBuffer(params, 0,
        new Float32Array([p.relaxation, p.compliance / (o.dt * o.dt), p.maxPush, p.damping]));
    setParams(o);
    const stage = (name, shaderCode, bindings, count) => {
        const s = engine.createComputeStage({ name: `S9 ${name}`, shaderCode, dispatchCount: [count, 1, 1] });
        engine.setComputePassBindGroup(s, bindings.map((b, binding) => ({ binding, resource: { buffer: b } })), 0);
        made.push(s);
        return s;
    };
    const apply = stage('apply', applyWGSL(o), [records, deltas, params], groups);
    const gathers = [], passes = [];
    for (let k = 0; k < o.iterations; k++) {
        gathers.push(stage(`gather ${k}`, gatherWGSL(o, layout, k),
            [records, grid.cellOffsets, grid.cellBodies, deltas, params, counters], groups));
        passes.push(gathers[k], apply);
    }
    const tick = stage('tick', TICK_WGSL, [counters], 1);
    return {
        o, layout, grid, deltas, counters, gridStages, gathers, apply, setParams,
        stages: [...gridStages, ...passes, tick],
        destroy() {
            for (const s of made) s.enabled = false;
            if (!o.shared) grid.destroy();
            for (const b of owned) b.destroy();
        },
    };
}

/** An AgentSystem's one-step chain with `stages` right after act, before the app stages. */
function afterAct(agents, stages) {
    if (agents.advance(agents.definition.dt) !== 1) throw new Error('S9: advance(dt) did not select one step');
    const chain = [...agents.getComputeStages()], at = chain.findIndex(s => s.name === 'AgentSystem act');
    if (at < 0) throw new Error('S9: no act stage');
    return [...chain.slice(0, at + 1), ...stages, ...chain.slice(at + 1)];
}

const worldOf = def => ({ minX: def.world.minX, minZ: def.world.minZ, maxX: def.world.maxX, maxZ: def.world.maxZ });

// ---------------------------------------------------------------------------------------------
// Worlds
// ---------------------------------------------------------------------------------------------
/** Herd & Wolves on the rule (createHerd), with contacts after act, or none. */
function herdSim(engine, count, contacts, extra = {}) {
    const herd = createHerd(engine, { count });
    const c = contacts ? buildContacts(engine, herd.agents.buffer, { capacity: count, world: worldOf(herd.definition),
        dt: herd.definition.dt, ...contacts, ...extra }) : null;
    const chain = afterAct(herd.agents, c ? c.stages : []);
    return { kind: 'herd', count, agents: herd.agents, def: herd.definition, stats: herd.stats, wolves: herd.wolves,
        contacts: c, chain, destroy() { c?.destroy(); herd.destroy(); } };
}

const STILL = `
fn think(i: u32, species: u32, tick: u32) {
    setAct(i, ACT_TURN, 0.0);
    setAct(i, ACT_THROTTLE, -1.0);
}`;
const INWARD = `
fn think(i: u32, species: u32, tick: u32) {
    let a = agents[i];
    let toOrigin = -a.posRadius.xz;
    let sn = sin(a.velHeading.w);
    let cs = cos(a.velHeading.w);
    let side = dot(toOrigin, vec2<f32>(cs, -sn));
    let fwd = dot(toOrigin, vec2<f32>(sn, cs));
    setAct(i, ACT_TURN, clamp(2.5 * atan2(side, fwd), -1.0, 1.0));
    setAct(i, ACT_THROTTLE, 1.0);
}`;

/** A crowd of two radii (one body in 16 is large): a pile (still bodies in a box, or on one point) or a jam. */
function crowdSim(engine, { count, half, spread, inward = false, allLarge = false, contacts, extra = {}, seed = 77 }) {
    const def = A.defineAgents({ capacity: count, world: { min: [-half, -half], max: [half, half] }, cellSize: 8,
        species: [
            { name: 'small', maxSpeed: 3.2, maxTurn: 4, accel: 6, senseRadius: 8, radius: RSMALL },
            { name: 'large', maxSpeed: 3.2, maxTurn: 4, accel: 6, senseRadius: 8, radius: RMAX },
        ],
        sensors: [A.Sensors.self(), A.Sensors.bounds()] });
    const agents = new A.AgentSystem(engine, def, { thinkEvery: 1 });
    agents.setBrain({ kind: 'stage', wgsl: inward ? INWARD : STILL, buffers: [], destroy() {} });
    for (let i = 0; i < count; i++) {
        const ring = inward ? Math.sqrt(A.rand01(i, seed, 0)) * spread : 0;
        const angle = A.rand01(i, seed, 3) * Math.PI * 2;
        agents.spawn(i, {
            position: inward ? [Math.cos(angle) * ring, 0.08, Math.sin(angle) * ring]
                : [(A.rand01(i, seed, 0) * 2 - 1) * spread, 0.08, (A.rand01(i, seed, 1) * 2 - 1) * spread],
            species: allLarge || i % 16 === 0 ? 'large' : 'small', heading: A.rand01(i, seed, 2) * Math.PI * 2 - Math.PI,
        });
    }
    agents.upload();
    const c = contacts ? buildContacts(engine, agents.buffer, { capacity: count, world: worldOf(def), dt: def.dt,
        ...contacts, ...extra }) : null;
    const chain = afterAct(agents, c ? c.stages : []);
    return { kind: 'crowd', count, agents, def, contacts: c, chain, wolves: 0,
        destroy() { c?.destroy(); agents.destroy(); } };
}

// S7's 64 arenas (S8's layout), on the rule: arena walls from a custom sensor and custom actions.
function arenaLayout(arenas, wolves = 16, prey = 256, gap = 10) {
    const perArena = wolves + prey, L = 2 * 80 * Math.sqrt(perArena / HERD_DEFAULTS.count);
    const cols = Math.ceil(Math.sqrt(arenas)), rows = Math.ceil(arenas / cols), pitch = L + gap;
    const spanX = cols * pitch - gap, spanZ = rows * pitch - gap;
    return { arenas, wolves, prey, perArena, L, cols, pitch, origin: [-spanX / 2, -spanZ / 2], spanX, spanZ,
        count: arenas * perArena, wolfTotal: arenas * wolves };
}
const arenaConstsWGSL = lay => `
const ARENA_ORIGIN_X: f32 = ${flt(lay.origin[0])};
const ARENA_ORIGIN_Z: f32 = ${flt(lay.origin[1])};
const ARENA_PITCH: f32 = ${flt(lay.pitch)};
const ARENA_L: f32 = ${flt(lay.L)};
const ARENA_COLS = ${lay.cols}u;
fn arenaLo(arena: u32) -> vec2<f32> {
    return vec2<f32>(ARENA_ORIGIN_X + f32(arena % ARENA_COLS) * ARENA_PITCH, ARENA_ORIGIN_Z + f32(arena / ARENA_COLS) * ARENA_PITCH);
}`;
const arenaWalls = lay => A.Sensors.custom({ name: 'bounds', width: 2, wgsl: `${arenaConstsWGSL(lay)}
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
    if ((bitcast<u32>(a0) & 0x7f800000u) != 0x7f800000u && (bitcast<u32>(a1) & 0x7f800000u) != 0x7f800000u) {
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
    fx = sin(h);
    fz = cos(h);
    a.posRadius.x = x;
    a.posRadius.z = z;
    a.velHeading = vec4<f32>(fx * speed, 0.0, fz * speed, h);
    agents[i] = a;
}` });
const arenaRespawnWGSL = lay => `${arenaConstsWGSL(lay)}
@group(1) @binding(0) var<storage, read> settings: array<f32>;
@group(1) @binding(1) var<storage, read_write> stats: array<atomic<u32>, 4>;
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
        respawn(i, randomPointInArena(i, tick), rand01(i, tick, 2u) * 6.2831853 - 3.1415927);
        agents[i].ids.y &= ~256u;
    }
}`;
const ARENA_CATCHES = `
fn step(i: u32, tick: u32) {
    let me = agents[i];
    if (me.ids.x == SPECIES_PREY) {
        let w = neighbours[i * NEIGHBOURS_PER_AGENT + NEIGHBOURS_WOLF];
        if (w != NO_AGENT && agents[w].state.w <= 0.0 && distance(agents[w].posRadius.xz, me.posRadius.xz) < 0.9) {
            agents[i].ids.y |= 256u;
        }
    } else if (me.state.w <= 0.0) {
        let p = neighbours[i * NEIGHBOURS_PER_AGENT + NEIGHBOURS_PREY];
        if (p != NO_AGENT && distance(agents[p].posRadius.xz, me.posRadius.xz) < 0.9) { agents[i].ids.y |= 512u; }
    }
}`;

function arenaSim(engine, arenas, contacts) {
    const lay = arenaLayout(arenas), o = HERD_DEFAULTS, device = engine.device;
    const def = A.defineAgents({
        capacity: lay.count,
        world: { min: lay.origin, max: [lay.origin[0] + lay.spanX, lay.origin[1] + lay.spanZ] },
        cellSize: 8, maxScanPerCell: o.scanBudget,
        species: [
            { name: 'prey', maxSpeed: o.preySpeed, maxTurn: 4, accel: 6, senseRadius: 8, radius: 0.4 },
            { name: 'wolf', maxSpeed: o.wolfSpeed, maxTurn: 3, accel: 5, senseRadius: 8, radius: 0.6 },
        ],
        sensors: [A.Sensors.self(), A.Sensors.nearest({ name: 'wolf', species: ['wolf'], k: 1 }),
            A.Sensors.nearest({ name: 'prey', species: ['prey'], k: 3 }), arenaWalls(lay)],
        actions: arenaSteering(lay),
    });
    const settings = device.createBuffer({ size: 16, usage: STORAGE | COPY_DST });
    const data = new Float32Array(4);
    data[SETTINGS.FLOCKING] = 1; data[SETTINGS.FLEEING] = 1; data[SETTINGS.REST_SECONDS] = o.restSeconds;
    device.queue.writeBuffer(settings, 0, data);
    const stats = device.createBuffer({ size: 16, usage: STORAGE | COPY_SRC | COPY_DST });
    device.queue.writeBuffer(stats, 0, new Uint32Array(4));
    const agents = new A.AgentSystem(engine, def, { thinkEvery: o.thinkEvery });
    // The rule writes steering's two actions by name; the arenas' custom actions are the same two floats.
    const rule = ruleBrain(DRIVER, settings);
    agents.setBrain({ ...rule, wgsl: `const ACT_TURN = 0u;\nconst ACT_THROTTLE = 1u;\n${rule.wgsl}` });
    agents.addStepStage({ name: 'catches', wgsl: ARENA_CATCHES });
    agents.addStepStage({ name: 'respawn', wgsl: arenaRespawnWGSL(lay), buffers: [settings, stats] });
    for (let a = 0; a < lay.arenas; a++) {
        const lo = [lay.origin[0] + (a % lay.cols) * lay.pitch, lay.origin[1] + Math.floor(a / lay.cols) * lay.pitch];
        const cx = lo[0] + lay.L / 2, cz = lo[1] + lay.L / 2, inset = lay.L / 2 - 1, s = 900001 + a;
        for (let slot = 0; slot < lay.perArena; slot++) {
            const wolf = slot < lay.wolves;
            const i = wolf ? a * lay.wolves + slot : lay.wolfTotal + a * lay.prey + (slot - lay.wolves);
            agents.spawn(i, { position: [cx + (A.rand01(slot, s, 0) * 2 - 1) * inset, 0.08, cz + (A.rand01(slot, s, 1) * 2 - 1) * inset],
                species: wolf ? 'wolf' : 'prey', heading: A.rand01(slot, s, 2) * Math.PI * 2 - Math.PI, user1: a });
        }
    }
    agents.upload();
    const c = contacts ? buildContacts(engine, agents.buffer, { capacity: lay.count, world: worldOf(def), dt: def.dt, ...contacts }) : null;
    const chain = afterAct(agents, c ? c.stages : []);
    return { kind: 'arenas', count: lay.count, lay, agents, def, stats, contacts: c, chain, wolves: lay.wolfTotal,
        destroy() { c?.destroy(); agents.destroy(); settings.destroy(); stats.destroy(); } };
}

// ---------------------------------------------------------------------------------------------
// Running, reading, measuring
// ---------------------------------------------------------------------------------------------
async function readU32(device, src, words) {
    const dst = device.createBuffer({ size: words * 4, usage: COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, dst, 0, words * 4);
    device.queue.submit([enc.finish()]);
    await dst.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(dst.getMappedRange().slice(0));
    dst.unmap();
    dst.destroy();
    return out;
}
const records = (device, sim) => readU32(device, sim.agents.buffer, sim.count * 16);

/** Steps a sim, a few per submission, never one near the timeout. */
async function run(engine, sim, steps) {
    if (!sim.perSubmit) {
        const t = performance.now();
        engine.render(sim.chain);
        await engine.device.queue.onSubmittedWorkDone();
        const probe = performance.now() - t;
        if (probe > STEP_STOP_MS) throw new Error(`S9: one ${sim.kind} step of ${fmt(sim.count)} took ${probe.toFixed(0)} ms`);
        sim.perSubmit = Math.max(1, Math.min(16, Math.floor(60 / Math.max(probe, 0.5))));
        steps--;
    }
    for (let done = 0; done < steps; ) {
        const n = Math.min(sim.perSubmit, steps - done);
        engine.render(Array.from({ length: n }, () => sim.chain).flat());
        await engine.device.queue.onSubmittedWorkDone();
        done += n;
    }
}

/** Overlap over live bodies: worst penetration (m, % of summed radii), % of bodies over 5%, and the
 * mean nearest-neighbour distance among bodies from `firstSmall` on (prey, or the small discs). */
function overlap(rec, { groups = null, firstSmall = 0 } = {}) {
    const f = new Float32Array(rec.buffer), count = rec.length / 16;
    let worstM = 0, worstPct = 0, over5 = 0, over20 = 0, live = 0, nnSum = 0, nnN = 0;
    const sweep = indices => {
        indices = indices.filter(i => rec[i * 16 + 13] & 1);
        live += indices.length;
        indices.sort((a, b) => f[a * 16] - f[b * 16] || a - b);
        const n = indices.length, flagged = new Uint8Array(n), severe = new Uint8Array(n);
        let rMax = 0;
        for (const i of indices) rMax = Math.max(rMax, f[i * 16 + 3]);
        for (let a = 0; a < n; a++) {
            const i = indices[a], xi = f[i * 16], zi = f[i * 16 + 2], ri = f[i * 16 + 3];
            for (let b = a + 1; b < n; b++) {
                const j = indices[b], dx = f[j * 16] - xi;
                if (dx > ri + rMax) break;
                const dz = f[j * 16 + 2] - zi, rij = ri + f[j * 16 + 3], d2 = dx * dx + dz * dz;
                if (d2 < rij * rij) {
                    const pen = rij - Math.sqrt(d2), pct = pen / rij;
                    if (pen > worstM) worstM = pen;
                    if (pct > worstPct) worstPct = pct;
                    if (pct > 0.05) { flagged[a] = 1; flagged[b] = 1; }
                    if (pct > 0.2) { severe[a] = 1; severe[b] = 1; }
                }
            }
            if (i >= firstSmall) {
                let best = Infinity;
                for (const dir of [-1, 1]) {
                    for (let b = a + dir; b >= 0 && b < n; b += dir) {
                        const j = indices[b], dx = f[j * 16] - xi;
                        if (dx * dx >= best) break;
                        if (j < firstSmall) continue;
                        const dz = f[j * 16 + 2] - zi;
                        best = Math.min(best, dx * dx + dz * dz);
                    }
                }
                if (best < Infinity) { nnSum += Math.sqrt(best); nnN++; }
            }
        }
        for (let a = 0; a < n; a++) { over5 += flagged[a]; over20 += severe[a]; }
    };
    if (groups) for (const g of groups) sweep(g);
    else sweep(Array.from({ length: count }, (_, i) => i));
    return { worstM, worstPct: 100 * worstPct, over5Pct: live ? 100 * over5 / live : 0,
        over20Pct: live ? 100 * over20 / live : 0, nearestM: nnN ? nnSum / nnN : 0 };
}
const arenaGroups = lay => Array.from({ length: lay.arenas }, (_, a) => [
    ...Array.from({ length: lay.wolves }, (_, w) => a * lay.wolves + w),
    ...Array.from({ length: lay.prey }, (_, p) => lay.wolfTotal + a * lay.prey + p)]);

/** Every step for `steps`: the worst share over 5% and worst penetration, and the final state. */
async function herdQuality(engine, sim, steps = 600) {
    const opts = sim.kind === 'arenas' ? { groups: arenaGroups(sim.lay), firstSmall: sim.wolves } : { firstSmall: sim.wolves };
    let maxOver5 = 0, maxWorstPct = 0, last;
    for (let s = 0; s < steps; s++) {
        await run(engine, sim, 1);
        last = overlap(await records(engine.device, sim), opts);
        maxOver5 = Math.max(maxOver5, last.over5Pct);
        maxWorstPct = Math.max(maxWorstPct, last.worstPct);
    }
    const catches = sim.stats ? (await readU32(engine.device, sim.stats, 1))[0] : 0;
    return { maxOver5, maxWorstPct, final: last, catches };
}

/** A still pile: steps until under 5% of bodies over 5% overlap (sampled every 3), capped. */
async function pileQuality(engine, sim, cap = 180) {
    let clearedAt = null, last;
    for (let s = 3; s <= cap; s += 3) {
        await run(engine, sim, 3);
        last = overlap(await records(engine.device, sim));
        if (clearedAt === null && last.over5Pct < 5) clearedAt = s;
    }
    return { clearedAt, final: last };
}

/** The jam: steady share over 5% (mean of the last 120 steps, every 2nd), worst penetration, and
 * displacement reversals over the last 60 steps (cos < −0.5 between consecutive moves). */
async function jamQuality(engine, sim, steps = 480) {
    await run(engine, sim, steps - 120);
    const over5 = [], over20 = [];
    let worstPct = 0, prev = null, prevD = null, reversals = 0, moves = 0;
    for (let s = steps - 120; s < steps; s++) {
        await run(engine, sim, 1);
        const rec = await records(engine.device, sim), f = new Float32Array(rec.buffer);
        if (s % 2 === 0) {
            const m = overlap(rec);
            over5.push(m.over5Pct);
            over20.push(m.over20Pct);
            worstPct = Math.max(worstPct, m.worstPct);
        }
        if (s >= steps - 60) {
            const d = new Float32Array(sim.count * 2);
            for (let i = 0; i < sim.count; i++) {
                d[i * 2] = prev ? f[i * 16] - prev[i * 16] : 0;
                d[i * 2 + 1] = prev ? f[i * 16 + 2] - prev[i * 16 + 2] : 0;
                if (prev && prevD) {
                    const len = Math.hypot(d[i * 2], d[i * 2 + 1]), plen = Math.hypot(prevD[i * 2], prevD[i * 2 + 1]);
                    if (len > 1e-4 && plen > 1e-4) {
                        moves++;
                        if ((d[i * 2] * prevD[i * 2] + d[i * 2 + 1] * prevD[i * 2 + 1]) / (len * plen) < -0.5) reversals++;
                    }
                }
            }
            if (prev) prevD = d;
        }
        prev = f;
    }
    const mean = v => v.reduce((a, b) => a + b, 0) / v.length;
    return { steadyOver5: mean(over5), steadyOver20: mean(over20), worstPct,
        reversalsPct: moves ? 100 * reversals / moves : 0 };
}

// ---------------------------------------------------------------------------------------------
// GPU timestamps: S6's two marks around a submitted sequence, and a restore stage so every
// repetition starts from the same snapshot (gotchas: compare builds from one state).
// ---------------------------------------------------------------------------------------------
function timer(engine) {
    const device = engine.device;
    const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
    const resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | COPY_SRC });
    const read = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | COPY_DST });
    const mark = index => {
        const enc = device.createCommandEncoder();
        enc.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: index } }).end();
        device.queue.submit([enc.finish()]);
    };
    return async sequence => {
        mark(0);
        engine.render(sequence);
        mark(1);
        const enc = device.createCommandEncoder();
        enc.resolveQuerySet(querySet, 0, 2, resolve, 0);
        enc.copyBufferToBuffer(resolve, 0, read, 0, 16);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const t = new BigUint64Array(read.getMappedRange().slice(0));
        read.unmap();
        return Number(t[1] - t[0]) / 1e6;
    };
}

function snapshot(engine, sim) {
    const device = engine.device, words = sim.count * 16;
    const copy = device.createBuffer({ label: 'S9 snapshot', size: words * 4, usage: STORAGE | COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(sim.agents.buffer, 0, copy, 0, words * 4);
    device.queue.submit([enc.finish()]);
    const restore = engine.createComputeStage({ name: 'S9 restore', dispatchCount: [Math.ceil(words / 4 / 256), 1, 1], shaderCode: `
@group(0) @binding(0) var<storage, read> src: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<u32>>;
@compute @workgroup_size(256) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < ${words / 4}u) { dst[id.x] = src[id.x]; }
}` });
    engine.setComputePassBindGroup(restore, [copy, sim.agents.buffer].map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
    return { restore, destroy() { restore.enabled = false; copy.destroy(); } };
}

/** GPU ms per repetition of each named sequence, from the snapshot, alternated over rounds. */
async function costs(engine, time, snap, sequences, rounds = 5) {
    const reps = {}, samples = {};
    const base = [snap.restore];
    for (const [name, seq] of Object.entries({ restore: [], ...sequences })) {
        const probe = await time([...base, ...seq]);
        if (probe > STEP_STOP_MS) throw new Error(`S9: ${name} took ${probe.toFixed(0)} ms in one repetition`);
        reps[name] = Math.max(1, Math.min(64, Math.ceil(8 / Math.max(probe, 0.05)), Math.floor(250 / Math.max(probe, 0.05))));
        samples[name] = [];
    }
    for (let r = 0; r < rounds; r++) {
        for (const [name, seq] of Object.entries({ restore: [], ...sequences })) {
            const n = reps[name];
            samples[name].push(await time(Array.from({ length: n }, () => [...base, ...seq]).flat()) / n);
        }
    }
    const restoreMs = median(samples.restore), out = {};
    for (const name of Object.keys(sequences)) out[name] = Math.max(0, median(samples[name]) - restoreMs);
    return out;
}

// ---------------------------------------------------------------------------------------------
// The parts
// ---------------------------------------------------------------------------------------------
const pointName = p => `K=${p.iterations} ω=${p.relaxation} B=${p.budget} β=${p.damping} ${p.mass} α=${p.compliance}` +
    `${p.averaging !== undefined && p.averaging !== 1 ? ` n^${p.averaging}` : ''}${p.maxPush !== undefined && p.maxPush !== 0.25 ? ` m=${p.maxPush}` : ''}`;

async function partChecks(engine, name, result) {
    const device = engine.device;
    device.pushErrorScope('validation');
    // Completeness: exhaustive scans, every pass against brute force at that pass's positions. A dense
    // pile (mixed radii, many contacts, small moves) and a loose crowd of large discs (few contacts,
    // moves up to a 0.5 m push), each at the cell rule and at 2·ρmax, below it.
    result.completeness = {};
    const loose = { allLarge: true, count: 4096, half: 48, spread: 45, push: { maxPush: 0.5, relaxation: 1.5 } };
    const dense = { allLarge: false, count: 4096, half: 40, spread: 8, push: {} };
    for (const [label, layout, cellSize] of [
        ['dense pile, the cell rule', dense, undefined], ['dense pile, 2·ρmax', dense, 2 * RMAX],
        ['loose crowd, the cell rule', loose, undefined], ['loose crowd, 2·ρmax', loose, 2 * RMAX],
    ]) {
        const sim = crowdSim(engine, { count: layout.count, half: layout.half, spread: layout.spread, allLarge: layout.allLarge,
            contacts: { ...POINT, iterations: 2, budget: Infinity, ...layout.push, cellSize } });
        const c = sim.contacts, k0 = sim.chain.indexOf(c.gathers[0]);
        let missed = 0, extra = 0, ambiguous = 0, maxContacts = 0;
        engine.render(sim.chain.slice(0, k0));            // the step up to the contact grid
        for (let k = 0; k < 2; k++) {
            engine.render(k === 0 ? [c.gathers[0]] : [c.apply, c.gathers[1]]);
            await device.queue.onSubmittedWorkDone();
            const rec = new Float32Array((await records(device, sim)).buffer);
            const counts = new Float32Array((await readU32(device, c.deltas, sim.count * 4)).buffer);
            for (let i = 0; i < sim.count; i++) {
                let n = 0, unsure = false;
                for (let j = 0; j < sim.count; j++) {
                    if (j === i) continue;
                    const dx = rec[i * 16] - rec[j * 16], dz = rec[i * 16 + 2] - rec[j * 16 + 2];
                    const r = rec[i * 16 + 3] + rec[j * 16 + 3], d2 = dx * dx + dz * dz;
                    if (Math.abs(d2 - r * r) < 1e-5 * r * r) unsure = true;
                    if (d2 < r * r) n++;
                }
                maxContacts = Math.max(maxContacts, n);
                const got = counts[i * 4 + 2];
                if (got === n) continue;
                if (unsure) { ambiguous++; continue; }
                if (got < n) missed += n - got; else extra += got - n;
            }
        }
        result.completeness[label] = { cellSize: c.layout.cellSize, missed, extra, ambiguous, maxContacts };
        log(`   completeness, ${label} (h = ${c.layout.cellSize.toFixed(2)} m): missed ${missed}, extra ${extra}, ambiguous ${ambiguous}, most contacts ${maxContacts}`);
        sim.destroy();
    }
    const c = result.completeness;
    check(`${name}: at the cell rule, both passes find every contact brute force finds, dense and loose (C1)`,
        [c['dense pile, the cell rule'], c['loose crowd, the cell rule']].every(r => r.missed === 0 && r.extra === 0) &&
        c['dense pile, the cell rule'].maxContacts > 8, JSON.stringify(c));
    check(`${name}: watched failing: a cell of 2·ρmax misses contacts at the second pass of the loose crowd`,
        c['loose crowd, 2·ρmax'].missed > 0, JSON.stringify(c['loose crowd, 2·ρmax']));

    // One pair, one pass: two discs end exactly touching (C3), each moving its share.
    {
        const sim = crowdSim(engine, { count: 2, half: 10, spread: 0,
            contacts: { ...POINT, iterations: 1, relaxation: 1, compliance: 0, maxPush: 1, damping: 0, mass: 'area' } });
        sim.agents.spawn(0, { position: [0, 0.08, 0], species: 'small', heading: 0 });
        sim.agents.spawn(1, { position: [0.5, 0.08, 0.3], species: 'large', heading: 0 });
        sim.agents.upload();
        const c = sim.contacts;
        engine.render([...c.stages]);
        await device.queue.onSubmittedWorkDone();
        const f = new Float32Array((await records(device, sim)).buffer);
        const d = Math.hypot(f[16] - f[0], f[18] - f[2]);
        const w0 = (RMAX / RSMALL) ** 2, w1 = 1, d0 = Math.hypot(0.5, 0.3), pen = 1.0 - d0;
        const moved0 = Math.hypot(f[0], f[2]), share = w0 / (w0 + w1);
        result.onePair = { distance: d, moved0, expected0: pen * share };
        check(`${name}: one pair ends exactly touching, each moving its share (C3)`,
            Math.abs(d - 1.0) < 1e-5 && Math.abs(moved0 - pen * share) < 1e-5, JSON.stringify(result.onePair));
        sim.destroy();
    }

    // Exact reruns at the point: three worlds, two runs of 300 steps each, word for word (C2).
    result.reruns = {};
    for (const [label, make] of [
        ['herd 4,096', () => herdSim(engine, 4096, POINT)],
        ['pile 256', () => crowdSim(engine, { count: 256, half: 24, spread: 2, contacts: POINT })],
        ['jam 4,096', () => crowdSim(engine, { count: 4096, half: 40, spread: 20, inward: true, contacts: POINT })],
    ]) {
        const runs = [];
        for (let r = 0; r < 2; r++) {
            const sim = make();
            await run(engine, sim, 300);
            runs.push(await records(device, sim));
            sim.destroy();
        }
        let differ = 0;
        for (let w = 0; w < runs[0].length; w++) if (runs[0][w] !== runs[1][w]) differ++;
        result.reruns[label] = differ;
        check(`${name}: ${label}, two 300-step runs at ${pointName(POINT)} match word for word (C2)`, differ === 0, `${differ} words differ`);
    }
    const error = await device.popErrorScope();
    check(`${name}: checks ran without a WebGPU validation error`, error === null, error?.message ?? '');
}

/** A herd warmed to a flocked state with contacts at the point, and a snapshot of it. */
async function warmHerd(engine, count, steps) {
    const sim = herdSim(engine, count, POINT);
    await run(engine, sim, steps);
    await engine.device.queue.onSubmittedWorkDone();
    return { sim, snap: snapshot(engine, sim) };
}

async function partGrid(engine, name, result, time) {
    result.grid = [];
    for (const [count, warm] of [[16384, 600], [65536, 600]]) {
        const { sim, snap } = await warmHerd(engine, count, warm);
        const shared = { grid: sim.agents.grid, layout: sim.def };
        const variants = {
            'G1 exhaustive': { shared, budget: Infinity },
            'G1 B=64': { shared, budget: 64 },
            'G2 B=16': { budget: 16 }, 'G2 B=32': { budget: 32 }, 'G2 B=64': { budget: 64 },
        };
        const built = Object.fromEntries(Object.entries(variants).map(([k, v]) =>
            [k, buildContacts(engine, sim.agents.buffer, { capacity: count, world: worldOf(sim.def), ...POINT, ...v })]));
        // The herd's own step, without its contact stages: the denominator of T4.
        const stepChain = sim.chain.filter(s => !sim.contacts.stages.includes(s));
        const ms = await costs(engine, time, snap, {
            step: stepChain, ...Object.fromEntries(Object.entries(built).map(([k, c]) => [k, c.stages])) });
        result.grid.push({ count, stepMs: ms.step, contacts: Object.fromEntries(Object.keys(variants).map(k => [k, ms[k]])) });
        log(`   ${fmt(count)} herd: step without contacts ${ms.step.toFixed(3)} ms; ` +
            Object.keys(variants).map(k => `${k} ${ms[k].toFixed(3)} ms (${(100 * ms[k] / ms.step).toFixed(1)}%)`).join(', '));
        for (const c of Object.values(built)) c.destroy();
        snap.destroy();
        sim.destroy();
    }
}

async function searchPoint(engine, point) {
    const herd = herdSim(engine, 4096, point);
    const h = await herdQuality(engine, herd);
    herd.destroy();
    const p256 = crowdSim(engine, { count: 256, half: 24, spread: 2, contacts: point });
    const pile256 = await pileQuality(engine, p256);
    p256.destroy();
    const p4k = crowdSim(engine, { count: 4096, half: 40, spread: 8, contacts: point });
    const pile4096 = await pileQuality(engine, p4k);
    p4k.destroy();
    const jamSim = crowdSim(engine, { count: 4096, half: 40, spread: 20, inward: true, contacts: point });
    const jam = await jamQuality(engine, jamSim);
    jamSim.destroy();
    const meets = {
        T1: h.maxOver5 <= 2 && h.final.worstPct < 10,
        T2: pile256.clearedAt !== null,
        T3: jam.steadyOver5 <= 10 && jam.reversalsPct <= 8,
    };
    return { point: { ...point }, herd: h, pile256, pile4096, jam, meets };
}

function logPoint(r) {
    const p = r.point;
    log(`   ${pointName(p).padEnd(46)} herd max>5% ${r.herd.maxOver5.toFixed(2)}% worst ${r.herd.final.worstPct.toFixed(1)}% ` +
        `nn ${r.herd.final.nearestM.toFixed(2)} m catches ${r.herd.catches} · pile256 ${r.pile256.clearedAt ?? '>180'} (${r.pile256.final.over5Pct.toFixed(1)}%) · ` +
        `pile4096 ${r.pile4096.clearedAt ?? '>180'} (${r.pile4096.final.over5Pct.toFixed(1)}%) · jam ${r.jam.steadyOver5.toFixed(1)}% ` +
        `(>20%: ${r.jam.steadyOver20.toFixed(1)}%) worst ${r.jam.worstPct.toFixed(1)}% rev ${r.jam.reversalsPct.toFixed(2)}% · ` +
        `${['T1', 'T2', 'T3'].filter(t => r.meets[t]).join(' ') || '—'}`);
}

/** A few points past the search's grid: a larger push, more passes, and weaker averaging. */
async function partFrontier(engine, name, result, time) {
    const points = [
        { iterations: 4, relaxation: 1.75, maxPush: 0.5 },
        { iterations: 8, relaxation: 1.5 },
        { iterations: 2, relaxation: 1.75, averaging: 0.5 },
        { iterations: 4, relaxation: 1.5, averaging: 0.5 },
        { iterations: 2, relaxation: 1, averaging: 0 },
    ].map(p => ({ ...POINT, averaging: 1, ...p }));
    result.frontier = [];
    for (const point of points) {
        const r = await searchPoint(engine, point);
        result.frontier.push(r);
        logPoint(r);
    }
    const { sim, snap } = await warmHerd(engine, 16384, 600);
    const stepChain = sim.chain.filter(s => !sim.contacts.stages.includes(s));
    const built = points.map(p => buildContacts(engine, sim.agents.buffer, { capacity: 16384, world: worldOf(sim.def), ...p }));
    const ms = await costs(engine, time, snap, { step: stepChain, ...Object.fromEntries(built.map((c, k) => [`p${k}`, c.stages])) });
    result.frontierCost = { step: ms.step, points: points.map((p, k) => ({ point: p, ms: ms[`p${k}`], cellSize: built[k].layout.cellSize })) };
    log(`   16,384 herd: step ${ms.step.toFixed(3)} ms; ` + points.map((p, k) =>
        `${pointName(p)}: ${ms[`p${k}`].toFixed(3)} ms (${(100 * ms[`p${k}`] / ms.step).toFixed(1)}%)`).join('; '));
    for (const c of built) c.destroy();
    snap.destroy();
    sim.destroy();
}

async function partSearch(engine, name, result, time) {
    result.search = [];
    for (const iterations of [1, 2, 4]) {
        for (const relaxation of [1, 1.25, 1.5, 1.75]) {
            const r = await searchPoint(engine, { ...POINT, iterations, relaxation });
            result.search.push(r);
            logPoint(r);
        }
    }
    // Contact cost per K at a 16,384 flocked state (relaxation does not change the work).
    const { sim, snap } = await warmHerd(engine, 16384, 600);
    const stepChain = sim.chain.filter(s => !sim.contacts.stages.includes(s));
    const built = Object.fromEntries([1, 2, 4].map(k => [`K=${k}`,
        buildContacts(engine, sim.agents.buffer, { capacity: 16384, world: worldOf(sim.def), ...POINT, iterations: k })]));
    const ms = await costs(engine, time, snap, { step: stepChain, ...Object.fromEntries(Object.entries(built).map(([k, c]) => [k, c.stages])) });
    result.searchCost = ms;
    log(`   16,384 herd: step ${ms.step.toFixed(3)} ms; ${[1, 2, 4].map(k => `K=${k} ${ms[`K=${k}`].toFixed(3)} ms (${(100 * ms[`K=${k}`] / ms.step).toFixed(1)}%)`).join(', ')}`);
    for (const c of Object.values(built)) c.destroy();
    snap.destroy();
    sim.destroy();
}

async function partFinish(engine, name, result) {
    result.finish = [];
    for (const damping of [0, 1]) {
        for (const mass of ['area', 'equal']) {
            for (const compliance of [0, 1e-4, 1e-3]) {
                const r = await searchPoint(engine, { ...POINT, damping, mass, compliance });
                result.finish.push(r);
                logPoint(r);
            }
        }
    }
}

async function partExtreme(engine, name, result, time) {
    result.extreme = [];
    // One point: every body spawned at the origin. Contact stages' time per budget, from the snapshot.
    let predicted = 0;
    for (const count of [4096, 16384, 65536]) {
        for (const budget of [16, 32, 64]) {
            const sim = crowdSim(engine, { count, half: 64, spread: 0, contacts: { ...POINT, budget } });
            const snap = snapshot(engine, sim);
            const ms = await costs(engine, time, snap, { contacts: sim.contacts.stages }, 3);
            engine.render(sim.contacts.gridStages);
            await engine.device.queue.onSubmittedWorkDone();
            const skipped = (await readU32(engine.device, sim.contacts.grid.overflow, 1))[0];
            result.extreme.push({ count, budget, contactsMs: ms.contacts, skipped });
            log(`   ${fmt(count)} on one point, B=${budget}: contact stages ${ms.contacts.toFixed(3)} ms, ${fmt(skipped)} entries skipped per build`);
            snap.destroy();
            sim.destroy();
        }
    }
    // Exhaustive scans grow with the square of a pile: small counts only, the rest predicted.
    for (const count of [1024, 4096]) {
        const sim = crowdSim(engine, { count, half: 64, spread: 0, contacts: { ...POINT, budget: Infinity } });
        const snap = snapshot(engine, sim);
        const ms = await costs(engine, time, snap, { contacts: sim.contacts.stages }, 3);
        predicted = ms.contacts * (65536 / count) ** 2;
        result.extreme.push({ count, budget: 'exhaustive', contactsMs: ms.contacts });
        log(`   ${fmt(count)} on one point, exhaustive: contact stages ${ms.contacts.toFixed(3)} ms`);
        snap.destroy();
        sim.destroy();
    }
    result.extremeExhaustive65536PredictedMs = predicted;
    log(`   65,536 on one point, exhaustive: about ${fmt(Math.round(predicted))} ms predicted (not run)`);
    // How a 4,096-body point opens at the point's budget.
    const sim = crowdSim(engine, { count: 4096, half: 64, spread: 0, contacts: POINT });
    const opening = [];
    for (const at of [60, 120, 180, 300]) {
        await run(engine, sim, at - (opening.at(-1)?.step ?? 0));
        opening.push({ step: at, ...overlap(await records(engine.device, sim)) });
    }
    result.opening = opening;
    log(`   4,096 on one point opening at ${pointName(POINT)}: ${opening.map(o => `${o.step}: ${o.over5Pct.toFixed(1)}%`).join(', ')}`);
    sim.destroy();
}

async function partConfirm(engine, name, result, time, adapterName) {
    result.confirm = { point: { ...POINT } };
    // Quality at more counts and in the arenas, with catches against the same runs without contacts.
    for (const [label, make] of [
        ['herd 4,096', c => herdSim(engine, 4096, c)],
        ['herd 16,384', c => herdSim(engine, 16384, c)],
        ['64 arenas', c => arenaSim(engine, 64, c)],
    ]) {
        const row = {};
        for (const [mode, c] of [['none', null], ['contacts', POINT]]) {
            const sim = make(c);
            row[mode] = await herdQuality(engine, sim);
            sim.destroy();
        }
        result.confirm[label] = row;
        log(`   ${label}: max>5% ${row.none.maxOver5.toFixed(2)}% → ${row.contacts.maxOver5.toFixed(2)}%, final worst ` +
            `${row.none.final.worstPct.toFixed(1)}% → ${row.contacts.final.worstPct.toFixed(1)}%, nn ${row.none.final.nearestM.toFixed(2)} → ` +
            `${row.contacts.final.nearestM.toFixed(2)} m, catches ${row.none.catches} → ${row.contacts.catches}`);
    }
    // Whole-step costs with and without contacts, from flocked states.
    result.confirm.cost = [];
    const counts = [4096, 16384, 65536, ...(/nvidia/i.test(adapterName) && !inBrowser ? [262144, 1048576] : [])];
    for (const count of counts) {
        const warm = count > 65536 ? 60 : 600;
        const { sim, snap } = await warmHerd(engine, count, warm);
        const stepChain = sim.chain.filter(s => !sim.contacts.stages.includes(s));
        const ms = await costs(engine, time, snap, { step: stepChain, contacts: sim.contacts.stages, both: sim.chain }, 3);
        result.confirm.cost.push({ count, warm, cellSize: sim.contacts.layout.cellSize, ...ms });
        log(`   ${fmt(count)} herd (${warm} steps in): step ${ms.step.toFixed(3)} ms, contacts ${ms.contacts.toFixed(3)} ms ` +
            `(${(100 * ms.contacts / ms.step).toFixed(1)}%), together ${ms.both.toFixed(3)} ms; cell ${sim.contacts.layout.cellSize.toFixed(2)} m`);
        snap.destroy();
        sim.destroy();
    }
}

// ---------------------------------------------------------------------------------------------
const results = [];
const seen = new Set();
for (const powerPreference of ['high-performance', 'low-power']) {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference });
    if (!adapter) continue;
    const info = adapter.info ?? {};
    const key = [info.vendor, info.architecture, info.device, info.description].join('|');
    if (seen.has(key)) continue;
    seen.add(key);
    const name = [info.vendor, info.architecture].filter(Boolean).join(' ') || powerPreference;
    if (!adapter.features.has('timestamp-query')) { fatal(`S9: ${name} has no timestamp-query`); throw new Error('S9: no timestamp-query'); }
    log(`S9: measuring on ${name}${info.description ? ` (${info.description})` : ''} (${inBrowser ? 'Chrome' : 'Node/Dawn'}), part ${PART}, point ${pointName(POINT)}…`);
    const device = await adapter.requestDevice({ requiredFeatures: ['timestamp-query'] });
    const engine = new NullGraph();
    const init = await engine.init(null, { device, width: 64, height: 64 });
    if (!init.success) throw new Error(`S9: engine: ${init.error}`);
    const time = timer(engine), result = { adapter: name, host: inBrowser ? 'chrome' : 'node', part: PART, point: { ...POINT } };
    const t0 = performance.now();
    if (want('checks')) { log(`\n== S9 checks · ${name}`); await partChecks(engine, name, result); }
    if (want('grid')) { log(`\n== S9 grid · ${name} — contact stages' GPU ms from one flocked state (K=${POINT.iterations})`); await partGrid(engine, name, result, time); }
    if (want('search')) { log(`\n== S9 search · ${name} — passes × relaxation`); await partSearch(engine, name, result, time); }
    if (want('frontier')) { log(`\n== S9 frontier · ${name} — a larger push, more passes, weaker averaging`); await partFrontier(engine, name, result, time); }
    if (want('finish')) { log(`\n== S9 finish · ${name} — damping × mass × compliance at K=${POINT.iterations}, ω=${POINT.relaxation}`); await partFinish(engine, name, result); }
    if (want('extreme')) { log(`\n== S9 extreme · ${name} — bodies on one point`); await partExtreme(engine, name, result, time); }
    if (want('confirm')) { log(`\n== S9 confirm · ${name} — ${pointName(POINT)}`); await partConfirm(engine, name, result, time, name); }
    result.seconds = (performance.now() - t0) / 1000;
    check(`${name}: part ${PART} ran to the end (${result.seconds.toFixed(0)} s)`, true);
    results.push(result);
    device.destroy();
}
if (!results.length) { fatal('No WebGPU adapter'); throw new Error('No WebGPU adapter'); }
log(`\nS9 JSON ${JSON.stringify(results)}`);
globalThis.__S9_MEASUREMENTS__ = results;
if (!inBrowser) {
    const { mkdirSync, writeFileSync } = await import('node:fs');
    const out = new URL('./out/', import.meta.url);
    mkdirSync(out, { recursive: true });
    const tag = results.map(r => r.adapter.split(' ')[0]).join('-');
    writeFileSync(new URL(`node-${PART}-${tag}.json`, out), JSON.stringify(results, null, 2));
}
await finish(`S9: part ${PART} measured on ${results.length} adapter(s).`);
log('S9 done');
