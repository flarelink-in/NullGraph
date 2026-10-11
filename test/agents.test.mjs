// Headless agents tests (docs/agents-design.md §10) — C1 step 1: the record layout (U1), the
// sensor mirrors (U4), steering (U5), the derived layout (U6), the fixed step (U7), and the
// construction-time errors (U14, the ones step 1 owns). No device: these are the DEFINITIONS the
// GPU stages of later steps are checked against.

const A = await import('../dist/agents.js');
const { Sensors, Actions } = A;

let failures = 0;
async function check(name, fn) {
    try { await fn(); console.log(`  ok    ${name}`); }
    catch (e) { failures++; console.log(`  FAIL  ${name}\n        ${e.message}`); }
}
function assert(v, m) { if (!v) throw new Error(m); }
function near(a, b, eps, m) { if (!(Math.abs(a - b) <= eps)) throw new Error(`${m}: expected ~${b}, got ${a}`); }
function throws(fn, pattern, label) {
    let msg = '';
    try { fn(); } catch (e) { msg = e.message; }
    if (!pattern.test(msg)) throw new Error(`${label}: expected ${pattern}, got "${msg}"`);
}

const PREY = { name: 'prey', maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 4, radius: 0.4, mesh: 2, boundsRadius: 0.8 };
const WOLF = { name: 'wolf', maxSpeed: 4.5, maxTurn: 3, accel: 5, senseRadius: 4, radius: 0.6, mesh: 3 };
/** The design doc's example (§7.1): obsDim 24, actDim 2. */
const opts = (over = {}) => ({
    capacity: 64,
    world: { min: [-20, -20], max: [20, 20] },
    cellSize: 4,
    species: [PREY, WOLF],
    sensors: [Sensors.self(), Sensors.nearest({ species: 'wolf', k: 1 }), Sensors.nearest({ species: 'prey', k: 3 }), Sensors.bounds()],
    ...over,
});
const fresh = (over) => {
    const def = A.defineAgents(opts(over));
    return { def, rec: new A.AgentRecords(def) };
};
const C = A.defineAgents(opts()).constants;
const W0 = C.OBS_WOLF_0, P0 = C.OBS_PREY_0;
const F = A.AGENT_FLOATS;

// ---------------------------------------------------------------------------
// U1 — the record
// ---------------------------------------------------------------------------

await check('U1: record offsets, 64 bytes, floats 0–3 = P1 body record (position, radius)', () => {
    const want = { POSITION: 0, RADIUS: 3, VELOCITY: 4, HEADING: 7, ENERGY: 8, AGE: 9, SPEED_SCALE: 10, USER0: 11, SPECIES: 12, FLAGS: 13, GENOME: 14, USER1: 15 };
    for (const [k, v] of Object.entries(want)) assert(A[`AGENT_${k}_OFFSET`] === v, `AGENT_${k}_OFFSET = ${A[`AGENT_${k}_OFFSET`]}, want ${v}`);
    assert(A.AGENT_FLOATS === 16 && A.AGENT_BYTES === 64, 'record size');
    // physics P1 §4.2: position at floats 0–2, world radius at 3 — one vec4 load for the grid.
    assert(A.AGENT_POSITION_OFFSET === 0 && A.AGENT_RADIUS_OFFSET === 3, 'P1-compatible first vec4');
    const fields = [...A.AGENT_STRUCT_WGSL.matchAll(/^\s+(\w+): (vec4<[fu]32>),$/gm)].map(m => `${m[1]}:${m[2]}`);
    assert(fields.slice(0, 4).join() === 'posRadius:vec4<f32>,velHeading:vec4<f32>,state:vec4<f32>,ids:vec4<u32>', `Agent struct: ${fields}`);
    assert(!A.AGENT_STRUCT_WGSL.includes('`'), 'a backtick in WGSL ends the template string');
});

await check('U1b: the u32 view inherits byteOffset and length; guards around an adopted view stay untouched', () => {
    const def = A.defineAgents(opts());
    const GUARD = 8, n = def.capacity * F;
    const backing = new Float32Array(GUARD + n + GUARD).fill(777);
    const view = new Float32Array(backing.buffer, GUARD * 4, n);
    view.fill(0);
    const rec = new A.AgentRecords(def, view, 10);
    assert(rec.u32.byteOffset === view.byteOffset && rec.u32.length === view.length, 'u32 view offset/length');
    assert(rec.dirtyFirst === 0 && rec.dirtyLast === 9, `adopted count marks 0..9 dirty, got ${rec.dirtyFirst}..${rec.dirtyLast}`);
    rec.clearDirty();
    rec.spawn(1, { position: [1, 2, 3], species: 'wolf', appFlags: 5, genome: 9, user1: 42 });
    const b = 1 * F;
    assert(rec.u32[b + 12] === 1 && rec.u32[b + 13] === (A.AGENT_FLAG_ALIVE | (5 << 8)) && rec.u32[b + 14] === 9 && rec.u32[b + 15] === 42,
        `ids ${rec.u32.slice(b + 12, b + 16)}`);
    assert(view[b + 3] === Math.fround(0.6), 'radius defaults to the species radius');
    for (let k = 0; k < GUARD; k++) assert(backing[k] === 777 && backing[GUARD + n + k] === 777, `guard word ${k} overwritten`);
    rec.spawn(5, { position: [0, 0, 0], species: 0 });
    assert(rec.dirtyFirst === 1 && rec.dirtyLast === 5, 'dirty range widens');
    rec.clearDirty();
    assert(rec.dirtyFirst === -1 && rec.dirtyLast === -1, 'clean after clearDirty');
});

await check('U1c: spawn keeps velocity along the heading, wraps the heading, kill clears only the alive bit', () => {
    const { rec } = fresh();
    rec.spawn(0, { position: [0, 0, 0], species: 'prey', heading: Math.PI / 2, speed: 2 });
    near(rec.data[4], 2, 1e-6, 'vx'); near(rec.data[6], 0, 1e-6, 'vz');
    rec.spawn(1, { position: [0, 0, 0], species: 'prey', heading: 3 * Math.PI });
    near(rec.data[F + 7], -Math.PI, 1e-6, 'heading 3π wraps to −π');
    rec.spawn(2, { position: [0, 0, 0], species: 'prey', appFlags: 3 });
    rec.kill(2);
    assert(!rec.isAlive(2) && rec.u32[2 * F + 13] === 3 << 8, 'kill clears bit 0 and keeps app bits');
    const fz = (rec.spawn(3, { position: [0, 0, 0], species: 'prey', speed: 2, frozen: true }), rec.data.slice(3 * F + 4, 3 * F + 7));
    assert(fz.every(v => v === 0) && (rec.u32[3 * F + 13] & A.AGENT_FLAG_FROZEN), 'frozen spawn: zero velocity, frozen bit');
    rec.spawn(4, { position: [0, 0, 0], species: 'prey', speed: 9, speedScale: 0.5 });
    near(Math.hypot(rec.data[4 * F + 4], rec.data[4 * F + 6]), 1.5, 1e-6, 'spawn speed clamps to maxSpeed × speedScale');
});

await check('U1d: packSpecies — 16 rows of 8 words, u32 mesh fields as raw bits, hidden mesh by default', () => {
    const def = A.defineAgents(opts({ species: [PREY, WOLF, { ...WOLF, name: 'ghost', mesh: undefined }] }));
    const t = A.packSpecies(def);
    const u = new Uint32Array(t.buffer);
    assert(t.length === 128, `length ${t.length}`);
    assert([t[8], t[9], t[10], t[11]].join() === '4.5,3,5,4' && u[12] === 3 && u[13] === 3 && u[14] === 3 && t[15] === 0, 'wolf row');
    near(t[7], 0.8, 1e-6, 'prey boundsRadius');
    assert(u[16 + 4] === A.HIDDEN_MESH && u[16 + 5] === A.HIDDEN_MESH, 'no mesh → hidden');
    assert(t.subarray(24).every(v => v === 0), 'undeclared rows are zero');
});

// ---------------------------------------------------------------------------
// U4 — sensor mirrors on hand-built scenes (prey 0 at the origin facing +Z unless stated)
// ---------------------------------------------------------------------------

const row = (def, rec, i = 0) => A.senseAgent(def, rec, i);
const block = (r, at, w = 5) => Array.from(r.subarray(at, at + w), v => Math.round(v * 1e5) / 1e5);

await check('U4a: a wolf 2 m dead ahead reads (side 0, forward 0.5); 2 m to +X reads (side 0.5, forward 0)', () => {
    const { def, rec } = fresh();
    rec.spawn(0, { position: [0, 0, 0], species: 'prey' });
    rec.spawn(1, { position: [0, 0, 2], species: 'wolf' });
    assert(block(row(def, rec), W0).join() === '0,0.5,0,0,1', `ahead: ${block(row(def, rec), W0)}`);
    rec.spawn(1, { position: [2, 0, 0], species: 'wolf' });
    assert(block(row(def, rec), W0).join() === '0.5,0,0,0,1', `+X: ${block(row(def, rec), W0)}`);
});

await check('U4b: the frame turns with the agent — facing +X, a wolf at +X is dead ahead', () => {
    const { def, rec } = fresh();
    rec.spawn(0, { position: [0, 0, 0], species: 'prey', heading: Math.PI / 2 });
    rec.spawn(1, { position: [2, 0, 0], species: 'wolf' });
    assert(block(row(def, rec), W0).join() === '0,0.5,0,0,1', `${block(row(def, rec), W0)}`);
});

await check('U4c: relative velocity — a wolf closing at 3 m/s reads velForward −3 / (2·3) = −0.5', () => {
    const { def, rec } = fresh();
    rec.spawn(0, { position: [0, 0, 0], species: 'prey' });
    rec.spawn(1, { position: [0, 0, 2], species: 'wolf', heading: Math.PI, speed: 3 });
    assert(block(row(def, rec), W0).join() === '0,0.5,0,-0.5,1', `${block(row(def, rec), W0)}`);
});

await check('U4d: k nearest, nearest first, an exact distance tie goes to the LOWER index', () => {
    const { def, rec } = fresh();
    rec.spawn(0, { position: [0, 0, 0], species: 'prey' });
    rec.spawn(3, { position: [1, 0, 0], species: 'prey' });
    rec.spawn(2, { position: [-1, 0, 0], species: 'prey' });
    rec.spawn(4, { position: [0, 0, 3], species: 'prey' });
    rec.spawn(5, { position: [0, 0, 3.5], species: 'prey' });   // 4th nearest — k = 3 drops it
    const r = row(def, rec);
    const sides = [0, 1, 2].map(k => block(r, P0 + 5 * k));
    assert(sides[0].join() === '-0.25,0,0,0,1' && sides[1].join() === '0.25,0,0,0,1' && sides[2].join() === '0,0.75,0,0,1',
        `got ${sides.map(s => `[${s}]`).join(' ')} — index 2 (at −X) must precede index 3 (at +X)`);
    const out = new Int32Array(8);
    assert(A.nearestNeighbours(def, rec, 0, 1, 4, 3, out) === 3 && [...out.subarray(0, 3)].join() === '2,3,4', `indices ${out}`);
});

await check('U4e: range is inclusive at senseRadius; dead agents, self and other species are invisible', () => {
    const { def, rec } = fresh();
    rec.spawn(0, { position: [0, 0, 0], species: 'prey' });
    rec.spawn(1, { position: [0, 0, 4], species: 'wolf' });
    assert(block(row(def, rec), W0)[4] === 1, 'a wolf at exactly 4 m is seen');
    rec.spawn(1, { position: [0, 0, 4.001], species: 'wolf' });
    assert(block(row(def, rec), W0).join() === '0,0,0,0,0', 'a wolf past 4 m is not');
    rec.spawn(1, { position: [0, 0, 1], species: 'wolf' });
    rec.kill(1);
    assert(block(row(def, rec), W0)[4] === 0, 'dead wolf invisible');
    const r = row(def, rec);
    for (let k = 0; k < 3; k++) assert(r[P0 + 5 * k + 4] === 0, 'no prey but itself — and itself excluded');
    rec.spawn(1, { position: [0, 0, 1], species: 'wolf' });
    assert(row(def, rec)[P0 + 4] === 0, 'the prey sensor ignores wolves');
});

await check('U4f: self reads speed / (maxSpeed × speedScale) and energy, both clamped to [−1, 1]', () => {
    const { def, rec } = fresh();
    rec.spawn(0, { position: [0, 0, 0], species: 'prey', speed: 1.5, energy: 0.25 });
    assert(block(row(def, rec), C.OBS_SELF, 2).join() === '0.5,0.25', `${block(row(def, rec), 0, 2)}`);
    rec.spawn(0, { position: [0, 0, 0], species: 'prey', speed: 1.5, speedScale: 2, energy: 7 });
    assert(block(row(def, rec), C.OBS_SELF, 2).join() === '0.25,1', `${block(row(def, rec), 0, 2)}`);
});

await check('U4g: bounds — 1 m from +X reads 0.75 toward it; facing the wall it is ahead; centre reads zero; corner ties are ordered', () => {
    const { def, rec } = fresh();
    rec.spawn(0, { position: [19, 0, 0], species: 'prey' });
    assert(block(row(def, rec), C.OBS_BOUNDS, 2).join() === '0.75,0', `facing +Z: ${block(row(def, rec), C.OBS_BOUNDS, 2)}`);
    rec.spawn(0, { position: [19, 0, 0], species: 'prey', heading: Math.PI / 2 });
    assert(block(row(def, rec), C.OBS_BOUNDS, 2).join() === '0,0.75', `facing +X: ${block(row(def, rec), C.OBS_BOUNDS, 2)}`);
    rec.spawn(0, { position: [0, 0, 0], species: 'prey' });
    assert(block(row(def, rec), C.OBS_BOUNDS, 2).join() === '0,0', 'centre');
    rec.spawn(0, { position: [18, 0, 18], species: 'prey' });
    assert(block(row(def, rec), C.OBS_BOUNDS, 2).join() === '0.5,0', 'equidistant corner: +X precedes +Z');
});

await check('U4h: a dead agent\'s row is zeros; custom sensors use their mirror, and refuse without one', () => {
    const { def, rec } = fresh();
    rec.spawn(0, { position: [0, 0, 0], species: 'prey', speed: 2 });
    rec.kill(0);
    assert(row(def, rec).every(v => v === 0), 'dead row');
    const custom = A.defineAgents(opts({ sensors: [Sensors.custom({ name: 'clock', width: 2, wgsl: '', mirror: (d, u, i, out, o) => { out[o] = i; out[o + 1] = 7; } })] }));
    const r2 = new A.AgentRecords(custom);
    r2.spawn(3, { position: [0, 0, 0], species: 'prey' });
    assert([...A.senseAgent(custom, r2, 3)].join() === '3,7', 'custom mirror');
    const bare = A.defineAgents(opts({ sensors: [Sensors.custom({ name: 'clock', width: 2, wgsl: '' })] }));
    const r3 = new A.AgentRecords(bare);
    r3.spawn(0, { position: [0, 0, 0], species: 'prey' });
    throws(() => A.senseAgent(bare, r3, 0), /custom sensor "clock" has no CPU mirror/, 'mirrorless custom sensor');
});

// ---------------------------------------------------------------------------
// U5 — steering
// ---------------------------------------------------------------------------

await check('U5a: steer-toward closes the loop — turning by the sensed side ends FACING the target (catches a flipped frame)', () => {
    const { def, rec } = fresh();
    rec.spawn(0, { position: [0, 0, 0], species: 'prey' });
    rec.spawn(1, { position: [3, 0, -1], species: 'wolf', frozen: true });   // behind-right of a +Z-facing prey
    const obs = new Float32Array(def.obsDim), act = new Float32Array(2);
    for (let s = 0; s < 400; s++) {
        A.senseAgent(def, rec, 0, obs);
        act[0] = Math.max(-1, Math.min(1, 4 * obs[W0 + C.NEAR_SIDE]));
        act[1] = -1;                                                         // target speed 0: turn in place
        A.actAgent(def, rec, 0, act);
    }
    A.senseAgent(def, rec, 0, obs);
    assert(Math.abs(obs[W0]) < 0.01 && obs[W0 + 1] > 0.7, `side ${obs[W0]}, forward ${obs[W0 + 1]} — ` +
        'a flipped side axis or turn sign settles with the target BEHIND (forward < 0)');
    near(rec.data[0], 0, 1e-6, 'turning in place does not move');
});

await check('U5b: throttle maps [−1, 1] to [0, max], approached at accel; turn +1 for 0.5 s yaws 2 rad', () => {
    const { def, rec } = fresh();
    rec.spawn(0, { position: [0, 0, -15], species: 'prey' });
    const speed = () => Math.hypot(rec.data[4], rec.data[6]);
    for (let s = 0; s < 15; s++) A.actAgent(def, rec, 0, [0, 1]);
    near(speed(), 1.5, 1e-5, 'after 0.25 s at accel 6');
    for (let s = 0; s < 30; s++) A.actAgent(def, rec, 0, [0, 1]);
    near(speed(), 3, 1e-5, 'capped at maxSpeed');
    for (let s = 0; s < 60; s++) A.actAgent(def, rec, 0, [0, 0]);
    near(speed(), 1.5, 1e-5, 'throttle 0 → half speed');
    for (let s = 0; s < 60; s++) A.actAgent(def, rec, 0, [0, -1]);
    near(speed(), 0, 1e-5, 'throttle −1 → stop');
    const h0 = rec.data[7];
    for (let s = 0; s < 30; s++) A.actAgent(def, rec, 0, [1, -1]);
    near(A.wrapAngle(rec.data[7] - h0), 2, 1e-4, 'yaw after 0.5 s at maxTurn 4');
    A.actAgent(def, rec, 0, [5, -1]);
    near(A.wrapAngle(rec.data[7] - h0 - 2), 4 / 60, 1e-4, 'turn 5 is clamped to 1');
});

await check('U5c: a non-finite action row stops the agent and never corrupts the record', () => {
    for (const bad of [[NaN, 1], [0.3, Infinity], [-Infinity, NaN]]) {
        const { def, rec } = fresh();
        rec.spawn(0, { position: [0, 0, 0], species: 'prey', speed: 3, heading: 1 });
        const h0 = rec.data[7];
        let last = 3;
        for (let s = 0; s < 40; s++) {
            A.actAgent(def, rec, 0, bad);
            const sp = Math.hypot(rec.data[4], rec.data[6]);
            assert(sp <= last + 1e-6, `speed rose under [${bad}]`);
            last = sp;
        }
        near(last, 0, 1e-6, `[${bad}] stops`);
        near(rec.data[7], h0, 1e-6, `[${bad}] does not turn`);
        assert(rec.data.subarray(0, F).every(Number.isFinite), `[${bad}] left a non-finite value`);
    }
});

await check('U5d: walls clamp the centre (inset by radius) and reflect a heading that drove into them', () => {
    const { def, rec } = fresh();
    rec.spawn(0, { position: [19.5, 0, 0], species: 'prey', heading: Math.PI / 2, speed: 3 });
    for (let s = 0; s < 3; s++) A.actAgent(def, rec, 0, [0, 1]);
    near(rec.data[0], 19.6, 1e-5, 'x clamped to max − radius');
    near(rec.data[7], -Math.PI / 2, 1e-5, 'heading reflected to −X');
    assert(rec.data[4] < 0, 'velocity now points away from the wall');
    rec.spawn(1, { position: [0, 0, -19.5], species: 'prey', heading: Math.PI, speed: 3 });
    for (let s = 0; s < 3; s++) A.actAgent(def, rec, 1, [0, 1]);
    near(rec.data[F + 2], -19.6, 1e-5, 'z clamped to min + radius');
    near(Math.abs(rec.data[F + 7]), 0, 1e-5, 'heading reflected to +Z');
});

await check('U5e: frozen agents only age; dead agents are byte-identical after a step', () => {
    const { def, rec } = fresh();
    rec.spawn(0, { position: [1, 0, 1], species: 'prey', speed: 2, frozen: true });
    rec.spawn(1, { position: [2, 0, 2], species: 'prey', speed: 2 });
    rec.kill(1);
    const dead = rec.data.slice(F, 2 * F);
    for (let s = 0; s < 10; s++) { A.actAgent(def, rec, 0, [1, 1]); A.actAgent(def, rec, 1, [1, 1]); }
    assert(rec.data[0] === 1 && rec.data[2] === 1 && rec.data[4] === 0 && rec.data[6] === 0, 'frozen moved');
    near(rec.data[9], 10 / 60, 1e-6, 'frozen still ages');
    assert(rec.data.subarray(F, 2 * F).every((v, k) => Object.is(v, dead[k])), 'dead record changed');
});

await check('U5f: 600 random steps × 48 agents — in bounds, under max speed, turn-rate bound, velocity on heading, finite', () => {
    const { def, rec } = fresh({ capacity: 48 });
    let seed = 12345;
    const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
    for (let i = 0; i < 48; i++) {
        rec.spawn(i, { position: [rnd() * 38 - 19, 0, rnd() * 38 - 19], species: i % 2, heading: rnd() * 7 - 3.5, speed: rnd() * 3, speedScale: 0.5 + rnd() });
    }
    const act = new Float32Array(48 * 2);
    for (let s = 0; s < 600; s++) {
        for (let k = 0; k < act.length; k++) act[k] = rnd() < 0.03 ? NaN : rnd() * 3 - 1.5;
        const before = rec.data.slice();
        A.stepAll(def, rec, act);
        for (let i = 0; i < 48; i++) {
            const b = i * F, sp = def.species[rec.u32[b + 12]], rad = rec.data[b + 3];
            const x = rec.data[b], z = rec.data[b + 2], h = rec.data[b + 7];
            const speed = Math.hypot(rec.data[b + 4], rec.data[b + 6]);
            assert(rec.data.subarray(b, b + 12).every(Number.isFinite), `agent ${i} non-finite at step ${s}`);
            assert(x >= -20 + rad - 1e-5 && x <= 20 - rad + 1e-5 && z >= -20 + rad - 1e-5 && z <= 20 - rad + 1e-5, `agent ${i} left the world`);
            assert(speed <= sp.maxSpeed * rec.data[b + 10] + 1e-5, `agent ${i} speed ${speed}`);
            assert(h >= -Math.PI - 1e-6 && h < Math.PI + 1e-6, `heading ${h} unwrapped`);
            near(rec.data[b + 4], Math.sin(h) * speed, 1e-4, `agent ${i} vx off heading`);
            near(rec.data[b + 6], Math.cos(h) * speed, 1e-4, `agent ${i} vz off heading`);
            const atWall = Math.abs(Math.abs(x) - (20 - rad)) < 1e-4 || Math.abs(Math.abs(z) - (20 - rad)) < 1e-4;
            if (!atWall) {
                const dh = Math.abs(A.wrapAngle(h - before[b + 7]));
                assert(dh <= sp.maxTurn * def.dt + 1e-4, `agent ${i} yawed ${dh} rad in one step`);
            }
        }
    }
});

// ---------------------------------------------------------------------------
// U6 — the derived layout
// ---------------------------------------------------------------------------

await check('U6a: obsDim, actDim, named offsets and slot names derive from the declarations (doc example: 24 → 2)', () => {
    const def = A.defineAgents(opts());
    assert(def.obsDim === 24 && def.actDim === 2, `${def.obsDim}→${def.actDim}`);
    const c = def.constants;
    const got = ['OBS_SELF', 'OBS_WOLF_0', 'OBS_PREY_0', 'OBS_PREY_1', 'OBS_PREY_2', 'OBS_BOUNDS', 'OBS_DIM', 'ACT_DIM', 'ACT_TURN', 'ACT_THROTTLE'].map(k => c[k]).join();
    assert(got === '0,2,7,12,17,22,24,2,0,1', `constants ${got}`);
    assert(def.obsNames.length === 24 && def.obsNames[2] === 'wolf[0].side' && def.obsNames[6] === 'wolf[0].present' &&
        def.obsNames[17] === 'prey[2].side' && def.obsNames[23] === 'bounds.forward', `names ${def.obsNames}`);
    assert(def.actNames.join() === 'turn,throttle', 'action names');
    assert(def.nearestMasks.join() === '2,1', `masks ${def.nearestMasks}`);
    assert(def.grid.cellsX === 10 && def.grid.cellsZ === 10 && def.grid.cellTotal === 100, 'grid 40 m / 4 m');
    const any = A.defineAgents(opts({ sensors: [Sensors.nearest({ k: 2 }), Sensors.nearest({ species: ['wolf', 'prey'], k: 1, name: 'both' })] }));
    assert(any.nearestMasks.join() === '3,3' && any.constants.OBS_ANY_1 === 5 && any.constants.OBS_BOTH_0 === 10, 'any-species and lists');
});

await check('U6b: constantsWGSL is one valid const per offset, with no backtick', () => {
    const text = A.constantsWGSL(A.defineAgents(opts()));
    const lines = text.trim().split('\n');
    assert(lines.every(l => /^const [A-Z][A-Z0-9_]* = \d+u;$/.test(l)), `bad line in ${text}`);
    assert(text.includes('const OBS_PREY_2 = 17u;') && !text.includes('`'), 'content');
});

await check('U6c: custom sensors and actions add their widths; a brain of other dimensions is refused', () => {
    const def = A.defineAgents(opts({
        sensors: [Sensors.self(), Sensors.custom({ name: 'eyes', width: 16, wgsl: '' })],
        actions: Actions.custom({ name: 'move', width: 3, wgsl: '' }),
    }));
    assert(def.obsDim === 18 && def.actDim === 3 && def.constants.OBS_EYES === 2 && !('ACT_TURN' in def.constants), `${def.obsDim}→${def.actDim}`);
    A.assertBrainDims(def, 18, 3);
    throws(() => A.assertBrainDims(def, 18, 2), /the brain maps 18→2; the layout is 18→3\./, 'brain dims');
});

// ---------------------------------------------------------------------------
// U7 — the fixed step
// ---------------------------------------------------------------------------

await check('U7a: steps per frame, the clamp only under real debt, the tab-switch guard, bad frame times', () => {
    const f = new A.FixedStep(1 / 64, 3, 0.25);
    for (let k = 0; k < 10; k++) assert(f.advance(1 / 64) === 1, 'one step per 1/64 frame');
    assert(f.advance(1 / 128) === 0 && f.advance(1 / 128) === 1, 'half frames accumulate');
    assert(f.advance(3 / 64) === 3 && f.clampedFrames === 0 && f.accumulator === 0,
        `exactly maxSteps of time is not a clamp (physics P1's pseudocode counts one): clamped ${f.clampedFrames}`);
    assert(f.advance(5 / 64) === 3 && f.clampedFrames === 1 && f.accumulator === 0, 'real debt is dropped, not carried');
    assert(f.advance(1 / 64) === 1, 'the dropped debt does not come back');
    assert(f.advance(10) === 3 && f.clampedFrames === 2, 'a 10 s tab switch runs maxSteps and is counted');
    for (const bad of [NaN, -1, 0, -Infinity]) assert(f.advance(bad) === 0, `frame time ${bad} → 0 steps`);
    assert(f.totalSteps === 10 + 1 + 3 + 3 + 1 + 3, `totalSteps ${f.totalSteps}`);
    throws(() => new A.FixedStep(0), /FixedStep needs dt > 0/, 'dt 0');
});

await check('U7b: advance, senseAll and stepAll allocate no typed arrays once their outputs exist (counted constructors)', () => {
    const { def, rec } = fresh({ capacity: 32 });
    for (let i = 0; i < 32; i++) rec.spawn(i, { position: [i - 16, 0, (i % 5) - 2], species: i % 2 });
    const obs = new Float32Array(def.capacity * def.obsDim), act = new Float32Array(def.capacity * 2).fill(0.5);
    const f = new A.FixedStep();
    const names = ['Float32Array', 'Float64Array', 'Int32Array', 'Uint32Array', 'Uint8Array', 'ArrayBuffer'];
    const saved = names.map(n => globalThis[n]);
    let made = 0;
    names.forEach((n, k) => { globalThis[n] = new Proxy(saved[k], { construct(t, args) { made++; return Reflect.construct(t, args); } }); });
    try {
        for (let k = 0; k < 200; k++) { f.advance(1 / 60); A.senseAll(def, rec, obs); A.stepAll(def, rec, act); }
    } finally {
        names.forEach((n, k) => { globalThis[n] = saved[k]; });
    }
    assert(made === 0, `${made} typed arrays constructed`);
});

// ---------------------------------------------------------------------------
// U14 — every construction-time refusal names the offender (the ones step 1 owns)
// ---------------------------------------------------------------------------

await check('U14: errors name the offender and the way out', () => {
    const t = (over, re, label) => throws(() => A.defineAgents(opts(over)), re, label);
    t({ capacity: 0 }, /capacity 0 must be a positive integer\./, 'capacity 0');
    t({ capacity: 1.5 }, /capacity 1\.5 must be a positive integer\./, 'capacity 1.5');
    t({ world: { min: [0, 0], max: [0, 5] } }, /world min must be finite and below max/, 'flat world');
    t({ cellSize: 0 }, /cellSize 0 must be positive\./, 'cellSize');
    t({ species: [{ ...PREY, senseRadius: 5 }] }, /species "prey" senses 5 m but cells are 4 m — a neighbour two cells away would be missed\. Raise cellSize to ≥ 5\./, 'sense radius');
    t({ world: { min: [-2000, -2000], max: [2000, 2000] } }, /1000×1000 cells exceeds maxCells 262144; raise cellSize or maxCells\./, 'grid size');
    t({ sensors: [Sensors.nearest({ species: 'bear' })] }, /species "bear" is not declared\./, 'unknown species');
    t({ species: [PREY, PREY] }, /species "prey" is declared twice\./, 'duplicate species');
    t({ species: Array.from({ length: 17 }, (_, k) => ({ ...PREY, name: `s${k}` })) }, /declare 1–16 species; got 17\./, 'species count');
    t({ species: [{ ...PREY, maxSpeed: -1 }] }, /species "prey" maxSpeed -1 must be positive\./, 'maxSpeed');
    t({ sensors: [Sensors.self(), Sensors.self()] }, /two sensors are both named "self"; pass a distinct name\./, 'sensor names');
    t({ sensors: [Sensors.nearest({ k: 9 })] }, /sensor "any" k 9 must be 1–8\./, 'k');
    t({ sensors: [Sensors.custom({ name: '9lives', width: 1, wgsl: '' })] }, /sensor name "9lives" must be an identifier/, 'identifier');
    t({ sensors: [Sensors.self(), Sensors.custom({ name: 'big', width: 300, wgsl: '' })] }, /the sensors produce 302 observation floats; the limit is 256\./, 'obsDim');
    t({ sensors: [] }, /declare at least one sensor/, 'no sensors');
    t({ sensors: [Sensors.custom({ name: 'dim', width: 1, wgsl: '' })] }, /both produce the WGSL constant OBS_DIM/, 'constant collision');
    t({ actions: Actions.custom({ name: 'm', width: 17, wgsl: '' }) }, /actions "m" width 17 must be 1–16\./, 'actDim');
    t({ dt: 0 }, /dt 0 must be positive\./, 'dt');
    t({ maxFrameDelta: 0.001 }, /maxFrameDelta 0\.001 must be at least dt/, 'maxFrameDelta');
    const def = A.defineAgents(opts());
    const rec = new A.AgentRecords(def);
    throws(() => rec.spawn(64, { position: [0, 0, 0], species: 'prey' }), /index 64 is outside capacity 64\./, 'spawn index');
    throws(() => rec.spawn(3, { position: [NaN, 0, 0], species: 'prey' }), /agent 3 has a non-finite position, heading or speed\./, 'spawn NaN');
    throws(() => rec.spawn(3, { position: [0, 0, 0], species: 7 }), /species index 7 is not declared\./, 'spawn species index');
    throws(() => rec.spawn(3, { position: [0, 0, 0], species: 'prey', speed: -1 }), /agent 3 speed -1 is negative; turn the heading instead\./, 'negative speed');
    throws(() => rec.spawn(3, { position: [0, 0, 0], species: 'prey', appFlags: 1 << 24 }), /appFlags 16777216 must fit in 24 bits\./, 'appFlags');
    throws(() => new A.AgentRecords(def, new Float32Array(10)), /adopted data holds 10 floats; capacity 64 needs 1024\./, 'short adoption');
});

console.log(failures === 0 ? '\nAll agents tests passed.' : `\n${failures} test(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
