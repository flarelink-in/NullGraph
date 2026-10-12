// Step 4, phase 4.4 (docs/agents-gpu-loop-design.md §9.6): inspect(i) — the staging layout, the
// decoder (U20) and the request queue. The mock device keeps real bytes and performs the copies, so
// a wrong offset reads the wrong row instead of passing silently.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as A from '../dist/agents.js';

globalThis.GPUBufferUsage = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };
globalThis.GPUMapMode = { READ: 1 };
const NO_AGENT = 0xFFFFFFFF;

/** Herd & Wolves' layout: self (2) + wolf k=1 (5) + prey k=3 (15) + bounds (2) = 24 obs floats. */
const herd = (capacity = 16) => A.defineAgents({ capacity, world: { min: [-8, -8], max: [8, 8] }, cellSize: 8,
    species: [
        { name: 'prey', maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 8, radius: 0.4 },
        { name: 'wolf', maxSpeed: 4, maxTurn: 3, accel: 5, senseRadius: 8, radius: 0.6 },
    ],
    sensors: [A.Sensors.self(), A.Sensors.nearest({ name: 'wolf', species: ['wolf'], k: 1 }),
        A.Sensors.nearest({ name: 'prey', species: ['prey'], k: 3 }), A.Sensors.bounds()] });

/** A device whose buffers hold bytes; submitted copies happen; mapAsync waits on `gate` when set. */
function mock() {
    const buffers = [], copies = [], log = [];
    let gate = null;
    const device = {
        limits: { maxStorageBufferBindingSize: 128 * 1024 * 1024, maxBufferSize: 256 * 1024 * 1024,
            maxComputeWorkgroupsPerDimension: 65535, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256,
            maxStorageBuffersPerShaderStage: 8 },
        createBuffer(o) {
            const b = { ...o, bytes: new ArrayBuffer(o.size), destroyed: false,
                destroy() { this.destroyed = true; },
                async mapAsync() {
                    log.push(`map ${this.label}`);
                    if (gate) await gate;
                    if (this.destroyed) throw new Error('AbortError: buffer destroyed');
                },
                getMappedRange() { return this.bytes; }, unmap() {} };
            buffers.push(b);
            return b;
        },
        createCommandEncoder() {
            const pending = [];
            return { copyBufferToBuffer(src, so, dst, d, size) { pending.push([src, so, dst, d, size]); }, finish: () => pending };
        },
        queue: {
            writeBuffer(b, offset, data, dataOffset = 0, size) {
                const src = ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset + dataOffset, size ?? data.byteLength)
                    : new Uint8Array(data, dataOffset, size ?? data.byteLength - dataOffset);
                new Uint8Array(b.bytes, offset, src.length).set(src);
            },
            submit(lists) {
                for (const pending of lists) for (const [src, so, dst, d, size] of pending) {
                    copies.push({ src: src.label, srcOffset: so, dst: dst.label, dstOffset: d, size });
                    new Uint8Array(dst.bytes, d, size).set(new Uint8Array(src.bytes, so, size));
                }
                log.push('submit');
            },
            async onSubmittedWorkDone() {},
        },
    };
    const engine = {
        device,
        createComputeStage(o) { return { ...o, enabled: true, preClearBuffers: [] }; },
        setComputePassBindGroup() {},
        render() {},
    };
    return { engine, buffers, copies, log, setGate(p) { gate = p; } };
}
function system(m, def = herd()) {
    const s = new A.AgentSystem(m.engine, def);
    s.setBrain(new A.RuleBrain('fn think(i: u32, species: u32, tick: u32) {}'));
    return s;
}
/** Fill a GPU-only buffer's row `i` with `values` (the mock's stand-in for a stage writing it). */
function seed(buffer, i, words, values, asU32 = false) {
    const view = asU32 ? new Uint32Array(buffer.bytes) : new Float32Array(buffer.bytes);
    view.set(values, i * words);
}

test('U20: the staging layout — 204 bytes for Herd & Wolves, every part 4-byte aligned', () => {
    const l = A.agentSnapshotLayout(herd());
    assert.deepEqual(l, { record: 0, obs: 64, act: 160, reason: 168, neighbours: 172, state: 188, neighbourStride: 4, size: 204 });
    const lone = A.agentSnapshotLayout(A.defineAgents({ capacity: 4, world: { min: [0, 0], max: [4, 4] }, cellSize: 4,
        species: [{ name: 'a', maxSpeed: 1, maxTurn: 1, accel: 1, senseRadius: 2, radius: 0.5 }], sensors: [A.Sensors.self()] }));
    assert.deepEqual(lone, { record: 0, obs: 64, act: 72, reason: 80, neighbours: 84, state: 84, neighbourStride: 0, size: 100 });
});

test('U20: the decoder — names, flags, the record\'s words, neighbours per sensor and the counters', () => {
    const def = herd(), l = A.agentSnapshotLayout(def);
    const bytes = new ArrayBuffer(l.size), f = new Float32Array(bytes), u = new Uint32Array(bytes);
    f.set([1.5, 0.25, -2, 0.6, 0.6, 0, 0.8, 0.6435, 0.9, 12.5, 1.25, 3.5], 0);    // pos, radius, vel, heading, energy, age, speedScale, user0
    u[12] = 1; u[13] = A.AGENT_FLAG_ALIVE | A.AGENT_FLAG_FROZEN | (5 << A.AGENT_FLAG_APP_SHIFT); u[14] = 77; u[15] = 9;
    for (let k = 0; k < def.obsDim; k++) f[l.obs / 4 + k] = k / 10;
    f.set([0.5, -1], l.act / 4);
    u[l.reason / 4] = 3;
    u.set([42, 7, NO_AGENT, 12], l.neighbours / 4);
    u.set([1234, 5, 17, 0], l.state / 4);
    const s = A.decodeAgentSnapshot(def, 6, bytes);
    assert.equal(s.index, 6);
    assert.equal(s.species, 'wolf');
    assert.deepEqual([s.alive, s.frozen], [true, true]);
    assert.deepEqual(s.position, [1.5, 0.25, -2].map(Math.fround));
    assert.ok(Math.abs(s.speed - 1) < 1e-6, `speed ${s.speed}`);
    assert.deepEqual([s.radius, s.heading, s.energy, s.age, s.speedScale, s.user0].map(v => +v.toFixed(4)), [0.6, 0.6435, 0.9, 12.5, 1.25, 3.5]);
    assert.deepEqual([s.genome, s.user1, s.flags >>> A.AGENT_FLAG_APP_SHIFT], [77, 9, 5]);
    assert.equal(s.obs.length, 24);
    assert.deepEqual(s.obs.slice(0, 3).map(o => o.name), ['self.speed', 'self.energy', 'wolf[0].side']);
    assert.equal(s.obs[23].name, 'bounds.forward');
    assert.ok(Math.abs(s.obs[23].value - 2.3) < 1e-6);
    assert.deepEqual(s.act.map(a => [a.name, a.value]), [['turn', 0.5], ['throttle', -1]]);
    assert.equal(s.reason, 3);
    assert.deepEqual(s.neighbours, [{ sensor: 'wolf', indices: [42] }, { sensor: 'prey', indices: [7, 12] }]);
    assert.deepEqual([s.tick, s.overflowSteps, s.overflowTotal], [1234, 5, 17]);
});

test('inspect: one submission copies agent i\'s rows, and only its rows, into the staging buffer', async () => {
    const m = mock(), def = herd(), s = system(m, def);
    s.spawn(4, { position: [1, 0, 2], species: 'prey', heading: 0.5, speed: 2 });
    s.spawn(5, { position: [-3, 0, 1], species: 'wolf', heading: -1 });
    s.upload();
    for (const i of [4, 5]) {           // what sense/think/queries would have written
        seed(s.obs, i, def.obsDim, Array.from({ length: def.obsDim }, (_, k) => i * 100 + k));
        seed(s.act, i, def.actDim, [i, -i]);
        seed(s.reason, i, 1, [i + 1], true);
        seed(s.neighbours, i, 4, [i + 10, NO_AGENT, i + 11, NO_AGENT], true);
    }
    seed(s.state, 0, 4, [600, 2, 9, 0], true);
    const snap = await s.inspect(5);
    assert.equal(snap.species, 'wolf');
    assert.deepEqual(snap.position, [-3, 0, 1]);
    assert.ok(Math.abs(snap.heading + 1) < 1e-6);
    assert.equal(snap.obs[0].value, 500);
    assert.equal(snap.obs[23].value, 523);
    assert.deepEqual(snap.act.map(a => a.value), [5, -5]);
    assert.equal(snap.reason, 6);
    assert.deepEqual(snap.neighbours, [{ sensor: 'wolf', indices: [15] }, { sensor: 'prey', indices: [16] }]);
    assert.deepEqual([snap.tick, snap.overflowSteps, snap.overflowTotal], [600, 2, 9]);
    assert.equal(m.copies.length, 6, 'record, obs, act, reason, neighbours and state: one copy each');
    assert.ok(m.copies.every(c => c.srcOffset % 4 === 0 && c.dstOffset % 4 === 0 && c.size % 4 === 0));
    const prey = await s.inspect(4);
    assert.deepEqual([prey.species, prey.obs[1].value, prey.reason], ['prey', 401, 5]);
    assert.ok(Math.abs(prey.speed - 2) < 1e-6);
    assert.equal(m.buffers.filter(b => b.label === 'AgentSystem inspect staging').length, 1, 'the staging buffer is allocated once');
    s.destroy();
    assert.equal(m.buffers.find(b => b.label === 'AgentSystem inspect staging').destroyed, true);
});

test('inspect: requests run one at a time, each resolving its own agent', async () => {
    const m = mock(), s = system(m);
    for (const i of [1, 2, 3]) s.spawn(i, { position: [i, 0, 0], species: 'prey' });
    s.upload();
    let open;
    m.setGate(new Promise(resolve => { open = resolve; }));
    const pending = [s.inspect(1), s.inspect(2), s.inspect(3)];
    await new Promise(resolve => setTimeout(resolve, 0));
    assert.deepEqual(m.log, ['submit', 'map AgentSystem inspect staging'], 'a second request copied before the first was read');
    open();
    m.setGate(null);
    const xs = (await Promise.all(pending)).map(snap => snap.position[0]);
    assert.deepEqual(xs, [1, 2, 3]);
    s.destroy();
});

test('inspect: refuses an index outside capacity and use after destroy; a queued or in-flight request resolves null', async () => {
    const m = mock(), s = system(m);
    for (const bad of [16, -1, 1.5, NaN]) assert.throws(() => s.inspect(bad), new RegExp(`index ${bad} is outside capacity 16`));
    let open;
    m.setGate(new Promise(resolve => { open = resolve; }));
    const inFlight = s.inspect(0), queued = s.inspect(1);
    await new Promise(resolve => setTimeout(resolve, 0));
    s.destroy();
    open();
    assert.deepEqual(await Promise.all([inFlight, queued]), [null, null]);
    assert.equal(m.log.filter(e => e === 'submit').length, 1, 'the queued request copied after destroy()');
    assert.throws(() => s.inspect(0), /used after destroy/);
});
