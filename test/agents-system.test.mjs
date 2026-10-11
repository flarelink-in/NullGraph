import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { build } from 'esbuild';
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';
import * as A from '../dist/agents.js';

globalThis.GPUBufferUsage = { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64 };
const definition = (extra = {}) => A.defineAgents({ capacity: 128,
    world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
    species: [{ name: 'prey', maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 4, radius: 0.4 }],
    sensors: [A.Sensors.nearest({ name: 'one', k: 1 }), A.Sensors.nearest({ name: 'three', k: 3 })], ...extra });
function mock() {
    const writes = [], buffers = [], stages = [], submissions = [];
    const engine = {
        device: { limits: { maxStorageBufferBindingSize: 128 * 1024 * 1024, maxBufferSize: 256 * 1024 * 1024,
            maxComputeWorkgroupsPerDimension: 65535, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256,
            maxStorageBuffersPerShaderStage: 8 },
        createBuffer(o) { const b = { ...o, destroyed: false, destroy() { this.destroyed = true; } }; buffers.push(b); return b; },
        queue: { writeBuffer(...args) { writes.push(args); }, async onSubmittedWorkDone() { submissions.push('await'); } } },
        createComputeStage(o) { const s = { ...o, enabled: true, preClearBuffers: [] }; stages.push(s); return s; },
        setComputePassBindGroup(stage, entries, group) {
            (stage.groups ??= [])[group] = entries;
            if (group === 0) stage.entries = entries;
        },
        render(chain) { submissions.push(chain); },
    };
    return { engine, writes, buffers, stages, submissions };
}

test('U8: cached 0..3 chains repeat the same nodes, with no frame typed-array allocations', () => {
    const m = mock(), s = new A.AgentSystem(m.engine, definition());
    const zero = s.getComputeStages();
    assert.equal(zero.length, 0);
    const chains = [zero];
    for (let n = 1; n <= 3; n++) { s.advance(n * s.definition.dt); chains.push(s.getComputeStages()); }
    assert.equal(chains[2].length, chains[1].length * 2);
    assert.equal(chains[3].length, chains[1].length * 3);
    chains[3].forEach((node, i) => assert.equal(node, chains[1][i % chains[1].length]));
    const original = [globalThis.Float32Array, globalThis.Uint32Array];
    let allocations = 0;
    try {
        [globalThis.Float32Array, globalThis.Uint32Array] = original.map(T => new Proxy(T, {
            construct(target, args) { allocations++; return Reflect.construct(target, args); },
        }));
        for (let i = 0; i < 100; i++) { s.advance(s.definition.dt); assert.equal(s.getComputeStages(), chains[1]); s.upload(); }
        assert.equal(allocations, 0);
    } finally { [globalThis.Float32Array, globalThis.Uint32Array] = original; }
    assert.equal(m.writes.length, 1);
    assert.ok(m.stages.every(stage => stage.preClearBuffers.length === 0));
    s.destroy();
});

test('U16: query defaults match committed step-2 text byte for byte; slices parse and never overlap', async () => {
    const old = execFileSync('git', ['show', '4ebfe99:src/extras/agents/agentGridShaders.ts'], { encoding: 'utf8' });
    const compiled = await build({ stdin: { contents: old, loader: 'ts', resolveDir: 'src/extras/agents' },
        bundle: true, write: false, format: 'esm', platform: 'node' });
    const legacy = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);
    const def = definition();
    for (const k of [1, 3, 8]) assert.equal(A.buildAgentGridQueryWGSL(def, 1, 4, k), legacy.buildAgentGridQueryWGSL(def, 1, 4, k));
    const m = mock(), s = new A.AgentSystem(m.engine, def);
    const queries = m.stages.filter(stage => stage.name.startsWith('AgentSystem query'));
    assert.equal(s.neighbourStride, 4);
    assert.match(queries[0].shaderCode, /result\[i \* 4u \+ n\]/);
    assert.match(queries[1].shaderCode, /result\[i \* 4u \+ 1u \+ n\]/);
    for (const stage of m.stages) new WgslReflect(stage.shaderCode);
    assert.equal(queries[0].entries[4].resource.buffer, queries[1].entries[4].resource.buffer);
    assert.throws(() => A.buildAgentGridQueryWGSL(def, 1, 4, 3, { stride: 3, base: 1 }), /stride and base/);
    s.destroy();
});

test('uploads dirty records only; species tuning validates atomically and changes spawn defaults', () => {
    const m = mock(), def = definition(), s = new A.AgentSystem(m.engine, def);
    s.upload(); assert.equal(m.writes[0][0], s.species);
    s.upload(); assert.equal(m.writes.length, 1);
    s.spawn(3, { position: [0, 0, 0], species: 'prey' });
    s.spawn(5, { position: [0, 0, 0], species: 'prey' });
    s.upload();
    assert.deepEqual(m.writes[1].slice(1), [192, s.records.data.buffer, 192, 192]);
    assert.equal(s.records.dirtyFirst, -1);
    assert.throws(() => s.setSpecies('prey', { senseRadius: 5 }), /senses 5/);
    assert.equal(s.definition.species[0].senseRadius, 4);
    assert.throws(() => s.setSpecies('missing', {}), /not declared/);
    s.setSpecies('prey', { maxSpeed: 1, radius: 0.7, mesh: 3 });
    s.spawn(7, { position: [0, 0, 0], species: 'prey', speed: 3 });
    assert.ok(Math.abs(s.records.data[7 * 16 + 3] - 0.7) < 1e-6);
    assert.equal(s.records.data[7 * 16 + 6], 1);
    assert.equal(def.species[0].maxSpeed, 3);
    s.upload(); assert.equal(m.writes.at(-1)[0], s.species);
    s.kill(7); assert.equal(s.records.isAlive(7), false);
    s.destroy();
});

test('lockstep uploads/submits/awaits without changing selected frame chain; teardown is idempotent', async () => {
    const m = mock(), s = new A.AgentSystem(m.engine, definition());
    s.advance(3 * s.definition.dt);
    const frame = s.getComputeStages();
    await s.step();
    assert.equal(m.submissions[0].length * 3, frame.length);
    assert.equal(m.submissions[1], 'await');
    assert.equal(s.getComputeStages(), frame);
    s.destroy(); s.destroy();
    assert.ok(m.buffers.every(b => b.destroyed));
    assert.ok(frame.every(stage => !stage.enabled));
    for (const call of [() => s.advance(0), () => s.getComputeStages(), () => s.upload(),
        () => s.spawn(0, {}), () => s.kill(0), () => s.setSpecies('prey', {})]) assert.throws(call, /used after destroy/);
    await assert.rejects(s.step(), /used after destroy/);
});

test('validation happens before buffer allocation; no-nearest layouts have a valid dummy buffer', () => {
    for (const thinkEvery of [0, -1, 1.5, NaN, Infinity]) {
        const m = mock(); assert.throws(() => new A.AgentSystem(m.engine, definition(), { thinkEvery }), /positive integer/);
        assert.equal(m.buffers.length, 0);
    }
    const m = mock(); m.engine.device.limits.maxStorageBufferBindingSize = 100;
    assert.throws(() => new A.AgentSystem(m.engine, definition()), /agents needs 8192 bytes/);
    assert.equal(m.buffers.length, 0);
    const n = mock(), s = new A.AgentSystem(n.engine, definition({ sensors: [A.Sensors.self()] }));
    assert.equal(s.neighbourStride, 0); assert.equal(s.neighbours.size, 4);
    assert.equal(n.stages.filter(stage => stage.name.startsWith('AgentSystem query')).length, 0);
    s.destroy();
});

test('U9 sense: every sensor kind and sparse layout parses, bindings remain referenced, predicate emitted once', () => {
    const custom = A.Sensors.custom({ name: 'extra', width: 2,
        wgsl: 'fn extra(i: u32, a: Agent) -> array<f32, 2> { return array<f32, 2>(f32(i), a.state.x); }' });
    for (const sensors of [[A.Sensors.self()], [A.Sensors.bounds()], [custom],
        [A.Sensors.self(), A.Sensors.nearest({ k: 8 }), A.Sensors.bounds(), custom]]) {
        const code = A.buildAgentSenseWGSL(definition({ sensors }), 3);
        const reflect = new WgslReflect(code);
        assert.equal(reflect.entry.compute.length, 1);
        assert.equal(reflect.entry.compute[0].name, 'cs_main');
        assert.equal((code.match(/fn thinksNow\(/g) ?? []).length, 1);
        assert.equal((code.match(/const OBS_DIM\b/g) ?? []).length, 1);
        assert.ok(!code.includes('`'));
        assert.match(code, /if \(!thinksNow\(i, a\)\)/);
        assert.match(code, /_ = arrayLength\(&neighbours\)/);
        assert.match(code, /_ = arrayLength\(&obs\)/);
        assert.match(code, /species\[a.ids.x\]/);
        const m = mock(), s = new A.AgentSystem(m.engine, definition({ sensors }));
        const sense = m.stages.find(stage => stage.name === 'AgentSystem sense');
        assert.equal(sense.entries.length, 5);
        s.advance(s.definition.dt);
        assert.equal(s.getComputeStages().at(-2), sense);
        s.destroy();
    }
});

test('U17 sense: missing/wrong/comment-only custom signatures fail by name before allocation', () => {
    for (const wgsl of ['', '// fn extra(i: u32, a: Agent) -> array<f32, 2> {}',
        'fn extra(i: u32, a: Agent) -> array<f32, 3> { return array<f32, 3>(); }']) {
        const def = definition({ sensors: [A.Sensors.custom({ name: 'extra', width: 2, wgsl })] });
        const m = mock();
        assert.throws(() => new A.AgentSystem(m.engine, def), /custom sensor "extra" must define fn extra/);
        assert.equal(m.buffers.length, 0);
    }
});
