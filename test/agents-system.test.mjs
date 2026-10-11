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
    const m = mock(), s = makeSystem(m.engine, definition());
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
    const m = mock(), s = makeSystem(m.engine, def);
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
    const m = mock(), def = definition(), s = makeSystem(m.engine, def);
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
    const m = mock(), s = makeSystem(m.engine, definition());
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
        const m = mock(); assert.throws(() => makeSystem(m.engine, definition(), { thinkEvery }), /positive integer/);
        assert.equal(m.buffers.length, 0);
    }
    const m = mock(); m.engine.device.limits.maxStorageBufferBindingSize = 100;
    assert.throws(() => makeSystem(m.engine, definition()), /agents needs 8192 bytes/);
    assert.equal(m.buffers.length, 0);
    const n = mock(), s = makeSystem(n.engine, definition({ sensors: [A.Sensors.self()] }));
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
        const m = mock(), s = makeSystem(m.engine, definition({ sensors }));
        const sense = m.stages.find(stage => stage.name === 'AgentSystem sense');
        assert.equal(sense.entries.length, 5);
        s.advance(s.definition.dt);
        assert.equal(s.getComputeStages().at(-4), sense);
        s.destroy();
    }
});

test('U17 sense: missing/wrong/comment-only custom signatures fail by name before allocation', () => {
    for (const wgsl of ['', '// fn extra(i: u32, a: Agent) -> array<f32, 2> {}',
        'fn extra(i: u32, a: Agent) -> array<f32, 3> { return array<f32, 3>(); }']) {
        const def = definition({ sensors: [A.Sensors.custom({ name: 'extra', width: 2, wgsl })] });
        const m = mock();
        assert.throws(() => makeSystem(m.engine, def), /custom sensor "extra" must define fn extra/);
        assert.equal(m.buffers.length, 0);
    }
});

test('U9 act: steering/custom modules parse, preserve bindings and integrate before tick', () => {
    for (const actions of [A.Actions.steering(), A.Actions.custom({ name: 'customMove', width: 1,
        wgsl: 'fn customMove(i: u32) { agents[i].state.w = act[i]; }' })]) {
        const def = definition({ actions }), code = A.buildAgentActWGSL(def);
        const reflected = new WgslReflect(code);
        assert.equal(reflected.entry.compute[0].name, 'cs_main');
        assert.equal((code.match(/const ACT_DIM\b/g) ?? []).length, 1);
        assert.ok(!code.includes('`'));
        assert.match(code, /arrayLength\(&act\)/);
        assert.match(code, /species\[a.ids.x\]/);
        if (actions.kind === 'steering') assert.match(code, /bitcast<u32>\(a0\) & 0x7f800000u/);
        const m = mock(), s = makeSystem(m.engine, def);
        s.advance(def.dt);
        assert.equal(s.getComputeStages().at(-2).name, 'AgentSystem act');
        s.destroy();
    }
});

test('U17 act: wrong/comment-only custom signatures fail by name before allocation', () => {
    for (const wgsl of ['', '// fn customMove(i: u32) {}', 'fn customMove(i: f32) {}']) {
        const m = mock(), def = definition({ actions: A.Actions.custom({ name: 'customMove', width: 1, wgsl }) });
        assert.throws(() => makeSystem(m.engine, def), /custom actions "customMove" must define fn customMove/);
        assert.equal(m.buffers.length, 0);
    }
});
function makeSystem(...args) { const s = new A.AgentSystem(...args); s.setBrain(new A.RuleBrain('fn think(i: u32, species: u32, tick: u32) {}')); return s; }

test('U17: no brain, signatures, dimensions and binding budgets fail before registering stages', async () => {
    const m = mock(), s = new A.AgentSystem(m.engine, definition());
    assert.throws(() => s.getComputeStages(), /no brain/);
    await assert.rejects(s.step(), /no brain/);
    for (const wgsl of ['', '// fn think(i: u32, species: u32, tick: u32) {}',
        'fn think(i: u32, species: u32) {}']) assert.throws(() => new A.RuleBrain(wgsl), /RuleBrain:.*fn think/);
    const good = 'fn think(i: u32, species: u32, tick: u32) {}';
    const count = m.stages.length;
    assert.throws(() => s.setBrain({ kind: 'stage', wgsl: good, dims: { obs: 999, act: 2 } }), /maps 999/);
    assert.throws(() => s.setBrain({ kind: 'stage', wgsl: good, buffers: [{}, {}, {}, {}] }), /brain binds 4.*3 are left/);
    assert.throws(() => s.setBrain({ kind: 'async' }), /async brains.*C2/);
    assert.throws(() => s.setBrain({ kind: 'stage' }), /RuleBrain/);
    for (const wgsl of ['', '/* fn step(i: u32, tick: u32) {} */', 'fn step(i: f32, tick: u32) {}']) {
        assert.throws(() => s.addStepStage({ name: 'bad', wgsl }), /step stage "bad" must define/);
    }
    assert.throws(() => s.addStepStage({ name: 'tooMany', wgsl: 'fn step(i: u32, tick: u32) {}',
        buffers: [{}, {}, {}, {}] }), /step stage "tooMany" binds 4.*3 are left/);
    m.engine.device.limits.maxStorageBuffersPerShaderStage = 6;
    assert.throws(() => s.addStepStage({ name: 'limited', wgsl: 'fn step(i: u32, tick: u32) {}',
        buffers: [{}, {}] }), /2 buffers; 1 are left/);
    assert.equal(m.stages.length, count);
    s.destroy();
});

test('U9/U8: think/app modules parse, predicate shared, chains ordered, replacement disables old think', () => {
    const def = definition(), m = mock(), s = makeSystem(m.engine, def);
    const brain = new A.RuleBrain(`fn think(i: u32, species: u32, tick: u32) {
        setAct(i, ACT_TURN, rand01(i, tick, species)); setAct(i, ACT_THROTTLE, obs(i, OBS_ONE_0)); setReason(i, tick);
    }`);
    const code = A.buildAgentThinkWGSL(def, brain.wgsl, 3);
    assert.equal(new WgslReflect(code).entry.compute[0].name, 'cs_main');
    assert.equal((code.match(/fn thinksNow\(/g) ?? []).length, 1);
    assert.ok(code.includes(A.buildAgentThinkPredicateWGSL(3)));
    assert.ok(A.buildAgentSenseWGSL(def, 3).includes(A.buildAgentThinkPredicateWGSL(3)));
    assert.equal((code.match(/const OBS_DIM\b/g) ?? []).length, 1);
    assert.ok(!code.includes('`'));
    s.advance(def.dt); const oldChain = s.getComputeStages(), oldThink = oldChain.at(-3);
    let destroyed = false;
    const extra = { destroy() { destroyed = true; } };
    s.setBrain({ ...brain, buffers: [extra] });
    assert.equal(oldThink.enabled, false);
    assert.notEqual(s.getComputeStages(), oldChain);
    for (const name of ['catches', 'respawn']) {
        const wgsl = 'fn step(i: u32, tick: u32) { respawn(i, randomPointInWorld(i, tick), rand01(i, tick, 2u)); }';
        const shader = A.buildAgentStepWGSL(def, name, wgsl);
        assert.equal(new WgslReflect(shader).entry.compute[0].name, 'cs_main');
        assert.equal((shader.match(/const ACT_DIM\b/g) ?? []).length, 1);
        assert.ok(!shader.includes('`'));
        s.addStepStage({ name, wgsl, buffers: [extra] });
    }
    assert.throws(() => s.addStepStage({ name: 'catches', wgsl: '' }), /added twice/);
    assert.deepEqual(s.getComputeStages().slice(-6).map(stage => stage.name),
        ['AgentSystem sense', 'AgentSystem think', 'AgentSystem act', 'AgentSystem step catches', 'AgentSystem step respawn', 'AgentSystem tick']);
    const one = s.getComputeStages(); s.advance(3 * def.dt);
    assert.equal(s.getComputeStages().length, one.length * 3);
    s.getComputeStages().forEach((stage, i) => assert.equal(stage, one[i % one.length]));
    const think = one.find(stage => stage.name === 'AgentSystem think');
    assert.equal(think.groups[0].length, 6); assert.equal(think.groups[1][0].resource.buffer, extra);
    s.destroy(); assert.equal(destroyed, false);
    assert.ok(one.every(stage => !stage.enabled));
    assert.throws(() => s.setBrain(brain), /after destroy/);
    assert.throws(() => s.addStepStage({ name: 'late', wgsl: '' }), /after destroy/);
});

test('U17: case-folded species constants cannot collide', () => {
    const def = definition({ species: ['prey', 'PREY'].map(name => ({ name, maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 4, radius: 0.4 })) });
    assert.throws(() => A.buildAgentThinkWGSL(def, 'fn think(i: u32, species: u32, tick: u32) {}'), /SPECIES_PREY.*twice/);
});

test('shape change bind failure disables new stage and preserves previous chain', () => {
    const m = mock(), s = makeSystem(m.engine, definition());
    s.advance(s.definition.dt); const chain = s.getComputeStages(), previous = chain.at(-3);
    const bind = m.engine.setComputePassBindGroup;
    m.engine.setComputePassBindGroup = () => { throw new Error('fixture binding failed'); };
    assert.throws(() => s.setBrain(new A.RuleBrain('fn think(i: u32, species: u32, tick: u32) {}')), /binding failed/);
    assert.equal(m.stages.at(-1).enabled, false);
    assert.equal(previous.enabled, true); assert.equal(s.getComputeStages(), chain);
    assert.throws(() => s.addStepStage({ name: 'retry', wgsl: 'fn step(i: u32, tick: u32) {}' }), /binding failed/);
    assert.equal(m.stages.at(-1).enabled, false);
    m.engine.setComputePassBindGroup = bind;
    s.addStepStage({ name: 'retry', wgsl: 'fn step(i: u32, tick: u32) {}' });
    s.destroy();
});

test('rand01 CPU twin is deterministic, f32-exact and half-open', () => {
    for (let i = 0; i < 10000; i++) for (const tick of [0, 0xffffffff]) {
        const v = A.rand01(i, tick, i % 7);
        assert.ok(v >= 0 && v < 1); assert.equal(Math.fround(v), v);
        assert.equal(A.rand01(i, tick, i % 7), v);
    }
});
