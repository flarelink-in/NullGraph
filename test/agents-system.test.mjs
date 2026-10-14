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
    // Setup wrote one thing: setBrain's 16-byte species mask (phase 7.3). Frames add only uploads.
    assert.deepEqual(m.writes.map(([b]) => b.label), ['AgentSystem brain species']);
    const original = [globalThis.Float32Array, globalThis.Uint32Array];
    let allocations = 0;
    try {
        [globalThis.Float32Array, globalThis.Uint32Array] = original.map(T => new Proxy(T, {
            construct(target, args) { allocations++; return Reflect.construct(target, args); },
        }));
        for (let i = 0; i < 100; i++) { s.advance(s.definition.dt); assert.equal(s.getComputeStages(), chains[1]); s.upload(); }
        assert.equal(allocations, 0);
    } finally { [globalThis.Float32Array, globalThis.Uint32Array] = original; }
    assert.equal(m.writes.length, 2, 'a hundred frames wrote more than the first species upload');
    assert.equal(m.writes[1][0], s.species);
    assert.ok(m.stages.every(stage => stage.preClearBuffers.length === 0));
    s.destroy();
});

test('U16/U23: windowed query slices parse and never overlap; tick is GPU-owned', async () => {
    const def = definition();
    for (const k of [1, 3, 8]) new WgslReflect(A.buildAgentGridQueryWGSL(def, 1, 4, k));
    const m = mock(), s = makeSystem(m.engine, def);
    const queries = m.stages.filter(stage => stage.name.startsWith('AgentSystem query'));
    assert.equal(s.neighbourStride, 4);
    assert.match(queries[0].shaderCode, /result\[i \* 4u \+ n\]/);
    assert.match(queries[1].shaderCode, /result\[i \* 4u \+ 1u \+ n\]/);
    for (const stage of m.stages) new WgslReflect(stage.shaderCode);
    assert.equal(queries[0].entries[4].resource.buffer, queries[1].entries[4].resource.buffer);
    assert.ok(queries.every(q => q.entries[5].resource.buffer === s.state));
    // A crowded cell's window is keyed by the cell, so every query of that cell reads the same entries.
    for (const q of queries) {
        assert.match(q.shaderCode, /at \+= gridWindow\(c, tick\) % count;/);
        assert.doesNotMatch(q.shaderCode, /gridWindow\(i,/);
    }
    assert.throws(() => A.buildAgentGridQueryWGSL(def, 1, 4, 3, { stride: 3, base: 1 }), /stride and base/);
    s.destroy();
});

test('uploads dirty records only; species tuning validates atomically and changes spawn defaults', () => {
    const m = mock(), def = definition(), s = makeSystem(m.engine, def);
    const setup = m.writes.length;            // setBrain's species mask (phase 7.3)
    s.upload(); assert.equal(m.writes[setup][0], s.species);
    s.upload(); assert.equal(m.writes.length, setup + 1);
    s.spawn(3, { position: [0, 0, 0], species: 'prey' });
    s.spawn(5, { position: [0, 0, 0], species: 'prey' });
    s.upload();
    assert.deepEqual(m.writes[setup + 1].slice(1), [192, s.records.data.buffer, 192, 192]);
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

test('U23: radix stage counts, AgentGrid counter ordering and named limit refusals', () => {
    for (const total of [255, 6400, 1048576]) {
        const m = mock(), records = m.engine.device.createBuffer({ size: 128 * 64, usage: GPUBufferUsage.STORAGE });
        const layout = { ...definition(), grid: { cellsX: total, cellsZ: 1, cellTotal: total } };
        const grid = new A.PointGrid(m.engine, records, { ...layout, stride: 16 });
        assert.equal(m.stages.filter(s => s.name === 'PointGrid stable scatter').length, A.gridRadixLayout(total).passes);
        assert.ok(!m.buffers.some(b => b.label?.includes('cursors')));
        grid.destroy();
    }
    const m = mock(), records = m.engine.device.createBuffer({ size: 128 * 64, usage: GPUBufferUsage.STORAGE });
    const grid = new A.AgentGrid(m.engine, records, definition());
    assert.deepEqual(grid.getComputeStages().slice(-2).map(s => s.name), ['AgentGrid nearest', 'AgentGrid tick']);
    assert.equal(grid.getComputeStages().at(-2).entries[5].resource.buffer, grid.getComputeStages().at(-1).entries[0].resource.buffer);
    grid.destroy();
    const limited = mock(); limited.engine.device.limits.maxStorageBufferBindingSize = 9000;
    const input = limited.engine.device.createBuffer({ size: 6156, usage: GPUBufferUsage.STORAGE });
    const layout = { ...definition(), grid: { cellsX: 1023, cellsZ: 1, cellTotal: 1023 } };
    assert.throws(() => new A.PointGrid(limited.engine, input, { ...layout, stride: 3, capacity: 513 }), /radix histogram/);
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

test('U23: grid constructor failures release owned resources without destroying records', () => {
    for (const label of ['PointGrid radix keys 1', 'AgentGrid tick']) {
        const m = mock(), records = m.engine.device.createBuffer({ size: 8192, usage: GPUBufferUsage.STORAGE });
        const create = m.engine.device.createBuffer;
        m.engine.device.createBuffer = o => { if (o.label === label) throw new Error('injected allocation'); return create(o); };
        assert.throws(() => new A.AgentGrid(m.engine, records, definition()), /injected allocation/);
        assert.ok(!records.destroyed);
        assert.ok(m.buffers.filter(b => b !== records).every(b => b.destroyed));
        assert.ok(m.stages.every(s => !s.enabled));
    }
    for (const name of ['PointGrid stable scatter', 'AgentGrid nearest', 'AgentGrid tick']) {
        const m = mock(), records = m.engine.device.createBuffer({ size: 8192, usage: GPUBufferUsage.STORAGE });
        const bind = m.engine.setComputePassBindGroup;
        m.engine.setComputePassBindGroup = (stage, ...args) => {
            if (stage.name === name) throw new Error('injected binding');
            bind(stage, ...args);
        };
        assert.throws(() => new A.AgentGrid(m.engine, records, definition()), /injected binding/);
        assert.ok(!records.destroyed);
        assert.ok(m.buffers.filter(b => b !== records).every(b => b.destroyed));
        assert.ok(m.stages.every(s => !s.enabled));
    }
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
    assert.throws(() => s.setBrain({ kind: 'async' }), /^Error: AgentSystem: a brain is a stage brain, or an async brain with bind\(\) and run\(\)\.$/);
    assert.throws(() => s.setBrain({ kind: 'async', run() {} }), /an async brain with bind\(\) and run\(\)/);
    assert.throws(() => s.setBrain({ kind: 'other', wgsl: good }), /a brain is a stage brain/);
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

test('U17: the binding budget follows the device: a negotiated 16-buffer device leaves 11', () => {
    const m = mock(), s = new A.AgentSystem(m.engine, definition());
    m.engine.device.limits.maxStorageBuffersPerShaderStage = 16;
    const good = 'fn think(i: u32, species: u32, tick: u32) {}';
    s.setBrain({ kind: 'stage', wgsl: good, buffers: [{}, {}, {}, {}] });
    assert.throws(() => s.setBrain({ kind: 'stage', wgsl: good, buffers: Array.from({ length: 12 }, () => ({})) }),
        /brain binds 12 buffers; 11 are left/);
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
    assert.equal(think.groups[0].length, 7); assert.equal(think.groups[1][0].resource.buffer, extra);
    s.destroy(); assert.equal(destroyed, false);
    assert.ok(one.every(stage => !stage.enabled));
    assert.throws(() => s.setBrain(brain), /after destroy/);
    assert.throws(() => s.addStepStage({ name: 'late', wgsl: '' }), /after destroy/);
});

// U22 (phase 7.3, docs/agents-species-brains-design.md §6): one think pass per brain, gated to its
// species by the stage's 16-byte mask uniform; moving species between attached brains rewrites masks
// and builds nothing.
const threeSpecies = () => definition({ species: ['prey', 'wolf', 'deer'].map(name =>
    ({ name, maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 4, radius: 0.4 })) });
const tagged = tag => new A.RuleBrain(`fn think(i: u32, species: u32, tick: u32) { setReason(i, ${tag}u); }`);
const passesOf = s => { s.advance(s.definition.dt); return s.getComputeStages().filter(st => st.name === 'AgentSystem think'); };
const maskOf = (m, stage) => m.writes.filter(([b]) => b === stage.groups[0][6].resource.buffer).at(-1)?.[2][0];

test('U22: one pass per distinct brain, each gated to its species; sense → passes → act', () => {
    const m = mock(), s = new A.AgentSystem(m.engine, threeSpecies());
    s.setBrain(tagged(1));
    let passes = passesOf(s);
    assert.equal(passes.length, 1);
    assert.equal(maskOf(m, passes[0]), 0b111);
    s.setBrain(tagged(2), { species: 'wolf' });
    s.setBrain(tagged(3), { species: ['deer'] });
    passes = passesOf(s);
    assert.deepEqual(passes.map(p => maskOf(m, p)), [0b001, 0b010, 0b100]);
    assert.ok(passes.every(p => p.groups[0].length === 7 && p.enabled));
    const names = s.getComputeStages().map(st => st.name), first = names.indexOf('AgentSystem think');
    assert.deepEqual(names.slice(first - 1, first + 4),
        ['AgentSystem sense', 'AgentSystem think', 'AgentSystem think', 'AgentSystem think', 'AgentSystem act']);
    s.destroy();
    assert.ok(passes.every(p => !p.enabled));
});

test('U22: moving species to an attached brain builds nothing; a brain serving nobody leaves every chain', () => {
    const m = mock(), s = new A.AgentSystem(m.engine, threeSpecies());
    const rules = tagged(1), wolfBrain = tagged(2);
    s.setBrain(rules);
    s.setBrain(wolfBrain, { species: 'wolf' });
    const [rulesPass, wolfPass] = passesOf(s), built = m.stages.length;
    s.setBrain(rules, { species: 'wolf' });
    assert.equal(m.stages.length, built, 'moving species to an attached brain built a stage');
    assert.equal(maskOf(m, rulesPass), 0b111);
    assert.equal(wolfPass.enabled, false);
    for (let n = 0; n <= s.definition.maxStepsPerFrame; n++) {
        s.advance(n * s.definition.dt);
        assert.ok(!s.getComputeStages().includes(wolfPass), `a chain of ${n} steps still runs the retired pass`);
    }
    s.setBrain(wolfBrain, { species: 'wolf' });
    assert.equal(m.stages.length, built + 1);
    assert.equal(m.stages.at(-1).shaderCode, wolfPass.shaderCode, 're-attaching changed the shader text (a pipeline-cache miss)');
    assert.equal(maskOf(m, rulesPass), 0b101);
    s.destroy();
});

test('U22: species are checked by name: unknown at setBrain, missing before the first step', async () => {
    const m = mock(), s = new A.AgentSystem(m.engine, threeSpecies());
    const before = m.stages.length;
    assert.throws(() => s.setBrain(tagged(1), { species: 'bear' }), /AgentSystem: "bear" is not a declared species\./);
    assert.equal(m.stages.length, before, 'a refused setBrain built a stage');
    s.setBrain(tagged(1), { species: ['prey', 'deer'] });
    assert.throws(() => s.getComputeStages(),
        /AgentSystem: no brain for species "wolf"; call setBrain\(brain, \{ species: 'wolf' \}\)\./);
    await assert.rejects(s.step(), /no brain for species "wolf"/);
    s.setBrain(tagged(2), { species: 'wolf' });
    assert.doesNotThrow(() => s.getComputeStages());
    s.destroy();
});

test('U22: an attached brain destroyed since is refused when re-assigned, and nothing changes', () => {
    const m = mock(), s = new A.AgentSystem(m.engine, threeSpecies());
    const brain = { kind: 'stage', wgsl: 'fn think(i: u32, species: u32, tick: u32) {}', destroyed: false,
        assertDevice() { if (this.destroyed) throw new Error('MlpBrain: cannot use after destroy().'); }, destroy() { this.destroyed = true; } };
    s.setBrain(brain);
    s.setBrain(tagged(2), { species: 'wolf' });
    const [pass] = passesOf(s), writes = m.writes.length;
    brain.destroy();
    assert.throws(() => s.setBrain(brain, { species: 'wolf' }), /after destroy/);
    assert.equal(m.writes.length, writes, 'a refused reassignment rewrote a mask');
    assert.equal(maskOf(m, pass), 0b101);
    s.destroy();
});

test('U22: the species gate is generated beside the cadence gate, and the wrapper parses', () => {
    const def = threeSpecies(), code = A.buildAgentThinkWGSL(def, 'fn think(i: u32, species: u32, tick: u32) {}', 3);
    assert.match(code, /@group\(0\) @binding\(6\) var<uniform> brainSpecies: vec4<u32>;/);
    assert.match(code, /if \(!thinksNow\(i, a\)\) \{ return; \}\s*if \(\(brainSpecies\.x & \(1u << a\.ids\.x\)\) == 0u\) \{ return; \}/);
    const reflect = new WgslReflect(code);
    assert.equal(reflect.entry.compute[0].name, 'cs_main');
    assert.ok(reflect.uniforms.some(u => u.name === 'brainSpecies' && u.group === 0 && u.binding === 6));
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
    assert.equal(m.buffers.at(-1).label, 'AgentSystem brain species');
    assert.equal(m.buffers.at(-1).destroyed, true, 'a failed setBrain kept its species mask');
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

// U13 (step 9, phase 9.3; agents-ort-design.md §8): async brains, against a fake whose runs wait
// until the test settles them. The brain is bound before its pass is built; frames start at most
// one run each, never in a frame without a step and at most once per thinkEvery steps; lockstep
// renders sense, awaits the run, then renders the pass and act; a failed run is thrown once.
function asyncBrain(log = []) {
    const runs = [], binds = [];
    const brain = { kind: 'async', wgsl: 'fn think(i: u32, species: u32, tick: u32) { setReason(i, 7u); }', buffers: [],
        bind(io) { binds.push(io); log.push('bind'); },
        run() { log.push('run'); return new Promise((done, failed) => runs.push({ done, failed })); },
        destroy() {} };
    const settle = async (error) => {
        const run = runs.find(r => !r.settled);
        run.settled = true;
        if (error) run.failed(error); else run.done();
        for (let k = 0; k < 5; k++) await Promise.resolve();
    };
    return { brain, runs, binds, settle };
}
const named = chain => chain.map(stage => stage.name);
const ticks = async () => { for (let k = 0; k < 5; k++) await Promise.resolve(); };

test('U13: an async brain is bound before its pass is built; its pass is a think pass, gated like any', () => {
    const m = mock(), log = [], s = new A.AgentSystem(m.engine, threeSpecies());
    m.engine.createComputeStage = (create => o => { log.push(`stage ${o.name}`); return create(o); })(m.engine.createComputeStage);
    const { brain, binds } = asyncBrain(log);
    s.setBrain(tagged(1));
    log.length = 0;
    s.setBrain(brain, { species: 'wolf' });
    assert.deepEqual(log, ['bind', 'stage AgentSystem think'], 'bound first, then its pass is built');
    assert.deepEqual(binds, [{ obs: s.obs, act: s.act, capacity: 128, obsDim: s.definition.obsDim, actDim: s.definition.actDim }]);
    const passes = passesOf(s);
    assert.equal(passes.length, 2);
    assert.deepEqual(passes.map(p => maskOf(m, p)), [0b101, 0b010], 'the async brain\'s pass serves the wolves alone');
    assert.match(passes[1].shaderCode, /setReason\(i, 7u\)/);
    // A brain whose bind refuses is not attached: nothing built, the species keeps its brain.
    const count = m.stages.length;
    assert.throws(() => s.setBrain({ ...asyncBrain().brain, bind() { throw new Error('not this one'); } }, { species: 'deer' }), /not this one/);
    assert.equal(m.stages.length, count);
    assert.equal(passesOf(s).length, 2);
    // obs is rounded up to 16 B: ONNX Runtime reads 16 B at a time.
    assert.equal(s.obs.size % 16, 0);
    assert.ok(s.obs.size >= 128 * s.definition.obsDim * 4);
    s.destroy();
});

test('U13: frames start one run at a time, none in a frame without a step, and at most one per thinkEvery steps', async () => {
    for (const thinkEvery of [1, 3]) {
        const m = mock(), s = new A.AgentSystem(m.engine, definition(), { thinkEvery }), { brain, runs, settle } = asyncBrain();
        s.setBrain(brain);
        const dt = s.definition.dt;
        assert.equal(s.advance(0), 0);
        assert.equal(runs.length, 0, 'no step, no run: the observations have not changed');
        s.advance(dt);
        assert.equal(runs.length, 1, 'the first frame with a step starts a run');
        s.advance(dt); s.advance(dt);
        assert.equal(runs.length, 1, 'never two in flight');
        await settle();
        s.advance(dt);
        assert.equal(runs.length, 2, thinkEvery === 1 ? 'a run per step' : 'three steps since the last run started, two of them while it was in flight');
        if (thinkEvery === 3) {
            await settle();
            s.advance(dt); s.advance(dt);
            assert.equal(runs.length, 2, 'two steps since: no run');
            s.advance(dt);
            assert.equal(runs.length, 3, 'the third: one run per three steps');
        }
        await settle();
        s.destroy();
    }
});

test('U13: lockstep renders sense, awaits the run, then renders the passes, act and the rest', async () => {
    const m = mock(), log = [], s = new A.AgentSystem(m.engine, threeSpecies()), { brain, runs, settle } = asyncBrain(log);
    m.engine.render = chain => { m.submissions.push(chain); log.push(`render ${named(chain).includes('AgentSystem sense') ? 'sense' : 'act'}`); };
    s.setBrain(tagged(1));
    s.setBrain(brain, { species: 'wolf' });
    log.length = 0;
    const step = s.step();
    await ticks();
    assert.deepEqual(log, ['render sense', 'run'], 'the run waits for this step\'s observations');
    await settle();
    await step;
    assert.deepEqual(log, ['render sense', 'run', 'render act']);
    const [first, second] = m.submissions.slice(-3, -1);
    assert.ok(named(first).at(-1) === 'AgentSystem sense' && !named(first).includes('AgentSystem think'), 'the first half ends with sense');
    assert.equal(named(second).filter(n => n === 'AgentSystem think').length, 2, 'every think pass is in the second half, the copy pass among them');
    assert.deepEqual(named(second).slice(-2), ['AgentSystem act', 'AgentSystem tick'], 'act and tick follow');
    assert.equal(m.submissions.at(-1), 'await');
    // An interactive run in flight is awaited first: two runs of one brain never overlap.
    log.length = 0;
    s.advance(s.definition.dt);
    assert.deepEqual(log, ['run']);
    const next = s.step();
    await ticks();
    assert.deepEqual(log, ['run'], 'nothing rendered while the interactive run is in flight');
    await settle();
    await ticks();
    assert.deepEqual(log, ['run', 'render sense', 'run']);
    await settle();
    await next;
    assert.deepEqual(log, ['run', 'render sense', 'run', 'render act']);
    assert.equal(runs.length, 3);
    s.destroy();
});

test('U13: a failed run is thrown once, naming the species; its brain starts no other run', async () => {
    const m = mock(), s = new A.AgentSystem(m.engine, threeSpecies()), { brain, runs, settle } = asyncBrain();
    s.setBrain(tagged(1));
    s.setBrain(brain, { species: ['wolf', 'deer'] });
    s.advance(s.definition.dt);
    await settle(new Error('failed to call OrtRun(). ERROR_CODE: 1\nmore'));
    assert.throws(() => s.advance(s.definition.dt),
        /^Error: AgentSystem: the brain of species "wolf", "deer" failed in a run: failed to call OrtRun\(\)\. ERROR_CODE: 1\. Give the species another brain with setBrain\(\)\.$/);
    s.advance(s.definition.dt);
    s.advance(s.definition.dt);
    assert.equal(runs.length, 1, 'thrown once; no run starts for that brain again');
    await s.step();
    assert.equal(runs.length, 1, 'nor in lockstep, which still steps');
    // A brain that throws instead of rejecting fails the same way.
    const other = asyncBrain();
    other.brain.run = () => { throw new Error('at once'); };
    s.setBrain(other.brain, { species: 'deer' });
    s.advance(s.definition.dt);
    await ticks();
    assert.throws(() => s.advance(s.definition.dt), /species "deer" failed in a run: at once\. Give/);
    // Another brain for the species ends it.
    s.setBrain(tagged(2), { species: ['wolf', 'deer'] });
    s.advance(s.definition.dt);
    s.destroy();
});

test('U13: a run in flight keeps the buffers until it settles; a detached brain\'s failure is nobody\'s', async () => {
    const m = mock(), s = new A.AgentSystem(m.engine, threeSpecies()), { brain, settle } = asyncBrain();
    s.setBrain(brain);
    s.advance(s.definition.dt);
    s.setBrain(tagged(1));                              // detached, its run still in flight
    await settle(new Error('late'));
    s.advance(s.definition.dt);                         // nothing thrown: the brain serves no species
    const again = asyncBrain();
    s.setBrain(again.brain, { species: 'prey' });
    s.advance(s.definition.dt);
    s.destroy();
    assert.ok(m.buffers.some(b => b.label === 'AgentSystem obs' && !b.destroyed), 'obs is kept while a run may still read it');
    await again.settle();
    await ticks();
    assert.ok(m.buffers.filter(b => b.label?.startsWith('AgentSystem')).every(b => b.destroyed), 'then every buffer is released');
});

test('U13: a brain attached again while its run is in flight starts no second run until that one settles', async () => {
    const m = mock(), s = new A.AgentSystem(m.engine, threeSpecies()), { brain, runs, settle } = asyncBrain();
    s.setBrain(brain);
    s.advance(s.definition.dt);
    s.setBrain(tagged(1));                              // detached, its run still in flight
    s.setBrain(brain);                                  // and attached again
    s.advance(s.definition.dt);
    assert.equal(runs.length, 1, 'never two runs of one brain in flight');
    const step = s.step();
    await ticks();
    assert.equal(runs.length, 1, 'lockstep awaits the run of the earlier attach');
    await settle();
    await ticks();
    assert.equal(runs.length, 2, 'then starts its own');
    await settle();
    await step;
    s.advance(s.definition.dt);
    assert.equal(runs.length, 3, 'and frames start runs again');
    await settle();
    s.destroy();
});
