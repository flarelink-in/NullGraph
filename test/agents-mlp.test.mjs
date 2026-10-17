import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as A from '../dist/agents.js';
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';

globalThis.GPUBufferUsage = { STORAGE: 128, COPY_DST: 8 };

const layer = (input, output, activation = 'identity', weights, biases) => ({ input, output, activation,
    weights: new Float32Array(weights ?? input * output), biases: new Float32Array(biases ?? output) });
const near = (a, b) => assert.ok(Math.abs(a - b) <= 1e-6 + Math.abs(b) * 1e-6, `${a} != ${b}`);

test('U10: row-major affine and two-layer forward are hand-computable', () => {
    const net = [layer(3, 2, 'identity', [1, 2, 3, -1, 0, 2], [1, -2])];
    assert.deepEqual([...A.mlpForward(net, new Float32Array([2, -1, 3]))], [10, 2]);
    net.push(layer(2, 1, 'relu', [2, -3], [-1]));
    assert.deepEqual([...A.mlpForward(net, new Float32Array([2, -1, 3]))], [13]);
});

test('U10: activation definitions include stable sigmoid and clamped tanh', () => {
    for (const activation of ['identity', 'relu', 'tanh', 'sigmoid']) {
        for (const x of [-1000, -15, -1, 0, 1, 15, 1000]) {
            const net = [layer(1, 1, activation, [0], [x])];
            const y = A.mlpForward(net, new Float32Array([0]))[0];
            const expected = activation === 'identity' ? x : activation === 'relu' ? Math.max(0, x)
                : activation === 'tanh' ? Math.tanh(Math.max(-15, Math.min(15, x))) : 1 / (1 + Math.exp(-x));
            near(y, expected);
        }
    }
});

test('U10: odd-width packing has zero padding and honors typed-array views', () => {
    const storage = new Float32Array([99, 1, 2, 3, 4, 5, 6, 99]);
    const net = [layer(3, 2, 'identity', [], [7, 8]), layer(2, 1, 'identity', [9, 10], [11])];
    net[0].weights = storage.subarray(1, 7);
    const layout = A.mlpLayout(net);
    assert.equal(layout.floats, 40);
    assert.equal(layout.layers[0].biasOffset, 16);
    assert.equal(layout.layers[1].weightOffset, 20);
    const packed = A.packMlpWeights(net);
    assert.deepEqual([...packed.slice(0, 20)], [1, 2, 3, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 0, 0]);
    assert.deepEqual([...packed.slice(20)], [9, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0]);
    packed.fill(99);
    A.packMlpWeights(net, packed);
    assert.equal(packed[3], 0);
    assert.equal(storage[0], 99);
});

test('U10: validation refuses every unsupported shape and parameter category', () => {
    const bad = net => assert.throws(() => A.mlpLayout(net), /MlpBrain:/);
    bad([]); bad(Array.from({ length: 5 }, () => layer(1, 1)));
    bad([layer(0, 1)]); bad([layer(1.5, 1)]); bad([layer(129, 1)]); bad([layer(1, 17)]);
    bad([layer(1, 65), layer(65, 1)]); bad([layer(1, 2), layer(3, 1)]);
    bad([layer(1, 1, 'softmax')]); bad([layer(2, 1, 'relu', [1])]);
    bad([layer(1, 1, 'relu', [NaN])]); bad([layer(1, 1, 'relu', [Infinity])]);
    bad([layer(1, 1, 'relu', [0], [Infinity])]); bad([layer(1, 1, 'relu', [0], [])]);
    const wrongType = layer(1, 1); wrongType.weights = [1]; bad([wrongType]);
    assert.doesNotThrow(() => A.mlpLayout([layer(128, 64), layer(64, 64), layer(64, 64), layer(64, 16)]));
});

test('U10: output and scratch reuse, dimensions, and overlapping views', () => {
    const net = [layer(2, 3, 'relu', [1, 0, 0, 1, 1, 1]), layer(3, 1, 'identity', [1, 2, 3])];
    const input = new Float32Array([2, 3]), out = new Float32Array(1), scratch = new Float32Array(6);
    for (let i = 0; i < 100; i++) {
        assert.equal(A.mlpForward(net, input, out, scratch), out);
        assert.equal(out[0], 23);
    }
    assert.throws(() => A.mlpForward(net, new Float32Array(1)), /input/);
    assert.throws(() => A.mlpForward(net, input, new Float32Array(0)), /output/);
    assert.throws(() => A.mlpForward(net, input, out, new Float32Array(5)), /scratch/);
    assert.throws(() => A.mlpForward(net, input, input.subarray(0, 1), scratch), /overlap/);
    assert.throws(() => A.mlpForward(net, input, out, input), /scratch/);
    assert.throws(() => A.mlpForward(net, input, scratch.subarray(0, 1), scratch), /overlap/);
    assert.throws(() => A.mlpForward(net, input, net[1].biases, scratch), /overlap/);
    assert.throws(() => A.packMlpWeights(net, net[0].weights), /output|overlap/);
    const shared = new Float32Array(9);
    shared.set(input);
    A.mlpForward(net, shared.subarray(0, 2), shared.subarray(2, 3), shared.subarray(3));
    assert.equal(shared[2], 23);
});

test('U10: deterministic mixed network agrees with independent f64 oracle', () => {
    let seed = 42;
    const random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32 - 0.5);
    const net = [[7, 5, 'tanh'], [5, 3, 'sigmoid'], [3, 2, 'identity']].map(([i, o, a]) =>
        layer(i, o, a, Array.from({ length: i * o }, random), Array.from({ length: o }, random)));
    for (let n = 0; n < 32; n++) {
        const input = Float32Array.from({ length: 7 }, random);
        let expected = [...input];
        for (const l of net) expected = Array.from({ length: l.output }, (_, r) => {
            const z = expected.reduce((sum, x, k) => sum + x * l.weights[r * l.input + k], l.biases[r]);
            return l.activation === 'tanh' ? Math.tanh(z) : l.activation === 'sigmoid' ? 1 / (1 + Math.exp(-z)) : z;
        });
        const actual = A.mlpForward(net, input);
        actual.forEach((x, i) => near(x, expected[i]));
        assert.deepEqual(A.mlpForward(net, input), actual);
    }
});

test('U10: supplied storage makes repeated forward allocate no typed arrays', () => {
    const net = [layer(2, 1, 'identity', [2, 3], [1])];
    const input = new Float32Array([4, 5]), out = new Float32Array(1), scratch = new Float32Array(4);
    const Native = globalThis.Float32Array;
    let allocations = 0;
    globalThis.Float32Array = new Proxy(Native, { construct(target, args) {
        allocations++;
        return Reflect.construct(target, args);
    } });
    try {
        for (let i = 0; i < 100; i++) A.mlpForward(net, input, out, scratch);
    } finally { globalThis.Float32Array = Native; }
    assert.equal(allocations, 0);
    assert.equal(out[0], 24);
});

test('U10: S3 shape logical count and packed buffer size are distinct', () => {
    const net = [layer(24, 32, 'tanh'), layer(32, 32, 'tanh'), layer(32, 2, 'tanh')];
    assert.equal(net.reduce((n, l) => n + l.weights.length + l.biases.length, 0), 1922);
    const packed = A.packMlpWeights(net);
    assert.equal(packed.byteLength, 7952);
    assert.equal(A.mlpLayout(net).scratchWidth, 32);
    assert.deepEqual([...A.mlpForward(net, new Float32Array(24))], [0, 0]);
});

function mock() {
    const buffers = [], writes = [], stages = [];
    const engine = {
        device: { limits: { maxStorageBufferBindingSize: 128 * 1024 * 1024, maxBufferSize: 256 * 1024 * 1024,
            maxComputeWorkgroupsPerDimension: 65535, maxComputeInvocationsPerWorkgroup: 256,
            maxComputeWorkgroupSizeX: 256, maxStorageBuffersPerShaderStage: 8 },
        createBuffer(o) { const b = { ...o, destroys: 0, destroy() { this.destroys++; } }; buffers.push(b); return b; },
        queue: { writeBuffer(buffer, offset, data) { writes.push({ buffer, offset, data: new Float32Array(data) }); } } },
        createComputeStage(o) { const s = { ...o, enabled: true, preClearBuffers: [] }; stages.push(s); return s; },
        setComputePassBindGroup(s, entries, group) { (s.groups ??= [])[group] = entries; },
    };
    return { engine, buffers, writes, stages };
}
const def = () => A.defineAgents({ capacity: 65, world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
    species: [{ name: 'unit', maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 4, radius: 0.4 }],
    sensors: [A.Sensors.self()] });

test('U9/U10: full MLP shader parses for all activations, odd and maximum widths', () => {
    for (const net of [
        [layer(2, 2)],
        [layer(2, 5, 'relu'), layer(5, 3, 'sigmoid'), layer(3, 2, 'tanh')],
        [layer(128, 64, 'tanh'), layer(64, 64, 'relu'), layer(64, 64, 'sigmoid'), layer(64, 16)],
    ]) {
        const wgsl = A.buildMlpThinkWGSL(net);
        const definition = A.defineAgents({ capacity: 65, world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
            species: [{ name: 'unit', maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 4, radius: 0.4 }],
            sensors: [A.Sensors.custom({ name: 'features', width: net[0].input,
                wgsl: `fn features(i: u32, a: Agent) -> array<f32, ${net[0].input}> { return array<f32, ${net[0].input}>(); }` })],
            actions: A.Actions.custom({ name: 'controls', width: net.at(-1).output, wgsl: 'fn controls(i: u32) {}' }) });
        const full = A.buildAgentThinkWGSL(definition, wgsl, 3);
        const reflect = new WgslReflect(full);
        assert.equal(reflect.entry.compute[0].name, 'cs_main');
        assert.match(wgsl, /@group\(1\) @binding\(0\)/);
        assert.match(wgsl, /setReason\(i, 0u\)/);
        assert.ok(!wgsl.includes('workgroupBarrier') && !wgsl.includes('`'));
        assert.match(wgsl, new RegExp(`k < ${net[0].input}u`));
    }
});

test('MlpBrain: copied parameters, transactional updates and stable GPU buffer identity', () => {
    const m = mock(), net = [layer(2, 2, 'tanh', [1, 2, 3, 4], [5, 6])];
    const brain = new A.MlpBrain(m.engine, net);
    assert.equal(brain.kind, 'stage');
    assert.deepEqual(brain.dims, { obs: 2, act: 2 });
    assert.equal(m.buffers.length, 1);
    const buffer = brain.buffers[0], wgsl = brain.wgsl;
    const original = m.writes[0].data.slice();
    net[0].weights.fill(99);
    assert.deepEqual(m.writes[0].data, original);
    brain.setWeights([layer(2, 2, 'tanh', [4, 3, 2, 1], [0, 1])]);
    assert.equal(brain.buffers[0], buffer);
    assert.equal(brain.wgsl, wgsl);
    assert.equal(m.writes.length, 2);
    assert.deepEqual(m.writes[1].data, A.packMlpWeights([layer(2, 2, 'tanh', [4, 3, 2, 1], [0, 1])]));
    for (const invalid of [[layer(2, 2, 'relu')], [layer(2, 1, 'tanh')], [layer(2, 2, 'tanh', [NaN, 0, 0, 0])]]) {
        assert.throws(() => brain.setWeights(invalid), /MlpBrain:/);
    }
    assert.equal(m.writes.length, 2);
    brain.destroy(); brain.destroy();
    assert.equal(buffer.destroys, 1);
    assert.throws(() => brain.setWeights(net), /destroy/);
});

test('MlpBrain: validation, buffer limits and upload failure clean up before attachment', () => {
    const m = mock();
    assert.throws(() => new A.MlpBrain(m.engine, []), /layers/);
    assert.equal(m.buffers.length, 0);
    m.engine.device.limits.maxBufferSize = 4;
    assert.throws(() => new A.MlpBrain(m.engine, [layer(2, 2)]), /limit/);
    assert.equal(m.buffers.length, 0);
    m.engine.device.limits.maxBufferSize = 256 * 1024 * 1024;
    m.engine.device.queue.writeBuffer = () => { throw new Error('upload failed'); };
    assert.throws(() => new A.MlpBrain(m.engine, [layer(2, 2)]), /upload failed/);
    assert.equal(m.buffers[0].destroys, 1);
});

test('MlpBrain: AgentSystem attachment, same-shape update, replacement and device/lifetime refusal', () => {
    const m = mock(), system = new A.AgentSystem(m.engine, def(), { thinkEvery: 3 });
    const brain = new A.MlpBrain(m.engine, [layer(2, 2)]);
    system.setBrain(brain);
    const stage = m.stages.at(-1);
    assert.equal(stage.groups[1][0].resource.buffer, brain.buffers[0]);
    system.advance(1 / 60);
    const chain = system.getComputeStages();
    brain.setWeights([layer(2, 2, 'identity', [1, 0, 0, 1])]);
    assert.equal(system.getComputeStages(), chain);
    assert.equal(m.stages.at(-1), stage);
    const other = mock(), wrong = new A.MlpBrain(other.engine, [layer(2, 2)]);
    const count = m.stages.length;
    assert.throws(() => system.setBrain(wrong), /device/);
    assert.equal(m.stages.length, count);
    assert.equal(stage.enabled, true);
    system.setBrain(new A.RuleBrain('fn think(i: u32, species: u32, tick: u32) {}'));
    assert.equal(stage.enabled, false);
    brain.destroy();
    assert.throws(() => system.setBrain(brain), /destroy/);
    system.destroy(); wrong.destroy();
});

// U21 (phase 7.1, docs/agents-species-brains-design.md §5): one network per species, as rows of
// one weight table, chosen by species name inside the same kernel.
const twoSpecies = () => A.defineAgents({ capacity: 65, world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
    species: [{ name: 'prey', maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 4, radius: 0.4 },
        { name: 'wolf', maxSpeed: 4, maxTurn: 3, accel: 5, senseRadius: 4, radius: 0.6 }],
    sensors: [A.Sensors.self()] });
const seq = (n, start) => Array.from({ length: n }, (_, k) => start + k);
const preyNet = () => [layer(2, 3, 'tanh', seq(6, 1), seq(3, 10)), layer(3, 2, 'sigmoid', seq(6, 20), seq(2, 30))];
const wolfNet = () => [layer(2, 3, 'tanh', seq(6, -6), seq(3, -10)), layer(3, 2, 'sigmoid', seq(6, -20), seq(2, -30))];
// buildMlpThinkWGSL([layer(3, 5, 'tanh'), layer(5, 2, 'sigmoid')]) as phase 6 generated it.
const GOLDEN = "@group(1) @binding(0) var<storage, read> mlpWeights: array<vec4<f32>>;\nfn think(i: u32, species: u32, tick: u32) {\n    var x0: array<vec4<f32>, 1>;\n    for (var k = 0u; k < 3u; k++) {\n        x0[k / 4u][k % 4u] = obs(i, k);\n    }\n    var x1: array<vec4<f32>, 2>;\n    for (var r = 0u; r < 5u; r++) {\n        var z = mlpWeights[8u + r / 4u][r % 4u];\n        for (var k = 0u; k < 1u; k++) {\n            z += dot(mlpWeights[0u + r * 1u + k], x0[k]);\n        }\n        z = tanh(clamp(z, -15.0, 15.0));\n        x1[r / 4u][r % 4u] = z;\n    }\n    var x2: array<vec4<f32>, 1>;\n    for (var r = 0u; r < 2u; r++) {\n        var z = mlpWeights[18u + r / 4u][r % 4u];\n        for (var k = 0u; k < 2u; k++) {\n            z += dot(mlpWeights[10u + r * 2u + k], x1[k]);\n        }\n        let e = exp(-abs(z));\n        z = select(e / (1.0 + e), 1.0 / (1.0 + e), z >= 0.0);\n        x2[r / 4u][r % 4u] = z;\n    }\n    for (var k = 0u; k < 2u; k++) {\n        setAct(i, k, x2[k / 4u][k % 4u]);\n    }\n    setReason(i, 0u);\n}";

test('U21: a single-network brain keeps its phase-6 WGSL byte for byte', () => {
    const net = [layer(3, 5, 'tanh'), layer(5, 2, 'sigmoid')];
    assert.equal(A.buildMlpThinkWGSL(net), GOLDEN);
    assert.equal(new A.MlpBrain(mock().engine, net).wgsl, GOLDEN);
});

test('U21: bySpecies packs one row per species, in the definition\'s order whatever the map\'s', () => {
    const m = mock();
    const brain = A.MlpBrain.bySpecies(m.engine, twoSpecies(), { wolf: wolfNet(), prey: preyNet() });
    const F = A.mlpLayout(preyNet()).floats;
    assert.equal(m.buffers.length, 1);
    assert.equal(m.buffers[0].size, 2 * F * 4);
    assert.equal(m.writes.length, 1);
    assert.deepEqual(m.writes[0].data.subarray(0, F), A.packMlpWeights(preyNet()));
    assert.deepEqual(m.writes[0].data.subarray(F), A.packMlpWeights(wolfNet()));
    assert.deepEqual(brain.dims, { obs: 2, act: 2 });
});

test('U21: rows are chosen by species name, and the shader parses in the full wrapper', () => {
    const definition = twoSpecies();
    const brain = A.MlpBrain.bySpecies(mock().engine, definition, { prey: preyNet(), wolf: wolfNet() });
    const quads = A.mlpLayout(preyNet()).floats / 4;
    assert.match(brain.wgsl, /case SPECIES_PREY: \{ row = 0u; \}/);
    assert.match(brain.wgsl, new RegExp(`case SPECIES_WOLF: \\{ row = ${quads}u; \\}`));
    assert.ok(brain.wgsl.includes('mlpWeights[row + '), 'every weight index is offset by the row');
    assert.ok(!/mlpWeights\[(?!row \+ )/.test(brain.wgsl), 'an index ignores the row');
    const full = A.buildAgentThinkWGSL(definition, brain.wgsl, 3);
    assert.equal(new WgslReflect(full).entry.compute[0].name, 'cs_main');
});

test('U21: setWeights for one species is one writeBuffer at its row; the shader is untouched', () => {
    const m = mock();
    const brain = A.MlpBrain.bySpecies(m.engine, twoSpecies(), { prey: preyNet(), wolf: wolfNet() });
    const F = A.mlpLayout(preyNet()).floats, buffer = brain.buffers[0], wgsl = brain.wgsl;
    const retrained = [layer(2, 3, 'tanh', seq(6, 50), seq(3, 60)), layer(3, 2, 'sigmoid', seq(6, 70), seq(2, 80))];
    brain.setWeights(retrained, { species: 'wolf' });
    assert.equal(m.writes.length, 2);
    assert.equal(m.writes[1].buffer, buffer);
    assert.equal(m.writes[1].offset, F * 4);
    assert.deepEqual(m.writes[1].data, A.packMlpWeights(retrained));
    brain.setWeights(retrained, { species: 'prey' });
    assert.equal(m.writes[2].offset, 0);
    assert.equal(brain.wgsl, wgsl);
});

test('U21: bySpecies and per-row setWeights refuse by name, before any upload', () => {
    const m = mock(), definition = twoSpecies();
    assert.throws(() => A.MlpBrain.bySpecies(m.engine, definition, { prey: preyNet() }),
        /MlpBrain: no network for species "wolf"; give one per species\./);
    assert.throws(() => A.MlpBrain.bySpecies(m.engine, definition, { prey: preyNet(), wolf: wolfNet(), bear: preyNet() }),
        /MlpBrain: "bear" is not a declared species\./);
    const wide = [layer(2, 4, 'tanh'), layer(4, 2, 'sigmoid')];
    assert.throws(() => A.MlpBrain.bySpecies(m.engine, definition, { prey: preyNet(), wolf: wide }),
        /MlpBrain: the network for "wolf" differs in shape from "prey" at layer 0; give it its own brain: setBrain\(brain, \{ species: 'wolf' \}\)\./);
    assert.throws(() => A.MlpBrain.bySpecies(m.engine, definition, { prey: preyNet(), wolf: [layer(2, 3, 'relu'), layer(3, 2, 'sigmoid')] }),
        /differs in shape from "prey" at layer 0/);
    assert.throws(() => A.MlpBrain.bySpecies(m.engine, definition, { prey: preyNet(), wolf: [layer(2, 3, 'tanh'), layer(3, 2, 'tanh')] }),
        /differs in shape from "prey" at layer 1/);
    assert.equal(m.buffers.length, 0);
    const brain = A.MlpBrain.bySpecies(m.engine, definition, { prey: preyNet(), wolf: wolfNet() });
    assert.throws(() => brain.setWeights(preyNet()),
        /MlpBrain: this brain holds one network per species; call setWeights\(network, \{ species \}\)\./);
    assert.throws(() => brain.setWeights(preyNet(), { species: 'bear' }), /MlpBrain: "bear" is not a declared species\./);
    assert.throws(() => brain.setWeights(wide, { species: 'wolf' }), /same shape and activations/);
    const single = new A.MlpBrain(m.engine, preyNet());
    assert.throws(() => single.setWeights(preyNet(), { species: 'prey' }),
        /MlpBrain: setWeights\(\{ species \}\) needs a brain from MlpBrain\.bySpecies\./);
    assert.equal(m.writes.length, 2, 'only the two constructions uploaded');
});

test('U21: a per-species brain attaches like any brain, its table bound at group 1', () => {
    const m = mock(), definition = twoSpecies(), system = new A.AgentSystem(m.engine, definition);
    const brain = A.MlpBrain.bySpecies(m.engine, definition, { prey: preyNet(), wolf: wolfNet() });
    system.setBrain(brain);
    assert.equal(m.stages.at(-1).groups[1][0].resource.buffer, brain.buffers[0]);
    system.destroy();
    brain.destroy();
});

// U43 (phase 11.2, docs/agents-genome-rows-design.md §5, §6, §9, §10): a network per agent
// (MlpBrain.byGenome), staged in workgroup memory with a per-lane fallback, and its row updates.
const thirdNet = () => [layer(2, 3, 'tanh', seq(6, 100), seq(3, 110)), layer(3, 2, 'sigmoid', seq(6, 120), seq(2, 130))];

test('U43: byGenome packs row r at r × floats, reports rowCount, and marks the brain cooperative', () => {
    const m = mock(), nets = [preyNet(), wolfNet(), thirdNet()];
    const brain = A.MlpBrain.byGenome(m.engine, nets);
    const F = A.mlpLayout(nets[0]).floats;
    assert.equal(brain.kind, 'stage');
    assert.equal(brain.cooperative, true);
    assert.equal(brain.rowCount, 3);
    assert.deepEqual(brain.dims, { obs: 2, act: 2 });
    assert.equal(m.buffers.length, 1);
    assert.equal(m.buffers[0].size, 3 * F * 4);
    assert.equal(m.writes.length, 1);
    for (let r = 0; r < 3; r++) {
        assert.deepEqual(m.writes[0].data.subarray(r * F, (r + 1) * F), A.packMlpWeights(nets[r]));
    }
    const copy = m.writes[0].data.slice();
    nets[1][0].weights.fill(99);
    assert.deepEqual(m.writes[0].data, copy, 'caller mutation after construction leaves the uploaded table untouched');
    assert.equal(new A.MlpBrain(mock().engine, preyNet()).rowCount, 1);
    assert.equal(new A.MlpBrain(mock().engine, preyNet()).cooperative, undefined);
    assert.equal(A.MlpBrain.bySpecies(mock().engine, twoSpecies(), { prey: preyNet(), wolf: wolfNet() }).rowCount, 2);
    brain.destroy();
});

test('U43: setWeights({ row }) writes one row and setRows writes a range of packed rows; the shader and buffer stay fixed', () => {
    const m = mock(), nets = [preyNet(), wolfNet(), thirdNet(), preyNet()];
    const brain = A.MlpBrain.byGenome(m.engine, nets);
    const F = A.mlpLayout(nets[0]).floats, buffer = brain.buffers[0], wgsl = brain.wgsl;
    const retrained = [layer(2, 3, 'tanh', seq(6, 50), seq(3, 60)), layer(3, 2, 'sigmoid', seq(6, 70), seq(2, 80))];
    brain.setWeights(retrained, { row: 2 });
    assert.equal(m.writes.length, 2);
    assert.equal(m.writes[1].buffer, buffer);
    assert.equal(m.writes[1].offset, 2 * F * 4);
    assert.deepEqual(m.writes[1].data, A.packMlpWeights(retrained));
    // setRows from a subarray view: two rows starting at row 1.
    const backing = new Float32Array((2 * F) + 8).fill(77);
    const twoRows = backing.subarray(4, 4 + 2 * F);
    A.packMlpWeights(wolfNet(), twoRows.subarray(0, F));
    A.packMlpWeights(thirdNet(), twoRows.subarray(F));
    brain.setRows(twoRows, { first: 1 });
    assert.equal(m.writes.length, 3);
    assert.equal(m.writes[2].buffer, buffer);
    assert.equal(m.writes[2].offset, 1 * F * 4);
    assert.deepEqual(m.writes[2].data, twoRows);
    // setRows with first omitted writes from row 0.
    const allFour = new Float32Array(4 * F);
    for (let r = 0; r < 4; r++) A.packMlpWeights(retrained, allFour.subarray(r * F, (r + 1) * F));
    brain.setRows(allFour);
    assert.equal(m.writes.length, 4);
    assert.equal(m.writes[3].offset, 0);
    assert.deepEqual(m.writes[3].data, allFour);
    assert.equal(brain.buffers[0], buffer);
    assert.equal(brain.wgsl, wgsl);
    brain.destroy();
});

test('U43: byGenome, setWeights({ row }) and setRows refuse every invalid argument of §10, before any upload', () => {
    const m = mock();
    assert.throws(() => A.MlpBrain.byGenome(m.engine, []), /^Error: MlpBrain: byGenome needs at least one network\.$/);
    assert.throws(() => A.MlpBrain.byGenome(m.engine, null), /^Error: MlpBrain: byGenome needs at least one network\.$/);
    assert.throws(() => A.buildMlpThinkWGSL(preyNet(), 0), /^Error: MlpBrain: byGenome needs at least one network\.$/);
    const wide = [layer(2, 4, 'tanh'), layer(4, 2, 'sigmoid')];
    assert.throws(() => A.MlpBrain.byGenome(m.engine, [preyNet(), wolfNet(), wide]),
        /^Error: MlpBrain: the network for row 2 differs in shape from row 0 at layer 0; give it its own brain\.$/);
    assert.throws(() => A.MlpBrain.byGenome(m.engine, [preyNet(), [layer(2, 3, 'relu'), layer(3, 2, 'sigmoid')]]),
        /^Error: MlpBrain: the network for row 1 differs in shape from row 0 at layer 0; give it its own brain\.$/);
    assert.throws(() => A.MlpBrain.byGenome(m.engine, [preyNet(), [layer(2, 3, 'tanh'), layer(3, 2, 'tanh')]]),
        /^Error: MlpBrain: the network for row 1 differs in shape from row 0 at layer 1; give it its own brain\.$/);
    const F = A.mlpLayout(preyNet()).floats;
    m.engine.device.limits.maxStorageBufferBindingSize = 2 * F * 4;
    assert.throws(() => A.MlpBrain.byGenome(m.engine, [preyNet(), wolfNet(), thirdNet()]),
        /^Error: MlpBrain: weights exceed device buffer limits\.$/);
    assert.equal(m.buffers.length, 0, 'a refused construction allocated a buffer');
    m.engine.device.limits.maxStorageBufferBindingSize = 128 * 1024 * 1024;
    const brain = A.MlpBrain.byGenome(m.engine, [preyNet(), wolfNet(), thirdNet()]);
    const single = new A.MlpBrain(m.engine, preyNet());
    const perSpecies = A.MlpBrain.bySpecies(m.engine, twoSpecies(), { prey: preyNet(), wolf: wolfNet() });
    assert.throws(() => brain.setWeights(preyNet()),
        /^Error: MlpBrain: this brain holds one network per row; call setWeights\(network, \{ row \}\)\.$/);
    assert.throws(() => brain.setWeights(preyNet(), { species: 'prey' }),
        /^Error: MlpBrain: setWeights\(\{ species \}\) needs a brain from MlpBrain\.bySpecies\.$/);
    assert.throws(() => single.setWeights(preyNet(), { row: 0 }),
        /^Error: MlpBrain: setWeights\(\{ row \}\) needs a brain from MlpBrain\.byGenome\.$/);
    assert.throws(() => perSpecies.setWeights(preyNet(), { row: 0 }),
        /^Error: MlpBrain: setWeights\(\{ row \}\) needs a brain from MlpBrain\.byGenome\.$/);
    assert.throws(() => single.setRows(A.packMlpWeights(preyNet())),
        /^Error: MlpBrain: setRows needs a brain from MlpBrain\.byGenome\.$/);
    assert.throws(() => perSpecies.setRows(A.packMlpWeights(preyNet())),
        /^Error: MlpBrain: setRows needs a brain from MlpBrain\.byGenome\.$/);
    for (const bad of [-1, 3, 4, 1.5, NaN]) {
        assert.throws(() => brain.setWeights(preyNet(), { row: bad }),
            new RegExp(`^Error: MlpBrain: row ${bad} is outside this brain's 3 rows\\.$`));
    }
    assert.throws(() => brain.setWeights(wide, { row: 1 }), /same shape and activations/);
    for (const bad of [new Float32Array(0), new Float32Array(F - 1), new Float32Array(F + 1)]) {
        assert.throws(() => brain.setRows(bad),
            new RegExp(`^Error: MlpBrain: setRows needs whole rows of ${F} floats; got ${bad.length}\\.$`));
    }
    const oneRow = A.packMlpWeights(preyNet());
    for (const bad of [-1, 3, 1.5, NaN]) {
        assert.throws(() => brain.setRows(oneRow, { first: bad }),
            new RegExp(`^Error: MlpBrain: row ${bad} is outside this brain's 3 rows\\.$`));
    }
    assert.throws(() => brain.setRows(new Float32Array(2 * F), { first: 2 }),
        /^Error: MlpBrain: row 3 is outside this brain's 3 rows\.$/);
    const nonFinite = oneRow.slice();
    nonFinite[5] = NaN;
    assert.throws(() => brain.setRows(nonFinite), /MlpBrain:.*finite/);
    assert.equal(m.writes.length, 3, 'only the three constructions uploaded');
    brain.destroy(); single.destroy(); perSpecies.destroy();
    assert.throws(() => brain.setRows(oneRow), /after destroy/);
});

test('U43: the staged genome shader parses in the cooperative wrapper, tiles wide layers, reads ids.z and guards out-of-range genomes', () => {
    for (const [net, rows] of [
        [preyNet(), 4],
        [[layer(128, 64, 'tanh'), layer(64, 64, 'relu'), layer(64, 64, 'sigmoid'), layer(64, 16)], 64],
    ]) {
        const m = mock();
        const brain = A.MlpBrain.byGenome(m.engine, Array.from({ length: rows }, () => net));
        const quads = A.mlpLayout(net).floats / 4;
        assert.match(brain.wgsl, /\bfn thinkBlock\(i: u32, species: u32, tick: u32, thinking: bool, lane: u32\)/);
        assert.match(brain.wgsl, new RegExp(`stagedRow = min\\(agents\\[i\\]\\.ids\\.z, ${rows - 1}u\\) \\* ${quads}u;`));
        assert.match(brain.wgsl, /let rowBase = workgroupUniformLoad\(&stagedRow\);/);
        assert.match(brain.wgsl, new RegExp(`let g = agents\\[i\\]\\.ids\\.z;\\s*if \\(g < ${rows}u\\)`));
        assert.match(brain.wgsl, /if \(myBase == rowBase\)/);
        assert.match(brain.wgsl, /else \{ setAct\(i, k, bitcast<f32>\(nan\)\); \}/);
        assert.ok(!brain.wgsl.includes('`'));
        if (net[0].input === 128) {
            // 128 × 64 = 2048 vec4s in layer 0 → 4 tiles of 16 rows × 32 vec4s (512 vec4s each).
            assert.match(brain.wgsl, /for \(var r = 0u; r < 16u; r\+\+\)/);
            assert.match(brain.wgsl, /for \(var r = 48u; r < 64u; r\+\+\)/);
        }
        const definition = A.defineAgents({ capacity: 65, world: { min: [-8, -8], max: [8, 8] }, cellSize: 4,
            species: [{ name: 'unit', maxSpeed: 3, maxTurn: 4, accel: 6, senseRadius: 4, radius: 0.4 }],
            sensors: [A.Sensors.custom({ name: 'features', width: net[0].input,
                wgsl: `fn features(i: u32, a: Agent) -> array<f32, ${net[0].input}> { return array<f32, ${net[0].input}>(); }` })],
            actions: A.Actions.custom({ name: 'controls', width: net.at(-1).output, wgsl: 'fn controls(i: u32) {}' }) });
        const sys = new A.AgentSystem(m.engine, definition, { thinkEvery: 3 });
        sys.setBrain(brain);
        const stage = m.stages.at(-1);
        assert.equal(stage.groups[1][0].resource.buffer, brain.buffers[0]);
        const reflect = new WgslReflect(stage.shaderCode);
        assert.equal(reflect.entry.compute[0].name, 'cs_main');
        assert.ok(stage.shaderCode.includes('thinkBlock(i, kind, state.tick, thinking, lane);'));
        sys.destroy();
        brain.destroy();
    }
});