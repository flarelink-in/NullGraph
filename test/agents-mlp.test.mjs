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