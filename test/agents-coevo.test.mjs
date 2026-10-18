import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mlpLayout } from '../dist/agents.js';
import { flatParams, networkOf } from '../tools/agents/mlp-train.mjs';
import { NOISE_VERSION, MAX_PARAMETERS, tupleHash, noise, centeredRanks, gradient,
    createAdamState, adamAscent, masterParams, candidateNetworks } from '../tools/agents/es.mjs';

const near = (a, b, tolerance = 1e-14) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);
const snapshot = state => ({ x: state.x.slice(), m: state.m.slice(), v: state.v.slice(), steps: state.steps });
const shape = [{ input: 2, output: 2, activation: 'tanh' }, { input: 2, output: 1, activation: 'identity' }];

test('U51 E4 versioned tuple hash: exact goldens and independent BigInt multiplication oracle', () => {
    assert.equal(NOISE_VERSION, 'es-tuple-box-muller-v1');
    const fixtures = [[[0], 2812203024], [[1, 2, 3], 995390908],
        [[0xffffffff, 0, 0xffffffff], 1340915906], [[0x45530101, 1, 1, 0, 0, 0], 2188908081]];
    const mask = 0xffffffffn;
    for (const [fields, golden] of fixtures) {
        let h = 0x85ebca6bn;
        for (const field of fields) {
            h = ((h ^ BigInt(field)) * 0x9e3779b1n) & mask;
            h ^= h >> 16n; h = (h * 0x7feb352dn) & mask; h ^= h >> 15n;
        }
        assert.equal(tupleHash(...fields), golden);
        assert.equal(tupleHash(...fields), Number(h));
    }
    for (const value of [-1, 2 ** 32, 0.5, NaN, Infinity, '1']) assert.throws(() => tupleHash(value), /^Error: coevolve:/);
});

test('U51 E4 Gaussian golden on Node 24, odd prefixes, deterministic tuple separation and budget refusals', () => {
    // Node v24.12.0 / V8; allow last-bit transcendental differences, not cross-engine bit promises.
    const golden = [0.6839898543014522, 0.9382111212418293, -0.6543989593212926,
        0.7900636767115494, -0.31766940287577583];
    const actual = noise(1, 0, 0, 5);
    actual.forEach((value, j) => near(value, golden[j], 2e-15));
    assert.deepEqual(actual, noise(1, 0, 0, 5));
    assert.deepEqual(actual, noise(1, 0, 0, 6).slice(0, 5));
    for (const args of [[0x80000001, 0, 0, 5], [1, 1, 0, 5], [1, 0, 1, 5]]) assert.notDeepEqual(actual, noise(...args));
    assert.notDeepEqual(noise(1, 1, 0, 5), noise(1, 0, 4096, 5));
    assert.ok(noise(0xffffffff, 0xffffffff, 0xffffffff, 1).every(Number.isFinite));
    for (const args of [[-1, 0, 0, 1], [0, 2 ** 32, 0, 1], [0, 0, 0.5, 1],
        [0, 0, 0, 0], [0, 0, 0, MAX_PARAMETERS + 1], [0, 0, 0, Infinity]])
        assert.throws(() => noise(...args), /^Error: coevolve:/);
    const sample = noise(191, 4, 7, 20000);
    near(sample.reduce((s, x) => s + x, 0) / sample.length, 0, 0.04);
    near(sample.reduce((s, x) => s + x * x, 0) / sample.length, 1, 0.06);
});

test('U51 E2/E3 average ranks: ties, permutations, monotone invariance and neutral paired gradients', () => {
    assert.deepEqual([...centeredRanks([7, 7, 7, 7])], [0, 0, 0, 0]);
    const ranks = centeredRanks([1, 1, 2, 3]);
    near(ranks[0], -1 / 3); near(ranks[1], -1 / 3); near(ranks[2], 1 / 6); near(ranks[3], 0.5);
    assert.deepEqual(ranks, centeredRanks([3, 3, 5, 7]));
    const p = [3, 1, 0, 2], scores = [1, 1, 2, 3];
    assert.deepEqual([...centeredRanks(p.map(i => scores[i]))], p.map(i => ranks[i]));
    const directions = [new Float64Array([1, -4]), new Float64Array([2, 5])];
    assert.deepEqual(gradient(directions, centeredRanks([7, 7, 7, 7]), 0.25), new Float64Array(2));
    assert.deepEqual(gradient(directions, centeredRanks([1, 1, 2, 2]), 0.25), new Float64Array(2));
    // Independent normalization fixture: K=2, sigma=.5, utility differences 1 and .5.
    assert.deepEqual([...gradient([[1, 2], [-1, 4]], [0.5, -0.5, 0.25, -0.25], 0.5)], [0.25, 2]);
    // Watched-failing index-ranked ties would produce a nonzero update.
    assert.notDeepEqual(gradient(directions, [-0.5, -1 / 6, 1 / 6, 0.5], 0.25), new Float64Array(2));
});

test('U51 rank/gradient refusals: empty, sparse, wrong dimensions and nonfinite arithmetic', () => {
    for (const scores of [[], [1], [1, NaN], [1, Infinity], [1, , 2], null])
        assert.throws(() => centeredRanks(scores), /^Error: coevolve:/);
    for (const args of [[[], [], 1], [[[1], [1, 2]], [0, 1, 2, 3], 1], [[[1]], [0], 1],
        [[[NaN]], [0, 1], 1], [[[1]], [0, 1], 0], [[[1]], [0, 1], -1],
        [[[1]], [0, 1], Infinity], [[[1]], [0, 1], Number.MAX_VALUE],
        [[[Number.MAX_VALUE]], [Number.MAX_VALUE, -Number.MAX_VALUE], 0.01]])
        assert.throws(() => gradient(...args), /^Error: coevolve:/);
});

test('U51 E1 Adam golden two steps: coupled decay, moments, correction and independent species', () => {
    const master = new Float64Array([1, -2]), state = createAdamState(master), wolf = createAdamState(master);
    assert.notEqual(master, state.x);
    adamAscent(state, [0.5, -0.25], { rate: 0.01, decay: 0.1 });
    [1.00999999975, -2.0099999980000005].forEach((x, j) => near(state.x[j], x));
    [0.04, -0.005].forEach((x, j) => near(state.m[j], x));
    [0.00016, 0.0000025].forEach((x, j) => near(state.v[j], x));
    adamAscent(state, [-0.125, 0.75], { rate: 0.01, decay: 0.1 });
    [1.012171214686597, -2.002920513332063].forEach((x, j) => near(state.x[j], x));
    [0.013400000002500002, 0.09059999997999998].forEach((x, j) => near(state.m[j], x));
    [0.00021091599998870024, 0.000906898499619601].forEach((x, j) => near(state.v[j], x));
    assert.equal(state.steps, 2); assert.equal(wolf.steps, 0); assert.deepEqual(wolf.x, master);
});

test('U51 E1/E2 quadratic ascent and zero-gradient behavior distinguish momentum, decay and neutrality', () => {
    const s = createAdamState([3, -2]);
    const loss = () => s.x.reduce((n, x) => n + x * x, 0), before = loss();
    for (let t = 0; t < 100; t++) adamAscent(s, Float64Array.from(s.x, x => -2 * x), { rate: 0.03, decay: 0 });
    assert.ok(loss() < before / 10);
    const old = s.x.slice(); adamAscent(s, [0, 0], { rate: 0.03, decay: 0 });
    assert.notDeepEqual(s.x, old, 'old momentum still moves with zero fitness gradient');
    const neutral = createAdamState([2]);
    adamAscent(neutral, [0], { rate: 0.01, decay: 0 }); assert.equal(neutral.x[0], 2);
    const decaying = createAdamState([2]);
    adamAscent(decaying, [0], { rate: 0.01, decay: 0.1 }); assert.ok(decaying.x[0] < 2);
    near(decaying.m[0], -0.02); near(decaying.v[0], 0.00004);
});

test('U51 optimizer refuses malformed/overflow states atomically before mutation', () => {
    const state = createAdamState([1, -2]), original = snapshot(state);
    for (const options of [{ rate: 0, decay: 0 }, { rate: 0.1, decay: -1 }, { rate: Infinity, decay: 0 },
        { rate: 0.1 }, {}]) {
        assert.throws(() => adamAscent(state, [0, 0], options), /^Error: coevolve:/);
        assert.deepEqual(snapshot(state), original);
    }
    for (const g of [[0], [0, NaN], [0, Number.MAX_VALUE]]) {
        assert.throws(() => adamAscent(state, g, { rate: 0.1, decay: 0 }), /^Error: coevolve:/);
        assert.deepEqual(snapshot(state), original);
    }
    for (const change of [{ steps: 2 ** 32 - 1 }, { steps: -1 }, { v: new Float64Array([-1, 0]) },
        { m: new Float64Array([Infinity, 0]) }, { x: new Float32Array([1, 2]) }, { m: state.x }])
        assert.throws(() => adamAscent({ ...state, ...change }, [0, 0], { rate: 0.1, decay: 0 }), /^Error: coevolve:/);
    assert.deepEqual(snapshot(state), original);
    assert.throws(() => createAdamState([Infinity]), /^Error: coevolve:/);
});

test('U51 parameter order and f32 candidates reuse canonical packing without mutating f64 masters', () => {
    const master = Float64Array.from({ length: 9 }, (_, i) => i / 10 + 1e-10);
    const direction = Float64Array.from({ length: 9 }, (_, i) => i % 2 ? -0.5 : 0.25);
    const saved = master.slice(), result = candidateNetworks(shape, master, direction, 0.2);
    const network = networkOf(shape, master);
    assert.deepEqual(masterParams(network), flatParams(network));
    const expected = Float64Array.from(master, (x, j) => Math.fround(x + 0.2 * direction[j]));
    assert.deepEqual(masterParams(result.plus), expected);
    assert.deepEqual(masterParams(result.minus), Float64Array.from(master, (x, j) => Math.fround(x - 0.2 * direction[j])));
    assert.doesNotThrow(() => mlpLayout(result.plus));
    assert.equal(result.diagnostics.collisions, 0);
    near(result.diagnostics.perturbationNorm, Math.sqrt([...direction].reduce((s, x) => s + (0.2 * x) ** 2, 0)));
    near(result.diagnostics.plusNorm, Math.sqrt([...expected].reduce((s, x, j) => s + (x - Math.fround(master[j])) ** 2, 0)));
    assert.deepEqual(master, saved);
    const collision = candidateNetworks([{ input: 1, output: 1, activation: 'identity' }], [1, 1], [1, 1], 1e-10);
    assert.equal(collision.diagnostics.collisions, 2);
    assert.equal(collision.diagnostics.plusNorm, 0); assert.equal(collision.diagnostics.minusNorm, 0);
    assert.deepEqual(masterParams(collision.plus), masterParams(collision.minus));
});

test('U51 parameter/candidate refusals reject shape drift, length errors and f32 overflow', () => {
    const basic = [{ input: 1, output: 1, activation: 'identity' }];
    for (const badShape of [[], [{ input: 1, output: 1, activation: 'linear' }],
        [{ input: 1, output: 0, activation: 'tanh' }], [...basic, ...shape],
        [{ input: MAX_PARAMETERS, output: MAX_PARAMETERS, activation: 'tanh' }],
        [{ input: 129, output: 1, activation: 'identity' }],
        [{ input: 1, output: 17, activation: 'identity' }], Array(5).fill(basic[0])])
        assert.throws(() => candidateNetworks(badShape, [1, 2], [0, 0], 1), /^Error: coevolve:/);
    for (const args of [[basic, [1], [0, 0], 1], [basic, [1, 2], [NaN, 0], 1],
        [basic, [1e40, 2], [0, 0], 1], [basic, [1, 2], [1e40, 0], 1],
        [basic, [1, 2], [0, 0], 0]])
        assert.throws(() => candidateNetworks(...args), /^Error: coevolve:/);
    assert.throws(() => masterParams([{ ...basic[0], weights: [1, 2], biases: [0] }]), /^Error: coevolve:/);
});

test('U51 repeatable CPU pipeline, separate timebases and no direct GPU/filesystem/scenario imports', () => {
    const run = () => {
        const species = [createAdamState([0.25, -0.5]), createAdamState([-0.25, 0.5])];
        for (let t = 0; t < 8; t++) {
            const s = species[t % 2], directions = Array.from({ length: 4 }, (_, k) => noise(123, t, k, 2));
            const scores = directions.flatMap(e => [1, -1].map(sign =>
                -s.x.reduce((sum, x, j) => sum + (x + sign * 0.1 * e[j]) ** 2, 0)));
            adamAscent(s, gradient(directions, centeredRanks(scores), 0.1), { rate: 0.01, decay: 0.005 });
        }
        return species.map(snapshot);
    };
    assert.deepEqual(run(), run()); assert.deepEqual(run().map(s => s.steps), [4, 4]);
    const source = readFileSync(new URL('../tools/agents/es.mjs', import.meta.url), 'utf8');
    const imports = [...source.matchAll(/^import .* from '([^']+)';/gm)].map(m => m[1]);
    assert.deepEqual(imports, ['./mlp-train.mjs']);
});