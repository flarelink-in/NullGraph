import { test } from 'node:test';
import assert from 'node:assert/strict';
import { hash, noise, ranks, gradient, adam, permutation, pairedInterval, scoreQuality, correlation } from './math.mjs';

test('equal and partial ties give neutral paired gradients; monotone invariance', () => {
    assert.deepEqual(Array.from(ranks([7, 7, 7, 7])), [0, 0, 0, 0]);
    const partial = ranks([1, 1, 2, 2]);
    assert.equal(partial[0], partial[1]); assert.equal(partial[2], partial[3]);
    assert.ok(Math.abs(partial[0] + 1 / 3) < 1e-15);
    assert.deepEqual(ranks([1, 1, 2, 3]), ranks([3, 3, 5, 7]));
    assert.deepEqual(gradient([new Float64Array([1]), new Float64Array([2])], ranks([7, 7, 7, 7]), 0.02), new Float64Array(1));
    assert.throws(() => ranks([0, NaN]));
});
test('versioned tuple streams repeat without original seed alias; Gaussian odd size', () => {
    assert.deepEqual(noise(9, 1, 0, 0), noise(9, 1, 0, 0));
    assert.notDeepEqual(noise(9, 1, 0, 0), noise(9, 2147483649, 0, 0));
    assert.notEqual(hash(1, 0, 4096, 0), hash(1, 1, 0, 0));
    assert.throws(() => hash(-1));
    assert.deepEqual([...permutation(8, 1, 0)].sort((a, b) => a - b), [0, 1, 2, 3, 4, 5, 6, 7]);
    assert.ok(noise(9, 1, 0, 0).every(Number.isFinite));
});
test('Adam ascent reduces quadratic loss; neutral gradient does not erase old momentum', () => {
    const s = { x: new Float64Array([3, -2]), m: new Float64Array(2), v: new Float64Array(2), steps: 0 };
    const before = s.x.reduce((n, x) => n + x * x, 0);
    for (let t = 0; t < 100; t++) adam(s, Float64Array.from(s.x, x => -2 * x), 0.03, 0);
    assert.ok(s.x.reduce((n, x) => n + x * x, 0) < before / 10);
    const old = s.x.slice(); adam(s, new Float64Array(2), 0.03, 0);
    assert.notDeepEqual(old, s.x);
});
test('paired bootstrap preserves constant differences', () => {
    assert.deepEqual(pairedInterval(Array(20).fill(-2)), { mean: -2, low: -2, high: -2 });
});
test('score quality rejects ties and unstable halves; accepts stable nonzero signal', () => {
    const thresholds = { splitHalfRankCorrelation: 0.7, signalToResidual: 1, nonzeroIntervals: 4 };
    const directions = Array.from({ length: 8 }, (_, k) => ({ mean: k, sd: 0.1,
        differences: Array(20).fill(k), interval: { low: k - 0.01, high: k + 0.01 } }));
    assert.equal(scoreQuality(directions, thresholds).accepted, true);
    assert.equal(scoreQuality(directions.map(d => ({ ...d, differences: [...Array(10).fill(d.mean), ...Array(10).fill(-d.mean)] })), thresholds).accepted, false);
    assert.equal(scoreQuality(directions.map(d => ({ ...d, mean: 0, differences: Array(20).fill(0) })), thresholds).accepted, false);
    assert.equal(correlation([0, 0], [1, 1]), null);
});