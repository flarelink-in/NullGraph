// D2.3: guard policy with injected timings; no unsafe GPU submissions needed to test stops.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { calibrateSteps, predictStep, checkStep, readCosts, saveCosts } from '../tools/film/guard.mjs';

test('F6 policy: prediction refuses before submitting a probe; measured time stops immediately', async () => {
    let calls = 0;
    await assert.rejects(calibrateSteps({ count: 65536, limitMs: 10,
        samples: [{ count: 4096, ms: 2 }], measure: async () => { calls++; return 1; } }), /predicted.*over the 10 ms/);
    assert.equal(calls, 0);
    await assert.rejects(calibrateSteps({ count: 65536, limitMs: 10, stopMs: 15,
        measure: async () => { calls++; return 16; } }), /over the 15 ms stop/);
    assert.equal(calls, 1);
    assert.equal(checkStep(1500), 1500);
    assert.throws(() => checkStep(1501), /1,500 ms stop/);
    assert.throws(() => checkStep(NaN), /invalid/);
    await assert.rejects(calibrateSteps({ count: 65536, limitMs: 1001 }), /only be lowered/);
});

test('U3: conservative predictions use largest eligible size and worst equal-size sample', () => {
    const samples = [{ count: 64, ms: 1 }, { count: 256, ms: 2 }, { count: 256, ms: 3 }, { count: 4096, ms: 99 }, { count: 128, ms: NaN }];
    assert.equal(predictStep(samples, 1024).predictedMs, 3 * 4 ** 1.25);
    assert.equal(predictStep(samples, 32), null);
});

test('U3: calibration climbs safely, persists observations and reuses compatible evidence', async () => {
    const measured = [], saved = [];
    const result = await calibrateSteps({ count: 1048576,
        measure: async count => { measured.push(count); return count / 4096; },
        save: samples => saved.push(samples.map(s => ({ ...s }))) });
    assert.deepEqual(measured, [4096, 16384, 65536, 262144]);
    assert.equal(result.cached, false);
    assert.equal(result.from.count, 262144);
    const reused = await calibrateSteps({ count: 1048576, samples: saved.at(-1),
        measure: async () => assert.fail('compatible cache must avoid calibration') });
    assert.equal(reused.cached, true);
    assert.equal(reused.predictedMs, result.predictedMs);
    const probes = [];
    await assert.rejects(calibrateSteps({ count: 2 ** 40,
        measure: async count => { probes.push(count); return count / 4096; } }), /predicted/);
    assert.equal(probes[0], 4096, 'even enormous requests start at a small count');
    assert.ok(probes.length < 10, 'unsafe extrapolation halts before large allocations');
});

test('U3: disk cache isolates adapters/workloads, retains maxima and discards malformed evidence', () => {
    const dir = mkdtempSync(join(tmpdir(), 'film-costs-')), file = join(dir, 'costs.json');
    try {
        assert.deepEqual(readCosts(file, 'nvidia-network'), []);
        saveCosts(file, 'nvidia-network', [{ count: 4096, ms: 5 }]);
        saveCosts(file, 'amd-network', [{ count: 4096, ms: 10 }]);
        saveCosts(file, 'nvidia-network', [{ count: 4096, ms: 2 }, { count: 16384, ms: 20 }]);
        assert.deepEqual(readCosts(file, 'nvidia-network'), [{ count: 4096, ms: 5 }, { count: 16384, ms: 20 }]);
        assert.deepEqual(readCosts(file, 'amd-network'), [{ count: 4096, ms: 10 }]);
        assert.deepEqual(readCosts(file, 'nvidia-ort'), []);
        writeFileSync(file, '{broken');
        assert.deepEqual(readCosts(file, 'nvidia-network'), []);
        saveCosts(file, 'new-build', [{ count: -1, ms: 5 }, { count: 64, ms: Infinity }, { count: 64, ms: 1 }]);
        assert.deepEqual(readCosts(file, 'new-build'), [{ count: 64, ms: 1 }]);
        writeFileSync(file, JSON.stringify({ version: 999, entries: { 'new-build': [{ count: 64, ms: 99 }] } }));
        assert.deepEqual(readCosts(file, 'new-build'), []);
    } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('F6: marginal quarter-size refusal permits only a safely predicted half-size refinement', async () => {
    const calls = [];
    const result = await calibrateSteps({ count: 1048576, samples: [{ count: 262144, ms: 181 }],
        measure: async count => { calls.push(count); return 300; } });
    assert.deepEqual(calls, [524288]);
    assert.ok(result.predictedMs < 1000);
    await assert.rejects(calibrateSteps({ count: 1048576, samples: [{ count: 262144, ms: 500 }],
        measure: async () => assert.fail('unsafe half-size probe') }), /predicted/);
});