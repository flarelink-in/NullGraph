// Phase 14.3 infrastructure smoke only, NOT research acceptance. Always uses new directories.
// `--only clones` runs just the clone-start part (used for watched-failing GPU controls).
import assert from 'node:assert/strict';
import { join, resolve } from 'node:path';
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { installNodeGPU } from '../../test/gpu/node-gpu.mjs';
import { NullGraph } from '../../dist/index.js';
import { buildLane, readBuffers } from './scenario/herd-wolves-arenas.mjs';
import { herdDefinition } from './scenario/herd-wolves.mjs';
import { networkFromModel } from './model.mjs';
import { runCoevolution } from './coevolve.mjs';
import { coevoProfile } from '../../test/fixtures/coevo-profile.mjs';

const only = process.argv.includes('--only') ? process.argv[process.argv.indexOf('--only') + 1] : null;
const base = resolve('tools/agents/out/coevo', `phase143-regression-${Date.now()}`);
mkdirSync(base, { recursive: true });
const host = installNodeGPU();
const report = { purpose: 'infrastructure smoke only; not research acceptance', base, only, checks: [], gpuErrors: [] };
const contactOptions = { iterations: 2, maxPush: 0.25, relaxation: 1, compliance: 0,
    damping: 0.05, mass: 'equal', maxScanPerCell: 64, cellSize: 2 };
const lines = (dir, file) => readFileSync(join(dir, file), 'utf8').trim().split('\n').map(line => JSON.parse(line));
const generations = dir => lines(dir, 'log.jsonl').filter(entry => entry.t !== undefined);

/** Counts mappable buffers on the trainer's own device: each probe's are logged by its "guard:"
 * line, each generation's by its "coevolve: gen" line. Nothing in the path keeps one for reuse. */
function readbackCounter() {
    const counter = { calibration: 0, generations: [], pending: 0 };
    counter.log = message => {
        if (message.includes('started on')) {
            const device = host.devices.at(-1), create = device.createBuffer.bind(device);
            device.createBuffer = descriptor => {
                if (descriptor.usage & GPUBufferUsage.MAP_READ) counter.pending++;
                return create(descriptor);
            };
        } else if (message.startsWith('guard:')) { counter.calibration += counter.pending; counter.pending = 0; }
        else if (message.startsWith('coevolve: gen')) { counter.generations.push(counter.pending); counter.pending = 0; }
    };
    return counter;
}

try {
    if (!only) {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        assert.ok(adapter);
        report.adapter = Object.fromEntries(['vendor', 'architecture', 'device', 'description'].map(key => [key, adapter.info[key]]));
        const device = await adapter.requestDevice(), engine = new NullGraph();
        try {
            assert.equal((await engine.init(null, { device, width: 8, height: 8 })).success, true);
            const geometry = { arenas: 4, wolves: 16, prey: 192, cols: 2, L: 24, gap: 10 };
            const lane = buildLane(engine, geometry, { environment: 'contact-v2', contacts: true, contactOptions,
                catchMargin: 0.05, dt: 0.1, thinkEvery: 3, seedOf: () => 11, rowOf: () => 0, monitorFinite: true });
            try {
                assert.equal(lane.solver.options.dt, 0.1);
                engine.render(lane.chain); await device.queue.onSubmittedWorkDone();
                const [records] = await readBuffers(device, [{ buffer: lane.sys.buffer, words: lane.lay.count * 16 }]);
                const floats = new Float32Array(records.buffer);
                assert.ok(Math.abs(floats[9] - 0.1) < 1e-6, `actual GPU age was ${floats[9]}`);
                report.checks.push('dt=0.1 reaches contacts and GPU age');
                device.queue.writeBuffer(lane.sys.buffer, 8 * 4, new Float32Array([NaN]));
                engine.render(lane.chain); await device.queue.onSubmittedWorkDone();
                const [stats] = await readBuffers(device, [{ buffer: lane.stats, words: 4 }]);
                assert.equal(stats[3], 1);
                report.checks.push('nonfinite state detected on GPU with shared score readback');
            } finally { lane.destroy(); }
        } finally { device.destroy(); }
        for (const label of ['one', 'two']) {
            const outDir = join(base, label);
            report[label] = await runCoevolution(coevoProfile({ name: label, outDir, T: 50, dt: 0.1,
                contacts: true, contactOptions, exportOnnx: true }));
            assert.equal(JSON.parse(readFileSync(join(outDir, 'status.json'))).state, 'complete');
        }
        for (const generation of ['gen-0000', 'gen-0010']) for (const file of [
            'prey.mlp.json', 'wolf.mlp.json', 'manifest.json', 'prey.onnx', 'wolf.onnx'])
            assert.deepEqual(readFileSync(join(base, 'one', generation, file)), readFileSync(join(base, 'two', generation, file)));
        const archived = readdirSync(join(base, 'one', 'archive')).sort();
        assert.deepEqual(archived, readdirSync(join(base, 'two', 'archive')).sort());
        for (const file of archived) assert.deepEqual(readFileSync(join(base, 'one', 'archive', file)), readFileSync(join(base, 'two', 'archive', file)));
        // H = 5 over ten updates: both species at 5 and 10, whichever species had just evolved.
        assert.deepEqual(lines(join(base, 'one'), 'archive/index.jsonl').map(entry => entry.id),
            ['prey-0000', 'wolf-0000', 'prey-0005', 'wolf-0005', 'prey-0010', 'wolf-0010']);
        const logs = ['one', 'two'].map(label => generations(join(base, label)));
        assert.equal(logs[0].length, 10);
        for (let t = 0; t < 10; t++) for (const key of ['scores', 'replicaCatches', 'noiseChecksum', 'gradientNorm', 'updateNorm', 'opponentIds', 'permutation', 'archive'])
            assert.deepEqual(logs[0][t][key], logs[1][t][key]);
        assert.deepEqual(logs[0][4].archive, [{ species: 'prey', added: 'prey-0005', evicted: null },
            { species: 'wolf', added: 'wolf-0005', evicted: null }]);
        assert.ok(logs[0].every(entry => !JSON.stringify(entry.archive).includes('params')), 'log carries archive IDs, not weights');
        report.checks.push('two fresh contacts-on T=50 ten-generation runs: exact scores/updates/models/manifests/ONNX/archive; both species archived every 5 updates');
        await host.settle();
        assert.equal(host.events.length, 0, JSON.stringify(host.events));
    }

    // H1's start: committed clones, E = 2 replicas, an archive small enough to evict, and I6
    // instrumented on the trainer's own device.
    {
        const outDir = join(base, 'clones'), counter = readbackCounter();
        const raw = coevoProfile({ name: 'clones', outDir, start: 'clones', hidden: [64, 64, 64], E: 2, cols: 4,
            T: 20, generations: 4, contacts: true, contactOptions, archive: { every: 1, cap: 2, share: 0.5 } });
        report.clones = await runCoevolution(raw, { log: counter.log });
        assert.equal(JSON.parse(readFileSync(join(outDir, 'status.json'))).state, 'complete');
        report.readbacks = { calibration: counter.calibration, generations: counter.generations.slice(0, 4) };
        assert.equal(counter.calibration, 0, 'calibration maps nothing');
        assert.deepEqual(counter.generations.slice(0, 4), [1, 1, 1, 1], 'exactly one score mapping per generation (I6)');
        report.checks.push('I6 instrumented: no mapping during calibration, exactly one per generation');
        const log = generations(outDir), demo = herdDefinition();
        for (const entry of log) {
            const sign = entry.species === 'wolf' ? 1 : -1;
            entry.replicaCatches.forEach((replicas, k) => {
                assert.equal(replicas.plus.length, 2); assert.equal(replicas.minus.length, 2);
                assert.equal(entry.scores[2 * k], sign * (replicas.plus[0] + replicas.plus[1]) / 2);
                assert.equal(entry.scores[2 * k + 1], sign * (replicas.minus[0] + replicas.minus[1]) / 2);
            });
        }
        assert.deepEqual(log.map(entry => entry.archive.map(change => change.evicted)),
            [[null, null], ['prey-0001', 'wolf-0001'], ['prey-0002', 'wolf-0002'], ['prey-0003', 'wolf-0003']]);
        assert.ok(log.some(entry => entry.opponentIds.some(id => /-000[1-4]$/.test(id))), 'an archived opponent beyond the start was sampled');
        for (const species of ['prey', 'wolf']) {
            const clone = networkFromModel(JSON.parse(readFileSync(`tools/agents/models/herd-wolves/${species}.mlp.json`, 'utf8')), demo, species);
            const start = networkFromModel(JSON.parse(readFileSync(join(outDir, 'gen-0000', `${species}.mlp.json`), 'utf8')), demo, species);
            start.forEach((layer, j) => assert.deepEqual(layer.weights, clone[j].weights));
            // Checkpoints load under the demo's own definition, as the clones do.
            const last = networkFromModel(JSON.parse(readFileSync(join(outDir, 'gen-0004', `${species}.mlp.json`), 'utf8')), demo, species);
            assert.equal(last.length, 4);
            assert.ok(existsSync(join(outDir, 'archive', `${species}-0004.mlp.json`)));
        }
        report.checks.push('clone start (E=2, archive cap 2 with evictions): replicas, archive opponents, demo-named checkpoints');
    }
    await host.settle();
    assert.equal(host.events.length, 0, JSON.stringify(host.events));

    if (!only) {
        // Genuine error on an already existing navigator host must not be ignored by the trainer.
        await assert.rejects(runCoevolution(coevoProfile({ name: 'failure', outDir: join(base, 'failure') }), {
            log(message) {
                if (message.includes('started on')) host.devices.at(-1).createBuffer({ size: 4, usage: 0 });
            },
        }), /WebGPU failure/);
        assert.equal(JSON.parse(readFileSync(join(base, 'failure', 'status.json'))).state, 'incomplete');
        report.expectedInjectedErrors = host.events.splice(0);
        assert.ok(report.expectedInjectedErrors.length > 0);
        report.checks.push('real uncaptured validation error rejects trainer and persists incomplete status');
    }
    report.ok = true;
} catch (error) {
    report.ok = false; report.error = String(error.stack ?? error);
} finally {
    await host.settle(); report.gpuErrors = host.events;
    if (host.events.length) report.ok = false;
    for (const device of host.devices) device.destroy();
    writeFileSync(join(base, 'report.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
    process.exit(report.ok ? 0 : 1);
}
