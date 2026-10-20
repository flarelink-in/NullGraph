// Phase 14.3/14.4 infrastructure smoke only, NOT research acceptance. Always uses new directories.
// `--only clones` runs just the clone-start part (used for watched-failing GPU controls);
// `--only evaluate` runs it and then the arena-matched evaluation of its checkpoints (A48-A51).
import assert from 'node:assert/strict';
import { join, resolve } from 'node:path';
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { installNodeGPU } from '../../test/gpu/node-gpu.mjs';
import { NullGraph } from '../../dist/index.js';
import { MlpBrain } from '../../dist/agents.js';
import { buildLane, buildMirroredLanes, readBuffers } from './scenario/herd-wolves-arenas.mjs';
import { herdDefinition } from './scenario/herd-wolves.mjs';
import { initNetwork } from './mlp-train.mjs';
import { networkFromModel } from './model.mjs';
import { runCoevolution } from './coevolve.mjs';
import { loadClones, runEvaluation } from './coevolve-eval.mjs';
import { replay } from './coevolve-evaluate.mjs';
import { coevoProfile } from '../../test/fixtures/coevo-profile.mjs';

const only = process.argv.includes('--only') ? process.argv[process.argv.indexOf('--only') + 1] : null;
if (only && !['clones', 'evaluate', 'v3'].includes(only)) throw new Error(`--only takes clones, evaluate or v3, not "${only}"`);
const base = resolve('tools/agents/out/coevo', `phase143-regression-${Date.now()}`);
mkdirSync(base, { recursive: true });
const host = installNodeGPU();
const report = { purpose: 'infrastructure smoke only; not research acceptance', base, only, checks: [], gpuErrors: [] };
const contactOptions = { iterations: 2, maxPush: 0.25, relaxation: 1, compliance: 0,
    damping: 0.05, mass: 'equal', maxScanPerCell: 64, cellSize: 2 };
const lines = (dir, file) => readFileSync(join(dir, file), 'utf8').trim().split('\n').map(line => JSON.parse(line));
const generations = dir => lines(dir, 'log.jsonl').filter(entry => entry.t !== undefined);

/** Counts mappable buffers on the tool's own device: each probe's are logged by its "guard:" line,
 * each generation's or matchup's by its progress line (`unit`). Nothing in the path keeps one. */
function readbackCounter(started = 'started on', unit = /^coevolve: gen/) {
    const counter = { calibration: 0, generations: [], pending: 0 };
    counter.log = message => {
        if (message.includes(started)) {
            const device = host.devices.at(-1), create = device.createBuffer.bind(device);
            device.createBuffer = descriptor => {
                if (descriptor.usage & GPUBufferUsage.MAP_READ) counter.pending++;
                return create(descriptor);
            };
        } else if (message.startsWith('guard:')) { counter.calibration += counter.pending; counter.pending = 0; }
        else if (unit.test(message)) { counter.generations.push(counter.pending); counter.pending = 0; }
    };
    return counter;
}

/** Steps a lane `steps` times, one submission each, then reads every record word and counter. */
async function rollout(engine, lane, steps) {
    for (let s = 0; s < steps; s++) engine.render(lane.chain);
    const [records, counters] = await readBuffers(engine.device, [{ buffer: lane.sys.buffer, words: lane.lay.count * 16 },
        { buffer: lane.counters, words: lane.lay.arenas }]);
    return { records, counters };
}

try {
    // A52: contact-v3, contact-v2 with initial prey spawns kept spawnSeparation from the tile's wolves.
    if (!only || only === 'v3') {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        const device = await adapter.requestDevice(), engine = new NullGraph();
        try {
            assert.equal((await engine.init(null, { device, width: 8, height: 8 })).success, true);
            const geometry = { arenas: 8, wolves: 16, prey: 192, cols: 4, L: 36, gap: 10 };
            const common = { contacts: true, contactOptions, catchMargin: 0.05, thinkEvery: 3, seedOf: a => 500 + a, rowOf: () => 0,
                monitorFinite: true, brains: { prey: 'rule', wolf: 'rule' } };
            const v3 = buildLane(engine, geometry, { ...common, environment: 'contact-v3', spawnSeparation: 2 });
            const v2 = buildLane(engine, geometry, { ...common, environment: 'contact-v2' });
            try {
                const lay = v3.lay, read = async lane => new Float32Array((await readBuffers(device,
                    [{ buffer: lane.sys.buffer, words: lay.count * 16 }]))[0].buffer);
                const [p3, p2] = [await read(v3), await read(v2)];
                let redrawn = 0;
                for (let a = 0; a < lay.arenas; a++) for (let s = 0; s < lay.perArena; s++) {
                    const i = s < lay.wolves ? a * lay.wolves + s : lay.wolfTotal + a * lay.prey + s - lay.wolves;
                    const same = [0, 2].every(c => p3[i * 16 + c] === p2[i * 16 + c]);
                    if (s < lay.wolves) { assert.ok(same, 'wolves spawn as in contact-v2'); continue; }
                    const gap = (p, w) => Math.hypot(p[i * 16] - p[w * 16], p[i * 16 + 2] - p[w * 16 + 2]);
                    const tileWolves = Array.from({ length: lay.wolves }, (_, k) => a * lay.wolves + k);
                    assert.ok(tileWolves.every(w => gap(p3, w) >= 2 - 1e-4), `prey ${i} keeps 2 m from its tile's wolves`);
                    const close = tileWolves.some(w => gap(p2, w) < 2);
                    assert.equal(!same, close, `prey ${i} is re-drawn exactly when contact-v2 put it within 2 m`);
                    if (close) redrawn++;
                }
                assert.ok(redrawn > 0, 'some prey were re-drawn');
                const [one3, one2] = [await rollout(engine, v3, 1), await rollout(engine, v2, 1)];
                assert.ok(one3.counters.every(n => n === 0), 'no catch on the first step');
                assert.ok(one2.counters.some(n => n > 0), 'contact-v2 gives free first-step catches on the same seeds');
                report.v3Redrawn = redrawn;
            } finally { v3.destroy(); v2.destroy(); }
            // With a separation no draw falls inside (1 mm), contact-v3 plays contact-v2 bit for bit.
            const near3 = buildLane(engine, geometry, { ...common, environment: 'contact-v3', spawnSeparation: 0.001 });
            const near2 = buildLane(engine, geometry, { ...common, environment: 'contact-v2' });
            try {
                const [a3, a2] = [await rollout(engine, near3, 120), await rollout(engine, near2, 120)];
                assert.deepEqual(a3.records, a2.records); assert.deepEqual(a3.counters, a2.counters);
            } finally { near3.destroy(); near2.destroy(); }
            const table = MlpBrain.byGenome(engine, [initNetwork([{ input: 24, output: 16, activation: 'tanh' },
                { input: 16, output: 2, activation: 'tanh' }], 51)]);
            const pair = buildMirroredLanes(engine, geometry, { ...common, environment: 'contact-v3', spawnSeparation: 2,
                brains: { prey: table, wolf: table }, rowOf: () => 0 });
            try {
                const [plus, minus] = [await rollout(engine, pair.lanes[0], 120), await rollout(engine, pair.lanes[1], 120)];
                assert.deepEqual(plus.records, minus.records); assert.deepEqual(plus.counters, minus.counters);
            } finally { pair.destroy(); table.destroy(); }
            report.checks.push('A52 contact-v3: separated spawns on the GPU, contact-v2 otherwise (bit for bit at 1 mm), no first-step catch, mirrored lanes exact');
        } finally { device.destroy(); }

        const v3Dir = join(base, 'v3');
        await runCoevolution(coevoProfile({ name: 'v3', outDir: v3Dir, environment: 'contact-v3', spawnSeparation: 2, L: 36,
            T: 20, generations: 2, contacts: true, contactOptions }), { log() {} });
        assert.equal(JSON.parse(readFileSync(join(v3Dir, 'gen-0002', 'manifest.json'))).experiment.spawnSeparation, 2);
        const v3Eval = await runEvaluation({ mode: 'baselines', run: v3Dir, generations: [2], purpose: 'validation', seed: 7,
            domain: 14, analysisSeed: 99, clones: true, T: 60, outDir: join(base, 'v3-eval') }, { log() {} });
        assert.equal(v3Eval.evaluation.experiment.environment, 'contact-v3');
        assert.deepEqual(replay(JSON.parse(readFileSync(join(base, 'v3-eval', 'evaluation.json'), 'utf8'))).results, v3Eval.evaluation.results);
        report.checks.push('contact-v3 trains, checkpoints its separation and evaluates in it');
        await host.settle();
        assert.equal(host.events.length, 0, JSON.stringify(host.events));
    }

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
    const clonesDir = join(base, 'clones');
    {
        const outDir = clonesDir, counter = readbackCounter();
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

    // 14.4: arena-matched evaluation. A48/A49 on a device of their own, then the clone run's checkpoints.
    if (!only || only === 'evaluate') {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        const device = await adapter.requestDevice(), engine = new NullGraph();
        try {
            assert.equal((await engine.init(null, { device, width: 8, height: 8 })).success, true);
            const geometry = { arenas: 4, wolves: 16, prey: 192, cols: 2, L: 24, gap: 10 };
            const common = { environment: 'contact-v2', contacts: true, contactOptions, catchMargin: 0.05, thinkEvery: 3,
                seedOf: a => 300 + a, rowOf: () => 0, monitorFinite: true };
            const play = async options => {
                const lane = buildLane(engine, geometry, { ...common, ...options });
                try { return await rollout(engine, lane, 120); } finally { lane.destroy(); }
            };
            const combined = await play({}), split = await play({ brains: { prey: 'rule', wolf: 'rule' } });
            assert.deepEqual(split.records, combined.records); assert.deepEqual(split.counters, combined.counters);
            assert.notDeepEqual((await play({ brains: { prey: 'label', wolf: 'label' } })).records, combined.records, 'the label is not the rule');
            await play({ brains: { prey: 'rule', wolf: 'label' } });
            report.checks.push('A48 per-species rule drivers equal the combined rule bit for bit; the label differs');

            const shape = [{ input: 24, output: 16, activation: 'tanh' }, { input: 16, output: 16, activation: 'tanh' },
                { input: 16, output: 2, activation: 'tanh' }];
            const [prey, wolf, other] = [41, 42, 43].map(seed => initNetwork(shape, seed));
            const tables = [MlpBrain.byGenome(engine, [other, other, prey, other]), MlpBrain.byGenome(engine, [other, wolf, other]),
                MlpBrain.byGenome(engine, [prey]), MlpBrain.byGenome(engine, [wolf])];
            try {
                const words = r => Array.from(r.records).filter((_, w) => w % 16 !== 14);   // word 14: the genome row
                const trained = await play({ brains: { prey: tables[0], wolf: tables[1] }, rowOf: (a, sp) => (sp === 'prey' ? 2 : 1) });
                const evaluated = await play({ brains: { prey: tables[2], wolf: tables[3] } });
                assert.deepEqual(words(evaluated), words(trained)); assert.deepEqual(evaluated.counters, trained.counters);
                const wrong = await play({ brains: { prey: tables[0], wolf: tables[1] } });
                assert.notDeepEqual(words(wrong), words(trained), 'another row plays differently');
            } finally { for (const table of tables) table.destroy(); }
            report.checks.push('A49 one-row evaluation tables play exactly as the trainer\'s rows');
        } finally { device.destroy(); }

        const counter = readbackCounter('" on ', /^coevolve: +\d+\//);
        // 300 steps (5 s), longer than the clone run's 20, so batching and the guard see real episodes.
        const held = { run: clonesDir, purpose: 'validation', seed: 7, domain: 14, analysisSeed: 99, clones: true,
            generations: [0, 4], T: 300 };
        const baselines = await runEvaluation({ ...held, mode: 'baselines', outDir: join(base, 'eval-baselines') }, { log: counter.log });
        const { results } = baselines.evaluation;
        const catches = (r, id) => r.matchups.find(m => m.id === id)?.catches ?? assert.fail(`no matchup ${id}`);
        assert.equal(results.matchups.length, 17); assert.equal(results.comparisons.length, 12);
        report.evalReadbacks = { calibration: counter.calibration, matchups: counter.generations };
        assert.equal(counter.calibration, 0, 'calibration maps nothing');
        assert.deepEqual(counter.generations, Array(17).fill(1), 'exactly one mapping per matchup');
        // The clone start's generation 0 is the clones: against the clones it plays their own pairing.
        for (const id of ['gen-0000/gen-0000', 'gen-0000/clone', 'clone/gen-0000'])
            assert.deepEqual(catches(results, id), catches(results, 'clone/clone'), `${id} plays as clone/clone`);
        // No catch-level rule/label check: spawn overlaps and the 5 s rest make their counts tie here;
        // A48 shows the label driver at the record level.
        for (const c of results.comparisons.filter(c => c.subject === 'gen-0000' && c.baseline === 'clone'))
            assert.deepEqual([c.difference.low, c.difference.high, c.superior], [0, 0, false]);
        assert.equal(JSON.parse(readFileSync(join(base, 'eval-baselines', 'status.json'))).state, 'complete');
        const storedBaselines = JSON.parse(readFileSync(join(base, 'eval-baselines', 'evaluation.json'), 'utf8'));
        assert.deepEqual(replay(storedBaselines), { results: storedBaselines.results, crossplay: null }, 'raw-data replay');
        report.baselineMeans = Object.fromEntries(results.matchups.map(m => [m.id, m.interval.mean]));
        {
            // The evaluator's mapping replayed by hand: generation 0's prey (the clone prey, bit for
            // bit) against rule wolves, an asymmetric pairing, in single steps.
            const e = baselines.evaluation, x = e.experiment, replay = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
            const replayDevice = await replay.requestDevice(), replayEngine = new NullGraph();
            try {
                assert.equal((await replayEngine.init(null, { device: replayDevice, width: 8, height: 8 })).success, true);
                const table = MlpBrain.byGenome(replayEngine, [loadClones().prey.network]);
                const lane = buildLane(replayEngine, e.evaluation.geometry, { environment: x.environment, contacts: x.contacts,
                    contactOptions: x.contactOptions, catchMargin: x.catchMargin, spawnSeparation: x.spawnSeparation,
                    thinkEvery: x.thinkEvery, dt: x.dt,
                    monitorFinite: true, brains: { prey: table, wolf: 'rule' }, seedOf: a => e.heldOut.seeds[a], rowOf: () => 0 });
                try {
                    const { counters } = await rollout(replayEngine, lane, e.evaluation.T);
                    assert.deepEqual(Array.from(counters), catches(results, 'gen-0000/rule'), 'the evaluator plays the drivers it names');
                } finally { lane.destroy(); table.destroy(); }
            } finally { replayDevice.destroy(); }
        }
        report.checks.push('A50 baselines on the clone run: generation 0 plays exactly as the clones; one mapping per matchup');

        const cross = await runEvaluation({ ...held, mode: 'crossplay', anchors: [0], outDir: join(base, 'eval-crossplay') }, { log() {} });
        const matrix = cross.evaluation.crossplay;
        assert.deepEqual(matrix.generations, ['gen-0000', 'gen-0004']);
        for (const id of ['gen-0000/gen-0000', 'gen-0004/gen-0004'])
            assert.deepEqual(catches(cross.evaluation.results, id), catches(results, id), `${id} replays exactly`);
        assert.equal(matrix.mean[1][1], results.matchups.find(m => m.id === 'gen-0004/gen-0004').interval.mean);
        const storedCross = JSON.parse(readFileSync(join(base, 'eval-crossplay', 'evaluation.json'), 'utf8'));
        assert.deepEqual(replay(storedCross), { results: storedCross.results, crossplay: storedCross.crossplay }, 'raw-data replay');
        report.crossplayMean = matrix.mean;
        report.checks.push('A51 cross-play replays the baselines\' shared pairings exactly; anchors recorded');
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
