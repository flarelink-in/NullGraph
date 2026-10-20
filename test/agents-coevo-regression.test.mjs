import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, mkdirSync, readdirSync, cpSync, writeFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { validateConfig, parseCliArgs, experimentOf, configHash } from '../tools/agents/coevolve-config.mjs';
import { arenaLayout } from '../tools/agents/scenario/arena-layout.mjs';
import { arenaDefinition, modelNames } from '../tools/agents/scenario/herd-wolves-arenas.mjs';
import { herdDefinition } from '../tools/agents/scenario/herd-wolves.mjs';
import { networkFromModel } from '../tools/agents/model.mjs';
import { calibrateArenaSteps, createSubmissionGuard, probeLayouts, guardIdentity, WORKLOADS } from '../tools/agents/coevolve-guard.mjs';
import { managedRun, watchDevice, buildCompiled } from '../tools/agents/coevolve-runtime.mjs';
import { createRunDirectory, writeGenerationCheckpoint, writeRunJson } from '../tools/agents/coevolve-checkpoint.mjs';
import { saveCosts } from '../tools/film/guard.mjs';
import { coevoProfile } from './fixtures/coevo-profile.mjs';
import { runCoevolution } from '../tools/agents/coevolve.mjs';

const temporary = async body => {
    const dir = mkdtempSync(join(tmpdir(), 'coevo-regression-'));
    try { await body(dir); } finally { rmSync(dir, { recursive: true, force: true }); }
};
const geometry = { arenas: 32, wolves: 16, prey: 192, cols: 2, L: 24, gap: 10 };

// Everything before the first adapter request runs headless; a failing adapter ends the run there.
async function withoutAdapter(body) {
    const previous = Object.getOwnPropertyDescriptor(globalThis.navigator, 'gpu');
    Object.defineProperty(globalThis.navigator, 'gpu', { configurable: true,
        value: { async requestAdapter() { throw new Error('injected adapter failure'); } } });
    try { await body(); }
    finally {
        if (previous) Object.defineProperty(globalThis.navigator, 'gpu', previous);
        else delete globalThis.navigator.gpu;
    }
}

test('trainer orchestration: early device initialization failure leaves durable incomplete output', async () => temporary(async dir => {
    const outDir = join(dir, 'failed');
    await withoutAdapter(async () => {
        await assert.rejects(runCoevolution(coevoProfile({ outDir }), { log() {} }), /injected adapter failure/);
        assert.equal(JSON.parse(readFileSync(join(outDir, 'status.json'))).state, 'incomplete');
        assert.equal(JSON.parse(readFileSync(join(outDir, 'run.json'))).config.T, 300);
    });
}));

test('U53 regression: arena models carry the demo\'s feature names, not the custom stages\' indexed ones', () => {
    const def = arenaDefinition(arenaLayout(geometry)), demo = herdDefinition();
    // The library names custom fields by index; before the re-review fix models carried these.
    assert.deepEqual(def.obsNames.slice(-2), ['bounds.0', 'bounds.1']);
    assert.deepEqual(def.actNames, ['steer.0', 'steer.1']);
    const names = modelNames(def);
    assert.deepEqual(names.obsNames, demo.obsNames);
    assert.deepEqual(names.actNames, demo.actNames);
    for (const species of ['prey', 'wolf']) {
        const model = JSON.parse(readFileSync(new URL(`../tools/agents/models/herd-wolves/${species}.mlp.json`, import.meta.url)));
        assert.throws(() => networkFromModel(model, def, species), /bounds\.side/, 'raw arena names refuse the clones');
        assert.equal(networkFromModel(model, names, species).length, 4);
    }
    // Only the four reviewed renames: any other drift from the demo's definition is refused.
    assert.throws(() => modelNames({ obsNames: [...def.obsNames.slice(0, -1), 'bounds.2'], actNames: def.actNames }), /obsNames\[23\]/);
    assert.throws(() => modelNames({ obsNames: def.obsNames, actNames: ['steer.1', 'steer.0'] }), /actNames\[0\]/);
    assert.throws(() => modelNames({ obsNames: def.obsNames.slice(1), actNames: def.actNames }), /obsNames\[0\]/);
});

test('trainer orchestration: clone start loads the committed clones and exports them under the demo\'s names', async () => temporary(async dir => {
    const outDir = join(dir, 'clones');
    const raw = coevoProfile({ outDir, start: 'clones', hidden: [64, 64, 64] });
    await withoutAdapter(async () => {
        await assert.rejects(runCoevolution(raw, { log() {} }), /injected adapter failure/);
    });
    const config = validateConfig({ ...raw, outDir: join(dir, 'unused') });
    const demo = herdDefinition();
    for (const species of ['prey', 'wolf']) {
        const source = readFileSync(new URL(`../tools/agents/models/herd-wolves/${species}.mlp.json`, import.meta.url), 'utf8');
        const written = JSON.parse(readFileSync(join(outDir, 'gen-0000', `${species}.mlp.json`)));
        const clone = networkFromModel(JSON.parse(source), demo, species);
        const start = networkFromModel(written, demo, `gen-0000 ${species}`);
        start.forEach((layer, j) => {
            assert.deepEqual(layer.weights, clone[j].weights);
            assert.deepEqual(layer.biases, clone[j].biases);
        });
        const run = JSON.parse(readFileSync(join(outDir, 'run.json')));
        assert.equal(run.start.files[species].hash, createHash('sha256').update(source).digest('hex'));
        assert.equal(JSON.parse(readFileSync(join(outDir, 'archive', `${species}-0000.mlp.json`))).obsNames.at(-1), 'bounds.forward');
    }
    const manifest = JSON.parse(readFileSync(join(outDir, 'gen-0000', 'manifest.json')));
    assert.deepEqual(manifest.experiment, JSON.parse(JSON.stringify(experimentOf(config))));
    assert.equal(manifest.configHash, configHash(manifest.experiment), 'a manifest alone identifies its experiment');
    const index = readFileSync(join(outDir, 'archive', 'index.jsonl'), 'utf8').trim().split('\n').map(line => JSON.parse(line));
    assert.deepEqual(index.map(entry => entry.id), ['prey-0000', 'wolf-0000']);
    await assert.rejects(withoutAdapter(() => runCoevolution(coevoProfile({ outDir: join(dir, 'shape'), start: 'clones' }), { log() {} })),
        /clone shape differs/);
}));

test('U50 regression: explicit profile, strict CLI, nested types and counter bounds', () => {
    assert.throws(() => validateConfig({ name: 'missing' }), /explicit research profile/);
    for (const field of ['K', 'T', 'dt', 'environment', 'contactOptions', 'optimizer', 'seeds']) {
        const raw = coevoProfile(); delete raw[field];
        assert.throws(() => validateConfig(raw), new RegExp(field));
    }
    for (const key of ['optimizer', 'archive', 'seeds', 'contactOptions'])
        assert.throws(() => validateConfig(coevoProfile({ [key]: null })), new RegExp(key));
    assert.throws(() => validateConfig(coevoProfile({ optimizer: { sigma: 0.1 } })), /optimizer.rate/);
    for (const [key, field] of [['optimizer', 'sigma'], ['archive', 'every'], ['seeds', 'master']]) {
        const raw = coevoProfile(); raw[key] = { ...raw[key], [field]: null };
        assert.throws(() => validateConfig(raw), new RegExp(`${key}.${field}`));
    }
    assert.throws(() => validateConfig(coevoProfile({ T: 22369622 })), /per-arena catch bound/);
    assert.throws(() => validateConfig(coevoProfile({ T: 3000000 })), /lane catch bound/);
    assert.throws(() => validateConfig(coevoProfile({ exportOnnx: 'false' })), /boolean/);
    assert.throws(() => validateConfig(coevoProfile({ contactOptions: { typo: 1 } })), /unknown contactOptions/);
    for (const args of [['--sigam', '0.5'], ['--T'], ['--T', '--K', '4'], ['unexpected'],
        ['--T', '5', '--T', '6'], ['--contacts', '--no-contacts'], ['--contacts', 'false']])
        assert.throws(() => parseCliArgs(args), /^Error: coevolve:/);
    assert.deepEqual(parseCliArgs(['--T', '50', '--no-contacts']), { T: 50, contacts: false });
});

test('U50 regression: accepted timestep reaches definition and changes WGSL', () => {
    const config = validateConfig(coevoProfile({ dt: 0.1 }));
    const lay = arenaLayout(geometry);
    assert.equal(arenaDefinition(lay, { dt: config.dt }).dt, 0.1);
    assert.equal(arenaDefinition(lay).dt, 1 / 60);
    assert.throws(() => arenaDefinition(lay, { dt: NaN }), /invalid dt/);
});

test('U53 regression: independent labels/paths share model, manifest and ONNX bytes', async () => temporary(async dir => {
    const a = validateConfig(coevoProfile({ name: 'one', outDir: join(dir, 'one') }));
    const b = validateConfig(coevoProfile({ name: 'two', outDir: join(dir, 'two') }));
    assert.equal(a.configHash, b.configHash);
    assert.notEqual(a.configHash, validateConfig(coevoProfile({ dt: 0.1 })).configHash);
    const network = [{ input: 24, output: 2, activation: 'tanh', weights: new Float32Array(48), biases: new Float32Array(2) }];
    for (const config of [a, b]) {
        createRunDirectory(config.targetDir);
        writeGenerationCheckpoint(config.targetDir, { generation: 0, preyNetwork: network, wolfNetwork: network,
            obsNames: [], actNames: [], configHash: config.configHash, experiment: experimentOf(config), exportOnnx: true });
        writeRunJson(config.targetDir, { config, seedManifest: { seedsByGeneration: [[42]], retries: [] },
            adapterInfo: Object.create({ vendor: 'test-vendor', description: 'test-device' }) });
    }
    for (const file of ['manifest.json', 'prey.mlp.json', 'wolf.mlp.json', 'prey.onnx', 'wolf.onnx'])
        assert.deepEqual(readFileSync(join(a.targetDir, 'gen-0000', file)), readFileSync(join(b.targetDir, 'gen-0000', file)));
    const metadata = JSON.parse(readFileSync(join(a.targetDir, 'run.json')));
    assert.equal(metadata.adapter.vendor, 'test-vendor');
    assert.deepEqual(metadata.seedManifest.seedsByGeneration, [[42]]);
    assert.throws(() => writeGenerationCheckpoint(a.targetDir, { generation: 0 }), /already exists/);
}));

test('U55 regression: progressive arena-valid calibration, all workloads, and unsafe cache refuses before probe', async () => temporary(async dir => {
    const layouts = probeLayouts(geometry);
    assert.deepEqual(layouts.map(p => p.arenas), [4, 8, 16, 32]);
    layouts.forEach(p => assert.doesNotThrow(() => arenaLayout(p)));
    const calls = [];
    const recorded = [];
    const calibrated = await calibrateArenaSteps({ geometry, async measure(layout, workload) {
        calls.push([layout.arenas, workload]); return { ms: layout.arenas, setupCompileMs: 12 };
    }, record: sample => recorded.push(sample) });
    assert.equal(calls.length, layouts.length * WORKLOADS.length);
    assert.equal(recorded.length, calls.length);
    assert.equal(calibrated.stepMs, 32);
    const cacheFile = join(dir, 'cache.json');
    saveCosts(cacheFile, 'key:ordinary', [{ count: 832, ms: 1200 }]);
    await assert.rejects(calibrateArenaSteps({ geometry, cacheFile, key: 'key',
        measure: () => assert.fail('predicted unsafe probe was submitted') }), /unsafe.*predicted/);
    let submissions = 0;
    await assert.rejects(calibrateArenaSteps({ geometry, measure: async () => { submissions++; return { ms: 1501 }; } }), /1,500 ms stop/);
    assert.equal(submissions, 1);
    await assert.rejects(calibrateArenaSteps({ geometry: { ...geometry, prey: 4096 },
        measure: () => assert.fail('unbounded cold initial probe') }), /without small-probe evidence/);
    const config = validateConfig(coevoProfile());
    const key = guardIdentity(config, { vendor: 'test', driver: 'one' }, []);
    assert.notEqual(key, guardIdentity(config, { vendor: 'test', driver: 'two' }, []));
    assert.notEqual(key, guardIdentity({ ...config, dt: 0.1 }, { vendor: 'test', driver: 'one' }, []));
}));

test('U55 regression: actual submission guard floors, backs off and refuses unsafe continuation', () => {
    const guard = createSubmissionGuard(10);
    assert.equal(guard.next(100), 50);
    guard.observe(50, 800);
    assert.equal(guard.next(100), 25);
    assert.throws(() => guard.observe(1, 1501), /1,500 ms stop/);
    const slow = createSubmissionGuard(600);
    assert.equal(slow.next(100), 1);
    slow.observe(1, 1100);
    assert.throws(() => slow.next(1), /unsafe/);
    assert.throws(() => createSubmissionGuard(NaN), /unsafe/);
});

function fakeDevice() {
    const device = new EventTarget();
    device.queue = { onSubmittedWorkDone: async () => {} };
    device.lost = new Promise(resolve => { device.lose = resolve; });
    return device;
}

test('U55 regression: externally supplied device errors and loss fail production health checks', async () => {
    const device = fakeDevice(), health = watchDevice(device);
    const event = new Event('uncapturederror'); event.error = new Error('injected validation error');
    device.dispatchEvent(event);
    await assert.rejects(health.check(), /injected validation error/);
    health.destroy();
    const lost = fakeDevice(), watcher = watchDevice(lost);
    lost.lose({ reason: 'unknown', message: 'injected loss' });
    await assert.rejects(watcher.check(), /injected loss/);
    watcher.destroy();
});

test('U55 regression: early allocation/export failures persist incomplete status and clean partial ownership', async () => temporary(async dir => {
    const order = [];
    await assert.rejects(managedRun({ targetDir: dir }, async ({ own, progress }) => {
        own({ destroy() { order.push('device'); } });
        own({ destroy() { order.push('prey brain'); } });
        progress(2);
        throw new Error('wolf allocation failed');
    }), /wolf allocation failed/);
    assert.deepEqual(order, ['prey brain', 'device']);
    const status = JSON.parse(readFileSync(join(dir, 'status.json')));
    assert.equal(status.state, 'incomplete'); assert.equal(status.completedUpdates, 2);
    assert.match(readFileSync(join(dir, 'log.jsonl'), 'utf8'), /wolf allocation failed/);
    await assert.rejects(managedRun({ targetDir: dir }, async () => {
        writeGenerationCheckpoint(dir, { generation: 1, preyNetwork: null });
    }));
    assert.equal(JSON.parse(readFileSync(join(dir, 'status.json'))).state, 'incomplete');
    assert.ok(!readdirSync(dir).some(name => name.endsWith('.tmp')));
}));

test('U55 regression: compilation completes without simulation, restores hook and destroys on failure', async () => {
    let compiled = 0, destroyed = false;
    const device = { createComputePipeline: () => 'pipeline', createComputePipelineAsync: async () => { compiled++; } };
    const original = device.createComputePipeline;
    const result = await buildCompiled(device, () => { device.createComputePipeline({}); return { destroy() {} }; });
    assert.equal(compiled, 1); assert.equal(device.createComputePipeline, original); assert.ok(result.setupCompileMs >= 0);
    device.createComputePipelineAsync = async () => { throw new Error('bad shader'); };
    await assert.rejects(buildCompiled(device, () => { device.createComputePipeline({}); return { destroy() { destroyed = true; } }; }), /bad shader/);
    assert.equal(destroyed, true); assert.equal(device.createComputePipeline, original);
});

test('lifecycle: completion follows cleanup, while cleanup errors retain incomplete verdict', async () => temporary(async dir => {
    const result = await managedRun({ targetDir: dir }, async ({ own, progress }) => {
        own({ destroy() {
            assert.equal(JSON.parse(readFileSync(join(dir, 'status.json'))).state, 'incomplete');
        } });
        progress(3); return 'done';
    });
    assert.equal(result, 'done');
    assert.deepEqual(JSON.parse(readFileSync(join(dir, 'status.json'))), { state: 'complete', completedUpdates: 3 });
    await assert.rejects(managedRun({ targetDir: dir }, async ({ own }) => {
        own({ destroy() { throw new Error('injected cleanup failure'); } });
    }), /injected cleanup failure/);
    assert.equal(JSON.parse(readFileSync(join(dir, 'status.json'))).state, 'incomplete');
}));

test('U54 evaluation orchestration: checkpoints verified, held-out seeds, durable incomplete on adapter failure', async () => temporary(async dir => {
    const { planEvaluation, runEvaluation, loadRun, loadCheckpoint, loadClones } = await import('../tools/agents/coevolve-eval.mjs');
    // A real run directory: the clone start writes run.json and gen-0000 before it asks for a GPU.
    const runDir = join(dir, 'run');
    await withoutAdapter(() => assert.rejects(runCoevolution(coevoProfile({ outDir: runDir, start: 'clones', hidden: [64, 64, 64] }),
        { log() {} }), /injected adapter failure/));
    const raw = { mode: 'baselines', run: runDir, generations: [0], purpose: 'validation', seed: 7, domain: 3,
        analysisSeed: 99, clones: true, outDir: join(dir, 'eval') };
    const plan = planEvaluation(raw);
    assert.equal(plan.plan.length, 10); assert.equal(plan.comparisons.length, 6);
    assert.deepEqual(plan.geometry, { arenas: 20, wolves: 16, prey: 192, L: 24, gap: 10, cols: 5, contacts: null });
    assert.equal(plan.T, 300);
    const training = new Set(loadRun(runDir).run.seedManifest.seedsByGeneration.flat());
    assert.equal(plan.heldOut.seeds.length, 20);
    assert.equal(plan.heldOut.excluded, training.size, 'every training seed of the run is excluded');
    assert.ok(plan.heldOut.seeds.every(seed => !training.has(seed)), 'held-out seeds are not training seeds');
    // The clone start's generation 0 is the committed clones, read back under the demo's names.
    const start = loadCheckpoint(loadRun(runDir), 0), clones = loadClones();
    for (const species of ['prey', 'wolf']) start[species].network.forEach((layer, j) =>
        assert.deepEqual(layer.weights, clones[species].network[j].weights));

    // A tampered model, manifest or config is refused before any GPU work.
    const tampered = join(dir, 'tampered');
    cpSync(runDir, tampered, { recursive: true });
    const model = join(tampered, 'gen-0000', 'prey.mlp.json');
    writeFileSync(model, readFileSync(model, 'utf8').replace('"weights": [', '"weights": [ '));
    assert.throws(() => loadCheckpoint(loadRun(tampered), 0), /does not match its manifest/);
    cpSync(join(runDir, 'gen-0000', 'prey.mlp.json'), model);
    const manifestPath = join(tampered, 'gen-0000', 'manifest.json'), manifest = JSON.parse(readFileSync(manifestPath));
    writeFileSync(manifestPath, JSON.stringify({ ...manifest, experiment: { ...manifest.experiment, T: 301 } }));
    assert.throws(() => loadCheckpoint(loadRun(tampered), 0), /does not belong/);
    const runPath = join(tampered, 'run.json'), runJson = JSON.parse(readFileSync(runPath));
    writeFileSync(runPath, JSON.stringify({ ...runJson, config: { ...runJson.config, dt: 0.1 } }));
    assert.throws(() => loadRun(tampered), /does not hash its config/);
    assert.throws(() => loadCheckpoint(loadRun(runDir), 10), /no checkpoint gen-0010/);

    for (const [change, message] of [[{ purpose: undefined }, /purpose/], [{ seed: undefined }, /--seed/],
        [{ analysisSeed: 2 ** 32 }, /--analysis-seed/], [{ generations: [0, 0] }, /repeats/], [{ generations: [] }, /at least one/],
        [{ episodes: 16 }, /at least 20/], [{ T: 2 ** 31 }, /u32/], [{ anchors: [0] }, /crossplay/],
        [{ mode: 'crossplay', clones: false }, /baselines/], [{ mode: 'crossplay', anchors: [10] }, /anchor gen-0010/],
        [{ margin: 0 }, /--margin/], [{ mode: 'crossplay', margin: 1 }, /--margin/],
        [{ run: join(dir, 'missing') }, /no run.json/]])
        assert.throws(() => planEvaluation({ ...raw, ...change }), message, JSON.stringify(change));
    assert.equal(planEvaluation({ ...raw, mode: 'crossplay', clones: true }).plan.length, 1);

    // The GPU is asked for only after plan.json is durable; a failure leaves the directory incomplete.
    await withoutAdapter(() => assert.rejects(runEvaluation(raw, { log() {} }), /injected adapter failure/));
    assert.equal(JSON.parse(readFileSync(join(dir, 'eval', 'status.json'))).state, 'incomplete');
    const written = JSON.parse(readFileSync(join(dir, 'eval', 'plan.json')));
    assert.equal(written.drivers['clone:prey'].hash, clones.prey.hash);
    assert.deepEqual(written.heldOut.seeds, plan.heldOut.seeds);
    assert.equal(written.plan.length, 10);
    assert.ok(!existsSync(join(dir, 'eval', 'evaluation.json')));
    assert.throws(() => planEvaluation(raw), /already exists/);
}));