// Phase 14.4, design §10. Arena-matched evaluation of a co-evolution run's checkpoints.
//
//   npm run agents:arena-eval -- --run <run dir> --generations 10,20 --purpose validation \
//       --seed <u32> --domain <u32> --analysis-seed <u32> [--episodes 20] [--T <steps>] [--no-clones]
//   npm run agents:crossplay -- --run <run dir> --generations 0,10,20 [--anchors 0] ...same flags
//
// arena-eval plays each listed checkpoint against the rule, the label (the rule without wander)
// and the clones, both ways, beside each baseline's own pairing; crossplay plays every listed prey
// generation against every listed wolf generation. Episodes are held out: their seeds have their
// own stream per purpose and are checked against the run's training seeds. Results go to a fresh
// directory with raw seed-level catches, intervals and every driver's identity; nothing is reused.
// NG_DAWN_FLAGS=adapter=NVIDIA picks the GPU. Final-test seeds judge; they never tune.
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { NullGraph } from '../../dist/index.js';
import * as A from '../../dist/agents.js';
import { herdDefinition } from './scenario/herd-wolves.mjs';
import { buildLane } from './scenario/herd-wolves-arenas.mjs';
import { networkFromModel } from './model.mjs';
import { configHash } from './coevolve-config.mjs';
import { buildHeldOutManifest } from './coevolve-schedule.mjs';
import { BOOTSTRAP, BOOTSTRAP_VERSION } from './coevolve-stats.mjs';
import { appendLog, atomicWriteFile, getRepoMetadata, writeStatus } from './coevolve-checkpoint.mjs';
import { managedRun, watchDevice, buildCompiled } from './coevolve-runtime.mjs';
import { calibrateArenaSteps, guardIdentity, createSubmissionGuard, workloadFixture, timeSingles, runGuarded } from './coevolve-guard.mjs';
import { LABEL_PAIR, RULE_PAIR, baselineComparisons, baselineMatchups, crossplayMatchups, crossplaySummary,
    evaluationGeometry, playMatchup, summarize } from './coevolve-evaluate.mjs';

export const EVALUATION_VERSION = 'coevo-eval-v1';
const fail = message => { throw new Error(`coevolve: ${message}`); };
const root = fileURLToPath(new URL('../../', import.meta.url));
const sha256 = text => createHash('sha256').update(text).digest('hex');
const SOURCES = ['./coevolve-eval.mjs', './coevolve-evaluate.mjs', './coevolve-stats.mjs', './coevolve-schedule.mjs',
    './coevolve-guard.mjs', './coevolve-runtime.mjs', './model.mjs', './scenario/arena-layout.mjs',
    './scenario/herd-wolves-arenas.mjs', './scenario/herd-wolves.mjs', '../film/guard.mjs',
    '../../dist/index.js', '../../dist/agents.js', '../../dist/physics.js'];
const u32 = (value, name) => {
    if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) fail(`${name} must be an explicit u32`);
    return value;
};
const gen = n => `gen-${String(n).padStart(4, '0')}`;

/** Strict flags: unknown, repeated or valueless options are refused; numbers must be whole. */
export function parseEvalArgs(argv) {
    const [mode, ...args] = argv;
    if (mode !== 'baselines' && mode !== 'crossplay') fail(`mode must be baselines or crossplay, not "${mode}"`);
    const values = new Set(['run', 'generations', 'anchors', 'purpose', 'seed', 'domain', 'analysis-seed',
        'episodes', 'T', 'margin', 'name', 'out-dir']), flags = new Set(['no-clones']), raw = { mode };
    for (let i = 0; i < args.length; i++) {
        const key = args[i].startsWith('--') ? args[i].slice(2) : '';
        if (!values.has(key) && !flags.has(key)) fail(`unknown argument ${args[i]}`);
        if (key in raw) fail(`duplicate argument ${args[i]}`);
        if (flags.has(key)) { raw[key] = true; continue; }
        const value = args[++i];
        if (value === undefined || value.startsWith('--')) fail(`${args[i - 1]} needs a value`);
        raw[key] = value;
    }
    const whole = text => (/^\d+$/.test(text) ? Number(text) : fail(`"${text}" is not a whole number`));
    const list = text => text.split(',').map(s => whole(s.trim()));
    return {
        mode, run: raw.run, purpose: raw.purpose, name: raw.name, outDir: raw['out-dir'],
        generations: raw.generations === undefined ? undefined : list(raw.generations),
        anchors: raw.anchors === undefined ? undefined : list(raw.anchors),
        seed: raw.seed === undefined ? undefined : whole(raw.seed),
        domain: raw.domain === undefined ? undefined : whole(raw.domain),
        analysisSeed: raw['analysis-seed'] === undefined ? undefined : whole(raw['analysis-seed']),
        episodes: raw.episodes === undefined ? undefined : whole(raw.episodes),
        T: raw.T === undefined ? undefined : whole(raw.T),
        margin: raw.margin === undefined ? undefined
            : (/^\d+(\.\d+)?$/.test(raw.margin) ? Number(raw.margin) : fail(`--margin "${raw.margin}" is not a number`)),
        clones: !raw['no-clones'],
    };
}

/** A run directory's settings, training seeds and status; an evaluation never writes into it. */
export function loadRun(dir) {
    if (typeof dir !== 'string' || !existsSync(join(dir, 'run.json'))) fail(`no run.json in run directory "${dir}"`);
    const run = JSON.parse(readFileSync(join(dir, 'run.json'), 'utf8'));
    const status = existsSync(join(dir, 'status.json')) ? JSON.parse(readFileSync(join(dir, 'status.json'), 'utf8')) : null;
    if (run.configHash !== configHash(run.config)) fail(`run.json's configHash does not hash its config`);
    if (!Array.isArray(run.seedManifest?.seedsByGeneration)) fail('run.json lacks its training seed manifest');
    return { dir, run, status };
}

/**
 * A checkpoint as a driver pair, its models read under the demo's names. The manifest must hash to
 * the run's experiment and the model files to the manifest's hashes.
 */
export function loadCheckpoint({ dir, run }, generation, demo = herdDefinition()) {
    const at = join(dir, gen(generation)), manifestPath = join(at, 'manifest.json');
    if (!existsSync(manifestPath)) fail(`no checkpoint ${gen(generation)} in "${dir}"`);
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
    if (manifest.generation !== generation || manifest.configHash !== run.configHash || configHash(manifest.experiment) !== run.configHash)
        fail(`${gen(generation)}'s manifest does not belong to this run's experiment`);
    const pair = { id: gen(generation), generation };
    for (const species of ['prey', 'wolf']) {
        const path = join(at, `${species}.mlp.json`), text = readFileSync(path, 'utf8'), model = JSON.parse(text);
        if (sha256(text) !== manifest.models?.[species]?.hash) fail(`${gen(generation)}/${species}.mlp.json does not match its manifest`);
        if (model.species !== species) fail(`${path} holds species "${model.species}", not "${species}"`);
        pair[species] = { kind: 'network', id: pair.id, network: networkFromModel(model, demo, path),
            file: relative(root, path).replaceAll('\\', '/'), hash: sha256(text) };
    }
    return pair;
}

/** The committed clones (C2) as a driver pair. */
export function loadClones(demo = herdDefinition()) {
    const pair = { id: 'clone' };
    for (const species of ['prey', 'wolf']) {
        const path = join(root, 'tools/agents/models/herd-wolves', `${species}.mlp.json`), text = readFileSync(path, 'utf8');
        pair[species] = { kind: 'network', id: 'clone', network: networkFromModel(JSON.parse(text), demo, path),
            file: relative(root, path).replaceAll('\\', '/'), hash: sha256(text) };
    }
    return pair;
}

const parameters = network => network.reduce((n, layer) => n + layer.weights.length + layer.biases.length, 0);
const identity = driver => (driver.kind === 'network'
    ? { kind: 'network', file: driver.file, hash: driver.hash, parameters: parameters(driver.network) } : { kind: driver.kind });

/** Validates the options, loads drivers and plans matchups; no GPU and no writes. */
export function planEvaluation(raw) {
    if (raw.purpose !== 'validation' && raw.purpose !== 'final-test') fail('--purpose must be validation or final-test');
    u32(raw.seed, '--seed'); u32(raw.domain, '--domain'); u32(raw.analysisSeed, '--analysis-seed');
    if (!raw.generations?.length) fail('--generations needs at least one checkpoint');
    if (new Set(raw.generations).size !== raw.generations.length) fail('--generations repeats a checkpoint');
    if (raw.mode === 'baselines' && raw.anchors) fail('--anchors is for crossplay');
    if (raw.mode === 'crossplay' && !raw.clones) fail('--no-clones is for baselines');
    if (raw.margin !== undefined && (raw.mode !== 'baselines' || !(Number.isFinite(raw.margin) && raw.margin > 0)))
        fail('--margin is a positive number of catches per episode, for baselines');
    const loaded = loadRun(resolve(raw.run ?? fail('--run is required')));
    const experiment = loaded.run.config, episodes = raw.episodes ?? 20, T = raw.T ?? experiment.T;
    if (!Number.isSafeInteger(T) || T < 1 || experiment.P * T >= 2 ** 32 || episodes * experiment.P * T >= 2 ** 32)
        fail(`T ${T} must be positive and keep catch counters within u32`);
    const geometry = evaluationGeometry(experiment, episodes), demo = herdDefinition();
    const subjects = raw.generations.map(n => loadCheckpoint(loaded, n, demo));
    let plan, comparisons = [], anchors = null;
    if (raw.mode === 'baselines') {
        const baselines = [RULE_PAIR, LABEL_PAIR, ...(raw.clones ? [loadClones(demo)] : [])];
        plan = [...new Map(subjects.flatMap(s => baselineMatchups(s, baselines)).map(m => [m.id, m])).values()];
        comparisons = subjects.flatMap(s => baselineComparisons(s, baselines));
    } else {
        anchors = (raw.anchors ?? [raw.generations[0]]).map(gen);
        for (const id of anchors) if (!subjects.some(s => s.id === id)) fail(`anchor ${id} is not among --generations`);
        plan = crossplayMatchups(subjects);
    }
    const heldOut = buildHeldOutManifest({ masterSeed: raw.seed, domain: raw.domain, purpose: raw.purpose,
        episodes, exclude: [loaded.run.seedManifest] });
    const name = raw.name ?? `${loaded.run.config.name}-${raw.mode}-${raw.purpose}-${Date.now()}`;
    if (!/^[a-zA-Z0-9_-]+$/.test(name)) fail(`invalid evaluation name "${name}"`);
    const targetDir = resolve(raw.outDir ?? join(root, 'tools/agents/out/coevo-eval', name));
    if (existsSync(targetDir)) fail(`evaluation directory already exists: "${targetDir}" (nothing is reused)`);
    return { ...raw, name, targetDir, loaded, experiment, episodes, T, geometry, subjects, plan, comparisons, anchors, heldOut };
}

const laneOptions = experiment => ({ environment: experiment.environment, contacts: experiment.contacts,
    contactOptions: experiment.contactOptions, catchMargin: experiment.catchMargin,
    thinkEvery: experiment.thinkEvery, dt: experiment.dt, monitorFinite: true });

export async function runEvaluation(raw, { log = console.log } = {}) {
    const p = planEvaluation(raw);
    mkdirSync(resolve(p.targetDir, '..'), { recursive: true });
    mkdirSync(p.targetDir);  // exclusive claim
    writeStatus(p.targetDir, { state: 'incomplete', completedUpdates: 0 });
    return managedRun({ targetDir: p.targetDir }, async ({ own, progress }) => {
        const description = {
            schema: 1, kind: 'coevo-evaluation', version: EVALUATION_VERSION, mode: p.mode, purpose: p.purpose,
            run: { dir: relative(root, p.loaded.dir).replaceAll('\\', '/'), name: p.loaded.run.config.name,
                configHash: p.loaded.run.configHash, status: p.loaded.status?.state ?? 'unknown' },
            experiment: p.experiment, evaluation: { episodes: p.episodes, T: p.T, seconds: p.T * p.experiment.dt,
                geometry: p.geometry, layout: 'episode e in tile e of one fresh lane per matchup, tick zero' },
            heldOut: p.heldOut, bootstrap: { ...BOOTSTRAP, seed: p.analysisSeed, version: BOOTSTRAP_VERSION },
            margin: p.margin ?? null,
            drivers: Object.fromEntries([...new Map(p.plan.flatMap(game => ['prey', 'wolf'].map(species =>
                [`${game[species].id}:${species}`, { species, id: game[species].id, ...identity(game[species]) }]))).entries()]),
            plan: p.plan.map(game => ({ id: game.id, prey: game.prey.id, wolf: game.wolf.id })),
            comparisons: p.comparisons, anchors: p.anchors,
            sourceHashes: Object.fromEntries(SOURCES.map(path => [path, sha256(readFileSync(new URL(path, import.meta.url)))])),
            repo: getRepoMetadata(),
        };
        atomicWriteFile(join(p.targetDir, 'plan.json'), JSON.stringify(description, null, 1) + '\n');

        let host;
        if (!globalThis.navigator?.gpu) host = (await import('../../test/gpu/node-gpu.mjs')).installNodeGPU();
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) fail('no WebGPU adapter available');
        const device = own(await adapter.requestDevice());
        const health = own(watchDevice(device));
        const engine = new NullGraph();
        const init = await engine.init(null, { device, width: 8, height: 8 });
        if (!init.success) fail(`engine initialization failed: ${init.error}`);
        log(`coevolve: ${p.mode} "${p.name}" on ${adapter.info.vendor} ${adapter.info.description}: ` +
            `${p.plan.length} matchups x ${p.episodes} ${p.purpose} episodes of ${p.T} steps`);

        // One-row tables per network driver and species (a checkpoint's prey and wolves share its
        // id), caller-owned for the whole evaluation: the trainer's kernel, one row.
        const tables = new Map();
        for (const game of p.plan) for (const species of ['prey', 'wolf']) {
            const driver = game[species];
            if (driver.kind !== 'network') continue;
            if (!tables.has(driver.id)) tables.set(driver.id, {});
            tables.get(driver.id)[species] ??= own(A.MlpBrain.byGenome(engine, [driver.network]));
        }
        // Calibrate on the heaviest driver of each species in the plan.
        const heaviest = species => p.plan.map(game => game[species]).filter(d => d.kind === 'network')
            .sort((a, b) => parameters(b.network) - parameters(a.network))[0];
        const heavy = Object.fromEntries(['prey', 'wolf'].map(species => {
            const driver = heaviest(species);
            return [species, driver ? tables.get(driver.id)[species] : 'rule'];
        }));
        const guard = await calibrateArenaSteps({ geometry: p.geometry,
            key: guardIdentity({ ...p.experiment, evaluation: { episodes: p.episodes, cols: p.geometry.cols } }, adapter.info,
                Object.fromEntries(['prey', 'wolf'].map(species => [species, heaviest(species)?.network ?? 'rule']))),
            cacheFile: join(root, 'tools/agents/out/coevo-costs.json'), log,
            record: sample => appendLog(p.targetDir, { event: 'calibration-sample', ...sample }),
            async measure(probe, workload) {
                const built = await buildCompiled(device, () => buildLane(engine, probe, { ...laneOptions(p.experiment),
                    brains: heavy, seedOf: () => 18001, rowOf: () => 0, fixture: workloadFixture(workload, probe.L) }));
                const lane = built.value;
                try {
                    await health.check();
                    const ms = await timeSingles(engine, lane, health, Math.max(8, 2 * p.experiment.thinkEvery));
                    return { ms, setupCompileMs: built.setupCompileMs };
                } finally { lane.destroy(); }
            },
        });
        appendLog(p.targetDir, { event: 'calibration', ...guard });
        const submissions = createSubmissionGuard(guard.stepMs);

        const catches = [], timing = [];
        for (const [j, game] of p.plan.entries()) {
            const start = performance.now();
            const result = await playMatchup(engine, { experiment: p.experiment, geometry: p.geometry,
                seeds: p.heldOut.seeds, game, tables, T: p.T, compile: make => buildCompiled(device, make),
                run: lane => runGuarded(engine, lane, p.T, submissions, health) });
            catches.push(result.catches);
            timing.push({ id: game.id, wallMs: performance.now() - start, ...result.observed });
            appendLog(p.targetDir, { event: 'matchup', id: game.id, catches: result.catches, observed: result.observed });
            log(`coevolve: ${String(j + 1).padStart(3)}/${p.plan.length} ${game.id}: mean catches ` +
                `${(result.catches.reduce((a, b) => a + b, 0) / result.catches.length).toFixed(2)}`);
            progress(j + 1);
        }
        await health.check({ final: true });
        if (host) {
            await host.settle();
            if (host.events.length) fail(`Node WebGPU failures: ${JSON.stringify(host.events)}`);
        }
        const options = { seed: p.analysisSeed, resamples: BOOTSTRAP.resamples, level: BOOTSTRAP.level };
        const results = summarize(p.plan, catches, p.comparisons, { ...options, margin: p.margin });
        const evaluation = { ...description, results,
            crossplay: p.mode === 'crossplay' ? crossplaySummary(p.subjects, p.plan, catches, p.anchors, options) : null,
            device: { adapter: Object.fromEntries(['vendor', 'architecture', 'device', 'description'].map(key => [key, adapter.info[key] ?? 'unknown'])),
                dawnFlags: process.env.NG_DAWN_FLAGS ?? '' },
            calibration: { stepMs: guard.stepMs }, timing };
        atomicWriteFile(join(p.targetDir, 'evaluation.json'), JSON.stringify(evaluation, null, 1) + '\n');
        log(`coevolve: wrote ${join(p.targetDir, 'evaluation.json')}`);
        return { targetDir: p.targetDir, evaluation };
    });
}

// Standalone CLI invocation
if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
    runEvaluation(parseEvalArgs(process.argv.slice(2))).catch(error => {
        console.error(error.message || error);
        process.exit(1);
    });
}
