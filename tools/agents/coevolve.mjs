// Phase 14.3, design §4, §6, §8, §9, §12. Headless ES Co-evolution CLI & Trainer.
import { resolve, join } from 'node:path';
import { createHash } from 'node:crypto';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { NullGraph } from '../../dist/index.js';
import * as A from '../../dist/agents.js';
import { arenaLayout } from './scenario/arena-layout.mjs';
import { arenaDefinition, buildLane, buildMirroredLanes, modelNames, readBuffers } from './scenario/herd-wolves-arenas.mjs';
import { noise, centeredRanks, gradient, createAdamState, adamAscent, masterParams, candidateNetworks } from './es.mjs';
import { shapeOf, initNetwork, networkOf } from './mlp-train.mjs';
import { networkFromModel } from './model.mjs';
import { managedRun, watchDevice, buildCompiled } from './coevolve-runtime.mjs';
import { calibrateArenaSteps, guardIdentity, createSubmissionGuard } from './coevolve-guard.mjs';
import { validateConfig, parseCliArgs, experimentOf } from './coevolve-config.mjs';
import {
    buildSeedManifest, permutationForGeneration, opponentForPair, policyRows, initSeed,
    createArchiveManager, evolvingSpecies, opponentSpecies, noiseChecksum,
} from './coevolve-schedule.mjs';
import {
    createRunDirectory, writeRunJson, appendLog,
    writeGenerationCheckpoint, writeArchiveSnapshot
} from './coevolve-checkpoint.mjs';

const fail = message => { throw new Error(`coevolve: ${message}`); };
const root = fileURLToPath(new URL('../../', import.meta.url));

export async function runCoevolution(rawConfig, { log = console.log } = {}) {
    const config = validateConfig(rawConfig);
    createRunDirectory(config.targetDir);
    return managedRun(config, async ({ own, progress }) => {

    const seedManifest = buildSeedManifest({
        masterSeed: config.seeds.master,
        domain: config.seeds.domain,
        generations: config.generations,
        K: config.K,
        E: config.E,
    });
    // Preserve settings and full seed manifest even when adapter/engine initialization fails.
    writeRunJson(config.targetDir, { config, seedManifest });

    const geometry = {
        arenas: config.arenasPerLane,
        wolves: config.W,
        prey: config.P,
        cols: config.cols,
        L: config.L,
        gap: config.gap,
        contacts: config.contacts ? config.contactOptions : null,
    };
    const lay = arenaLayout(geometry);
    const def = arenaDefinition(lay, { dt: config.dt });
    // Models read and write the demo's feature names, not the custom stages' indexed ones.
    const names = modelNames(def);
    const experiment = experimentOf(config);

    // Initialize start networks
    let preyNetwork, wolfNetwork, start = { mode: config.start };
    if (config.start === 'clones') {
        const preyPath = join(root, 'tools/agents/models/herd-wolves/prey.mlp.json');
        const wolfPath = join(root, 'tools/agents/models/herd-wolves/wolf.mlp.json');
        if (!existsSync(preyPath) || !existsSync(wolfPath)) {
            fail('clone models not found in tools/agents/models/herd-wolves');
        }
        const [preyText, wolfText] = [preyPath, wolfPath].map(path => readFileSync(path, 'utf8'));
        preyNetwork = networkFromModel(JSON.parse(preyText), names, 'prey.mlp.json');
        wolfNetwork = networkFromModel(JSON.parse(wolfText), names, 'wolf.mlp.json');
        for (const network of [preyNetwork, wolfNetwork]) {
            if (JSON.stringify(network.slice(0, -1).map(layer => layer.output)) !== JSON.stringify(config.hidden))
                fail('clone shape differs from explicit hidden profile');
        }
        start.files = Object.fromEntries([['prey', preyText], ['wolf', wolfText]].map(([sp, text]) =>
            [sp, { path: `tools/agents/models/herd-wolves/${sp}.mlp.json`, hash: createHash('sha256').update(text).digest('hex') }]));
    } else {
        const buildShape = () => {
            const layers = [];
            let inDim = 24;
            for (const h of config.hidden) {
                layers.push({ input: inDim, output: h, activation: 'tanh' });
                inDim = h;
            }
            layers.push({ input: inDim, output: 2, activation: 'tanh' });
            return layers;
        };
        const seedOf = species => initSeed({ masterSeed: config.seeds.master, domain: config.seeds.domain, species });
        preyNetwork = initNetwork(buildShape(), seedOf('prey'));
        wolfNetwork = initNetwork(buildShape(), seedOf('wolf'));
        start.seeds = { prey: seedOf('prey'), wolf: seedOf('wolf') };
    }
    writeRunJson(config.targetDir, { config, seedManifest, start });

    const shape = {
        prey: shapeOf(preyNetwork),
        wolf: shapeOf(wolfNetwork),
    };
    const master = {
        prey: masterParams(preyNetwork),
        wolf: masterParams(wolfNetwork),
    };
    const adam = {
        prey: createAdamState(master.prey),
        wolf: createAdamState(master.wolf),
    };

    const archive = createArchiveManager({
        cap: config.archive.cap,
        every: config.archive.every,
    });
    archive.init({
        prey: { params: master.prey },
        wolf: { params: master.wolf },
    });

    const exportSnapshot = (snapshot, sp) => writeArchiveSnapshot(config.targetDir, {
        snapshot,
        species: sp,
        network: networkOf(shape[sp], snapshot.params),
        obsNames: names.obsNames,
        actNames: names.actNames,
        configHash: config.configHash,
    });
    const exportCheckpoint = generation => writeGenerationCheckpoint(config.targetDir, {
        generation,
        preyNetwork: networkOf(shape.prey, master.prey),
        wolfNetwork: networkOf(shape.wolf, master.wolf),
        obsNames: names.obsNames,
        actNames: names.actNames,
        configHash: config.configHash,
        experiment,
        exportOnnx: config.exportOnnx,
    });

    // Write initial archive snapshots and gen-0000 start checkpoint
    for (const sp of ['prey', 'wolf']) exportSnapshot(archive.get(sp, 0), sp);
    exportCheckpoint(0);

    // Acquire WebGPU device & engine
    let host;
    if (!globalThis.navigator?.gpu) {
        const { installNodeGPU } = await import('../../test/gpu/node-gpu.mjs');
        host = installNodeGPU();
    }
    if (typeof navigator === 'undefined' || !navigator?.gpu) fail('navigator.gpu not found. WebGPU environment required.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) fail('no WebGPU adapter available');
    const device = own(await adapter.requestDevice());
    const health = own(watchDevice(device));
    const engine = new NullGraph();
    const initResult = await engine.init(null, { device, width: 8, height: 8 });
    if (!initResult.success) fail(`engine initialization failed: ${initResult.error}`);

    writeRunJson(config.targetDir, {
        config,
        seedManifest,
        start,
        adapterInfo: adapter.info,
        limits: device.limits,
    });

    log(`coevolve: run "${config.name}" started on ${adapter.info.vendor} ${adapter.info.description}`);
    log(`coevolve: layout ${config.arenasPerLane} arenas/lane (${config.totalArenas} total), ${config.P} prey, ${config.W} wolves`);

    // Shared caller-owned MlpBrain.byGenome tables (2K + 1 + M rows per species)
    const rows = policyRows({ K: config.K, cap: config.archive.cap });
    const brains = {};
    for (const sp of ['prey', 'wolf']) {
        const bytes = rows.count * A.mlpLayout(networkOf(shape[sp], master[sp])).floats * 4;
        if (!Number.isSafeInteger(bytes) || bytes > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize))
            fail('policy rows exceed device buffer limits');
        brains[sp] = own(A.MlpBrain.byGenome(engine,
            Array.from({ length: rows.count }, () => networkOf(shape[sp], master[sp]))));
    }
    // One upload per species per generation: the current opponent and its archive are contiguous.
    const packOpponentRows = sp => {
        const members = [master[sp], ...archive.list(sp).map(snapshot => snapshot.params)];
        const packed = new Float32Array(members.length * brains[sp].floats);
        members.forEach((params, j) => A.packMlpWeights(networkOf(shape[sp], params),
            packed.subarray(j * brains[sp].floats, (j + 1) * brains[sp].floats)));
        brains[sp].setRows(packed, { first: rows.current });
    };
    for (const sp of ['prey', 'wolf']) packOpponentRows(sp);

    {
        const laneOptions = {
            environment: config.environment,
            contacts: config.contacts,
            contactOptions: config.contactOptions,
            catchMargin: config.catchMargin,
            thinkEvery: config.thinkEvery,
            dt: config.dt,
            monitorFinite: true,
            brains,
            seedOf: () => 18001,
            rowOf: a => a % rows.current,
        };
        // Candidate rows are genuinely perturbed during calibration, not duplicate master rows.
        for (const sp of ['prey', 'wolf']) for (let k = 0; k < config.K; k++) {
            const candidate = candidateNetworks(shape[sp], master[sp], noise(config.seeds.master, 0, k, master[sp].length), config.optimizer.sigma);
            brains[sp].setRows(A.packMlpWeights(candidate.plus), { first: rows.candidate(k, 0) });
            brains[sp].setRows(A.packMlpWeights(candidate.minus), { first: rows.candidate(k, 1) });
        }
        const guard = await calibrateArenaSteps({ geometry,
            key: guardIdentity(config, adapter.info, { prey: preyNetwork, wolf: wolfNetwork }),
            cacheFile: join(root, 'tools/agents/out/coevo-costs.json'), log,
            record: sample => appendLog(config.targetDir, { event: 'calibration-sample', ...sample }),
            async measure(probe, workload) {
                const built = await buildCompiled(device, () => buildLane(engine, probe, { ...laneOptions,
                    fixture(init, { lo, slot }) {
                        if (workload === 'ordinary') return;
                        const shift = workload === 'coincident' ? 0 : (slot % 8) * 0.02;
                        init.position = [lo[0] + (workload === 'wall' ? 0.61 : probe.L / 2) + shift,
                            0.08, lo[1] + probe.L / 2 + shift];
                        if (workload === 'frozen') { init.frozen = true; init.age = 1; }
                    },
                }));
                const lane = built.value;
                try {
                    await health.check();
                    const single = createSubmissionGuard(1);
                    let ms = 0;
                    // Singles at advancing cadence; compilation is complete before the first step.
                    for (let step = 0; step < Math.max(8, 2 * config.thinkEvery); step++) {
                        const start = performance.now(); engine.render(lane.chain);
                        await health.check();
                        const took = performance.now() - start;
                        single.observe(1, took); ms = Math.max(ms, took);
                        if (took > 1000) fail('calibration single exceeds admission limit');
                    }
                    return { ms, setupCompileMs: built.setupCompileMs, contactGrid: lane.solver?.options ?? null };
                } finally { lane.destroy(); }
            },
        });
        appendLog(config.targetDir, { event: 'calibration', ...guard });
        const submissions = createSubmissionGuard(guard.stepMs);
        // Exactly T one-step chains in guarded batches; each batch is waited for and checked.
        const runLane = async lane => {
            const observed = { ms: 0, batches: 0, maxBatchMs: 0, maxStepMs: 0 }, start = performance.now();
            for (let remaining = config.T; remaining > 0;) {
                const count = submissions.next(remaining), seq = [];
                for (let s = 0; s < count; s++) seq.push(...lane.chain);
                const submitted = performance.now();
                engine.render(seq);
                await health.check();
                const ms = performance.now() - submitted;
                submissions.observe(count, ms);
                Object.assign(observed, { batches: observed.batches + 1, maxBatchMs: Math.max(observed.maxBatchMs, ms),
                    maxStepMs: Math.max(observed.maxStepMs, ms / count) });
                remaining -= count;
            }
            observed.ms = performance.now() - start;
            return observed;
        };

        const B = config.arenasPerLane;

        // Generation Loop
        for (let t = 0; t < config.generations; t++) {
            const tGenStart = performance.now();
            const tPackStart = performance.now();
            const sEvo = evolvingSpecies(t);
            const sOpp = opponentSpecies(t);
            const speciesSign = sEvo === 'wolf' ? 1 : -1;
            const paramCount = master[sEvo].length;

            // Generate candidate directions and perturbations
            const directions = [];
            const candidates = [];
            let totalCollisions = 0;
            for (let k = 0; k < config.K; k++) {
                const dir = noise(config.seeds.master, t, k, paramCount);
                directions.push(dir);
                const cand = candidateNetworks(shape[sEvo], master[sEvo], dir, config.optimizer.sigma);
                candidates.push(cand);
                totalCollisions += cand.diagnostics.collisions;
            }
            const noiseCk = noiseChecksum(directions);

            // In-place setRows: upload all 2K candidates into evolving species brain table
            const floats = brains[sEvo].floats;
            const packedCandidates = new Float32Array(2 * config.K * floats);
            for (let k = 0; k < config.K; k++) for (const [sign, network] of [[0, candidates[k].plus], [1, candidates[k].minus]]) {
                const row = rows.candidate(k, sign);
                A.packMlpWeights(network, packedCandidates.subarray(row * floats, (row + 1) * floats));
            }
            brains[sEvo].setRows(packedCandidates, { first: 0 });

            // In-place setRows: the opponent's current master (row 2K) and its archive after it
            const oppArchive = archive.list(sOpp);
            packOpponentRows(sOpp);

            // Tile permutation and opponent assignments
            const perm = permutationForGeneration({
                masterSeed: config.seeds.master,
                domain: config.seeds.domain,
                t,
                B,
            });
            const invPerm = new Int32Array(B);
            for (let a = 0; a < B; a++) invPerm[perm[a]] = a;

            const opponents = Array.from({ length: config.K }, (_, k) => opponentForPair({
                masterSeed: config.seeds.master,
                domain: config.seeds.domain,
                t,
                k,
                archive: oppArchive,
                h: config.archive.share,
                rows,
            }));

            const genSeeds = seedManifest.seedsByGeneration[t];

            // Build fresh mirrored lanes for generation episode
            const packingMs = performance.now() - tPackStart;
            const built = await buildCompiled(device, () => buildMirroredLanes(engine, geometry, {
                environment: config.environment,
                contacts: config.contacts,
                contactOptions: config.contactOptions,
                catchMargin: config.catchMargin,
                thinkEvery: config.thinkEvery,
                dt: config.dt,
                monitorFinite: true,
                brains,
                seedOf: a => genSeeds[perm[a]],
                rowOf: (sign, a, sp) => {
                    const k = Math.floor(perm[a] / config.E);
                    return sp === sEvo ? rows.candidate(k, sign) : opponents[k].row;
                },
            }));
            const pair = built.value;

            let plus, minus, readbackMs = 0;
            let plusCounters, minusCounters, plusStats, minusStats;

            try {
                // Lanes run one after the other: plus (sign 0), then minus (sign 1)
                plus = await runLane(pair.lanes[0]);
                minus = await runLane(pair.lanes[1]);

                // Single readback after both lanes complete (I6)
                const t0Read = performance.now();
                [plusCounters, minusCounters, plusStats, minusStats] = await readBuffers(device, [
                    { buffer: pair.lanes[0].counters, words: B },
                    { buffer: pair.lanes[1].counters, words: B },
                    { buffer: pair.lanes[0].stats, words: 4 },
                    { buffer: pair.lanes[1].stats, words: 4 },
                ]);
                await health.check();
                for (const [counters, stats] of [[plusCounters, plusStats], [minusCounters, minusStats]]) {
                    if (stats[3]) fail('nonfinite runtime state');
                    if (counters.reduce((sum, n) => sum + n, 0) !== stats[0] || counters.some(n => n > config.P * config.T))
                        fail('runtime catch counters are inconsistent');
                }
                readbackMs = performance.now() - t0Read;
            } finally {
                pair.destroy();
            }

            const tOptimizerStart = performance.now();
            // Aggregate catches per candidate and replica
            const replicaCatches = [];
            const scores = new Float64Array(2 * config.K);

            for (let k = 0; k < config.K; k++) {
                let sumPlus = 0, sumMinus = 0;
                const kCatches = { plus: [], minus: [] };
                for (let e = 0; e < config.E; e++) {
                    const b = k * config.E + e;
                    const a = invPerm[b];
                    const cPlus = plusCounters[a];
                    const cMinus = minusCounters[a];
                    kCatches.plus.push(cPlus);
                    kCatches.minus.push(cMinus);
                    sumPlus += cPlus;
                    sumMinus += cMinus;
                }
                replicaCatches.push(kCatches);
                scores[2 * k] = speciesSign * (sumPlus / config.E);
                scores[2 * k + 1] = speciesSign * (sumMinus / config.E);
            }

            // Utilities via centered ranks
            const utilities = centeredRanks(scores);

            // Fitness gradient and Adam update
            const grad = gradient(directions, utilities, config.optimizer.sigma);
            const prevX = Float64Array.from(master[sEvo]);
            adamAscent(adam[sEvo], grad, { rate: config.optimizer.rate, decay: config.optimizer.decay });
            master[sEvo] = adam[sEvo].x;

            // Compute diagnostic norms
            let gradNorm = 0, penaltyNorm = 0, updateNorm = 0, weightNorm = 0;
            for (let j = 0; j < paramCount; j++) {
                gradNorm = Math.hypot(gradNorm, grad[j]);
                penaltyNorm = Math.hypot(penaltyNorm, config.optimizer.decay * prevX[j]);
                updateNorm = Math.hypot(updateNorm, master[sEvo][j] - prevX[j]);
                weightNorm = Math.hypot(weightNorm, master[sEvo][j]);
            }

            // Tie share
            const scoreCounts = new Map();
            for (let i = 0; i < scores.length; i++) {
                scoreCounts.set(scores[i], (scoreCounts.get(scores[i]) ?? 0) + 1);
            }
            let tiedScores = 0;
            for (const [, count] of scoreCounts) if (count > 1) tiedScores += count;
            const tieShare = tiedScores / scores.length;
            const optimizerMs = performance.now() - tOptimizerStart;
            const tExportStart = performance.now();

            // Archive: every H completed updates, both species' current masters (design §8)
            const archiveChanges = archive.snapshotDue(t, master);
            for (const change of archiveChanges) exportSnapshot(change.snapshotted, change.species);

            const tGenTotal = performance.now() - tGenStart;

            // Append to log.jsonl
            const logEntry = {
                t,
                completedUpdates: t + 1,
                species: sEvo,
                scores: Array.from(scores),
                replicaCatches,
                tieShare,
                opponentIds: opponents.map(o => o.id),
                permutation: Array.from(perm),
                noiseChecksum: noiseCk,
                gradientNorm: gradNorm,
                penaltyNorm,
                updateNorm,
                weightNorm,
                f32Collisions: totalCollisions,
                // IDs only: the snapshots' weights are their archive/ files
                archive: archiveChanges.length ? archiveChanges.map(change => ({ species: change.species,
                    added: change.snapshotted.id, evicted: change.evicted?.id ?? null })) : null,
                timing: {
                    genWallMs: tGenTotal,
                    lanePlusMs: plus.ms,
                    laneMinusMs: minus.ms,
                    setupCompileMs: built.setupCompileMs,
                    packingMs,
                    readbackMs,
                    optimizerMs,
                },
                guard: {
                    batchSteps: submissions.next(config.T),
                    batches: plus.batches + minus.batches,
                    maxBatchMs: Math.max(plus.maxBatchMs, minus.maxBatchMs),
                    maxStepMs: Math.max(plus.maxStepMs, minus.maxStepMs),
                },
            };
            // Periodic and final checkpoint
            if ((t + 1) % 10 === 0 || t === config.generations - 1) exportCheckpoint(t + 1);

            logEntry.timing.exportMs = performance.now() - tExportStart;
            logEntry.timing.genWallMs = performance.now() - tGenStart;
            appendLog(config.targetDir, logEntry);
            log(`coevolve: gen ${String(t + 1).padStart(3)}/${config.generations} (${sEvo.padEnd(4)}) ` +
                `scores [${scores.slice(0, 4).map(s => s.toFixed(1)).join(', ')}...] ` +
                `|g| ${gradNorm.toFixed(3)} |Δx| ${updateNorm.toFixed(4)} in ${logEntry.timing.genWallMs.toFixed(0)} ms`);
            progress(t + 1);
        }
        await health.check({ final: true });
        if (host) {
            await host.settle();
            if (host.events.length) fail(`Node WebGPU failures: ${JSON.stringify(host.events)}`);
        }
    }

    log(`coevolve: completed ${config.generations} generations for "${config.name}". Output saved to ${config.targetDir}`);
    return {
        name: config.name,
        targetDir: config.targetDir,
        configHash: config.configHash,
        generations: config.generations,
    };
    });
}

// Standalone CLI invocation
if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
    const raw = parseCliArgs(process.argv.slice(2));
    if (!raw.name) {
        raw.name = `run-${Date.now()}`;
    }
    runCoevolution(raw).catch(err => {
        console.error(err.message || err);
        process.exit(1);
    });
}
