// Phase 14.3, design §4, §6, §8, §9, §12. Headless ES Co-evolution CLI & Trainer.
import { resolve, join } from 'node:path';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { NullGraph } from '../../dist/index.js';
import * as A from '../../dist/agents.js';
import { arenaLayout } from './scenario/arena-layout.mjs';
import { arenaDefinition, buildLane, buildMirroredLanes, readBuffers } from './scenario/herd-wolves-arenas.mjs';
import { noise, centeredRanks, gradient, createAdamState, adamAscent, masterParams, candidateNetworks } from './es.mjs';
import { shapeOf, initNetwork, networkOf } from './mlp-train.mjs';
import { networkFromModel } from './model.mjs';
import { checkStep } from '../film/guard.mjs';
import { validateConfig, parseCliArgs } from './coevolve-config.mjs';
import {
    buildSeedManifest, permutationForGeneration, opponentForPair,
    createArchiveManager, evolvingSpecies, opponentSpecies, noiseChecksum,
    STREAM_INIT
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

    const seedManifest = buildSeedManifest({
        masterSeed: config.seeds.master,
        domain: config.seeds.domain,
        generations: config.generations,
        K: config.K,
        E: config.E,
    });

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
    const def = arenaDefinition(lay);

    // Initialize start networks
    let preyNetwork, wolfNetwork;
    if (config.start === 'clones') {
        const preyPath = join(root, 'tools/agents/models/herd-wolves/prey.mlp.json');
        const wolfPath = join(root, 'tools/agents/models/herd-wolves/wolf.mlp.json');
        if (!existsSync(preyPath) || !existsSync(wolfPath)) {
            fail('clone models not found in tools/agents/models/herd-wolves');
        }
        preyNetwork = networkFromModel(JSON.parse(readFileSync(preyPath, 'utf8')), def, 'prey.mlp.json');
        wolfNetwork = networkFromModel(JSON.parse(readFileSync(wolfPath, 'utf8')), def, 'wolf.mlp.json');
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
        const preyShape = buildShape(), wolfShape = buildShape();
        preyNetwork = initNetwork(preyShape, (config.seeds.master ^ 0x12345678) >>> 0);
        wolfNetwork = initNetwork(wolfShape, (config.seeds.master ^ 0x87654321) >>> 0);
    }

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

    // Write initial archive snapshots and gen-0000 start checkpoint
    for (const sp of ['prey', 'wolf']) {
        writeArchiveSnapshot(config.targetDir, {
            snapshot: archive.get(sp, 0),
            species: sp,
            network: networkOf(shape[sp], master[sp]),
            obsNames: def.obsNames,
            actNames: def.actNames,
            configHash: config.configHash,
        });
    }
    writeGenerationCheckpoint(config.targetDir, {
        generation: 0,
        preyNetwork: networkOf(shape.prey, master.prey),
        wolfNetwork: networkOf(shape.wolf, master.wolf),
        obsNames: def.obsNames,
        actNames: def.actNames,
        configHash: config.configHash,
        exportOnnx: config.exportOnnx,
    });

    // Acquire WebGPU device & engine
    if (typeof navigator === 'undefined' || !navigator?.gpu) {
        try {
            const { installNodeGPU } = await import('../../test/gpu/node-gpu.mjs');
            installNodeGPU();
        } catch {
            // ignore
        }
    }
    if (typeof navigator === 'undefined' || !navigator?.gpu) fail('navigator.gpu not found. WebGPU environment required.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) fail('no WebGPU adapter available');
    const device = await adapter.requestDevice();
    const engine = new NullGraph();
    const initResult = await engine.init(null, { device, width: 8, height: 8 });
    if (!initResult.success) fail(`engine initialization failed: ${initResult.error}`);

    writeRunJson(config.targetDir, {
        config,
        seedManifest,
        adapterInfo: adapter.info,
    });

    log(`coevolve: run "${config.name}" started on ${adapter.info.vendor} ${adapter.info.description}`);
    log(`coevolve: layout ${config.arenasPerLane} arenas/lane (${config.totalArenas} total), ${config.P} prey, ${config.W} wolves`);

    // Shared caller-owned MlpBrain.byGenome tables (2K + 1 + M rows per species)
    const rowCapacity = 2 * config.K + 1 + config.archive.cap;
    const brains = {
        prey: A.MlpBrain.byGenome(engine, Array.from({ length: rowCapacity }, () => networkOf(shape.prey, master.prey))),
        wolf: A.MlpBrain.byGenome(engine, Array.from({ length: rowCapacity }, () => networkOf(shape.wolf, master.wolf))),
    };

    // Pack initial start master into row 2K and row 2K+1 (archive 0)
    for (const sp of ['prey', 'wolf']) {
        const packed = A.packMlpWeights(networkOf(shape[sp], master[sp]));
        brains[sp].setRows(packed, { first: 2 * config.K });
        brains[sp].setRows(packed, { first: 2 * config.K + 1 });
    }

    try {
        // Calibration using disposable lane
        const calibLane = buildLane(engine, geometry, {
            environment: config.environment,
            contacts: config.contacts,
            contactOptions: config.contactOptions,
            catchMargin: config.catchMargin,
            thinkEvery: config.thinkEvery,
            brains,
            seedOf: () => 18001,
            rowOf: () => 0,
        });
        let singleStepMs = 10;
        try {
            const t0 = performance.now();
            await calibLane.sys.step();
            singleStepMs = Math.max(0.1, performance.now() - t0);
            checkStep(singleStepMs, 1500);
        } finally {
            calibLane.destroy();
        }

        // Batch target: 500 ms (never > 1000 ms, safety stop at 1500 ms)
        let batchSteps = Math.max(1, Math.min(config.T, Math.floor(500 / singleStepMs)));
        if (batchSteps * singleStepMs > 1000) batchSteps = Math.max(1, Math.floor(1000 / singleStepMs));
        log(`coevolve: calibrated ~${singleStepMs.toFixed(1)} ms/step; initial batch size ${batchSteps} steps (submission target <= 500 ms)`);

        const B = config.arenasPerLane;

        // Generation Loop
        for (let t = 0; t < config.generations; t++) {
            const tGenStart = performance.now();
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
            const packedCandidates = new Float32Array(2 * config.K * brains[sEvo].floats);
            for (let k = 0; k < config.K; k++) {
                A.packMlpWeights(candidates[k].plus, packedCandidates.subarray(2 * k * brains[sEvo].floats, (2 * k + 1) * brains[sEvo].floats));
                A.packMlpWeights(candidates[k].minus, packedCandidates.subarray((2 * k + 1) * brains[sEvo].floats, (2 * k + 2) * brains[sEvo].floats));
            }
            brains[sEvo].setRows(packedCandidates, { first: 0 });

            // In-place setRows: update current master in row 2K of opponent species brain table
            const packedOpponent = A.packMlpWeights(networkOf(shape[sOpp], master[sOpp]));
            brains[sOpp].setRows(packedOpponent, { first: 2 * config.K });

            // Update opponent archive rows if any
            const oppArchive = archive.list(sOpp);
            for (let idx = 0; idx < oppArchive.length; idx++) {
                const packedArch = A.packMlpWeights(networkOf(shape[sOpp], oppArchive[idx].params));
                brains[sOpp].setRows(packedArch, { first: 2 * config.K + 1 + idx });
            }

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
                K: config.K,
            }));

            const genSeeds = seedManifest.seedsByGeneration[t];

            // Build fresh mirrored lanes for generation episode
            const pair = buildMirroredLanes(engine, geometry, {
                environment: config.environment,
                contacts: config.contacts,
                contactOptions: config.contactOptions,
                catchMargin: config.catchMargin,
                thinkEvery: config.thinkEvery,
                brains,
                seedOf: a => genSeeds[perm[a]],
                rowOf: (sign, a, sp) => {
                    const b = perm[a];
                    const k = Math.floor(b / config.E);
                    if (sp === sEvo) return 2 * k + sign;
                    return opponents[k].row;
                },
            });

            let tEpisodePlus = 0, tEpisodeMinus = 0;
            let plusCounters, minusCounters, plusStats, minusStats;

            try {
                // Submit Lane 0 (plus) in guarded batches
                const t0Plus = performance.now();
                let remPlus = config.T;
                while (remPlus > 0) {
                    const count = Math.min(remPlus, batchSteps);
                    const seq = [];
                    for (let s = 0; s < count; s++) seq.push(...pair.lanes[0].chain);
                    const tSub0 = performance.now();
                    engine.render(seq);
                    await device.queue.onSubmittedWorkDone();
                    const subMs = performance.now() - tSub0;
                    checkStep(subMs, 1500);
                    if (subMs > 500 && batchSteps > 1) {
                        batchSteps = Math.max(1, Math.floor(batchSteps * (400 / subMs)));
                    }
                    remPlus -= count;
                }
                tEpisodePlus = performance.now() - t0Plus;

                // Submit Lane 1 (minus) in guarded batches
                const t0Minus = performance.now();
                let remMinus = config.T;
                while (remMinus > 0) {
                    const count = Math.min(remMinus, batchSteps);
                    const seq = [];
                    for (let s = 0; s < count; s++) seq.push(...pair.lanes[1].chain);
                    const tSub0 = performance.now();
                    engine.render(seq);
                    await device.queue.onSubmittedWorkDone();
                    const subMs = performance.now() - tSub0;
                    checkStep(subMs, 1500);
                    if (subMs > 500 && batchSteps > 1) {
                        batchSteps = Math.max(1, Math.floor(batchSteps * (400 / subMs)));
                    }
                    remMinus -= count;
                }
                tEpisodeMinus = performance.now() - t0Minus;

                // Single readback after both lanes complete (I6)
                const t0Read = performance.now();
                [plusCounters, minusCounters, plusStats, minusStats] = await readBuffers(device, [
                    { buffer: pair.lanes[0].counters, words: B },
                    { buffer: pair.lanes[1].counters, words: B },
                    { buffer: pair.lanes[0].stats, words: 4 },
                    { buffer: pair.lanes[1].stats, words: 4 },
                ]);
            } finally {
                pair.destroy();
            }

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

            // Archive snapshot check
            const archiveChange = archive.maybeSnapshot(t, sEvo, master[sEvo]);
            if (archiveChange?.snapshotted) {
                writeArchiveSnapshot(config.targetDir, {
                    snapshot: archiveChange.snapshotted,
                    species: sEvo,
                    network: networkOf(shape[sEvo], master[sEvo]),
                    obsNames: def.obsNames,
                    actNames: def.actNames,
                    configHash: config.configHash,
                });
            }

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
                archive: archiveChange,
                timing: {
                    genWallMs: tGenTotal,
                    lanePlusMs: tEpisodePlus,
                    laneMinusMs: tEpisodeMinus,
                },
                guard: {
                    batchSteps,
                },
            };
            appendLog(config.targetDir, logEntry);

            // Periodic and final checkpoint
            if ((t + 1) % 10 === 0 || t === config.generations - 1) {
                writeGenerationCheckpoint(config.targetDir, {
                    generation: t + 1,
                    preyNetwork: networkOf(shape.prey, master.prey),
                    wolfNetwork: networkOf(shape.wolf, master.wolf),
                    obsNames: def.obsNames,
                    actNames: def.actNames,
                    configHash: config.configHash,
                    exportOnnx: config.exportOnnx,
                });
            }

            log(`coevolve: gen ${String(t + 1).padStart(3)}/${config.generations} (${sEvo.padEnd(4)}) ` +
                `scores [${scores.slice(0, 4).map(s => s.toFixed(1)).join(', ')}...] ` +
                `|g| ${gradNorm.toFixed(3)} |Δx| ${updateNorm.toFixed(4)} in ${tGenTotal.toFixed(0)} ms`);
        }
    } finally {
        brains.prey.destroy();
        brains.wolf.destroy();
        device.destroy();
    }

    log(`coevolve: completed ${config.generations} generations for "${config.name}". Output saved to ${config.targetDir}`);
    return {
        name: config.name,
        targetDir: config.targetDir,
        configHash: config.configHash,
        generations: config.generations,
    };
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
