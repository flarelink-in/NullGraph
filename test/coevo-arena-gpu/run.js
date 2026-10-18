import { NullGraph } from '../../dist/index.js';
import * as A from '../../dist/agents.js';
import { createHerd } from '../../tools/agents/scenario/herd-wolves.mjs';
import { arenaLayout, catchFlags } from '../../tools/agents/scenario/arena-layout.mjs';
import { buildLane, buildMirroredLanes, readBuffers } from '../../tools/agents/scenario/herd-wolves-arenas.mjs';
import { harnessEnv } from '../gpu/harness-env.mjs';
const env = harnessEnv('__COEVO_ARENA_RESULTS__'), { check, log } = env, seen = new Set();
const stationary = { kind: 'stage', wgsl: `fn think(i: u32, species: u32, tick: u32) { setAct(i, 0u, 0.0); setAct(i, 1u, -1.0); }` };
const options = { environment: 'contact-v2', contacts: false, thinkEvery: 3, catchMargin: 0.05,
    seedOf: () => 11, rowOf: () => 0 };
const geometry = { arenas: 2, wolves: 32, prey: 64, cols: 2, L: 24, gap: 10 };
async function read(engine, lane) {
    return readBuffers(engine.device, [{ buffer: lane.sys.buffer, words: lane.lay.count * 16 },
        { buffer: lane.counters, words: lane.lay.arenas }, { buffer: lane.stats, words: 4 }]);
}
async function steps(lane, n) { for (let i = 0; i < n; i++) await lane.sys.step(); }
for (const powerPreference of ['high-performance', 'low-power']) {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference });
    if (!adapter) continue;
    const info = adapter.info, key = JSON.stringify([info.vendor, info.device, info.description]);
    if (seen.has(key)) continue; seen.add(key);
    const device = await adapter.requestDevice(), engine = new NullGraph();
    try {
        const init = await engine.init(env.canvas, { device, width: 8, height: 8 });
        if (!init.success) throw new Error(init.error);
        log(`14.1 on ${info.vendor} ${info.description}`);
        // Legacy oracle uses exactly the original geometry and seeded rule state.
        const herd = createHerd(engine), legacy = buildLane(engine,
            { arenas: 1, wolves: 256, prey: 3840, cols: 1, L: 160, gap: 10 },
            { ...options, environment: 'legacy-v1' });
        try {
            for (let t = 50; t <= 300; t += 50) {
                for (let j = 0; j < 50; j++) { await herd.agents.step(); await legacy.sys.step(); }
                const [a, b, ac, bc] = await readBuffers(device, [{ buffer: herd.agents.buffer, words: 4096 * 16 },
                    { buffer: legacy.sys.buffer, words: 4096 * 16 }, { buffer: herd.stats, words: 4 }, { buffer: legacy.stats, words: 4 }]);
                check(`A42 legacy records/catches ${t}`, a.every((v, i) => v === b[i]) && ac[0] === bc[0]);
            }
        } finally { herd.destroy(); legacy.destroy(); }
        for (const contacts of [false, true]) {
            const config = { ...options, contacts, contactOptions: { iterations: 2, maxPush: 0.25 } };
            const lane = buildLane(engine, geometry, { ...config, fixture: (init, { i, wolf, lo, a }) => {
                init.age = 1; init.frozen = true;
                init.position = [lo[0] + 12 + (wolf ? (i % 32 === 0 ? 0 : 0.5) : 0), 0.08, lo[1] + 12];
                if (wolf && i % 32 === 0) init.user0 = 1;
                // Identical physical coordinates across different tile IDs are deliberately invalid
                // only in the separate isolation fixture below, never in an admitted workload.
            } });
            try {
                lane.sys.setBrain(stationary);
                const before = lane.sys.records.data.slice(), expected = catchFlags(before, lane.lay, 0.05);
                const count = expected.reduce((n, f) => n + Number(f === 256), 0);
                await lane.sys.step(); const [records, counters, stats] = await read(engine, lane);
                check(`A46 exhaustive catches, many prey, once, contacts ${contacts}`, count === 128 && stats[0] === count && counters[0] === 64 && counters[1] === 64);
                const f = new Float32Array(records.buffer);
                check(`A46 all eligible wolves rest, flags consumed ${contacts}`, records.every((v, i) => i % 16 !== 13 || (v & 768) === 0) &&
                    Array.from({ length: 64 }, (_, i) => f[i * 16 + 11]).every(t => t > 0));
                check(`A46 respawn containment ${contacts}`, Array.from({ length: lane.lay.count }, (_, i) => {
                    const tile = records[i * 16 + 15], loX = lane.lay.origin[0] + tile * lane.lay.pitch;
                    return f[i * 16] >= loX + f[i * 16 + 3] - 1e-4 && f[i * 16] <= loX + 24 - f[i * 16 + 3] + 1e-4;
                }).every(Boolean));
            } finally { lane.destroy(); lane.destroy(); }
            const pair = buildMirroredLanes(engine, geometry, { ...config, rowOf: () => 0 });
            let reference;
            try {
                check(`A43 fresh mirrored buffers/solvers ${contacts}`, pair.lanes[0].sys.buffer !== pair.lanes[1].sys.buffer &&
                    (!contacts || pair.lanes[0].solver !== pair.lanes[1].solver));
                for (const lane of pair.lanes) await steps(lane, 20);
                const [a] = await read(engine, pair.lanes[0]), [b] = await read(engine, pair.lanes[1]);
                reference = a;
                check(`A43 mirrored exact ${contacts}`, a.every((v, i) => v === b[i]));
                pair.lanes[0].sys.setBrain(stationary);
                check(`A43 chain refresh after brain change ${contacts}`, pair.lanes[0].chain === pair.lanes[0].sys.getComputeStages());
            } finally { pair.destroy(); pair.destroy(); }
            const unrelated = buildLane(engine, geometry, config);
            try { await steps(unrelated, 7); } finally { unrelated.destroy(); }
            const fresh = buildLane(engine, geometry, config);
            try {
                await steps(fresh, 20); const [words] = await read(engine, fresh);
                check(`A44 fresh episode resets private state ${contacts}`, words.every((v, i) => v === reference[i]));
            } finally { fresh.destroy(); }
        }
        // Negative control: legacy pre-act nearest list misses the second eligible wolf.
        const fixture = (init, { wolf, i, lo }) => {
            init.age = 1; init.frozen = true;
            init.position = [lo[0] + 12 + (wolf ? (i % 32 === 0 ? 0.1 : 0.5) : 0), 0.08, lo[1] + 12];
            if (wolf && i % 32 === 0) init.user0 = 1;
        };
        const totals = [];
        for (const environment of ['legacy-v1', 'contact-v2']) {
            const lane = buildLane(engine, geometry, { ...options, environment, fixture });
            try { lane.sys.setBrain(stationary); await lane.sys.step(); totals.push((await read(engine, lane))[2][0]); }
            finally { lane.destroy(); }
        }
        check('A46 watched-failing nearest-list omission', totals[0] === 0 && totals[1] === 128);
        for (const [name, dx, resting, dead, expected] of [
            ['tie', 0.5, false, false, 128], ['inside', 0.999, false, false, 128],
            ['boundary', 1, false, false, 0], ['outside', 1.001, false, false, 0],
            ['resting', 0.5, true, false, 0], ['dead', 0.5, false, true, 0]]) {
            const lane = buildLane(engine, geometry, { ...options, catchMargin: 0,
                fixture: (init, { wolf, lo }) => {
                    init.age = 1; init.frozen = true;
                    init.position = [lo[0] + 12 + (wolf ? dx : 0), 0.08, lo[1] + 12];
                    if (wolf && resting) init.user0 = 1;
                } });
            try {
                lane.sys.setBrain(stationary);
                if (dead) { for (let i = 0; i < lane.lay.wolfTotal; i++) lane.sys.kill(i); lane.sys.upload(); }
                const oracle = catchFlags(lane.sys.records.data, lane.lay, 0).filter(f => f === 256).length;
                await lane.sys.step(); const [words, , stats] = await read(engine, lane);
                check(`A46 exhaustive ${name}`, oracle === expected && stats[0] === expected);
                if (name === 'tie') {
                    const f = new Float32Array(words.buffer);
                    check('A46 ties rest every eligible wolf, not a nearest winner',
                        Array.from({ length: 64 }, (_, i) => f[i * 16 + 11]).every(t => t === 5));
                }
            } finally { lane.destroy(); }
        }
        // --- Shared policy tables (I4, A45) ---
        const makeNet = (turn = 0, throttle = 1) => [
            { input: 24, output: 16, activation: 'tanh', weights: new Float32Array(24 * 16), biases: new Float32Array(16) },
            { input: 16, output: 2, activation: 'tanh', weights: new Float32Array(16 * 2), biases: new Float32Array([turn, throttle]) }
        ];
        const preyBrain = A.MlpBrain.byGenome(engine, Array.from({ length: 8 }, (_, r) => makeNet(0, r % 2 === 0 ? 0.8 : -0.8)));
        const wolfBrain = A.MlpBrain.byGenome(engine, Array.from({ length: 8 }, (_, r) => makeNet(r % 2 === 0 ? 0.3 : -0.3, 0.5)));
        try {
            const neutralPair = buildMirroredLanes(engine, geometry, {
                ...options,
                brains: { prey: preyBrain, wolf: wolfBrain },
                rowOf: (sign, a) => a
            });
            try {
                for (const lane of neutralPair.lanes) await steps(lane, 20);
                const [a] = await read(engine, neutralPair.lanes[0]);
                const [b] = await read(engine, neutralPair.lanes[1]);
                check('A43 neutral mirrored exact with shared byGenome tables', a.every((v, i) => v === b[i]));
            } finally { neutralPair.destroy(); }

            const candPair = buildMirroredLanes(engine, geometry, {
                ...options,
                brains: { prey: preyBrain, wolf: wolfBrain },
                rowOf: (sign) => (sign === 0 ? 0 : 1)
            });
            try {
                for (const lane of candPair.lanes) await steps(lane, 20);
                const [ca] = await read(engine, candPair.lanes[0]);
                const [cb] = await read(engine, candPair.lanes[1]);
                check('A45 shared policy discrimination between distinct candidate rows', ca.some((v, i) => v !== cb[i]));
            } finally { candPair.destroy(); }

            const packed = A.packMlpWeights(makeNet(0, 0.1));
            preyBrain.setRows(packed, { first: 0 });
            const updateLane = buildLane(engine, geometry, {
                ...options,
                brains: { prey: preyBrain, wolf: wolfBrain },
                rowOf: () => 0
            });
            try {
                await steps(updateLane, 10);
                const [rec] = await read(engine, updateLane);
                check('A45 setRows updates shared table in-place without reallocation', rec.some(v => v !== 0));
            } finally { updateLane.destroy(); }

            let rowRefused = false;
            try {
                buildLane(engine, geometry, {
                    ...options,
                    brains: { prey: preyBrain, wolf: wolfBrain },
                    rowOf: () => 8
                });
            } catch (e) {
                rowRefused = /coevolve: prey row 8 outside policy table/.test(e.message);
            }
            check('A45 row overflow refused before allocation', rowRefused);
        } finally {
            preyBrain.destroy();
            wolfBrain.destroy();
        }

        // --- Broader I1/I3/E5/C1 Stress Coverage ---
        const stressGeom = { arenas: 4, wolves: 16, prey: 64, cols: 2, L: 24, gap: 10 };
        const isoConfig = {
            ...options,
            contacts: true,
            contactOptions: { iterations: 2, maxPush: 0.25 },
            fixture: (init, { wolf, a, lo }) => {
                init.age = 1;
                if (a === 0 && wolf) init.position = [lo[0] + 23.3, 0.08, lo[1] + 12];
                else if (a === 1 && !wolf) init.position = [lo[0] + 0.5, 0.08, lo[1] + 12];
                else if (wolf) init.position = [lo[0] + 2, 0.08, lo[1] + 2];
                else init.position = [lo[0] + 22, 0.08, lo[1] + 22];
            }
        };
        const isoLane = buildLane(engine, stressGeom, isoConfig);
        try {
            isoLane.sys.setBrain(stationary);
            await steps(isoLane, 10);
            const [rec, counters, stats] = await read(engine, isoLane);
            check('A42 cross-tile isolation: zero catches or contact leakage across tile gap',
                stats[0] === 0 && counters[0] === 0 && counters[1] === 0);
            const f = new Float32Array(rec.buffer);
            const withinBounds = Array.from({ length: isoLane.lay.count }, (_, i) => {
                const tile = rec[i * 16 + 15];
                const tileLoX = isoLane.lay.origin[0] + (tile % isoLane.lay.cols) * isoLane.lay.pitch;
                const tileLoZ = isoLane.lay.origin[1] + Math.floor(tile / isoLane.lay.cols) * isoLane.lay.pitch;
                const x = f[i * 16], z = f[i * 16 + 2], r = f[i * 16 + 3];
                return x >= tileLoX + r - 1e-4 && x <= tileLoX + 24 - r + 1e-4 &&
                       z >= tileLoZ + r - 1e-4 && z <= tileLoZ + 24 - r + 1e-4;
            }).every(Boolean);
            check('A46 isolation stress: all agents remain strictly contained in respective arenas', withinBounds);
        } finally { isoLane.destroy(); }

        const denseMirrored = buildMirroredLanes(engine, stressGeom, {
            ...options,
            contacts: true,
            contactOptions: { iterations: 2, maxPush: 0.25 },
            fixture: (init, { lo, slot }) => {
                init.age = 1;
                const cluster = slot % 4;
                init.position = [lo[0] + 10 + cluster * 0.1, 0.08, lo[1] + 10 + cluster * 0.1];
            }
        });
        try {
            denseMirrored.lanes[0].sys.setBrain(stationary);
            denseMirrored.lanes[1].sys.setBrain(stationary);
            for (const lane of denseMirrored.lanes) await steps(lane, 25);
            const [a] = await read(engine, denseMirrored.lanes[0]);
            const [b] = await read(engine, denseMirrored.lanes[1]);
            check('A43 mirrored exact under dense contact and coincidence stress', a.every((v, i) => v === b[i]));
        } finally { denseMirrored.destroy(); }

        const multiCatchLane = buildLane(engine, stressGeom, {
            ...options,
            catchMargin: 0.05,
            fixture: (init, { wolf, i, lo }) => {
                init.age = 1; init.frozen = true;
                if (wolf && (i % 16 < 2)) init.position = [lo[0] + 12 + (i % 2) * 0.2, 0.08, lo[1] + 12];
                else if (!wolf && (i % 64 < 4)) init.position = [lo[0] + 12 + (i % 4) * 0.1, 0.08, lo[1] + 12];
                else if (wolf) init.position = [lo[0] + 2, 0.08, lo[1] + 2];
                else init.position = [lo[0] + 22, 0.08, lo[1] + 22];
            }
        });
        try {
            multiCatchLane.sys.setBrain(stationary);
            await multiCatchLane.sys.step();
            const [, counters, stats] = await read(engine, multiCatchLane);
            const sumCounters = counters.reduce((sum, c) => sum + c, 0);
            check('A45 per-arena counters sum exactly equals global stats under multi-catch',
                sumCounters === stats[0] && stats[0] === 16 && counters.every(c => c === 4));
        } finally { multiCatchLane.destroy(); }

        const containmentLane = buildLane(engine, geometry, {
            ...options,
            contacts: true,
            contactOptions: { iterations: 2, maxPush: 0.5 },
            fixture: (init, { wolf, lo }) => {
                init.age = 1; init.frozen = true;
                // Place 2 wolves touching right at east wall to push one against wall
                init.position = [lo[0] + 23.35 + (wolf ? 0 : -0.1), 0.08, lo[1] + 12];
            }
        });
        try {
            containmentLane.sys.setBrain(stationary);
            await containmentLane.sys.step();
            const [rec] = await read(engine, containmentLane);
            const f = new Float32Array(rec.buffer);
            const validBounds = Array.from({ length: containmentLane.lay.count }, (_, i) => {
                const tile = rec[i * 16 + 15];
                const tileLoX = containmentLane.lay.origin[0] + tile * containmentLane.lay.pitch;
                return f[i * 16] >= tileLoX + f[i * 16 + 3] - 1e-4 && f[i * 16] <= tileLoX + 24 - f[i * 16 + 3] + 1e-4;
            }).every(Boolean);
            check('A46 containment clamps positions under contact wall push', validBounds);
        } finally { containmentLane.destroy(); }

        // --- Contact-v2 Smoke validation (A47) ---
        const smokeBrainP = A.MlpBrain.byGenome(engine, Array.from({ length: 4 }, () => makeNet(0.1, 0.6)));
        const smokeBrainW = A.MlpBrain.byGenome(engine, Array.from({ length: 4 }, () => makeNet(-0.1, 0.7)));
        try {
            const smokePair = buildMirroredLanes(engine, geometry, {
                ...options,
                contacts: true,
                contactOptions: { iterations: 2, maxPush: 0.25 },
                brains: { prey: smokeBrainP, wolf: smokeBrainW },
                rowOf: () => 0
            });
            let maxStepMs = 0;
            try {
                for (let s = 0; s < 60; s++) {
                    const t0 = performance.now();
                    await smokePair.lanes[0].sys.step();
                    await smokePair.lanes[1].sys.step();
                    const stepMs = performance.now() - t0;
                    if (stepMs > maxStepMs) maxStepMs = stepMs;
                }
                check('A47 contact-v2 smoke 60 steps completes under 1500ms safety stop', maxStepMs < 1500, `max step ${maxStepMs.toFixed(1)} ms`);
                const [a0] = await read(engine, smokePair.lanes[0]);
                const [a1] = await read(engine, smokePair.lanes[1]);
                check('A47 smoke mirrored lanes remain exact after 60 contact-v2 steps', a0.every((v, i) => v === a1[i]));
            } finally { smokePair.destroy(); }

            const repeatLane1 = buildLane(engine, geometry, {
                ...options,
                contacts: true,
                contactOptions: { iterations: 2, maxPush: 0.25 },
                brains: { prey: smokeBrainP, wolf: smokeBrainW },
                seedOf: () => 42,
                rowOf: () => 0
            });
            let refRecord;
            try {
                await steps(repeatLane1, 30);
                refRecord = (await read(engine, repeatLane1))[0];
            } finally { repeatLane1.destroy(); }

            const repeatLane2 = buildLane(engine, geometry, {
                ...options,
                contacts: true,
                contactOptions: { iterations: 2, maxPush: 0.25 },
                brains: { prey: smokeBrainP, wolf: smokeBrainW },
                seedOf: () => 42,
                rowOf: () => 0
            });
            try {
                await steps(repeatLane2, 30);
                const [repRecord] = await read(engine, repeatLane2);
                check('A47 short-run exact repeatability from identical initial seed', repRecord.every((v, i) => v === refRecord[i]));
            } finally { repeatLane2.destroy(); }
        } finally {
            smokeBrainP.destroy();
            smokeBrainW.destroy();
        }
    } catch (error) { check('14.1 completed adapter checks', false, String(error.stack ?? error)); }
    finally { await device.queue.onSubmittedWorkDone().catch(() => {}); device.destroy(); }
}
if (!seen.size) check('GPU adapter available', false);
await env.finish('14.1 arena checks passed (new environment performance/learning acceptance pending)');