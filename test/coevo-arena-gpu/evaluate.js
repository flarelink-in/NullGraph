import { NullGraph } from '../../dist/index.js';
import { MlpBrain } from '../../dist/agents.js';
import { initNetwork } from '../../tools/agents/mlp-train.mjs';
import { buildLane, readBuffers } from '../../tools/agents/scenario/herd-wolves-arenas.mjs';
import { decodeDiagnostics } from '../../tools/agents/coevolve-diagnostics.mjs';
import { RULE_PAIR, LABEL_PAIR, baselineMatchups, baselineComparisons, crossplayMatchups,
    crossplaySummary, summarize, replay, evaluationGeometry, playMatchup } from '../../tools/agents/coevolve-evaluate.mjs';
import { harnessEnv } from '../gpu/harness-env.mjs';

const env = harnessEnv('__COEVO_EVALUATE_RESULTS__'), { check, log } = env, seen = new Set();
const experiment = { P: 64, W: 16, L: 36, gap: 10, environment: 'contact-v3', contacts: false,
    contactOptions: {}, catchMargin: 0.05, spawnSeparation: 2, thinkEvery: 3, dt: 1 / 60 };
const geometry = evaluationGeometry(experiment, 20), seeds = Array.from({ length: 20 }, (_, i) => 19001 + i);
const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const close = (got, want) => got.every((v, k) => Math.abs(v - want[k]) <= 1e-4 * Math.max(1, Math.abs(want[k])));

// CPU twin of the diagnostics stage: tile a's 16 sums from records copied back after a step.
function sampleOracle(lay, records, a, wallDistance) {
    const f = new Float32Array(records.buffer, records.byteOffset, records.length), sums = new Array(16).fill(0);
    const lo = [lay.origin[0] + a % lay.cols * lay.pitch, lay.origin[1] + Math.floor(a / lay.cols) * lay.pitch];
    const live = [...Array.from({ length: lay.wolves }, (_, s) => a * lay.wolves + s),
        ...Array.from({ length: lay.prey }, (_, s) => lay.wolfTotal + a * lay.prey + s)]
        .filter(i => records[i * 16 + 13] & 1 && records[i * 16 + 15] === a);
    for (const [k, i] of live.entries()) {
        const x = i * 16, wolf = i < lay.wolfTotal, s = wolf ? 1 : 0;
        sums[s]++;
        const wall = Math.min(f[x] - lo[0], f[x + 2] - lo[1], lay.L - f[x] + lo[0], lay.L - f[x + 2] + lo[1]) - f[x + 3];
        if (wall <= wallDistance) sums[2 + s]++;
        sums[4 + s] += Math.hypot(f[x + 4], f[x + 6]);
        if (wolf && f[x + 11] > 0) sums[6]++;
        let nearest = Infinity;
        for (const [m, j] of live.entries()) {
            if (j === i) continue;
            const y = j * 16, dist = Math.hypot(f[x] - f[y], f[x + 2] - f[y + 2]);
            if (wolf && j >= lay.wolfTotal) nearest = Math.min(nearest, dist);
            if (m < k) continue;
            sums[12]++;
            const p = f[x + 3] + f[y + 3] - dist;
            if (p > 0) { sums[9]++; sums[10] += p; sums[11] = Math.max(sums[11], p); }
        }
        if (wolf && nearest < Infinity) { sums[7] += nearest; sums[8]++; }
    }
    return sums;
}
for (const powerPreference of ['high-performance', 'low-power']) {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference });
    if (!adapter) continue;
    const key = JSON.stringify([adapter.info.vendor, adapter.info.device, adapter.info.description]);
    if (seen.has(key)) continue; seen.add(key);
    log(`14.4 evaluator on ${key}`);
    const device = await adapter.requestDevice(), engine = new NullGraph(), tables = new Map();
    try {
        const ready = await engine.init(env.canvas, { device, width: 8, height: 8 });
        if (!ready.success) throw new Error(ready.error);
        const run = async lane => {
            for (let t = 0; t < 3; t++) {
                const started = performance.now(); engine.render(lane.chain); await device.queue.onSubmittedWorkDone();
                if (performance.now() - started > 1500) throw new Error('evaluation fixture safety stop');
            }
            return { steps: 3 };
        };
        const subject = { id: 'gen-0000' };
        for (const species of ['prey', 'wolf']) {
            const network = initNetwork([{ input: 24, output: 16, activation: 'tanh' },
                { input: 16, output: 16, activation: 'tanh' }, { input: 16, output: 2, activation: 'tanh' }], 23);
            subject[species] = { kind: 'network', id: subject.id, network };
            if (!tables.has(subject.id)) tables.set(subject.id, {});
            tables.get(subject.id)[species] = MlpBrain.byGenome(engine, [network]);
        }
        const plan = baselineMatchups(subject, [RULE_PAIR, LABEL_PAIR]), catches = [];
        for (const game of plan) {
            const plain = await playMatchup(engine, { experiment, geometry, seeds, game, tables, T: 3, run });
            const createBuffer = device.createBuffer;
            let maps = 0, diagnostic;
            device.createBuffer = function (descriptor) {
                if (descriptor.usage & GPUBufferUsage.MAP_READ) maps++;
                return createBuffer.call(this, descriptor);
            };
            try {
                diagnostic = await playMatchup(engine, { experiment, geometry, seeds, game, tables, T: 3, run,
                    diagnostics: { steps: 3, stride: 2 } });
            } finally { device.createBuffer = createBuffer; }
            check(`A53 single final readback ${game.id}`, maps === 1);
            check(`A53 diagnostic catch invariance ${game.id}`, equal(plain.catches, diagnostic.catches));
            check(`A53 history length and catch accounting ${game.id}`, diagnostic.diagnostics.episodes.every((e, i) =>
                e.history.length === 2 && e.history[1].tick === 2 && e.catches === plain.catches[i]));
            catches.push(plain.catches);
        }
        const bootstrap = { seed: 77, resamples: 10000, level: 0.95 }, comparisons = baselineComparisons(subject, [RULE_PAIR, LABEL_PAIR]);
        const results = summarize(plan, catches, comparisons, bootstrap);
        const stored = { mode: 'baselines', plan: plan.map(m => ({ id: m.id, prey: m.prey.id, wolf: m.wolf.id })),
            comparisons, bootstrap, results };
        check('A54 evaluator raw-data replay', equal(replay(stored), { results, crossplay: null }));
        const cross = crossplayMatchups([subject]), summary = crossplaySummary([subject], cross, [catches[0]], [subject.id], bootstrap);
        check('A54 cross-play baseline cell and signed anchors', summary.mean[0][0] === results.matchups[0].interval.mean &&
            summary.anchorScores.prey[0].mean === -summary.anchorScores.wolf[0].mean);
        const contactExperiment = { ...experiment, contacts: true,
            contactOptions: { iterations: 2, maxPush: 0.25, relaxation: 1, compliance: 0,
                damping: 0.05, mass: 'equal', maxScanPerCell: 64, cellSize: 2 } };
        const contactGeometry = evaluationGeometry(contactExperiment, 20);
        let plainRecords, diagnosticRecords;
        const recordedRun = diagnostic => async lane => {
            const observed = await run(lane);
            const [records] = await readBuffers(device, [{ buffer: lane.sys.buffer, words: lane.lay.count * 16 }]);
            if (diagnostic) diagnosticRecords = records; else plainRecords = records;
            return observed;
        };
        const plainContact = await playMatchup(engine, { experiment: contactExperiment, geometry: contactGeometry,
            seeds, game: plan[0], tables, T: 3, run: recordedRun(false) });
        const diagnosticContact = await playMatchup(engine, { experiment: contactExperiment, geometry: contactGeometry,
            seeds, game: plan[0], tables, T: 3, run: recordedRun(true), diagnostics: { steps: 3, stride: 2 } });
        check('A53 contacts-on records and catches unchanged by diagnostics', equal(plainContact.catches, diagnosticContact.catches) &&
            equal(Array.from(plainRecords), Array.from(diagnosticRecords)));

        // Nonvacuous fixture: one caught prey at a known cell, living/dead denominators, rest,
        // overlaps and wall proximity. Records copied back give an independent CPU geometric oracle.
        const fixtureGeometry = { arenas: 4, wolves: 16, prey: 64, cols: 2, L: 36, gap: 10 };
        const lane = buildLane(engine, fixtureGeometry, { ...experiment, diagnostics: { steps: 1, stride: 1 },
            seedOf: () => 11, rowOf: () => 0, fixture(init, { lo, wolf, slot }) {
                init.position = [lo[0] + 0.65 + slot * 0.01, 0.08, lo[1] + 0.65];
                init.frozen = true; init.age = 1;
                if (wolf && slot !== 0) init.user0 = 2;
            } });
        try {
            // First wolf is dead in tile 1: reduction must elect another living record.
            lane.sys.kill(16); lane.sys.upload();
            engine.render(lane.chain); await device.queue.onSubmittedWorkDone();
            const [records, samples, locations, counts] = await readBuffers(device, [
                { buffer: lane.sys.buffer, words: lane.lay.count * 16 },
                { buffer: lane.diagnosticBuffers.samples, words: lane.diagnostic.words },
                { buffer: lane.diagnosticBuffers.catches, words: lane.diagnostic.words },
                { buffer: lane.counters, words: 4 }]);
            const d = decodeDiagnostics(lane.lay, lane.diagnostic, samples, locations), f = new Float32Array(records.buffer);
            check('A53 forced catches at known pre-respawn cell', d.episodes.every((e, a) => (a === 1 ? e.catches === 0 : e.catches > 0) && e.catches === counts[a] &&
                e.history[0].catchLocations[0] === counts[a]));
            for (let a = 0; a < 4; a++) {
                const indices = [...Array.from({ length: 16 }, (_, i) => a * 16 + i),
                    ...Array.from({ length: 64 }, (_, i) => 64 + a * 64 + i)];
                let pairs = 0, penetration = 0, max = 0;
                for (let i = 0; i < indices.length; i++) for (let j = i + 1; j < indices.length; j++) {
                    const x = indices[i] * 16, y = indices[j] * 16;
                    if (!(records[x + 13] & 1) || !(records[y + 13] & 1)) continue;
                    const p = f[x + 3] + f[y + 3] - Math.hypot(f[x] - f[y], f[x + 2] - f[y + 2]);
                    if (p > 0) { pairs++; penetration += p; max = Math.max(max, p); }
                }
                const raw = d.episodes[a].history[0].raw;
                check(`A53 overlap CPU oracle tile ${a}`, raw[9] === pairs && Math.abs(raw[10] - penetration) < 0.02 && Math.abs(raw[11] - max) < 1e-5);
                const wolves = a === 1 ? 15 : 16, alive = wolves + 64;
                check(`A53 rest and body denominators tile ${a}`, raw[0] === 64 && raw[1] === wolves && raw[6] === wolves && raw[12] === alive * (alive - 1) / 2);
                const lo = [lane.lay.origin[0] + a % 2 * lane.lay.pitch, lane.lay.origin[1] + Math.floor(a / 2) * lane.lay.pitch];
                const totals = [0, 0, 0, 0]; let nearest = 0;
                for (const index of indices) {
                    const x = index * 16; if (!(records[x + 13] & 1)) continue;
                    const wolf = index < lane.lay.wolfTotal, s = wolf ? 1 : 0;
                    totals[s] += Math.hypot(f[x + 4], f[x + 6]);
                    const wall = Math.min(f[x] - lo[0], f[x + 2] - lo[1], 36 - f[x] + lo[0], 36 - f[x + 2] + lo[1]) - f[x + 3];
                    if (wall <= 1) totals[2 + s]++;
                    if (wolf) nearest += Math.min(...indices.slice(16).map(j => Math.hypot(f[x] - f[j * 16], f[x + 2] - f[j * 16 + 2])));
                }
                check(`A53 wall/speed/nearest CPU oracle tile ${a}`, Math.abs(raw[4] - totals[0]) < 1e-4 &&
                    Math.abs(raw[5] - totals[1]) < 1e-4 && raw[2] === totals[2] && raw[3] === totals[3] && Math.abs(raw[7] - nearest) < 0.001);
            }
        } finally { lane.destroy(); }

        // Over time: tile a's first wolf wakes after tick 2 + 2a and catches its tile's prey at tick
        // 3 + 2a (tile 3's falls after the run); the other wolves rest throughout and the prey move.
        // Second-row tiles start one frozen prey across their low wall, in the gap: steering keeps
        // moving bodies in their tile itself, so only containment moves this one, before the sample.
        // Records and counters read back after every step (a test, not the production loop) are the
        // oracle for each sampled tick and each stride window.
        const timed = buildLane(engine, fixtureGeometry, { ...experiment, diagnostics: { steps: 8, stride: 2 },
            seedOf: () => 11, rowOf: () => 0, fixture(init, { a, lo, wolf, slot }) {
                const outside = slot === 16 && a >= 2;
                init.position = [lo[0] + 0.65 + slot * 0.01, 0.08, lo[1] + (outside ? -0.2 : 0.65 + slot % 3 * 0.25)];
                init.frozen = wolf || outside;
                if (wolf) init.user0 = slot ? 2 : (2.5 + 2 * a) / 60;
            } });
        try {
            const after = [];
            for (let t = 0; t < 8; t++) {
                engine.render(timed.chain); await device.queue.onSubmittedWorkDone();
                after.push(await readBuffers(device, [{ buffer: timed.sys.buffer, words: timed.lay.count * 16 },
                    { buffer: timed.counters, words: 4 }]));
            }
            const [samples, locations] = await readBuffers(device, ['samples', 'catches'].map(key =>
                ({ buffer: timed.diagnosticBuffers[key], words: timed.diagnostic.words })));
            const d = decodeDiagnostics(timed.lay, timed.diagnostic, samples, locations);
            const windows = [0, 1, 2, 3].map(a => [0, 1, 2, 3].map(w => after[2 * w + 1][1][a] - (w ? after[2 * w - 1][1][a] : 0)));
            check('A53 timed catches fall in the expected windows', windows.every((row, a) => row.every((n, w) => (n > 0) === (w === a + 1))));
            check('A53 catch histogram windows equal counter deltas', d.episodes.every((e, a) => e.history.every((h, w) =>
                h.catchLocations[0] === windows[a][w] && h.catchLocations.reduce((s, n) => s + n, 0) === windows[a][w])));
            const wants = [0, 1, 2, 3].map(a => [0, 1, 2, 3].map(w => sampleOracle(timed.lay, after[2 * w][0], a, 1)));
            for (let a = 0; a < 4; a++) check(`A53 samples over time equal their ticks' records, tile ${a}`,
                d.episodes[a].history.every((h, w) => h.tick === 2 * w && close(h.raw, wants[a][w])));
            const all = wants.flat();
            check('A53 timed fixture nonvacuous: wolves wake, prey move', all.some(s => s[6] === 15) && all.some(s => s[6] === 16) &&
                all.every(s => s[4] > 0));
        } finally { timed.destroy(); }
    } finally {
        for (const pair of tables.values()) for (const brain of Object.values(pair)) brain.destroy();
        device.destroy();
    }
}
if (!seen.size) env.fatal('no adapter');
else await env.finish('14.4 evaluator checks complete');