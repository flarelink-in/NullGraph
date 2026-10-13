// Step 7, part 2 (docs/agents-cloning-design.md §10): the training tools' headless checks.
// Part of `npm run test:agents`; build first (`npm run build:lib`).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as A from '../dist/agents.js';
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';
import { DRIVER, LABEL, LABEL_DRIVING, TAG, HERD_DEFAULTS, herdDefinition, herdRule, herdWolves } from '../tools/agents/scenario/herd-wolves.mjs';
import { ROW, ROW_WORDS, Rows, flagsReason, hashText, keepRow, readDataset, rowFlags, rowFloats, splitRows, trainingSet, writeDataset } from '../tools/agents/dataset.mjs';
import { isFresh, recordEpisodes } from '../tools/agents/recorder.mjs';
import { RUNS, addDecision, addTally, describeCatches, describeTally, evaluateRun, newTally } from '../tools/agents/evaluator.mjs';
import { modelText, networkFromModel } from '../tools/agents/model.mjs';
import { ONNX_OPSET, onnxFromNetwork } from '../tools/agents/onnx.mjs';
import { flatParams, initNetwork, lossAndGradient, meanAbsoluteError, quantile, trainNetwork } from '../tools/agents/mlp-train.mjs';

const words = values => new Uint32Array(new Float32Array(values).buffer);

/** A small dataset with awkward values: negative zero, a denormal, the largest f32. */
function sample() {
    const definition = herdDefinition(), floats = rowFloats(definition), prey = new Rows(floats), wolf = new Rows(floats);
    for (let row = 0; row < 3000; row++) {
        const obs = Float32Array.from({ length: definition.obsDim }, (_, k) => A.rand01(row, k, 1) * 2 - 1);
        obs[0] = [-0, 1e-45, 3.4028234663852886e38][row % 3];
        const label = [A.rand01(row, 0, 2) * 2 - 1, A.rand01(row, 1, 2)], act = [A.rand01(row, 0, 3) * 2 - 1, A.rand01(row, 1, 3)];
        (row % 4 ? prey : wolf).push(words(obs), words(label), words(act), row % 3, row * 3, row, rowFlags(row % 7 === 0, row % 6));
    }
    const header = { scenario: { name: 'herd-wolves' }, definition: { obsDim: definition.obsDim, actDim: definition.actDim },
        episodes: [{ seed: 11, heldOut: false }, { seed: 12, heldOut: false }, { seed: 13, heldOut: true }] };
    return { definition, header, species: { prey: prey.finish(), wolf: wolf.finish() } };
}

test('U24: a dataset written and read back is byte-identical; a changed or short file is refused', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'ng-dataset-'));
    try {
        const { definition, header, species } = sample();
        assert.equal(rowFloats(definition), 28);
        const written = await writeDataset(dir, header, species), back = await readDataset(dir);
        assert.deepEqual(back.header, written);
        assert.deepEqual(Object.keys(back.species), ['prey', 'wolf']);
        for (const name of ['prey', 'wolf']) {
            assert.equal(back.species[name].rows, species[name].rows);
            // Compared as words: a float comparison would call -0 and 0, or two NaNs, equal or unequal wrongly.
            assert.deepEqual(new Uint32Array(back.species[name].f32.buffer), new Uint32Array(species[name].f32.buffer));
            assert.deepEqual(back.species[name].u32, species[name].u32);
        }
        assert.equal(species.prey.rows + species.wolf.rows, 3000);
        assert.equal(written.files['prey.f32'].bytes, species.prey.rows * 28 * 4);
        assert.equal(written.files['wolf.u32'].bytes, species.wolf.rows * ROW_WORDS * 4);
        assert.match(written.files['prey.f32'].sha256, /^[0-9a-f]{64}$/);

        const file = join(dir, 'prey.f32'), bytes = readFileSync(file);
        const flipped = Buffer.from(bytes);
        flipped[1234] ^= 1;
        writeFileSync(file, flipped);
        await assert.rejects(readDataset(dir), /dataset: prey\.f32 does not match header\.json's SHA-256\. It was changed or cut short\./);
        writeFileSync(file, bytes.subarray(0, bytes.length - 4));
        await assert.rejects(readDataset(dir), new RegExp(`dataset: prey\\.f32 is ${bytes.length - 4} bytes; header\\.json promises ` +
            `${bytes.length} \\(${species.prey.rows} rows × 112 bytes\\)\\.`));
        writeFileSync(file, bytes);
        await readDataset(dir);

        // A row is the observation, then the label, then the driver's action; its words are
        // episode, tick, agent, flags. A reader that swapped two of them would still round-trip.
        const rows = new Rows(28), obs = Float32Array.from({ length: 24 }, (_, k) => k + 1);
        rows.push(words(obs), words([-0.25, 0.5]), words([0.75, -1]), 2, 1799, 4095, rowFlags(true, 3));
        const one = rows.finish();
        assert.deepEqual([...one.f32], [...obs, -0.25, 0.5, 0.75, -1]);
        assert.deepEqual([...one.u32], [2, 1799, 4095, (3 << 4) | 1]);
        assert.deepEqual([one.u32[ROW.EPISODE], one.u32[ROW.TICK], one.u32[ROW.AGENT], one.u32[ROW.FLAGS] & 1, flagsReason(one.u32[ROW.FLAGS])],
            [2, 1799, 4095, 1, 3]);
        assert.throws(() => rows.push(words([1]), words([1, 2]), words([1, 2]), 0, 0, 0, 0), /a row has 5 floats; this dataset's rows have 28/);
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});

test('U25: the rule\'s texts parse in the think wrapper and differ only in their four functions; tags round-trip', () => {
    const definition = herdDefinition(), wrapped = seams => A.buildAgentThinkWGSL(definition, herdRule(seams), HERD_DEFAULTS.thinkEvery);
    for (const seams of [DRIVER, LABEL, LABEL_DRIVING]) new WgslReflect(wrapped(seams));
    // With the four functions taken out, every text is the same text.
    const body = seams => {
        let text = herdRule(seams);
        for (const name of ['wander', 'coin', 'resting', 'tag']) {
            const begin = text.indexOf(`fn ${name}(`), end = text.indexOf('\nfn ', begin);   // up to the next function
            assert.ok(begin >= 0 && end > begin, `fn ${name} is missing`);
            text = text.slice(0, begin) + text.slice(end + 1);
        }
        return text;
    };
    assert.equal(body(LABEL), body(DRIVER));
    assert.equal(body(LABEL_DRIVING), body(DRIVER));
    assert.ok(body(DRIVER).includes('fn think(i: u32, species: u32, tick: u32)') && !/fn (wander|coin|resting|tag)\(/.test(body(DRIVER)));
    // The driver keeps the demo's hidden inputs; the label has none of them and carries the step.
    const driver = herdRule(DRIVER), label = herdRule(LABEL);
    assert.ok(driver.includes('0x9e3779b1u') && driver.includes('(i % 2u) == 0u') && driver.includes('agents[i].state.w > 0.0'));
    assert.ok(driver.includes('fn tag(i: u32, tick: u32, why: u32) -> u32 { return why; }'));
    assert.ok(!label.includes('0x9e3779b1u') && !label.includes('% 2u') && !label.includes('state.w'));
    assert.ok(label.includes('fn wander(i: u32, tick: u32) -> f32 { return 0.0; }') && label.includes('fn coin(i: u32) -> bool { return true; }') &&
        label.includes('fn resting(i: u32) -> bool { return false; }'));
    assert.ok(label.includes('return ((tick + 1u) << 8u) | (((agents[i].ids.y >> 1u) & 1u) << 4u) | why;'));
    assert.throws(() => herdRule({ wander: true, coin: true, resting: true }), /herdRule needs a boolean "tag"/);
    assert.deepEqual(Object.keys(herdWolves.texts()), ['driver', 'label', 'catches', 'respawn']);
    assert.equal(herdWolves.texts().label, label);

    for (const [tick, frozen, why] of [[0, 0, 0], [0, 1, 5], [299, 1, 3], [1799, 0, 4], [(1 << 24) - 2, 1, 15]]) {
        const word = TAG.pack(tick, frozen, why);
        assert.deepEqual([TAG.tick(word), TAG.frozen(word), TAG.why(word)], [tick, frozen, why]);
        assert.notEqual(word, 0);
    }
    assert.equal(TAG.tick(0), -1);                                  // a row no pass has written
    assert.equal(flagsReason(rowFlags(true, 5)), 5);
    assert.equal(rowFlags(true, 5) & 1, 1);
    assert.equal(rowFlags(false, 3), 3 << 4);
});

test('U27: the row sampler repeats, keeps the asked share, and a held-out episode never trains', async () => {
    for (const [species, rate] of [[0, 1 / 16], [1, 1 / 4], [0, 0.9]]) {
        let kept = 0;
        const first = [];
        for (let row = 0; row < 100_000; row++) {
            const keep = keepRow(row % 4096, Math.floor(row / 4096) * 6 + 5, species, rate);
            first.push(keep);
            if (keep) kept++;
        }
        assert.ok(Math.abs(kept / 100_000 - rate) < 0.01, `kept ${kept} of 100,000 at rate ${rate}`);
        for (let row = 0; row < 100_000; row += 997) {
            assert.equal(keepRow(row % 4096, Math.floor(row / 4096) * 6 + 5, species, rate), first[row]);
        }
    }
    assert.ok(keepRow(7, 11, 0, 1) && keepRow(7, 11, 1, 2));
    // Another species draws from another stream: the two keep different rows.
    let differ = 0;
    for (let agent = 0; agent < 4096; agent++) if (keepRow(agent, 5, 0, 0.5) !== keepRow(agent, 5, 1, 0.5)) differ++;
    assert.ok(differ > 1500 && differ < 2600, `${differ} of 4,096 rows differ between species`);

    // A decision is kept once: in the stretch that follows the last copy, up to this step.
    assert.deepEqual([5, 6, 11, 12].map(tick => isFresh(tick, 5, 11)), [false, true, true, false]);
    assert.ok(isFresh(0, -1, 5) && !isFresh(TAG.tick(0), -1, 5), 'a row no pass has written is never fresh');
    // A copy rarer than a think cycle is fine; one more frequent would see some agents twice and is refused.
    await assert.rejects(recordEpisodes(null, herdWolves, { stride: 2 }),
        /record: stride 2 must be a whole number of steps, at least thinkEvery \(3\)\./);
    await assert.rejects(recordEpisodes(null, herdWolves, { seeds: [] }), /record: needs at least one seed/);
    await assert.rejects(recordEpisodes(null, herdWolves, { steps: 5 }), /steps \(5\) of at least one stride \(6\)/);

    const { header, species } = sample(), u32 = species.prey.u32, rows = species.prey.rows;
    const split = splitRows(header, u32), all = splitRows(header, u32, { dropFrozen: false });
    const episode = row => u32[row * ROW_WORDS + ROW.EPISODE], frozen = row => u32[row * ROW_WORDS + ROW.FLAGS] & 1;
    assert.ok(split.train.length > 0 && split.heldOut.length > 0);
    assert.ok(split.train.every(row => episode(row) !== 2 && !frozen(row)));
    assert.ok(split.heldOut.every(row => episode(row) === 2 && !frozen(row)));
    assert.equal(all.train.length + all.heldOut.length, rows);
    assert.ok(all.train.every(row => episode(row) !== 2) && all.heldOut.every(row => episode(row) === 2));
    assert.ok(all.train.length > split.train.length);
    assert.throws(() => splitRows({ episodes: header.episodes.slice(0, 2) }, u32), /names episode 2; header\.json lists 2/);
});

/** A training set built row by row: inputs, then targets, in one f32 array, as a dataset holds them. */
function makeSet(rows, inputs, outputs, fill) {
    const stride = inputs + outputs, data = new Float32Array(rows * stride);
    for (let r = 0; r < rows; r++) fill(r, data.subarray(r * stride, r * stride + inputs), data.subarray(r * stride + inputs, (r + 1) * stride));
    return { data, stride, x: 0, y: inputs, inputs, outputs, rows: Uint32Array.from({ length: rows }, (_, r) => r) };
}
/** Mean-squared error of an f32 network, by the library's mlpForward. */
function meanSquared(network, set) {
    let sum = 0;
    for (const row of set.rows) {
        const at = row * set.stride, out = A.mlpForward(network, set.data.subarray(at, at + set.inputs));
        for (let j = 0; j < set.outputs; j++) sum += (out[j] - set.data[at + set.y + j]) ** 2;
    }
    return sum / (set.rows.length * set.outputs);
}
const bitsOf = network => network.flatMap(l => [...new Uint32Array(l.weights.buffer), ...new Uint32Array(l.biases.buffer)]);

test('U11: the trainer fits a linear map and XOR, repeats byte for byte, and its gradients are right', () => {
    // y = Ax + b with 10 inputs and 3 outputs, no noise.
    const matrix = Float64Array.from({ length: 30 }, (_, k) => A.rand01(k, 1, 7) * 2 - 1), offset = [0.25, -0.5, 0.125];
    const linearRow = stream => (r, x, y) => {
        for (let k = 0; k < 10; k++) x[k] = A.rand01(r, k, stream) * 2 - 1;
        for (let j = 0; j < 3; j++) {
            let s = offset[j];
            for (let k = 0; k < 10; k++) s += matrix[j * 10 + k] * x[k];
            y[j] = s;
        }
    };
    const linear = makeSet(4096, 10, 3, linearRow(8)), shape = [{ input: 10, output: 3, activation: 'identity' }];
    const options = { epochs: 60, batch: 64, rate: 1e-2, finalRate: 1e-5, seed: 1 };
    const fit = trainNetwork(shape, linear, options);
    assert.ok(meanSquared(fit.network, linear) < 1e-6, `linear fit: mean-squared error ${meanSquared(fit.network, linear)}`);

    // The same rows and seed give the same weights, byte for byte; another seed does not.
    assert.deepEqual(bitsOf(trainNetwork(shape, linear, options).network), bitsOf(fit.network));
    assert.notDeepEqual(bitsOf(trainNetwork(shape, linear, { ...options, seed: 2 }).network), bitsOf(fit.network));

    // XOR needs the hidden layer.
    const xor = makeSet(256, 2, 1, (r, x, y) => {
        x[0] = r & 1 ? 1 : -1;
        x[1] = r & 2 ? 1 : -1;
        y[0] = (r & 1) ^ ((r >> 1) & 1);
    });
    const xorNet = trainNetwork([{ input: 2, output: 4, activation: 'tanh' }, { input: 4, output: 1, activation: 'sigmoid' }], xor,
        { epochs: 300, batch: 32, rate: 3e-2, finalRate: 1e-3, seed: 3 }).network;
    for (const [a, b] of [[0, 0], [0, 1], [1, 0], [1, 1]]) {
        const out = A.mlpForward(xorNet, new Float32Array([a ? 1 : -1, b ? 1 : -1]))[0];
        assert.ok(Math.abs(out - (a ^ b)) < 0.05, `XOR(${a}, ${b}) gave ${out}`);
    }

    // The gradient against central differences, through every activation, at the deepest shape.
    const deep = [{ input: 5, output: 6, activation: 'tanh' }, { input: 6, output: 4, activation: 'relu' },
        { input: 4, output: 3, activation: 'sigmoid' }, { input: 3, output: 2, activation: 'identity' }];
    const params = flatParams(initNetwork(deep, 5)), batch = makeSet(16, 5, 2, (r, x, y) => {
        for (let k = 0; k < 5; k++) x[k] = A.rand01(r, k, 9) * 2 - 1;
        for (let j = 0; j < 2; j++) y[j] = A.rand01(r, j, 10) * 2 - 1;
    });
    const { gradient } = lossAndGradient(deep, params, batch);
    let worst = 0, active = 0;
    for (let k = 0; k < params.length; k++) {
        const h = 1e-6, up = params.slice(), down = params.slice();
        up[k] += h;
        down[k] -= h;
        const numeric = (lossAndGradient(deep, up, batch).loss - lossAndGradient(deep, down, batch).loss) / (2 * h);
        worst = Math.max(worst, Math.abs(numeric - gradient[k]) / Math.max(Math.abs(numeric), Math.abs(gradient[k]), 1e-5));
        if (Math.abs(gradient[k]) > 1e-5) active++;
    }
    assert.ok(worst < 1e-4, `worst relative gradient error ${worst.toExponential(2)}`);
    assert.ok(active > params.length / 2, `${active} of ${params.length} gradients are large enough to compare`);
});

test('U11: the best pass wins, errors split by group, and a species with nothing to train on is refused', () => {
    // A constant, large step makes the held-out error jump about, so the best pass is not the last.
    const matrix = Float64Array.from({ length: 6 }, (_, k) => A.rand01(k, 2, 7) * 2 - 1);
    const row = stream => (r, x, y) => {
        for (let k = 0; k < 3; k++) x[k] = A.rand01(r, k, stream) * 2 - 1;
        for (let j = 0; j < 2; j++) y[j] = Math.tanh(matrix[j * 3] * x[0] + matrix[j * 3 + 1] * x[1] + matrix[j * 3 + 2] * x[2]);
    };
    const train = makeSet(1024, 3, 2, row(11)), heldOut = makeSet(512, 3, 2, row(12));
    const shape = [{ input: 3, output: 8, activation: 'tanh' }, { input: 8, output: 2, activation: 'tanh' }];
    const run = trainNetwork(shape, train, { epochs: 12, batch: 16, rate: 0.2, finalRate: 0.2, seed: 4, heldOut });
    const scores = run.history.map(e => e.heldOut[0] + e.heldOut[1]), lowest = Math.min(...scores);
    assert.equal(run.best, scores.indexOf(lowest));
    assert.ok(run.best < run.history.length - 1, `the best pass is ${run.best} of ${run.history.length}: the test needs one before the last`);
    const kept = meanAbsoluteError(run.network, heldOut);
    assert.ok(Math.abs(kept.mae[0] + kept.mae[1] - lowest) < 1e-5, 'the network returned is the best pass, judged in f32');
    assert.notDeepEqual(bitsOf(run.network), bitsOf(run.last));

    // Split by group: every row counted once, and each group's error is its rows' mean.
    const split = meanAbsoluteError(run.network, heldOut, r => r % 3);
    assert.deepEqual(Object.keys(split.groups), ['0', '1', '2']);
    assert.equal(Object.values(split.groups).reduce((n, g) => n + g.rows, 0), 512);
    for (let j = 0; j < 2; j++) {
        const pooled = Object.values(split.groups).reduce((s, g) => s + g.mae[j] * g.rows, 0) / 512;
        assert.ok(Math.abs(pooled - split.mae[j]) < 1e-12);
    }
    // Kept errors, row by row, average to the mean; quantiles read the sorted values.
    const each = meanAbsoluteError(run.network, heldOut, undefined, { keep: true });
    assert.ok(each.errors.every((e, j) => e.length === 512 && Math.abs(e.reduce((s, v) => s + v, 0) / 512 - each.mae[j]) < 1e-6));
    assert.equal(meanAbsoluteError(run.network, heldOut).errors, undefined);
    assert.deepEqual([quantile([3, 1, 2], 0.5), quantile([4, 1, 3, 2], 0.9), quantile([5], 0), quantile([5], 1)], [2, 4, 5, 5]);

    // A dataset's species, read in place: observation in, label out; refused by name when empty.
    const { header, species } = sample(), set = trainingSet({ header, species }, 'prey');
    const obsDim = header.definition.obsDim, first = set.train.rows[0];
    assert.equal(set.train.data, species.prey.f32);
    assert.deepEqual(set.train.data.subarray(first * set.train.stride + set.train.y, first * set.train.stride + set.train.y + 2),
        species.prey.f32.subarray(first * set.train.stride + obsDim, first * set.train.stride + obsDim + 2));
    assert.equal(set.reason(first), flagsReason(species.prey.u32[first * ROW_WORDS + ROW.FLAGS]));
    // Frozen rows never train or judge; the held-out episode (2) only judges.
    const word = (r, w) => species.prey.u32[r * ROW_WORDS + w];
    assert.ok(set.train.rows.length > 0 && set.heldOut.rows.length > 0);
    assert.ok(set.train.rows.every(r => !(word(r, ROW.FLAGS) & 1) && word(r, ROW.EPISODE) !== 2));
    assert.ok(set.heldOut.rows.every(r => !(word(r, ROW.FLAGS) & 1) && word(r, ROW.EPISODE) === 2));
    const allHeldOut = { ...header, episodes: header.episodes.map(e => ({ ...e, heldOut: true })) };
    assert.throws(() => trainingSet({ header: allHeldOut, species }, 'prey'),
        /^Error: train: no rows for species "prey" after dropping frozen rows and the held-out episode\.$/);
    assert.throws(() => trainingSet({ header, species }, 'bear'), /train: the dataset has no species "bear"; it has prey, wolf\./);
    assert.throws(() => trainNetwork([{ input: 3, output: 2, activation: 'tanh' }], { ...train, rows: new Uint32Array(0) }), /train: the set has no rows\./);
    assert.throws(() => trainNetwork([{ input: 4, output: 2, activation: 'tanh' }], train), /train: the shape maps 4 → 2; the set has 3 → 2\./);
    assert.throws(() => initNetwork([{ input: 3, output: 65, activation: 'tanh' }, { input: 65, output: 2, activation: 'tanh' }]),
        /MlpBrain: layer 0 output must be 1–64\./);
});

test('U28: an evaluation counts each decision once, compares live agents only, and pools its seeds', async () => {
    const actNames = ['turn', 'throttle'], reasons = herdWolves.reasons, measures = [{ name: 'speed' }, { name: 'nearest prey' }];
    const act = Float32Array.of(0.5, 1, -0.25, 0.5, 9, 9), label = Float32Array.of(0.25, 1, 0.25, 0, 0, 0);
    const a = newTally(2, reasons.length, 2);
    assert.equal(addDecision(a, act, label, 0, false, 2), true);
    assert.equal(addDecision(a, act, label, 2, false, 4), true);
    assert.equal(addDecision(a, act, label, 4, true, 3), false);      // frozen: counted, compared with nothing
    a.measures[0].rows = 2;
    a.measures[0].sum = 5;
    const one = describeTally(a, actNames, reasons, measures);
    assert.deepEqual(one, {
        decisions: 3, frozen: 1, mae: { turn: 0.375, throttle: 0.25 },
        byReason: { flee: { rows: 1, mae: { turn: 0.25, throttle: 0 } }, wall: { rows: 1, mae: { turn: 0.5, throttle: 0.5 } } },
        measures: { speed: 2.5, 'nearest prey': null },
    });

    // Pooled seeds weigh every decision alike: a seed with more rows counts for more.
    const b = newTally(2, reasons.length, 2);
    for (let k = 0; k < 6; k++) addDecision(b, Float32Array.of(1, 0), Float32Array.of(0, 0), 0, false, 2);
    b.measures[1].rows = 4;
    b.measures[1].sum = 2;
    const pooled = describeTally(addTally(addTally(newTally(2, reasons.length, 2), a), b), actNames, reasons, measures);
    assert.equal(pooled.decisions, 9);
    assert.equal(pooled.frozen, 1);
    assert.equal(pooled.mae.turn, (0.25 + 0.5 + 6) / 8);
    assert.deepEqual(pooled.byReason.flee, { rows: 7, mae: { turn: 6.25 / 7, throttle: 0 } });
    assert.deepEqual(pooled.measures, { speed: 2.5, 'nearest prey': 0.5 });
    assert.deepEqual(describeTally(newTally(2, reasons.length, 2), actNames, reasons, measures).mae, { turn: null, throttle: null });

    assert.deepEqual(describeCatches([90, 100, 110]), { values: [90, 100, 110], mean: 100, low: 90 / 100 - 1, high: 110 / 100 - 1 });
    assert.deepEqual(Object.keys(RUNS), ['T', 'C', 'S', 'Sp', 'Sw']);
    assert.deepEqual(RUNS.Sp, { prey: 'student', wolf: 'teacher' });

    // The scenario's measures read one observation row.
    const definition = herdDefinition(), [speed, nearest, alone] = herdWolves.measures(definition);
    const obs = new Float32Array(2 * definition.obsDim), row = definition.obsDim, at = name => row + definition.obsNames.indexOf(name);
    const prey = definition.speciesIndex('prey'), wolf = definition.speciesIndex('wolf');
    obs[at('self.speed')] = 0.5;
    assert.equal(speed.of(obs, row, prey), 0.5 * HERD_DEFAULTS.preySpeed);
    assert.equal(speed.of(obs, row, wolf), 0.5 * HERD_DEFAULTS.wolfSpeed);
    assert.equal(nearest.of(obs, row, prey), null);
    assert.equal(alone.of(obs, row, prey), 1);
    obs[at('prey[0].side')] = 0.375;
    obs[at('prey[0].forward')] = -0.5;
    obs[at('prey[0].present')] = 1;
    assert.equal(nearest.of(obs, row, prey), 0.625 * 8);
    assert.equal(alone.of(obs, row, prey), 0);
    assert.equal(nearest.of(obs, row, wolf), null);
    assert.equal(alone.of(obs, row, wolf), null);

    // Refused before anything is built.
    await assert.rejects(evaluateRun(null, herdWolves, RUNS.T, { stride: 2 }),
        /evaluate: stride 2 must be a whole number of steps, at least thinkEvery \(3\)\./);
    await assert.rejects(evaluateRun(null, herdWolves, RUNS.T, { seeds: [] }), /evaluate: needs at least one seed/);
    await assert.rejects(evaluateRun(null, herdWolves, { prey: 'teacher', wolf: 'rule' }),
        /evaluate: the driver of "wolf" must be teacher, label or student; got "rule"\./);
    await assert.rejects(evaluateRun(null, herdWolves, RUNS.Sw, { students: { prey: {} } }), /evaluate: no student network for species "wolf"\./);
});

test('U26 (the reader): a network file\'s layers come back as f32, and names that are not the definition\'s are refused', () => {
    const definition = herdDefinition();
    const model = { format: 'null-graph/mlp', version: 1, species: 'prey', obsNames: [...definition.obsNames], actNames: [...definition.actNames],
        layers: [{ input: 24, output: 2, activation: 'tanh', weights: Array.from({ length: 48 }, (_, k) => Math.fround(A.rand01(k, 0, 4) - 0.5)), biases: [0.125, -0.5] }] };
    const network = networkFromModel(model, definition, 'prey.mlp.json');
    assert.equal(network.length, 1);
    assert.deepEqual({ ...network[0], weights: null, biases: null }, { input: 24, output: 2, activation: 'tanh', weights: null, biases: null });
    assert.ok(network[0].weights instanceof Float32Array && network[0].biases instanceof Float32Array);
    assert.deepEqual([...network[0].weights], model.layers[0].weights);
    assert.deepEqual([...network[0].biases], [0.125, -0.5]);
    assert.equal(A.mlpForward(network, new Float32Array(24)).length, 2);

    const swapped = [...definition.obsNames];
    [swapped[2], swapped[3]] = [swapped[3], swapped[2]];
    assert.throws(() => networkFromModel({ ...model, obsNames: swapped }, definition, 'prey.mlp.json'),
        /model: prey\.mlp\.json obsNames\[2\] is "wolf\[0\]\.forward"; the definition has "wolf\[0\]\.side"\./);
    assert.throws(() => networkFromModel({ ...model, obsNames: definition.obsNames.slice(0, 23) }, definition, 'prey.mlp.json'),
        /obsNames\[23\] is nothing; the definition has "bounds\.forward"\./);
    assert.throws(() => networkFromModel({ ...model, actNames: ['turn', 'speed'] }, definition, 'prey.mlp.json'),
        /actNames\[1\] is "speed"; the definition has "throttle"\./);
    assert.throws(() => networkFromModel({ ...model, version: 2 }, definition, 'prey.mlp.json'),
        /model: prey\.mlp\.json is "null-graph\/mlp" version 2; this reader takes "null-graph\/mlp" version 1\./);
});

test('U26 (the writer): a network written and read back is bit-identical, and the same input gives the same bytes', () => {
    const definition = herdDefinition();
    // Awkward f32 values: a denormal, the largest and the smallest normal, and neighbours one bit apart.
    const next = value => new Float32Array(Uint32Array.of(new Uint32Array(Float32Array.of(value).buffer)[0] + 1).buffer)[0];
    const awkward = [1e-45, 3.4028234663852886e38, 1.1754943508222875e-38, 0.1, next(0.1), -1 / 3, 0];
    const weights = Float32Array.from({ length: 48 }, (_, k) => (k < awkward.length ? awkward[k] : A.rand01(k, 0, 5) * 2 - 1));
    const network = [{ input: 24, output: 2, activation: 'tanh', weights, biases: Float32Array.of(0.125, -0.7) }];
    const input = { species: 'prey', obsNames: [...definition.obsNames], actNames: [...definition.actNames], network,
        trainedOn: { seed: 1 }, measured: { rows: 3 } };
    const text = modelText(input), model = JSON.parse(text);
    assert.deepEqual(Object.keys(model), ['format', 'version', 'species', 'obsNames', 'actNames', 'layers', 'trainedOn', 'measured']);
    assert.equal(text, modelText(input));
    assert.ok(text.endsWith('}\n'));
    const back = networkFromModel(model, definition, 'prey.mlp.json');
    const bits = array => [...new Uint32Array(array.buffer, array.byteOffset, array.length)];
    assert.deepEqual(bits(back[0].weights), bits(weights));
    assert.deepEqual(bits(back[0].biases), bits(network[0].biases));
    assert.equal(bits(weights)[4], bits(weights)[3] + 1);      // the two neighbours stayed apart

    // JSON cannot hold these, so they are refused rather than changed in silence.
    for (const value of [-0, NaN, Infinity]) {
        const broken = [{ ...network[0], weights: Float32Array.from(weights, (w, k) => (k === 9 ? value : w)) }];
        assert.throws(() => modelText({ ...input, network: broken }), /model: the network for "prey" does not read back bit for bit/);
    }
});

test('U26 (the committed files): the Herd & Wolves networks fit the scenario and were trained on its label', async () => {
    const definition = herdDefinition(), teacher = await hashText(herdWolves.texts().label);
    for (const species of ['prey', 'wolf']) {
        const path = new URL(`../tools/agents/models/herd-wolves/${species}.mlp.json`, import.meta.url);
        const text = readFileSync(path, 'utf8'), model = JSON.parse(text);
        assert.equal(model.species, species);
        const network = networkFromModel(model, definition, `${species}.mlp.json`);
        assert.deepEqual(network.map(l => [l.input, l.output, l.activation]),
            [[24, 64, 'tanh'], [64, 64, 'tanh'], [64, 64, 'tanh'], [64, 2, 'tanh']]);
        const { format, version, layers, ...rest } = model;
        assert.equal(modelText({ ...rest, network }), text, `${species}.mlp.json is what the writer writes`);
        // The file names the rule it copies. If this fails, the scenario's label changed: train again.
        assert.equal(model.trainedOn.teacher, teacher, `${species}.mlp.json was trained on another label text`);
        assert.deepEqual(model.trainedOn.settings, herdWolves.settings());
        // The report beside it is this file's: a changed weight no longer matches its hash.
        const report = JSON.parse(readFileSync(new URL(`../tools/agents/models/herd-wolves/${species}.report.json`, import.meta.url), 'utf8'));
        assert.equal(report.file.sha256, await hashText(text), `${species}.report.json is not ${species}.mlp.json's report`);
        assert.equal(report.dataset, model.trainedOn.dataset);
        assert.deepEqual(report.heldOutMae, model.measured.heldOutMae);
        for (const action of definition.actNames) {
            assert.ok(model.measured.heldOutMae[action] <= 0.05, `${species} ${action}: held-out error ${model.measured.heldOutMae[action]}`);
        }
        const action = A.mlpForward(network, new Float32Array(definition.obsDim));
        assert.ok(action.every(v => Math.abs(v) <= 1));
    }
});

// A protobuf message as its fields, in order: { field, value }, with a BigInt for a varint and the
// bytes for a length-delimited field. Enough to read back what onnx.mjs writes.
function protoFields(bytes) {
    const fields = [];
    let at = 0;
    const varint = () => {
        let value = 0n, shift = 0n;
        for (;;) {
            const b = bytes[at++];
            value |= BigInt(b & 0x7f) << shift;
            if (!(b & 0x80)) return value;
            shift += 7n;
        }
    };
    while (at < bytes.length) {
        const tag = Number(varint()), field = tag >> 3, wire = tag & 7;
        if (wire === 0) fields.push({ field, value: varint() });
        else if (wire === 2) {
            const length = Number(varint());
            fields.push({ field, value: bytes.subarray(at, at + length) });
            at += length;
        } else throw new Error(`wire type ${wire} at byte ${at}`);
    }
    return fields;
}

test('U12: the ONNX writer\'s bytes decode to the layers that were written, weights bit for bit', () => {
    const text = bytes => new TextDecoder().decode(bytes);
    const all = (fields, n) => fields.filter(f => f.field === n).map(f => f.value), one = (fields, n) => all(fields, n)[0];
    const layer = (input, output, activation, stream) => ({ input, output, activation,
        weights: Float32Array.from({ length: input * output }, (_, k) => A.rand01(k, stream, 6) * 4 - 2),
        biases: Float32Array.from({ length: output }, (_, k) => A.rand01(k, stream, 7) - 0.5) });
    const network = [layer(5, 7, 'tanh', 0), layer(7, 4, 'relu', 1), layer(4, 4, 'identity', 2), layer(4, 3, 'sigmoid', 3), layer(3, 2, 'identity', 4)];
    network[0].weights.set([1e-45, 3.4028234663852886e38, -0, -1 / 3]);     // raw bytes keep even a -0
    const bytes = onnxFromNetwork(network, { name: 'five', metadata: { species: 'prey', obsNames: '["a","b"]' } });
    assert.deepEqual([...bytes], [...onnxFromNetwork(network, { name: 'five', metadata: { species: 'prey', obsNames: '["a","b"]' } })]);

    const model = protoFields(bytes), graph = protoFields(one(model, 7));
    assert.equal(one(model, 1), 8n);
    assert.deepEqual(all(model, 8).map(o => one(protoFields(o), 2)), [BigInt(ONNX_OPSET)]);
    assert.equal(ONNX_OPSET, 13);
    assert.deepEqual(all(model, 14).map(e => protoFields(e).map(f => text(f.value))), [['species', 'prey'], ['obsNames', '["a","b"]']]);
    assert.equal(text(one(graph, 2)), 'five');

    // The nodes, in order: a Gemm per layer, then its activation unless it is the identity.
    const nodes = all(graph, 1).map(n => {
        const f = protoFields(n);
        return { op: text(one(f, 4)), inputs: all(f, 1).map(text), outputs: all(f, 2).map(text),
            attributes: all(f, 5).map(a => { const x = protoFields(a); return [text(one(x, 1)), one(x, 3), one(x, 20)]; }) };
    });
    assert.deepEqual(nodes.map(n => n.op), ['Gemm', 'Tanh', 'Gemm', 'Relu', 'Gemm', 'Gemm', 'Sigmoid', 'Gemm']);
    assert.deepEqual(nodes.map(n => [n.inputs.join(), n.outputs.join()]), [
        ['obs,w0,b0', 'z0'], ['z0', 'h0'], ['h0,w1,b1', 'z1'], ['z1', 'h1'], ['h1,w2,b2', 'h2'],
        ['h2,w3,b3', 'z3'], ['z3', 'h3'], ['h3,w4,b4', 'act']]);
    for (const node of nodes) assert.deepEqual(node.attributes, node.op === 'Gemm' ? [['transB', 1n, 2n]] : []);

    // The initializers: each layer's weights as [output, input] and its biases, as raw f32.
    const tensors = Object.fromEntries(all(graph, 5).map(t => {
        const f = protoFields(t), raw = one(f, 9);
        return [text(one(f, 8)), { dims: all(f, 1).map(Number), type: one(f, 2), words: [...new Uint32Array(raw.slice().buffer)] }];
    }));
    assert.deepEqual(Object.keys(tensors), ['w0', 'b0', 'w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4']);
    const bits = array => [...new Uint32Array(array.buffer, array.byteOffset, array.length)];
    network.forEach((l, k) => {
        assert.deepEqual(tensors[`w${k}`], { dims: [l.output, l.input], type: 1n, words: bits(l.weights) });
        assert.deepEqual(tensors[`b${k}`], { dims: [l.output], type: 1n, words: bits(l.biases) });
    });
    assert.equal(tensors.w0.words[2], 0x80000000);

    // "obs" is float32 [batch, 5] and "act" float32 [batch, 2], with a free first dimension.
    const info = value => {
        const f = protoFields(value), tensor = protoFields(one(protoFields(one(f, 2)), 1));
        return [text(one(f, 1)), one(tensor, 1), all(protoFields(one(tensor, 2)), 1).map(d => protoFields(d)[0]).map(d => (d.field === 2 ? text(d.value) : Number(d.value)))];
    };
    assert.deepEqual(all(graph, 11).map(info), [['obs', 1n, ['batch', 5]]]);
    assert.deepEqual(all(graph, 12).map(info), [['act', 1n, ['batch', 2]]]);

    assert.throws(() => onnxFromNetwork([]), /onnx: a network needs at least one layer\./);
    assert.throws(() => onnxFromNetwork([{ ...network[0], activation: 'gelu' }]),
        /onnx: layer 0 has activation "gelu"; the writer takes identity, relu, tanh, sigmoid\./);
    assert.throws(() => onnxFromNetwork([{ ...network[0], weights: new Float32Array(34) }]), /onnx: layer 0 is 5→7 but holds 34 weights and 7 biases\./);
    assert.throws(() => onnxFromNetwork([network[0], network[2]]), /onnx: layer 1 reads 4 values; layer 0 writes 7\./);
});
