// Step 7, part 2 (docs/agents-cloning-design.md §7–§9): train one network per species on a recording.
//
//   npm run build:lib
//   npm run agents:record
//   npm run agents:train -- [--data <dir>] [--out <dir>] [--hidden 64x64x64] [--epochs 200] [--seed 1]
//                           [--batch 256] [--rate 0.01] [--final-rate 0.0001] [--species prey,wolf]
//
// The defaults are what phase 7.7 measured to meet the 0.05 bar for both species. They took about 45
// minutes for prey and 20 for wolves, one core each, with other runs beside them on a 6-core laptop
// CPU. `--hidden 16x16 --epochs 2` is a quick check.
//
// Reads a dataset (dataset.mjs checks its sizes and hashes). For each species it trains on the live
// rows outside the held-out episode, scores every pass on the held-out episode and keeps the best,
// then writes <species>.mlp.json (design §4.2) and <species>.report.json to --out: by default
// <data>-students, beside the recording, which git ignores. Every error it reports is measured with
// the library's mlpForward on the f32 weights, per action and per reason code. CPU only.
//
// --hidden lists the hidden layers' widths ("64x64x64"; "0" for none); every layer uses tanh, as the
// actions lie in [-1, 1].
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { assertDistCurrent, options, root } from './cli.mjs';

const { option, whole, positive } = options(process.argv.slice(2), 'train');
assertDistCurrent();
const { readDataset, trainingSet } = await import('./dataset.mjs');
const { TRAIN_DEFAULTS, meanAbsoluteError, quantile, trainNetwork } = await import('./mlp-train.mjs');

const data = option('data', join(root, 'tools/agents/out/herd-wolves'));
const out = option('out', `${data}-students`);
const hiddenText = option('hidden', '64x64x64');
const hidden = hiddenText === '0' ? [] : hiddenText.split('x').map(Number);
if (!hidden.every(w => Number.isInteger(w) && w > 0)) throw new Error(`train: --hidden must be widths joined by "x", such as 32x32, or 0; got "${hiddenText}".`);
const settings = {
    epochs: whole('epochs', TRAIN_DEFAULTS.epochs), batch: whole('batch', TRAIN_DEFAULTS.batch), seed: whole('seed', TRAIN_DEFAULTS.seed),
    rate: positive('rate', TRAIN_DEFAULTS.rate), finalRate: positive('final-rate', TRAIN_DEFAULTS.finalRate),
};

const dataset = await readDataset(data), { header } = dataset, { obsDim, actDim, obsNames, actNames } = header.definition;
const datasetHash = createHash('sha256').update(readFileSync(join(data, 'header.json'))).digest('hex');
const names = option('species', Object.keys(dataset.species).join(',')).split(',');
const reasonName = code => header.reasons?.[code] ?? `reason ${code}`;
const byAction = values => Object.fromEntries(actNames.map((name, j) => [name, values[j]]));
const fixed = values => values.map(v => v.toFixed(4)).join(' / ');

/** f32 weights as JSON numbers that read back to the same f32: 9 significant digits, and no −0. */
const nine = value => (value === 0 ? 0 : Number(value.toPrecision(9)));
const words = network => network.flatMap(l => [...new Uint32Array(l.weights.buffer), ...new Uint32Array(l.biases.buffer)]);

mkdirSync(out, { recursive: true });
console.log(`Training on ${data} (${header.episodes.length} episodes; held out: ${header.episodes.filter(e => e.heldOut).map(e => e.seed).join(', ')}).`);
for (const name of names) {
    const set = trainingSet(dataset, name);
    const widths = [obsDim, ...hidden, actDim];
    const shape = widths.slice(1).map((output, l) => ({ input: widths[l], output, activation: 'tanh' }));
    console.log(`${name}: ${widths.join('→')} tanh, ${set.train.rows.length.toLocaleString('en-US')} rows to train on, ` +
        `${set.heldOut.rows.length.toLocaleString('en-US')} held out`);
    const started = performance.now();
    const run = trainNetwork(shape, set.train, { ...settings, heldOut: set.heldOut, onEpoch: e => {
        if (e.epoch % 5 === 4 || e.epoch === settings.epochs - 1) {
            console.log(`   pass ${String(e.epoch + 1).padStart(3)}: loss ${e.loss.toExponential(3)}, held out ${fixed(e.heldOut)}`);
        }
    } });
    const seconds = (performance.now() - started) / 1000;
    for (const layer of run.network) for (const array of [layer.weights, layer.biases]) array.forEach((v, k) => { if (v === 0) array[k] = 0; });
    const best = meanAbsoluteError(run.network, set.heldOut, set.reason, { keep: true }), last = meanAbsoluteError(run.last, set.heldOut);
    const trained = meanAbsoluteError(run.network, set.train);
    const median = best.errors.map(e => quantile(e, 0.5)), p90 = best.errors.map(e => quantile(e, 0.9));

    const file = {
        format: 'null-graph/mlp', version: 1, species: name, obsNames, actNames,
        layers: run.network.map(l => ({ input: l.input, output: l.output, activation: l.activation,
            weights: Array.from(l.weights, nine), biases: Array.from(l.biases, nine) })),
        trainedOn: { dataset: datasetHash, teacher: header.scenario.wgsl?.label ?? null, settings: header.scenario.settings,
            seed: settings.seed, rounds: 1 },
        // Nothing that varies from run to run (such as the time taken): the same dataset and seed write the same bytes.
        training: { ...settings, hidden, bestPass: run.best + 1, rows: set.train.rows.length },
        measured: { heldOutMae: byAction(best.mae), rows: best.rows },
    };
    const text = JSON.stringify(file, null, 1) + '\n', path = join(out, `${name}.mlp.json`);
    const back = JSON.parse(text).layers.map(l => ({ ...l, weights: Float32Array.from(l.weights), biases: Float32Array.from(l.biases) }));
    if (words(back).some((w, k) => w !== words(run.network)[k])) throw new Error(`train: ${path} does not read back bit for bit.`);
    writeFileSync(path, text);

    // One report per species, so species trained in separate runs into one folder keep both.
    writeFileSync(join(out, `${name}.report.json`), JSON.stringify({
        species: name, dataset: datasetHash, data, settings, shape: widths,
        rows: { train: set.train.rows.length, heldOut: set.heldOut.rows.length }, seconds, bestPass: run.best + 1,
        heldOutMae: byAction(best.mae), heldOutMedian: byAction(median), heldOutP90: byAction(p90),
        lastPassMae: byAction(last.mae), trainMae: byAction(trained.mae),
        byReason: Object.fromEntries(Object.entries(best.groups).map(([code, g]) => [reasonName(Number(code)), { rows: g.rows, mae: byAction(g.mae) }])),
        passes: run.history.map(e => ({ rate: e.rate, loss: e.loss, heldOut: byAction(e.heldOut) })),
        file: { path, bytes: Buffer.byteLength(text), sha256: createHash('sha256').update(text).digest('hex') },
    }, null, 1) + '\n');
    console.log(`   ${seconds.toFixed(1)} s; best pass ${run.best + 1} of ${settings.epochs}`);
    console.log(`   held-out error (${actNames.join(' / ')}): ${fixed(best.mae)}; last pass ${fixed(last.mae)}; on training rows ${fixed(trained.mae)}`);
    console.log(`   held-out spread: median ${fixed(median)}, 90th percentile ${fixed(p90)}`);
    for (const [code, g] of Object.entries(best.groups)) {
        console.log(`      ${reasonName(Number(code)).padEnd(7)} ${String(g.rows).padStart(7)} rows: ${fixed(g.mae)}`);
    }
}
console.log(`Wrote ${names.map(n => `${n}.mlp.json and ${n}.report.json`).join(', ')} to ${out}`);
