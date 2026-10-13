// Step 7, part 2 (docs/agents-cloning-design.md §8): judge the copy in the running simulation.
//
//   npm run build:lib
//   npm run agents:evaluate -- [--students <dir>] [--runs T,C,S,Sp,Sw] [--episodes 5] [--seed 21]
//                              [--steps 1800] [--stride 6] [--out <file>]
//
// Runs Herd & Wolves headless under Node with Dawn, in lockstep, once per run and seed:
//   T   the teacher drives both species: the reference, with its spread across seeds
//   C   the rule without wander and coin-flips drives both: what the label, followed exactly, does
//   S   the students drive both: the copy
//   Sp  a student drives the prey and the teacher the wolves; Sw is the other way round
// --students is a folder of <species>.mlp.json files, as train.mjs writes them. The seeds are none
// of a recording's defaults. NG_DAWN_FLAGS=adapter=NVIDIA picks the GPU, as for the GPU suites.
//
// It prints, and writes to --out (by default <students>/evaluation.json): catches in 30 s, each
// driver's mean absolute difference from the label on its own run, mean speed, and how tight the
// herds are. The design's bar is S's mean catches within 10% of C's.
//
// Exit 0 when every run finished, the device reported no error, and run C differed from its own
// label by nothing; 1 otherwise. A missed bar is a measurement, not a failure of the tool.
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { installNodeGPU, describeDevice } from '../../test/gpu/node-gpu.mjs';
import { assertDistCurrent, options, root } from './cli.mjs';

const { option, whole } = options(process.argv.slice(2), 'evaluate');
assertDistCurrent();

const host = installNodeGPU();
const { NullGraph } = await import('../../dist/index.js');
const { MlpBrain } = await import('../../dist/agents.js');
const { herdWolves, herdDefinition } = await import('./scenario/herd-wolves.mjs');
const { EVALUATE_DEFAULTS, RUNS, evaluateRun } = await import('./evaluator.mjs');
const { networkFromModel } = await import('./model.mjs');

const BAR = 0.10;       // S's mean catches against C's (design §8)
const students = option('students', join(root, 'tools/agents/out/herd-wolves-students'));
const out = option('out', join(students, 'evaluation.json'));
const runNames = option('runs', Object.keys(RUNS).join(',')).split(',');
for (const name of runNames) if (!RUNS[name]) throw new Error(`evaluate: --runs takes ${Object.keys(RUNS).join(', ')}; got "${name}".`);
const episodes = whole('episodes', EVALUATE_DEFAULTS.seeds.length), first = whole('seed', EVALUATE_DEFAULTS.seeds[0]);
const seeds = Array.from({ length: episodes }, (_, k) => first + k);
const steps = whole('steps', EVALUATE_DEFAULTS.steps), stride = whole('stride', EVALUATE_DEFAULTS.stride);

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
    console.error('evaluate: no WebGPU adapter.');
    process.exit(1);
}
const device = await adapter.requestDevice();
const engine = new NullGraph();
const init = await engine.init(null, { device, width: 64, height: 64 });
if (!init.success) throw new Error(init.error);

// One MlpBrain per species that some run puts a student on.
const definition = herdDefinition(), brains = {}, models = {};
for (const species of new Set(runNames.flatMap(run => Object.keys(RUNS[run]).filter(s => RUNS[run][s] === 'student')))) {
    const path = join(students, `${species}.mlp.json`), text = readFileSync(path, 'utf8'), model = JSON.parse(text);
    if (model.species !== species) throw new Error(`model: ${path} was trained for species "${model.species}", not "${species}".`);
    brains[species] = new MlpBrain(engine, networkFromModel(model, definition, path));
    models[species] = { path: relative(root, path).replaceAll('\\', '/'), sha256: createHash('sha256').update(text).digest('hex'), trainedOn: model.trainedOn,
        heldOutMae: model.measured?.heldOutMae };
}

const percent = share => `${share >= 0 ? '+' : '−'}${(100 * Math.abs(share)).toFixed(1)}%`;
const started = performance.now(), results = {};
console.log(`Evaluating on ${describeDevice(device)}: seeds ${seeds.join(', ')}, ${steps} steps each.`);
for (const name of runNames) {
    const at = performance.now();
    const run = results[name] = await evaluateRun(engine, herdWolves, RUNS[name], { seeds, steps, stride, students: brains });
    const c = run.catches;
    console.log(`${name.padEnd(2)} (${Object.entries(run.drivers).map(([s, d]) => `${s}: ${d}`).join(', ')}) — ${((performance.now() - at) / 1000).toFixed(1)} s`);
    console.log(`   catches: ${c.values.join(', ')}; mean ${c.mean.toFixed(1)} (${percent(c.low)} to ${percent(c.high)})`);
    for (const [species, s] of Object.entries(run.species)) {
        const mae = Object.entries(s.mae).map(([action, e]) => `${action} ${e === null ? '—' : e.toFixed(4)}`).join(', ');
        const reasons = Object.entries(s.byReason).map(([why, g]) => `${why} ${Object.values(g.mae)[0].toFixed(3)} (${(100 * g.rows / (s.decisions - s.frozen)).toFixed(0)}%)`).join(', ');
        const measured = herdWolves.measures(definition).flatMap(m => (s.measures[m.name] === null ? []
            : [`${m.name} ${m.unit === 'share' ? `${(100 * s.measures[m.name]).toFixed(1)}%` : `${s.measures[m.name].toFixed(2)} ${m.unit}`}`]));
        console.log(`   ${species}: driver against label: ${mae}`);
        console.log(`      ${Object.keys(s.mae)[0]} by reason: ${reasons}`);
        console.log(`      ${measured.join('; ')}; ${(100 * s.frozen / s.decisions).toFixed(1)}% of decisions frozen`);
    }
}
await host.settle();

// How each run's mean catches compare with T's and C's.
const against = {};
for (const name of runNames) {
    against[name] = Object.fromEntries(['T', 'C'].flatMap(other => (results[other] && other !== name
        ? [[other, results[name].catches.mean / results[other].catches.mean - 1]] : [])));
}
console.log('Mean catches: ' + runNames.map(name => `${name} ${results[name].catches.mean.toFixed(1)}` +
    (Object.keys(against[name]).length ? ` (${Object.entries(against[name]).map(([other, share]) => `${percent(share)} on ${other}`).join(', ')})` : '')).join('; '));
let bar = null;
if (results.S && results.C) {
    bar = { share: against.S.C, limit: BAR, met: Math.abs(against.S.C) <= BAR };
    console.log(`The bar: S's mean catches within ${100 * BAR}% of C's: ${percent(bar.share)} — ${bar.met ? 'met' : 'MISSED'}.`);
}

// Run C drives with the label's own decisions, so a live agent's action must equal its label.
let selfCheck = true;
if (results.C) {
    const worst = Math.max(...Object.values(results.C.species).flatMap(s => Object.values(s.mae).map(e => e ?? 0)));
    selfCheck = worst === 0;
    console.log(selfCheck ? 'Self-check: run C differs from its own label by nothing.'
        : `Self-check FAILED: run C differs from its own label by up to ${worst}; the label pass and the driver disagree.`);
}

const git = (...a) => execFileSync('git', a, { cwd: root, encoding: 'utf8' }).trim();
writeFileSync(out, JSON.stringify({
    scenario: herdWolves.name, seeds, steps, stride, students: models, runs: results, against, bar, selfCheck,
    madeWith: { commit: git('rev-parse', 'HEAD'), dirty: git('status', '--porcelain') !== '', node: process.version,
        adapter: describeDevice(device) },
}, null, 1) + '\n');
console.log(`Wrote ${out} in ${((performance.now() - started) / 1000).toFixed(1)} s.`);
for (const event of host.events) console.error(`WebGPU ${event.kind}: ${event.message.split('\n')[0]}`);
for (const brain of Object.values(brains)) brain.destroy();
device.destroy();
process.exit(host.events.length || !selfCheck ? 1 : 0);
