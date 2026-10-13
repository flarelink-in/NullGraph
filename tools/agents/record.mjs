// Step 7, part 2 (docs/agents-cloning-design.md §6): record what the teacher decides.
//
//   npm run build:lib
//   npm run agents:record -- [--out <dir>] [--episodes <n>] [--seed <first>] [--steps <n>] [--stride <n>] [--count <n>]
//
// Runs Herd & Wolves headless under Node with Dawn, in lockstep, and writes a dataset folder
// (dataset.mjs): by default tools/agents/out/herd-wolves, which git ignores. The last episode is
// held out. NG_DAWN_FLAGS=adapter=NVIDIA picks the GPU, as for the GPU suites.
//
// Exit 0 when the dataset is written and the device reported no error; 1 otherwise.
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { installNodeGPU, describeDevice } from '../../test/gpu/node-gpu.mjs';
import { assertDistCurrent, options, root } from './cli.mjs';

const { option, whole } = options(process.argv.slice(2), 'record');
assertDistCurrent();

const host = installNodeGPU();
const { NullGraph } = await import('../../dist/index.js');
const { herdWolves } = await import('./scenario/herd-wolves.mjs');
const { RECORD_DEFAULTS, recordEpisodes, describeRecording } = await import('./recorder.mjs');
const { writeDataset, hashText } = await import('./dataset.mjs');

const episodes = whole('episodes', RECORD_DEFAULTS.seeds.length), first = whole('seed', RECORD_DEFAULTS.seeds[0]);
const out = option('out', join(root, 'tools/agents/out', herdWolves.name));
const scene = { count: whole('count', herdWolves.defaults.count) };

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
    console.error('record: no WebGPU adapter.');
    process.exit(1);
}
const device = await adapter.requestDevice();
const engine = new NullGraph();
const init = await engine.init(null, { device, width: 64, height: 64 });
if (!init.success) throw new Error(init.error);

const started = performance.now();
const recording = await recordEpisodes(engine, herdWolves, {
    seeds: Array.from({ length: episodes }, (_, k) => first + k),
    steps: whole('steps', RECORD_DEFAULTS.steps), stride: whole('stride', RECORD_DEFAULTS.stride), scene,
});
await host.settle();

const git = (...a) => execFileSync('git', a, { cwd: root, encoding: 'utf8' }).trim();
const texts = herdWolves.texts(scene), hashes = {};
for (const [role, text] of Object.entries(texts)) hashes[role] = await hashText(text);
const header = await writeDataset(out, {
    ...recording.header,
    scenario: { ...recording.header.scenario, wgsl: hashes },
    madeWith: { commit: git('rev-parse', 'HEAD'), dirty: git('status', '--porcelain') !== '', node: process.version,
        adapter: describeDevice(device) },
}, recording.species);

console.log(`Recorded ${episodes} episode(s) on ${describeDevice(device)} in ${((performance.now() - started) / 1000).toFixed(1)} s → ${out}`);
for (const line of describeRecording(recording)) console.log(line);
for (const [file, entry] of Object.entries(header.files)) {
    console.log(`   ${file.padEnd(10)} ${String(entry.rows).padStart(8)} rows ${String(entry.bytes).padStart(10)} B  sha256 ${entry.sha256.slice(0, 16)}…`);
}
for (const event of host.events) console.error(`WebGPU ${event.kind}: ${event.message.split('\n')[0]}`);
device.destroy();
process.exit(host.events.length ? 1 : 0);
