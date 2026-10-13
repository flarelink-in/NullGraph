// Step 7, part 2 (docs/agents-cloning-design.md §9, phase 7.10): write trained networks as .onnx.
//
//   npm run agents:export -- [--students <dir>] [--out <dir>]
//
// Reads every <species>.mlp.json in --students (by default the committed Herd & Wolves networks),
// checks its names against the scenario's definition, and writes <species>.onnx to --out: by
// default tools/agents/out/onnx, which git ignores. Each file stores its species and its
// observation and action names as metadata. CPU only; the same network gives the same bytes.
//
// The library reads these files back with mlpFromOnnx (null-graph/inference; check U12), and ONNX
// Runtime runs them: check A27 against mlpForward, check A28 against the network read back.
import { createHash } from 'node:crypto';
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { assertDistCurrent, options, root } from './cli.mjs';

const { option } = options(process.argv.slice(2), 'export');
assertDistCurrent();
const { herdDefinition } = await import('./scenario/herd-wolves.mjs');
const { networkFromModel } = await import('./model.mjs');
const { onnxFromNetwork } = await import('./onnx.mjs');

const students = option('students', join(root, 'tools/agents/models/herd-wolves'));
const out = option('out', join(root, 'tools/agents/out/onnx'));
const files = readdirSync(students).filter(f => f.endsWith('.mlp.json')).sort();
if (!files.length) throw new Error(`export: no <species>.mlp.json in ${students}.`);

const definition = herdDefinition();
mkdirSync(out, { recursive: true });
for (const file of files) {
    const path = join(students, file), model = JSON.parse(readFileSync(path, 'utf8'));
    const network = networkFromModel(model, definition, path);
    const bytes = onnxFromNetwork(network, { name: `${model.species}-mlp`, metadata: {
        species: model.species, obsNames: JSON.stringify(model.obsNames), actNames: JSON.stringify(model.actNames) } });
    const target = join(out, `${model.species}.onnx`);
    writeFileSync(target, bytes);
    console.log(`${target}: ${[network[0].input, ...network.map(l => l.output)].join('→')}, ${bytes.length.toLocaleString('en-US')} B, ` +
        `sha256 ${createHash('sha256').update(bytes).digest('hex').slice(0, 16)}…`);
}
