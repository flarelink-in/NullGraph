// Step 10 (docs/agents-live-switch-design.md, phases 10.1 and 10.2): the test-engine site's copies
// against this repository. Run it after changing the scenario, training again, or editing the
// site's demo. CPU only; it reads the site's files and changes nothing.
//
//   node test/spikes/live-switch/check-site.mjs [--site <the site's folder>]
//
// 1. The demo's rule (HerdWolvesGPU/brain.ts) is tools/agents/scenario/herd-wolves.mjs's texts,
//    byte for byte (D3), and its label text hashes to each network's trainedOn.teacher.
// 2. The demo's networks/<species>.mlp.json are the committed networks (tools/agents/models).
// 3. Its networks/<species>.onnx hold the same networks, bit for bit, as mlpFromOnnx reads them
//    (D2), and are what `npm run agents:export` writes from them.
// brain.ts is TypeScript: the site's own esbuild turns it into JavaScript first.
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const at = process.argv.indexOf('--site');
const site = resolve(at > 0 ? process.argv[at + 1] : process.env.NG_SITE ?? 'D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts');
const demo = join(site, 'src/data/engine-demos/demos/AgentsExperiments/HerdWolvesGPU');
const models = join(repo, 'tools/agents/models/herd-wolves');

const scenario = await import(pathToFileURL(join(repo, 'tools/agents/scenario/herd-wolves.mjs')).href);
const { networkFromModel } = await import(pathToFileURL(join(repo, 'tools/agents/model.mjs')).href);
const { onnxFromNetwork } = await import(pathToFileURL(join(repo, 'tools/agents/onnx.mjs')).href);
const { mlpFromOnnx } = await import(pathToFileURL(join(repo, 'dist/inference.js')).href);

const esbuild = createRequire(join(site, 'package.json'))('esbuild');
const scratch = mkdtempSync(join(tmpdir(), 'ng-check-site-'));
const transpiled = join(scratch, 'brain.mjs');
writeFileSync(transpiled, esbuild.transformSync(readFileSync(join(demo, 'brain.ts'), 'utf8'), { loader: 'ts', format: 'esm' }).code);
const brain = await import(pathToFileURL(transpiled).href);
rmSync(scratch, { recursive: true, force: true });

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
let failures = 0;
function check(name, ok, detail = '') {
    if (!ok) failures++;
    console.log(`  ${ok ? 'ok  ' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`);
}
function sameText(name, site, ours) {
    let k = 0;
    while (k < Math.min(site.length, ours.length) && site[k] === ours[k]) k++;
    check(name, site === ours, site === ours ? '' : `first difference at ${k}: site ${JSON.stringify(site.slice(k, k + 40))}, scenario ${JSON.stringify(ours.slice(k, k + 40))}`);
}
const bits = a => new Uint32Array(a.buffer, a.byteOffset, a.length);
const sameFloats = (a, b) => a.length === b.length && bits(a).every((w, i) => w === bits(b)[i]);

console.log(`The site: ${site}`);
// 1. The rule.
sameText('brain.ts: herdRule(DRIVER), the rule the demo runs', brain.HERD_BRAIN, scenario.herdRule(scenario.DRIVER));
sameText('brain.ts: herdRule(LABEL), the text the networks copy', brain.HERD_LABEL, scenario.herdRule(scenario.LABEL));
sameText('brain.ts: the catches', brain.HERD_CATCHES, scenario.HERD_CATCHES);
for (const every of [1, 4, 16, 64, 256]) {
    sameText(`brain.ts: the respawn, one flash per ${every}`, brain.herdRespawnWGSL(every), scenario.herdRespawnWGSL(every));
}
const label = sha256(brain.HERD_LABEL);
const definition = scenario.herdDefinition();
for (const species of ['prey', 'wolf']) {
    const text = readFileSync(join(demo, `networks/${species}.mlp.json`), 'utf8');
    const model = JSON.parse(text);
    check(`${species}.mlp.json was trained to copy the demo's rule`, model.trainedOn?.teacher === label,
        `teacher ${model.trainedOn?.teacher?.slice(0, 16)}…, the demo's label ${label.slice(0, 16)}…`);
    // 2. The networks.
    const committed = readFileSync(join(models, `${species}.mlp.json`), 'utf8');
    check(`${species}.mlp.json is the committed network`, sha256(text) === sha256(committed), `sha256 ${sha256(text).slice(0, 16)}…`);
    // 3. The .onnx files.
    const onnx = readFileSync(join(demo, `networks/${species}.onnx`));
    const network = networkFromModel(model, definition, `${species}.mlp.json`);
    let read = null, why = '';
    try { read = mlpFromOnnx(onnx).network; } catch (error) { why = `mlpFromOnnx refused it: ${error.message}`; }
    const same = !!read && read.length === network.length && read.every((l, i) => l.input === network[i].input && l.output === network[i].output &&
        l.activation === network[i].activation && sameFloats(l.weights, network[i].weights) && sameFloats(l.biases, network[i].biases));
    check(`${species}.onnx holds ${species}.mlp.json's network, bit for bit`, same,
        why || `${[read[0].input, ...read.map(l => l.output)].join('→')}, ${onnx.length.toLocaleString('en-US')} B`);
    const exported = onnxFromNetwork(network, { name: `${model.species}-mlp`, metadata: {
        species: model.species, obsNames: JSON.stringify(model.obsNames), actNames: JSON.stringify(model.actNames) } });
    check(`${species}.onnx is what npm run agents:export writes`, Buffer.compare(Buffer.from(exported), onnx) === 0);
}
console.log(failures ? `\n${failures} check(s) failed.` : '\nThe site matches this repository.');
process.exit(failures ? 1 : 0);
