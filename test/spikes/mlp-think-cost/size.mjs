// Step 6, phase 6.3 (docs/agents-mlp-design.md §7): what MlpBrain adds to an app's bundle.
//
//   node test/spikes/mlp-think-cost/size.mjs
//
// Same settings as phase 6.0's baseline: esbuild, browser ESM, default target, minified, from
// src/, gzipSync defaults. Each line bundles one import set and lets esbuild tree-shake the rest,
// as an app's bundler would. A rule-only app must not carry neural code: its bundle is searched
// for the MLP's WGSL and labels.
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';

const root = fileURLToPath(new URL('../../../', import.meta.url));
const esbuild = createRequire(root + 'package.json')('esbuild');
const entry = (root + 'src/extras/agents/index.ts').replace(/\\/g, '/');
// Step 8 (docs/agents-onnx-reader-design.md §12): the ONNX reader lives in another entry,
// null-graph/inference. Its own line (2,600 B) is asserted by test/inference-onnx.test.mjs; the
// two lines here report what an app pays for it.
const inference = (root + 'src/extras/inference/index.ts').replace(/\\/g, '/');
// Step 9, phase 9.5 (docs/agents-ort-design.md §12.1): core gains createComputeStageAsync; this line
// reports what core's entry, null-graph's main import, weighs.
const core = (root + 'src/core/index.ts').replace(/\\/g, '/');
const RULES = 'AgentSystem, defineAgents, Sensors, Actions, RuleBrain';
const sets = [
    ['core entry', `export * from '${core}';`],
    ['whole agents entry', `export * from '${entry}';`],
    ['rule path', `export { ${RULES} } from '${entry}';`],
    ['rule path + MlpBrain', `export { ${RULES}, MlpBrain } from '${entry}';`],
    ['MlpBrain + mlpForward', `export { ${RULES}, MlpBrain, mlpForward } from '${entry}';`],
    ['whole inference entry', `export * from '${inference}';`],
    ['MlpBrain + mlpFromOnnx', `export { ${RULES}, MlpBrain } from '${entry}'; export { mlpFromOnnx } from '${inference}';`],
];
const MLP_MARKERS = ['mlpWeights', 'MlpBrain weights', 'setWeights requires'];
const sizes = {};
for (const [name, contents] of sets) {
    const out = await esbuild.build({ stdin: { contents, resolveDir: root, loader: 'ts' }, bundle: true, minify: true,
        format: 'esm', platform: 'browser', external: ['gl-matrix'], write: false, logLevel: 'error' });
    const code = out.outputFiles[0].contents, text = out.outputFiles[0].text;
    sizes[name] = { min: code.length, gzip: gzipSync(code).length, mlp: MLP_MARKERS.filter(m => text.includes(m)) };
    sizes[name].reader = text.includes('mlpFromOnnx');
    console.log(`${name.padEnd(24)} ${String(code.length).padStart(7)} B min  ${String(sizes[name].gzip).padStart(6)} B min+gzip` +
        `  MLP code: ${sizes[name].mlp.length ? sizes[name].mlp.join(', ') : 'none'}${sizes[name].reader ? '  + the ONNX reader' : ''}`);
}
const rule = sizes['rule path'], mlp = sizes['rule path + MlpBrain'];
console.log(`MlpBrain adds ${mlp.gzip - rule.gzip} B min+gzip to a rule-path app.`);
console.log(`mlpFromOnnx adds ${sizes['MlpBrain + mlpFromOnnx'].gzip - mlp.gzip} B min+gzip to an app that runs an MlpBrain.`);
if (rule.mlp.length || !mlp.mlp.length) {
    console.error('FAIL: rule-only imports must tree-shake every MLP marker, and the MLP path must contain them.');
    process.exit(1);
}
if (sizes['whole agents entry'].reader || mlp.reader || !sizes['whole inference entry'].reader) {
    console.error('FAIL: the ONNX reader belongs to the inference entry alone; no agents bundle may hold it.');
    process.exit(1);
}
