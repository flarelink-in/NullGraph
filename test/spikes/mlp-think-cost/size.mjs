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
const RULES = 'AgentSystem, defineAgents, Sensors, Actions, RuleBrain';
const sets = [
    ['whole agents entry', `export * from '${entry}';`],
    ['rule path', `export { ${RULES} } from '${entry}';`],
    ['rule path + MlpBrain', `export { ${RULES}, MlpBrain } from '${entry}';`],
    ['MlpBrain + mlpForward', `export { ${RULES}, MlpBrain, mlpForward } from '${entry}';`],
];
const MLP_MARKERS = ['mlpWeights', 'MlpBrain weights', 'setWeights requires'];
const sizes = {};
for (const [name, contents] of sets) {
    const out = await esbuild.build({ stdin: { contents, resolveDir: root, loader: 'ts' }, bundle: true, minify: true,
        format: 'esm', platform: 'browser', external: ['gl-matrix'], write: false, logLevel: 'error' });
    const code = out.outputFiles[0].contents, text = out.outputFiles[0].text;
    sizes[name] = { min: code.length, gzip: gzipSync(code).length, mlp: MLP_MARKERS.filter(m => text.includes(m)) };
    console.log(`${name.padEnd(24)} ${String(code.length).padStart(7)} B min  ${String(sizes[name].gzip).padStart(6)} B min+gzip` +
        `  MLP code: ${sizes[name].mlp.length ? sizes[name].mlp.join(', ') : 'none'}`);
}
const rule = sizes['rule path'], mlp = sizes['rule path + MlpBrain'];
console.log(`MlpBrain adds ${mlp.gzip - rule.gzip} B min+gzip to a rule-path app.`);
if (rule.mlp.length || !mlp.mlp.length) {
    console.error('FAIL: rule-only imports must tree-shake every MLP marker, and the MLP path must contain them.');
    process.exit(1);
}
