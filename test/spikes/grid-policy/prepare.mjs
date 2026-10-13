// Bundles the two earlier versions run.js compares with today's source, into dist/ (ignored by git).
// Run after every build, before the spike or its test:
//
//   node test/spikes/grid-policy/prepare.mjs
//
// A bundle takes every src/extras file from one revision, so it is that revision's code and nothing
// newer. Core is today's: step 5 did not change it.
import { build } from 'esbuild';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../../../', import.meta.url));
const at = (revision, entry, outfile) => build({
    entryPoints: [root + entry], outfile: root + outfile, bundle: true, format: 'esm',
    platform: 'browser', external: ['gl-matrix'], logLevel: 'error',
    plugins: [{ name: 'source-at-revision', setup(b) {
        b.onLoad({ filter: /src[\\/]extras[\\/].*\.ts$/ }, args => ({
            contents: execFileSync('git', ['show', `${revision}:${args.path.slice(root.length).replace(/\\/g, '/')}`],
                { cwd: root, encoding: 'utf8' }), loader: 'ts',
        }));
    } }],
});
// Before step 5: the atomic scatter, and a query that reads a crowded cell's first entries.
await at('b7050b7', 'src/extras/agents/index.ts', 'dist/grid-policy-before.js');
// Step 5 as phase 5.6 built it: a query and a CPU mirror with a window per querying agent.
await at('da262ad', 'src/extras/agents/index.ts', 'dist/grid-policy-window.js');
console.log('Prepared dist/grid-policy-before.js (b7050b7) and dist/grid-policy-window.js (da262ad).');
