// npm run test:gpu — every GPU harness, headless, on this machine's real GPU through Dawn.
//
//   npm run test:gpu                       all suites
//   npm run test:gpu -- --only shadows     one suite (repeatable)
//   npm run test:gpu -- --json-dir out     also write one JSON report per suite
//   NG_GPU_ARTIFACTS=out npm run test:gpu  also save the frames harnesses render (PNG)
//
// Each suite runs in its own process (test/gpu/run-harness.mjs) so a lost device or a leaked
// global cannot decide the next suite's result. Exits non-zero if any suite failed a check,
// reported an uncaptured WebGPU error, or could not run (design-principles §13.2).
//
// The harnesses import ../../dist — build first (`npm run build:lib`). A dist older than src is
// reported, because a GPU suite passing against stale code is a green light with no bulb in it.

import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { SUITES, RENDER_SCENES } from './suites.mjs';

const argv = process.argv.slice(2);
const only = argv.flatMap((a, i) => (a === '--only' ? [argv[i + 1]] : []));
const jsonDirAt = argv.indexOf('--json-dir');
const jsonDir = jsonDirAt >= 0 ? argv[jsonDirAt + 1] : null;

const known = [...SUITES, ...RENDER_SCENES].map(s => s.name);
const unknown = only.filter(n => !known.includes(n));
if (unknown.length) {
    console.error(`Unknown suite(s): ${unknown.join(', ')}. Known: ${known.join(', ')}.`);
    process.exit(2);
}

if (!existsSync('dist/index.js')) {
    console.error('dist/ is missing — run `npm run build:lib` first (the harnesses import the built library).');
    process.exit(2);
}
const newest = (dir) => readdirSync(dir, { recursive: true })
    .map(f => join(dir, String(f)))
    .filter(f => statSync(f).isFile())
    .reduce((t, f) => Math.max(t, statSync(f).mtimeMs), 0);
if (newest('src') > statSync('dist/index.js').mtimeMs) {
    console.warn('WARNING: src/ has changes newer than dist/ — these suites test the BUILT library. ' +
        'Run `npm run build:lib` first or the results describe old code.\n');
}

const scratch = mkdtempSync(join(tmpdir(), 'ng-gpu-'));
const rows = [];
for (const suite of SUITES.filter(s => only.length === 0 || only.includes(s.name))) {
    console.log(`\n=== ${suite.name} (${suite.file}) ===`);
    const report = join(jsonDir ?? scratch, `${suite.name}.json`);
    const res = spawnSync(process.execPath, ['test/gpu/run-harness.mjs', suite.file, suite.key, '--json', report], {
        stdio: 'inherit',
    });
    let r = null;
    try { r = JSON.parse(readFileSync(report, 'utf8')); } catch { /* the child died before writing it */ }
    rows.push({
        suite: suite.name,
        ok: res.status === 0,
        checks: r ? `${r.total - r.failed}/${r.total}` : '—',
        gpuErrors: r ? r.gpuErrors.length : '—',
        ms: r ? r.ms : '—',
        note: !r ? 'did not report' : r.fatal ? 'fatal: ' + r.fatal : r.threw && !r.total ? 'crashed' : '',
    });
}
for (const scene of RENDER_SCENES.filter(s => only.length === 0 || only.includes(s.name))) {
    console.log(`\n=== ${scene.name} (${scene.file}) ===`);
    const base = join(jsonDir ?? scratch, scene.name.replace(/[^a-z0-9-]+/gi, '-'));
    const res = spawnSync(process.execPath,
        ['test/gpu/render.mjs', scene.file, '--out', `${base}.png`, '--json', `${base}.json`, '--frames', '2', '--width', '256', '--height', '256'],
        { stdio: 'inherit' });
    let r = null;
    try { r = JSON.parse(readFileSync(`${base}.json`, 'utf8')); } catch { /* render failed before reporting */ }
    rows.push({
        suite: scene.name,
        ok: res.status === 0,
        checks: res.status === 0 ? '1/1' : '0/1',
        gpuErrors: r ? r.gpuErrors.length : '—',
        ms: r ? r.ms : '—',
        note: !r ? 'did not render' : r.blank ? 'uniform frame' : '',
    });
}
rmSync(scratch, { recursive: true, force: true });

console.log('\n=== GPU summary ===');
console.table(rows);
const failed = rows.filter(r => !r.ok);
console.log(failed.length === 0
    ? `All ${rows.length} GPU suite(s) passed.`
    : `${failed.length} GPU suite(s) FAILED: ${failed.map(r => r.suite).join(', ')}.`);
process.exit(failed.length === 0 ? 0 : 1);
