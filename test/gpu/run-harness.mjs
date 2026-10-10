// Run ONE GPU harness under Node/Dawn and turn its result into an exit code.
//
//   node test/gpu/run-harness.mjs <harness.js> <resultsKey> [--json <report.json>]
//
// The harness is an ordinary ES module with top-level await (test/*-gpu/run.js): importing it IS
// running it, and its last act is publishing globalThis[resultsKey]. This file adds the two things
// a page cannot do for itself — an exit code (design-principles §13.2), and failing the run on any
// uncaptured WebGPU error even when every numeric check passed, because "the checks passed but
// the device reported a validation error" means some check was not looking at what it thought.
//
// One harness per process, on purpose: a lost device or a leaked global in one suite must not
// decide the next suite's result. test/gpu/run-gpu-tests.mjs runs them in sequence.

import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { installNodeGPU, describeDevice } from './node-gpu.mjs';

const args = process.argv.slice(2);
const jsonAt = args.indexOf('--json');
const jsonPath = jsonAt >= 0 ? args[jsonAt + 1] : null;
const [file, key] = jsonAt >= 0 ? args.filter((_, i) => i !== jsonAt && i !== jsonAt + 1) : args;

if (!file || !key) {
    console.error('usage: node test/gpu/run-harness.mjs <harness.js> <resultsKey> [--json <report.json>]');
    process.exit(2);
}

const host = installNodeGPU();
const t0 = performance.now();

let threw = null;
try {
    await import(pathToFileURL(resolve(file)).href);
} catch (e) {
    threw = e;
}
await host.settle();

const ms = Math.round(performance.now() - t0);
const r = globalThis[key];
const device = host.devices[0];
const events = host.events;

const report = {
    harness: file,
    device: device ? describeDevice(device) : null,
    ms,
    total: r?.total ?? 0,
    failed: r?.failed ?? 0,
    fatal: r?.fatal ?? null,
    threw: threw ? String(threw?.stack ?? threw) : null,
    gpuErrors: events,
    results: r?.results ?? [],
};

console.log(`\n[${file}] on ${report.device ?? 'no device'} — ${ms} ms`);

if (events.length > 0) {
    console.log(`\n${events.length} uncaptured WebGPU error(s) — the run fails even if every check passed:`);
    for (const e of events.slice(0, 8)) {
        console.log(`  [${e.kind}] ${e.message.split('\n').slice(0, 4).join('\n           ')}`);
    }
    if (events.length > 8) console.log(`  … and ${events.length - 8} more (see --json)`);
}

let ok;
if (threw && !r) {
    console.log(`\nHARNESS CRASHED before publishing ${key}:\n${report.threw}`);
    ok = false;
} else if (!r) {
    console.log(`\nThe harness finished without publishing ${key} — it never reached finish().`);
    ok = false;
} else if (r.fatal) {
    console.log(`\nFATAL (environment, not a test result): ${r.fatal}`);
    ok = false;
} else {
    ok = r.failed === 0 && events.length === 0 && !threw;
    if (threw) console.log(`\nThe harness threw after publishing results:\n${report.threw}`);
    console.log(`${r.total - r.failed}/${r.total} GPU checks passed` +
        (events.length ? `, ${events.length} WebGPU error(s)` : '') + '.');
}

if (jsonPath) writeFileSync(jsonPath, JSON.stringify(report, null, 2));
for (const d of host.devices) { try { d.destroy(); } catch { /* already gone */ } }
process.exit(ok ? 0 : 1);
