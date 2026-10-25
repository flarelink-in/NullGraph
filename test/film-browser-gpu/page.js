// F2's browser G tests (docs/offline-films-browser-design.md §10): one case per load, named by the query,
//   index.html?case=G1-pattern
// It logs "[g] done" or "[g] failed: …" and leaves the result in globalThis.__FILM_G__ for run.mjs: the
// case's own checks, what it measured, and the files it uploaded for run.mjs to read back.

import { CASES } from './cases.js';

const logEl = document.querySelector('#log');
const log = (line, level = 'log') => {
    console[level](`[g] ${line}`);
    logEl.textContent += `${line}\n`;
};

async function upload(name, body) {
    const res = await fetch(`/__g/upload/${encodeURIComponent(name)}`, { method: 'PUT', body });
    if (!res.ok) throw new Error(`upload of ${name} failed: HTTP ${res.status}`);
}

/** Each adapter a case was given, so run.mjs can tell the case ran on the GPU it asked for. */
const adapters = [];

/** The app's device: no powerPreference (Chrome on Windows ignores it); run.mjs picks the GPU by flag. */
async function requestDevice() {
    if (!navigator.gpu) throw new Error('WebGPU is not available');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('no WebGPU adapter');
    const device = await adapter.requestDevice();
    const { vendor, architecture, description } = adapter.info ?? {};
    adapters.push({ vendor, architecture, description });
    return { adapter, device, info: { vendor, architecture, description } };
}

const name = new URLSearchParams(location.search).get('case');
const checks = [];
/** A case's assertion: recorded whatever the outcome, so one failure does not hide the others. */
const check = (what, ok, detail = null) => {
    checks.push({ name: what, ok: !!ok, detail });
    log(`${ok ? 'ok  ' : 'FAIL'} ${what}${detail != null && !ok ? ` — ${typeof detail === 'string' ? detail : JSON.stringify(detail)}` : ''}`, ok ? 'log' : 'error');
};
try {
    const run = CASES[name];
    if (!run) throw new Error(`unknown case ${name}; cases: ${Object.keys(CASES).join(', ')}`);
    log(`${name}: ${navigator.userAgent}`);
    const result = await run({ requestDevice, upload, log, check, gc: () => globalThis.gc?.() });
    globalThis.__FILM_G__ = { ok: true, case: name, checks, ...result, adapters, browser: navigator.userAgent };
    log('done');
} catch (e) {
    globalThis.__FILM_G__ = { ok: false, case: name, checks, error: String(e?.stack ?? e), adapters, browser: navigator.userAgent };
    log(`failed: ${e?.message ?? e}`, 'error');
}
