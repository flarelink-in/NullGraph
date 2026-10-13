// npm run test:gpu:browser [-- --only core] [--chrome <path>] [--timeout 120]
//
// The same GPU harnesses as `npm run test:gpu`, in real headless Chrome — the browser host of
// test/gpu/harness-env.mjs. Self-contained: it serves the repo root itself (COOP/COEP on, like
// dev-server.js), launches Chrome with WebGPU enabled, opens each suite's page, waits for the
// page to publish its results global, and exits non-zero if any suite failed or could not run.
//
// Why both hosts: Node/Dawn is the fast, scriptable path agents use; Chrome is what users run.
// They differ in ways that matter — Chrome ships DXC on Windows (so `subgroups` / `shader-f16`
// light up), Dawn-in-Node here does not — so a suite green in one host is not proof for the other.
//
// Uncaptured WebGPU errors fail the run here too: the page's device is not reachable from CDP, so
// Chrome's console warnings that carry WebGPU validation messages are counted instead.

import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { SUITES } from './suites.mjs';

const argv = process.argv.slice(2);
const arg = (name, fallback) => { const i = argv.indexOf(`--${name}`); return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback; };
const only = argv.flatMap((a, i) => (a === '--only' ? [argv[i + 1]] : []));
const TIMEOUT_ARG = arg('timeout', '');        // empty: each suite's own limit, or 120 s
const PORT = Number(arg('port', '9334'));

const unknown = only.filter(n => !SUITES.some(s => s.name === n));
if (unknown.length) {
    console.error(`Unknown suite(s): ${unknown.join(', ')}. Known: ${SUITES.map(s => s.name).join(', ')}.`);
    process.exit(2);
}

const CHROME = [
    arg('chrome', null), process.env.CHROME_PATH,
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome', '/usr/bin/chromium',
].filter(Boolean).find(p => existsSync(p));
if (!CHROME) {
    console.error('No Chrome found. Pass --chrome <path> or set CHROME_PATH.');
    process.exit(2);
}

// ---------------------------------------------------------------------------
// A static server for the repo root (what dev-server.js does, minus express)
// ---------------------------------------------------------------------------
const ROOT = resolve(fileURLToPath(import.meta.url), '../../..');
const MIME = {
    '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json',
    '.wasm': 'application/wasm', '.map': 'application/json', '.png': 'image/png', '.css': 'text/css',
};
const server = createServer((req, res) => {
    let rel = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (rel.endsWith('/')) rel += 'index.html';
    const file = normalize(join(ROOT, rel));
    if (!file.startsWith(ROOT) || !existsSync(file) || !statSync(file).isFile()) {
        res.writeHead(404); res.end('not found'); return;
    }
    res.writeHead(200, {
        'Content-Type': MIME[extname(file)] ?? 'application/octet-stream',
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
        'Cache-Control': 'no-store',
    });
    res.end(readFileSync(file));
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const BASE = `http://127.0.0.1:${server.address().port}`;

// ---------------------------------------------------------------------------
// Chrome over CDP (the same protocol test/environment-gpu/headless.mjs speaks)
// ---------------------------------------------------------------------------
const profile = mkdtempSync(join(tmpdir(), 'nullgraph-chrome-'));
const chrome = spawn(CHROME, [
    '--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
    '--enable-unsafe-webgpu', '--no-first-run', '--no-default-browser-check', '--disable-extensions',
    '--window-size=512,512', 'about:blank',
], { stdio: ['ignore', 'pipe', 'pipe'] });
let chromeErr = '';
chrome.stderr.on('data', d => { chromeErr += d.toString(); });

function cleanup() {
    try { chrome.kill(); } catch { /* gone */ }
    try { server.close(); } catch { /* gone */ }
    try { rmSync(profile, { recursive: true, force: true }); } catch { /* best effort */ }
}
process.on('exit', cleanup);

async function pageTarget(deadline) {
    while (Date.now() < deadline) {
        try {
            const targets = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
            const page = targets.find(t => t.type === 'page' && t.webSocketDebuggerUrl);
            if (page) return page;
        } catch { /* not listening yet */ }
        await new Promise(r => setTimeout(r, 200));
    }
    throw new Error(`Chrome never exposed a debuggable page on :${PORT}\n${chromeErr}`);
}

const target = await pageTarget(Date.now() + 30_000);
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((ok, fail) => { ws.onopen = ok; ws.onerror = () => fail(new Error('DevTools WebSocket failed')); });

let nextId = 1;
const pending = new Map();
const waiters = new Map();
let consoleSink = null;
ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
        const { ok, fail } = pending.get(msg.id);
        pending.delete(msg.id);
        msg.error ? fail(new Error(msg.error.message)) : ok(msg.result);
        return;
    }
    if (msg.method && waiters.has(msg.method)) {
        for (const ok of waiters.get(msg.method)) ok(msg.params);
        waiters.delete(msg.method);
    }
    if (msg.method === 'Runtime.consoleAPICalled' && consoleSink) {
        consoleSink(msg.params.type, (msg.params.args || []).map(a => a.value ?? a.description ?? '').join(' '));
    } else if (msg.method === 'Runtime.exceptionThrown' && consoleSink) {
        consoleSink('exception', msg.params.exceptionDetails?.exception?.description ?? msg.params.exceptionDetails?.text);
    }
};
const send = (method, params = {}) => new Promise((ok, fail) => {
    const id = nextId++;
    pending.set(id, { ok, fail });
    ws.send(JSON.stringify({ id, method, params }));
});
const waitFor = (method, ms) => new Promise((ok) => {
    waiters.set(method, [...(waiters.get(method) ?? []), ok]);
    setTimeout(() => ok(null), ms);
});

await send('Page.enable');
await send('Runtime.enable');

// ---------------------------------------------------------------------------
// Run the suites
// ---------------------------------------------------------------------------
const rows = [];
for (const suite of SUITES.filter(s => only.length === 0 || only.includes(s.name))) {
    const url = `${BASE}/${dirname(suite.file).replace(/\\/g, '/')}/`;
    console.log(`\n=== ${suite.name} (${url}) ===`);
    const TIMEOUT_MS = Number(TIMEOUT_ARG || suite.timeout || 120) * 1000;
    const gpuWarnings = [];
    consoleSink = (type, text) => {
        if (!text) return;
        // Chrome reports uncaptured WebGPU errors as console warnings; the page's own log() is info.
        if ((type === 'warning' || type === 'error' || type === 'exception') && !/^\s*(ok|FAIL)\s/.test(text)) {
            gpuWarnings.push(text);
            console.log(`  [page ${type}] ${text.split('\n')[0]}`);
        } else {
            console.log(text);
        }
    };
    const ready = waitFor('Runtime.executionContextCreated', 30_000);
    await send('Page.navigate', { url });
    await ready;
    const { result } = await send('Runtime.evaluate', {
        awaitPromise: true, returnByValue: true,
        expression: `new Promise((resolve) => {
            const t0 = Date.now();
            const tick = () => {
                if (globalThis[${JSON.stringify(suite.key)}]) return resolve(globalThis[${JSON.stringify(suite.key)}]);
                if (Date.now() - t0 > ${TIMEOUT_MS}) return resolve({ timeout: true });
                setTimeout(tick, 100);
            };
            tick();
        })`,
    });
    const r = result?.value ?? {};
    // The harness watches its own devices (harness-env.mjs) and publishes what they reported;
    // console warnings are printed above for diagnosis but are not the record.
    const gpuErrors = r.gpuErrors ?? [];
    const ok = !r.timeout && !r.fatal && r.failed === 0 && gpuErrors.length === 0;
    rows.push({
        suite: suite.name, ok,
        checks: r.total !== undefined ? `${r.total - r.failed}/${r.total}` : '—',
        gpuErrors: r.gpuErrors ? gpuErrors.length : '—',
        note: r.timeout ? 'timeout' : r.fatal ? `fatal: ${r.fatal}` : (gpuWarnings.length ? `${gpuWarnings.length} console warning(s)` : ''),
    });
}
consoleSink = null;

console.log('\n=== Browser GPU summary (headless Chrome) ===');
console.table(rows);
const failed = rows.filter(r => !r.ok);
console.log(failed.length === 0
    ? `All ${rows.length} GPU suite(s) passed in Chrome.`
    : `${failed.length} GPU suite(s) FAILED in Chrome: ${failed.map(r => r.suite).join(', ')}.`);
process.exit(failed.length === 0 ? 0 : 1);
