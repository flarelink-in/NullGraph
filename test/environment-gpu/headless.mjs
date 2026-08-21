// Drive test/environment-gpu/ in headless Chrome and report to the terminal.
//
// The page itself is the test (run.js); this is only a way to run it without a human clicking.
// It launches Chrome with remote debugging, waits for `window.__SKY_RESULTS__`, prints the same
// summary the page prints, and exits non-zero on failure (design-principles §13.2) — an exit
// path that is the whole reason to have this file rather than a screenshot.
//
//   node test/environment-gpu/headless.mjs [--url http://localhost:3000/test/environment-gpu/]
//                                          [--chrome <path>] [--timeout 120]
//
// Needs the dev server running (`npm run dev:serve`) and a build (`npm run build:lib`).

import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const argv = process.argv.slice(2);
const arg = (name, fallback) => {
    const i = argv.indexOf(`--${name}`);
    return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback;
};

const URL_ = arg('url', 'http://localhost:3000/test/environment-gpu/');
const TIMEOUT_MS = Number(arg('timeout', '120')) * 1000;
const PORT = Number(arg('port', '9333'));

const CHROME_CANDIDATES = [
    arg('chrome', null),
    process.env.CHROME_PATH,
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
].filter(Boolean);
const chromePath = CHROME_CANDIDATES.find(p => existsSync(p));
if (!chromePath) {
    console.error(`No Chrome found. Tried:\n  ${CHROME_CANDIDATES.join('\n  ')}\n` +
        `Pass one with --chrome <path> or set CHROME_PATH.`);
    process.exit(1);
}

const profile = mkdtempSync(join(tmpdir(), 'nullgraph-gpu-'));
const chrome = spawn(chromePath, [
    '--headless=new',
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${profile}`,
    '--enable-unsafe-webgpu',            // WebGPU in headless is still behind this flag
    '--enable-features=Vulkan',
    '--no-first-run', '--no-default-browser-check', '--disable-extensions',
    '--window-size=512,512',
    'about:blank',                       // the suite is loaded by Page.navigate below, once
], { stdio: ['ignore', 'pipe', 'pipe'] });

let chromeErr = '';
chrome.stderr.on('data', d => { chromeErr += d.toString(); });

function cleanup() {
    try { chrome.kill(); } catch { /* already gone */ }
    try { rmSync(profile, { recursive: true, force: true }); } catch { /* best effort */ }
}
process.on('exit', cleanup);

// Top-level await means any driver fault would otherwise land as a raw stack trace, which reads
// as a failing test. Say which it is.
process.on('unhandledRejection', (e) => {
    console.error(`\nDRIVER ERROR: ${e?.message ?? e}`);
    console.error(`This is a harness failure, not a test result — open ${URL_} in a browser to ` +
        `run the suite directly.`);
    process.exit(1);
});

/** Poll the DevTools HTTP endpoint until the page target appears. */
async function findTarget(deadline) {
    while (Date.now() < deadline) {
        try {
            const res = await fetch(`http://127.0.0.1:${PORT}/json/list`);
            const targets = await res.json();
            const page = targets.find(t => t.type === 'page' && t.webSocketDebuggerUrl);
            if (page) return page;
        } catch { /* Chrome is not listening yet */ }
        await new Promise(r => setTimeout(r, 200));
    }
    throw new Error(`Chrome never exposed a debuggable page on :${PORT}\n${chromeErr}`);
}

const deadline = Date.now() + TIMEOUT_MS;
const target = await findTarget(deadline);

const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = () => reject(new Error('DevTools WebSocket failed to open'));
});

let nextId = 1;
const pending = new Map();
const eventWaiters = new Map();          // method -> [resolve, ...]
ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.method && eventWaiters.has(msg.method)) {
        for (const resolve of eventWaiters.get(msg.method)) resolve(msg.params);
        eventWaiters.delete(msg.method);
    }
    if (msg.id && pending.has(msg.id)) {
        const { resolve, reject } = pending.get(msg.id);
        pending.delete(msg.id);
        msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result);
    } else if (msg.method === 'Runtime.consoleAPICalled') {
        // The page logs each check as it runs; mirror it so a hang shows where it stopped.
        const text = (msg.params.args || []).map(a => a.value ?? a.description ?? '').join(' ');
        if (text) console.log(text);
    } else if (msg.method === 'Runtime.exceptionThrown') {
        console.error('PAGE ERROR:', msg.params.exceptionDetails?.exception?.description
            ?? msg.params.exceptionDetails?.text);
    }
};
const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
});

const waitForEvent = (method, ms) => new Promise((resolve) => {
    const list = eventWaiters.get(method) ?? [];
    list.push(resolve);
    eventWaiters.set(method, list);
    setTimeout(() => resolve(null), ms);
});

// Drive the navigation ourselves rather than trusting the command-line URL to have finished.
// Chrome exposes a debuggable target before the document has an execution context, and a
// Runtime.evaluate that lands in that window fails with "Cannot find default execution context"
// — an intermittent that reads like a WebGPU problem and is not one.
await send('Page.enable');
await send('Runtime.enable');
const contextReady = waitForEvent('Runtime.executionContextCreated', 30_000);
await send('Page.navigate', { url: URL_ });
await contextReady;

// Wait for the page to publish its results. The page sets __SKY_RESULTS__ last, including the
// `{ fatal }` shape when WebGPU is unavailable — so this resolves either way and never hangs
// silently on a device that cannot run the suite.
const waitMs = Math.max(1000, deadline - Date.now());
const { result, exceptionDetails } = await send('Runtime.evaluate', {
    awaitPromise: true,
    returnByValue: true,
    expression: `new Promise((resolve) => {
        const t0 = Date.now();
        const tick = () => {
            if (window.__SKY_RESULTS__) return resolve(window.__SKY_RESULTS__);
            if (Date.now() - t0 > ${waitMs}) {
                return resolve({ timeout: true, log: document.getElementById('log').textContent });
            }
            setTimeout(tick, 100);
        };
        tick();
    })`,
});

if (exceptionDetails) {
    console.error('Evaluation failed:', exceptionDetails.text);
    process.exit(1);
}

const r = result.value;
if (r.fatal) {
    console.error(`\nFATAL: the page could not initialise WebGPU — ${r.fatal}`);
    console.error(`This is an environment result, not a test result: headless Chrome needs a ` +
        `working WebGPU adapter. Open ${URL_} in a normal browser to run the suite.`);
    process.exit(1);
}
if (r.timeout) {
    console.error(`\nTIMEOUT after ${TIMEOUT_MS / 1000}s. Page log so far:${r.log}`);
    process.exit(1);
}

console.log(`\n${r.total - r.failed}/${r.total} GPU checks passed.`);
process.exit(r.failed === 0 ? 0 : 1);
