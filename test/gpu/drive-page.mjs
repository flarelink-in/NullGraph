// npm run drive -- <url | /path> [options]
//
// Drive a WebGPU page in headless Chrome the way a person would — move, click, press keys, scroll,
// wait, screenshot, evaluate — while capturing every console line from the page AND its workers, and
// watching every GPU device the page creates. For interactive demos (the test-engine site) and the
// repository's own harness pages alike. Grew out of test/gpu/browser-run.mjs: same Chrome launch,
// same DevTools-protocol plumbing.
//
//   npm run drive -- http://localhost:8081/dashboard/agents --wait-for "Demo loaded:" --actions a.json
//   npm run drive -- --serve . /test/text-gpu/index.html --out drive-out
//
// Options
//   --actions <file.json>    run these steps after loading, then exit
//   --interactive            after loading, run <out>/actions-1.json, actions-2.json, … as each appears,
//                            writing done-N after it; a step {"end": true} finishes. For an agent that
//                            looks at each screenshot before deciding the next click.
//   --out <dir>              screenshots, console.log and report.json (default: drive-out)
//   --serve <dir>            serve <dir> with COOP/COEP (as browser-run does) and resolve a /path URL on it
//   --wait-for <regex>       before acting, wait until a console line matches (e.g. "Demo loaded:")
//   --canvas <selector>      element whose box the fractional coordinates refer to (default: canvas)
//   --allow-error <regex>    a console error that is known noise (repeatable)
//   --width <n> --height <n> viewport in CSS px (default 1440 × 900)
//   --timeout <s>            load and wait-for timeout (default 120)
//   --chrome <path>
//
// Steps (a JSON array; fx, fy are fractions of the --canvas element's box, x, y are page CSS px)
//   {"move":[fx,fy]}  {"click":[fx,fy]}  {"pageMove":[x,y]}  {"pageClick":[x,y]}
//   {"key":"KeyW","ms":500}  {"wheel":[fx,fy,deltaY]}  {"wait":ms}  {"waitFor":"regex","timeout":s}
//   {"shot":"name.png"}  {"eval":"expression"}  {"end":true}
//
// Exit codes
//   0  no uncaught exception, no console error, no WebGPU uncaptured error or device loss
//   1  any of those — each is listed in report.json
//   2  bad arguments, no Chrome, the page never loaded, or a wait-for timed out
//
// What it proves: the page ran these interactions with no recorded error, and the screenshots show
// what was on screen. What it does not: that the screenshots look right — read them.

import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { extname, join, normalize, resolve } from 'node:path';

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------
const argv = process.argv.slice(2);
const VALUE = new Set(['--actions', '--out', '--serve', '--wait-for', '--canvas', '--allow-error',
    '--width', '--height', '--timeout', '--chrome']);
const opts = { allowError: [] };
let target = null;
for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--interactive') { opts.interactive = true; continue; }
    if (VALUE.has(a)) {
        const v = argv[++i];
        if (v === undefined) usage(`${a} needs a value`);
        const key = a.slice(2).replace(/-(\w)/g, (_, c) => c.toUpperCase());
        if (key === 'allowError') opts.allowError.push(new RegExp(v)); else opts[key] = v;
        continue;
    }
    if (a.startsWith('--')) usage(`unknown option ${a}`);
    if (target) usage(`one URL only (got ${target} and ${a})`);
    target = a;
}
if (!target) usage('give a URL, or --serve <dir> and a /path');

function usage(message) {
    console.error(`drive-page: ${message}.\nUsage: npm run drive -- <url | /path> [--actions a.json | --interactive] ` +
        '[--out dir] [--serve dir] [--wait-for regex] [--canvas selector] [--allow-error regex] (see the header).');
    process.exit(2);
}

const OUT = resolve(opts.out ?? 'drive-out');
const WIDTH = Number(opts.width ?? 1440);
const HEIGHT = Number(opts.height ?? 900);
const TIMEOUT_MS = Number(opts.timeout ?? 120) * 1000;
const CANVAS = opts.canvas ?? 'canvas';
mkdirSync(OUT, { recursive: true });

const CHROME = [
    opts.chrome, process.env.CHROME_PATH,
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome', '/usr/bin/chromium',
].filter(Boolean).find(p => existsSync(p));
if (!CHROME) usage('no Chrome found — pass --chrome <path> or set CHROME_PATH');

// ---------------------------------------------------------------------------
// Optional static server (browser-run.mjs's, for repository pages)
// ---------------------------------------------------------------------------
let server = null;
let url = target;
if (opts.serve) {
    const ROOT = resolve(opts.serve);
    const MIME = {
        '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json',
        '.wasm': 'application/wasm', '.map': 'application/json', '.png': 'image/png', '.css': 'text/css',
    };
    server = createServer((req, res) => {
        let rel = decodeURIComponent(new URL(req.url, 'http://x').pathname);
        // Browsers ask for this unprompted; a 404 here would be a console error the page never made.
        if (rel === '/favicon.ico') { res.writeHead(204); res.end(); return; }
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
    // Git Bash rewrites an argument starting with "/" into a Windows path ("/x.html" arrives as
    // "C:/Program Files/Git/x.html"), which Chrome then opens as a missing file. Refuse it by name.
    if (/^[a-zA-Z]:[\\/]/.test(target)) {
        usage(`the page path arrived as "${target}" — Git Bash rewrites a leading "/" into a Windows path. ` +
            'Pass it relative to --serve (test/text-gpu/index.html), or set MSYS_NO_PATHCONV=1');
    }
    url = new URL(target.replace(/^\/+/, ''), `http://127.0.0.1:${server.address().port}/`).href;
} else if (!/^https?:\/\//.test(target)) {
    usage(`"${target}" is not an http(s) URL — use --serve <dir> for a local page`);
}

// ---------------------------------------------------------------------------
// Chrome over the DevTools protocol
// ---------------------------------------------------------------------------
const profile = mkdtempSync(join(tmpdir(), 'nullgraph-drive-'));
const chrome = spawn(CHROME, [
    '--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`,
    '--enable-unsafe-webgpu', '--no-first-run', '--no-default-browser-check', '--disable-extensions',
    `--window-size=${WIDTH},${HEIGHT}`, 'about:blank',
], { stdio: ['ignore', 'pipe', 'pipe'] });
let chromeErr = '';
chrome.stderr.on('data', d => { chromeErr += d.toString(); });
function cleanup() {
    try { chrome.kill(); } catch { /* gone */ }
    try { server?.close(); } catch { /* gone */ }
    try { rmSync(profile, { recursive: true, force: true }); } catch { /* best effort */ }
}
process.on('exit', cleanup);
for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) process.on(sig, () => { cleanup(); process.exit(2); });
const sleep = ms => new Promise(r => setTimeout(r, ms));

// Port 0 lets Chrome pick a free port and write it to <profile>/DevToolsActivePort. A fixed port can
// reach a previous run's browser that is still shutting down — and silently drive the wrong page.
let page = null;
for (const deadline = Date.now() + 30_000; !page && Date.now() < deadline;) {
    try {
        const port = Number(readFileSync(join(profile, 'DevToolsActivePort'), 'utf8').split('\n')[0]);
        page = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find(t => t.type === 'page' && t.webSocketDebuggerUrl);
    } catch { /* not listening yet */ }
    if (!page) await sleep(200);
}
if (!page) { console.error(`drive-page: Chrome exposed no debuggable page\n${chromeErr}`); process.exit(2); }
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((ok, fail) => { ws.onopen = ok; ws.onerror = () => fail(new Error('DevTools WebSocket failed')); });

let nextId = 1;
const pending = new Map();
const send = (method, params = {}, sessionId) => new Promise((ok, fail) => {
    const id = nextId++;
    pending.set(id, { ok, fail });
    ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
});

// Every device the page or a worker creates reports its uncaptured errors and its loss as console
// errors — evidence that does not depend on how Chrome words its own console warnings.
const WATCH = `(() => {
    if (!globalThis.GPUAdapter || globalThis.__ngDriveWatch) return;
    globalThis.__ngDriveWatch = true;
    const requestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (...args) {
        const device = await requestDevice.apply(this, args);
        device.addEventListener('uncapturederror', (e) => console.error('[drive-page] WebGPU uncaptured error: ' + e.error.message));
        device.lost.then((info) => { if (info.reason !== 'destroyed') console.error('[drive-page] WebGPU device lost: ' + info.message); });
        return device;
    };
})();`;

const t0 = Date.now();
const lines = [];
const errors = [];
const watchBreakpoints = new Map();
let warnings = 0;
// Chrome's own report of an uncaptured WebGPU error is a WARNING; Dawn's messages carry a
// " - While calling [Device].CreateBuffer(...)" context line. The device watch is the primary
// evidence; this catches a device the watch missed.
const DAWN = /\n\s*- While (calling|encoding|validating|finishing|processing)|^\[Invalid \w+|WebGPU: too many warnings/;
const record = (where, kind, text) => {
    if (!text || kind === 'log-verbose') return;
    const line = `[${((Date.now() - t0) / 1000).toFixed(1)}s ${where} ${kind}] ${text}`;
    lines.push(line);
    if (opts.interactive) console.log(line);
    const isError = kind === 'error' || kind === 'log-error' || kind === 'exception' ||
        ((kind === 'warning' || kind === 'log-warning') && DAWN.test(text));
    if (isError && !opts.allowError.some(re => re.test(text)) && !errors.some(e => e.text === text)) {
        errors.push({ where, kind, text });
    } else if (kind === 'warning' || kind === 'log-warning') {
        warnings++;
    }
};
let loaded = false;
ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) {
        const { ok, fail } = pending.get(m.id);
        pending.delete(m.id);
        m.error ? fail(new Error(m.error.message)) : ok(m.result);
        return;
    }
    const where = m.sessionId ? 'worker' : 'page';
    if (m.method === 'Target.attachedToTarget') {
        // A worker starts paused (waitForDebuggerOnStart), but its global scope does not exist yet,
        // so the watch cannot simply be evaluated now. A "before script execution" instrumentation
        // breakpoint stops it at its first script — scope present, nothing run — where the watch is
        // installed (Debugger.paused below). Nested workers attach the same way. Always resume.
        const s = m.params.sessionId;
        record('cdp', 'attach', `${m.params.targetInfo.type} ${m.params.targetInfo.url}`);
        Promise.allSettled([
            send('Runtime.enable', {}, s), send('Log.enable', {}, s),
            send('Debugger.enable', {}, s).then(() =>
                send('Debugger.setInstrumentationBreakpoint', { instrumentation: 'beforeScriptExecution' }, s))
                .then(r => { watchBreakpoints.set(s, r.breakpointId); }),
            send('Target.setAutoAttach', { autoAttach: true, waitForDebuggerOnStart: true, flatten: true }, s),
        ]).then(() => send('Runtime.runIfWaitingForDebugger', {}, s).catch(() => {}));
    } else if (m.method === 'Debugger.paused' && m.sessionId && watchBreakpoints.has(m.sessionId)) {
        const s = m.sessionId;
        const id = watchBreakpoints.get(s);
        watchBreakpoints.delete(s);
        Promise.allSettled([send('Runtime.evaluate', { expression: WATCH }, s)])
            .then(() => send('Debugger.removeBreakpoint', { breakpointId: id }, s).catch(() => {}))
            .then(() => send('Debugger.resume', {}, s).catch(() => {}))
            .then(() => send('Debugger.disable', {}, s).catch(() => {}));
    } else if (m.method === 'Runtime.consoleAPICalled') {
        record(where, m.params.type, (m.params.args || []).map(a => a.value ?? a.description ?? '').join(' '));
    } else if (m.method === 'Runtime.exceptionThrown') {
        const d = m.params.exceptionDetails;
        record(where, 'exception', d?.exception?.description ?? d?.text);
    } else if (m.method === 'Log.entryAdded') {
        record(where, `log-${m.params.entry.level}`, m.params.entry.text);
    } else if (m.method === 'Page.loadEventFired') {
        loaded = true;
    }
};

const report = { url, viewport: [WIDTH, HEIGHT], steps: [], screenshots: [], errors, warnings: 0, consoleLines: 0 };
function finish(code, note) {
    if (note) report.note = note;
    report.warnings = warnings;
    report.consoleLines = lines.length;
    report.exit = code;
    writeFileSync(join(OUT, 'console.log'), lines.join('\n') + '\n');
    writeFileSync(join(OUT, 'report.json'), JSON.stringify(report, null, 2));
    console.log(`drive-page: ${code === 0 ? 'clean' : code === 1 ? `${errors.length} error(s)` : 'could not run'}` +
        ` — ${report.steps.length} step(s), ${report.screenshots.length} screenshot(s), ${lines.length} console line(s)` +
        `${note ? ` — ${note}` : ''}. Output: ${OUT}`);
    for (const e of errors.slice(0, 10)) console.log(`  [${e.where} ${e.kind}] ${e.text.split('\n')[0]}`);
    cleanup();
    process.exit(code);
}

await send('Page.enable');
await send('Runtime.enable');
await send('Log.enable');
await send('Page.addScriptToEvaluateOnNewDocument', { source: WATCH });
await send('Target.setAutoAttach', { autoAttach: true, waitForDebuggerOnStart: true, flatten: true });
await send('Emulation.setDeviceMetricsOverride', { width: WIDTH, height: HEIGHT, deviceScaleFactor: 1, mobile: false });
const nav = await send('Page.navigate', { url });
if (nav.errorText) finish(2, `navigation to ${url} failed: ${nav.errorText}`);

const until = async (test, ms) => {
    for (const deadline = Date.now() + ms; Date.now() < deadline;) {
        if (test()) return true;
        await sleep(100);
    }
    return test();
};
if (!await until(() => loaded, TIMEOUT_MS)) finish(2, `the page did not finish loading within ${TIMEOUT_MS / 1000} s`);
{
    const { result } = await send('Runtime.evaluate', {
        returnByValue: true,
        expression: `[location.href, performance.getEntriesByType('navigation')[0]?.responseStatus ?? 0]`,
    });
    const [href, status] = result.value;
    if (String(href).startsWith('chrome-error:')) finish(2, `Chrome showed an error page for ${url}`);
    if (status >= 400) finish(2, `${url} answered HTTP ${status}`);
}
if (opts.waitFor) {
    const re = new RegExp(opts.waitFor);
    if (!await until(() => lines.some(l => re.test(l)), TIMEOUT_MS)) finish(2, `no console line matched /${opts.waitFor}/`);
}
await sleep(500);

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------
async function box() {
    const { result } = await send('Runtime.evaluate', {
        returnByValue: true,
        expression: `(() => { const e = document.querySelector(${JSON.stringify(CANVAS)}); if (!e) return null;
            const r = e.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; })()`,
    });
    if (!result.value) throw new Error(`no element matches --canvas ${CANVAS}`);
    return result.value;
}
const mouse = (type, x, y, extra = {}) => send('Input.dispatchMouseEvent', { type, x, y, button: 'none', ...extra });
async function at([fx, fy]) {
    const b = await box();
    return [b.x + fx * b.w, b.y + fy * b.h];
}
async function click(x, y) {
    await mouse('mouseMoved', x, y);
    await sleep(40);
    await mouse('mousePressed', x, y, { button: 'left', buttons: 1, clickCount: 1 });
    await sleep(60);
    await mouse('mouseReleased', x, y, { button: 'left', buttons: 0, clickCount: 1 });
}
async function key(code, ms) {
    const keyName = /^Key[A-Z]$/.test(code) ? code.slice(3).toLowerCase() : /^Digit\d$/.test(code) ? code.slice(5) : code;
    const vk = /^Key[A-Z]$/.test(code) ? code.charCodeAt(3) : /^Digit\d$/.test(code) ? code.charCodeAt(5) : undefined;
    await send('Input.dispatchKeyEvent', { type: 'keyDown', code, key: keyName, windowsVirtualKeyCode: vk });
    await sleep(ms);
    await send('Input.dispatchKeyEvent', { type: 'keyUp', code, key: keyName, windowsVirtualKeyCode: vk });
}

async function runSteps(steps, source) {
    if (!Array.isArray(steps)) throw new Error(`${source}: steps must be a JSON array`);
    for (const s of steps) {
        const entry = { step: s };
        if (s.move) { const [x, y] = await at(s.move); await mouse('mouseMoved', x, y); }
        if (s.click) { const [x, y] = await at(s.click); await click(x, y); }
        if (s.pageMove) await mouse('mouseMoved', s.pageMove[0], s.pageMove[1]);
        if (s.pageClick) await click(s.pageClick[0], s.pageClick[1]);
        if (s.key) await key(s.key, s.ms ?? 100);
        if (s.wheel) {
            const [x, y] = await at(s.wheel);
            await send('Input.dispatchMouseEvent', { type: 'mouseWheel', x, y, deltaX: 0, deltaY: s.wheel[2] ?? 100 });
        }
        if (s.wait) await sleep(s.wait);
        if (s.waitFor) {
            const re = new RegExp(s.waitFor);
            entry.matched = await until(() => lines.some(l => re.test(l)), (s.timeout ?? 30) * 1000);
            if (!entry.matched) { report.steps.push(entry); finish(2, `step waitFor /${s.waitFor}/ timed out`); }
        }
        if (s.shot) {
            const { data } = await send('Page.captureScreenshot', { format: 'png' });
            writeFileSync(join(OUT, s.shot), Buffer.from(data, 'base64'));
            report.screenshots.push(s.shot);
        }
        if (s.eval) {
            const { result, exceptionDetails } = await send('Runtime.evaluate', { expression: s.eval, returnByValue: true, awaitPromise: true });
            entry.result = exceptionDetails ? { exception: exceptionDetails.text } : result.value;
            record('cdp', 'eval', JSON.stringify(entry.result));
        }
        report.steps.push(entry);
        if (s.end) return true;
    }
    return false;
}

try {
    if (opts.actions) {
        await runSteps(JSON.parse(readFileSync(opts.actions, 'utf8')), opts.actions);
    } else if (opts.interactive) {
        writeFileSync(join(OUT, 'ready'), JSON.stringify(await box()));
        console.log(`drive-page: ready — write ${join(OUT, 'actions-1.json')}, then actions-2.json, … ({"end":true} finishes)`);
        for (let n = 1; ; n++) {
            const file = join(OUT, `actions-${n}.json`);
            if (!await until(() => existsSync(file), 15 * 60_000)) finish(errors.length ? 1 : 0, 'no new actions for 15 minutes');
            await sleep(100);
            const ended = await runSteps(JSON.parse(readFileSync(file, 'utf8')), file);
            writeFileSync(join(OUT, `done-${n}`), '');
            if (ended) break;
        }
    } else {
        await runSteps([{ shot: 'loaded.png' }], 'default');
    }
} catch (e) {
    finish(2, e.message);
}
finish(errors.length ? 1 : 0);
