// Spike S1 of D2.4 (docs/offline-films-browser-design.md §14): browser film export, four ways, both GPUs.
//
//   node test/spikes/film-browser/run.mjs [--gpu amd,nvidia] [--only alpha,pace,export,noyield,opfs,cancel]
//        [--ways visible,offscreen,worker,readback,headless] [--sizes 1080,4k] [--frames 300] [--headed]
//        [--keep] [--timeout 300] [--out dir] [--chrome path]
//   Not in the default set: --only encoder (each size with prefer-hardware, then prefer-software) and
//   --only memory (a 4K film of --memory-frames, default 1800, into memory and then OPFS; --memory-hw,
//   --memory-qp), --only qp (prefer-hardware at each of --qps, default 26,30,34) and --only cpuframes
//   (JavaScript-made frames, the same bytes in every run, to tell whether two runs share an encoder).
//
// For each GPU it starts its own Chrome (headless unless --headed; the GTX 1650 through
// --force_high_performance_gpu, since Chrome on Windows ignores powerPreference), serves the repository
// with COOP/COEP as test/gpu/drive-page.mjs does, plus a PUT endpoint the page uploads its files to, and
// loads index.html once per case. While a case runs, it samples the working set of Chrome's renderer and
// GPU processes. Afterwards ffprobe reads every MP4 (packets, timestamps, codec, the order of the top-level
// boxes), ffmpeg decodes the alpha pattern clips, and each way's sampled raw frame is compared with the
// readback's. Results: <out>/results.json and <out>/summary.md. By default the raw frames and MP4s over
// 32 MB are deleted once read (--keep keeps them). A measurement, not a suite: it exits 0 unless it could
// not run at all.

import { execFile, spawn, spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import {
    closeSync, createWriteStream, existsSync, fstatSync, mkdirSync, mkdtempSync, openSync, readFileSync, readSync, rmSync, statSync,
    unlinkSync, writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { build } from 'esbuild';
import { PATCHES, patchCentre } from './pattern.js';

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, '../../..');
const argv = process.argv.slice(2);
const flag = name => argv.includes(`--${name}`);
const option = (name, fallback) => {
    const at = argv.indexOf(`--${name}`);
    return at >= 0 ? argv[at + 1] : fallback;
};
const list = (name, fallback) => new Set(option(name, fallback).split(',').map(s => s.trim()).filter(Boolean));

const GPUS = [...list('gpu', 'amd,nvidia')];
const ONLY = list('only', 'alpha,pace,export,noyield,opfs,cancel');
const WAYS = option('ways') ? list('ways') : null;
const SIZES = [...list('sizes', '1080,4k')].map(s => ({ 1080: [1920, 1080], '4k': [3840, 2160], 720: [1280, 720] })[s]);
const FRAMES = Number(option('frames', 300));
const HEADED = flag('headed');
const KEEP = flag('keep');
const TIMEOUT_MS = Number(option('timeout', 300)) * 1000;
const OUT = resolve(option('out', join(here, 'out', new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19))));
const CHROME = [option('chrome'), process.env.CHROME_PATH,
    'C:/Program Files/Google/Chrome/Application/chrome.exe', 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe']
    .filter(Boolean).find(p => existsSync(p));
if (!CHROME) { console.error('s1: no Chrome found; pass --chrome <path>.'); process.exit(2); }
if (SIZES.some(s => !s)) { console.error('s1: --sizes takes 720, 1080 and 4k.'); process.exit(2); }
mkdirSync(OUT, { recursive: true });
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ---------------------------------------------------------------------------------------------------
// The cases, in the order they run on each GPU
// ---------------------------------------------------------------------------------------------------

const CANVAS_WAYS = ['visible', 'offscreen', 'worker', 'readback'];
const HW = { 'no-preference': '-np', 'prefer-hardware': '-hw', 'prefer-software': '-sw' };
const idOf = c => `${c.case}-${c.way}-${c.w}x${c.h}${c.tag ? `-${c.tag}` : ''}${c.target === 'opfs' ? '-opfs' : ''}` +
    `${c.yield === 'none' ? '-noyield' : ''}${HW[c.hw] ?? ''}${c.brmode === 'constant' ? '-cbr' : ''}${c.qp != null ? `-qp${c.qp}` : ''}`;
const MEMORY_FRAMES = Number(option('memory-frames', 1800));
const MEMORY_HW = option('memory-hw', 'no-preference');
const MEMORY_QP = option('memory-qp') === undefined ? null : Number(option('memory-qp'));
function cases() {
    const out = [];
    // Frames are sampled (copied back, hashed, the middle one uploaded raw) at 1080p only: at 4K each copy
    // is ~90 MB of transient JS memory, which would blur the memory-against-OPFS comparison.
    const film = (way, w, h, extra = {}) => ({
        case: 'export', way, w, h, frames: FRAMES, count: 1048576, target: 'memory',
        ...(w === 1920 ? { sample: [0, FRAMES >> 1, FRAMES - 1].join(','), raw: FRAMES >> 1 } : { sample: '' }), ...extra,
    });
    if (ONLY.has('alpha')) for (const way of CANVAS_WAYS) out.push({ case: 'alpha', way, w: 1280, h: 720, frames: 30 });
    if (ONLY.has('pace')) for (const way of ['headless', ...CANVAS_WAYS]) out.push({ case: 'pace', way, w: 1920, h: 1080, frames: 120, count: 65536 });
    if (ONLY.has('export')) for (const [w, h] of SIZES) for (const way of CANVAS_WAYS) out.push(film(way, w, h));
    if (ONLY.has('noyield')) for (const way of ['visible', 'offscreen']) out.push(film(way, 1920, 1080, { yield: 'none' }));
    if (ONLY.has('opfs')) out.push(film('offscreen', 3840, 2160, { target: 'opfs' }));
    if (ONLY.has('cancel')) out.push({ case: 'cancel', way: 'offscreen', w: 1920, h: 1080, frames: 120, count: 65536 });
    // Which encoder 'no-preference' lands on: the same frames under each preference. Equal mdat hashes
    // mean the same encoder (frames are deterministic per GPU, as the export cases show).
    if (ONLY.has('encoder')) {
        for (const [w, h] of SIZES) {
            for (const hw of Object.keys(HW)) out.push(film('offscreen', w, h, { hw }));
            out.push(film('offscreen', w, h, { hw: 'prefer-hardware', brmode: 'constant' }));
            for (const hw of ['prefer-hardware', 'prefer-software']) out.push(film('offscreen', w, h, { hw, qp: 20 }));
        }
    }
    // Constant-QP on the hardware path: what each quantizer costs in bitrate (--qps, default 26,30,34).
    if (ONLY.has('qp')) {
        for (const [w, h] of SIZES) for (const qp of option('qps', '26,30,34').split(',').map(Number)) out.push(film('offscreen', w, h, { hw: 'prefer-hardware', qp }));
    }
    // Frames made in JavaScript, identical on every run: equal mdat hashes across GPUs = one encoder.
    if (ONLY.has('cpuframes')) {
        for (const [w, h] of SIZES) for (const hw of ['prefer-hardware', 'prefer-software']) out.push({ case: 'cpuframes', way: 'cpu', w, h, frames: 120, hw });
    }
    // A film long enough that its MP4 is large: memory against OPFS, one after the other.
    if (ONLY.has('memory')) {
        for (const target of ['memory', 'opfs']) {
            out.push(film('offscreen', 3840, 2160, { tag: 'long', frames: MEMORY_FRAMES, target, hw: MEMORY_HW, ...(MEMORY_QP != null ? { qp: MEMORY_QP } : {}) }));
        }
    }
    return out.filter(c => !WAYS || WAYS.has(c.way)).map(c => ({ ...c, id: idOf(c) }));
}

// ---------------------------------------------------------------------------------------------------
// Bundle (a module worker has no import map), serve, receive uploads
// ---------------------------------------------------------------------------------------------------

await build({
    entryPoints: { page: join(here, 'page.js'), worker: join(here, 'worker.js') },
    outdir: join(here, 'out', 'bundle'), bundle: true, format: 'esm', platform: 'browser', target: 'es2022',
    sourcemap: true, logLevel: 'warning',
    plugins: [{
        name: 'node-crypto-shim',
        setup(b) { b.onResolve({ filter: /^node:crypto$/ }, () => ({ path: join(here, 'crypto-shim.js') })); },
    }],
});

let uploadDir = OUT;
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.map': 'application/json' };
const server = createServer(async (req, res) => {
    const path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (req.method === 'PUT' && path.startsWith('/__s1/upload/')) {
        const name = path.slice('/__s1/upload/'.length);
        if (!/^[\w.-]+$/.test(name)) { res.writeHead(400); res.end(); return; }
        try { await pipeline(req, createWriteStream(join(uploadDir, name))); res.writeHead(204); } catch { res.writeHead(500); }
        res.end();
        return;
    }
    if (path === '/favicon.ico') { res.writeHead(204); res.end(); return; }
    const file = normalize(join(repo, path.endsWith('/') ? `${path}index.html` : path));
    if (!file.startsWith(repo) || !existsSync(file) || !statSync(file).isFile()) { res.writeHead(404); res.end('not found'); return; }
    res.writeHead(200, {
        'Content-Type': MIME[extname(file)] ?? 'application/octet-stream',
        'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp', 'Cache-Control': 'no-store',
    });
    res.end(readFileSync(file));
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const origin = `http://127.0.0.1:${server.address().port}`;

// ---------------------------------------------------------------------------------------------------
// Chrome over the DevTools protocol (the page target only; workers report through the page)
// ---------------------------------------------------------------------------------------------------

async function openChrome(gpu) {
    const profile = mkdtempSync(join(tmpdir(), 'nullgraph-s1-'));
    const args = [
        ...(HEADED ? ['--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding'] : ['--headless=new']),
        '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--enable-unsafe-webgpu', '--no-first-run',
        '--no-default-browser-check', '--disable-extensions', '--window-size=1440,900',
        // performance.memory unquantized, and measureUserAgentSpecificMemory() answering at once.
        '--enable-precise-memory-info', '--enable-blink-features=ForceEagerMeasureMemory',
        // Without this a finished case's page waits in the back/forward cache, its renderer holding
        // hundreds of MiB for a minute or more while the next case runs (seen in the first full run).
        '--disable-features=BackForwardCache',
        ...(gpu === 'nvidia' ? ['--force_high_performance_gpu'] : []), 'about:blank',
    ];
    const proc = spawn(CHROME, args, { stdio: ['ignore', 'ignore', 'pipe'] });
    let stderr = '';
    proc.stderr.on('data', d => { stderr = (stderr + d).slice(-20000); });
    let target = null;
    for (const deadline = Date.now() + 30_000; !target && Date.now() < deadline;) {
        try {
            const port = Number(readFileSync(join(profile, 'DevToolsActivePort'), 'utf8').split('\n')[0]);
            target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find(t => t.type === 'page' && t.webSocketDebuggerUrl);
        } catch { /* not listening yet */ }
        if (!target) await sleep(200);
    }
    if (!target) throw new Error(`Chrome exposed no page\n${stderr}`);
    const lines = [];
    const page = await connect(target.webSocketDebuggerUrl, m => {
        if (m.method === 'Runtime.consoleAPICalled') {
            lines.push({ t: Date.now(), kind: m.params.type, text: (m.params.args ?? []).map(a => a.value ?? a.description ?? '').join(' ') });
        } else if (m.method === 'Runtime.exceptionThrown') {
            const d = m.params.exceptionDetails;
            lines.push({ t: Date.now(), kind: 'exception', text: d?.exception?.description ?? d?.text });
        } else if (m.method === 'Log.entryAdded') {
            lines.push({ t: Date.now(), kind: `log-${m.params.entry.level}`, text: m.params.entry.text });
        }
    });
    await page.send('Page.enable');
    await page.send('Runtime.enable');
    await page.send('Log.enable');
    // The browser endpoint (second line of DevToolsActivePort) says which of Chrome's processes is the
    // renderer and which the GPU process; polled, since a navigation can start a new renderer.
    const [port, browserPath] = readFileSync(join(profile, 'DevToolsActivePort'), 'utf8').split('\n').map(s => s.trim());
    const browser = await connect(`ws://127.0.0.1:${port}${browserPath}`, () => {});
    const types = new Map();
    // CPU seconds per process over time: the case's renderer is the one that worked during the case.
    const cpu = [];
    const poll = async () => {
        try {
            const t = Date.now();
            for (const p of (await browser.send('SystemInfo.getProcessInfo')).processInfo) {
                types.set(p.id, p.type);
                cpu.push({ t, pid: p.id, cpuTime: p.cpuTime });
            }
        } catch { /* closing */ }
    };
    await poll();
    const poller = setInterval(poll, 500);
    return {
        pid: proc.pid, lines, types, cpu, send: page.send, poll,
        async close() {
            clearInterval(poller);
            page.close();
            browser.close();
            try { proc.kill(); } catch { /* gone */ }
            await sleep(1000);
            try { rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 300 }); } catch { /* best effort */ }
        },
    };
}

async function connect(url, onEvent) {
    const ws = new WebSocket(url);
    await new Promise((ok, fail) => { ws.onopen = ok; ws.onerror = () => fail(new Error(`DevTools WebSocket ${url} failed`)); });
    let nextId = 1;
    const pending = new Map();
    ws.onmessage = ev => {
        const m = JSON.parse(ev.data);
        if (m.id && pending.has(m.id)) {
            const { ok, fail } = pending.get(m.id);
            pending.delete(m.id);
            if (m.error) fail(new Error(m.error.message)); else ok(m.result);
        } else {
            onEvent(m);
        }
    };
    return {
        send: (method, params = {}) => new Promise((ok, fail) => {
            const id = nextId++;
            pending.set(id, { ok, fail });
            ws.send(JSON.stringify({ id, method, params }));
        }),
        close: () => { try { ws.close(); } catch { /* gone */ } },
    };
}

/**
 * The working set of every chrome.exe, polled with tasklist about four times a second; run.mjs keeps the
 * ones SystemInfo names as this Chrome's. A dedicated worker lives in its page's renderer process.
 * Not PowerShell: on this machine a PowerShell loop enumerating chrome.exe was terminated after about
 * 12 s (exit 0; the same loop over explorer.exe ran on), and a WMI Win32_Process query stalls for good
 * while processes are being created and ended, as Chrome's are on every navigation (both 2026-10-24).
 */
function startSampler() {
    const samples = [];
    let stopped = false;
    const tick = () => {
        if (stopped) return;
        const t = Date.now();
        execFile('tasklist', ['/fo', 'csv', '/nh', '/fi', 'IMAGENAME eq chrome.exe'], { windowsHide: true }, (error, out) => {
            if (!error) {
                for (const line of out.split(/\r?\n/)) {
                    const cols = line.match(/"[^"]*"/g)?.map(s => s.slice(1, -1));
                    if (cols?.length >= 5) samples.push({ t, pid: Number(cols[1]), ws: Number(cols[4].replace(/\D/g, '')) * 1024 });
                }
            }
            setTimeout(tick, 150);
        });
    };
    tick();
    return { samples, stop: () => { stopped = true; } };
}

/**
 * Working-set MiB of the case's renderer and of the GPU process: the median over the 2 s before `from`
 * (one sample can catch a passing spike), the peak in [from, to], the median inside, and the value at
 * `to`. The case's renderer is the one whose CPU time grew most in [from, to] (SystemInfo); without CPU
 * data, the one with the highest peak.
 */
function memoryWindow(samples, types, cpu, from, to) {
    const mib = b => (b == null ? null : Math.round(b / 1048576));
    const median = v => (v.length ? [...v].sort((a, b) => a - b)[v.length >> 1] : null);
    const cpuGrowth = pid => {
        const mine = cpu.filter(c => c.pid === pid);
        const a = mine.filter(c => c.t <= from).at(-1) ?? mine.find(c => c.t >= from);
        const b = mine.filter(c => c.t <= to).at(-1);
        return a && b ? b.cpuTime - a.cpuTime : -1;
    };
    const out = {};
    for (const type of ['renderer', 'GPU']) {
        const series = new Map();
        for (const s of samples) {
            if (types.get(s.pid) !== type || s.t < from - 2000 || s.t > to + 300) continue;
            if (!series.has(s.pid)) series.set(s.pid, []);
            series.get(s.pid).push(s);
        }
        let best = null;
        for (const [pid, list] of series) {
            const inside = list.filter(s => s.t >= from && s.t <= to).map(s => s.ws);
            const score = { cpu: cpuGrowth(pid), peak: inside.length ? Math.max(...inside) : -1 };
            if (!best || score.cpu > best.score.cpu || (score.cpu === best.score.cpu && score.peak > best.score.peak)) {
                best = { pid, list, inside, score };
            }
        }
        if (!best || !best.inside.length) { out[type] = { base: null, peak: null, end: null, samples: 0 }; continue; }
        out[type] = {
            pid: best.pid, cpuSeconds: Math.round(best.score.cpu * 10) / 10,
            base: mib(median(best.list.filter(s => s.t < from).map(s => s.ws))), peak: mib(best.score.peak),
            median: mib(median(best.inside)), end: mib(best.list.filter(s => s.t <= to).at(-1)?.ws), samples: best.inside.length,
        };
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------
// Reading the files back
// ---------------------------------------------------------------------------------------------------

function topLevelBoxes(file) {
    const fd = openSync(file, 'r');
    const size = fstatSync(fd).size;
    const head = Buffer.alloc(16);
    const boxes = [];
    for (let at = 0; at + 8 <= size;) {
        readSync(fd, head, 0, 16, at);
        let len = head.readUInt32BE(0);
        if (len === 1) len = Number(head.readBigUInt64BE(8));
        else if (len === 0) len = size - at;
        boxes.push({ type: head.toString('latin1', 4, 8), at, size: len });
        if (len < 8) break;
        at += len;
    }
    closeSync(fd);
    return boxes;
}

function probeMp4(file, fps) {
    const ff = args => spawnSync('ffprobe', args, { encoding: 'utf8', maxBuffer: 1 << 28 });
    const s = ff(['-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries',
        'stream=codec_name,profile,level,width,height,pix_fmt,r_frame_rate,avg_frame_rate,time_base,nb_read_packets,nb_frames,duration,' +
        'color_range,color_space,color_transfer,color_primaries:format=format_name,duration,size,bit_rate', '-of', 'json', file]);
    if (s.status !== 0) return { error: s.stderr.trim() };
    const { streams: [stream = {}] = [], format = {} } = JSON.parse(s.stdout);
    const [tbNum, tbDen] = (stream.time_base ?? '1/1').split('/').map(Number);
    const packets = ff(['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'packet=pts,flags', '-of', 'csv=p=0', file])
        .stdout.trim().split(/\r?\n/).filter(Boolean).map(l => { const [pts, flags] = l.split(','); return { pts: Number(pts), key: flags.includes('K') }; });
    const pts = packets.map(p => p.pts).sort((a, b) => a - b);
    let maxTickError = 0;
    for (let k = 0; k < pts.length; k++) maxTickError = Math.max(maxTickError, Math.abs(pts[k] - pts[0] - Math.round(k * tbDen / (fps * tbNum))));
    const keyPts = packets.filter(p => p.key).map(p => p.pts).sort((a, b) => a - b);
    const trace = ff(['-v', 'trace', file]).stderr;
    const boxes = topLevelBoxes(file);
    const index = type => boxes.findIndex(b => b.type === type);
    return {
        codec: stream.codec_name, profile: stream.profile, level: stream.level, width: stream.width, height: stream.height,
        pixFmt: stream.pix_fmt, rFrameRate: stream.r_frame_rate, timeBase: stream.time_base, packets: Number(stream.nb_read_packets),
        nbFrames: Number(stream.nb_frames), duration: Number(stream.duration ?? format.duration), bitRate: Number(format.bit_rate),
        color: { range: stream.color_range, space: stream.color_space, transfer: stream.color_transfer, primaries: stream.color_primaries },
        firstPts: pts[0], maxTickError, keyFrames: keyPts.length,
        keyEverySeconds: keyPts.length > 1 ? Math.round((keyPts[1] - keyPts[0]) * tbNum / tbDen * 1000) / 1000 : null,
        boxes: boxes.map(b => `${b.type}:${b.size}`), moovBeforeMdat: index('moov') >= 0 && index('moov') < index('mdat'),
        ffprobeRootBoxes: [...trace.matchAll(/type:'(.{4})' parent:'root'/g)].map(m => m[1]),
        // The encoded payload alone: two files from one encoder and one set of frames share it, whatever
        // the container's own fields (creation times) hold.
        mdatSha256: index('mdat') >= 0 ? createHash('sha256').update(readFileSync(file).subarray(boxes[index('mdat')].at, boxes[index('mdat')].at + boxes[index('mdat')].size)).digest('hex') : null,
    };
}

/**
 * Patch centres of decoded frame 15, each the mean of a 9 × 9 block, beside what an opaque video should
 * hold. The YUV → RGB matrix and range follow the stream's own tags: swscale's default is BT.601, which
 * reads BT.709 red (255, 0, 0) as (232, 0, 1).
 */
function decodePatches(file, width, height, color = {}) {
    const matrix = { bt709: 'bt709', smpte170m: 'bt601', bt470bg: 'bt601' }[color.space] ?? 'bt709';
    const range = color.range === 'pc' ? 'pc' : 'tv';
    const r = spawnSync('ffmpeg', ['-v', 'error', '-i', file, '-vf', `select=eq(n\\,15),scale=in_color_matrix=${matrix}:in_range=${range}:out_range=pc`,
        '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], { maxBuffer: width * height * 3 + (1 << 20) });
    if (r.status !== 0 || r.stdout.length < width * height * 3) return { error: String(r.stderr) };
    const rgb = r.stdout;
    return PATCHES.map((p, i) => {
        const [cx, cy] = patchCentre(i, width, height);
        const sum = [0, 0, 0];
        for (let y = cy - 4; y <= cy + 4; y++) for (let x = cx - 4; x <= cx + 4; x++) for (let c = 0; c < 3; c++) sum[c] += rgb[(y * width + x) * 3 + c];
        return { name: p.name, opaque: p.rgba.slice(0, 3), got: sum.map(v => Math.round(v / 81)) };
    });
}

/** The way's raw RGBA frame against the readback's: how many pixels differ in RGB, and by how much. */
function compareRaw(a, b) {
    const x = readFileSync(a);
    const y = readFileSync(b);
    if (x.length !== y.length) return { error: `sizes differ (${x.length} vs ${y.length})` };
    let differing = 0;
    let maxDiff = 0;
    for (let i = 0; i < x.length; i += 4) {
        const d = Math.max(Math.abs(x[i] - y[i]), Math.abs(x[i + 1] - y[i + 1]), Math.abs(x[i + 2] - y[i + 2]));
        if (d) { differing++; if (d > maxDiff) maxDiff = d; }
    }
    return { pixels: x.length / 4, differing, maxDiff };
}

// ---------------------------------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------------------------------

const results = { started: new Date().toISOString(), argv, origin, headed: HEADED, chrome: CHROME, runs: [] };
const plan = cases();
console.log(`s1: ${plan.length} case(s) on ${GPUS.join(', ')}${HEADED ? ' (headed)' : ' (headless)'}; output ${OUT}`);

for (const gpu of GPUS) {
    uploadDir = join(OUT, gpu);
    mkdirSync(uploadDir, { recursive: true });
    const chrome = await openChrome(gpu);
    const sampler = startSampler();
    console.log(`\n== ${gpu}: Chrome pid ${chrome.pid}`);
    for (const c of plan) {
        const query = new URLSearchParams(Object.fromEntries(Object.entries(c).map(([k, v]) => [k, String(v)])));
        const url = `${origin}/test/spikes/film-browser/index.html?${query}`;
        const from = chrome.lines.length;
        const t0 = Date.now();
        await chrome.send('Page.navigate', { url });
        let finished = false;
        for (const deadline = t0 + TIMEOUT_MS; !finished && Date.now() < deadline;) {
            finished = chrome.lines.slice(from).some(l => /^\[s1\] (done|failed)/.test(l.text));
            if (!finished) await sleep(200);
        }
        let result = null;
        try {
            const { result: r } = await chrome.send('Runtime.evaluate', { expression: 'JSON.stringify(globalThis.__FILM_S1__ ?? null)', returnByValue: true });
            result = JSON.parse(r.value ?? 'null');
        } catch (e) { result = { ok: false, error: `could not read the result: ${e.message}` }; }
        if (!finished) result = { ...(result ?? {}), ok: false, error: `timed out after ${TIMEOUT_MS / 1000} s` };
        const log = chrome.lines.slice(from).map(l => `[${((l.t - t0) / 1000).toFixed(1)}s ${l.kind}] ${l.text}`);
        const entry = { gpu, ...c, ms: Date.now() - t0, result, log };

        // The case's own window for the process memory: export-start … export-end, or the pace loop.
        const marks = Object.fromEntries((result?.marks ?? []).map(m => [m.name, m.at]));
        const from_ = marks['export-start'] ?? marks['pace-start'];
        const to = marks['export-end'] ?? marks['pace-end'];
        if (from_ && to) {
            await chrome.poll();
            entry.memory = memoryWindow(sampler.samples, chrome.types, chrome.cpu, from_, to);
        }
        const mp4 = join(uploadDir, `${c.id}.mp4`);
        if (existsSync(mp4)) {
            entry.probe = probeMp4(mp4, Number(c.fps ?? 60));
            if (c.case === 'alpha') entry.decoded = decodePatches(mp4, c.w, c.h, entry.probe.color);
        }
        results.runs.push(entry);
        const r = result ?? {};
        console.log(`${c.id}: ${r.ok ? 'ok' : `FAILED ${String(r.error).split('\n')[0]}`}` +
            (r.msPerFrame ? ` — ${r.msPerFrame} ms/frame, ${(r.bytes / 1048576).toFixed(1)} MB ${r.sink?.codec}` : '') +
            (r.frame ? ` — ${r.frame.mean} ms/frame` : '') + (entry.probe ? `, ffprobe ${entry.probe.packets} packets, moov first ${entry.probe.moovBeforeMdat}` : '') +
            (entry.memory ? `, renderer ${entry.memory.renderer.base}→${entry.memory.renderer.peak} MiB` : '') + ` (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
        writeFileSync(join(OUT, 'results.json'), JSON.stringify(results, null, 1));
        await chrome.send('Page.navigate', { url: 'about:blank' }).catch(() => {});
        await sleep(1500);
    }
    sampler.stop();
    writeFileSync(join(uploadDir, 'process-memory.json'), JSON.stringify({
        types: Object.fromEntries(chrome.types), cpu: chrome.cpu, samples: sampler.samples.filter(s => chrome.types.has(s.pid)),
    }));
    await chrome.close();

    // Each way's raw frame against the readback's, at the same size, on this GPU.
    for (const e of results.runs.filter(x => x.gpu === gpu && x.case === 'export' && x.raw !== undefined && x.way !== 'readback')) {
        const mine = join(uploadDir, `${e.id}-f${e.raw}.rgba`);
        const ref = join(uploadDir, `${idOf({ ...e, way: 'readback', yield: undefined })}-f${e.raw}.rgba`);
        if (existsSync(mine) && existsSync(ref)) e.compare = compareRaw(mine, ref);
    }
    if (!KEEP) {
        for (const e of results.runs.filter(x => x.gpu === gpu)) {
            for (const name of [`${e.id}-f${e.raw}.rgba`, `${e.id}.mp4`]) {
                const f = join(uploadDir, name);
                if (existsSync(f) && (name.endsWith('.rgba') || statSync(f).size > 32 << 20)) unlinkSync(f);
            }
        }
    }
    writeFileSync(join(OUT, 'results.json'), JSON.stringify(results, null, 1));
}
server.close();

// ---------------------------------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------------------------------

const md = [`# Spike S1 results (${results.started}, ${HEADED ? 'headed' : 'headless'} Chrome)`, ''];
const fmt = (x, d = 1) => (x == null || Number.isNaN(x) ? '–' : typeof x === 'number' ? x.toFixed(d) : String(x));
for (const gpu of GPUS) {
    const runs = results.runs.filter(r => r.gpu === gpu);
    const a = runs.find(r => r.result?.adapter)?.result.adapter ?? {};
    md.push(`## ${gpu}: ${a.vendor ?? '?'} ${a.architecture ?? ''} ${a.description ?? ''}`.trim(), '');
    md.push(`Browser: ${runs.find(r => r.result?.browser)?.result.browser.userAgent ?? '?'}`, '');

    const exports = runs.filter(r => r.case === 'export');
    if (exports.length) {
        md.push('| case | ms/frame | frame p50 / p90 / max | add p50 / sum | fence sum | capture p50 | finalize | MB | codec (asked) | hw / sw supported | ffprobe packets | tick err | moov first | renderer WS MiB base→peak (median) | GPU proc WS MiB base→peak | rAF p90 | long tasks | vs readback |',
            '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|');
        for (const r of exports) {
            const x = r.result ?? {};
            if (!x.ok) { md.push(`| ${r.id} | FAILED: ${String(x.error).split('\n')[0]} |`); continue; }
            const p = r.probe ?? {};
            const m = r.memory ?? {};
            md.push(`| ${r.id} | ${fmt(x.msPerFrame, 2)} | ${fmt(x.per.total?.p50)} / ${fmt(x.per.total?.p90)} / ${fmt(x.per.total?.max)} | ` +
                `${fmt(x.per.add?.p50)} / ${fmt(x.per.add?.sum, 0)} | ${fmt(x.per.fence?.sum, 0)} | ${fmt(x.per.capture?.p50, 2)} | ${x.finalizeMs} | ` +
                `${fmt(x.bytes / 1048576)} | ${x.sink?.config?.codec ?? x.sink?.codec} (${x.sink?.config?.hardwareAcceleration ?? '?'}) | ` +
                `${x.sink?.support?.['prefer-hardware']} / ${x.sink?.support?.['prefer-software']} | ` +
                `${p.packets ?? '–'} | ${p.maxTickError ?? '–'} | ${p.moovBeforeMdat ?? '–'} | ${m.renderer?.base ?? '–'}→${m.renderer?.peak ?? '–'} (${m.renderer?.median ?? '–'}) | ` +
                `${m.GPU?.base ?? '–'}→${m.GPU?.peak ?? '–'} | ${x.mainThread?.rafP90 ?? '–'} | ${x.mainThread ? `${x.mainThread.longTasks} (${x.mainThread.longTaskMs} ms)` : '–'} | ` +
                `${r.compare ? (r.compare.differing === 0 ? 'identical' : `${r.compare.differing} px, max ${r.compare.maxDiff}`) : '–'} |`);
        }
        md.push('');
    }
    const paces = runs.filter(r => r.case === 'pace');
    if (paces.length) {
        // The canvas ways fence every 8 frames, so single frames are bimodal (submit-only, or the fence's
        // wait); the mean is the throughput.
        md.push('| pace (65,536 particles, 1080p, a task per frame) | mean ms/frame | max | ≥ 15 ms | capture p50 | onSubmittedWorkDone p50 | mapAsync p50 | rAF count |',
            '|---|---|---|---|---|---|---|---|');
        for (const r of paces) {
            const x = r.result ?? {};
            if (!x.ok) { md.push(`| ${r.id} | FAILED: ${String(x.error).split('\n')[0]} |`); continue; }
            md.push(`| ${r.id} | ${fmt(x.frame?.mean, 2)} | ${fmt(x.frame?.max, 2)} | ${x.over15ms} | ${fmt(x.capture?.p50, 2)} | ` +
                `${fmt(x.fence?.onSubmittedWorkDone?.p50, 2)} | ${fmt(x.fence?.mapAsync?.p50, 2)} | ${x.mainThread?.rafCount ?? '–'} |`);
        }
        md.push('');
    }
    const alphas = runs.filter(r => r.case === 'alpha');
    if (alphas.length) {
        md.push('| alpha pattern | canvas format | frame format | before encoding: a=0.5 / a=0 / clear | decoded: a=0.5 / a=0 / clear (opaque expects 102,51,26 / 204,102,51 / 0,0,0) | decoded grey | colour tags |',
            '|---|---|---|---|---|---|---|');
        for (const r of alphas) {
            const x = r.result ?? {};
            if (!x.ok) { md.push(`| ${r.id} | FAILED: ${String(x.error).split('\n')[0]} |`); continue; }
            const pre = x.captured?.patches;
            const dec = Array.isArray(r.decoded) ? r.decoded : null;
            md.push(`| ${r.id} | ${x.canvasFormat} | ${x.captured?.format} | ${pre ? [5, 6, 7].map(i => pre[i].got.join(',')).join(' / ') : '–'} | ` +
                `${dec ? [5, 6, 7].map(i => dec[i].got.join(',')).join(' / ') : '–'} | ${dec ? dec[3].got.join(',') : '–'} | ` +
                `${r.probe?.color ? Object.values(r.probe.color).join(' ') : '–'} |`);
        }
        md.push('');
    }
    const cpuRuns = runs.filter(r => r.case === 'cpuframes');
    if (cpuRuns.length) {
        md.push('| CPU-made frames (120) | MB | wall ms | codec (asked) | mdat SHA-256 (first 16) | same payload on the other GPU |', '|---|---|---|---|---|---|');
        for (const r of cpuRuns) {
            const x = r.result ?? {};
            if (!x.ok) { md.push(`| ${r.id} | FAILED: ${String(x.error).split('\n')[0]} |`); continue; }
            const other = results.runs.find(o => o.id === r.id && o.gpu !== r.gpu);
            const hash = r.probe?.mdatSha256;
            md.push(`| ${r.id} | ${fmt(x.bytes / 1048576, 2)} | ${x.wallMs} | ${x.sink?.config?.codec} (${x.sink?.config?.hardwareAcceleration}) | ${hash?.slice(0, 16) ?? '–'} | ` +
                `${other?.probe?.mdatSha256 ? (other.probe.mdatSha256 === hash ? 'yes' : 'no') : '–'} |`);
        }
        md.push('');
    }
    for (const r of runs.filter(x => x.case === 'cancel')) {
        const x = r.result ?? {};
        md.push(x.ok ? `Cancel over OPFS: ${x.variants.map(v => `${v.variant}: ${v.sizeAfterCancel} bytes after cancel (${v.encodedBytes} encoded; entries ${v.entries.join(', ') || 'none'})`).join('; ')}` : `Cancel: FAILED ${x.error}`, '');
    }
}
writeFileSync(join(OUT, 'summary.md'), md.join('\n'));
console.log(`\n${md.join('\n')}\n\ns1: done — ${join(OUT, 'results.json')}`);
