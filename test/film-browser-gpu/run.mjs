// The browser G tests of F2 and F3 (docs/offline-films-browser-design.md §10, §14): null-graph/film's managed
// runner in Chrome, on each GPU, read back with ffprobe and ffmpeg. A suite: it exits 1 when any check fails.
//
//   npm run test:film:browser -- [--gpu amd,nvidia] [--only G1-pattern,G2,…] [--long] [--headed] [--keep]
//        [--timeout 300] [--out dir] [--chrome path]
//
// Run `npm run build:lib` first: the page loads dist/film.js and dist/index.js through an import map, as an
// app without a bundler would. Cases: G1-pattern, G1-kerr, G2, G3, G3-kerr-4k, G5, G6, G9-cancel, G9-responsive,
// G9-background, G9-lost, G9-crash (cases.js, F2); G2-png, G4, G5-storage, G7 (with its other tab, G7-peer) and
// G8 (storage-cases.js, F3); G4-long, a 60 s 4K film, only when named or with --long. For each GPU this starts its
// own Chrome (headless unless --headed; the GTX 1650 through --force_high_performance_gpu, since Chrome on Windows
// ignores powerPreference; --expose-gc so a case can tell whether a film's engine was kept alive), serves the
// repository with COOP/COEP and a PUT endpoint the page uploads its films to, and loads index.html once per case.
// A page asks for what it cannot do itself (hide its tab, freeze, crash the GPU process, run out of quota, open
// and close another tab) by logging "[g] action <name>" (act()). Every case must leave nothing in OPFS. A DevTools
// call that does not answer in 60 s fails; a Chrome whose DevTools connection is lost (a system sleep closes it)
// fails its case and is replaced for the next. Then, for every uploaded film:
//   - ffprobe: J packets and J decoded frames, each timestamp within one container tick of j / fps, a key frame
//     every 2 s, H.264 or VP9 4:2:0 at the film's size, and an MP4's top-level boxes as its target makes them
//     (memory: ftyp moov mdat; OPFS or a picked file: ftyp moov free mdat);
//   - the pattern's: the bar code of every decoded frame (frame k shows k) and the colour of every patch,
//     decoded with the stream's own matrix and range, against the pattern's bytes; and its PNGs, exactly;
//   - G6's pair: PSNR, frame by frame, between the canvas film and the readback film.
// Results: <out>/results.json and <out>/summary.md. The films are deleted once read unless --keep.

import { spawn, spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import {
    closeSync, createWriteStream, existsSync, fstatSync, mkdirSync, mkdtempSync, openSync, readFileSync, readSync, rmSync, statSync,
    unlinkSync, writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import { readPNG } from '../../tools/film/png-read.mjs';
import { BITS, barcodeBand, patternFrame, patternSamples } from './pattern.js';

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, '../..');
const argv = process.argv.slice(2);
const flag = name => argv.includes(`--${name}`);
const option = (name, fallback) => {
    const at = argv.indexOf(`--${name}`);
    return at >= 0 ? argv[at + 1] : fallback;
};
const list = (name, fallback) => option(name, fallback).split(',').map(s => s.trim()).filter(Boolean);

const ALL = ['G1-pattern', 'G1-kerr', 'G2', 'G2-png', 'G3', 'G3-kerr-4k', 'G4', 'G4-long', 'G5', 'G5-storage', 'G6', 'G7', 'G8', 'G9-cancel', 'G9-responsive',
    'G9-background', 'G9-lost', 'G9-crash'];
/** G4-long (a 60 s 4K film: minutes on the iGPU) runs when named, or with --long. */
const LONG = flag('long');
const GPUS = list('gpu', 'amd,nvidia');
const ONLY = list('only', ALL.filter(c => c !== 'G4-long' || LONG).join(','));
const HEADED = flag('headed');
const KEEP = flag('keep');
const TIMEOUT_MS = Number(option('timeout', 300)) * 1000;
const OUT = resolve(option('out', join(here, 'out', new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19))));
const CHROME = [option('chrome'), process.env.CHROME_PATH,
    'C:/Program Files/Google/Chrome/Application/chrome.exe', 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe']
    .filter(Boolean).find(p => existsSync(p));
if (!CHROME) { console.error('film G: no Chrome found; pass --chrome <path>.'); process.exit(2); }
for (const c of ONLY) if (!ALL.includes(c)) { console.error(`film G: unknown case ${c}; cases: ${ALL.join(', ')}.`); process.exit(2); }
for (const tool of ['ffprobe', 'ffmpeg']) {
    if (spawnSync(tool, ['-version']).status !== 0) { console.error(`film G: ${tool} is not on PATH; the films cannot be read back.`); process.exit(2); }
}
if (!existsSync(join(repo, 'dist/film.js'))) { console.error('film G: dist/film.js is missing; run npm run build:lib first.'); process.exit(2); }
mkdirSync(OUT, { recursive: true });
const sleep = ms => new Promise(r => setTimeout(r, ms));

/**
 * Decoded colours against the pattern's, per codec and range: the largest channel error allowed at a patch
 * centre (a 16 × 16 mean, far from 4:2:0 edges). Measured on both GPUs on 2026-10-25 (Phase F2 results) and
 * set a little above the worst seen, so a wrong matrix, range, channel order or alpha handling fails.
 */
const COLOUR_TOLERANCE = { 'h264 tv': 4, 'h264 pc': 3, 'vp9 tv': 4, 'vp9 pc': 3 };
/** G6: the canvas and readback films of one scene, decoded with their own tags, frame by frame. */
const PAIR_MIN_PSNR_DB = 30;

// ---------------------------------------------------------------------------------------------------
// Serve the repository; receive uploads
// ---------------------------------------------------------------------------------------------------

let uploadDir = OUT;
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.map': 'application/json' };
const server = createServer(async (req, res) => {
    const path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (req.method === 'PUT' && path.startsWith('/__g/upload/')) {
        const name = path.slice('/__g/upload/'.length);
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
// Chrome over the DevTools protocol
// ---------------------------------------------------------------------------------------------------

async function openChrome(gpu) {
    const profile = mkdtempSync(join(tmpdir(), 'nullgraph-film-g-'));
    const args = [
        ...(HEADED ? ['--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding'] : ['--headless=new']),
        '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--enable-unsafe-webgpu', '--no-first-run',
        '--no-default-browser-check', '--disable-extensions', '--window-size=1440,900', '--js-flags=--expose-gc',
        // performance.memory unquantized, and measureUserAgentSpecificMemory() answering at once (G4, as in S1).
        '--enable-precise-memory-info', '--enable-blink-features=ForceEagerMeasureMemory',
        // A finished case's page would otherwise wait in the back/forward cache, holding its memory (S1).
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
    // The browser target (DevToolsActivePort's second line), for what only the browser can do: crash its GPU process.
    const [port, browserPath] = readFileSync(join(profile, 'DevToolsActivePort'), 'utf8').split('\n').map(s => s.trim());
    const browser = await connect(`ws://127.0.0.1:${port}${browserPath}`, () => {});
    return {
        pid: proc.pid, lines, send: page.send, browser: browser.send, targetId: target.id,
        /** Both DevTools sockets are open: a system sleep can close them, and then this Chrome is no use to the run. */
        alive: () => !page.closed && !browser.closed,
        async close() {
            page.close();
            browser.close();
            try { proc.kill(); } catch { /* gone */ }
            await sleep(1000);
            try { rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 300 }); } catch { /* best effort */ }
        },
    };
}

/** A DevTools call that has not answered in this long has been lost: every call the runner makes answers in seconds. */
const CDP_TIMEOUT_MS = 60_000;

/**
 * A DevTools connection. Every call answers, fails, or times out: when the socket closes (a system sleep closed
 * both of the runner's on 2026-10-25), the calls waiting on it fail at once, and later ones at the call.
 */
async function connect(url, onEvent) {
    const ws = new WebSocket(url);
    await new Promise((ok, fail) => { ws.onopen = ok; ws.onerror = () => fail(new Error(`DevTools WebSocket ${url} failed`)); });
    let nextId = 1;
    const pending = new Map();
    const conn = {
        closed: false,
        send: (method, params = {}) => new Promise((ok, fail) => {
            if (conn.closed) { fail(new Error(`DevTools socket closed; ${method} not sent`)); return; }
            const id = nextId++;
            const timer = setTimeout(() => { pending.delete(id); fail(new Error(`DevTools did not answer ${method} in ${CDP_TIMEOUT_MS / 1000} s`)); }, CDP_TIMEOUT_MS);
            pending.set(id, { ok: value => { clearTimeout(timer); ok(value); }, fail: error => { clearTimeout(timer); fail(error); } });
            ws.send(JSON.stringify({ id, method, params }));
        }),
        close: () => { try { ws.close(); } catch { /* gone */ } },
    };
    const drop = why => {
        conn.closed = true;
        for (const { fail } of pending.values()) fail(new Error(`DevTools socket closed (${why}) before it answered`));
        pending.clear();
    };
    ws.onclose = ev => drop(`code ${ev.code}`);
    ws.onerror = () => drop('error');
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
    return conn;
}

/**
 * What a page asks for and cannot do itself ("[g] action <name>"):
 * - gpucrash: end Chrome's GPU process (chrome://gpucrash works only when typed into the address bar; DevTools
 *   has a command for it). Not awaited: the browser need not answer a command that ends one of its processes.
 * - hide / show: put a blank tab in front of the page, then bring the page back and close the blank tab.
 * - freeze [ms]: freeze the page (no task of its runs) for ms (default 1000), then thaw it; the page cannot ask
 *   to be thawed.
 * - quota [bytes]: override this origin's storage quota (0 restores it): how G8 runs OPFS out of room.
 * - open-peer <case> / close-peer: a second tab of this origin running that case, in the background (G7's other
 *   writer: it reports over a BroadcastChannel, its console unread), and closing it, as a user would.
 */
async function act(chrome, action, arg) {
    if (action === 'gpucrash') {
        chrome.browser('Browser.crashGpuProcess').catch(e => console.log(`   gpucrash: ${e.message}`));
    } else if (action === 'hide') {
        chrome.blank = (await chrome.browser('Target.createTarget', { url: 'about:blank' })).targetId;
        await chrome.browser('Target.activateTarget', { targetId: chrome.blank });
    } else if (action === 'show') {
        await chrome.browser('Target.activateTarget', { targetId: chrome.targetId });
        if (chrome.blank) await chrome.browser('Target.closeTarget', { targetId: chrome.blank });
        chrome.blank = null;
    } else if (action === 'quota') {
        await chrome.send('Storage.overrideQuotaForOrigin', { origin, ...(Number(arg) ? { quotaSize: Number(arg) } : {}) });
    } else if (action === 'open-peer') {
        const url = `${origin}/test/film-browser-gpu/index.html?case=${encodeURIComponent(arg)}`;
        chrome.peer = (await chrome.browser('Target.createTarget', { url, background: true })).targetId;
    } else if (action === 'close-peer') {
        if (chrome.peer) await chrome.browser('Target.closeTarget', { targetId: chrome.peer });
        chrome.peer = null;
    } else if (action === 'freeze') {
        await chrome.send('Page.setWebLifecycleState', { state: 'frozen' });
        await sleep(Number(arg ?? 1000));
        await chrome.send('Page.setWebLifecycleState', { state: 'active' });
    } else {
        throw new Error(`unknown action ${action}`);
    }
}

// ---------------------------------------------------------------------------------------------------
// Reading the films back
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

const ff = (tool, args, maxBuffer = 1 << 28) => spawnSync(tool, args, { maxBuffer });

/** What ffprobe says of the film: stream, packets (pts and key flags in the stream's time base), decoded frames. */
function probe(file) {
    const s = ff('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries',
        'stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,time_base,nb_read_frames,nb_frames,duration,' +
        'color_range,color_space,color_transfer,color_primaries:format=format_name,duration,size,bit_rate', '-of', 'json', file]);
    if (s.status !== 0) return { error: String(s.stderr).trim() };
    const { streams: [stream = {}] = [], format = {} } = JSON.parse(String(s.stdout));
    const [tbNum, tbDen] = (stream.time_base ?? '1/1').split('/').map(Number);
    const packets = String(ff('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'packet=pts,flags', '-of', 'csv=p=0', file]).stdout)
        .trim().split(/\r?\n/).filter(Boolean).map(l => { const [pts, flags] = l.split(','); return { pts: Number(pts), key: flags.includes('K') }; });
    const boxes = file.endsWith('.mp4') ? topLevelBoxes(file) : [];
    const index = type => boxes.findIndex(b => b.type === type);
    return {
        codec: stream.codec_name, profile: stream.profile, width: stream.width, height: stream.height, pixFmt: stream.pix_fmt,
        timeBase: [tbNum, tbDen], decodedFrames: Number(stream.nb_read_frames), packets,
        color: { range: stream.color_range, space: stream.color_space, transfer: stream.color_transfer, primaries: stream.color_primaries },
        bitRate: Number(format.bit_rate), bytes: Number(format.size), format: format.format_name,
        boxes: boxes.map(b => `${b.type}:${b.size}`), moovBeforeMdat: boxes.length ? index('moov') >= 0 && index('moov') < index('mdat') : null,
    };
}

/** ffmpeg's YUV → RGB with the stream's own tags: swscale's default, BT.601, reads BT.709 red as 232, 0, 1 (S1). */
function rgbFilter(color = {}) {
    const matrix = { bt709: 'bt709', smpte170m: 'bt601', bt470bg: 'bt601' }[color.space] ?? 'bt709';
    const range = color.range === 'pc' ? 'pc' : 'tv';
    return { filter: `scale=in_color_matrix=${matrix}:in_range=${range}:out_range=pc`, matrix, range };
}

/** The bar code of every decoded frame: each bit's column averaged to one grey byte, then thresholded. */
function barcodes(file, width, height, color) {
    const band = barcodeBand(width, height);
    const r = ff('ffmpeg', ['-v', 'error', '-i', file, '-vf',
        `${rgbFilter(color).filter},crop=${band.width}:${band.height}:${band.x}:${band.y},scale=${BITS}:1:flags=area,format=gray`,
        '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1']);
    if (r.status !== 0) return { error: String(r.stderr) };
    const out = [];
    let ambiguous = 0;
    for (let at = 0; at + BITS <= r.stdout.length; at += BITS) {
        let value = 0;
        for (let b = 0; b < BITS; b++) {
            const v = r.stdout[at + b];
            if (v > 64 && v < 192) ambiguous++;
            value = value * 2 + (v >= 128 ? 1 : 0);
        }
        out.push(value);
    }
    return { indices: out, ambiguous };
}

/** Frame `index` decoded to RGB with the stream's tags, and each pattern sample's 16 × 16 mean against its bytes. */
function patchColours(file, width, height, index, color) {
    const r = ff('ffmpeg', ['-v', 'error', '-i', file, '-vf', `select=eq(n\\,${index}),${rgbFilter(color).filter}`,
        '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], width * height * 3 + (1 << 20));
    if (r.status !== 0 || r.stdout.length < width * height * 3) return { error: String(r.stderr) };
    const rgb = r.stdout;
    return patternSamples(width, height).map(s => {
        const sum = [0, 0, 0];
        for (let y = s.y - 8; y < s.y + 8; y++) for (let x = s.x - 8; x < s.x + 8; x++) for (let c = 0; c < 3; c++) sum[c] += rgb[(y * width + x) * 3 + c];
        const got = sum.map(v => Math.round(v / 256));
        return { name: s.name, want: s.rgb, got, error: Math.max(...got.map((v, c) => Math.abs(v - s.rgb[c]))) };
    });
}

/** PSNR (RGB, dB) of every frame of `b` against `a`, each decoded with its own tags. */
function pairPSNR(a, b, colorA, colorB) {
    const r = ff('ffmpeg', ['-v', 'error', '-i', a, '-i', b, '-lavfi',
        `[0:v]${rgbFilter(colorA).filter},format=rgb24[a];[1:v]${rgbFilter(colorB).filter},format=rgb24[b];[a][b]psnr=stats_file=-`, '-f', 'null', '-']);
    if (r.status !== 0) return { error: String(r.stderr) };
    const values = [...String(r.stdout).matchAll(/psnr_avg:(\S+)/g)].map(m => (m[1] === 'inf' ? Infinity : Number(m[1])));
    const finite = values.filter(Number.isFinite);
    return { frames: values.length, min: Math.min(...values), mean: finite.length ? finite.reduce((x, y) => x + y, 0) / finite.length : Infinity };
}

/**
 * An MP4's top-level boxes, in order, as its target makes them (§4.5): in memory `ftyp moov mdat`
 * (fastStart in-memory); written to OPFS or a picked file `ftyp moov free mdat`, the index reserved at the
 * front for J packets and its unused room left as a free box (fastStart reserve).
 */
function boxOrder(v, p, add) {
    if (v.container !== 'mp4') return;
    const reserve = v.target === 'opfs' || v.target === 'file';
    const want = reserve ? ['ftyp', 'moov', 'free', 'mdat'] : ['ftyp', 'moov', 'mdat'];
    const got = p.boxes.map(b => b.split(':')[0]);
    add(`boxes ${want.join(' ')}: moov first (${v.target ?? 'memory'}, fastStart ${reserve ? 'reserve' : 'in-memory'})`, JSON.stringify(got) === JSON.stringify(want), p.boxes);
}

/** The checks of one uploaded film, against what the page said it should hold. */
function checkVideo(dir, v, checks) {
    const path = join(dir, v.file);
    const at = v.id ?? v.file;
    const add = (name, ok, detail = null) => checks.push({ name: `${at}: ${name}`, ok: !!ok, detail });
    if (!existsSync(path)) { add('was uploaded', false); return null; }
    const p = probe(path);
    if (p.error) { add('ffprobe reads it', false, p.error); return p; }
    const facts = { codec: p.codec, profile: p.profile, pixFmt: p.pixFmt, color: p.color, timeBase: p.timeBase.join('/'), boxes: p.boxes, bytes: p.bytes };
    if (v.partial) {
        add(`a kept partial decodes to its ${v.frames} frames`, p.packets.length === v.frames && p.decodedFrames === v.frames, { packets: p.packets.length, decoded: p.decodedFrames });
        boxOrder(v, p, add);
        return { ...p, facts };
    }
    const J = v.frames;
    add(`${J} packets and ${J} decoded frames`, p.packets.length === J && p.decodedFrames === J, { packets: p.packets.length, decoded: p.decodedFrames });
    add(`${v.width}×${v.height} ${v.container === 'webm' ? 'vp9' : 'h264'} yuv420p`,
        p.width === v.width && p.height === v.height && p.codec === (v.container === 'webm' ? 'vp9' : 'h264') && p.pixFmt === 'yuv420p', facts);
    const [num, den] = p.timeBase;
    const pts = p.packets.map(x => x.pts).sort((a, b) => a - b);
    // The container stores T(j) = round(j · 10⁶ / fps) µs in its own time base: within one tick of it.
    const tick = pts.map((t, j) => Math.abs((t - pts[0]) - Math.round(j * den / (v.fps * num))));
    const worst = Math.max(0, ...tick);
    add('every timestamp within one container tick of j / fps, from 0', pts[0] === 0 && worst <= 1, { firstPts: pts[0], worstTicks: worst, timeBase: `${num}/${den}` });
    const keys = p.packets.filter(x => x.key).map(x => x.pts).sort((a, b) => a - b);
    const wantKeys = Math.ceil(J / (2 * v.fps));
    const spacing = keys.slice(1).map((k, i) => Math.round((k - keys[i]) * num / den * 1000) / 1000);
    add(`a key frame every 2 s (${wantKeys})`, keys.length === wantKeys && keys[0] === 0 && spacing.every(s => Math.abs(s - 2) < 0.02), { keys: keys.length, spacing });
    boxOrder(v, p, add);
    if (v.pattern) {
        const codes = barcodes(path, v.width, v.height, p.color);
        const wrong = codes.indices ? codes.indices.map((k, j) => (k === j ? null : [j, k])).filter(Boolean) : null;
        add('decoded frame k shows frame k: none dropped, doubled or reordered', codes.indices?.length === J && wrong.length === 0 && codes.ambiguous === 0,
            codes.error ?? { frames: codes.indices.length, wrong: wrong.slice(0, 8), ambiguous: codes.ambiguous });
        const middle = Math.floor(J / 2);
        const patches = patchColours(path, v.width, v.height, middle, p.color);
        const key = `${p.codec} ${p.color.range === 'pc' ? 'pc' : 'tv'}`;
        const tolerance = COLOUR_TOLERANCE[key] ?? 0;
        const worstPatch = patches.error ? null : patches.reduce((a, b) => (b.error > a.error ? b : a));
        add(`decoded colours within ${key}'s ${tolerance} of the pattern's bytes (matrix ${rgbFilter(p.color).matrix}, range ${rgbFilter(p.color).range})`,
            !patches.error && worstPatch.error <= tolerance, patches.error ?? { worst: worstPatch, patches: patches.map(x => `${x.name} ${x.got.join(',')} (${x.error})`) });
        const glow = patches.error ? null : patches.find(x => x.name === 'additive a=0');
        add('the additive glow at alpha 0 survives as opaque colour', glow && glow.error <= tolerance, glow);
        return { ...p, facts, patches, barcodes: { frames: codes.indices?.length, ambiguous: codes.ambiguous } };
    }
    return { ...p, facts };
}

// ---------------------------------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------------------------------

const results = { started: new Date().toISOString(), argv, origin, headed: HEADED, chrome: CHROME, runs: [] };
console.log(`film G: ${ONLY.length} case(s) on ${GPUS.join(', ')}${HEADED ? ' (headed)' : ' (headless)'}; output ${OUT}`);

for (const gpu of GPUS) {
    uploadDir = join(OUT, gpu);
    mkdirSync(uploadDir, { recursive: true });
    let chrome = await openChrome(gpu);
    console.log(`\n== ${gpu}: Chrome pid ${chrome.pid}`);
    for (const name of ONLY) {
        // A Chrome whose DevTools connection was lost (a system sleep closes it) is replaced before the next case.
        if (!chrome.alive()) {
            await chrome.close();
            chrome = await openChrome(gpu);
            console.log(`== ${gpu}: DevTools connection lost; Chrome restarted, pid ${chrome.pid}`);
        }
        const from = chrome.lines.length;
        const t0 = Date.now();
        let lost = null;
        await chrome.send('Page.navigate', { url: `${origin}/test/film-browser-gpu/index.html?case=${encodeURIComponent(name)}` }).catch(e => { lost = e.message; });
        let finished = false;
        let seen = from;
        const actions = [];
        for (const deadline = t0 + TIMEOUT_MS; !finished && !lost && Date.now() < deadline;) {
            for (; seen < chrome.lines.length; seen++) {
                const [, action, arg] = /^\[g\] action ([\w-]+)(?: (\S+))?/.exec(chrome.lines[seen].text) ?? [];
                if (action) actions.push(await act(chrome, action, arg).then(() => `${action}: ok`, e => `${action}: ${e.message}`));
            }
            finished = chrome.lines.slice(from).some(l => /^\[g\] (done|failed)/.test(l.text));
            if (!chrome.alive()) lost = 'DevTools socket closed';
            if (!finished && !lost) await sleep(100);
        }
        let result = null;
        if (lost) {
            result = { ok: false, error: `the runner lost Chrome's DevTools connection (${lost}); a system sleep closes it` };
        } else {
            try {
                const { result: r } = await chrome.send('Runtime.evaluate', { expression: 'JSON.stringify(globalThis.__FILM_G__ ?? null)', returnByValue: true });
                result = JSON.parse(r.value ?? 'null');
            } catch (e) { result = { ok: false, error: `could not read the result: ${e.message}` }; }
            if (!finished) result = { ...(result ?? {}), ok: false, error: `timed out after ${TIMEOUT_MS / 1000} s` };
        }
        const log = chrome.lines.slice(from).map(l => `[${((l.t - t0) / 1000).toFixed(1)}s ${l.kind}] ${l.text}`);
        const checks = [...(result?.checks ?? [])];
        if (!result?.ok) checks.push({ name: 'the case ran to its end', ok: false, detail: result?.error ?? 'no result' });
        // Chrome picks the GPU, not the page: every adapter the case was given must be the one asked for.
        const vendors = [...new Set((result?.adapters ?? []).map(a => a.vendor))];
        checks.push({ name: `every adapter was the ${gpu} GPU`, ok: vendors.length === 1 && vendors[0] === gpu, detail: result?.adapters?.[0] ?? null });
        // Every film a case made it removed (a kept one after reading it back): what is left in OPFS leaked.
        checks.push({ name: 'nothing left in OPFS', ok: (result?.leftoverFilms ?? []).length === 0, detail: result?.leftoverFilms ?? null });
        if (chrome.peer) await act(chrome, 'close-peer').catch(() => {});
        const files = {};
        for (const v of result?.videos ?? []) files[v.file] = checkVideo(uploadDir, v, checks);
        for (const pair of result?.pairs ?? []) {
            const a = (result.videos ?? []).find(v => v.id === pair.a), b = (result.videos ?? []).find(v => v.id === pair.b);
            if (!a || !b || !files[a.file] || !files[b.file]) { checks.push({ name: `${pair.a} against ${pair.b}: both films`, ok: false }); continue; }
            const psnr = pairPSNR(join(uploadDir, a.file), join(uploadDir, b.file), files[a.file].color, files[b.file].color);
            files[`${pair.a}~${pair.b}`] = psnr;
            checks.push({ name: `${pair.a} against ${pair.b}: every decoded frame within ${PAIR_MIN_PSNR_DB} dB PSNR`, ok: !psnr.error && psnr.frames === a.frames && psnr.min >= PAIR_MIN_PSNR_DB, detail: psnr });
        }
        // PNGs of the pattern (a snapshot, a sequence's frames): each must decode to the pattern's bytes exactly.
        for (const png of result?.pngs ?? []) {
            const path = join(uploadDir, png.file);
            const want = patternFrame(png.width, png.height, png.frame);
            const got = existsSync(path) ? readPNG(readFileSync(path)) : null;
            checks.push({ name: `${png.file} decodes to the pattern's exact bytes (frame ${png.frame})`, ok: !!got && got.width === png.width && got.height === png.height && Buffer.compare(Buffer.from(got.rgba), Buffer.from(want)) === 0 });
        }
        const errors = log.filter(l => /\b(exception|error)\]/.test(l) && !/\[g\] (FAIL|failed)/.test(l));
        const entry = { gpu, case: name, ms: Date.now() - t0, ok: checks.length > 0 && checks.every(c => c.ok), checks, result, files, actions, log, consoleErrors: errors };
        results.runs.push(entry);
        const failed = checks.filter(c => !c.ok);
        console.log(`${gpu} ${name}: ${entry.ok ? 'ok' : 'FAILED'} — ${checks.length - failed.length}/${checks.length} checks (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
        for (const c of failed) console.log(`   FAIL ${c.name}${c.detail != null ? ` — ${JSON.stringify(c.detail).slice(0, 400)}` : ''}`);
        writeFileSync(join(OUT, 'results.json'), JSON.stringify(results, null, 1));
        await chrome.send('Page.navigate', { url: 'about:blank' }).catch(() => {});
        await sleep(1000);
        if (!KEEP) {
            for (const f of [...(result?.videos ?? []).map(v => v.file), ...(result?.pngs ?? []).map(p => p.file)].filter(Boolean)) {
                const p = join(uploadDir, f);
                if (existsSync(p)) unlinkSync(p);
            }
        }
    }
    await chrome.close();
}
server.close();

// ---------------------------------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------------------------------

const md = [`# Film G tests (${results.started}, ${HEADED ? 'headed' : 'headless'} Chrome)`, ''];
for (const gpu of GPUS) {
    const runs = results.runs.filter(r => r.gpu === gpu);
    const a = runs.find(r => r.result?.adapters?.length)?.result.adapters[0] ?? {};
    md.push(`## ${gpu}: ${a.vendor ?? '?'} ${a.architecture ?? ''} ${a.description ?? ''}`.trim(), '', `Browser: ${runs.find(r => r.result?.browser)?.result.browser ?? '?'}`, '');
    md.push('| case | result | checks | s |', '|---|---|---|---|');
    for (const r of runs) md.push(`| ${r.case} | ${r.ok ? 'ok' : '**FAILED**'} | ${r.checks.filter(c => c.ok).length}/${r.checks.length} | ${(r.ms / 1000).toFixed(1)} |`);
    md.push('');
    for (const r of runs.filter(x => !x.ok)) {
        md.push(`### ${r.case}: failed checks`, '');
        for (const c of r.checks.filter(x => !x.ok)) md.push(`- ${c.name}${c.detail != null ? `: \`${JSON.stringify(c.detail).slice(0, 600)}\`` : ''}`);
        md.push('');
    }
}
const failed = results.runs.filter(r => !r.ok);
writeFileSync(join(OUT, 'summary.md'), md.join('\n'));
console.log(`\n${md.join('\n')}\n\nfilm G: ${failed.length ? `${failed.length} case(s) FAILED` : 'all passed'} — ${join(OUT, 'results.json')}`);
process.exit(failed.length ? 1 : 0);
