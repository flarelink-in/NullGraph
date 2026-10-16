// Cycle D, step D2, phase D2.0 (docs/offline-films-design.md §11, §17): what an offline film costs on this
// machine, before any tool is built. A measurement, not a suite. Node only: it reads the test site's
// files, pipes frames to ffmpeg and compresses with zstd.
//
//   NG_DAWN_FLAGS=adapter=NVIDIA npm run spike:films      NG_SPIKE_CASE=steps|draw|encode|pack (default steps)
//   NG_SPIKE_MAX      stops at a count (default 1,048,576)
//   NG_SPIKE_LIMITMS  a step or a draw predicted longer is not run (default 1000, half the ~2 s after which
//                     Windows resets the GPU); one measured over 1.5 s stops the case
//   NG_SITE           the test site's folder (look.mjs reads its modules and public/ assets)
//   Results go to test/spikes/films/out/ (ignored by git): <case>-<gpu>.json, the draw case's PNGs, the
//   encode case's videos.
//
// Cases. The simulation is tools/agents/scenario/herd-wolves.mjs with both species on the committed
// networks (MlpBrain), stepped as a film steps it: one step per submission, waited for.
//   steps   16K, 64K, 256K, 1M; thinkEvery 3 and 60; GPU time (timestamps) and wall time per step, at
//           spawn and after settling (herds form, and a step's cost moves); GPU bytes allocated
//   draw    the demo's Models look (look.mjs) at 1920 × 1080 over the stepped simulation; three views
//   encode  16K, the default view: frames stepped, drawn, read back and piped to ffmpeg, per encoder
//   pack    the 20-byte agents preset of the design (§4.2) packed on the GPU and read back; zstd on it
import { mkdirSync, statSync, writeFileSync, readFileSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { zstdCompressSync, constants as zlibConstants } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { harnessEnv } from '../../gpu/harness-env.mjs';
import { readTextureRGBA8 } from '../../gpu/readback.mjs';
import { encodePNG } from '../../gpu/png.mjs';
import { NullGraph } from '../../../dist/index.js';
import { MlpBrain, AGENT_STRUCT_WGSL } from '../../../dist/agents.js';
import { herdWolves } from '../../../tools/agents/scenario/herd-wolves.mjs';
import { networkFromModel } from '../../../tools/agents/model.mjs';

const env = harnessEnv('__FILM_SPIKE_RESULTS__');
const { log, check } = env;
const CASE = process.env.NG_SPIKE_CASE ?? 'steps';
const COUNTS = [16_384, 65_536, 262_144, 1_048_576];
const MAX = Number(process.env.NG_SPIKE_MAX ?? COUNTS[COUNTS.length - 1]);
const LIMIT_MS = Number(process.env.NG_SPIKE_LIMITMS ?? 1000);
const STOP_MS = 1500;
const WIDTH = 1920, HEIGHT = 1080;
const OUT = fileURLToPath(new URL('./out/', import.meta.url));
const fmt = n => n.toLocaleString('en-US');
const median = v => { const s = [...v].sort((a, b) => a - b); return s.length ? s[Math.floor(s.length / 2)] : NaN; };
const max = v => Math.max(...v);
const MB = bytes => (bytes / 1048576).toFixed(1);
mkdirSync(OUT, { recursive: true });

// --- the engine, its GPU, and an account of what is allocated on it ---------------------------
const engine = new NullGraph();
const init = await engine.init(null, { width: WIDTH, height: HEIGHT, desiredFeatures: ['timestamp-query'] });
if (!init.success) { env.fatal(init.error); throw new Error(init.error); }
const { device } = engine;
if (!device.features.has('timestamp-query')) { env.fatal('no timestamp-query'); throw new Error('no timestamp-query'); }
const info = device.adapterInfo ?? {};
const gpu = /nvidia/i.test(`${info.vendor} ${info.description}`) ? 'nvidia' : /amd|ati/i.test(`${info.vendor} ${info.description}`) ? 'amd' : 'gpu';
log(`films D2.0 (${CASE}): Node/Dawn · ${[info.vendor, info.architecture, info.description].filter(Boolean).join(' ')} · ` +
    `limit ${LIMIT_MS} ms per submission, stop above ${STOP_MS} ms`);

const BPP = { 'rgba8unorm': 4, 'rgba8unorm-srgb': 4, 'bgra8unorm': 4, 'rgba16float': 8, 'rgba32float': 16, 'depth24plus': 4,
    'depth32float': 4, 'r8unorm': 1, 'r32float': 4, 'rg16float': 4, 'r16float': 2 };
const account = { live: 0, peak: 0 };
const track = (size, thing) => {
    account.live += size;
    account.peak = Math.max(account.peak, account.live);
    const destroy = thing.destroy.bind(thing);
    let gone = false;
    thing.destroy = () => { if (!gone) { gone = true; account.live -= size; } destroy(); };
    return thing;
};
const createBuffer = device.createBuffer.bind(device), createTexture = device.createTexture.bind(device);
device.createBuffer = d => track(d.size, createBuffer(d));
device.createTexture = d => {
    const [w, h = 1, layers = 1] = Array.isArray(d.size) ? d.size : [d.size.width, d.size.height, d.size.depthOrArrayLayers];
    return track(w * h * layers * (BPP[d.format] ?? 4) * (d.mipLevelCount > 1 ? 4 / 3 : 1), createTexture(d));
};
/** Measures the peak of GPU bytes allocated from now until `done()`. */
const allocation = () => {
    const base = account.live;
    account.peak = account.live;
    return () => account.peak - base;
};

// --- timestamps: two marker passes around a submission ---------------------------------------
const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
const resolveBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
const readBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
const mark = index => {
    const enc = device.createCommandEncoder();
    enc.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: index } }).end();
    device.queue.submit([enc.finish()]);
};
async function gpuMs() {
    const enc = device.createCommandEncoder();
    enc.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
    enc.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, 16);
    device.queue.submit([enc.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const t = new BigUint64Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    return Number(t[1] - t[0]) / 1e6;
}
/** Submits `sequence` alone between two marks and waits for it: { gpu, wall } in ms. */
async function timed(sequence) {
    const t0 = performance.now();
    mark(0);
    engine.render(sequence);
    mark(1);
    await device.queue.onSubmittedWorkDone();
    const wall = performance.now() - t0;
    return { gpu: await gpuMs(), wall };
}

// --- the simulation: the scenario, both species on the committed networks -----------------------
const MODELS = new URL('../../../tools/agents/models/herd-wolves/', import.meta.url);
const files = Object.fromEntries(['prey', 'wolf'].map(s => [s, JSON.parse(readFileSync(new URL(`${s}.mlp.json`, MODELS), 'utf8'))]));
function buildSim(count, thinkEvery) {
    const sim = herdWolves.create(engine, { count, thinkEvery });
    const brains = ['prey', 'wolf'].map(species => {
        const brain = new MlpBrain(engine, networkFromModel(files[species], sim.definition, `${species}.mlp.json`));
        sim.agents.setBrain(brain, { species });
        return brain;
    });
    return { ...sim, destroy() { sim.destroy(); brains.forEach(b => b.destroy()); } };
}
/** The one-step chain, as tools/agents/label-pass.mjs's lockstep takes it: after every stage is attached. */
function stepChain(agents) {
    if (agents.advance(agents.definition.dt) !== 1) throw new Error('films spike: advance(dt) did not select one step');
    agents.upload();
    return agents.getComputeStages();
}

/** The demo's guard rule: ns per agent × N × (N / M)^0.25, from the largest measured count M below N. */
function predict(measured, count) {
    const below = [...measured.keys()].filter(c => c < count);
    if (!below.length) return { ms: 0, from: null };
    const m = Math.max(...below);
    return { ms: (measured.get(m) / m) * count * (count / m) ** 0.25, from: m };
}
const save = (name, data) => writeFileSync(`${OUT}${name}-${gpu}.json`, JSON.stringify(data, null, 2));

/** Steps `n` times, each alone, timed; stops (throws) on a step over STOP_MS. */
async function steps(chain, n) {
    const out = [];
    for (let k = 0; k < n; k++) {
        const t = await timed(chain);
        if (t.wall > STOP_MS) throw new Error(`a step took ${t.wall.toFixed(0)} ms, over the ${STOP_MS} ms stop`);
        out.push(t);
    }
    return out;
}
const summary = runs => ({ gpu: median(runs.map(r => r.gpu)), wall: median(runs.map(r => r.wall)),
    gpuMax: max(runs.map(r => r.gpu)), wallMax: max(runs.map(r => r.wall)) });

// =============================================================================================
// steps
// =============================================================================================
async function caseSteps() {
    const SAMPLES = { 16384: 30, 65536: 30, 262144: 20, 1048576: 10 };
    const SETTLE = { 16384: 600, 65536: 600, 262144: 300, 1048576: 0 };
    const rows = [];
    for (const thinkEvery of [3, 60]) {
        const measured = new Map();
        for (const count of COUNTS.filter(c => c <= MAX)) {
            const p = predict(measured, count);
            if (p.ms > LIMIT_MS) {
                log(`   thinkEvery ${thinkEvery}, ${fmt(count)}: not run, ~${p.ms.toFixed(0)} ms per step predicted from ${fmt(p.from)} (limit ${LIMIT_MS})`);
                rows.push({ thinkEvery, count, skipped: true, predicted: p });
                break;
            }
            const done = allocation();
            const built = performance.now();
            const sim = buildSim(count, thinkEvery);
            const chain = stepChain(sim.agents);
            const buildMs = performance.now() - built;
            try {
                await steps(chain, 3);                                    // pipelines compile on the first
                const spawn = summary(await steps(chain, SAMPLES[count]));
                let settled = null;
                if (SETTLE[count]) {
                    await steps(chain, SETTLE[count]);
                    settled = summary(await steps(chain, SAMPLES[count]));
                }
                const bytes = done();
                const worst = Math.max(spawn.gpu, settled?.gpu ?? 0);
                measured.set(count, worst);
                const row = { thinkEvery, count, predicted: p, buildMs, spawn, settled, settleSteps: SETTLE[count], bytes };
                rows.push(row);
                log(`   thinkEvery ${String(thinkEvery).padStart(2)}, ${fmt(count).padStart(9)}: step ${spawn.gpu.toFixed(2)} ms GPU ` +
                    `(${spawn.wall.toFixed(2)} wall) at spawn` +
                    (settled ? `, ${settled.gpu.toFixed(2)} (${settled.wall.toFixed(2)}) after ${SETTLE[count]} steps` : '') +
                    ` · ${((worst / count) * 1e6).toFixed(0)} ns per agent · predicted ${p.from ? `${p.ms.toFixed(1)} ms` : '—'}` +
                    ` · ${MB(bytes)} MB allocated · built in ${buildMs.toFixed(0)} ms`);
                check(`steps: thinkEvery ${thinkEvery}, ${fmt(count)} measured`, Number.isFinite(spawn.gpu) && spawn.gpu > 0);
            } finally {
                sim.destroy();
                await device.queue.onSubmittedWorkDone();
            }
        }
    }
    save('steps', { device: info, limitMs: LIMIT_MS, rows });
}

// =============================================================================================
// draw
// =============================================================================================
async function caseDraw() {
    const { siteModules } = await import('./site.mjs');
    const { loadAssets, buildLook, VIEWS } = await import('./look.mjs');
    const site = await siteModules();
    const assets = loadAssets(engine);
    const stepMeasured = new Map(), drawMeasured = new Map();
    const rows = [];
    for (const count of COUNTS.filter(c => c <= MAX)) {
        const p = predict(stepMeasured, count);
        if (p.ms > LIMIT_MS) {
            log(`   ${fmt(count)}: not built, ~${p.ms.toFixed(0)} ms per step predicted from ${fmt(p.from)}`);
            rows.push({ count, skipped: 'step', predicted: p });
            break;
        }
        const done = allocation();
        const sim = buildSim(count, 3);
        const look = buildLook(engine, sim, site, assets, { width: WIDTH, height: HEIGHT });
        const chain = stepChain(sim.agents);
        try {
            const settle = await steps(chain, 30);
            const step = summary(settle.slice(5));
            stepMeasured.set(count, step.gpu);
            const row = { count, palette: look.usePalette, step, views: {} };
            for (const name of Object.keys(VIEWS)) {
                const key = `${name}`;
                const measuredHere = new Map([...drawMeasured].filter(([k]) => k.endsWith(`/${key}`)).map(([k, v]) => [Number(k.split('/')[0]), v]));
                const pd = predict(measuredHere, count);
                if (pd.ms > LIMIT_MS) {
                    log(`   ${fmt(count)} ${name}: not drawn, ~${pd.ms.toFixed(0)} ms predicted from ${fmt(pd.from)}`);
                    row.views[name] = { skipped: true, predicted: pd };
                    continue;
                }
                look.view(name, 0.5);
                const sequence = look.sequence();
                for (let f = 0; f < 3; f++) await timed(sequence);
                const frames = [];
                for (let f = 0; f < 10; f++) {
                    const t = await timed(sequence);
                    if (t.wall > STOP_MS) throw new Error(`a draw took ${t.wall.toFixed(0)} ms, over the ${STOP_MS} ms stop`);
                    frames.push(t);
                }
                const s = summary(frames);
                drawMeasured.set(`${count}/${key}`, s.gpu);
                row.views[name] = { ...s, predicted: pd };
                const rgba = await readTextureRGBA8(device, engine.screenTexture, WIDTH, HEIGHT, engine.ctx.format);
                writeFileSync(`${OUT}draw-${count}-${name}-${gpu}.png`, encodePNG(rgba, WIDTH, HEIGHT));
            }
            row.bytes = done();
            rows.push(row);
            log(`   ${fmt(count).padStart(9)}: step ${step.gpu.toFixed(2)} ms · draw ` +
                Object.entries(row.views).map(([n, v]) => `${n} ${v.skipped ? 'skipped' : `${v.gpu.toFixed(2)} ms (wall ${v.wall.toFixed(1)})`}`).join(', ') +
                ` · ${look.usePalette ? 'palette' : 'hook'} · ${MB(row.bytes)} MB allocated`);
            check(`draw: ${fmt(count)} measured`, Object.values(row.views).some(v => Number.isFinite(v.gpu)));
        } finally {
            sim.destroy();
            look.destroy();
            await device.queue.onSubmittedWorkDone();
        }
    }
    assets.destroy();
    save('draw', { device: info, width: WIDTH, height: HEIGHT, rows });
}

// =============================================================================================
// encode
// =============================================================================================
// ffmpeg 9 tags the stream from the frames, so the tags go in the filter (setparams); output flags alone left
// primaries and transfer "unknown".
const COLOUR = ['-vf', 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv'];
const ENCODERS = {
    none: null,
    'x264-slow': ['-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-tune', 'animation'],
    'x264-medium': ['-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-tune', 'animation'],
    nvenc: ['-c:v', 'h264_nvenc', '-preset', 'p5', '-tune', 'hq', '-rc', 'vbr', '-cq', '19', '-b:v', '0'],
};
function ffmpeg(args, file) {
    const proc = spawn(process.env.NG_FFMPEG ?? 'ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${WIDTH}x${HEIGHT}`, '-r', '60', '-i', '-', ...args, ...COLOUR, file],
    { stdio: ['pipe', 'ignore', 'pipe'] });
    let errors = '';
    proc.stderr.on('data', d => { errors += d; });
    proc.stdin.on('error', () => {});                // an ffmpeg that quit is reported by write()
    const exited = once(proc, 'close');
    const reason = () => errors.trim().split('\n').filter(l => !/Terminating thread|Task finished/.test(l)).slice(0, 2).join(' / ');
    return {
        async write(bytes) {
            if (proc.exitCode !== null || proc.stdin.destroyed) {
                await exited;
                throw new Error(`ffmpeg quit: ${reason()}`);
            }
            try {
                if (!proc.stdin.write(bytes)) await Promise.race([once(proc.stdin, 'drain'), exited]);
            } catch {
                await exited;
                throw new Error(`ffmpeg quit: ${reason()}`);
            }
        },
        async close() {
            proc.stdin.end();
            const [code] = await exited;
            if (code !== 0) throw new Error(`ffmpeg exited ${code}: ${errors.trim().split('\n').slice(-3).join(' / ')}`);
        },
    };
}
async function caseEncode() {
    const count = Number(process.env.NG_SPIKE_COUNT ?? 16_384), FRAMES = Number(process.env.NG_SPIKE_FRAMES ?? 300);
    const { siteModules } = await import('./site.mjs');
    const { loadAssets, buildLook } = await import('./look.mjs');
    const site = await siteModules();
    const assets = loadAssets(engine);
    const rows = [];
    const size = WIDTH * HEIGHT * 4;
    if ((WIDTH * 4) % 256) throw new Error('films spike: rows need padding at this width');
    const staging = [0, 1, 2].map(() => device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
    for (const [name, args] of Object.entries(ENCODERS)) {
        const sim = buildSim(count, 3);
        const look = buildLook(engine, sim, site, assets, { width: WIDTH, height: HEIGHT });
        const chain = stepChain(sim.agents);
        const file = `${OUT}encode-${name}-${gpu}.mp4`;
        try {
            await steps(chain, 600);                                      // the film's 10 s warm-up
            look.view('default', 0);
            const sink = args ? ffmpeg(args, file) : { write: async () => {}, close: async () => {} };
            const wait = { map: 0, write: 0 };
            const inflight = [];
            const drain = async ({ buffer, map }) => {
                let t = performance.now();
                await map;
                const bytes = Buffer.from(new Uint8Array(buffer.getMappedRange()));
                buffer.unmap();
                wait.map += performance.now() - t;
                t = performance.now();
                await sink.write(bytes);
                wait.write += performance.now() - t;
            };
            const t0 = performance.now();
            for (let f = 0; f < FRAMES; f++) {
                engine.render(chain);
                look.view('default', f / 60);
                engine.render(look.sequence());
                const buffer = staging[f % 3];
                const enc = device.createCommandEncoder();
                enc.copyTextureToBuffer({ texture: engine.screenTexture }, { buffer, bytesPerRow: WIDTH * 4 }, [WIDTH, HEIGHT]);
                device.queue.submit([enc.finish()]);
                inflight.push({ buffer, map: buffer.mapAsync(GPUMapMode.READ) });
                if (inflight.length >= 2) await drain(inflight.shift());
            }
            while (inflight.length) await drain(inflight.shift());
            const loop = performance.now() - t0;
            await sink.close();
            const total = performance.now() - t0;
            const bytes = args ? statSync(file).size : 0;
            const row = { encoder: name, count, frames: FRAMES, loopMs: loop, totalMs: total, fps: FRAMES / (total / 1000),
                mapWaitMs: wait.map, writeWaitMs: wait.write, fileBytes: bytes };
            rows.push(row);
            log(`   ${name.padEnd(12)} ${FRAMES} frames in ${(total / 1000).toFixed(1)} s: ${row.fps.toFixed(1)} fps ` +
                `(waiting on readback ${(wait.map / FRAMES).toFixed(1)} ms, on the encoder ${(wait.write / FRAMES).toFixed(1)} ms a frame)` +
                (args ? ` · ${MB(bytes)} MB, ${(bytes * 8 / (FRAMES / 60) / 1e6).toFixed(1)} Mbit/s` : ''));
            check(`encode: ${name} ran`, row.fps > 0);
        } catch (error) {
            // An encoder this machine cannot run (h264_nvenc needs a newer driver than ffmpeg 9 finds) is a finding.
            rows.push({ encoder: name, count, unavailable: error.message });
            log(`   ${name.padEnd(12)} unavailable: ${error.message}`);
        } finally {
            sim.destroy();
            look.destroy();
            await device.queue.onSubmittedWorkDone();
        }
    }
    staging.forEach(b => b.destroy());
    assets.destroy();
    save('encode', { device: info, width: WIDTH, height: HEIGHT, rows });
}

// =============================================================================================
// pack
// =============================================================================================
const PACK_WORDS = 5;               // x, z (f32); velocity xz (f16 × 2); heading, rest timer (f16 × 2); flags (u8)
const packWGSL = count => `${AGENT_STRUCT_WGSL}
@group(0) @binding(0) var<storage, read> agents: array<Agent>;
@group(0) @binding(1) var<storage, read_write> packed: array<u32>;
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= ${count}u) { return; }
    let a = agents[i];
    let o = i * ${PACK_WORDS}u;
    packed[o] = bitcast<u32>(a.posRadius.x);
    packed[o + 1u] = bitcast<u32>(a.posRadius.z);
    packed[o + 2u] = pack2x16float(a.velHeading.xz);
    packed[o + 3u] = pack2x16float(vec2<f32>(a.velHeading.w, a.state.w));
    packed[o + 4u] = a.ids.y & 255u;
}`;
const zstd = (bytes, level) => {
    const t = performance.now();
    const out = zstdCompressSync(bytes, { params: { [zlibConstants.ZSTD_c_compressionLevel]: level } });
    return { ratio: bytes.length / out.length, mbps: bytes.length / 1048576 / ((performance.now() - t) / 1000) };
};
async function casePack() {
    const plan = { 65536: { settle: 600, saves: 10 }, 1048576: { settle: 0, saves: 2 } };
    const rows = [];
    const stepMeasured = new Map();
    for (const count of [16_384, 65_536, 262_144, 1_048_576].filter(c => c <= MAX)) {
        const p = predict(stepMeasured, count);
        if (p.ms > LIMIT_MS) { log(`   ${fmt(count)}: not run, ~${p.ms.toFixed(0)} ms per step predicted`); break; }
        const sim = buildSim(count, 3);
        const chain = stepChain(sim.agents);
        const bytes = count * PACK_WORDS * 4;
        const packed = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const stage = engine.createComputeStage({ name: 'films pack', shaderCode: packWGSL(count), dispatchCount: [Math.ceil(count / 64), 1, 1] });
        engine.setComputePassBindGroup(stage, [sim.agents.buffer, packed].map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
        try {
            const measuredSteps = await steps(chain, 10);
            stepMeasured.set(count, summary(measuredSteps.slice(3)).gpu);
            const { settle = 0, saves = 1 } = plan[count] ?? {};
            if (settle) await steps(chain, settle);
            const packMs = [], readMs = [], saved = [];
            for (let k = 0; k < saves; k++) {
                if (k) await steps(chain, 6);
                packMs.push((await timed([stage])).gpu);
                const t = performance.now();
                const enc = device.createCommandEncoder();
                enc.copyBufferToBuffer(packed, 0, staging, 0, bytes);
                device.queue.submit([enc.finish()]);
                await staging.mapAsync(GPUMapMode.READ);
                saved.push(new Uint8Array(staging.getMappedRange().slice(0)));
                staging.unmap();
                readMs.push(performance.now() - t);
            }
            const row = { count, recordBytes: PACK_WORDS * 4, saveBytes: bytes, packMs: median(packMs), readMs: median(readMs), zstd: {} };
            if (saves > 1) {
                const chunk = Buffer.concat(saved);
                for (const level of [1, 3, 9]) row.zstd[`level ${level}`] = zstd(chunk, level);
                // Each save XORed with the one before: f32 positions six steps apart share their high bits.
                const xored = Buffer.alloc(chunk.length);
                saved.forEach((s, k) => {
                    const prev = k ? saved[k - 1] : null;
                    for (let i = 0; i < s.length; i++) xored[k * bytes + i] = prev ? s[i] ^ prev[i] : s[i];
                });
                row.zstd['level 3, XOR with the save before'] = zstd(xored, 3);
            } else {
                row.zstd['level 3'] = zstd(Buffer.from(saved[0]), 3);
            }
            rows.push(row);
            log(`   ${fmt(count).padStart(9)}: a save of ${MB(bytes)} MB packed in ${row.packMs.toFixed(3)} ms GPU, read back in ${row.readMs.toFixed(1)} ms · zstd ` +
                Object.entries(row.zstd).map(([k, z]) => `${k} ×${z.ratio.toFixed(2)} at ${z.mbps.toFixed(0)} MB/s`).join(', '));
            check(`pack: ${fmt(count)} measured`, row.packMs > 0);
        } finally {
            stage.enabled = false;
            packed.destroy();
            staging.destroy();
            sim.destroy();
            await device.queue.onSubmittedWorkDone();
        }
    }
    save('pack', { device: info, rows });
}

try {
    await ({ steps: caseSteps, draw: caseDraw, encode: caseEncode, pack: casePack }[CASE] ?? (() => { throw new Error(`NG_SPIKE_CASE: no case "${CASE}"`); }))();
} catch (error) {
    log(`   stopped: ${error.message}`);
    check(`${CASE}: ran to the end`, false, error.message);
}
await env.finish(`films ${CASE} done`);
