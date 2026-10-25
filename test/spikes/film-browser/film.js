// Spike S1 of D2.4 (docs/offline-films-browser-design.md §14): the part the page and its worker share.
// The Node film tool's own Kerr scene (tools/film/scenes/kerr-black-hole.mjs) is filmed four ways, each
// frame wrapped in a mediabunny VideoSample and handed to a VideoSampleSource (design §4.3, §4.4):
//   canvas    — the engine presents to the page's visible <canvas> (MetaForge's exporter today);
//   offscreen — to an OffscreenCanvas nobody displays, on the main thread or in a worker;
//   headless  — into the engine's screenTexture, read back through three staging buffers.
// A measurement, not a test. Its results are the design's "Spike S1 results".

import {
    BufferTarget, getFirstEncodableVideoCodec, Mp4OutputFormat, Output, Quality, StreamTarget, VideoSample, VideoSampleSource,
} from 'mediabunny';
import { NullGraph } from '../../../dist/index.js';
import kerrBlackHole from '../../../tools/film/scenes/kerr-black-hole.mjs';
import { PATCHES, PATTERN_WGSL, patchCentre } from './pattern.js';

export const now = () => performance.now();

/** T(j) = round(j · 10⁶ / fps) µs; a frame lasts T(j + 1) − T(j) (design §4.3). */
export const stampUs = (j, fps) => Math.round(j * 1e6 / fps);

const KERR_DT = 1 / 60;
/** The guard's limits (design §4.7): refuse a frame over 1,000 ms, fence every frame above 150 ms. */
const FRAME_LIMIT_MS = 1000;
const PER_FRAME_FENCE_MS = 150;

/** A real task boundary, not a resolved promise: the browser may render, present and run rAF here. */
const channel = new MessageChannel();
export const yieldTask = () => new Promise(resolve => {
    channel.port1.onmessage = () => resolve();
    channel.port2.postMessage(0);
});

export function stats(values) {
    const v = values.filter(Number.isFinite).sort((a, b) => a - b);
    if (!v.length) return null;
    const at = p => v[Math.min(v.length - 1, Math.floor(p * v.length))];
    const sum = v.reduce((a, b) => a + b, 0);
    const r = x => Math.round(x * 100) / 100;
    return { n: v.length, mean: r(sum / v.length), p50: r(at(0.5)), p90: r(at(0.9)), p99: r(at(0.99)), max: r(v[v.length - 1]), sum: r(sum) };
}

export async function upload(name, body) {
    const res = await fetch(`/__s1/upload/${encodeURIComponent(name)}`, { method: 'PUT', body });
    if (!res.ok) throw new Error(`upload of ${name} failed: HTTP ${res.status}`);
}

export function parseCase(params) {
    const get = (k, d) => params.get(k) ?? d;
    const cfg = {
        case: get('case', 'export'), way: get('way', 'offscreen'), id: get('id', 'case'),
        w: Number(get('w', 1920)), h: Number(get('h', 1080)), fps: Number(get('fps', 60)),
        frames: Number(get('frames', 300)), count: Number(get('count', 1048576)),
        target: get('target', 'memory'), yield: get('yield', 'task'), hw: get('hw', 'no-preference'), brmode: get('brmode', ''),
        qp: get('qp', '') === '' ? null : Number(get('qp', '')),
        sample: get('sample', '0,150,299').split(',').filter(Boolean).map(Number),
        rawAt: get('raw', '') === '' ? null : Number(get('raw', '')),
    };
    if (cfg.w % 2 || cfg.h % 2) throw new Error(`width and height must be even for H.264; got ${cfg.w}×${cfg.h}`);
    return cfg;
}

/** Which target the engine draws into for a way; the worker way is an OffscreenCanvas inside the worker. */
export const surfaceFor = way => ({ visible: 'canvas', offscreen: 'offscreen', worker: 'offscreen', readback: 'headless', headless: 'headless' })[way];

export async function requestDevice(log) {
    if (!navigator.gpu) throw new Error('WebGPU is not available');
    // No powerPreference: Chrome on Windows ignores it and says so in the console. run.mjs picks the
    // NVIDIA GPU with --force_high_performance_gpu.
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('no WebGPU adapter');
    const device = await adapter.requestDevice();
    const errors = [];
    device.addEventListener('uncapturederror', e => { errors.push(e.error.message); log(`WebGPU uncaptured error: ${e.error.message}`, 'error'); });
    device.lost.then(info => {
        if (info.reason !== 'destroyed') { errors.push(`device lost: ${info.message}`); log(`WebGPU device lost: ${info.message}`, 'error'); }
    });
    const { vendor, architecture, description, device: id } = adapter.info ?? {};
    return { device, errors, adapter: { vendor, architecture, description, device: id } };
}

export async function runCase(cfg, env) {
    const gpu = await requestDevice(env.log);
    const started = now();
    const run = { alpha: alphaCase, pace: paceCase, export: exportCase, cancel: cancelCase, cpuframes: cpuFramesCase }[cfg.case];
    if (!run) throw new Error(`unknown case ${cfg.case}`);
    const result = await run(cfg, env, gpu);
    await gpu.device.queue.onSubmittedWorkDone();
    return { cfg, surface: env.surfaceKind, adapter: gpu.adapter, gpuErrors: gpu.errors, caseMs: Math.round(now() - started), ...result };
}

// ---------------------------------------------------------------------------------------------------
// Engine, scene and capture
// ---------------------------------------------------------------------------------------------------

async function createEngine(device, kind, width, height, canvas) {
    const engine = new NullGraph();
    let surface = null;
    let init;
    if (kind === 'headless') {
        init = await engine.init(null, { device, width, height });
    } else {
        surface = kind === 'canvas' ? canvas : new OffscreenCanvas(width, height);
        surface.width = width;
        surface.height = height;
        init = await engine.init(surface, { device });
    }
    if (!init.success) throw new Error(`engine init failed: ${init.error}`);
    return { engine, surface };
}

async function buildKerr(engine, width, height, count) {
    const scene = await kerrBlackHole({ engine, width, height, settings: { count }, film: { track: () => null } });
    return {
        // The Node runner awaits each step's own fence. The browser loop fences every 8 frames instead
        // (design §4.7), so the step's fence promise is left to settle by itself.
        step() { scene.step().catch(() => {}); },
        // A slow orbit, so that the encoder sees motion as well as the particles' own.
        view(seconds) { scene.view({ target: [0, 0, 0], distance: 17.5, azimuth: -0.35 + 0.25 * seconds, polar: 1.32 }, seconds); },
        draw: () => scene.draw(),
    };
}

/**
 * Four warm-up steps and draws, each fenced and timed (queue-completion wall ms). The first compiles the
 * pipelines (seconds on a cold NVIDIA D3D12 cache, MetaForge's readiness notes), which is not GPU work,
 * so the guard judges the other three. Every way runs the same warm-up, so their films start alike.
 */
async function warmUp(engine, kerr, device) {
    const ms = [];
    for (let i = 0; i < 4; i++) {
        const t = now();
        kerr.step();
        kerr.view(0);
        engine.render(kerr.draw());
        await device.queue.onSubmittedWorkDone();
        ms.push(Math.round((now() - t) * 100) / 100);
        if (i > 0 && ms[i] > FRAME_LIMIT_MS) {
            throw new Error(`guard: a warm-up frame took ${ms[i].toFixed(0)} ms, over the ${FRAME_LIMIT_MS} ms limit`);
        }
    }
    const steady = Math.max(...ms.slice(1));
    return { ms, fenceEvery: steady > PER_FRAME_FENCE_MS ? 1 : 8 };
}

function canvasFrame(surface, j, fps) {
    const t = stampUs(j, fps);
    return new VideoFrame(surface, { timestamp: t, duration: stampUs(j + 1, fps) - t });
}

/**
 * Source B (design §4.3): each grab copies the texture into one of three staging buffers and starts
 * mapping it; takeFrame() waits for the oldest and builds a VideoFrame straight from the mapped range, so
 * the frame's bytes are copied once on the CPU, by the VideoFrame itself. 'RGBX' ignores the alpha byte.
 */
class FrameReadback {
    constructor(device, texture, width, height) {
        this.device = device;
        this.texture = texture;
        this.width = width;
        this.height = height;
        this.paddedRow = Math.ceil(width * 4 / 256) * 256;
        this.staging = [0, 1, 2].map(() => device.createBuffer({
            label: 's1 readback', size: this.paddedRow * height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        }));
        this.inFlight = [];
        this.next = 0;
    }

    grab(frame) {
        if (this.inFlight.length >= this.staging.length - 1) throw new Error('readback: take a frame before grabbing two more');
        const buffer = this.staging[this.next++ % this.staging.length];
        const encoder = this.device.createCommandEncoder({ label: 's1 readback' });
        encoder.copyTextureToBuffer({ texture: this.texture }, { buffer, bytesPerRow: this.paddedRow }, [this.width, this.height]);
        this.device.queue.submit([encoder.finish()]);
        this.inFlight.push({ frame, buffer, mapped: buffer.mapAsync(GPUMapMode.READ) });
    }

    get pending() { return this.inFlight.length; }

    async takeFrame(fps) {
        const { frame, buffer, mapped } = this.inFlight.shift();
        await mapped;
        try {
            const t = stampUs(frame, fps);
            return new VideoFrame(buffer.getMappedRange(), {
                format: 'RGBX', codedWidth: this.width, codedHeight: this.height,
                layout: [{ offset: 0, stride: this.paddedRow }],
                timestamp: t, duration: stampUs(frame + 1, fps) - t,
            });
        } finally {
            buffer.unmap();
        }
    }

    destroy() { for (const b of this.staging) b.destroy(); }
}

/** A frame's visible pixels as tightly packed RGBA8, whatever 8-bit RGB layout the browser chose. */
async function frameRGBA(frame) {
    const { width, height } = frame.visibleRect;
    const format = frame.format;
    if (!['RGBA', 'RGBX', 'BGRA', 'BGRX'].includes(format)) return { format, width, height, rgba: null };
    const bytes = new Uint8Array(frame.allocationSize());
    const [plane] = await frame.copyTo(bytes);
    const bgr = format.startsWith('B');
    const rgba = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) {
        let s = plane.offset + y * plane.stride;
        let d = y * width * 4;
        for (let x = 0; x < width; x++, s += 4, d += 4) {
            rgba[d] = bytes[s + (bgr ? 2 : 0)];
            rgba[d + 1] = bytes[s + 1];
            rgba[d + 2] = bytes[s + (bgr ? 0 : 2)];
            rgba[d + 3] = bytes[s + 3];
        }
    }
    return { format, width, height, rgba };
}

/** SHA-256 of the RGB bytes (alpha left out: the encoder discards it) and how many pixels had alpha < 255. */
async function rgbDigest(rgba) {
    const rgb = new Uint8Array(rgba.length / 4 * 3);
    let translucent = 0;
    for (let s = 0, d = 0; s < rgba.length; s += 4, d += 3) {
        rgb[d] = rgba[s]; rgb[d + 1] = rgba[s + 1]; rgb[d + 2] = rgba[s + 2];
        if (rgba[s + 3] !== 255) translucent++;
    }
    const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', rgb));
    return { sha256: [...digest].map(b => b.toString(16).padStart(2, '0')).join(''), translucent };
}

// ---------------------------------------------------------------------------------------------------
// The video sink (design §4.4, §4.5): VideoSampleSource, sample closed in finally
// ---------------------------------------------------------------------------------------------------

async function createVideoSink({ width, height, fps, frames, target, writable, chunkSize, hardwareAcceleration = 'no-preference', bitrateMode, quantizer }) {
    const bitrate = Math.round(width * height * fps * 0.16); // MetaForge's measured rule (design §4.4)
    // A quantizer asks for constant-QP encoding first, with the bitrate as mediabunny's fallback.
    const quality = new Quality({ bitrate, ...(bitrateMode ? { bitrateMode } : {}), ...(quantizer != null ? { quantizer } : {}) });
    // mediabunny's probe takes no hardware preference (1.61.3); the sink's own config does.
    const codec = await getFirstEncodableVideoCodec(['avc', 'vp9', 'av1'], { width, height, quality, frameRate: fps });
    if (!codec) throw new Error(`no WebCodecs encoder for ${width}×${height} (avc, vp9, av1; ${hardwareAcceleration})`);
    const configs = [];
    const sink = { codec, bitrate, configs, packets: 0, keyPackets: 0, bytes: 0 };
    const source = new VideoSampleSource({
        codec, quality, keyFrameInterval: 2, latencyMode: 'quality', alpha: 'discard', hardwareAcceleration,
        // Called once per candidate before mediabunny tries it; the last one is the one it kept.
        onEncoderConfig: c => configs.push(c),
        onEncodedPacket: p => { sink.packets++; sink.bytes += p.byteLength; if (p.type === 'key') sink.keyPackets++; },
    });
    const output = new Output({
        format: new Mp4OutputFormat({ fastStart: target === 'memory' ? 'in-memory' : 'reserve' }),
        target: target === 'memory' ? new BufferTarget() : new StreamTarget(writable, { chunked: true, ...(chunkSize ? { chunkSize } : {}) }),
    });
    output.addVideoTrack(source, { frameRate: fps, maximumPacketCount: frames });
    await output.start();
    /** Takes ownership of the frame: it is closed whether the add succeeds or fails (design §4.6). */
    sink.add = async frame => {
        let sample;
        try { sample = new VideoSample(frame); } catch (e) { frame.close(); throw e; }
        try { await source.add(sample); } finally { sample.close(); }
    };
    sink.finish = async () => {
        await output.finalize();
        return target === 'memory' ? new Blob([output.target.buffer], { type: 'video/mp4' }) : null;
    };
    sink.cancel = () => output.cancel();
    return sink;
}

/** What the encoder was given, and whether this browser has hardware and software encoders for it. */
async function describeSink(sink, fps) {
    const used = sink.configs.at(-1);
    const config = used && JSON.parse(JSON.stringify(used));
    const support = {};
    if (config) {
        for (const hardwareAcceleration of ['prefer-hardware', 'prefer-software']) {
            try {
                support[hardwareAcceleration] = (await VideoEncoder.isConfigSupported({ ...config, framerate: fps, hardwareAcceleration })).supported;
            } catch (e) { support[hardwareAcceleration] = `error: ${e.message}`; }
        }
    }
    return { codec: sink.codec, bitrate: sink.bitrate, config, candidates: sink.configs.length, support, packets: sink.packets, keyPackets: sink.keyPackets, encodedBytes: sink.bytes };
}

// ---------------------------------------------------------------------------------------------------
// Case "export": 300 frames of the full Kerr scene into an MP4, in memory or in OPFS
// ---------------------------------------------------------------------------------------------------

async function exportCase(cfg, env, gpu) {
    const { w: width, h: height, fps, frames } = cfg;
    const { device } = gpu;
    const { engine, surface } = await createEngine(device, env.surfaceKind, width, height, env.canvas);
    const kerr = await buildKerr(engine, width, height, cfg.count);
    const guard = await warmUp(engine, kerr, device);
    const stepsPerFrame = Math.round(1 / fps / KERR_DT);
    const readback = env.surfaceKind === 'headless' ? new FrameReadback(device, engine.screenTexture, width, height) : null;
    // mapAsync already waits for each read-back frame, so only the canvas ways fence (every 8 frames,
    // or every frame when the warm-up measured over 150 ms: design §4.7).
    const fenceEvery = readback ? 0 : guard.fenceEvery;

    let opfs = null;
    if (cfg.target === 'opfs') {
        const root = await navigator.storage.getDirectory();
        const films = await root.getDirectoryHandle('null-graph-films', { create: true });
        const exportId = crypto.randomUUID();
        const dir = await films.getDirectoryHandle(exportId, { create: true });
        const handle = await dir.getFileHandle('film.mp4', { create: true });
        opfs = { films, exportId, handle, writable: await handle.createWritable(), before: await navigator.storage.estimate() };
    }
    const sink = await createVideoSink({
        width, height, fps, frames, target: cfg.target, writable: opfs?.writable, hardwareAcceleration: cfg.hw, bitrateMode: cfg.brmode || undefined,
        quantizer: cfg.qp,
    });

    const cols = { step: [], render: [], capture: [], sample: [], add: [], fence: [], yield: [], total: [] };
    const heap = [];
    const samples = [];
    const sampled = new Set(cfg.sample);
    // Hands a frame to the sink; sampled frames are copied back and hashed first. That time is kept apart
    // from the add's, and the marks let the page leave it out of its main-thread figures.
    const handOff = async frame => {
        const j = Math.round(frame.timestamp * fps / 1e6);
        let sampleMs = 0;
        if (sampled.has(j)) {
            env.mark('sample-start');
            const t = now();
            const { format, rgba } = await frameRGBA(frame);
            const entry = { frame: j, format, colorSpace: frame.colorSpace?.toJSON?.() ?? null };
            if (rgba) {
                Object.assign(entry, await rgbDigest(rgba));
                if (cfg.rawAt === j) { await env.upload(`${cfg.id}-f${j}.rgba`, rgba); entry.raw = `${cfg.id}-f${j}.rgba`; }
            }
            samples.push(entry);
            sampleMs = now() - t;
            env.mark('sample-end');
        }
        const t = now();
        await sink.add(frame);
        return { sampleMs, addMs: now() - t };
    };

    await env.measure?.('built');
    env.mark('export-start');
    const t0 = now();
    for (let j = 0; j < frames; j++) {
        const a = now();
        for (let s = 0; s < stepsPerFrame; s++) kerr.step();
        const b = now();
        kerr.view(j / fps);
        engine.render(kerr.draw());
        const c = now();
        // Capture in the same task as the render, before any await (design §4.3).
        let frame = null;
        if (readback) {
            readback.grab(j);
            if (readback.pending > 1) frame = await readback.takeFrame(fps);
        } else {
            frame = canvasFrame(surface, j, fps);
        }
        const d = now();
        const { sampleMs, addMs } = frame ? await handOff(frame) : { sampleMs: 0, addMs: 0 };
        const e = now();
        if (fenceEvery && (j + 1) % fenceEvery === 0) await device.queue.onSubmittedWorkDone();
        const f = now();
        if (cfg.yield !== 'none') await yieldTask();
        const g = now();
        cols.step.push(b - a); cols.render.push(c - b); cols.capture.push(d - c); cols.sample.push(sampleMs);
        cols.add.push(addMs); cols.fence.push(f - e); cols.yield.push(g - f); cols.total.push(g - a - sampleMs);
        if (j % 8 === 0 && globalThis.performance?.memory) heap.push(performance.memory.usedJSHeapSize);
    }
    while (readback?.pending) await handOff(await readback.takeFrame(fps));
    const framesMs = now() - t0 - cols.sample.reduce((x, y) => x + y, 0);
    env.mark('frames-done');
    await env.measure?.('frames-done');
    const tf = now();
    const blob = await sink.finish();
    const finalizeMs = now() - tf;
    env.mark('export-end');

    let file;
    if (opfs) {
        file = await opfs.handle.getFile();
        const after = await navigator.storage.estimate();
        await env.measure?.('finished');
        await env.upload(`${cfg.id}.mp4`, file);
        await opfs.films.removeEntry(opfs.exportId, { recursive: true });
        opfs = { bytes: file.size, quotaBefore: opfs.before.quota, usageBefore: opfs.before.usage, usageAfter: after.usage };
    } else {
        await env.measure?.('finished');
        await env.upload(`${cfg.id}.mp4`, blob);
    }
    readback?.destroy();

    const per = Object.fromEntries(Object.entries(cols).map(([k, v]) => [k, stats(v)]));
    return {
        guard, stepsPerFrame, fenceEvery, yield: cfg.yield, target: cfg.target,
        canvasFormat: env.surfaceKind === 'headless' ? engine.ctx.format : navigator.gpu.getPreferredCanvasFormat(),
        framesMs: Math.round(framesMs), msPerFrame: Math.round(framesMs / frames * 100) / 100, finalizeMs: Math.round(finalizeMs),
        bytes: (blob ?? file).size, file: `${cfg.id}.mp4`, opfs, per,
        raw: Object.fromEntries(Object.entries(cols).map(([k, v]) => [k, v.map(x => Math.round(x * 100) / 100)])),
        heapPeak: heap.length ? Math.max(...heap) : null, heapFirst: heap[0] ?? null,
        samples, sink: await describeSink(sink, fps),
    };
}

// ---------------------------------------------------------------------------------------------------
// Case "pace": is a way held to the display's rhythm? A light scene, a task between frames, no encoder.
// ---------------------------------------------------------------------------------------------------

async function paceCase(cfg, env, gpu) {
    const { w: width, h: height, fps, frames } = cfg;
    const { device } = gpu;
    const { engine, surface } = await createEngine(device, env.surfaceKind, width, height, env.canvas);
    const kerr = await buildKerr(engine, width, height, cfg.count);
    const guard = await warmUp(engine, kerr, device);
    const readback = cfg.way === 'readback' ? new FrameReadback(device, engine.screenTexture, width, height) : null;
    const ms = [];
    const capture = [];
    const WARM = 10;
    for (let j = 0; j < WARM + frames; j++) {
        if (j === WARM) env.mark('pace-start');
        const a = now();
        kerr.step();
        kerr.view(j / fps);
        engine.render(kerr.draw());
        const b = now();
        if (readback) {
            readback.grab(j);
            if (readback.pending > 1) (await readback.takeFrame(fps)).close();
        } else if (surface) {
            canvasFrame(surface, j, fps).close();
        }
        const c = now();
        if ((j + 1) % 8 === 0 && !readback) await device.queue.onSubmittedWorkDone();
        await yieldTask();
        if (j >= WARM) { ms.push(now() - a); capture.push(c - b); }
    }
    while (readback?.pending) (await readback.takeFrame(fps)).close();
    await device.queue.onSubmittedWorkDone();
    env.mark('pace-end');
    const fence = await fenceLatency(engine, kerr, device);
    readback?.destroy();
    return {
        guard, frame: stats(ms), capture: stats(capture), raw: ms.map(x => Math.round(x * 100) / 100),
        over15ms: ms.filter(x => x >= 15).length, fence,
    };
}

/**
 * How long Chrome takes to report finished work: onSubmittedWorkDone against a 4-byte mapAsync queued
 * behind the same frame. The guard (design §4.7) times queue completion with the first.
 */
async function fenceLatency(engine, kerr, device) {
    const map = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const src = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_SRC });
    const done = [];
    const mapped = [];
    for (let i = 0; i < 20; i++) {
        kerr.view(0);
        engine.render(kerr.draw());
        let t = now();
        await device.queue.onSubmittedWorkDone();
        done.push(now() - t);
        kerr.view(0);
        engine.render(kerr.draw());
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(src, 0, map, 0, 4);
        device.queue.submit([encoder.finish()]);
        t = now();
        await map.mapAsync(GPUMapMode.READ);
        mapped.push(now() - t);
        map.unmap();
    }
    map.destroy();
    src.destroy();
    return { onSubmittedWorkDone: stats(done), mapAsync: stats(mapped) };
}

// ---------------------------------------------------------------------------------------------------
// Case "alpha": what each way does with alpha below 1 (design §4.4 "Colour", Q1), on a fixed pattern
// ---------------------------------------------------------------------------------------------------

async function alphaCase(cfg, env, gpu) {
    const { w: width, h: height, fps, frames } = cfg;
    const { device } = gpu;
    let surface = null;
    let context = null;
    let texture = null;
    let format;
    if (env.surfaceKind === 'headless') {
        format = 'rgba8unorm';
        texture = device.createTexture({ size: [width, height], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    } else {
        surface = env.surfaceKind === 'canvas' ? env.canvas : new OffscreenCanvas(width, height);
        surface.width = width;
        surface.height = height;
        context = surface.getContext('webgpu');
        format = navigator.gpu.getPreferredCanvasFormat();
        context.configure({ device, format, alphaMode: 'premultiplied' }); // as WebGPUContext.init configures it
    }
    const module = device.createShaderModule({ code: PATTERN_WGSL });
    const pipeline = device.createRenderPipeline({
        layout: 'auto', vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    });
    const dims = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(dims, 0, new Float32Array([width, height, 0, 0]));
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: dims } }] });
    const draw = () => {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({ colorAttachments: [{
            view: (texture ?? context.getCurrentTexture()).createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0],
        }] });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.draw(3);
        pass.end();
        device.queue.submit([encoder.finish()]);
    };
    const readback = texture ? new FrameReadback(device, texture, width, height) : null;
    const sink = await createVideoSink({ width, height, fps, frames, target: 'memory' });
    let captured = null;
    for (let j = 0; j < frames; j++) {
        draw();
        let frame;
        if (readback) {
            readback.grab(j);
            frame = await readback.takeFrame(fps);
        } else {
            frame = canvasFrame(surface, j, fps);
        }
        if (j === 0) {
            const { format: frameFormat, rgba } = await frameRGBA(frame);
            captured = {
                format: frameFormat, colorSpace: frame.colorSpace?.toJSON?.() ?? null,
                patches: rgba && PATCHES.map((p, i) => {
                    const [x, y] = patchCentre(i, width, height);
                    const at = (y * width + x) * 4;
                    return { name: p.name, expected: p.rgba, got: [...rgba.subarray(at, at + 4)] };
                }),
            };
        }
        await sink.add(frame);
        await yieldTask();
    }
    const blob = await sink.finish();
    await env.upload(`${cfg.id}.mp4`, blob);
    readback?.destroy();
    texture?.destroy();
    return { canvasFormat: format, captured, bytes: blob.size, file: `${cfg.id}.mp4`, sink: await describeSink(sink, fps) };
}

// ---------------------------------------------------------------------------------------------------
// Case "cpuframes": the same frames, made in JavaScript, encoded on each GPU's Chrome. Equal encoded
// payloads from the AMD and the NVIDIA runs mean one encoder serves both, whatever WebGPU's adapter.
// ---------------------------------------------------------------------------------------------------

/** A gradient, a moving square and seeded noise in the lower third: the same bytes on every machine. */
function synthFrame(rgba, width, height, j) {
    let seed = (Math.imul(j + 1, 2654435761)) >>> 0;
    const square = Math.floor(height / 6);
    const sx = Math.floor((j * 7) % (width - square));
    const sy = Math.floor(height / 6);
    for (let y = 0; y < height; y++) {
        const noise = y > (height * 2) / 3;
        for (let x = 0, i = y * width * 4; x < width; x++, i += 4) {
            if (noise) {
                seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
                const n = seed >>> 24;
                rgba[i] = n; rgba[i + 1] = n; rgba[i + 2] = n;
            } else {
                const inSquare = x >= sx && x < sx + square && y >= sy && y < sy + square;
                rgba[i] = inSquare ? 255 : (x * 255 / width) | 0;
                rgba[i + 1] = inSquare ? 255 : (y * 255 / height) | 0;
                rgba[i + 2] = inSquare ? 255 : (j * 4) & 255;
            }
            rgba[i + 3] = 255;
        }
    }
}

async function cpuFramesCase(cfg, env) {
    const { w: width, h: height, fps, frames } = cfg;
    const sink = await createVideoSink({ width, height, fps, frames, target: 'memory', hardwareAcceleration: cfg.hw, bitrateMode: cfg.brmode || undefined });
    const rgba = new Uint8Array(width * height * 4);
    const t0 = now();
    let encodeMs = 0;
    for (let j = 0; j < frames; j++) {
        synthFrame(rgba, width, height, j);
        const t = stampUs(j, fps);
        const e = now();
        await sink.add(new VideoFrame(rgba, { format: 'RGBX', codedWidth: width, codedHeight: height, timestamp: t, duration: stampUs(j + 1, fps) - t }));
        encodeMs += now() - e;
        if (j % 4 === 3) await yieldTask();
    }
    const blob = await sink.finish();
    await env.upload(`${cfg.id}.mp4`, blob);
    return { bytes: blob.size, file: `${cfg.id}.mp4`, wallMs: Math.round(now() - t0), addMs: Math.round(encodeMs), sink: await describeSink(sink, fps) };
}

// ---------------------------------------------------------------------------------------------------
// Case "cancel": does output.cancel() over an OPFS writable commit a partial file (design §4.5)?
// ---------------------------------------------------------------------------------------------------

/**
 * mediabunny 1.61.3's StreamTarget closes its writer on Output.cancel(), and closing a
 * FileSystemWritableFileStream commits it. The design's remedy: hand mediabunny a stream of our own that
 * aborts the file's writable instead, once the session has decided to discard.
 */
function interpose(fileWritable) {
    const writer = fileWritable.getWriter();
    let discard = false;
    const stream = new WritableStream({
        write: chunk => writer.write(chunk),
        close: () => (discard ? writer.abort(new DOMException('Export discarded', 'AbortError')) : writer.close()),
        abort: reason => writer.abort(reason),
    });
    return { stream, discard() { discard = true; } };
}

async function cancelCase(cfg, env, gpu) {
    const { w: width, h: height, fps, frames } = cfg;
    const { engine, surface } = await createEngine(gpu.device, env.surfaceKind, width, height, env.canvas);
    const kerr = await buildKerr(engine, width, height, cfg.count);
    await warmUp(engine, kerr, gpu.device);
    const root = await navigator.storage.getDirectory();
    const films = await root.getDirectoryHandle('null-graph-films', { create: true });
    const variants = [];
    for (const variant of ['direct', 'interposed']) {
        const exportId = crypto.randomUUID();
        const dir = await films.getDirectoryHandle(exportId, { create: true });
        const handle = await dir.getFileHandle('film.mp4', { create: true });
        const writable = await handle.createWritable();
        const gate = variant === 'interposed' ? interpose(writable) : null;
        // 1 MiB chunks so that data reaches the file before the cancel at 1080p.
        const sink = await createVideoSink({ width, height, fps, frames, target: 'opfs', writable: gate?.stream ?? writable, chunkSize: 1 << 20 });
        const added = Math.floor(frames / 2);
        for (let j = 0; j < added; j++) {
            kerr.step();
            kerr.view(j / fps);
            engine.render(kerr.draw());
            await sink.add(canvasFrame(surface, j, fps));
        }
        gate?.discard();
        let cancelError = null;
        try { await sink.cancel(); } catch (e) { cancelError = String(e?.message ?? e); }
        const size = (await handle.getFile()).size;
        const entries = [];
        for await (const name of dir.keys()) entries.push(name);
        variants.push({ variant, framesAdded: added, encodedBytes: sink.bytes, sizeAfterCancel: size, entries, cancelError });
        await films.removeEntry(exportId, { recursive: true });
    }
    return { variants };
}
