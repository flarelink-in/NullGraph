// F2's browser G tests (docs/offline-films-browser-design.md §10, §14 F2's exit): the cases, one per page load.
// Each films through the managed runner, recordFilm, exactly as an app would: on the app's own device, with
// the library's dist/film.js. A case checks what the page alone can see (hashes against the pattern's exact
// bytes, error messages, what was left on the device) and uploads its films for run.mjs to read back with
// ffprobe and ffmpeg (frame counts, timestamps, colour, the bar code, canvas against readback).
//
//   G1  P1: the same film twice hashes the same, frame for frame (the pattern exactly, Kerr against itself);
//       nothing is left on the device and no engine is kept alive.
//   G2  P4: the pattern's readback bytes are exact; through each source and codec its decoded colours are
//       within the codec's loss; the additive glow survives as opaque on both sources.
//   G3  J frames, timestamps T(j) within one container tick, key frames every 2 s, moov before mdat, and
//       frame k of the file showing frame k; and Kerr at 4K.
//   G5  the guard on a real device with injected wall time: refused by calibration, stopped by one slow
//       operation, not stopped by a slow 8-frame batch, switched to fencing every operation by a congested one.
//   G6  canvas against readback on Kerr: the bytes each hands the encoder, and the decoded films.
//   G9  cancellation in every phase, the page's main thread during a film, the preview paused and restored,
//       hidden and frozen pages, and a lost device (destroyed, or its GPU process crashed) with the app's
//       recovery.

import { recordFilm, sha256, snapshot } from 'null-graph/film';
import { NullGraph } from 'null-graph';
import { patternFrame } from './pattern.js';
import { capturingMediabunny, compareRGB, kerrWorkload, orbit, patternWorkload, sleep, trackDevice } from './workloads.js';

const range = n => Array.from({ length: n }, (_, i) => i);
const ms = t => Math.round(t * 10) / 10;

/** The device, its tracker, and a check that a film left nothing behind and kept nothing alive. */
async function setup(env) {
    const gpu = await env.requestDevice();
    const tracker = trackDevice(gpu.device);
    const uncaptured = [];
    gpu.device.addEventListener('uncapturederror', e => uncaptured.push(e.error.message));
    const baseline = tracker.live();
    /** Live objects back to the baseline, and every engine the workload was given collectable. */
    const settled = async (label, engines = []) => {
        await gpu.device.queue.onSubmittedWorkDone();
        const now = tracker.live();
        env.check(`${label}: nothing left on the device`, now.count === baseline.count, { before: baseline.count, after: now.count, alive: now.labels.slice(0, 12) });
        if (engines.length && globalThis.gc) {
            // A WeakRef keeps its target alive until the job that made it ends: collect in later tasks.
            for (let i = 0; i < 3; i++) { await sleep(20); env.gc(); }
            const kept = engines.filter(ref => ref.deref() !== undefined).length;
            env.check(`${label}: no film engine kept alive`, kept === 0, `${kept} of ${engines.length} engines still reachable after GC`);
        }
    };
    return { ...gpu, tracker, uncaptured, settled };
}

/** recordFilm, and what came of it, whether a film was made or not. */
async function film(options) {
    const reports = [];
    const t = performance.now();
    try {
        const result = await recordFilm({ ...options, onReport: r => { reports.push(r); options.onReport?.(r); } });
        return { result, error: null, report: reports[0] ?? null, ms: ms(performance.now() - t) };
    } catch (error) {
        return { result: null, error, report: reports[0] ?? null, ms: ms(performance.now() - t) };
    }
}

const describe = run => (run.error ? `${run.error.name}: ${run.error.message}` : run.result?.status);
const summary = run => run.report && {
    status: run.report.status, source: run.report.source, sourceReason: run.report.sourceReason, codec: run.report.output?.codec ?? null,
    encoderConfig: run.report.encoderConfig, frames: run.report.frames, bytes: run.report.bytes, achievedBitrate: run.report.achievedBitrate,
    timings: run.report.timings, guard: run.report.guard, gpuErrors: run.report.gpuErrors, deviceLost: run.report.deviceLost,
    failure: run.report.failure, cleanupErrors: run.report.cleanupErrors, wallMs: run.ms,
};

/** Upload a made film for run.mjs, with what it should hold. */
async function keep(env, id, run, expected) {
    if (!run.result?.file) return null;
    const ext = run.result.file.type === 'video/webm' ? 'webm' : 'mp4';
    const file = `${id}.${ext}`;
    await env.upload(file, run.result.file);
    return { file, ...expected, codec: run.result.codec, source: run.report?.source, encoderConfig: run.report?.encoderConfig };
}

// ---------------------------------------------------------------------------------------------------
// G1: P1, the same film twice; nothing left behind
// ---------------------------------------------------------------------------------------------------

async function g1Pattern(env) {
    const s = await setup(env);
    // 1366 wide: 5,464-byte rows, padded to 5,632 by copyTextureToBuffer (B4's widths, on a real device).
    const W = 1366, H = 768, fps = 30, J = 30;
    const work = patternWorkload({ dt: 1 / fps });
    const options = { device: s.device, createWorkload: work.create, film: { width: W, height: H, fps, seconds: J / fps }, output: false, hashes: true };
    const a = await film(options);
    env.check('the pattern films', a.result?.status === 'completed', describe(a));
    const want = await Promise.all(range(J).map(j => sha256(patternFrame(W, H, j))));
    const wrong = (a.report?.hashes ?? []).map((h, j) => (h === want[j] ? null : j)).filter(j => j !== null);
    env.check('G2: every read-back frame is exactly the pattern (bytes, orientation, channels, bar code)',
        a.report?.hashes?.length === J && wrong.length === 0, { frames: a.report?.hashes?.length, wrong: wrong.slice(0, 8) });
    await s.settled('first film', work.log.engines);
    const b = await film(options);
    env.check('G1: a second film hashes the same, frame for frame (P1)', JSON.stringify(b.report?.hashes) === JSON.stringify(a.report?.hashes), describe(b));
    env.check('the stateless workload was built once and reset, not rebuilt', work.log.builds === 2 && work.log.disposals === 2, work.log);
    await s.settled('second film', work.log.engines);
    // A snapshot at an odd size: rows of 1,332 bytes padded to 1,536, and a PNG with no yuv420p rule.
    const shot = await snapshot({ device: s.device, createWorkload: work.create, width: 333, height: 187 });
    const expected = patternFrame(333, 187, 0);
    env.check('snapshot: the frame\'s bytes are the pattern exactly, at 333×187', compareRGB(shot.rgba, expected).differing === 0 && shot.rgba.every((v, i) => v === expected[i]));
    await env.upload('G1-pattern-snapshot.png', shot.png);
    await s.settled('snapshot', work.log.engines);
    env.check('no uncaptured WebGPU error', s.uncaptured.length === 0, s.uncaptured);
    return { reports: [summary(a), summary(b)], png: { file: 'G1-pattern-snapshot.png', width: 333, height: 187, frame: 0 } };
}

async function g1Kerr(env) {
    const s = await setup(env);
    const work = kerrWorkload();
    const options = {
        device: s.device, createWorkload: work.create, camera: orbit, output: false, hashes: true,
        film: { width: 1920, height: 1080, fps: 60, seconds: 1, warmupSeconds: 0.25 },
    };
    const runs = [];
    for (let i = 0; i < 3; i++) {
        const run = await film(options);
        runs.push(run);
        env.check(`Kerr film ${i + 1} completes`, run.result?.status === 'completed', describe(run));
        await s.settled(`Kerr film ${i + 1}`, work.log.engines);
    }
    const [a, ...rest] = runs.map(r => JSON.stringify(r.report?.hashes ?? null));
    env.check('G1: three Kerr films hash the same, frame for frame (P1)', rest.every(h => h === a) && runs[0].report?.hashes?.length === 60);
    const distinct = new Set(runs[0].report?.hashes ?? []).size;
    env.check('the frames differ from one another (the scene moves)', distinct === 60, `${distinct} distinct hashes`);
    env.check('Kerr has no reset(): rebuilt after calibration, each film', work.log.builds === 6 && work.log.disposals === 6, work.log);
    env.check('plan: 60 frames, 1 step each, 15 warm-up steps', JSON.stringify(runs[0].report?.plan) === JSON.stringify({ frames: 60, stepsPerFrame: 1, warmupSteps: 15, dt: 1 / 60 }), runs[0].report?.plan);
    env.check('no uncaptured WebGPU error', s.uncaptured.length === 0, s.uncaptured);
    return { reports: runs.map(summary) };
}

// ---------------------------------------------------------------------------------------------------
// G2 and G3: the pattern through each source and codec, and the timeline
// ---------------------------------------------------------------------------------------------------

/** Films of the pattern, uploaded for run.mjs: `[id, source, container, width, height, fps, seconds]`. */
async function patternFilms(env, list) {
    const s = await setup(env);
    const videos = [];
    const reports = [];
    for (const [id, source, container, width, height, fps, seconds] of list) {
        const work = patternWorkload({ dt: 1 / fps });
        const run = await film({
            device: s.device, createWorkload: work.create, source, film: { width, height, fps, seconds },
            output: { container, ...(container === 'webm' ? { codecs: ['vp9'] } : {}), name: `${id}.${container}` },
        });
        env.check(`${id}: films`, run.result?.status === 'completed', describe(run));
        env.check(`${id}: from the ${source} source`, run.report?.source === source, run.report?.sourceReason);
        const J = Math.round(fps * seconds);
        env.check(`${id}: ${J} frames accepted and encoded`, run.report?.frames.accepted === J && run.report?.frames.encoded === J, run.report?.frames);
        reports.push({ id, ...summary(run) });
        const video = await keep(env, id, run, { id, pattern: true, width, height, fps, frames: J, container });
        if (video) videos.push(video);
        await s.settled(id, work.log.engines);
    }
    env.check('no uncaptured WebGPU error', s.uncaptured.length === 0, s.uncaptured);
    return { videos, reports };
}

const g2 = env => patternFilms(env, [
    ['G2-canvas-avc', 'canvas', 'mp4', 1280, 720, 30, 1],
    ['G2-readback-avc', 'readback', 'mp4', 1280, 720, 30, 1],
    ['G2-canvas-vp9', 'canvas', 'webm', 1280, 720, 30, 1],
    ['G2-readback-vp9', 'readback', 'webm', 1280, 720, 30, 1],
]);

const g3 = env => patternFilms(env, [
    ['G3-canvas-24fps', 'canvas', 'mp4', 640, 360, 24, 3],
    ['G3-canvas-60fps', 'canvas', 'mp4', 640, 360, 60, 3],
    ['G3-readback-60fps', 'readback', 'mp4', 640, 360, 60, 3],
    ['G3-readback-25fps', 'readback', 'mp4', 640, 360, 25, 3],
    ['G3-canvas-30fps-vp9', 'canvas', 'webm', 640, 360, 30, 3],
]);

/** Kerr at 4K through the managed runner, into memory: the file, the guard's fencing, the encoder's start-up. */
async function g3Kerr4k(env) {
    const s = await setup(env);
    const work = kerrWorkload();
    const longTasks = [];
    const t0 = performance.now();
    const observer = new PerformanceObserver(list => longTasks.push(...list.getEntries().map(e => ({ at: ms(e.startTime - t0), ms: ms(e.duration) }))));
    observer.observe({ type: 'longtask' });
    const phases = [];
    const run = await film({
        device: s.device, createWorkload: work.create, camera: orbit, film: { width: 3840, height: 2160, fps: 60, seconds: 2 },
        onProgress: p => { if (phases.at(-1)?.phase !== p.phase) phases.push({ phase: p.phase, at: ms(performance.now() - t0) }); },
    });
    observer.disconnect();
    env.check('Kerr 4K films', run.result?.status === 'completed', describe(run));
    const video = await keep(env, 'G3-kerr-4k', run, { id: 'G3-kerr-4k', width: 3840, height: 2160, fps: 60, frames: 120, container: 'mp4' });
    await s.settled('Kerr 4K', work.log.engines);
    env.check('no uncaptured WebGPU error', s.uncaptured.length === 0, s.uncaptured);
    return { videos: video ? [video] : [], reports: [summary(run)], longTasks, phases };
}

// ---------------------------------------------------------------------------------------------------
// G5: the guard, with injected wall time (never GPU work, never a reset)
// ---------------------------------------------------------------------------------------------------

async function g5(env) {
    const s = await setup(env);
    const size = { width: 320, height: 180 };
    const runs = {};
    const guarded = async (key, { dt, fps, frames, stepMs, guard }) => {
        const work = patternWorkload({ dt, stepMs });
        const run = await film({ device: s.device, createWorkload: work.create, output: false, guard, film: { ...size, fps, seconds: frames / fps } });
        runs[key] = summary(run);
        await s.settled(key, work.log.engines);
        return run;
    };
    // Calibration measures steps 2–4 (step 1 compiles, recorded but not judged); 60 ms against a 50 ms limit.
    const refused = await guarded('refused', { dt: 1 / 30, fps: 30, frames: 6, stepMs: () => 60, guard: { limitMs: 50 } });
    env.check('a step over the lowered limit is refused by calibration, in §8\'s words',
        /^film: a step at 320×180 took \d+ ms, over the 50 ms per-submission limit \(Windows resets a GPU busy for ~2 s\)\. Try a smaller size, fewer particles, or a faster GPU\.$/.test(refused.error?.message ?? ''),
        describe(refused));
    env.check('refused: nothing was filmed', refused.report?.status === 'failed' && refused.report.frames.accepted === 0, refused.report?.frames);
    // 160 ms a step calibrates above 150 ms a frame (and ≥ 300 / 4 = 75 ms): every operation is fenced, and step 8 (frame 3) stops at once.
    const stopped = await guarded('stopped', { dt: 1 / 10, fps: 10, frames: 10, stepMs: n => (n === 8 ? 350 : 160), guard: { limitMs: 200, stopMs: 300 } });
    env.check('one operation over the lowered stop (calibrated ≥ stop / 4) ends the film at once, in §8\'s words',
        /^film: submission at frame 3 took \d+ ms, over the 300 ms safety stop \(a GPU stalled by other work can cause this too; try again on an idle GPU\)\.$/.test(stopped.error?.message ?? ''),
        describe(stopped));
    env.check('stopped: fenced every operation from calibration, worst step measured', stopped.report?.guard.fencing === 'every operation' && stopped.report.guard.worstStepMs >= 350, stopped.report?.guard);
    // Q10: when calibrated under a quarter of the stop (160 ms < 800 / 4 = 200 ms), one isolated stall (850 ms at frame 3)
    // is measured once more and survives when frame 4 returns to 160 ms; a second stall at frame 4 stops the film.
    const outlier = await guarded('outlier', { dt: 1 / 10, fps: 10, frames: 6, stepMs: n => (n === 8 ? 850 : 160), guard: { limitMs: 200, stopMs: 800 } });
    env.check('Q10: one outlier over the stop (calibrated < stop / 4) is measured once more and completes',
        outlier.result?.status === 'completed' && outlier.report?.guard.worstStepMs >= 850, { outcome: describe(outlier), guard: outlier.report?.guard });
    const sustained = await guarded('sustained', { dt: 1 / 10, fps: 10, frames: 6, stepMs: n => (n >= 8 ? 850 : 160), guard: { limitMs: 200, stopMs: 800 } });
    env.check('Q10: a second operation over the stop after the outlier ends the film at frame 4',
        /^film: submission at frame 4 took \d+ ms, over the 800 ms safety stop/.test(sustained.error?.message ?? ''), describe(sustained));
    // 40 ms steps after a fast calibration: an 8-frame batch of ~320 ms passes the 250 ms stop, but no operation does.
    const batch = await guarded('batch', { dt: 1 / 30, fps: 30, frames: 16, stepMs: n => (n <= 4 ? 0 : 40), guard: { stopMs: 250 } });
    env.check('an 8-frame batch slower than the stop, each operation under it, does not stop the film', batch.result?.status === 'completed', describe(batch));
    env.check('batch: still fenced every 8 frames (under the 1,200 ms congestion signal)', batch.report?.guard.fencing === 'every 8 frames' && batch.report.guard.switchedAt === null, batch.report?.guard);
    // 160 ms steps after a fast calibration: frames 0–7 take ~1,280 ms, past the congestion signal.
    const congested = await guarded('congested', { dt: 1 / 30, fps: 30, frames: 12, stepMs: n => (n <= 4 ? 0 : 160), guard: { stopMs: 400 } });
    env.check('a congested batch switches to fencing every operation at frame 8, and the film completes',
        congested.result?.status === 'completed' && congested.report?.guard.switchedAt === 8 && congested.report.guard.fencing === 'every operation', { status: describe(congested), guard: congested.report?.guard });
    env.check('congested: operations after the switch are measured', congested.report?.guard.worstStepMs >= 150 && congested.report.guard.worstStepMs < 400, congested.report?.guard);
    // The first step compiles pipelines: recorded as firstMs and disclosed, never judged (§4.7).
    const cold = await guarded('cold', { dt: 1 / 30, fps: 30, frames: 6, stepMs: n => (n === 1 ? 1200 : 0), guard: {} });
    env.check('a slow first submission is recorded, disclosed and not judged', cold.result?.status === 'completed' && cold.report?.guard.calibration.firstMs >= 1200 &&
        cold.report.guard.calibration.limitations.some(l => l.includes('unverified first-submission risk')), cold.report?.guard.calibration);
    const after = await guarded('after', { dt: 1 / 30, fps: 30, frames: 6, stepMs: () => 0, guard: {} });
    env.check('the device films normally after the guard stopped films on it', after.result?.status === 'completed', describe(after));
    env.check('no uncaptured WebGPU error', s.uncaptured.length === 0, s.uncaptured);
    return { runs };
}

// ---------------------------------------------------------------------------------------------------
// G6: canvas against readback, on Kerr
// ---------------------------------------------------------------------------------------------------

async function g6(env) {
    const s = await setup(env);
    const fps = 60, J = 120;
    const sampled = [0, 60, 119];
    const stamps = new Set(sampled.map(j => Math.round(j * 1e6 / fps)));
    const videos = [];
    const reports = [];
    const frames = {};
    for (const source of ['canvas', 'readback']) {
        const captured = new Map();
        const work = kerrWorkload();
        const run = await film({
            device: s.device, createWorkload: work.create, camera: orbit, source, mediabunny: await capturingMediabunny(stamps, captured),
            film: { width: 1920, height: 1080, fps, seconds: J / fps, warmupSeconds: 0.25 }, output: { name: `G6-kerr-${source}.mp4` },
        });
        env.check(`Kerr from the ${source} source films`, run.result?.status === 'completed' && run.report?.source === source, describe(run));
        frames[source] = captured;
        reports.push({ id: `G6-kerr-${source}`, ...summary(run) });
        const video = await keep(env, `G6-kerr-${source}`, run, { id: `G6-kerr-${source}`, width: 1920, height: 1080, fps, frames: J, container: 'mp4' });
        if (video) videos.push(video);
        await s.settled(`Kerr ${source}`, work.log.engines);
    }
    const pairs = sampled.map(j => {
        const t = Math.round(j * 1e6 / fps);
        const a = frames.canvas.get(t), b = frames.readback.get(t);
        return { frame: j, formats: [a?.format, b?.format], ...(a?.rgba && b?.rgba ? compareRGB(a.rgba, b.rgba) : { differing: -1 }) };
    });
    env.check('G6: what each source hands the encoder is the same image, byte for byte in RGB', pairs.every(p => p.differing === 0), pairs);
    env.check('no uncaptured WebGPU error', s.uncaptured.length === 0, s.uncaptured);
    return { videos, reports, pairs: [{ a: 'G6-kerr-canvas', b: 'G6-kerr-readback' }], captured: pairs };
}

// ---------------------------------------------------------------------------------------------------
// G9: cancellation, the main thread and the preview, device loss
// ---------------------------------------------------------------------------------------------------

async function g9Cancel(env) {
    const s = await setup(env);
    const results = {};
    const size = { width: 640, height: 360 };
    /** A film of 60 frames at 30 fps, 5 ms a step, cancelled where `arm(controller, …)` says. */
    const cancelled = async (key, arm, extra = {}) => {
        const controller = new AbortController();
        let abortedAt = null;
        const abort = () => { if (!abortedAt) { abortedAt = performance.now(); controller.abort(); } };
        const work = patternWorkload({ dt: 1 / 30, stepMs: n => { arm.step?.(n, abort); return 5; } });
        const createWorkload = arm.create ? context => arm.create(context, abort, work.create) : work.create;
        const run = await film({
            device: s.device, createWorkload, signal: controller.signal, film: { ...size, fps: 30, seconds: 2, warmupSeconds: 0.2 },
            onProgress: p => arm.progress?.(p, abort), partial: 'keep-if-finalizable', ...extra,
        });
        const latencyMs = abortedAt === null ? null : ms(performance.now() - abortedAt);
        results[key] = { outcome: describe(run), latencyMs, status: run.report?.status ?? null, states: run.report?.states.map(x => x.state) };
        // abort() made the signal's reason, a DOMException whose unread stack keeps the frames it was made in
        // (here, inside a step) alive. It is the app's, as the controller is: read it, as a logged error would be.
        void controller.signal.reason?.stack;
        await s.settled(key, work.log.engines);
        return { run, latencyMs };
    };
    const expectCancelled = (key, { run, latencyMs }) => {
        env.check(`cancelled ${key}: AbortError, nothing kept`, run.error?.name === 'AbortError' && !run.result, describe(run));
        env.check(`cancelled ${key}: the report says cancelled, not failed`, run.report === null || (run.report.status === 'cancelled' && run.report.cancelled && run.report.failure === null),
            run.report && { status: run.report.status, failure: run.report.failure });
        env.check(`cancelled ${key}: within 1 s`, latencyMs !== null && latencyMs < 1000, `${latencyMs} ms`);
    };
    // While the app's createWorkload is still building (it ignores the signal and takes 100 ms more).
    expectCancelled('while building', await cancelled('building', {
        create: async (context, abort, create) => { abort(); await sleep(100); return create(context); },
    }));
    expectCancelled('while calibrating', await cancelled('calibrating', { progress: (p, abort) => { if (p.phase === 'calibrating') abort(); } }));
    expectCancelled('while resetting', await cancelled('resetting', { progress: (p, abort) => { if (p.phase === 'resetting') abort(); } }));
    // Calibration takes steps 1–4 and the 0.2 s warm-up steps 5–10 at 30 fps.
    expectCancelled('during the warm-up', await cancelled('warm-up', { step: (n, abort) => { if (n === 7) abort(); } }));
    expectCancelled('while recording', await cancelled('recording', { progress: (p, abort) => { if (p.phase === 'recording' && p.frame === 20) abort(); } }));
    // While the encoder holds frame 30: the abort comes from inside the add, before mediabunny encodes it.
    const mb = await import('mediabunny');
    let holdAbort = null;
    class SlowSource extends mb.VideoSampleSource {
        async add(sample, options) {
            if (sample.microsecondTimestamp === 1e6) { holdAbort?.(); await sleep(150); }
            return super.add(sample, options);
        }
    }
    expectCancelled('while the encoder holds a frame', await cancelled('add', {
        progress: (p, abort) => { holdAbort = abort; },
    }, { mediabunny: { ...mb, VideoSampleSource: SlowSource } }));
    // A cancel after finalizing began joins finish(): the first terminal request wins (§4.9).
    let finalizing = null;
    class SlowOutput extends mb.Output {
        async finalize() { finalizing?.(); await sleep(150); return super.finalize(); }
    }
    const late = await cancelled('finalizing', { progress: (p, abort) => { finalizing = abort; } }, { mediabunny: { ...mb, Output: SlowOutput } });
    env.check('a cancel during the muxer\'s finalize joins it: the film completes, as §4.9\'s first terminal request',
        late.run.result?.status === 'completed' && late.run.report?.cancelled === false, describe(late.run));
    // A cancel the moment the runner reports finalizing, before it asks the session to finish, wins.
    expectCancelled('as finalizing is reported', await cancelled('finalizing-reported', { progress: (p, abort) => { if (p.phase === 'finalizing') abort(); } }));
    // A film waiting for its turn on the device.
    const running = film({ device: s.device, createWorkload: patternWorkload({ dt: 1 / 30, stepMs: () => 5 }).create, film: { ...size, fps: 30, seconds: 1 } });
    const controller = new AbortController();
    const waiting = film({ device: s.device, createWorkload: patternWorkload({ dt: 1 / 30 }).create, signal: controller.signal, film: { ...size, fps: 30, seconds: 1 } });
    await sleep(50);
    const t = performance.now();
    controller.abort();
    const w = await waiting;
    const waitMs = ms(performance.now() - t);
    const r = await running;
    env.check('a film waiting for its turn is cancelled at once, before the running one ends', w.error?.name === 'AbortError' && waitMs < 100 && r.result?.status === 'completed', { waiting: describe(w), waitMs, running: describe(r) });
    results.waiting = { outcome: describe(w), latencyMs: waitMs };
    await s.settled('after the cancels');
    const after = await film({ device: s.device, createWorkload: patternWorkload({ dt: 1 / 30 }).create, film: { ...size, fps: 30, seconds: 1 } });
    env.check('the device films normally after every cancel', after.result?.status === 'completed', describe(after));
    env.check('no uncaptured WebGPU error', s.uncaptured.length === 0, s.uncaptured);
    return { results };
}

/** The app's preview: an engine of its own on the page's canvas, drawing the pattern each animation frame. */
async function startPreview(device, canvas) {
    const engine = new NullGraph();
    const init = await engine.init(canvas, { device });
    if (!init.success) throw new Error(`preview: ${init.error}`);
    const workload = await patternWorkload().create({ engine, width: canvas.width, height: canvas.height });
    let raf = 0, frames = 0, running = false;
    const tick = () => {
        if (!running) return;
        engine.render(workload.draw({ index: frames++, seconds: 0, timestampUs: 0, durationUs: 0, view: null }));
        raf = requestAnimationFrame(tick);
    };
    return {
        engine,
        get frames() { return frames; },
        start() { if (!running) { running = true; raf = requestAnimationFrame(tick); } },
        stop() { running = false; cancelAnimationFrame(raf); },
        destroy() { this.stop(); workload.dispose(); engine.destroy(); },
    };
}

/**
 * rAF intervals and long tasks while a film runs: how free the page's own thread stayed. Each long task is
 * given with its start, in ms from the watch's start, so it can be set against the film's phases.
 */
function watchMainThread() {
    const intervals = [], longTasks = [];
    const t0 = performance.now();
    let last = 0, raf = 0, on = true;
    const observer = new PerformanceObserver(list => longTasks.push(...list.getEntries().map(e => ({ at: ms(e.startTime - t0), ms: ms(e.duration) }))));
    observer.observe({ type: 'longtask' });
    const tick = t => { if (!on) return; if (last) intervals.push(t - last); last = t; raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
    return () => {
        on = false;
        cancelAnimationFrame(raf);
        observer.disconnect();
        const sorted = [...intervals].sort((a, b) => a - b);
        const at = p => (sorted.length ? ms(sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))]) : null);
        return {
            t0, rafCount: intervals.length, rafP50: at(0.5), rafP90: at(0.9), rafMax: at(1),
            longTasks: longTasks.length, longTaskMs: ms(longTasks.reduce((a, b) => a + b.ms, 0)), longTaskList: longTasks,
        };
    };
}

async function g9Responsive(env) {
    const s = await setup(env);
    const canvas = document.querySelector('#preview');
    const preview = await startPreview(s.device, canvas);
    preview.start();
    await sleep(500);
    const before = preview.frames;
    env.check('the preview runs', before >= 15, `${before} preview frames in 0.5 s`);
    // The app pauses its preview (one call), films, and restores it in its own finally (§4.9).
    preview.stop();
    const work = kerrWorkload();
    const progress = [];
    const stop = watchMainThread();
    let run;
    try {
        run = await film({
            device: s.device, createWorkload: work.create, camera: orbit, film: { width: 1920, height: 1080, fps: 60, seconds: 3 },
            onProgress: p => progress.push({ phase: p.phase, frame: p.frame, background: p.background, t: performance.now() }),
        });
    } finally {
        preview.start();
    }
    const thread = stop();
    // When each phase began, in the watch's ms, to place the long tasks.
    thread.phases = progress.filter((p, i) => i === 0 || p.phase !== progress[i - 1].phase).map(p => ({ phase: p.phase, at: ms(p.t - thread.t0) }));
    env.check('Kerr 1080p films beside the paused preview', run.result?.status === 'completed', describe(run));
    env.check('the main thread kept its animation frames during the film (rAF p90 under 25 ms)', thread.rafP90 !== null && thread.rafP90 < 25, thread);
    env.check('progress: every frame reported, never as background', progress.filter(p => p.phase === 'recording').length === 180 && progress.every(p => !p.background));
    const video = await keep(env, 'G9-kerr-1080p', run, { id: 'G9-kerr-1080p', width: 1920, height: 1080, fps: 60, frames: 180, container: 'mp4' });
    await sleep(500);
    const after = preview.frames - before;
    env.check('the preview resumed after the film, on its own canvas at its own size', after >= 15 && canvas.width === 640 && canvas.height === 360 && !preview.engine.isDestroyed,
        { framesAfter: after, size: [canvas.width, canvas.height] });
    preview.destroy();
    await s.settled('film and preview', work.log.engines);
    env.check('no uncaptured WebGPU error', s.uncaptured.length === 0, s.uncaptured);
    return { videos: video ? [video] : [], reports: [summary(run)], mainThread: thread };
}

async function g9Lost(env) {
    const outcomes = {};
    for (const [key, source, partial] of [['canvas', 'canvas', 'discard'], ['readback', 'readback', 'discard'], ['canvas-keep', 'canvas', 'keep-if-finalizable'], ['readback-keep', 'readback', 'keep-if-finalizable']]) {
        const s = await setup(env);
        const canvas = new OffscreenCanvas(640, 360);
        const preview = await startPreview(s.device, canvas);
        preview.start();
        await sleep(200);
        preview.stop();
        const work = kerrWorkload({ count: 65536 });
        let lostAt = null;
        const run = await film({
            device: s.device, createWorkload: work.create, camera: orbit, source, partial, film: { width: 640, height: 360, fps: 60, seconds: 2 },
            onProgress: p => { if (p.phase === 'recording' && p.frame === 30 && lostAt === null) { lostAt = p.frame; s.device.destroy(); } },
        });
        const words = run.error?.message ?? run.result?.report.failure ?? '';
        outcomes[key] = { outcome: describe(run), failure: run.report?.failure, deviceLost: run.report?.deviceLost, frames: run.result?.frames ?? null, status: run.report?.status };
        env.check(`${key}: the lost device ends the film in §8's words`, /^film: GPU device lost at frame \d+ \(destroyed[^)]*\); partial file (kept|discarded)\.$/.test(words), words);
        env.check(`${key}: the report names the loss`, typeof run.report?.deviceLost === 'string' && run.report.deviceLost.startsWith('destroyed'), run.report?.deviceLost);
        if (partial === 'discard') env.check(`${key}: nothing kept`, !run.result && run.report?.status === 'failed', describe(run));
        else if (run.result) {
            const video = await keep(env, `G9-lost-${key}`, run, { id: `G9-lost-${key}`, width: 640, height: 360, fps: 60, frames: run.result.frames, container: 'mp4', partial: true });
            outcomes[key].video = video;
            env.check(`${key}: a kept partial is finalized, with its frame count`, run.result.status === 'partial' && run.result.frames > 0 && run.result.frames <= 31, outcomes[key]);
        }
        // The app's recovery (§4.9): a new device, the preview rebuilt on it, and a film that completes.
        const fresh = await setup(env);
        preview.destroy();
        const rebuilt = await startPreview(fresh.device, canvas);
        rebuilt.start();
        await sleep(200);
        rebuilt.stop();
        const again = await film({ device: fresh.device, createWorkload: kerrWorkload({ count: 65536 }).create, camera: orbit, source, film: { width: 640, height: 360, fps: 60, seconds: 0.5 } });
        env.check(`${key}: after recovery on a new device, the preview draws and a film completes`, rebuilt.frames > 0 && again.result?.status === 'completed', describe(again));
        rebuilt.destroy();
        await fresh.settled(`${key}, recovered`);
        env.check(`${key}: no uncaptured WebGPU error on the new device`, fresh.uncaptured.length === 0, fresh.uncaptured);
    }
    return { outcomes, videos: Object.values(outcomes).map(o => o.video).filter(Boolean) };
}

/**
 * A real loss, not a destroy(): mid-film the page asks run.mjs to open chrome://gpucrash, which ends this Chrome's
 * GPU process (the video encoder lives there too). The film must end in §8's words, its report naming the loss,
 * and the app recover on a new device once the GPU process is back. Not a TDR: no GPU work is made to hang.
 */
async function g9Crash(env) {
    const s = await setup(env);
    const work = kerrWorkload({ count: 65536 });
    let asked = false;
    const lostAt = [];
    void s.device.lost.then(info => { lostAt.push(performance.now()); env.log(`device lost: ${info.reason} ${info.message}`); });
    // Says so when the encoder holds an add for over a second: what the film used to wait on for ever.
    const mb = await import('mediabunny');
    const held = [];
    class TracingSource extends mb.VideoSampleSource {
        async add(sample, options) {
            const t = sample.microsecondTimestamp;
            const timer = setTimeout(() => { held.push(t); env.log(`the encoder has held the add of ${t} µs for 1 s`); }, 1000);
            try { return await super.add(sample, options); } finally { clearTimeout(timer); }
        }
    }
    const run = await film({
        device: s.device, createWorkload: work.create, camera: orbit, partial: 'keep-if-finalizable', film: { width: 640, height: 360, fps: 60, seconds: 4 },
        mediabunny: { ...mb, VideoSampleSource: TracingSource },
        onProgress: p => { if (p.phase === 'recording' && p.frame === 30 && !asked) { asked = true; env.log('action gpucrash'); } },
    });
    const endedMs = lostAt.length ? ms(performance.now() - lostAt[0]) : null;
    env.check('the film ends within 3 s of the loss, though its encoder stopped answering', endedMs !== null && endedMs < 3000, { endedMs, held });
    const words = run.error?.message ?? run.result?.report.failure ?? '';
    env.check('the crashed GPU process ends the film in §8\'s words', /^film: GPU device lost at frame \d+ \(.+\); partial file (kept|discarded)\.$/.test(words), words);
    env.check('the report names the loss', typeof run.report?.deviceLost === 'string', run.report?.deviceLost);
    const outcome = {
        outcome: describe(run), failure: run.report?.failure, deviceLost: run.report?.deviceLost, frames: run.result?.frames ?? null,
        cleanupErrors: run.report?.cleanupErrors, endedMsAfterLoss: endedMs, heldAdds: held,
    };
    let video = null;
    if (run.result) video = await keep(env, 'G9-crash-kept', run, { id: 'G9-crash-kept', width: 640, height: 360, fps: 60, frames: run.result.frames, container: 'mp4', partial: true });
    // Recovery: a new adapter and device, once Chrome has its GPU process back.
    let fresh = null;
    for (let i = 0; i < 20 && !fresh; i++) {
        try { fresh = await setup(env); } catch { await sleep(250); }
    }
    env.check('a new device is had after the crash', !!fresh);
    if (fresh) {
        const again = await film({ device: fresh.device, createWorkload: kerrWorkload({ count: 65536 }).create, camera: orbit, film: { width: 640, height: 360, fps: 60, seconds: 1 } });
        env.check('on the new device a film completes', again.result?.status === 'completed', describe(again));
        await fresh.settled('recovered');
    }
    return { outcome, videos: video ? [video] : [] };
}

/**
 * Visibility (§4.9): mid-film the page asks run.mjs to put another tab in front ("hide") and later to bring it
 * back ("show"), then to freeze it for a second ("freeze"). Film time never follows the wall clock, so the film's
 * hashes equal an undisturbed film's; progress says when the page was hidden.
 */
async function g9Background(env) {
    const s = await setup(env);
    // Kerr, not the pattern: real GPU work and no timers in its steps (a hidden page's timers fire at most once
    // a second, which any guard would read as a slow operation).
    const options = { device: s.device, camera: orbit, output: false, hashes: true, film: { width: 1920, height: 1080, fps: 30, seconds: 6 } };
    const calm = await film({ ...options, createWorkload: kerrWorkload().create });
    env.check('an undisturbed film completes', calm.result?.status === 'completed', describe(calm));
    const progress = [];
    const lifecycle = [];
    const note = e => lifecycle.push({ event: e.type, visibility: document.visibilityState, t: performance.now() });
    for (const type of ['visibilitychange', 'freeze', 'resume']) document.addEventListener(type, note);
    const disturbed = await film({
        ...options, createWorkload: kerrWorkload().create,
        onProgress: p => {
            progress.push({ frame: p.frame, background: p.background, phase: p.phase, t: performance.now() });
            if (p.phase !== 'recording') return;
            // Hidden from frame 20, frozen for a second at frame 60 (hidden, as browsers freeze pages), shown at 120.
            if (p.frame === 20) env.log('action hide');
            if (p.frame === 60) env.log('action freeze');
            if (p.frame === 120) env.log('action show');
        },
    });
    for (const type of ['visibilitychange', 'freeze', 'resume']) document.removeEventListener(type, note);
    env.check('the hidden, frozen and shown film completes', disturbed.result?.status === 'completed', describe(disturbed));
    env.check('film time never follows the wall clock: its frames hash as the undisturbed film\'s', JSON.stringify(disturbed.report?.hashes) === JSON.stringify(calm.report?.hashes));
    const hidden = progress.filter(p => p.background).map(p => p.frame);
    env.check('progress said when the page was hidden', hidden.length > 0, { hiddenFrames: [hidden[0], hidden.at(-1)], count: hidden.length });
    // The frozen second: the longest gap between two frames' progress.
    const gaps = progress.slice(1).map((p, i) => ({ frame: p.frame, ms: ms(p.t - progress[i].t) })).filter(g => g.frame > 1);
    const longest = gaps.reduce((a, b) => (b.ms > a.ms ? b : a), { ms: 0 });
    env.check('the freeze held the film (a gap of most of a second), and it went on after', longest.ms > 700, longest);
    // The pace hidden against visible, frames 30–55 against 130–155 (after the page came back).
    const pace = (a, b) => { const x = progress.find(p => p.frame === a), y = progress.find(p => p.frame === b); return x && y ? ms((y.t - x.t) / (b - a)) : null; };
    const paces = { hiddenMsPerFrame: pace(30, 55), visibleMsPerFrame: pace(130, 155), calmMsPerFrame: calm.report?.timings.msPerFrame };
    // A 2 s freeze while every operation is fenced (160 ms of work a step, busy, not on a timer): the frozen
    // operation is not judged against the 1,500 ms stop, and the film completes (it stopped before §4.9's fix).
    const frozenOps = await film({
        device: s.device, output: false, film: { width: 640, height: 360, fps: 10, seconds: 3 },
        createWorkload: patternWorkload({ dt: 1 / 10, busy: true, stepMs: () => 160 }).create,
        onProgress: p => {
            if (p.phase !== 'recording') return;
            if (p.frame === 3) env.log('action hide');
            if (p.frame === 5) env.log('action freeze 2000');
            if (p.frame === 8) env.log('action show');
        },
    });
    env.check('a 2 s freeze while every operation is fenced: not judged, and the film completes',
        frozenOps.result?.status === 'completed' && frozenOps.report?.guard.fencing === 'every operation' && frozenOps.report.guard.unjudged >= 1,
        { outcome: describe(frozenOps), guard: frozenOps.report?.guard && { fencing: frozenOps.report.guard.fencing, unjudged: frozenOps.report.guard.unjudged, worstStepMs: frozenOps.report.guard.worstStepMs } });
    await s.settled('background films', []);
    env.check('no uncaptured WebGPU error', s.uncaptured.length === 0, s.uncaptured);
    return {
        reports: [summary(calm), summary(disturbed), summary(frozenOps)], lifecycle: lifecycle.map(l => ({ ...l, t: ms(l.t) })), longestGap: longest, paces,
        guard: disturbed.report?.guard,
    };
}

export const CASES = {
    'G9-background': g9Background,
    'G9-crash': g9Crash,
    'G1-pattern': g1Pattern,
    'G1-kerr': g1Kerr,
    G2: g2,
    G3: g3,
    'G3-kerr-4k': g3Kerr4k,
    G5: g5,
    G6: g6,
    'G9-cancel': g9Cancel,
    'G9-responsive': g9Responsive,
    'G9-lost': g9Lost,
};
