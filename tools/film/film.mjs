// D2.1 (docs/offline-films-design.md §4, §5, §7): records a film.
//
//   npm run film -- <film.json> [--frames ffmpeg|png|none] [--out <dir>] [--seconds <n>] [--size <W>x<H>]
//   NG_DAWN_FLAGS=adapter=NVIDIA picks the GPU; NG_FFMPEG the ffmpeg to run; NG_SITE the test site.
//
// The scene is stepped exactly, `stepsPerFrame` steps a frame, each its own submission. The camera follows
// the film's keyframes; the frame is drawn, read back and handed to the sink. ffmpeg writes one .mp4 per
// segment, joined at the end into film.mp4 without encoding again. While a frame's steps run, the frame
// before it is hashed (sha256) and written. The film's folder (tools/film/out/<name>/, or --out) gets
// progress.json after each segment and report.json at the end: provenance, timings, frame hashes.
//
// Exit 0: the film is written and the device reported no error. 1: a GPU error, a failed encode, or a
// step over the 1.5 s stop. 2: a bad film file or argument, or a refusal before any frame.
import { existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';
import { hash } from 'node:crypto';
import { createRequire } from 'node:module';
import { installNodeGPU, describeDevice } from '../../test/gpu/node-gpu.mjs';
import { assertDistCurrent, root } from '../agents/cli.mjs';
import { readManifest, planFilm } from './manifest.mjs';
import { cameraPath } from './camera.mjs';
import { concatSegments, ffmpegSegment, findFfmpeg, noSink, pngSink } from './sinks.mjs';
import { FrameCapture } from './capture.mjs';
import { compareCopies } from './check.mjs';

const STOP_MS = 1500;
const VALUE = new Set(['--frames', '--out', '--seconds', '--size']);
const argv = process.argv.slice(2);
const flag = name => { const i = argv.indexOf(`--${name}`); return i >= 0 ? argv[i + 1] : undefined; };
const positional = argv.filter((a, i) => !a.startsWith('--') && !VALUE.has(argv[i - 1]));
const refuse = message => { console.error(message); process.exit(2); };
const fmt = n => n.toLocaleString('en-US');
const ms = t => `${(t / 1000).toFixed(1)} s`;

// --- the film file, and what this run changes of it ----------------------------------------------
if (positional.length !== 1) refuse('usage: npm run film -- <film.json> [--frames ffmpeg|png|none] [--out <dir>] [--seconds <n>] [--size <W>x<H>]');
assertDistCurrent();
const filmFile = resolve(positional[0]);
let manifest;
try {
    const json = JSON.parse(readFileSync(filmFile, 'utf8'));
    if (flag('seconds') !== undefined) json.film = { ...json.film, seconds: Number(flag('seconds')) };
    if (flag('size') !== undefined) {
        const m = /^(\d+)x(\d+)$/.exec(flag('size'));
        if (!m) refuse(`film: --size must be <width>x<height>, such as 1280x720; got ${flag('size')}.`);
        json.film = { ...json.film, width: Number(m[1]), height: Number(m[2]) };
    }
    manifest = readManifest(json, basename(filmFile, '.json'));
} catch (error) {
    refuse(error.message.startsWith('film:') ? error.message : `film: cannot read ${filmFile}: ${error.message}`);
}
const { film, encode } = manifest;
const frames = flag('frames') ?? 'ffmpeg';
if (!['ffmpeg', 'png', 'none'].includes(frames)) refuse(`film: --frames must be ffmpeg, png or none; got ${frames}.`);
const ffmpeg = frames === 'ffmpeg' ? findFfmpeg() : null;
if (frames === 'ffmpeg' && !ffmpeg) refuse('film: ffmpeg not found (NG_FFMPEG, PATH). Install it (choco install ffmpeg), or pass --frames png.');
const out = resolve(flag('out') ?? join(root, 'tools/film/out', manifest.name));
mkdirSync(out, { recursive: true });
for (const old of ['segments', 'frames', 'film.mp4', 'film.mp4.segments.txt', 'progress.json', 'report.json']) rmSync(join(out, old), { recursive: true, force: true });
const scenePath = join(root, 'tools/film/scenes', `${manifest.scene}.mjs`);
if (!existsSync(scenePath)) refuse(`film: no scene "${manifest.scene}" (tools/film/scenes/${manifest.scene}.mjs).`);

// --- the GPU, the engine, the scene --------------------------------------------------------------
const host = installNodeGPU();
const { NullGraph } = await import('../../dist/index.js');
const engine = new NullGraph();
const init = await engine.init(null, { width: film.width, height: film.height });
if (!init.success) refuse(`film: the engine did not start: ${init.error}`);
const device = describeDevice(engine.device);
const log = line => console.log(`   ${line}`);
const drift = compareCopies().filter(c => c.scene === manifest.scene && c.state !== 'same');
for (const c of drift) console.warn(`film: the copy ${c.scene}/${c.file} is ${c.state} against the site (npm run film:check).`);

let scene, plan;
try {
    const make = (await import(pathToFileURL(scenePath).href)).default;
    scene = await make({ engine, width: film.width, height: film.height, scale: film.scale, settings: manifest.settings, mode: 'record', log });
    plan = planFilm(manifest, { dt: scene.dt });
} catch (error) {
    await host.settle();
    refuse(error.message.startsWith('film:') ? error.message : `film: the scene "${manifest.scene}" failed to start: ${error.stack ?? error}`);
}
console.log(`film "${manifest.name}": ${manifest.scene} ${JSON.stringify(scene.settings)} on ${device}`);
console.log(`   ${fmt(plan.frames)} frames (${film.seconds} s at ${film.fps} fps), ${film.width}×${film.height}` +
    `${film.scale !== 1 ? ` rendered at ×${film.scale}` : ''}, ${plan.stepsPerFrame} step${plan.stepsPerFrame > 1 ? 's' : ''} a frame, ` +
    `${fmt(plan.warmupSteps)} warm-up steps; ${plan.segments.length} segment${plan.segments.length > 1 ? 's' : ''}; frames to ${frames} → ${out}`);

// --- recording -----------------------------------------------------------------------------------
const started = performance.now();
const path = cameraPath(manifest.camera);
const capture = new FrameCapture(engine, film.width, film.height);
const progress = { name: manifest.name, frames: plan.frames, segments: [] };
const timing = { warmupMs: 0, segments: [] };
let worstStep = 0, failure = null;

async function step() {
    const t = performance.now();
    await scene.step();
    const took = performance.now() - t;
    worstStep = Math.max(worstStep, took);
    if (took > STOP_MS) throw new Error(`film: a step took ${fmt(Math.round(took))} ms, over the ${fmt(STOP_MS)} ms stop. Film fewer agents, or on a faster GPU.`);
    return took;
}

try {
    const t = performance.now();
    for (let k = 0; k < plan.warmupSteps; k++) await step();
    timing.warmupMs = performance.now() - t;
    if (plan.warmupSteps) log(`warm-up: ${fmt(plan.warmupSteps)} steps in ${ms(timing.warmupMs)}`);

    for (const segment of plan.segments) {
        const t0 = performance.now();
        const file = join(out, 'segments', `${String(segment.index).padStart(4, '0')}.mp4`);
        mkdirSync(join(out, 'segments'), { recursive: true });
        const sink = frames === 'ffmpeg' ? ffmpegSegment(ffmpeg, encode, film, file) : frames === 'png' ? pngSink(join(out, 'frames'), film) : noSink();
        const hashes = new Array(segment.count);
        let stepping = 0;
        const drain = async () => {
            const { frame, bytes } = await capture.take();
            hashes[frame - segment.first] = hash('sha256', bytes);
            await sink.write(bytes, frame);
        };
        for (let j = segment.first; j < segment.first + segment.count; j++) {
            const steps = (async () => { for (let k = 0; k < plan.stepsPerFrame; k++) stepping += await step(); })();
            await Promise.all([steps, capture.pending ? drain() : null]);
            const seconds = j / film.fps;
            scene.view(path ? path(seconds) : null, seconds);
            engine.render(scene.draw(j));
            capture.grab(j);
        }
        while (capture.pending) await drain();
        await sink.close();
        const wall = performance.now() - t0;
        timing.segments.push({ index: segment.index, frames: segment.count, wallMs: wall, stepMsPerFrame: stepping / segment.count,
            fps: segment.count / (wall / 1000), bytes: frames === 'ffmpeg' ? statSync(file).size : null });
        progress.segments.push({ index: segment.index, first: segment.first, count: segment.count, file: frames === 'ffmpeg' ? file : null, hashes });
        writeFileSync(join(out, 'progress.json'), JSON.stringify(progress, null, 1));
        const done = segment.first + segment.count, eta = (performance.now() - started) / done * (plan.frames - done);
        log(`segment ${segment.index + 1}/${plan.segments.length}: ${segment.count} frames in ${ms(wall)} ` +
            `(${(segment.count / (wall / 1000)).toFixed(1)} frames a second; steps ${(stepping / segment.count).toFixed(1)} ms a frame)` +
            (done < plan.frames ? `; about ${ms(eta)} left` : ''));
    }
    if (frames === 'ffmpeg') await concatSegments(ffmpeg, progress.segments.map(s => s.file), join(out, 'film.mp4'));
} catch (error) {
    failure = error;
    console.error(error.message.startsWith('film:') ? error.message : `film: ${error.stack ?? error}`);
}

// --- the report ----------------------------------------------------------------------------------
await host.settle();
const git = (...a) => { try { return execFileSync('git', a, { cwd: root, encoding: 'utf8' }).trim(); } catch { return null; } };
const require = createRequire(import.meta.url);
const report = {
    name: manifest.name, file: filmFile, manifest, plan: { frames: plan.frames, stepsPerFrame: plan.stepsPerFrame, warmupSteps: plan.warmupSteps,
        segments: plan.segments.length }, scene: scene.info, settings: scene.settings, sink: frames,
    madeWith: { commit: git('rev-parse', 'HEAD'), dirty: git('status', '--porcelain') !== '', node: process.version,
        dawn: require('webgpu/package.json').version, ffmpeg: ffmpeg?.version ?? null, device,
        copies: compareCopies().filter(c => c.scene === manifest.scene).map(({ file, state, copyHash }) => ({ file, state, sha256: copyHash })) },
    timing: { ...timing, totalMs: performance.now() - started, worstStepMs: worstStep },
    frames: progress.segments.flatMap(s => s.hashes),
    gpuErrors: host.events, failure: failure ? String(failure.message) : null,
};
writeFileSync(join(out, 'report.json'), JSON.stringify(report, null, 1));
scene.destroy();
capture.destroy();
for (const event of host.events) console.error(`WebGPU ${event.kind}: ${event.message.split('\n')[0]}`);
const ok = !failure && !host.events.length;
if (ok) {
    const result = frames === 'ffmpeg' ? join(out, 'film.mp4') : frames === 'png' ? join(out, 'frames') : out;
    console.log(`film "${manifest.name}" done in ${ms(report.timing.totalMs)}: ${result}` +
        (frames === 'ffmpeg' ? ` (${(statSync(result).size / 1048576).toFixed(1)} MB)` : ''));
}
for (const d of host.devices) { try { d.destroy(); } catch { /* gone */ } }
process.exit(ok ? 0 : 1);
