// D2.1 (docs/offline-films-design.md §4, §5, §7): records a film.
//
//   npm run film -- <film.json> [--frames ffmpeg|png|none] [--out <dir>] [--seconds <n>] [--size <W>x<H>] [--font <ttf>]
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
import { existsSync, mkdirSync, openSync, closeSync, readFileSync, readdirSync, unlinkSync, statSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { execFileSync, spawn } from 'node:child_process';
import { hash } from 'node:crypto';
import { createRequire } from 'node:module';
import { installNodeGPU, describeDevice } from '../../test/gpu/node-gpu.mjs';
import { assertDistCurrent, root } from '../agents/cli.mjs';
import { readManifest, planFilm } from './manifest.mjs';
import { cameraPath } from './camera.mjs';
import { concatSegments, ffmpegSegment, findFfmpeg, noSink, pngSink } from './sinks.mjs';
import { burnOverlays, DEFAULT_FONT } from './burn.mjs';
import { FrameCapture } from './capture.mjs';
import { compareCopies } from './check.mjs';
import { atomicJSON, inputHash, number, track, saveCheckpoint, restoreCheckpoint, TrackWriter, TrackReader, resumeSegments } from './records.mjs';
import { checkStep } from './guard.mjs';

const VALUE = new Set(['--frames', '--out', '--seconds', '--size', '--camera', '--speed', '--track-every', '--stop-after', '--replay-out', '--font']);
const SWITCH = new Set(['--replay', '--resume', '--detach', '--status']);
const argv = process.argv.slice(2);
const flag = name => { const i = argv.indexOf(`--${name}`); return i >= 0 ? argv[i + 1] : undefined; };
const positional = argv.filter((a, i) => !a.startsWith('--') && !VALUE.has(argv[i - 1]));
const refuse = message => { console.error(message); process.exit(2); };
const fmt = n => n.toLocaleString('en-US');
const ms = t => `${(t / 1000).toFixed(1)} s`;
for (let i = 0; i < argv.length; i++) {
    if (VALUE.has(argv[i])) { if (!argv[i + 1] || argv[i + 1].startsWith('--')) refuse(`film: ${argv[i]} needs a value.`); i++; }
    else if (argv[i].startsWith('--') && !SWITCH.has(argv[i])) refuse(`film: unknown argument ${argv[i]}.`);
}
const replay = argv.includes('--replay'), resume = argv.includes('--resume');
const speed = Number(flag('speed') ?? 1);
let trackEvery = Number(flag('track-every') ?? 6);
if (!Number.isFinite(speed) || speed <= 0) refuse('film: --speed must be a number above zero.');
if (!Number.isInteger(trackEvery) || trackEvery < 1) refuse('film: --track-every must be a whole number above zero.');
if (!replay && (flag('speed') || flag('camera'))) refuse('film: --speed and --camera require --replay.');
if (!replay && flag('replay-out')) refuse('film: --replay-out requires --replay.');
if (flag('stop-after') && (!Number.isInteger(Number(flag('stop-after'))) || Number(flag('stop-after')) < 1)) refuse('film: --stop-after must be a whole number of segments above zero.');

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
const source = resolve(join(root, 'tools/film/out', manifest.name));
const recording = resolve(flag('out') ?? source);
if (resume && !replay && flag('track-every') === undefined) {
    try { trackEvery = JSON.parse(readFileSync(join(recording, 'film.json'), 'utf8')).trackEvery; }
    catch (error) { refuse(`film: cannot resume ${recording}: ${error.message}`); }
}
let recorded;
if (replay) {
    try { recorded = JSON.parse(readFileSync(join(recording, 'film.json'), 'utf8')); }
    catch (error) { refuse(`film: cannot replay ${recording}: ${error.message}`); }
    // The recording is authoritative for the simulation; replay overrides only the view and output.
    if (manifest.scene !== recorded.scene || JSON.stringify(manifest.settings) !== JSON.stringify(recorded.settings)) refuse('film: replay settings differ from the recording.');
    manifest = { ...recorded, film: { ...manifest.film, seconds: recorded.film.seconds / speed } };
    if (flag('camera')) {
        try { manifest.camera = readManifest({ ...manifest, camera: JSON.parse(readFileSync(resolve(flag('camera')), 'utf8')) }).camera; }
        catch (error) { refuse(`film: cannot read replay camera: ${error.message}`); }
    }
}
const { film, encode } = manifest;
const frames = flag('frames') ?? 'ffmpeg';
if (!['ffmpeg', 'png', 'none'].includes(frames)) refuse(`film: --frames must be ffmpeg, png or none; got ${frames}.`);
const ffmpeg = frames === 'ffmpeg' ? findFfmpeg() : null;
if (frames === 'ffmpeg' && !ffmpeg) refuse('film: ffmpeg not found (NG_FFMPEG, PATH). Install it (choco install ffmpeg), or pass --frames png.');
const font = flag('font') ?? DEFAULT_FONT;
if (frames === 'ffmpeg' && manifest.overlays.length && !existsSync(font)) refuse(`film: overlay font ${font} not found (pass another with --font).`);
const out = replay ? resolve(flag('replay-out') ?? join(recording, 'replay')) : recording;
if (replay && out === recording) refuse('film: replay output must differ from the recording directory.');
mkdirSync(out, { recursive: true });
if (argv.includes('--status')) {
    const p = existsSync(join(out, 'progress.json')) ? JSON.parse(readFileSync(join(out, 'progress.json'), 'utf8')) : null;
    const done = p?.segments.reduce((sum, s) => sum + s.count, 0) ?? 0;
    const job = existsSync(join(out, 'job.json')) ? JSON.parse(readFileSync(join(out, 'job.json'), 'utf8')) : null;
    let running = false; if (job) { try { process.kill(job.pid, 0); running = true; } catch { /* finished */ } }
    const elapsed = job ? (Date.now() - Date.parse(job.started)) / 1000 : null;
    console.log(JSON.stringify({ out, running, pid: job?.pid, completedFrames: done, totalFrames: p?.frames,
        etaSeconds: running && done ? elapsed / done * (p.frames - done) : null, segments: p?.segments.map(({ hashes, ...s }) => s) ?? [] }, null, 2)); process.exit(0);
}
if (argv.includes('--detach')) {
    const logFile = join(out, 'film.log'), fd = openSync(logFile, 'a');
    const child = spawn(process.execPath, [process.argv[1], ...argv.filter(a => a !== '--detach')], { cwd: root, detached: true, windowsHide: true, stdio: ['ignore', fd, fd] });
    child.on('error', error => { console.error(`film: detach failed: ${error.message}`); process.exitCode = 2; });
    child.once('spawn', () => { atomicJSON(join(out, 'job.json'), { pid: child.pid, log: logFile, started: new Date().toISOString() }); child.unref(); });
    closeSync(fd);
    console.log(`film: detached; log ${logFile}`); await new Promise(resolve => child.once('spawn', resolve)); process.exit(0);
}
const lock = join(out, 'running.json');
if (existsSync(lock)) {
    const job = JSON.parse(readFileSync(lock, 'utf8'));
    let alive = false;
    try { process.kill(job.pid, 0); alive = true; } catch { /* dead owner */ }
    if (alive) refuse(`film: ${out} is already being written by process ${job.pid}.`);
    unlinkSync(lock);
}
try { const fd = openSync(lock, 'wx'); writeFileSync(fd, JSON.stringify({ pid: process.pid })); closeSync(fd); }
catch { refuse(`film: another process acquired ${out}.`); }
process.on('exit', () => { try { unlinkSync(lock); } catch { /* already gone */ } });
if (!resume && existsSync(join(out, 'progress.json'))) refuse(`film: output already contains a film: ${out}. Use --resume or another --out (remove the old output explicitly to replace it).`);
const scenePath = join(root, 'tools/film/scenes', `${manifest.scene}.mjs`);
if (!existsSync(scenePath)) refuse(`film: no scene "${manifest.scene}" (tools/film/scenes/${manifest.scene}.mjs).`);

// --- the GPU, the engine, the scene --------------------------------------------------------------
const host = installNodeGPU();
const { NullGraph } = await import('../../dist/index.js');
const engine = new NullGraph();
const runtime = !replay && manifest.settings.brains === 'ort' ? await (await import('./ort.mjs')).filmRuntime() : null;
const init = await engine.init(null, { width: film.width, height: film.height, ...(runtime ? { device: runtime.device } : {}) });
if (!init.success) refuse(`film: the engine did not start: ${init.error}`);
const device = describeDevice(engine.device);
const log = line => console.log(`   ${line}`);
const drift = compareCopies().filter(c => c.scene === manifest.scene && c.state !== 'same');
for (const c of drift) console.warn(`film: the copy ${c.scene}/${c.file} is ${c.state} against the site (npm run film:check).`);

let scene, plan;
try {
    const make = (await import(pathToFileURL(scenePath).href)).default;
    scene = await make({ engine, width: film.width, height: film.height, scale: film.scale, settings: manifest.settings, mode: replay ? 'replay' : 'record', trackEvery: recorded?.trackEvery ?? trackEvery, runtime, film: { track }, log });
    plan = planFilm(replay ? { ...manifest, warmupSeconds: 0, film: { ...film, seconds: Math.ceil(film.seconds * film.fps) / film.fps } } : manifest, { dt: replay ? 1 / film.fps : scene.dt });
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
const buildHashes = dir => readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name)).flatMap(e => e.name === 'out' ? [] : e.isDirectory() ? buildHashes(join(dir, e.name)) : /\.(mjs|js|ts|json)$/.test(e.name) ? [[join(dir, e.name).replace(root, ''), hash('sha256', readFileSync(join(dir, e.name)))]] : []);
const tracks = scene.tracks ?? [], checkpoint = scene.checkpoint ?? [];
if (new Set(tracks.map(t => t.name)).size !== tracks.length || new Set(checkpoint.map(t => t.name)).size !== checkpoint.length) refuse('film: track and checkpoint names must be unique.');
for (const { name, buffer } of checkpoint) {
    if ((buffer.usage & (GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST)) !== (GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST)) refuse(`film: checkpoint ${name} needs COPY_SRC and COPY_DST usage.`);
}
const provenance = { manifest: replay ? recorded : manifest, tracks: tracks.map(({ name, layout, size }) => ({ name, layout, size })),
    build: buildHashes(join(root, 'dist')), tools: buildHashes(join(root, 'tools/film')), scenario: buildHashes(join(root, 'tools/agents/scenario')), scene: scene.info?.wgsl, networks: scene.info?.networks, device };
// guard calibration timings are not inputs. The serialized manifest excludes its resolved track table.
if (replay) { const { tracks: ignored, trackEvery: ignoredRate, inputs: ignoredHash, ...original } = recorded; provenance.manifest = original; }
const inputs = replay ? recorded.inputs : inputHash(provenance);
let progress = resume ? JSON.parse(readFileSync(join(out, 'progress.json'), 'utf8')) : { name: manifest.name, frames: plan.frames, segments: [], inputs, sink: frames };
let remaining = resumeSegments(progress, plan);
if (resume && (progress.inputs !== inputs || progress.sink !== frames)) refuse('film: resume inputs differ from the checkpoint (or the frame sink changed).');
if (!replay && !resume) atomicJSON(join(out, 'film.json'), { ...manifest, trackEvery, tracks: provenance.tracks, inputs });
const previousReport = resume && existsSync(join(out, 'report.json')) ? JSON.parse(readFileSync(join(out, 'report.json'), 'utf8')) : null;
const timing = { warmupMs: previousReport?.timing.warmupMs ?? 0, segments: previousReport?.timing.segments ?? [] };
let worstStep = 0, failure = null;
let currentStep = 0, reader, writer, activeSink;
let stopping = false;
process.on('SIGINT', () => { stopping = true; });
process.on('SIGTERM', () => { stopping = true; });
const checkpointFile = index => join(out, 'checkpoints', `${number(index)}.bin`);

async function step() {
    const t = performance.now();
    await scene.step();
    const took = performance.now() - t;
    worstStep = Math.max(worstStep, took);
    checkStep(took);
    return took;
}

try {
    const t = performance.now();
    if (replay) {
        const original = JSON.parse(readFileSync(join(recording, 'progress.json'), 'utf8'));
        const originalPlan = planFilm(recorded, { dt: scene.dt });
        resumeSegments(original, originalPlan);
        if (original.segments.length !== originalPlan.segments.length) throw new Error('film: replay needs a complete recording; resume it first.');
        if (JSON.stringify(recorded.tracks) !== JSON.stringify(provenance.tracks)) throw new Error('film: replay track declarations differ from recording.');
        restoreCheckpoint(engine.device, checkpoint, join(recording, 'checkpoints', '0000.bin'), inputs);
        reader = new TrackReader(engine.device, tracks, recording, original.segments);
    } else if (resume) {
        currentStep = restoreCheckpoint(engine.device, checkpoint, checkpointFile(progress.segments.length), inputs);
    } else {
        for (let k = 0; k < plan.warmupSteps; k++) await step();
        await saveCheckpoint(engine.device, checkpoint, checkpointFile(0), inputs, 0);
        atomicJSON(join(out, 'progress.json'), progress);
    }
    if (!resume) timing.warmupMs = performance.now() - t;
    if (plan.warmupSteps && !resume && !replay) log(`warm-up: ${fmt(plan.warmupSteps)} steps in ${ms(timing.warmupMs)}`);

    for (const segment of remaining) {
        const t0 = performance.now();
        const file = join(out, 'segments', `${String(segment.index).padStart(4, '0')}.mp4`);
        mkdirSync(join(out, 'segments'), { recursive: true });
        const sink = frames === 'ffmpeg' ? ffmpegSegment(ffmpeg, encode, film, file) : frames === 'png' ? pngSink(join(out, 'frames'), film) : noSink();
        activeSink = sink;
        if (!replay) { writer = new TrackWriter(engine.device, tracks, out, segment.index); await writer.save(currentStep, true); }
        const hashes = new Array(segment.count);
        let stepping = 0;
        const drain = async () => {
            const { frame, bytes } = await capture.take();
            hashes[frame - segment.first] = hash('sha256', bytes);
            await sink.write(bytes, frame);
        };
        for (let j = segment.first; j < segment.first + segment.count; j++) {
            const steps = (async () => {
                if (replay) reader.restore(Math.min((j + 1) * speed / film.fps / scene.dt, recorded.film.seconds / scene.dt));
                else for (let k = 0; k < plan.stepsPerFrame; k++) { stepping += await step(); currentStep++; await writer.save(currentStep); }
            })();
            await Promise.all([steps, capture.pending ? drain() : null]);
            const seconds = j / film.fps;
            scene.view(path ? path(seconds) : null, replay ? seconds * speed : seconds);
            if (frames !== 'none') {
                // Optional bounded, awaited reconstruction work before the final draw/readback.
                await scene.prepareSurface?.(j);
                engine.render(scene.draw(j)); capture.grab(j);
            }
        }
        while (capture.pending) await drain();
        await sink.close();
        activeSink = null;
        await host.settle();
        if (host.events.length) throw new Error('film: GPU errors occurred; this segment is not committed (see report.json).');
        if (writer) {
            await writer.save(currentStep, true); writer.close(); writer = null;
            await saveCheckpoint(engine.device, checkpoint, checkpointFile(segment.index + 1), inputs, currentStep);
        }
        const wall = performance.now() - t0;
        timing.segments.push({ index: segment.index, frames: segment.count, wallMs: wall, stepMsPerFrame: stepping / segment.count,
            fps: segment.count / (wall / 1000), bytes: frames === 'ffmpeg' ? statSync(file).size : null });
        progress.segments.push({ index: segment.index, first: segment.first, count: segment.count, file: frames === 'ffmpeg' ? file : null, hashes });
        if (frames === 'none') hashes.fill(null);
        progress.updated = new Date().toISOString();
        atomicJSON(join(out, 'progress.json'), progress);
        const done = segment.first + segment.count, eta = (performance.now() - started) / done * (plan.frames - done);
        log(`segment ${segment.index + 1}/${plan.segments.length}: ${segment.count} frames in ${ms(wall)} ` +
            `(${(segment.count / (wall / 1000)).toFixed(1)} frames a second; steps ${(stepping / segment.count).toFixed(1)} ms a frame)` +
            (done < plan.frames ? `; about ${ms(eta)} left` : ''));
        if (stopping || (flag('stop-after') && progress.segments.length >= Number(flag('stop-after')))) break;
    }
    if (frames === 'ffmpeg' && progress.segments.length === plan.segments.length) {
        await concatSegments(ffmpeg, progress.segments.map(s => s.file), join(out, 'film.mp4'));
        // The film is finished and exact; the labelled copy is made from it, beside it (docs/film-overlays-design.md §3).
        if (manifest.overlays.length) burnOverlays({ input: join(out, 'film.mp4'), output: join(out, 'film-labelled.mp4'),
            film: { ...film, seconds: plan.frames / film.fps }, encode, overlays: manifest.overlays, font, ffmpeg });
    }
} catch (error) {
    failure = error;
    console.error(error.message.startsWith('film:') ? error.message : `film: ${error.stack ?? error}`);
} finally {
    if (writer) writer.close(false);
    if (activeSink) { try { await activeSink.close(); } catch { /* original error is reported */ } }
    reader?.destroy();
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
await runtime?.destroy();
for (const event of host.events) console.error(`WebGPU ${event.kind}: ${event.message.split('\n')[0]}`);
const ok = !failure && !host.events.length;
if (ok) {
    const result = frames === 'ffmpeg' ? join(out, 'film.mp4') : frames === 'png' ? join(out, 'frames') : out;
    console.log(`film "${manifest.name}" ${progress.segments.length === plan.segments.length ? 'done' : 'paused (use --resume)'} in ${ms(report.timing.totalMs)}: ${result}` +
        (frames === 'ffmpeg' && existsSync(result) ? ` (${(statSync(result).size / 1048576).toFixed(1)} MB)` : ''));
}
for (const d of host.devices) { try { d.destroy(); } catch { /* gone */ } }
process.exit(ok ? 0 : 1);
