// D2.4, phases F1 and F2 (docs/offline-films-browser-design.md §10): null-graph/film's headless checks,
// against dist/film.js. B1: the clock, timestamps, the camera and reframing. B2: refusals that need no GPU.
// B3: the rate and size rules, progress, and the guard's fencing. B4: readback. B5: PNG. B6: the session,
// and a film's teardown wherever it fails. B7: reset, step counts, timestamps, hashes and the memory budget.
// B8: the entry without mediabunny, its declarations in a bare consumer, and the probe against stand-in
// browsers. F2's runner runs whole on test/film-stand-ins.mjs. `npm run test:film` runs them.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { copyFileSync, cpSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import v8 from 'node:v8';
import vm from 'node:vm';
import { crc32 as zlibCrc32 } from 'node:zlib';
import {
    BATCH_FRAMES, CONGESTED_BATCH_MS, CleanupLedger, DEFAULT_CODECS, DEFAULT_MEMORY_BUDGET, FILM_LIMIT_MS, FILM_STOP_MS, FILM_TRANSITIONS,
    FilmLifecycle, FrameReadback, LOST_ENCODER_GRACE_MS, PER_OPERATION_MS, bitrateFor, calibrateFilm, cameraPath, checkStorage, containAspect, crc32,
    createFilmSession, encodePNG, fitAspect, frameStamp, isTerminal, planFilm, predictBytes, probeFilm, readCamera, readFilm, readGuardLimits,
    readOutput, recordFilm, sha256, snapshot,
} from '../dist/film.js';
import { readPNG } from '../tools/film/png-read.mjs';
import {
    FakeOffscreenCanvas, FakeVideoFrame, counterWorkload, fakeClock, fakeMediabunny, filmDevice, frames, installWebCodecs, pixels, shown, sleep,
} from './film-stand-ins.mjs';

// The film's engine announces each device it adopts; against stand-in devices that is noise.
const log = console.log;
console.log = (...a) => { if (!/^NullGraph: Adopted/.test(String(a[0]))) log(...a); };

const repo = fileURLToPath(new URL('..', import.meta.url));
const DT = 1 / 60;
const key = (t, more = {}) => ({ t, target: [0, 0, 0], distance: 10, azimuth: 0.2, polar: 1.1, ...more });
const near = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;

// ---------------------------------------------------------------------------------------------------
// B1: the clock and the camera
// ---------------------------------------------------------------------------------------------------

test('B1: planFilm — frames, steps a frame and warm-up; segments only when asked for', () => {
    const p = planFilm({ fps: 60, seconds: 2.5, warmupSeconds: 0.5 }, { dt: DT });
    assert.deepEqual(p, { frames: 150, stepsPerFrame: 1, warmupSteps: 30, dt: DT });
    assert.equal(planFilm({ fps: 30, seconds: 1 }, { dt: DT }).stepsPerFrame, 2, 'a 30 fps film takes two 60 Hz steps a frame');
    const s = planFilm({ fps: 60, seconds: 2.5, segmentSeconds: 1 }, { dt: DT });
    assert.equal(s.perSegment, 60);
    assert.deepEqual(s.segments, [{ index: 0, first: 0, count: 60 }, { index: 1, first: 60, count: 60 }, { index: 2, first: 120, count: 30 }]);
});

test('B1: planFilm refuses what it cannot film, in the Node tool\'s words (§8)', () => {
    assert.throws(() => planFilm({ fps: 45, seconds: 1 }, { dt: DT }), /^Error: film: one frame at 45 fps \(0\.0222\d* s\) is not a whole number of steps of 0\.01666\d* s\.$/);
    assert.throws(() => planFilm({ fps: 60, seconds: 0.01 }, { dt: DT }), /^Error: film: film\.seconds \(0\.01\) is not a whole number of frames at 60 fps\.$/);
    assert.throws(() => planFilm({ fps: 60, seconds: 1, warmupSeconds: 0.01 }, { dt: DT }), /warmupSeconds \(0\.01 s\) is not a whole number of steps/);
    assert.throws(() => planFilm({ fps: 60, seconds: 1, warmupSeconds: -1 }, { dt: DT }), /warmupSeconds must be 0 or more; got -1/);
    assert.throws(() => planFilm({ fps: 60, seconds: 1 }, { dt: 0 }), /the scene's step dt must be a number above zero; got 0/);
    assert.throws(() => planFilm({ fps: 29.97, seconds: 1 }, { dt: DT }), /film\.fps must be a whole number above zero; got 29\.97/);
    assert.throws(() => planFilm({ fps: 60, seconds: 1, segmentSeconds: 0.001 }, { dt: DT }), /segmentSeconds \(0\.001\) is not a whole number of frames at 60 fps/);
});

test('B1: frameStamp — T(j) = round(j·10⁶/fps) µs, and durations add up to the next timestamp exactly', () => {
    assert.deepEqual(frameStamp(0, 60), { index: 0, seconds: 0, timestampUs: 0, durationUs: 16667 });
    assert.deepEqual(frameStamp(1, 60), { index: 1, seconds: 1 / 60, timestampUs: 16667, durationUs: 16666 });
    for (const fps of [24, 30, 60, 144]) {
        let sum = 0;
        for (let j = 0; j < fps; j++) {
            const s = frameStamp(j, fps);
            assert.equal(s.timestampUs, sum, `fps ${fps}, frame ${j}: T(j) is not the sum of the durations before it`);
            sum += s.durationUs;
        }
        assert.equal(sum, 1e6, `${fps} frames at ${fps} fps last ${sum} µs, not one second`);
    }
    // Truncating j/fps seconds to whole µs would give 16666 here: the drift §4.3 avoids.
    assert.equal(frameStamp(1, 60).timestampUs, 16667);
    assert.equal(frameStamp(3600 * 60, 60).timestampUs, 3600 * 1e6, 'an hour into a film, a frame still lands on the second');
    assert.throws(() => frameStamp(-1, 60), /frame index must be a whole number, 0 or more; got -1/);
    assert.throws(() => frameStamp(1.5, 60), /got 1\.5/);
    assert.throws(() => frameStamp(Number.MAX_SAFE_INTEGER + 1, 60), /frame index must be a whole number/);
});

test('B1: readCamera keeps the Node tool\'s keyframes as they were, and takes fov on every key or on none', () => {
    const node = readCamera([key(0), key(2, { distance: 40 })]);
    assert.deepEqual(node, [{ t: 0, target: [0, 0, 0], distance: 10, azimuth: 0.2, polar: 1.1 }, { t: 2, target: [0, 0, 0], distance: 40, azimuth: 0.2, polar: 1.1 }]);
    assert.ok(!('fov' in node[0]), 'a keyframe without fov came back with one');
    assert.deepEqual(readCamera([key(0, { fov: 45 }), key(1, { fov: 60 })]).map(k => k.fov), [45, 60]);
    assert.throws(() => readCamera([key(0, { fov: 45 }), key(1)]), /camera keyframe 1: fov must be given on every keyframe or on none/);
    assert.throws(() => readCamera([key(0), key(1, { fov: 50 })]), /camera keyframe 1: fov must be given on every keyframe or on none/);
    for (const fov of [0, 180, -5, '45', NaN]) assert.throws(() => readCamera([key(0, { fov })]), /camera keyframe 0: fov must be degrees above 0 and below 180/);
    const held = [key(0)];
    readCamera(held)[0].target[0] = 99;
    assert.equal(held[0].target[0], 0, 'readCamera returned the caller\'s target array, not a copy');
});

test('B1: cameraPath carries fov when the keyframes do, and its views are the caller\'s to keep', () => {
    const path = cameraPath([key(0, { fov: 40 }), key(10, { fov: 70, distance: 1000 })]);
    assert.equal(path(0).fov, 40);
    assert.equal(path(10).fov, 70);
    assert.ok(near(path(5).fov, 55, 1e-9), `fov eases through the middle: ${path(5).fov}`);
    assert.ok(Math.abs((path(1e-4).fov - path(0).fov) / 1e-4) < 1e-2, 'fov starts at rest, as every value does');
    assert.ok(near(path(5).distance, 100, 1e-9), 'distance stays on a log scale with fov present');
    assert.ok(!('fov' in cameraPath([key(0), key(1)])(0.5)), 'a path without fov keys gained a fov');
    const a = path(3), b = path(3);
    a.target[0] = 99;
    assert.equal(b.target[0], 0, 'two calls share a target array');
    assert.equal(cameraPath([]), null);
    assert.throws(() => cameraPath([key(1), key(1)]), /camera keyframe 1: t must be at least 0 and above the keyframe before it/);
});

/** Half the frame's width at the target, in metres: what a viewer sees across. */
const halfWidth = (v, aspect) => Math.tan(((v.fov ?? 45) * Math.PI) / 360) * aspect * v.distance;
const design = { target: [1, 2, 3], distance: 20, azimuth: 0.4, polar: 1.2, fov: 45 };

test('B1: fitAspect keeps 0.55 of the 16:9 width in a narrower frame — by field of view, then by distance', () => {
    const want = 0.55 * halfWidth(design, 16 / 9);
    const tall = fitAspect(design, 9 / 16);
    assert.equal(tall.fov, 60, '9:16 widens the field of view to the 60° cap first');
    assert.ok(tall.distance > design.distance, '9:16 then pulls back');
    assert.ok(near(halfWidth(tall, 9 / 16), want, 1e-9), `9:16 shows ${halfWidth(tall, 9 / 16)} m across, not ${want}`);
    const mid = fitAspect(design, 0.8);
    assert.ok(mid.fov > 45 && mid.fov < 60 && mid.distance === design.distance, `0.8 needs only the field of view: ${mid.fov}°, ${mid.distance} m`);
    assert.ok(near(halfWidth(mid, 0.8), want, 1e-9), '0.8 does not keep 0.55 of the width');
    for (const aspect of [1, 4 / 3, 16 / 9, 21 / 9]) {
        const v = fitAspect(design, aspect);
        assert.equal(v.fov, 45, `${aspect}: a frame that already shows 0.55 of the width changed its field of view`);
        assert.ok(halfWidth(v, aspect) >= want - 1e-12, `${aspect}: shows less than 0.55 of the 16:9 width`);
    }
    const noFov = fitAspect({ ...design, fov: undefined }, 9 / 16);
    assert.equal(noFov.fov, 60, 'a view without fov is taken as 45° and comes back with the fov it needs');
    assert.ok(!('fov' in fitAspect({ target: [0, 0, 0], distance: 1, azimuth: 0, polar: 1 }, 16 / 9)), 'an unchanged view without fov gained one');
    const opts = fitAspect(design, 9 / 16, { keepWidth: 0.4, maxFov: 90 });
    assert.ok(near(halfWidth(opts, 9 / 16), 0.4 * halfWidth(design, 16 / 9), 1e-9) && opts.distance === design.distance, 'options are not honoured');
    tall.target[0] = 99;
    assert.equal(design.target[0], 1, 'fitAspect returned the caller\'s target array');
    assert.throws(() => fitAspect(design, 0), /fitAspect: aspect must be a number above zero; got 0/);
});

test('B1: containAspect widens the field of view until the whole frame fits a narrower canvas', () => {
    const shown = containAspect(design, 16 / 9, 1);
    const fitted = fitAspect(design, 16 / 9);
    // Equal widths across: the canvas's (aspect 1) and the frame's (16:9) at the target.
    assert.ok(near(halfWidth(shown, 1), halfWidth(fitted, 16 / 9), 1e-9), `the frame does not fit the canvas: ${halfWidth(shown, 1)} vs ${halfWidth(fitted, 16 / 9)}`);
    assert.deepEqual(containAspect(design, 9 / 16, 16 / 9), fitAspect(design, 9 / 16), 'a wider canvas changes the fitted view');
    assert.throws(() => containAspect(design, 16 / 9, -1), /containAspect: canvasAspect must be a number above zero/);
});

// ---------------------------------------------------------------------------------------------------
// B2: refusals that need no GPU (§8), and B3: the rate and size rules
// ---------------------------------------------------------------------------------------------------

test('B2: readFilm completes a film and refuses, by name, what no encoder takes', () => {
    assert.deepEqual(readFilm({ width: 1920, height: 1080, fps: 60, seconds: 12 }), { width: 1920, height: 1080, fps: 60, seconds: 12, warmupSeconds: 0 });
    assert.throws(() => readFilm({ width: 1919, height: 1080, fps: 60, seconds: 1 }), /^Error: film: width and height must be even for video encoding \(yuv420p\); got 1919×1080\.$/);
    assert.throws(() => readFilm({ width: 1920, height: 1080, fps: 29.97, seconds: 1 }), /film\.fps must be a whole number above zero; got 29\.97/);
    assert.throws(() => readFilm({ width: 1920, height: 1080, fps: 60, seconds: 0 }), /film\.seconds must be a number above zero; got 0/);
    assert.throws(() => readFilm({ width: 1920, height: 1080, fps: 60, seconds: 1, warmupSeconds: -2 }), /warmupSeconds must be 0 or more; got -2/);
    assert.throws(() => readFilm({ width: '1920', height: 1080, fps: 60, seconds: 1 }), /film\.width must be a whole number above zero; got "1920"/);
    assert.throws(() => readFilm(null), /film must be an object: \{ width, height, fps, seconds \}/);
});

test('B2: readOutput — the decided defaults (Q3, Q7), and refusals naming the option', () => {
    const d = readOutput();
    assert.deepEqual(d, { container: 'mp4', codecs: ['avc'], bitrate: undefined, target: 'auto', name: 'film.mp4', memoryBudgetBytes: 256 * 1024 * 1024 });
    assert.equal(DEFAULT_MEMORY_BUDGET, 268435456);
    assert.deepEqual(readOutput({ container: 'webm' }).codecs, ['vp9', 'av1']);
    assert.deepEqual(DEFAULT_CODECS.mp4, ['avc'], 'Q7: MP4 is H.264 alone unless the app opts in');
    assert.deepEqual(readOutput({ codecs: ['avc', 'vp9'], bitrate: 8e6, name: 'x.mp4', memoryBudgetBytes: 1024 }).codecs, ['avc', 'vp9']);
    const handle = { createWritable: async () => ({}) };
    assert.equal(readOutput({ target: { file: handle } }).target.file, handle, 'a picked file handle is accepted as it is');
    const refuses = (output, pattern) => assert.throws(() => readOutput(output), pattern);
    refuses({ container: 'avi' }, /output\.container must be "mp4" or "webm"; got "avi"/);
    refuses({ codecs: ['h264'] }, /output\.codecs: mp4 cannot hold "h264"; it holds avc, hevc, vp9, av1/);
    refuses({ container: 'webm', codecs: ['avc'] }, /output\.codecs: webm cannot hold "avc"; it holds vp8, vp9, av1/);
    refuses({ codecs: [] }, /output\.codecs must be a non-empty list, in preference order/);
    refuses({ target: 'cloud' }, /output\.target must be "auto", "memory", "opfs" or \{ file \} with a FileSystemFileHandle from the save picker/);
    refuses({ target: { file: {} } }, /output\.target must be/);
    refuses({ name: ' ' }, /output\.name must be a non-empty file name/);
    refuses({ memoryBudgetBytes: 0 }, /output\.memoryBudgetBytes must be a whole number above zero; got 0/);
    refuses({ bitrate: 1.5 }, /output\.bitrate \(bits a second\) must be a whole number above zero; got 1\.5/);
    refuses([], /output must be an object/);
});

test('B2: checkStorage refuses a film the OPFS estimate cannot hold, with both numbers', () => {
    assert.throws(() => checkStorage(60_000_001, { quota: 100e6, usage: 88e6 }), /^Error: film: not enough storage in OPFS — 61 MB needed, 12 MB available\.$/);
    assert.deepEqual(checkStorage(5e6, { quota: 100e6, usage: 88e6 }), { needed: 5e6, available: 12e6 });
    assert.deepEqual(checkStorage(5e6, {}), { needed: 5e6, available: null }, 'without quota figures the check must refuse nothing');
    assert.throws(() => checkStorage(1, { quota: 10, usage: 20 }), /1 MB needed, 0 MB available/, 'usage above quota must read as nothing free');
});

test('B3: bitrateFor and predictBytes — MetaForge\'s rule, and 15% over the rate', () => {
    assert.equal(bitrateFor(1920, 1080, 60), 19906560);
    assert.equal(bitrateFor(3840, 2160, 60), 79626240);
    assert.equal(bitrateFor(1280, 720, 30), 4423680);
    assert.equal(predictBytes(19906560, 12), Math.ceil(19906560 / 8 * 12 * 1.15));
    assert.equal(predictBytes(8e6, 1), 1150000);
    assert.throws(() => bitrateFor(1920.5, 1080, 60), /width must be a whole number above zero/);
    assert.throws(() => predictBytes(0, 1), /bitrate must be a number above zero/);
});

// ---------------------------------------------------------------------------------------------------
// B5: PNG
// ---------------------------------------------------------------------------------------------------

test('B5: encodePNG gives the bytes back exactly — alpha between 0 and 255 included — and its checksums are zlib\'s', async () => {
    const width = 37, height = 23; // odd sizes: rows that are not a multiple of anything convenient
    const rgba = new Uint8Array(width * height * 4);
    let seed = 7;
    for (let i = 0; i < rgba.length; i++) { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; rgba[i] = seed >>> 24; }
    rgba.set([200, 100, 50, 1], 0); // a canvas would premultiply this pixel to (1, 0, 0, 1)
    const png = await encodePNG(rgba, width, height);
    const back = readPNG(png);
    assert.equal(back.width, width);
    assert.equal(back.height, height);
    assert.deepEqual(back.rgba, rgba, 'the decoded pixels differ from the encoded ones');
    assert.deepEqual([...png.subarray(0, 8)], [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
    const view = new DataView(png.buffer, png.byteOffset, png.byteLength);
    const types = [];
    for (let at = 8; at < png.length;) {
        const length = view.getUint32(at), type = String.fromCharCode(...png.subarray(at + 4, at + 8));
        types.push(type);
        assert.equal(view.getUint32(at + 8 + length), zlibCrc32(png.subarray(at + 4, at + 8 + length)), `${type}'s CRC is not zlib's`);
        at += 12 + length;
    }
    assert.deepEqual(types, ['IHDR', 'IDAT', 'IEND']);
    await assert.rejects(encodePNG(rgba.subarray(4), width, height), /encodePNG: 37×23 RGBA needs 3404 bytes; got 3400/);
});

test('B5: crc32 is zlib\'s CRC-32, and continues as zlib.crc32(data, value) does', () => {
    const data = new TextEncoder().encode('The quick brown fox jumps over the lazy dog');
    assert.equal(crc32(data), 0x414fa339);
    assert.equal(crc32(data), zlibCrc32(data));
    assert.equal(crc32(data.subarray(10), crc32(data.subarray(0, 10))), crc32(data), 'a continued CRC differs from the whole');
    assert.equal(crc32(new Uint8Array(0)), 0);
});

// ---------------------------------------------------------------------------------------------------
// B6: the session's skeleton (§4.9)
// ---------------------------------------------------------------------------------------------------

test('B6: the lifecycle goes through disposing to one terminal state, and refuses anything else by name', () => {
    const life = new FilmLifecycle();
    for (const s of ['initializing', 'calibrating', 'resetting', 'recording', 'finalizing', 'disposing', 'completed']) life.to(s);
    assert.deepEqual(life.history.map(h => h.state), ['probing', 'initializing', 'calibrating', 'resetting', 'recording', 'finalizing', 'disposing', 'completed']);
    assert.ok(life.history.every((h, i) => i === 0 || h.at >= life.history[i - 1].at), 'entry times go backwards');
    assert.throws(() => life.to('recording'), /film: a session cannot go from completed to recording; from completed it goes to nothing \(it has ended\)/);
    const failing = new FilmLifecycle();
    failing.to('initializing');
    assert.throws(() => failing.to('completed'), /cannot go from initializing to completed; from initializing it goes to calibrating, recording or disposing/);
    const recording = new FilmLifecycle();
    for (const s of ['initializing', 'calibrating', 'resetting', 'recording']) recording.to(s);
    assert.throws(() => recording.to('disposing'), /from recording it goes to finalizing or stopping/, 'recording must settle its samples (stopping) before disposing');
    for (const [state, next] of Object.entries(FILM_TRANSITIONS)) {
        assert.equal(isTerminal(state), next.length === 0, `${state}: terminal and transitions disagree`);
        if (!isTerminal(state) && state !== 'disposing') {
            const reach = new Set([state]);
            for (let grew = true; grew;) { grew = false; for (const s of [...reach]) for (const t of FILM_TRANSITIONS[s]) if (!reach.has(t)) { reach.add(t); grew = true; } }
            assert.ok(reach.has('disposing'), `${state} cannot reach disposing: teardown would not run`);
        }
    }
    assert.deepEqual(FILM_TRANSITIONS.disposing, ['completed', 'partial', 'cancelled', 'failed']);
});

test('B6: the ledger releases last-first, attempts every release, and returns the failures', async () => {
    const ledger = new CleanupLedger();
    const order = [];
    ledger.defer('device listener', () => { order.push('listener'); });
    ledger.defer('engine', async () => { order.push('engine'); throw new Error('engine refused'); });
    ledger.defer('encoder', () => { order.push('encoder'); throw new Error('encoder refused'); });
    ledger.defer('capture pool', () => { order.push('pool'); });
    assert.equal(ledger.size, 4);
    const errors = await ledger.unwind();
    assert.deepEqual(order, ['pool', 'encoder', 'engine', 'listener'], 'not released last-first, or one failure stopped the rest');
    assert.deepEqual(errors.map(e => [e.name, e.error.message]), [['encoder', 'encoder refused'], ['engine', 'engine refused']]);
    assert.deepEqual(await ledger.unwind(), [], 'a second unwind released again');
    assert.deepEqual(order.length, 4);
    // Acquired after the unwind (a slow promise settling late): released at once, not leaked.
    ledger.defer('late readback', () => { order.push('late'); });
    ledger.defer('late failure', () => { throw new Error('late refused'); });
    await ledger.unwind();
    assert.equal(order.at(-1), 'late');
    assert.deepEqual(ledger.lateErrors.map(e => e.name), ['late failure']);
});

// ---------------------------------------------------------------------------------------------------
// B8: the entry without mediabunny, its declarations, and the probe
// ---------------------------------------------------------------------------------------------------

test('B8: dist/film.js imports mediabunny only when it encodes, and the declarations never import it', () => {
    const bundle = readFileSync(join(repo, 'dist/film.js'), 'utf8');
    assert.match(bundle, /import\("mediabunny"\)/, 'the dynamic import is gone: the app\'s bundler could not find mediabunny');
    assert.doesNotMatch(bundle, /^\s*(import|export)\b[^\n]*["']mediabunny["']/m, 'a static import of mediabunny loads it for every caller');
    const dir = join(repo, 'dist/extras/film');
    for (const file of readdirSync(dir)) {
        const text = readFileSync(join(dir, file), 'utf8');
        assert.doesNotMatch(text, /^\s*(import|export)\b[^\n]*["']mediabunny["']/m, `${file} imports mediabunny`);
        assert.doesNotMatch(text, /^\/\/\/\s*<reference/m, `${file} carries a reference directive into consumers' builds`);
    }
});

test('B8: dist/film.js imports the core at run time, as "null-graph", and carries no copy of it', () => {
    const bundle = readFileSync(join(repo, 'dist/film.js'), 'utf8');
    assert.match(bundle, /import\("null-graph"\)/, 'the film no longer loads the app\'s core');
    assert.doesNotMatch(bundle, /^\s*(import|export)\b[^\n]*["']null-graph["']/m, 'a static import of the core loads it for the pure helpers too');
    for (const name of ['NullGraph', 'WebGPUContext', 'BatchManager', 'RenderPipeline']) {
        assert.doesNotMatch(bundle, new RegExp(`class ${name}\\b`), `film.js bundles its own ${name}: an app would run two cores`);
    }
});

/** A folder outside the repository, with dist/film.js copied in and no node_modules anywhere above it. */
function isolated() {
    const dir = mkdtempSync(join(tmpdir(), 'ng-film-iso-'));
    copyFileSync(join(repo, 'dist/film.js'), join(dir, 'film.js'));
    return dir;
}
const run = (dir, source) => execFileSync(process.execPath, ['--input-type=module', '-e', source], { cwd: dir, encoding: 'utf8' }).trim();

test('B8: the pure helpers load and run where mediabunny is not installed; the probe says so in plain words', () => {
    const dir = isolated();
    try {
        assert.equal(run(dir, 'try { import.meta.resolve("mediabunny"); console.log("found"); } catch { console.log("absent"); }'), 'absent',
            'mediabunny resolves from the temporary folder, so this test proves nothing here');
        const out = run(dir, `
            const film = await import('./film.js');
            const plan = film.planFilm({ fps: 60, seconds: 2 }, { dt: 1 / 60 });
            const view = film.cameraPath([{ t: 0, target: [0, 0, 0], distance: 5, azimuth: 0, polar: 1 }, { t: 1, target: [0, 0, 0], distance: 50, azimuth: 1, polar: 1 }])(0.5);
            const png = await film.encodePNG(new Uint8Array(16), 2, 2);
            const scope = { VideoEncoder: function () {}, VideoFrame: function () {}, navigator: { gpu: {} } };
            const support = await film.probeFilm({ width: 1920, height: 1080, fps: 60, scope });
            console.log(JSON.stringify({ frames: plan.frames, distance: view.distance, png: png.length > 50, ok: support.ok, reasons: support.reasons }));`);
        const result = JSON.parse(out);
        assert.equal(result.frames, 120);
        assert.ok(near(result.distance, Math.sqrt(250), 1e-9));
        assert.equal(result.png, true);
        assert.equal(result.ok, false);
        assert.ok(result.reasons.includes('film: mediabunny is not installed; install mediabunny to encode video in the browser.'), result.reasons.join(' | '));
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});

test('B8: the declarations type-check in a consumer with neither mediabunny nor @types installed', () => {
    // The package's own layout: the film's declarations name the core's (dist/core) by relative path, and
    // the core's name gl-matrix, a dependency every install has.
    const dir = mkdtempSync(join(tmpdir(), 'ng-film-dts-'));
    try {
        cpSync(join(repo, 'dist/extras/film'), join(dir, 'dist/extras/film'), { recursive: true, filter: f => !f.endsWith('.map') });
        cpSync(join(repo, 'dist/core'), join(dir, 'dist/core'), { recursive: true });
        mkdirSync(join(dir, 'node_modules'));
        symlinkSync(join(repo, 'node_modules/gl-matrix'), join(dir, 'node_modules/gl-matrix'), 'junction');
        const webgpu = join(repo, 'node_modules/@webgpu/types/dist/index.d.ts').replace(/\\/g, '/');
        writeFileSync(join(dir, 'consumer.ts'), `/// <reference path="${webgpu}" />
import {
    planFilm, cameraPath, fitAspect, probeFilm, readOutput, encodePNG, CleanupLedger, recordFilm, createFilmSession, snapshot, FrameReadback,
    calibrateFilm, type FilmSupport, type CameraView, type FilmTarget, type CreateFilmWorkload, type FilmWorkload, type FilmReport, type FilmProgress,
} from './dist/extras/film/index';
const frames: number = planFilm({ fps: 60, seconds: 2 }, { dt: 1 / 60 }).frames;
const view: CameraView | undefined = cameraPath([{ t: 0, target: [0, 0, 0], distance: 1, azimuth: 0, polar: 1 }])?.(0);
const fitted: CameraView = fitAspect({ target: [0, 0, 0], distance: 1, azimuth: 0, polar: 1, fov: 45 }, 9 / 16);
const target: FilmTarget = readOutput({}).target;
const createWorkload: CreateFilmWorkload = async ({ engine, width, height, defer }) => {
    defer(() => {});
    const workload: FilmWorkload = { dt: 1 / 60, step() {}, draw: frame => (void frame.view, []), dispose() {}, info: { width, height } };
    return engine.device ? workload : workload;
};
async function main(device: GPUDevice): Promise<[FilmSupport, FilmReport, Uint8Array]> {
    await encodePNG(new Uint8Array(4), 1, 1);
    new CleanupLedger().defer('x', () => {});
    const film = await recordFilm({ device, createWorkload, film: { width: 64, height: 36, fps: 60, seconds: 1 }, onProgress: (p: FilmProgress) => void p.etaMs });
    const session = await createFilmSession({ film: { width: 64, height: 36, fps: 60, seconds: 1 }, output: false });
    session.noteFrame();
    await session.finish();
    const shot = await snapshot({ device, createWorkload, width: 8, height: 8 });
    void FrameReadback, void calibrateFilm;
    return [await probeFilm({ device, width: 1920, height: 1080, fps: 60 }), film.report, shot.png];
}
export { frames, view, fitted, target, main };
`);
        writeFileSync(join(dir, 'tsconfig.json'), JSON.stringify({
            compilerOptions: { strict: true, noEmit: true, target: 'ES2022', module: 'ESNext', moduleResolution: 'bundler', lib: ['ES2022', 'DOM'], types: [], skipLibCheck: false },
            files: ['consumer.ts'],
        }));
        try {
            execFileSync(process.execPath, [join(repo, 'node_modules/typescript/bin/tsc'), '-p', join(dir, 'tsconfig.json')], { encoding: 'utf8' });
        } catch (e) {
            assert.fail(`the declarations do not type-check without mediabunny:\n${e.stdout}${e.stderr}`);
        }
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});

/** A stand-in browser with everything, and a stand-in mediabunny whose encoders stop at `maxPixels`. */
function standIns({ maxPixels = 1920 * 1080, held = ['avc', 'hevc', 'vp9', 'av1'], encodable = ['avc', 'vp9'], estimate = { quota: 1e9, usage: 1e8 } } = {}) {
    const calls = { encodable: [], getDirectory: 0, estimate: 0 };
    const scope = {
        navigator: {
            gpu: {},
            storage: {
                getDirectory() { calls.getDirectory++; throw new Error('the cheap probe opened OPFS'); },
                async estimate() { calls.estimate++; return estimate; },
            },
        },
        VideoFrame: function VideoFrame() {}, VideoEncoder: function VideoEncoder() {}, Worker: function Worker() {},
        OffscreenCanvas: class { getContext(kind) { return kind === 'webgpu' ? {} : null; } },
    };
    const mediabunny = {
        Quality: class { constructor(o) { this.bitrate = o.bitrate; } },
        Mp4OutputFormat: class { getSupportedVideoCodecs() { return held; } },
        WebMOutputFormat: class { getSupportedVideoCodecs() { return ['vp8', 'vp9', 'av1']; } },
        // Answers in its own order, not the app's, so the probe must restore the app's.
        async getEncodableVideoCodecs(codecs, o) {
            calls.encodable.push({ codecs, ...o, bitrate: o.quality.bitrate });
            return o.width * o.height <= maxPixels ? encodable.filter(c => codecs.includes(c)) : [];
        },
    };
    return { scope, mediabunny, calls };
}

test('B8: probe — an empty scope says what is missing, and nothing is claimed', async () => {
    const support = await probeFilm({ width: 1920, height: 1080, fps: 60, scope: {} });
    assert.equal(support.ok, false);
    assert.deepEqual(support.codecs, []);
    assert.deepEqual(support.sources, { canvas: { api: false, operational: 'untested' }, readback: { api: false } });
    assert.deepEqual(support.targets, { memory: true, opfs: { api: false, writable: 'untested' }, picker: { api: false } });
    assert.equal(support.worker, false);
    assert.equal(support.storage, null);
    for (const words of ['WebGPU is not available', 'VideoFrame is not available', 'VideoEncoder is not available', 'OPFS (navigator.storage.getDirectory) is not available']) {
        assert.ok(support.reasons.some(r => r.includes(words)), `no reason says "${words}": ${support.reasons.join(' | ')}`);
    }
    await assert.rejects(probeFilm({ width: 1921, height: 1080, fps: 60, scope: {} }), /width and height must be even for video encoding \(yuv420p\); got 1921×1080/);
    await assert.rejects(probeFilm({ width: 1920, height: 1080, fps: 60, container: 'avi', scope: {} }), /output\.container must be "mp4" or "webm"/);
});

test('B8: probe — codecs are the app\'s, held by the container and encodable here, in the app\'s order', async () => {
    const s = standIns();
    const support = await probeFilm({ width: 1920, height: 1080, fps: 60, codecs: ['vp9', 'av1', 'avc'], scope: s.scope, mediabunny: s.mediabunny });
    assert.equal(support.ok, true);
    assert.deepEqual(support.codecs, ['vp9', 'avc'], 'not the app\'s order, or an unencodable codec kept');
    assert.deepEqual(s.calls.encodable[0], { codecs: ['vp9', 'av1', 'avc'], width: 1920, height: 1080, quality: s.calls.encodable[0].quality, frameRate: 60, bitrate: 19906560 },
        'the encoder was not asked at the film\'s own size, rate and default bitrate');
    assert.deepEqual(support.sources, { canvas: { api: true, operational: 'untested' }, readback: { api: true } });
    assert.equal(support.targets.opfs.api, true);
    assert.deepEqual(support.storage, { quota: 1e9, usage: 1e8, available: 9e8 });
    assert.equal(s.calls.getDirectory, 0, 'the cheap probe opened OPFS');
    assert.equal((await probeFilm({ width: 1920, height: 1080, fps: 60, scope: s.scope, mediabunny: s.mediabunny })).codecs.join(), 'avc', 'Q7: MP4 tries H.264 alone by default');
    const notHeld = standIns({ held: ['avc'] });
    const held = await probeFilm({ width: 1920, height: 1080, fps: 60, codecs: ['vp9', 'avc'], scope: notHeld.scope, mediabunny: notHeld.mediabunny });
    assert.deepEqual(held.codecs, ['avc']);
    assert.ok(held.reasons.includes('film: this mediabunny\'s mp4 cannot hold vp9.'), held.reasons.join(' | '));
    assert.deepEqual(notHeld.calls.encodable[0].codecs, ['avc'], 'the encoder was asked for a codec the container cannot hold');
    const bitrate = standIns();
    await probeFilm({ width: 1280, height: 720, fps: 30, bitrate: 5e6, scope: bitrate.scope, mediabunny: bitrate.mediabunny });
    assert.equal(bitrate.calls.encodable[0].bitrate, 5e6, 'the app\'s bitrate was not the one probed');
});

test('B8: probe — a size with no encoder is refused with the sizes it probed that have one', async () => {
    const s = standIns({ maxPixels: 1920 * 1080 });
    const support = await probeFilm({ width: 3840, height: 2160, fps: 60, scope: s.scope, mediabunny: s.mediabunny });
    assert.equal(support.ok, false);
    assert.ok(support.reasons.includes('film: no WebCodecs encoder for 3840×2160 (avc); supported sizes on this browser: 1920×1080, 1280×720, 854×480.'), support.reasons.join(' | '));
    const sizes = s.calls.encodable.map(c => `${c.width}×${c.height}`);
    assert.deepEqual(sizes, ['3840×2160', '2560×1440', '1920×1080', '1280×720', '854×480'], 'a size named in the refusal was not probed, or the asked size was probed again');
    const tall = standIns({ maxPixels: 1080 * 1920 });
    const vertical = await probeFilm({ width: 2160, height: 3840, fps: 60, scope: tall.scope, mediabunny: tall.mediabunny });
    assert.ok(vertical.reasons.some(r => r.includes('supported sizes on this browser: 1080×1920, 720×1280, 480×854')), `vertical sizes are not offered vertically: ${vertical.reasons.join(' | ')}`);
    const none = standIns({ maxPixels: 0 });
    assert.ok((await probeFilm({ width: 1920, height: 1080, fps: 60, scope: none.scope, mediabunny: none.mediabunny })).reasons.some(r => r.endsWith('supported sizes on this browser: none of those probed.')));
});

test('B8: probe — no OffscreenCanvas with WebGPU means readback only, and a worker needs OffscreenCanvas', async () => {
    const s = standIns();
    const scope = { ...s.scope, OffscreenCanvas: class { getContext() { return null; } } };
    const support = await probeFilm({ width: 1920, height: 1080, fps: 60, scope, mediabunny: s.mediabunny });
    assert.equal(support.ok, true, 'readback alone is a source');
    assert.deepEqual(support.sources, { canvas: { api: false, operational: 'untested' }, readback: { api: true } });
    assert.ok(support.reasons.includes('film: no OffscreenCanvas with a WebGPU context here; frames would be read back.'));
    const noOffscreen = await probeFilm({ width: 1920, height: 1080, fps: 60, scope: { ...s.scope, OffscreenCanvas: undefined }, mediabunny: s.mediabunny });
    assert.equal(noOffscreen.worker, false);
    const device = await probeFilm({ width: 1920, height: 1080, fps: 60, device: {}, scope: { ...s.scope, navigator: { storage: s.scope.navigator.storage } }, mediabunny: s.mediabunny });
    assert.equal(device.sources.readback.api, true, 'an app\'s device counts as WebGPU even where navigator.gpu is missing (a worker)');
});

// ---------------------------------------------------------------------------------------------------
// F2: the runner and its pieces, whole, on stand-ins (test/film-stand-ins.mjs)
// ---------------------------------------------------------------------------------------------------

/** An F2 test: the stand-in VideoFrame and OffscreenCanvas installed for it alone. */
const f2 = (name, fn) => test(name, async () => {
    const uninstall = installWebCodecs();
    try { await fn(); } finally { uninstall(); }
});

/** 6 frames at 60 fps, no warm-up: frame j shows j + 1 steps. */
const SMALL = { width: 64, height: 36, fps: 60, seconds: 0.1 };

/** recordFilm on stand-ins, and what happened, whether a film was made or not. */
async function film({ device = filmDevice(), workload = {}, mb = {}, ...options } = {}) {
    const work = counterWorkload(device, workload);
    const mediabunny = fakeMediabunny(mb);
    const reports = [];
    let result = null, error = null;
    try {
        result = await recordFilm({ device, createWorkload: work.create, film: SMALL, mediabunny, onReport: r => reports.push(r), ...options });
    } catch (e) {
        error = e;
    }
    return { result, error, report: reports[0] ?? null, reports, device, workload: work.log, mb: mediabunny.log };
}

/** Everything a film made is released, once, and the device is as the film found it. */
function released(run, label) {
    const { log } = run.device;
    const twice = log.buffers.filter(b => b.destroyed !== 1);
    assert.deepEqual(twice.map(b => `${b.label}: ${b.destroyed}`), [], `${label}: buffers not destroyed exactly once`);
    assert.deepEqual(log.textures.filter(t => t.destroyed !== 1).map(t => `${t.label}: ${t.destroyed}`), [], `${label}: textures not destroyed exactly once`);
    assert.equal(log.listeners, 0, `${label}: an uncapturederror listener was left on the device`);
    assert.equal(run.device.scopeDepth(), 0, `${label}: the error-scope stack is unbalanced (${log.scopes.join(', ')})`);
    for (const output of run.mb.outputs) assert.ok(output.state === 'finalized' || output.state === 'canceled', `${label}: the muxer was left ${output.state}`);
    const open = FakeVideoFrame.made.filter(f => f.closes !== 1);
    assert.deepEqual(open.map(f => `${f.timestamp}: closed ${f.closes}×`), [], `${label}: VideoFrames not closed exactly once`);
    assert.deepEqual(run.mb.samples.filter(s => s.closes !== 1).map(s => s.timestamp), [], `${label}: VideoSamples not closed exactly once`);
}

const rgbx = (j, { fps = 60, timestamp = frameStamp(j, fps).timestampUs } = {}) =>
    new FakeVideoFrame(pixels(j, 64, 36), { format: 'RGBX', codedWidth: 64, codedHeight: 36, timestamp, duration: frameStamp(j, fps).durationUs });

f2('review: cancelling the last waiter never unlocks a device still in use', async () => {
    const device = filmDevice();
    const controller = new AbortController();
    const order = [];
    const first = film({ device, workload: { stepMs: () => 2 }, onProgress: p => order.push(`first ${p.phase}`) });
    const waiting = film({ device, signal: controller.signal });
    await sleep(0);
    controller.abort();
    assert.equal((await waiting).error?.name, 'AbortError');
    const third = film({ device, onProgress: p => order.push(`third ${p.phase}`) });
    const results = await Promise.all([first, third]);
    assert.ok(results.every(r => r.result), results.map(r => r.error).join(', '));
    assert.ok(order.indexOf('first finalizing') < order.indexOf('third calibrating'), order.join(', '));
});

f2('review: stop waits for queued frames and never finalizes a failed encoder', async () => {
    for (const failAddAt of [-1, 0]) {
        const mb = fakeMediabunny({ addMs: 5, failAddAt });
        const session = await createFilmSession({ film: SMALL, mediabunny: mb });
        const adding = session.add(rgbx(0)).catch(error => error);
        const result = await session.stop();
        await adding;
        assert.equal(result?.status ?? null, failAddAt < 0 ? 'partial' : null);
        assert.equal(session.state, failAddAt < 0 ? 'partial' : 'failed');
        assert.equal(mb.log.outputs[0].state, failAddAt < 0 ? 'finalized' : 'canceled');
    }
});

f2('review: an abort during asynchronous session setup discards the initialized muxer', async () => {
    const controller = new AbortController();
    const mb = fakeMediabunny();
    mb.getEncodableVideoCodecs = async () => { controller.abort(); return ['avc']; };
    let session;
    try {
        await assert.rejects(async () => { session = await createFilmSession({ film: SMALL, mediabunny: mb, signal: controller.signal }); }, { name: 'AbortError' });
        assert.equal(mb.log.outputs[0].state, 'canceled');
    } finally {
        await session?.cancel();
    }
});

f2('review: packets emitted only during finalize still obey the memory budget', async () => {
    const mb = fakeMediabunny();
    const BaseOutput = mb.Output;
    mb.Output = class extends BaseOutput {
        async finalize() {
            this.tracks[0].source.config.onEncodedPacket({ byteLength: 2e6, type: 'delta' });
            await super.finalize();
        }
    };
    const session = await createFilmSession({ film: SMALL, mediabunny: mb, output: { memoryBudgetBytes: 1048576 } });
    await session.add(rgbx(0));
    const result = await session.stop();
    assert.equal(result, null, 'an over-budget Blob escaped finalize');
    assert.equal(session.state, 'failed');
    assert.match(session.report().failure, /in-memory budget/);
});

f2('review: a failed readback allocation releases every staging buffer already acquired', () => {
    const device = filmDevice();
    const create = device.createBuffer.bind(device);
    device.createBuffer = descriptor => {
        if (device.log.buffers.length === 1) throw new Error('staging allocation failed');
        return create(descriptor);
    };
    assert.throws(() => new FrameReadback(device, { format: 'rgba8unorm' }, 64, 36), /staging allocation failed/);
    assert.equal(device.log.buffers[0].destroyed, 1);
});

f2('review: failed muxer setup is cancelled before a session can own it', async () => {
    for (const point of ['track', 'start']) {
        const mb = fakeMediabunny();
        const BaseOutput = mb.Output;
        mb.Output = class extends BaseOutput {
            addVideoTrack(...args) {
                if (point === 'track') throw new Error('track setup failed');
                super.addVideoTrack(...args);
            }
            async start() { throw new Error('start setup failed'); }
        };
        await assert.rejects(createFilmSession({ film: SMALL, mediabunny: mb }), new RegExp(`${point} setup failed`));
        assert.equal(mb.log.outputs[0].state, 'canceled');
    }
});

f2('review: a map rejection is observed immediately and still reaches the readback consumer', async () => {
    const device = filmDevice();
    const reader = new FrameReadback(device, { format: 'rgba8unorm' }, 64, 36);
    device.log.buffers[0].mapAsync = () => Promise.reject(new Error('mapping failed'));
    try {
        reader.grab(0);
        await sleep(5); // map rejects before the runner gets to take(), as on device loss
        await assert.rejects(reader.take(), /mapping failed/);
    } finally {
        reader.destroy();
    }
});

test('review: camera keyframe times must be finite', () => {
    assert.throws(() => readCamera([key(0), key(Infinity)]), /camera keyframe 1: t must be/);
});

// --- B3: progress and the guard's fencing -----------------------------------------------------------

f2('B3: calibrateFilm fences every 8 frames up to 150 ms a frame, and every operation above it (§4.7)', async () => {
    const clock = fakeClock();
    try {
        const calibrate = (stepMs, drawMs, stepsPerFrame, o = {}) => {
            const engine = { device: filmDevice(), render() {} };
            let n = 0;
            const workload = {
                dt: 1 / 60, dispose() {},
                step: () => clock.advance(o.first && n++ === 0 ? o.first : stepMs),
                draw: () => { clock.advance(drawMs); return []; },
            };
            return calibrateFilm(engine, workload, { width: 64, height: 36, stepsPerFrame, frame: { ...frameStamp(0, 60), view: null }, ...o.options });
        };
        const edge = await calibrate(50, 50, 2);
        assert.equal(edge.msPerFrame, 150, 'stepsPerFrame × the slowest step + the slowest draw');
        assert.equal(edge.fenceEvery, BATCH_FRAMES, 'exactly 150 ms a frame is fenced every 8 frames');
        assert.equal((await calibrate(50, 50.5, 2)).fenceEvery, 1, 'above 150 ms a frame, every operation is fenced');
        assert.equal(PER_OPERATION_MS, 150);
        assert.deepEqual(edge.stepMs, [50, 50, 50], 'three measured rounds after the first');
        assert.equal(edge.timing, 'queue-completion wall ms');
        assert.ok(edge.limitations.some(l => l.includes('unverified first-submission risk')), 'the unmeasured first submission is not disclosed');
        // The first step and draw compile pipelines: recorded, never judged.
        const cold = await calibrate(1, 1, 1, { first: 5000 });
        assert.equal(cold.firstMs, 5000 + 1);
        await assert.rejects(calibrate(1001, 1, 1), /^Error: film: a step at 64×36 took 1,001 ms, over the 1,000 ms per-submission limit \(Windows resets a GPU busy for ~2 s\)\. Try a smaller size, fewer particles, or a faster GPU\.$/);
        await assert.rejects(calibrate(1, 61, 1, { options: { limits: { limitMs: 60 } } }), /^Error: film: a draw at 64×36 took 61 ms, over the 60 ms per-submission limit/);
    } finally {
        clock.restore();
    }
    assert.deepEqual(readGuardLimits(), { limitMs: FILM_LIMIT_MS, stopMs: FILM_STOP_MS });
    assert.deepEqual(readGuardLimits({ limitMs: 50, stopMs: 80 }), { limitMs: 50, stopMs: 80 });
    assert.throws(() => readGuardLimits({ stopMs: 2000 }), /^Error: film: guard limits may only be lowered: limitMs ≤ 1000 and stopMs ≤ 1500; got 1000 and 2000\.$/);
    assert.throws(() => readGuardLimits({ limitMs: 0 }), /guard limits may only be lowered/);
});

f2('B3: progress — phases in order, every frame counted, ms a frame and ETA from the frames made so far', async () => {
    const seen = [];
    const run = await film({ onProgress: p => seen.push(p) });
    assert.equal(run.result.status, 'completed');
    assert.deepEqual(seen.map(p => p.phase), ['calibrating', 'resetting', ...Array(6).fill('recording'), 'finalizing']);
    assert.ok(seen.every(p => p.frames === 6 && p.background === false));
    assert.deepEqual(seen.slice(0, 2).map(p => [p.frame, p.msPerFrame, p.etaMs, p.elapsedMs]), [[0, null, null, 0], [0, null, null, 0]],
        'before frame 0 there is no rate to estimate from');
    const recording = seen.filter(p => p.phase === 'recording');
    assert.deepEqual(recording.map(p => p.frame), [1, 2, 3, 4, 5, 6]);
    for (const p of recording) {
        assert.equal(p.msPerFrame, p.elapsedMs / p.frame);
        assert.equal(p.etaMs, p.msPerFrame * (6 - p.frame));
    }
    assert.ok(recording.every((p, i) => i === 0 || p.elapsedMs >= recording[i - 1].elapsedMs), 'elapsed time went backwards');
    assert.equal(recording.at(-1).etaMs, 0);
    assert.equal(seen.at(-1).bytes, 6 * 1000, 'bytes is not the encoded payload so far');
});

f2('B3/G5: one slow operation stops a film; an 8-frame batch slow only in total switches to fencing every operation', async () => {
    const clock = fakeClock();
    try {
        // Calibrated at 400 ms a frame (≥ stopMs / 4 = 375 ms): every operation is fenced, and frame 3's 1,600 ms
        // step is not an outlier from a fast calibration, so it stops at once (Q10).
        const slow = await film({ workload: { clock, stepMs: n => (n === 8 ? 1600 : 400) } });
        assert.equal(slow.error?.message, 'film: submission at frame 3 took 1,600 ms, over the 1,500 ms safety stop (a GPU stalled by other work can cause this too; try again on an idle GPU).');
        assert.equal(slow.report.status, 'failed');
        assert.equal(slow.report.guard.fencing, 'every operation');
        assert.equal(slow.report.guard.calibration.msPerFrame, 400);
        assert.equal(slow.report.guard.worstStepMs, 1600);
        released(slow, 'stopped by the guard');
        // Q10: calibrated at 160 ms (< stopMs / 4 = 375 ms), a single 1,600 ms stall at frame 3 is measured once
        // more; frame 4's 160 ms step is under the stop, so the film completes and records the stall.
        const outlier = await film({ workload: { clock, stepMs: n => (n === 8 ? 1600 : 160) } });
        assert.equal(outlier.result?.status, 'completed', String(outlier.error));
        assert.equal(outlier.report.guard.worstStepMs, 1600);
        released(outlier, 'outlier survived');
        // Q10: when the re-measured operation is over the stop too (or draw right after step is over it), it stops.
        const sustained = await film({ workload: { clock, stepMs: n => (n >= 8 ? 1600 : 160) } });
        assert.equal(sustained.error?.message, 'film: submission at frame 4 took 1,600 ms, over the 1,500 ms safety stop (a GPU stalled by other work can cause this too; try again on an idle GPU).');
        assert.equal(sustained.report.status, 'failed');
        released(sustained, 'sustained stall stopped');
        const bothOps = await film({ workload: { clock, stepMs: n => (n === 8 ? 1600 : 160), drawWork: n => (n === 8 ? 1600 : 0) } });
        assert.equal(bothOps.error?.message, 'film: submission at frame 3 took 1,600 ms, over the 1,500 ms safety stop (a GPU stalled by other work can cause this too; try again on an idle GPU).');
        released(bothOps, 'step and draw both over stop');
        // Calibrated at 10 ms; then 190 ms a step: frames 0–7 take 1,520 ms together, past the 1,200 ms congestion
        // signal (and the 1,500 ms stop) but no operation is over the stop, so the film goes on, fenced per operation.
        const batch = await film({ workload: { clock, stepMs: n => (n <= 4 ? 10 : 190) }, film: { ...SMALL, seconds: 0.2 } });
        assert.equal(batch.result?.status, 'completed', String(batch.error));
        assert.equal(batch.report.guard.switchedAt, 8, 'the congested batch did not switch fencing at frame 8');
        assert.equal(batch.report.guard.fencing, 'every operation');
        assert.equal(batch.report.guard.worstStepMs, 190, 'operations after the switch were not measured');
        assert.equal(CONGESTED_BATCH_MS, 1200);
        // Lowered limits refuse at calibration.
        const refused = await film({ workload: { clock, stepMs: () => 60 }, guard: { limitMs: 50 } });
        assert.match(refused.error?.message, /^film: a step at 64×36 took 60 ms, over the 50 ms per-submission limit/);
        assert.equal(refused.report.status, 'failed');
        released(refused, 'refused by calibration');
    } finally {
        clock.restore();
    }
});

// --- B4: readback --------------------------------------------------------------------------------------

f2('B4: FrameReadback unpads 256-byte rows into tight RGBA8, top row first (720, 1080, 1366 wide)', async () => {
    for (const width of [720, 1080, 1366, 64]) {
        const device = filmDevice();
        device.screen.steps = 513;
        const texture = device.createTexture({ size: [width, 3], format: 'rgba8unorm' });
        const readback = new FrameReadback(device, texture, width, 3);
        assert.equal(device.log.buffers[0].size, Math.ceil(width * 4 / 256) * 256 * 3, `${width}: staging rows are not 256-byte aligned`);
        readback.grab(7);
        const { frame, bytes } = await readback.take();
        assert.equal(frame, 7);
        assert.equal(bytes.length, width * 3 * 4);
        assert.deepEqual(bytes, pixels(513, width, 3), `${width}: the padding (0xEE) or a row offset leaked into the frame`);
        readback.destroy();
    }
});

f2('B4: take(out) writes into out and allocates nothing; take() reuses one pooled buffer; misuse is refused by name', async () => {
    const device = filmDevice();
    const texture = device.createTexture({ size: [720, 2], format: 'rgba8unorm' });
    const readback = new FrameReadback(device, texture, 720, 2);
    const out = new Uint8Array(720 * 2 * 4 + 16);
    device.screen.steps = 1; readback.grab(0);
    device.screen.steps = 2; readback.grab(1);
    assert.throws(() => readback.grab(2), /^Error: film: readback: take\(\) a frame before grabbing two more\.$/);
    assert.equal(readback.pending, 2);
    const a = await readback.take(out);
    assert.equal(a.bytes, out, 'take(out) did not hand back out itself');
    assert.equal(shown(out), 1);
    const b = await readback.take();
    device.screen.steps = 3; readback.grab(2);
    const c = await readback.take();
    assert.equal(b.bytes.buffer, c.bytes.buffer, 'take() allocated a new buffer for a frame');
    assert.equal(shown(c.bytes), 3);
    assert.equal(device.log.buffers.length, 3, 'three staging buffers, and no more');
    assert.ok(device.log.buffers.every(s => s.mapState === 'unmapped'), 'a staging buffer was left mapped');
    await assert.rejects(readback.take(), /^Error: film: readback: nothing to take; grab\(\) a frame first\.$/);
    readback.grab(3);
    await assert.rejects(readback.take(new Uint8Array(4)), /^Error: film: readback: take\(out\) needs 5760 bytes for 720×2 RGBA; got 4\.$/);
    assert.throws(() => new FrameReadback(device, { format: 'bgra8unorm' }, 4, 4), /readback reads rgba8unorm frames, top row first; the screen texture is bgra8unorm/);
    readback.destroy();
    readback.destroy();
    assert.ok(device.log.buffers.every(s => s.destroyed === 1), 'a staging buffer was not destroyed exactly once');
    assert.throws(() => readback.grab(4), /readback: grab\(\) after destroy\(\)/);
});

f2('B4: takeVideoFrame builds an RGBX frame on the mapped range at the padded stride, before unmapping it', async () => {
    const device = filmDevice();
    device.screen.steps = 300;
    const texture = device.createTexture({ size: [1366, 2], format: 'rgba8unorm' });
    const readback = new FrameReadback(device, texture, 1366, 2);
    readback.grab(0);
    const { frame, videoFrame } = await readback.takeVideoFrame(16667, 16666);
    assert.equal(frame, 0);
    assert.equal(videoFrame.format, 'RGBX', 'alpha must be ignored (§4.3)');
    assert.deepEqual([videoFrame.timestamp, videoFrame.duration], [16667, 16666]);
    assert.deepEqual(videoFrame.data, pixels(300, 1366, 2), 'the frame was not read at the padded stride');
    assert.equal(device.log.buffers[0].mapState, 'unmapped');
    readback.destroy();
});

// --- B6: the session --------------------------------------------------------------------------------

f2('B6: a session takes each frame at exactly T(j), owns it, and refuses by name what does not fit', async () => {
    const mb = fakeMediabunny();
    const session = await createFilmSession({ film: { width: 64, height: 36, fps: 60, seconds: 0.05 }, mediabunny: mb });
    assert.equal(session.frames, 3);
    assert.equal(session.state, 'initializing');
    await session.add(rgbx(0));
    assert.equal(session.state, 'recording', 'the first frame did not start recording');
    const seconds = rgbx(1, { timestamp: 1 / 60 });
    await assert.rejects(session.add(seconds), /^Error: film: frame 1 must be stamped 16667 µs \(T\(j\) = round\(j × 10⁶ \/ 60\)\); got 0\.0166+\d*\.$/);
    const skipped = rgbx(2);
    await assert.rejects(session.add(skipped), /frame 1 must be stamped 16667 µs .*; got 33333\./);
    await session.add(rgbx(1));
    await session.add(rgbx(2));
    const extra = rgbx(3);
    await assert.rejects(session.add(extra), /^Error: film: the film has all its 3 frames; finish\(\) it\.$/);
    const result = await session.finish();
    assert.equal(result.status, 'completed');
    assert.deepEqual([result.frames, result.name, result.codec, result.file.type], [3, 'film.mp4', 'avc', 'video/mp4']);
    assert.deepEqual(await frames(result.file), [[0, 16667, 0], [16667, 16666, 1], [33333, 16667, 2]]);
    assert.equal(await session.finish(), result, 'a second finish() made another film');
    assert.equal(await session.stop(), result, 'the first terminal call does not win');
    await session.cancel();
    assert.equal(session.state, 'completed');
    const late = rgbx(3);
    await assert.rejects(session.add(late), /^Error: film: the session has ended \(completed\); it takes no more frames\.$/);
    await result.dispose();
    await result.dispose();
    assert.ok([seconds, skipped, extra, late].every(f => f.closes === 1), 'a refused frame was not closed exactly once');
    assert.ok(FakeVideoFrame.made.every(f => f.closes === 1) && mb.log.samples.every(s => s.closes === 1));
    const report = session.report();
    assert.deepEqual(report.states.map(s => s.state), ['probing', 'initializing', 'recording', 'finalizing', 'disposing', 'completed']);
    assert.deepEqual(report.frames, { planned: 3, accepted: 3, encoded: 3, keyFrames: 1 });
    assert.equal(report.achievedBitrate, Math.round(3000 * 8 / (3 / 60)));
    assert.equal(mb.log.outputs[0].tracks[0].options.maximumPacketCount, 3, 'the track does not declare J packets');
});

f2('B6: finish() wants all J frames; stop() keeps a finalized partial; cancel() discards; a cancelled session refuses frames', async () => {
    const open = (more = {}) => createFilmSession({ film: { width: 64, height: 36, fps: 60, seconds: 0.05 }, mediabunny: fakeMediabunny(), ...more });
    const early = await open();
    await early.add(rgbx(0));
    await assert.rejects(early.finish(), /^Error: film: finish\(\) after 1 of 3 frames; add the rest, or stop\(\{ keepPartial: true \}\) to keep a partial film\.$/);
    const partial = await early.stop();
    assert.deepEqual([partial.status, partial.frames, early.state], ['partial', 1, 'partial']);
    assert.deepEqual(await frames(partial.file), [[0, 16667, 0]]);
    await early.cancel();
    assert.equal(early.state, 'partial', 'a cancel after the end changed it');

    const cancelled = await open();
    await cancelled.add(rgbx(0));
    await cancelled.cancel();
    assert.equal(cancelled.state, 'cancelled');
    await assert.rejects(cancelled.add(rgbx(1)), { name: 'AbortError', message: 'Export cancelled' });
    assert.equal(await cancelled.stop(), null);
    assert.equal(cancelled.report().failure, null, 'a cancel is not a failure');

    const empty = await open();
    assert.equal(await empty.stop(), null, 'a partial with no frames was kept');
    assert.equal(empty.state, 'cancelled');
    const refusedFirst = await open();
    await assert.rejects(refusedFirst.add(rgbx(1)), /frame 0 must be stamped 0 µs/);
    assert.equal(refusedFirst.state, 'recording');
    assert.equal(await refusedFirst.stop(), null, 'a recording session with no frames kept a partial');
    const unkept = await open();
    await unkept.add(rgbx(0));
    assert.equal(await unkept.stop({ keepPartial: false }), null);
    const insisted = await open();
    await insisted.add(rgbx(0));
    assert.equal(await insisted.abort(new DOMException('Export cancelled', 'AbortError'), { keepPartial: true }), null, 'a cancel kept a partial');
    assert.equal(insisted.state, 'cancelled');

    const controller = new AbortController();
    const aborted = await open({ signal: controller.signal });
    await aborted.add(rgbx(0));
    controller.abort();
    await assert.rejects(aborted.add(rgbx(1)), { name: 'AbortError' }, 'aborting the signal did not cancel at once');
    await assert.rejects(aborted.finish(), { name: 'AbortError' });
    assert.equal(aborted.state, 'cancelled');
    assert.ok(FakeVideoFrame.made.every(f => f.closes === 1), 'a frame outlived its session');
});

f2('B6: an encoder failure fails the session, and every frame and sample is still closed exactly once', async () => {
    const mb = fakeMediabunny({ failAddAt: 1 });
    const session = await createFilmSession({ film: { width: 64, height: 36, fps: 60, seconds: 0.05 }, mediabunny: mb });
    await session.add(rgbx(0));
    await assert.rejects(session.add(rgbx(1)), /^Error: film: video encoder failed at frame 1: EncodingError: the stand-in encoder failed\.$/);
    await assert.rejects(session.add(rgbx(2)), /video encoder failed at frame 1/, 'a frame after the failure was taken');
    await assert.rejects(session.finish(), /video encoder failed at frame 1/);
    assert.equal(session.state, 'failed');
    assert.equal(mb.log.outputs[0].state, 'canceled');
    assert.ok(FakeVideoFrame.made.every(f => f.closes === 1) && mb.log.samples.every(s => s.closes === 1));
    // Wrapping fails: the sink closes the raw frame it was given.
    const wrap = fakeMediabunny({ failWrapAt: 0 });
    const refused = await createFilmSession({ film: { width: 64, height: 36, fps: 60, seconds: 0.05 }, mediabunny: wrap });
    const frame = rgbx(0);
    await assert.rejects(refused.add(frame), /The stand-in VideoSample refused this frame/);
    assert.equal(frame.closes, 1, 'the raw frame was not closed when wrapping it failed');
    await refused.cancel();
});

// --- B6: a film's teardown, wherever it fails -------------------------------------------------------------

f2('B6: a film that fails anywhere unwinds whole: each resource released once, listeners off, scopes balanced, the device free', async () => {
    // Calibration takes steps and draws 1–4; three warm-up steps are 5–7; frame j is step 8 + j and draw 5 + j.
    const points = [
        ['construct', /^the stand-in workload failed at construct$/],
        ['step:1', /failed at step:1$/],
        ['draw:2', /failed at draw:2$/],
        ['reset', /failed at reset$/],
        ['step:7', /failed at step:7$/],
        ['prepare:2', /failed at prepare:2$/],
        ['draw:7', /failed at draw:7$/],
        ['step:12', /failed at step:12$/],
    ];
    for (const source of ['canvas', 'readback']) {
        for (const [point, pattern] of points) {
            const label = `${source}, ${point}`;
            const run = await film({ source, workload: { fail: point }, film: { ...SMALL, warmupSeconds: 0.05 } });
            assert.match(String(run.error?.message), pattern, label);
            assert.equal(run.report?.status, 'failed', label);
            assert.equal(run.report.failure, run.error.message, `${label}: the report does not carry the film's own error`);
            released(run, label);
            if (point === 'construct') assert.equal(run.workload.deferred, 1, 'a half-built workload\'s registered cleanup did not run');
            else assert.equal(run.workload.deferred, 0, `${label}: a built workload's construction cleanups ran as well as dispose()`);
            const again = await film({ device: run.device, source });
            assert.equal(again.result?.status, 'completed', `${label}: the device was not free for the next film (${again.error})`);
        }
    }
});

f2('B6: when the engine cannot start, the film says so and releases what it made', async () => {
    const device = filmDevice();
    device.createTexture = () => { throw new Error('No textures on this stand-in.'); };
    const quiet = console.error;
    console.error = () => {};
    try {
        const run = await film({ device });
        assert.equal(run.error?.message, 'film: the film\'s engine did not start: No textures on this stand-in.');
        assert.equal(run.report.status, 'failed');
        released(run, 'engine start');
    } finally {
        console.error = quiet;
    }
});

f2('B6: when teardown fails too, the film\'s own error stands, and the teardown failure is reported beside it', async () => {
    const failed = await film({ workload: { fail: ['draw:7', 'dispose'] } });
    assert.match(failed.error.message, /failed at draw:7$/);
    assert.deepEqual(failed.report.cleanupErrors, ['workload: the stand-in workload failed at dispose']);
    released(failed, 'two failures');
    const made = await film({ workload: { fail: 'dispose' } });
    assert.equal(made.result?.status, 'completed', 'a film that was made failed because its teardown did');
    assert.deepEqual(made.result.report.cleanupErrors, ['workload: the stand-in workload failed at dispose']);
});

f2('B6: what onProgress or onReport throws changes nothing; onProgress\'s errors are reported', async () => {
    const reports = [];
    const run = await film({
        onProgress: () => { throw new Error('the progress bar fell over'); },
        onReport: report => { reports.push(report); throw new Error('the report panel fell over'); },
    });
    assert.equal(run.result?.status, 'completed', String(run.error));
    assert.equal(reports.length, 1);
    assert.equal(run.result.report, reports[0]);
    assert.equal(run.result.report.observerErrors.length, 9);
    assert.equal(run.result.report.observerErrors[0], 'the progress bar fell over');
});

f2('B6: every frame and sample is closed once whether its add succeeded, the encoder failed, or wrapping it did', async () => {
    for (const source of ['canvas', 'readback']) {
        FakeVideoFrame.made = [];
        released(await film({ source }), `${source}, made`);
        FakeVideoFrame.made = [];
        const encoder = await film({ source, mb: { failAddAt: 2 } });
        assert.equal(encoder.error?.message, 'film: video encoder failed at frame 2: EncodingError: the stand-in encoder failed.');
        released(encoder, `${source}, encoder failure`);
        FakeVideoFrame.made = [];
        const wrap = await film({ source, mb: { failWrapAt: 1 } });
        assert.match(String(wrap.error?.message), /The stand-in VideoSample refused this frame/);
        released(wrap, `${source}, wrapping failure`);
    }
});

f2('B6: aborting cancels wherever the film is: AbortError, everything discarded, the report says cancelled', async () => {
    // 'finalizing': the abort comes as the runner reports it, before it asks the session to finish (G9 found
    // that this one was ignored and the film completed).
    for (const phase of ['calibrating', 'resetting', 'recording', 'finalizing']) {
        const controller = new AbortController();
        const run = await film({ signal: controller.signal, partial: 'keep-if-finalizable', onProgress: p => { if (p.phase === phase) controller.abort(); } });
        assert.equal(run.error?.name, 'AbortError', phase);
        assert.equal(run.error.message, 'Export cancelled');
        assert.deepEqual([run.report.status, run.report.cancelled, run.report.failure], ['cancelled', true, null], phase);
        released(run, `cancelled while ${phase}`);
    }
    const controller = new AbortController();
    controller.abort();
    const before = await film({ signal: controller.signal });
    assert.equal(before.error?.name, 'AbortError');
    assert.equal(before.report, null, 'a film cancelled before it began has no report');
    assert.equal(before.device.log.buffers.length, 0);
});

f2('B6: a cancel closes the frames still queued behind an add, unencoded', async () => {
    const mb = fakeMediabunny({ addMs: 20 });
    const session = await createFilmSession({ film: SMALL, mediabunny: mb });
    const adds = [0, 1, 2].map(j => session.add(rgbx(j)).catch(error => error));
    await sleep(5); // frame 0 is in the encoder; frames 1 and 2 wait behind it
    await session.cancel();
    const outcomes = await Promise.all(adds);
    assert.equal(mb.log.sources[0].added.length, 1, 'frames queued before the cancel were encoded after it');
    assert.deepEqual(outcomes.map(o => o?.name ?? 'added'), ['added', 'AbortError', 'AbortError']);
    assert.equal(session.state, 'cancelled');
    assert.ok(FakeVideoFrame.made.every(f => f.closes === 1) && mb.log.samples.every(s => s.closes === 1), 'a queued frame was not closed exactly once');
});

/** Node's own collector, exposed at run time (the test runner gives its files no --expose-gc). */
function exposeGC() {
    v8.setFlagsFromString('--expose-gc');
    return vm.runInNewContext('gc');
}

f2('G1 headless: no film keeps its engine alive, whether it was made, failed or cancelled', async () => {
    const collect = exposeGC();
    const device = filmDevice();
    const refs = [];
    const errors = [];
    for (const variant of ['made', 'failed', 'cancelled', 'made again']) {
        const controller = new AbortController();
        // Not counterWorkload: its log holds every engine it was given.
        const createWorkload = async ({ engine }) => {
            refs.push({ variant, ref: new WeakRef(engine) });
            let steps = 0;
            return {
                dt: 1 / 60,
                reset() { steps = 0; },
                step() { if (++steps === 6 && variant === 'failed') throw new Error('the workload failed at its sixth step'); },
                draw() { device.screen.steps = steps; return []; },
                dispose() {},
            };
        };
        try {
            await recordFilm({
                device, createWorkload, film: SMALL, mediabunny: fakeMediabunny(), signal: controller.signal,
                onProgress: p => { if (variant === 'cancelled' && p.phase === 'recording' && p.frame === 2) controller.abort(); },
            });
        } catch (error) {
            errors.push(error); // kept, as an app showing the error would keep it
        }
        // The abort's reason is the app's, made in its own frames: read it, as a logged error would be.
        void controller.signal.reason?.stack;
    }
    assert.deepEqual(errors.map(e => e.message), ['the workload failed at its sixth step', 'Export cancelled']);
    for (let i = 0; i < 3; i++) { await sleep(5); collect(); }
    const kept = refs.filter(r => r.ref.deref() !== undefined).map(r => r.variant);
    assert.deepEqual(kept, [], 'a film\'s engine stayed reachable after its film: a device.lost reaction or an unread error stack holds its scope');
});

f2('G1 headless: a device\'s loss is watched once, however many films it makes', async () => {
    const device = filmDevice();
    const lost = device.lost;
    let reactions = 0;
    device.lost = { then(onLost, onError) { reactions++; return lost.then(onLost, onError); } };
    for (let i = 0; i < 3; i++) assert.equal((await film({ device })).result?.status, 'completed');
    assert.equal(reactions, 1, 'each film added a reaction to device.lost, which keeps its scope alive as long as the device');
});

f2('G9 headless: a loss whose symptom comes first (mapAsync rejects, then device.lost settles) is reported as the loss', async () => {
    const device = filmDevice();
    const run = await film({
        device, source: 'readback', workload: {
            onStep: n => {
                if (n !== 9) return;
                for (const buffer of device.log.buffers.filter(b => b.usage & GPUBufferUsage.MAP_READ)) {
                    buffer.mapAsync = () => Promise.reject(new DOMException('Buffer was unmapped before mapping was resolved.', 'AbortError'));
                }
                setTimeout(() => device.lose('destroyed', 'Device was destroyed.'), 30);
            },
        },
    });
    assert.equal(run.error?.message, 'film: GPU device lost at frame 5 (destroyed: Device was destroyed); partial file discarded.');
    assert.equal(run.report.deviceLost, 'destroyed: Device was destroyed');
    released(run, 'a loss after its symptom');
    // A failure that is not the platform's (the workload's own) does not wait for a loss.
    const t = performance.now();
    const quick = await film({ workload: { fail: 'step:9' } });
    const waited = performance.now() - t;
    assert.match(quick.error?.message, /failed at step:9$/);
    assert.ok(waited < 80, `a workload's own failure took ${Math.round(waited)} ms: it waited for a device loss`);
});

f2('G9 headless: a device lost with its encoder ends the film; the wait on the dead encoder is given up', async () => {
    // The add of frame 3 never returns, as when Chrome's GPU process (and its hardware encoder) is gone, and
    // the device is lost as it begins. Without abandon() the film waits for ever (G9-crash, before the fix).
    const device = filmDevice();
    const t = performance.now();
    const run = await film({ device, mb: { hangAddAt: 3, onHang: () => device.lose('unknown', 'A valid external Instance reference no longer exists.') } });
    const took = performance.now() - t;
    assert.equal(run.error?.message, 'film: GPU device lost at frame 3 (unknown: A valid external Instance reference no longer exists); partial file discarded.');
    assert.ok(took >= LOST_ENCODER_GRACE_MS && took < LOST_ENCODER_GRACE_MS + 1500, `the film ended ${Math.round(took)} ms after it began`);
    assert.deepEqual([run.report.status, run.report.deviceLost], ['failed', 'unknown: A valid external Instance reference no longer exists']);
    assert.equal(run.mb.outputs[0].state, 'canceled', 'the dead encoder\'s output was not cancelled');
    assert.equal(device.log.listeners, 0);
    assert.equal(device.scopeDepth(), 0);
    assert.ok(device.log.buffers.every(b => b.destroyed === 1), 'a buffer was not released after the encoder was given up');
    // The film gave its turn back: the next film on the device is refused at once, the device being lost.
    const next = await film({ device });
    assert.equal(next.error?.message, 'film: GPU device lost before frame 0 (unknown: A valid external Instance reference no longer exists); partial file discarded.');
    // A finalize that never returns, the device lost as it begins, is given up the same way.
    const finalizing = filmDevice();
    const stuck = await film({ device: finalizing, mb: { hangFinalize: true, onHang: () => finalizing.lose('unknown', 'Device was lost.') } });
    assert.equal(stuck.error?.message, 'film: GPU device lost at frame 5 (unknown: Device was lost); partial file discarded.');
    assert.equal(stuck.report.status, 'failed');
});

f2('G9 headless: what is measured across a page freeze is not judged: an operation, a calibration round, a batch', async () => {
    const clock = fakeClock();
    // Node has no document: a stand-in that dispatches Page Lifecycle's freeze and resume.
    const page = new EventTarget();
    globalThis.document = page;
    const freeze = ms => {
        page.dispatchEvent(new Event('freeze'));
        clock.advance(ms);
        page.dispatchEvent(new Event('resume'));
    };
    try {
        // 160 ms a step fences every operation; step 8 (frame 3) spans a 3 s freeze: not judged, and the film goes on.
        const op = await film({ workload: { clock, stepMs: () => 160, onStep: n => { if (n === 8) freeze(3000); } } });
        assert.equal(op.result?.status, 'completed', String(op.error));
        assert.deepEqual([op.result.report.guard.fencing, op.result.report.guard.unjudged, op.result.report.guard.worstStepMs], ['every operation', 1, 160]);
        // A calibration round across a freeze is measured again, not refused.
        const cal = await film({ workload: { clock, stepMs: () => 10, onStep: n => { if (n === 3) freeze(3000); } } });
        assert.equal(cal.result?.status, 'completed', String(cal.error));
        assert.deepEqual(cal.result.report.guard.calibration.stepMs, [10, 10, 10]);
        assert.ok(cal.result.report.guard.calibration.limitations.some(l => /frozen during 1 round/.test(l)));
        // Fenced every 8 frames: a batch across a 2 s freeze is not taken for congestion.
        const batch = await film({ film: { ...SMALL, seconds: 0.2 }, workload: { clock, stepMs: () => 1, onStep: n => { if (n === 9) freeze(2000); } } });
        assert.equal(batch.result?.status, 'completed', String(batch.error));
        assert.deepEqual([batch.result.report.guard.fencing, batch.result.report.guard.switchedAt], ['every 8 frames', null]);
        released(batch, 'frozen batch');
    } finally {
        delete globalThis.document;
        clock.restore();
    }
});

f2('B6: a cancel does not wait out an add the encoder holds: it cancels the encoder first, and its refusal is no failure', async () => {
    const mb = fakeMediabunny({ addMs: 1000, releaseOnCancel: true });
    const session = await createFilmSession({ film: SMALL, mediabunny: mb });
    const adding = session.add(rgbx(0)).catch(error => error);
    await sleep(5);
    const t = performance.now();
    await session.cancel();
    const took = performance.now() - t;
    assert.ok(took < 300, `the cancel waited ${Math.round(took)} ms for the encoder to take the frame`);
    assert.deepEqual([session.state, session.report().failure, session.report().cancelled], ['cancelled', null, true]);
    assert.match(String((await adding)?.message), /closed codec/);
    assert.ok(FakeVideoFrame.made.every(f => f.closes === 1));
    // A failure's discard cancels the encoder first too, and the film's own error stands.
    const failing = fakeMediabunny({ addMs: 1000, releaseOnCancel: true });
    const failed = await createFilmSession({ film: SMALL, mediabunny: failing });
    const held = failed.add(rgbx(0)).catch(error => error);
    await sleep(5);
    assert.equal(await failed.abort(new Error('film: the workload failed')), null);
    await held;
    assert.deepEqual([failed.state, failed.report().failure], ['failed', 'film: the workload failed']);
});

f2('B6: abandon() gives up a wait on an encoder that will not answer, and the session ends discarded', async () => {
    const mb = fakeMediabunny({ hangAddAt: 1 });
    const session = await createFilmSession({ film: SMALL, mediabunny: mb });
    await session.add(rgbx(0));
    const hung = session.add(rgbx(1));
    const queued = session.add(rgbx(2));
    await sleep(5);
    const gone = new Error('film: the encoder is gone');
    assert.equal(await session.abandon(gone), null);
    await assert.rejects(hung, error => error === gone);
    await assert.rejects(queued, error => error === gone);
    assert.deepEqual([session.state, session.report().failure], ['failed', 'film: the encoder is gone']);
    assert.equal(mb.log.outputs[0].state, 'canceled');
    await assert.rejects(session.finish(), error => error === gone);
    assert.equal(await session.stop(), null);
    const queuedFrame = FakeVideoFrame.made[2];
    assert.equal(queuedFrame.closes, 1, 'the frame queued behind the hung add was not closed');
    // Once a session has ended, abandon() changes nothing: a give-up timer can fire during a film's teardown.
    const done = await createFilmSession({ film: { ...SMALL, seconds: 1 / 60 }, mediabunny: fakeMediabunny() });
    await done.add(rgbx(0));
    const made = await done.finish();
    assert.equal(await done.abandon(new Error('film: too late')), made, 'an abandon after the end changed the outcome');
    assert.deepEqual([done.state, done.report().failure], ['completed', null]);
});

f2('B6: partial "keep-if-finalizable" keeps a finalized partial after a failure; "discard", the default, keeps nothing', async () => {
    const kept = await film({ partial: 'keep-if-finalizable', workload: { fail: 'draw:8' } });
    assert.equal(kept.result?.status, 'partial');
    assert.equal(kept.result.frames, 3);
    assert.deepEqual((await frames(kept.result.file)).map(f => f[2]), [1, 2, 3]);
    assert.match(kept.result.report.failure, /failed at draw:8$/);
    assert.equal(kept.result.report.status, 'partial');
    released(kept, 'kept partial');
    // Readback keeps one frame in flight: the partial ends at the last frame handed over.
    const readback = await film({ source: 'readback', partial: 'keep-if-finalizable', workload: { fail: 'draw:8' } });
    assert.equal(readback.result?.frames, 2);
    const discarded = await film({ workload: { fail: 'draw:8' } });
    assert.match(discarded.error?.message, /failed at draw:8$/);
    assert.equal(discarded.report.status, 'failed');
    const nothing = await film({ partial: 'keep-if-finalizable', workload: { fail: 'draw:5' } });
    assert.equal(nothing.result, null, 'a partial with no frames was kept');
    assert.equal(nothing.report.status, 'failed');
});

f2('B6: a lost device or an uncaptured error stops the film before its next submission, in §8\'s words', async () => {
    const device = filmDevice();
    const kept = await film({ device, partial: 'keep-if-finalizable', workload: { onStep: n => { if (n === 9) device.lose(); } } });
    assert.equal(kept.result?.status, 'partial');
    assert.equal(kept.result.report.failure, 'film: GPU device lost at frame 4 (unknown: The stand-in GPU went away); partial file kept.');
    assert.equal(kept.result.report.deviceLost, 'unknown: The stand-in GPU went away');
    assert.equal(kept.result.frames, 4);
    const lost = filmDevice();
    const discarded = await film({ device: lost, workload: { onStep: n => { if (n === 9) lost.lose('unknown', 'Device was lost.'); } } });
    assert.equal(discarded.error?.message, 'film: GPU device lost at frame 4 (unknown: Device was lost); partial file discarded.');
    const early = filmDevice();
    const calibrating = await film({ device: early, workload: { onStep: n => { if (n === 2) early.lose(); } } });
    assert.equal(calibrating.error?.message, 'film: GPU device lost before frame 0 (unknown: The stand-in GPU went away); partial file discarded.');
    released(calibrating, 'lost while calibrating');

    const raising = filmDevice();
    const uncaptured = await film({ device: raising, workload: { onStep: n => { if (n === 9) raising.raise('validation', 'Bind group is invalid.'); } } });
    assert.equal(uncaptured.error?.message, 'film: GPU validation error while building or filming the scene: Bind group is invalid.');
    assert.deepEqual(uncaptured.report.gpuErrors, ['validation: Bind group is invalid.']);
    released(uncaptured, 'uncaptured error');
    const starving = filmDevice();
    const uncapturedOom = await film({ device: starving, workload: { onStep: n => { if (n === 9) starving.raise('out-of-memory', 'Out of memory.'); } } });
    assert.equal(uncapturedOom.error?.message, 'film: GPU out-of-memory error while building or filming the scene: Out of memory.');
    const scoped = filmDevice();
    const oom = await film({ device: scoped, workload: { onStep: n => { if (n === 2) scoped.raise('out-of-memory', 'Not enough memory for the particle buffer.'); } } });
    assert.equal(oom.error?.message, 'film: GPU out-of-memory error while building or filming the scene: Not enough memory for the particle buffer.');
    assert.deepEqual(oom.report.gpuErrors, ['out-of-memory: Not enough memory for the particle buffer.']);
    released(oom, 'scoped error');
});

f2('§4.6: films on one device take turns; on two devices they run side by side; a film waiting its turn can be cancelled', async () => {
    const device = filmDevice();
    const order = [];
    const [a, b] = await Promise.all([
        film({ device, onProgress: p => order.push(`a ${p.phase}`) }),
        film({ device, onProgress: p => order.push(`b ${p.phase}`) }),
    ]);
    assert.ok(a.result && b.result, `${a.error} ${b.error}`);
    assert.ok(order.indexOf('a finalizing') < order.indexOf('b calibrating'), `the second film began before the first had finished: ${order.join(', ')}`);
    const side = [];
    await Promise.all([
        film({ onProgress: p => side.push(`a ${p.phase}`) }),
        film({ onProgress: p => side.push(`b ${p.phase}`) }),
    ]);
    assert.ok(side.indexOf('b calibrating') < side.indexOf('a finalizing'), `films on two devices ran one after the other: ${side.join(', ')}`);
    // A first film of 30 frames that sleep 2 ms a step: the cancelled one must end while it is still running.
    const controller = new AbortController();
    let made = 0;
    const first = film({ device, film: { ...SMALL, seconds: 0.5 }, workload: { stepMs: () => 2 }, onProgress: p => { made = p.frame; } });
    const waiting = film({ device, signal: controller.signal });
    await sleep(0);
    controller.abort();
    assert.equal((await waiting).error?.name, 'AbortError');
    assert.ok(made < 30, `the cancelled film went on waiting for its turn (the first had made ${made} of 30 frames)`);
    assert.equal((await first).result?.status, 'completed', 'cancelling the waiting film disturbed the running one');
    assert.equal((await film({ device })).result?.status, 'completed', 'the cancelled wait left the device locked');
});

// --- B7: reset, steps, timestamps, hashes, the budget -----------------------------------------------------

f2('B7: frame j shows W + s·(j + 1) steps from placement, after an exact reset or a rebuild that discards calibration', async () => {
    for (const reset of [true, false]) {
        for (const fps of [60, 30]) {
            const label = `reset ${reset}, ${fps} fps`;
            const run = await film({ workload: { reset }, film: { width: 64, height: 36, fps, seconds: 0.2, warmupSeconds: 0.1 } });
            const s = 60 / fps, W = 6, J = fps / 5;
            assert.equal(run.result?.status, 'completed', `${label}: ${run.error}`);
            assert.deepEqual((await frames(run.result.file)).map(f => f[2]), Array.from({ length: J }, (_, j) => W + s * (j + 1)), label);
            assert.deepEqual(run.result.report.plan, { frames: J, stepsPerFrame: s, warmupSteps: W, dt: 1 / 60 }, label);
            assert.equal(run.workload.steps, 4 + W + s * J, `${label}: not calibration's 4, W warm-up and s·J filmed steps`);
            assert.deepEqual([run.workload.builds, run.workload.resets, run.workload.disposals], reset ? [1, 1, 1] : [2, 0, 2], label);
            assert.equal(run.workload.engines[0], run.workload.engines.at(-1), `${label}: the rebuild was given another engine`);
            released(run, label);
        }
    }
});

f2('B7: frames carry the camera path\'s view at j / fps, and calibration draws frame 0', async () => {
    const run = await film({ camera: t => ({ target: [0, 0, 0], distance: 10 + t, azimuth: 0, polar: 1 }) });
    const draws = run.workload.draws;
    assert.deepEqual(draws.slice(0, 4).map(d => [d.index, d.view.distance]), [[0, 10], [0, 10], [0, 10], [0, 10]]);
    assert.deepEqual(draws.slice(4).map(d => [d.index, d.seconds, d.view.distance]), Array.from({ length: 6 }, (_, j) => [j, j / 60, 10 + j / 60]));
    assert.ok((await film()).workload.draws.every(d => d.view === null), 'without a camera path a view was invented');
});

f2('B7: both sources stamp frames T(j) and D(j) in µs, and the track declares exactly J packets at the film\'s rate', async () => {
    for (const source of ['readback', 'canvas']) {
        const run = await film({ source, film: { width: 64, height: 36, fps: 30, seconds: 0.3 } });
        assert.equal(run.result?.status, 'completed', String(run.error));
        assert.deepEqual((await frames(run.result.file)).map(f => [f[0], f[1]]), Array.from({ length: 9 }, (_, j) => [frameStamp(j, 30).timestampUs, frameStamp(j, 30).durationUs]), source);
        const [output] = run.mb.outputs;
        assert.deepEqual(output.tracks[0].options, { frameRate: 30, maximumPacketCount: 9 });
        assert.equal(output.format.options.fastStart, 'in-memory');
        const { config } = run.mb.sources[0];
        assert.deepEqual([config.codec, config.quality.bitrate, config.keyFrameInterval, config.latencyMode, config.alpha, config.hardwareAcceleration],
            ['avc', bitrateFor(64, 36, 30), 2, 'quality', 'discard', 'no-preference']);
        assert.deepEqual(run.mb.sources[0].added.map(f => f.format), Array(9).fill(source === 'readback' ? 'RGBX' : 'BGRA'));
        assert.equal(run.result.report.source, source);
        released(run, source);
    }
});

f2('B7: hashes are SHA-256 of each frame\'s exact bytes, equal run to run; a film can be hashed or measured with no video', async () => {
    const a = await film({ output: false, hashes: true });
    const b = await film({ output: false, hashes: true });
    const want = await Promise.all(Array.from({ length: 6 }, (_, j) => sha256(pixels(j + 1, 64, 36))));
    assert.deepEqual(a.result.report.hashes, want);
    assert.deepEqual(b.result.report.hashes, a.result.report.hashes);
    assert.deepEqual([a.result.file, a.result.report.source, a.result.report.sourceReason], [null, 'readback', 'hashes need exact bytes']);
    assert.equal(a.mb.outputs.length, 0, 'a film with no video made a muxer');
    released(a, 'hashes only');
    const both = await film({ hashes: true });
    assert.deepEqual(both.result.report.hashes, want);
    assert.deepEqual((await frames(both.result.file)).map(f => f[2]), [1, 2, 3, 4, 5, 6]);
    const measured = await film({ output: false });
    assert.deepEqual([measured.result.report.source, measured.result.report.frames.accepted, measured.result.report.hashes], ['none', 6, null]);
    assert.ok(!measured.device.log.buffers.some(buffer => buffer.usage & GPUBufferUsage.MAP_READ), 'a film that captures nothing read frames back');
    await assert.rejects(recordFilm({ device: filmDevice(), createWorkload: counterWorkload(filmDevice()).create, film: SMALL, hashes: true, source: 'canvas' }),
        /^Error: film: hashes need exact bytes, and canvas frames are not exact \(§4\.3\): use source "readback" or "auto"\.$/);
});

f2('B7: the memory budget refuses a film predicted over it before frame 0, and stops one that grows past it', async () => {
    const predicted = await film({ output: { bitrate: 8e6, memoryBudgetBytes: 1048576 }, film: { ...SMALL, seconds: 2 } });
    assert.equal(predicted.error?.message, 'film: a 2 s film at 8.0 Mbit/s is expected to take 2.3 MB, over the in-memory budget of 1.0 MB; ' +
        'raise output.memoryBudgetBytes or shorten the film (OPFS streaming is phase F3).');
    assert.equal(predicted.report, null);
    assert.equal(predicted.device.log.buffers.length, 0, 'a refused film made something first');
    const grown = await film({ output: { memoryBudgetBytes: 1048576 }, mb: { packetBytes: 400000 } });
    assert.equal(grown.error?.message, 'film: the film passed its in-memory budget of 1.0 MB at 3 frames (1.2 MB encoded); raise output.memoryBudgetBytes or shorten the film.');
    assert.equal(grown.report.status, 'failed');
    released(grown, 'over budget');
    const opfs = await film({ output: { target: 'opfs' } });
    assert.match(opfs.error?.message, /OPFS and picked-file targets arrive in phase F3/);
});

// --- Q1: the frame source, and the report -------------------------------------------------------------------

f2('Q1: source "auto" films from the offscreen canvas where a frame can be taken from it, and reads back otherwise, saying why', async () => {
    const canvas = await film();
    assert.deepEqual([canvas.result.report.source, canvas.result.report.sourceReason], ['canvas', 'an offscreen canvas works here (Q1)']);
    assert.ok(canvas.device.log.configured.every(c => c.alphaMode === 'premultiplied' && c.format === 'bgra8unorm'), 'not configured as the engine configures');
    FakeVideoFrame.refuseCanvas = 'The canvas is not usable.';
    const fallback = await film();
    assert.equal(fallback.result.report.source, 'readback');
    assert.equal(fallback.result.report.sourceReason,
        'canvas frames are unavailable (a VideoFrame could not be taken from an OffscreenCanvas here (The canvas is not usable)); frames are read back');
    const insisted = await film({ source: 'canvas' });
    assert.equal(insisted.error?.message, 'film: source "canvas" cannot be used here: a VideoFrame could not be taken from an OffscreenCanvas here (The canvas is not usable).');
    assert.deepEqual([insisted.report.status, insisted.report.source], ['failed', null]);
    released(insisted, 'canvas refused');
    FakeVideoFrame.refuseCanvas = null;
    FakeOffscreenCanvas.noWebGPU = true;
    assert.match((await film()).result.report.sourceReason, /an OffscreenCanvas here has no WebGPU context/);
    FakeOffscreenCanvas.noWebGPU = false;
    delete globalThis.OffscreenCanvas;
    assert.match((await film()).result.report.sourceReason, /there is no OffscreenCanvas or VideoFrame here/);
    assert.equal((await film({ source: 'readback' })).result.report.sourceReason, 'asked for');
});

f2('§4.8: the report says what was made, how, on what, and when each state began', async () => {
    const run = await film({ provenance: { scene: 'counter', commit: 'abc123' }, workload: { info: { seed: 7 } } });
    const report = run.result.report;
    assert.equal(run.result.status, 'completed');
    assert.deepEqual(report.states.map(s => s.state), ['probing', 'initializing', 'calibrating', 'resetting', 'recording', 'finalizing', 'disposing', 'completed']);
    assert.deepEqual(report.frames, { planned: 6, accepted: 6, encoded: 6, keyFrames: 1 });
    assert.deepEqual([report.bytes, report.achievedBitrate], [6000, Math.round(6000 * 8 / (6 / 60))]);
    assert.deepEqual(report.encoderConfig, { codec: 'avc1.640028', width: 64, height: 36, hardwareAcceleration: 'no-preference' });
    assert.deepEqual(report.output, { container: 'mp4', codecs: ['avc'], codec: 'avc', bitrate: bitrateFor(64, 36, 60), target: 'memory', memoryBudgetBytes: DEFAULT_MEMORY_BUDGET, name: 'film.mp4' });
    assert.deepEqual(report.adapter, { vendor: 'stand-in', architecture: 'test', description: 'the film stand-in device' });
    assert.equal(report.browser, 'film stand-in');
    assert.deepEqual(report.provenance, { scene: 'counter', commit: 'abc123' });
    assert.deepEqual(report.workload, { seed: 7 });
    assert.deepEqual(report.film, { width: 64, height: 36, fps: 60, seconds: 0.1, warmupSeconds: 0 });
    assert.equal(report.guard.fencing, 'every 8 frames');
    assert.deepEqual([report.guard.switchedAt, report.guard.worstStepMs, report.guard.worstDrawMs], [null, null, null], 'periodic fencing reported per-operation measurements');
    assert.ok(report.timings.recordingMs >= 0 && report.timings.msPerFrame >= 0);
    assert.deepEqual([report.failure, report.cancelled, report.cleanupErrors, report.gpuErrors, report.deviceLost], [null, false, [], [], null]);
    assert.doesNotThrow(() => JSON.stringify(report));
});

f2('B2: recordFilm refuses, by name and before anything is made, options it cannot film', async () => {
    const device = filmDevice();
    const { create } = counterWorkload(device);
    const base = { device, createWorkload: create, film: SMALL, mediabunny: fakeMediabunny() };
    const refuses = (more, pattern) => assert.rejects(recordFilm({ ...base, ...more }), pattern);
    await refuses({ device: null }, /^Error: film: recordFilm: device must be the app's GPUDevice; the film's engine adopts it\.$/);
    await refuses({ createWorkload: null }, /recordFilm: createWorkload must be a function that resolves to the workload/);
    await refuses({ film: { ...SMALL, width: 63 } }, /width and height must be even for video encoding \(yuv420p\); got 63×36/);
    await refuses({ partial: 'keep' }, /partial must be "discard" or "keep-if-finalizable"; got "keep"/);
    await refuses({ source: 'webgl' }, /source must be "auto", "canvas" or "readback"; got "webgl"/);
    await refuses({ camera: 'orbit' }, /camera must be \(seconds\) => view, such as cameraPath\(keyframes\), or omitted/);
    await refuses({ yieldEveryMs: -1 }, /yieldEveryMs must be a number of milliseconds, 0 or more; got -1/);
    await refuses({ guard: { limitMs: 2000 } }, /guard limits may only be lowered/);
    await refuses({ output: { container: 'avi' } }, /output\.container must be "mp4" or "webm"/);
    assert.equal(device.log.buffers.length + device.log.listeners, 0, 'a refusal made something first');
    // A workload that is not one is refused once built, and released.
    const shapeless = await film({ workload: {}, createWorkload: async () => ({ dt: 1 / 60 }) });
    assert.equal(shapeless.error?.message, 'film: createWorkload must resolve to a workload with step(), draw(frame) and dispose() (§4.6).');
    let disposed = 0;
    const timeless = await film({ createWorkload: async () => ({ step() {}, draw: () => [], dispose() { disposed++; } }) });
    assert.equal(timeless.error?.message, 'film: the workload\'s dt must be a number above zero; got undefined.');
    assert.equal(disposed, 1, 'a workload refused for its dt was not disposed');
    const mismatched = await film({ workload: { dt: 1 / 50 } });
    assert.match(mismatched.error?.message, /^film: one frame at 60 fps \(0\.0166+\d* s\) is not a whole number of steps of 0\.02 s\.$/);
    released(mismatched, 'planFilm refused');
});

f2('§6: snapshot draws one exact frame, at any size, stepping nothing, and releases everything', async () => {
    const device = filmDevice();
    const work = counterWorkload(device);
    const shot = await snapshot({ device, createWorkload: work.create, width: 33, height: 7, seconds: 1.5, view: { target: [0, 0, 0], distance: 3, azimuth: 0, polar: 1 } });
    assert.deepEqual([shot.width, shot.height], [33, 7], 'an odd size: a PNG needs no yuv420p rule');
    assert.deepEqual(shot.rgba, pixels(0, 33, 7));
    assert.deepEqual(readPNG(shot.png).rgba, shot.rgba, 'the PNG is not the frame\'s exact bytes');
    assert.equal(work.log.steps, 0, 'the snapshot stepped the workload');
    assert.deepEqual([work.log.draws[0].seconds, work.log.draws[0].view.distance], [1.5, 3]);
    assert.ok(shot.limitations.some(l => l.includes('first-submission risk')));
    assert.ok(device.log.buffers.every(b => b.destroyed === 1) && device.log.textures.every(t => t.destroyed === 1));
    assert.equal(device.scopeDepth(), 0);
    const failing = counterWorkload(device, { fail: 'draw:1' });
    await assert.rejects(snapshot({ device, createWorkload: failing.create, width: 8, height: 8 }), /failed at draw:1$/);
    assert.ok(device.log.buffers.every(b => b.destroyed === 1), 'a failed snapshot leaked a buffer');
    await assert.rejects(snapshot({ device, createWorkload: work.create, width: 0, height: 8 }), /snapshot width must be a whole number above zero; got 0/);
});
