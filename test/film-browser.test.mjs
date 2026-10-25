// D2.4, phase F1 (docs/offline-films-browser-design.md §10): null-graph/film's headless checks, against
// dist/film.js. B1: the clock, timestamps, the camera and reframing. B2: refusals that need no GPU. B3: the
// rate and size rules. B5: PNG. B6: the session's skeleton. B8: the entry without mediabunny, its
// declarations in a bare consumer, and the probe against stand-in browsers. `npm run test:film` runs them.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { copyFileSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { crc32 as zlibCrc32 } from 'node:zlib';
import {
    CleanupLedger, DEFAULT_CODECS, DEFAULT_MEMORY_BUDGET, FILM_TRANSITIONS, FilmLifecycle, bitrateFor, cameraPath, checkStorage,
    containAspect, crc32, encodePNG, fitAspect, frameStamp, isTerminal, planFilm, predictBytes, probeFilm, readCamera, readFilm, readOutput,
} from '../dist/film.js';
import { readPNG } from '../tools/film/png-read.mjs';

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
    assert.throws(() => failing.to('completed'), /cannot go from initializing to completed; from initializing it goes to calibrating or disposing/);
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
    const dir = mkdtempSync(join(tmpdir(), 'ng-film-dts-'));
    try {
        mkdirSync(join(dir, 'film'));
        for (const file of readdirSync(join(repo, 'dist/extras/film'))) copyFileSync(join(repo, 'dist/extras/film', file), join(dir, 'film', file));
        const webgpu = join(repo, 'node_modules/@webgpu/types/dist/index.d.ts').replace(/\\/g, '/');
        writeFileSync(join(dir, 'consumer.ts'), `/// <reference path="${webgpu}" />
import { planFilm, cameraPath, fitAspect, probeFilm, readOutput, encodePNG, CleanupLedger, type FilmSupport, type CameraView, type FilmTarget } from './film/index';
const frames: number = planFilm({ fps: 60, seconds: 2 }, { dt: 1 / 60 }).frames;
const view: CameraView | undefined = cameraPath([{ t: 0, target: [0, 0, 0], distance: 1, azimuth: 0, polar: 1 }])?.(0);
const fitted: CameraView = fitAspect({ target: [0, 0, 0], distance: 1, azimuth: 0, polar: 1, fov: 45 }, 9 / 16);
const target: FilmTarget = readOutput({}).target;
async function main(device: GPUDevice): Promise<FilmSupport> {
    await encodePNG(new Uint8Array(4), 1, 1);
    new CleanupLedger().defer('x', () => {});
    return probeFilm({ device, width: 1920, height: 1080, fps: 60 });
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
