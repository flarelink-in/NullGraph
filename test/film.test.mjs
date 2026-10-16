// Cycle D, step D2, phase D2.1 (docs/offline-films-design.md §14): the film tool's headless checks.
// `npm run test:film`. U1: the film file and its plan. U4: ffmpeg's arguments and the camera path. And the
// copies' drift check, which film:check and every film's report rely on.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { FILM_DEFAULTS, planFilm, readManifest } from '../tools/film/manifest.mjs';
import { cameraPath } from '../tools/film/camera.mjs';
import { BT709, encoderArgs } from '../tools/film/sinks.mjs';
import { compareCopies } from '../tools/film/check.mjs';
import { readSettings } from '../tools/film/scenes/herd-wolves.mjs';

const DT = 1 / 60;
const minimal = (more = {}) => ({ scene: 'herd-wolves', film: { seconds: 2 }, ...more });
const keyframe = (t, distance = 40, more = {}) => ({ t, target: [0, 0, 0], distance, azimuth: 0.3, polar: 0.4, ...more });

test('U1: a film file gets every default, and its name', () => {
    const m = readManifest(minimal(), 'from-file');
    assert.equal(m.name, 'from-file');
    assert.deepEqual(m.film, { ...FILM_DEFAULTS.film, seconds: 2 });
    assert.deepEqual(m.encode, FILM_DEFAULTS.encode);
    assert.equal(m.warmupSeconds, 0);
    assert.equal(m.segmentSeconds, 10);
    assert.deepEqual(m.camera, []);
    assert.equal(readManifest(minimal({ name: 'given' }), 'from-file').name, 'given');
});

test('U1: every refusal names what is wrong', () => {
    const refuses = (json, pattern) => assert.throws(() => readManifest(json, 'f'), pattern);
    refuses([], /f is not a JSON object/);
    refuses({ film: { seconds: 1 } }, /names no scene/);
    refuses(minimal({ scene: 'Herd Wolves' }), /names no scene/);
    refuses(minimal({ film: {} }), /film\.seconds must be a number above zero; got undefined/);
    refuses(minimal({ film: { seconds: 0 } }), /film\.seconds must be a number above zero; got 0/);
    refuses(minimal({ film: { seconds: 1, fps: 29.97 } }), /film\.fps must be a whole number above zero/);
    refuses(minimal({ film: { seconds: 1, width: 1919 } }), /width and height must be even for H\.264 \(yuv420p\); got 1919×1080/);
    refuses(minimal({ film: { seconds: 1, scale: 0 } }), /film\.scale must be a number above zero/);
    refuses(minimal({ encode: { crf: 52 } }), /encode\.crf must be 0–51; got 52/);
    refuses(minimal({ encode: { codec: '' } }), /encode\.codec must name an ffmpeg encoder/);
    refuses(minimal({ warmupSeconds: -1 }), /warmupSeconds must be 0 or more/);
    refuses(minimal({ segmentSeconds: 0 }), /segmentSeconds must be a number above zero/);
    refuses(minimal({ settings: [] }), /settings must be an object/);
    refuses(minimal({ camera: {} }), /camera must be a list of keyframes/);
    refuses(minimal({ camera: [keyframe(1), keyframe(1)] }), /camera keyframe 1: t must be at least 0 and above the keyframe before it; got 1/);
    refuses(minimal({ camera: [keyframe(-1)] }), /camera keyframe 0: t must be at least 0/);
    refuses(minimal({ camera: [keyframe(0, 40, { target: [0, 0] })] }), /camera keyframe 0: target must be three numbers/);
    refuses(minimal({ camera: [keyframe(0, 0)] }), /camera keyframe 0: distance must be a number above zero/);
    refuses(minimal({ camera: [keyframe(0, 40, { polar: undefined })] }), /camera keyframe 0: polar must be a number \(radians\)/);
});

test('U1: the plan: steps per frame, warm-up, segments', () => {
    const plan = (more, dt = DT) => planFilm(readManifest(minimal(more), 'f'), { dt });
    const p = plan({ film: { seconds: 2.5 }, warmupSeconds: 0.5, segmentSeconds: 1 });
    assert.equal(p.frames, 150);
    assert.equal(p.stepsPerFrame, 1);
    assert.equal(p.warmupSteps, 30);
    assert.deepEqual(p.segments, [{ index: 0, first: 0, count: 60 }, { index: 1, first: 60, count: 60 }, { index: 2, first: 120, count: 30 }]);
    assert.equal(plan({ film: { seconds: 1, fps: 30 } }).stepsPerFrame, 2);
    assert.throws(() => plan({ film: { seconds: 1, fps: 45 } }), /one frame at 45 fps \(0\.0222\d* s\) is not a whole number of steps of 0\.01666\d* s/);
    assert.throws(() => plan({ film: { seconds: 0.01 } }), /film\.seconds \(0\.01\) is not a whole number of frames at 60 fps/);
    assert.throws(() => plan({ warmupSeconds: 0.01 }), /warmupSeconds \(0\.01 s\) is not a whole number of steps/);
    assert.throws(() => plan({ segmentSeconds: 0.001 }), /segmentSeconds \(0\.001\) is not a whole number of frames at 60 fps/);
    assert.throws(() => plan({}, 0), /the scene's step dt must be a number above zero/);
});

test('U1: the Herd & Wolves scene refuses settings it cannot film', () => {
    assert.deepEqual(readSettings({}), { count: 4096, seed: 11, thinkEvery: 3, brains: 'network', look: 'anime', flashes: true });
    assert.throws(() => readSettings({ count: 1000 }), /count must be a whole number of at least 64, a multiple of 16; got 1000/);
    assert.throws(() => readSettings({ brains: 'ort' }), /brains "ort" come later/);
    assert.throws(() => readSettings({ brains: 'teacher' }), /brains must be "network" or "rules"/);
    assert.throws(() => readSettings({ look: 'toon' }), /look must be "anime" or "faithful"/);
    assert.throws(() => readSettings({ zoom: 2 }), /herd-wolves has no setting "zoom"/);
});

test('U4: ffmpeg is fed raw RGBA of the film\'s size, and tags BT.709 in the filter', () => {
    const film = { width: 1920, height: 1080, fps: 60 };
    const args = encoderArgs({ codec: 'libx264', crf: 16, preset: 'medium', tune: 'animation' }, film, 'out.mp4');
    const after = flag => args[args.indexOf(flag) + 1];
    assert.equal(after('-f'), 'rawvideo');
    assert.equal(after('-pix_fmt'), 'rgba');
    assert.equal(after('-s'), '1920x1080');
    assert.equal(after('-r'), '60');
    assert.equal(after('-i'), '-');
    assert.equal(after('-c:v'), 'libx264');
    assert.equal(after('-preset'), 'medium');
    assert.equal(after('-crf'), '16');
    assert.equal(after('-tune'), 'animation');
    assert.equal(after('-vf'), BT709);
    assert.match(BT709, /setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv/);
    assert.equal(args.at(-1), 'out.mp4');
    assert.ok(!encoderArgs({ codec: 'libx264', crf: 16, preset: 'medium', tune: null }, film, 'o.mp4').includes('-tune'));
});

test('U4: the camera path hits its keyframes, eases at the ends, and zooms on a log scale', () => {
    assert.equal(cameraPath([]), null);
    const one = cameraPath([keyframe(0, 50)]);
    assert.deepEqual(one(0), one(100));
    const path = cameraPath([keyframe(0, 10, { azimuth: 0 }), keyframe(10, 1000, { azimuth: 1, target: [10, 0, -5] })]);
    assert.equal(path(0).distance, 10);
    assert.equal(path(10).distance, 1000);
    assert.deepEqual(path(10).target, [10, 0, -5]);
    assert.ok(Math.abs(path(5).distance - 100) < 1e-9, 'half way is the geometric mean');
    assert.ok(Math.abs(path(5).azimuth - 0.5) < 1e-12);
    assert.deepEqual(path(-3), path(0), 'held before the first keyframe');
    assert.deepEqual(path(30), path(10), 'held after the last');
    const slope = (f, t, h = 1e-4) => (f(t + h) - f(t - h)) / (2 * h);
    assert.ok(Math.abs(slope(t => path(t).azimuth, 1e-4)) < 1e-3, 'the path starts at rest');
    // Through an inner keyframe the value moves smoothly: the slope from either side agrees.
    const three = cameraPath([keyframe(0, 10, { azimuth: 0 }), keyframe(4, 10, { azimuth: 2 }), keyframe(10, 10, { azimuth: 2.5 })]);
    const left = (three(4).azimuth - three(4 - 1e-5).azimuth) / 1e-5, right = (three(4 + 1e-5).azimuth - three(4).azimuth) / 1e-5;
    assert.ok(Math.abs(left - right) < 1e-3, `slopes ${left} and ${right} at the inner keyframe`);
    assert.ok(Math.abs(left - 2.5 / 10) < 1e-3, 'the Catmull-Rom slope: through the neighbours');
});

test('the drift check: a copy equal to the site, changed in it, or missing from it', () => {
    const scenes = fileURLToPath(new URL('../tools/film/scenes/', import.meta.url));
    const { from, files } = JSON.parse(readFileSync(join(scenes, 'herd-wolves', 'copies.json'), 'utf8'));
    const site = mkdtempSync(join(tmpdir(), 'film-site-'));
    try {
        mkdirSync(join(site, from), { recursive: true });
        for (const f of files) copyFileSync(join(scenes, 'herd-wolves', f), join(site, from, f));
        const states = () => Object.fromEntries(compareCopies(site).filter(c => c.scene === 'herd-wolves').map(c => [c.file, c.state]));
        assert.ok(Object.values(states()).every(s => s === 'same'), JSON.stringify(states()));
        writeFileSync(join(site, from, files[0]), readFileSync(join(site, from, files[0]), 'utf8') + '\n// changed in the site\n');
        rmSync(join(site, from, files[1]));
        const after = states();
        assert.equal(after[files[0]], 'changed');
        assert.equal(after[files[1]], 'missing in the site');
        assert.ok(files.slice(2).every(f => after[f] === 'same'));
    } finally {
        rmSync(site, { recursive: true, force: true });
    }
});
