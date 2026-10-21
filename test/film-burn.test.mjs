// O.2 (docs/film-overlays-design.md §3): the text pass over a finished film, with the real ffmpeg.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { burnArgs, burnFilm, burnOverlays } from '../tools/film/burn.mjs';
import { findFfmpeg } from '../tools/film/sinks.mjs';

const FONT = 'C:/Windows/Fonts/segoeui.ttf';
const ffmpeg = findFfmpeg();
const film = { width: 320, height: 180, seconds: 2, fps: 10 };
const encode = { codec: 'libx264', crf: 16, preset: 'medium', tune: null };

test('O7 burn arguments: the film\'s own encoder and colour tags, over the finished video', () => {
    const args = burnArgs({ input: 'in.mp4', output: 'out.mp4', filters: ['drawtext=a', 'drawtext=b'], encode });
    assert.deepEqual(args.slice(0, 6), ['-hide_banner', '-loglevel', 'error', '-y', '-i', 'in.mp4']);
    assert.match(args[args.indexOf('-vf') + 1], /^drawtext=a,drawtext=b,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setparams=/);
    assert.deepEqual(args.slice(args.indexOf('-c:v')), ['-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-movflags', '+faststart', 'out.mp4']);
    assert.ok(burnArgs({ input: 'i', output: 'o', filters: [], encode: { ...encode, tune: 'animation' } }).includes('animation'));
    assert.throws(() => burnOverlays({ input: 'i', output: 'o', film, encode, overlays: [], font: FONT, ffmpeg }), /no overlays/);
});

/** The bright pixels of one frame at `t`, from a decoded gray frame. */
const bright = (file, t) => {
    const run = spawnSync(ffmpeg.path, ['-hide_banner', '-loglevel', 'error', '-ss', String(t), '-i', file, '-frames:v', '1',
        '-f', 'rawvideo', '-pix_fmt', 'gray', '-'], { maxBuffer: 1 << 24 });
    assert.equal(run.status, 0, String(run.stderr));
    return run.stdout.reduce((n, v) => n + (v > 128 ? 1 : 0), 0);
};

test('O8 burnOverlays: text appears in its time range and nowhere else', { skip: !ffmpeg || !existsSync(FONT) }, () => {
    const dir = mkdtempSync(join(tmpdir(), 'film-burn-'));
    try {
        const input = join(dir, 'film.mp4'), output = join(dir, 'film-labelled.mp4');
        const make = spawnSync(ffmpeg.path, ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i', `color=c=black:s=320x180:r=10:d=2`,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', input]);
        assert.equal(make.status, 0, String(make.stderr));
        assert.equal(bright(input, 0.5), 0, 'the source video is black');
        const overlays = [{ text: 'Title: it\'s here, now', size: 28, from: 0.2, to: 1 }];
        assert.equal(burnOverlays({ input, output, film, encode, overlays, font: FONT, ffmpeg }), output);
        assert.ok(bright(output, 0.6) > 100, 'text inside its range');
        assert.equal(bright(output, 1.5), 0, 'no text after its range');
        assert.equal(bright(output, 0), 0, 'no text before its range');
        assert.throws(() => burnOverlays({ input, output, film, encode, overlays, font: join(dir, 'missing.ttf'), ffmpeg }), /font .* not found/);
    } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('O9 burnFilm: a finished film directory is labelled from its own film.json', { skip: !ffmpeg || !existsSync(FONT) }, () => {
    const dir = mkdtempSync(join(tmpdir(), 'film-burn-'));
    try {
        assert.throws(() => burnFilm(dir, { font: FONT }), /no film\.json/);
        writeFileSync(join(dir, 'film.json'), JSON.stringify({ film, encode, overlays: [] }));
        assert.throws(() => burnFilm(dir, { font: FONT }), /no film\.mp4/);
        writeFileSync(join(dir, 'film.mp4'), '');
        assert.throws(() => burnFilm(dir, { font: FONT }), /no overlays/);
    } finally { rmSync(dir, { recursive: true, force: true }); }
});
