// O.2 (docs/film-overlays-design.md §3): burns a film's overlay text into its finished video with ffmpeg's drawtext. The
// filters are overlay.mjs's (pure); this file runs ffmpeg. The simulation and the render are not touched, so a film's
// frames stay exact and the labelled video can be made again, or differently, without rendering.
//
//   node tools/film/burn.mjs <film output directory> [--font <ttf>]
//
// Writes film-labelled.mp4 beside film.mp4, from the "overlays" the directory's film.json recorded.
import { existsSync, readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { FILM_DEFAULTS } from './manifest.mjs';
import { overlayFilters } from './overlay.mjs';
import { BT709, findFfmpeg, qualityArgs } from './sinks.mjs';

export const DEFAULT_FONT = 'C:/Windows/Fonts/segoeui.ttf';

const fail = message => { throw new Error(`film: overlays: ${message}`); };

/** ffmpeg's arguments to draw `filters` over the finished video `input`, with the film's encoder and colour tags. Pure. */
export function burnArgs({ input, output, filters, encode }) {
    return ['-hide_banner', '-loglevel', 'error', '-y', '-i', input, '-vf', [...filters, BT709].join(','),
        '-c:v', encode.codec, ...qualityArgs(encode), '-movflags', '+faststart', output];
}

/** Draws `overlays` over `input` into `output`; `film` gives the size and length. Returns `output`. */
export function burnOverlays({ input, output, film, encode, overlays, font = DEFAULT_FONT, ffmpeg = findFfmpeg() }) {
    if (!overlays.length) fail('no overlays to burn.');
    if (!existsSync(font)) fail(`font ${font} not found (pass another with --font).`);
    if (!ffmpeg) fail('ffmpeg is not on PATH (or NG_FFMPEG).');
    const filters = overlayFilters({ overlays, width: film.width, height: film.height, seconds: film.seconds, font });
    const run = spawnSync(ffmpeg.path, burnArgs({ input, output, filters, encode }), { encoding: 'utf8' });
    if (run.status !== 0) fail(`ffmpeg failed: ${run.stderr.trim().split('\n').slice(-3).join(' ')}`);
    return output;
}

/** A finished film's directory, labelled from the overlays its film.json recorded. Returns the labelled video's path. */
export function burnFilm(dir, options = {}) {
    for (const file of ['film.json', 'film.mp4']) if (!existsSync(join(dir, file))) fail(`no ${file} in "${dir}".`);
    const manifest = JSON.parse(readFileSync(join(dir, 'film.json'), 'utf8'));
    return burnOverlays({ input: join(dir, 'film.mp4'), output: join(dir, 'film-labelled.mp4'), film: manifest.film,
        encode: { ...FILM_DEFAULTS.encode, ...(manifest.encode ?? {}) }, overlays: manifest.overlays ?? [], ...options });
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
    const [dir, ...rest] = process.argv.slice(2);
    const font = rest.indexOf('--font') < 0 ? undefined : rest[rest.indexOf('--font') + 1];
    try { console.log(`film: wrote ${burnFilm(dir, font ? { font } : {})}`); }
    catch (error) { console.error(error.message); process.exit(1); }
}
