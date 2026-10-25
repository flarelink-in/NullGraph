// D2.1 (docs/offline-films-design.md §5, §13): a film file, read and checked, and the plan it gives: frames,
// steps, the warm-up and the segments. Pure: no node: imports, so the headless tests check every refusal.
// The clock and the camera keyframes are null-graph/film's (D2.4, offline-films-browser-design.md §4.1,
// §4.2): this file plans through them, so the browser's films and Node's cannot drift. Build first.

import { planFilm as planTiming, readCamera } from '../../dist/film.js';
import { readOverlays } from './overlay.mjs';

export { readCamera };

export const FILM_DEFAULTS = Object.freeze({
    warmupSeconds: 0,
    segmentSeconds: 10,
    film: Object.freeze({ fps: 60, width: 1920, height: 1080, scale: 1 }),
    encode: Object.freeze({ codec: 'libx264', crf: 16, preset: 'medium', tune: null }),
    camera: Object.freeze([]),
    overlays: Object.freeze([]),
    settings: Object.freeze({}),
});

const fail = message => { throw new Error(`film: ${message}`); };
const whole = (value, what) => (Number.isInteger(value) && value > 0 ? value : fail(`${what} must be a whole number above zero; got ${JSON.stringify(value)}.`));
const positive = (value, what) => (typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : fail(`${what} must be a number above zero; got ${JSON.stringify(value)}.`));
const isObject = value => value !== null && typeof value === 'object' && !Array.isArray(value);

/**
 * A film file's object as a complete manifest, every default filled. `name` names it in errors and is
 * its name when the file gives none. Refuses, by name, anything the plan or the encoder could not use.
 */
export function readManifest(json, name = 'film') {
    if (!isObject(json)) fail(`${name} is not a JSON object.`);
    if (typeof json.scene !== 'string' || !/^[a-z0-9-]+$/.test(json.scene)) fail(`${name} names no scene ("scene": "herd-wolves", say).`);
    const film = { ...FILM_DEFAULTS.film, ...(json.film ?? {}) };
    positive(film.seconds, 'film.seconds');
    whole(film.fps, 'film.fps');
    whole(film.width, 'film.width');
    whole(film.height, 'film.height');
    if (film.width % 2 || film.height % 2) fail(`width and height must be even for H.264 (yuv420p); got ${film.width}×${film.height}.`);
    positive(film.scale, 'film.scale');
    const encode = { ...FILM_DEFAULTS.encode, ...(json.encode ?? {}) };
    if (typeof encode.codec !== 'string' || !encode.codec) fail('encode.codec must name an ffmpeg encoder ("libx264", say).');
    if (!(Number.isFinite(encode.crf) && encode.crf >= 0 && encode.crf <= 51)) fail(`encode.crf must be 0–51; got ${JSON.stringify(encode.crf)}.`);
    const warmupSeconds = json.warmupSeconds ?? FILM_DEFAULTS.warmupSeconds;
    if (!(Number.isFinite(warmupSeconds) && warmupSeconds >= 0)) fail(`warmupSeconds must be 0 or more; got ${JSON.stringify(warmupSeconds)}.`);
    const segmentSeconds = positive(json.segmentSeconds ?? FILM_DEFAULTS.segmentSeconds, 'segmentSeconds');
    if (json.settings !== undefined && !isObject(json.settings)) fail('settings must be an object.');
    return {
        name: typeof json.name === 'string' && json.name ? json.name : name,
        scene: json.scene,
        settings: { ...(json.settings ?? {}) },
        warmupSeconds, segmentSeconds, film, encode,
        camera: readCamera(json.camera ?? []),
        overlays: readOverlays(json.overlays ?? [], film.seconds),
    };
}

/**
 * What a manifest asks of a scene whose step is `dt` seconds: the frames, steps per frame, warm-up steps,
 * and the segments (each `{ index, first, count }` in frames; the last may be shorter).
 */
export function planFilm(manifest, { dt }) {
    const { fps, seconds } = manifest.film;
    return planTiming({ fps, seconds, warmupSeconds: manifest.warmupSeconds, segmentSeconds: manifest.segmentSeconds }, { dt });
}
