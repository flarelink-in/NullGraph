// D2.1 (docs/offline-films-design.md §5, §13): a film file, read and checked, and the plan it gives: frames,
// steps, the warm-up and the segments. Pure: no node: imports, so the headless tests check every refusal.

export const FILM_DEFAULTS = Object.freeze({
    warmupSeconds: 0,
    segmentSeconds: 10,
    film: Object.freeze({ fps: 60, width: 1920, height: 1080, scale: 1 }),
    encode: Object.freeze({ codec: 'libx264', crf: 16, preset: 'medium', tune: null }),
    camera: Object.freeze([]),
    settings: Object.freeze({}),
});

const fail = message => { throw new Error(`film: ${message}`); };
const whole = (value, what) => (Number.isInteger(value) && value > 0 ? value : fail(`${what} must be a whole number above zero; got ${JSON.stringify(value)}.`));
const positive = (value, what) => (typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : fail(`${what} must be a number above zero; got ${JSON.stringify(value)}.`));
const isObject = value => value !== null && typeof value === 'object' && !Array.isArray(value);

/** The camera keyframes, checked: `t` from 0 and rising; a target of three numbers; a distance above zero. */
export function readCamera(keyframes) {
    if (!Array.isArray(keyframes)) fail('camera must be a list of keyframes.');
    let last = -Infinity;
    return keyframes.map((k, i) => {
        const at = `camera keyframe ${i}`;
        if (!isObject(k)) fail(`${at} must be an object.`);
        if (!(typeof k.t === 'number' && k.t >= 0 && k.t > last)) fail(`${at}: t must be at least 0 and above the keyframe before it; got ${JSON.stringify(k.t)}.`);
        last = k.t;
        if (!(Array.isArray(k.target) && k.target.length === 3 && k.target.every(Number.isFinite))) fail(`${at}: target must be three numbers.`);
        positive(k.distance, `${at}: distance`);
        for (const angle of ['azimuth', 'polar']) {
            if (!Number.isFinite(k[angle])) fail(`${at}: ${angle} must be a number (radians).`);
        }
        return { t: k.t, target: [...k.target], distance: k.distance, azimuth: k.azimuth, polar: k.polar };
    });
}

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
    };
}

/** `seconds / step` as a whole number of steps, or a refusal naming `what`. */
function steps(seconds, step, what) {
    const n = Math.round(seconds / step);
    if (Math.abs(n * step - seconds) > 1e-9 * Math.max(1, seconds)) fail(`${what} (${seconds} s) is not a whole number of steps of ${step} s.`);
    return n;
}

/**
 * What a manifest asks of a scene whose step is `dt` seconds: the frames, steps per frame, warm-up steps,
 * and the segments (each `{ index, first, count }` in frames; the last may be shorter).
 */
export function planFilm(manifest, { dt }) {
    positive(dt, 'the scene\'s step dt');
    const { fps, seconds } = manifest.film;
    const stepsPerFrame = steps(1 / fps, dt, `one frame at ${fps} fps`);
    if (stepsPerFrame < 1) fail(`one frame at ${fps} fps must contain at least one step.`);
    const frames = Math.round(seconds * fps);
    if (Math.abs(frames - seconds * fps) > 1e-9 * Math.max(1, frames)) fail(`film.seconds (${seconds}) is not a whole number of frames at ${fps} fps.`);
    const warmupSteps = steps(manifest.warmupSeconds, dt, 'warmupSeconds');
    const perSegment = Math.round(manifest.segmentSeconds * fps);
    if (perSegment < 1 || Math.abs(perSegment - manifest.segmentSeconds * fps) > 1e-9 * perSegment) {
        fail(`segmentSeconds (${manifest.segmentSeconds}) is not a whole number of frames at ${fps} fps.`);
    }
    const segments = [];
    for (let first = 0, index = 0; first < frames; first += perSegment, index++) segments.push({ index, first, count: Math.min(perSegment, frames - first) });
    return { frames, stepsPerFrame, warmupSteps, perSegment, segments, dt };
}
