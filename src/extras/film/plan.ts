// D2.4 (offline-films-browser-design.md §4.1, §4.3, §4.4): a film's clock (frames, steps a frame, warm-up
// steps, each frame's timestamp) and its rate and size rules. Time comes from the frame number, never the
// wall clock. Pure and shared: the Node film tool plans from this file too (tools/film/manifest.mjs, through
// dist/film.js), so the browser's films and Node's cannot drift.

import { fail, isObject, positive, whole } from './util';

/** What a plan needs of a film. */
export interface FilmTiming {
    /** Frames a second: a whole number. */
    readonly fps: number;
    /** The film's length: a whole number of frames. */
    readonly seconds: number;
    /** Simulated seconds before frame 0, not filmed: a whole number of steps. Default 0. */
    readonly warmupSeconds?: number;
    /** The Node tool's segment files: when given, the plan also cuts the frames into segments this long. */
    readonly segmentSeconds?: number;
}

/** A run of frames written as one file by the Node tool: frames `first … first + count − 1`. */
export interface FilmSegment {
    readonly index: number;
    readonly first: number;
    readonly count: number;
}

/**
 * The clock of §2: J frames, s steps a frame (each frame, frame 0 included, is drawn after its s steps),
 * and W warm-up steps before frame 0.
 */
export interface FilmPlan {
    readonly frames: number;
    readonly stepsPerFrame: number;
    readonly warmupSteps: number;
    /** Present when `segmentSeconds` was given: frames per segment, and the segments. */
    readonly perSegment?: number;
    readonly segments?: readonly FilmSegment[];
    readonly dt: number;
}

/** `seconds / step` as a whole number of steps, or a refusal naming `what`. */
function steps(seconds: number, step: number, what: string): number {
    const n = Math.round(seconds / step);
    if (Math.abs(n * step - seconds) > 1e-9 * Math.max(1, seconds)) fail(`${what} (${seconds} s) is not a whole number of steps of ${step} s.`);
    return n;
}

/** The plan for a film whose workload steps `dt` seconds at a time. Refuses, by name, what it cannot film. */
export function planFilm(film: FilmTiming, { dt }: { readonly dt: number }): FilmPlan {
    positive(dt, 'the scene\'s step dt');
    const fps = whole(film.fps, 'film.fps');
    const seconds = positive(film.seconds, 'film.seconds');
    const stepsPerFrame = steps(1 / fps, dt, `one frame at ${fps} fps`);
    if (stepsPerFrame < 1) fail(`one frame at ${fps} fps must contain at least one step.`);
    const frames = Math.round(seconds * fps);
    if (Math.abs(frames - seconds * fps) > 1e-9 * Math.max(1, frames)) fail(`film.seconds (${seconds}) is not a whole number of frames at ${fps} fps.`);
    const warmupSeconds = film.warmupSeconds ?? 0;
    if (!(typeof warmupSeconds === 'number' && Number.isFinite(warmupSeconds) && warmupSeconds >= 0)) {
        fail(`warmupSeconds must be 0 or more; got ${JSON.stringify(warmupSeconds)}.`);
    }
    const warmupSteps = steps(warmupSeconds, dt, 'warmupSeconds');
    if (film.segmentSeconds === undefined) return { frames, stepsPerFrame, warmupSteps, dt };
    const segmentSeconds = positive(film.segmentSeconds, 'segmentSeconds');
    const perSegment = Math.round(segmentSeconds * fps);
    if (perSegment < 1 || Math.abs(perSegment - segmentSeconds * fps) > 1e-9 * perSegment) {
        fail(`segmentSeconds (${segmentSeconds}) is not a whole number of frames at ${fps} fps.`);
    }
    const segments: FilmSegment[] = [];
    for (let first = 0, index = 0; first < frames; first += perSegment, index++) segments.push({ index, first, count: Math.min(perSegment, frames - first) });
    return { frames, stepsPerFrame, warmupSteps, perSegment, segments, dt };
}

/** Frame `index`'s place on the film's timeline. */
export interface FrameStamp {
    readonly index: number;
    /** `index / fps`: the time the workload and the camera see. */
    readonly seconds: number;
    /** T(j) = round(j · 10⁶ / fps), in whole microseconds, as WebCodecs takes it. */
    readonly timestampUs: number;
    /** T(j + 1) − T(j): durations add up to the next timestamp exactly, never drifting by truncation. */
    readonly durationUs: number;
}

/**
 * Frame j's timestamp and duration (§4.3): quantized once, the same for canvas and readback frames. A
 * `VideoFrame` takes microseconds; mediabunny's `VideoSample` reads seconds back from it.
 */
export function frameStamp(index: number, fps: number): FrameStamp {
    if (!Number.isSafeInteger(index) || index < 0) fail(`a frame index must be a whole number, 0 or more; got ${JSON.stringify(index)}.`);
    whole(fps, 'fps');
    const at = (j: number) => Math.round(j * 1e6 / fps);
    const timestampUs = at(index);
    return { index, seconds: index / fps, timestampUs, durationUs: at(index + 1) - timestampUs };
}

/**
 * The default bitrate, bits a second: MetaForge's measured rule, w × h × fps × 0.16 (about 20 Mbit/s
 * at 1080p60 and 80 at 4K60). Spike S1 found a hardware encoder that delivers far less at 4K; rate
 * control is the user's parked Q9.
 */
export function bitrateFor(width: number, height: number, fps: number): number {
    return Math.round(whole(width, 'width') * whole(height, 'height') * whole(fps, 'fps') * 0.16);
}

/** The bytes a film is expected to take at `bitrate` (§4.5): 15% over the rate, for the container and slack. */
export function predictBytes(bitrate: number, seconds: number): number {
    return Math.ceil(positive(bitrate, 'bitrate') / 8 * positive(seconds, 'seconds') * 1.15);
}

/** A browser film's size and clock, as the app asks for it. */
export interface FilmSettings {
    readonly width: number;
    readonly height: number;
    readonly fps: number;
    readonly seconds: number;
    readonly warmupSeconds: number;
}

/**
 * The app's `film` option, checked and completed. Its width and height must be even: H.264 and VP9 encode
 * 4:2:0, one chroma sample per 2 × 2 pixels.
 */
export function readFilm(film: unknown): FilmSettings {
    if (!isObject(film)) fail('film must be an object: { width, height, fps, seconds }.');
    const width = whole(film.width, 'film.width');
    const height = whole(film.height, 'film.height');
    if (width % 2 || height % 2) fail(`width and height must be even for video encoding (yuv420p); got ${width}×${height}.`);
    const fps = whole(film.fps, 'film.fps');
    const seconds = positive(film.seconds, 'film.seconds');
    const warmupSeconds = film.warmupSeconds ?? 0;
    if (!(typeof warmupSeconds === 'number' && Number.isFinite(warmupSeconds) && warmupSeconds >= 0)) {
        fail(`warmupSeconds must be 0 or more; got ${JSON.stringify(warmupSeconds)}.`);
    }
    return { width, height, fps, seconds, warmupSeconds };
}
