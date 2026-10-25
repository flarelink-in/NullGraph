// D2.4 (offline-films-browser-design.md §4.2): camera keyframes in orbit terms (target, distance, azimuth,
// polar, and optionally the vertical field of view), a smooth path through them, and reframing for other
// aspects. Pure and shared with the Node film tool (tools/film/camera.mjs re-exports cameraPath).
//
// Each value follows a cubic through its keyframes: at an inner keyframe the slope is the Catmull-Rom
// slope (the line through its neighbours), at the first and the last it is zero, so a path starts and
// stops gently. Distance moves on a log scale, so a pull back from 40 m to 4 km looks even. Before the
// first keyframe the camera holds the first; after the last, the last.

import { fail, isObject, positive } from './util';

/** An orbit view. Angles in radians (polar from +Y); `fov` is vertical, in degrees. */
export interface CameraView {
    readonly target: readonly [number, number, number];
    readonly distance: number;
    readonly azimuth: number;
    readonly polar: number;
    /** Absent: the scene's own field of view holds. */
    readonly fov?: number;
}

/** A view at film time `t` seconds. */
export interface CameraKeyframe extends CameraView {
    readonly t: number;
}

/**
 * The keyframes, checked and copied: `t` from 0 and rising; a target of three numbers; a distance above
 * zero; angles that are numbers. `fov`, when any keyframe gives one, must be on every keyframe, above 0
 * and below 180 degrees. The Node tool's films give none; MetaForge's give it on every key.
 */
export function readCamera(keyframes: unknown): CameraKeyframe[] {
    if (!Array.isArray(keyframes)) fail('camera must be a list of keyframes.');
    const withFov = keyframes.length > 0 && isObject(keyframes[0]) && keyframes[0].fov !== undefined;
    let last = -Infinity;
    return keyframes.map((k: unknown, i: number) => {
        const at = `camera keyframe ${i}`;
        if (!isObject(k)) fail(`${at} must be an object.`);
        if (!(typeof k.t === 'number' && k.t >= 0 && k.t > last)) fail(`${at}: t must be at least 0 and above the keyframe before it; got ${JSON.stringify(k.t)}.`);
        last = k.t;
        const target = k.target;
        if (!(Array.isArray(target) && target.length === 3 && target.every(Number.isFinite))) fail(`${at}: target must be three numbers.`);
        const distance = positive(k.distance, `${at}: distance`);
        for (const angle of ['azimuth', 'polar']) {
            if (!Number.isFinite(k[angle])) fail(`${at}: ${angle} must be a number (radians).`);
        }
        const key = { t: k.t, target: [target[0], target[1], target[2]] as const, distance, azimuth: k.azimuth as number, polar: k.polar as number };
        if ((k.fov !== undefined) !== withFov) fail(`${at}: fov must be given on every keyframe or on none.`);
        if (!withFov) return key;
        if (!(typeof k.fov === 'number' && k.fov > 0 && k.fov < 180)) fail(`${at}: fov must be degrees above 0 and below 180; got ${JSON.stringify(k.fov)}.`);
        return { ...key, fov: k.fov };
    });
}

/** Cubic Hermite between (t0, p0, m0) and (t1, p1, m1), at t. */
function hermite(t: number, t0: number, t1: number, p0: number, p1: number, m0: number, m1: number): number {
    const h = t1 - t0, u = (t - t0) / h, u2 = u * u, u3 = u2 * u;
    return (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * h * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * h * m1;
}

/** One value's path through `times` and `values`. */
function track(times: readonly number[], values: readonly number[]): (t: number) => number {
    const n = values.length;
    const slope = values.map((_, k) => (k === 0 || k === n - 1 ? 0 : (values[k + 1] - values[k - 1]) / (times[k + 1] - times[k - 1])));
    return t => {
        if (n === 1 || t <= times[0]) return values[0];
        if (t >= times[n - 1]) return values[n - 1];
        let k = 0;
        while (t > times[k + 1]) k++;
        return hermite(t, times[k], times[k + 1], values[k], values[k + 1], slope[k], slope[k + 1]);
    };
}

/**
 * `(t) => view` through the keyframes (checked by readCamera), or null when there are none: the scene's
 * own view then holds. Each call returns a new object.
 */
export function cameraPath(keyframes: readonly CameraKeyframe[]): ((t: number) => CameraView) | null {
    const keys = readCamera(keyframes);
    if (!keys.length) return null;
    const times = keys.map(k => k.t);
    const tracks = {
        x: track(times, keys.map(k => k.target[0])),
        y: track(times, keys.map(k => k.target[1])),
        z: track(times, keys.map(k => k.target[2])),
        logDistance: track(times, keys.map(k => Math.log(k.distance))),
        azimuth: track(times, keys.map(k => k.azimuth)),
        polar: track(times, keys.map(k => k.polar)),
    };
    const fov = keys[0].fov === undefined ? null : track(times, keys.map(k => k.fov as number));
    const at = (k: CameraKeyframe): CameraView => {
        const view = { target: [k.target[0], k.target[1], k.target[2]] as const, distance: k.distance, azimuth: k.azimuth, polar: k.polar };
        return k.fov === undefined ? view : { ...view, fov: k.fov };
    };
    return t => {
        // On a keyframe, or outside the path, its own values exactly (exp(log(d)) is not always d).
        if (t <= times[0]) return at(keys[0]);
        if (t >= times[times.length - 1]) return at(keys[keys.length - 1]);
        const k = times.indexOf(t);
        if (k >= 0) return at(keys[k]);
        const view = {
            target: [tracks.x(t), tracks.y(t), tracks.z(t)] as const,
            distance: Math.exp(tracks.logDistance(t)),
            azimuth: tracks.azimuth(t),
            polar: tracks.polar(t),
        };
        return fov ? { ...view, fov: fov(t) } : view;
    };
}

/** How fitAspect adapts a view composed for one aspect to a narrower frame. */
export interface FitAspectOptions {
    /** The aspect the keyframes were composed for. Default 16:9. */
    readonly designAspect?: number;
    /** The share of the design frame's width a narrower frame keeps. Default 0.55: a vertical video crops the sides, and a bigger subject reads better on a phone. */
    readonly keepWidth?: number;
    /** Widen the vertical field of view up to this many degrees; past it wide-angle stretch shows, so pull back instead. Default 60. */
    readonly maxFov?: number;
}

const RAD = Math.PI / 180;
/** The vertical field of view assumed for a view that gives none: the engine's `Camera` default. */
const DEFAULT_FOV = 45;

const copyOf = (v: CameraView, change: Partial<CameraView> = {}): CameraView =>
    ({ ...v, target: [v.target[0], v.target[1], v.target[2]] as const, ...change });

/**
 * The view for a frame of `aspect` (width / height), from MetaForge's `fitView`. A frame narrower than the
 * design aspect keeps `keepWidth` of the design width: first by widening the vertical field of view up to
 * `maxFov`, then by moving the camera back. A wider frame keeps the vertical field of view and shows more at
 * the sides. A view with no `fov` is taken as 45° and comes back with one once it had to change.
 */
export function fitAspect(view: CameraView, aspect: number, options: FitAspectOptions = {}): CameraView {
    positive(aspect, 'fitAspect: aspect');
    const designAspect = positive(options.designAspect ?? 16 / 9, 'fitAspect: designAspect');
    const keepWidth = positive(options.keepWidth ?? 0.55, 'fitAspect: keepWidth');
    const maxFov = positive(options.maxFov ?? 60, 'fitAspect: maxFov');
    if (aspect >= designAspect) return copyOf(view);
    const fov = view.fov ?? DEFAULT_FOV;
    const tanV = Math.tan((fov * RAD) / 2);
    const want = (tanV * designAspect * keepWidth) / aspect;
    if (want <= tanV) return copyOf(view);
    const capFov = Math.max(fov, maxFov);
    const cap = Math.tan((capFov * RAD) / 2);
    if (want <= cap) return copyOf(view, { fov: (2 * Math.atan(want)) / RAD });
    return copyOf(view, { fov: capFov, distance: (view.distance * want) / cap });
}

/**
 * A frame of `frameAspect`, shown whole on a canvas of `canvasAspect` (MetaForge's `containView`). On a
 * narrower canvas the vertical field of view widens until the frame's full width fits, so a guide drawn
 * min(width, height × frameAspect) wide marks exactly what the film will contain.
 */
export function containAspect(view: CameraView, frameAspect: number, canvasAspect: number, options: FitAspectOptions = {}): CameraView {
    positive(canvasAspect, 'containAspect: canvasAspect');
    const fitted = fitAspect(view, frameAspect, options);
    if (canvasAspect >= frameAspect) return fitted;
    const tan = (Math.tan(((fitted.fov ?? DEFAULT_FOV) * RAD) / 2) * frameAspect) / canvasAspect;
    return copyOf(fitted, { fov: (2 * Math.atan(tan)) / RAD });
}
