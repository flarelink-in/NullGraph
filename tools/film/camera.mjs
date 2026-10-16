// D2.1 (docs/offline-films-design.md §9): a camera path from keyframes in orbit terms — target, distance,
// azimuth, polar — evaluated at film time. Pure.
//
// Each value follows a cubic through its keyframes: at an inner keyframe the slope is the Catmull-Rom
// slope (the line through its neighbours), at the first and the last it is zero, so a path starts and
// stops gently. Distance moves on a log scale, so a pull back from 40 m to 4 km looks even. Before the
// first keyframe the camera holds the first; after the last, the last.

/** Cubic Hermite between (t0, p0, m0) and (t1, p1, m1), at t. */
function hermite(t, t0, t1, p0, p1, m0, m1) {
    const h = t1 - t0, u = (t - t0) / h, u2 = u * u, u3 = u2 * u;
    return (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * h * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * h * m1;
}

/** One value's path through `times` and `values`. */
function track(times, values) {
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
 * `(t) => { target, distance, azimuth, polar }` through the keyframes (readCamera's), or null when there
 * are none: the scene's own view then holds.
 */
export function cameraPath(keyframes) {
    if (!keyframes.length) return null;
    const times = keyframes.map(k => k.t);
    const tracks = {
        x: track(times, keyframes.map(k => k.target[0])),
        y: track(times, keyframes.map(k => k.target[1])),
        z: track(times, keyframes.map(k => k.target[2])),
        logDistance: track(times, keyframes.map(k => Math.log(k.distance))),
        azimuth: track(times, keyframes.map(k => k.azimuth)),
        polar: track(times, keyframes.map(k => k.polar)),
    };
    const at = k => ({ target: [...k.target], distance: k.distance, azimuth: k.azimuth, polar: k.polar });
    return t => {
        // On a keyframe, or outside the path, its own values exactly (exp(log(d)) is not always d).
        if (t <= times[0]) return at(keyframes[0]);
        if (t >= times[times.length - 1]) return at(keyframes[keyframes.length - 1]);
        const k = times.indexOf(t);
        if (k >= 0) return at(keyframes[k]);
        return {
            target: [tracks.x(t), tracks.y(t), tracks.z(t)],
            distance: Math.exp(tracks.logDistance(t)),
            azimuth: tracks.azimuth(t),
            polar: tracks.polar(t),
        };
    };
}
