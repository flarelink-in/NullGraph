// Pure, worker-safe night-sky math (docs/environment-night-design.md).

import { DEG_TO_RAD, airMassKastenYoung } from './skyMath';

export const NIGHT_MOON_RADIUS_DEG = 1.0;
export const NIGHT_MOON_RADIUS_RAD = NIGHT_MOON_RADIUS_DEG * DEG_TO_RAD;
export const NIGHT_FULL_MOON_RADIANCE = 14.0;

function clamp(x: number, lo: number, hi: number): number {
    return Math.min(hi, Math.max(lo, x));
}

/** Convert catalogue right ascension/declination to +Y-up horizon position (east, up, north). */
export function celestialToHorizon(
    rightAscensionRad: number, declinationRad: number, gmstRad: number,
    latitudeRad: number, longitudeRad: number = 0,
    out: Float32Array | Float64Array | number[] = [0, 0, 0],
): typeof out {
    const hour = gmstRad + longitudeRad - rightAscensionRad;
    const cd = Math.cos(declinationRad), sd = Math.sin(declinationRad);
    const cl = Math.cos(latitudeRad), sl = Math.sin(latitudeRad);
    out[0] = -cd * Math.sin(hour);
    out[1] = sl * sd + cl * cd * Math.cos(hour);
    out[2] = cl * sd - sl * cd * Math.cos(hour);
    return out;
}

/** Astronomical magnitude is logarithmic by definition. */
export function magnitudeToFlux(magnitude: number): number {
    return Math.pow(10, -0.4 * magnitude);
}

/** Opposition surge: about 30% at full phase, rapidly disappearing away from opposition. */
export function lunarOppositionBoost(phaseAngleRad: number): number {
    const a = Math.abs(phaseAngleRad);
    return 1 + 0.3 * Math.exp(-Math.pow(a / (3 * DEG_TO_RAD), 2));
}

/** Scene-relative lunar disc radiance, mirrored by the scalar values uploaded to WGSL. */
export function moonRadiance(
    illumination: number, elevationDeg: number, phaseAngleRad: number,
    nightGain: number = 1,
): number {
    const extinction = Math.exp(-0.12 * Math.max(airMassKastenYoung(elevationDeg) - 1, 0));
    return NIGHT_FULL_MOON_RADIANCE * clamp(illumination, 0, 1)
        * lunarOppositionBoost(phaseAngleRad) / 1.3 * extinction * nightGain;
}

/** Lommel-Seeliger single-scattering term used by the lunar disc. */
export function lommelSeeliger(nDotL: number, nDotV: number): number {
    if (nDotL <= 0) return 0;
    return nDotL / (nDotL + Math.max(nDotV, 1e-3));
}

/** Approximate B-V colour index as a blackbody RGB colour, normalized to max channel 1. */
export function bvToRGB(
    bv: number, out: Float32Array | Float64Array | number[] = [0, 0, 0],
): typeof out {
    const b = clamp(bv, -0.4, 2.0);
    const t = 4600 * (1 / (0.92 * b + 1.7) + 1 / (0.92 * b + 0.62));
    const k = t / 100;
    let r: number, g: number, bl: number;
    r = k <= 66 ? 255 : 329.698727446 * Math.pow(k - 60, -0.1332047592);
    g = k <= 66 ? 99.4708025861 * Math.log(k) - 161.1195681661
        : 288.1221695283 * Math.pow(k - 60, -0.0755148492);
    bl = k >= 66 ? 255 : k <= 19 ? 0 : 138.5177312231 * Math.log(k - 10) - 305.044792731;
    const m = Math.max(r, g, bl, 1);
    out[0] = clamp(r, 0, 255) / m;
    out[1] = clamp(g, 0, 255) / m;
    out[2] = clamp(bl, 0, 255) / m;
    return out;
}

/** Integral of the shipped radial sprite profile in pixel², useful for resolution-invariance tests. */
export function starSpriteEnergy(magnitude: number, radiusPx: number = 1.5, samples: number = 4096): number {
    let sum = 0;
    const step = radiusPx / samples;
    for (let i = 0; i < samples; i++) {
        const r = (i + 0.5) * step;
        const x = r / radiusPx;
        const profile = Math.pow(Math.max(1 - x * x, 0), 2);
        sum += profile * 2 * Math.PI * r * step;
    }
    return sum * magnitudeToFlux(magnitude);
}

/** Deterministic no-asset fallback with a naked-eye-like magnitude distribution. */
export function proceduralStarCatalogue(count: number = 2000, seed: number = 0x51a7): Float32Array {
    const out = new Float32Array(count * 4);
    let state = seed >>> 0;
    const random = () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 4294967296;
    };
    for (let i = 0; i < count; i++) {
        out[i * 4] = random() * Math.PI * 2;
        out[i * 4 + 1] = Math.asin(random() * 2 - 1);
        // More faint stars than bright ones; range approximately [-1.5, 6.5].
        out[i * 4 + 2] = -1.5 + 8 * Math.pow(random(), 0.35);
        out[i * 4 + 3] = -0.3 + 1.9 * random();
    }
    return out;
}