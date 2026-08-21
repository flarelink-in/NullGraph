// src/extras/environment/skyMath.ts
//
// Pure, GPU-free sky math — the CPU mirror of the WGSL the SkySource seam emits
// (docs/environment-sky-design.md §5, §10.2). Everything here runs in a plain unit test with no
// device, so the gradient refactor (U1), the Preetham model (U4/U5), and the sun-radiance curve
// (U6) are pinned by numbers, not by eyeballing a render (design-principles §13). Worker-safe:
// plain numbers and typed arrays in, typed arrays out, no allocation in the frame-path helpers.
//
// Shared constants (Kasten–Young, the XYZ→RGB matrix, the Preetham fit) live HERE and are
// interpolated into the generated WGSL by skyShaders.ts — one source of truth, so the CPU mirror
// and the shader cannot disagree about a constant even in principle (U6 pins the rest).

import type { ProceduralSkyConfig } from './types';

// ---------------------------------------------------------------------------
// The sky params header (§4.5) — floats 0..3 of the sky batch's storage buffer
// ---------------------------------------------------------------------------

/** Floats the skybox owns before any source params: [exposure, yaw, blurMip, cheapSky] (§4.5). */
export const SKY_PARAM_FLOATS = 4;

/** Cheap-sky flag values (§4.5): 0 = evaluate sampleSky per pixel, 1 = the prefiltered lookup. */
export const SKY_CHEAP_EVALUATE = 0;
export const SKY_CHEAP_CUBE = 1;

/** Degrees → radians, printed at full precision wherever the WGSL needs it. */
export const DEG_TO_RAD = Math.PI / 180;

// ---------------------------------------------------------------------------
// Sun radiance — the air-mass extinction curve (design §4.2, Phase 2 §5.2)
// ---------------------------------------------------------------------------

/**
 * Kasten–Young 1989 relative optical air mass, `AM(h) = 1 / (sin h + a·(h + b)^-c)` with h in
 * degrees — AM = 1 at the zenith, ≈ 38 at the horizon. The fit is valid to the horizon; below it
 * the elevation is clamped to −2°, which keeps the curve finite and monotone where the model
 * itself stops being meaningful (U5 sweeps down to −20°). Interpolated into the WGSL of every
 * source's skySunRadiance (skyShaders.ts) — change it here and both sides move together.
 */
export const KASTEN_YOUNG: readonly [number, number, number] = [0.50572, 6.07995, 1.6364];

/**
 * Air mass at a solar elevation in degrees (Kasten–Young above). Mirrors the WGSL bodies emitted
 * by skyShaders.ts — the same clamp, the same constants.
 */
export function airMassKastenYoung(elevDeg: number): number {
    const h = Math.max(elevDeg, -2.0);
    const sinH = Math.sin(h * DEG_TO_RAD);
    return 1.0 / (sinH + KASTEN_YOUNG[0] * Math.pow(h + KASTEN_YOUNG[1], -KASTEN_YOUNG[2]));
}

/**
 * The CPU mirror of the WGSL `skySunRadiance` the analytic sources emit (§4.2): the sun disc's
 * colour WITHOUT its angular extent — unit magnitude at the zenith, reddening through the
 * air-mass transmittance `exp(-beta·AM)`. Phase 2's todMath consumes this mirror; Phase 4
 * replaces the body with a transmittance-LUT evaluation behind the same signature.
 */
export function sunRadiance(
    elevDeg: number, beta: ArrayLike<number>,
    out: Float32Array | Float64Array | number[] = [0, 0, 0],
): typeof out {
    const am = airMassKastenYoung(elevDeg);
    out[0] = Math.exp(-beta[0] * am);
    out[1] = Math.exp(-beta[1] * am);
    out[2] = Math.exp(-beta[2] * am);
    return out;
}

function clampf(x: number, lo: number, hi: number): number {
    return Math.min(hi, Math.max(lo, x));
}
function smoothstepf(e0: number, e1: number, x: number): number {
    const t = clampf((x - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// ---------------------------------------------------------------------------
// GradientSky — the pre-refactor skyColor, transcribed (§5.1, U1)
// ---------------------------------------------------------------------------

/**
 * GradientSky's parameter layout in the sky params array, relative to the base
 * (`SKY_PARAM_FLOATS`): 14 floats, the old SkyParams uniform minus its alignment padding.
 * The WGSL in skyShaders.ts reads exactly these indices — keep the two lists together.
 */
export const GRADIENT_SKY_FLOATS = 14;
export const GRADIENT_OFF = {
    ZENITH: 0, SUN_COS: 3, HORIZON: 4, SUN_INTENSITY: 7, GROUND: 8, SUN_DIR: 11,
} as const;

/**
 * The CPU mirror of GradientSky's `sampleSky` — line-for-line the pre-refactor `skyColor` body
 * (the old envShaders.ts procedural kernel), reading the same flat param layout the WGSL reads.
 * U1 pins this against a transcription of the pre-refactor code; T4 pins the GPU image.
 */
export function gradientSkyColor(
    dir: ArrayLike<number>, params: ArrayLike<number>, base: number,
    out: Float32Array | Float64Array | number[] = [0, 0, 0],
): typeof out {
    const t = clampf(dir[1], -1.0, 1.0);
    const kUp = Math.pow(Math.max(t, 0.0), 0.6);
    const kDown = Math.pow(Math.max(-t, 0.0), 0.5);
    const sunDot = -(dir[0] * params[base + GRADIENT_OFF.SUN_DIR]
        + dir[1] * params[base + GRADIENT_OFF.SUN_DIR + 1]
        + dir[2] * params[base + GRADIENT_OFF.SUN_DIR + 2]);
    const sunAmt = smoothstepf(params[base + GRADIENT_OFF.SUN_COS],
        params[base + GRADIENT_OFF.SUN_COS] + 0.0003, sunDot);
    const sun = params[base + GRADIENT_OFF.SUN_INTENSITY] * sunAmt;
    for (let i = 0; i < 3; i++) {
        const horizon = params[base + GRADIENT_OFF.HORIZON + i];
        const zenith = params[base + GRADIENT_OFF.ZENITH + i];
        const ground = params[base + GRADIENT_OFF.GROUND + i];
        let c = horizon + (zenith - horizon) * kUp;
        c = c + (ground - c) * kDown;
        // The disc tint is skySunRadiance for this source: the authored constant (1, 0.98, 0.95).
        const tint = i === 0 ? 1.0 : i === 1 ? 0.98 : 0.95;
        out[i] = c + sun * tint;
    }
    return out;
}

/** GradientSky's skySunRadiance: the authored disc tint, constant by construction (§4.2). */
export const GRADIENT_SUN_TINT: readonly [number, number, number] = [1.0, 0.98, 0.95];

/** Normalize a direction into `out` (zero-safe); returns `out`. No allocation. */
export function normalize3(
    v: ArrayLike<number>, out: Float32Array | Float64Array | number[],
): typeof out {
    const l = Math.hypot(v[0], v[1], v[2]) || 1;
    out[0] = v[0] / l; out[1] = v[1] / l; out[2] = v[2] / l;
    return out;
}

/** Serialize a gradient sky into the flat param layout (no allocation; mirrors writeParams). */
export function gradientWriteParams(
    sky: {
        zenith: ArrayLike<number>; horizon: ArrayLike<number>; ground: ArrayLike<number>;
        sunDir: ArrayLike<number>; sunIntensity: number; sunCos: number;
    },
    out: { [index: number]: number }, base: number,
): void {
    out[base + GRADIENT_OFF.ZENITH] = sky.zenith[0];
    out[base + GRADIENT_OFF.ZENITH + 1] = sky.zenith[1];
    out[base + GRADIENT_OFF.ZENITH + 2] = sky.zenith[2];
    out[base + GRADIENT_OFF.SUN_COS] = sky.sunCos;
    out[base + GRADIENT_OFF.HORIZON] = sky.horizon[0];
    out[base + GRADIENT_OFF.HORIZON + 1] = sky.horizon[1];
    out[base + GRADIENT_OFF.HORIZON + 2] = sky.horizon[2];
    out[base + GRADIENT_OFF.SUN_INTENSITY] = sky.sunIntensity;
    out[base + GRADIENT_OFF.GROUND] = sky.ground[0];
    out[base + GRADIENT_OFF.GROUND + 1] = sky.ground[1];
    out[base + GRADIENT_OFF.GROUND + 2] = sky.ground[2];
    const l = Math.hypot(sky.sunDir[0], sky.sunDir[1], sky.sunDir[2]) || 1;
    out[base + GRADIENT_OFF.SUN_DIR] = sky.sunDir[0] / l;
    out[base + GRADIENT_OFF.SUN_DIR + 1] = sky.sunDir[1] / l;
    out[base + GRADIENT_OFF.SUN_DIR + 2] = sky.sunDir[2] / l;
}

/** The documented default: the old ProceduralSkyConfig's sunCos when none is supplied. */
export const GRADIENT_DEFAULT_SUN_COS = 0.9995;

/** Snapshot the flat fields a GradientSky needs out of a ProceduralSkyConfig. */
export function gradientFromConfig(cfg: ProceduralSkyConfig): {
    zenith: [number, number, number]; horizon: [number, number, number];
    ground: [number, number, number]; sunDir: [number, number, number];
    sunIntensity: number; sunCos: number;
} {
    return {
        zenith: [cfg.zenith[0], cfg.zenith[1], cfg.zenith[2]],
        horizon: [cfg.horizon[0], cfg.horizon[1], cfg.horizon[2]],
        ground: [cfg.ground[0], cfg.ground[1], cfg.ground[2]],
        sunDir: [cfg.sunDir[0], cfg.sunDir[1], cfg.sunDir[2]],
        sunIntensity: cfg.sunIntensity,
        sunCos: cfg.sunCos ?? GRADIENT_DEFAULT_SUN_COS,
    };
}

// ---------------------------------------------------------------------------
// PreethamSky — Preetham et al. 1999 analytic daylight (§5.3)
// ---------------------------------------------------------------------------
//
// The model's FAILURE MODE is documented, not hidden: the fit degrades below roughly 5° solar
// elevation and the per-channel Perez factors can go negative near the horizon at high turbidity.
// sampleSky clamps the result to non-negative (U5 sweeps it); Phase 3 blends twilight over the
// handoff and Phase 4 replaces the model outright.
//
// Layout decision: the A–E distribution coefficients and the zenith terms are derived ON THE CPU
// (below) and ride in the params array, so the per-texel WGSL body is the ~40-flop Perez
// evaluation only (design §6.3: Stage A runs 1.57 M invocations per amortized frame).

/** Preetham params, relative to the base: 30 floats. The WGSL reads exactly these indices. */
export const PREETHAM_SKY_FLOATS = 30;
export const PREETHAM_OFF = {
    SUN_DIR: 0, SUN_COS: 3, DISC_EDGE: 4, DISC_INTENSITY: 5,
    GROUND: 6, BETA: 9, ZENITH_OVER_F0: 12, PEREZ: 15,
} as const;

/**
 * CIE XYZ → linear sRGB (D65). Interpolated into the Preetham WGSL — the CPU mirror and the
 * shader share these nine numbers by construction.
 */
export const XYZ_TO_RGB: readonly [
    number, number, number,
    number, number, number,
    number, number, number,
] = [
    3.2406, -1.5372, -0.4986,
    -0.9689, 1.8758, 0.0415,
    0.0557, -0.2040, 1.0570,
];

/**
 * Preetham's absolute zenith luminance is kcd/m²; the engine is scene-relative
 * (§4.4: 1.0 = a white Lambertian surface in direct midday sun). The conversion gain is chosen
 * so a T = 2.8 midday zenith lands near 0.5 scene units — same order as the shipped gradient's
 * authored zenith. Phase 3's nightGain rides on top of this reference.
 */
export const PREETHAM_SCENE_GAIN = 0.0185;

/**
 * Per-channel Rayleigh optical depth at sea level [red, green, blue] — blue scatters most, which
 * is what makes the extinguished sun redden. Paired with the aerosol slope below; together they
 * are the "three-element optical-depth constant" Phase 2 §5.2 mirrors.
 */
export const PREETHAM_RAYLEIGH: readonly [number, number, number] = [0.040, 0.094, 0.224];

/** Aerosol optical depth added per unit of turbidity above the pristine floor of 1.8. */
export const PREETHAM_AEROSOL_PER_TURBIDITY = 0.05;
export const PREETHAM_TURBIDITY_FLOOR = 1.8;

/** The extinction beta (§4.2's curve) for a turbidity: Rayleigh + aerosol, per channel. */
export function preethamBeta(
    turbidity: number, out: Float32Array | Float64Array | number[] = [0, 0, 0],
): typeof out {
    const aerosol = PREETHAM_AEROSOL_PER_TURBIDITY * Math.max(turbidity - PREETHAM_TURBIDITY_FLOOR, 0.0);
    out[0] = PREETHAM_RAYLEIGH[0] + aerosol;
    out[1] = PREETHAM_RAYLEIGH[1] + aerosol;
    out[2] = PREETHAM_RAYLEIGH[2] + aerosol;
    return out;
}


/**
 * The fifteen Perez distribution coefficients A–E for the Y, x, y channels (Preetham 1999,
 * Table 1), as linear functions of turbidity. Order: AY..EY, Ax..Ex, Ay..Ey — the order the
 * params array carries them and the WGSL reads them.
 */
export function preethamPerezCoeffs(
    turbidity: number, out: Float32Array | Float64Array | number[] = new Array(15).fill(0),
): typeof out {
    const T = turbidity;
    out[0] = 0.1787 * T - 1.4630;   // AY
    out[1] = -0.3554 * T + 0.4275;  // BY
    out[2] = -0.0227 * T + 5.3251;  // CY
    out[3] = 0.1206 * T - 2.5771;   // DY
    out[4] = -0.0670 * T + 0.3703;  // EY
    out[5] = -0.0193 * T - 0.2592;  // Ax
    out[6] = -0.0665 * T + 0.0008;  // Bx
    out[7] = -0.0004 * T + 0.2125;  // Cx
    out[8] = -0.0641 * T - 0.8989;  // Dx
    out[9] = -0.0033 * T + 0.0452;  // Ex
    out[10] = -0.0167 * T - 0.2608; // Ay
    out[11] = -0.0950 * T + 0.0092; // By
    out[12] = -0.0079 * T + 0.2102; // Cy
    out[13] = -0.0441 * T - 1.6537; // Dy
    out[14] = -0.0109 * T + 0.0529; // Ey
    return out;
}

/**
 * The Perez luminance distribution factor F(θ, γ) = (1 + A·e^(B/cosθ))·(1 + C·e^(Dγ) + E·cos²γ),
 * with cosθ clamped to 0.02 — the same clamp the WGSL applies (B/cosθ diverges at the horizon;
 * for the fitted B < 0 the true limit is e^-∞ = 0, so the clamp errs toward a touch more horizon
 * glow — the conservative direction, design-principles §3).
 */
export function perezF(
    cosTheta: number, gamma: number,
    A: number, B: number, C: number, D: number, E: number,
): number {
    const ct = Math.max(cosTheta, 0.02);
    const cg = Math.cos(gamma);
    return (1.0 + A * Math.exp(B / ct)) * (1.0 + C * Math.exp(D * gamma) + E * cg * cg);
}

/**
 * Zenith terms (Yz in SCENE units, xz, yz) at turbidity T and solar zenith angle thetaS
 * (radians). U4 pins Yz monotonically increasing in T and in sin(solar elevation).
 */
export function preethamZenith(
    turbidity: number, thetaS: number,
    out: Float32Array | Float64Array | number[] = [0, 0, 0],
): typeof out {
    const T = turbidity;
    const chi = (4.0 / 9.0 - T / 120.0) * (Math.PI - 2.0 * thetaS);
    out[0] = ((4.0453 * T - 4.9710) * Math.tan(chi) - 0.2155 * T + 2.4192) * PREETHAM_SCENE_GAIN;
    const t3 = thetaS * thetaS * thetaS, t2 = thetaS * thetaS;
    // Zenith chromaticities (Preetham 1999), transcription of the paper's polynomials.
    out[1] = t3 * (0.00166 * T * T - 0.02903 * T + 0.11693)
        + t2 * (-0.00375 * T * T + 0.06377 * T - 0.21196)
        + thetaS * (0.00209 * T * T - 0.03202 * T + 0.06052)
        + (0.00394 * T + 0.25886);
    out[2] = t3 * (0.00275 * T * T - 0.04210 * T + 0.15346)
        + t2 * (-0.00610 * T * T + 0.08970 * T - 0.26756)
        + thetaS * (0.00317 * T * T - 0.04153 * T + 0.06670)
        + (0.00516 * T + 0.26688);
    return out;
}

/**
 * The zenith-normalization terms F_channel(0, θs) — the Perez factor straight up with the sun at
 * θs. Dividing the zenith terms by these (done on the CPU) folds the normalization into the
 * params, so the WGSL never divides per texel.
 */
export function preethamF0(
    perez15: ArrayLike<number>, thetaS: number,
    out: Float32Array | Float64Array | number[] = [0, 0, 0],
): typeof out {
    out[0] = perezF(1.0, thetaS, perez15[0], perez15[1], perez15[2], perez15[3], perez15[4]);
    out[1] = perezF(1.0, thetaS, perez15[5], perez15[6], perez15[7], perez15[8], perez15[9]);
    out[2] = perezF(1.0, thetaS, perez15[10], perez15[11], perez15[12], perez15[13], perez15[14]);
    return out;
}


/** xyY → XYZ → linear RGB for one sky sample; clamps `y` away from zero (U5 finite). */
function preethamXYYToRGB(
    Y: number, x: number, y: number,
    out: Float32Array | Float64Array | number[],
): void {
    const yy = Math.max(y, 1.0e-4);
    const X = (x * Y) / yy;
    const Z = ((1.0 - x - y) * Y) / yy;
    out[0] = XYZ_TO_RGB[0] * X + XYZ_TO_RGB[1] * Y + XYZ_TO_RGB[2] * Z;
    out[1] = XYZ_TO_RGB[3] * X + XYZ_TO_RGB[4] * Y + XYZ_TO_RGB[5] * Z;
    out[2] = XYZ_TO_RGB[6] * X + XYZ_TO_RGB[7] * Y + XYZ_TO_RGB[8] * Z;
}

/**
 * The CPU mirror of PreethamSky's `sampleSky` — line-for-line the WGSL body in skyShaders.ts.
 * Reads the flat param layout `PREETHAM_OFF`; returns finite, non-negative radiance for every
 * unit direction, straight down included (U5). `dir` is a world-space unit vector.
 */
export function preethamSkyColor(
    dir: ArrayLike<number>, params: ArrayLike<number>, base: number,
    out: Float32Array | Float64Array | number[] = [0, 0, 0],
): typeof out {
    // Direction TOWARD the sun (params carry the TRAVEL direction, as everywhere in the engine).
    const sx = -params[base + PREETHAM_OFF.SUN_DIR];
    const sy = -params[base + PREETHAM_OFF.SUN_DIR + 1];
    const sz = -params[base + PREETHAM_OFF.SUN_DIR + 2];

    const cosGamma = clampf(dir[0] * sx + dir[1] * sy + dir[2] * sz, -1.0, 1.0);
    const gamma = Math.acos(cosGamma);
    const cosTheta = dir[1];                 // y-up zenith angle cosine, may be negative
    const ct = Math.max(cosTheta, 0.02);     // Perez is a sky model; clamp at the horizon

    const p = base + PREETHAM_OFF.PEREZ;
    const FY = perezF(ct, gamma, params[p], params[p + 1], params[p + 2], params[p + 3], params[p + 4]);
    const Fx = perezF(ct, gamma, params[p + 5], params[p + 6], params[p + 7], params[p + 8], params[p + 9]);
    const Fy = perezF(ct, gamma, params[p + 10], params[p + 11], params[p + 12], params[p + 13], params[p + 14]);

    const z = base + PREETHAM_OFF.ZENITH_OVER_F0;
    const Y = params[z] * FY;
    const x = params[z + 1] * Fx;
    const y = params[z + 2] * Fy;
    preethamXYYToRGB(Y, x, y, out);

    // Below the horizon, ease toward the ground: albedo × the horizon sky light (continuous).
    if (cosTheta < 0.02) {
        const t = clampf((0.02 - cosTheta) / 0.1, 0.0, 1.0);
        const g0 = base + PREETHAM_OFF.GROUND;
        const gr = params[g0] * out[0], gg = params[g0 + 1] * out[1], gb = params[g0 + 2] * out[2];
        out[0] += (gr - out[0]) * t;
        out[1] += (gg - out[1]) * t;
        out[2] += (gb - out[2]) * t;
    }

    // The analytic sun disc (§5.3): colour from the SAME skySunRadiance the light record uses,
    // so the disc on screen and the directional light cannot drift apart (design §4.2).
    const sunCos = params[base + PREETHAM_OFF.SUN_COS];
    const edge = params[base + PREETHAM_OFF.DISC_EDGE];
    const discAmt = smoothstepf(sunCos - edge, sunCos + edge, cosGamma);
    if (discAmt > 0.0) {
        const beta = [
            params[base + PREETHAM_OFF.BETA],
            params[base + PREETHAM_OFF.BETA + 1],
            params[base + PREETHAM_OFF.BETA + 2],
        ];
        const elevDeg = preethamSunElevDeg(params, base);
        const sr = sunRadiance(elevDeg, beta);
        const intensity = params[base + PREETHAM_OFF.DISC_INTENSITY];
        out[0] += discAmt * intensity * sr[0];
        out[1] += discAmt * intensity * sr[1];
        out[2] += discAmt * intensity * sr[2];
    }

    // The model can dip negative near the horizon at high turbidity — clamp, don't hide (U5).
    out[0] = Math.max(out[0], 0.0);
    out[1] = Math.max(out[1], 0.0);
    out[2] = Math.max(out[2], 0.0);
    return out;
}

/**
 * Solar elevation in degrees implied by the stored sun TRAVEL direction — the argument Phase 2's
 * clock will feed skySunRadiance with. asin(−sunDir.y): −1 → −90° (below horizon), +1 → +90°.
 */
export function preethamSunElevDeg(params: ArrayLike<number>, base: number): number {
    const sy = params[base + PREETHAM_OFF.SUN_DIR + 1];
    return Math.asin(clampf(-sy, -1.0, 1.0)) / DEG_TO_RAD;
}

/** The soft-disc half-width in cosine space that keeps the disc's 50% point at `sunCos`. */
export function preethamDiscEdge(sunCos: number): number {
    return 0.5 * Math.max(1.0 - sunCos, 1.0e-6);
}


