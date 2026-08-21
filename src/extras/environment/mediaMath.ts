// src/extras/environment/mediaMath.ts
//
// Phase 5 — participating media, unified (docs/environment-aerial-design.md §6).
//
// One medium set, one integration. Rayleigh, Mie, ozone and height fog are four density profiles
// sharing a single extinction and a single in-scatter accumulation — NOT four systems. Shipping
// them separately double-counts extinction (§6.2), a bug that is only wrong when two are enabled
// and therefore survives every test that enables one.
//
// This is the CPU mirror of `aerialShaders.buildAerialVolumeWGSL`. It is a SECOND IMPLEMENTATION
// of the same algorithm and it drifts (design-principles §13.1), so it is pinned by invariants —
// slice composition (U3), the analytic closed form (U9), monotonicity (U1) — rather than by
// reading it side by side with the kernel.
//
// Units, stated once because a mismatch here is silent (§13.1): DISTANCES ARE KILOMETRES and
// COEFFICIENTS ARE m^-1, exactly as Phase 4 defines them. The x1000 conversion happens at the
// integration boundary and nowhere else. Callers work in world units and convert through
// `unitsPerKm`.

import {
    type AtmosphereMedium, type RGB, EARTH_ATMOSPHERE,
    cornetteShanksPhase, mieDensity, ozoneDensity, rayleighDensity, rayleighPhase,
    transmittance as atmosphereTransmittance,
} from './atmosphereMath';

// ---------------------------------------------------------------------------
// §6.1 — the medium set. Height fog is Mie with different numbers; it is a row, not a feature.
// ---------------------------------------------------------------------------

/** An exponential height-fog medium: `sigma0 * exp(-(h - h0) / H_f)` (§6.1). */
export interface HeightFogMedium {
    /** Scattering coefficient at `referenceHeight`, m^-1, per channel. */
    scattering: RGB;
    /** Absorption coefficient at `referenceHeight`, m^-1, per channel. */
    absorption: RGB;
    /** `h0` — the altitude in km at which the coefficients above hold. */
    referenceHeight: number;
    /** `H_f` — e-folding height in km. Small (0.1–1) is ground fog; large is uniform haze. */
    scaleHeight: number;
    /** Cornette-Shanks anisotropy in (-1, 1). Fog is forward-scattering; 0.6 is a good default. */
    g: number;
}

/** A ground-hugging fog: dense, shallow, strongly forward-scattering. */
export const DEFAULT_HEIGHT_FOG: Readonly<HeightFogMedium> = Object.freeze({
    scattering: [2.0e-5, 2.0e-5, 2.0e-5] as RGB,
    absorption: [1.0e-6, 1.0e-6, 1.0e-6] as RGB,
    referenceHeight: 0,
    scaleHeight: 0.3,
    g: 0.6,
});

/**
 * Everything the integration marches through. `atmosphere` contributes Rayleigh + Mie + ozone
 * (Phase 4's rows); `fog` contributes the height-fog row. Either may be absent; both present is
 * the case §6.2 exists for, and it is handled by summing coefficients BEFORE the exponential —
 * never by integrating twice and multiplying the results.
 */
export interface AerialMedia {
    atmosphere?: AtmosphereMedium | null;
    fog?: HeightFogMedium | null;
}

/** Sun state the in-scatter integration needs. */
export interface AerialSun {
    /** The sun's TRAVEL direction (where the light goes) — the engine-wide convention. */
    direction: ArrayLike<number>;
    /** Linear radiance scale applied to the sun's contribution. */
    intensity: number;
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/**
 * How the W axis maps to view distance (§4.2). The two media want opposite things, so this is a
 * config choice rather than a constant: `'linear'` spends resolution at range (atmospheric haze),
 * `'log'` packs it near the camera (ground fog, interiors).
 */
export type FroxelDistribution = 'linear' | 'log';

export interface AerialConfig {
    /** Froxel volume dimensions. Each axis must be a multiple of 4 (§10). Default [32, 32, 32]. */
    dims?: [number, number, number];
    /**
     * View distance the volume covers, in WORLD units. Beyond it the volume clamps to its last
     * slice, which is correct: the medium's transmittance is already near its asymptote (§4.2).
     * Default 32000 (= 32 km at the default `unitsPerKm`).
     */
    maxDistance?: number;
    /** Where the volume starts — the camera near plane, in world units. Default 0.1. */
    nearDistance?: number;
    /** W-axis mapping (§4.2). Default `'linear'`. */
    distribution?: FroxelDistribution;
    /** World units in one kilometre. Default 1000 (world units are metres). */
    unitsPerKm?: number;
    /** The world Y that is altitude 0 in the medium. Default 0. */
    groundLevel?: number;
    /**
     * March sub-steps per slice. 1 makes the whole volume cost exactly one ray per (x, y) with
     * `dims[2]` accumulation steps — the §4.3 cost argument. Default 1.
     */
    stepsPerSlice?: number;
    /**
     * Reject froxel samples whose depth disagrees with the pixel's, the mitigation for §4.3's
     * silhouette bleeding. Only the fullscreen apply pass can do this — it owns the depth buffer
     * and sees a neighbourhood; the injected path sees one fragment (§5.1). Default true.
     */
    depthAwareUpsampling?: boolean;
    /** Depth agreement tolerance for the above, as a fraction of view distance. Default 0.05. */
    upsampleDepthTolerance?: number;
    /** Linear radiance scale on the sun's in-scatter contribution. Default 1. */
    sunIntensity?: number;
    /** The bind group the injected layout occupies — must match the LightSystem's. Default 2. */
    aerialGroup?: number;
}

export type ResolvedAerialConfig = Required<AerialConfig>;

export function resolveAerialConfig(config: AerialConfig = {}): ResolvedAerialConfig {
    const resolved: ResolvedAerialConfig = {
        dims: config.dims ? [...config.dims] as [number, number, number] : [32, 32, 32],
        maxDistance: config.maxDistance ?? 32000,
        nearDistance: config.nearDistance ?? 0.1,
        distribution: config.distribution ?? 'linear',
        unitsPerKm: config.unitsPerKm ?? 1000,
        groundLevel: config.groundLevel ?? 0,
        stepsPerSlice: config.stepsPerSlice ?? 1,
        depthAwareUpsampling: config.depthAwareUpsampling ?? true,
        upsampleDepthTolerance: config.upsampleDepthTolerance ?? 0.05,
        sunIntensity: config.sunIntensity ?? 1,
        aerialGroup: config.aerialGroup ?? 2,
    };
    validateAerialConfig(resolved);
    return resolved;
}

export function validateAerialConfig(c: ResolvedAerialConfig): void {
    const [x, y, z] = c.dims;
    if (![x, y, z].every(d => Number.isInteger(d) && d > 0 && d % 4 === 0)) {
        throw new Error(`AerialPerspective: volume dimensions ${x}x${y}x${z} must be multiples of 4.`);
    }
    if (!(c.nearDistance > 0)) {
        throw new Error(`AerialPerspective: nearDistance ${c.nearDistance} must be positive; the 'log' distribution divides by it.`);
    }
    if (!(c.maxDistance > c.nearDistance)) {
        throw new Error(`AerialPerspective: maxDistance ${c.maxDistance} must exceed the camera near plane ${c.nearDistance}.`);
    }
    if (c.distribution !== 'linear' && c.distribution !== 'log') {
        throw new Error(`AerialPerspective: distribution "${c.distribution}" is not 'linear' or 'log'.`);
    }
    if (!(c.unitsPerKm > 0)) {
        throw new Error(`AerialPerspective: unitsPerKm ${c.unitsPerKm} must be positive.`);
    }
    if (!Number.isInteger(c.stepsPerSlice) || c.stepsPerSlice < 1) {
        throw new Error(`AerialPerspective: stepsPerSlice ${c.stepsPerSlice} must be a positive integer.`);
    }
}

export function validateHeightFog(fog: HeightFogMedium): void {
    for (const [name, channels] of [['scattering', fog.scattering], ['absorption', fog.absorption]] as const) {
        for (let i = 0; i < 3; i++) {
            if (!Number.isFinite(channels[i]) || channels[i] < 0) {
                throw new Error(`HeightFog: ${name}[${i}] ${channels[i]} is negative; extinction must be non-negative or transmittance exceeds 1.`);
            }
        }
    }
    if (!(fog.scaleHeight > 0)) {
        throw new Error(`HeightFog: scaleHeight ${fog.scaleHeight} must be positive.`);
    }
    if (!(fog.g > -1 && fog.g < 1)) {
        throw new Error(`HeightFog: g ${fog.g} is outside (-1, 1); Cornette-Shanks is singular at the endpoints.`);
    }
}

// ---------------------------------------------------------------------------
// §4.2 — the W axis, and its exact inverse (U4)
// ---------------------------------------------------------------------------

/** Normalized slice coordinate `w` in [0, 1] to view distance in world units. */
export function froxelDistance(w: number, near: number, maxDistance: number,
    distribution: FroxelDistribution = 'linear'): number {
    const t = w < 0 ? 0 : (w > 1 ? 1 : w);
    return distribution === 'log'
        ? near * Math.pow(maxDistance / near, t)
        : near + (maxDistance - near) * t;
}

/** The exact inverse of {@link froxelDistance}. Beyond `maxDistance` it clamps (§4.2). */
export function froxelW(distance: number, near: number, maxDistance: number,
    distribution: FroxelDistribution = 'linear'): number {
    const d = distance < near ? near : distance;
    const w = distribution === 'log'
        ? Math.log(d / near) / Math.log(maxDistance / near)
        : (d - near) / (maxDistance - near);
    return w < 0 ? 0 : (w > 1 ? 1 : w);
}

/**
 * View distance stored by texel `z` — the texel CENTRE, so a lookup at `froxelW(d)` lands exactly
 * on the texel that integrated to `d` and trilinear filtering interpolates between neighbouring
 * accumulations. Getting this half-texel wrong shifts the whole fog toward or away from the
 * camera by one slice, which reads as "the fog density is off" rather than as an indexing bug.
 */
export function froxelSliceDistance(z: number, c: ResolvedAerialConfig): number {
    return froxelDistance((z + 0.5) / c.dims[2], c.nearDistance, c.maxDistance, c.distribution);
}

// ---------------------------------------------------------------------------
// §6.1 — density profiles and the COMBINED coefficients
// ---------------------------------------------------------------------------

/** `exp(-(h - h0) / H_f)`, the height-fog density profile. */
export function heightFogDensity(heightKm: number, fog: HeightFogMedium): number {
    return Math.exp(-(heightKm - fog.referenceHeight) / fog.scaleHeight);
}

/**
 * Total extinction at `heightKm`, m^-1 per channel — every active medium summed.
 *
 * This summation is the whole of §6.2. `exp(-(sigma_a + sigma_b) * ds)` is not the bug;
 * `exp(-sigma_a * ds) * exp(-sigma_b * ds)` applied by two independent systems that each also add
 * their own in-scatter IS. The two forms agree on transmittance and disagree on radiance.
 */
export function combinedExtinction(heightKm: number, media: AerialMedia,
    out: Float32Array = new Float32Array(3)): Float32Array {
    out[0] = out[1] = out[2] = 0;
    const atmosphere = media.atmosphere;
    if (atmosphere) {
        const dR = rayleighDensity(heightKm, atmosphere);
        const dM = mieDensity(heightKm, atmosphere);
        const dO = ozoneDensity(heightKm, atmosphere);
        const mieExtinction = (atmosphere.mieScattering + atmosphere.mieAbsorption) * dM;
        for (let c = 0; c < 3; c++) {
            out[c] += atmosphere.rayleighScattering[c] * dR + mieExtinction + atmosphere.ozoneAbsorption[c] * dO;
        }
    }
    const fog = media.fog;
    if (fog) {
        const dF = heightFogDensity(heightKm, fog);
        for (let c = 0; c < 3; c++) out[c] += (fog.scattering[c] + fog.absorption[c]) * dF;
    }
    return out;
}

/**
 * Total scattering at `heightKm`, m^-1 per channel, WITHOUT the phase function — the coefficient
 * a multiple-scattering or ambient term multiplies.
 */
export function combinedScattering(heightKm: number, media: AerialMedia,
    out: Float32Array = new Float32Array(3)): Float32Array {
    out[0] = out[1] = out[2] = 0;
    const atmosphere = media.atmosphere;
    if (atmosphere) {
        const dR = rayleighDensity(heightKm, atmosphere);
        const mie = atmosphere.mieScattering * mieDensity(heightKm, atmosphere);
        for (let c = 0; c < 3; c++) out[c] += atmosphere.rayleighScattering[c] * dR + mie;
    }
    const fog = media.fog;
    if (fog) {
        const dF = heightFogDensity(heightKm, fog);
        for (let c = 0; c < 3; c++) out[c] += fog.scattering[c] * dF;
    }
    return out;
}

/**
 * Total scattering at `heightKm` weighted by each medium's own phase function at `cosTheta` — the
 * single-scattering source term. Each row carries its own phase, which is exactly why they cannot
 * be collapsed into one coefficient before the phase is applied.
 */
export function combinedPhaseScattering(heightKm: number, cosTheta: number, media: AerialMedia,
    out: Float32Array = new Float32Array(3)): Float32Array {
    out[0] = out[1] = out[2] = 0;
    const atmosphere = media.atmosphere;
    if (atmosphere) {
        const rayleigh = rayleighDensity(heightKm, atmosphere) * rayleighPhase(cosTheta);
        const mie = atmosphere.mieScattering * mieDensity(heightKm, atmosphere)
            * cornetteShanksPhase(cosTheta, atmosphere.mieG);
        for (let c = 0; c < 3; c++) out[c] += atmosphere.rayleighScattering[c] * rayleigh + mie;
    }
    const fog = media.fog;
    if (fog) {
        const dF = heightFogDensity(heightKm, fog) * cornetteShanksPhase(cosTheta, fog.g);
        for (let c = 0; c < 3; c++) out[c] += fog.scattering[c] * dF;
    }
    return out;
}

// ---------------------------------------------------------------------------
// §6.2 — the analytic single-medium integral. This IS the v1.2 exponential height fog:
// closed form, no depth texture, no volume, shippable standalone and fragment-side.
// ---------------------------------------------------------------------------

/**
 * Optical depth through one exponential medium over `distanceKm`, in closed form.
 *
 * With `h(s) = h0 + k*s` (planar; `k` is the ray's vertical component):
 *
 *   tau = sigma * exp(-(h0 - href)/H) * (H/k) * (1 - exp(-k*D/H))     for k != 0
 *   tau = sigma * exp(-(h0 - href)/H) * D                             for k == 0
 *
 * The `k == 0` branch is not a degenerate special case to be avoided — a horizontal ray through
 * ground fog is the single most common configuration there is.
 */
export function analyticFogOpticalDepth(altitudeKm: number, dirY: number, distanceKm: number,
    fog: HeightFogMedium, out: Float32Array = new Float32Array(3)): Float32Array {
    const H = fog.scaleHeight;
    const base = Math.exp(-(altitudeKm - fog.referenceHeight) / H);
    // The k -> 0 limit of (H/k)(1 - e^(-kD/H)) is D; switch below the scale at which the
    // subtraction loses its significant digits rather than at exactly zero.
    const k = dirY;
    const lengthKm = Math.abs(k * distanceKm / H) < 1e-4
        ? distanceKm
        : (H / k) * (1 - Math.exp(-k * distanceKm / H));
    const columnMetres = base * lengthKm * 1000;
    for (let c = 0; c < 3; c++) out[c] = (fog.scattering[c] + fog.absorption[c]) * columnMetres;
    return out;
}

/** `exp(-tau)` for {@link analyticFogOpticalDepth}. */
export function analyticFogTransmittance(altitudeKm: number, dirY: number, distanceKm: number,
    fog: HeightFogMedium, out: Float32Array = new Float32Array(3)): Float32Array {
    analyticFogOpticalDepth(altitudeKm, dirY, distanceKm, fog, out);
    for (let c = 0; c < 3; c++) out[c] = Math.exp(-out[c]);
    return out;
}

/**
 * In-scattered radiance through one exponential medium, in closed form.
 *
 * Scattering and absorption share the density profile, so the single-scattering albedo
 * `alpha = sigma_s / sigma_t` is constant along the ray and
 *
 *   L = E * p * alpha * integral(T(s) * sigma_t(s) ds) = E * p * alpha * (1 - T(D))
 *
 * — the familiar `fogColor * (1 - T)`. `sunIrradiance` is the sun's contribution after its own
 * transmittance, held CONSTANT along the ray; that constancy is the only approximation separating
 * this path from the froxel volume, and it is why it is cheap. The phase factor is not folded out
 * into a 4*PI: the march's source term is `sigma_s * p`, so this one must be too, or the two
 * disagree by 4*PI and U9 fails for a reason that looks like a coefficient error.
 */
export function analyticFogInScatter(altitudeKm: number, dirY: number, distanceKm: number,
    cosTheta: number, sunIrradiance: ArrayLike<number>, fog: HeightFogMedium,
    out: Float32Array = new Float32Array(3)): Float32Array {
    analyticFogTransmittance(altitudeKm, dirY, distanceKm, fog, out);
    const phase = cornetteShanksPhase(cosTheta, fog.g);
    for (let c = 0; c < 3; c++) {
        const extinction = fog.scattering[c] + fog.absorption[c];
        const albedo = extinction > 0 ? fog.scattering[c] / extinction : 0;
        out[c] = sunIrradiance[c] * phase * albedo * (1 - out[c]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// The march — the CPU mirror of the froxel kernel (§4.3)
// ---------------------------------------------------------------------------

export interface AerialMarchOptions {
    /**
     * `'spherical'` (default) derives height from the radial distance to the planet centre, which
     * is what the kernel does. `'planar'` uses `h = h0 + dirY*s` — the assumption the analytic
     * integral above makes, so U9 compares like with like instead of against a curvature term.
     */
    geometry?: 'spherical' | 'planar';
    /**
     * Sun transmittance override. The kernel samples Phase 4's transmittance LUT; supplying a
     * constant here integrates the same thing the analytic form does (U9). Default: evaluate
     * Phase 4's `transmittance` per step.
     */
    sunTransmittance?: ArrayLike<number> | null;
    /** Marching starts here (world units). Default `config.nearDistance`. */
    startDistance?: number;
    /**
     * Accumulator to march into. Supply one to read the final PER-CHANNEL transmittance, which
     * the column's alpha channel has already collapsed to a mean. Defaults to module scratch —
     * so a caller who wants to hold the result must pass their own (design-principles §6).
     */
    state?: AerialAccumulator;
}

/** Result of {@link marchAerialColumn}: one froxel column, `[r, g, b, meanT]` per slice. */
export function createAerialColumn(sliceCount: number): Float32Array {
    return new Float32Array(sliceCount * 4);
}

/**
 * A running march state: in-scattered radiance and per-channel transmittance from the camera.
 *
 * Per-CHANNEL, deliberately — the froxel texel stores their mean in one alpha channel (§4.1), but
 * the accumulation itself must stay chromatic or extinction stops being wavelength-dependent and
 * distant objects fade to grey instead of to blue.
 */
export interface AerialAccumulator {
    inScatter: Float32Array;      // 3
    transmittance: Float32Array;  // 3
}

export function createAerialAccumulator(): AerialAccumulator {
    return { inScatter: new Float32Array(3), transmittance: new Float32Array([1, 1, 1]) };
}

/** Reset an accumulator to "at the camera": no in-scatter, full transmittance. */
export function resetAerialAccumulator(state: AerialAccumulator): AerialAccumulator {
    state.inScatter[0] = state.inScatter[1] = state.inScatter[2] = 0;
    state.transmittance[0] = state.transmittance[1] = state.transmittance[2] = 1;
    return state;
}

// Module scratch. Non-reentrant by design (single-threaded frame path, one module per Worker) —
// and none of it escapes into a return value (design-principles §6): every write lands in the
// caller's `out` or accumulator.
const scratchExtinction = new Float32Array(3);
const scratchScatter = new Float32Array(3);
const scratchSunT = new Float32Array(3);
const scratchPos = new Float64Array(3);
const scratchColumnState = createAerialAccumulator();

/**
 * Advance `state` through `[fromKm, toKm]`, in place.
 *
 * This is the primitive the whole design rests on: because the state carries forward, marching
 * `[a, b]` then `[b, c]` equals marching `[a, c]`, which is why the froxel volume can cost one ray
 * per lateral froxel instead of one per (froxel, slice). U3 pins exactly that property, because a
 * version that restarts the march at every slice renders an almost identical image and costs
 * `dims[2]` times more — a performance bug with no visual symptom.
 */
export function accumulateAerialSegment(
    altitudeKm: number,
    dir: ArrayLike<number>,
    sun: AerialSun,
    media: AerialMedia,
    fromKm: number,
    toKm: number,
    steps: number,
    state: AerialAccumulator,
    options: AerialMarchOptions = {},
): AerialAccumulator {
    const spherical = (options.geometry ?? 'spherical') === 'spherical';
    const ground = media.atmosphere?.groundRadius ?? EARTH_ATMOSPHERE.groundRadius;
    const r0 = ground + altitudeKm;
    const fixedSunT = options.sunTransmittance ?? null;

    // Toward the sun, not the direction it travels — the engine stores travel direction.
    const sunLength = Math.hypot(sun.direction[0], sun.direction[1], sun.direction[2]) || 1;
    const sunX = -sun.direction[0] / sunLength;
    const sunY = -sun.direction[1] / sunLength;
    const sunZ = -sun.direction[2] / sunLength;
    const cosTheta = dir[0] * sunX + dir[1] * sunY + dir[2] * sunZ;

    const L = state.inScatter;
    const T = state.transmittance;
    const ds = (toKm - fromKm) / steps;
    const dsMetres = ds * 1000;

    for (let step = 0; step < steps; step++) {
        const s = fromKm + (step + 0.5) * ds;

        let heightKm: number, muS: number;
        if (spherical) {
            scratchPos[0] = dir[0] * s;
            scratchPos[1] = r0 + dir[1] * s;
            scratchPos[2] = dir[2] * s;
            const r = Math.hypot(scratchPos[0], scratchPos[1], scratchPos[2]) || 1e-6;
            heightKm = r - ground;
            muS = (scratchPos[0] * sunX + scratchPos[1] * sunY + scratchPos[2] * sunZ) / r;
        } else {
            heightKm = altitudeKm + dir[1] * s;
            muS = sunY;
        }

        combinedExtinction(heightKm, media, scratchExtinction);
        combinedPhaseScattering(heightKm, cosTheta, media, scratchScatter);

        if (fixedSunT) {
            scratchSunT[0] = fixedSunT[0]; scratchSunT[1] = fixedSunT[1]; scratchSunT[2] = fixedSunT[2];
        } else if (media.atmosphere) {
            sunTransmittanceAt(heightKm, muS, media.atmosphere, scratchSunT);
        } else {
            scratchSunT[0] = scratchSunT[1] = scratchSunT[2] = 1;
        }

        for (let c = 0; c < 3; c++) {
            // In-scatter uses the transmittance ENTERING the step, then the step attenuates —
            // the same order as the sky-view kernel. Swapping them biases the whole volume
            // brighter with no other symptom.
            L[c] += T[c] * scratchSunT[c] * sun.intensity * scratchScatter[c] * dsMetres;
            T[c] *= Math.exp(-scratchExtinction[c] * dsMetres);
        }
    }
    return state;
}

/**
 * March one froxel column and write `[inScatter.rgb, meanTransmittance]` per slice into `out`.
 *
 * The march CONTINUES across slices rather than restarting from the camera — the whole cost
 * argument in §4.3 and the correctness property U3 pins. A version that restarts per slice
 * produces a plausible image and costs `dims[2]` times more.
 *
 * `dir` is a normalized world-space view direction; +Y is up, matching the frame Phase 4's
 * sky-view kernel puts the camera in.
 */
export function marchAerialColumn(
    altitudeKm: number,
    dir: ArrayLike<number>,
    sun: AerialSun,
    media: AerialMedia,
    config: ResolvedAerialConfig,
    out: Float32Array = createAerialColumn(config.dims[2]),
    options: AerialMarchOptions = {},
): Float32Array {
    const slices = config.dims[2];
    if (out.length < slices * 4) {
        throw new Error(`marchAerialColumn: out needs ${slices * 4} floats for ${slices} slices, got ${out.length}.`);
    }
    const state = resetAerialAccumulator(options.state ?? scratchColumnState);
    const toKm = 1 / config.unitsPerKm;
    let previous = (options.startDistance ?? config.nearDistance) * toKm;

    for (let z = 0; z < slices; z++) {
        const target = froxelSliceDistance(z, config) * toKm;
        accumulateAerialSegment(altitudeKm, dir, sun, media, previous, target,
            config.stepsPerSlice, state, options);
        const base = z * 4;
        out[base] = state.inScatter[0];
        out[base + 1] = state.inScatter[1];
        out[base + 2] = state.inScatter[2];
        // §4.1: the alpha channel is the MEAN of the three transmittances. The chromatic part of
        // extinction survives in the in-scatter (which is blue), not in this scalar — an
        // approximation the four-channel packing forces, stated rather than discovered.
        out[base + 3] = (state.transmittance[0] + state.transmittance[1] + state.transmittance[2]) / 3;
        previous = target;
    }
    return out;
}

/**
 * Sun transmittance from a point at `heightKm` with sun cosine `muS`, matching what the kernel
 * reads out of Phase 4's transmittance LUT. Ground-occluded directions return 0 — sunlight does
 * not arrive through the planet, and omitting that check lights the night side.
 */
export function sunTransmittanceAt(heightKm: number, muS: number, medium: AtmosphereMedium,
    out: Float32Array = new Float32Array(3)): Float32Array {
    // Phase 4's integral; it already returns zero for a ray that hits the ground.
    return atmosphereTransmittance(heightKm, muS, out, medium);
}

/**
 * Build the whole CPU volume — every column of a `dims[0] x dims[1] x dims[2]` froxel grid — for
 * a camera with the given basis. This is U11's reference and the readback target of build-order
 * step 2 ("the froxel volume, built but unread").
 *
 * `right`/`up`/`forward` are the camera's normalized world-space basis vectors and `tanHalfFovY`
 * / `aspect` its projection, exactly as the kernel reconstructs them from `invViewProj`.
 */
export function buildAerialVolumeCPU(
    altitudeKm: number,
    basis: { right: ArrayLike<number>; up: ArrayLike<number>; forward: ArrayLike<number>; tanHalfFovY: number; aspect: number },
    sun: AerialSun,
    media: AerialMedia,
    config: ResolvedAerialConfig,
    out: Float32Array = new Float32Array(config.dims[0] * config.dims[1] * config.dims[2] * 4),
    options: AerialMarchOptions = {},
): Float32Array {
    const [nx, ny, nz] = config.dims;
    if (out.length < nx * ny * nz * 4) {
        throw new Error(`buildAerialVolumeCPU: out needs ${nx * ny * nz * 4} floats, got ${out.length}.`);
    }
    const column = createAerialColumn(nz);
    const dir = new Float32Array(3);
    for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
            // NDC of the froxel centre. The Y flip matches the texture convention: v = 0 is the
            // top row, which is +1 in NDC.
            const ndcX = ((x + 0.5) / nx) * 2 - 1;
            const ndcY = 1 - ((y + 0.5) / ny) * 2;
            const sx = ndcX * basis.tanHalfFovY * basis.aspect;
            const sy = ndcY * basis.tanHalfFovY;
            for (let c = 0; c < 3; c++) {
                dir[c] = basis.forward[c] + basis.right[c] * sx + basis.up[c] * sy;
            }
            const length = Math.hypot(dir[0], dir[1], dir[2]) || 1;
            dir[0] /= length; dir[1] /= length; dir[2] /= length;

            marchAerialColumn(altitudeKm, dir, sun, media, config, column, options);
            for (let z = 0; z < nz; z++) {
                const dst = ((z * ny + y) * nx + x) * 4;
                const src = z * 4;
                out[dst] = column[src];
                out[dst + 1] = column[src + 1];
                out[dst + 2] = column[src + 2];
                out[dst + 3] = column[src + 3];
            }
        }
    }
    return out;
}

/**
 * `color * T + inScatter` — extinction THEN addition, in that order (§4.1).
 *
 * Reversed, distant objects glow instead of fading. It is the single most common bug in a fog
 * implementation and it looks superficially like "too much fog", which is why U5 asserts the
 * order against a reference rather than trusting that it reads correctly.
 */
export function applyAerialCPU(color: ArrayLike<number>, inScatter: ArrayLike<number>,
    transmittance: number, out: Float32Array = new Float32Array(3)): Float32Array {
    for (let c = 0; c < 3; c++) out[c] = color[c] * transmittance + inScatter[c];
    return out;
}
