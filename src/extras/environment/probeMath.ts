// src/extras/environment/probeMath.ts
//
// Pure, GPU-free math for Phase 2 (docs/lighting-probes-design.md): the irradiance-volume grid,
// the SH-L1 projection/evaluation, the parallax-corrected reflection-probe blend, the capture
// scheduler, and the config validation. Everything here runs headless — the CPU mirrors are kept
// in lockstep with the WGSL in probeShaders.ts by test/environment-probes.test.mjs (U1–U8), the
// same discipline envMath.ts ↔ envShaders.ts follow for Phase 1.
//
// Worker-safe: plain numbers and typed arrays in, typed arrays out.

import { faceDirection, cubeTexelSolidAngle } from './envMath';

// ---------------------------------------------------------------------------
// Constants — the WGSL-facing layouts (probeShaders.ts must match float-for-float)
// ---------------------------------------------------------------------------

/** Floats in the `ProbeParams` uniform (§4.3): 4 vec4 = 64 bytes. */
export const PROBE_PARAMS_FLOATS = 16;

/** Floats per `ReflProbe` record (§4.4): 3 vec4 = 48 bytes. */
export const REFL_PROBE_FLOATS = 12;

/** Floats of the `ReflProbeBuffer` header ahead of the runtime array: count + pad (16 bytes).
 *  The count lives IN the buffer so the blend loop's bound crosses the CPU/GPU boundary with the
 *  data — a uniform in ProbeParams would make the volume write another system's field. */
export const REFL_PROBE_HEADER_FLOATS = 4;

/** Cosine-convolution constants folded into the stored coefficients at projection time (§7.2):
 *  stored.w = Σ L·sa / (4π) (mean radiance = irradiance/π), stored.xyz = Σ L·d·sa / (2π).
 *  Derived from A0·B0² = π·(1/4π) = ¼ and A1·B1² = (2π/3)·(3/4π) = ½, divided by π so a uniform
 *  radiance c projects to irradiance c (U3) — matching Phase 1's /π convention (§7.4 there). */
export const SH_L1_L0_SCALE = 1 / (4 * Math.PI);
export const SH_L1_L1_SCALE = 1 / (2 * Math.PI);

/** A capture texel closer than this fraction of `spacing` marks its probe enclosed (§7.2). */
export const ENCLOSED_DIST_SPACING = 0.25;

/** A probe whose enclosed-texel fraction exceeds this is written as validity 0 (§7.2). */
export const ENCLOSED_VALIDITY_CUTOFF = 0.6;

// ---------------------------------------------------------------------------
// §4.1 The grid — probeIndex, probePosition, worldToUVW
// ---------------------------------------------------------------------------

/** Linear index of probe (i,j,k) — x fastest, z slowest; the 3D-texture texel address. */
export function probeIndex(dims: [number, number, number], i: number, j: number, k: number): number {
    return i + dims[0] * (j + dims[1] * k);
}

/** Inverse of {@link probeIndex}. */
export function probeCoord(dims: [number, number, number], index: number, out: [number, number, number] = [0, 0, 0]): typeof out {
    const i = index % dims[0];
    const j = Math.floor(index / dims[0]) % dims[1];
    const k = Math.floor(index / (dims[0] * dims[1]));
    out[0] = i; out[1] = j; out[2] = k;
    return out;
}

/** World position of probe (i,j,k): `origin + spacing * (i,j,k)` (§4.1). */
export function probePosition(
    origin: ArrayLike<number>, spacing: number, i: number, j: number, k: number,
    out: [number, number, number] = [0, 0, 0],
): typeof out {
    out[0] = origin[0] + spacing * i;
    out[1] = origin[1] + spacing * j;
    out[2] = origin[2] + spacing * k;
    return out;
}

/**
 * Normalized probe-grid coordinate of a world point (U1): `worldToUVW(probePosition(i,j,k))`
 * recovers `(i,j,k)/(dims-1)` exactly. This NORMALIZED coordinate feeds the bounds test and the
 * border fade — the texture sample itself happens at {@link probeSampleUVW}, because hardware
 * trilinear taps texel CENTRES and the grid's texel at (i,j,k) sits at `(i+0.5)/dims`, not at
 * `i/(dims-1)` (U2's "at a probe centre, one weight of 1" pins the alignment).
 */
export function worldToUVW(
    origin: ArrayLike<number>, spacing: number, dims: [number, number, number],
    worldPos: ArrayLike<number>, out: [number, number, number] = [0, 0, 0],
): typeof out {
    out[0] = (worldPos[0] - origin[0]) / (spacing * (dims[0] - 1));
    out[1] = (worldPos[1] - origin[1]) / (spacing * (dims[1] - 1));
    out[2] = (worldPos[2] - origin[2]) / (spacing * (dims[2] - 1));
    return out;
}

/** The texture coordinate of the hardware sample: `(uvw * (dims-1) + 0.5) / dims` — the mapping
 *  that places probe (i,j,k) exactly on texel centre ((i+0.5)/dims). Mirrors the WGSL lookup. */
export function probeSampleUVW(
    uvw: ArrayLike<number>, dims: [number, number, number],
    out: [number, number, number] = [0, 0, 0],
): typeof out {
    out[0] = (uvw[0] * (dims[0] - 1) + 0.5) / dims[0];
    out[1] = (uvw[1] * (dims[1] - 1) + 0.5) / dims[1];
    out[2] = (uvw[2] * (dims[2] - 1) + 0.5) / dims[2];
    return out;
}

// ---------------------------------------------------------------------------
// §5 The trilinear blend — the CPU mirror of the hardware sample (U2)
// ---------------------------------------------------------------------------

/** Eight (probe index, weight) pairs. Allocate once and pass as `out` on the hot path. */
export interface TrilinearResult {
    indices: Int32Array;
    weights: Float32Array;
}

export function createTrilinearResult(): TrilinearResult {
    return { indices: new Int32Array(8), weights: new Float32Array(8) };
}

/**
 * Mirror of the hardware trilinear tap at normalized grid coordinate `uvw`: texel-space
 * coordinate `tc = uvw * (dims-1)`, base = floor(tc), f = tc - base, corner indices clamped to
 * the volume (the clamped tap carries weight 0, exactly like the hardware's edge behaviour).
 * U2: a point exactly between eight probes yields eight 0.125s; a probe centre yields one 1.
 */
export function trilinearWeights(
    uvw: ArrayLike<number>, dims: [number, number, number],
    out: TrilinearResult = createTrilinearResult(),
): TrilinearResult {
    const bx = Math.min(Math.floor(uvw[0] * (dims[0] - 1)), dims[0] - 1);
    const by = Math.min(Math.floor(uvw[1] * (dims[1] - 1)), dims[1] - 1);
    const bz = Math.min(Math.floor(uvw[2] * (dims[2] - 1)), dims[2] - 1);
    const fx = uvw[0] * (dims[0] - 1) - bx;
    const fy = uvw[1] * (dims[1] - 1) - by;
    const fz = uvw[2] * (dims[2] - 1) - bz;
    const i1 = Math.min(bx + 1, dims[0] - 1);
    const j1 = Math.min(by + 1, dims[1] - 1);
    const k1 = Math.min(bz + 1, dims[2] - 1);
    let n = 0;
    for (let dz = 0; dz < 2; dz++) {
        const wz = dz === 0 ? 1 - fz : fz;
        const kk = dz === 0 ? bz : k1;
        for (let dy = 0; dy < 2; dy++) {
            const wy = dy === 0 ? 1 - fy : fy;
            const jj = dy === 0 ? by : j1;
            for (let dx = 0; dx < 2; dx++) {
                const wx = dx === 0 ? 1 - fx : fx;
                const ii = dx === 0 ? bx : i1;
                out.indices[n] = probeIndex(dims, ii, jj, kk);
                out.weights[n] = wx * wy * wz;
                n++;
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// §7.2 SH-L1 projection + §5 evaluation
// ---------------------------------------------------------------------------

export interface SHL1ProjectionOptions {
    /** Captured radiance, 6 × faceSize² × 4 (rgba), tile row-major per face. */
    color: ArrayLike<number>;
    faceSize: number;
    /**
     * Optional captured NDC depth, 6 × faceSize² (same tiling). When supplied, texels with
     * ndcZ == 1 get `skyRadiance` and texels closer than `enclosedDist` count toward validity.
     */
    depth?: ArrayLike<number>;
    near?: number;
    far?: number;
    /** World distance below which a texel marks the probe enclosed. Default: ignored. */
    enclosedDist?: number;
    /** Radiance substituted for rays that hit nothing (the environment sky). Default 0. */
    skyRadiance?: [number, number, number];
}


/**
 * Project a captured cube (6 tiles of the capture atlas) to SH-L1, EXACTLY as the projection
 * kernel does (probeShaders.ts `buildProbeProjectionWGSL`): true per-texel solid angles, the
 * viewport y-flip (`v = 1 - uv.y` — a rendered tile's row 0 is NDC +y, cube-face v = 1), and the
 * cosine-convolution constants folded in. Returns 13 floats: the channel-blocked coefficients
 * `[rX,rY,rZ,r0, gX,gY,gZ,g0, bX,bY,bZ,b0]` plus the validity in slot 12.
 */
export function projectSHL1FromCapture(
    opts: SHL1ProjectionOptions,
    out: Float32Array = new Float32Array(13),
): Float32Array {
    const F = opts.faceSize;
    const sky = opts.skyRadiance ?? [0, 0, 0];
    out.fill(0);
    let enclosed = 0;
    const dir = [0, 0, 0];
    for (let face = 0; face < 6; face++) {
        const faceBase = face * F * F;
        for (let py = 0; py < F; py++) {
            for (let px = 0; px < F; px++) {
                const t = faceBase + py * F + px;
                const u = (px + 0.5) / F;
                const v = (py + 0.5) / F;
                // The viewport y-flip: tile row y sees cube-face v = 1 - v (§7.1).
                faceDirection(face, u, 1 - v, dir);
                // INTEGER texel indices, not the normalized uv. The CPU signature is
                // (x, y, size) and derives the centre itself; the WGSL twin takes the already-
                // normalized uv. Passing uv here silently shrinks every texel's solid angle —
                // Σsa comes to 7.78 instead of 4π, and the whole projection scales down ~1.6×
                // with no error anywhere. U3's "L0 of a constant == c" is what catches it.
                const sa = cubeTexelSolidAngle(px, py, F);
                let r: number, g: number, b: number;
                // No depth supplied means "trust the colour for every texel" — NOT "everything is
                // sky". Defaulting ndcZ to 1 hits the sky sentinel and substitutes skyRadiance
                // (default black) for the whole capture, so the projection returns all zeros with
                // no error anywhere. On the GPU depth is always present; this branch exists for
                // the CPU mirror and its tests.
                const hasDepth = opts.depth !== undefined;
                const ndcZ = hasDepth ? opts.depth![t] : 0;
                if (hasDepth && ndcZ >= 1) {
                    r = sky[0]; g = sky[1]; b = sky[2];
                } else {
                    r = opts.color[t * 4]; g = opts.color[t * 4 + 1]; b = opts.color[t * 4 + 2];
                    if (opts.enclosedDist !== undefined && opts.near !== undefined && opts.far !== undefined) {
                        const cx = u * 2 - 1, cy = v * 2 - 1;
                        const rayLen = Math.sqrt(1 + cx * cx + cy * cy);
                        const linearZ = (opts.near * opts.far) / (opts.far - ndcZ * (opts.far - opts.near));
                        if (linearZ * rayLen < opts.enclosedDist) enclosed++;
                    }
                }
                // Channel-blocked accumulation, constants folded at the store (§7.2).
                out[0] += r * dir[0] * sa; out[1] += r * dir[1] * sa; out[2] += r * dir[2] * sa; out[3] += r * sa;
                out[4] += g * dir[0] * sa; out[5] += g * dir[1] * sa; out[6] += g * dir[2] * sa; out[7] += g * sa;
                out[8] += b * dir[0] * sa; out[9] += b * dir[1] * sa; out[10] += b * dir[2] * sa; out[11] += b * sa;
            }
        }
    }
    for (let c = 0; c < 3; c++) {
        out[c * 4] *= SH_L1_L1_SCALE; out[c * 4 + 1] *= SH_L1_L1_SCALE; out[c * 4 + 2] *= SH_L1_L1_SCALE;
        out[c * 4 + 3] *= SH_L1_L0_SCALE;
    }
    const frac = enclosed / (6 * F * F);
    out[12] = frac > ENCLOSED_VALIDITY_CUTOFF ? 0 : 1 - frac;
    return out;
}

/**
 * Irradiance(/π) along `N` from packed SH-L1 coefficients — the mirror of the WGSL
 * `evalSHL1probe`. `r/g/b` are the four packed coefficients (L1x, L1y, L1z, L0) per channel.
 * The `max(·, 0)` is load-bearing: an L1 projection rings facing away from the dominant
 * direction (§5, U4).
 */
export function evalSHL1CPU(
    r: ArrayLike<number>, g: ArrayLike<number>, b: ArrayLike<number>,
    N: ArrayLike<number>, out: [number, number, number] = [0, 0, 0],
): typeof out {
    out[0] = Math.max(r[0] * N[0] + r[1] * N[1] + r[2] * N[2] + r[3], 0);
    out[1] = Math.max(g[0] * N[0] + g[1] * N[1] + g[2] * N[2] + g[3], 0);
    out[2] = Math.max(b[0] * N[0] + b[1] * N[1] + b[2] * N[2] + b[3], 0);
    return out;
}


// ---------------------------------------------------------------------------
// §5 The volume lookup — sampleProbeVolumeCPU, mirroring getIndirectDiffuse exactly
// ---------------------------------------------------------------------------

/** smoothstep(0, fadeUV, min distance to a face) — the border exit weight. */
export function volumeBorderWeightCPU(uvw: ArrayLike<number>, fadeUV: number): number {
    const m = Math.min(
        uvw[0], 1 - uvw[0],
        uvw[1], 1 - uvw[1],
        uvw[2], 1 - uvw[2],
    );
    const t = Math.min(Math.max(m / fadeUV, 0), 1);
    return t * t * (3 - 2 * t);
}

export interface ProbeVolumeCPUState {
    origin: [number, number, number];
    spacing: number;
    dims: [number, number, number];
    normalBiasTexels: number;
    borderFade: number;
    intensity: number;
    /** Packed SH-L1 texels, count × 4, per channel. */
    probeR: Float32Array;
    probeG: Float32Array;
    probeB: Float32Array;
    /** count × 1. */
    validity: Float32Array;
}

const _tri = createTrilinearResult();

/**
 * The full CPU mirror of the WGSL `getIndirectDiffuse` (§5): normal bias, the normalized-grid
 * bounds test, the trilinear 8-probe blend, and **both fallbacks toward the global environment,
 * never toward zero** (U5: validity 0 everywhere ⇒ the result EQUALS the global, exactly).
 */
export function sampleProbeVolumeCPU(
    vol: ProbeVolumeCPUState,
    worldPos: ArrayLike<number>, N: ArrayLike<number>,
    globalFn: (N: ArrayLike<number>, out: [number, number, number]) => ArrayLike<number>,
    out: [number, number, number] = [0, 0, 0],
): typeof out {
    const { dims, spacing } = vol;
    const biasLen = vol.normalBiasTexels * spacing;
    const biased = [
        worldPos[0] + N[0] * biasLen,
        worldPos[1] + N[1] * biasLen,
        worldPos[2] + N[2] * biasLen,
    ];
    const uvw = worldToUVW(vol.origin, spacing, dims, biased);
    if (uvw[0] < 0 || uvw[0] > 1 || uvw[1] < 0 || uvw[1] > 1 || uvw[2] < 0 || uvw[2] > 1) {
        const gOut = globalFn(N, out);
        out[0] = gOut[0]; out[1] = gOut[1]; out[2] = gOut[2];
        return out;
    }
    trilinearWeights(uvw, dims, _tri);
    const r = [0, 0, 0, 0], g = [0, 0, 0, 0], b = [0, 0, 0, 0];
    let valid = 0;
    for (let c = 0; c < 8; c++) {
        const idx = _tri.indices[c], w = _tri.weights[c];
        if (w === 0) continue;
        for (let k = 0; k < 4; k++) {
            r[k] += vol.probeR[idx * 4 + k] * w;
            g[k] += vol.probeG[idx * 4 + k] * w;
            b[k] += vol.probeB[idx * 4 + k] * w;
        }
        valid += vol.validity[idx] * w;
    }
    const local = evalSHL1CPU(r, g, b, N);
    const global = globalFn(N, [0, 0, 0]);
    const fadeUV = vol.borderFade / (spacing * (dims[0] - 1));
    const t = valid * volumeBorderWeightCPU(uvw, fadeUV);
    out[0] = global[0] * (1 - t) + local[0] * vol.intensity * t;
    out[1] = global[1] * (1 - t) + local[1] * vol.intensity * t;
    out[2] = global[2] * (1 - t) + local[2] * vol.intensity * t;
    return out;
}


// ---------------------------------------------------------------------------
// §8 Reflection probes — parallax + influence
// ---------------------------------------------------------------------------

/**
 * Parallax-corrected lookup direction (§8.2): intersect the reflection ray with the proxy box
 * (slab method, furthest slab exit), then look up FROM THE CAPTURE POINT — using the box centre
 * in that last step is the classic sliding-reflection bug (U6 exists to catch a regression to it).
 */
export function parallaxCorrect(
    center: ArrayLike<number>, halfExtent: ArrayLike<number>, capture: ArrayLike<number>,
    worldPos: ArrayLike<number>, R: ArrayLike<number>,
    out: [number, number, number] = [0, 0, 0],
): typeof out {
    let dist = Infinity;
    for (let a = 0; a < 3; a++) {
        const invR = 1 / (R[a] as number);
        const first = (center[a] + halfExtent[a] - worldPos[a]) * invR;
        const second = (center[a] - halfExtent[a] - worldPos[a]) * invR;
        dist = Math.min(dist, Math.max(first, second));
    }
    const hx = worldPos[0] + R[0] * dist;
    const hy = worldPos[1] + R[1] * dist;
    const hz = worldPos[2] + R[2] * dist;
    const dx = hx - capture[0], dy = hy - capture[1], dz = hz - capture[2];
    const l = Math.hypot(dx, dy, dz) || 1;
    out[0] = dx / l; out[1] = dy / l; out[2] = dz / l;
    return out;
}

/**
 * The per-probe blend weight (§8.3): 1 deep inside the proxy box, fading to 0 across
 * `blend` world units at the boundary, 0 outside the box and outside the influence radius.
 * `captured === false` (the WGSL's negative layer) reads as 0 — the global environment covers it.
 */
export function reflInfluenceWeight(
    centerW: ArrayLike<number>,     // [x,y,z, blend distance]
    halfExtentW: ArrayLike<number>, // [x,y,z, influence radius]
    worldPos: ArrayLike<number>,
    captured: boolean = true,
): number {
    if (!captured) return 0;
    const qx = Math.abs(worldPos[0] - centerW[0]) - halfExtentW[0];
    const qy = Math.abs(worldPos[1] - centerW[1]) - halfExtentW[1];
    const qz = Math.abs(worldPos[2] - centerW[2]) - halfExtentW[2];
    const inside = -Math.max(qx, Math.max(qy, qz));
    if (inside < 0) return 0;
    const dc = Math.hypot(worldPos[0] - centerW[0], worldPos[1] - centerW[1], worldPos[2] - centerW[2]);
    if (dc > halfExtentW[3]) return 0;
    const t = Math.min(Math.max(inside / centerW[3], 0), 1);
    return t * t * (3 - 2 * t);
}

/**
 * Mirror of the WGSL blend loop (§8.3): sorted probes claim weight front-to-back,
 * `wi = min(influence, 1 - w)`, the environment gets exactly the remainder. U7 asserts the
 * invariant that makes the blend energy-preserving: total probe weight ≤ 1 at every point.
 */
export function blendReflectionProbes(
    records: Array<{ centerW: [number, number, number, number]; halfExtentW: [number, number, number, number]; captured: boolean }>,
    worldPos: ArrayLike<number>,
    outWeights: Float32Array,
): number {
    let w = 0;
    for (let i = 0; i < records.length; i++) {
        const wi = Math.min(
            reflInfluenceWeight(records[i].centerW, records[i].halfExtentW, worldPos, records[i].captured),
            1 - w);
        outWeights[i] = wi > 0 ? wi : 0;
        w += outWeights[i];
    }
    return w;
}


// ---------------------------------------------------------------------------
// Uniform / storage packing
// ---------------------------------------------------------------------------

/** Pack `ProbeParams` (§4.3). */
export function packProbeParams(
    cfg: {
        origin: [number, number, number]; spacing: number; dims: [number, number, number];
        normalBiasTexels: number; borderFade: number; intensity: number;
    },
    out: Float32Array = new Float32Array(PROBE_PARAMS_FLOATS),
): Float32Array {
    out[0] = cfg.origin[0]; out[1] = cfg.origin[1]; out[2] = cfg.origin[2]; out[3] = cfg.intensity;
    out[4] = 1 / (cfg.spacing * (cfg.dims[0] - 1));
    out[5] = 1 / (cfg.spacing * (cfg.dims[1] - 1));
    out[6] = 1 / (cfg.spacing * (cfg.dims[2] - 1));
    out[7] = cfg.spacing;
    out[8] = cfg.dims[0]; out[9] = cfg.dims[1]; out[10] = cfg.dims[2]; out[11] = cfg.normalBiasTexels;
    out[12] = cfg.borderFade; out[13] = 0; out[14] = 0; out[15] = 0;
    return out;
}

/**
 * Pack one `ReflProbe` record (12 floats) at float offset `o`. An UNCAPTURED probe packs its
 * layer as `-(layer + 1)` (negative): the WGSL blend loop skips it, so a scene of never-captured
 * probes degrades to exactly Phase 1 (§5 principle: fall back to the environment, never to black).
 */
export function packReflProbe(
    out: Float32Array, o: number,
    center: ArrayLike<number>, blend: number,
    halfExtent: ArrayLike<number>, radius: number,
    capture: ArrayLike<number>, layer: number, captured: boolean,
): void {
    out[o] = center[0]; out[o + 1] = center[1]; out[o + 2] = center[2]; out[o + 3] = blend;
    out[o + 4] = halfExtent[0]; out[o + 5] = halfExtent[1]; out[o + 6] = halfExtent[2]; out[o + 7] = radius;
    out[o + 8] = capture[0]; out[o + 9] = capture[1]; out[o + 10] = capture[2];
    out[o + 11] = captured ? layer : -(layer + 1);
}

// ---------------------------------------------------------------------------
// §7.4 The scheduler
// ---------------------------------------------------------------------------

export interface ProbeScheduler {
    /** Frame of each probe's last capture; -1 = never (the "highest priority" input, §7.4). */
    lastCaptureFrame: Int32Array;
    /** Explicit invalidation (the caller moved a light or opened a door, §7.4). */
    invalidated: Uint8Array;
    /** This frame's picked probe indices, length = the returned count. */
    picks: Int32Array;
    /** Parallel score scratch for the top-N selection. */
    pickScores: Float64Array;
}

export function createProbeScheduler(probeCount: number, maxPicks: number): ProbeScheduler {
    return {
        lastCaptureFrame: new Int32Array(probeCount).fill(-1),
        invalidated: new Uint8Array(probeCount),
        picks: new Int32Array(maxPicks),
        pickScores: new Float64Array(maxPicks),
    };
}

/**
 * Pick up to `picks.length` probes to recapture this frame (§7.4). Priority, in order:
 *   1. never captured            — the volume fills in from nothing on first run
 *   2. explicitly invalidated    — a light moved, a door opened
 *   3. score = age × distance weight — near probes refresh up to `bias`× more often than a probe
 *      past the reference distance, but a far probe's score grows linearly with age, so no probe
 *      starves (U8).
 * Deterministic: earlier probe index wins ties. Marks picks captured at SCHEDULE time (the volume
 * holds bounce ⁿ after this frame's projection runs — §7.3).
 */
export function scheduleProbes(
    sch: ProbeScheduler,
    positions: Float32Array,       // 3 per probe — world positions
    probeCount: number,
    camPos: ArrayLike<number>,
    frame: number,
    bias: number,
    invDistRef: number,            // 1 / reference distance for the near weight
): number {
    let n = 0;
    const maxPicks = sch.picks.length;
    for (let i = 0; i < probeCount && n < maxPicks; i++) {           // 1. never captured
        if (sch.lastCaptureFrame[i] < 0) sch.picks[n++] = i;
    }
    for (let i = 0; i < probeCount && n < maxPicks; i++) {           // 2. invalidated
        if (sch.invalidated[i] !== 0 && sch.lastCaptureFrame[i] >= 0) sch.picks[n++] = i;
    }
    if (n < maxPicks) {                                              // 3. scored sweep
        const need = maxPicks - n;
        for (let k = 0; k < need; k++) sch.pickScores[k] = -Infinity;
        let minSlot = 0;
        for (let i = 0; i < probeCount; i++) {
            if (sch.lastCaptureFrame[i] < 0 || sch.invalidated[i] !== 0) continue;
            const age = frame - sch.lastCaptureFrame[i];
            if (age <= 0) continue;
            const dx = positions[i * 3] - camPos[0];
            const dy = positions[i * 3 + 1] - camPos[1];
            const dz = positions[i * 3 + 2] - camPos[2];
            const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
            const w = 1 + (bias - 1) * Math.max(0, 1 - d * invDistRef);
            const score = age * w;
            if (score > sch.pickScores[minSlot]) {                    // strict: earliest index wins ties
                sch.picks[n + minSlot] = i;
                sch.pickScores[minSlot] = score;
                minSlot = 0;                                          // re-locate the new minimum
                for (let k = 1; k < need; k++) if (sch.pickScores[k] < sch.pickScores[minSlot]) minSlot = k;
            }
        }
        // Compact the filled corner of the top-N set to the front of the picks array.
        let m = n;
        for (let k = 0; k < need; k++) {
            if (sch.pickScores[k] > -Infinity) sch.picks[m++] = sch.picks[n + k];
        }
        n = m;
    }
    for (let k = 0; k < n; k++) {                                     // mark at schedule time
        sch.lastCaptureFrame[sch.picks[k]] = frame;
        sch.invalidated[sch.picks[k]] = 0;
    }
    return n;
}


// ---------------------------------------------------------------------------
// §11 Config validation — every failure names the offender and the way out
// ---------------------------------------------------------------------------

export function validateProbeVolumeConfig(cfg: {
    origin: [number, number, number];
    spacing: number;
    dims: [number, number, number];
    captureFaceSize: number;
    probesPerFrame: number;
    captureAtlasSize: number;
    maxProbes: number;
    shBands: number;
    environment: unknown;
    visibility: unknown;
    shadows?: unknown;
    allowShadowlessCapture: boolean;
    reflectionFaceSize: number;   // 0 disables the reflection band
}): void {
    const [dx, dy, dz] = cfg.dims;
    const n = dx * dy * dz;
    if (n > cfg.maxProbes) {
        const bytes = n * (3 * 8 + 1);
        throw new Error(
            `ProbeVolume: dims ${dx}×${dy}×${dz} = ${n} probes exceeds maxProbes ${cfg.maxProbes}; ` +
            `a 3D texture of ${bytes} bytes would be allocated.`);
    }
    const tilesPerRow = Math.floor(cfg.captureAtlasSize / cfg.captureFaceSize);
    const reflTilesPerRow = cfg.reflectionFaceSize > 0
        ? Math.floor(cfg.captureAtlasSize / cfg.reflectionFaceSize) : 0;
    const reflBandPx = cfg.reflectionFaceSize > 0
        ? Math.ceil(6 / reflTilesPerRow) * cfg.reflectionFaceSize : 0;
    const usedRows = Math.ceil((cfg.probesPerFrame * 6) / tilesPerRow);
    if (tilesPerRow === 0 || usedRows * cfg.captureFaceSize + reflBandPx > cfg.captureAtlasSize) {
        throw new Error(
            `ProbeVolume: probesPerFrame ${cfg.probesPerFrame} needs ${cfg.probesPerFrame * 6} tiles of ` +
            `${cfg.captureFaceSize}px, exceeding the ${cfg.captureAtlasSize}px capture atlas; ` +
            `lower probesPerFrame or captureFaceSize.`);
    }
    if (!cfg.environment) {
        throw new Error(
            `ProbeVolume: config.environment is required — the volume falls back to the global ` +
            `environment outside its bounds and uses it as the capture ambient ` +
            `(lighting-probes-design.md §5, §7.3).`);
    }
    if (!cfg.visibility) {
        throw new Error(`ProbeVolume: config.visibility is required; probe capture has no CPU draw path.`);
    }
    if (!cfg.shadows && !cfg.allowShadowlessCapture) {
        throw new Error(
            `ProbeVolume: capturing without a ShadowSystem produces shadowless bounce light — ` +
            `pass shadows, or set allowShadowlessCapture:true to accept it.`);
    }
    if (cfg.shBands === 2) {
        throw new Error(
            `ProbeVolume: shBands 2 needs 7 rgba16float 3D textures but only 4 group-2 texture ` +
            `bindings remain (lighting-probes-design.md §4.2); use L1.`);
    }
    if (cfg.spacing <= 0) {
        throw new Error(`ProbeVolume: spacing ${cfg.spacing} must be > 0; the grid is parameterized by it.`);
    }
    if (dx < 2 || dy < 2 || dz < 2) {
        throw new Error(
            `ProbeVolume: dims ${dx}×${dy}×${dz} must be ≥ 2 on every axis; ` +
            `the trilinear blend needs a real neighbourhood.`);
    }
}

