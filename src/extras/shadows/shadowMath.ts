// src/extras/shadows/shadowMath.ts
//
// Pure, GPU-free cascade shadow math — the CPU mirror of the shadow WGSL kernels in
// shadowShaders.ts (docs/shadows-design.md §5). Everything here runs in a plain unit test with
// no device, so the split/fit/snap/extrusion math is pinned by numbers, not by eyeballing a
// render. It is also worker-safe: it consumes matrices and numbers and produces Float32Arrays.
//
// Conventions (identical to cullMath.ts, both verified against core/Camera.ts):
//  1. Matrices are COLUMN-MAJOR (gl-matrix / WGSL). Element at column c, row r is m[c*4 + r].
//  2. Clip z spans [0, 1] (WebGPU ZO). Cascade orthos are built with mat4.orthoZO — the NO form
//     silently moves the near plane, the same trap Camera.ts:62-66 documents for perspective.

import { mat4, vec3 } from 'gl-matrix';

// ---------------------------------------------------------------------------
// Scratch — module-scope temporaries so the per-frame fit allocates nothing
// ---------------------------------------------------------------------------
//
// Roadmap constraint #2 ("no per-frame allocation, no GC spikes"). Cascade fitting runs every
// frame, so every gl-matrix temporary lives here rather than being created per call. Module
// scope makes these functions non-reentrant, which is correct for a single-threaded frame path;
// a Worker gets its own module instance, so the `setRawCascades` hand-off (§9.2) is unaffected.
//
// Every public function below still allocates by default when no `out` is supplied, so the
// headless tests (and any one-shot caller) keep their existing signatures.

const _corners = new Float32Array(24);
const _sphere = new Float32Array(4);
const _ldIn = vec3.create();
const _ld = vec3.create();
const _ls = vec3.create();
const _snapped = vec3.create();
const _center = vec3.create();
const _eye = vec3.create();
const _snapView = mat4.create();
const _invSnap = mat4.create();
const _view = mat4.create();
const _proj = mat4.create();
const _viewProj = mat4.create();
const _invViewProj = mat4.create();
const _invView = mat4.create();
const _invProj = mat4.create();

/** Up vectors, hoisted — a ternary over two array literals allocated one per cascade per frame. */
const UP_Y: vec3 = vec3.fromValues(0, 1, 0);
const UP_Z: vec3 = vec3.fromValues(0, 0, 1);
/** The texel-snap frame's eye (§5.5): fixed in the world, so the grid does not travel with the camera. */
const ORIGIN: vec3 = vec3.fromValues(0, 0, 0);

/**
 * A cached `Uint32Array` view over a `Float32Array`'s memory, for the packers that write integer
 * fields as raw bits. Creating the view per call is a small allocation, but these run per view
 * per frame — enough to show up as GC pressure in a 60 Hz loop. The map is weak, so a discarded
 * staging array takes its view with it.
 */
// Keyed on the Float32Array itself, NOT its buffer: the view must inherit `byteOffset`/`length`
// so float index i and u32 index i address the same word. Node's typed-array pool hands out
// non-zero byteOffsets, so a whole-buffer view would silently write to the wrong words.
const _u32Views = new WeakMap<Float32Array, Uint32Array>();
export function u32ViewOf(f32: Float32Array): Uint32Array {
    let v = _u32Views.get(f32);
    if (!v) {
        v = new Uint32Array(f32.buffer, f32.byteOffset, f32.length);
        _u32Views.set(f32, v);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Maximum cascades. Sizes the `ShadowData` uniform's `array<ShadowCascade, MAX_CASCADES>`
 * (design doc §4.2) — a fixed-size uniform array, so this is a hard cap, not a hint.
 */
export const MAX_CASCADES = 4;

/** Floats per cascade camera buffer — the exact `Camera` layout (core/Camera.ts:35). */
export const SHADOW_VIEW_FLOATS = 108;

/**
 * Floats in the per-cascade caster params uniform (`ShadowCasterParams` in shadowShaders.ts):
 * `[texelWorld, normalOffsetTexels, constantBias, _pad]` = 16 bytes. One per cascade, so the
 * caster's bias constants are exact for the cascade being rendered (§7) — a single shared
 * ShadowData uniform cannot serve the caster, because the cascade index is a per-PASS property
 * and a uniform array index would be a per-vertex one.
 */
export const SHADOW_CASTER_PARAMS_FLOATS = 4;


/** Floats per `ShadowCascade` record in the receiver uniform (80 bytes, 16-aligned). */
export const SHADOW_CASCADE_FLOATS = 20;

/** Floats in the whole `ShadowData` receiver uniform: 20 × MAX_CASCADES + 8 = 88 (352 B). */
export const SHADOW_DATA_FLOATS = SHADOW_CASCADE_FLOATS * MAX_CASCADES + 8;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * The camera state cascade fitting needs. A plain-data snapshot (not the `Camera` class) so a
 * worker can compute cascades from a transferred struct — the `setRawCascades` contract (§9.2).
 */
export interface ShadowCameraState {
    /** Column-major 4×4 world-from-view (inverse view matrix). */
    invView: ArrayLike<number>;
    /** Vertical field of view in radians. */
    fovRads: number;
    /** Viewport aspect (width / height). */
    aspect: number;
    /** Main camera near plane. */
    near: number;
    /** `bufferData[99]` — copied verbatim into every cascade's simTime slot. */
    simTime?: number;
}

/** Per-cascade fit parameters (design doc §5.1–§5.5). */
export interface CascadeFitConfig {
    /** Number of cascades, 1..MAX_CASCADES. */
    cascades: number;
    /** Shadow map resolution (square). Must be a power of two — texel snapping assumes it. */
    resolution: number;
    /** World-space distance from the camera beyond which shadows stop. NOT camera.far. */
    shadowDistance: number;
    /** Practical-split blend: 0 = uniform, 1 = logarithmic. Default 0.6. */
    lambda: number;
    /**
     * How far behind the cascade (along the light direction) the ortho near plane is pushed so
     * off-screen casters still land in the map (§5.4). Default = shadowDistance.
     */
    casterExtrusion: number;
    /** Normal offset in texels, applied on the caster (§7.2). Used here only for the §11 guard. */
    normalOffsetTexels: number;
}

/** The result of fitting one cascade. */
export interface CascadeResult {
    /** 108 floats, the exact `Camera` layout — written verbatim into the cascade's GPUBuffer. */
    viewData: Float32Array;
    /** View-space depth this cascade covers up to (its split far plane). */
    splitFar: number;
    /** World size of one shadow texel — sets the caster normal offset and the PCF radius. */
    texelWorld: number;
    /** World-space centre of the fitted (snapped) bounding sphere. */
    center: [number, number, number];
    /** Radius of the fitted bounding sphere (quantized to 1/16 to kill jitter). */
    radius: number;
}

// ---------------------------------------------------------------------------
// Validation (design doc §11 — loud, naming the offender)
// ---------------------------------------------------------------------------

/** Validate the constructor-level invariants. Throws the §11 errors verbatim. */
export function validateShadowConfig(cascades: number, resolution: number): void {
    if (!Number.isInteger(cascades) || cascades < 1 || cascades > MAX_CASCADES) {
        throw new Error(
            `ShadowSystem: cascades must be between 1 and ${MAX_CASCADES} (got ${cascades}); ` +
            `MAX_CASCADES sizes the ShadowData uniform.`);
    }
    if (!Number.isInteger(resolution) || resolution < 1 || (resolution & (resolution - 1)) !== 0) {
        throw new Error(
            `ShadowSystem: resolution ${resolution} must be a power of two; ` +
            `texel snapping (§5.5) assumes it.`);
    }
}

// ---------------------------------------------------------------------------
// §5.1 Splits — practical split scheme (Zhang et al.)
// ---------------------------------------------------------------------------

/**
 * Blend logarithmic (good near) with uniform (good far) by `lambda`. Returns `count` split far
 * distances, strictly increasing, with `splits[count-1] === shadowDistance` exactly.
 */
export function cascadeSplits(
    near: number,
    shadowDistance: number,
    count: number,
    lambda: number,
    out: number[] = [],
): number[] {
    out.length = count;
    for (let i = 1; i <= count; i++) {
        const p = i / count;
        const log = near * Math.pow(shadowDistance / near, p);
        const uni = near + (shadowDistance - near) * p;
        out[i - 1] = lambda * log + (1 - lambda) * uni;
    }
    return out;
}

// ---------------------------------------------------------------------------
// §5.2 Split frustum corners
// ---------------------------------------------------------------------------

/**
 * The eight world-space corners of the split `[nearD, farD)`, taken directly from `invView`,
 * `fovRads`, and `aspect` (less numeric drift than unprojecting the NDC box). Writes 24 floats
 * (8 × xyz) into `out`, near quad first, each quad ordered (-x,-y) (+x,-y) (-x,+y) (+x,+y).
 */
export function splitFrustumCorners(
    invView: ArrayLike<number>,
    fovRads: number,
    aspect: number,
    nearD: number,
    farD: number,
    out: Float32Array = new Float32Array(24),
): Float32Array {
    const tanH = Math.tan(fovRads * 0.5);
    const tanW = tanH * aspect;
    let o = 0;
    // Index arithmetic rather than `for (const d of [nearD, farD])` — the array literals in the
    // original allocated three arrays per call, and this runs once per cascade per frame.
    for (let q = 0; q < 2; q++) {
        const d = q === 0 ? nearD : farD;
        for (let iy = 0; iy < 2; iy++) {
            const sy = iy === 0 ? -1 : 1;
            for (let ix = 0; ix < 2; ix++) {
                const sx = ix === 0 ? -1 : 1;
                // viewToWorld: invView * (sx·tanW·d, sy·tanH·d, -d, 1), column-major.
                const vx = sx * tanW * d, vy = sy * tanH * d, vz = -d;
                out[o++] = invView[0] * vx + invView[4] * vy + invView[8] * vz + invView[12];
                out[o++] = invView[1] * vx + invView[5] * vy + invView[9] * vz + invView[13];
                out[o++] = invView[2] * vx + invView[6] * vy + invView[10] * vz + invView[14];
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// §5.3 Sphere fit — rotation-invariant, so the map does not re-quantize on camera yaw
// ---------------------------------------------------------------------------

/**
 * Fit a bounding sphere to the 24-float corner array from {@link splitFrustumCorners}:
 * centroid, then max radius, quantized UP to 1/16 world units so sub-texel camera jitter does
 * not re-fit the cascade. Writes `[cx, cy, cz, r]` into `out`.
 */
export function fitSphereToCorners(
    corners: ArrayLike<number>,
    out: Float32Array | number[] = [0, 0, 0, 0],
): Float32Array | number[] {
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < 8; i++) {
        cx += corners[i * 3]; cy += corners[i * 3 + 1]; cz += corners[i * 3 + 2];
    }
    cx /= 8; cy /= 8; cz /= 8;
    let r = 0;
    for (let i = 0; i < 8; i++) {
        const dx = corners[i * 3] - cx, dy = corners[i * 3 + 1] - cy, dz = corners[i * 3 + 2] - cz;
        r = Math.max(r, Math.hypot(dx, dy, dz));
    }
    out[0] = cx; out[1] = cy; out[2] = cz;
    out[3] = Math.ceil(r * 16) / 16; // quantize to kill jitter (§5.3)
    return out;
}

// ---------------------------------------------------------------------------
// §5.4 + §5.5 The full cascade fit
// ---------------------------------------------------------------------------

/**
 * Fit one cascade: sphere fit → texel snap in light space → extruded ZO ortho → pack the exact
 * 108-float `Camera` layout. `lightDir` is the normalized light TRAVEL direction (where the
 * light goes, e.g. `[-0.5, -1, -0.3]`); the light looks along it.
 *
 * The near plane is pushed `extrusion` behind the sphere along the light direction so casters
 * outside the visible region still land in the map (§5.4) — the conservative choice, because a
 * wrongly-culled caster is a missing shadow in the LIT image.
 */
export function fitCascade(
    camera: ShadowCameraState,
    lightDir: ArrayLike<number>,
    nearD: number,
    farD: number,
    config: CascadeFitConfig,
    out: CascadeResult = createCascadeResult(),
): CascadeResult {
    splitFrustumCorners(camera.invView, camera.fovRads, camera.aspect, nearD, farD, _corners);
    fitSphereToCorners(_corners, _sphere);
    const radius = _sphere[3];
    const extrusion = config.casterExtrusion;

    // Light orientation: the light looks along lightDir, so its eye sits on the far side of the
    // sphere (toward the sun) and its target on the near side. The up vector must not be
    // collinear with the light direction — a vertical sun gets the Z axis instead.
    vec3.set(_ldIn, lightDir[0], lightDir[1], lightDir[2]);
    const ld = vec3.normalize(_ld, _ldIn);
    const up: vec3 = Math.abs(ld[1]) > 0.999 ? UP_Z : UP_Y;

    // §5.5 — snap the ortho centre to whole shadow texels IN LIGHT SPACE, or the cascade
    // shimmers as the camera rotates and crawls as it translates. The snap frame must be FIXED in
    // the world — the light's orientation, origin at the world origin — so the texel grid stays put
    // while the cascade slides across it. Extrusion only moves the final eye along the light
    // direction (this frame's Z), so it never shifts the grid either.
    //
    // This frame used to put its eye AT the sphere centre. That maps the centre to (0, 0) every
    // frame, so the snap was a no-op; U3 measured it in the same moving frame and passed
    // tautologically, and the first real-device run (test/shadows-gpu T4) measured a 0.31-texel
    // slide per degree of yaw.
    const texelWorld = (2 * radius) / config.resolution;
    vec3.set(_center, _sphere[0], _sphere[1], _sphere[2]);
    const snapView = mat4.lookAt(_snapView, ORIGIN, ld, up);
    const ls = vec3.transformMat4(_ls, _center, snapView);
    ls[0] = Math.floor(ls[0] / texelWorld) * texelWorld;
    ls[1] = Math.floor(ls[1] / texelWorld) * texelWorld;
    // A lookAt matrix is always invertible; gl-matrix types the result nullable regardless.
    const invSnap = mat4.invert(_invSnap, snapView)!;
    const snapped = vec3.transformMat4(_snapped, ls, invSnap);

    // §5.4 — the extruded eye: pushed back along -lightDir so the near plane (z=0) sits
    // `extrusion` units behind the sphere's far side.
    const eye = vec3.set(_eye,
        snapped[0] - ld[0] * (radius + extrusion),
        snapped[1] - ld[1] * (radius + extrusion),
        snapped[2] - ld[2] * (radius + extrusion));
    const zFar = 2 * radius + extrusion;

    const view = mat4.lookAt(_view, eye, snapped, up);
    const proj = mat4.orthoZO(_proj, -radius, radius, -radius, radius, 0, zFar);
    const viewProj = mat4.multiply(_viewProj, proj, view);
    // lookAt/orthoZO are always invertible; gl-matrix types invert() nullable regardless.
    const invViewProj = mat4.invert(_invViewProj, viewProj)!;
    const invView = mat4.invert(_invView, view)!;
    const invProj = mat4.invert(_invProj, proj)!;

    // §5.5 — the virtual fov: the perspective fov that reproduces this cascade's true texel
    // density at its centre (dist = radius + extrusion), so the stock LOD kernel keeps working
    // in the shadow view without a shadow-specific branch.
    const fovRads = 2 * Math.atan(radius / (radius + extrusion));

    // Pack the exact Camera layout (core/Camera.ts:35) — the cascade is then indistinguishable
    // from a camera to every existing shader.
    const viewData = out.viewData;
    viewData.set(viewProj, 0);
    viewData.set(view, 16);
    viewData.set(proj, 32);
    viewData.set(invViewProj, 48);
    viewData.set(invView, 64);
    viewData.set(invProj, 80);
    viewData[96] = eye[0]; viewData[97] = eye[1]; viewData[98] = eye[2];
    viewData[99] = camera.simTime ?? 0;
    viewData[100] = 0;          // the extruded near plane
    viewData[101] = zFar;
    viewData[102] = fovRads;
    viewData[103] = 1.0;        // cascades are square

    out.splitFar = farD;
    out.texelWorld = texelWorld;
    out.center[0] = snapped[0]; out.center[1] = snapped[1]; out.center[2] = snapped[2];
    out.radius = radius;
    return out;
}

/** An empty {@link CascadeResult} with its 108-float `viewData` preallocated. */
export function createCascadeResult(): CascadeResult {
    return {
        viewData: new Float32Array(SHADOW_VIEW_FLOATS),
        splitFar: 0,
        texelWorld: 0,
        center: [0, 0, 0],
        radius: 0,
    };
}

/** `n` reusable {@link CascadeResult}s — the persistent target for {@link buildCascades}. */
export function createCascadeResults(n: number): CascadeResult[] {
    const out: CascadeResult[] = [];
    for (let k = 0; k < n; k++) out.push(createCascadeResult());
    return out;
}

/**
 * Fit the whole cascade set for one frame.
 *
 * Pass `out` — a {@link createCascadeResults} array of exactly `config.cascades` entries — to run
 * allocation-free, which is what `ShadowSystem.update()` does every frame (roadmap constraint #2).
 * Omitting it allocates a fresh set, which is what the headless tests and one-shot callers want.
 *
 * When `out` is supplied its entries are **overwritten in place**, so a caller holding a previous
 * frame's result holds live data, not a snapshot.
 */
export function buildCascades(
    camera: ShadowCameraState,
    lightDir: ArrayLike<number>,
    config: CascadeFitConfig,
    out?: CascadeResult[],
): CascadeResult[] {
    const results = out ?? createCascadeResults(config.cascades);
    if (results.length !== config.cascades) {
        throw new Error(
            `ShadowSystem: buildCascades received an out array of ${results.length} entries for ` +
            `${config.cascades} cascades; it must match exactly (createCascadeResults(cascades)).`);
    }
    const splits = cascadeSplits(camera.near, config.shadowDistance, config.cascades, config.lambda,
        out ? _splitsScratch : undefined);
    let nearD = camera.near;
    for (let k = 0; k < config.cascades; k++) {
        fitCascade(camera, lightDir, nearD, splits[k], config, results[k]);
        nearD = splits[k];
    }
    return results;
}

/** Split scratch, used only on the allocation-free path (when the caller supplied `out`). */
const _splitsScratch: number[] = [];

// ---------------------------------------------------------------------------
// §4.2 ShadowData packing
// ---------------------------------------------------------------------------

/**
 * Pack the receiver uniform (`ShadowData`, 352 bytes at MAX_CASCADES = 4) from the fitted
 * cascades. `out` must be at least {@link SHADOW_DATA_FLOATS} long. Integer fields
 * (`cascadeCount`, `pcfRadius`) are written as raw u32 bits into the same memory.
 */
export function packShadowData(
    cascades: CascadeResult[],
    lightDir: ArrayLike<number>,
    pcfRadius: number,
    blendTexels: number,
    constantBias: number,
    out: Float32Array = new Float32Array(SHADOW_DATA_FLOATS),
): Float32Array {
    // Uint32Array's third arg is a LENGTH in elements, not bytes — out.length floats = u32s.
    // Cached per staging array: this runs every frame, and the view must inherit byteOffset.
    const u32 = u32ViewOf(out);
    for (let k = 0; k < cascades.length; k++) {
        const base = k * SHADOW_CASCADE_FLOATS;
        // viewProj is floats 0..15 of the packed Camera layout. Copied by index rather than
        // `set(subarray(0,16))` — the subarray view was a per-cascade, per-frame allocation.
        const vd = cascades[k].viewData;
        for (let i = 0; i < 16; i++) out[base + i] = vd[i];
        out[base + 16] = cascades[k].splitFar;
        out[base + 17] = cascades[k].texelWorld;
        out[base + 18] = 0;
        out[base + 19] = 0;
    }
    const tail = SHADOW_CASCADE_FLOATS * MAX_CASCADES;
    vec3.set(_ldIn, lightDir[0], lightDir[1], lightDir[2]);
    const ld = vec3.normalize(_ld, _ldIn);
    out[tail] = ld[0]; out[tail + 1] = ld[1]; out[tail + 2] = ld[2]; out[tail + 3] = 0;
    u32[tail + 4] = cascades.length;
    u32[tail + 5] = pcfRadius;
    out[tail + 6] = blendTexels;
    out[tail + 7] = constantBias;
    return out;
}

// ---------------------------------------------------------------------------
// §7.3 The peter-panning guard (§11)
// ---------------------------------------------------------------------------

/**
 * Throws when a cascade's caster normal offset exceeds 2% of its extent — the point where
 * contact shadows visibly detach. Called from `update()`, once per frame, naming the cascade.
 */
export function checkNormalOffset(cascades: CascadeResult[], normalOffsetTexels: number): void {
    for (let k = 0; k < cascades.length; k++) {
        const offset = cascades[k].texelWorld * normalOffsetTexels;
        const extent = 2 * cascades[k].radius;
        if (offset > 0.02 * extent) {
            throw new Error(
                `ShadowSystem: cascade ${k} normal offset (${offset.toFixed(4)} world units) exceeds 2% of ` +
                `its extent (${extent.toFixed(4)}); raise resolution or lower normalOffsetTexels — ` +
                `contact shadows will detach.`);
        }
    }
}
