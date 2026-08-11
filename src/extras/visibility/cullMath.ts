// src/extras/visibility/cullMath.ts
//
// Pure, GPU-free culling math — the CPU mirror of the WGSL kernels in visibilityShaders.ts
// (docs/visibility-design.md §6). Everything here runs in a plain unit test with no device, so
// the frustum/sphere/cone/Hi-Z-level/LOD math is pinned by numbers, not by eyeballing a render.
//
// Two load-bearing conventions, both verified against the engine's Camera (core/Camera.ts):
//  1. Matrices are COLUMN-MAJOR (gl-matrix / WGSL). Element at column c, row r is m[c*4 + r].
//  2. Camera uses gl-matrix `mat4.perspectiveZO` — WebGPU's projection, whose NDC z spans
//     [0, 1] (near plane maps to 0, not -1). The frustum-plane extraction below matches: the
//     near plane is row2 ALONE (clip z >= 0), not row3 + row2 (clip z >= -w, the OpenGL form).
//     The Hi-Z depth test needs no remap: the depth buffer stores the SAME projection's raw
//     NDC z, so `ndcNear = clipZ/clipW` compares like-for-like (§6.4).


// ---------------------------------------------------------------------------
// Instance layout (design doc §4.1) — 96 bytes = 24 floats per instance.
// ---------------------------------------------------------------------------

export const INSTANCE_FLOATS = 24;
/** column-major mat4, floats 0..15. */
export const INSTANCE_MODEL_OFFSET = 0;
/** vec4 (xyz = object-space center, w = radius), floats 16..19. */
export const INSTANCE_BOUNDS_OFFSET = 16;
/** u32 index into meshTable, float 20 (written as integer bits). */
export const INSTANCE_MESH_INDEX_OFFSET = 20;
/** u32, float 21. LOD_AUTO = auto-select. */
export const INSTANCE_LOD_OVERRIDE_OFFSET = 21;
/** u32, float 22. bit0: disable occlusion cull; bit1: disable cone cull. */
export const INSTANCE_FLAGS_OFFSET = 22;
/** padding, float 23. */
export const INSTANCE_PAD_OFFSET = 23;

/** `lodOverride` value meaning "select LOD automatically from screen-space error" (§8). */
export const LOD_AUTO = 0xFFFFFFFF;
/** Instance.flags bit 0 — skip the Hi-Z occlusion test for this instance. */
export const INSTANCE_FLAG_NO_OCCLUSION = 1 << 0;
/** Instance.flags bit 1 — skip the meshlet normal-cone (backface) test for this instance. */
export const INSTANCE_FLAG_NO_CONE_CULL = 1 << 1;

/**
 * A zero-copy, worker-fillable instance pool (design doc §4.1, §9). Holds the shared
 * `Float32Array` the caller fills and the `Uint32Array` view over the SAME memory used to write
 * the integer fields (meshIndex / lodOverride / flags) as raw bits. Adopt by reference into the
 * VisibilitySystem via `setInstances(buffer.data, count)`; nothing is copied.
 */
export class InstanceBuffer {
    /** Float view — write the model matrix and bounding sphere through this. */
    public readonly data: Float32Array;
    /** Integer view over the same memory — used internally for the u32 fields. */
    private readonly u32: Uint32Array;
    /** Number of live instances written so far. */
    public count: number = 0;

    constructor(public readonly capacity: number) {
        const buffer = new ArrayBuffer(capacity * INSTANCE_FLOATS * 4);
        this.data = new Float32Array(buffer);
        this.u32 = new Uint32Array(buffer);
    }

    /**
     * Write one instance. `model` is a column-major 16-float matrix (gl-matrix `mat4`).
     * `boundsCenter` is `[x, y, z, radius]` in object space (copied from the mesh at build time).
     */
    public write(
        index: number,
        model: ArrayLike<number>,
        boundsCenter: ArrayLike<number>,
        meshIndex: number,
        lodOverride: number = LOD_AUTO,
        flags: number = 0,
    ): void {
        if (index < 0 || index >= this.capacity) {
            throw new Error(`InstanceBuffer: write index ${index} out of range (capacity ${this.capacity}).`);
        }
        const f = index * INSTANCE_FLOATS;
        for (let i = 0; i < 16; i++) this.data[f + i] = model[i];
        this.data[f + 16] = boundsCenter[0];
        this.data[f + 17] = boundsCenter[1];
        this.data[f + 18] = boundsCenter[2];
        this.data[f + 19] = boundsCenter[3];
        this.u32[f + INSTANCE_MESH_INDEX_OFFSET] = meshIndex;
        this.u32[f + INSTANCE_LOD_OVERRIDE_OFFSET] = lodOverride;
        this.u32[f + INSTANCE_FLAGS_OFFSET] = flags;
        this.u32[f + INSTANCE_PAD_OFFSET] = 0;
        if (index + 1 > this.count) this.count = index + 1;
    }
}

// ---------------------------------------------------------------------------
// Frustum (design doc §6.1) — Gribb–Hartmann plane extraction + sphere test.
// ---------------------------------------------------------------------------

/**
 * Extract the 6 frustum planes from a column-major view-projection matrix.
 * Returns a flat 24-float array: planes L, R, B, T, N, F, each `[nx, ny, nz, w]` with a
 * normalized normal (so the sphere-radius test is a true distance comparison). Points inside
 * the frustum have `dot(n, p) + w >= 0` on every plane.
 */
export function extractFrustumPlanes(viewProj: ArrayLike<number>): Float32Array {
    const m = viewProj;
    // Row r of the column-major matrix = (m[r], m[4+r], m[8+r], m[12+r]).
    const row = (r: number): number[] => [m[r], m[4 + r], m[8 + r], m[12 + r]];
    const rows = [row(0), row(1), row(2), row(3)];
    // L = r3+r0, R = r3-r0, B = r3+r1, T = r3-r1, F = r3-r2 (design doc §6.1). N = r2 ALONE
    // (WebGPU ZO: inside means clip z >= 0), NOT r3+r2 (OpenGL: clip z >= -w) — [2,2,0] is
    // r2 + 0·r2. Mixing the two forms shrinks the near plane to half its distance.
    const spec: [number, number, number][] = [
        [3, 0, 1], [3, 0, -1], [3, 1, 1], [3, 1, -1], [2, 2, 0], [3, 2, -1],
    ];
    const out = new Float32Array(24);
    for (let i = 0; i < 6; i++) {
        const [a, b, sign] = spec[i];
        const ra = rows[a], rb = rows[b];
        let nx = ra[0] + sign * rb[0];
        let ny = ra[1] + sign * rb[1];
        let nz = ra[2] + sign * rb[2];
        let w = ra[3] + sign * rb[3];
        const len = Math.hypot(nx, ny, nz) || 1;
        nx /= len; ny /= len; nz /= len; w /= len;
        out[i * 4 + 0] = nx; out[i * 4 + 1] = ny; out[i * 4 + 2] = nz; out[i * 4 + 3] = w;
    }
    return out;
}


/**
 * Sphere vs frustum. `planes` is the 24-float result of {@link extractFrustumPlanes}.
 * Returns false only when the sphere is fully outside some plane by more than its radius.
 */
export function sphereInFrustum(
    planes: ArrayLike<number>,
    cx: number, cy: number, cz: number,
    radius: number,
): boolean {
    for (let i = 0; i < 6; i++) {
        const o = i * 4;
        if (planes[o] * cx + planes[o + 1] * cy + planes[o + 2] * cz + planes[o + 3] < -radius) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// World-space sphere from an instance (design doc §6.2).
// ---------------------------------------------------------------------------

/**
 * Transform an object-space bounding sphere by a column-major model matrix. The radius scales
 * by the LARGEST absolute scale factor (max column length of the upper 3×3), which keeps the
 * sphere conservative under non-uniform scale. Writes `[cx, cy, cz, r]` into `out`.
 */
export function worldSphereFromModel(
    model: ArrayLike<number>,
    boundsCenter: ArrayLike<number>,
    out: number[] | Float32Array = [0, 0, 0, 0],
): number[] | Float32Array {
    const x = boundsCenter[0], y = boundsCenter[1], z = boundsCenter[2];
    // center = model * vec4(center, 1); column-major: out[i] = Σ_j model[j*4+i] * v[j] + model[12+i].
    out[0] = model[0] * x + model[4] * y + model[8] * z + model[12];
    out[1] = model[1] * x + model[5] * y + model[9] * z + model[13];
    out[2] = model[2] * x + model[6] * y + model[10] * z + model[14];
    const sx = Math.hypot(model[0], model[1], model[2]);
    const sy = Math.hypot(model[4], model[5], model[6]);
    const sz = Math.hypot(model[8], model[9], model[10]);
    out[3] = boundsCenter[3] * Math.max(sx, Math.max(sy, sz));
    return out;
}

// ---------------------------------------------------------------------------
// Meshlet normal cone (design doc §6.3) — backface cluster test.
// ---------------------------------------------------------------------------

/**
 * True when every triangle in the cluster faces away from the camera. `axisW`/`apexW` are the
 * cone axis/apex in world space (axis already transformed by the normal matrix and renormalized
 * by the caller). `cutoff` is the meshlet's `coneAxis.w`; the 2.0 sentinel means "never cull".
 *
 * The sign convention is the one pinned by geometry-meshlets-design.md §6.2: `>=`, and the
 * direction is camera→apex (`d`), not its negation. Flip either and you cull the VISIBLE clusters.
 */
export function coneIsBackfacing(
    axisW: ArrayLike<number>,
    cutoff: number,
    apexW: ArrayLike<number>,
    eye: ArrayLike<number>,
): boolean {
    if (cutoff >= 2.0) return false; // sentinel: cone too wide to ever cull
    let dx = apexW[0] - eye[0], dy = apexW[1] - eye[1], dz = apexW[2] - eye[2];
    const dl = Math.hypot(dx, dy, dz) || 1;
    dx /= dl; dy /= dl; dz /= dl;
    let ax = axisW[0], ay = axisW[1], az = axisW[2];
    const al = Math.hypot(ax, ay, az) || 1;
    ax /= al; ay /= al; az /= al;
    return (ax * dx + ay * dy + az * dz) >= cutoff;
}

// ---------------------------------------------------------------------------
// Hi-Z mip level selection (design doc §6.4).
// ---------------------------------------------------------------------------

/**
 * The pyramid level whose texel footprint covers a screen-space sphere of `sizePx` pixels:
 * `ceil(log2(max(sizePx, 1)))`, clamped to `[0, maxLevel]`.
 */
export function hiZLevelForFootprint(sizePx: number, maxLevel: number): number {
    const level = Math.ceil(Math.log2(Math.max(sizePx, 1)));
    return Math.min(Math.max(level, 0), maxLevel);
}

/** Number of mip levels in a Hi-Z pyramid for a `width`×`height` depth source (§5.1). */
export function hiZMipCount(width: number, height: number): number {
    return Math.ceil(Math.log2(Math.max(width, height))) + 1;
}

/** Width/height of Hi-Z mip `level` for a `width`×`height` base (floor, min 1). */
export function hiZMipSize(width: number, height: number, level: number): [number, number] {
    const s = Math.pow(2, level);
    return [Math.max(1, Math.floor(width / s)), Math.max(1, Math.floor(height / s))];
}

// ---------------------------------------------------------------------------
// LOD selection (design doc §8).
// ---------------------------------------------------------------------------

/**
 * Projected screen-space error (in pixels) of a LOD level at a given distance.
 * `error` is the level's world-space geometric error; `screenHeightPx`, `fovY` rads.
 */
export function projectedErrorPx(
    error: number,
    distance: number,
    screenHeightPx: number,
    fovYRads: number,
): number {
    const pxPerUnit = screenHeightPx / (2 * distance * Math.tan(fovYRads / 2));
    return error * pxPerUnit;
}

/**
 * Pick the coarsest LOD whose projected error is within `errorThresholdPx`. `errors` are the
 * mesh's per-LOD world-space errors (finest first, LOD0 = 0). Returns the LOD index. A
 * single-LOD mesh (errors = [0]) always returns 0 — it never LODs, the correct degenerate case.
 */
export function selectLod(
    errors: ArrayLike<number>,
    distance: number,
    screenHeightPx: number,
    fovYRads: number,
    errorThresholdPx: number = 1.0,
): number {
    let chosen = 0;
    for (let k = 0; k < errors.length; k++) {
        if (projectedErrorPx(errors[k], distance, screenHeightPx, fovYRads) <= errorThresholdPx) {
            chosen = k; // coarsen while under budget
        } else {
            break;
        }
    }
    return chosen;
}
