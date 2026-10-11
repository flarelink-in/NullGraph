// src/extras/picking/pickMath.ts
//
// The pure half of picking (docs/picking-design.md §4, §7): the pick camera, unprojection, and
// hit selection. No device, so every number here is pinned headless (test/picking.test.mjs).
//
// Matrices are column-major 16-float arrays (gl-matrix / WGSL layout), the camera block is the
// engine's 108-float `Camera` layout (src/core/Camera.ts).

/** Floats in the engine camera block (432 bytes). */
export const CAMERA_FLOATS = 108;

/** Offsets into the camera block (src/core/Camera.ts). */
export const CAM_VIEW_PROJ = 0;
export const CAM_PROJ = 32;
export const CAM_INV_VIEW_PROJ = 48;
export const CAM_INV_PROJ = 80;

/** Texel channels of the pick target (rgba32uint). */
export const PICK_TAG = 0;
export const PICK_INDEX = 1;
export const PICK_DEPTH_BITS = 2;
export const PICK_EXTRA = 3;

/** Largest pick radius the target can be built for — 15×15 texels. */
export const MAX_PICK_RADIUS = 7;

const _m = new Float32Array(16);
const _mInv = new Float32Array(16);
/** Bit-cast scratch: a texel's depth channel is f32 bits stored as u32. */
const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);

/** out = a · b (column-major). `out` may not alias `a` or `b`. */
export function mat4Multiply(a: ArrayLike<number>, b: ArrayLike<number>, out: Float32Array, outOffset = 0): Float32Array {
    for (let c = 0; c < 4; c++) {
        const b0 = b[c * 4], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
        for (let r = 0; r < 4; r++) {
            out[outOffset + c * 4 + r] = a[r] * b0 + a[4 + r] * b1 + a[8 + r] * b2 + a[12 + r] * b3;
        }
    }
    return out;
}

/**
 * The centre of pixel (px, py) in NDC. `py` counts DOWN from the top row, as screen and texture
 * coordinates do; NDC +y is up.
 */
export function pixelCenterNDC(px: number, py: number, width: number, height: number, out: number[] | Float32Array = [0, 0]) {
    out[0] = ((px + 0.5) / width) * 2 - 1;
    out[1] = 1 - ((py + 0.5) / height) * 2;
    return out;
}

/**
 * The clip-space "zoom" onto the (2R+1)² pixels centred on (px, py): after it, those pixels span
 * NDC [-1, 1] exactly. Depth (row z) and w are untouched, so the depth test and the depth a pick
 * returns are the main view's own. Writes M and M⁻¹.
 */
export function pickZoom(
    px: number, py: number, radius: number, width: number, height: number,
    outM: Float32Array, outMInv: Float32Array,
): void {
    const n = 2 * radius + 1;
    const s = width / n, t = height / n;
    const cx = ((px + 0.5) / width) * 2 - 1;
    const cy = 1 - ((py + 0.5) / height) * 2;
    outM.fill(0);
    outM[0] = s; outM[5] = t; outM[10] = 1; outM[15] = 1;
    outM[12] = -s * cx; outM[13] = -t * cy;
    outMInv.fill(0);
    outMInv[0] = 1 / s; outMInv[5] = 1 / t; outMInv[10] = 1; outMInv[15] = 1;
    outMInv[12] = cx; outMInv[13] = cy;
}

/**
 * Build the pick camera: a copy of `main` (the engine's 108-float camera block) whose projection is
 * zoomed onto the (2R+1)² pixels around (px, py). view, invView, eye and the scalars are kept, so
 * anything that reads them — the LOD maths, a shader's eye position — sees the main view.
 */
export function buildPickCamera(
    main: ArrayLike<number>, px: number, py: number, radius: number, width: number, height: number,
    out: Float32Array,
): Float32Array {
    for (let i = 0; i < CAMERA_FLOATS; i++) out[i] = main[i];
    pickZoom(px, py, radius, width, height, _m, _mInv);
    const sub = (o: number) => (main as Float32Array).subarray
        ? (main as Float32Array).subarray(o, o + 16)
        : Array.prototype.slice.call(main, o, o + 16);
    mat4Multiply(_m, sub(CAM_VIEW_PROJ), out, CAM_VIEW_PROJ);
    mat4Multiply(_m, sub(CAM_PROJ), out, CAM_PROJ);
    mat4Multiply(sub(CAM_INV_VIEW_PROJ), _mInv, out, CAM_INV_VIEW_PROJ);
    mat4Multiply(sub(CAM_INV_PROJ), _mInv, out, CAM_INV_PROJ);
    return out;
}

/**
 * World position of the point at pixel (px, py)'s centre and clip depth `depth` ∈ [0, 1], through
 * the MAIN camera's inverse view-projection.
 */
export function unprojectPixel(
    invViewProj: ArrayLike<number>, px: number, py: number, depth: number, width: number, height: number,
    out: number[] | Float32Array = [0, 0, 0],
) {
    const x = ((px + 0.5) / width) * 2 - 1;
    const y = 1 - ((py + 0.5) / height) * 2;
    const m = invViewProj;
    const wx = m[0] * x + m[4] * y + m[8] * depth + m[12];
    const wy = m[1] * x + m[5] * y + m[9] * depth + m[13];
    const wz = m[2] * x + m[6] * y + m[10] * depth + m[14];
    const ww = m[3] * x + m[7] * y + m[11] * depth + m[15];
    out[0] = wx / ww; out[1] = wy / ww; out[2] = wz / ww;
    return out;
}

/** The texel a hit came from, in target coordinates and in screen pixels. */
export interface TexelHit {
    tx: number; ty: number;
    px: number; py: number;
    tag: number; index: number; depth: number; extra: number;
}

/**
 * Choose the hit from a read-back pick target (§7): among non-empty texels within `radius`
 * (Euclidean) of the centre AND inside the W×H screen, the nearest to the centre wins; equal
 * distances go to the smaller depth. `texels` holds 4 u32 per texel, rows `rowStrideU32` apart.
 */
export function selectHit(
    texels: Uint32Array, rowStrideU32: number, maxRadius: number, radius: number,
    px: number, py: number, width: number, height: number,
): TexelHit | null {
    const n = 2 * maxRadius + 1;
    let best: TexelHit | null = null;
    let bestD2 = Infinity;
    for (let ty = 0; ty < n; ty++) {
        const sy = py - maxRadius + ty;
        if (sy < 0 || sy >= height) continue;
        for (let tx = 0; tx < n; tx++) {
            const sx = px - maxRadius + tx;
            if (sx < 0 || sx >= width) continue;
            const dx = tx - maxRadius, dy = ty - maxRadius;
            const d2 = dx * dx + dy * dy;
            if (d2 > radius * radius) continue;
            const o = ty * rowStrideU32 + tx * 4;
            const tag = texels[o + PICK_TAG];
            if (tag === 0) continue;
            _u32[0] = texels[o + PICK_DEPTH_BITS];
            const depth = _f32[0];
            if (d2 < bestD2 || (d2 === bestD2 && best && depth < best.depth)) {
                bestD2 = d2;
                best = { tx, ty, px: sx, py: sy, tag, index: texels[o + PICK_INDEX], depth, extra: texels[o + PICK_EXTRA] };
            }
        }
    }
    return best;
}
