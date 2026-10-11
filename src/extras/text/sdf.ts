// src/extras/text/sdf.ts
//
// Signed distance fields from coverage bitmaps (docs/text-labels-design.md §6) — the engine of
// FontAtlas.fromCanvas. Pure: no canvas, no device, pinned headless against brute force.
//
// Exact squared Euclidean distance transform, separable, after Felzenszwalb & Huttenlocher
// ("Distance Transforms of Sampled Functions", 2012). Anti-aliased coverage seeds each edge pixel
// with its sub-pixel distance to the edge (coverage 0.5 = the edge passes through the centre), the
// refinement Mapbox's TinySDF popularised — so a rasterised glyph's field is accurate to a fraction
// of a pixel rather than biased by half of one.

const INF = 1e20;

/**
 * One row of the 1-D transform, in place on `f` (length n, stride 1 within the scratch arrays):
 * f[q] ← min_p ((q − p)² + f[p]). `v`, `z` are scratch (lengths n and n + 1).
 */
function edt1d(f: Float64Array, d: Float64Array, v: Int32Array, z: Float64Array, n: number): void {
    v[0] = 0; z[0] = -INF; z[1] = INF;
    for (let q = 1, k = 0; q < n; q++) {
        let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
        while (s <= z[k]) {
            k--;
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
        }
        k++; v[k] = q; z[k] = s; z[k + 1] = INF;
    }
    for (let q = 0, k = 0; q < n; q++) {
        while (z[k + 1] < q) k++;
        const dq = q - v[k];
        d[q] = dq * dq + f[v[k]];
    }
    for (let q = 0; q < n; q++) f[q] = d[q];
}

/** 2-D squared EDT, in place on `grid` (width × height, row-major). */
export function squaredEDT(grid: Float64Array, width: number, height: number): Float64Array {
    const n = Math.max(width, height);
    const f = new Float64Array(n), d = new Float64Array(n), v = new Int32Array(n), z = new Float64Array(n + 1);
    for (let x = 0; x < width; x++) {
        for (let y = 0; y < height; y++) f[y] = grid[y * width + x];
        edt1d(f, d, v, z, height);
        for (let y = 0; y < height; y++) grid[y * width + x] = f[y];
    }
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) f[x] = grid[y * width + x];
        edt1d(f, d, v, z, width);
        for (let x = 0; x < width; x++) grid[y * width + x] = f[x];
    }
    return grid;
}

/**
 * Signed distance, in pixels, from each pixel centre to the shape's edge — POSITIVE INSIDE (the
 * MSDF convention). `coverage` is 0..1 per pixel (an alpha channel / 255).
 */
export function computeSDF(coverage: ArrayLike<number>, width: number, height: number, out = new Float32Array(width * height)): Float32Array {
    const size = width * height;
    const outer = new Float64Array(size), inner = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        const a = coverage[i];
        if (a >= 1) { outer[i] = 0; inner[i] = INF; }
        else if (a <= 0) { outer[i] = INF; inner[i] = 0; }
        else {
            const o = Math.max(0, 0.5 - a), n = Math.max(0, a - 0.5);
            outer[i] = o * o; inner[i] = n * n;
        }
    }
    squaredEDT(outer, width, height);
    squaredEDT(inner, width, height);
    for (let i = 0; i < size; i++) out[i] = Math.sqrt(inner[i]) - Math.sqrt(outer[i]);
    return out;
}

/**
 * Encode signed distances (px, positive inside) into bytes the shaders read as
 * `(value − 0.5) · distanceRange`: distances ±distanceRange/2 map to 0..255.
 */
export function encodeSDF(dist: ArrayLike<number>, distanceRange: number, out: Uint8Array, outOffset = 0, outStride = 1): Uint8Array {
    for (let i = 0; i < dist.length; i++) {
        const v = 0.5 + dist[i] / distanceRange;
        out[outOffset + i * outStride] = Math.round(Math.min(1, Math.max(0, v)) * 255);
    }
    return out;
}

/** A packed rectangle. */
export interface ShelfRect { x: number; y: number; w: number; h: number }

/**
 * Shelf-pack rectangles into a `width`-wide atlas (1 px gutters). Returns positions in input order
 * and the height used. Throws when a rectangle is wider than the atlas.
 */
export function shelfPack(sizes: { w: number; h: number }[], width: number): { rects: ShelfRect[]; height: number } {
    const order = sizes.map((s, i) => i).sort((a, b) => sizes[b].h - sizes[a].h);
    const rects: ShelfRect[] = new Array(sizes.length);
    let x = 1, y = 1, shelf = 0;
    for (const i of order) {
        const { w, h } = sizes[i];
        if (w + 2 > width) throw new Error(`shelfPack: a ${w}px-wide glyph does not fit a ${width}px atlas.`);
        if (x + w + 1 > width) { x = 1; y += shelf + 1; shelf = 0; }
        rects[i] = { x, y, w, h };
        x += w + 1;
        shelf = Math.max(shelf, h);
    }
    return { rects, height: y + shelf + 1 };
}
