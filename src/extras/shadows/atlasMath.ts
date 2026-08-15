// src/extras/shadows/atlasMath.ts
//
// Pure, GPU-free math for the Phase 2 shadow atlas (docs/shadows-atlas-design.md). Everything
// here runs in a plain unit test with no device, so the allocator, the view builders, the cube
// face select, the tap clamp, and the slot packing are pinned by numbers — the same discipline
// shadowMath.ts (Phase 1) and cullMath.ts (visibility) follow.
//
// Conventions (identical to shadowMath.ts / cullMath.ts, verified against core/Camera.ts):
//  1. Matrices are COLUMN-MAJOR (gl-matrix / WGSL). Element at column c, row r is m[c*4 + r].
//  2. Clip z spans [0, 1] (WebGPU ZO). Spot/point projections use mat4.perspectiveZO — the NO
//     form silently moves the near plane, the trap Camera.ts:62-66 documents and Phase 1 §10 U4
//     pins.

import { mat4, vec3 } from 'gl-matrix';
import { SHADOW_VIEW_FLOATS, u32ViewOf } from './shadowMath';

// ---------------------------------------------------------------------------
// Scratch — module-scope temporaries (same contract as shadowMath.ts's)
// ---------------------------------------------------------------------------
//
// The punctual view builders run once per view per frame (six times for a single point light),
// so their gl-matrix temporaries are hoisted here. Roadmap constraint #2. Every builder still
// allocates by default when no `out` is supplied, so the headless tests are unaffected.

const _eye = vec3.create();
const _dir = vec3.create();
const _target = vec3.create();
const _view = mat4.create();
const _proj = mat4.create();
const _viewProj = mat4.create();
const _invViewProj = mat4.create();
const _invView = mat4.create();
const _invProj = mat4.create();
const UP_Y: vec3 = vec3.fromValues(0, 1, 0);
const UP_Z: vec3 = vec3.fromValues(0, 0, 1);

// ---------------------------------------------------------------------------
// Constants — the WGSL-facing layouts (atlasShaders.ts must match float-for-float)
// ---------------------------------------------------------------------------

/** Floats per `ShadowView` record in the receiver storage array (96 bytes, 16-aligned). §4.1. */
export const SHADOW_VIEW_RECORD_FLOATS = 24;

/** Floats per `ShadowSlot` record in the receiver storage array (16 bytes). §4.2. */
export const SHADOW_SLOT_RECORD_FLOATS = 4;

/** `ShadowView.kind`: a directional cascade. */
export const SHADOW_VIEW_KIND_CASCADE = 0;
/** `ShadowView.kind`: a spot cone. */
export const SHADOW_VIEW_KIND_SPOT = 1;
/** `ShadowView.kind`: one of a point light's six cube faces. */
export const SHADOW_VIEW_KIND_CUBE = 2;

/** `ShadowSlot.kind`: a directional light (owns `cascadeCount` consecutive views). */
export const SHADOW_SLOT_KIND_DIRECTIONAL = 0;
/** `ShadowSlot.kind`: a spot light (owns 1 view). */
export const SHADOW_SLOT_KIND_SPOT = 1;
/** `ShadowSlot.kind`: a point light (owns exactly 6 views, +X,-X,+Y,-Y,+Z,-Z). */
export const SHADOW_SLOT_KIND_POINT = 2;

/**
 * The tile-size ladder (§5.1). Importance maps to one of these power-of-two sizes; hysteresis
 * keeps a resident tile from re-rendering on a small importance wobble.
 */
export const TILE_LADDER: readonly number[] = [2048, 1024, 512, 256, 128];

/** The default hysteresis band (§5.1): a tile resizes only when importance crosses a ladder
 *  threshold by more than this fraction. */
export const DEFAULT_RESIZE_HYSTERESIS = 0.25;

/** The largest shadow slot index `spot_params.z` can encode (§11): the field carries
 *  `slot + 1` as a float, and 0 means "no shadow", so slots top out at 254. */
export const MAX_SHADOW_SLOT = 254;

// ---------------------------------------------------------------------------
// §5.1 Importance → tile size, with hysteresis
// ---------------------------------------------------------------------------

/**
 * A light's shadow importance, evaluated on the CPU once per frame (§5.1). Cascades dominate by
 * construction (`1 / (cascadeIndex + 1)`); a punctual light's importance is its screen-space
 * footprint fraction scaled by the caller's `userImportance`.
 */
export interface ShadowViewRequest {
    kind: 'cascade' | 'spot' | 'point';
    /** Cascades only: which split this view covers (0 = nearest). */
    cascadeIndex?: number;
    /** Punctual only: world-space light position. */
    position?: ArrayLike<number>;
    /** Punctual only: light range (world units). */
    range?: number;
    /** Caller-supplied importance multiplier. Default 1. */
    userImportance?: number;
}

/** The camera state the importance heuristic needs (plain data, worker-safe). */
export interface ImportanceCameraState {
    eye: ArrayLike<number>;
    near: number;
    screenH: number;
}

/**
 * Screen-space footprint heuristic (§5.1): how many pixels does this light's influence cover?
 * Returns a value in `[0, 1]` for punctual lights; cascades return `1/(i+1)` (cascade 0 = 1).
 */
export function importance(view: ShadowViewRequest, camera: ImportanceCameraState): number {
    if (view.kind === 'cascade') return 1.0 / ((view.cascadeIndex ?? 0) + 1);
    const pos = view.position ?? [0, 0, 0];
    const dx = camera.eye[0] - pos[0], dy = camera.eye[1] - pos[1], dz = camera.eye[2] - pos[2];
    const dist = Math.hypot(dx, dy, dz);
    const range = view.range ?? 1;
    const px = (range / Math.max(dist, camera.near)) * camera.screenH;
    const frac = Math.min(Math.max(px / camera.screenH, 0), 1);
    return frac * (view.userImportance ?? 1);
}

/**
 * Map an importance value to a ladder tile size. Higher importance → larger tile. The ladder is
 * scanned coarse-to-fine so the result is the LARGEST tile whose threshold the importance meets.
 *
 * Thresholds are the ladder midpoints in importance space: a tile of size `LADDER[i]` is chosen
 * when `importance >= threshold(i)`, where `threshold` spaces the ladder evenly over [0,1].
 */
/**
 * Index of the first ladder rung not larger than `maxTileSize`. The ladder is sorted descending,
 * so this is the start of the usable window — `TILE_LADDER.filter(...)` allocated an array on
 * every call, and these run once per view per frame (roadmap constraint #2).
 */
function ladderStart(maxTileSize: number): number {
    for (let i = 0; i < TILE_LADDER.length; i++) {
        if (TILE_LADDER[i] <= maxTileSize) return i;
    }
    return TILE_LADDER.length - 1;   // maxTileSize below the whole ladder → the smallest rung
}

export function tileSizeForImportance(imp: number, maxTileSize: number = TILE_LADDER[0]): number {
    // Clamp importance to [0,1] and pick the ladder rung. Rung 0 (largest) needs imp near 1.
    const t = Math.min(Math.max(imp, 0), 1);
    // The usable ladder window, capped at maxTileSize.
    const start = ladderStart(maxTileSize);
    const n = TILE_LADDER.length - start;
    // Map t∈[0,1] to a rung index: t=1 → 0 (largest), t=0 → n-1 (smallest).
    const rung = Math.min(n - 1, Math.floor((1 - t) * n));
    return TILE_LADDER[start + rung];
}

/**
 * Hysteresis (§5.1): given a resident tile's current size and a freshly-computed importance,
 * return the size the tile SHOULD have — which equals the current size unless importance has
 * crossed a ladder threshold by more than `hysteresis`. This is what keeps a cached tile from
 * re-rendering (a resize is a full re-render and a visible quality step).
 *
 * Returns `{ size, changed }`. `changed` is true only when the size actually moved.
 */
export function resolveTileSize(
    currentSize: number,
    imp: number,
    hysteresis: number = DEFAULT_RESIZE_HYSTERESIS,
    maxTileSize: number = TILE_LADDER[0],
): { size: number; changed: boolean } {
    const start = ladderStart(maxTileSize);
    const n = TILE_LADDER.length - start;
    const currentRung = Math.max(0, TILE_LADDER.indexOf(currentSize) - start);
    // The importance band the current rung occupies: [rungLo, rungHi] in t-space.
    // rung i covers t ∈ [1 - (i+1)/n, 1 - i/n].
    const rungLo = 1 - (currentRung + 1) / n;
    const rungHi = 1 - currentRung / n;
    const t = Math.min(Math.max(imp, 0), 1);
    // Only move when t leaves the band by more than the hysteresis fraction of the band width.
    const band = (rungHi - rungLo) || (1 / n);
    if (t > rungHi + hysteresis * band || t < rungLo - hysteresis * band) {
        const size = tileSizeForImportance(imp, maxTileSize);
        return { size, changed: size !== currentSize };
    }
    return { size: currentSize, changed: false };
}

// ---------------------------------------------------------------------------
// §5.1 Quad-tree free-list allocator
// ---------------------------------------------------------------------------

/**
 * A quad-tree node over the square atlas. Leaves are either free or occupied; internal nodes
 * have four children in row-major quadrant order (TL, TR, BL, BR). The tree is implicit — nodes
 * are created lazily on subdivision and coalesced on free.
 */
interface QuadNode {
    x: number; y: number; size: number;
    occupied: boolean;
    children: QuadNode[] | null;
}

function makeNode(x: number, y: number, size: number): QuadNode {
    return { x, y, size, occupied: false, children: null };
}

/**
 * A quad-tree free-list over a power-of-two square atlas (§5.1). Allocation returns the
 * top-left corner of a free power-of-two tile of the requested size; freeing returns it. Splits
 * and coalesces lazily. Pure CPU — the GPU never sees the tree, only the resulting rects.
 *
 * The allocator is deterministic: same request sequence → same rects. That determinism is what
 * makes U1 (no overlap) and U8 (eviction) testable by number rather than by eye.
 */
export class QuadTreeAllocator {
    private root: QuadNode;
    public readonly size: number;
    /** Smallest allocatable tile — subdivision stops here. */
    public readonly minTile: number;

    constructor(atlasSize: number, minTile: number = TILE_LADDER[TILE_LADDER.length - 1]) {
        if (!Number.isInteger(atlasSize) || atlasSize < 1 || (atlasSize & (atlasSize - 1)) !== 0) {
            throw new Error(
                `ShadowAtlas: atlasSize ${atlasSize} must be a power of two; the quad-tree assumes it.`);
        }
        this.size = atlasSize;
        this.minTile = minTile;
        this.root = makeNode(0, 0, atlasSize);
    }

    /**
     * Allocate a `size`×`size` tile. Returns `[x, y]` in atlas pixels, or `null` when no free
     * region is large enough (fragmentation included — §11's "free but fragmented" error reads
     * this as null and reports the free area).
     */
    public alloc(size: number): [number, number] | null {
        if (size > this.size || size < this.minTile || (size & (size - 1)) !== 0) return null;
        return this.allocIn(this.root, size);
    }

    private allocIn(node: QuadNode, size: number): [number, number] | null {
        if (node.occupied) return null;
        if (node.size === size) {
            if (node.children) return null; // an internal node can't be handed out whole
            node.occupied = true;
            return [node.x, node.y];
        }
        // Need to descend. Split if this is an untouched leaf large enough to hold the tile.
        if (!node.children) {
            if (node.size / 2 < size) return null; // can't subdivide finely enough here
            const h = node.size / 2;
            node.children = [
                makeNode(node.x, node.y, h),
                makeNode(node.x + h, node.y, h),
                makeNode(node.x, node.y + h, h),
                makeNode(node.x + h, node.y + h, h),
            ];
        }
        for (const c of node.children) {
            const r = this.allocIn(c, size);
            if (r) return r;
        }
        return null;
    }

    /** Free the tile at `[x, y]` of the given size. Coalesces fully-free parents. */
    public free(x: number, y: number, size: number): void {
        this.freeIn(this.root, x, y, size);
    }

    private freeIn(node: QuadNode, x: number, y: number, size: number): boolean {
        if (node.size === size && node.x === x && node.y === y) {
            node.occupied = false;
            return true;
        }
        if (!node.children) return false;
        let freed = false;
        for (const c of node.children) {
            if (this.freeIn(c, x, y, size)) { freed = true; break; }
        }
        // Coalesce: if all four children are free leaves, collapse back to one free leaf.
        if (node.children && node.children.every(c => !c.occupied && !c.children)) {
            node.children = null;
        }
        return freed;
    }

    /** Total free area in px², and the number of free regions (for the §11 fragmentation error). */
    public freeStats(): { freePx: number; regions: number } {
        let freePx = 0, regions = 0;
        const walk = (n: QuadNode): void => {
            if (n.occupied) return;
            if (!n.children) { freePx += n.size * n.size; regions++; return; }
            for (const c of n.children) walk(c);
        };
        walk(this.root);
        return { freePx, regions };
    }

    /** Every occupied rect, as `[x, y, size]`. Used by U1's pairwise-overlap assertion. */
    public occupiedRects(): [number, number, number][] {
        const out: [number, number, number][] = [];
        const walk = (n: QuadNode): void => {
            if (n.occupied) { out.push([n.x, n.y, n.size]); return; }
            if (n.children) for (const c of n.children) walk(c);
        };
        walk(this.root);
        return out;
    }
}

// ---------------------------------------------------------------------------
// §9.1 The three view builders — all produce the same 108-float Camera layout
// ---------------------------------------------------------------------------

/** The result of building one shadow view: the packed Camera block plus bookkeeping. */
export interface BuiltView {
    /** 108 floats, the exact `Camera` layout — written verbatim into the view's GPUBuffer. */
    viewData: Float32Array;
    /** `ShadowView.kind` for the receiver table. */
    kind: number;
    /** World size of one texel at the view's reference depth (bias + filter radius). */
    texelWorld: number;
    /** Cascades: view-space depth covered; punctual: light range. */
    splitFar: number;
}

/**
 * Pack a view/proj pair into the exact 108-float `Camera` layout (core/Camera.ts:35), the same
 * packing shadowMath.ts's fitCascade uses. `eye`, `zNear`, `zFar`, `fovRads`, `aspect` fill the
 * scalar tail; `simTime` rides slot 99 verbatim.
 */
export function packViewCamera(
    view: mat4,
    proj: mat4,
    eye: ArrayLike<number>,
    zNear: number,
    zFar: number,
    fovRads: number,
    aspect: number,
    simTime: number = 0,
    out: Float32Array = new Float32Array(SHADOW_VIEW_FLOATS),
): Float32Array {
    const viewProj = mat4.multiply(_viewProj, proj, view);
    const invViewProj = mat4.invert(_invViewProj, viewProj)!;
    const invView = mat4.invert(_invView, view)!;
    const invProj = mat4.invert(_invProj, proj)!;

    const d = out;
    d.set(viewProj, 0);
    d.set(view, 16);
    d.set(proj, 32);
    d.set(invViewProj, 48);
    d.set(invView, 64);
    d.set(invProj, 80);
    d[96] = eye[0]; d[97] = eye[1]; d[98] = eye[2];
    d[99] = simTime;
    d[100] = zNear;
    d[101] = zFar;
    d[102] = fovRads;
    d[103] = aspect;
    return d;
}

/**
 * §9.1 spot view: `perspectiveZO(2·acos(outerCos), 1, near, range)` looking along `direction`.
 * The fov comes straight from `spot_params.y = cos(outer)`, so the cone and its shadow can never
 * disagree (U6 pins `fovRads ≈ 2·acos(outerCos)`).
 */
export function spotView(
    position: ArrayLike<number>,
    direction: ArrayLike<number>,
    outerCos: number,
    range: number,
    near: number,
    tileSize: number,
    simTime: number = 0,
    out: BuiltView = createBuiltView(),
): BuiltView {
    const fovRads = 2 * Math.acos(Math.min(Math.max(outerCos, -1), 1));
    const eye = vec3.set(_eye, position[0], position[1], position[2]);
    vec3.set(_target, direction[0], direction[1], direction[2]);
    const dir = vec3.normalize(_dir, _target);
    const target = vec3.add(_target, eye, dir);
    const up: vec3 = Math.abs(dir[1]) > 0.999 ? UP_Z : UP_Y;
    const view = mat4.lookAt(_view, eye, target, up);
    const proj = mat4.perspectiveZO(_proj, fovRads, 1.0, near, range);
    // texelWorld at the range far plane: the frustum is `2·range·tan(fov/2)` wide there.
    packViewCamera(view, proj, eye, near, range, fovRads, 1.0, simTime, out.viewData);
    out.kind = SHADOW_VIEW_KIND_SPOT;
    out.texelWorld = (2 * range * Math.tan(fovRads / 2)) / tileSize;
    out.splitFar = range;
    return out;
}

/**
 * The six cube-face directions and ups, in the §4.2 build order `+X,−X,+Y,−Y,+Z,−Z`. Face
 * selection at sample time is arithmetic (cubeFace below), so the build order is load-bearing.
 */
const CUBE_DIRS: [number, number, number][] = [
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
];
const CUBE_UPS: [number, number, number][] = [
    [0, -1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [0, -1, 0], [0, -1, 0],
];

/**
 * §9.1 cube-face view: `perspectiveZO(π/2 + margin, 1, near, range)` at `position`, looking along
 * face `faceIndex`'s axis. The fov is widened by `2·atan(1 + 2·filterTexels/tileSize)` so PCF taps
 * near a face edge read real data inside the tile rather than past the frustum (§9.1, §9.4).
 */
export function cubeFaceView(
    position: ArrayLike<number>,
    faceIndex: number,
    range: number,
    near: number,
    tileSize: number,
    filterTexels: number,
    simTime: number = 0,
    out: BuiltView = createBuiltView(),
): BuiltView {
    if (faceIndex < 0 || faceIndex > 5) {
        throw new Error(`ShadowSystem: cubeFaceView face index ${faceIndex} out of range [0,6).`);
    }
    // The widened fov (§9.1): exactly 90° tiles the sphere; the margin covers the filter kernel.
    const fovRads = 2 * Math.atan(1 + (2 * filterTexels) / tileSize);
    const eye = vec3.set(_eye, position[0], position[1], position[2]);
    const d = CUBE_DIRS[faceIndex];
    const target = vec3.set(_target, eye[0] + d[0], eye[1] + d[1], eye[2] + d[2]);
    const view = mat4.lookAt(_view, eye, target, CUBE_UPS[faceIndex]);
    const proj = mat4.perspectiveZO(_proj, fovRads, 1.0, near, range);
    packViewCamera(view, proj, eye, near, range, fovRads, 1.0, simTime, out.viewData);
    out.kind = SHADOW_VIEW_KIND_CUBE;
    out.texelWorld = (2 * range * Math.tan(fovRads / 2)) / tileSize;
    out.splitFar = range;
    return out;
}

/** An empty {@link BuiltView} with its 108-float `viewData` preallocated — the reusable target
 *  the atlas path passes to {@link spotView} / {@link cubeFaceView} every frame. */
export function createBuiltView(): BuiltView {
    return {
        viewData: new Float32Array(SHADOW_VIEW_FLOATS),
        kind: SHADOW_VIEW_KIND_CUBE,
        texelWorld: 0,
        splitFar: 0,
    };
}

// ---------------------------------------------------------------------------
// §9.2 cubeFace — the CPU mirror of the WGSL major-axis select
// ---------------------------------------------------------------------------

/**
 * Major-axis cube face select, matching the `+X,−X,+Y,−Y,+Z,−Z` build order (§9.2). `d` is the
 * direction FROM the light TO the shaded point. Returns a face index in `[0, 6)`. This is the
 * exact CPU mirror of the WGSL `cubeFace()` in atlasShaders.ts — U2 pins them to agree.
 */
export function cubeFace(d: ArrayLike<number>): number {
    const ax = Math.abs(d[0]), ay = Math.abs(d[1]), az = Math.abs(d[2]);
    if (ax >= ay && ax >= az) return d[0] > 0 ? 0 : 1;
    if (ay >= az) return d[1] > 0 ? 2 : 3;
    return d[2] > 0 ? 4 : 5;
}

// ---------------------------------------------------------------------------
// §9.3 The tap clamp — the CPU mirror of sampleView's atlas-UV math
// ---------------------------------------------------------------------------

/**
 * The atlas UV of one PCF tap (§9.3). `uv` is the view-local `[0,1]²` coordinate; `offset` is the
 * integer tap offset; `texel` is `filterTexels / atlasSize` in ATLAS uv units; `rect` is the
 * view's `[scaleX, scaleY, offsetX, offsetY]` atlas transform. The clamp to `[0,1]²` BEFORE the
 * rect transform is the atlas's defining sampling rule (§9.4): a tap can never read a
 * neighbouring light's depth. U3 pins this.
 */
export function tapAtlasUV(
    uv: [number, number],
    offset: [number, number],
    texel: number,
    rectScaleX: number,
    rect: [number, number, number, number],
): [number, number] {
    // The view-local texel size is the atlas texel divided by the rect's uv scale (the tile's
    // share of the atlas). Step in view-local uv, clamp inside the tile, then map to the atlas.
    const localTexel = texel / rectScaleX;
    const lx = Math.min(Math.max(uv[0] + offset[0] * localTexel, 0), 1);
    const ly = Math.min(Math.max(uv[1] + offset[1] * localTexel, 0), 1);
    return [lx * rect[0] + rect[2], ly * rect[1] + rect[3]];
}

// ---------------------------------------------------------------------------
// §4.3 Slot packing — the spot_params.z contract
// ---------------------------------------------------------------------------

/**
 * Encode a shadow slot for `spot_params.z` (§4.3): the field carries `slot + 1` so `0` means
 * "this light does not cast." Throws the §11 error when the slot exceeds the encodable range.
 */
export function packSlotIndex(slot: number): number {
    if (slot > MAX_SHADOW_SLOT) {
        throw new Error(
            `ShadowSystem: shadow slot ${slot} exceeds ${MAX_SHADOW_SLOT}; ` +
            `spot_params.z encodes slot+1 as a float.`);
    }
    return slot + 1;
}

/** The value a non-casting light carries in `spot_params.z` (§4.3). */
export const SHADOW_SLOT_NONE = 0;

// ---------------------------------------------------------------------------
// §4.1 / §4.2 receiver-table packing
// ---------------------------------------------------------------------------

/**
 * Pack one `ShadowView` receiver record (24 floats, §4.1) into `out` at float offset `o`.
 * `viewData` is the 108-float Camera block (only its viewProj, floats 0..15, is read).
 * `rect` is the atlas UV transform `[scaleX, scaleY, offsetX, offsetY]`.
 */
export function packShadowView(
    out: Float32Array,
    o: number,
    viewData: ArrayLike<number>,
    rect: [number, number, number, number],
    texelWorld: number,
    splitFar: number,
    kind: number,
    /** Float offset of this view's Camera block within `viewData`. Lets a caller pass the whole
     *  `maxViews × 108` staging array instead of a per-view `subarray` — the subarray was a
     *  per-view, per-frame allocation (roadmap constraint #2). */
    viewDataOffset: number = 0,
): void {
    for (let i = 0; i < 16; i++) out[o + i] = viewData[viewDataOffset + i]; // viewProj
    out[o + 16] = rect[0]; out[o + 17] = rect[1]; out[o + 18] = rect[2]; out[o + 19] = rect[3];
    out[o + 20] = texelWorld;
    out[o + 21] = splitFar;
    const u32 = u32ViewOf(out);
    u32[o + 22] = kind;
    u32[o + 23] = 0;
}

/**
 * Pack one `ShadowSlot` record (4 floats, §4.2) into `out` at float offset `o`. Integer fields
 * are written as raw u32 bits into the same memory.
 */
export function packShadowSlot(
    out: Float32Array,
    o: number,
    firstView: number,
    viewCount: number,
    kind: number,
    lightRange: number,
): void {
    const u32 = u32ViewOf(out);
    u32[o] = firstView;
    u32[o + 1] = viewCount;
    u32[o + 2] = kind;
    out[o + 3] = lightRange;
}
