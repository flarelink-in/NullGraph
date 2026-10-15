// What the rigid and skinned bakes share (docs/agents-animated-models-design.md §3,
// docs/agents-skinned-models-design.md §3): sampling a GLB's animation at fixed frames, the nodes'
// transforms, the clip table, the 3×4 rows a frame stores, and welding. Internal: the bakes export
// their own types and entry points.
import type { GLBAnimation, GLBData, GLBNode } from './types';
import { MathUtils } from './MathUtils';

/** Throws with the calling bake's name. */
export type Fail = (message: string) => never;

/** A clip to bake, by its name in the file. */
export interface BakeClipRequest {
    readonly name: string;
    /** The clip repeats (idle, walk, run), or plays once and holds its end (open, a fall). Default true. */
    readonly loop?: boolean;
}

export interface BakeOptions {
    /** Frames per second of each clip in the table. Default 30. */
    readonly fps?: number;
    /** The clips to bake, in this order. Default: every clip of the file, in its order, looping. */
    readonly clips?: readonly (string | BakeClipRequest)[];
    /**
     * Merge the vertices that share position, uv and influences. Default true. Low-poly models repeat
     * a vertex for every face around it, so that each face has its own normal; merged, the vertex stage
     * runs about half as often, and a face's normal is found per pixel instead (from the derivatives of
     * its position). A merged vertex's normal is the mean of the normals it replaces, a smooth normal.
     * `false` keeps the parser's vertices as they are.
     */
    readonly weld?: boolean;
}

/** One clip in the table. */
export interface BakedClip {
    readonly name: string;
    /** Its first frame in the whole table. */
    readonly firstFrame: number;
    readonly frames: number;
    /** Its length in the file. */
    readonly seconds: number;
    /**
     * Looping, frame f is the pose at f / frames of the clip, and the frame after the last is the
     * first. Not looping, frame f is at f / (frames − 1): the last frame is the clip's end.
     */
    readonly loop: boolean;
}

/** Every vertex in every frame, and so in any blend of frames: a box, and the radius about the origin. */
export interface BakedBounds { readonly min: readonly number[]; readonly max: readonly number[]; readonly radius: number }

type Sampler = GLBAnimation['samplers'][number];
interface TRS { t: number[]; r: number[]; s: number[] }
const KEYS = { translation: 't', rotation: 'r', scale: 's' } as const;

function normalize(q: number[]): number[] {
    const length = Math.hypot(...q);
    return q.map(x => x / length);
}

/** The shorter way round between two rotations; close ones are blended and normalized. */
function slerp(a: number[], b: number[], u: number): number[] {
    let cos = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    const sign = cos < 0 ? -1 : 1;
    cos *= sign;
    let wa = 1 - u, wb = u;
    if (cos < 0.9999) {
        const angle = Math.acos(cos), sin = Math.sin(angle);
        wa = Math.sin((1 - u) * angle) / sin;
        wb = Math.sin(u * angle) / sin;
    }
    return normalize(a.map((x, j) => wa * x + sign * wb * b[j]));
}

/** A sampler's value at time t (glTF 2.0, "Animations"): the first key's before it, the last key's after. */
function sample(s: Sampler, n: number, rotation: boolean, t: number, clip: string, fail: Fail): number[] {
    const times = s.input, values = s.output, last = times.length - 1, kind = s.interpolation;
    if (kind !== 'LINEAR' && kind !== 'STEP' && kind !== 'CUBICSPLINE') {
        fail(`clip "${clip}" interpolates by ${kind}; glTF's are LINEAR, STEP and CUBICSPLINE.`);
    }
    const cubic = kind === 'CUBICSPLINE';
    // A cubic key holds its in-tangent, its value, then its out-tangent.
    const at = (key: number, part = 1) => (cubic ? 3 * key + part : key) * n;
    const value = (key: number) => Array.from(values.subarray(at(key), at(key) + n));
    if (t <= times[0]) return value(0);
    if (t >= times[last]) return value(last);
    let k = 0;
    while (times[k + 1] <= t) k++;
    if (kind === 'STEP') return value(k);
    const dt = times[k + 1] - times[k], u = (t - times[k]) / dt;
    if (!cubic) {
        const a = value(k), b = value(k + 1);
        return rotation ? slerp(a, b, u) : a.map((x, j) => x + (b[j] - x) * u);
    }
    const u2 = u * u, u3 = u2 * u;
    const out = Array.from({ length: n }, (_, j) =>
        (2 * u3 - 3 * u2 + 1) * values[at(k) + j] + (u3 - 2 * u2 + u) * dt * values[at(k, 2) + j] +
        (3 * u2 - 2 * u3) * values[at(k + 1) + j] + (u3 - u2) * dt * values[at(k + 1, 0) + j]);
    return rotation ? normalize(out) : out;
}

/** A node's local transform: its matrix, or its translation, rotation and scale (`trs` when animated). */
function local(node: GLBNode, trs?: TRS): Float32Array {
    if (trs) return MathUtils.fromTRS(trs.t, trs.r, trs.s);
    if (node.matrix) return new Float32Array(node.matrix);
    return MathUtils.fromTRS(node.translation ?? [0, 0, 0], node.rotation ?? [0, 0, 0, 1], node.scale ?? [1, 1, 1]);
}

/** The inverse of an affine transform (column-major), or null when it has none. */
export function invertAffine(m: Float32Array): Float32Array | null {
    const [a, d, g, , b, e, h, , c, f, i] = m;
    const c11 = e * i - f * h, c12 = f * g - d * i, c13 = d * h - e * g;
    const det = a * c11 + b * c12 + c * c13;
    if (!(Math.abs(det) > 0)) return null;
    const r = [c11, c12, c13, 0, c * h - b * i, a * i - c * g, b * g - a * h, 0, b * f - c * e, c * d - a * f, a * e - b * d, 0].map(x => x / det);
    for (let k = 0; k < 3; k++) r[12 + k] = -(r[k] * m[12] + r[4 + k] * m[13] + r[8 + k] * m[14]);
    r[15] = 1;
    return new Float32Array(r);
}

/** The 3×4 a table stores for a column-major transform: three rows of four. */
export const rowsOf = (m: Float32Array) => [m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14]];

export interface Timeline {
    /** World transforms at rest, multiplied from the top of each chain down, as GLBParser bakes meshes. */
    rest(node: number): Float32Array;
    readonly clips: (BakedClip & { animation: GLBAnimation })[];
    readonly frames: number;
    /** Calls `frame` for every frame of every clip, with each node's world transform at that time. */
    each(frame: (index: number, world: (node: number) => Float32Array) => void): void;
}

/** The clips asked for, their frames at `fps`, and each frame's node transforms. */
export function timeline(glb: GLBData, options: BakeOptions, fail: Fail): Timeline {
    const fps = options.fps ?? 30, nodes = glb.nodes ?? [];
    if (!(fps > 0 && fps < Infinity)) fail(`fps must be a number above 0; got ${fps}.`);
    const animations = glb.animations ?? [], names = animations.map(a => a.name);
    const asked = (options.clips ?? names).map(c => (typeof c === 'string' ? { name: c, loop: true } : { name: c.name, loop: c.loop ?? true }));
    if (!asked.length) fail(options.clips ? 'clips is empty.' : 'the file has no clips.');
    let frames = 0;
    const clips = asked.map(({ name, loop }) => {
        const animation = animations.find(a => a.name === name);
        if (!animation) fail(`no clip "${name}" in the file; it has ${names.length ? names.map(n => `"${n}"`).join(', ') : 'none'}.`);
        const seconds = animation.maxTime, steps = Math.round(seconds * fps);
        const clip = { name, firstFrame: frames, frames: loop ? Math.max(1, steps) : steps + 1, seconds, loop, animation };
        frames += clip.frames;
        return clip;
    });
    const parent = new Int32Array(nodes.length).fill(-1);
    nodes.forEach((node, n) => node.children?.forEach(child => { parent[child] = n; }));
    const world = (node: number, localOf: (node: number) => Float32Array) => {
        const chain: number[] = [];
        for (let n = node; n >= 0; n = parent[n]) chain.push(n);
        let m = localOf(chain.pop()!);
        while (chain.length) m = MathUtils.multiply(m, localOf(chain.pop()!));
        return m;
    };
    return {
        rest: node => world(node, n => local(nodes[n])),
        clips,
        frames,
        each(frame) {
            for (const clip of clips) {
                for (let f = 0; f < clip.frames; f++) {
                    const time = clip.loop ? f * clip.seconds / clip.frames : clip.frames > 1 ? f * clip.seconds / (clip.frames - 1) : 0;
                    const pose = new Map<number, TRS>();
                    for (const { targetNode, path, samplerIndex } of clip.animation.channels) {
                        const key = KEYS[path];
                        if (!key || targetNode === undefined) continue;
                        const node = nodes[targetNode];
                        if (node.matrix) fail(`node ${targetNode} ("${node.name ?? ''}") has a matrix and is animated; glTF animates translation, rotation and scale only.`);
                        const trs = pose.get(targetNode) ?? { t: node.translation ?? [0, 0, 0], r: node.rotation ?? [0, 0, 0, 1], s: node.scale ?? [1, 1, 1] };
                        trs[key] = sample(clip.animation.samplers[samplerIndex], key === 'r' ? 4 : 3, key === 'r', time, clip.name, fail);
                        pose.set(targetNode, trs);
                    }
                    frame(clip.firstFrame + f, node => world(node, n => local(nodes[n], pose.get(n))));
                }
            }
        },
    };
}

/**
 * The vertices that share position, uv and `extra` (`width` values per vertex, compared by their bits)
 * merged: their normals averaged, their `extra` kept once, the indices remapped.
 */
export function weld<T extends Uint32Array | Float32Array>(vertices: Float32Array, extra: T, width: number, indices: Uint32Array) {
    const bits = new Uint32Array(vertices.buffer, vertices.byteOffset, vertices.length);
    const extraBits = new Uint32Array(extra.buffer, extra.byteOffset, extra.length);
    const count = extra.length / width, merged = new Map<string, number>(), remap = new Uint32Array(count);
    const out: number[] = [], kept: number[] = [], normals: number[] = [];
    for (let v = 0; v < count; v++) {
        const o = v * 8;
        let key = `${bits[o]},${bits[o + 1]},${bits[o + 2]},${bits[o + 6]},${bits[o + 7]}`;
        for (let k = 0; k < width; k++) key += `,${extraBits[v * width + k]}`;
        let w = merged.get(key);
        if (w === undefined) {
            merged.set(key, w = kept.length / width);
            for (let k = 0; k < width; k++) kept.push(extra[v * width + k]);
            for (let j = 0; j < 8; j++) out.push(vertices[o + j]);
            normals.push(0, 0, 0);
        }
        for (let j = 0; j < 3; j++) normals[w * 3 + j] += vertices[o + 3 + j];
        remap[v] = w;
    }
    for (let w = 0; w < normals.length / 3; w++) {
        const n = normals.slice(w * 3, w * 3 + 3), length = Math.hypot(...n);
        // Opposite faces at one place (a card seen from both sides) cancel: keep the first normal.
        if (length > 0) for (let j = 0; j < 3; j++) out[w * 8 + 3 + j] = n[j] / length;
    }
    const Kept = extra.constructor as new (values: number[]) => T;
    return { vertices: new Float32Array(out), extra: new Kept(kept), indices: indices.map(i => remap[i]) };
}

/** Bounds over every frame, each vertex posed as `pose` poses it from the table's own floats. */
export function boundsOf(vertexCount: number, frames: number, pose: (frame: number, vertex: number, out: number[]) => void): BakedBounds {
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity], q = [0, 0, 0];
    let radius2 = 0;
    for (let f = 0; f < frames; f++) {
        for (let v = 0; v < vertexCount; v++) {
            pose(f, v, q);
            for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], q[k]); max[k] = Math.max(max[k], q[k]); }
            radius2 = Math.max(radius2, q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
        }
    }
    return { min, max, radius: Math.sqrt(radius2) };
}

/** The meshes' nodes must be known: the parser records them (loadMerged drops them). */
export function nodeOf(glb: GLBData, k: number, fail: Fail): number {
    const node = glb.meshes![k].node;
    if (node === undefined || !glb.nodes?.[node]) fail(`mesh ${k} has no node: parse the file with GLBParser.parse or load (loadMerged drops the nodes).`);
    return node;
}
