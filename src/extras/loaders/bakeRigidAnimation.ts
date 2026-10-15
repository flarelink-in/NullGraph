// Baked rigid animation (docs/agents-animated-models-design.md §3): a GLB whose meshes are moved by
// their nodes, with no skin (the low-poly animals of most asset packs), baked into one mesh, the part
// that moves each vertex, and one table of part transforms per frame of each clip. Uploaded once, the
// table serves every instance of the model: a vertex shader reads its vertex's part and its instance's
// clip and phase, and poses the vertex with no CPU work per instance per frame.
//
// A rigid part is a bone with weight 1, so the same table can hold a skinned model's bone matrices.
import type { GLBAnimation, GLBData, GLBNode } from './types';
import { MathUtils } from './MathUtils';

/** A clip to bake, by its name in the file. */
export interface RigidClipRequest {
    readonly name: string;
    /** The clip repeats (idle, walk, run), or plays once and holds its end (open, a fall). Default true. */
    readonly loop?: boolean;
}

export interface RigidBakeOptions {
    /** Frames per second of each clip in the table. Default 30. */
    readonly fps?: number;
    /** The clips to bake, in this order. Default: every clip of the file, in its order, looping. */
    readonly clips?: readonly (string | RigidClipRequest)[];
    /**
     * Merge the vertices that share position, uv and part. Default true. Low-poly models repeat a
     * vertex for every face around it, so that each face has its own normal; merged, the vertex stage
     * runs about half as often, and a face's normal is found per pixel instead (from the derivatives of
     * its position). A merged vertex's normal is the mean of the normals it replaces, a smooth normal.
     * `false` keeps the parser's vertices as they are.
     */
    readonly weld?: boolean;
}

/** One clip in the table. */
export interface RigidClip {
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

export interface RigidAnimation {
    /** The model's meshes as one, at rest: position, normal, uv per vertex (8 floats, StandardLayout). */
    readonly vertices: Float32Array;
    readonly indices: Uint32Array;
    readonly vertexCount: number;
    readonly indexCount: number;
    /** The part that moves each vertex. */
    readonly partOfVertex: Uint32Array;
    readonly parts: number;
    /** The node of each part, an index into `glb.nodes`. */
    readonly partNodes: readonly number[];
    /**
     * For each frame, for each part, the transform from the rest pose to the frame's pose (the part's
     * world transform at that time over its world transform at rest): three rows of four floats.
     * Part p of frame f starts at float (f × parts + p) × 12, and a vertex at position v is posed at
     * (dot(row0, v1), dot(row1, v1), dot(row2, v1)), with v1 = (v, 1).
     */
    readonly transforms: Float32Array;
    /** Frames in the table, every clip's together. */
    readonly frames: number;
    readonly clips: readonly RigidClip[];
    /**
     * Every vertex in every frame, and so in any blend of frames: a box, and the largest distance from
     * the model's origin (the radius of an instance row's bounds `[0, 0, 0, radius]`).
     */
    readonly bounds: { readonly min: readonly number[]; readonly max: readonly number[]; readonly radius: number };
}

function fail(message: string): never { throw new Error(`bakeRigidAnimation: ${message}`); }

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
function sample(s: Sampler, n: number, rotation: boolean, t: number, clip: string): number[] {
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

/** A node's world transform, multiplied from the top of its chain down, as GLBParser bakes the meshes. */
function world(node: number, localOf: (node: number) => Float32Array, parent: Int32Array): Float32Array {
    const chain: number[] = [];
    for (let n = node; n >= 0; n = parent[n]) chain.push(n);
    let m = localOf(chain.pop()!);
    while (chain.length) m = MathUtils.multiply(m, localOf(chain.pop()!));
    return m;
}

/** The inverse of an affine transform (column-major), or null when it has none. */
function invertAffine(m: Float32Array): Float32Array | null {
    const [a, d, g, , b, e, h, , c, f, i] = m;
    const c11 = e * i - f * h, c12 = f * g - d * i, c13 = d * h - e * g;
    const det = a * c11 + b * c12 + c * c13;
    if (!(Math.abs(det) > 0)) return null;
    const r = [c11, c12, c13, 0, c * h - b * i, a * i - c * g, b * g - a * h, 0, b * f - c * e, c * d - a * f, a * e - b * d, 0].map(x => x / det);
    for (let k = 0; k < 3; k++) r[12 + k] = -(r[k] * m[12] + r[4 + k] * m[13] + r[8 + k] * m[14]);
    r[15] = 1;
    return new Float32Array(r);
}

/** The vertices that share position, uv and part, merged: their normals averaged, the indices remapped. */
function weld(vertices: Float32Array, partOf: number[], indices: Uint32Array) {
    const bits = new Uint32Array(vertices.buffer, vertices.byteOffset, vertices.length);
    const merged = new Map<string, number>(), remap = new Uint32Array(partOf.length);
    const out: number[] = [], parts: number[] = [], normals: number[] = [];
    for (let v = 0; v < partOf.length; v++) {
        const o = v * 8;
        const key = `${bits[o]},${bits[o + 1]},${bits[o + 2]},${bits[o + 6]},${bits[o + 7]},${partOf[v]}`;
        let w = merged.get(key);
        if (w === undefined) {
            merged.set(key, w = parts.length);
            parts.push(partOf[v]);
            for (let j = 0; j < 8; j++) out.push(vertices[o + j]);
            normals.push(0, 0, 0);
        }
        for (let j = 0; j < 3; j++) normals[w * 3 + j] += vertices[o + 3 + j];
        remap[v] = w;
    }
    for (let w = 0; w < parts.length; w++) {
        const n = normals.slice(w * 3, w * 3 + 3), length = Math.hypot(...n);
        // Opposite faces at one place (a card seen from both sides) cancel: keep the first normal.
        if (length > 0) for (let j = 0; j < 3; j++) out[w * 8 + 3 + j] = n[j] / length;
    }
    return { vertices: new Float32Array(out), partOf: parts, indices: indices.map(i => remap[i]) };
}

/**
 * Bakes a parsed GLB's rigid animation: its meshes as one mesh (welded by default), the part of each
 * vertex, and for each asked clip, sampled at `fps`, every part's transform from rest to pose. Every
 * mesh must come from `GLBParser.parse` or `load` (which record each mesh's node) and be unskinned.
 * The whole model draws with one material, and morph-target (`weights`) channels are not baked.
 */
export function bakeRigidAnimation(glb: GLBData, options: RigidBakeOptions = {}): RigidAnimation {
    const fps = options.fps ?? 30, meshes = glb.meshes ?? [], nodes = glb.nodes ?? [];
    if (!(fps > 0 && fps < Infinity)) fail(`fps must be a number above 0; got ${fps}.`);
    if (!meshes.length) fail('the file has no meshes.');

    // The meshes as one, and a part for each node that holds a mesh, in the parser's order.
    const partNodes: number[] = [], partOf: number[] = [], vertices: number[] = [], indices: number[] = [];
    meshes.forEach((mesh, k) => {
        if (mesh.isSkinned) fail(`mesh ${k} is skinned; this bakes meshes moved by their nodes (rigid parts) only.`);
        if (mesh.node === undefined || !nodes[mesh.node]) {
            fail(`mesh ${k} has no node: parse the file with GLBParser.parse or load (loadMerged drops the nodes).`);
        }
        let part = partNodes.indexOf(mesh.node);
        if (part < 0) part = partNodes.push(mesh.node) - 1;
        const base = partOf.length;
        for (let v = 0; v < mesh.vertexCount; v++) partOf.push(part);
        for (let i = 0; i < mesh.indexCount; i++) indices.push(base + mesh.indices[i]);
        for (const x of mesh.vertices) vertices.push(x);
    });
    let mesh = { vertices: new Float32Array(vertices), partOf, indices: new Uint32Array(indices) };
    if (options.weld ?? true) mesh = weld(mesh.vertices, partOf, mesh.indices);

    // The clips asked for, and their frames.
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

    // Each frame's transforms: every part's world transform at that time over its world transform at rest.
    const parent = new Int32Array(nodes.length).fill(-1);
    nodes.forEach((node, n) => node.children?.forEach(child => { parent[child] = n; }));
    const restInverse = partNodes.map(node => invertAffine(world(node, n => local(nodes[n]), parent)) ??
        fail(`node ${node} ("${nodes[node].name ?? ''}") has a rest transform with no inverse (a scale of 0?).`));
    const parts = partNodes.length;
    const transforms = new Float32Array(frames * parts * 12);
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
                trs[key] = sample(clip.animation.samplers[samplerIndex], key === 'r' ? 4 : 3, key === 'r', time, clip.name);
                pose.set(targetNode, trs);
            }
            partNodes.forEach((node, p) => {
                const m = MathUtils.multiply(world(node, n => local(nodes[n], pose.get(n)), parent), restInverse[p]);
                transforms.set([m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14]], ((clip.firstFrame + f) * parts + p) * 12);
            });
        }
    }

    // Bounds over every frame, posed as a vertex shader poses them, from the table's own floats.
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity], v = mesh.vertices;
    let radius2 = 0;
    for (let f = 0; f < frames; f++) {
        for (let i = 0; i < mesh.partOf.length; i++) {
            const o = (f * parts + mesh.partOf[i]) * 12, x = v[i * 8], y = v[i * 8 + 1], z = v[i * 8 + 2];
            let length2 = 0;
            for (let k = 0; k < 3; k++) {
                const q = transforms[o + 4 * k] * x + transforms[o + 4 * k + 1] * y + transforms[o + 4 * k + 2] * z + transforms[o + 4 * k + 3];
                min[k] = Math.min(min[k], q);
                max[k] = Math.max(max[k], q);
                length2 += q * q;
            }
            radius2 = Math.max(radius2, length2);
        }
    }

    return {
        vertices: mesh.vertices, indices: mesh.indices,
        vertexCount: mesh.partOf.length, indexCount: mesh.indices.length,
        partOfVertex: new Uint32Array(mesh.partOf), parts, partNodes,
        transforms, frames,
        clips: clips.map(({ name, firstFrame, frames, seconds, loop }) => ({ name, firstFrame, frames, seconds, loop })),
        bounds: { min, max, radius: Math.sqrt(radius2) },
    };
}
