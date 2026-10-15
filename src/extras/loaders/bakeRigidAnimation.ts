// Baked rigid animation (docs/agents-animated-models-design.md §3): a GLB whose meshes are moved by
// their nodes, with no skin (the low-poly animals of most asset packs), baked into one mesh, the part
// that moves each vertex, and one table of part transforms per frame of each clip. Uploaded once, the
// table serves every instance of the model: a vertex shader reads its vertex's part and its instance's
// clip and phase, and poses the vertex with no CPU work per instance per frame.
//
// A rigid part is a bone with weight 1: bakeSkinnedAnimation keeps the same table for skins.
import type { GLBData } from './types';
import { MathUtils } from './MathUtils';
import { boundsOf, invertAffine, nodeOf, rowsOf, timeline, weld } from './bakeShared';
import type { BakeClipRequest, BakedBounds, BakedClip, BakeOptions } from './bakeShared';

export type RigidClipRequest = BakeClipRequest;
export type RigidBakeOptions = BakeOptions;
export type RigidClip = BakedClip;

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
    readonly bounds: BakedBounds;
}

function fail(message: string): never { throw new Error(`bakeRigidAnimation: ${message}`); }

/**
 * Bakes a parsed GLB's rigid animation: its meshes as one mesh (welded by default), the part of each
 * vertex, and for each asked clip, sampled at `fps`, every part's transform from rest to pose. Every
 * mesh must come from `GLBParser.parse` or `load` (which record each mesh's node) and be unskinned.
 * The whole model draws with one material, and morph-target (`weights`) channels are not baked.
 */
export function bakeRigidAnimation(glb: GLBData, options: RigidBakeOptions = {}): RigidAnimation {
    const fps = options.fps ?? 30, meshes = glb.meshes ?? [];
    if (!(fps > 0 && fps < Infinity)) fail(`fps must be a number above 0; got ${fps}.`);
    if (!meshes.length) fail('the file has no meshes.');

    // The meshes as one, and a part for each node that holds a mesh, in the parser's order.
    const partNodes: number[] = [], partOf: number[] = [], vertices: number[] = [], indices: number[] = [];
    meshes.forEach((mesh, k) => {
        if (mesh.isSkinned) fail(`mesh ${k} is skinned; this bakes meshes moved by their nodes (rigid parts) only.`);
        const node = nodeOf(glb, k, fail);
        let part = partNodes.indexOf(node);
        if (part < 0) part = partNodes.push(node) - 1;
        const base = partOf.length;
        for (let v = 0; v < mesh.vertexCount; v++) partOf.push(part);
        for (let i = 0; i < mesh.indexCount; i++) indices.push(base + mesh.indices[i]);
        for (const x of mesh.vertices) vertices.push(x);
    });
    let mesh = { vertices: new Float32Array(vertices), extra: new Uint32Array(partOf), indices: new Uint32Array(indices) };
    if (options.weld ?? true) mesh = weld(mesh.vertices, mesh.extra, 1, mesh.indices);

    // Each frame's transforms: every part's world transform at that time over its world transform at rest.
    const time = timeline(glb, options, fail), nodes = glb.nodes!;
    const restInverse = partNodes.map(node => invertAffine(time.rest(node)) ??
        fail(`node ${node} ("${nodes[node].name ?? ''}") has a rest transform with no inverse (a scale of 0?).`));
    const parts = partNodes.length, frames = time.frames;
    const transforms = new Float32Array(frames * parts * 12);
    time.each((frame, world) => partNodes.forEach((node, p) =>
        transforms.set(rowsOf(MathUtils.multiply(world(node), restInverse[p])), (frame * parts + p) * 12)));

    // Bounds over every frame, posed as a vertex shader poses them, from the table's own floats.
    const v = mesh.vertices, partOfVertex = mesh.extra;
    const bounds = boundsOf(partOfVertex.length, frames, (f, i, q) => {
        const o = (f * parts + partOfVertex[i]) * 12, x = v[i * 8], y = v[i * 8 + 1], z = v[i * 8 + 2];
        for (let k = 0; k < 3; k++) q[k] = transforms[o + 4 * k] * x + transforms[o + 4 * k + 1] * y + transforms[o + 4 * k + 2] * z + transforms[o + 4 * k + 3];
    });

    return {
        vertices: mesh.vertices, indices: mesh.indices,
        vertexCount: partOfVertex.length, indexCount: mesh.indices.length,
        partOfVertex, parts, partNodes,
        transforms, frames,
        clips: time.clips.map(({ name, firstFrame, frames, seconds, loop }) => ({ name, firstFrame, frames, seconds, loop })),
        bounds,
    };
}
