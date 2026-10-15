// Baked skinned animation (docs/agents-skinned-models-design.md §3): a GLB with a skin baked into one
// mesh, four influences and weights per vertex, and one table of influence matrices per frame of each
// clip: the rigid bake's table, with the skin's joints where it has parts. Uploaded once, it serves
// every instance; a palette pass or a vertex hook blends a vertex's four influences.
//
// glTF skins a vertex in its mesh's bind space: Σ weight × W_t(joint) · IBM(joint) · v. The bind space
// can sit at another scale or angle than the model (Mixamo's armature is scaled 0.01 and turned), so
// the bake moves the vertices into the model's space at rest with one matrix M (the rest skinning
// matrix of the joint with the most weight; when the rest pose is the bind pose, every joint's is
// the same) and stores W_t(joint) · IBM(joint) · M⁻¹: exact for any invertible M. At rest the
// table is identity only when the file's rest pose is its bind pose.
import type { GLBData } from './types';
import { MathUtils } from './MathUtils';
import { boundsOf, invertAffine, nodeOf, rowsOf, timeline, weld } from './bakeShared';
import type { BakedBounds, BakedClip, BakeOptions } from './bakeShared';

export type SkinnedBakeOptions = BakeOptions;

export interface SkinnedAnimation {
    /** The model's meshes as one, in a fixed rebased bind space: position, normal, uv (8 floats, StandardLayout). */
    readonly vertices: Float32Array;
    readonly indices: Uint32Array;
    readonly vertexCount: number;
    readonly indexCount: number;
    /** Four influences per vertex, indices into the table; an unused one is 0 with weight 0. */
    readonly influencesOfVertex: Uint32Array;
    /** Their weights, four per vertex, summing to 1. */
    readonly weightsOfVertex: Float32Array;
    readonly influences: number;
    /** The node of each influence: the skin's joints in its order, then each unskinned mesh's node. */
    readonly influenceNodes: readonly number[];
    /**
     * For each frame, for each influence, the transform from the rest pose to the frame's pose: three
     * rows of four floats, as the rigid table. Influence i of frame f starts at float
     * (f × influences + i) × 12; a vertex is posed at Σ weight × (dot(row0, v1), dot(row1, v1),
     * dot(row2, v1)) over its four influences, with v1 = (v, 1).
     */
    readonly transforms: Float32Array;
    /** Frames in the table, every clip's together. */
    readonly frames: number;
    readonly clips: readonly BakedClip[];
    /** Every vertex in every frame, blended as a shader blends it: a box, and the radius about the origin. */
    readonly bounds: BakedBounds;
}

function fail(message: string): never { throw new Error(`bakeSkinnedAnimation: ${message}`); }

/**
 * Bakes a parsed GLB's skinned animation: its meshes as one mesh (welded by default), each vertex's
 * four influences and weights, and for each asked clip, sampled at `fps`, every influence's transform
 * from rest to pose. The influences are the file's first skin's joints, then one per unskinned mesh
 * (a hat on a head bone moves with its node, weight 1). Meshes must come from `GLBParser.parse` or
 * `load`. One material draws the model; additional joint sets are refused; morph targets are not baked.
 */
export function bakeSkinnedAnimation(glb: GLBData, options: SkinnedBakeOptions = {}): SkinnedAnimation {
    const fps = options.fps ?? 30, meshes = glb.meshes ?? [], skin = glb.skin;
    if (!(fps > 0 && fps < Infinity)) fail(`fps must be a number above 0; got ${fps}.`);
    if (!meshes.length) fail('the file has no meshes.');
    if (!skin || !meshes.some(m => m.isSkinned)) fail('the file has no skinned mesh; bakeRigidAnimation bakes meshes moved by their nodes.');
    const nodes = glb.nodes ?? [], joints = skin.joints.length;
    if ((glb.skinCount ?? 1) > 1) fail(`the file has ${glb.skinCount} skins; only one skin is supported.`);
    if (!joints || joints > 0xffffffff) fail(`the skin has ${joints} joints; the influence table requires 1 to 4294967295.`);
    if (skin.joints.some(j => !Number.isInteger(j) || !nodes[j])) fail('the skin names a missing joint node.');
    if (skin.inverseBindMatrices.length !== joints * 16 || !skin.inverseBindMatrices.every(Number.isFinite)) fail('inverse bind matrices must contain 16 finite floats per joint.');

    // The meshes as one: per vertex 8 floats, and 8 words of influences (4 indices, 4 weight bits).
    const influenceNodes = [...skin.joints], rigid: number[] = [];
    const vertices: number[] = [], indices: number[] = [], extra: number[] = [], skinned: boolean[] = [];
    const weightOf = new Float64Array(joints), word = new Float32Array(1), wordBits = new Uint32Array(word.buffer);
    const asBits = (x: number) => { word[0] = x; return wordBits[0]; };
    meshes.forEach((mesh, k) => {
        const node = nodeOf(glb, k, fail), base = skinned.length;
        if (mesh.hasAdditionalInfluences) fail(`mesh ${k} has JOINTS_1 or WEIGHTS_1 (additional influence sets); only four influences per vertex are supported.`);
        if (mesh.isSkinned && (nodes[node].skin ?? 0) !== 0) fail(`mesh ${k} uses skin ${nodes[node].skin}; only the file's first skin is read.`);
        let own = -1;
        if (!mesh.isSkinned) {
            own = rigid.indexOf(node);
            if (own < 0) own = rigid.push(node) - 1;
        }
        const stride = mesh.isSkinned ? 16 : 8, v = mesh.vertices;
        if (v.length !== mesh.vertexCount * stride) fail(`mesh ${k} has an invalid vertex layout; expected ${stride} floats per vertex.`);
        for (let i = 0; i < mesh.vertexCount; i++) {
            const o = i * stride;
            for (let j = 0; j < 8; j++) vertices.push(v[o + j]);
            skinned.push(mesh.isSkinned);
            if (!mesh.isSkinned) { extra.push(joints + own, 0, 0, 0, asBits(1), 0, 0, 0); continue; }
            const w = [v[o + 12], v[o + 13], v[o + 14], v[o + 15]], sum = w[0] + w[1] + w[2] + w[3];
            if (w.some(x => !Number.isFinite(x) || x < 0) || !Number.isFinite(sum)) fail(`mesh ${k} vertex ${i} has invalid weights; weights must be finite and non-negative.`);
            if (!(sum > 0)) fail(`mesh ${k} vertex ${i} has no weight.`);
            const j4 = [v[o + 8], v[o + 9], v[o + 10], v[o + 11]];
            j4.forEach((j, n) => {
                if (!(Number.isInteger(j) && j >= 0 && j < joints)) fail(`mesh ${k} vertex ${i} names joint ${j}; the skin has ${joints}.`);
                weightOf[j] += w[n] / sum;
            });
            extra.push(...j4.map((j, k) => w[k] === 0 ? 0 : j), ...w.map(x => asBits(x / sum)));
        }
        for (let i = 0; i < mesh.indexCount; i++) indices.push(base + mesh.indices[i]);
    });
    influenceNodes.push(...rigid);
    if (influenceNodes.length > 0xffffffff) fail('the influence count exceeds the u32 table index.');

    // Bind space to the model at rest: the skinning matrix at rest of the joint with the most weight.
    const time = timeline(glb, options, fail);
    const ibm = (j: number) => skin.inverseBindMatrices.subarray(j * 16, j * 16 + 16);
    let heaviest = 0;
    for (let j = 1; j < joints; j++) if (weightOf[j] > weightOf[heaviest]) heaviest = j;
    const M = MathUtils.multiply(time.rest(skin.joints[heaviest]), ibm(heaviest));
    const toBind = invertAffine(M) ?? fail(`joint ${heaviest} ("${nodes[skin.joints[heaviest]].name ?? ''}") has a skinning matrix with no inverse at rest.`);
    for (let i = 0; i < skinned.length; i++) {
        if (!skinned[i]) continue;
        const o = i * 8, p = [vertices[o], vertices[o + 1], vertices[o + 2]], n = [vertices[o + 3], vertices[o + 4], vertices[o + 5]];
        for (let r = 0; r < 3; r++) vertices[o + r] = M[r] * p[0] + M[4 + r] * p[1] + M[8 + r] * p[2] + M[12 + r];
        // Normals by the inverse transpose: row r of toBind's columns.
        const q = [0, 1, 2].map(r => toBind[r * 4] * n[0] + toBind[r * 4 + 1] * n[1] + toBind[r * 4 + 2] * n[2]), l = Math.hypot(...q);
        for (let r = 0; r < 3; r++) vertices[o + 3 + r] = l > 0 ? q[r] / l : n[r];
    }
    let mesh = { vertices: new Float32Array(vertices), extra: new Uint32Array(extra), indices: new Uint32Array(indices) };
    if (options.weld ?? true) mesh = weld(mesh.vertices, mesh.extra, 8, mesh.indices);
    const count = mesh.extra.length / 8;
    const influencesOfVertex = new Uint32Array(count * 4), weightsOfVertex = new Float32Array(count * 4);
    const weightView = new Float32Array(mesh.extra.buffer, mesh.extra.byteOffset, mesh.extra.length);
    for (let v = 0; v < count; v++) {
        for (let k = 0; k < 4; k++) {
            influencesOfVertex[v * 4 + k] = mesh.extra[v * 8 + k];
            weightsOfVertex[v * 4 + k] = weightView[v * 8 + 4 + k];
        }
    }

    // Each frame: a joint's W_t · IBM · M⁻¹; an unskinned mesh's node, its world over its rest.
    const restInverse = rigid.map(node => invertAffine(time.rest(node)) ??
        fail(`node ${node} ("${nodes[node].name ?? ''}") has a rest transform with no inverse (a scale of 0?).`));
    const influences = influenceNodes.length, frames = time.frames;
    const transforms = new Float32Array(frames * influences * 12);
    time.each((frame, world) => {
        skin.joints.forEach((node, j) => transforms.set(rowsOf(MathUtils.multiply(MathUtils.multiply(world(node), ibm(j)), toBind)), (frame * influences + j) * 12));
        rigid.forEach((node, r) => transforms.set(rowsOf(MathUtils.multiply(world(node), restInverse[r])), (frame * influences + joints + r) * 12));
    });

    // Bounds over every frame, each vertex blended from the table's own floats.
    const v = mesh.vertices;
    const bounds = boundsOf(count, frames, (f, i, q) => {
        const x = v[i * 8], y = v[i * 8 + 1], z = v[i * 8 + 2];
        q[0] = q[1] = q[2] = 0;
        for (let k = 0; k < 4; k++) {
            const w = weightsOfVertex[i * 4 + k];
            if (w === 0) continue;
            const o = (f * influences + influencesOfVertex[i * 4 + k]) * 12;
            for (let r = 0; r < 3; r++) q[r] += w * (transforms[o + 4 * r] * x + transforms[o + 4 * r + 1] * y + transforms[o + 4 * r + 2] * z + transforms[o + 4 * r + 3]);
        }
    });

    return {
        vertices: mesh.vertices, indices: mesh.indices, vertexCount: count, indexCount: mesh.indices.length,
        influencesOfVertex, weightsOfVertex, influences, influenceNodes,
        transforms, frames,
        clips: time.clips.map(({ name, firstFrame, frames, seconds, loop }) => ({ name, firstFrame, frames, seconds, loop })),
        bounds,
    };
}
