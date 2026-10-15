// Cycle D, step 1, phase 0 (docs/agents-animated-models-design.md §3): bake a GLB's rigid-part
// animation into a table of part transforms per frame. A spike: the library's bakeRigidAnimation
// (phase 1) replaces it. Pure JavaScript, no GPU; runs in Node and in a browser.
//
// The models of the test site (fox, deer, cow) are rigid parts — body, head or tail, four legs —
// moved by node transforms, with no skin. Their vertices are baked at the rest pose into model
// space; a part's transform at time t is W_t · W_rest⁻¹, its world transform then over its world
// transform at rest. A rigid part is a bone with weight 1, so the same table can later hold the
// bone matrices of a skinned model.

import { mat3, mat4, quat, vec3 } from 'gl-matrix';

const COMPONENTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
const ARRAYS = { 5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };

/** The JSON and binary chunks of a .glb. */
export function readGlb(bytes) {
    const buffer = bytes instanceof ArrayBuffer ? bytes : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const view = new DataView(buffer);
    if (view.getUint32(0, true) !== 0x46546c67) throw new Error('not a .glb');
    const jsonLength = view.getUint32(12, true);
    const gltf = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 20, jsonLength)));
    const binStart = 20 + jsonLength + 8;
    return { gltf, buffer, binStart };
}

/** An accessor's values, one plain array per element. */
function accessor(glb, index) {
    const { gltf, buffer, binStart } = glb;
    const a = gltf.accessors[index];
    const bv = gltf.bufferViews[a.bufferView];
    const Type = ARRAYS[a.componentType];
    const n = COMPONENTS[a.type];
    const stride = bv.byteStride || n * Type.BYTES_PER_ELEMENT;
    const base = binStart + (bv.byteOffset ?? 0) + (a.byteOffset ?? 0);
    const out = [];
    for (let i = 0; i < a.count; i++) {
        out.push(Array.from(new Type(buffer, base + i * stride, n)));
    }
    return out;
}

function restTRS(node) {
    if (node.matrix) {
        const m = mat4.clone(node.matrix);
        const t = vec3.create(), r = quat.create(), s = vec3.create();
        mat4.getTranslation(t, m); mat4.getRotation(r, m); mat4.getScaling(s, m);
        return { t: [...t], r: [...r], s: [...s] };
    }
    return { t: node.translation ?? [0, 0, 0], r: node.rotation ?? [0, 0, 0, 1], s: node.scale ?? [1, 1, 1] };
}

const local = ({ t, r, s }) => mat4.fromRotationTranslationScale(mat4.create(), r, t, s);

/** Every node's parent, or -1. */
function parents(gltf) {
    const parent = gltf.nodes.map(() => -1);
    gltf.nodes.forEach((n, i) => (n.children ?? []).forEach(c => { parent[c] = i; }));
    return parent;
}

/** World matrix of a node from per-node TRS (rest, or animated). */
function world(node, trs, parent) {
    const chain = [];
    for (let n = node; n >= 0; n = parent[n]) chain.unshift(n);
    const m = mat4.create();
    for (const n of chain) mat4.multiply(m, m, local(trs[n]));
    return m;
}

/** A sampler's value at time t: LINEAR (slerp for rotations) or STEP; the clip loops. */
function sample(times, values, path, interpolation, t) {
    if (t <= times[0]) return values[0];
    const last = times.length - 1;
    if (t >= times[last]) return values[last];
    let k = 0;
    while (times[k + 1] < t) k++;
    if (interpolation === 'STEP') return values[k];
    if (interpolation !== 'LINEAR') throw new Error(`${interpolation} interpolation is not read by the spike`);
    const f = (t - times[k]) / (times[k + 1] - times[k]);
    if (path === 'rotation') return [...quat.slerp(quat.create(), values[k], values[k + 1], f)];
    return values[k].map((v, i) => v + (values[k + 1][i] - v) * f);
}

/**
 * The same mesh with vertices merged where position, uv and part agree. The models split every
 * vertex per face for flat normals; merged, a face's normal is found per pixel instead (from the
 * derivatives of its world position), and the vertex stage runs 1.7–2.8× fewer times.
 */
export function weld(baked) {
    const { vertices, indices, partOfVertex } = baked;
    const seen = new Map(), outV = [], outP = [], remap = new Uint32Array(partOfVertex.length);
    for (let i = 0; i < partOfVertex.length; i++) {
        const v = vertices.subarray(i * 8, i * 8 + 8);
        const key = `${v[0].toFixed(4)},${v[1].toFixed(4)},${v[2].toFixed(4)},${v[6].toFixed(4)},${v[7].toFixed(4)},${partOfVertex[i]}`;
        let j = seen.get(key);
        if (j === undefined) {
            j = outP.length;
            seen.set(key, j);
            outV.push(...v);
            outP.push(partOfVertex[i]);
        }
        remap[i] = j;
    }
    return { ...baked, vertices: new Float32Array(outV), partOfVertex: new Uint32Array(outP), indices: indices.map(i => remap[i]) };
}

/**
 * The rest-pose mesh, one part per mesh node, and every asked clip sampled at `fps`.
 * @returns {{ vertices: Float32Array, indices: Uint32Array, partOfVertex: Uint32Array, parts: number,
 *   transforms: Float32Array, clips: { name: string, firstFrame: number, frames: number, seconds: number }[],
 *   bounds: { min: number[], max: number[], center: number[], radius: number } }}
 */
export function bakeRigid(glb, { fps = 30, clips = ['idle', 'walk', 'run', 'eat'] } = {}) {
    const { gltf } = glb;
    const parent = parents(gltf);
    const rest = gltf.nodes.map(restTRS);
    const meshNodes = gltf.nodes.map((n, i) => (n.mesh !== undefined ? i : -1)).filter(i => i >= 0);
    const restWorld = meshNodes.map(n => world(n, rest, parent));

    // Geometry: position, normal, uv (StandardLayout, 8 floats), baked into model space at rest.
    const vertices = [], indices = [], partOfVertex = [];
    meshNodes.forEach((node, part) => {
        const m = restWorld[part];
        const normalMatrix = mat3.normalFromMat4(mat3.create(), m);
        for (const prim of gltf.meshes[gltf.nodes[node].mesh].primitives) {
            const base = partOfVertex.length;
            const pos = accessor(glb, prim.attributes.POSITION);
            const nor = prim.attributes.NORMAL !== undefined ? accessor(glb, prim.attributes.NORMAL) : pos.map(() => [0, 1, 0]);
            const uv = prim.attributes.TEXCOORD_0 !== undefined ? accessor(glb, prim.attributes.TEXCOORD_0) : pos.map(() => [0, 0]);
            pos.forEach((p, i) => {
                const wp = vec3.transformMat4(vec3.create(), p, m);
                const wn = vec3.normalize(vec3.create(), vec3.transformMat3(vec3.create(), nor[i], normalMatrix));
                vertices.push(wp[0], wp[1], wp[2], wn[0], wn[1], wn[2], uv[i][0], uv[i][1]);
                partOfVertex.push(part);
            });
            const idx = prim.indices !== undefined ? accessor(glb, prim.indices).map(v => v[0]) : pos.map((_, i) => i);
            for (const i of idx) indices.push(base + i);
        }
    });

    // Clips: frames of part transforms, 3 rows of 4 floats each (row r = [m0r, m1r, m2r, m3r]).
    const restInverse = restWorld.map(m => mat4.invert(mat4.create(), m));
    const table = [], clipTable = [];
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    const posed = [];
    for (const name of clips) {
        const anim = gltf.animations.find(a => a.name === name);
        if (!anim) throw new Error(`no clip "${name}"`);
        const samplers = anim.samplers.map(s => ({
            times: accessor(glb, s.input).map(v => v[0]),
            values: accessor(glb, s.output),
            interpolation: s.interpolation ?? 'LINEAR',
        }));
        const seconds = Math.max(...samplers.map(s => s.times[s.times.length - 1]));
        const frames = Math.max(2, Math.round(seconds * fps));
        clipTable.push({ name, firstFrame: table.length / (meshNodes.length * 12), frames, seconds });
        for (let f = 0; f < frames; f++) {
            const t = (f / frames) * seconds;                       // loops: the last frame is not the first
            const trs = rest.map(r => ({ ...r }));
            for (const c of anim.channels) {
                const s = samplers[c.sampler];
                const key = { translation: 't', rotation: 'r', scale: 's' }[c.target.path];
                if (key) trs[c.target.node] = { ...trs[c.target.node], [key]: sample(s.times, s.values, c.target.path, s.interpolation, t) };
            }
            meshNodes.forEach((node, part) => {
                const m = mat4.multiply(mat4.create(), world(node, trs, parent), restInverse[part]);
                table.push(m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14]);
                posed.push(m);
            });
        }
    }
    // Bounds over every frame: each vertex through each pose of its part.
    const framesTotal = table.length / (meshNodes.length * 12);
    for (let v = 0; v < partOfVertex.length; v++) {
        const p = vertices.slice(v * 8, v * 8 + 3);
        for (let f = 0; f < framesTotal; f++) {
            const q = vec3.transformMat4(vec3.create(), p, posed[f * meshNodes.length + partOfVertex[v]]);
            for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], q[k]); hi[k] = Math.max(hi[k], q[k]); }
        }
    }
    const center = lo.map((v, k) => (v + hi[k]) / 2);
    const radius = Math.hypot(...hi.map((v, k) => v - center[k]));
    return {
        vertices: new Float32Array(vertices), indices: new Uint32Array(indices),
        partOfVertex: new Uint32Array(partOfVertex), parts: meshNodes.length,
        transforms: new Float32Array(table), clips: clipTable,
        bounds: { min: lo, max: hi, center, radius },
    };
}
