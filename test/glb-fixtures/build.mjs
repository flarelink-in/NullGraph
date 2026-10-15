// Cycle D, step 1 (docs/agents-animated-models-design.md §9): GLB files written in memory for the
// loaders' tests. 3D model files are never committed here, so each test describes the model it needs
// and this writes it as a .glb. Plain JS, no node: imports, so a GPU suite can use it in a browser too.
//
// A description is glTF's own JSON with the binary data inline:
//   nodes       glTF nodes as they are (name, children, mesh, matrix, translation, rotation, scale, skin)
//   meshes      [{ primitives: [{ positions, normals?, uvs?, indices, joints?, weights? }] }], flat arrays
//   animations  [{ name?, samplers: [{ times, values, interpolation? }], channels: [{ sampler, node, path }] }]
//   skins       [{ joints, inverseBindMatrices }] (16 floats per joint)
//   scene       the root nodes of the scene (default [0])

const COMPONENTS = { f32: [5126, Float32Array], u8: [5121, Uint8Array], u16: [5123, Uint16Array], u32: [5125, Uint32Array] };
const TYPES = { 1: 'SCALAR', 2: 'VEC2', 3: 'VEC3', 4: 'VEC4', 16: 'MAT4' };

/** The description as a .glb, an ArrayBuffer. */
export function glb({ nodes, meshes = [], animations = [], skins, scene = [0], extra = {} }) {
    const parts = [], bufferViews = [], accessors = [];
    let length = 0;
    const accessor = (values, size, kind = 'f32', minMax = false, normalized = false) => {
        const [componentType, Type] = COMPONENTS[kind];
        const bytes = new Uint8Array(new Type(values).buffer);
        length = Math.ceil(length / 4) * 4;
        parts.push([length, bytes]);
        bufferViews.push({ buffer: 0, byteOffset: length, byteLength: bytes.length });
        length += bytes.length;
        const a = { bufferView: bufferViews.length - 1, componentType, count: values.length / size, type: TYPES[size] };
        if (normalized) a.normalized = true;
        if (minMax) {
            a.min = Array.from({ length: size }, (_, k) => Math.min(...values.filter((_, i) => i % size === k)));
            a.max = Array.from({ length: size }, (_, k) => Math.max(...values.filter((_, i) => i % size === k)));
        }
        accessors.push(a);
        return accessors.length - 1;
    };
    const json = {
        asset: { version: '2.0', generator: 'null-graph test/glb-fixtures' },
        scene: 0,
        scenes: [{ nodes: scene }],
        nodes,
        meshes: meshes.map(mesh => ({
            primitives: mesh.primitives.map(p => ({
                attributes: {
                    POSITION: accessor(p.positions, 3, 'f32', true),
                    ...(p.normals ? { NORMAL: accessor(p.normals, 3) } : {}),
                    ...(p.uvs ? { TEXCOORD_0: accessor(p.uvs, 2) } : {}),
                    ...(p.joints ? { JOINTS_0: accessor(p.joints, 4, p.jointType ?? 'u8') } : {}),
                    ...(p.weights ? { WEIGHTS_0: accessor(p.weights, 4, p.weightType ?? 'f32', false, p.weightsNormalized ?? false) } : {}),
                    ...(p.joints1 ? { JOINTS_1: accessor(p.joints1, 4, 'u8') } : {}),
                    ...(p.weights1 ? { WEIGHTS_1: accessor(p.weights1, 4) } : {}),
                },
                indices: accessor(p.indices, 1, p.indexType ?? 'u16'),
            })),
        })),
        ...(animations.length ? {
            animations: animations.map(a => ({
                ...(a.name === undefined ? {} : { name: a.name }),
                samplers: a.samplers.map(s => ({
                    input: accessor(s.times, 1, 'f32', true),
                    output: accessor(s.values, s.values.length / s.times.length / (s.interpolation === 'CUBICSPLINE' ? 3 : 1)),
                    ...(s.interpolation ? { interpolation: s.interpolation } : {}),
                })),
                channels: a.channels.map(c => ({ sampler: c.sampler, target: { node: c.node, path: c.path } })),
            })),
        } : {}),
        ...(skins ? { skins: skins.map(s => ({ joints: s.joints, ...(s.inverseBindMatrices ? { inverseBindMatrices: accessor(s.inverseBindMatrices, 16) } : {}) })) } : {}),
        ...extra,
    };
    json.buffers = [{ byteLength: length }];
    json.bufferViews = bufferViews;
    json.accessors = accessors;

    const text = new TextEncoder().encode(JSON.stringify(json));
    const jsonLength = Math.ceil(text.length / 4) * 4, binLength = Math.ceil(length / 4) * 4;
    const out = new Uint8Array(12 + 8 + jsonLength + 8 + binLength);
    const view = new DataView(out.buffer);
    view.setUint32(0, 0x46546c67, true);                     // "glTF"
    view.setUint32(4, 2, true);
    view.setUint32(8, out.length, true);
    view.setUint32(12, jsonLength, true);
    view.setUint32(16, 0x4e4f534a, true);                    // "JSON"
    out.fill(0x20, 20, 20 + jsonLength);                     // JSON is padded with spaces
    out.set(text, 20);
    view.setUint32(20 + jsonLength, binLength, true);
    view.setUint32(24 + jsonLength, 0x004e4942, true);       // "BIN\0"
    for (const [offset, bytes] of parts) out.set(bytes, 28 + jsonLength + offset);
    return out.buffer;
}

/**
 * A box from `min` to `max`, its 24 vertices split per face, as low-poly exporters write them: each
 * face has its own normal. Its uv is a function of its corner, so a corner's three copies share it
 * (and weld), unless `seam` gives each face its own uvs.
 */
export function box(min = [-0.5, -0.5, -0.5], max = [0.5, 0.5, 0.5], { seam = false } = {}) {
    const positions = [], normals = [], uvs = [], indices = [];
    const faces = [[0, 1], [0, -1], [1, 1], [1, -1], [2, 1], [2, -1]];
    faces.forEach(([axis, sign], face) => {
        const u = (axis + 1) % 3, v = (axis + 2) % 3, base = positions.length / 3;
        for (const [a, b] of [[0, 0], [1, 0], [1, 1], [0, 1]]) {
            const p = [0, 0, 0];
            p[axis] = sign > 0 ? max[axis] : min[axis];
            p[u] = a ? max[u] : min[u];
            p[v] = b ? max[v] : min[v];
            positions.push(...p);
            normals.push(...[0, 1, 2].map(k => (k === axis ? sign : 0)));
            const corner = (p[0] === max[0] ? 1 : 0) + (p[1] === max[1] ? 2 : 0) + (p[2] === max[2] ? 4 : 0);
            uvs.push(seam ? face / 6 + a / 12 : corner / 8, seam ? b : 0.5);
        }
        indices.push(...(sign > 0 ? [base, base + 1, base + 2, base, base + 2, base + 3] : [base, base + 2, base + 1, base, base + 3, base + 2]));
    });
    return { positions, normals, uvs, indices };
}

/** A unit quaternion for a rotation of `degrees` about `axis`. */
export function quat(axis, degrees) {
    const length = Math.hypot(...axis), half = degrees * Math.PI / 360, s = Math.sin(half) / length;
    return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(half)];
}
