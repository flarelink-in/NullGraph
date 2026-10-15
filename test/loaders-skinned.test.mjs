// D1.5.1: in-memory fixtures are mandatory; external models remain outside this repository.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { Animator, GLBParser, bakeRigidAnimation, bakeSkinnedAnimation } from '../dist/loaders.js';
import { mat4, quat as quaternion, vec3 } from 'gl-matrix';
import { glb, box, quat } from './glb-fixtures/build.mjs';

const identity = [...mat4.create()];
const near = (a, b, tolerance = 2e-5) => a.forEach((x, k) => assert.ok(Math.abs(x - b[k]) < tolerance, `${k}: ${x} != ${b[k]}`));
function fixture({ primitive = {}, skins, interpolation = 'LINEAR' } = {}) {
    return glb({
        nodes: [{ translation: [3, 1, -2], scale: [2, 3, 2], children: [3, 1] },
            { children: [2] }, { translation: [0, 1, 0], children: [4] },
            { mesh: 0, skin: 0, translation: [999, 0, 0] }, { mesh: 1, translation: [0, 0.5, 0] }],
        meshes: [{ primitives: [{ positions: [0, 0, 0, 1, 0, 0, 0, 1, 0], normals: [0, 1, 0, 0, 1, 0, 0, 1, 0], indices: [0, 1, 2],
            joints: [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], weights: [1, 1, 0, 0, 1, 3, 0, 0, 0, 2, 0, 0], ...primitive }] },
            { primitives: [box()] }],
        skins: skins ?? [{ joints: [1, 2], inverseBindMatrices: [...identity, ...identity] }],
        animations: [{ name: 'bend', samplers: [{ times: [0, 1], values: [0, 0, 0, 0, 2, 0], interpolation },
            { times: [0, 1], values: [...quat([0, 0, 1], 0), ...quat([0, 0, 1], 90)] }],
        channels: [{ sampler: 0, node: 1, path: 'translation' }, { sampler: 1, node: 2, path: 'rotation' }] }],
    });
}
function posed(bake, frame, vertex) {
    const p = [...bake.vertices.subarray(vertex * 8, vertex * 8 + 3), 1], out = [0, 0, 0];
    for (let k = 0; k < 4; k++) {
        const w = bake.weightsOfVertex[vertex * 4 + k], j = bake.influencesOfVertex[vertex * 4 + k];
        const o = (frame * bake.influences + j) * 12;
        for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) out[r] += w * bake.transforms[o + r * 4 + c] * p[c];
    }
    return out;
}

// Independent evaluator: gl-matrix TRS and glTF interpolation, not the bake's MathUtils/timeline.
function matrices(data, animation, time) {
    const trs = data.nodes.map(n => ({ t: n.translation ?? [0, 0, 0], r: n.rotation ?? [0, 0, 0, 1], s: n.scale ?? [1, 1, 1] }));
    for (const channel of animation.channels) {
        const s = animation.samplers[channel.samplerIndex], n = channel.path === 'rotation' ? 4 : 3;
        assert.ok(['LINEAR', 'STEP'].includes(s.interpolation));
        const last = s.input.length - 1;
        let a = 0;
        while (a < last && s.input[a + 1] <= time) a++;
        const b = Math.min(a + 1, last), u = b === a || s.interpolation === 'STEP' ? 0 : Math.max(0, (time - s.input[a]) / (s.input[b] - s.input[a]));
        const x = s.output.subarray(a * n, a * n + n), y = s.output.subarray(b * n, b * n + n);
        trs[channel.targetNode][{ translation: 't', rotation: 'r', scale: 's' }[channel.path]] =
            n === 4 ? [...quaternion.slerp(quaternion.create(), x, y, u)] : [...vec3.lerp(vec3.create(), x, y, u)];
    }
    const parent = data.nodes.map(() => -1), cache = [];
    data.nodes.forEach((n, i) => n.children?.forEach(c => { parent[c] = i; }));
    const world = i => {
        if (cache[i]) return cache[i];
        const p = trs[i], local = data.nodes[i].matrix ?? mat4.fromRotationTranslationScale(mat4.create(), p.r, p.t, p.s);
        return cache[i] = parent[i] < 0 ? local : mat4.multiply(mat4.create(), world(parent[i]), local);
    };
    return data.skin.joints.map((node, j) => mat4.multiply(mat4.create(), world(node), data.skin.inverseBindMatrices.subarray(j * 16, j * 16 + 16)));
}
const sourceVertices = new WeakMap();
function referenceVertex(data, jointMatrices, vertex) {
    if (!sourceVertices.has(data)) sourceVertices.set(data, data.meshes.filter(m => m.isSkinned).flatMap(m =>
        Array.from({ length: m.vertexCount }, (_, i) => m.vertices.subarray(i * 16, i * 16 + 16))));
    const v = sourceVertices.get(data)[vertex];
    const sum = v[12] + v[13] + v[14] + v[15], out = [0, 0, 0];
    for (let k = 0; k < 4; k++) {
        const q = vec3.transformMat4(vec3.create(), v, jointMatrices[v[8 + k]]);
        for (let r = 0; r < 3; r++) out[r] += q[r] * v[12 + k] / sum;
    }
    return out;
}

test('skinned bake matches direct glTF and Animator; mixed rigid attachment, bounds and frame blends', () => {
    const data = GLBParser.parse(fixture()), bake = bakeSkinnedAnimation(data, { fps: 4, weld: false, clips: [{ name: 'bend', loop: false }] });
    assert.equal(bake.influences, 3);
    assert.deepEqual(bake.influenceNodes, [1, 2, 4]);
    assert.equal(bake.frames, 5);
    for (let f = 0; f < 5; f++) {
        const direct = matrices(data, data.animations[0], f / 4), animator = new Animator(data.skin);
        animator.play(data.animations[0]); animator.update(f / 4);
        direct.forEach((m, j) => near(m, animator.getSkinningMatrices().subarray(j * 16, j * 16 + 16)));
        for (let v = 0; v < 3; v++) near(posed(bake, f, v), referenceVertex(data, direct, v));
        for (let v = 0; v < bake.vertexCount; v++) {
            const p = posed(bake, f, v);
            p.forEach((x, k) => assert.ok(x >= bake.bounds.min[k] - 1e-6 && x <= bake.bounds.max[k] + 1e-6));
            assert.ok(Math.hypot(...p) <= bake.bounds.radius + 1e-6);
            near([bake.weightsOfVertex.subarray(v * 4, v * 4 + 4).reduce((a, b) => a + b, 0)], [1]);
        }
        // The rigid attachment follows joint 2's world transform, independently of the ignored mesh transform.
        const source = box().positions.slice(0, 3); source[1] += 0.5;
        near(posed(bake, f, 3), vec3.transformMat4(vec3.create(), source, direct[1]));
    }
    // Matrix interpolation is linear in positions, including across the looping seam.
    const loop = bakeSkinnedAnimation(data, { fps: 4, weld: false });
    for (const [a, b] of [[0, 1], [3, 0]]) for (let v = 0; v < 3; v++) {
        const blended = { ...loop, transforms: loop.transforms.map((x, i) => {
            const width = loop.influences * 12, k = i % width;
            return (loop.transforms[a * width + k] + loop.transforms[b * width + k]) / 2;
        }) };
        near(posed(blended, 0, v), posed(loop, a, v).map((x, k) => (x + posed(loop, b, v)[k]) / 2));
    }
});

test('normalized byte/ushort weights, ushort joints, omitted inverse binds and welding', () => {
    for (const [weightType, max] of [['u8', 255], ['u16', 65535]]) {
        const data = GLBParser.parse(fixture({ primitive: { jointType: 'u16', weightType, weightsNormalized: true,
            weights: [max, 0, 0, 0, 0, max, 0, 0, max - 1, 1, 0, 0] }, skins: [{ joints: [1, 2] }] }));
        near(data.meshes[0].vertices.subarray(12, 16), [1, 0, 0, 0]);
        near(data.meshes[0].vertices.subarray(44, 48), [(max - 1) / max, 1 / max, 0, 0]);
        assert.deepEqual([...data.skin.inverseBindMatrices], [...identity, ...identity]);
        const welded = bakeSkinnedAnimation(data), loose = bakeSkinnedAnimation(data, { weld: false });
        assert.equal(welded.vertexCount, 11); // 3 skin vertices + 8 welded attachment corners
        assert.equal(loose.vertexCount, 27);
        near([welded.bounds.radius], [loose.bounds.radius]);
    }
});

test('skinned welding preserves different weights and averages normals only for exact influence matches', () => {
    const p = { positions: [0, 0, 0, 0, 0, 0, 0, 0, 0], normals: [1, 0, 0, 0, 1, 0, 0, 0, 1],
        weights: [1, 1, 0, 0, 1, 1, 0, 0, 3, 1, 0, 0] };
    const bake = bakeSkinnedAnimation(GLBParser.parse(fixture({ primitive: p })));
    assert.equal(bake.vertexCount, 10); // two skin vertices, eight attachment corners
    assert.equal(bake.indices[0], bake.indices[1]);
    assert.notEqual(bake.indices[1], bake.indices[2]);
    assert.ok(Math.abs(Math.hypot(...bake.vertices.subarray(3, 6)) - 1) < 1e-6);
});

test('STEP skinned channels hold until the next key, independent of Animator interpolation', () => {
    const data = GLBParser.parse(fixture({ interpolation: 'STEP' }));
    const bake = bakeSkinnedAnimation(data, { fps: 4, weld: false, clips: [{ name: 'bend', loop: false }] });
    for (let f = 0; f < bake.frames; f++) {
        const direct = matrices(data, data.animations[0], f / 4);
        for (let v = 0; v < 3; v++) near(posed(bake, f, v), referenceVertex(data, direct, v));
    }
});

test('CUBICSPLINE skinned translation uses interval-scaled tangents and includes the held end', () => {
    const data = GLBParser.parse(fixture());
    data.animations[0] = { name: 'cubic', maxTime: 2,
        channels: [{ samplerIndex: 0, targetNode: 1, path: 'translation' }],
        samplers: [{ input: new Float32Array([0, 2]), interpolation: 'CUBICSPLINE',
            // in tangent, value, out tangent at each endpoint
            output: new Float32Array([0, 0, 0, 0, 0, 0, 0, 3, 0, 0, -1, 0, 0, 2, 0, 0, 0, 0]) }] };
    const bake = bakeSkinnedAnimation(data, { fps: 2, weld: false, clips: [{ name: 'cubic', loop: false }] });
    const rest = posed(bake, 0, 0);
    for (let f = 0; f < 5; f++) {
        const u = f / 4;
        // The Hermite curve as a cubic Bezier: P0=0, P1=2, P2=8/3, P3=2.
        const y = 3 * (1 - u) ** 2 * u * 2 + 3 * (1 - u) * u ** 2 * (8 / 3) + u ** 3 * 2;
        near(posed(bake, f, 0), [rest[0], rest[1] + 3 * y, rest[2]]);
    }
});

test('skinned bake refuses unsupported input by name rather than silently dropping influences', () => {
    const rejects = (change, pattern) => { const data = GLBParser.parse(fixture()); change(data); assert.throws(() => bakeSkinnedAnimation(data), pattern); };
    rejects(d => { d.skinCount = 2; }, /bakeSkinnedAnimation:.*2 skins/);
    rejects(d => { d.nodes[3].skin = 1; }, /skin 1/);
    rejects(d => { d.meshes[0].vertices.fill(0, 12, 16); }, /vertex 0 has no weight/);
    for (const bad of [-1, Infinity, NaN]) rejects(d => { d.meshes[0].vertices[12] = bad; }, /invalid weights/);
    rejects(d => { d.meshes[0].vertices[8] = 2; }, /names joint 2/);
    rejects(d => { d.skin.inverseBindMatrices = new Float32Array(0); }, /inverse bind matrices/);
    rejects(d => { d.skin.joints[0] = 100; }, /missing joint node/);
    rejects(d => { d.skin.joints = []; }, /0 joints/);
    rejects(d => { d.meshes = []; }, /no meshes/);
    rejects(d => { d.skin = null; }, /no skinned mesh/);
    rejects(d => { d.nodes[0].scale = [0, 0, 0]; }, /no inverse at rest/);
    rejects(d => { delete d.meshes[0].node; }, /mesh 0 has no node/);
    for (const primitive of [{ joints1: Array(12).fill(0) }, { weights1: Array(12).fill(0) }]) {
        assert.throws(() => bakeSkinnedAnimation(GLBParser.parse(fixture({ primitive }))), /JOINTS_1 or WEIGHTS_1/);
    }
    const two = [{ joints: [1, 2], inverseBindMatrices: [...identity, ...identity] }, { joints: [1], inverseBindMatrices: identity }];
    assert.throws(() => bakeSkinnedAnimation(GLBParser.parse(fixture({ skins: two }))), /2 skins/);
    for (const fps of [0, -1, Infinity, NaN]) assert.throws(() => bakeSkinnedAnimation(GLBParser.parse(fixture()), { fps }), /fps/);
    assert.throws(() => bakeSkinnedAnimation(GLBParser.parse(fixture()), { clips: [] }), /clips is empty/);
    assert.throws(() => bakeSkinnedAnimation(GLBParser.parse(fixture()), { clips: ['absent'] }), /no clip "absent"/);
});

test('shared bake refactor preserves rigid output byte-for-byte against the pre-refactor build', {
    skip: process.env.NG_RIGID_OLD_LOADER ? false : 'set NG_RIGID_OLD_LOADER to the pre-refactor loaders build',
}, async () => {
    const old = await import(pathToFileURL(process.env.NG_RIGID_OLD_LOADER).href);
    const bytes = glb({ nodes: [{ translation: [2, 1, -3], children: [1] }, { mesh: 0 }],
        meshes: [{ primitives: [box()] }], animations: [{ name: 'move',
            samplers: [{ times: [0, 1], values: [0, 0, 0, 4, 2, 1] }], channels: [{ sampler: 0, node: 1, path: 'translation' }] }] });
    for (const weld of [true, false]) for (const loop of [true, false]) {
        const options = { weld, fps: 7, clips: [{ name: 'move', loop }] };
        assert.deepEqual(bakeRigidAnimation(GLBParser.parse(bytes), options), old.bakeRigidAnimation(old.GLBParser.parse(bytes), options));
    }
    const directory = process.env.NG_GLB_DIR;
    if (directory) for (const file of ['animal-cow.glb', 'animal-deer.glb', 'animal-fox.glb']) {
        const buffer = readFileSync(join(directory, file)), bytes = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
        for (const weld of [true, false]) assert.deepEqual(bakeRigidAnimation(GLBParser.parse(bytes), { weld }), old.bakeRigidAnimation(old.GLBParser.parse(bytes), { weld }), file);
    }
});

const corpus = process.env.NG_SKINNED_GLB_DIR;
for (const file of ['AnimatedMixamo/BreakDanceGirl.glb', 'AnimatedMixamo/HipHopDancingGirl.glb', '3dAssetsNotIncluded/butterfly.glb']) {
    test(`external skinned model: ${file}`, { skip: !corpus || !existsSync(join(corpus, file)) ? 'set NG_SKINNED_GLB_DIR to the site public directory' : false }, () => {
        const bytes = readFileSync(join(corpus, file)), data = GLBParser.parse(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
        try {
            const bake = bakeSkinnedAnimation(data, { fps: 5, weld: false });
            const count = data.meshes.filter(m => m.isSkinned).reduce((n, m) => n + m.vertexCount, 0);
            const fullFrames = new Set([0, Math.floor(bake.frames / 2), bake.frames - 1]);
            for (let f = 0; f < bake.frames; f++) {
                const direct = matrices(data, data.animations[0], f * bake.clips[0].seconds / bake.frames);
                const stride = fullFrames.has(f) ? 1 : Math.max(1, Math.floor(count / 64));
                for (let v = 0; v < count; v += stride) near(posed(bake, f, v), referenceVertex(data, direct, v), 2e-4);
            }
            console.log(`${file}: ${bake.vertexCount} vertices, ${bake.influences} influences, ${bake.frames} frames, radius ${bake.bounds.radius.toFixed(3)}`);
        } finally { GLBParser.disposeImages(data); }
    });
}