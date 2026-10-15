// Cycle D, step 1, phase 1 (docs/agents-animated-models-design.md §3, §9): GLBParser's node per mesh,
// and bakeRigidAnimation, headless.
//
//   U37  GLBParser records each mesh's node and the file's nodes; what it returned before is unchanged
//   U38  the bake against glTF evaluated here, by other formulas than the library's: at every frame,
//        between frames, across a looping clip's end; the clip table
//   U39  welding (exact position, uv and part; normals averaged; never across parts), and the bounds
//   U40  every refusal fires and names what it refuses
//
// The test site's animals are checked as well when they are here: copy them from the site's
// public/3dAssetsNotIncluded/ into test/spikes/animated-models/models/ (ignored by git), or name a
// folder with NG_GLB_DIR. 3D model files are never committed: the other tests write theirs in memory.
//
// Imports dist/: run `npm run build:lib` first.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { GLBParser, bakeRigidAnimation } from '../dist/loaders.js';
import { glb, box, quat } from './glb-fixtures/build.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const fround = Math.fround;

// --- glTF evaluated directly: points carried up the node chain, in doubles ---------------------
// No 4×4 matrix and no inverse, unlike the library: each node rotates its scaled point by its
// quaternion (q p q*) and translates it; a cubic segment is evaluated as a Bézier curve.
const qmul = (a, b) => [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
const unit = q => { const l = Math.hypot(...q); return q.map(x => x / l); };
const rotate = (q, p) => { const u = unit(q); return qmul(qmul(u, [...p, 0]), [-u[0], -u[1], -u[2], u[3]]).slice(0, 3); };
/** From a to b by the shorter way, as a rotation about one axis by a share of the angle. */
function slerpRef(a, b, u) {
    if (a.reduce((s, x, k) => s + x * b[k], 0) < 0) b = b.map(x => -x);
    const r = qmul([-a[0], -a[1], -a[2], a[3]], b), sin = Math.hypot(r[0], r[1], r[2]);
    if (sin < 1e-12) return a;
    const angle = 2 * Math.atan2(sin, r[3]), s = Math.sin(u * angle / 2) / sin;
    return qmul(a, [r[0] * s, r[1] * s, r[2] * s, Math.cos(u * angle / 2)]);
}
/** A sampler `{ times, values, interpolation }` (floats as the file stores them) at time t. */
function sampleRef(s, n, rotation, t) {
    const times = s.times, cubic = s.interpolation === 'CUBICSPLINE', last = times.length - 1;
    const part = (k, p) => s.values.slice((cubic ? 3 * k + p : k) * n, (cubic ? 3 * k + p : k) * n + n);
    const value = k => part(k, 1);
    if (t <= times[0]) return value(0);
    if (t >= times[last]) return value(last);
    const k = times.findLastIndex(x => x <= t);
    if (s.interpolation === 'STEP') return value(k);
    const dt = times[k + 1] - times[k], u = (t - times[k]) / dt;
    if (!cubic) return rotation ? slerpRef(value(k), value(k + 1), u) : value(k).map((x, j) => x + (value(k + 1)[j] - x) * u);
    const p = [value(k), value(k).map((x, j) => x + part(k, 2)[j] * dt / 3), value(k + 1).map((x, j) => x - part(k + 1, 0)[j] * dt / 3), value(k + 1)];
    const lerp = (a, b) => a.map((x, j) => x + (b[j] - x) * u);
    const [q0, q1, q2] = [lerp(p[0], p[1]), lerp(p[1], p[2]), lerp(p[2], p[3])];
    const out = lerp(lerp(q0, q1), lerp(q1, q2));
    return rotation ? unit(out) : out;
}
/** Each animated node's translation, rotation and scale at time t; the rest come from the nodes. */
function poseRef(animation, t) {
    const pose = new Map();
    for (const c of animation.channels) {
        const key = { translation: 't', rotation: 'r', scale: 's' }[c.path];
        if (!key) continue;
        pose.set(c.node, { ...pose.get(c.node), [key]: sampleRef(animation.samplers[c.sampler], key === 'r' ? 4 : 3, key === 'r', t) });
    }
    return pose;
}
/** A point of node n's mesh, carried to the model's space through every node above it. */
function placeRef(nodes, parent, n, pose, p) {
    for (; n >= 0; n = parent[n]) {
        const node = nodes[n], trs = pose.get(n);
        if (node.matrix && !trs) {
            const m = node.matrix;
            p = [0, 1, 2].map(r => m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r]);
            continue;
        }
        const t = trs?.t ?? node.translation ?? [0, 0, 0], r = trs?.r ?? node.rotation ?? [0, 0, 0, 1], s = trs?.s ?? node.scale ?? [1, 1, 1];
        p = rotate(r, p.map((x, k) => x * s[k])).map((x, k) => x + t[k]);
    }
    return p;
}
const parentsOf = nodes => {
    const parent = nodes.map(() => -1);
    nodes.forEach((node, n) => (node.children ?? []).forEach(c => { parent[c] = n; }));
    return parent;
};
/** The parser's order of meshes: the scene's nodes depth first, each mesh's primitives in order. */
function meshNodesRef(nodes, scene) {
    const out = [];
    const walk = n => { if (nodes[n].mesh !== undefined) out.push(n); (nodes[n].children ?? []).forEach(walk); };
    scene.forEach(walk);
    return out;
}

// --- the bake, read as a vertex shader reads it ----------------------------------------------
const rowsOf = (baked, frame, part) => baked.transforms.subarray((frame * baked.parts + part) * 12, (frame * baked.parts + part) * 12 + 12);
const pose = (rows, v) => [0, 1, 2].map(k => rows[4 * k] * v[0] + rows[4 * k + 1] * v[1] + rows[4 * k + 2] * v[2] + rows[4 * k + 3]);
const mix = (a, b, w) => a.map((x, k) => x + (b[k] - x) * w);
const vertexOf = (baked, i) => [...baked.vertices.subarray(i * 8, i * 8 + 3)];
const distance = (a, b) => Math.hypot(...a.map((x, k) => x - b[k]));
const bits = array => [...new Uint32Array(array.buffer, array.byteOffset, array.length)];
/** The time of each frame of a clip, by the rule the clip table states. */
const frameTime = (clip, f) => (clip.loop ? f * clip.seconds / clip.frames : clip.frames > 1 ? f * clip.seconds / (clip.frames - 1) : 0);

// --- the rig: one model with every case the bake must read ----------------------------------
const flagA = { positions: [0, 0, 0, 0.5, 0, 0, 0, 0.3, 0], normals: [0, 0, 1, 0, 0, 1, 0, 0, 1], uvs: [0, 0, 1, 0, 0, 1], indices: [0, 1, 2] };
const flagB = { positions: [0.5, 0, 0, 0.5, 0.3, 0, 0, 0.3, 0], normals: [0, 0, 1, 0, 0, 1, 0, 0, 1], uvs: [1, 0, 1, 1, 0, 1], indices: [0, 1, 2] };
const c30 = Math.cos(Math.PI / 6), s30 = Math.sin(Math.PI / 6);
const RIG = {
    nodes: [
        { name: 'model', children: [1], scale: [2, 2, 2] },                                            // 0: a scale over everything
        { name: 'root', children: [2, 5, 6, 7] },                                                       // 1: spins and bobs
        { name: 'body', mesh: 0, translation: [0, 1, 0], children: [3] },                               // 2
        { name: 'head', mesh: 1, translation: [0, 0.5, 0.7], rotation: quat([1, 0, 0], 10), children: [4] }, // 3: nods, cubic
        { name: 'hat', mesh: 3, matrix: [c30, s30, 0, 0, -s30, c30, 0, 0, 0, 0, 1, 0, 0, 0.3, 0, 1] },  // 4: a matrix, rides the head
        { name: 'leg', mesh: 2, translation: [0.25, 0.7, 0.4], scale: [1, 1.5, 1] },                    // 5: swings; scaled unevenly at rest
        { name: 'leg-twin', mesh: 2, translation: [0.25, 0.7, 0.4], scale: [1, 1.5, 1] },               // 6: the same mesh at the same place
        { name: 'flag', mesh: 4, translation: [0, 1.5, -0.6] },                                         // 7: two primitives; steps and stretches
        { name: 'stray', mesh: 0, translation: [9, 9, 9] },                                             // 8: in no scene: no mesh, no part
    ],
    meshes: [
        { primitives: [box([-0.4, -0.3, -0.6], [0.4, 0.3, 0.6])] },             // body: its corners weld, 24 → 8
        { primitives: [box([-0.2, -0.2, -0.2], [0.2, 0.2, 0.2], { seam: true })] }, // head: a uv seam on every edge, 24 stay
        { primitives: [box([-0.1, -0.5, -0.1], [0.1, 0, 0.1])] },               // leg
        { primitives: [box([-0.15, 0, -0.15], [0.15, 0.1, 0.15])] },            // hat
        { primitives: [flagA, flagB] },                                          // flag: the two share an edge, 6 → 4
    ],
    animations: [
        { name: 'spin', samplers: [
            { times: [0, 0.25, 0.5, 0.75, 1], values: [0, 90, 180, 270, 360].flatMap(d => quat([0, 1, 0], d)) },
            { times: [0, 0.5, 1], values: [0, 0, 0, 0, 0.1, 0, 0, 0, 0] },
            { times: [0, 0.5, 1], interpolation: 'CUBICSPLINE', values: [
                0, 0, 0, 0, ...quat([1, 0, 0], 10), 0.6, 0, 0, -0.1,
                0, 0, 0, 0, ...quat([1, 0, 0], 40), 0, 0, 0, 0,
                -0.6, 0, 0, 0.1, ...quat([1, 0, 0], 10), 0, 0, 0, 0] },
            { times: [0, 0.5, 1], values: [-30, 30, -30].flatMap(d => quat([1, 0, 0], d)) },
        ], channels: [
            { sampler: 0, node: 1, path: 'rotation' }, { sampler: 1, node: 1, path: 'translation' },
            { sampler: 2, node: 3, path: 'rotation' }, { sampler: 3, node: 5, path: 'rotation' },
        ] },
        { name: 'wave', samplers: [
            { times: [0, 0.15], interpolation: 'STEP', values: [0, 1.5, -0.6, 0, 1.7, -0.6] },
            { times: [0, 0.4], values: [1, 1, 1, 1, 2, 1] },
        ], channels: [{ sampler: 0, node: 7, path: 'translation' }, { sampler: 1, node: 7, path: 'scale' }] },
        { name: 'pose', samplers: [{ times: [0], values: quat([1, 0, 0], 45) }], channels: [{ sampler: 0, node: 5, path: 'rotation' }] },
        { name: 'late', samplers: [{ times: [0.5, 1.5], values: [...quat([1, 0, 0], 0), ...quat([1, 0, 0], 90)] }],
            channels: [{ sampler: 0, node: 6, path: 'rotation' }] },
    ],
};
const rigBytes = glb(RIG);
/** The rig's animations as the file stores them: every key and value rounded to a float. */
const rigClips = Object.fromEntries(RIG.animations.map(a => [a.name, {
    ...a, samplers: a.samplers.map(s => ({ ...s, times: s.times.map(fround), values: s.values.map(fround) })),
}]));
const ASKED = ['spin', { name: 'wave', loop: false }, 'pose', 'late'];

test('U37: GLBParser records each mesh\'s node and the file\'s nodes, and bakes the meshes as before', () => {
    const parsed = GLBParser.parse(rigBytes.slice(0));
    // Depth first from the scene: body, head, hat, leg, leg-twin, then the flag's two primitives.
    assert.deepEqual(parsed.meshes.map(m => m.node), [2, 3, 4, 5, 6, 7, 7]);
    assert.deepEqual(parsed.nodes, RIG.nodes, 'every node of the file, the stray one too, as the file holds them');
    assert.equal(parsed.skin, null);
    assert.deepEqual(parsed.animations.map(a => a.name), ['spin', 'wave', 'pose', 'late']);
    // The meshes are baked at their nodes' rest transforms, as before: positions to float rounding,
    // uvs and indices exactly, normals turned with their node. (The parser carries a normal by the
    // node's matrix, which is a normal's transform only without an uneven scale: the legs are left out.)
    const parent = parentsOf(RIG.nodes), rest = new Map();
    const sources = [RIG.meshes[0], RIG.meshes[1], RIG.meshes[3], RIG.meshes[2], RIG.meshes[2], RIG.meshes[4], RIG.meshes[4]];
    let worst = 0;
    parsed.meshes.forEach((mesh, k) => {
        const p = sources[k].primitives[k === 6 ? 1 : 0];
        assert.equal(mesh.vertexCount, p.positions.length / 3);
        assert.deepEqual([...mesh.indices], p.indices);
        assert.equal(mesh.isSkinned, false);
        for (let v = 0; v < mesh.vertexCount; v++) {
            const want = placeRef(RIG.nodes, parent, mesh.node, rest, p.positions.slice(v * 3, v * 3 + 3));
            worst = Math.max(worst, distance([...mesh.vertices.subarray(v * 8, v * 8 + 3)], want));
            assert.deepEqual([...mesh.vertices.subarray(v * 8 + 6, v * 8 + 8)], p.uvs.slice(v * 2, v * 2 + 2).map(fround));
            const n = mesh.vertices.subarray(v * 8 + 3, v * 8 + 6), up = placeRef(RIG.nodes, parent, mesh.node, rest, [0, 0, 0]);
            const wantN = placeRef(RIG.nodes, parent, mesh.node, rest, p.normals.slice(v * 3, v * 3 + 3)).map((x, j) => x - up[j]);
            const cos = n.reduce((s, x, j) => s + x * wantN[j], 0) / Math.hypot(...n) / Math.hypot(...wantN);
            if (mesh.node !== 5 && mesh.node !== 6) assert.ok(cos > 1 - 1e-6, `mesh ${k} vertex ${v}: its normal turned with its node`);
        }
    });
    assert.ok(worst < 2e-6, `the rest positions are within ${worst} of glTF's`);

    // A skinned mesh records its node too, and is left in its own space as before.
    const skinned = GLBParser.parse(glb({
        nodes: [{ name: 'armature', children: [1, 2] }, { name: 'skinned', mesh: 0, skin: 0, translation: [5, 0, 0] }, { name: 'bone' }],
        meshes: [{ primitives: [{ ...box(), joints: new Array(96).fill(0), weights: Array.from({ length: 24 }, () => [1, 0, 0, 0]).flat() }] }],
        skins: [{ joints: [2], inverseBindMatrices: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] }],
    }));
    assert.equal(skinned.meshes[0].node, 1);
    assert.equal(skinned.meshes[0].isSkinned, true);
    assert.equal(skinned.meshes[0].vertices[0], box().positions[0], 'not moved by its node');
});

test('U38: the bake poses every vertex as glTF does, at every frame of every clip', () => {
    const parsed = GLBParser.parse(rigBytes.slice(0));
    const baked = bakeRigidAnimation(parsed, { clips: ASKED, weld: false });
    const parent = parentsOf(RIG.nodes);

    // The clip table: frames at 30 fps; a looping clip stops one frame short of its end.
    assert.deepEqual(baked.clips, [
        { name: 'spin', firstFrame: 0, frames: 30, seconds: 1, loop: true },
        { name: 'wave', firstFrame: 30, frames: 13, seconds: fround(0.4), loop: false },
        { name: 'pose', firstFrame: 43, frames: 1, seconds: 0, loop: true },
        { name: 'late', firstFrame: 44, frames: 45, seconds: 1.5, loop: true },
    ]);
    assert.equal(baked.frames, 89);
    assert.deepEqual(baked.partNodes, [2, 3, 4, 5, 6, 7]);
    assert.equal(baked.parts, 6);
    assert.equal(baked.transforms.length, 89 * 6 * 12);
    const partNode = [...baked.partOfVertex].map(p => baked.partNodes[p]);
    // Without welding, the vertices are the parser's, in its order, and each knows its source point.
    const sources = [];
    meshNodesRef(RIG.nodes, [0]).forEach(n => RIG.meshes[RIG.nodes[n].mesh].primitives.forEach(p => {
        for (let v = 0; v < p.positions.length / 3; v++) sources.push(p.positions.slice(v * 3, v * 3 + 3));
    }));
    assert.equal(sources.length, baked.vertexCount);

    let worst = 0;
    const errors = {};
    for (const clip of baked.clips) {
        const animation = rigClips[clip.name];
        for (let f = 0; f < clip.frames; f++) {
            const t = frameTime(clip, f), want = poseRef(animation, t);
            for (let i = 0; i < baked.vertexCount; i++) {
                const got = pose(rowsOf(baked, clip.firstFrame + f, baked.partOfVertex[i]), vertexOf(baked, i));
                const e = distance(got, placeRef(RIG.nodes, parent, partNode[i], want, sources[i]));
                worst = Math.max(worst, e);
                errors[clip.name] = Math.max(errors[clip.name] ?? 0, e);
            }
        }
    }
    assert.ok(worst < 1e-5, `a vertex is ${worst} from glTF's pose (per clip: ${JSON.stringify(errors)})`);

    // Particular frames: the flag's origin (at 2 × (0, 1.5, −0.6) at rest) steps up 2 × 0.2 at 0.15 s,
    // between frames 4 and 5 of wave; the last frame of a clip played once is its end (the flag's
    // height scaled by 2); the late clip holds its first key until 0.5 s.
    const flagPart = baked.partNodes.indexOf(7), y = frame => pose(rowsOf(baked, frame, flagPart), [0, 3, -1.2])[1];
    assert.ok(Math.abs(y(30 + 4) - 3) < 1e-6 && Math.abs(y(30 + 5) - 3.4) < 1e-6, `the step falls between frames 4 and 5: ${y(34)}, ${y(35)}`);
    assert.ok(Math.abs(rowsOf(baked, 42, flagPart)[5] - 2) < 1e-6, 'the last frame of wave is its end (scale 2)');
    const twin = baked.partNodes.indexOf(6);
    for (let f = 0; f <= 15; f++) assert.ok(Math.abs(rowsOf(baked, 44 + f, twin)[5] - 1) < 1e-6, `late, frame ${f}: still the first key`);
    assert.ok(Math.abs(rowsOf(baked, 44 + 16, twin)[5] - 1) > 1e-4, 'late moves after 0.5 s');

    // Between frames, as a shader blends them: no further from glTF than any 30 fps sampling is, and
    // across the end of a clip that loops without a seam (its end is its start: spin), the blend runs
    // from the last frame into the first. Late loops with a seam (0° to 90°): its end is not checked.
    let between = 0, wrapped = 0;
    const seamless = clip => baked.partNodes.every(n => distance(placeRef(RIG.nodes, parentsOf(RIG.nodes), n, poseRef(rigClips[clip.name], 0), [0.1, 0.2, 0.3]),
        placeRef(RIG.nodes, parentsOf(RIG.nodes), n, poseRef(rigClips[clip.name], clip.seconds), [0.1, 0.2, 0.3])) < 1e-6);
    for (const clip of baked.clips.filter(c => c.frames > 1)) {
        const animation = rigClips[clip.name];
        for (let f = 0; f < clip.frames; f++) {
            const wraps = f === clip.frames - 1;
            if (wraps && !(clip.loop && seamless(clip))) continue;
            if (wraps) wrapped++;
            const g = wraps ? 0 : f + 1, t0 = frameTime(clip, f), t1 = wraps ? clip.seconds : frameTime(clip, g);
            const [want0, want1, wantMid] = [t0, t1, (t0 + t1) / 2].map(t => poseRef(animation, t));
            for (let i = 0; i < baked.vertexCount; i++) {
                const v = vertexOf(baked, i), part = baked.partOfVertex[i];
                const got = pose(mix(rowsOf(baked, clip.firstFrame + f, part), rowsOf(baked, clip.firstFrame + g, part), 0.5), v);
                const ends = [want0, want1].map(w => placeRef(RIG.nodes, parent, partNode[i], w, sources[i]));
                const mid = placeRef(RIG.nodes, parent, partNode[i], wantMid, sources[i]);
                const chord = distance(mix(ends[0], ends[1], 0.5), mid);
                assert.ok(distance(got, mid) <= chord + 1e-5, `${clip.name}, between frames ${f} and ${g}: ${distance(got, mid)} from glTF, the chord ${chord}`);
                between = Math.max(between, distance(got, mid));
            }
        }
    }
    assert.ok(between > 0.005, `the spin turns 12° a frame: blends between frames cut its arc (${between})`);
    assert.equal(wrapped, 1, "spin's end, and only spin's, is checked");
    console.log(`U38: the rig within ${worst.toExponential(1)} of glTF at the frames; ${between.toFixed(3)} at most between frames (the chord of a 12° step)`);

    // Every clip of the file by default, looping; the weights channels of morph targets are passed over.
    assert.deepEqual(bakeRigidAnimation(parsed).clips.map(c => [c.name, c.loop]), [['spin', true], ['wave', true], ['pose', true], ['late', true]]);
    const morphs = GLBParser.parse(glb({ ...RIG, animations: [{ ...RIG.animations[1], samplers: [...RIG.animations[1].samplers,
        { times: [0, 0.4], values: [0, 1] }], channels: [...RIG.animations[1].channels, { sampler: 2, node: 7, path: 'weights' }] }] }));
    const withMorphs = bakeRigidAnimation(morphs, { clips: [{ name: 'wave', loop: false }], weld: false });
    assert.deepEqual(bits(withMorphs.transforms), bits(baked.transforms.subarray(30 * 72, 43 * 72)));
    // Another rate: 12 frames a second.
    assert.deepEqual(bakeRigidAnimation(parsed, { fps: 12, clips: ['spin', { name: 'wave', loop: false }] }).clips.map(c => c.frames), [12, 6]);
});

test('U39: welding merges exactly the vertices that share position, uv and part; the bounds hold every pose', () => {
    const parsed = GLBParser.parse(rigBytes.slice(0));
    const loose = bakeRigidAnimation(parsed, { clips: ASKED, weld: false });
    const welded = bakeRigidAnimation(parsed, { clips: ASKED });

    // Unwelded: the parser's vertices bit for bit, its indices offset mesh by mesh.
    const concat = parsed.meshes.flatMap(m => [...m.vertices]);
    assert.deepEqual(bits(loose.vertices), bits(new Float32Array(concat)));
    let base = 0;
    const indices = parsed.meshes.flatMap(m => { const out = [...m.indices].map(i => i + base); base += m.vertexCount; return out; });
    assert.deepEqual([...loose.indices], indices);
    assert.deepEqual([...loose.partOfVertex], parsed.meshes.flatMap(m => new Array(m.vertexCount).fill(loose.partNodes.indexOf(m.node))));

    // Welded: body 24 → 8, the seamed head keeps 24, hat and both legs 8 each, the flag 6 → 4.
    // The twin leg sits on the first one, vertex for vertex: it is another part, so nothing merges.
    const perPart = loose.partNodes.map((_, p) => [...welded.partOfVertex].filter(x => x === p).length);
    assert.deepEqual(perPart, [8, 24, 8, 8, 8, 4]);
    assert.equal(welded.vertexCount, 60);
    assert.deepEqual(bits(welded.transforms), bits(loose.transforms), 'welding leaves the table alone');
    // The same triangles, corner for corner: position and uv bits, and part.
    assert.equal(welded.indexCount, loose.indexCount);
    for (let k = 0; k < loose.indexCount; k++) {
        const a = welded.indices[k], b = loose.indices[k];
        for (const j of [0, 1, 2, 6, 7]) assert.equal(bits(welded.vertices.subarray(a * 8 + j, a * 8 + j + 1))[0], bits(loose.vertices.subarray(b * 8 + j, b * 8 + j + 1))[0]);
        assert.equal(welded.partOfVertex[a], loose.partOfVertex[b]);
    }
    // A full weld: no two vertices left that share position, uv and part.
    const keys = new Set();
    for (let i = 0; i < welded.vertexCount; i++) {
        keys.add([...bits(welded.vertices.subarray(i * 8, i * 8 + 3)), ...bits(welded.vertices.subarray(i * 8 + 6, i * 8 + 8)), welded.partOfVertex[i]].join());
    }
    assert.equal(keys.size, welded.vertexCount);
    // Each normal is the mean of those merged: a box corner's three faces give (±1, ±1, ±1) / √3.
    const merged = Array.from({ length: welded.vertexCount }, () => new Map());
    for (let k = 0; k < loose.indexCount; k++) merged[welded.indices[k]].set(loose.indices[k], [...loose.vertices.subarray(loose.indices[k] * 8 + 3, loose.indices[k] * 8 + 6)]);
    merged.forEach((sources, w) => {
        const sum = [0, 1, 2].map(j => [...sources.values()].reduce((s, n) => s + n[j], 0)), l = Math.hypot(...sum);
        const n = [...welded.vertices.subarray(w * 8 + 3, w * 8 + 6)];
        assert.ok(distance(n, sum.map(x => x / l)) < 1e-6, `vertex ${w}: normal ${n}, the mean of ${sources.size}`);
    });
    const corner = [...welded.vertices.subarray(3, 6)];
    assert.ok(corner.every(x => Math.abs(Math.abs(x) - 1 / Math.sqrt(3)) < 1e-6), `a body corner's normal: ${corner}`);

    // The bounds: every vertex in every frame, and any blend of frames, is inside them; they are tight.
    const { min, max, radius } = welded.bounds;
    let far = 0;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let f = 0; f < welded.frames; f++) {
        for (let i = 0; i < welded.vertexCount; i++) {
            const q = pose(rowsOf(welded, f, welded.partOfVertex[i]), vertexOf(welded, i));
            q.forEach((x, k) => { lo[k] = Math.min(lo[k], x); hi[k] = Math.max(hi[k], x); });
            far = Math.max(far, Math.hypot(...q));
        }
    }
    [lo, hi].forEach((got, k) => got.forEach((x, j) => assert.ok(Math.abs(x - [min, max][k][j]) < 1e-9, `the box: ${got} against ${[min, max][k]}`)));
    assert.ok(Math.abs(far - radius) < 1e-9, `the radius ${radius}, the farthest vertex ${far}`);
    let seed = 7;
    const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
    for (let n = 0; n < 2000; n++) {
        const i = Math.floor(random() * welded.vertexCount), [f, g] = [random(), random()].map(r => Math.floor(r * welded.frames));
        const q = pose(mix(rowsOf(welded, f, welded.partOfVertex[i]), rowsOf(welded, g, welded.partOfVertex[i]), random()), vertexOf(welded, i));
        assert.ok(Math.hypot(...q) <= radius + 1e-9 && q.every((x, k) => x >= min[k] - 1e-9 && x <= max[k] + 1e-9), `a blend of frames ${f} and ${g} leaves the bounds`);
    }
    // The rest pose alone stays well inside: the spin and the swing reach further.
    const restFar = Math.max(...Array.from({ length: welded.vertexCount }, (_, i) => Math.hypot(...vertexOf(welded, i))));
    assert.ok(radius > restFar, `the radius ${radius} covers more than the rest pose (${restFar})`);
});

test('U40: every refusal fires and names what it refuses', async () => {
    const refused = (glbData, options, pattern, label) => assert.throws(() => bakeRigidAnimation(glbData, options),
        error => { assert.match(error.message, /^bakeRigidAnimation: /, label); assert.match(error.message, pattern, label); return true; });
    const parsed = GLBParser.parse(rigBytes.slice(0));
    const bad = { matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] };

    for (const fps of [0, -30, NaN, Infinity]) refused(parsed, { fps }, /^bakeRigidAnimation: fps must be a number above 0; got /, `fps ${fps}`);
    refused({ ...parsed, meshes: [] }, {}, /the file has no meshes\.$/, 'no meshes');
    refused({ ...parsed, meshes: null }, {}, /the file has no meshes\.$/, 'meshes null');
    refused({ ...parsed, nodes: undefined }, {}, /mesh 0 has no node: parse the file with GLBParser\.parse or load \(loadMerged drops the nodes\)\.$/, 'no nodes');
    refused({ ...parsed, meshes: parsed.meshes.map((m, k) => (k === 3 ? { ...m, node: undefined } : m)) }, {}, /mesh 3 has no node/, 'a mesh without node');
    refused({ ...parsed, meshes: parsed.meshes.map((m, k) => (k === 2 ? { ...m, isSkinned: true } : m)) }, {}, /mesh 2 is skinned; this bakes meshes moved by their nodes \(rigid parts\) only\.$/, 'skinned');
    refused(parsed, { clips: ['spin', 'trot'] }, /no clip "trot" in the file; it has "spin", "wave", "pose", "late"\.$/, 'a missing clip');
    refused(parsed, { clips: [{ name: 'Spin' }] }, /no clip "Spin"/, 'names are exact');
    refused(parsed, { clips: [] }, /clips is empty\.$/, 'no clip asked');
    refused({ ...parsed, animations: [] }, {}, /the file has no clips\.$/, 'a file with no clips');
    refused({ ...parsed, animations: [] }, { clips: ['spin'] }, /no clip "spin" in the file; it has none\.$/, 'a clip asked of a file with none');
    refused(GLBParser.parse(glb({ ...RIG, nodes: RIG.nodes.map((n, k) => (k === 7 ? { ...n, translation: undefined, ...bad } : n)) })),
        {}, /node 7 \("flag"\) has a matrix and is animated; glTF animates translation, rotation and scale only\.$/, 'an animated matrix');
    refused(GLBParser.parse(glb({ ...RIG, nodes: RIG.nodes.map((n, k) => (k === 2 ? { ...n, scale: [1, 0, 1] } : n)) })),
        {}, /node 2 \("body"\) has a rest transform with no inverse \(a scale of 0\?\)\.$/, 'a flat part');
    refused(GLBParser.parse(glb({ ...RIG, nodes: RIG.nodes.map((n, k) => (k === 1 ? { ...n, scale: [0, 0, 0] } : n)) })),
        {}, /node 2 \("body"\) has a rest transform with no inverse/, 'a flat parent');
    refused(GLBParser.parse(glb({ ...RIG, animations: [{ ...RIG.animations[1], samplers: [{ ...RIG.animations[1].samplers[0], interpolation: 'SMOOTH' }, RIG.animations[1].samplers[1]] }] })),
        {}, /clip "wave" interpolates by SMOOTH; glTF's are LINEAR, STEP and CUBICSPLINE\.$/, 'an unknown interpolation');
    // A skinned file, as the parser reads it.
    const skinnedFile = glb({
        nodes: [{ name: 'armature', children: [1, 2] }, { name: 'skinned', mesh: 0, skin: 0 }, { name: 'bone' }],
        meshes: [{ primitives: [{ ...box(), joints: new Array(96).fill(0), weights: Array.from({ length: 24 }, () => [1, 0, 0, 0]).flat() }] }],
        skins: [{ joints: [2], inverseBindMatrices: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] }],
        animations: [{ name: 'wiggle', samplers: [{ times: [0, 1], values: [0, 0, 0, 0, 1, 0] }], channels: [{ sampler: 0, node: 2, path: 'translation' }] }],
    });
    refused(GLBParser.parse(skinnedFile), {}, /mesh 0 is skinned/, 'a skinned file');
    // loadMerged's output: the meshes merged into one, which has no node.
    const merged = await GLBParser.loadMerged(`data:model/gltf-binary;base64,${Buffer.from(rigBytes).toString('base64')}`);
    assert.equal(merged.meshes.length, 1);
    refused(merged, {}, /mesh 0 has no node: parse the file with GLBParser\.parse or load/, 'loadMerged');
});

// --- the test site's animals, when they are here ----------------------------------------------
const MODELS = process.env.NG_GLB_DIR ?? join(root, 'test/spikes/animated-models/models');
const glbFiles = existsSync(MODELS) ? readdirSync(MODELS).filter(f => f.endsWith('.glb')).sort() : [];
/** A .glb's JSON, and an accessor's values as plain numbers: read here, apart from GLBParser. */
function readFile(bytes) {
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), view = new DataView(buffer);
    const jsonLength = view.getUint32(12, true), json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 20, jsonLength)));
    const TYPES = { 5126: Float32Array, 5123: Uint16Array, 5121: Uint8Array, 5125: Uint32Array }, SIZES = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
    const accessor = index => {
        const a = json.accessors[index], view_ = json.bufferViews[a.bufferView], Type = TYPES[a.componentType], n = SIZES[a.type];
        const stride = view_.byteStride || n * Type.BYTES_PER_ELEMENT, start = 28 + jsonLength + (view_.byteOffset ?? 0) + (a.byteOffset ?? 0);
        return Array.from({ length: a.count }, (_, i) => [...new Type(buffer, start + i * stride, n)]);
    };
    return { json, buffer, accessor };
}

test('U38–U39 on the test site\'s animals: posed as glTF poses them, the static clip at rest, welded as phase 0 measured', { skip: glbFiles.length ? false : `no .glb in ${MODELS}: copy the site's animals there to run this` }, () => {
    const lines = [];
    const welds = { 'animal-fox.glb': 423, 'animal-deer.glb': 524, 'animal-cow.glb': 418 };   // phase 0's counts
    for (const file of glbFiles) {
        const bytes = readFileSync(join(MODELS, file)), { json, accessor } = readFile(bytes);
        if (!json.animations?.length || json.skins?.length) { lines.push(`${file}: skipped (${json.skins?.length ? 'skinned' : 'no clips'})`); continue; }
        const parsed = GLBParser.parse(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
        const t0 = performance.now();
        const loose = bakeRigidAnimation(parsed, { weld: false });
        const ms = performance.now() - t0;
        const welded = bakeRigidAnimation(parsed);
        const four = json.animations.some(a => a.name === 'run') ? bakeRigidAnimation(parsed, { clips: ['idle', 'walk', 'run', 'eat'] }) : null;

        // Every vertex, at every frame of every clip, against glTF evaluated here from the file's own bytes.
        const nodes = json.nodes, parent = parentsOf(nodes), sources = [], owner = [];
        meshNodesRef(nodes, json.scenes[json.scene ?? 0].nodes).forEach(n => json.meshes[nodes[n].mesh].primitives.forEach(p => {
            for (const v of accessor(p.attributes.POSITION)) { sources.push(v); owner.push(n); }
        }));
        assert.equal(sources.length, loose.vertexCount, file);
        const clipsOf = Object.fromEntries(json.animations.map(a => [a.name, {
            channels: a.channels.map(c => ({ sampler: c.sampler, node: c.target.node, path: c.target.path })),
            samplers: a.samplers.map(s => ({ times: accessor(s.input).flat(), values: accessor(s.output).flat(), interpolation: s.interpolation ?? 'LINEAR' })),
        }]));
        // Between frames: halfway, as a shader blends two frames, against glTF there (a clip's end
        // into its start is left out: not every clip of these files loops).
        let worst = 0, between = 0;
        const betweenOf = {};
        for (const clip of loose.clips) {
            for (let f = 0; f < clip.frames; f++) {
                const want = poseRef(clipsOf[clip.name], frameTime(clip, f)), g = f + 1;
                const mid = g < clip.frames ? poseRef(clipsOf[clip.name], (frameTime(clip, f) + frameTime(clip, g)) / 2) : null;
                for (let i = 0; i < loose.vertexCount; i++) {
                    const part = loose.partOfVertex[i], v = vertexOf(loose, i);
                    worst = Math.max(worst, distance(pose(rowsOf(loose, clip.firstFrame + f, part), v), placeRef(nodes, parent, owner[i], want, sources[i])));
                    if (!mid) continue;
                    const blend = pose(mix(rowsOf(loose, clip.firstFrame + f, part), rowsOf(loose, clip.firstFrame + g, part), 0.5), v);
                    const e = distance(blend, placeRef(nodes, parent, owner[i], mid, sources[i]));
                    between = Math.max(between, e);
                    betweenOf[clip.name] = Math.max(betweenOf[clip.name] ?? 0, e);
                }
            }
        }
        assert.ok(worst < 1e-5, `${file}: a vertex is ${worst} from glTF's pose`);
        // The static clip's first frame is the rest pose.
        const statics = loose.clips.find(c => c.name === 'static');
        let restError = 0;
        if (statics) {
            for (let p = 0; p < loose.parts; p++) rowsOf(loose, statics.firstFrame, p).forEach((x, k) => { restError = Math.max(restError, Math.abs(x - (k % 5 === 0 ? 1 : 0))); });
            assert.ok(restError < 1e-6, `${file}: static's first frame is ${restError} from the rest pose`);
        }
        if (welds[file]) assert.equal(welded.vertexCount, welds[file], `${file}: welded vertices`);
        assert.ok(welded.bounds.radius >= loose.bounds.radius - 1e-6 && welded.bounds.radius <= loose.bounds.radius + 1e-6, `${file}: welding moves no vertex`);
        lines.push(`${file}: ${loose.indexCount / 3} triangles, ${loose.parts} parts, ${loose.vertexCount} → ${welded.vertexCount} vertices welded; ` +
            `${loose.clips.length} clips, ${loose.frames} frames, ${loose.transforms.byteLength} B${four ? ` (idle, walk, run, eat: ${four.transforms.byteLength} B)` : ''}; ` +
            `baked in ${ms.toFixed(1)} ms; within ${worst.toExponential(1)} of glTF at the frames, ${(between * 1000).toFixed(1)} mm between them ` +
            `(${Object.entries(betweenOf).map(([c, e]) => `${c} ${(e * 1000).toFixed(1)}`).join(', ')})` +
            `${statics ? `; static at rest within ${restError.toExponential(1)}` : ''}; radius ${welded.bounds.radius.toFixed(3)}`);
    }
    console.log(lines.join('\n'));
});
