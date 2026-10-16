// GPU integration tests for the VisibilitySystem — the README's headline claims, on a real device
// (roadmap §8.1 gates 1–2, which were "awaiting a browser harness"; docs/handoff.md, Cycle A).
//
//   V1  draw calls scale with MESH count, not instance count (one drawIndexedIndirect per mesh,
//       counted on the encoder the engine actually used)
//   V2  the GPU's frustum cull keeps EXACTLY the instances the CPU mirror (cullMath) keeps —
//       the mirror is pinned headless; this pins the kernel to the mirror
//   V3  Hi-Z: geometry behind a wall is not drawn — and the inverted half, so the check can fail:
//       frame 0 (no depth history) and a frame with the wall moved both draw everything
//   V4  rows owned by the GPU: after setInstanceCount, upload() writes no rows, so rows written
//       straight into vis.instanceBuffer are culled and drawn (agents design §9.1)
//   V6  level of detail by size on screen: two batches with adjoining bands over one mesh index
//       draw each in-view row exactly once, as the CPU mirror of the size says
//
// Two hosts, one file (test/gpu/harness-env.mjs): `npm run test:gpu`, or
// `npm run dev:serve` → http://localhost:3000/test/visibility-gpu/.

import { NullGraph, Camera } from '../../dist/index.js';
import {
    VisibilitySystem, InstanceBuffer, extractFrustumPlanes, sphereInFrustum, worldSphereFromModel, INSTANCE_SOURCE_INDEX_OFFSET,
} from '../../dist/visibility.js';
import { MegabufferBuilder, Primitives, StandardLayout } from '../../dist/geometry.js';
import { mat4 } from 'gl-matrix';
import { harnessEnv } from '../gpu/harness-env.mjs';
import { GLBParser, bakeRigidAnimation } from '../../dist/loaders.js';
import { glb, box, quat } from '../glb-fixtures/build.mjs';

const { log, check, fatal, finish, saveImage } = harnessEnv('__VISIBILITY_RESULTS__');

const S = 256;
const engine = new NullGraph();
const init = await engine.init(null, { width: S, height: S });
if (!init.success) {
    fatal(init.error);
    throw new Error(init.error);
}
const device = engine.device;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/**
 * Count the draws the engine issues, on the encoders it really creates. RenderPipeline asks
 * `ctx.device.createCommandEncoder()` for every frame; the wrapper hands back the real encoder
 * with beginRenderPass instrumented. Nothing about the commands themselves changes.
 */
const draws = { indirect: 0, indexed: 0, plain: 0 };
{
    const real = device.createCommandEncoder.bind(device);
    device.createCommandEncoder = (desc) => {
        const enc = real(desc);
        const begin = enc.beginRenderPass.bind(enc);
        enc.beginRenderPass = (d) => {
            const pass = begin(d);
            const di = pass.drawIndexedIndirect.bind(pass), dx = pass.drawIndexed.bind(pass), dr = pass.draw.bind(pass);
            pass.drawIndexedIndirect = (...a) => { draws.indirect++; return di(...a); };
            pass.drawIndexed = (...a) => { draws.indexed++; return dx(...a); };
            pass.draw = (...a) => { draws.plain++; return dr(...a); };
            return pass;
        };
        return enc;
    };
}
const resetDraws = () => { draws.indirect = 0; draws.indexed = 0; draws.plain = 0; };

/** [indexCount, instanceCount, …] of an indirect batch. Its buffer has no COPY_SRC, so via compute. */
async function readArgs(batch) {
    const dst = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const stage = engine.createComputeStage({
        name: 'args readback', dispatchCount: [1, 1, 1],
        shaderCode: /* wgsl */ `
            @group(0) @binding(0) var<storage, read> src: array<u32, 5>;
            @group(0) @binding(1) var<storage, read_write> dst: array<u32, 8>;
            @compute @workgroup_size(1) fn cs_main() { for (var i = 0u; i < 5u; i++) { dst[i] = src[i]; } }`,
    });
    engine.setComputePassBindGroup(stage, [
        { binding: 0, resource: { buffer: batch.indirectBuffer } },
        { binding: 1, resource: { buffer: dst } },
    ], 0);
    engine.render([stage]);
    const read = device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(dst, 0, read, 0, 32);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const out = Array.from(new Uint32Array(read.getMappedRange().slice(0)).subarray(0, 5));
    read.unmap(); read.destroy(); dst.destroy();
    return out;
}

async function readColor(texture) {
    const read = device.createBuffer({ size: S * S * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture }, { buffer: read, bytesPerRow: S * 4 }, [S, S]);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const px = new Uint8Array(read.getMappedRange().slice(0));
    read.unmap(); read.destroy();
    return px;
}

/** An offscreen scene pass with a SAMPLABLE depth target — what Hi-Z builds from. */
function makeTargets(name) {
    const color = device.createTexture({ size: [S, S], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const depth = device.createTexture({ size: [S, S], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    const depthView = depth.createView();
    const pass = engine.createPass({
        name, isMainScreenPass: false,
        colorAttachments: [{ view: color.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
        depthStencilAttachment: { view: depthView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    return { color, depth, depthView, pass };
}

const mega = new MegabufferBuilder(StandardLayout.strideFloats);
const sphereGeo = Primitives.createSphere(StandardLayout, 0.5, 12, 8);
mega.addMesh('sphere', sphereGeo.vertices, sphereGeo.indices);
const cubeGeo = Primitives.createCube(StandardLayout, 1, 1, 1);
mega.addMesh('cube', cubeGeo.vertices, cubeGeo.indices);
const SPHERE = 0, CUBE = 1;
const LOCAL_BOUNDS = { [SPHERE]: [0, 0, 0, 0.5], [CUBE]: [0, 0, 0, Math.sqrt(3) / 2] };

// ---------------------------------------------------------------------------
// V1 + V2 — one draw per mesh at any instance count; the GPU cull equals the CPU mirror
// ---------------------------------------------------------------------------
{
    const vis = new VisibilitySystem(engine, { maxInstances: 20000 });
    vis.setGeometry(mega);
    const t = makeTargets('V1 scene');
    const spheres = vis.createInstanceDrawBatch(t.pass, 'sphere', SPHERE);
    const cubes = vis.createInstanceDrawBatch(t.pass, 'cube', CUBE);

    const camera = new Camera(60, 1, 0.1, 400);
    camera.updateView([0, 25, 70], [0, 0, 0]);
    engine.updateCamera(camera);
    const planes = extractFrustumPlanes(camera.viewProjectionMatrix);

    /** Fill `n` instances on a jittered grid much wider than the view; return CPU-mirror counts. */
    function fill(n) {
        const buf = new InstanceBuffer(n);
        const m = mat4.create(), ws = [0, 0, 0, 0];
        const kept = { [SPHERE]: 0, [CUBE]: 0 }, keptIndices = { [SPHERE]: [], [CUBE]: [] };
        const side = Math.ceil(Math.sqrt(n));
        for (let i = 0; i < n; i++) {
            const mesh = i % 4 === 3 ? CUBE : SPHERE;
            const x = (i % side) * 3 - side * 1.5 + ((i * 7919) % 13) * 0.1;
            const z = Math.floor(i / side) * 3 - side * 1.5;
            mat4.fromTranslation(m, [x, (i % 5) * 0.7, z]);
            buf.write(i, m, LOCAL_BOUNDS[mesh], mesh);
            worldSphereFromModel(m, LOCAL_BOUNDS[mesh], ws);
            if (sphereInFrustum(planes, ws[0], ws[1], ws[2], ws[3])) { kept[mesh]++; keptIndices[mesh].push(i); }
        }
        return { buf, kept, keptIndices };
    }

    const runs = [];
    for (const n of [1000, 13000]) {
        const { buf, kept } = fill(n);
        vis.setInstances(buf.data, n);
        vis.upload();
        resetDraws();
        engine.render([...vis.getComputeStages(), t.pass]);
        const sceneDraws = draws.indirect + draws.indexed + draws.plain;
        const [, gpuSpheres] = await readArgs(spheres);
        const [, gpuCubes] = await readArgs(cubes);
        runs.push({ n, sceneDraws, indirect: draws.indirect, kept, gpu: { [SPHERE]: gpuSpheres, [CUBE]: gpuCubes } });
        if (n === 13000) await saveImage('visibility-v1-13k', await readColor(t.color), S, S);
    }

    check('V1: one indirect draw per MESH, whatever the instance count (1k and 13k instances, 2 meshes)',
        runs.every(r => r.indirect === 2 && r.sceneDraws === 2),
        runs.map(r => `${r.n} instances → ${r.sceneDraws} draw(s), ${r.indirect} indirect`).join('; '));

    const big = runs[1];
    check('V2: the GPU frustum cull keeps exactly what the CPU mirror keeps — and culls something',
        big.gpu[SPHERE] === big.kept[SPHERE] && big.gpu[CUBE] === big.kept[CUBE] &&
        big.kept[SPHERE] > 0 && big.kept[SPHERE] + big.kept[CUBE] < big.n,
        `13k: spheres gpu=${big.gpu[SPHERE]} cpu=${big.kept[SPHERE]}, cubes gpu=${big.gpu[CUBE]} cpu=${big.kept[CUBE]} ` +
        `(culled ${big.n - big.kept[SPHERE] - big.kept[CUBE]})`);

    // V7 (visibility-design §14): the cull writes survivors in source order, so the draw order is the
    // same every frame and every run. Read back the compacted lists: their source indices must be the
    // CPU mirror's, ascending, and two runs must write the same bytes.
    {
        const n = 13000, { buf, keptIndices } = fill(n);
        vis.setInstances(buf.data, n);
        vis.upload();
        const survivors = async (batch, count) => {
            const size = Math.max(16, count * 24 * 4), read = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
            const enc = device.createCommandEncoder();
            if (count) enc.copyBufferToBuffer(batch.storageBuffer, 0, read, 0, count * 24 * 4);
            device.queue.submit([enc.finish()]);
            await read.mapAsync(GPUMapMode.READ);
            const words = new Uint32Array(read.getMappedRange().slice(0, count * 24 * 4));
            read.unmap(); read.destroy();
            return { bytes: words, sources: Array.from({ length: count }, (_, k) => words[k * 24 + INSTANCE_SOURCE_INDEX_OFFSET]) };
        };
        const run = async () => {
            engine.render([...vis.getComputeStages(), t.pass]);
            const [, s] = await readArgs(spheres), [, c] = await readArgs(cubes);
            return { [SPHERE]: await survivors(spheres, s), [CUBE]: await survivors(cubes, c) };
        };
        const a = await run(), b = await run();
        const same = (x, y) => x.length === y.length && x.every((v, k) => v === y[k]);
        check('V7: survivors come out in source order — exactly the CPU mirror\'s list, ascending',
            same(a[SPHERE].sources, keptIndices[SPHERE]) && same(a[CUBE].sources, keptIndices[CUBE]),
            `spheres ${a[SPHERE].sources.length} (first ${a[SPHERE].sources.slice(0, 4).join(', ')}), cpu ${keptIndices[SPHERE].length} (first ${keptIndices[SPHERE].slice(0, 4).join(', ')}); ` +
            `cubes ${a[CUBE].sources.length}, cpu ${keptIndices[CUBE].length}`);
        check('V7: … and two runs write the same lists, byte for byte',
            same(a[SPHERE].bytes, b[SPHERE].bytes) && same(a[CUBE].bytes, b[CUBE].bytes));
    }

    vis.destroy();
    t.color.destroy(); t.depth.destroy();
    engine.clearPasses();
}

// ---------------------------------------------------------------------------
// V3 — Hi-Z: what is behind the wall is not drawn
// ---------------------------------------------------------------------------
{
    const t = makeTargets('V3 scene');
    const vis = new VisibilitySystem(engine, { maxInstances: 1024, hiZ: { depthSource: t.depthView, width: S, height: S } });
    vis.setGeometry(mega);
    const spheres = vis.createInstanceDrawBatch(t.pass, 'sphere', SPHERE);
    vis.createInstanceDrawBatch(t.pass, 'cube', CUBE);

    const camera = new Camera(60, 1, 0.1, 200);
    camera.updateView([0, 0, 12], [0, 0, 0]);
    engine.updateCamera(camera);

    const HIDDEN = 400, FRONT = 25;
    const buf = new InstanceBuffer(1 + HIDDEN + FRONT);
    const m = mat4.create();
    // The wall: a cube scaled to 60 x 60 x 1 at z = -5 — it fills the whole view.
    const wallAt = (z) => { mat4.fromScaling(m, [60, 60, 1]); m[14] = z; buf.write(0, m, LOCAL_BOUNDS[CUBE], CUBE); };
    wallAt(-5);
    // HIDDEN spheres behind it (z = -20), in a block well inside the frustum.
    for (let i = 0; i < HIDDEN; i++) {
        mat4.fromTranslation(m, [(i % 20) - 9.5, Math.floor(i / 20) - 9.5, -20]);
        buf.write(1 + i, m, LOCAL_BOUNDS[SPHERE], SPHERE);
    }
    // FRONT spheres in front of it (z = 0).
    for (let i = 0; i < FRONT; i++) {
        mat4.fromTranslation(m, [(i % 5) * 1.5 - 3, Math.floor(i / 5) * 1.5 - 3, 0]);
        buf.write(1 + HIDDEN + i, m, LOCAL_BOUNDS[SPHERE], SPHERE);
    }
    vis.setInstances(buf.data, 1 + HIDDEN + FRONT);

    const frame = async () => {
        vis.upload();
        engine.render([...vis.getComputeStages(), t.pass]);
        return (await readArgs(spheres))[1];
    };

    const frame0 = await frame();     // no depth history yet: Hi-Z occludes nothing (documented)
    const frame1 = await frame();     // built from frame 0's depth: the wall hides the back block
    await saveImage('visibility-v3-hiz', await readColor(t.color), S, S);
    wallAt(-40);                      // move the wall BEHIND the hidden block
    vis.setInstances(buf.data, 1 + HIDDEN + FRONT);
    await frame();                    // this frame's pyramid still holds the old wall…
    const frame3 = await frame();     // …the next one does not

    check('V3a: frame 0 has no depth history, so nothing is occluded (all spheres drawn)',
        frame0 === HIDDEN + FRONT, `frame0 spheres=${frame0} want=${HIDDEN + FRONT}`);
    check('V3b: Hi-Z culls the spheres behind the wall and keeps the ones in front',
        frame1 === FRONT, `frame1 spheres=${frame1} want=${FRONT} (hidden=${HIDDEN} must be culled)`);
    check('V3c: with the wall moved behind them, the same spheres are drawn again (the cull is the wall)',
        frame3 === HIDDEN + FRONT, `after moving the wall: spheres=${frame3} want=${HIDDEN + FRONT}`);

    vis.destroy();
    t.color.destroy(); t.depth.destroy();
    engine.clearPasses();
}

// ---------------------------------------------------------------------------
// V4 — rows owned by the GPU (setInstanceCount): upload() no longer overwrites them
// ---------------------------------------------------------------------------
{
    const vis = new VisibilitySystem(engine, { maxInstances: 64 });
    vis.setGeometry(mega);
    const t = makeTargets('V4 scene');
    const spheres = vis.createInstanceDrawBatch(t.pass, 'sphere', SPHERE);
    const camera = new Camera(60, 1, 0.1, 200);
    camera.updateView([0, 0, 12], [0, 0, 0]);
    engine.updateCamera(camera);

    const N = 16, IN_VIEW = 10;
    const m = mat4.create();
    // A CPU array whose rows are all behind the camera: if upload() still copied it, nothing would draw.
    const stale = new InstanceBuffer(N);
    for (let i = 0; i < N; i++) { mat4.fromTranslation(m, [i, 0, 40]); stale.write(i, m, LOCAL_BOUNDS[SPHERE], SPHERE); }
    vis.setInstances(stale.data, N);
    // The rows the "simulation" owns: IN_VIEW in front of the camera, the rest behind it.
    const rows = new InstanceBuffer(N);
    for (let i = 0; i < N; i++) {
        mat4.fromTranslation(m, i < IN_VIEW ? [(i % 5) * 1.5 - 3, Math.floor(i / 5) * 1.5, 0] : [i, 0, 40]);
        rows.write(i, m, LOCAL_BOUNDS[SPHERE], SPHERE);
    }
    vis.setInstanceCount(N);
    device.queue.writeBuffer(vis.instanceBuffer, 0, rows.data);
    const frame = async () => {
        vis.upload();
        engine.render([...vis.getComputeStages(), t.pass]);
        return (await readArgs(spheres))[1];
    };
    const first = await frame();
    const second = await frame();   // a second upload() must not bring the stale array back
    await saveImage('visibility-v4-gpu-rows', await readColor(t.color), S, S);
    check('V4: after setInstanceCount, rows written to vis.instanceBuffer survive upload() and are culled',
        vis.instanceSource === 'gpu' && first === IN_VIEW && second === IN_VIEW,
        `source=${vis.instanceSource}, spheres drawn ${first} then ${second}, want ${IN_VIEW}`);

    vis.destroy();
    t.color.destroy(); t.depth.destroy();
    engine.clearPasses();
}

// V5 — public hook: a baked two-part GLB against independently CPU-posed geometry.
// Padding geometry tests absolute vertex_index; sparse source rows test sourceIndex after cull.
{
    const nodes = [{ children: [1, 2] }, { mesh: 0, translation: [-0.7, 0, 0] }, { mesh: 0, translation: [0.7, 0.3, 0] }];
    const primitive = box([-0.3, -0.25, -0.2], [0.3, 0.25, 0.2]);
    const bytes = glb({ nodes, meshes: [{ primitives: [primitive] }], animations: [{ name: 'turn',
        samplers: [{ times: [0, 1], values: [...quat([0, 0, 1], 0), ...quat([0, 0, 1], 90)] },
            { times: [0, 1], values: [-0.7, 0, 0, -0.4, 0.4, 0] }],
        channels: [{ sampler: 0, node: 0, path: 'rotation' }, { sampler: 1, node: 1, path: 'translation' }] }] });
    const baked = bakeRigidAnimation(GLBParser.parse(bytes), { clips: ['turn'], weld: false });
    const pool = new MegabufferBuilder(8);
    pool.addMesh('padding', new Float32Array(24), [0, 1, 2]);
    const offset = pool.addMesh('animated', baked.vertices, baked.indices);
    const frames = [0, 7, 0, 19];
    // Direct glTF pose: rotate each node's translated local box. No baked transforms are read here.
    const posed = frame => {
        const t = frame / 30, a = t * Math.PI / 2, c = Math.cos(a), s = Math.sin(a), verts = [];
        for (let part = 0; part < 2; part++) {
            const tx = part === 0 ? -0.7 + 0.3 * t : 0.7, ty = part === 0 ? 0.4 * t : 0.3;
            for (let i = 0; i < primitive.positions.length / 3; i++) {
                const x = primitive.positions[i * 3] + tx, y = primitive.positions[i * 3 + 1] + ty;
                const nx = primitive.normals[i * 3], ny = primitive.normals[i * 3 + 1];
                verts.push(c * x - s * y, s * x + c * y, primitive.positions[i * 3 + 2],
                    c * nx - s * ny, s * nx + c * ny, primitive.normals[i * 3 + 2], ...primitive.uvs.slice(i * 2, i * 2 + 2));
            }
        }
        return new Float32Array(verts);
    };
    pool.addMesh('cpu1', posed(frames[1]), baked.indices);
    pool.addMesh('cpu3', posed(frames[3]), baked.indices);
    const partOf = new Uint32Array(pool.build().megaVertices.length / 8).fill(99);
    partOf.set(baked.partOfVertex, offset.baseVertex);
    const upload = data => {
        const b = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(b, 0, data);
        return b;
    };
    const buffers = [upload(partOf), upload(baked.transforms), upload(new Uint32Array(frames))];
    const camera = new Camera(45, 1, 0.1, 100);
    camera.updateView([0, 0, 8], [0, 0, 0]);
    engine.updateCamera(camera);
    const hook = {
        bindings: `@group(2) @binding(0) var<storage, read> partOf: array<u32>;
            @group(2) @binding(1) var<storage, read> poses: array<vec4<f32>>;
            @group(2) @binding(2) var<storage, read> frameOf: array<u32>;`,
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        wgsl: `fn deform(pos: vec3<f32>, normal: vec3<f32>, vi: u32, inst: Instance) -> Deformed {
            let base = (frameOf[inst.sourceIndex] * ${baked.parts}u + partOf[vi]) * 3u;
            let a = poses[base]; let b = poses[base + 1u]; let c = poses[base + 2u];
            let p = vec4<f32>(pos, 1.0);
            return Deformed(vec3<f32>(dot(a, p), dot(b, p), dot(c, p)),
                vec3<f32>(dot(a.xyz, normal), dot(b.xyz, normal), dot(c.xyz, normal)));
        }`,
    };
    const fragmentBody = `@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
        return vec4<f32>((normalize(in.worldNormal) * 0.3 + vec3<f32>(0.5)) * (0.8 + 0.2 * in.uv.x), 1.0);
    }`;
    const render = async kind => {
        const vis = new VisibilitySystem(engine, { maxInstances: 4 });
        vis.setGeometry(pool);
        const target = makeTargets(`V5 ${kind}`);
        if (kind === 'cpu') {
            vis.createInstanceDrawBatch(target.pass, 'cpu1', 1, { fragmentBody });
            vis.createInstanceDrawBatch(target.pass, 'cpu3', 3, { fragmentBody });
        } else vis.createInstanceDrawBatch(target.pass, 'animated', 0, { fragmentBody, ...(kind === 'hook' ? { vertexHook: hook } : {}) });
        const rows = new InstanceBuffer(4);
        for (let i = 0; i < 4; i++) {
            const model = mat4.create();
            mat4.translate(model, model, [i === 1 ? -1.1 : 1.1, -0.3, 0]);
            mat4.rotateY(model, model, 0.25);
            mat4.scale(model, model, [0.8, 1.1, 0.7]);
            rows.write(i, model, [0, 0, 0, baked.bounds.radius], i % 2 === 0 ? 99 : kind === 'cpu' ? i : 0);
        }
        vis.setInstances(rows.data, 4);
        vis.upload();
        engine.render([...vis.getComputeStages(), target.pass]);
        const pixels = await readColor(target.color);
        await saveImage(`visibility-v5-${kind}`, pixels, S, S);
        vis.destroy(); target.color.destroy(); target.depth.destroy(); engine.clearPasses();
        return pixels;
    };
    const cpu = await render('cpu'), gpu = await render('hook'), rest = await render('rest');
    let bad = 0, changed = 0, covered = 0;
    for (let i = 0; i < cpu.length; i += 4) {
        if (cpu[i] || cpu[i + 1] || cpu[i + 2]) covered++;
        if ([0, 1, 2].some(k => Math.abs(cpu[i + k] - gpu[i + k]) > 2)) bad++;
        if ([0, 1, 2].some(k => Math.abs(cpu[i + k] - rest[i + k]) > 2)) changed++;
    }
    check('V5: baked hook matches CPU glTF poses, including normals, baseVertex and sparse sourceIndex',
        covered > 500 && bad <= 8, `${covered} covered pixels; ${bad} differ by >2/255 (at most 8 edge pixels allowed)`);
    check('V5 watched failing: removing the hook differs from the posed reference',
        changed > 200, `${changed} pixels differ in the rest-pose draw`);
    buffers.forEach(b => b.destroy());
}

// ---------------------------------------------------------------------------
// V6 — level of detail by size on screen (animated-models design, "Phase 4 review"): one set of
// rows, a near batch (sphere, minPixels) and a far batch (cube, maxPixels) under the same mesh
// index. Each in-view row is drawn by exactly one of them, as the CPU mirror of the band says.
// ---------------------------------------------------------------------------
{
    const T = 24;                                          // px: the band boundary
    const vis = new VisibilitySystem(engine, { maxInstances: 2000 });
    vis.setGeometry(mega);
    const t = makeTargets('V6 scene');
    const near = vis.createInstanceDrawBatch(t.pass, 'sphere', SPHERE, { lod: { minPixels: T } });
    const far = vis.createInstanceDrawBatch(t.pass, 'cube', SPHERE, { lod: { maxPixels: T } });
    const fov = 60, eye = [0, 6, 12];
    const camera = new Camera(fov, 1, 0.1, 600);
    camera.updateView(eye, [0, 0, -100]);
    engine.updateCamera(camera);
    const planes = extractFrustumPlanes(camera.viewProjectionMatrix);
    const sizeOf = ws => ws[3] * S / (Math.max(Math.hypot(ws[0] - eye[0], ws[1] - eye[1], ws[2] - eye[2]) - ws[3], camera.near) * Math.tan(fov * Math.PI / 360));
    // Rows spread evenly in size on screen, from 3 to 150 px, placed along the view at the distance
    // that gives each its size, with a sideways offset inside the view; one row in ten behind the
    // eye. Rows within 2% of the boundary are left out, so float rounding cannot move one across.
    const dir = [0, -6, -112].map(v => v / Math.hypot(6, 112)), side = [1, 0, 0];
    const buf = new InstanceBuffer(2000), m = mat4.create(), ws = [0, 0, 0, 0];
    const kept = { near: 0, far: 0, out: 0 };
    let n = 0;
    for (let i = 0; n < 2000 && i < 4000; i++) {
        const s = 0.5 + (i % 7) * 0.25, target = 3 * 50 ** ((i % 997) / 997);
        const d = 0.5 * s * S / (target * Math.tan(fov * Math.PI / 360)) * (i % 10 === 9 ? -1 : 1);
        const lateral = Math.sin(i * 1.7) * 0.4 * Math.abs(d) * Math.tan(fov * Math.PI / 360);
        mat4.fromTranslation(m, [0, 1, 2].map(k => eye[k] + dir[k] * d + side[k] * lateral));
        mat4.scale(m, m, [s, s, s]);
        worldSphereFromModel(m, LOCAL_BOUNDS[SPHERE], ws);
        const px = sizeOf(ws);
        if (Math.abs(px - T) < 0.02 * T) continue;
        buf.write(n++, m, LOCAL_BOUNDS[SPHERE], SPHERE);
        if (!sphereInFrustum(planes, ws[0], ws[1], ws[2], ws[3])) kept.out++;
        else if (px >= T) kept.near++;
        else kept.far++;
    }
    vis.setInstances(buf.data, n);
    vis.upload();
    engine.render([...vis.getComputeStages(), t.pass]);
    const gpu = { near: (await readArgs(near))[1], far: (await readArgs(far))[1] };
    await saveImage('visibility-v6-lod', await readColor(t.color), S, S);
    check('V6: each row is drawn by exactly one LOD band, as the CPU mirror of the size says',
        gpu.near === kept.near && gpu.far === kept.far && kept.near > 50 && kept.far > 50 && kept.out > 0,
        `${n} rows: near (sphere, ≥ ${T} px) gpu=${gpu.near} cpu=${kept.near}; far (cube, < ${T} px) gpu=${gpu.far} cpu=${kept.far}; out of view ${kept.out}`);
    vis.destroy();
    t.color.destroy(); t.depth.destroy();
    engine.clearPasses();
}

await finish('All visibility GPU tests passed.');
