// GPU integration tests for the VisibilitySystem — the README's headline claims, on a real device
// (roadmap §8.1 gates 1–2, which were "awaiting a browser harness"; docs/handoff.md, Cycle A).
//
//   V1  draw calls scale with MESH count, not instance count (one drawIndexedIndirect per mesh,
//       counted on the encoder the engine actually used)
//   V2  the GPU's frustum cull keeps EXACTLY the instances the CPU mirror (cullMath) keeps —
//       the mirror is pinned headless; this pins the kernel to the mirror
//   V3  Hi-Z: geometry behind a wall is not drawn — and the inverted half, so the check can fail:
//       frame 0 (no depth history) and a frame with the wall moved both draw everything
//
// Two hosts, one file (test/gpu/harness-env.mjs): `npm run test:gpu`, or
// `npm run dev:serve` → http://localhost:3000/test/visibility-gpu/.

import { NullGraph, Camera } from '../../dist/index.js';
import {
    VisibilitySystem, InstanceBuffer, extractFrustumPlanes, sphereInFrustum, worldSphereFromModel,
} from '../../dist/visibility.js';
import { MegabufferBuilder, Primitives, StandardLayout } from '../../dist/geometry.js';
import { mat4 } from 'gl-matrix';
import { harnessEnv } from '../gpu/harness-env.mjs';

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
        const kept = { [SPHERE]: 0, [CUBE]: 0 };
        const side = Math.ceil(Math.sqrt(n));
        for (let i = 0; i < n; i++) {
            const mesh = i % 4 === 3 ? CUBE : SPHERE;
            const x = (i % side) * 3 - side * 1.5 + ((i * 7919) % 13) * 0.1;
            const z = Math.floor(i / side) * 3 - side * 1.5;
            mat4.fromTranslation(m, [x, (i % 5) * 0.7, z]);
            buf.write(i, m, LOCAL_BOUNDS[mesh], mesh);
            worldSphereFromModel(m, LOCAL_BOUNDS[mesh], ws);
            if (sphereInFrustum(planes, ws[0], ws[1], ws[2], ws[3])) kept[mesh]++;
        }
        return { buf, kept };
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

await finish('All visibility GPU tests passed.');
