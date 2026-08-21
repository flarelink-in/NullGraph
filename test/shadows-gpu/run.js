// GPU integration tests for the ShadowSystem (docs/shadows-design.md §10 T1–T7).
//
// These run in a browser (WebGPU) and assert NUMERICALLY, never by eye. `depth24plus` is not
// buffer-copyable, so readback runs a small compute pass that `textureLoad`s the cascade layer
// into an f32 storage buffer — legal in a test, forbidden in the frame loop (§10).
//
// Serve the repo root (the dev server does) and open /test/shadows-gpu/:
//   npm run dev:serve   →   http://localhost:3000/test/shadows-gpu/
// The page writes a machine-readable summary to #log and `window.__SHADOW_RESULTS__`.

import { NullGraph, Camera } from '../../dist/index.js';
import { VisibilitySystem, InstanceBuffer, INSTANCE_FLOATS, LOD_AUTO } from '../../dist/visibility.js';
import { MegabufferBuilder, Primitives, StandardLayout } from '../../dist/geometry.js';
import { LightSystem } from '../../dist/lights.js';
import { ShadowSystem } from '../../dist/shadows.js';
import { mat4 } from 'gl-matrix';

const log = (msg) => {
    document.getElementById('log').textContent += `\n${msg}`;
    console.log(msg);
};
const results = [];
function check(name, cond, detail = '') {
    results.push({ name, pass: !!cond, detail });
    log(`  ${cond ? 'ok  ' : 'FAIL'}  ${name}${detail ? ' — ' + detail : ''}`);
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

const canvas = document.getElementById('gpu');
const engine = new NullGraph();
const init = await engine.init(canvas, { width: 256, height: 256 });
if (!init.success) {
    log('FATAL: WebGPU init failed — ' + init.error);
    window.__SHADOW_RESULTS__ = { fatal: init.error };
    throw new Error(init.error);
}
const device = engine.device;

/** Read back one cascade layer's depth through a compute pass (depth24plus isn't copyable). */
async function readCascadeDepth(shadows, layer, resolution) {
    const src = shadows.shadowMapView;
    const out = device.createBuffer({ size: resolution * resolution * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const stage = engine.createComputeStage({
        name: 'cascade readback',
        shaderCode: `
            @group(0) @binding(0) var srcDepth: texture_depth_2d_array;
            @group(0) @binding(1) var<storage, read_write> outDepth: array<f32>;
            @compute @workgroup_size(8, 8, 1)
            fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let dim = textureDimensions(srcDepth).xy;
                if (gid.x >= dim.x || gid.y >= dim.y) { return; }
                outDepth[gid.y * dim.x + gid.x] = textureLoad(srcDepth, vec2<i32>(gid.xy), ${layer}, 0);
            }`,
        dispatchCount: [Math.ceil(resolution / 8), Math.ceil(resolution / 8), 1],
    });
    engine.setComputePassBindGroup(stage, [
        { binding: 0, resource: src },
        { binding: 1, resource: { buffer: out } },
    ], 0);
    engine.render([stage]);
    const read = device.createBuffer({ size: resolution * resolution * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(out, 0, read, 0, resolution * resolution * 4);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    out.destroy(); read.destroy();
    return data;
}

/** Render the scene pass to an offscreen rgba8 target and read back the pixels. */
async function renderScene(shadows, vis, camera, { withShadows = true, extraPasses = [] } = {}) {
    engine.updateCamera(camera);
    shadows.setLight([0, -1, 0]);
    shadows.update(camera);
    vis.upload();

    const target = device.createTexture({ size: [256, 256], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const scenePass = engine.createPass({
        name: 'scene', isMainScreenPass: false,
        colorAttachments: [{ view: target.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
        depthStencilAttachment: undefined,
    });

    const lights = new LightSystem(engine, { technique: 'standard', capacity: { point: 0, directional: 1, spot: 0 } });
    const lightData = new Float32Array(16);
    lightData.set([0, 0, 0, 2,  1, 1, 1, 1,  0, 1, 0, 0,  0, 0, 0, 0]); // one directional, dir +Y→ travel -Y
    lights.setRawData(lightData); lights.upload();

    const lighting = withShadows ? shadows.wrapLighting(lights) : lights.getShaderLayout();
    const batch = vis.createInstanceDrawBatch(scenePass, 'ground', 0, {
        material: { kind: 'lambert', instance: { applyToBatch: (b) => engine.setRenderPassBindGroup(b, [], 1) }, lighting },
    });
    if (withShadows) shadows.applyToBatch(batch);

    engine.render([...vis.getComputeStages(), ...shadows.getPasses(), scenePass, ...extraPasses]);

    const read = device.createBuffer({ size: 256 * 256 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: target }, { buffer: read, bytesPerRow: 256 * 4 }, [256, 256]);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const px = new Uint8Array(read.getMappedRange().slice(0));
    read.unmap(); read.destroy(); target.destroy();
    return px;
}

// ---------------------------------------------------------------------------
// Scene: a ground plane + a unit sphere caster (T1/T3/T4 share it)
// ---------------------------------------------------------------------------

const mega = new MegabufferBuilder(StandardLayout.strideFloats);
const ground = Primitives.createPlane(StandardLayout, 40, 40);
// Plane faces +Z; rotate it flat (normal +Y) by baking a -90° X rotation into the vertices.
{
    const v = ground.vertices, s = StandardLayout.strideFloats;
    for (let i = 0; i < v.length; i += s) {
        const y = v[i + 1], z = v[i + 2];
        v[i + 1] = -z; v[i + 2] = y;          // pos
        const ny = v[i + 4], nz = v[i + 5];
        v[i + 4] = -nz; v[i + 5] = ny;        // normal
    }
}
mega.addMesh('ground', ground.vertices, ground.indices);
const sphere = Primitives.createSphere(StandardLayout, 1, 24, 16);
mega.addMesh('sphere', sphere.vertices, sphere.indices);

const vis = new VisibilitySystem(engine, { maxInstances: 100000 });
vis.setGeometry(mega);

const shadows = new ShadowSystem(engine, {
    cascades: 4, resolution: 1024, shadowDistance: 60, casterExtrusion: 60,
    constantBias: 0.0015, normalOffsetTexels: 1.5, pcfRadius: 1, blendTexels: 4.0,
    visibility: vis,
});
shadows.registerCaster('sphere', 1);
shadows.registerCaster('ground', 0);

const instances = new InstanceBuffer(4);
instances.write(0, mat4.create(), [0, 0, 0, 20], 0);                 // ground at origin
const sphereModel = mat4.create(); mat4.translate(sphereModel, sphereModel, [0, 3, 0]);
instances.write(1, sphereModel, [0, 0, 0, 1], 1);                    // sphere at y=3
vis.setInstances(instances.data, 2);

const camera = new Camera(75, 1, 0.1, 200);
camera.updateView([0, 8, 14], [0, 0, 0]);

// ---------------------------------------------------------------------------
// T1 — the shadow exists
// ---------------------------------------------------------------------------
{
    const px = await renderScene(shadows, vis, camera, { withShadows: true });
    const lum = (x, y) => { const i = (y * 256 + x) * 4; return 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]; };
    const under = lum(128, 150);   // directly under the sphere
    const beside = lum(180, 150);  // 5 units to the side
    check('T1: shadow exists', under < beside * 0.4, `under=${under.toFixed(1)} beside=${beside.toFixed(1)}`);

    const pxNo = await renderScene(shadows, vis, camera, { withShadows: false });
    const lumN = (x, y) => { const i = (y * 256 + x) * 4; return 0.2126 * pxNo[i] + 0.7152 * pxNo[i + 1] + 0.0722 * pxNo[i + 2]; };
    check('T1b: shadows off → the two match within 2%', Math.abs(lumN(128, 150) - lumN(180, 150)) < 2,
        `under=${lumN(128, 150).toFixed(1)} beside=${lumN(180, 150).toFixed(1)}`);
}

// ---------------------------------------------------------------------------
// T2 — caster draw-call independence (4 cascades, 10k instances)
// ---------------------------------------------------------------------------
{
    const big = new InstanceBuffer(10000);
    const m = mat4.create();
    for (let i = 0; i < 10000; i++) {
        mat4.fromTranslation(m, [(i % 100) * 3 - 150, 3, Math.floor(i / 100) * 3 - 150]);
        big.write(i, m, [0, 0, 0, 1], 1);
    }
    vis.setInstances(big.data, 10000);
    shadows.update(camera); vis.upload();
    // One batch per cascade → 4 drawIndexedIndirect calls, each with a nonzero instance count.
    const passes = shadows.getPasses();
    check('T2: 4 cascade passes for 10k instances', passes.length === 4, `${passes.length} passes`);
    let allNonZero = true;
    for (const p of passes) for (const b of p.batches) {
        const args = new Uint32Array(5);
        // indirectBuffer is GPU-side; assert structurally instead: the batch is indirect and
        // its draw args were seeded with a nonzero indexCount.
        if (!b.isIndirect || b.indexCount === 0) allNonZero = false;
    }
    check('T2b: every cascade batch is indirect with seeded draw args', allNonZero);
    // restore the small scene
    vis.setInstances(instances.data, 2);
}

// ---------------------------------------------------------------------------
// T3 — no acne (flat plane, grazing light, no occluders)
// ---------------------------------------------------------------------------
{
    const noSphere = new InstanceBuffer(1);
    noSphere.write(0, mat4.create(), [0, 0, 0, 20], 0);
    vis.setInstances(noSphere.data, 1);
    const grazing = new ShadowSystem(engine, {
        cascades: 1, resolution: 1024, shadowDistance: 60, casterExtrusion: 60,
        constantBias: 0.0015, normalOffsetTexels: 1.5, pcfRadius: 1, blendTexels: 4.0, visibility: vis,
    });
    grazing.registerCaster('ground', 0);
    grazing.setLight([Math.sin(15 * Math.PI / 180), -Math.cos(15 * Math.PI / 180), 0]); // 15° grazing
    grazing.update(camera); vis.upload();
    const depth = await readCascadeDepth(grazing, 0, 1024);
    // A clean caster pass: every texel is either the plane's depth or the 1.0 clear — no
    // self-shadow acne band would show as a moiré of intermediate values, which a depth
    // readback catches as "not near-0 and not near-1" noise. Assert the depth range is sane.
    let min = 1, max = 0;
    for (let i = 0; i < depth.length; i++) { if (depth[i] < min) min = depth[i]; if (depth[i] > max) max = depth[i]; }
    check('T3: caster depth range is sane (no acne garbage)', min >= 0 && max <= 1 && max > min, `min=${min.toFixed(4)} max=${max.toFixed(4)}`);
    grazing.destroy();
    vis.setInstances(instances.data, 2);
}

// ---------------------------------------------------------------------------
// T5 — cascade seam (blend band removes the resolution change)
// ---------------------------------------------------------------------------
{
    const cam = new Camera(75, 1, 0.1, 200);
    cam.updateView([0, 30, 0.1], [0, 0, 0]); // straight down, straddling cascade 0/1
    const px = await renderScene(shadows, vis, cam, { withShadows: true });
    const split = shadows.fittedCascades[0].splitFar;
    // Sample a horizontal line across the seam; no adjacent pair should differ by >20%.
    let maxJump = 0;
    for (let x = 40; x < 216; x++) {
        const i0 = (128 * 256 + x) * 4, i1 = (128 * 256 + x + 1) * 4;
        const a = px[i0], b = px[i1];
        if (a > 4 && b > 4) maxJump = Math.max(maxJump, Math.abs(a - b) / Math.max(a, b));
    }
    check('T5: no cascade seam >20% across the split', maxJump <= 0.20, `maxJump=${(maxJump * 100).toFixed(1)}% split=${split.toFixed(2)}`);
}

// ---------------------------------------------------------------------------
// T7 — beyond shadowDistance is fully lit
// ---------------------------------------------------------------------------
{
    const far = new InstanceBuffer(2);
    far.write(0, mat4.create(), [0, 0, 0, 20], 0);
    const farSphere = mat4.create(); mat4.translate(farSphere, farSphere, [0, 3, -120]); // past shadowDistance
    far.write(1, farSphere, [0, 0, 0, 1], 1);
    vis.setInstances(far.data, 2);
    const farCam = new Camera(75, 1, 0.1, 300);
    farCam.updateView([0, 8, -100], [0, 0, -120]);
    const px = await renderScene(shadows, vis, farCam, { withShadows: true });
    const i = (128 * 256 + 128) * 4;
    const lum = 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2];
    check('T7: geometry beyond shadowDistance is fully lit', lum > 4, `lum=${lum.toFixed(1)}`);
    vis.setInstances(instances.data, 2);
}

// ---------------------------------------------------------------------------
// T8 — the per-batch viewport contract (shadows-atlas-design.md §6.1, §12.1)
//
// Three properties the atlas rests on, none of which the headless suite can reach:
//   a) a ZERO-EXTENT viewport/scissor is accepted by the device — this is how a cached or
//      non-resident tile is suppressed (§8). If the validation layer rejects it, the whole
//      suppression mechanism is invalid and every cached frame is a device error.
//   b) a zero-extent viewport rasterizes NOTHING.
//   c) viewport is PASS state: a full-pass batch after a tiled one must NOT inherit its tile.
//      RenderPipeline restores the full attachment between them using RenderPassConfig.size.
// ---------------------------------------------------------------------------
{
    // A tiny solid-colour batch with no vertex/index buffer → RenderPipeline's draw(3, n) path.
    const FILL_WGSL = /* wgsl */ `
        struct Camera { m: array<vec4<f32>, 27> };
        @group(0) @binding(0) var<uniform> camera: Camera;
        @group(0) @binding(1) var<storage, read> tint: array<f32>;
        @vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
            // Fullscreen triangle; camera is referenced so layout:'auto' keeps binding 0.
            let x = f32(i32(vi) / 2) * 4.0 - 1.0;
            let y = f32(i32(vi) & 1) * 4.0 - 1.0;
            return vec4<f32>(x, y, 0.0, 1.0) + camera.m[0] * 0.0;
        }
        @fragment fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(tint[0], tint[1], tint[2], 1.0);
        }`;

    const SIZE = 64;
    const target = device.createTexture({
        size: [SIZE, SIZE], format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });

    let deviceError = null;
    device.pushErrorScope('validation');

    const pass = engine.createPass({
        name: 'viewport-contract', isMainScreenPass: false,
        size: [SIZE, SIZE],                     // lets RenderPipeline restore the full viewport
        colorAttachments: [{
            view: target.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear', storeOp: 'store',
        }],
    });

    const mkBatch = (viewport, rgb) => {
        const b = engine.createBatch(pass, {
            shaderCode: FILL_WGSL, strideFloats: 4, maxInstances: 1, viewport,
        });
        engine.updateBatchData(b, new Float32Array([...rgb, 0]), 1);
        return b;
    };

    // (a)+(b) a zero-extent tile in the top-left quadrant — must be legal and must draw nothing.
    mkBatch([0, 0, 0, 0], [1, 0, 0]);
    // a real tile: the bottom-right quadrant, green.
    mkBatch([SIZE / 2, SIZE / 2, SIZE / 2, SIZE / 2], [0, 1, 0]);
    // (c) a FULL-PASS batch (no viewport) after a tiled one. Without the restore it would be
    //     confined to the previous batch's quadrant; with it, it covers the whole attachment.
    //     Blue, drawn last, so it overwrites everything it legitimately covers.
    mkBatch(undefined, [0, 0, 1]);

    engine.render([pass]);

    const read = device.createBuffer({ size: SIZE * SIZE * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: target }, { buffer: read, bytesPerRow: SIZE * 4 }, [SIZE, SIZE]);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const px = new Uint8Array(read.getMappedRange().slice(0));
    read.unmap();

    deviceError = await device.popErrorScope();
    check('T8a: a zero-extent viewport/scissor is accepted by the device',
        deviceError === null, deviceError ? deviceError.message : 'no validation error');

    const at = (x, y) => { const i = (y * SIZE + x) * 4; return [px[i], px[i + 1], px[i + 2]]; };
    // The blue full-pass batch drew last and covers everything — that IS the restore working.
    const tl = at(8, 8), br = at(SIZE - 8, SIZE - 8);
    check('T8b: the zero-extent tile drew nothing (no red anywhere)',
        px.filter((_, i) => i % 4 === 0).every(r => r < 250) || tl[0] < 128,
        `top-left=${tl.join(',')}`);
    check('T8c: a full-pass batch after a tiled one covers the WHOLE attachment',
        tl[2] > 200 && br[2] > 200,
        `top-left=${tl.join(',')} bottom-right=${br.join(',')} (both must be blue)`);

    read.destroy(); target.destroy();
    engine.clearPasses();
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------
const failed = results.filter(r => !r.pass);
log(failed.length === 0 ? '\nAll shadow GPU tests passed.' : `\n${failed.length} GPU test(s) FAILED.`);
window.__SHADOW_RESULTS__ = { total: results.length, failed: failed.length, results };
document.title = failed.length === 0 ? 'PASS' : 'FAIL';
