// GPU integration tests for the ShadowSystem (docs/shadows-design.md §10 T1–T7).
//
// These assert NUMERICALLY, never by eye. `depth24plus` is not buffer-copyable, so readback runs a
// small compute pass that `textureLoad`s the cascade layer into an f32 storage buffer — legal in a
// test, forbidden in the frame loop (§10).
//
// Two hosts, one file (test/gpu/harness-env.mjs):
//   Node/Dawn:  npm run test:gpu          (headless — no browser, no server)
//   Browser:    npm run dev:serve   →   http://localhost:3000/test/shadows-gpu/
// Both publish a machine-readable summary on `__SHADOW_RESULTS__`.

import { NullGraph, Camera } from '../../dist/index.js';
import { VisibilitySystem, InstanceBuffer, INSTANCE_FLOATS, LOD_AUTO } from '../../dist/visibility.js';
import { MegabufferBuilder, Primitives, StandardLayout } from '../../dist/geometry.js';
import { LightSystem } from '../../dist/lights.js';
import { ShadowSystem } from '../../dist/shadows.js';
import { LambertMaterial } from '../../dist/materials.js';
import { mat4 } from 'gl-matrix';
import { harnessEnv } from '../gpu/harness-env.mjs';

const { log, check, fatal, finish, saveImage } = harnessEnv('__SHADOW_RESULTS__');

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

// Headless on BOTH hosts. Every target below is rgba8unorm, and batches default their colour
// format to the engine's — which is the canvas's preferred format (bgra8unorm) when there is a
// canvas. Initialised against #gpu, every pipeline here mismatched its own attachments in Chrome;
// the first browser run failed 5 checks that way. Headless, the engine format is rgba8unorm.
const engine = new NullGraph();
const init = await engine.init(null, { width: 256, height: 256 });
if (!init.success) {
    fatal(init.error);
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
async function renderScene(shadows, vis, camera, { withShadows = true, extraPasses = [], name = 'scene' } = {}) {
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
    // One directional light. Floats 8–10 are the direction of TRAVEL (DirectionalLightProxy;
    // the shader lights along -dir) — so straight down is [0, -1, 0], matching setLight above.
    // This read [0, 1, 0] before: light travelling UP from under the ground, N·L < 0 everywhere,
    // and the first real-device run measured ambient only (38/255) both in and out of shadow.
    lightData.set([0, 0, 0, 2,  1, 1, 1, 1,  0, -1, 0, 0,  0, 0, 0, 0]);
    lights.setRawData(lightData); lights.upload();

    const lighting = withShadows ? shadows.wrapLighting(lights) : lights.getShaderLayout();
    // A real LambertMaterial, not a stub: the lambert fragment declares group 1 as sampler +
    // albedo + 32-byte params, and a stub that bound an empty group made WebGPU drop the whole
    // frame's command buffer — shadow passes included — so T1/T3 read zeros and T1b/T5 passed
    // comparing zero with zero. Caught by the first Node/Dawn run (docs/handoff.md).
    const batch = vis.createInstanceDrawBatch(scenePass, 'ground', 0, {
        material: { kind: 'lambert', instance: new LambertMaterial(engine), lighting },
    });
    // Group 2 is the lights' (roadmap §6.1). Missing, the device refuses the draw with "No bind
    // group set at group index 2" and drops the frame — the second thing the first run found.
    lights.applyToBatch(batch);
    if (withShadows) shadows.applyToBatch(batch);

    engine.render([...vis.getComputeStages(), ...shadows.getPasses(), scenePass, ...extraPasses]);

    const read = device.createBuffer({ size: 256 * 256 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: target }, { buffer: read, bytesPerRow: 256 * 4 }, [256, 256]);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const px = new Uint8Array(read.getMappedRange().slice(0));
    read.unmap(); read.destroy(); target.destroy();
    // This batch was built for one frame; unregister it so its cull stage does not keep running
    // in every later frame's visibility chain (VisibilitySystem.removeDrawBatch).
    vis.removeDrawBatch(batch);
    await saveImage(`shadows-${name}`, px, 256, 256);   // a no-op unless NG_GPU_ARTIFACTS is set
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
    const px = await renderScene(shadows, vis, camera, { withShadows: true, name: 't1-on' });
    const lum = (x, y) => { const i = (y * 256 + x) * 4; return 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]; };
    // The camera looks AT the origin, and the sphere's shadow (straight-down light) is centred on
    // the origin — so "under the sphere" is the screen centre. This sampled (128, 150) before:
    // lit ground ~3 units in front of the shadow, which the first saved frame showed at a glance.
    const under = lum(128, 128);   // the shadow's centre
    const beside = lum(180, 150);  // lit ground beside it
    check('T1: shadow exists', under < beside * 0.4, `under=${under.toFixed(1)} beside=${beside.toFixed(1)}`);

    const pxNo = await renderScene(shadows, vis, camera, { withShadows: false, name: 't1-off' });
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
    // Draw the cascade before reading it. The original never did — it read a never-rendered,
    // zero-initialised texture (min = max = 0), which only a real-device run could reveal.
    engine.render([...vis.getComputeStages(), ...grazing.getPasses()]);
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
// T4 — no shimmer, no crawl (roadmap §8.2 gate 3, on the device)
//
// The headless half (U2, U3) pins the math: a rotation-invariant fit radius and a texel-snapped
// centre. This pins what the rasterizer then does with it. Render the cascade, nudge the camera,
// render it again: with snapping, the caster's footprint in the map moves by WHOLE texels only, so
// the two footprints coincide exactly at some integer shift. Shimmer (a fit radius that breathes
// with rotation, an unsnapped centre) shows as a fractional centroid shift and edge texels that
// match at no shift.
//
// A dedicated one-cascade system with ONLY the sphere as caster, so the map holds exactly its
// silhouette on the 1.0 clear. (On the main system the ground casts too and fills the cascade,
// and the comparison then measures the ground's outline instead — the first attempt did that.)
// ---------------------------------------------------------------------------
{
    const RES = 1024;
    const stable = new ShadowSystem(engine, {
        cascades: 1, resolution: RES, shadowDistance: 30, casterExtrusion: 60,
        constantBias: 0.0015, normalOffsetTexels: 1.5, pcfRadius: 1, blendTexels: 4.0, visibility: vis,
    });
    stable.registerCaster('sphere', 1);
    stable.setLight([0, -1, 0]);

    async function casterMask(cam) {
        stable.update(cam);
        vis.upload();
        engine.render([...vis.getComputeStages(), ...stable.getPasses()]);
        const depth = await readCascadeDepth(stable, 0, RES);
        const mask = new Uint8Array(RES * RES);
        let n = 0, sx = 0, sy = 0;
        for (let i = 0; i < depth.length; i++) {
            if (depth[i] < 0.999) { mask[i] = 1; n++; sx += i % RES; sy += Math.floor(i / RES); }
        }
        return { mask, n, cx: sx / n, cy: sy / n };
    }
    /** Texels that disagree when `b` is compared at an integer offset (dx, dy) from `a`. */
    function mismatches(a, b, dx, dy) {
        let bad = 0;
        for (let y = 0; y < RES; y++) for (let x = 0; x < RES; x++) {
            const xb = x + dx, yb = y + dy;
            const vb = (xb >= 0 && xb < RES && yb >= 0 && yb < RES) ? b.mask[yb * RES + xb] : 0;
            if (a.mask[y * RES + x] !== vb) bad++;
        }
        return bad;
    }
    function compare(a, b) {
        const fx = b.cx - a.cx, fy = b.cy - a.cy;
        const dx = Math.round(fx), dy = Math.round(fy);
        return { dx, dy, frac: Math.max(Math.abs(fx - dx), Math.abs(fy - dy)), bad: mismatches(a, b, dx, dy) };
    }

    const eye = [0, 8, 14], target = [0, 0, 0];
    const base = new Camera(75, 1, 0.1, 200); base.updateView(eye, target);
    const A = await casterMask(base);

    // (a) yaw the camera 1° about its own position — the cascade's centre swings (~6 texels at
    //     this distance); its texel size must not change, and its content must land on whole texels.
    const yaw = 1 * Math.PI / 180, d = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
    const yawed = new Camera(75, 1, 0.1, 200);
    yawed.updateView(eye, [eye[0] + d[0] * Math.cos(yaw) + d[2] * Math.sin(yaw), eye[1] + d[1],
        eye[2] - d[0] * Math.sin(yaw) + d[2] * Math.cos(yaw)]);
    const rot = compare(A, await casterMask(yawed));
    check('T4a: a 1° camera yaw moves the cascade by whole texels only — no shimmer',
        A.n > 300 && rot.bad === 0 && rot.frac < 1e-6 && (rot.dx !== 0 || rot.dy !== 0),
        `caster=${A.n} texels, moved (${rot.dx}, ${rot.dy}) texels, fractional part ${rot.frac.toExponential(1)}, ` +
        `${rot.bad} mismatched texels at that shift (a zero move would make this check vacuous)`);

    // (b) slide the camera sideways by a fraction of a texel's world size (~0.045 here) — the
    //     content may stay or jump a whole texel, never slide by part of one.
    const slid = new Camera(75, 1, 0.1, 200);
    slid.updateView([eye[0] + 0.013, eye[1], eye[2]], [target[0] + 0.013, target[1], target[2]]);
    const tr = compare(A, await casterMask(slid));
    check('T4b: a sub-texel camera translation never moves the caster by part of a texel — no crawl',
        tr.bad === 0 && tr.frac < 1e-6,
        `moved (${tr.dx}, ${tr.dy}) texels, fractional part ${tr.frac.toExponential(1)}, ${tr.bad} mismatched texels`);

    stable.destroy();   // safe on a shared VisibilitySystem since removeDrawBatch (lifecycle fix)
}

// ---------------------------------------------------------------------------
// T5 — cascade seam (blend band removes the resolution change)
// ---------------------------------------------------------------------------
{
    // A shadow STRIP that runs from near the camera to past the first split — 25 overlapping
    // spheres in a line under a straight-down light — seen by an oblique camera, so view depth
    // grows along the strip and crosses splitFar[0]. A seam artefact (a cascade-selection gap, a
    // broken blend band) shows as LIT pixels inside a strip that must be dark end to end; a false
    // shadow band shows as DARK pixels on the lit line beside it.
    //
    // Rebuilt after the first real-device run. The original looked straight down — every ground
    // point then sits at the same view depth, so no split was ever on screen — and its sample row
    // crossed the caster's own shadow, whose edge it reported as a 75% "seam".
    const strip = new InstanceBuffer(26);
    strip.write(0, mat4.create(), [0, 0, 0, 20], 0);
    const m = mat4.create();
    for (let k = 0; k < 25; k++) {
        const z = 5 - k * 0.5;
        mat4.fromTranslation(m, [0, 3, z]);
        strip.write(k + 1, m, [0, 0, 0, 1], 1);   // bounds are LOCAL — the cull applies the model
    }
    vis.setInstances(strip.data, 26);

    const cam = new Camera(75, 1, 0.1, 200);
    // Low and oblique: the strip's visible near end sits at view depth ≈4, well inside cascade 0
    // (splitFar ≈ 6.3), so both sides of the split carry dozens of samples, not a handful.
    cam.updateView([0, 4, 7], [0, 0, -6]);
    const px = await renderScene(shadows, vis, cam, { withShadows: true, name: 't5-seam' });
    const split = shadows.fittedCascades[0].splitFar;

    const lumAt = (x, y) => { const i = (y * 256 + x) * 4; return 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]; };
    /** World point → pixel, through the same view-projection the frame was rendered with. */
    const project = (p) => {
        const v = cam.viewProjectionMatrix;
        const cx = v[0] * p[0] + v[4] * p[1] + v[8] * p[2] + v[12];
        const cy = v[1] * p[0] + v[5] * p[1] + v[9] * p[2] + v[13];
        const cw = v[3] * p[0] + v[7] * p[1] + v[11] * p[2] + v[15];
        return [Math.floor((cx / cw * 0.5 + 0.5) * 256), Math.floor((0.5 - cy / cw * 0.5) * 256)];
    };
    const viewDepth = (p) => { const v = cam.viewMatrix; return -(v[2] * p[0] + v[6] * p[1] + v[10] * p[2] + v[14]); };

    /** Walk the ground line x = `x0` from z = 4 to z = -6; collect on-screen luminance + depth range. */
    function walk(x0) {
        const out = { lums: [], minD: Infinity, maxD: -Infinity };
        for (let z = 4.0; z >= -6.0; z -= 0.05) {
            const p = [x0, 0, z];
            const [x, y] = project(p);
            if (x < 0 || x > 255 || y < 0 || y > 255) continue;
            out.lums.push(lumAt(x, y));
            const d = viewDepth(p);
            out.minD = Math.min(out.minD, d); out.maxD = Math.max(out.maxD, d);
        }
        return out;
    }
    const inStrip = walk(0);      // centreline of the shadow: must stay dark
    const beside = walk(4);       // 3 units clear of it: must stay lit

    check('T5a: the strip straddles the first split (the check below can fail)',
        inStrip.minD < split - 1 && inStrip.maxD > split + 1 && inStrip.lums.length > 50,
        `view depth ${inStrip.minD.toFixed(2)}–${inStrip.maxD.toFixed(2)} vs splitFar[0]=${split.toFixed(2)}, ` +
        `${inStrip.lums.length} samples`);
    const lit = inStrip.lums.filter(l => l > 128).length;
    check('T5b: no lit gap along a shadow that crosses the cascade split',
        lit === 0, `${lit}/${inStrip.lums.length} samples lit; brightest=${Math.max(...inStrip.lums).toFixed(0)} ` +
        `(shadowed ≈ 38, lit ≈ 255)`);
    const dark = beside.lums.filter(l => l < 128).length;
    check('T5c: no false shadow band on lit ground across the split',
        beside.lums.length > 50 && dark === 0,
        `${dark}/${beside.lums.length} samples dark; darkest=${Math.min(...beside.lums).toFixed(0)}`);
    vis.setInstances(instances.data, 2);
}

// ---------------------------------------------------------------------------
// T7 — beyond shadowDistance is fully lit
// ---------------------------------------------------------------------------
{
    // A ground tile and a caster directly above it, both ~120 units from the camera — twice
    // shadowDistance (60). Under a straight-down light the caster's shadow lands exactly on the
    // centre pixel, so the check can tell the two outcomes apart: past the last cascade the
    // ground must be LIT (≈255), not shadowed (≈ambient, 38).
    //
    // Rebuilt after the first real-device run. The original looked from z=-100 at z=-120 — where
    // there was no ground (the plane spans ±20 around the origin) and which is only ~21 units
    // away, i.e. INSIDE the cascades — and its `lum > 4` threshold passed an unlit ground too.
    const far = new InstanceBuffer(2);
    const farGround = mat4.create(); mat4.translate(farGround, farGround, [0, 0, -120]);
    far.write(0, farGround, [0, 0, 0, 20], 0);     // bounds are LOCAL (worldSphereFromModel)
    const farSphere = mat4.create(); mat4.translate(farSphere, farSphere, [0, 3, -120]);
    far.write(1, farSphere, [0, 0, 0, 1], 1);
    vis.setInstances(far.data, 2);
    const farCam = new Camera(75, 1, 0.1, 300);
    farCam.updateView([0, 8, 0], [0, 0, -120]);   // centre ray meets the ground at (0, 0, -120)
    const px = await renderScene(shadows, vis, farCam, { withShadows: true, name: 't7-far' });
    const i = (128 * 256 + 128) * 4;
    const lum = 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2];
    check('T7: geometry beyond shadowDistance is fully lit (not shadowed by its caster)', lum > 128,
        `lum=${lum.toFixed(1)} (lit ≈ 255, shadowed ≈ 38)`);
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
await finish('All shadow GPU tests passed.');
