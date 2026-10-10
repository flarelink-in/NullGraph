// GPU integration tests for the core execution model — Cycle A (docs/handoff.md).
//
// The headless half (core-compute + core-headless, mock devices) pins WHICH commands the engine
// issues. These pin what a real device does with them:
//   C1  headless init: the main-screen pass draws into engine.screenTexture
//   C2  device adoption: an engine on someone else's device renders, and the owner keeps it
//   C3  zero copy: a buffer another library wrote drives instances, GPU-counted — the ONNX seam,
//       proven without ONNX Runtime (no model, no download, deterministic)
//   C4  indirect dispatch: a stage dispatches exactly the workgroup count a prior stage wrote,
//       inside one compute pass (RenderPipeline's "no barrier needed" comment, measured)
//   C5  a validation error names the stage it came from (labels)
//   C6  headless resize: the screen texture follows engine.resize
//
// Two hosts, one file (test/gpu/harness-env.mjs): `npm run test:gpu` (Node/Dawn), or
// `npm run dev:serve` → http://localhost:3000/test/core-gpu/. Both run headless (canvas null) —
// a browser can render without a canvas too, and that is part of what C1 claims.

import { NullGraph } from '../../dist/index.js';
import { harnessEnv } from '../gpu/harness-env.mjs';

const { log, check, fatal, finish } = harnessEnv('__CORE_GPU_RESULTS__');

const W = 64, H = 64;
const engine = new NullGraph();
const init = await engine.init(null, { width: W, height: H });
if (!init.success) {
    fatal(init.error);
    throw new Error(init.error);
}
const device = engine.device;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/** Read mip 0 of an rgba8unorm texture into tightly packed RGBA8, top row first. */
async function readRGBA8(texture, w, h) {
    const bpr = Math.ceil(w * 4 / 256) * 256;
    const buf = device.createBuffer({ size: bpr * h, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture }, { buffer: buf, bytesPerRow: bpr }, [w, h]);
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) out.set(src.subarray(y * bpr, y * bpr + w * 4), y * w * 4);
    buf.unmap(); buf.destroy();
    return out;
}

async function readU32(buffer, count, dev = device) {
    const buf = dev.createBuffer({ size: count * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, buf, 0, count * 4);
    dev.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(buf.getMappedRange().slice(0));
    buf.unmap(); buf.destroy();
    return out;
}

const near = (a, b, tol = 1) => Math.abs(a - b) <= tol;
const px = (data, w, x, y) => Array.from(data.subarray((y * w + x) * 4, (y * w + x) * 4 + 4));

// Group 0 is engine-owned on every batch: binding 0 the 432-byte camera uniform, binding 1 the
// instance storage. layout:'auto' drops bindings a shader never touches, and the engine always
// supplies both — so each test shader references both, at zero cost.
const GROUP0 = /* wgsl */ `
@group(0) @binding(0) var<uniform> camera: array<vec4<f32>, 27>;
@group(0) @binding(1) var<storage, read> instances: array<vec4<f32>>;
fn touchGroup0() -> f32 { return camera[0].x * 0.0 + instances[0].x * 0.0; }
`;

/** A batch that fills the screen with one colour: a single full-screen triangle. */
function solidBatch(target, pass, rgba) {
    const batch = target.createBatch(pass, {
        label: 'solid fill',
        shaderCode: GROUP0 + /* wgsl */ `
            @vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
                return vec4<f32>(p * 2.0 - 1.0 + touchGroup0(), 0.5, 1.0);
            }
            @fragment fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(${rgba.map(v => v.toFixed(4)).join(', ')});
            }`,
        strideFloats: 4, maxInstances: 1,
    });
    batch.currentInstanceCount = 1;   // a non-indirect batch with 0 instances is skipped
    return batch;
}

// ---------------------------------------------------------------------------
// C1 — headless: the main-screen pass draws into engine.screenTexture
// ---------------------------------------------------------------------------
{
    const pass = engine.createPass({ name: 'screen', isMainScreenPass: true });
    solidBatch(engine, pass, [0.25, 0.5, 0.75, 1]);
    engine.render([pass]);
    const data = await readRGBA8(engine.screenTexture, W, H);
    const want = [64, 128, 191, 255];
    let bad = 0;
    for (let i = 0; i < W * H; i++) {
        for (let c = 0; c < 4; c++) if (!near(data[i * 4 + c], want[c])) { bad++; break; }
    }
    check('C1: headless init renders the main-screen pass into engine.screenTexture',
        engine.ctx.isHeadless && engine.screenTexture && engine.ctx.format === 'rgba8unorm' && bad === 0,
        `format=${engine.ctx.format} centre=${px(data, W, 32, 32)} want≈${want} — ${bad}/${W * H} px off`);
    engine.clearPasses();
}

// ---------------------------------------------------------------------------
// C2 — adoption: render on a device someone else created; the owner keeps using it
// ---------------------------------------------------------------------------
{
    const adapter = await navigator.gpu.requestAdapter();
    const owned = await adapter.requestDevice();            // "another library's" device
    const guest = new NullGraph();
    const r = await guest.init(null, { device: owned, width: 32, height: 32 });
    let ok = r.success && guest.device === owned && guest.ctx.ownsDevice === false;
    let detail = r.success ? '' : r.error;
    if (ok) {
        const pass = guest.createPass({ name: 'guest screen', isMainScreenPass: true });
        solidBatch(guest, pass, [1, 0, 0, 1]);
        guest.render([pass]);
        const bpr = 256;
        const read = owned.createBuffer({ size: bpr * 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const enc = owned.createCommandEncoder();
        enc.copyTextureToBuffer({ texture: guest.screenTexture }, { buffer: read, bytesPerRow: bpr }, [32, 32]);
        owned.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const c = Array.from(new Uint8Array(read.getMappedRange(), 16 * bpr + 16 * 4, 4));
        read.unmap(); read.destroy();
        ok = c[0] === 255 && c[1] === 0 && c[2] === 0;
        detail = `guest centre=${c}`;

        // The owner's own work still runs after the guest's — the guest did not take the device.
        const mine = owned.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        owned.queue.writeBuffer(mine, 0, new Uint32Array([0xC0FFEE]));
        const [back] = await readU32(mine, 1, owned);
        ok = ok && back === 0xC0FFEE;
        detail += ` owner round-trip=0x${back.toString(16)}`;
    }
    check('C2: an engine on an adopted device renders, and the owner keeps the device', ok, detail);
}

// ---------------------------------------------------------------------------
// C3 — zero copy: a buffer ANOTHER library wrote drives instances, counted on the GPU
//
// The ONNX seam without ONNX Runtime: the "model" is a raw compute pass this test encodes itself
// (its own encoder, its own pipeline — the engine never sees it) that writes 4 points into a
// storage buffer. A NullGraph compute stage binds that buffer AS-IS, expands it into the batch's
// instance data and writes the indirect instanceCount; the batch draws. The CPU never reads the
// points, and never learns how many there are.
// ---------------------------------------------------------------------------
{
    const N = 4;
    // "Model output": N points [x_ndc, y_ndc, r, g] + [b, 1, 0, 0] — 8 floats each.
    const modelOut = device.createBuffer({ size: N * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const producer = device.createComputePipeline({
        label: 'foreign model',
        layout: 'auto',
        compute: {
            module: device.createShaderModule({ code: /* wgsl */ `
                @group(0) @binding(0) var<storage, read_write> outPts: array<vec4<f32>>;
                @compute @workgroup_size(4) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
                    let q = vec2<f32>(f32(g.x & 1u), f32(g.x >> 1u)) - 0.5;    // 4 quadrants
                    let colours = array<vec3<f32>, 4>(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0),
                                                      vec3(0.0, 0.0, 1.0), vec3(1.0, 1.0, 0.0));
                    let c = colours[g.x];
                    outPts[g.x * 2u] = vec4<f32>(q, c.r, c.g);
                    outPts[g.x * 2u + 1u] = vec4<f32>(c.b, 1.0, 0.0, 0.0);
                }` }),
            entryPoint: 'main',
        },
    });
    {
        const enc = device.createCommandEncoder();
        const cp = enc.beginComputePass();
        cp.setPipeline(producer);
        cp.setBindGroup(0, device.createBindGroup({ layout: producer.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: modelOut } }] }));
        cp.dispatchWorkgroups(1);
        cp.end();
        device.queue.submit([enc.finish()]);
    }

    // NullGraph side: an indirect batch drawing one small quad per instance.
    const target = device.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const pass = engine.createPass({
        name: 'points', isMainScreenPass: false,
        colorAttachments: [{ view: target.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
    });
    const quadVB = engine.bufferManager.createVertexBuffer(new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]));
    const quadIB = engine.bufferManager.createIndexBuffer(new Uint16Array([0, 1, 2, 2, 1, 3]));
    const batch = engine.createBatch(pass, {
        label: 'model points',
        isIndirect: true,
        strideFloats: 8, maxInstances: N,
        vertexLayouts: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }] }],
        shaderCode: /* wgsl */ `
            @group(0) @binding(0) var<uniform> camera: array<vec4<f32>, 27>;
            @group(0) @binding(1) var<storage, read> inst: array<vec4<f32>>;
            struct V { @builtin(position) pos: vec4<f32>, @location(0) col: vec3<f32> };
            @vertex fn vs_main(@location(0) corner: vec2<f32>, @builtin(instance_index) i: u32) -> V {
                let a = inst[i * 2u]; let b = inst[i * 2u + 1u];
                var o: V;
                o.pos = vec4<f32>(a.xy + corner * 0.08 + camera[0].x * 0.0, 0.5, 1.0);
                o.col = vec3<f32>(a.z, a.w, b.x);
                return o;
            }
            @fragment fn fs_main(v: V) -> @location(0) vec4<f32> { return vec4<f32>(v.col, 1.0); }`,
    });
    engine.setBatchGeometry(batch, quadVB, quadIB, 6);

    // The expand stage: reads the foreign buffer, writes instance data + the GPU-side count.
    const expand = engine.createComputeStage({
        name: 'expand model output', dispatchCount: [1, 1, 1],
        shaderCode: /* wgsl */ `
            @group(0) @binding(0) var<storage, read> pts: array<vec4<f32>>;
            @group(0) @binding(1) var<storage, read_write> inst: array<vec4<f32>>;
            @group(0) @binding(2) var<storage, read_write> args: array<atomic<u32>, 5>;
            @compute @workgroup_size(${N}) fn cs_main(@builtin(global_invocation_id) g: vec3<u32>) {
                inst[g.x * 2u] = pts[g.x * 2u];
                inst[g.x * 2u + 1u] = pts[g.x * 2u + 1u];
                if (g.x == 0u) { atomicStore(&args[0], 6u); }    // indexCount
                atomicAdd(&args[1], 1u);                          // instanceCount: counted, not told
            }`,
        preClearBuffers: [{ buffer: batch.indirectBuffer, offset: 0, data: new Uint32Array([0, 0, 0, 0, 0]) }],
    });
    engine.setComputePassBindGroup(expand, [
        { binding: 0, resource: { buffer: modelOut } },          // the foreign buffer, unwrapped
        { binding: 1, resource: { buffer: batch.storageBuffer } },
        { binding: 2, resource: { buffer: batch.indirectBuffer } },
    ], 0);
    engine.render([expand, pass]);

    const data = await readRGBA8(target, W, H);
    // Quadrant centres (NDC ±0.5 → pixels 16/48; NDC +y is up, row 0 is the top).
    const at = (ndcX, ndcY) => px(data, W, Math.round((ndcX * 0.5 + 0.5) * W), Math.round((0.5 - ndcY * 0.5) * H));
    const got = [at(-0.5, -0.5), at(0.5, -0.5), at(-0.5, 0.5), at(0.5, 0.5)];
    const want = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]];
    const okPts = got.every((c, i) => c[0] === want[i][0] && c[1] === want[i][1] && c[2] === want[i][2]);
    const bg = px(data, W, 32, 32);
    check('C3: a buffer another library wrote drives 4 instances through a GPU-counted indirect draw',
        okPts && bg[0] === 0 && bg[1] === 0 && bg[2] === 0,
        `points=${got.map(c => c.slice(0, 3).join(',')).join(' | ')} centre(bg)=${bg.slice(0, 3)}`);
    target.destroy(); modelOut.destroy();
    engine.clearPasses();
}

// ---------------------------------------------------------------------------
// C4 — indirect dispatch: the consumer runs exactly the workgroups the producer wrote
// ---------------------------------------------------------------------------
{
    const args = engine.bufferManager.createDispatchIndirectBuffer(1, 'C4 dispatch args');
    const counter = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const want = new Uint32Array([0]);
    // The counter reset rides on the PRODUCER: RenderPipeline closes the compute pass before any
    // stage with preClearBuffers, so a reset on the consumer would split the two into separate
    // passes — and the claim under test is the single-pass case.
    const producer = engine.createComputeStage({
        name: 'C4 producer', dispatchCount: [1, 1, 1],
        shaderCode: /* wgsl */ `
            struct P { n: u32 };
            @group(0) @binding(0) var<storage, read_write> args: array<u32, 3>;
            @group(0) @binding(1) var<uniform> p: P;
            @compute @workgroup_size(1) fn cs_main() { args[0] = p.n; args[1] = 1u; args[2] = 1u; }`,
        preClearBuffers: [{ buffer: counter, offset: 0, data: want }],
    });
    const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    engine.setComputePassBindGroup(producer, [
        { binding: 0, resource: { buffer: args } },
        { binding: 1, resource: { buffer: params } },
    ], 0);
    const consumer = engine.createComputeStage({
        name: 'C4 consumer', indirectBuffer: args,
        shaderCode: /* wgsl */ `
            @group(0) @binding(0) var<storage, read_write> count: atomic<u32>;
            @compute @workgroup_size(64) fn cs_main() { atomicAdd(&count, 1u); }`,
    });
    engine.setComputePassBindGroup(consumer, [{ binding: 0, resource: { buffer: counter } }], 0);

    const counts = [];
    for (const n of [7, 3]) {
        device.queue.writeBuffer(params, 0, new Uint32Array([n, 0, 0, 0]));
        engine.render([producer, consumer]);             // one render() → one compute pass
        counts.push((await readU32(counter, 1))[0]);
    }
    check('C4: an indirect stage dispatches exactly the count a prior stage wrote, in the same pass',
        counts[0] === 7 * 64 && counts[1] === 3 * 64,
        `invocations=${counts.join(', ')} want=${7 * 64}, ${3 * 64}`);
    args.destroy(); counter.destroy(); params.destroy();
}

// ---------------------------------------------------------------------------
// C5 — a validation error names the stage that caused it
// ---------------------------------------------------------------------------
{
    device.pushErrorScope('validation');
    engine.createComputeStage({ name: 'C5 deliberately broken', shaderCode: 'fn cs_main( { }' });
    const err = await device.popErrorScope();
    check('C5: a WGSL error names the compute stage it came from (labels)',
        err !== null && /C5 deliberately broken/.test(err.message),
        err ? err.message.split('\n').find(l => /broken/.test(l))?.trim() ?? err.message.split('\n')[0] : 'no error raised');
}

// ---------------------------------------------------------------------------
// C6 — headless resize: the screen texture follows engine.resize
// ---------------------------------------------------------------------------
{
    engine.resize(48, 32);
    const pass = engine.createPass({ name: 'resized screen', isMainScreenPass: true });
    solidBatch(engine, pass, [0, 1, 0, 1]);
    engine.render([pass]);
    const t = engine.screenTexture;
    const data = await readRGBA8(t, 48, 32);
    const corner = px(data, 48, 47, 31);
    check('C6: headless resize recreates the screen target at the new size, and it renders',
        t.width === 48 && t.height === 32 && corner[1] === 255 && corner[0] === 0,
        `screen=${t.width}x${t.height} bottom-right=${corner}`);
    engine.clearPasses();
}

await finish('All core GPU tests passed.');
