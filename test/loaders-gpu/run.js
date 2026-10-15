// D1.5.2: actual cull -> visible-slot palette -> public vertex hook, against direct CPU skinning.
import { NullGraph, Camera } from '../../dist/index.js';
import { VisibilitySystem, InstanceBuffer, INSTANCE_STRUCT_WGSL } from '../../dist/visibility.js';
import { MegabufferBuilder } from '../../dist/geometry.js';
import { GLBParser, bakeSkinnedAnimation, BAKED_POSE_WGSL, BAKED_SAMPLE_WGSL,
    BAKED_PALETTE_WRITE_WGSL, BAKED_PALETTE_READ_WGSL } from '../../dist/loaders.js';
import { mat4 } from 'gl-matrix';
import { glb, box } from '../glb-fixtures/build.mjs';
import { harnessEnv } from '../gpu/harness-env.mjs';

const { check, fatal, finish, saveImage } = harnessEnv('__LOADERS_RESULTS__');
const S = 256, engine = new NullGraph();
const init = await engine.init(null, { width: S, height: S });
if (!init.success) { fatal(init.error); throw new Error(init.error); }
const device = engine.device, resources = [];
const upload = (data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) => {
    const b = device.createBuffer({ size: data.byteLength, usage });
    device.queue.writeBuffer(b, 0, data); resources.push(b); return b;
};
async function readBuffer(buffer, size) {
    const b = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = device.createCommandEncoder(); e.copyBufferToBuffer(buffer, 0, b, 0, size); device.queue.submit([e.finish()]);
    await b.mapAsync(GPUMapMode.READ); const out = b.getMappedRange().slice(0); b.unmap(); b.destroy(); return out;
}
async function readColor(texture) {
    const b = device.createBuffer({ size: S * S * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = device.createCommandEncoder(); e.copyTextureToBuffer({ texture }, { buffer: b, bytesPerRow: S * 4 }, [S, S]);
    device.queue.submit([e.finish()]); await b.mapAsync(GPUMapMode.READ);
    const out = new Uint8Array(b.getMappedRange().slice(0)); b.unmap(); b.destroy(); return out;
}
const camera = new Camera(45, 1, 0.1, 100); camera.updateView([0, 0, 8], [0, 0, 0]); engine.updateCamera(camera);
const primitive = box([-0.45, -0.6, -0.2], [0.45, 0.6, 0.2]);
primitive.joints = []; primitive.weights = [];
for (let v = 0; v < primitive.positions.length / 3; v++) {
    primitive.joints.push(0, 1, 2, 3);
    primitive.weights.push(0.1, 0.2, 0.3, 0.4); // all four fetches matter
}
const translations = [[1.2, 0, 0], [0, 1.8, 0], [0.6, 0, 0.2], [0, 0.8, 0]];
const bytes = glb({ nodes: [{ children: [1, 2, 3, 4, 5] }, { mesh: 0, skin: 0 }, {}, {}, {}, {}],
    meshes: [{ primitives: [primitive] }], skins: [{ joints: [2, 3, 4, 5], inverseBindMatrices: Array(4).fill([...mat4.create()]).flat() }],
    animations: [1, -1].map((sign, clip) => ({ name: `clip${clip}`,
        samplers: translations.map(t => ({ times: [0, 1], values: [0, 0, 0, ...t.map(x => x * sign)] })),
        channels: translations.map((_, j) => ({ sampler: j, node: j + 2, path: 'translation' })) })) });
const baked = bakeSkinnedAnimation(GLBParser.parse(bytes), { fps: 4, weld: false,
    clips: [{ name: 'clip0', loop: false }, { name: 'clip1', loop: false }] });
const pool = new MegabufferBuilder(8);
pool.addMesh('padding', new Float32Array(24), [0, 1, 2]);
const mesh = pool.addMesh('skin', baked.vertices, baked.indices);
const phases = [0, 0.25, 0, 0.75], mixWeight = 0.25;
function cpuVertices(source, rest = false) {
    const verts = baked.vertices.slice(), phase = phases[source] * (1 - 2 * mixWeight);
    const delta = [0, 0, 0];
    translations.forEach((t, j) => t.forEach((x, k) => { delta[k] += x * [0.1, 0.2, 0.3, 0.4][j] * phase; }));
    if (!rest) for (let v = 0; v < verts.length / 8; v++) for (let k = 0; k < 3; k++) verts[v * 8 + k] += delta[k];
    return verts;
}
pool.addMesh('cpu1', cpuVertices(1), baked.indices); pool.addMesh('cpu3', cpuVertices(3), baked.indices);
const vertexCount = pool.build().megaVertices.length / 8;
const joints = new Uint32Array(vertexCount * 4), weights = new Float32Array(vertexCount * 4);
joints.set(baked.influencesOfVertex, mesh.baseVertex * 4); weights.set(baked.weightsOfVertex, mesh.baseVertex * 4);
// Offset regions test nonzero bases; padding slots/unused slots must retain their sentinel values.
const tableBase = 3, paletteBase = 3, stride = 5, capacity = 4;
const tableData = new Float32Array(tableBase * 4 + baked.transforms.length); tableData.set(baked.transforms, tableBase * 4);
const paletteData = new Float32Array((paletteBase + capacity * stride * 3) * 4).fill(-777);
const table = upload(tableData), palette = upload(paletteData), jointBuffer = upload(joints), weightBuffer = upload(weights);
const state = upload(new Float32Array(phases));
const fragmentBody = `@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(normalize(in.worldNormal) * 0.3 + vec3<f32>(0.5), 1.0);
}`;
const hook = bad => ({ instanceIndex: true,
    bindings: `@group(2) @binding(0) var<storage, read> bakedPalette: array<vec4<f32>>;
        @group(2) @binding(1) var<storage, read> joints: array<vec4<u32>>;
        @group(2) @binding(2) var<storage, read> weights: array<vec4<f32>>;`,
    entries: [palette, jointBuffer, weightBuffer].map((buffer, binding) => ({ binding, resource: { buffer } })),
    wgsl: `${BAKED_POSE_WGSL} ${BAKED_PALETTE_READ_WGSL}
        fn deform(pos: vec3<f32>, normal: vec3<f32>, vi: u32, inst: Instance, slot: u32) -> Deformed {
            let pose = bakedSkinPose(${paletteBase}u, ${stride}u, ${bad ? 'inst.sourceIndex' : 'slot'}, joints[vi], weights[vi]);
            return Deformed(bakedPosition(pose, pos), bakedNormal(pose, normal));
        }`,
});
let activeVis;
async function render(kind, onlyOne = false) {
    const vis = new VisibilitySystem(engine, { maxInstances: capacity }); activeVis = vis; vis.setGeometry(pool);
    const color = device.createTexture({ size: [S, S], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const depth = device.createTexture({ size: [S, S], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const pass = engine.createPass({ name: `palette ${kind}`, isMainScreenPass: false,
        colorAttachments: [{ view: color.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
        depthStencilAttachment: { view: depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } });
    let batch;
    if (kind === 'cpu') {
        vis.createInstanceDrawBatch(pass, 'cpu1', 1, { fragmentBody }); vis.createInstanceDrawBatch(pass, 'cpu3', 3, { fragmentBody });
    } else batch = vis.createInstanceDrawBatch(pass, 'skin', 0, { fragmentBody, ...(kind === 'rest' ? {} : { vertexHook: hook(kind === 'wrong') }) });
    const rows = new InstanceBuffer(capacity);
    for (let i = 0; i < capacity; i++) {
        const m = mat4.create(); mat4.translate(m, m, [i === 1 ? -1 : 1, 0, 0]);
        // Row 0 is filtered, row 2 is outside the frustum. Original rows 1 and 3 survive.
        if (i === 2 || (onlyOne && i === 3)) mat4.translate(m, m, [1000, 0, 0]);
        rows.write(i, m, [0, 0, 0, baked.bounds.radius], i === 0 ? 99 : kind === 'cpu' ? (i === 1 ? 1 : 3) : 0);
    }
    vis.setInstances(rows.data, capacity); vis.upload();
    let stage;
    if (kind === 'palette' || kind === 'wrong') {
        device.queue.writeBuffer(palette, 0, paletteData);
        stage = engine.createComputeStage({ name: 'visible palette', dispatchCount: [1, 1, 1], shaderCode: `
            ${INSTANCE_STRUCT_WGSL}
            @group(0) @binding(0) var<storage, read> visible: array<Instance>;
            @group(0) @binding(1) var<storage, read> args: array<u32>;
            @group(0) @binding(2) var<storage, read> bakedTransforms: array<vec4<f32>>;
            @group(0) @binding(3) var<storage, read_write> bakedPalette: array<vec4<f32>>;
            @group(0) @binding(4) var<storage, read> phases: array<f32>;
            ${BAKED_POSE_WGSL} ${BAKED_SAMPLE_WGSL} ${BAKED_PALETTE_WRITE_WGSL}
            @compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let slot = gid.x / 4u; let joint = gid.x % 4u;
                if (slot >= args[1]) { return; }
                let phase = phases[visible[slot].sourceIndex];
                let a = bakedClipPose(${tableBase}u, 4u, vec2<u32>(0u, 5u), phase, false, joint);
                let b = bakedClipPose(${tableBase}u, 4u, vec2<u32>(5u, 5u), phase, false, joint);
                bakedWritePalette(${paletteBase}u, ${stride}u, slot, joint, bakedMix(a, b, 0.25));
            }` });
        engine.setComputePassBindGroup(stage, [batch.storageBuffer, batch.indirectBuffer, table, palette, state].map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
    }
    engine.render([...vis.getComputeStages(), ...(stage ? [stage] : []), pass]);
    const pixels = await readColor(color);
    let values;
    if (stage) {
        values = new Float32Array(await readBuffer(palette, paletteData.byteLength));
        const sentinel = (slot, j) => values.slice((paletteBase + (slot * stride + j) * 3) * 4, (paletteBase + (slot * stride + j + 1) * 3) * 4).every(x => x === -777);
        check(`${kind}${onlyOne ? ' shrinking' : ''}: no writes beyond GPU visible count or influence stride`,
            sentinel(onlyOne ? 1 : 2, 0) && sentinel(0, 4) && values.slice(0, paletteBase * 4).every(x => x === -777));
    }
    if (kind === 'palette' && !onlyOne) {
        const inspect = upload(new Uint32Array(4));
        const read = engine.createComputeStage({ name: 'visible ids', dispatchCount: [1, 1, 1], shaderCode: `
            ${INSTANCE_STRUCT_WGSL}
            @group(0) @binding(0) var<storage, read> visible: array<Instance>;
            @group(0) @binding(1) var<storage, read> args: array<u32>;
            @group(0) @binding(2) var<storage, read_write> ids: array<u32>;
            @compute @workgroup_size(1) fn cs_main() { ids[0] = args[1]; ids[1] = visible[0].sourceIndex; ids[2] = visible[1].sourceIndex; }` });
        engine.setComputePassBindGroup(read, [batch.storageBuffer, batch.indirectBuffer, inspect].map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
        engine.render([read]); const ids = new Uint32Array(await readBuffer(inspect, 16));
        check('actual cull compacts sparse source rows 1 and 3 into slots 0 and 1', ids[0] === 2 && [ids[1], ids[2]].sort().join() === '1,3');
        let worst = 0;
        for (let slot = 0; slot < 2; slot++) for (let j = 0; j < 4; j++) {
            const o = (paletteBase + (slot * stride + j) * 3) * 4;
            const t = translations[j].map(x => x * phases[ids[slot + 1]] * 0.5);
            const expected = [1, 0, 0, t[0], 0, 1, 0, t[1], 0, 0, 1, t[2]];
            expected.forEach((x, k) => { worst = Math.max(worst, Math.abs(values[o + k] - x)); });
        }
        check('visible palette rows match independent clip blend for each survivor', worst < 1e-6, `max error ${worst}`);
        const rigidOutput = upload(new Float32Array(8));
        const rigidStage = engine.createComputeStage({ name: 'rigid palette read', dispatchCount: [1, 1, 1], shaderCode: `
            @group(0) @binding(0) var<storage, read> bakedPalette: array<vec4<f32>>;
            @group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
            ${BAKED_POSE_WGSL} ${BAKED_PALETTE_READ_WGSL}
            @compute @workgroup_size(1) fn cs_main() {
                let p = bakedPalettePose(${paletteBase}u, ${stride}u, 0u, 1u);
                let s = bakedSkinPose(${paletteBase}u, ${stride}u, 0u, vec4<u32>(1u, 999999u, 999999u, 999999u), vec4<f32>(1.0, 0.0, 0.0, 0.0));
                output[0] = vec4<f32>(bakedPosition(p, vec3<f32>(1.0)), 1.0);
                output[1] = vec4<f32>(bakedPosition(s, vec3<f32>(1.0)), 1.0);
            }` });
        engine.setComputePassBindGroup(rigidStage, [palette, rigidOutput].map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
        engine.render([rigidStage]);
        const rigid = new Float32Array(await readBuffer(rigidOutput, 32));
        const expected = translations[1].map(x => 1 + x * phases[ids[1]] * 0.5);
        check('one-influence palette read equals weight-one skin read; zero weights skip unused indices',
            expected.every((x, k) => Math.abs(rigid[k] - x) < 1e-6 && Math.abs(rigid[k + 4] - x) < 1e-6));
        // Reuse this exact batch/stage across frames with a shrinking count, then zero survivors.
        for (const survivors of [1, 0]) {
            const moved = new InstanceBuffer(capacity);
            for (let i = 0; i < capacity; i++) {
                const model = mat4.create(); mat4.translate(model, model, [i === 1 && survivors ? -1 : 1000, 0, 0]);
                moved.write(i, model, [0, 0, 0, baked.bounds.radius], 0);
            }
            vis.setInstances(moved.data, capacity); vis.upload(); device.queue.writeBuffer(palette, 0, paletteData);
            engine.render([...vis.getComputeStages(), stage, pass]);
            const next = new Float32Array(await readBuffer(palette, paletteData.byteLength));
            check(`same batch next frame: ${survivors} survivor(s), no stale-count palette writes`,
                next.slice((paletteBase + survivors * stride * 3) * 4).every(x => x === -777) &&
                (survivors === 0 || next[paletteBase * 4] === 1));
        }
    }
    await saveImage(`palette-${kind}${onlyOne ? '-shrunk' : ''}`, pixels, S, S);
    vis.destroy(); activeVis = null; color.destroy(); depth.destroy(); engine.clearPasses(); return pixels;
}
try {
    // Timing and rigid (single-influence) reads independently of the raster test. Each input is
    // (phase, looping, firstFrame, frames); include negative wrap, seam, clamp and one-frame clips.
    const cases = [[-0.25, 1, 0, 4], [0.875, 1, 0, 4], [1.25, 1, 0, 4],
        [-1, 0, 0, 5], [1.5, 0, 0, 5], [0.375, 0, 0, 5], [12, 1, 2, 1], [1, 0, 0, 5]];
    const input = upload(new Float32Array(cases.flat())), output = upload(new Float32Array(cases.length * 4));
    const sampleStage = engine.createComputeStage({ name: 'palette sampling edges', dispatchCount: [1, 1, 1], shaderCode: `
        @group(0) @binding(0) var<storage, read> bakedTransforms: array<vec4<f32>>;
        @group(0) @binding(1) var<storage, read> samples: array<vec4<f32>>;
        @group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;
        ${BAKED_POSE_WGSL} ${BAKED_SAMPLE_WGSL}
        @compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
            if (gid.x >= ${cases.length}u) { return; }
            let c = samples[gid.x];
            let p = bakedClipPose(${tableBase}u, 4u, vec2<u32>(u32(c.z), u32(c.w)), c.x, c.y > 0.0, 0u);
            output[gid.x] = vec4<f32>(bakedPosition(p, vec3<f32>(0.0)), bakedNormal(p, vec3<f32>(0.0, 1.0, 0.0)).y);
        }` });
    engine.setComputePassBindGroup(sampleStage, [table, input, output].map((buffer, binding) => ({ binding, resource: { buffer } })), 0);
    engine.render([sampleStage]);
    const sampled = new Float32Array(await readBuffer(output, cases.length * 16));
    let error = 0;
    cases.forEach(([phase, looping, first, frames], i) => {
        const f = looping ? (phase - Math.floor(phase)) * frames : Math.max(0, Math.min(1, phase)) * (frames - 1);
        const a = Math.min(Math.floor(f), frames - 1), b = looping ? (a + 1) % frames : Math.min(a + 1, frames - 1);
        const t = ((first + a) * (1 - (f % 1)) + (first + b) * (f % 1)) / 4;
        [...translations[0].map(x => x * t), 1].forEach((x, k) => { error = Math.max(error, Math.abs(sampled[i * 4 + k] - x)); });
    });
    check('rigid sampling: negative wrap, looping seam, held end, clamps, one-frame clip and normals', error < 1e-6, `max error ${error}`);
    const cpu = await render('cpu'), gpu = await render('palette'), wrong = await render('wrong'), rest = await render('rest');
    let bad = 0, wrongPixels = 0, restPixels = 0, covered = 0;
    for (let i = 0; i < cpu.length; i += 4) {
        if (cpu[i] || cpu[i + 1] || cpu[i + 2]) covered++;
        const differs = other => [0, 1, 2].some(k => Math.abs(cpu[i + k] - other[i + k]) > 2);
        if (differs(gpu)) bad++; if (differs(wrong)) wrongPixels++; if (differs(rest)) restPixels++;
    }
    check('four-influence palette hook matches independently CPU-posed pixels', covered > 1000 && bad <= 8, `${covered} covered; ${bad} differ >2/255`);
    check('watched failing: sourceIndex is not the visible palette slot', wrongPixels > 200, `${wrongPixels} wrong pixels`);
    check('watched failing: removing palette deformation changes the image', restPixels > 200, `${restPixels} wrong pixels`);
    await render('palette', true);
} finally {
    activeVis?.destroy(); resources.forEach(b => b.destroy());
}
await finish('All baked palette GPU checks passed.');