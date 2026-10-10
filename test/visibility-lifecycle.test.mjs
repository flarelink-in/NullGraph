// Headless lifecycle tests for VisibilitySystem draw batches, and for a ShadowSystem sharing one.
//
// Both bugs pinned here were found by the first real-device run of test/shadows-gpu (Cycle A,
// docs/handoff.md) and both failed SILENTLY on a GPU:
//   L1 — a batch created after upload() drew nothing on its first frame: its CullParams buffer
//        starts zeroed, and instanceCount = 0 culls everything in.
//   L2/L3 — destroying a ShadowSystem left its caster cull stages in the shared visibility chain,
//        bound to destroyed camera buffers; every later frame was refused at submit.
//
// The engine runs on a mock device ADOPTED through init(null, { device }) — the same seam ONNX
// Runtime uses — so these are real NullGraph/VisibilitySystem/ShadowSystem objects end to end,
// with only the device faked. The mock records buffer writes and destroys; that is all the
// assertions read.

globalThis.GPUBufferUsage ??= {
    MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16,
    VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512,
};
globalThis.GPUTextureUsage ??= {
    COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16,
};
globalThis.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };

const { NullGraph } = await import('../dist/index.js');
const { VisibilitySystem, InstanceBuffer } = await import('../dist/visibility.js');
const { MegabufferBuilder, Primitives, StandardLayout } = await import('../dist/geometry.js');
const { ShadowSystem } = await import('../dist/shadows.js');
const { mat4 } = await import('gl-matrix');

// init announces the GPU it adopted; against a mock that line is noise.
const log = console.log;
console.log = (...a) => { if (!/^NullGraph: (Using GPU|Adopted)/.test(String(a[0]))) log(...a); };

let failures = 0;
async function check(name, fn) {
    try { await fn(); console.log(`  ok    ${name}`); }
    catch (e) { failures++; console.log(`  FAIL  ${name}\n        ${e.message}`); }
}
function assert(v, m) { if (!v) throw new Error(m); }

/** A permissive device that records writes (by target) and destroys. */
function mockDevice() {
    const writes = new Map();     // GPUBuffer -> [ArrayBuffer copies]
    const destroyed = new Set();
    const res = (extra = {}) => ({ ...extra, destroy() { destroyed.add(this); } });
    const pipeline = () => ({ getBindGroupLayout: () => ({}) });
    const device = {
        features: new Set(), adapterInfo: { vendor: 'mock' },
        createBuffer: (d) => res({ ...d, getMappedRange: () => new ArrayBuffer(d.size), unmap() {} }),
        createTexture: (d) => res({ ...d, createView: () => ({}) }),
        createSampler: () => ({}),
        createShaderModule: () => ({}),
        createRenderPipeline: pipeline,
        createComputePipeline: pipeline,
        createBindGroup: () => ({}),
        createBindGroupLayout: () => ({}),
        createPipelineLayout: () => ({}),
        createCommandEncoder: () => ({}),
        queue: {
            writeBuffer: (buffer, offset, data, dataOffset = 0, size) => {
                const src = ArrayBuffer.isView(data) ? data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) : data.slice(0);
                if (!writes.has(buffer)) writes.set(buffer, []);
                writes.get(buffer).push(src);
            },
            submit() {},
        },
    };
    return { device, writes, destroyed };
}

async function scene() {
    const mock = mockDevice();
    const engine = new NullGraph();
    const r = await engine.init(null, { device: mock.device, width: 256, height: 256 });
    if (!r.success) throw new Error(`mock engine init failed: ${r.error}`);
    const mega = new MegabufferBuilder(StandardLayout.strideFloats);
    const plane = Primitives.createPlane(StandardLayout, 10, 10);
    mega.addMesh('ground', plane.vertices, plane.indices);
    const sphere = Primitives.createSphere(StandardLayout, 1, 8, 6);
    mega.addMesh('sphere', sphere.vertices, sphere.indices);
    const vis = new VisibilitySystem(engine, { maxInstances: 64 });
    vis.setGeometry(mega);
    const inst = new InstanceBuffer(2);
    inst.write(0, mat4.create(), [0, 0, 0, 10], 0);
    inst.write(1, mat4.create(), [0, 0, 0, 1], 1);
    vis.setInstances(inst.data, 2);
    return { mock, engine, vis };
}

const paramsBuffers = (mock) =>
    [...mock.writes.keys()].filter(b => b.label === 'NullGraph_VisCullParams');

// ---------------------------------------------------------------------------

await check('L1: a draw batch\'s CullParams are written at registration — no upload() needed', async () => {
    const { mock, engine, vis } = await scene();
    // Deliberately NO vis.upload(): this is the order test/shadows-gpu's renderScene uses.
    const pass = engine.createPass({ name: 'scene', isMainScreenPass: true });
    vis.createInstanceDrawBatch(pass, 'ground', 0);
    const params = paramsBuffers(mock);
    assert(params.length === 1,
        `expected the new stage's params buffer to be written once at creation, saw ${params.length} — ` +
        `a zeroed params buffer culls every instance (instanceCount = 0) and the batch silently draws nothing`);
    const u32 = new Uint32Array(mock.writes.get(params[0]).at(-1));
    assert(u32[0] === 2, `instanceCount written as ${u32[0]}, expected the 2 set by setInstances`);
    assert(u32[1] === 0, `meshIndex written as ${u32[1]}, expected 0`);
});

await check('L2: removeDrawBatch drops the stage, releases its params, detaches the batch', async () => {
    const { mock, engine, vis } = await scene();
    const pass = engine.createPass({ name: 'scene', isMainScreenPass: true });
    const keep = vis.createInstanceDrawBatch(pass, 'ground', 0);
    const drop = vis.createInstanceDrawBatch(pass, 'sphere', 1);
    const dropStage = vis.getCullStage(drop);
    const keepStage = vis.getCullStage(keep);
    assert(vis.getComputeStages().includes(dropStage), 'precondition: both stages in the chain');

    // The record is TypeScript-private, not runtime-private; reading it is the only way to name
    // the exact params buffer that must be released.
    const paramsOfDrop = vis['instanceTierCulls'].find(r => r.batch === drop).paramsBuffer;
    assert(vis.removeDrawBatch(drop) === true, 'removeDrawBatch returned false for its own batch');
    assert(mock.destroyed.has(paramsOfDrop), 'the removed stage\'s CullParams buffer was not destroyed');

    const chain = vis.getComputeStages();
    assert(!chain.includes(dropStage), 'the removed batch\'s cull stage is still in the compute chain (cache not invalidated?)');
    assert(chain.includes(keepStage), 'removing one batch dropped another');
    assert(vis.getCullStage(drop) === null, 'getCullStage still resolves the removed batch');
    assert(!pass.batches.includes(drop) && pass.batches.includes(keep), 'the batch was not detached from its pass');
    assert(mock.destroyed.has(drop.storageBuffer) && mock.destroyed.has(drop.indirectBuffer),
        'the removed batch\'s own draw buffers were not destroyed');
    assert(!mock.destroyed.has(vis['instanceBuffer']), 'the SHARED instance buffer was destroyed with one batch');
    assert(vis.removeDrawBatch(drop) === false, 'a second remove of the same batch should report false');
    assert(vis.removeDrawBatch({}) === false, 'removing a foreign batch should report false');
});

await check('L3: ShadowSystem.destroy unregisters its casters from the SHARED VisibilitySystem', async () => {
    const { engine, vis } = await scene();
    const main = engine.createPass({ name: 'scene', isMainScreenPass: true });
    vis.createInstanceDrawBatch(main, 'ground', 0);
    const before = vis.getComputeStages().length;

    const shadows = new ShadowSystem(engine, { cascades: 2, resolution: 256, shadowDistance: 20, visibility: vis });
    shadows.registerCaster('sphere', 1);
    shadows.registerCaster('ground', 0);
    const during = vis.getComputeStages().length;
    assert(during === before + 4, `expected 2 casters x 2 cascades = 4 new cull stages, got ${during - before}`);

    shadows.destroy();
    const after = vis.getComputeStages().length;
    assert(after === before,
        `${after - before} caster cull stage(s) outlived ShadowSystem.destroy — on a GPU every later frame ` +
        `is refused at submit ("used in submit while destroyed")`);
});

console.log(failures === 0 ? '\nAll visibility lifecycle tests passed.' : `\n${failures} test(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
