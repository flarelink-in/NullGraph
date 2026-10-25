// Headless core tests — device adoption and canvas-less init (docs/handoff.md, Cycle A).
//
// Same discipline as core-compute.test.mjs: a mock device that RECORDS what the engine asked for.
// The questions here are "did init request an adapter or not", "which texture did the main pass
// draw into", "did the error name the option" — none of which a rendered image answers. The real-
// device half lives in test/gpu/ (Dawn), and exercises the same paths end to end.

globalThis.GPUBufferUsage = globalThis.GPUBufferUsage ?? {
    MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32,
    UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512,
};
globalThis.GPUTextureUsage = globalThis.GPUTextureUsage ?? {
    COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16,
};

const { NullGraph, WebGPUContext, RenderPipeline, RenderPassNode, BatchManager, HEADLESS_SCREEN_FORMAT } =
    await import('../dist/index.js');

// init announces the GPU it got ("NullGraph: Using GPU — …"); against mock devices that is noise.
const log = console.log;
console.log = (...a) => { if (!/^NullGraph: (Using GPU|Adopted)/.test(String(a[0]))) log(...a); };

let failures = 0;
async function check(name, fn) {
    try { await fn(); console.log(`  ok    ${name}`); }
    catch (e) { failures++; console.log(`  FAIL  ${name}\n        ${e.message}`); }
}
function assert(v, m) { if (!v) throw new Error(m); }
async function rejects(fn, pattern, label) {
    let message = '';
    try { await fn(); } catch (e) { message = e.message; }
    if (!pattern.test(message)) throw new Error(`${label}: expected ${pattern}, got "${message}"`);
    return message;
}

const TU = globalThis.GPUTextureUsage;

/**
 * A device that records textures, buffers, labels, draws, submits, and every destroy() call (once per call,
 * so a double release shows). `features` is what an owner requested.
 */
function mockDevice({ features = [], adapterInfo = { vendor: 'mock', subgroupMinSize: 16, subgroupMaxSize: 64 } } = {}) {
    const log = { textures: [], buffers: [], labels: [], destroyed: [], destroyedBuffers: [], colorViews: [], submits: 0 };
    const passEncoder = {
        setPipeline() {}, setBindGroup() {}, setViewport() {}, setScissorRect() {},
        setVertexBuffer() {}, setIndexBuffer() {}, drawIndexed() {}, draw() {},
        drawIndexedIndirect() {}, end() {},
    };
    const device = {
        features: new Set(features),
        adapterInfo,
        createBuffer: (d) => {
            const buf = { ...d, getMappedRange: () => new ArrayBuffer(d.size), unmap() {}, destroy() { log.destroyedBuffers.push(buf); } };
            log.buffers.push(buf);
            return buf;
        },
        createTexture: (d) => {
            const tex = {
                ...d, views: 0,
                createView() { tex.views++; return { texture: tex }; },
                destroy() { log.destroyed.push(tex); },
            };
            log.textures.push(tex);
            return tex;
        },
        createSampler: () => ({}),
        createShaderModule: (d) => { log.labels.push(['module', d.label]); return {}; },
        createRenderPipeline: (d) => { log.labels.push(['render', d.label]); return { getBindGroupLayout: () => ({}) }; },
        createComputePipeline: (d) => { log.labels.push(['compute', d.label]); return { getBindGroupLayout: () => ({}) }; },
        createBindGroup: () => ({}),
        createCommandEncoder: () => ({
            beginRenderPass: (d) => { log.colorViews.push(d.colorAttachments[0]?.view); return passEncoder; },
            beginComputePass: () => ({ setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }),
            finish: () => ({}),
        }),
        queue: { writeBuffer() {}, writeTexture() {}, submit() { log.submits++; } },
        destroy() { log.deviceDestroyed = true; },
    };
    return { device, log };
}

/** Install a navigator.gpu whose adapter hands out `device`, recording what was requested. */
function installNavigatorGpu(device, adapterFeatures = [], adapterLimits = {}) {
    const calls = { requestAdapter: [], requestDevice: [] };
    const adapter = {
        features: new Set(adapterFeatures),
        limits: adapterLimits,
        info: { vendor: 'mock-adapter' },
        requestDevice: async (d) => { calls.requestDevice.push(d); device.features = new Set(d?.requiredFeatures ?? []); return device; },
    };
    Object.defineProperty(globalThis, 'navigator', {
        configurable: true, writable: true,
        value: { gpu: { requestAdapter: async (o) => { calls.requestAdapter.push(o); return adapter; }, getPreferredCanvasFormat: () => 'bgra8unorm' } },
    });
    return calls;
}

/** A navigator.gpu that fails the test if anyone touches it. */
function installTrapNavigatorGpu() {
    Object.defineProperty(globalThis, 'navigator', {
        configurable: true, writable: true,
        value: { gpu: { requestAdapter: async () => { throw new Error('requestAdapter was called on an adoption path'); } } },
    });
}

function removeNavigatorGpu() {
    Object.defineProperty(globalThis, 'navigator', { configurable: true, writable: true, value: {} });
}

/** Render one frame of a main-screen pass with no batches; return the color view it drew into. */
function renderMainPass(ctx) {
    const pass = new RenderPassNode({ name: 'main', isMainScreenPass: true });
    new RenderPipeline(ctx).render([pass]);
}

// ---------------------------------------------------------------------------

await check('H1: init(null, { device }) adopts the device — no adapter request, ownsDevice false', async () => {
    installTrapNavigatorGpu();
    const { device } = mockDevice({ features: ['subgroups', 'timestamp-query'] });
    const ctx = new WebGPUContext();
    await ctx.init(null, { device, width: 64, height: 32 });
    assert(ctx.device === device, 'the adopted device is not the one in use');
    assert(ctx.ownsDevice === false, 'an adopted device is reported as owned');
    assert(ctx.capabilities.subgroups === true, 'capabilities ignored the adopted device features');
    assert(ctx.capabilities.has('timestamp-query'), 'a non-progression feature the owner enabled is not reported');
    assert(ctx.capabilities.subgroupMinSize === 16 && ctx.capabilities.subgroupMaxSize === 64,
        `subgroup sizes were not read from device.adapterInfo: ${ctx.capabilities.subgroupMinSize}/${ctx.capabilities.subgroupMaxSize}`);
});

await check('H1b: adoption with no adapterInfo reports subgroup sizes as 0, never throws', async () => {
    installTrapNavigatorGpu();
    const { device } = mockDevice({ features: ['subgroups'] });
    // Absent, as in a browser that does not expose GPUDevice.adapterInfo yet. (Passing
    // `adapterInfo: undefined` to mockDevice would just select its default.)
    delete device.adapterInfo;
    const ctx = new WebGPUContext();
    await ctx.init(null, { device, width: 8, height: 8 });
    assert(ctx.capabilities.subgroupMinSize === 0 && ctx.capabilities.subgroupMaxSize === 0,
        'unreported subgroup sizes should read as 0');
});

await check('H2: headless — the main pass draws into the screen texture, and the view is reused', async () => {
    installTrapNavigatorGpu();
    const { device, log } = mockDevice();
    const ctx = new WebGPUContext();
    await ctx.init(null, { device, width: 64, height: 32 });
    renderMainPass(ctx);
    renderMainPass(ctx);
    assert(ctx.isHeadless === true, 'isHeadless is false without a canvas');
    assert(log.colorViews.length === 2, `expected two main passes, got ${log.colorViews.length}`);
    assert(log.colorViews[0]?.texture === ctx.screenTexture,
        'the main pass did not render into ctx.screenTexture — if this fails with "getCurrentTexture", RenderPipeline still reads ctx.context');
    assert(log.colorViews[0] === log.colorViews[1], 'a new view was created per frame — the headless frame path must not allocate');
});

await check('H3: headless screen texture — format, usage and size', async () => {
    installTrapNavigatorGpu();
    const { device } = mockDevice();
    const ctx = new WebGPUContext();
    await ctx.init(null, { device, width: 640, height: 360 });
    const t = ctx.screenTexture;
    assert(HEADLESS_SCREEN_FORMAT === 'rgba8unorm', `default headless format changed to ${HEADLESS_SCREEN_FORMAT}`);
    assert(ctx.format === 'rgba8unorm' && t.format === 'rgba8unorm', `format ${ctx.format}/${t.format}`);
    assert((t.usage & TU.RENDER_ATTACHMENT) && (t.usage & TU.COPY_SRC), 'screen texture must be RENDER_ATTACHMENT | COPY_SRC');
    assert(t.size[0] === 640 && t.size[1] === 360, `screen size ${t.size}`);
    assert(ctx.depthTexture.size[0] === 640 && ctx.depthTexture.size[1] === 360, `depth size ${ctx.depthTexture.size}`);

    const ctx2 = new WebGPUContext();
    await ctx2.init(null, { device, width: 16, height: 16, format: 'rgba16float' });
    assert(ctx2.format === 'rgba16float' && ctx2.screenTexture.format === 'rgba16float', 'options.format was ignored headless');
});

await check('H4: headless resize recreates the screen texture and its view', async () => {
    installTrapNavigatorGpu();
    const { device, log } = mockDevice();
    const ctx = new WebGPUContext();
    await ctx.init(null, { device, width: 64, height: 64 });
    const before = ctx.screenTexture;
    ctx.resize(128, 32);
    assert(ctx.screenTexture !== before, 'resize kept the old screen texture');
    assert(log.destroyed.includes(before), 'the old screen texture was not destroyed');
    assert(ctx.screenTexture.size[0] === 128 && ctx.screenTexture.size[1] === 32, `resized to ${ctx.screenTexture.size}`);
    renderMainPass(ctx);
    assert(log.colorViews.at(-1)?.texture === ctx.screenTexture, 'the main pass still draws into the pre-resize texture');
});

await check('H5: the request path is unchanged — high-performance adapter, progression features negotiated, owned', async () => {
    const { device } = mockDevice();
    const calls = installNavigatorGpu(device, ['subgroups', 'shader-f16', 'texture-compression-bc']);
    const ctx = new WebGPUContext();
    await ctx.init(null, { width: 8, height: 8 });
    assert(calls.requestAdapter.length === 1 && calls.requestAdapter[0]?.powerPreference === 'high-performance',
        `adapter request ${JSON.stringify(calls.requestAdapter)}`);
    const req = calls.requestDevice[0]?.requiredFeatures ?? [];
    assert(req.includes('subgroups') && req.includes('shader-f16'), `progression features not requested: ${req}`);
    assert(!req.includes('texture-compression-bc'), 'a non-progression feature was requested without being desired');
    assert(ctx.ownsDevice === true, 'a requested device is reported as adopted');
});

await check('H8: autoNegotiate raises the curated limits to what the adapter supports, and only those', async () => {
    // What a real adapter reports (Node/Dawn and Chrome, AMD GCN-5 and NVIDIA Turing): 16 storage
    // buffers and ~2 GB bindings, where a device gets 8 and 128 MB unless it asks.
    const adapterLimits = { maxStorageBuffersPerShaderStage: 16, maxStorageBufferBindingSize: 2147483644,
        maxBufferSize: 2147483648, maxTextureDimension2D: 16384, minUniformBufferOffsetAlignment: 64 };
    const calls = installNavigatorGpu(mockDevice().device, [], adapterLimits);
    await new WebGPUContext().init(null, { width: 8, height: 8 });
    const raised = calls.requestDevice[0]?.requiredLimits ?? {};
    const sorted = o => JSON.stringify(Object.entries(o).sort(([a], [b]) => a.localeCompare(b)));
    assert(sorted(raised) === sorted({ maxStorageBuffersPerShaderStage: 16, maxStorageBufferBindingSize: 2147483644,
        maxBufferSize: 2147483648 }), `requiredLimits ${JSON.stringify(raised)}`);
    const off = installNavigatorGpu(mockDevice().device, [], adapterLimits);
    await new WebGPUContext().init(null, { width: 8, height: 8, autoNegotiate: false });
    const none = off.requestDevice[0]?.requiredLimits ?? {};
    assert(Object.keys(none).length === 0, `autoNegotiate: false still raised ${JSON.stringify(none)}`);
});

await check('H6: a canvas still presents — getCurrentTexture is used, and headless is false', async () => {
    const { device, log } = mockDevice();
    installNavigatorGpu(device);
    let configured = null, current = 0;
    const canvasTexture = { createView: () => ({ texture: canvasTexture }) };
    const canvas = {
        width: 300, height: 150,
        getContext: (kind) => kind === 'webgpu' ? {
            configure: (c) => { configured = c; },
            getCurrentTexture: () => { current++; return canvasTexture; },
        } : null,
    };
    const ctx = new WebGPUContext();
    await ctx.init(canvas, {});
    assert(configured?.device === device && configured.format === 'bgra8unorm', `configure got ${JSON.stringify(configured?.format)}`);
    assert(ctx.isHeadless === false && ctx.screenTexture === undefined, 'a canvas init allocated a screen texture');
    assert(ctx.depthTexture.size[0] === 300 && ctx.depthTexture.size[1] === 150, 'depth no longer follows the canvas size');
    renderMainPass(ctx);
    assert(current === 1 && log.colorViews.at(-1)?.texture === canvasTexture, 'the main pass did not draw into the canvas');
});

await check('H7: NullGraph.init(null, …) wires the engine and exposes screenTexture', async () => {
    installTrapNavigatorGpu();
    const { device } = mockDevice();
    const engine = new NullGraph();
    const res = await engine.init(null, { device, width: 32, height: 32 });
    assert(res.success, `init failed: ${res.error}`);
    assert(engine.device === device && engine.canvas === null, 'engine fields not wired');
    assert(engine.screenTexture === engine.ctx.screenTexture && engine.screenTexture, 'engine.screenTexture missing');
});

await check('errors: every refusal names the option and the way out; NullGraph.init surfaces them', async () => {
    installTrapNavigatorGpu();
    const { device } = mockDevice();
    const make = (canvas, options) => () => new WebGPUContext().init(canvas, options);

    let m = await rejects(make(null, { device }), /needs options\.width and options\.height/, 'headless without size');
    assert(/canvas = null/.test(m), `the message does not say why a size is needed: ${m}`);
    await rejects(make(null, { device, width: 0, height: 4 }), /positive integers, got 0 x 4/, 'zero width');
    await rejects(make(null, { device, width: 10.5, height: 4 }), /positive integers/, 'fractional width');
    m = await rejects(make(null, { device, width: 4, height: 4, powerPreference: 'low-power' }),
        /powerPreference does nothing on an adopted device/, 'device + powerPreference');
    assert(/where the device is requested/.test(m), `no way out given: ${m}`);
    m = await rejects(make(null, { device, width: 4, height: 4, autoNegotiate: true }),
        /autoNegotiate does nothing on an adopted device/, 'device + autoNegotiate');
    assert(/engine\.capabilities/.test(m), `no way out given: ${m}`);

    removeNavigatorGpu();
    m = await rejects(make(null, { width: 4, height: 4 }), /navigator\.gpu is missing/, 'no WebGPU, no device');
    assert(/init\(…, \{ device \}\)/.test(m), `the adoption way out is not offered: ${m}`);

    // The engine wrapper reports instead of throwing — the shape every harness checks. (It also
    // console.errors the failure; captured so a passing run reads clean.)
    const logged = [];
    const error = console.error;
    console.error = (...a) => logged.push(a.map(String).join(' '));
    let res;
    try { res = await new NullGraph().init(null, { device }); }
    finally { console.error = error; }
    assert(res.success === false && /needs options\.width/.test(res.error ?? ''), `NullGraph.init returned ${JSON.stringify(res)}`);
    assert(logged.some(l => /needs options\.width/.test(l)), 'the failure was not logged');

    // A missing desired feature on an adopted device warns (it is optional), naming the feature.
    installTrapNavigatorGpu();
    const warnings = [];
    const warn = console.warn;
    console.warn = (msg) => warnings.push(String(msg));
    try { await new WebGPUContext().init(null, { device, width: 4, height: 4, desiredFeatures: ['shader-f16'] }); }
    finally { console.warn = warn; }
    assert(warnings.some(w => /"shader-f16" is not enabled on the adopted device/.test(w)), `warnings: ${JSON.stringify(warnings)}`);
});

await check('labels: compute stages and batches label their module and pipeline', async () => {
    installTrapNavigatorGpu();
    const { device, log } = mockDevice();
    const ctx = new WebGPUContext();
    await ctx.init(null, { device, width: 8, height: 8 });
    const manager = new BatchManager(ctx);
    manager.createComputeStage({ name: 'cull instances', shaderCode: '' });
    const pass = new RenderPassNode({ name: 'scene', isMainScreenPass: true });
    manager.createBatch(pass, { shaderCode: '', strideFloats: 4, maxInstances: 1 });
    manager.createBatch(pass, { shaderCode: '', strideFloats: 4, maxInstances: 1, label: 'ground' });
    const got = JSON.stringify(log.labels);
    const want = JSON.stringify([
        ['module', 'cull instances'], ['compute', 'cull instances'],
        ['module', 'scene #0'], ['render', 'scene #0'],
        ['module', 'ground'], ['render', 'ground'],
    ]);
    assert(got === want, `labels were ${got}`);
});

// ---------------------------------------------------------------------------
// Teardown (offline-films-browser-design.md §12): NullGraph.destroy() releases what the engine allocated,
// and only that. The film runner builds an engine per film on the APP's device, so a release too few
// leaks 33–66 MB a 4K film, and a release too many breaks the app's preview.

/**
 * An engine with three batches: a plain one with the caller's geometry, an indirect one whose source
 * buffer the engine allocated, and an indirect one over a SHARED source buffer and the caller's camera
 * buffer; plus a 1×1 solid texture from the texture manager.
 */
async function engineWithBatches({ adopt = true } = {}) {
    const mock = mockDevice();
    if (adopt) installTrapNavigatorGpu(); else installNavigatorGpu(mock.device);
    const engine = new NullGraph();
    const res = await engine.init(null, { ...(adopt ? { device: mock.device } : {}), width: 16, height: 16 });
    assert(res.success, `init failed: ${res.error}`);
    const pass = engine.createPass({ name: 'scene', isMainScreenPass: true });
    const shaderCode = '// the mock compiles nothing';
    const plain = engine.createBatch(pass, { shaderCode, strideFloats: 4, maxInstances: 8 });
    const owned = engine.createBatch(pass, { shaderCode, strideFloats: 4, maxInstances: 8, isIndirect: true });
    const shared = mock.device.createBuffer({ label: 'caller: shared source', size: 128, usage: 0 });
    const cameraBuffer = mock.device.createBuffer({ label: 'caller: camera', size: 432, usage: 0 });
    const viaShared = engine.createBatch(pass, { shaderCode, strideFloats: 4, maxInstances: 8, isIndirect: true, sharedSourceBuffer: shared, cameraBuffer });
    const vb = engine.bufferManager.createVertexBuffer(new Float32Array(9));
    const ib = engine.bufferManager.createIndexBuffer(new Uint16Array(3));
    engine.setBatchGeometry(plain, vb, ib, 3);
    engine.textureManager.fallbackWhite;
    const solid = mock.log.textures.find(t => t.size[0] === 1 && t.size[1] === 1);
    return { engine, mock, pass, plain, owned, viaShared, shared, cameraBuffer, vb, ib, solid,
        screen: engine.ctx.screenTexture, depth: engine.ctx.depthTexture, camera: engine.ctx.cameraUniformBuffer };
}
const releases = (list, item) => list.filter(x => x === item).length;

await check('H9: destroy() frees every buffer and texture the engine allocated, and nothing the caller owns', async () => {
    const s = await engineWithBatches();
    s.engine.destroy();
    const gone = b => s.mock.log.destroyedBuffers.includes(b);
    const mine = { 'plain storage': s.plain.storageBuffer, 'indirect storage': s.owned.storageBuffer, 'indirect args': s.owned.indirectBuffer,
        'indirect own source': s.owned.sourceStorageBuffer, 'shared-source storage': s.viaShared.storageBuffer,
        'shared-source args': s.viaShared.indirectBuffer, 'camera uniform': s.camera };
    for (const [name, buffer] of Object.entries(mine)) assert(gone(buffer), `the engine's ${name} buffer outlived destroy()`);
    assert(!gone(s.shared), 'the SHARED source buffer was destroyed with the engine — it is the caller\'s (TextSystem, VisibilitySystem)');
    assert(!gone(s.cameraBuffer), 'a camera buffer the caller passed was destroyed');
    assert(!gone(s.vb) && !gone(s.ib), 'the caller\'s geometry was destroyed');
    for (const [name, texture] of Object.entries({ screen: s.screen, depth: s.depth, 'solid 1×1': s.solid })) {
        assert(s.mock.log.destroyed.includes(texture), `the ${name} texture outlived destroy()`);
    }
    assert(!s.mock.log.deviceDestroyed, 'an ADOPTED device was destroyed — its owner still has work on it');
    assert(s.engine.isDestroyed && s.engine.ctx.isDestroyed, 'isDestroyed is false after destroy()');
});

await check('H10: destroy() is idempotent, and render() and resize() do nothing afterwards', async () => {
    const s = await engineWithBatches();
    s.engine.destroy();
    const before = { buffers: s.mock.log.destroyedBuffers.length, textures: s.mock.log.destroyed.length, made: s.mock.log.textures.length, submits: s.mock.log.submits };
    s.engine.destroy();
    s.engine.ctx.destroy(); // the context's own guard, apart from the engine's
    s.engine.render([s.pass]);
    s.engine.resize(64, 64);
    assert(s.mock.log.destroyedBuffers.length === before.buffers && s.mock.log.destroyed.length === before.textures,
        'a second destroy() released something again');
    assert(s.mock.log.submits === before.submits, 'render() submitted work after destroy()');
    assert(s.mock.log.textures.length === before.made, 'resize() made new targets after destroy() — they would outlive the engine');
});

await check('H11: an engine that requested its device destroys it', async () => {
    const s = await engineWithBatches({ adopt: false });
    assert(s.engine.ctx.ownsDevice === true, 'a requested device is reported as adopted');
    s.engine.destroy();
    assert(s.mock.log.deviceDestroyed === true, 'destroy() left the engine\'s own device alive');
});

await check('H12: destroy() after an init that failed part-way releases what was made and does not throw', async () => {
    installTrapNavigatorGpu();
    const early = new NullGraph();
    assert(!(await early.init(null, { width: 8 })).success, 'a headless init without a height succeeded');
    early.destroy();
    // The device is requested and the camera uniform made; then the canvas refuses its configuration.
    const { device, log } = mockDevice();
    installNavigatorGpu(device);
    const canvas = { width: 8, height: 8, getContext: () => ({ configure() { throw new Error('configure refused'); }, unconfigure() {} }) };
    const late = new NullGraph();
    const report = console.error;
    console.error = () => {}; // init reports its own failure; here the failure is the point
    try { assert(!(await late.init(canvas, {})).success, 'init succeeded although configure threw'); } finally { console.error = report; }
    late.destroy();
    assert(log.destroyedBuffers.length === 1 && log.destroyedBuffers[0].size === 432, 'the camera uniform made before the failure leaked');
    assert(log.deviceDestroyed === true, 'the device the failed init requested leaked');
});

await check('H13: every release is attempted, and failures are thrown together afterwards', async () => {
    const s = await engineWithBatches();
    s.depth.destroy = () => { throw new Error('depth refused'); };
    s.camera.destroy = () => { throw new Error('camera refused'); };
    let thrown = null;
    try { s.engine.destroy(); } catch (e) { thrown = e; }
    assert(thrown instanceof AggregateError && thrown.errors.map(e => e.message).join() === 'depth refused,camera refused',
        `expected one AggregateError of both failures, got ${thrown}`);
    assert(s.mock.log.destroyed.includes(s.screen), 'a failed release stopped the screen texture\'s');
    assert(s.mock.log.destroyedBuffers.includes(s.owned.sourceStorageBuffer), 'a failed release stopped the batches\'');
    assert(s.engine.isDestroyed, 'a failed release left isDestroyed false');
});

await check('H14: clearBatch frees an indirect batch\'s own source buffer, never a shared one, and destroy() does not free it twice', async () => {
    const s = await engineWithBatches();
    s.engine.clearBatch(s.pass, s.owned);
    assert(s.mock.log.destroyedBuffers.includes(s.owned.sourceStorageBuffer),
        'clearBatch leaked the source buffer createBatch allocated for an indirect batch');
    s.engine.clearBatch(s.pass, s.viaShared);
    assert(!s.mock.log.destroyedBuffers.includes(s.shared), 'clearBatch destroyed a SHARED source buffer');
    s.engine.destroy();
    for (const b of [s.owned.storageBuffer, s.owned.indirectBuffer, s.owned.sourceStorageBuffer, s.viaShared.storageBuffer]) {
        assert(releases(s.mock.log.destroyedBuffers, b) === 1, `a cleared batch's buffer was released ${releases(s.mock.log.destroyedBuffers, b)} times`);
    }
});

await check('H15: a canvas engine unconfigures its canvas and frees its depth target, and never touches the canvas texture', async () => {
    const { device, log } = mockDevice();
    installNavigatorGpu(device);
    let unconfigured = 0;
    const canvasTexture = { createView: () => ({ texture: canvasTexture }), destroy() { throw new Error('the canvas texture is the browser\'s'); } };
    const canvas = { width: 32, height: 32, getContext: () => ({ configure() {}, unconfigure() { unconfigured++; }, getCurrentTexture: () => canvasTexture }) };
    const engine = new NullGraph();
    assert((await engine.init(canvas, {})).success, 'canvas init failed');
    renderMainPass(engine.ctx);
    const depth = engine.ctx.depthTexture;
    engine.destroy();
    assert(unconfigured === 1, `the canvas was unconfigured ${unconfigured} times`);
    assert(log.destroyed.includes(depth), 'the depth target outlived destroy()');
    assert(log.deviceDestroyed === true, 'the engine\'s own device outlived destroy()');
});

console.log(failures === 0 ? '\nAll core headless tests passed.' : `\n${failures} test(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
