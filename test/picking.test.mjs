// Headless picking tests (docs/picking-design.md §10). Pure maths, hit selection, the generated
// WGSL, the visibility touchpoint, and the whole Picker flow on a mock device ADOPTED through
// init(null, { device }) — real NullGraph / VisibilitySystem / Picker objects, faked readback.

globalThis.GPUBufferUsage ??= {
    MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16,
    VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512,
};
globalThis.GPUTextureUsage ??= { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 };
globalThis.GPUMapMode ??= { READ: 1, WRITE: 2 };

const { mat4 } = await import('gl-matrix');
const { NullGraph, Camera } = await import('../dist/index.js');
const { VisibilitySystem, InstanceBuffer, buildInstanceCullWGSL, INSTANCE_SOURCE_INDEX_OFFSET } = await import('../dist/visibility.js');
const { MegabufferBuilder, Primitives, StandardLayout } = await import('../dist/geometry.js');
const P = await import('../dist/picking.js');

const log = console.log;
console.log = (...a) => { if (!/^NullGraph: (Using GPU|Adopted)/.test(String(a[0]))) log(...a); };

let failures = 0;
async function check(name, fn) {
    try { await fn(); console.log(`  ok    ${name}`); }
    catch (e) { failures++; console.log(`  FAIL  ${name}\n        ${e.message}`); }
}
function assert(v, m) { if (!v) throw new Error(m); }
function near(a, b, eps, m) { if (!(Math.abs(a - b) <= eps)) throw new Error(`${m}: expected ~${b}, got ${a}`); }
function throws(fn, pattern, label) {
    let msg = '';
    try { fn(); } catch (e) { msg = e.message; }
    if (!pattern.test(msg)) throw new Error(`${label}: expected ${pattern}, got "${msg}"`);
    return msg;
}

/** Transform p (xyz) by a column-major 4x4 → [x, y, z] after the divide. */
function project(m, p) {
    const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
    const y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
    const z = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
    const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
    return [x / w, y / w, z / w];
}

const W = 640, H = 360;
const camera = new Camera(60, W / H, 0.1, 500);
camera.updateView([3, 4, 12], [0, 0, 0]);

// ---------------------------------------------------------------------------
// Pure maths
// ---------------------------------------------------------------------------

await check('M1: the clicked pixel maps to NDC (0, 0) under the pick camera, depth untouched', () => {
    const R = 4, px = 417, py = 133;
    const pick = P.buildPickCamera(camera.bufferData, px, py, R, W, H, new Float32Array(108));
    const world = P.unprojectPixel(camera.bufferData.subarray(48, 64), px, py, 0.73, W, H);
    const main = project(camera.bufferData.subarray(0, 16), world);
    const zoomed = project(pick.subarray(0, 16), world);
    near(zoomed[0], 0, 1e-3, 'x'); near(zoomed[1], 0, 1e-3, 'y');
    near(zoomed[2], main[2], 1e-6, 'depth must be the main view\'s');
    near(main[2], 0.73, 1e-4, 'round trip through the main camera');
});

await check('M2: the pick region is exactly (2R+1) pixels — R inside, R+1 outside', () => {
    const R = 3, px = 200, py = 100;
    const pick = P.buildPickCamera(camera.bufferData, px, py, R, W, H, new Float32Array(108));
    const at = (dx, dy) => project(pick.subarray(0, 16),
        P.unprojectPixel(camera.bufferData.subarray(48, 64), px + dx, py + dy, 0.5, W, H));
    for (const [dx, dy] of [[R, 0], [-R, 0], [0, R], [0, -R]]) {
        const [x, y] = at(dx, dy);
        assert(Math.abs(x) < 1 && Math.abs(y) < 1, `pixel offset (${dx}, ${dy}) should be inside: ${x}, ${y}`);
    }
    for (const [dx, dy] of [[R + 1, 0], [-(R + 1), 0], [0, R + 1], [0, -(R + 1)]]) {
        const [x, y] = at(dx, dy);
        assert(Math.abs(x) > 1 || Math.abs(y) > 1, `pixel offset (${dx}, ${dy}) should be outside: ${x}, ${y}`);
    }
    // Screen +y is down, NDC +y is up: one pixel DOWN must land at negative NDC y.
    assert(at(0, 1)[1] < 0 && at(1, 0)[0] > 0, 'axis orientation inverted');
});

await check('M3: the pick camera\'s inverses invert it; view, eye and scalars are the main camera\'s', () => {
    const pick = P.buildPickCamera(camera.bufferData, 10, 20, 2, W, H, new Float32Array(108));
    const id = P.mat4Multiply(pick.subarray(48, 64), pick.subarray(0, 16), new Float32Array(16));
    const idp = P.mat4Multiply(pick.subarray(80, 96), pick.subarray(32, 48), new Float32Array(16));
    for (let i = 0; i < 16; i++) {
        near(id[i], i % 5 === 0 ? 1 : 0, 1e-4, `invViewProj·viewProj[${i}]`);
        near(idp[i], i % 5 === 0 ? 1 : 0, 1e-4, `invProj·proj[${i}]`);
    }
    for (const i of [16, 20, 31, 64, 79, 96, 97, 98, 100, 101, 102, 103]) {
        assert(pick[i] === camera.bufferData[i], `float ${i} changed`);
    }
});

/** A read-back target with the given hits: { tx, ty, tag, index, depth, extra }. */
function texelGrid(R, hits) {
    const n = 2 * R + 1, stride = Math.ceil(n * 16 / 256) * 64;   // u32 per row
    const t = new Uint32Array(stride * n), f = new Float32Array(1), u = new Uint32Array(f.buffer);
    for (const h of hits) {
        const o = h.ty * stride + h.tx * 4;
        f[0] = h.depth ?? 0.5;
        t[o] = h.tag; t[o + 1] = h.index; t[o + 2] = u[0]; t[o + 3] = h.extra ?? 0;
    }
    return { texels: t, stride };
}

await check('S1: hit selection — centre, nearest within radius, ties by depth, screen clipping', () => {
    const R = 4;
    let g = texelGrid(R, [{ tx: 4, ty: 4, tag: 1, index: 7 }, { tx: 5, ty: 4, tag: 2, index: 9 }]);
    let h = P.selectHit(g.texels, g.stride, R, 0, 100, 100, W, H);
    assert(h && h.tag === 1 && h.index === 7 && h.px === 100 && h.py === 100, `centre hit: ${JSON.stringify(h)}`);

    g = texelGrid(R, [{ tx: 7, ty: 4, tag: 1, index: 1 }, { tx: 4, ty: 6, tag: 2, index: 2 }]);
    assert(P.selectHit(g.texels, g.stride, R, 1, 100, 100, W, H) === null, 'radius 1 must not reach distance 2');
    h = P.selectHit(g.texels, g.stride, R, 3, 100, 100, W, H);
    assert(h && h.index === 2 && h.px === 100 && h.py === 102, `nearest (distance 2 beats 3): ${JSON.stringify(h)}`);

    g = texelGrid(R, [{ tx: 5, ty: 4, tag: 1, index: 1, depth: 0.9 }, { tx: 3, ty: 4, tag: 1, index: 2, depth: 0.2 }]);
    h = P.selectHit(g.texels, g.stride, R, 2, 100, 100, W, H);
    assert(h && h.index === 2, `equal distance → smaller depth: ${JSON.stringify(h)}`);

    // A texel left of the screen edge (px 1, tx 0 → screen x -3) is ignored even if nearer.
    g = texelGrid(R, [{ tx: 2, ty: 4, tag: 1, index: 1 }, { tx: 7, ty: 4, tag: 1, index: 2 }]);
    h = P.selectHit(g.texels, g.stride, R, 4, 1, 100, W, H);
    assert(h && h.index === 2, `off-screen texel must be skipped: ${JSON.stringify(h)}`);
    assert(P.selectHit(texelGrid(R, []).texels, g.stride, R, 4, 100, 100, W, H) === null, 'empty target');
});

// ---------------------------------------------------------------------------
// Generated WGSL + the visibility touchpoint
// ---------------------------------------------------------------------------

function balanced(code, label) {
    const pairs = { '{': '}', '(': ')', '[': ']' }, stack = [];
    for (const ch of code.replace(/\/\/[^\n]*/g, '')) {
        if (pairs[ch]) stack.push(pairs[ch]);
        else if ('})]'.includes(ch) && stack.pop() !== ch) throw new Error(`${label}: unbalanced '${ch}'`);
    }
    if (stack.length) throw new Error(`${label}: ${stack.length} unclosed`);
}

await check('W1: pick shaders — entry points, flat integer varyings, baked tag, every binding used', () => {
    for (const [label, code] of [['instance', P.buildInstancePickWGSL(3)], ['standard', P.buildStandardPickWGSL(5)]]) {
        balanced(code, label);
        assert(/fn vs_main\(/.test(code) && /fn fs_main\(/.test(code), `${label}: entry points`);
        assert((code.match(/@interpolate\(flat\)/g) || []).length === 2, `${label}: integer varyings must be flat`);
        assert(/-> @location\(0\) vec4<u32>/.test(code), `${label}: must write rgba32uint`);
        assert(/var<storage, read> (instances|ecs)/.test(code) && /camera\.viewProj/.test(code), `${label}: group 0 unused`);
        assert(!code.includes('`'), `${label}: a backtick inside a WGSL template ends the string (design-principles §13)`);
    }
    assert(/vec4<u32>\(3u,/.test(P.buildInstancePickWGSL(3)), 'tag 3 not baked');
    assert(/inst\.sourceIndex/.test(P.buildInstancePickWGSL(1)), 'instance pick must return the SOURCE index');
    throws(() => P.buildPickFragmentWGSL(0), /positive integer/, 'tag 0');
});

await check('W2: the instance cull writes each survivor\'s source index (picking-design §5)', () => {
    const cull = buildInstanceCullWGSL({});
    assert(/survivor\.sourceIndex = i;/.test(cull) && /outInstances\[slot\] = survivor;/.test(cull),
        'compacted copies would lose their source index — pick and per-instance lookups return garbage');
    assert(INSTANCE_SOURCE_INDEX_OFFSET === 23, `offset ${INSTANCE_SOURCE_INDEX_OFFSET}`);
});

// ---------------------------------------------------------------------------
// The Picker, on an adopted mock device
// ---------------------------------------------------------------------------

function mockDevice() {
    const state = { renders: 0, copies: 0, writes: [], readback: null, lastPassBatches: 0 };
    const res = (extra = {}) => ({ ...extra, destroy() { this.destroyed = true; } });
    const pipeline = () => ({ getBindGroupLayout: () => ({}) });
    const device = {
        features: new Set(), adapterInfo: { vendor: 'mock' },
        createBuffer: (d) => {
            const buf = res({ ...d, getMappedRange: () => (d.usage & 1) && state.readback ? state.readback.buffer : new ArrayBuffer(d.size), unmap() {} });
            buf.mapAsync = async () => { };
            return buf;
        },
        createTexture: (d) => res({ ...d, createView: () => ({}) }),
        createSampler: () => ({}), createShaderModule: () => ({}),
        createRenderPipeline: pipeline, createComputePipeline: pipeline,
        createBindGroup: (d) => ({ entries: d.entries }),
        createCommandEncoder: () => ({
            beginComputePass: () => ({ setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }),
            beginRenderPass: () => { state.renders++; return { setPipeline() {}, setBindGroup() {}, setViewport() {}, setScissorRect() {}, setVertexBuffer() {}, setIndexBuffer() {}, drawIndexed() {}, draw() {}, drawIndexedIndirect() {}, end() {} }; },
            copyTextureToBuffer: () => { state.copies++; },
            finish: () => ({}),
        }),
        queue: { writeBuffer: (b, o, d) => state.writes.push([b, o, d]), submit() {} },
    };
    return { device, state };
}

async function rig() {
    const m = mockDevice();
    const engine = new NullGraph();
    const r = await engine.init(null, { device: m.device, width: W, height: H });
    if (!r.success) throw new Error(r.error);
    const mega = new MegabufferBuilder(StandardLayout.strideFloats);
    const cube = Primitives.createCube(StandardLayout, 1, 1, 1);
    mega.addMesh('cube', cube.vertices, cube.indices);
    const vis = new VisibilitySystem(engine, { maxInstances: 8 });
    vis.setGeometry(mega);
    const inst = new InstanceBuffer(2);
    inst.write(0, mat4.create(), [0, 0, 0, 0.9], 0);
    inst.write(1, mat4.create(), [0, 0, 0, 0.9], 0);
    vis.setInstances(inst.data, 2);
    return { ...m, engine, vis };
}

await check('K1: addInstances — a pick batch on the system\'s own cull, disabled, zero instances', async () => {
    const { engine, vis, state } = await rig();
    const scene = engine.createPass({ name: 'scene', isMainScreenPass: true });
    vis.createInstanceDrawBatch(scene, 'cube', 0);
    const picker = new P.Picker(engine);
    const src = picker.addInstances(vis, 'cube', 0, { label: 'cubes' });
    assert(src.tag === 1 && src.kind === 'instances' && src.label === 'cubes', JSON.stringify(src));
    const chain = vis.getComputeStages();
    assert(chain.length === 2, `the pick cull must ride the system's chain (got ${chain.length} stages)`);
    assert(chain[1].enabled === false, 'the pick cull is enabled outside a pick — the app frame would pay for it');
    assert(state.writes.some(([, o, d]) => o === 4 && d instanceof Uint32Array && d[0] === 0),
        'indirect instanceCount was not zeroed — engine.render() with no sequence would draw the pick batch');
});

await check('K2: pick() — renders the pick pass, reads back, resolves source + index + world position', async () => {
    const { engine, vis, state } = await rig();
    const picker = new P.Picker(engine, { maxRadius: 2 });
    const src = picker.addInstances(vis, 'cube', 0);
    const R = 2, depth = 0.6;
    state.readback = texelGrid(R, [{ tx: 2, ty: 2, tag: 1, index: 1, depth, extra: 0 }]).texels;
    const hit = await picker.pick(300, 150, camera);
    assert(state.renders === 1 && state.copies === 1, `renders=${state.renders} copies=${state.copies}`);
    assert(hit && hit.source === src && hit.index === 1 && hit.pixel[0] === 300 && hit.pixel[1] === 150, JSON.stringify(hit));
    const want = P.unprojectPixel(camera.bufferData.subarray(48, 64), 300, 150, depth, W, H);
    for (let i = 0; i < 3; i++) near(hit.position[i], want[i], 1e-4, `position[${i}]`);
    assert(vis.getComputeStages().every((s, i) => i === 0 || !s.enabled), 'pick cull left enabled after the pick');
    state.readback = texelGrid(R, []).texels;
    assert(await picker.pick(10, 10, camera) === null, 'an empty target must resolve null');
});

await check('K3: addBatch — shares the source\'s storage, mirrors count + geometry at pick time, idles at 0', async () => {
    const { engine, device, state } = await rig();
    const scene = engine.createPass({ name: 'scene', isMainScreenPass: true });
    const source = engine.createBatch(scene, { shaderCode: '', strideFloats: 14, maxInstances: 4 });
    const vb = device.createBuffer({ size: 64, usage: 32 }), ib = device.createBuffer({ size: 12, usage: 16 });
    engine.setBatchGeometry(source, vb, ib, 6);
    source.currentInstanceCount = 3;
    const picker = new P.Picker(engine, { maxRadius: 0 });
    picker.addBatch(source, { label: 'quick-start cube' });
    const pickBatch = picker['sources'][0].batch;
    assert(pickBatch.bindGroup.entries[1].resource.buffer === source.storageBuffer,
        'group 0 must bind the SOURCE storage buffer (no second upload)');
    assert(pickBatch.currentInstanceCount === 0, 'a pick batch must idle at 0 instances');
    state.readback = texelGrid(0, [{ tx: 0, ty: 0, tag: 1, index: 2 }]).texels;
    let seen = -1;
    const realRender = engine.render.bind(engine);
    engine.render = (seq) => { seen = pickBatch.currentInstanceCount; assert(pickBatch.indexBuffer === ib && pickBatch.indexCount === 6, 'geometry not mirrored'); realRender(seq); };
    const hit = await picker.pick(5, 5, camera);
    assert(seen === 3, `instance count not mirrored at pick time (saw ${seen})`);
    assert(pickBatch.currentInstanceCount === 0, 'not reset after the pick');
    assert(hit && hit.index === 2 && hit.meshIndex === 0, JSON.stringify(hit));
});

await check('K4: concurrent picks are serialized and each resolves its own result', async () => {
    const { engine, vis, state } = await rig();
    const picker = new P.Picker(engine, { maxRadius: 0 });
    picker.addInstances(vis, 'cube', 0);
    const grids = [texelGrid(0, [{ tx: 0, ty: 0, tag: 1, index: 0 }]), texelGrid(0, []), texelGrid(0, [{ tx: 0, ty: 0, tag: 1, index: 1 }])];
    let k = 0;
    const realRender = engine.render.bind(engine);
    engine.render = (seq) => { state.readback = grids[k++].texels; realRender(seq); };
    const [a, b, c] = await Promise.all([picker.pick(1, 1, camera), picker.pick(2, 2, camera), picker.pick(3, 3, camera)]);
    assert(a?.index === 0 && b === null && c?.index === 1, `results crossed: ${JSON.stringify([a?.index, b, c?.index])}`);
});

// Found by the GPU agents demo (phase 4.3): it destroys and rebuilds its picker while hover picks
// are queued, and a run that started after destroy() wrote into the released buffers.
await check('K5: picks still queued when destroy() runs resolve null and touch nothing released', async () => {
    const { engine, vis, state } = await rig();
    const picker = new P.Picker(engine, { maxRadius: 0 });
    picker.addInstances(vis, 'cube', 0);
    state.readback = texelGrid(0, [{ tx: 0, ty: 0, tag: 1, index: 1 }]).texels;
    const renders = state.renders, writes = state.writes.length;
    const queued = [picker.pick(1, 1, camera), picker.pick(2, 2, camera)];
    picker.destroy();
    const results = await Promise.all(queued);
    assert(results.every(r => r === null), `queued picks resolved ${JSON.stringify(results.map(r => r?.index ?? r))} after destroy`);
    assert(state.renders === renders, `${state.renders - renders} pick pass(es) rendered after destroy`);
    const late = state.writes.slice(writes).filter(([b]) => b.destroyed);
    assert(late.length === 0, `${late.length} write(s) into released buffers after destroy`);
});

await check('errors: every refusal names the offender and the way out', async () => {
    const { engine, vis } = await rig();
    const picker = new P.Picker(engine, { maxRadius: 2 });
    throws(() => picker.pick(1, 1, camera), /nothing is pickable/, 'no sources');
    picker.addInstances(vis, 'cube', 0);
    let m = throws(() => picker.pick(W, 1, camera), /outside the 640×360 target/, 'x = width');
    assert(/devicePixelRatio/.test(m), `no way out: ${m}`);
    throws(() => picker.pick(-1, 1, camera), /outside/, 'negative x');
    throws(() => picker.pick(1, 1, camera, { radius: 3 }), /radius 3 exceeds maxRadius 2/, 'radius');
    throws(() => picker.pick(1, 1, { bufferData: new Float32Array(16) }), /has 16 floats/, 'not a camera');
    throws(() => new P.Picker(engine, { maxRadius: 8 }), /maxRadius 8/, 'maxRadius');
    const other = await rig();
    throws(() => picker.addInstances(other.vis, 'cube', 0), /different engine/, 'foreign system');
    const scene = engine.createPass({ name: 'scene', isMainScreenPass: true });
    const indirect = engine.createBatch(scene, { shaderCode: '', strideFloats: 24, maxInstances: 1, isIndirect: true });
    m = throws(() => picker.addBatch(indirect, { label: 'culled rocks' }), /"culled rocks" is indirect/, 'indirect batch');
    assert(/addInstances/.test(m), `no way out: ${m}`);
    picker.destroy();
    throws(() => picker.pick(1, 1, camera), /after destroy/, 'use after destroy');
    assert(vis['instanceTierCulls'].length === 0,
        'destroy() left its pick batch registered on the VisibilitySystem (removeDrawBatch not called)');
});

console.log(failures === 0 ? '\nAll picking tests passed.' : `\n${failures} test(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
