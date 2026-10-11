// GPU integration tests for text labels (docs/text-labels-design.md §10, X1–X9).
//
// The synthetic font's glyphs are exact rectangles with analytic fields (test/gpu/synthetic-font.mjs),
// so every expectation is a pixel rectangle or a coverage value computed on the CPU from the same
// projection and pixel snapping the shaders use.
//
// Two hosts, one file (test/gpu/harness-env.mjs): `npm run test:gpu`, or
// `npm run dev:serve` → http://localhost:3000/test/text-gpu/. X9 (Canvas-generated atlas) runs in
// the browser host only — Node has no OffscreenCanvas.

import { NullGraph, Camera } from '../../dist/index.js';
import { FontAtlas, TextSystem } from '../../dist/text.js';
import { VisibilitySystem, InstanceBuffer } from '../../dist/visibility.js';
import { MegabufferBuilder, Primitives, StandardLayout } from '../../dist/geometry.js';
import { mat4 } from 'gl-matrix';
import { syntheticFont, SHAPES } from '../gpu/synthetic-font.mjs';
import { harnessEnv } from '../gpu/harness-env.mjs';

const { log, check, fatal, finish, saveImage, inBrowser } = harnessEnv('__TEXT_RESULTS__');

const W = 400, H = 300;
const engine = new NullGraph();
const init = await engine.init(null, { width: W, height: H });
if (!init.success) {
    fatal(init.error);
    throw new Error(init.error);
}
const device = engine.device;
const atlas = FontAtlas.fromSDF(engine, syntheticFont());

const camera = new Camera(60, W / H, 0.1, 200);
const EYE = [0, 0, 10];
camera.updateView(EYE, [0, 0, 0]);
engine.updateCamera(camera);

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

function makeTargets() {
    const color = device.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const depth = device.createTexture({ size: [W, H], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    return { color, depth };
}

function scenePass(t, name) {
    return engine.createPass({
        name, isMainScreenPass: false,
        colorAttachments: [{ view: t.color.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
        depthStencilAttachment: { view: t.depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
}

async function readRGBA8(texture) {
    const bpr = Math.ceil(W * 4 / 256) * 256;
    const buf = device.createBuffer({ size: bpr * H, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture }, { buffer: buf, bytesPerRow: bpr }, [W, H]);
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange()), out = new Uint8Array(W * H * 4);
    for (let y = 0; y < H; y++) out.set(src.subarray(y * bpr, y * bpr + W * 4), y * W * 4);
    buf.unmap(); buf.destroy();
    return out;
}

const px = (img, x, y) => Array.from(img.subarray((y * W + x) * 4, (y * W + x) * 4 + 4));
const lum = (img, x, y) => { const p = px(img, x, y); return 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]; };

/** Bounding box of pixels brighter than `t` inside [x0,x1)×[y0,y1) — default the whole frame. */
function bbox(img, t = 128, x0 = 0, y0 = 0, x1 = W, y1 = H) {
    let minX = Infinity, minY = Infinity, maxX = -1, maxY = -1, n = 0;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
        if (lum(img, x, y) > t) { n++; minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y); }
    }
    return n ? { x0: minX, y0: minY, x1: maxX + 1, y1: maxY + 1, n } : null;
}

/** The shaders' anchor maths, on the CPU: clip, then the pixel-snapped screen position. */
function anchor(cam, p) {
    const v = cam.viewProjectionMatrix;
    const cx = v[0] * p[0] + v[4] * p[1] + v[8] * p[2] + v[12];
    const cy = v[1] * p[0] + v[5] * p[1] + v[9] * p[2] + v[13];
    const cz = v[2] * p[0] + v[6] * p[1] + v[10] * p[2] + v[14];
    const cw = v[3] * p[0] + v[7] * p[1] + v[11] * p[2] + v[15];
    return {
        clip: [cx, cy, cz, cw],
        px: [Math.floor((cx / cw * 0.5 + 0.5) * W + 0.5), Math.floor((-cy / cw * 0.5 + 0.5) * H + 0.5)],
    };
}

/** Expected screen rectangle of a glyph's filled shape (em box [l,b,r,t]) at pen (x, y). */
function shapeRect(a, size, pen, [l, b, r, t], off = [0, 0]) {
    return {
        x0: a[0] + off[0] + (pen[0] + l) * size, x1: a[0] + off[0] + (pen[0] + r) * size,
        y0: a[1] - (off[1] + (pen[1] + t) * size), y1: a[1] - (off[1] + (pen[1] + b) * size),
    };
}

const within = (got, want, tol = 1) => got && ['x0', 'y0', 'x1', 'y1'].every(k => Math.abs(got[k] - want[k]) <= tol);
const fmt = (r) => r ? `[${['x0', 'y0', 'x1', 'y1'].map(k => (+r[k]).toFixed(1)).join(', ')}]` : 'none';

/** Render one frame: the given label system (and optional extra batches drawn first). */
function frame(text, pass, before = []) {
    text.upload();
    engine.render([...text.getComputeStages(), ...before, pass]);
}

// ---------------------------------------------------------------------------
// X1 — a label lands at its projected anchor, at its pixel size
// ---------------------------------------------------------------------------
{
    const t = makeTargets(), pass = scenePass(t, 'X1');
    const text = new TextSystem(engine, atlas, { maxLabels: 4, maxGlyphs: 16 });
    text.createDrawBatch(pass, { targetFormat: 'rgba8unorm' });
    const S = 40, pos = [-1.5, 0.5, 0];
    text.setLabel(0, { text: 'B', position: pos, sizePx: S });
    frame(text, pass);
    const img = await readRGBA8(t.color);
    await saveImage('text-x1', img, W, H);
    const a = anchor(camera, pos).px;
    const want = shapeRect(a, S, [0, 0], SHAPES.B.outer);
    const got = bbox(img);
    check('X1: a label lands at its pixel-snapped anchor, at its size (±1 px)', within(got, want),
        `got ${fmt(got)} want ${fmt(want)} (anchor ${a}, ${S}px em, box ${S * 0.5}px)`);
}

// ---------------------------------------------------------------------------
// X2 — constant pixel size, at two very different depths
// ---------------------------------------------------------------------------
{
    const t = makeTargets(), pass = scenePass(t, 'X2');
    const text = new TextSystem(engine, atlas, { maxLabels: 4, maxGlyphs: 16 });
    text.createDrawBatch(pass, { targetFormat: 'rgba8unorm' });
    const S = 32, near = [-3, 0, 4], far = [20, 0, -80];
    text.setLabel(0, { text: 'B', position: near, sizePx: S });
    text.setLabel(1, { text: 'B', position: far, sizePx: S });
    frame(text, pass);
    const img = await readRGBA8(t.color);
    const an = anchor(camera, near).px, af = anchor(camera, far).px;
    const rn = bbox(img, 128, 0, 0, Math.floor((an[0] + af[0]) / 2), H);
    const rf = bbox(img, 128, Math.floor((an[0] + af[0]) / 2), 0, W, H);
    const size = (r) => r ? [r.x1 - r.x0, r.y1 - r.y0] : [0, 0];
    check('X2: constant pixel size — a label 6 and 90 units away measures the same (±1 px)',
        rn && rf && size(rn).every((v, k) => Math.abs(v - size(rf)[k]) <= 1 && Math.abs(v - S * 0.5) <= 1),
        `near ${size(rn)} far ${size(rf)} want ${S * 0.5}×${S * 0.5}`);
}

// ---------------------------------------------------------------------------
// X3 — the GPU cull keeps exactly what a CPU mirror of it keeps
// ---------------------------------------------------------------------------
async function readArgs(batch) {
    const dst = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const stage = engine.createComputeStage({
        name: 'args readback', dispatchCount: [1, 1, 1],
        shaderCode: `@group(0) @binding(0) var<storage, read> src: array<u32, 5>;
                     @group(0) @binding(1) var<storage, read_write> dst: array<u32, 8>;
                     @compute @workgroup_size(1) fn cs_main() { for (var i = 0u; i < 5u; i++) { dst[i] = src[i]; } }`,
    });
    engine.setComputePassBindGroup(stage, [{ binding: 0, resource: { buffer: batch.indirectBuffer } }, { binding: 1, resource: { buffer: dst } }], 0);
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
{
    const t = makeTargets(), pass = scenePass(t, 'X3');
    const L = 2000, MAXD = 60;
    const text = new TextSystem(engine, atlas, { maxLabels: L, maxGlyphs: L * 2, maxDistance: MAXD });
    const batch = text.createDrawBatch(pass, { targetFormat: 'rgba8unorm' });
    let seed = 7;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32);
    const labels = [];
    for (let i = 0; i < L; i++) {
        const pos = [(rnd() - 0.5) * 120, (rnd() - 0.5) * 80, (rnd() - 0.75) * 140];   // some behind (z > 10)
        const size = 8 + rnd() * 40, hidden = i % 13 === 0;
        text.setLabel(i, { text: 'AB', position: pos, sizePx: size, hidden });
        labels.push({ pos, size, hidden });
    }
    frame(text, pass);
    const [, gpu] = await readArgs(batch);
    // CPU mirror of buildTextCullWGSL, glyph by glyph.
    const b = atlas.glyphIndex(66), aG = atlas.glyphIndex(65);
    const planes = [aG, b].map(gi => {
        const g = syntheticFont().glyphs.find(x => x.unicode === (gi === aG ? 65 : 66));
        return [g.plane.left, g.plane.bottom, g.plane.right, g.plane.top];
    });
    const pens = [0, SHAPES.A.advance - 0.1];
    let cpu = 0, behind = 0, far = 0, off = 0, hid = 0;
    for (const l of labels) {
        if (l.hidden) { hid += 2; continue; }
        if (Math.hypot(l.pos[0] - EYE[0], l.pos[1] - EYE[1], l.pos[2] - EYE[2]) > MAXD) { far += 2; continue; }
        const { clip, px: a } = anchor(camera, l.pos);
        if (clip[3] <= 0 || clip[2] < 0 || clip[2] > clip[3]) { behind += 2; continue; }
        for (let k = 0; k < 2; k++) {
            const r = shapeRect(a, l.size, [pens[k], 0], planes[k]);
            const lo = [Math.min(r.x0, r.x1), Math.min(r.y0, r.y1)], hi = [Math.max(r.x0, r.x1), Math.max(r.y0, r.y1)];
            if (hi[0] <= 0 || hi[1] <= 0 || lo[0] >= W || lo[1] >= H) off++;
            else cpu++;
        }
    }
    check('X3: the GPU label cull keeps exactly the glyphs the CPU mirror keeps (4,000 glyphs)',
        gpu === cpu && behind > 0 && far > 0 && off > 0 && hid > 0,
        `gpu=${gpu} cpu=${cpu} — culled: behind ${behind}, beyond maxDistance ${far}, off-screen ${off}, hidden ${hid}`);
}

// ---------------------------------------------------------------------------
// X4 — labels follow positions only the GPU writes; the CPU writes nothing per frame
// ---------------------------------------------------------------------------
{
    const t = makeTargets(), pass = scenePass(t, 'X4');
    const follow = device.createBuffer({ size: 4 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(follow, 0, new Float32Array([-2, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
    const text = new TextSystem(engine, atlas, { maxLabels: 2, maxGlyphs: 4, follow: { buffer: follow, strideFloats: 4, positionOffset: 0 } });
    text.createDrawBatch(pass, { targetFormat: 'rgba8unorm' });
    text.setLabel(0, { text: 'B', follow: 0, position: [0, 0.5, 0], sizePx: 30 });
    frame(text, pass);
    const before = bbox(await readRGBA8(t.color), 128, 0, 0, W / 2, H);

    // "Simulation": a compute stage moves follow record 0 by +1.5 in x. No CPU label writes after this.
    const move = engine.createComputeStage({
        name: 'move followed', dispatchCount: [1, 1, 1],
        shaderCode: `@group(0) @binding(0) var<storage, read_write> p: array<f32>;
                     @compute @workgroup_size(1) fn cs_main() { p[0] = p[0] + 1.5; }`,
    });
    engine.setComputePassBindGroup(move, [{ binding: 0, resource: { buffer: follow } }], 0);
    let labelWrites = 0;
    const realWrite = device.queue.writeBuffer.bind(device.queue);
    device.queue.writeBuffer = (buf, ...rest) => { if (buf === text.labelBuffer || buf === text.glyphBuffer) labelWrites++; return realWrite(buf, ...rest); };
    text.upload();
    engine.render([move, ...text.getComputeStages(), pass]);
    device.queue.writeBuffer = realWrite;
    const after = bbox(await readRGBA8(t.color), 128, 0, 0, W, H);
    const a0 = anchor(camera, [-2, 0.5, 0]).px, a1 = anchor(camera, [-0.5, 0.5, 0]).px;
    check('X4: a label follows a position the GPU moved, with zero CPU writes to label data',
        before && after && labelWrites === 0 && Math.abs((after.x0 - before.x0) - (a1[0] - a0[0])) <= 1,
        `moved ${after && before ? after.x0 - before.x0 : '?'} px, expected ${a1[0] - a0[0]} px; label/glyph buffer writes: ${labelWrites}`);
}

// ---------------------------------------------------------------------------
// X4b — the documented one-liner: labels follow a VisibilitySystem's instances (stride 24, offset 12)
// ---------------------------------------------------------------------------
{
    const t = makeTargets(), pass = scenePass(t, 'X4b');
    const vis = new VisibilitySystem(engine, { maxInstances: 4 });
    const mega = new MegabufferBuilder(StandardLayout.strideFloats);
    const cube = Primitives.createCube(StandardLayout, 1, 1, 1);
    mega.addMesh('cube', cube.vertices, cube.indices);
    vis.setGeometry(mega);
    const inst = new InstanceBuffer(2);
    const m = mat4.create();
    mat4.fromTranslation(m, [-3, -1, 0]); inst.write(0, m, [0, 0, 0, 0.9], 0);
    mat4.fromTranslation(m, [2, 1, 0]); inst.write(1, m, [0, 0, 0, 0.9], 0);
    vis.setInstances(inst.data, 2);
    vis.upload();
    const text = new TextSystem(engine, atlas, { maxLabels: 2, maxGlyphs: 4, follow: { buffer: vis.instanceBuffer, strideFloats: 24, positionOffset: 12 } });
    text.createDrawBatch(pass, { targetFormat: 'rgba8unorm' });
    text.setLabel(0, { text: 'B', follow: 1, position: [0, 0.6, 0], sizePx: 30 });
    frame(text, pass);
    const got = bbox(await readRGBA8(t.color));
    const want = shapeRect(anchor(camera, [2, 1.6, 0]).px, 30, [0, 0], SHAPES.B.outer);
    check('X4b: a label follows a VisibilitySystem instance through vis.instanceBuffer (stride 24, offset 12)',
        within(got, want), `got ${fmt(got)} want ${fmt(want)} — instance #1 at (2, 1, 0), label 0.6 above`);
    vis.destroy();
}

// ---------------------------------------------------------------------------
// X5 — depth test: on hides a label behind geometry, off draws it on top
// ---------------------------------------------------------------------------
{
    // A wall: a full-screen triangle at the depth of z = 2, between the camera and a label at z = 0.
    const wallClip = anchor(camera, [0, 0, 2]).clip;
    const wallZ = wallClip[2] / wallClip[3];
    const results = [];
    for (const depthTest of [true, false]) {
        const t = makeTargets(), pass = scenePass(t, `X5 ${depthTest}`);
        const wall = engine.createBatch(pass, {
            label: 'wall', strideFloats: 4, maxInstances: 1, targetFormat: 'rgba8unorm',
            shaderCode: `@group(0) @binding(0) var<uniform> camera: array<vec4<f32>, 27>;
                @group(0) @binding(1) var<storage, read> inst: array<vec4<f32>>;
                @vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
                    return vec4<f32>(p + camera[0].x * 0.0 + inst[0].x * 0.0, ${wallZ.toFixed(7)}, 1.0);
                }
                @fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.25, 1.0); }`,
        });
        wall.currentInstanceCount = 1;
        const text = new TextSystem(engine, atlas, { maxLabels: 1, maxGlyphs: 4, depthTest });
        text.createDrawBatch(pass, { targetFormat: 'rgba8unorm' });
        text.setLabel(0, { text: 'B', position: [0, 0, 0], sizePx: 40 });
        frame(text, pass);
        results.push(bbox(await readRGBA8(t.color)));
    }
    check('X5: depthTest true hides a label behind closer geometry; false draws it on top',
        results[0] === null && results[1] && results[1].x1 - results[1].x0 === 20,
        `depthTest on → ${results[0] ? results[0].n + ' px' : 'hidden'}; off → ${results[1] ? fmt(results[1]) : 'hidden'}`);
}

// ---------------------------------------------------------------------------
// X6 — the halo rings the fill, at its width
// ---------------------------------------------------------------------------
{
    const t = makeTargets(), pass = scenePass(t, 'X6');
    const text = new TextSystem(engine, atlas, { maxLabels: 1, maxGlyphs: 4 });
    text.createDrawBatch(pass, { targetFormat: 'rgba8unorm' });
    const S = 40, pos = [0, 0, 0];
    text.setLabel(0, { text: 'B', position: pos, sizePx: S, color: [1, 1, 1, 1], halo: { color: [1, 0, 0, 1], width: 3 } });
    frame(text, pass);
    const img = await readRGBA8(t.color);
    await saveImage('text-x6', img, W, H);
    const r = shapeRect(anchor(camera, pos).px, S, [0, 0], SHAPES.B.outer);
    const cy = Math.floor((r.y0 + r.y1) / 2);
    const inside = px(img, Math.floor(r.x0) + 4, cy), ring = px(img, Math.floor(r.x0) - 2, cy), clear = px(img, Math.floor(r.x0) - 7, cy);
    check('X6: a 3 px halo rings the fill: white inside, red 2 px outside, black 7 px outside',
        inside[0] > 240 && inside[1] > 240 && ring[0] > 240 && ring[1] < 15 && clear[0] < 15,
        `inside ${inside.slice(0, 3)} ring ${ring.slice(0, 3)} beyond ${clear.slice(0, 3)}`);
}

// ---------------------------------------------------------------------------
// X7 — antialiasing: an edge at a fractional pixel gets exactly its coverage
// ---------------------------------------------------------------------------
{
    const t = makeTargets(), pass = scenePass(t, 'X7');
    const text = new TextSystem(engine, atlas, { maxLabels: 1, maxGlyphs: 4 });
    text.createDrawBatch(pass, { targetFormat: 'rgba8unorm' });
    const S = 37, pos = [0, 0, 0];                  // A's left edge at 0.1 em = 3.7 px from the anchor
    text.setLabel(0, { text: 'A', position: pos, sizePx: S });
    frame(text, pass);
    const img = await readRGBA8(t.color);
    const a = anchor(camera, pos).px;
    const edge = a[0] + 0.1 * S;                     // 3.7 px right of a pixel corner
    const y = a[1] - Math.round(0.35 * S);
    const straddle = Math.floor(edge);               // pixel whose centre is 0.2 px outside the edge
    const coverage = (straddle + 1 - edge);          // fraction of that pixel inside the box: 0.3
    const got = px(img, straddle, y)[0] / 255;
    const left = px(img, straddle - 1, y)[0], right = px(img, straddle + 1, y)[0];
    check('X7: an edge 0.3 px into a pixel gives that pixel 30% coverage — the field\'s pixel range is right',
        Math.abs(got - coverage) < 0.08 && left < 8 && right > 247,
        `coverage ${got.toFixed(3)} want ${coverage.toFixed(3)}; neighbours ${left} | ${right}`);
}

// ---------------------------------------------------------------------------
// X8 — 600,000 glyphs, one draw call
// ---------------------------------------------------------------------------
{
    const t = makeTargets(), pass = scenePass(t, 'X8');
    const LABELS = 100000;
    const text = new TextSystem(engine, atlas, { maxLabels: LABELS, maxGlyphs: LABELS * 6 });
    text.createDrawBatch(pass, { targetFormat: 'rgba8unorm' });
    const t0 = performance.now();
    for (let i = 0; i < LABELS; i++) {
        text.setLabel(i, { text: 'AB AB AB', position: [(i % 400 - 200) * 0.6, (Math.floor(i / 400) % 250 - 125) * 0.4, -20 - (i % 7)], sizePx: 6 });
    }
    const layoutMs = performance.now() - t0;
    let draws = 0;
    const real = device.createCommandEncoder.bind(device);
    device.createCommandEncoder = (d) => {
        const enc = real(d);
        const begin = enc.beginRenderPass.bind(enc);
        enc.beginRenderPass = (pd) => {
            const p = begin(pd);
            for (const fn of ['draw', 'drawIndexed', 'drawIndexedIndirect']) {
                const f = p[fn].bind(p);
                p[fn] = (...a) => { draws++; return f(...a); };
            }
            return p;
        };
        return enc;
    };
    frame(text, pass);
    device.createCommandEncoder = real;
    await device.queue.onSubmittedWorkDone();
    const g0 = performance.now();
    frame(text, pass);
    await device.queue.onSubmittedWorkDone();
    const frameMs = performance.now() - g0;
    check('X8: 100,000 labels (600,000 glyphs) draw in exactly one draw call',
        text.glyphCount === LABELS * 6 && draws === 1,
        `glyphs ${text.glyphCount}, draws ${draws}; layout ${(layoutMs / 1000).toFixed(2)} s once, frame ${frameMs.toFixed(1)} ms (cull + draw, submit to idle)`);
    text.destroy();
}

// ---------------------------------------------------------------------------
// X9 — a Canvas-generated atlas renders a real "O" with a hole (browser host only)
// ---------------------------------------------------------------------------
if (inBrowser) {
    const t = makeTargets(), pass = scenePass(t, 'X9');
    const real = FontAtlas.fromCanvas(engine, { fontFamily: 'sans-serif', chars: 'O' });
    const text = new TextSystem(engine, real, { maxLabels: 1, maxGlyphs: 2 });
    text.createDrawBatch(pass, { targetFormat: 'rgba8unorm' });
    text.setLabel(0, { text: 'O', position: [0, 0, 0], sizePx: 120, align: 'center', baseline: 'middle' });
    frame(text, pass);
    const img = await readRGBA8(t.color);
    const r = bbox(img);
    const cx = r ? Math.floor((r.x0 + r.x1) / 2) : 0, cy = r ? Math.floor((r.y0 + r.y1) / 2) : 0;
    check('X9: a Canvas-generated SDF atlas renders "O" — a ring with a hole',
        r && r.x1 - r.x0 > 40 && lum(img, cx, cy) < 64 && lum(img, r.x0 + 2, cy) > 128 && lum(img, r.x1 - 3, cy) > 128,
        r ? `bbox ${fmt(r)} centre ${lum(img, cx, cy).toFixed(0)} left ${lum(img, r.x0 + 2, cy).toFixed(0)} right ${lum(img, r.x1 - 3, cy).toFixed(0)}` : 'nothing drawn');
} else {
    log('  (X9 skipped: Canvas atlas generation needs OffscreenCanvas — browser host only)');
}

await finish('All text GPU tests passed.');
