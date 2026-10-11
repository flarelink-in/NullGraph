// Headless text tests (docs/text-labels-design.md §10): layout, atlas parsing, the distance
// transform against brute force, packing, the generated WGSL, and TextSystem wiring on a mock
// device adopted through init(null, { device }).

globalThis.GPUBufferUsage ??= {
    MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16,
    VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512,
};
globalThis.GPUTextureUsage ??= { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 };

const { NullGraph } = await import('../dist/index.js');
const T = await import('../dist/text.js');
const { syntheticFont, SHAPES, KERN_AB, SPACE_ADVANCE, METRICS } = await import('./gpu/synthetic-font.mjs');

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

/** A LayoutFont over the synthetic font's description — no device needed for layout. */
function fakeFont(desc) {
    const t = T.buildGlyphTable(desc);
    return {
        glyphIndex: (cp) => t.index.get(cp) ?? -1,
        advance: (g) => t.advances[g],
        kerning: (a, b) => t.kerning.get(a * 0x110000 + b) ?? 0,
        hasShape: (g) => t.shaped[g] === 1,
        ...METRICS,
    };
}
const font = fakeFont(syntheticFont());
const g = (ch) => font.glyphIndex(ch.codePointAt(0));

function lay(text, opts = {}, cap = 64) {
    const buf = new ArrayBuffer(cap * 16);
    const u32 = new Uint32Array(buf), f32 = new Float32Array(buf);
    const res = T.layoutText(font, text, 7, opts, u32, f32, 0, cap);
    const recs = [];
    for (let k = 0; k < res.glyphs; k++) recs.push({ label: u32[k * 4], glyph: u32[k * 4 + 1], x: f32[k * 4 + 2], y: f32[k * 4 + 3] });
    return { res, recs };
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

await check('L1: advances, kerning and letter spacing place each glyph', () => {
    let { recs } = lay('AB');
    assert(recs.length === 2 && recs[0].label === 7 && recs[0].glyph === g('A') && recs[1].glyph === g('B'), JSON.stringify(recs));
    near(recs[0].x, 0, 1e-6, 'A'); near(recs[1].x, SHAPES.A.advance + KERN_AB, 1e-6, 'B after kerning');
    ({ recs } = lay('AB', { letterSpacing: 0.05 }));
    near(recs[1].x, SHAPES.A.advance + 0.05 + KERN_AB, 1e-6, 'B with letter spacing');
    ({ recs } = lay('A B'));
    near(recs[1].x, SHAPES.A.advance + SPACE_ADVANCE, 1e-6, 'a space advances the pen, emits no record');
    assert(recs.length === 2, 'the space must not emit a record');
});

await check('L2: alignment shifts each line about the anchor by its width', () => {
    const w = SHAPES.A.advance + KERN_AB + SHAPES.B.advance;    // "AB"
    const c = lay('AB', { align: 'center' }), r = lay('AB', { align: 'right' });
    near(c.res.width, w, 1e-6, 'width');
    near(c.recs[0].x, -w / 2, 1e-6, 'center'); near(r.recs[0].x, -w, 1e-6, 'right');
});

await check('L3: baselines — alphabetic, top, bottom, middle, over two lines', () => {
    const { ascender: a, descender: d, lineHeight: lh } = METRICS;
    const y = (baseline, line) => lay('A\nA', { baseline }).recs[line].y;
    near(y('alphabetic', 0), 0, 1e-6, 'alphabetic'); near(y('alphabetic', 1), -lh, 1e-6, 'second line');
    near(y('top', 0), -a, 1e-6, 'top');
    near(y('bottom', 1), -d, 1e-6, 'bottom: the last line\'s descender sits on the anchor');
    near((y('middle', 0) + a + y('middle', 1) + d) / 2, 0, 1e-6, 'middle: the block is centred');
    near(lay('A\nA').res.height, a - d + lh, 1e-6, 'height');
});

await check('L4: greedy word wrap at spaces, and a lone long word is not split', () => {
    // "AB" is 1.2 em, "AB AB" 2.7 em: at 2.8 em two words fit and the third wraps.
    const { res, recs } = lay('AB AB AB', { maxWidth: 2.8 });
    assert(res.lines === 2, `expected 2 lines, got ${res.lines}`);
    assert(lay('AB AB AB', { maxWidth: 2.6 }).res.lines === 3, 'at 2.6 em only one word fits per line');
    assert(recs[4].x === 0 && recs[4].y < 0, `the third word should start line 2 at x 0: ${JSON.stringify(recs[4])}`);
    assert(lay('ABABABAB', { maxWidth: 1 }).res.lines === 1, 'a single word wider than maxWidth stays on one line');
});

await check('L5: missing glyphs fall back to "?" and are counted; without "?" they are skipped', () => {
    const { res, recs } = lay('AZB');
    assert(res.missing === 1 && recs.length === 3 && recs[1].glyph === g('?'), JSON.stringify({ res, recs }));
    const bare = fakeFont(syntheticFont({ omitFallback: true }));
    const buf = new ArrayBuffer(64 * 16);
    const r2 = T.layoutText(bare, 'AZB', 0, {}, new Uint32Array(buf), new Float32Array(buf), 0, 64);
    assert(r2.missing === 1 && r2.glyphs === 2, JSON.stringify(r2));
});

await check('L6: running past capacity throws, naming the text and the numbers, and writes nothing', () => {
    const buf = new ArrayBuffer(4 * 16), u32 = new Uint32Array(buf);
    u32.fill(0xABCD);
    const m = throws(() => T.layoutText(font, 'ABABA', 0, {}, u32, new Float32Array(buf), 1, 4), /needs 5 glyph records at offset 1, but capacity is 4/, 'capacity');
    assert(/"ABABA"/.test(m) && /maxGlyphs/.test(m), `message: ${m}`);
    assert(u32.every(v => v === 0xABCD), 'records were written before the capacity check');
});

// ---------------------------------------------------------------------------
// Atlas parsing
// ---------------------------------------------------------------------------

const msdfJson = {
    atlas: { type: 'msdf', distanceRange: 4, size: 40, width: 200, height: 100, yOrigin: 'bottom' },
    metrics: { emSize: 2, lineHeight: 2.4, ascender: 1.8, descender: -0.4 },
    glyphs: [
        { unicode: 32, advance: 0.6 },
        { unicode: 72, advance: 1.2, planeBounds: { left: 0.1, bottom: -0.1, right: 1.1, top: 1.5 }, atlasBounds: { left: 10, bottom: 20, right: 50, top: 84 } },
    ],
    kerning: [{ unicode1: 72, unicode2: 72, advance: -0.2 }],
};

await check('A1: msdf-atlas-gen JSON → em metrics, flipped UVs, kerning (y origin bottom, emSize 2)', () => {
    const d = T.parseMSDFJson(msdfJson);
    assert(d.type === 'msdf' && d.emSize === 40 && d.distanceRange === 4, JSON.stringify(d));
    near(d.lineHeight, 1.2, 1e-9, 'lineHeight normalised by emSize');
    near(d.glyphs[1].advance, 0.6, 1e-9, 'advance normalised');
    const t = T.buildGlyphTable(d);
    const i = t.index.get(72), o = i * T.ATLAS_GLYPH_FLOATS;
    near(t.table[o], 10 / 200, 1e-7, 'u at left'); near(t.table[o + 2], 50 / 200, 1e-7, 'u at right');
    near(t.table[o + 1], 1 - 20 / 100, 1e-7, 'v at plane-bottom (flipped from y-up)');
    near(t.table[o + 3], 1 - 84 / 100, 1e-7, 'v at plane-top');
    near(t.table[o + 7], 0.75, 1e-7, 'plane top normalised');
    near(t.kerning.get(72 * 0x110000 + 72), -0.1, 1e-9, 'kerning normalised');
    assert(t.shaped[i] === 1 && t.shaped[t.index.get(32)] === 0, 'whitespace must not be shaped');
});

await check('A2: refusals — not a distance field, not msdf-atlas-gen output', () => {
    throws(() => T.parseMSDFJson({ ...msdfJson, atlas: { ...msdfJson.atlas, type: 'softmask' } }), /"softmask" is not a distance field/, 'mask type');
    throws(() => T.parseMSDFJson({ glyphs: [] }), /no "atlas" block/, 'missing atlas');
    assert(T.parseMSDFJson({ ...msdfJson, atlas: { ...msdfJson.atlas, type: 'psdf' } }).type === 'sdf', 'psdf reads as one-channel sdf');
});

// ---------------------------------------------------------------------------
// Distance transform
// ---------------------------------------------------------------------------

/** The definition the EDT must match: min over pixels of squared distance + seed, both grids. */
function bruteSDF(cov, w, h) {
    const out = new Float32Array(w * h);
    const seed = (a, kind) => a >= 1 ? (kind === 'outer' ? 0 : Infinity) : a <= 0 ? (kind === 'outer' ? Infinity : 0)
        : Math.max(0, kind === 'outer' ? 0.5 - a : a - 0.5) ** 2;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        let o = Infinity, n = Infinity;
        for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
            const d2 = (x - i) ** 2 + (y - j) ** 2, a = cov[j * w + i];
            o = Math.min(o, d2 + seed(a, 'outer')); n = Math.min(n, d2 + seed(a, 'inner'));
        }
        out[y * w + x] = Math.sqrt(n) - Math.sqrt(o);
    }
    return out;
}

await check('D1: computeSDF equals the brute-force definition (binary and anti-aliased bitmaps)', () => {
    let seed = 12345;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32);
    for (const [w, h, grey] of [[9, 7, false], [12, 10, true], [5, 13, true]]) {
        const cov = Float32Array.from({ length: w * h }, () => (grey ? Math.round(rnd() * 4) / 4 : rnd() < 0.4 ? 1 : 0));
        const fast = T.computeSDF(cov, w, h), slow = bruteSDF(cov, w, h);
        for (let i = 0; i < w * h; i++) near(fast[i], slow[i], 1e-4, `${w}×${h} pixel ${i}`);
    }
});

// The bound is measured, not hoped for: coverage seeds each edge pixel with a sub-pixel offset and
// the transform sums offsets Pythagorean-style, which is exact for straight edges and ~0.6 px off on
// a curve at worst (0.585 measured here). A binary bitmap would carry a 0.5 px bias everywhere.
await check('D2: an anti-aliased disc\'s field is its true distance to within 0.75 px', () => {
    const w = 40, h = 40, cx = 19.7, cy = 20.2, r = 11.3;
    const cov = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        let inside = 0;                                       // 8×8 supersampled coverage
        for (let sy = 0; sy < 8; sy++) for (let sx = 0; sx < 8; sx++) {
            if (Math.hypot(x + (sx + 0.5) / 8 - 0.5 - cx, y + (sy + 0.5) / 8 - 0.5 - cy) < r) inside++;
        }
        cov[y * w + x] = inside / 64;
    }
    const sdf = T.computeSDF(cov, w, h);
    let worst = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const truth = r - Math.hypot(x - cx, y - cy);
        if (Math.abs(truth) < 6) worst = Math.max(worst, Math.abs(sdf[y * w + x] - truth));
    }
    assert(worst < 0.75, `max error within 6 px of the edge: ${worst.toFixed(3)} px`);
});

await check('D3: encodeSDF maps ±range/2 to 0..255 with the edge at 127.5; shelfPack never overlaps', () => {
    const b = T.encodeSDF([-4, 0, 4, 9], 8, new Uint8Array(4));
    assert(b[0] === 0 && (b[1] === 128 || b[1] === 127) && b[2] === 255 && b[3] === 255, `[${b}]`);
    const sizes = Array.from({ length: 40 }, (_, i) => ({ w: 5 + (i * 7) % 23, h: 4 + (i * 5) % 19 }));
    const { rects, height } = T.shelfPack(sizes, 128);
    for (let i = 0; i < rects.length; i++) {
        const a = rects[i];
        assert(a.x >= 1 && a.x + a.w <= 127 && a.y + a.h <= height, `rect ${i} out of bounds`);
        for (let j = i + 1; j < rects.length; j++) {
            const c = rects[j];
            assert(a.x + a.w <= c.x || c.x + c.w <= a.x || a.y + a.h <= c.y || c.y + c.h <= a.y, `rects ${i} and ${j} overlap`);
        }
    }
    throws(() => T.shelfPack([{ w: 200, h: 4 }], 128), /does not fit a 128px atlas/, 'too wide');
});

// ---------------------------------------------------------------------------
// WGSL
// ---------------------------------------------------------------------------

function balanced(code, label) {
    const pairs = { '{': '}', '(': ')', '[': ']' }, stack = [];
    for (const ch of code.replace(/\/\/[^\n]*/g, '')) {
        if (pairs[ch]) stack.push(pairs[ch]);
        else if ('})]'.includes(ch) && stack.pop() !== ch) throw new Error(`${label}: unbalanced '${ch}'`);
    }
    if (stack.length) throw new Error(`${label}: unclosed`);
}

await check('W1: cull and draw WGSL — entry points, every binding used, the right distance per type', () => {
    const cull = T.buildTextCullWGSL();
    balanced(cull, 'cull');
    assert(/fn cs_main/.test(cull) && /atomicAdd\(&drawArgs\[1\]/.test(cull), 'cull entry / count');
    for (const name of ['glyphs', 'labels', 'atlasGlyphs', 'follow', 'params', 'visible', 'drawArgs']) {
        assert((cull.match(new RegExp(`\\b${name}\\b`, 'g')) || []).length >= 2, `cull: binding ${name} declared but unused`);
    }
    for (const type of ['sdf', 'msdf', 'mtsdf']) {
        const draw = T.buildTextDrawWGSL(type);
        balanced(draw, type);
        assert(/fn vs_main/.test(draw) && /fn fs_main/.test(draw), `${type}: entry points`);
        assert(!draw.includes('`') && !cull.includes('`'), 'a backtick inside a WGSL template ends the string');
        const median = /max\(min\(s\.r, s\.g\), min\(max\(s\.r, s\.g\), s\.b\)\)/.test(draw);
        assert(type === 'sdf' ? !median && /let fillD = \(s\.r/.test(draw) : median, `${type}: wrong distance decode`);
        assert(/let haloD = \(s\.a/.test(draw) === (type === 'mtsdf'), `${type}: halo distance source`);
    }
});

// ---------------------------------------------------------------------------
// TextSystem on an adopted mock device
// ---------------------------------------------------------------------------

function mockDevice() {
    const state = { writes: [], textures: [] };
    const res = (extra = {}) => ({ ...extra, destroy() { this.destroyed = true; } });
    const pipeline = (d) => ({ desc: d, getBindGroupLayout: () => ({}) });
    const device = {
        features: new Set(), adapterInfo: { vendor: 'mock' },
        createBuffer: (d) => res({ ...d, getMappedRange: () => new ArrayBuffer(d.size), unmap() {} }),
        createTexture: (d) => { const t = res({ ...d, createView: () => ({}) }); state.textures.push(t); return t; },
        createSampler: () => ({}), createShaderModule: () => ({}),
        createRenderPipeline: pipeline, createComputePipeline: pipeline,
        createBindGroup: (d) => ({ entries: d.entries }),
        createCommandEncoder: () => ({}),
        queue: { writeBuffer: (b, o, d, dOff = 0, size) => state.writes.push({ b, o, d, dOff, size }), writeTexture: (...a) => state.writes.push({ texture: a }), submit() {} },
    };
    return { device, state };
}

async function rig(opts = {}) {
    const m = mockDevice();
    const engine = new NullGraph();
    const r = await engine.init(null, { device: m.device, width: 800, height: 600 });
    if (!r.success) throw new Error(r.error);
    const atlas = T.FontAtlas.fromSDF(engine, syntheticFont());
    const text = new T.TextSystem(engine, atlas, { maxLabels: 8, maxGlyphs: 32, ...opts });
    return { ...m, engine, atlas, text };
}

await check('S1: setLabel writes the 48-byte record and the glyph records; packRGBA8 is R-low', async () => {
    const { text, atlas } = await rig();
    text.setLabel(2, { text: 'AB', position: [1, 2, 3], sizePx: 24, color: [1, 0.5, 0, 1], halo: { color: [0, 0, 0, 0.5], width: 3 }, offsetPx: [4, -5] });
    const f = text.labelsF32, u = text.labelsU32, b = 2 * T.LABEL_FLOATS;
    assert(f[b] === 1 && f[b + 1] === 2 && f[b + 2] === 3 && f[b + 3] === 24, 'pos / sizePx');
    assert(u[b + 4] === T.packRGBA8([1, 0.5, 0, 1]) && T.packRGBA8([1, 0, 0, 0]) === 0xFF, 'colour packing (R in the low byte)');
    assert(f[b + 6] === 3 && u[b + 7] === T.FOLLOW_NONE && f[b + 8] === 4 && f[b + 9] === -5, 'halo / follow / offset');
    assert(text.glyphCount === 2 && text.glyphsU32[0] === 2 && text.glyphsU32[1] === atlas.glyphIndex(65), 'glyph records');
});

await check('S2: re-setting a label reuses its range when it fits, clears the tail, and grows at the end', async () => {
    const { text } = await rig();
    text.setLabel(0, { text: 'ABAB' });                     // range [0, 4)
    text.setLabel(1, { text: 'B' });                        // range [4, 5)
    text.setLabel(0, { text: 'A' });                        // fits: reuse, clear 3 tail records
    assert(text.glyphsU32[1] === text.atlas.glyphIndex(65), 'reused in place');
    for (const k of [1, 2, 3]) assert(text.glyphsU32[k * 4 + 1] === T.GLYPH_NONE, `tail record ${k} not cleared`);
    text.setLabel(0, { text: 'ABABAB' });                   // too big for 4: new range at 5, old cleared
    assert(text.glyphCount === 11, `glyphCount ${text.glyphCount}`);
    assert(text.glyphsU32[0 * 4 + 1] === T.GLYPH_NONE && text.glyphsU32[5 * 4] === 0, 'old range cleared, new range at the end');
});

await check('S3: createDrawBatch — premultiplied blend, anchor depth test, one indirect draw over a quad', async () => {
    const { engine, text, state } = await rig({ depthTest: true });
    const pass = engine.createPass({ name: 'scene', isMainScreenPass: true });
    text.setLabel(0, { text: 'ABA' });
    const batch = text.createDrawBatch(pass);
    const desc = batch.pipeline.desc;
    const blend = desc.fragment.targets[0].blend;
    assert(blend.color.srcFactor === 'one' && blend.color.dstFactor === 'one-minus-src-alpha', 'not premultiplied');
    assert(desc.depthStencil.depthWriteEnabled === false && desc.depthStencil.depthCompare === 'less-equal', 'depth state');
    assert(batch.isIndirect && batch.indexCount === 6 && batch.vertexBuffer === null && batch.sourceStorageBuffer === text.glyphBuffer,
        'indirect, vertex-pulled quad, sharing the glyph buffer');
    const [cull] = text.getComputeStages();
    assert(JSON.stringify(cull.dispatchCount) === '[1,1,1]', `dispatch ${cull.dispatchCount}`);
    assert(Array.from(cull.preClearBuffers[0].data).join() === '6,0,0,0,0', 'the args must reset to [6, 0, …] every frame');
    text.upload();
    const params = state.writes.find(w => w.b?.label === 'NullGraph_TextParams');
    assert(params && params.d[0] === 800 && params.d[1] === 600 && new Uint32Array(params.d.buffer)[5] === 3, 'params: viewport + glyphCount');
    throws(() => text.createDrawBatch(pass), /already called/, 'second batch');
});

await check('S4: depthTest: false draws on top; follow binds the given buffer', async () => {
    const follow = { usage: GPUBufferUsage.STORAGE, size: 960 };
    const { engine, text } = await rig({ depthTest: false, follow: { buffer: follow, strideFloats: 24, positionOffset: 12 } });
    const pass = engine.createPass({ name: 'scene', isMainScreenPass: true });
    text.setLabel(0, { text: 'A', follow: 3, position: [0, 1.5, 0] });
    const batch = text.createDrawBatch(pass);
    assert(batch.pipeline.desc.depthStencil.depthCompare === 'always', 'depthTest false must compare always');
    assert(batch.customBindGroups[1].entries[5].resource.buffer === follow, 'follow buffer not bound to the draw');
    assert(text.labelsU32[7] === 3, 'follow index');
});

await check('S5: destroy() takes the batch out of its pass — one left behind binds destroyed buffers', async () => {
    const { engine, text } = await rig();
    const pass = engine.createPass({ name: 'scene', isMainScreenPass: true });
    text.setLabel(0, { text: 'AB' });
    const batch = text.createDrawBatch(pass);
    assert(pass.batches.includes(batch), 'precondition: the batch is in its pass');
    text.destroy();
    assert(!pass.batches.includes(batch), 'the batch is still in its pass after destroy()');
    assert(batch.indirectBuffer.destroyed && batch.storageBuffer.destroyed && text.glyphBuffer.destroyed, 'buffers not released');
    text.destroy();                                         // a second call is harmless
});

await check('errors: each refusal names what is wrong and the way out', async () => {
    const { text, engine } = await rig();
    throws(() => text.setLabel(8, { text: 'A' }), /label index 8 is outside \[0, 8\)/, 'index');
    throws(() => text.setLabel(0, { text: 'A', sizePx: 0 }), /sizePx 0 must be positive/, 'size');
    throws(() => text.setLabel(0, { text: 'A', follow: 1 }), /without a follow buffer/, 'follow without buffer');
    throws(() => text.setLabel(0, { text: 'AB'.repeat(17) }), /needs 34 glyphs but only 32 of 32 remain/, 'capacity');
    throws(() => text.getComputeStages(), /before createDrawBatch/, 'stages before batch');
    throws(() => new T.TextSystem(engine, text.atlas, { maxLabels: 0, maxGlyphs: 1 }), /maxLabels 0/, 'maxLabels');
    throws(() => new T.TextSystem(engine, text.atlas, { maxLabels: 1, maxGlyphs: 1, follow: { buffer: { usage: 8 }, strideFloats: 24, positionOffset: 12 } }),
        /lacks GPUBufferUsage\.STORAGE/, 'follow usage');
    throws(() => T.FontAtlas.fromCanvas(engine, { fontFamily: 'x' }), /needs OffscreenCanvas/, 'canvas in Node');
    throws(() => T.FontAtlas.fromSDF(engine, { ...syntheticFont(), data: new Uint8Array(3) }), /needs \d+ bytes \(1 per texel\), got 3/, 'data size');
});

console.log(failures === 0 ? '\nAll text tests passed.' : `\n${failures} test(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
