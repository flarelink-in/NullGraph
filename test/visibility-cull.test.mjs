// Unit tests for the VisibilitySystem cull math + WGSL generation
// (docs/visibility-design.md §10). The cull math is pure logic — no GPU — so U1–U4 run in Node.
// The GPU integration tests (§10 T1–T7) require a browser/WebGPU harness and are not run here;
// instead the generated WGSL is validated structurally (delimiters, entry points, and the
// binding-usage invariant that `layout: 'auto'` relies on).
//   npm run build:lib && node test/visibility-cull.test.mjs
import { mat4 } from 'gl-matrix';
import {
    extractFrustumPlanes, sphereInFrustum, worldSphereFromModel, coneIsBackfacing,
    hiZLevelForFootprint, hiZMipCount, hiZMipSize, selectLod, projectedErrorPx,
    InstanceBuffer, INSTANCE_FLOATS, LOD_AUTO,
    buildHiZMipWGSL, buildInstanceCullWGSL, buildMeshletInstanceCullWGSL,
    buildMeshletCullWGSL, buildInstanceDrawWGSL, buildMeshletDrawWGSL,
    VisibilitySystem,
} from '../dist/visibility.js';
import { MegabufferBuilder, MeshletBuilder } from '../dist/geometry.js';

// WebGPU usage-flag enums are browser globals, absent in Node. The VisibilitySystem reads them
// at construction time (not import time), so polyfill them before any constructor runs.
globalThis.GPUBufferUsage ??= {
    MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16,
    VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512,
};
globalThis.GPUTextureUsage ??= {
    COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16,
};

let failures = 0;

function check(name, fn) {
    try { fn(); console.log(`  ok    ${name}`); }
    catch (e) { failures++; console.log(`  FAIL  ${name}\n        ${e.message}`); }
}
function assert(cond, what) { if (!cond) throw new Error(what); }
function assertEq(actual, expected, what) {
    if (actual !== expected) throw new Error(`${what}: expected ${expected}, got ${actual}`);
}
function near(a, b, eps, what) {
    if (Math.abs(a - b) > eps) throw new Error(`${what}: expected ~${b}, got ${a}`);
}

// A known view-projection: eye at origin looking down -Z, 75° fov, aspect 1, near 0.1, far 1000.
// perspectiveZO — WebGPU's [0,1] clip z, matching core/Camera.ts.
function knownViewProj() {
    const view = mat4.create();
    mat4.lookAt(view, [0, 0, 0], [0, 0, -1], [0, 1, 0]);
    const proj = mat4.create();
    mat4.perspectiveZO(proj, 75 * Math.PI / 180, 1, 0.1, 1000);
    const vp = mat4.create();
    mat4.multiply(vp, proj, view);
    return vp;
}

// === U1: frustum plane extraction ==========================================
check('U1: extractFrustumPlanes yields 6 correct planes', () => {
    const p = extractFrustumPlanes(knownViewProj());
    assertEq(p.length, 24, '24 floats (6 planes)');
    // All normals unit length.
    for (let i = 0; i < 6; i++) {
        const l = Math.hypot(p[i * 4], p[i * 4 + 1], p[i * 4 + 2]);
        near(l, 1, 1e-6, `plane ${i} normal length`);
    }
    // Near plane (index 4): normal (0,0,-1), passes through z=-0.1.
    near(p[16], 0, 1e-5, 'N nx'); near(p[17], 0, 1e-5, 'N ny'); near(p[18], -1, 1e-5, 'N nz');
    near(p[16] * 0 + p[17] * 0 + p[18] * -0.1 + p[19], 0, 1e-4, 'N plane through z=-0.1');
    // Far plane (index 5): normal (0,0,+1).
    near(p[20], 0, 1e-4, 'F nx'); near(p[22], 1, 1e-4, 'F nz');
    // Left plane (index 0) normal points +X-ish and into the frustum (-Z): (+cos, 0, -sin).
    assert(p[0] > 0.7 && p[2] < 0, `L normal into frustum, got ${p[0]},${p[2]}`);
    // Right plane (index 1) is the mirror: (-cos, 0, -sin).
    assert(p[4] < -0.7 && p[6] < 0, `R normal into frustum, got ${p[4]},${p[6]}`);
});

// === U2: sphere vs frustum =================================================
check('U2: sphereInFrustum classification', () => {
    const p = extractFrustumPlanes(knownViewProj());
    assert(sphereInFrustum(p, 0, 0, -10, 1) === true, 'center sphere inside');
    assert(sphereInFrustum(p, 0, 0, 10, 1) === false, 'behind camera culled');
    assert(sphereInFrustum(p, 100, 0, -10, 1) === false, 'far right culled');
    assert(sphereInFrustum(p, 0, 0, -0.1, 1) === true, 'straddling near plane kept');
    // A sphere just outside the left plane by more than its radius is culled; by less, kept.
    // At z=-10 the left half-extent is tan(37.5°)*10 ≈ 7.67.
    assert(sphereInFrustum(p, -9.5, 0, -10, 1) === false, 'fully outside left culled');
    assert(sphereInFrustum(p, -7.5, 0, -10, 1) === true, 'straddling left plane kept');
});

// === U2b: worldSphereFromModel (conservative radius under scale) ===========
check('U2b: worldSphereFromModel transforms center and scales radius conservatively', () => {
    const m = mat4.create();
    mat4.translate(m, m, [10, 0, 0]);
    mat4.scale(m, m, [2, 3, 1]); // non-uniform: max scale 3
    const out = worldSphereFromModel(m, [0, 0, 0, 2]);
    near(out[0], 10, 1e-5, 'cx'); near(out[1], 0, 1e-5, 'cy'); near(out[2], 0, 1e-5, 'cz');
    near(out[3], 6, 1e-5, 'radius × max scale (3)');
});

// === U3: Hi-Z level selection ==============================================
check('U3: hiZLevelForFootprint picks ceil(log2(sizePx)), clamped', () => {
    assertEq(hiZLevelForFootprint(1, 10), 0, 'sizePx 1');
    assertEq(hiZLevelForFootprint(2, 10), 1, 'sizePx 2');
    assertEq(hiZLevelForFootprint(3, 10), 2, 'sizePx 3');
    assertEq(hiZLevelForFootprint(256, 16), 8, 'sizePx 256');
    assertEq(hiZLevelForFootprint(10000, 10), 10, 'clamped to maxLevel');
    assertEq(hiZLevelForFootprint(0, 10), 0, 'sizePx 0 → level 0');
});
check('U3b: hiZMipCount / hiZMipSize', () => {
    assertEq(hiZMipCount(1920, 1080), 12, '1920×1080 mip count'); // ceil(log2(1920))=11, +1 = 12

    assertEq(hiZMipCount(8, 8), 4, '8×8 mip count');
    const [w1, h1] = hiZMipSize(1920, 1080, 1);
    assertEq(w1, 960, 'mip1 w'); assertEq(h1, 540, 'mip1 h');
    const [w4] = hiZMipSize(1920, 1080, 4);
    assertEq(w4, 120, 'mip4 w (floor)');
});

// === U4: cone sign (real cone from MeshletBuilder) =========================
check('U4: coneIsBackfacing culls the back face, keeps the front (sign is load-bearing)', () => {
    // Build a +Z-facing quad as a meshlet and read its computed cone.
    const STRIDE = 8;
    const verts = new Float32Array([
        -0.5, -0.5, 0.5, 0, 0, 1, 0, 0,
        0.5, -0.5, 0.5, 0, 0, 1, 0, 0,
        0.5, 0.5, 0.5, 0, 0, 1, 0, 0,
        -0.5, 0.5, 0.5, 0, 0, 1, 0, 0,
    ]);
    const mega = new MegabufferBuilder(STRIDE);
    const builder = new MeshletBuilder(mega, { positionOffsetFloats: 0 });
    builder.addMesh('face', { vertices: verts, indices: [0, 1, 2, 0, 2, 3] });
    const pack = builder.build();
    const f = new Float32Array(pack.meshlets);
    const axis = [f[8], f[9], f[10]];           // coneAxis.xyz ≈ +Z
    const cutoff = f[11];                        // flat cluster → 0
    const apex = [f[4], f[5], f[6]];            // coneApex = sphere center (0,0,0.5)
    near(axis[2], 1, 1e-4, 'cone axis +Z');
    near(cutoff, 0, 1e-4, 'flat cluster cutoff 0');

    // Camera in FRONT of the +Z face (eye at +Z): the face is visible → NOT backfacing.
    assert(coneIsBackfacing(axis, cutoff, apex, [0, 0, 5]) === false, 'front camera keeps the face');
    // Camera BEHIND (eye at -Z, looking at the face's back): every triangle faces away → culled.
    assert(coneIsBackfacing(axis, cutoff, apex, [0, 0, -5]) === true, 'back camera culls the face');
    // The 2.0 sentinel never culls, regardless of viewpoint.
    assert(coneIsBackfacing(axis, 2.0, apex, [0, 0, -5]) === false, 'sentinel cutoff 2.0 never culls');
});

// === LOD selection (§8) ====================================================
check('LOD: selectLod coarsens while under the error budget', () => {
    const errors = [0.0, 0.1, 0.4];
    const fov = 75 * Math.PI / 180, screenH = 1080;
    // Near: even LOD2 error is over budget → LOD0.
    assertEq(selectLod(errors, 5, screenH, fov, 1.0), 0, 'near → LOD0');
    // Far enough that all pass → coarsest (LOD2).
    assertEq(selectLod(errors, 10000, screenH, fov, 1.0), 2, 'very far → LOD2');
    // Single-LOD mesh never LODs.
    assertEq(selectLod([0.0], 10000, screenH, fov, 1.0), 0, 'single LOD → 0');
    // projectedErrorPx monotonic in error and inverse in distance.
    assert(projectedErrorPx(0.4, 100, screenH, fov) > projectedErrorPx(0.1, 100, screenH, fov), 'error monotonic');
    assert(projectedErrorPx(0.1, 10, screenH, fov) > projectedErrorPx(0.1, 100, screenH, fov), 'distance inverse');
});

// === InstanceBuffer packing (§4.1) =========================================
check('InstanceBuffer: 24-float layout, integer fields as raw bits', () => {
    const buf = new InstanceBuffer(4);
    const model = mat4.create();
    mat4.translate(model, model, [1, 2, 3]);
    buf.write(0, model, [5, 6, 7, 2.5], 3, LOD_AUTO, 1);
    assertEq(buf.count, 1, 'count');
    const d = buf.data;
    near(d[12], 1, 1e-5, 'model tx'); near(d[13], 2, 1e-5, 'model ty'); near(d[14], 3, 1e-5, 'model tz');
    near(d[16], 5, 1e-5, 'bounds x'); near(d[19], 2.5, 1e-5, 'bounds r');
    const u32 = new Uint32Array(d.buffer);
    assertEq(u32[20], 3, 'meshIndex as u32 bits');
    assertEq(u32[21], LOD_AUTO, 'lodOverride as u32 bits');
    assertEq(u32[22], 1, 'flags as u32 bits');
    assertEq(INSTANCE_FLOATS, 24, 'INSTANCE_FLOATS');
    // Second instance lands at offset 24.
    buf.write(1, mat4.create(), [0, 0, 0, 1], 0);
    assertEq(buf.count, 2, 'count after second write');
    assertEq(new Uint32Array(d.buffer)[24 + 20], 0, 'second instance meshIndex at offset 24+20');
    // Out of range throws.
    let threw = false;
    try { buf.write(4, mat4.create(), [0, 0, 0, 1], 0); } catch (e) { threw = true; }
    assert(threw, 'out-of-range write throws');
});

// === VisibilitySystem constructor validation (mock engine, no GPU) =========
check('VisibilitySystem: constructor + setInstances validation', () => {
    const mockDevice = {
        createBuffer: () => ({ destroy() {}, getMappedRange: () => new ArrayBuffer(256), unmap() {} }),
        createTexture: () => ({ destroy() {}, createView: () => ({}) }),
        queue: { writeBuffer() {} },
    };
    const mockEngine = { device: mockDevice, width: 1920, height: 1080, ctx: { cameraUniformBuffer: {} } };

    const vis = new VisibilitySystem(mockEngine, { maxInstances: 100 });
    assert(vis !== undefined, 'constructs with maxInstances');

    let threw = false;
    try { new VisibilitySystem(mockEngine, { maxInstances: 0 }); } catch (e) { threw = true; }
    assert(threw, 'maxInstances 0 throws');

    threw = false;
    try { new VisibilitySystem(mockEngine, { maxInstances: 10, mode: 'two-phase' }); } catch (e) {
        threw = /two-phase/.test(e.message);
    }
    assert(threw, 'two-phase throws a named reserved error');

    threw = false;
    try { vis.setInstances(new Float32Array(101 * 24), 101); } catch (e) { threw = /exceeding maxInstances/.test(e.message); }
    assert(threw, 'setInstances overflow throws');

    threw = false;
    try { vis.getComputeStages(); } catch (e) { threw = /before any draw batch/.test(e.message); }
    assert(threw, 'getComputeStages before any batch throws');
});

// === WGSL structural validation ============================================

function checkBalanced(code, label) {
    const pairs = { '{': '}', '(': ')', '[': ']' };
    const stack = [];
    // Strip line comments so `// {` doesn't unbalance the count.
    for (const ch of code.replace(/\/\/[^\n]*/g, '')) {
        if (pairs[ch]) stack.push(pairs[ch]);
        else if (ch === '}' || ch === ')' || ch === ']') {
            if (stack.pop() !== ch) throw new Error(`${label}: unbalanced '${ch}'`);
        }
    }
    if (stack.length !== 0) throw new Error(`${label}: ${stack.length} unclosed delimiter(s)`);
}

function checkBindingsUsed(code, label) {
    // Every declared @binding variable must be referenced again (layout:'auto' prunes unused
    // bindings, and the engine throws when it then supplies an entry for the pruned binding).
    const declRe = /@binding\(\d+\)\s+var(?:<[^>]*>)?\s+(\w+)\s*:/g;
    let m;
    while ((m = declRe.exec(code)) !== null) {
        const name = m[1];
        const uses = code.match(new RegExp('\\b' + name + '\\b', 'g')) || [];
        if (uses.length < 2) throw new Error(`${label}: binding var '${name}' declared but never used`);
    }
}

function validateWGSL(label, code, entries) {
    checkBalanced(code, label);
    for (const e of entries) {
        if (!new RegExp('fn\\s+' + e + '\\b').test(code)) throw new Error(`${label}: missing entry '${e}'`);
    }
    checkBindingsUsed(code, label);
    console.log(`  ok    WGSL ${label}`);
}

check('WGSL: all generated modules are structurally valid', () => {
    validateWGSL('HiZ mip0 (from depth)', buildHiZMipWGSL(true), ['cs_main']);
    validateWGSL('HiZ mipL (from mip)', buildHiZMipWGSL(false), ['cs_main']);
    validateWGSL('instance cull (no occ)', buildInstanceCullWGSL({ occlusion: false }), ['cs_main']);
    validateWGSL('instance cull (occ)', buildInstanceCullWGSL({ occlusion: true }), ['cs_main']);
    validateWGSL('meshlet instance cull (no occ)', buildMeshletInstanceCullWGSL({ occlusion: false }), ['cs_main']);
    validateWGSL('meshlet instance cull (occ)', buildMeshletInstanceCullWGSL({ occlusion: true }), ['cs_main']);
    validateWGSL('meshlet cull (no occ)', buildMeshletCullWGSL({ occlusion: false }), ['cs_main']);
    validateWGSL('meshlet cull (occ)', buildMeshletCullWGSL({ occlusion: true }), ['cs_main']);
    validateWGSL('meshlet cull (no cone)', buildMeshletCullWGSL({ cone: false }), ['cs_main']);
    validateWGSL('instance draw', buildInstanceDrawWGSL(), ['vs_main', 'fs_main']);
    validateWGSL('meshlet draw', buildMeshletDrawWGSL({ strideFloats: 8 }), ['vs_main', 'fs_main']);
});

console.log(failures === 0 ? '\nAll visibility cull tests passed.' : `\n${failures} test(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
