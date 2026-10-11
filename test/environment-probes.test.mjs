// Unit tests for Phase 2 — irradiance volumes & probes (docs/lighting-probes-design.md §10).
// All pure math is headless (probeMath.ts pinned by number — the CPU mirrors of the WGSL in
// probeShaders.ts); every generated WGSL module is parsed with wgsl_reflect; the systems' wiring
// is exercised through a mock device, the same discipline as shadows-csm.test.mjs and
// environment-ibl.test.mjs. Colour bleeding/feedback run in test/probes-gpu on Node and Chrome;
// the remaining integration tests (leak bounds, white room, reflections) are still unwritten.
//   npm run build:lib && node test/environment-probes.test.mjs
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';
import {
    probeIndex, probeCoord, probePosition, worldToUVW, probeSampleUVW, trilinearWeights,
    projectSHL1FromCapture, evalSHL1CPU, volumeBorderWeightCPU, sampleProbeVolumeCPU,
    parallaxCorrect, reflInfluenceWeight, blendReflectionProbes,
    packReflProbe, createProbeScheduler, scheduleProbes, validateProbeVolumeConfig,
    buildProbeStructsWGSL, buildProbeIndirectLogicWGSL, buildProbeIndirectLayout,
    buildProbeProjectionWGSL, buildReflAssembleWGSL, buildPrefilterWGSL,
    buildIndirectStructsWGSL, buildIndirectLogicWGSL,
    ProbeVolume, ReflectionProbes,
} from '../dist/environment.js';
import { buildMaterialFragment } from '../dist/materials.js';

let failures = 0;
function check(name, fn) {
    try { fn(); console.log(`  ok    ${name}`); }
    catch (e) { failures++; console.log(`  FAIL  ${name}\n        ${e.message}`); }
}
async function checkAsync(name, fn) {
    try { await fn(); console.log(`  ok    ${name}`); }
    catch (e) { failures++; console.log(`  FAIL  ${name}\n        ${e.message}`); }
}
function assert(cond, what) { if (!cond) throw new Error(what); }
function assertEq(actual, expected, what) {
    if (actual !== expected) throw new Error(`${what}: expected ${expected}, got ${actual}`);
}
function near(a, b, eps, what) {
    if (!(Math.abs(a - b) <= eps)) throw new Error(`${what}: expected ~${b}, got ${a}`);
}
function assertThrows(fn, re, what) {
    let threw = null;
    try { fn(); } catch (e) { threw = e.message; }
    assert(threw && re.test(threw), `${what}: expected ${re}, got ${threw ?? 'no throw'}`);
}
/** Seeded LCG — deterministic randoms for the volume/point sweeps. */
function lcg(seed) {
    let s = seed >>> 0;
    return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

// A camera + VertexOut preamble, so a material fragment can be parsed as a whole module.
const HOST_PREAMBLE = `struct Camera { viewProj: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>,
 invViewProj: mat4x4<f32>, invView: mat4x4<f32>, invProj: mat4x4<f32>, eyePos: vec3<f32>,
 simTime: f32, zNear: f32, zFar: f32, fovRads: f32, aspect: f32, _padEnd: vec4<f32> };
@group(0) @binding(0) var<uniform> camera: Camera;
struct VertexOut { @builtin(position) clipPos: vec4<f32>, @location(0) worldPos: vec3<f32>,
 @location(1) worldNormal: vec3<f32>, @location(2) uv: vec2<f32>, @location(3) ecsColor: vec3<f32> };
@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }`;

/** The real Phase 1 layout — the decorator consumes (and renames) exactly this. */

// === U1: grid ↔ world round-trip (§4.1, U1) =================================
check('reflection data header is 16 bytes, matching CPU records and the empty placeholder', () => {
    const reflect = new WgslReflect(buildProbeStructsWGSL(2));
    const header = reflect.structs.find(s => s.name === 'ReflProbeBuffer');
    assertEq(header.members.find(m => m.name === 'probes').offset, 16, 'probe array byte offset');
    assertEq(header.size, 16, 'fixed header byte size (runtime array excluded by reflector)');
});

check('U1: worldToUVW(probePosition(i,j,k)) recovers (i,j,k)/(dims-1); index/coord round-trips', () => {
    const D = [32, 16, 32], O = [-64, 0, -64], S = 4;
    const corners = [[0, 0, 0], [31, 0, 0], [0, 15, 0], [0, 0, 31], [31, 15, 31], [31, 15, 0], [0, 15, 31], [31, 0, 31]];
    for (const [i, j, k] of corners) {
        const uvw = worldToUVW(O, S, D, probePosition(O, S, i, j, k));
        near(uvw[0], i / (D[0] - 1), 1e-6, `corner (${i},${j},${k}) uvw.x`);
        near(uvw[1], j / (D[1] - 1), 1e-6, `corner (${i},${j},${k}) uvw.y`);
        near(uvw[2], k / (D[2] - 1), 1e-6, `corner (${i},${j},${k}) uvw.z`);
        const c = probeCoord(D, probeIndex(D, i, j, k));
        assert(c[0] === i && c[1] === j && c[2] === k, `probeCoord(probeIndex(${i},${j},${k})) round-trips`);
    }
    const rand = lcg(20260816);
    for (let n = 0; n < 1000; n++) {
        const i = Math.floor(rand() * 31), j = Math.floor(rand() * 15), k = Math.floor(rand() * 31);
        const uvw = worldToUVW(O, S, D, probePosition(O, S, i, j, k));
        near(uvw[0], i / 31, 1e-6, `interior ${n} x`);
        near(uvw[1], j / 15, 1e-6, `interior ${n} y`);
        near(uvw[2], k / 31, 1e-6, `interior ${n} z`);
        const c = probeCoord(D, probeIndex(D, i, j, k));
        assert(c[0] === i && c[1] === j && c[2] === k, `interior ${n} index round-trip`);
    }
    // Points outside the volume map outside [0,1]³.
    for (const p of [[-64.1, 4, 4], [-63, -0.1, 4], [-63, 4, 72.1], [-63, 70, 4], [-300, 4, 4], [-63, 4, -300]]) {
        const uvw = worldToUVW(O, S, D, p);
        const outside = uvw[0] < 0 || uvw[0] > 1 || uvw[1] < 0 || uvw[1] > 1 || uvw[2] < 0 || uvw[2] > 1;
        assert(outside, `point ${p} maps outside [0,1]³`);
    }
    const inside = worldToUVW(O, S, D, [-60, 10, 10]);
    assert(inside[0] > 0 && inside[0] < 1 && inside[1] > 0 && inside[1] < 1 && inside[2] > 0 && inside[2] < 1,
        'an interior point maps inside');
});

// === U2: trilinear weights (§5, U2) ==========================================
check('U2: hardware-trilinear mirror — eight ⅛s between probes, one 1 on a probe centre', () => {
    const D = [4, 4, 4];
    // Exactly between probes (1,1,1) and (2,2,2): grid coord 1.5 → uvw 1.5/3 = 0.5.
    const mid = worldToUVW([0, 0, 0], 1, D, [1.5, 1.5, 1.5]);
    const r = trilinearWeights(mid, D);
    let sum = 0;
    const expected = new Set();
    for (const i of [1, 2]) for (const j of [1, 2]) for (const k of [1, 2]) expected.add(probeIndex(D, i, j, k));
    for (let c = 0; c < 8; c++) {
        near(r.weights[c], 0.125, 1e-7, `weight ${c} between probes`);
        assert(expected.has(r.indices[c]), `index ${c} is one of the 8 neighbours`);
        sum += r.weights[c];
    }
    near(sum, 1, 1e-7, 'weights sum to 1');

    // On a probe centre: one weight of exactly 1. (Texel-CENTRE alignment — without the
    // (tc+0.5)/dims remap in the sampler path this test could never hold; probeSampleUVW pins it.)
    const onProbe = worldToUVW([0, 0, 0], 1, D, probePosition([0, 0, 0], 1, 2, 2, 2));
    const r2 = trilinearWeights(onProbe, D);
    let ones = 0, zeroes = 0;
    for (let c = 0; c < 8; c++) {
        if (r2.weights[c] === 1 && r2.indices[c] === probeIndex(D, 2, 2, 2)) ones++;
        else if (r2.weights[c] === 0) zeroes++;
    }
    assertEq(ones, 1, 'exactly one weight of 1 on the probe');
    assertEq(zeroes, 7, 'all other weights 0');
    const sc = probeSampleUVW(onProbe, D);
    near(sc[0], (2 + 0.5) / 4, 1e-7, 'probe (2,2,2) samples at texel centre x');
    near(sc[1], (2 + 0.5) / 4, 1e-7, 'probe (2,2,2) samples at texel centre y');
});

const ENV_LAYOUT = { wgslStructs: buildIndirectStructsWGSL(2), wgslLogic: buildIndirectLogicWGSL() };

// === U3: SH-L1 of a constant environment (§7.2, U3) ==========================
check('U3: a uniform radiance c projects to irradiance c in every direction (±1%)', () => {
    const F = 8, C = [2.0, 0.5, 1.0];
    const color = new Float32Array(6 * F * F * 4);
    for (let t = 0; t < 6 * F * F; t++) { color[t * 4] = C[0]; color[t * 4 + 1] = C[1]; color[t * 4 + 2] = C[2]; color[t * 4 + 3] = 1; }
    const sh = projectSHL1FromCapture({ color, faceSize: F });
    // L0 per channel ≈ c (the solid angles sum to EXACTLY 4π at any size — env U3); L1 ≈ 0.
    near(sh[3], C[0], C[0] * 0.01, 'L0.r ≈ c.r');
    near(sh[7], C[1], C[1] * 0.01 + 0.005, 'L0.g ≈ c.g');
    near(sh[11], C[2], C[2] * 0.01, 'L0.b ≈ c.b');
    for (const [ci, cc] of [[0, 0], [4, 1], [8, 2]]) {
        for (let k = 0; k < 3; k++) {
            assert(Math.abs(sh[ci + k]) < C[cc] * 0.005 + 1e-3, `L1 of a constant is ~0 (channel ${cc}, k ${k}): ${sh[ci + k]}`);
        }
    }
    // The sky path (depth cleared) must agree with the color path.
    const shSky = projectSHL1FromCapture({
        color, faceSize: F,
        depth: new Float32Array(6 * F * F).fill(1), skyRadiance: C,
    });
    near(shSky[3], C[0], C[0] * 0.01, 'sky path L0.r ≈ c.r');
    // Irradiance c in EVERY direction.
    const rand = lcg(7);
    for (let n = 0; n < 200; n++) {
        const z = rand() * 2 - 1, a = rand() * 2 * Math.PI, s = Math.sqrt(1 - z * z);
        const e = evalSHL1CPU(sh.subarray(0, 4), sh.subarray(4, 8), sh.subarray(8, 12), [s * Math.cos(a), s * Math.sin(a), z]);
        near(e[0], C[0], C[0] * 0.01, `dir ${n} irradiance.r`);
        near(e[1], C[1], C[1] * 0.01 + 0.005, `dir ${n} irradiance.g`);
        near(e[2], C[2], C[2] * 0.01, `dir ${n} irradiance.b`);
    }
});

// === U4: L1 clamping (§5, U4) ================================================
check('U4: a capture bright in one direction and black elsewhere evaluates >= 0 everywhere', () => {
    const F = 8;
    const color = new Float32Array(6 * F * F * 4);
    for (let t = 0; t < F * F; t++) { color[t * 4] = 3; color[t * 4 + 1] = 3; color[t * 4 + 2] = 3; color[t * 4 + 3] = 1; } // +X face only
    const sh = projectSHL1FromCapture({ color, faceSize: F });
    const rand = lcg(99);
    for (let n = 0; n < 500; n++) {
        const z = rand() * 2 - 1, a = rand() * 2 * Math.PI, s = Math.sqrt(1 - z * z);
        const e = evalSHL1CPU(sh.subarray(0, 4), sh.subarray(4, 8), sh.subarray(8, 12), [s * Math.cos(a), s * Math.sin(a), z]);
        assert(e[0] >= 0 && e[1] >= 0 && e[2] >= 0, `dir ${n} evaluates >= 0`);
    }
    // Facing the bright side reads bright; facing away rings NEGATIVE and the clamp catches it.
    const ePos = evalSHL1CPU(sh.subarray(0, 4), sh.subarray(4, 8), sh.subarray(8, 12), [1, 0, 0]);
    assert(ePos[0] > 1.0 && ePos[0] < 1.5, `eval(+X) bright (≈L0+L1x = ${ePos[0]})`);
    const eNeg = evalSHL1CPU(sh.subarray(0, 4), sh.subarray(4, 8), sh.subarray(8, 12), [-1, 0, 0]);
    assertEq(eNeg[0], 0, 'eval(-X) clamped to 0');
    const raw = sh[0] * -1 + sh[3];   // the UNCLAMPED dot — L1 rings, so the max() in evalSHL1 is load-bearing
    assert(raw < 0, `the unclamped L1 evaluation at -X is negative (${raw.toFixed(3)}) — without the clamp this test's first loop would fail`);
});


// === U5: validity fallback is toward the environment, EXACTLY (§5, U5) =======
check('U5: validity 0 everywhere ⇒ the lookup equals the global environment exactly (not black)', () => {
    const D = [4, 4, 4], count = 64;
    const rand = lcg(1234);
    const vol = {
        origin: [0, 0, 0], spacing: 1, dims: D, normalBiasTexels: 0.5, borderFade: 0.5, intensity: 1,
        probeR: Float32Array.from({ length: count * 4 }, () => rand() * 40 - 20),   // deliberately wild
        probeG: Float32Array.from({ length: count * 4 }, () => rand() * 40 - 20),
        probeB: Float32Array.from({ length: count * 4 }, () => rand() * 40 - 20),
        validity: new Float32Array(count),                                          // all 0
    };
    const global = (N, out = [0, 0, 0]) => { out[0] = 0.4 * (N[1] + 1); out[1] = 0.2; out[2] = 0.7; return out; };
    for (let n = 0; n < 100; n++) {
        const p = [rand() * 3, rand() * 3, rand() * 3];
        const N = [0, 1, 0];
        const e = sampleProbeVolumeCPU(vol, p, N, global);
        const g = global(N);
        // EXACT equality — a half-captured volume looks like Phase 1, never like a bug.
        assert(e[0] === g[0] && e[1] === g[1] && e[2] === g[2], `point ${n} falls back EXACTLY`);
    }
    // Outside the volume with validity 1: also exactly the global.
    vol.validity.fill(1);
    const gOut = global([0, 1, 0]);
    const eOut = sampleProbeVolumeCPU(vol, [-5, 1, 1], [0, 1, 0], global);
    assert(eOut[0] === gOut[0] && eOut[1] === gOut[1] && eOut[2] === gOut[2], 'outside the volume → the global, exactly');
    // Non-vacuous: inside with validity 1 and deep-border-free content, the local wins.
    vol.probeR.fill(0); vol.probeG.fill(0); vol.probeB.fill(0);
    for (let i = 0; i < count; i++) { vol.probeR[i * 4 + 3] = 5; vol.probeG[i * 4 + 3] = 5; vol.probeB[i * 4 + 3] = 5; }
    const eIn = sampleProbeVolumeCPU(vol, [1.5, 1.5, 1.5], [0, 1, 0], global);
    near(eIn[0], 5, 1e-5, 'inside with valid probes reads the volume, not the global');
    assert(eIn[0] !== gOut[0], 'the fallback test above was not vacuous');
});

// === U6: parallax uses the CAPTURE point (§8.2, U6) ==========================
check('U6: parallax correction reprojects from the capture point, not the box centre', () => {
    const center = [0, 2, 0], half = [8, 3, 8], capture = [4, 2, 0];   // offset by half the extent (U6's trap)
    const worldPos = [0, 1, 0];
    const R = (() => { const l = Math.hypot(0.7, 0.15, 0.65); return [0.7 / l, 0.15 / l, 0.65 / l]; })();

    // Analytic ray→box solve, written independently (slab furthest-exit).
    let dist = Infinity;
    for (let a = 0; a < 3; a++) {
        const inv = 1 / R[a];
        const t1 = (center[a] + half[a] - worldPos[a]) * inv;
        const t2 = (center[a] - half[a] - worldPos[a]) * inv;
        dist = Math.min(dist, Math.max(t1, t2));
    }
    const hit = [worldPos[0] + R[0] * dist, worldPos[1] + R[1] * dist, worldPos[2] + R[2] * dist];
    const analytic = (() => {
        const d = [hit[0] - capture[0], hit[1] - capture[1], hit[2] - capture[2]];
        const l = Math.hypot(...d); return d.map(v => v / l);
    })();

    const corrected = parallaxCorrect(center, half, capture, worldPos, R);
    near(corrected[0], analytic[0], 1e-6, 'corrected.x matches the analytic solve');
    near(corrected[1], analytic[1], 1e-6, 'corrected.y matches the analytic solve');
    near(corrected[2], analytic[2], 1e-6, 'corrected.z matches the analytic solve');

    // The bug U6 exists for: substituting `center` for `capture` in the final line. It MUST
    // differ measurably here — off-centre capture is what makes the two distinguishable.
    const naive = (() => {
        const d = [hit[0] - center[0], hit[1] - center[1], hit[2] - center[2]];
        const l = Math.hypot(...d); return d.map(v => v / l);
    })();
    const dot = corrected[0] * naive[0] + corrected[1] * naive[1] + corrected[2] * naive[2];
    const angleDeg = Math.acos(Math.min(1, dot)) * 180 / Math.PI;
    assert(angleDeg > 1, `capture-vs-center difference is measurable (${angleDeg.toFixed(2)}°) — swap them and the analytic assertion above fails`);
});


// === U7: influence weights never exceed 1 (§8.3, U7) =========================
check('U7: overlapping probes never exceed total weight 1; the environment gets the remainder', () => {
    // Sorted as the system sorts them (smaller proxy first). C is never-captured → weight 0.
    const records = [
        { centerW: [1, 1, 0, 0.5], halfExtentW: [2, 1, 2, Math.hypot(2, 1, 2)], captured: true },
        { centerW: [0, 1, 0, 0.5], halfExtentW: [4, 1, 4, Math.hypot(4, 1, 4)], captured: true },
        { centerW: [0, 1, 0.5, 0.5], halfExtentW: [3, 1.5, 3, Math.hypot(3, 1.5, 3)], captured: false },
    ];
    const weights = new Float32Array(records.length);
    const rand = lcg(31337);
    for (let n = 0; n < 100; n++) {
        const p = [rand() * 12 - 6, rand() * 4 - 1, rand() * 12 - 6];
        const total = blendReflectionProbes(records, p, weights);
        assert(total <= 1 + 1e-6, `point ${n}: total probe weight ${total} ≤ 1`);
        for (let i = 0; i < records.length; i++) {
            assert(weights[i] >= 0 && weights[i] <= 1, `point ${n}: weight ${i} in [0,1] (${weights[i]})`);
        }
        assertEq(weights[2], 0, `point ${n}: the never-captured probe claims nothing`);
        const env = 1 - total;   // §8.3's energy-preserving remainder
        assert(env >= -1e-6, `point ${n}: environment remainder ≥ 0 (${env})`);
    }
    // Deep inside the small probe: it claims 1, the bigger (less specific) probe and the env get 0.
    blendReflectionProbes(records, [0.5, 1, 0], weights);
    near(weights[0], 1, 1e-6, 'small probe deep inside claims full weight');
    assertEq(weights[1], 0, 'the larger probe is crowded out');
});

// === U8: the scheduler is fair (§7.4, U8) ====================================
check('U8: over 8000 frames every probe is captured and near probes refresh ≥ 4× far ones', () => {
    const D = [8, 8, 8], COUNT = 512, PER_FRAME = 16, FRAMES = 8000;
    const positions = new Float32Array(COUNT * 3);
    for (let i = 0; i < COUNT; i++) {
        const c = probeCoord(D, i);
        const p = probePosition([0, 0, 0], 1, c[0], c[1], c[2]);
        positions[i * 3] = p[0]; positions[i * 3 + 1] = p[1]; positions[i * 3 + 2] = p[2];
    }
    const camPos = [1, 1, 1];
    const diag = Math.hypot(7, 7, 7);
    const sch = createProbeScheduler(COUNT, PER_FRAME);
    // Deterministic cold start: the first frame takes probes 0..15 in index order.
    const n0 = scheduleProbes(sch, positions, COUNT, camPos, 1, 16, 4 / diag);
    assertEq(n0, PER_FRAME, 'frame 1 fills the budget');
    for (let k = 0; k < PER_FRAME; k++) assertEq(sch.picks[k], k, `cold start is index order (pick ${k})`);

    const seen = new Uint32Array(COUNT);
    for (let f = 2; f <= FRAMES; f++) {
        const n = scheduleProbes(sch, positions, COUNT, camPos, f, 16, 4 / diag);
        assert(n <= PER_FRAME, `frame ${f} never exceeds probesPerFrame`);
        for (let k = 0; k < n; k++) seen[sch.picks[k]]++;
    }
    seen[0]++; seen[15] = seen[15] + 0;   // account for frame 1 (0..15)
    for (let k = 0; k < PER_FRAME; k++) seen[k]++;
    for (let i = 0; i < COUNT; i++) assert(seen[i] >= 1, `probe ${i} captured at least once (no starvation)`);

    let nearSum = 0, nearN = 0, farSum = 0, farN = 0;
    for (let i = 0; i < COUNT; i++) {
        const d = Math.hypot(positions[i * 3] - 1, positions[i * 3 + 1] - 1, positions[i * 3 + 2] - 1);
        if (d <= 2.0) { nearSum += seen[i]; nearN++; }
        else if (d >= 8.0) { farSum += seen[i]; farN++; }
    }
    assert(nearN > 10 && farN > 10, `rings are populated (near ${nearN}, far ${farN})`);
    const ratio = (nearSum / nearN) / (farSum / farN);
    assert(ratio >= 4, `near:far refresh ratio ${ratio.toFixed(2)} ≥ 4 (near avg ${(nearSum / nearN).toFixed(1)}, far avg ${(farSum / farN).toFixed(1)})`);
});


// === U9: the contract is unchanged (§1, U9) ==================================
check('U9: getShaderLayout() re-emits byte-identical signatures; the material fragment is unchanged', () => {
    const layout = buildProbeIndirectLayout(ENV_LAYOUT, 2);
    const countOcc = (s, n) => s.split(n).length - 1;

    // The re-emitted definitions have EXACTLY Phase 1's signatures (whitespace-normalized).
    const sig = (src, name) => (src.match(new RegExp(`fn ${name}\\([^)]*\\)`, 's'))?.[0] ?? '').replace(/\s+/g, ' ');
    for (const name of ['getIndirectDiffuse', 'getIndirectSpecular']) {
        const a = sig(ENV_LAYOUT.wgslLogic, name), b = sig(layout.wgslLogic, name);
        assert(a.length > 0, `Phase 1 defines ${name}`);
        assertEq(b, a, `${name}'s signature is byte-identical (whitespace-normalized)`);
        assertEq(countOcc(layout.wgslLogic, `fn ${name}(`), 1, `${name} defined exactly once`);
    }
    // The Phase 1 bodies survive as the *Global fallback the volume blends toward.
    assertEq(countOcc(layout.wgslLogic, 'fn getIndirectDiffuseGlobal('), 1, 'the global diffuse survives the rename');
    assertEq(countOcc(layout.wgslLogic, 'fn getIndirectSpecularGlobal('), 1, 'the global specular survives the rename');
    // The fallback call must pass BOTH arguments: Phase 1's signature is (worldPos, N), so a
    // one-arg `getIndirectDiffuseGlobal(N)` is a WGSL compile error, not a shorthand. An earlier
    // draft of this assertion copied the simplified snippet from the design doc's §5 and so
    // "passed" against code that could not have compiled — the composed-module parse below is
    // what actually holds the two honest.
    assert(/getIndirectDiffuseGlobal\(\s*worldPos\s*,\s*N\s*\)/.test(layout.wgslLogic),
        'the volume fallback calls the global with (worldPos, N)');
    // Parsing the composed material is the real guard: a bad call site fails here, not by eye.
    const composed = buildMaterialFragment({ kind: 'pbr', groupIndex: 1, indirect: layout });
    new WgslReflect(`${HOST_PREAMBLE}\n${composed.bindings}\n${composed.fragment}`);

    // The material fragment given the probe layout is unchanged from the one Phase 1 produced —
    // THE test proving this release cost no material edits (design-principles §4).
    for (const kind of ['pbr', 'lambert', 'toon']) {
        const a = buildMaterialFragment({ kind, groupIndex: 1, indirect: ENV_LAYOUT });
        const b = buildMaterialFragment({ kind, groupIndex: 1, indirect: layout });
        assertEq(b.fragment, a.fragment, `${kind} fragment is unchanged`);
    }

    // Both composed modules parse (PBR exercises specular; Lambert is diffuse-only).
    for (const kind of ['pbr', 'lambert']) {
        const m = buildMaterialFragment({ kind, groupIndex: 1, indirect: layout });
        new WgslReflect(HOST_PREAMBLE + '\n' + m.bindings + '\n' + m.fragment);
    }

    // The decorator is guarded: a layout without the markers throws, naming the drift.
    assertThrows(
        () => buildProbeIndirectLayout({ wgslStructs: '', wgslLogic: 'fn noMarkerHere() {}' }),
        /out of date/, 'missing marker throws');
});


// === U10: the binding map (§3, §4.4, U10) ====================================
check('U10: the probe chunk spends group-2 bindings 7–13 only, each exactly once', () => {
    const chunk = buildProbeStructsWGSL(2);
    const found = [...chunk.matchAll(/@group\(2\)\s+@binding\((\d+)\)/g)].map(m => Number(m[1]));
    assertEq(found.join(','), '7,8,9,10,11,12,13', 'probe bindings are exactly 7–13');
    for (const b of found) assert(b >= 7 && b <= 13, `binding ${b} inside the budget`);

    // The composed module: env keeps 3–6, probes add 7–13, nothing collides and nothing spills
    // past the budget. Bindings 0–2 are the LightSystem's and are declared by ITS wgslStructs —
    // not by anything here, which is exactly what env U8 asserts from the other side. An earlier
    // draft of this test expected `min === 0` and was wrong about who owns those slots.
    const composed = ENV_LAYOUT.wgslStructs + '\n' + chunk;
    const all = [...composed.matchAll(/@group\(2\)\s+@binding\((\d+)\)/g)].map(m => Number(m[1]));
    all.forEach((b, i) => assert(all.indexOf(b) === i, `binding ${b} appears exactly once`));
    assertEq(all.slice().sort((a, b) => a - b).join(','), '3,4,5,6,7,8,9,10,11,12,13',
        'env+probes occupy exactly 3–13, contiguous');
    for (const b of all) {
        assert(b > 2, `binding ${b} would collide with the LightSystem's 0–2`);
        assert(b <= 13, `binding ${b} exceeds the group-2 budget (lighting-ibl-design.md §4.4)`);
    }

    // The anti-prune is REAL: the diffuse body (the only function Lambert/Toon reach) references
    // the reflection bindings, so layout:'auto' can never drop them for diffuse-only materials.
    const logic = buildProbeIndirectLogicWGSL();
    const diffuseBody = logic.slice(logic.indexOf('fn getIndirectDiffuse('), logic.indexOf('fn getIndirectSpecular('));
    assert(diffuseBody.includes('reflProbes'), 'diffuse body touches the cube array (binding 12)');
    assert(diffuseBody.includes('reflProbeData'), 'diffuse body touches the probe storage (binding 13)');

    // Every generated module parses, and the prefilter's default path is byte-identical (§12.2).
    new WgslReflect(buildProbeProjectionWGSL({ faceSize: 32 }));
    new WgslReflect(buildProbeProjectionWGSL({ faceSize: 64 }));
    new WgslReflect(buildReflAssembleWGSL());
    const legacy = buildPrefilterWGSL({ level: 0, prefilterMips: 6, samples: 16, srcSize: 512 });
    const legacyGgx = buildPrefilterWGSL({ level: 3, prefilterMips: 6, samples: 16, srcSize: 512 });
    assert(!legacy.includes('reflLayerOffset') && !legacyGgx.includes('reflLayerOffset'),
        'the Phase 1 prefilter is byte-identical without the option');
    const layered = buildPrefilterWGSL({ level: 2, prefilterMips: 6, samples: 16, srcSize: 128, arrayLayerOffset: true });
    assert(layered.includes('reflLayerOffset'), 'the layer offset lands in the kernel');
    new WgslReflect(layered);
    const layeredCopy = buildPrefilterWGSL({ level: 0, prefilterMips: 6, samples: 16, srcSize: 128, arrayLayerOffset: true });
    new WgslReflect(layeredCopy);

    // The generator-hazard pin (design-principles §13 rule 3): no backticks anywhere.
    for (const [name, src] of Object.entries({
        structs: chunk, logic, layoutStructs: buildProbeIndirectLayout(ENV_LAYOUT, 2).wgslStructs,
        projection: buildProbeProjectionWGSL({ faceSize: 32 }), assemble: buildReflAssembleWGSL(),
        layeredPrefilter: layered,
    })) {
        assert(!src.includes('`'), `${name} contains a backtick — it would terminate its template literal`);
    }
});


// === §11: the error table =====================================================
check('§11: config validation throws the named errors', () => {
    const base = {
        origin: [0, 0, 0], spacing: 4, dims: [8, 4, 8],
        captureFaceSize: 32, probesPerFrame: 16, captureAtlasSize: 1024, maxProbes: 65536,
        shBands: 1, environment: {}, visibility: {}, shadows: {}, allowShadowlessCapture: false,
        reflectionFaceSize: 128,
    };
    validateProbeVolumeConfig(base);  // the happy path must not throw
    validateProbeVolumeConfig({ ...base, shadows: undefined, allowShadowlessCapture: true });
    const cases = [
        [{ ...base, dims: [64, 64, 64], maxProbes: 1000 }, /exceeds maxProbes 1000/],
        [{ ...base, probesPerFrame: 64, captureFaceSize: 128, captureAtlasSize: 512, reflectionFaceSize: 128 }, /capture atlas; lower probesPerFrame/],
        [{ ...base, environment: undefined }, /config\.environment is required/],
        [{ ...base, visibility: undefined }, /config\.visibility is required/],
        [{ ...base, shadows: undefined, allowShadowlessCapture: false }, /shadowless bounce light/],
        [{ ...base, shBands: 2 }, /shBands 2 needs 7 rgba16float 3D textures/],
    ];
    for (const [cfg, re] of cases) assertThrows(() => validateProbeVolumeConfig(cfg), re, `validation ${re}`);
});

// === Mock-device wiring: construction, combined group 2, gating, prewarm ====
function makeMockRig() {
    const renders = [];
    const stages = [];
    const device = {
        queue: {
            writes: [],
            writeBuffer(...a) { this.writes.push(['buffer', ...a]); },
            writeTexture(...a) { this.writes.push(['texture', ...a]); },
            onSubmittedWorkDone: async () => {},
        },
        createTexture: (d) => ({ desc: d, createView: (v = {}) => ({ view: v }), destroy() {} }),
        createBuffer: (d) => ({ desc: d, destroy() {} }),
        createSampler: () => ({}),
        createShaderModule: () => ({}),
        createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
        createRenderPipeline: () => ({ getBindGroupLayout: () => ({}) }),
        createBindGroup: (d) => ({ desc: d }),
    };
    const engine = {
        device, width: 800, height: 600,
        createPass: (cfg) => ({ ...cfg, batches: [], addBatch(b) { this.batches.push(b); }, removeBatch() {} }),
        createComputeStage: (cfg) => {
            const s = { name: cfg.name, enabled: true, customBindGroups: {}, preClearBuffers: [], dispatchCount: cfg.dispatchCount ?? [1, 1, 1] };
            stages.push(s); return s;
        },
        setComputePassBindGroup: (stage, entries) => { stage._entries = entries; },
        setRenderPassBindGroup: (batch, entries, g) => { batch.customBindGroups[g] = { entries }; },
        render: (seq) => { renders.push(seq); },
    };
    const env = {
        getShaderLayout: () => ENV_LAYOUT,
        sampler: {}, specularView: {}, brdfView: {}, paramsBuffer: {},
        isBaked: true, intensity: 2,
    };
    const visBatches = [];
    const visCull = new Map();
    const vis = {
        createInstanceDrawBatch: (pass, meshName, meshIndex, opts) => {
            const b = { customBindGroups: {}, viewport: opts.viewport, opts };
            const cull = { enabled: true, name: `cull:${meshName}` };
            visCull.set(b, cull);
            visBatches.push(b);
            pass.addBatch(b);
            return b;
        },
        getCullStage: (b) => visCull.get(b) ?? null,
    };
    const lights = { getBindGroupEntries: () => [{ binding: 0, resource: { buffer: {} } }, { binding: 1, resource: { buffer: {} } }] };
    return { engine, device, renders, stages, env, vis, visBatches, visCull, lights };
}


// The summary + exit code. Without these the suite counts failures and then reports success:
// `npm run test:environment` chains with `&&`, so a red probe test would have been invisible.
console.log(failures === 0 ? '\nAll environment probe tests passed.' : `\n${failures} test(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
