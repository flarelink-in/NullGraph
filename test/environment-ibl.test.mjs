// Unit tests for the EnvironmentSystem — Phase 1 IBL (docs/lighting-ibl-design.md §10).
// The bake is pure math over known inputs, so U1–U9 run headless in Node. The GPU integration
// tests (§10 T1–T6) need a browser/WebGPU harness; here every generated WGSL module is parsed
// with wgsl_reflect and the CPU mirrors are pinned by number.
//   npm run build:lib && node test/environment-ibl.test.mjs
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';
import {
    faceDirection, dirToEquirectUV, cubeTexelSolidAngle, shBasisL2, projectSHL2FromCube,
    evalSHL2, rotateY, getIndirectDiffuseCPU, packSHChannel, packEnvParams,
    integrateBRDFCPU, hammersley, validateEnvironmentConfig,
    ENV_PARAMS_FLOATS, SH_SOURCE_SIZE,
    buildSourceCubeWGSL, buildCubeMipWGSL, buildPrefilterWGSL, buildBRDFLUTWGSL,
    buildSHProjectWGSL, buildIndirectStructsWGSL, buildIndirectLogicWGSL, buildSkyWGSL,
    ENV_MATH_WGSL, ENV_SAMPLING_WGSL,
    buildGradientSkyWGSL, buildEquirectSkyBakeWGSL, SKY_PARAM_FLOATS,
} from '../dist/environment.js';
import { buildMaterialFragment } from '../dist/materials.js';

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
    if (!(Math.abs(a - b) <= eps)) throw new Error(`${what}: expected ~${b}, got ${a}`);
}

// A camera + VertexOut preamble, so a material fragment can be parsed as a whole module.
const HOST_PREAMBLE = `struct Camera { viewProj: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>,
 invViewProj: mat4x4<f32>, invView: mat4x4<f32>, invProj: mat4x4<f32>, eyePos: vec3<f32>,
 simTime: f32, zNear: f32, zFar: f32, fovRads: f32, aspect: f32, _padEnd: vec4<f32> };
@group(0) @binding(0) var<uniform> camera: Camera;
struct VertexOut { @builtin(position) clipPos: vec4<f32>, @location(0) worldPos: vec3<f32>,
 @location(1) worldNormal: vec3<f32>, @location(2) uv: vec2<f32>, @location(3) ecsColor: vec3<f32> };
@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }`;

// === U1: cube face directions (§6.1) ========================================
check('U1: faceDirection maps the six faces to +X,-X,+Y,-Y,+Z,-Z', () => {
    const expect = [
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    ];
    for (let f = 0; f < 6; f++) {
        const d = faceDirection(f, 0.5, 0.5);
        for (let i = 0; i < 3; i++) {
            near(d[i], expect[f][i], 1e-6, `face ${f} centre component ${i}`);
        }
    }
    // Every corner of every face is unit length and in the correct octant for its major axis.
    for (let f = 0; f < 6; f++) {
        for (const [u, v] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
            const d = faceDirection(f, u, v);
            near(Math.hypot(d[0], d[1], d[2]), 1, 1e-6, `face ${f} corner is unit length`);
            const axis = f >> 1, sign = f % 2 === 0 ? 1 : -1;
            assert(Math.sign(d[axis]) === sign, `face ${f} corner major axis sign`);
            assert(Math.abs(d[axis]) >= Math.abs(d[(axis + 1) % 3]) - 1e-6,
                `face ${f} corner: major axis dominates`);
        }
    }
});

// === U2: equirect round-trip (§6.1) =========================================
check('U2: dirToEquirectUV(faceDirection(...)) round-trips to the same direction', () => {
    const uvToDir = (u, v) => {
        const phi = (u - 0.5) * 2 * Math.PI;
        const theta = v * Math.PI;
        return [Math.sin(theta) * Math.cos(phi), Math.cos(theta), Math.sin(theta) * Math.sin(phi)];
    };
    let worst = 0;
    for (let i = 0; i < 1000; i++) {
        const f = i % 6;
        const u = ((i * 37) % 97) / 97, v = ((i * 53) % 89) / 89;
        const d = faceDirection(f, u, v);
        const uv = dirToEquirectUV(d);
        const back = uvToDir(uv[0], uv[1]);
        worst = Math.max(worst, Math.hypot(back[0] - d[0], back[1] - d[1], back[2] - d[2]));
    }
    assert(worst < 1e-5, `worst round-trip error ${worst} (want < 1e-5)`);
});

// === U3: solid angles sum to 4π (§7.2) ======================================
check('U3: cube texel solid angles over 6 faces sum to 4π (a uniform weight does not)', () => {
    const S = SH_SOURCE_SIZE;
    let total = 0;
    for (let f = 0; f < 6; f++) {
        for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) total += cubeTexelSolidAngle(x, y, S);
    }
    near(total, 4 * Math.PI, 1e-4, 'Σ solid angle');
    // The corner/centre ratio is the thing a uniform weight throws away.
    const centre = cubeTexelSolidAngle(S / 2, S / 2, S);
    const corner = cubeTexelSolidAngle(0, 0, S);
    assert(corner < centre * 0.6, `corner texel (${corner}) must subtend far less than centre (${centre})`);
});

// === U4: SH packing round-trip (§7.3) =======================================
check('U4: packSHChannel order matches the shCoeff(k) the generated WGSL indexes', () => {
    // 9 coefficients × 3 channels, each distinguishable.
    const coeffs = new Float32Array(27);
    for (let k = 0; k < 9; k++) for (let c = 0; c < 3; c++) coeffs[k * 3 + c] = k * 10 + c;

    const packed = new Float32Array(12);
    packSHChannel(coeffs, 1, packed, 0);              // green channel
    // shCoeffG(k) in WGSL reads env.shG[k/3][k%3] — reproduce that indexing exactly.
    for (let k = 0; k < 9; k++) {
        const viaWGSL = packed[Math.floor(k / 3) * 4 + (k % 3)];
        assertEq(viaWGSL, k * 10 + 1, `coefficient ${k} of the green channel`);
    }
    // And the unused 12th slot of each vec4 stays zero.
    for (let v = 0; v < 3; v++) assertEq(packed[v * 4 + 3], 0, `vec4 ${v} pad slot`);

    const params = packEnvParams(coeffs, 0.5, 2.0, 5.0, 1.0);
    assertEq(params.length, ENV_PARAMS_FLOATS, 'EnvParams float count');
    assertEq(params[39], 0.5, 'yaw at float 39');
    assertEq(params[40], 2.0, 'intensity at float 40');
    assertEq(params[41], 5.0, 'specularMip at float 41');
    assertEq(params[42], 1.0, 'horizonFade at float 42');
});

// === U5: SH ringing is clamped (§7.4) =======================================
check('U5: a split-hemisphere environment rings negative, and the shader mirror clamps it', () => {
    // White above the horizon, black below — the classic high-contrast ringing case.
    const S = 8, cube = new Float32Array(6 * S * S * 4);
    for (let f = 0; f < 6; f++) {
        for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
            const d = faceDirection(f, (x + 0.5) / S, (y + 0.5) / S);
            const lit = d[1] > 0 ? 1 : 0;
            const t = (f * S * S + y * S + x) * 4;
            cube[t] = lit; cube[t + 1] = lit; cube[t + 2] = lit; cube[t + 3] = 1;
        }
    }
    const sh = projectSHL2FromCube(cube, S);
    // Raw evaluation must go negative SOMEWHERE (that is the ringing this clamp defends against).
    let sawNegative = false;
    for (let i = 0; i < 500; i++) {
        const a = (i / 500) * Math.PI * 2, b = (i % 50) / 50 * Math.PI;
        const N = [Math.sin(b) * Math.cos(a), Math.cos(b), Math.sin(b) * Math.sin(a)];
        if (evalSHL2(sh, N)[0] < -1e-4) { sawNegative = true; break; }
    }
    assert(sawNegative, 'the raw L2 evaluation should ring negative for this environment');
    // The clamped mirror — what the shader actually runs — never does.
    for (let i = 0; i < 500; i++) {
        const a = (i / 500) * Math.PI * 2, b = (i % 50) / 50 * Math.PI;
        const N = [Math.sin(b) * Math.cos(a), Math.cos(b), Math.sin(b) * Math.sin(a)];
        const c = getIndirectDiffuseCPU(sh, 0, 1, N);
        assert(c[0] >= 0 && c[1] >= 0 && c[2] >= 0, `clamped result went negative at sample ${i}`);
    }
});

// === U6: SH of a constant environment (§7.2) ================================
check('U6: a uniform radiance c projects to irradiance c in every direction', () => {
    const S = 8, C = 0.75;
    const cube = new Float32Array(6 * S * S * 4).fill(C);
    const sh = projectSHL2FromCube(cube, S);
    // This single number validates the cosine-convolution constants A0/A1/A2 end to end.
    for (let i = 0; i < 200; i++) {
        const a = (i / 200) * Math.PI * 2, b = (i % 20) / 20 * Math.PI;
        const N = [Math.sin(b) * Math.cos(a), Math.cos(b), Math.sin(b) * Math.sin(a)];
        const got = getIndirectDiffuseCPU(sh, 0, 1, N);
        // Stored value is irradiance = π·radiance for a constant field; the shader divides by π
        // implicitly through the A0 constant, so the evaluated result is the radiance back.
        near(got[0] / Math.PI, C, C * 0.01, `direction ${i} red`);
    }
});

check('U6b: rotation moves the SH response by exactly the yaw', () => {
    // A bright +X lobe: rotating the environment by +90° about Y must move the response to +Z.
    const S = 8, cube = new Float32Array(6 * S * S * 4);
    for (let f = 0; f < 6; f++) {
        for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
            const d = faceDirection(f, (x + 0.5) / S, (y + 0.5) / S);
            const v = Math.max(0, d[0]) ** 4;
            const t = (f * S * S + y * S + x) * 4;
            cube[t] = v; cube[t + 1] = v; cube[t + 2] = v; cube[t + 3] = 1;
        }
    }
    const sh = projectSHL2FromCube(cube, S);
    const atX = getIndirectDiffuseCPU(sh, 0, 1, [1, 0, 0])[0];
    const rotated = getIndirectDiffuseCPU(sh, Math.PI / 2, 1, rotateY([1, 0, 0], Math.PI / 2));
    near(rotated[0], atX, atX * 0.02, 'rotating the env and the normal together is a no-op');
});

// === U7: BRDF LUT endpoints (§6.4) ==========================================
check('U7: the BRDF LUT is (scale,bias)→(1,0) at roughness 0 and monotonic in NdotV', () => {
    const [s0, b0] = integrateBRDFCPU(1.0, 0.0, 512);
    near(s0, 1.0, 0.02, 'scale at roughness 0, NdotV 1');
    near(b0, 0.0, 0.02, 'bias at roughness 0, NdotV 1');
    // Monotonic in NdotV at fixed roughness — DECREASING. The split-sum A term is largest at
    // grazing incidence (it exceeds 1 there, which is correct: the Fresnel-weighted lobe gathers
    // more than the normal-incidence reflectance) and falls toward ~0.77 head-on. Asserting the
    // opposite direction is the mistake this comment exists to prevent; it was made once here.
    let prev = Infinity, ok = true;
    for (let i = 1; i <= 16; i++) {
        const [s] = integrateBRDFCPU(i / 16, 0.5, 512);
        if (s > prev + 0.02) { ok = false; break; }
        prev = s;
    }
    assert(ok, 'scale should be non-increasing in NdotV at roughness 0.5');
    // The bias term decays to ~0 head-on — it is the F90 lobe, which only matters at grazing.
    const [, biasGrazing] = integrateBRDFCPU(0.06, 0.5, 512);
    const [, biasHeadOn] = integrateBRDFCPU(1.0, 0.5, 512);
    assert(biasGrazing > biasHeadOn * 10, 'bias is a grazing-angle term');
    // Hammersley is a real low-discrepancy sequence, not a constant.
    const h0 = hammersley(0, 64), h1 = hammersley(1, 64);
    assert(h0[1] !== h1[1], 'radical inverse must vary');
    assert(h0[1] >= 0 && h0[1] < 1, 'radical inverse in [0,1)');
});

// === U8: WGSL structure + the binding budget (§4.4) =========================
check('U8: every generated module parses, and env bindings never touch the lights at 0–2', () => {
    const mods = {
        'sourceCube(gradient)': buildSourceCubeWGSL(buildGradientSkyWGSL(SKY_PARAM_FLOATS)),
        'sourceCube(equirect)': buildSourceCubeWGSL(buildEquirectSkyBakeWGSL()),
        'cubeMip': buildCubeMipWGSL(),
        'prefilter mip0': buildPrefilterWGSL({ level: 0, prefilterMips: 6, samples: 64, srcSize: 512 }),
        'prefilter mip3': buildPrefilterWGSL({ level: 3, prefilterMips: 6, samples: 64, srcSize: 512 }),
        'brdfLUT': buildBRDFLUTWGSL({ samples: 256 }),
        'shProject': buildSHProjectWGSL(),
        'sky': buildSkyWGSL({ environmentGroup: 2 }),
    };
    for (const [name, src] of Object.entries(mods)) {
        try { new WgslReflect(src); }
        catch (e) { throw new Error(`${name} failed to parse: ${String(e.message || e).split('\n')[0]}`); }
    }
    // The indirect layout composed into a real PBR fragment must parse as one module.
    const indirect = { wgslStructs: buildIndirectStructsWGSL(2), wgslLogic: buildIndirectLogicWGSL() };
    const { bindings, fragment } = buildMaterialFragment({ kind: 'pbr', groupIndex: 1, indirect });
    new WgslReflect(`${HOST_PREAMBLE}\n${bindings}\n${fragment}`);

    // §4.4: the environment owns group 2 bindings 3–6. Bindings 0–2 belong to the LightSystem,
    // and claiming one would silently overwrite the lights in the combined bind group.
    const structs = buildIndirectStructsWGSL(2);
    const claimed = [...structs.matchAll(/@group\(2\)\s*@binding\((\d+)\)/g)].map(m => +m[1]).sort();
    assertEq(JSON.stringify(claimed), JSON.stringify([3, 4, 5, 6]), 'environment claims exactly bindings 3–6');

    // Both contract functions are declared exactly once.
    for (const fn of ['getIndirectDiffuse', 'getIndirectSpecular']) {
        const n = (indirect.wgslLogic.match(new RegExp(`fn\\s+${fn}\\b`, 'g')) || []).length;
        assertEq(n, 1, `${fn} declared exactly once`);
    }
});

// === U9: the material hook is opt-in and byte-identical when absent (§5.3) ==
check('U9: `indirect` swaps the hardcoded ambient; without it the output is unchanged', () => {
    const indirect = { wgslStructs: buildIndirectStructsWGSL(2), wgslLogic: buildIndirectLogicWGSL() };
    const withIbl = buildMaterialFragment({ kind: 'pbr', groupIndex: 1, indirect });
    const without = buildMaterialFragment({ kind: 'pbr', groupIndex: 1 });

    assert(withIbl.fragment.includes('getIndirectDiffuse('), 'with indirect: calls getIndirectDiffuse');
    assert(withIbl.fragment.includes('getIndirectSpecular('), 'with indirect: calls getIndirectSpecular');
    assert(!withIbl.fragment.includes('vec3<f32>(0.3) * albedo'), 'with indirect: the 0.3 constant is gone');

    assert(without.fragment.includes('vec3<f32>(0.3) * albedo'), 'without indirect: the constant remains');
    assert(!without.fragment.includes('getIndirectDiffuse('), 'without indirect: no IBL call');
    assert(!without.bindings.includes('envSpecular'), 'without indirect: no env bindings leak in');

    // Lambert and Toon also take the diffuse half.
    for (const kind of ['lambert', 'toon']) {
        const w = buildMaterialFragment({ kind, groupIndex: 1, indirect });
        assert(w.fragment.includes('getIndirectDiffuse('), `${kind} consumes the indirect diffuse`);
    }
});

// === U10: the backtick trap — a generator-authoring hazard, pinned ==========
// A backtick inside a WGSL comment TERMINATES the JS template literal that carries it. It has
// bitten this file twice (once in ENV_SAMPLING_WGSL, once in the indirect logic), and the symptom
// is a cascade of TypeScript syntax errors pointing at WGSL, which reads as a corrupted file
// rather than a quoting mistake. Cheap to pin, so pin it.
check('U10: no generated WGSL contains a backtick (it would break the template literal)', () => {
    const all = {
        ENV_MATH_WGSL, ENV_SAMPLING_WGSL,
        sourceCubeProc: buildSourceCubeWGSL(buildGradientSkyWGSL(SKY_PARAM_FLOATS)),
        sourceCubeEqui: buildSourceCubeWGSL(buildEquirectSkyBakeWGSL()),
        cubeMip: buildCubeMipWGSL(),
        prefilter0: buildPrefilterWGSL({ level: 0, prefilterMips: 6, samples: 64, srcSize: 512 }),
        prefilter3: buildPrefilterWGSL({ level: 3, prefilterMips: 6, samples: 64, srcSize: 512 }),
        brdfLUT: buildBRDFLUTWGSL({}),
        shProject: buildSHProjectWGSL(),
        indirectStructs: buildIndirectStructsWGSL(2),
        indirectLogic: buildIndirectLogicWGSL(),
        sky: buildSkyWGSL({ environmentGroup: 2 }),
    };
    for (const [name, src] of Object.entries(all)) {
        assert(!src.includes('`'), `${name} contains a backtick — it would terminate its template literal`);
    }
});

// === §11: config validation ==================================================
check('§11: config validation throws the named errors', () => {
    const base = {
        sourceSize: 512, prefilterMips: 6, environmentGroup: 2,
        updateMode: 'once', keepSourceCube: false,
    };
    validateEnvironmentConfig(base);  // the happy path must not throw

    const cases = [
        [{ ...base, sourceSize: 500 }, /power of two/],
        [{ ...base, sourceSize: 4 }, /power of two/],
        [{ ...base, prefilterMips: 20 }, /exceeds log2/],
        [{ ...base, environmentGroup: 3 }, /must equal the LightSystem's group/],
        [{ ...base, format: 'rgba32float' }, /float32-filterable/],
        [{ ...base, updateMode: 'amortized', keepSourceCube: false }, /requires keepSourceCube/],
    ];
    for (const [cfg, re] of cases) {
        let threw = null;
        try { validateEnvironmentConfig(cfg); } catch (e) { threw = e.message; }
        assert(threw && re.test(threw), `expected ${re}, got ${threw ?? 'no throw'}`);
    }
});

console.log(failures === 0 ? '\nAll environment IBL tests passed.' : `\n${failures} test(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
