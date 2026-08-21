// Unit tests for the sky source seam — Phase 1 (docs/environment-sky-design.md §8 U1–U11).
// All pure math + every generated WGSL module (parse-validated with wgsl_reflect) + the system
// logic reachable through a mock device. The GPU integration half (T1–T5) lives in
// test/environment-gpu/ and needs a browser; here the CPU mirrors are pinned by number and the
// WGSL the seam emits is pinned structurally, per design-principles §13.1.
//   npm run build:lib && node test/environment-sky.test.mjs
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';
import {
    // seam + builders
    buildSourceCubeWGSL, buildSkyWGSL, ENV_MATH_WGSL,
    buildGradientSkyWGSL, buildPreethamSkyWGSL, buildEquirectSkyWGSL, buildEquirectSkyBakeWGSL,
    // sources + types
    GradientSky, PreethamSky, EquirectSky, EnvironmentSystem,
    // CPU mirrors + constants
    gradientSkyColor, preethamSkyColor, preethamZenith, airMassKastenYoung, sunRadiance,
    KASTEN_YOUNG, SKY_PARAM_FLOATS, GRADIENT_SKY_FLOATS, PREETHAM_SKY_FLOATS, PREETHAM_OFF,
} from '../dist/environment.js';

// Node has no WebGPU globals; EnvironmentSystem references the usage enums when it allocates.
// The values are the standard bitmasks — the mock device ignores them, but they must exist.
globalThis.GPUTextureUsage = globalThis.GPUTextureUsage ?? {
    COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16,
};
globalThis.GPUBufferUsage = globalThis.GPUBufferUsage ?? {
    MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, INDEX: 32, VERTEX: 64, UNIFORM: 128,
    STORAGE: 256, INDIRECT: 512,
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
    if (!(Math.abs(a - b) <= eps)) throw new Error(`${what}: expected ~${b}, got ${a}`);
}
function clamp(x, lo, hi) { return Math.min(hi, Math.max(lo, x)); }
function mix(a, b, t) { return a + (b - a) * t; }
function mix3(a, b, t) { return [mix(a[0], b[0], t), mix(a[1], b[1], t), mix(a[2], b[2], t)]; }
function smoothstep(e0, e1, x) { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); }
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// ---------------------------------------------------------------------------
// The PRE-REFACTOR skyColor body, transcribed verbatim from the old envShaders.ts procedural
// kernel — the reference U1 holds the refactor against. If this transcription and the new CPU
// mirror agree to 1e-6 at 64 directions, the refactor is numerically inert (design §5.1).
// ---------------------------------------------------------------------------
function oldSkyColor(dir, cfg) {
    // The old kernel received a NORMALIZED sunDir (setProceduralSky normalized before upload),
    // so reproduce that here or the disc smoothstep lands on a different dot and the test lies.
    const sl = Math.hypot(cfg.sunDir[0], cfg.sunDir[1], cfg.sunDir[2]) || 1;
    const sd = [cfg.sunDir[0] / sl, cfg.sunDir[1] / sl, cfg.sunDir[2] / sl];
    const t = clamp(dir[1], -1.0, 1.0);
    let c = mix3(cfg.horizon, cfg.zenith, Math.pow(Math.max(t, 0.0), 0.6));
    c = mix3(c, cfg.ground, Math.pow(Math.max(-t, 0.0), 0.5));
    const sunAmt = smoothstep(cfg.sunCos, cfg.sunCos + 0.0003, dot3(dir, [-sd[0], -sd[1], -sd[2]]));
    return [
        c[0] + cfg.sunIntensity * sunAmt * 1.0,
        c[1] + cfg.sunIntensity * sunAmt * 0.98,
        c[2] + cfg.sunIntensity * sunAmt * 0.95,
    ];
}

// 64 deterministic directions: 6 poles + 58 on a Fibonacci sphere — covers up, down, and oblique.
const DIRS = (() => {
    const out = [[0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]];
    const N = 58, GA = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < N; i++) {
        const y = 1 - 2 * (i + 0.5) / N, r = Math.sqrt(1 - y * y), a = GA * i;
        out.push([Math.cos(a) * r, y, Math.sin(a) * r]);
    }
    return out;
})();

// === U1: the GradientSky refactor is numerically inert (§5.1) ================
check('U1: GradientSky CPU mirror matches the pre-refactor skyColor at 64 directions to 1e-6', () => {
    const cfg = {
        zenith: [0.10, 0.22, 0.52], horizon: [0.65, 0.72, 0.85], ground: [0.28, 0.24, 0.20],
        sunDir: [-0.4, -1.0, -0.35], sunIntensity: 18,
    };
    const source = new GradientSky(cfg);
    const params = new Float32Array(SKY_PARAM_FLOATS + source.paramFloats);
    source.writeParams(params, SKY_PARAM_FLOATS);
    // sunCos default (0.9995) is the shipped default the old kernel used.
    const ref = { ...cfg, sunCos: 0.9995 };
    let maxErr = 0;
    for (const dir of DIRS) {
        const got = gradientSkyColor(dir, params, SKY_PARAM_FLOATS);
        const want = oldSkyColor(dir, ref);
        for (let i = 0; i < 3; i++) maxErr = Math.max(maxErr, Math.abs(got[i] - want[i]));
    }
    assert(maxErr <= 1e-6, `max channel error ${maxErr} exceeds 1e-6 — the refactor moved the sky`);
});

// === U2: one evaluator, byte-identical in both consumers (§6, principle 1) ===
function sampleSkyBody(module) {
    const i = module.indexOf('fn sampleSky(');
    assert(i >= 0, 'module lacks fn sampleSky');
    return module.slice(i);
}
check('U2: bake kernel and skybox fragment carry byte-identical sampleSky bodies', () => {
    for (const source of [new GradientSky({
        zenith: [0.1, 0.2, 0.5], horizon: [0.6, 0.7, 0.8], ground: [0.2, 0.2, 0.2],
        sunDir: [0, -1, 0], sunIntensity: 10,
    }), new PreethamSky()]) {
        const layout = source.getShaderLayout(SKY_PARAM_FLOATS);
        const bake = buildSourceCubeWGSL(layout);
        const skybox = buildSkyWGSL({ environmentGroup: 2, sky: layout });
        // The seam injects the SAME wgslLogic object into both — the bodies cannot drift.
        assert(bake.includes(layout.wgslLogic), `${source.name}: bake dropped the shared logic`);
        assert(skybox.includes(layout.wgslLogic), `${source.name}: skybox dropped the shared logic`);
        assertEq(sampleSkyBody(bake).startsWith('fn sampleSky('), true, `${source.name} bake body`);
    }
    // EquirectSky is the ONE sanctioned divergence (§5.2): the bake writes the cube, so it reads
    // the 2D asset; the skybox reads the cube. Pin both facets exactly.
    const equi = new EquirectSky({ equirect: {} });
    const skyL = equi.getShaderLayout(SKY_PARAM_FLOATS);
    const bakeL = equi.getBakeShaderLayout(SKY_PARAM_FLOATS);
    assert(skyL.wgslLogic.includes('skySrcCube'), 'equirect skybox must sample the source cube');
    assert(bakeL.wgslLogic.includes('dirToEquirectUV('), 'equirect bake must sample the 2D asset');
    assert(skyL.wgslLogic !== bakeL.wgslLogic, 'equirect facets are documented-different');
});


// === U3: every shipped source's composed modules parse (§8) ==================
check('U3: composed bake kernels and skybox modules parse for every shipped source', () => {
    const sources = [
        new GradientSky({ zenith: [0.1, 0.2, 0.5], horizon: [0.6, 0.7, 0.8], ground: [0.2, 0.2, 0.2], sunDir: [0, -1, 0], sunIntensity: 10 }),
        new PreethamSky(),
        new EquirectSky({ equirect: {} }),
    ];
    for (const s of sources) {
        const bake = buildSourceCubeWGSL(s.getBakeShaderLayout
            ? s.getBakeShaderLayout(SKY_PARAM_FLOATS) : s.getShaderLayout(SKY_PARAM_FLOATS));
        const skybox = buildSkyWGSL({ environmentGroup: 2, sky: s.getShaderLayout(SKY_PARAM_FLOATS) });
        for (const [name, src] of [[`${s.name} bake`, bake], [`${s.name} skybox`, skybox]]) {
            try { new WgslReflect(src); }
            catch (e) { throw new Error(`${name} failed to parse: ${String(e.message || e).split('\n')[0]}`); }
        }
    }
    // The pre-Phase-1 skybox (no source) still parses — it is the cheapSky/T4 baseline.
    new WgslReflect(buildSkyWGSL({ environmentGroup: 2 }));
});

// === U4: Preetham zenith luminance monotonicity (§5.3) =======================
check('U4: Preetham zenith luminance rises with turbidity and with sin(solar elevation)', () => {
    // Fixed solar elevation (thetaS = 45°): Yz strictly increases with turbidity.
    const thetaS = Math.PI / 4;
    let prev = -Infinity;
    for (const T of [1.8, 2.5, 3.5, 5, 7, 10]) {
        const Yz = preethamZenith(T, thetaS)[0];
        assert(Yz > prev, `Yz(${T})=${Yz} not > Yz(prev)=${prev} (turbidity sweep)`);
        prev = Yz;
    }
    // Fixed turbidity: Yz strictly increases with sin(elevation) = cos(thetaS) as the sun rises.
    const T = 2.8;
    prev = -Infinity;
    for (const elevDeg of [5, 15, 30, 45, 60, 75, 85]) {
        const th = (90 - elevDeg) * Math.PI / 180;      // solar zenith angle
        const Yz = preethamZenith(T, th)[0];
        assert(Yz > prev, `Yz(elev=${elevDeg})=${Yz} not increasing with elevation`);
        prev = Yz;
    }
});


// === U5: finite + non-negative everywhere, straight down included (§4.2) ====
const MANY_DIRS = (() => {
    const out = [];
    const N = 500, GA = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < N; i++) {
        const y = 1 - 2 * (i + 0.5) / N, r = Math.sqrt(1 - y * y), a = GA * i;
        out.push([Math.cos(a) * r, y, Math.sin(a) * r]);
    }
    out.push([0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1],
        [0.577, 0.577, 0.577], [-0.577, -0.577, 0.577], [0.707, -0.707, 0], [0, 0.707, -0.707],
        [0.001, -0.999, 0.044], [-0.3, 0.1, 0.95]);
    return out; // 512 directions
})();

function sunTravelDir(elevDeg) {
    const e = elevDeg * Math.PI / 180;
    return [-Math.cos(e), -Math.sin(e), 0];   // sun at elevation e, azimuth 0; travel = -toSun
}

check('U5: analytic sources are finite & non-negative for 512 dirs x 32 sun elevations (-20..90)', () => {
    assertEq(MANY_DIRS.length, 512, 'direction count');
    const elevations = [];
    for (let i = 0; i < 32; i++) elevations.push(-20 + (110 * i) / 31);

    const grad = new GradientSky({
        zenith: [0.1, 0.2, 0.5], horizon: [0.6, 0.7, 0.8], ground: [0.2, 0.2, 0.2],
        sunDir: [0, -1, 0], sunIntensity: 18,
    });
    const preetham = new PreethamSky({ turbidity: 10 });   // worst case: highest shipped turbidity

    for (const source of [grad, preetham]) {
        const params = new Float32Array(SKY_PARAM_FLOATS + source.paramFloats);
        for (const elev of elevations) {
            source.setSunDirection(sunTravelDir(elev));
            source.writeParams(params, SKY_PARAM_FLOATS);
            for (const dir of MANY_DIRS) {
                const rgb = source === grad
                    ? gradientSkyColor(dir, params, SKY_PARAM_FLOATS)
                    : preethamSkyColor(dir, params, SKY_PARAM_FLOATS);
                for (let i = 0; i < 3; i++) {
                    assert(Number.isFinite(rgb[i]), `${source.name} non-finite at elev=${elev}`);
                    assert(rgb[i] >= 0, `${source.name} negative ${rgb[i]} at elev=${elev}`);
                }
            }
        }
    }
});

// === U6: skySunRadiance CPU mirror matches the WGSL (§4.2, §13.1) ===========
// Headless cannot run WGSL, so we pull the WGSL's own constants back out of the generated body
// and re-evaluate them in JS — the CPU mirror must reproduce them to 1e-5 at 32 elevations.
function extractSunRadianceConsts(wgsl) {
    // Body shape: 1/(sin(h*A) + B*pow(h + C, -D))  → capture A, B, C, D.
    const m = wgsl.match(/sin\(h \* ([\d.eE+-]+)\) \+ ([\d.eE+-]+) \* pow\(h \+ ([\d.eE+-]+), -([\d.eE+-]+)\)/);
    assert(m, 'could not locate the Kasten-Young expression in the WGSL');
    return { A: +m[1], B: +m[2], C: +m[3], D: +m[4] };
}

check('U6: sunRadiance CPU mirror matches the WGSL constants at 32 elevations to 1e-5', () => {
    const layout = new PreethamSky().getShaderLayout(SKY_PARAM_FLOATS);
    const body = layout.wgslLogic.slice(layout.wgslLogic.indexOf('fn skySunRadiance('));
    const { A, B, C, D } = extractSunRadianceConsts(body);

    // The WGSL must embed the skyMath-owned constants (single source of truth): deg→rad for the
    // sin, and the Kasten–Young a/b/c for the air mass.
    near(A, Math.PI / 180, 1e-12, 'WGSL deg-to-rad constant');
    near(B, KASTEN_YOUNG[0], 1e-9, 'WGSL Kasten-Young a constant');
    near(C, KASTEN_YOUNG[1], 1e-9, 'WGSL Kasten-Young b constant');
    near(D, KASTEN_YOUNG[2], 1e-9, 'WGSL Kasten-Young c constant');

    // Independent JS evaluation of the SAME expression the WGSL runs, compared to the mirror.
    const beta = [0.05, 0.10, 0.20];
    let maxErr = 0;
    for (let i = 0; i < 32; i++) {
        const elev = -20 + (110 * i) / 31;
        const h = Math.max(elev, -2.0);
        const am = 1.0 / (Math.sin(h * A) + B * Math.pow(h + C, -D));
        const want = [Math.exp(-beta[0] * am), Math.exp(-beta[1] * am), Math.exp(-beta[2] * am)];
        const got = sunRadiance(elev, beta);
        for (let c = 0; c < 3; c++) maxErr = Math.max(maxErr, Math.abs(got[c] - want[c]));
    }
    assert(maxErr <= 1e-5, `mirror vs WGSL-expression error ${maxErr} exceeds 1e-5`);
});


// === U7: no PI redeclaration, no collision with ENV_MATH_WGSL (§4.2) =======
check('U7: no source declares PI or collides with an ENV_MATH_WGSL identifier', () => {
    const envMathNames = ['faceDirection', 'dirToEquirectUV', 'areaElement',
        'cubeTexelSolidAngle', 'shBasisL2', 'rotateY'];
    const sources = [
        buildGradientSkyWGSL(SKY_PARAM_FLOATS),
        buildPreethamSkyWGSL(SKY_PARAM_FLOATS),
        buildEquirectSkyWGSL(),
        buildEquirectSkyBakeWGSL(),
    ];
    for (const { wgslStructs, wgslLogic } of sources) {
        const chunk = wgslStructs + wgslLogic;
        assert(!/const\s+PI\b/.test(chunk), 'a source redeclares PI (the preamble owns it)');
        for (const name of envMathNames) {
            const re = new RegExp(`fn\\s+${name}\\s*\\(`);
            assert(!re.test(chunk), `a source redeclares ENV_MATH helper "${name}"`);
        }
        // sky_ prefix discipline: every helper a source declares (except the two entry points)
        // must be sky_-prefixed, so a Phase 3 composite can rename without collision.
        const declared = [...chunk.matchAll(/\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/g)].map(m => m[1]);
        for (const fn of declared) {
            assert(fn === 'sampleSky' || fn === 'skySunRadiance' || fn.startsWith('sky_'),
                `helper "${fn}" is not sky_-prefixed nor a seam entry point`);
        }
    }
});

// === U8: the backtick trap, pinned across every sky generator (§4.2) ========
check('U8: no generated sky WGSL contains a backtick (it would end the template literal)', () => {
    const all = {
        ENV_MATH_WGSL,
        gradient: buildGradientSkyWGSL(SKY_PARAM_FLOATS),
        preetham: buildPreethamSkyWGSL(SKY_PARAM_FLOATS),
        equirectSky: buildEquirectSkyWGSL(),
        equirectBake: buildEquirectSkyBakeWGSL(),
    };
    const composed = {
        gradientBake: buildSourceCubeWGSL(all.gradient),
        preethamSkybox: buildSkyWGSL({ environmentGroup: 2, sky: all.preetham }),
        equirectSkybox: buildSkyWGSL({ environmentGroup: 2, sky: all.equirectSky }),
    };
    for (const [name, src] of Object.entries({ ...all, ...composed })) {
        const text = typeof src === 'string' ? src : src.wgslStructs + src.wgslLogic;
        assert(!text.includes('`'), `${name} contains a backtick`);
    }
});


// ---------------------------------------------------------------------------
// Mock-device wiring: the system logic reachable headless (§11's build-time errors)
// ---------------------------------------------------------------------------
function makeMockEngine() {
    const device = {
        queue: { writeBuffer() {}, writeTexture() {}, onSubmittedWorkDone: async () => {} },
        createTexture: (d) => ({ desc: d, createView: () => ({ view: true }), destroy() {} }),
        createBuffer: (d) => ({ desc: d, destroy() {} }),
        createSampler: () => ({ sampler: true }),
        createShaderModule: () => ({}),
        createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
        createRenderPipeline: () => ({ getBindGroupLayout: () => ({}) }),
        createBindGroup: (d) => ({ desc: d }),
    };
    const engine = {
        device,
        createPass: (cfg) => ({ ...cfg, batches: [], addBatch(b) { this.batches.push(b); }, removeBatch() {} }),
        createComputeStage: (cfg) => ({
            name: cfg.name, enabled: true, customBindGroups: {}, preClearBuffers: [],
            dispatchCount: cfg.dispatchCount ?? [1, 1, 1],
        }),
        setComputePassBindGroup: (stage, entries) => { stage._entries = entries; },
        createBatch: (pass, cfg) => {
            const b = {
                pipeline: { getBindGroupLayout: () => ({}) }, storageBuffer: { buffer: true },
                customBindGroups: {}, stride: cfg.strideFloats, maxInstanceCount: cfg.maxInstances,
                currentInstanceCount: 0,
            };
            pass.addBatch(b);
            return b;
        },
        setRenderPassBindGroup: (batch, entries, g) => { batch.customBindGroups[g] = { entries }; },
        updateBatchData: (batch, data, count) => { batch.currentInstanceCount = count; },
    };
    return { engine, device };
}
function assertThrows(fn, re, what) {
    let msg = null;
    try { fn(); } catch (e) { msg = e.message; }
    assert(msg !== null, `${what}: expected a throw, got none`);
    assert(re.test(msg), `${what}: expected ${re}, got "${msg}"`);
}

// === U9: EquirectSky on a keepSourceCube:false system throws, naming the flag (§5.2, §9) ===
check('U9: EquirectSky + keepSourceCube:false is a named error; keepSourceCube:true is not', () => {
    const release = new EnvironmentSystem(makeMockEngine().engine, {}); // keepSourceCube → false
    assertThrows(
        () => release.setSky(new EquirectSky({ equirect: { view: true } })),
        /keepSourceCube/,
        'equirect on a cube-releasing system');

    const kept = new EnvironmentSystem(makeMockEngine().engine, { keepSourceCube: true });
    kept.setSky(new EquirectSky({ equirect: { view: true } })); // must NOT throw
    assert(kept.isBaked === false, 'not baked before getBakeStages');
    const stages = kept.getBakeStages();
    assert(Array.isArray(stages) && stages.length > 0, 'the bake chain is built for the equirect');
});

// === §9: the rest of the build-time error table ============================
check('§9: the remaining named errors fire at setSky / getSkyPass', () => {
    const { engine } = makeMockEngine();
    const env = new EnvironmentSystem(engine, {});
    env.setSky(new GradientSky({
        zenith: [0.1, 0.2, 0.5], horizon: [0.6, 0.7, 0.8], ground: [0.2, 0.2, 0.2],
        sunDir: [0, -1, 0], sunIntensity: 10,
    }));

    // A source whose logic lacks sampleSky.
    const noSample = {
        name: 'BrokenSky', paramFloats: 0,
        getShaderLayout: () => ({ wgslStructs: '', wgslLogic: 'fn skySunRadiance(e: f32) -> vec3<f32> { return vec3(1.0); }' }),
        writeParams: () => {},
    };
    assertThrows(() => env.setSky(noSample), /does not define sampleSky/, 'missing sampleSky');

    // A source that emits a backtick.
    const backtick = {
        name: 'TickSky', paramFloats: 0,
        getShaderLayout: () => ({ wgslStructs: '', wgslLogic: 'fn sampleSky(dir: vec3<f32>) -> vec3<f32> { return vec3(1.0); } // `' }),
        writeParams: () => {},
    };
    assertThrows(() => env.setSky(backtick), /backtick/, 'backtick in generated WGSL');

    // A source that changes paramFloats between setSky and getSkyPass. The stride is frozen into
    // the batch, so writeParams would run off the end of a bounds-clipped staging array —
    // silent truncation, which is exactly what this error exists to convert into a sentence.
    {
        const { engine: e2 } = makeMockEngine();
        const env2 = new EnvironmentSystem(e2, {});
        const mutable = {
            name: 'MutableSky', paramFloats: 4,
            getShaderLayout: () => ({ wgslStructs: '', wgslLogic: 'fn sampleSky(dir: vec3<f32>) -> vec3<f32> { return vec3(1.0); }' }),
            writeParams: () => {},
        };
        env2.setSky(mutable);
        mutable.paramFloats = 12;                    // the mistake this guards
        assertThrows(
            () => env2.getSkyPass({ colorAttachments: [{ view: {}, loadOp: 'load', storeOp: 'store' }] }),
            /declares paramFloats=12.*strideFloats=8/s,
            'paramFloats changed after setSky');
    }

    // setSky after the skybox pipeline was compiled (getSkyPass called).
    env.getSkyPass({ colorAttachments: [{ view: {}, loadOp: 'load', storeOp: 'store' }] });
    assertThrows(
        () => env.setSky(new GradientSky({
            zenith: [0.1, 0.2, 0.5], horizon: [0.6, 0.7, 0.8], ground: [0.2, 0.2, 0.2],
            sunDir: [0, -1, 0], sunIntensity: 10,
        })),
        /before the first getSkyPass/,
        'source swap after the batch was built');
});

// === U12: the shipped cheapSky default evaluates per pixel (§1, §4.5) =======
// The phase's deliverable, as one number. If this reads 1 the seam is landed but switched off,
// and every claim in §1 about a representable 0.266 degree sun is inert.
check('U12: a fresh source defaults to cheapSky 0 — the skybox evaluates sampleSky', () => {
    const { engine } = makeMockEngine();
    const env = new EnvironmentSystem(engine, {});
    let uploaded = null;
    engine.updateBatchData = (batch, data, count) => { uploaded = data.slice(); batch.currentInstanceCount = count; };
    env.setSky(new PreethamSky());
    env.getSkyPass({ colorAttachments: [{ view: {}, loadOp: 'load', storeOp: 'store' }] });
    assert(uploaded !== null, 'the sky batch never received its params');
    assertEq(uploaded[3], 0, 'cheapSky header float');
    assertEq(uploaded.length, SKY_PARAM_FLOATS + PREETHAM_SKY_FLOATS, 'sky params stride');

    // And the opt-out still reaches the buffer.
    env.setSkyParams({ cheapSky: 1, blurMip: 3 });
    assertEq(uploaded[3], 1, 'cheapSky after opting into the cube lookup');
    assertEq(uploaded[2], 3, 'blurMip rides along for the cheap path');
});

// === U13: every sky header float PERSISTS across a later partial call ======
// Found by the GPU suite's T4: `yaw` alone reverted to the system's rotation on the next
// setSkyParams that omitted it, so the skybox silently turned back while the SH and the
// specular lookup stayed put — the exact drift api/environment.md §1.5 promises cannot happen.
check('U13: exposure / yaw / blurMip / cheapSky all survive a later partial setSkyParams', () => {
    const { engine } = makeMockEngine();
    const env = new EnvironmentSystem(engine, {});
    let uploaded = null;
    engine.updateBatchData = (batch, data, count) => { uploaded = data.slice(); batch.currentInstanceCount = count; };
    env.setSky(new PreethamSky());
    env.getSkyPass({ colorAttachments: [{ view: {}, loadOp: 'load', storeOp: 'store' }] });

    // The staging array is f32, so compare at f32 precision, not with ===.
    const F32 = 1e-7;
    env.setSkyParams({ exposure: 1.3, yaw: 0.4, blurMip: 2, cheapSky: 1 });
    near(uploaded[1], 0.4, F32, 'yaw as set');

    env.setSkyParams({});                       // a later call that mentions none of them
    near(uploaded[0], 1.3, F32, 'exposure persisted');
    near(uploaded[1], 0.4, F32, 'yaw persisted');
    near(uploaded[2], 2, F32, 'blurMip persisted');
    near(uploaded[3], 1, F32, 'cheapSky persisted');

    // ...and yaw stayed the ENVIRONMENT's rotation, not a sky-local copy of it, so the SH and
    // specular lookup turned with it.
    env.setRotation(1.1);
    near(uploaded[1], 1.1, F32, 'setRotation moves the skybox too');
    env.setSkyParams({ exposure: 2.0 });
    near(uploaded[1], 1.1, F32, 'rotation held across an unrelated param change');
});


// === U10: writeParams allocates nothing (the out-parameter discipline, §7) ==
check('U10: writeParams allocates zero bytes across 1,000 calls per source', () => {
    const origs = {
        Float32Array: globalThis.Float32Array, Float64Array: globalThis.Float64Array,
        Uint8Array: globalThis.Uint8Array, Uint32Array: globalThis.Uint32Array,
        Int32Array: globalThis.Int32Array, Array: globalThis.Array,
    };
    let allocations = 0;
    const counting = (Ctor) => function (...args) { allocations++; return new Ctor(...args); };
    // Count only constructor-style allocation; class instances and literals are out of scope.
    globalThis.Float32Array = counting(origs.Float32Array);
    globalThis.Float64Array = counting(origs.Float64Array);
    globalThis.Uint8Array = counting(origs.Uint8Array);
    globalThis.Uint32Array = counting(origs.Uint32Array);
    globalThis.Int32Array = counting(origs.Int32Array);
    try {
        const sources = [
            new GradientSky({ zenith: [0.1, 0.2, 0.5], horizon: [0.6, 0.7, 0.8], ground: [0.2, 0.2, 0.2], sunDir: [0, -1, 0], sunIntensity: 10 }),
            new PreethamSky(),
            new EquirectSky({ equirect: {} }),
        ];
        // Allocate the destinations ONCE, outside the measured window.
        const bufs = sources.map(s => new origs.Float32Array(SKY_PARAM_FLOATS + s.paramFloats));
        allocations = 0;
        for (let i = 0; i < 1000; i++) {
            for (let s = 0; s < sources.length; s++) sources[s].writeParams(bufs[s], SKY_PARAM_FLOATS);
        }
    } finally {
        globalThis.Float32Array = origs.Float32Array; globalThis.Float64Array = origs.Float64Array;
        globalThis.Uint8Array = origs.Uint8Array; globalThis.Uint32Array = origs.Uint32Array;
        globalThis.Int32Array = origs.Int32Array;
    }
    assertEq(allocations, 0, 'typed-array allocations during 1000 writeParams calls');
});

// === U11: source group-2 bindings stay in the reserved 14–17 range (§4.3) ===
check('U11: sky-batch source bindings never touch the material group-2 map (0–13)', () => {
    for (const source of [new GradientSky({
        zenith: [0.1, 0.2, 0.5], horizon: [0.6, 0.7, 0.8], ground: [0.2, 0.2, 0.2],
        sunDir: [0, -1, 0], sunIntensity: 10,
    }), new PreethamSky(), new EquirectSky({ equirect: {} })]) {
        const { wgslStructs } = source.getShaderLayout(SKY_PARAM_FLOATS);
        const claimed = [...wgslStructs.matchAll(/@group\(2\)\s*@binding\((\d+)\)/g)].map(m => +m[1]);
        for (const b of claimed) {
            assert(b >= 14 && b <= 17,
                `${source.name} claims group-2 binding ${b}; the reserved source range is 14–17`);
        }
    }
    // The skybox module itself spends only 3,4 (env) plus the reserved range — never the lights
    // at 0–2 nor anything that would overwrite a fully-featured material's 5–13.
    const equiSkybox = buildSkyWGSL({ environmentGroup: 2, sky: new EquirectSky({ equirect: {} }).getShaderLayout(SKY_PARAM_FLOATS) });
    const group2 = [...equiSkybox.matchAll(/@group\(2\)\s*@binding\((\d+)\)/g)].map(m => +m[1]).sort((a, b) => a - b);
    assertEq(JSON.stringify(group2), JSON.stringify([3, 4, 14, 15]),
        'equirect skybox group-2 spends exactly {3,4} env + {14,15} source');
});

// The summary + a non-zero exit on failure (design-principles §13.2).
console.log(failures === 0 ? '\nAll environment sky tests passed.' : `\n${failures} test(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);

