// GPU integration tests for the sky-source seam — Phase 1 (docs/environment-sky-design.md §8,
// T1–T5). The headless half (U1–U12, `npm run test:sky`) pins the math and the generated WGSL as
// strings; these five pin the things only a real device can answer: what the skybox actually puts
// on screen, and whether the bake and the skybox really are one function once a driver has
// compiled both.
//
// Serve the repo root (the dev server does) and open /test/environment-gpu/:
//   npm run build:lib && npm run dev:serve   →   http://localhost:3000/test/environment-gpu/
// The page writes a machine-readable summary to #log and `window.__SKY_RESULTS__`.
//
// Every assertion is numeric. Readback is legal in a test and forbidden in the frame loop
// (design-principles §13).

import { NullGraph, Camera } from '../../dist/index.js';
import {
    EnvironmentSystem, GradientSky, PreethamSky,
    gradientSkyColor, SKY_PARAM_FLOATS, buildSkyWGSL,
} from '../../dist/environment.js';

const log = (msg) => {
    document.getElementById('log').textContent += `\n${msg}`;
    console.log(msg);
};
const results = [];
function check(name, cond, detail = '') {
    results.push({ name, pass: !!cond, detail });
    log(`  ${cond ? 'ok  ' : 'FAIL'}  ${name}${detail ? ' — ' + detail : ''}`);
}

const canvas = document.getElementById('gpu');
const engine = new NullGraph();
const init = await engine.init(canvas, { width: 256, height: 256 });
if (!init.success) {
    log('FATAL: WebGPU init failed — ' + init.error);
    window.__SKY_RESULTS__ = { fatal: init.error };
    throw new Error(init.error);
}
const device = engine.device;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

// The shipped skybox target is rgba16float. These tests render to rgba32float instead, so that
// half-float quantization (~5e-4 relative) never lands inside a tolerance — the textures being
// compared are still the engine's own rgba16float cubes, so nothing about the path under test
// changes. T4 renders BOTH images to the same format, so the comparison stays apples to apples.
const TARGET_FORMAT = 'rgba32float';
const BYTES_PER_PX = 16;

function makeTarget(N) {
    // copyTextureToBuffer wants bytesPerRow % 256 == 0; at 16 B/px that means N % 16 == 0.
    if (N % 16 !== 0) throw new Error(`target edge ${N} must be a multiple of 16`);
    const color = device.createTexture({
        size: [N, N], format: TARGET_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const depth = device.createTexture({
        size: [N, N], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const t = { N, color, depth, colorView: color.createView(), depthView: depth.createView() };
    // getSkyPass rewrites its attachments to loadOp 'load' (it is a background over the opaque
    // pass), so the clear has to be its own pass — exactly the shipped ordering, minus the scene.
    t.clearPass = engine.createPass({
        name: 'sky-clear', isMainScreenPass: false, size: [N, N],
        colorAttachments: [{
            view: t.colorView, clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear', storeOp: 'store',
        }],
        depthStencilAttachment: {
            view: t.depthView, depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store',
        },
    });
    t.attachments = {
        colorAttachments: [{ view: t.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: t.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
        size: [N, N],
        targetFormat: TARGET_FORMAT,
    };
    return t;
}

async function readTarget(t) {
    const bpr = t.N * BYTES_PER_PX;
    const buf = device.createBuffer({
        size: bpr * t.N, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: t.color }, { buffer: buf, bytesPerRow: bpr }, [t.N, t.N]);
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const px = new Float32Array(buf.getMappedRange().slice(0));
    buf.unmap(); buf.destroy();
    return px;                                  // RGBA f32, row-major
}

/** Point a camera down `dir` from the origin and push it to the engine's uniform. */
function aim(camera, dir) {
    const up = Math.abs(dir[1]) > 0.99 ? [0, 0, 1] : [0, 1, 0];
    camera.updateView([0, 0, 0], dir, up);
    engine.updateCamera(camera);
}

/** Render one skybox frame into `t` and read it back. */
async function renderSky(env, t, camera, dir, params) {
    if (params) env.setSkyParams(params);
    aim(camera, dir);
    engine.render([t.clearPass, env.getSkyPass(t.attachments)]);
    return readTarget(t);
}

// Sample a cube texture along arbitrary directions — the only way to compare what the BAKE wrote
// against what the SKYBOX drew. `dirs` entries are [x, y, z, mipLevel].
const CUBE_READ_WGSL = /* wgsl */ `
@group(0) @binding(0) var cubeTex: texture_cube<f32>;
@group(0) @binding(1) var cubeSamp: sampler;
@group(0) @binding(2) var<storage, read> dirs: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outCol: array<vec4<f32>>;
@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&dirs)) { return; }
    let d = dirs[gid.x];
    outCol[gid.x] = textureSampleLevel(cubeTex, cubeSamp, normalize(d.xyz), d.w);
}`;

async function sampleCube(view, sampler, dirs) {
    const n = dirs.length;
    const inData = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
        inData[i * 4] = dirs[i][0]; inData[i * 4 + 1] = dirs[i][1];
        inData[i * 4 + 2] = dirs[i][2]; inData[i * 4 + 3] = dirs[i][3] ?? 0;
    }
    const inBuf = device.createBuffer({
        size: inData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(inBuf, 0, inData);
    const outBuf = device.createBuffer({
        size: n * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const stage = engine.createComputeStage({
        name: 'cube readback', shaderCode: CUBE_READ_WGSL,
        dispatchCount: [Math.ceil(n / 64), 1, 1],
    });
    engine.setComputePassBindGroup(stage, [
        { binding: 0, resource: view },
        { binding: 1, resource: sampler },
        { binding: 2, resource: { buffer: inBuf } },
        { binding: 3, resource: { buffer: outBuf } },
    ], 0);
    engine.render([stage]);

    const read = device.createBuffer({
        size: n * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(outBuf, 0, read, 0, n * 16);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(read.getMappedRange().slice(0));
    read.unmap(); read.destroy(); inBuf.destroy(); outBuf.destroy();
    return out;
}

const norm = (v) => { const l = Math.hypot(v[0], v[1], v[2]); return [v[0] / l, v[1] / l, v[2] / l]; };
const relErr = (a, b) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-3);

/** N Fibonacci-sphere directions — deterministic, and the same construction the headless suite uses. */
function fibonacci(n) {
    const out = [], GA = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < n; i++) {
        const y = 1 - 2 * (i + 0.5) / n, r = Math.sqrt(1 - y * y), a = GA * i;
        out.push([Math.cos(a) * r, y, Math.sin(a) * r]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// T1 — the sun disc is the size it was asked to be
//
// The claim §1 rests on: at 0.266° angular radius the true sun is SUB-TEXEL in the 128²-per-face
// prefiltered cube the skybox used to sample (0.70°/texel), so this measurement is only possible
// at all once the fragment evaluates sampleSky. Measured by subtracting a disc-less render from a
// disc-bearing one, which isolates the disc's own smoothstep profile from the circumsolar glow.
// ---------------------------------------------------------------------------
{
    const N = 512, FOV = 4;                       // 4° across 512 px ≈ 0.0078°/px
    const RADIUS_DEG = 0.266;
    const ELEV = 45 * Math.PI / 180;
    const toSun = [Math.cos(ELEV), Math.sin(ELEV), 0];
    const travel = [-toSun[0], -toSun[1], -toSun[2]];
    const camera = new Camera(FOV, 1, 0.1, 100);
    const tanHalf = Math.tan(FOV * Math.PI / 360);

    async function lumRow(discIntensity) {
        const env = new EnvironmentSystem(engine, { sourceSize: 32 });
        env.setSky(new PreethamSky({
            turbidity: 2.8, sunDir: travel,
            sunAngularRadius: RADIUS_DEG, sunDiscIntensity: discIntensity,
        }));
        const t = makeTarget(N);
        const px = await renderSky(env, t, camera, toSun);
        const row = new Float64Array(N), y = N / 2;
        for (let x = 0; x < N; x++) {
            const i = (y * N + x) * 4;
            row[x] = 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2];
        }
        t.color.destroy(); t.depth.destroy(); env.destroy(); engine.clearPasses();
        return row;
    }

    const withDisc = await lumRow(40);
    const noDisc = await lumRow(0);
    const profile = Float64Array.from(withDisc, (v, i) => v - noDisc[i]);

    // discAmt saturates at 1 inside the disc, so the peak is a flat plateau and the 50% crossing
    // sits exactly at cosGamma == sunCos — i.e. exactly at the requested angular radius.
    let peak = 0;
    for (const v of profile) peak = Math.max(peak, v);
    const half = peak * 0.5;

    /** Sub-pixel crossing of `half`, scanning outward from the centre. */
    function crossing(step) {
        for (let x = N / 2; x > 0 && x < N - 1; x += step) {
            if (profile[x] >= half && profile[x + step] < half) {
                const f = (profile[x] - half) / (profile[x] - profile[x + step]);
                return x + step * f;
            }
        }
        return NaN;
    }
    const toNdc = (x) => 2 * (x + 0.5) / N - 1;
    const angle = (x) => Math.atan(toNdc(x) * tanHalf) * 180 / Math.PI;

    const left = crossing(-1), right = crossing(+1);
    const measured = (angle(right) - angle(left)) / 2;
    const pixelDeg = (FOV / N);

    check('T1: the disc has a nonzero, saturated profile to measure',
        peak > 1e-3 && Number.isFinite(left) && Number.isFinite(right),
        `peak=${peak.toFixed(4)} left=${left.toFixed(2)}px right=${right.toFixed(2)}px`);
    check('T1: sun disc angular radius matches sunAngularRadius within one pixel',
        Math.abs(measured - RADIUS_DEG) <= pixelDeg,
        `measured=${measured.toFixed(4)}° asked=${RADIUS_DEG}° tol=${pixelDeg.toFixed(4)}° ` +
        `(${(2 * measured / pixelDeg).toFixed(1)} px across — the prefiltered cube resolves 0.70°/texel)`);
}

// ---------------------------------------------------------------------------
// T2 — the two consumers are one function
//
// U2 pins that the bake kernel and the skybox carry byte-identical sampleSky SOURCE. That a
// rebuild cannot skip; this is the half a driver cannot fake. Both are held against the same CPU
// mirror (skyMath.gradientSkyColor, itself pinned to the pre-refactor kernel by U1), so agreeing
// with it means agreeing with each other — through a reference, not with each other's errors.
// ---------------------------------------------------------------------------
{
    const CFG = {
        zenith: [0.10, 0.22, 0.52], horizon: [0.65, 0.72, 0.85], ground: [0.28, 0.24, 0.20],
        sunDir: [-0.4, -1.0, -0.35],
        sunIntensity: 0,   // no disc: an 0.266° feature is not resolvable by a 512² cube texel,
                           // and T1 already measured the disc. This isolates the gradient.
    };
    const env = new EnvironmentSystem(engine, { sourceSize: 512, keepSourceCube: true });
    const source = new GradientSky(CFG);
    env.setSky(source);
    engine.render([...env.getBakeStages()]);
    env.copySHToParams();

    const params = new Float32Array(SKY_PARAM_FLOATS + source.paramFloats);
    source.writeParams(params, SKY_PARAM_FLOATS);

    // The gradient's pow(|y|, 0.6) / pow(|y|, 0.5) terms have an INFINITE derivative at y = 0, so
    // a cube texel straddling the horizon carries a real quadrature error that is not a bug in
    // anything. Excluded, and stated, rather than absorbed into a loose tolerance.
    const dirs = fibonacci(80).filter(d => Math.abs(d[1]) >= 0.08).slice(0, 64);

    // (a) the SKYBOX against the mirror — no discretization anywhere in this path, so it is tight.
    const N = 64, camera = new Camera(2, 1, 0.1, 100);
    const t = makeTarget(N);
    let maxSkyErr = 0, worstSky = null;
    for (const dir of dirs) {
        const px = await renderSky(env, t, camera, dir);
        const i = ((N / 2) * N + N / 2) * 4;
        const want = gradientSkyColor(dir, params, SKY_PARAM_FLOATS);
        for (let c = 0; c < 3; c++) {
            const e = relErr(px[i + c], want[c]);
            if (e > maxSkyErr) { maxSkyErr = e; worstSky = dir; }
        }
    }
    check('T2a: the SKYBOX evaluates the mirror at 64 directions within 0.5%',
        maxSkyErr <= 5e-3,
        `maxRelErr=${(maxSkyErr * 100).toFixed(3)}% worst=[${(worstSky || []).map(v => v.toFixed(2))}]`);

    // (b) the SOURCE CUBE against the same mirror — this one carries the cube's own 512²
    //     quantization plus rgba16float storage, so it gets the looser (and still numeric) bound.
    const cube = await sampleCube(env.sourceView, env.sampler, dirs.map(d => [...d, 0]));
    let maxCubeErr = 0, worstCube = null;
    for (let i = 0; i < dirs.length; i++) {
        const want = gradientSkyColor(dirs[i], params, SKY_PARAM_FLOATS);
        for (let c = 0; c < 3; c++) {
            const e = relErr(cube[i * 4 + c], want[c]);
            if (e > maxCubeErr) { maxCubeErr = e; worstCube = dirs[i]; }
        }
    }
    check('T2b: the BAKE writes the same mirror at the same 64 directions within 2%',
        maxCubeErr <= 2e-2,
        `maxRelErr=${(maxCubeErr * 100).toFixed(3)}% worst=[${(worstCube || []).map(v => v.toFixed(2))}]`);

    t.color.destroy(); t.depth.destroy(); env.destroy(); engine.clearPasses();
}

// ---------------------------------------------------------------------------
// T3 — no seam artefact where the cube faces meet
//
// Two halves, because one sky cannot answer both questions cleanly. A CONSTANT sky isolates the
// artefact (any variation at all is one), but would not notice a face that is rotated or flipped.
// A real gradient catches that, but carries its own quadrature error near the horizon. Run both.
// ---------------------------------------------------------------------------
{
    // The 12 cube edges: two axes pinned to ±1, the third sweeping.
    const EDGES = [];
    for (const [a, b, c] of [[0, 1, 2], [0, 2, 1], [1, 2, 0]]) {
        for (const sa of [-1, 1]) for (const sb of [-1, 1]) {
            EDGES.push({ a, b, c, sa, sb });
        }
    }
    const edgeDirs = (samples) => {
        const out = [];
        for (const e of EDGES) {
            for (let i = 0; i < samples; i++) {
                const t = -1 + 2 * i / (samples - 1);
                const v = [0, 0, 0];
                v[e.a] = e.sa; v[e.b] = e.sb; v[e.c] = t;
                out.push(norm(v));
            }
        }
        return out;
    };

    // (a) a constant sky: sampleSky is one colour in every direction, so ANY spread across a face
    //     boundary is an artefact of the cube, not of the model.
    {
        const K = [0.37, 0.61, 0.83];
        const env = new EnvironmentSystem(engine, { sourceSize: 128, keepSourceCube: true });
        env.setSky(new GradientSky({
            zenith: K, horizon: K, ground: K, sunDir: [0, -1, 0], sunIntensity: 0,
        }));
        engine.render([...env.getBakeStages()]);

        const dirs = edgeDirs(33);            // 12 edges × 33 samples, corners included
        const got = await sampleCube(env.sourceView, env.sampler, dirs.map(d => [...d, 0]));
        let worst = 0, at = null;
        for (let i = 0; i < dirs.length; i++) {
            for (let c = 0; c < 3; c++) {
                const e = relErr(got[i * 4 + c], K[c]);
                if (e > worst) { worst = e; at = dirs[i]; }
            }
        }
        check('T3a: radiance is continuous across all 12 cube edges within 1%',
            worst <= 1e-2,
            `maxDeviation=${(worst * 100).toFixed(3)}% over ${dirs.length} edge samples, ` +
            `worst=[${(at || []).map(v => v.toFixed(2))}]`);
        env.destroy(); engine.clearPasses();
    }

    // (b) a real gradient: catches a face whose orientation is wrong, which (a) cannot see.
    //     Horizon-adjacent samples are excluded for the same pow-derivative reason as T2b.
    {
        const CFG = {
            zenith: [0.10, 0.22, 0.52], horizon: [0.65, 0.72, 0.85], ground: [0.28, 0.24, 0.20],
            sunDir: [-0.4, -1.0, -0.35], sunIntensity: 0,
        };
        const env = new EnvironmentSystem(engine, { sourceSize: 512, keepSourceCube: true });
        const source = new GradientSky(CFG);
        env.setSky(source);
        engine.render([...env.getBakeStages()]);
        const params = new Float32Array(SKY_PARAM_FLOATS + source.paramFloats);
        source.writeParams(params, SKY_PARAM_FLOATS);

        const dirs = edgeDirs(17).filter(d => Math.abs(d[1]) >= 0.1);
        const got = await sampleCube(env.sourceView, env.sampler, dirs.map(d => [...d, 0]));
        let worst = 0, at = null;
        for (let i = 0; i < dirs.length; i++) {
            const want = gradientSkyColor(dirs[i], params, SKY_PARAM_FLOATS);
            for (let c = 0; c < 3; c++) {
                const e = relErr(got[i * 4 + c], want[c]);
                if (e > worst) { worst = e; at = dirs[i]; }
            }
        }
        check('T3b: edge directions carry the RIGHT radiance (face orientation) within 2%',
            worst <= 2e-2,
            `maxRelErr=${(worst * 100).toFixed(3)}% over ${dirs.length} samples, ` +
            `worst=[${(at || []).map(v => v.toFixed(2))}]`);
        env.destroy(); engine.clearPasses();
    }
}

// ---------------------------------------------------------------------------
// T4 — cheapSky: 1 reproduces the pre-Phase-1 image
//
// buildSkyWGSL still emits the pre-Phase-1 module when no sky layout is supplied. That module is
// the reference: the cheapSky branch of the composed skybox must draw the same pixels from the
// same cube. This is what makes "the old path is still reachable" a fact rather than an intention.
// ---------------------------------------------------------------------------
{
    const N = 128, BLUR = 2, EXPOSURE = 1.3, YAW = 0.4;
    const camera = new Camera(70, 1, 0.1, 100);
    const dir = norm([0.4, 0.25, 1]);

    const env = new EnvironmentSystem(engine, { sourceSize: 128, keepSourceCube: true });
    env.setSky(new PreethamSky({ turbidity: 3.2, sunDir: [-0.4, -1.0, -0.35] }));
    engine.render([...env.getBakeStages()]);
    env.copySHToParams();

    // The composed skybox, told to take the cheap branch.
    const t = makeTarget(N);
    const now = await renderSky(env, t, camera, dir,
        { cheapSky: 1, blurMip: BLUR, exposure: EXPOSURE, yaw: YAW });

    // The pre-Phase-1 module, hand-wired against the SAME cube, sampler and params.
    const refTarget = makeTarget(N);
    const refPass = engine.createPass({
        name: 'sky-pre-phase1', isMainScreenPass: false, size: [N, N],
        colorAttachments: [{ view: refTarget.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: {
            view: refTarget.depthView, depthLoadOp: 'load', depthStoreOp: 'store',
        },
    });
    const refBatch = engine.createBatch(refPass, {
        shaderCode: buildSkyWGSL({ environmentGroup: 2 }),      // no `sky` — the old module
        strideFloats: 4, maxInstances: 1, targetFormat: TARGET_FORMAT,
        depthWriteEnabled: false, depthCompare: 'less-equal',
    });
    engine.setRenderPassBindGroup(refBatch, [
        { binding: 3, resource: env.sampler },
        { binding: 4, resource: env.specularView },
    ], 2);
    engine.updateBatchData(refBatch, new Float32Array([EXPOSURE, YAW, BLUR, 0]), 1);
    aim(camera, dir);
    engine.render([refTarget.clearPass, refPass]);
    const ref = await readTarget(refTarget);

    let worst = 0, worstAt = 0, sum = 0;
    for (let i = 0; i < N * N; i++) {
        for (let c = 0; c < 3; c++) {
            const d = Math.abs(now[i * 4 + c] - ref[i * 4 + c]);
            sum += d;
            if (d > worst) { worst = d; worstAt = i; }
        }
    }
    if (worst > 0) {
        // A whole-image offset means the two modules disagree about a HEADER float (this is how
        // the non-persistent `yaw` was found); a localized one means they disagree about a
        // direction or a mip.
        const fmt = (a, i) => `[${a[i * 4].toFixed(4)},${a[i * 4 + 1].toFixed(4)},${a[i * 4 + 2].toFixed(4)}]`;
        log(`        T4 diag: mean|d|=${(sum / (N * N * 3)).toExponential(2)} ` +
            `worst at px(${worstAt % N},${Math.floor(worstAt / N)}) ` +
            `cheap=${fmt(now, worstAt)} ref=${fmt(ref, worstAt)}`);
    }
    let refMax = 0;
    for (let i = 0; i < ref.length; i += 4) refMax = Math.max(refMax, ref[i], ref[i + 1], ref[i + 2]);
    check('T4: the reference image is non-trivial (the comparison can fail)',
        refMax > 1e-2, `max reference channel=${refMax.toFixed(4)}`);
    check('T4: cheapSky 1 reproduces the pre-Phase-1 image within 1e-3',
        worst <= 1e-3, `maxAbsDiff=${worst.toExponential(2)} over ${N}x${N} px`);

    // And the shipped default is the OTHER branch — flipping it must visibly change the image,
    // or "the skybox evaluates per pixel" is a claim with no consequence.
    const evaluated = await renderSky(env, t, camera, dir,
        { cheapSky: 0, blurMip: BLUR, exposure: EXPOSURE, yaw: YAW });
    let diff = 0;
    for (let i = 0; i < N * N * 4; i++) diff = Math.max(diff, Math.abs(evaluated[i] - now[i]));
    check('T4b: cheapSky 0 is a different image (the branch is live)',
        diff > 1e-3, `maxAbsDiff vs cheap=${diff.toExponential(2)}`);

    t.color.destroy(); t.depth.destroy();
    refTarget.color.destroy(); refTarget.depth.destroy();
    env.destroy(); engine.clearPasses();
}

// ---------------------------------------------------------------------------
// T5 — switching sources rebuilds the bake and leaves no stale mip
//
// setSky rebuilds the chain and clears `baked`, so the next getBakeStages() returns the full
// chain again. The failure this catches is the quiet one: a chain that rebuilds but never re-runs
// leaves EVERY prefilter mip holding the previous sky, and the scene stays lit by a sky that is
// no longer on screen. Two constant skies make that unambiguous at every mip.
// ---------------------------------------------------------------------------
{
    const A = [0.80, 0.10, 0.10];               // red
    const B = [0.10, 0.75, 0.15];               // green
    const MIPS = 6;
    const env = new EnvironmentSystem(engine, {
        sourceSize: 128, prefilterMips: MIPS, keepSourceCube: true,
    });
    const constantSky = (K) => new GradientSky({
        zenith: K, horizon: K, ground: K, sunDir: [0, -1, 0], sunIntensity: 0,
    });

    const probes = fibonacci(16);
    async function mipColours() {
        const dirs = [];
        for (let m = 0; m < MIPS; m++) for (const d of probes) dirs.push([...d, m]);
        const got = await sampleCube(env.specularView, env.sampler, dirs);
        const perMip = [];
        for (let m = 0; m < MIPS; m++) {
            const acc = [0, 0, 0];
            for (let i = 0; i < probes.length; i++) {
                const o = (m * probes.length + i) * 4;
                acc[0] += got[o]; acc[1] += got[o + 1]; acc[2] += got[o + 2];
            }
            perMip.push(acc.map(v => v / probes.length));
        }
        return perMip;
    }

    env.setSky(constantSky(A));
    engine.render([...env.getBakeStages()]);
    const afterA = await mipColours();

    env.setSky(constantSky(B));
    check('T5: setSky re-arms the bake (getBakeStages is non-empty after a swap)',
        env.isBaked === false, `isBaked=${env.isBaked}`);
    const stages = env.getBakeStages();
    check('T5: the swapped source returns a full bake chain', stages.length > 0,
        `${stages.length} stages`);
    engine.render([...stages]);
    const afterB = await mipColours();

    let worstA = 0, worstB = 0, staleMip = -1;
    for (let m = 0; m < MIPS; m++) {
        for (let c = 0; c < 3; c++) {
            worstA = Math.max(worstA, relErr(afterA[m][c], A[c]));
            const e = relErr(afterB[m][c], B[c]);
            if (e > worstB) { worstB = e; staleMip = m; }
        }
    }
    check('T5: every prefilter mip carries source A after the first bake',
        worstA <= 3e-2, `maxRelErr=${(worstA * 100).toFixed(2)}% across ${MIPS} mips`);
    check('T5: every prefilter mip carries source B after the swap — no stale mip',
        worstB <= 3e-2,
        `maxRelErr=${(worstB * 100).toFixed(2)}% at mip ${staleMip}; ` +
        `mip colours=${afterB.map(c => '[' + c.map(v => v.toFixed(2)).join(',') + ']').join(' ')}`);

    env.destroy(); engine.clearPasses();
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------
const failed = results.filter(r => !r.pass);
log(failed.length === 0
    ? '\nAll environment sky GPU tests passed.'
    : `\n${failed.length} GPU test(s) FAILED.`);
window.__SKY_RESULTS__ = { total: results.length, failed: failed.length, results };
document.title = failed.length === 0 ? 'PASS' : 'FAIL';
