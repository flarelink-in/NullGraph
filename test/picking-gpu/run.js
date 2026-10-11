// GPU integration tests for picking (docs/picking-design.md §10, P1–P8).
//
// The oracle is a CPU ray cast: from the eye through the clicked pixel's centre, against every box
// in the scene (10,000 grid cubes, a floating cube, a pole, three stock-material cubes). Whatever
// the CPU says is nearest, the GPU pick must return — so the test never assumes what is visible,
// and a scene edit that puts one object in front of another cannot make it lie.
//
// Two hosts, one file (test/gpu/harness-env.mjs): `npm run test:gpu`, or
// `npm run dev:serve` → http://localhost:3000/test/picking-gpu/.

import { NullGraph, Camera } from '../../dist/index.js';
import { VisibilitySystem, InstanceBuffer } from '../../dist/visibility.js';
import { MegabufferBuilder, Primitives, StandardLayout } from '../../dist/geometry.js';
import { buildPBRShader, StandardPBRMaterial } from '../../dist/materials.js';
import { Picker, unprojectPixel } from '../../dist/picking.js';
import { mat4 } from 'gl-matrix';
import { harnessEnv } from '../gpu/harness-env.mjs';

const { log, check, fatal, finish } = harnessEnv('__PICKING_RESULTS__');

const W = 320, H = 240;
const engine = new NullGraph();
const init = await engine.init(null, { width: W, height: H });
if (!init.success) {
    fatal(init.error);
    throw new Error(init.error);
}
const device = engine.device;

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

const mega = new MegabufferBuilder(StandardLayout.strideFloats);
const cubeGeo = Primitives.createCube(StandardLayout, 1, 1, 1);
mega.addMesh('cube', cubeGeo.vertices, cubeGeo.indices);
const CUBE = 0, BOUNDS = [0, 0, 0, Math.sqrt(3) / 2];

const GRID = 100, SPACING = 2.5;
const FRONT = GRID * GRID, POLE = GRID * GRID + 1;
const N = GRID * GRID + 2;
const vis = new VisibilitySystem(engine, { maxInstances: N });
vis.setGeometry(mega);
const instances = new InstanceBuffer(N);

/** Every box in the scene, for the CPU oracle: { source, index, min, max }. */
const boxes = [];
const m = mat4.create();
function addCube(i, centre, half = [0.5, 0.5, 0.5]) {
    mat4.fromTranslation(m, centre);
    mat4.scale(m, m, half.map(h => h * 2));
    instances.write(i, m, BOUNDS, CUBE);
    boxes.push({ source: 'cubes', index: i, min: centre.map((c, k) => c - half[k]), max: centre.map((c, k) => c + half[k]) });
}
const centres = [];
for (let i = 0; i < GRID * GRID; i++) {
    const c = [(i % GRID - GRID / 2) * SPACING, 0, (Math.floor(i / GRID) - GRID / 2) * SPACING];
    centres.push(c);
    addCube(i, c);
}

const camera = new Camera(55, W / H, 0.5, 400);
const EYE = [4, 60, 40];
camera.updateView(EYE, [0, 0, 0]);
engine.updateCamera(camera);

// P3: a cube floating 6 units in front of grid cube #T, on the line from it to the eye.
const T = 50 * GRID + 47;
const toEye = EYE.map((v, k) => v - centres[T][k]);
const len = Math.hypot(...toEye);
addCube(FRONT, centres[T].map((v, k) => v + toEye[k] / len * 6));

// P6: a pole 0.06 wide (≈1.4 px from its own camera) and 6 tall, far from the grid, in clear sky.
const POLE_POS = [0, 4, 190];
addCube(POLE, POLE_POS, [0.03, 3, 0.03]);

vis.setInstances(instances.data, N);
vis.upload();

const color = device.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
const depth = device.createTexture({ size: [W, H], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
const scenePass = engine.createPass({
    name: 'scene', isMainScreenPass: false,
    colorAttachments: [{ view: color.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
    depthStencilAttachment: { view: depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
});
vis.createInstanceDrawBatch(scenePass, 'cube', CUBE, { targetFormat: 'rgba8unorm' });

// The quick start's stock PBR batch (14-float layout): three 2-unit cubes floating over the grid.
const qsGeo = Primitives.createCube(StandardLayout, 2, 2, 2);
qsGeo.upload(engine);
const qsBatch = engine.createBatch(scenePass, {
    shaderCode: buildPBRShader({ useSkinning: false }), strideFloats: 14, maxInstances: 3,
    vertexLayouts: qsGeo.layout.getWebGPUDescriptor(), depthWriteEnabled: true, targetFormat: 'rgba8unorm',
});
new StandardPBRMaterial(engine, {
    albedoMap: engine.textureManager.fallbackWhite, normalMap: engine.textureManager.fallbackNormal,
    packedMap: engine.textureManager.fallbackWhite, packedMapFormat: 'ARM', baseColor: [0.1, 0.5, 0.9, 1],
}).applyToBatch(qsBatch);
engine.setBatchGeometry(qsBatch, qsGeo.vertexBuffer, qsGeo.indexBuffer, qsGeo.indices.length);
const QS = [[20, 6, 10], [25, 6, 10], [30, 6, 10]];
const qsData = new Float32Array(14 * 3);
QS.forEach((p, k) => {
    qsData.set([0, ...p, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], k * 14);
    boxes.push({ source: 'quick-start', index: k, min: p.map(c => c - 1), max: p.map(c => c + 1) });
});
engine.updateBatchData(qsBatch, qsData, 3);

// The app's frame, once, so a pick reads the state a user would be looking at.
engine.render([...vis.getComputeStages(), scenePass]);

const picker = new Picker(engine, { maxRadius: 4 });
const cubes = picker.addInstances(vis, 'cube', CUBE, { label: 'cubes' });
const quick = picker.addBatch(qsBatch, { label: 'quick-start', vertexLayouts: qsGeo.layout.getWebGPUDescriptor() });

// ---------------------------------------------------------------------------
// Harness: projection + the CPU oracle
// ---------------------------------------------------------------------------

/** World point → render-target pixel through `cam` (x right, y down). */
function toPixel(cam, p) {
    const v = cam.viewProjectionMatrix;
    const x = v[0] * p[0] + v[4] * p[1] + v[8] * p[2] + v[12];
    const y = v[1] * p[0] + v[5] * p[1] + v[9] * p[2] + v[13];
    const w = v[3] * p[0] + v[7] * p[1] + v[11] * p[2] + v[15];
    return [Math.floor((x / w * 0.5 + 0.5) * W), Math.floor((0.5 - y / w * 0.5) * H)];
}
const onScreen = ([x, y]) => x >= 4 && x < W - 4 && y >= 4 && y < H - 4;

/** The nearest box along the ray through pixel (px, py)'s centre, or null. */
function oracle(cam, eye, px, py) {
    const far = unprojectPixel(cam.bufferData.subarray(48, 64), px, py, 1, W, H);
    const dir = far.map((v, k) => v - eye[k]);
    let best = null, bestT = Infinity;
    for (const b of boxes) {
        let t0 = -Infinity, t1 = Infinity;
        for (let k = 0; k < 3; k++) {
            const a = (b.min[k] - eye[k]) / dir[k], c = (b.max[k] - eye[k]) / dir[k];
            t0 = Math.max(t0, Math.min(a, c)); t1 = Math.min(t1, Math.max(a, c));
        }
        if (t0 <= t1 && t0 > 0 && t0 < bestT) { bestT = t0; best = { ...b, point: eye.map((v, k) => v + dir[k] * t0) }; }
    }
    return best;
}

const sourceOf = (label) => (label === 'cubes' ? cubes : quick);
const describe = (h) => (h ? `${h.source.label}#${h.index}` : 'null');
const agrees = (hit, want) => (want === null ? hit === null : !!hit && hit.source === sourceOf(want.source) && hit.index === want.index);

/** Grid cubes whose top-face centre is on screen, deterministically spread across the grid. */
function sampleCubes(count) {
    const out = [];
    for (let k = 0; out.length < count && k < 5000; k++) {
        const i = (k * 7919) % (GRID * GRID);
        if (i === T) continue;
        const px = toPixel(camera, [centres[i][0], 0.5, centres[i][2]]);
        if (onScreen(px)) out.push({ i, px });
    }
    return out;
}

// ---------------------------------------------------------------------------
// P1 — the pick at an instance's projected centre returns that instance (and agrees with the oracle)
// ---------------------------------------------------------------------------
{
    const samples = sampleCubes(24);
    const bad = [];
    for (const s of samples) {
        const hit = await picker.pick(s.px[0], s.px[1], camera);
        const want = oracle(camera, EYE, s.px[0], s.px[1]);
        if (!agrees(hit, want) || want.index !== s.i) bad.push(`#${s.i}@${s.px}: gpu ${describe(hit)}, cpu ${want ? `${want.source}#${want.index}` : 'null'}`);
    }
    check('P1: a pick at an instance\'s centre returns its InstanceBuffer index — 24 picks over 10,002',
        samples.length === 24 && bad.length === 0, bad.length ? bad.slice(0, 3).join('; ') : '24/24 agree with the CPU ray cast');
}

// ---------------------------------------------------------------------------
// P2 — an empty pixel returns null
// ---------------------------------------------------------------------------
{
    const g = centres[55 * GRID + 52];
    const px = toPixel(camera, [g[0] + SPACING / 2, 0, g[2] + SPACING / 2]);   // ground between 4 cubes
    const hit = await picker.pick(px[0], px[1], camera);
    const want = oracle(camera, EYE, px[0], px[1]);
    check('P2: an empty pixel resolves null (the CPU ray cast agrees nothing is there)',
        onScreen(px) && want === null && hit === null, `pixel ${px}: gpu ${describe(hit)}, cpu ${want ? want.index : 'null'}`);
}

// ---------------------------------------------------------------------------
// P3 — overlapping instances: the front one wins
// ---------------------------------------------------------------------------
{
    const px = toPixel(camera, [centres[T][0], 0.5, centres[T][2]]);
    const hit = await picker.pick(px[0], px[1], camera);
    const want = oracle(camera, EYE, px[0], px[1]);
    check('P3: where two instances overlap, the pick returns the FRONT one',
        want?.index === FRONT && agrees(hit, want), `pixel ${px}: gpu ${describe(hit)}, cpu ${want ? `${want.source}#${want.index}` : 'null'} (front #${FRONT} hides #${T})`);
}

// ---------------------------------------------------------------------------
// P4 — the returned world position is where the ray meets the surface
// ---------------------------------------------------------------------------
{
    const s = sampleCubes(30)[29];
    const hit = await picker.pick(s.px[0], s.px[1], camera);
    const want = oracle(camera, EYE, s.px[0], s.px[1]);
    const err = hit && want ? Math.hypot(...hit.position.map((v, k) => v - want.point[k])) : Infinity;
    check('P4: the hit\'s world position is the ray–surface intersection (within 2 cm)',
        agrees(hit, want) && err < 0.02,
        `error=${err.toFixed(4)} m  got=[${hit?.position.map(v => v.toFixed(3))}] want=[${want?.point.map(v => v.toFixed(3))}]`);
}

// ---------------------------------------------------------------------------
// P5 — a stock-material plain batch is pickable, by its instance index
// ---------------------------------------------------------------------------
{
    const results = [];
    for (let k = 0; k < QS.length; k++) {
        const px = toPixel(camera, QS[k]);
        const hit = await picker.pick(px[0], px[1], camera);
        const want = oracle(camera, EYE, px[0], px[1]);
        results.push(onScreen(px) && want?.source === 'quick-start' && want.index === k && agrees(hit, want)
            ? 'ok' : `${k}@${px}: gpu ${describe(hit)}, cpu ${want ? `${want.source}#${want.index}` : 'null'}`);
    }
    check('P5: a stock-material plain batch (the quick start\'s) is pickable, by instance index',
        results.every(r => r === 'ok'), results.join(' '));
}

// ---------------------------------------------------------------------------
// P6 — radius: a hairline is missed at radius 0 beside it, and caught at radius 4
// ---------------------------------------------------------------------------
{
    const eye2 = [0, 4, 200];
    const side = new Camera(55, W / H, 0.5, 400);
    side.updateView(eye2, [0, 6, 190]);                // tilted up: clear sky behind the pole
    const c = toPixel(side, POLE_POS);
    const beside = [c[0] + 3, c[1]];
    const miss = await picker.pick(beside[0], beside[1], side, { radius: 0 });
    const caught = await picker.pick(beside[0], beside[1], side, { radius: 4 });
    check('P6: a hairline pole is missed 3 px beside it at radius 0, and caught at radius 4',
        onScreen(c) && miss === null && caught?.source === cubes && caught.index === POLE && caught.pixel[0] !== beside[0],
        `radius 0 → ${describe(miss)}; radius 4 → ${describe(caught)} at ${caught?.pixel}`);
}

// ---------------------------------------------------------------------------
// P7 — the app's frame is unchanged: same draws, pick culls disabled
// ---------------------------------------------------------------------------
{
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
    engine.render([...vis.getComputeStages(), scenePass]);
    device.createCommandEncoder = real;
    const pickStage = vis.getComputeStages().at(-1);
    check('P7: the app\'s frame is unchanged by a picker — its own 2 draws, the pick cull disabled',
        draws === 2 && pickStage.enabled === false, `draws=${draws} (instance batch + quick-start batch), pick cull enabled=${pickStage.enabled}`);
}

// ---------------------------------------------------------------------------
// P8 — concurrent picks resolve independently; latency is one GPU round trip
// ---------------------------------------------------------------------------
{
    const targets = sampleCubes(40).slice(30, 35);
    const t0 = performance.now();
    const hits = await Promise.all(targets.map(t => picker.pick(t.px[0], t.px[1], camera)));
    const ms = (performance.now() - t0) / targets.length;
    const wants = targets.map(t => oracle(camera, EYE, t.px[0], t.px[1]));
    check('P8: five concurrent picks each resolve their own instance',
        targets.length === 5 && hits.every((h, k) => agrees(h, wants[k])),
        `${hits.map(describe).join(', ')} — ${ms.toFixed(2)} ms per pick`);
}

await finish('All picking GPU tests passed.');
