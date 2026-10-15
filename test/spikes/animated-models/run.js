// Cycle D, step 1, phase 0 (docs/agents-animated-models-design.md §8): what animated low-poly
// models cost against the Herd & Wolves demo's darts, at its counts. A measurement, not a suite.
//
//   Node + Dawn:  npm run spike:animated-models      NG_DAWN_FLAGS=adapter=AMD or adapter=NVIDIA
//                 NG_GPU_ARTIFACTS=<dir> also saves a close-up of the posed models as PNG
//   Chrome:       npm run drive -- --serve . "/test/spikes/animated-models/index.html" --wait-for "animated-models done" --timeout 900
//   NG_SPIKE_MAX (?max=) stops at a count.
//
// The models are the test site's (fox, deer, cow: rigid parts, no skin), baked by bake.mjs into a
// table of part transforms per frame. The rows are written once, as the demo's instance stage would
// write them: one agent in 16 a wolf (the fox), the rest prey, on a field that grows with the count,
// at random headings. The camera looks down on the whole field, as the demo opens: every agent is
// in view, the worst case for the draw.
//
// Each mode, at each count, draws one frame R times between two timestamps; the culls run each
// time too (one per batch, as in the demo). Modes:
//   darts       the demo's cones, the generated vertex stage
//   rest:<prey> the fox and the prey model at rest, the generated vertex stage (geometry cost alone)
//   hook:<prey> the same, posed by the hook this spike prototypes (design §4–§5): each vertex reads
//               its part (by vertex index), its agent's state (by sourceIndex) and up to four poses
//               (two clips blended by speed, two frames each), before the model matrix
import { harnessEnv } from '../../gpu/harness-env.mjs';
import { NullGraph, Camera } from '../../../dist/index.js';
import { Primitives, StandardLayout, MegabufferBuilder } from '../../../dist/geometry.js';
import {
    VisibilitySystem, InstanceBuffer, LOD_AUTO,
    CAMERA_STRUCT_WGSL, INSTANCE_STRUCT_WGSL, CULL_CORE_WGSL,
} from '../../../dist/visibility.js';
import { mat4 } from 'gl-matrix';
import { readGlb, bakeRigid, weld } from './bake.mjs';

const env = harnessEnv('__ANIM_SPIKE_RESULTS__');
const { log, check } = env;
const inBrowser = env.inBrowser;
const setting = (name, fallback) => (inBrowser
    ? new URLSearchParams(location.search).get(name)
    : globalThis.process?.env?.[`NG_SPIKE_${name.toUpperCase()}`]) ?? fallback;

const COUNTS = [4_096, 16_384, 65_536];
const MAX = Number(setting('max', 65_536));
const WOLF_SHARE = 16, BASE_HALF = 80;
const WIDTH = 1280, HEIGHT = 720, REPEAT = 3, WARM = 8, FRAMES = 24;
const LIMIT_MS = 400;                             // a render predicted longer is not run (GPU reset at ~2 s)
const CLIPS = ['idle', 'walk', 'run', 'eat'];
const median = v => { const s = [...v].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
const fmt = n => n.toLocaleString('en-US');

// --- the engine ------------------------------------------------------------------------------
const engine = new NullGraph();
const init = await engine.init(env.canvas, { width: WIDTH, height: HEIGHT, desiredFeatures: ['timestamp-query'] });
if (!init.success) { env.fatal(init.error); throw new Error(init.error); }
const { device } = engine;
if (!device.features.has('timestamp-query')) { env.fatal('no timestamp-query'); throw new Error('no timestamp-query'); }
const info = device.adapterInfo ?? {};
log(`animated-models: ${inBrowser ? 'Chrome' : 'Node/Dawn'} · ${[info.vendor, info.architecture, info.description].filter(Boolean).join(' ')}`);

// --- the models --------------------------------------------------------------------------------
const load = async name => {
    const url = new URL(`./models/animal-${name}.glb`, import.meta.url);
    const bytes = inBrowser
        ? await (await fetch(url)).arrayBuffer()
        : (await import('node:fs')).readFileSync(url);
    const t0 = performance.now();
    const baked = bakeRigid(readGlb(bytes), { clips: CLIPS });
    baked.bakeMs = performance.now() - t0;
    return baked;
};
const models = { fox: await load('fox'), deer: await load('deer'), cow: await load('cow') };
// Lengths as the demo's darts: wolves 1.7 m, prey 1.1 m (the rest pose's depth).
const restDepth = m => { let lo = Infinity, hi = -Infinity; for (let i = 2; i < m.vertices.length; i += 8) { lo = Math.min(lo, m.vertices[i]); hi = Math.max(hi, m.vertices[i]); } return hi - lo; };
const scaleOf = { fox: 1.7 / restDepth(models.fox), deer: 1.3 / restDepth(models.deer), cow: 1.3 / restDepth(models.cow) };
// Welded variants (bake.mjs, weld): the same shapes, vertices shared between faces.
for (const name of ['fox', 'deer', 'cow']) { models[`${name}-w`] = weld(models[name]); scaleOf[`${name}-w`] = scaleOf[name]; }
for (const [name, m] of Object.entries(models)) {
    log(`   ${name}: ${fmt(m.indices.length / 3)} triangles, ${m.parts} parts, clips ${m.clips.map(c => `${c.name} ${c.frames}`).join(', ')}; ` +
        `table ${fmt(m.transforms.byteLength)} B, baked in ${m.bakeMs.toFixed(0)} ms; scaled ×${scaleOf[name].toFixed(2)}`);
}

// One mega buffer: the two darts, then the three models. The hook finds a vertex's part by its
// absolute index in it (vertex_index includes the batch's base vertex), so partOf spans the mega.
const cone = (radius, length) => {
    const g = Primitives.createCone(StandardLayout, radius, length, 16, 1);
    const v = g.vertices, s = StandardLayout.strideFloats;
    for (let i = 0; i < v.length; i += s) {            // along +Z, as the demo's darts
        const y = v[i + 1]; v[i + 1] = -v[i + 2] + radius + 0.05; v[i + 2] = y;
        const ny = v[i + 4]; v[i + 4] = -v[i + 5]; v[i + 5] = ny;
    }
    return g;
};
const mega = new MegabufferBuilder(StandardLayout.strideFloats);
const parts = [];                                      // partOf, in mega order
const poses = [];                                      // all models' tables, concatenated (vec4s)
const meshIndex = {}, poseBase = {};
let next = 0;
const addMesh = (name, vertices, indices, partOfVertex, table) => {
    mega.addMesh(name, vertices, indices);
    meshIndex[name] = next++;
    if (partOfVertex) {
        poseBase[name] = poses.length / 4;
        poses.push(...table);
        parts.push(...partOfVertex);
    } else {
        parts.push(...new Array(vertices.length / StandardLayout.strideFloats).fill(0));
    }
};
const preyCone = cone(0.32, 1.1), wolfCone = cone(0.48, 1.7);
addMesh('prey-dart', preyCone.vertices, preyCone.indices);
addMesh('wolf-dart', wolfCone.vertices, wolfCone.indices);
for (const [name, m] of Object.entries(models)) addMesh(name, m.vertices, new Uint16Array(m.indices), m.partOfVertex, m.transforms);

const storage = (data, usage = GPUBufferUsage.STORAGE) => {
    const buffer = device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 16) * 16), usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
};
const partOfBuffer = storage(new Uint32Array(parts));
const posesBuffer = storage(new Float32Array(poses));
log(`   mega: ${fmt(parts.length)} vertices; part table ${fmt(parts.length * 4)} B; pose tables ${fmt(poses.length * 4)} B`);

// --- the hook (design §4–§5), prototyped as a whole vertex stage ---------------------------------
const HOOK_DRAW = /* wgsl */ `
${CAMERA_STRUCT_WGSL}
${INSTANCE_STRUCT_WGSL}
${CULL_CORE_WGSL}
struct VertexOut {
    @builtin(position) clipPos: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) worldNormal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) ecsColor: vec3<f32>,
};
struct Anim {
    clip: array<vec4<u32>, 4>,     // idle, walk, run, eat: first frame (in this model's table), frames
    poseBase: u32,                 // this model's first vec4 in poses
    parts: u32,
    _a: u32, _b: u32,
    color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(2) @binding(0) var<storage, read> partOf: array<u32>;
@group(2) @binding(1) var<storage, read> poses: array<vec4<f32>>;
@group(2) @binding(2) var<storage, read> state: array<vec4<f32>>;   // per row: speed / top speed, gait phase, resting, age offset
@group(2) @binding(3) var<uniform> anim: Anim;

struct Pose { r0: vec4<f32>, r1: vec4<f32>, r2: vec4<f32> };
fn poseAt(frame: u32, part: u32) -> Pose {
    let k = anim.poseBase + (frame * anim.parts + part) * 3u;
    return Pose(poses[k], poses[k + 1u], poses[k + 2u]);
}
// A clip at a phase in [0, 1): its two nearest frames, blended.
fn clipPose(c: u32, phase: f32, part: u32) -> Pose {
    let first = anim.clip[c].x;
    let frames = anim.clip[c].y;
    let f = fract(phase) * f32(frames);
    let f0 = u32(f) % frames;
    let w = fract(f);
    let a = poseAt(first + f0, part);
    let b = poseAt(first + (f0 + 1u) % frames, part);
    return Pose(mix(a.r0, b.r0, w), mix(a.r1, b.r1, w), mix(a.r2, b.r2, w));
}
fn blend(a: Pose, b: Pose, w: f32) -> Pose { return Pose(mix(a.r0, b.r0, w), mix(a.r1, b.r1, w), mix(a.r2, b.r2, w)); }

@vertex
fn vs_main(@location(0) pos: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>,
           @builtin(instance_index) ii: u32, @builtin(vertex_index) vi: u32) -> VertexOut {
    let inst = instances[ii];
    let s = state[inst.sourceIndex];
    let part = partOf[vi];
    let t = camera.simTime + s.w;
    var p: Pose;
    if (s.z > 0.0) {
        p = clipPose(3u, t / 0.5, part);                                   // resting: eat
    } else {
        let gait = s.y + t * mix(2.0, 4.0, smoothstep(0.45, 0.75, s.x));  // one gait phase for walk and run
        let moving = blend(clipPose(1u, gait, part), clipPose(2u, gait, part), smoothstep(0.45, 0.75, s.x));
        let still = smoothstep(0.08, 0.2, s.x);
        p = moving;
        if (still < 1.0) { p = blend(clipPose(0u, t, part), moving, still); }
    }
    let lp = vec4<f32>(pos, 1.0);
    let posed = vec3<f32>(dot(p.r0, lp), dot(p.r1, lp), dot(p.r2, lp));
    let n = vec3<f32>(dot(p.r0.xyz, normal), dot(p.r1.xyz, normal), dot(p.r2.xyz, normal));
    let world = inst.model * vec4<f32>(posed, 1.0);
    var out: VertexOut;
    out.clipPos = camera.viewProj * world;
    out.worldPos = world.xyz;
    out.worldNormal = normalize(normalMatrixFromModel(inst.model) * n);
    out.uv = uv;
    out.ecsColor = anim.color.rgb;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.worldNormal);
    let light = max(dot(n, normalize(vec3<f32>(0.4, 1.0, 0.3))), 0.0);
    let band = select(0.55, 1.0, light > 0.5);                        // a two-band toon ramp
    return vec4<f32>(in.ecsColor * band, 1.0);
}`;
const HOOK_DRAW_FACETED = HOOK_DRAW.replace('let n = normalize(in.worldNormal);', 'let n = normalize(cross(dpdx(in.worldPos), dpdy(in.worldPos)));');
const fragmentOf = rgb => /* wgsl */ `
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.worldNormal);
    let light = max(dot(n, normalize(vec3<f32>(0.4, 1.0, 0.3))), 0.0);
    return vec4<f32>(vec3<f32>(${rgb.join(', ')}) * select(0.55, 1.0, light > 0.5), 1.0);
}`;
const PREY_RGB = [0.75, 0.6, 0.42], WOLF_RGB = [0.9, 0.45, 0.2];

// --- timestamps --------------------------------------------------------------------------------
const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
const resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
const read = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
const mark = index => {
    const enc = device.createCommandEncoder();
    enc.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: index } }).end();
    device.queue.submit([enc.finish()]);
};
async function timed(fn) {
    mark(0);
    fn();
    mark(1);
    const enc = device.createCommandEncoder();
    enc.resolveQuerySet(querySet, 0, 2, resolve, 0);
    enc.copyBufferToBuffer(resolve, 0, read, 0, 16);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const t = new BigUint64Array(read.getMappedRange().slice(0));
    read.unmap();
    return Number(t[1] - t[0]) / 1e6;
}

// --- one count, one mode -----------------------------------------------------------------------
const camera = new Camera(60, WIDTH / HEIGHT, 0.5, 4000);
function field(count) {
    const half = BASE_HALF * Math.sqrt(count / COUNTS[0]);
    let seed = 11;
    const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
    const agents = [];
    for (let i = 0; i < count; i++) {
        agents.push({ wolf: i < count / WOLF_SHARE, x: (random() * 2 - 1) * (half - 1), z: (random() * 2 - 1) * (half - 1),
            heading: random() * Math.PI * 2, speed: random(), phase: random(), resting: random() < 0.2 ? 1 : 0, age: random() * 10 });
    }
    return { half, agents };
}

async function measure(count, mode, { closeUp = false } = {}) {
    const { half, agents } = field(count);
    const [kind, prey] = mode.split(':');
    const preyMesh = kind === 'darts' ? 'prey-dart' : kind === 'weld' ? `${prey}-w` : prey;
    const wolfMesh = kind === 'darts' ? 'wolf-dart' : kind === 'weld' ? 'fox-w' : 'fox';
    const pass = engine.createPass({ name: `spike ${mode}`, isMainScreenPass: true });
    const vis = new VisibilitySystem(engine, { maxInstances: count });
    vis.setGeometry(mega);

    const boundsOf = name => (name.endsWith('dart')
        ? [0, 0.4, 0, name === 'wolf-dart' ? 1.0 : 0.7]
        : [...models[name].bounds.center, models[name].bounds.radius]);
    const rows = new InstanceBuffer(count);
    const state = new Float32Array(count * 4);
    const m = mat4.create();
    agents.forEach((a, i) => {
        const name = a.wolf ? wolfMesh : preyMesh;
        const s = name in scaleOf ? scaleOf[name] : 1;
        mat4.fromTranslation(m, [a.x, 0, a.z]);
        mat4.rotateY(m, m, a.heading);
        mat4.scale(m, m, [s, s, s]);
        rows.write(i, m, boundsOf(name), meshIndex[name], LOD_AUTO, 0);
        state.set([a.speed, a.phase, a.wolf ? a.resting : 0, a.age], i * 4);
    });
    device.queue.writeBuffer(vis.instanceBuffer, 0, rows.data, 0, count * 24);
    vis.setInstanceCount(count);
    const stateBuffer = storage(state);

    const owned = [stateBuffer];
    const batches = [];
    for (const [name, rgb] of [[preyMesh, PREY_RGB], [wolfMesh, WOLF_RGB]]) {
        const layout = { vertexLayouts: StandardLayout.getWebGPUDescriptor() };
        if (kind !== 'hook') {
            batches.push(vis.createInstanceDrawBatch(pass, name, meshIndex[name], { ...layout, fragmentBody: fragmentOf(rgb) }));
            continue;
        }
        const batch = vis.createInstanceDrawBatch(pass, name, meshIndex[name], { ...layout, shaderCode: kind === 'weld' ? HOOK_DRAW_FACETED : HOOK_DRAW });
        const model = models[name];
        const u = new ArrayBuffer(96), u32 = new Uint32Array(u), f32 = new Float32Array(u);
        model.clips.forEach((c, k) => { u32[k * 4] = c.firstFrame; u32[k * 4 + 1] = c.frames; });
        u32[16] = poseBase[name]; u32[17] = model.parts;
        f32.set([...rgb, 1], 20);
        const uniform = storage(new Uint8Array(u), GPUBufferUsage.UNIFORM);
        owned.push(uniform);
        engine.setRenderPassBindGroup(batch, [
            { binding: 0, resource: { buffer: partOfBuffer } },
            { binding: 1, resource: { buffer: posesBuffer } },
            { binding: 2, resource: { buffer: stateBuffer } },
            { binding: 3, resource: { buffer: uniform } },
        ], 2);
        batches.push(batch);
    }

    if (closeUp) camera.updateView([6, 5, 9], [0, 0.6, 0]);
    else {
        const d = 190 * (half / BASE_HALF);
        camera.updateView([d * Math.sin(0.4) * Math.sin(0.3), d * Math.cos(0.4), d * Math.sin(0.4) * Math.cos(0.3)], [0, 0, 0]);
    }
    camera.far = closeUp ? 200 : 190 * 3 * (half / BASE_HALF);
    camera.updateProjection(WIDTH / HEIGHT);
    const sequence = [...vis.getComputeStages(), pass];
    const frame = (seconds) => {
        camera.updateTime(seconds);
        engine.updateCamera(camera);
        vis.upload();
        engine.render(sequence);
    };
    let result = NaN;
    if (closeUp) {
        frame(0.3);
        await device.queue.onSubmittedWorkDone();
    } else {
        const spans = [];
        for (let f = 0; f < WARM + FRAMES; f++) {
            const ms = await timed(() => { for (let r = 0; r < REPEAT; r++) frame((f * REPEAT + r) / 60); });
            if (f >= WARM) spans.push(ms / REPEAT);
        }
        result = median(spans);
    }
    if (closeUp && !inBrowser && globalThis.process?.env?.NG_GPU_ARTIFACTS) {
        const { readTextureRGBA8 } = await import('../../gpu/readback.mjs');
        const rgba = await readTextureRGBA8(device, engine.screenTexture, WIDTH, HEIGHT, engine.ctx.format);
        await env.saveImage(`animated-models-${mode.replace(':', '-')}-close`, rgba, WIDTH, HEIGHT);
    }
    for (const b of batches) vis.removeDrawBatch(b);
    vis.destroy();
    engine.clearPasses();
    owned.forEach(b => b.destroy());
    return result;
}

// --- the counts --------------------------------------------------------------------------------
const MODES = ['darts', 'rest:deer', 'hook:deer', 'hook:cow', 'weld:deer', 'weld:cow'];
const table = {};
let costPerAgent = null;                               // ms per agent at the last count, its slowest mode
for (const count of COUNTS.filter(c => c <= MAX)) {
    if (costPerAgent && costPerAgent * count * 2 > LIMIT_MS) {
        log(`   ${fmt(count)}: skipped, ~${(costPerAgent * count * 2).toFixed(0)} ms per render predicted (limit ${LIMIT_MS})`);
        break;
    }
    table[count] = {};
    for (const mode of MODES) {
        const ms = await measure(count, mode);
        table[count][mode] = ms;
        costPerAgent = Math.max(table[count].worst ?? 0, ms / count);
        table[count].worst = costPerAgent;
    }
    const t = table[count];
    log(`   ${fmt(count).padStart(6)} agents: ${MODES.map(m => `${m} ${t[m].toFixed(2)} ms`).join(' · ')} ` +
        `(culls + draw, one frame, ${WIDTH}x${HEIGHT}, the median of ${FRAMES}; hook over rest ×${(t['hook:deer'] / t['rest:deer']).toFixed(2)}, welded over not ×${(t['weld:deer'] / t['hook:deer']).toFixed(2)})`);
    check(`${fmt(count)}: every mode measured`, MODES.every(m => Number.isFinite(t[m]) && t[m] > 0));
}
for (const mode of ['rest:deer', 'hook:deer', 'hook:cow', 'weld:deer', 'weld:cow']) await measure(4_096, mode, { closeUp: true });
await env.finish('animated-models done');
