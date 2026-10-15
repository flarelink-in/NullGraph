// Cycle D, step 1, phase 5.0 (docs/agents-skinned-models-design.md §6): what skinning costs, drawn
// as Herd & Wolves' agents at its counts, against the rigid hook the demo ships. A measurement, not
// a suite.
//
//   Node + Dawn:  npm run spike:skinned-models      NG_DAWN_FLAGS=adapter=AMD or adapter=NVIDIA
//                 NG_GPU_ARTIFACTS=<dir> also saves a close-up of each mode as PNG
//   Chrome:       npm run drive -- --serve . "/test/spikes/skinned-models/index.html" --wait-for "skinned-models done" --timeout 900
//   NG_SPIKE_MAX (?max=) stops at a count.
//
// The models are the demo's: the test site's cow (prey) and fox (wolves), copied into
// test/spikes/animated-models/models/ (ignored by git), welded and baked by the library's
// bakeRigidAnimation, scaled to 1.3 m and 1.7 m as the demo scales them. A skinned stand-in is made
// from each in code: every vertex lists four influences (its own part, the body, two more), and the
// top 15 cm of each leg blends into the body, so the legs bend at the hip instead of hinging.
//
// Modes (each draws the same rows, one in 16 a fox; the camera sees the whole field):
//   rigid    the demo's hook: one part per vertex, two clips × two frames sampled per vertex
//   skin4    four influences per vertex, every one sampled (f32 weights)
//   skin4b   the same, influences of weight 0 skipped (most vertices have one)
//   skin4u8  four influences, weights packed as four 8-bit unorms in one u32
//   pal1     a compute pass blends each agent's part matrices once per frame (its "palette"); the
//            hook reads one matrix per vertex (rigid)
//   pal4     the palette, and four weighted matrices per vertex (skinned)
import { harnessEnv } from '../../gpu/harness-env.mjs';
import { NullGraph, Camera } from '../../../dist/index.js';
import { StandardLayout, MegabufferBuilder } from '../../../dist/geometry.js';
import { VisibilitySystem, InstanceBuffer, CAMERA_STRUCT_WGSL, INSTANCE_STRUCT_WGSL, CULL_CORE_WGSL } from '../../../dist/visibility.js';
import { GLBParser, bakeRigidAnimation } from '../../../dist/loaders.js';
import { mat4 } from 'gl-matrix';

const env = harnessEnv('__SKIN_SPIKE_RESULTS__');
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
const MODES = ['rigid', 'skin4', 'skin4b', 'skin4u8', 'pal1', 'pal4'];
const median = v => { const s = [...v].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
const fmt = n => n.toLocaleString('en-US');

// --- the engine ------------------------------------------------------------------------------
const engine = new NullGraph();
const init = await engine.init(env.canvas, { width: WIDTH, height: HEIGHT, desiredFeatures: ['timestamp-query'] });
if (!init.success) { env.fatal(init.error); throw new Error(init.error); }
const { device } = engine;
if (!device.features.has('timestamp-query')) { env.fatal('no timestamp-query'); throw new Error('no timestamp-query'); }
const info = device.adapterInfo ?? {};
log(`skinned-models: ${inBrowser ? 'Chrome' : 'Node/Dawn'} · ${[info.vendor, info.architecture, info.description].filter(Boolean).join(' ')}`);

// --- the models, and their skinned stand-ins ---------------------------------------------------
async function load(name, length) {
    const url = new URL(`../animated-models/models/animal-${name}.glb`, import.meta.url);
    const bytes = inBrowser ? await (await fetch(url)).arrayBuffer() : (await import('node:fs')).readFileSync(url);
    const buffer = bytes instanceof ArrayBuffer ? bytes : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const glb = GLBParser.parse(buffer);
    const baked = bakeRigidAnimation(glb, { clips: CLIPS });
    let lo = Infinity, hi = -Infinity;
    for (let v = 2; v < baked.vertices.length; v += 8) { lo = Math.min(lo, baked.vertices[v]); hi = Math.max(hi, baked.vertices[v]); }
    const scale = length / (hi - lo);
    for (let v = 0; v < baked.vertices.length; v += 8) for (let k = 0; k < 3; k++) baked.vertices[v + k] *= scale;
    for (let k = 3; k < baked.transforms.length; k += 4) baked.transforms[k] *= scale;
    const names = baked.partNodes.map(n => glb.nodes[n].name ?? '');
    return { ...baked, names, radius: baked.bounds.radius * scale };
}

/** Four influences per vertex: its part, the body, two more; leg tops blend into the body. */
function standInSkin(model) {
    const { partOfVertex: part, vertices, parts, names } = model;
    const body = Math.max(0, names.indexOf('body'));
    const top = new Array(parts).fill(-Infinity);
    for (let v = 0; v < part.length; v++) top[part[v]] = Math.max(top[part[v]], vertices[v * 8 + 1]);
    const joints = new Uint32Array(part.length), weights = new Float32Array(part.length * 4), packed = new Uint32Array(part.length);
    let blended = 0;
    for (let v = 0; v < part.length; v++) {
        const p = part[v];
        const others = [body, (p + 1) % parts, (p + 2) % parts].map(j => (j === p ? (j + 3) % parts : j));
        const j4 = [p, ...others];
        joints[v] = j4[0] | (j4[1] << 8) | (j4[2] << 16) | (j4[3] << 24);
        let toBody = 0;
        if (names[p].startsWith('leg')) toBody = 0.5 * Math.max(0, 1 - (top[p] - vertices[v * 8 + 1]) / 0.15);
        if (toBody > 0) blended++;
        const w = [1 - toBody, toBody, 0, 0];
        weights.set(w, v * 4);
        packed[v] = w.reduce((word, x, k) => word | (Math.round(x * 255) << (8 * k)), 0) >>> 0;
    }
    return { joints, weights, packed, blended };
}

const models = { prey: await load('cow', 1.3), wolf: await load('fox', 1.7) };
for (const [species, m] of Object.entries(models)) {
    m.skin = standInSkin(m);
    log(`   ${species}: ${fmt(m.indexCount / 3)} triangles, ${fmt(m.vertexCount)} vertices (welded), ${m.parts} parts ` +
        `(${m.names.join(', ')}), ${m.frames} frames; ${fmt(m.skin.blended)} vertices blend two influences`);
}

// One mega buffer, both models; per-vertex data spans the mega (the hook reads it by vertex_index).
const mega = new MegabufferBuilder(StandardLayout.strideFloats);
const partOf = [], joints = [], weights = [], packed = [], poses = [];
const poseBase = {};
for (const [species, m] of Object.entries(models)) {
    mega.addMesh(species, m.vertices, m.indices);
    poseBase[species] = poses.length / 4;
    for (const x of m.transforms) poses.push(x);
    partOf.push(...m.partOfVertex);
    joints.push(...m.skin.joints);
    weights.push(...m.skin.weights);
    packed.push(...m.skin.packed);
}
const storage = (data, usage = GPUBufferUsage.STORAGE) => {
    const buffer = device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 16) * 16), usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
};
const buffers = {
    partOf: storage(new Uint32Array(partOf)), joints: storage(new Uint32Array(joints)),
    weights: storage(new Float32Array(weights)), packed: storage(new Uint32Array(packed)), poses: storage(new Float32Array(poses)),
};
log(`   mega: ${fmt(partOf.length)} vertices; per vertex: part 4 B, influences 4 B, weights 16 B (f32) or 4 B (unorm8)`);

// The two models' clip tables and table offsets, for the hooks and the palette pass.
const ANIM_WGSL = /* wgsl */ `
struct Model {
    clip: array<vec4<u32>, 4>,     // idle, walk, run, eat: first frame (in this model's table), frames
    poseBase: u32,                 // this model's first vec4 in poses
    parts: u32,
    _a: u32, _b: u32,
};
struct Anim { models: array<Model, 2>, wolves: u32, maxParts: u32, _c: u32, _d: u32 };
struct Pose { r0: vec4<f32>, r1: vec4<f32>, r2: vec4<f32> };
fn blend(a: Pose, b: Pose, w: f32) -> Pose { return Pose(mix(a.r0, b.r0, w), mix(a.r1, b.r1, w), mix(a.r2, b.r2, w)); }
fn add(a: Pose, b: Pose, w: f32) -> Pose { return Pose(a.r0 + b.r0 * w, a.r1 + b.r1 * w, a.r2 + b.r2 * w); }`;
// Sampling a clip from the tables, and an agent's pose of one part: reads poses and state.
const SAMPLE_WGSL = /* wgsl */ `
fn poseAt(m: Model, frame: u32, part: u32) -> Pose {
    let k = m.poseBase + (frame * m.parts + part) * 3u;
    return Pose(poses[k], poses[k + 1u], poses[k + 2u]);
}
fn clipPose(m: Model, c: u32, phase: f32, part: u32) -> Pose {
    let f = fract(phase) * f32(m.clip[c].y);
    let f0 = u32(f) % m.clip[c].y;
    let w = fract(f);
    let a = poseAt(m, m.clip[c].x + f0, part);
    let b = poseAt(m, m.clip[c].x + (f0 + 1u) % m.clip[c].y, part);
    return Pose(mix(a.r0, b.r0, w), mix(a.r1, b.r1, w), mix(a.r2, b.r2, w));
}
// An agent's pose of one part: its clips and phases from its state, as the demo's hook takes them.
fn agentPose(row: u32, part: u32) -> Pose {
    let m = anim.models[select(0u, 1u, row < anim.wolves)];
    let s = state[row];
    let t = camera.simTime + s.w;
    if (s.z > 0.0) { return clipPose(m, 3u, t / 0.5, part); }
    let gait = s.y + t * mix(2.0, 4.0, smoothstep(0.45, 0.75, s.x));
    let moving = blend(clipPose(m, 1u, gait, part), clipPose(m, 2u, gait, part), smoothstep(0.45, 0.75, s.x));
    let still = smoothstep(0.08, 0.2, s.x);
    if (still < 1.0) { return blend(clipPose(m, 0u, t, part), moving, still); }
    return moving;
}`;

const unpackJoints = 'let j = vec4<u32>(skinJ[vi] & 255u, (skinJ[vi] >> 8u) & 255u, (skinJ[vi] >> 16u) & 255u, skinJ[vi] >> 24u);';
// How each mode poses a vertex: its declarations, and the body that sets `p` from `row`, `vi`.
const MODE_WGSL = {
    rigid: { decl: '@group(2) @binding(4) var<storage, read> partOf: array<u32>;', body: 'let p = agentPose(row, partOf[vi]);' },
    skin4: {
        decl: '@group(2) @binding(5) var<storage, read> skinJ: array<u32>;\n@group(2) @binding(6) var<storage, read> skinW: array<vec4<f32>>;',
        body: `${unpackJoints}
        let w = skinW[vi];
        var p = add(Pose(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)), agentPose(row, j.x), w.x);
        p = add(p, agentPose(row, j.y), w.y);
        p = add(p, agentPose(row, j.z), w.z);
        p = add(p, agentPose(row, j.w), w.w);`,
    },
    skin4b: {
        decl: '@group(2) @binding(5) var<storage, read> skinJ: array<u32>;\n@group(2) @binding(6) var<storage, read> skinW: array<vec4<f32>>;',
        body: `${unpackJoints}
        let w = skinW[vi];
        var p = add(Pose(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)), agentPose(row, j.x), w.x);
        if (w.y > 0.0) { p = add(p, agentPose(row, j.y), w.y); }
        if (w.z > 0.0) { p = add(p, agentPose(row, j.z), w.z); }
        if (w.w > 0.0) { p = add(p, agentPose(row, j.w), w.w); }`,
    },
    skin4u8: {
        decl: '@group(2) @binding(5) var<storage, read> skinJ: array<u32>;\n@group(2) @binding(7) var<storage, read> skinW8: array<u32>;',
        body: `${unpackJoints}
        let w = unpack4x8unorm(skinW8[vi]);
        var p = add(Pose(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)), agentPose(row, j.x), w.x);
        p = add(p, agentPose(row, j.y), w.y);
        p = add(p, agentPose(row, j.z), w.z);
        p = add(p, agentPose(row, j.w), w.w);`,
    },
    pal1: {
        decl: '@group(2) @binding(4) var<storage, read> partOf: array<u32>;\n@group(2) @binding(8) var<storage, read> palette: array<vec4<f32>>;',
        body: 'let p = paletteAt(row, partOf[vi]);',
    },
    pal4: {
        decl: '@group(2) @binding(5) var<storage, read> skinJ: array<u32>;\n@group(2) @binding(6) var<storage, read> skinW: array<vec4<f32>>;\n@group(2) @binding(8) var<storage, read> palette: array<vec4<f32>>;',
        body: `${unpackJoints}
        let w = skinW[vi];
        var p = add(Pose(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)), paletteAt(row, j.x), w.x);
        p = add(p, paletteAt(row, j.y), w.y);
        p = add(p, paletteAt(row, j.z), w.z);
        p = add(p, paletteAt(row, j.w), w.w);`,
    },
};
const usesPalette = mode => mode.startsWith('pal');
const PALETTE_AT = /* wgsl */ `
fn paletteAt(row: u32, part: u32) -> Pose {
    let k = (row * anim.maxParts + part) * 3u;
    return Pose(palette[k], palette[k + 1u], palette[k + 2u]);
}`;

const drawWGSL = (mode, rgb) => /* wgsl */ `
${CAMERA_STRUCT_WGSL}
${INSTANCE_STRUCT_WGSL}
${CULL_CORE_WGSL}
struct VertexOut {
    @builtin(position) clipPos: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
${usesPalette(mode) ? '' : `@group(2) @binding(0) var<storage, read> poses: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read> state: array<vec4<f32>>;`}
@group(2) @binding(2) var<uniform> anim: Anim;
${MODE_WGSL[mode].decl}
${ANIM_WGSL}
${usesPalette(mode) ? PALETTE_AT : SAMPLE_WGSL}

@vertex
fn vs_main(@location(0) pos: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>,
           @builtin(instance_index) ii: u32, @builtin(vertex_index) vi: u32) -> VertexOut {
    let inst = instances[ii];
    let row = inst.sourceIndex;
    ${MODE_WGSL[mode].body}
    let lp = vec4<f32>(pos, 1.0);
    let world = inst.model * vec4<f32>(dot(p.r0, lp), dot(p.r1, lp), dot(p.r2, lp), 1.0);
    var out: VertexOut;
    out.clipPos = camera.viewProj * world;
    out.worldPos = world.xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(cross(dpdx(in.worldPos), dpdy(in.worldPos)));
    let sun = abs(dot(n, normalize(vec3<f32>(0.4, 1.0, 0.3))));
    let light = 0.42 + 0.28 * smoothstep(0.25, 0.3, sun) + 0.3 * smoothstep(0.65, 0.7, sun);
    return vec4<f32>(vec3<f32>(${rgb.join(', ')}) * light, 1.0);
}`;

// The palette pass: one thread per agent and part, each part's matrix blended once per frame.
const PALETTE_WGSL = /* wgsl */ `
${CAMERA_STRUCT_WGSL}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> poses: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> state: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> anim: Anim;
@group(0) @binding(4) var<storage, read_write> palette: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> rows: vec4<u32>;
${ANIM_WGSL}
${SAMPLE_WGSL}
@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x / anim.maxParts;
    let part = gid.x % anim.maxParts;
    if (row >= rows.x) { return; }
    let m = anim.models[select(0u, 1u, row < anim.wolves)];
    if (part >= m.parts) { return; }
    let p = agentPose(row, part);
    let k = (row * anim.maxParts + part) * 3u;
    palette[k] = p.r0; palette[k + 1u] = p.r1; palette[k + 2u] = p.r2;
}`;

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
    return { half, agents: Array.from({ length: count }, (_, i) => ({
        wolf: i < count / WOLF_SHARE, x: (random() * 2 - 1) * (half - 1), z: (random() * 2 - 1) * (half - 1),
        heading: random() * Math.PI * 2, speed: random(), phase: random(), resting: random() < 0.2 ? 1 : 0, age: random() * 10 })) };
}
const animUniform = count => {
    const u = new ArrayBuffer(2 * 80 + 16), u32 = new Uint32Array(u);
    ['prey', 'wolf'].forEach((species, k) => {
        const m = models[species], o = k * 20;
        m.clips.forEach((c, i) => { u32[o + i * 4] = c.firstFrame; u32[o + i * 4 + 1] = c.frames; });
        u32[o + 16] = poseBase[species];
        u32[o + 17] = m.parts;
    });
    u32[40] = count / WOLF_SHARE;
    u32[41] = Math.max(models.prey.parts, models.wolf.parts);
    return storage(new Uint8Array(u), GPUBufferUsage.UNIFORM);
};

async function measure(count, mode, { closeUp = false } = {}) {
    const { half, agents } = field(count);
    const pass = engine.createPass({ name: `skin ${mode}`, isMainScreenPass: true });
    const vis = new VisibilitySystem(engine, { maxInstances: count });
    vis.setGeometry(mega);
    const rows = new InstanceBuffer(count), state = new Float32Array(count * 4), m = mat4.create();
    agents.forEach((a, i) => {
        mat4.fromTranslation(m, [a.x, 0, a.z]);
        mat4.rotateY(m, m, a.heading);
        rows.write(i, m, [0, 0, 0, a.wolf ? models.wolf.radius : models.prey.radius], a.wolf ? 1 : 0);
        state.set([a.speed, a.phase, a.wolf ? a.resting : 0, a.age], i * 4);
    });
    device.queue.writeBuffer(vis.instanceBuffer, 0, rows.data, 0, count * 24);
    vis.setInstanceCount(count);
    const owned = [storage(state), animUniform(count)];
    const [stateBuffer, anim] = owned;
    const maxParts = Math.max(models.prey.parts, models.wolf.parts);
    let palette = null, stages = [];
    if (usesPalette(mode)) {
        palette = device.createBuffer({ size: count * maxParts * 48, usage: GPUBufferUsage.STORAGE });
        const rowsUniform = storage(new Uint32Array([count, 0, 0, 0]), GPUBufferUsage.UNIFORM);
        owned.push(palette, rowsUniform);
        const stage = engine.createComputeStage({ name: 'palette', shaderCode: PALETTE_WGSL, dispatchCount: () => [Math.ceil(count * maxParts / 64), 1, 1] });
        engine.setComputePassBindGroup(stage, [
            { binding: 0, resource: { buffer: engine.ctx.cameraUniformBuffer } },
            { binding: 1, resource: { buffer: buffers.poses } }, { binding: 2, resource: { buffer: stateBuffer } },
            { binding: 3, resource: { buffer: anim } }, { binding: 4, resource: { buffer: palette } },
            { binding: 5, resource: { buffer: rowsUniform } },
        ], 0);
        stages.push(stage);
    }
    const resources = { 0: buffers.poses, 1: stateBuffer, 2: anim, 4: buffers.partOf, 5: buffers.joints, 6: buffers.weights, 7: buffers.packed, 8: palette };
    const batches = [['prey', 0, [0.93, 0.92, 0.88]], ['wolf', 1, [0.98, 0.52, 0.16]]].map(([name, mesh, rgb]) => {
        const code = drawWGSL(mode, rgb);
        const batch = vis.createInstanceDrawBatch(pass, name, mesh, { vertexLayouts: StandardLayout.getWebGPUDescriptor(), shaderCode: code });
        const used = [...code.matchAll(/@group\(2\) @binding\((\d+)\)/g)].map(x => Number(x[1]));
        engine.setRenderPassBindGroup(batch, used.map(binding => ({ binding, resource: { buffer: resources[binding] } })), 2);
        return batch;
    });

    if (closeUp) camera.updateView([5, 4, 7], [0, 0.6, 0]);
    else {
        const d = 190 * (half / BASE_HALF);
        camera.updateView([d * Math.sin(0.4) * Math.sin(0.3), d * Math.cos(0.4), d * Math.sin(0.4) * Math.cos(0.3)], [0, 0, 0]);
    }
    camera.far = closeUp ? 200 : 190 * 3 * (half / BASE_HALF);
    camera.updateProjection(WIDTH / HEIGHT);
    const sequence = [...stages, ...vis.getComputeStages(), pass];
    const frame = seconds => { camera.updateTime(seconds); engine.updateCamera(camera); vis.upload(); engine.render(sequence); };
    let result = NaN;
    if (closeUp) {
        frame(0.3);
        await device.queue.onSubmittedWorkDone();
        if (!inBrowser && globalThis.process?.env?.NG_GPU_ARTIFACTS) {
            const { readTextureRGBA8 } = await import('../../gpu/readback.mjs');
            const rgba = await readTextureRGBA8(device, engine.screenTexture, WIDTH, HEIGHT, engine.ctx.format);
            await env.saveImage(`skinned-models-${mode}-close`, rgba, WIDTH, HEIGHT);
        }
    } else {
        const spans = [];
        for (let f = 0; f < WARM + FRAMES; f++) {
            const ms = await timed(() => { for (let r = 0; r < REPEAT; r++) frame((f * REPEAT + r) / 60); });
            if (f >= WARM) spans.push(ms / REPEAT);
        }
        result = median(spans);
    }
    for (const b of batches) vis.removeDrawBatch(b);
    vis.destroy();
    engine.clearPasses();
    owned.forEach(b => b.destroy());
    return result;
}

// --- the counts --------------------------------------------------------------------------------
const table = {};
let costPerAgent = null;
for (const count of COUNTS.filter(c => c <= MAX)) {
    if (costPerAgent && costPerAgent * count * 2 > LIMIT_MS) {
        log(`   ${fmt(count)}: skipped, ~${(costPerAgent * count * 2).toFixed(0)} ms per render predicted (limit ${LIMIT_MS})`);
        break;
    }
    table[count] = {};
    for (const mode of MODES) {
        const ms = await measure(count, mode);
        table[count][mode] = ms;
        costPerAgent = Math.max(costPerAgent ?? 0, ms / count);
    }
    const t = table[count];
    log(`   ${fmt(count).padStart(6)} agents: ${MODES.map(m => `${m} ${t[m].toFixed(2)}`).join(' · ')} ms ` +
        `(palette and culls + draw, one frame, ${WIDTH}x${HEIGHT}, the median of ${FRAMES}; skin4 / rigid ×${(t.skin4 / t.rigid).toFixed(2)}, pal4 / rigid ×${(t.pal4 / t.rigid).toFixed(2)})`);
    check(`${fmt(count)}: every mode measured`, MODES.every(m => Number.isFinite(t[m]) && t[m] > 0));
}
for (const mode of MODES) await measure(4_096, mode, { closeUp: true });
await env.finish('skinned-models done');
