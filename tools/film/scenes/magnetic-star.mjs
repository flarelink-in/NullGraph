// A stylized magnetic accretion sculpture, not an astrophysical solver.
// Persistent particles evolve on the GPU; drawing never advances their phases.
import { hash } from 'node:crypto';
import { Camera } from '../../../dist/index.js';
import { PostChain, Bloom, Tonemap } from '../../../dist/post.js';

const CAMERA = `struct Camera {
    viewProj: mat4x4f, view: mat4x4f, proj: mat4x4f, invViewProj: mat4x4f,
    invView: mat4x4f, invProj: mat4x4f, eye: vec3f, time: f32,
    near: f32, far: f32, fov: f32, aspect: f32, pad: vec4f,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> particles: array<vec4f>;
`;

export const EVOLVE = `
@group(0) @binding(0) var<storage, read_write> particles: array<vec4f>;
@compute @workgroup_size(128) fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x; if (i * 2u >= arrayLength(&particles)) { return; }
    var a = particles[i * 2u]; let b = particles[i * 2u + 1u];
    let phase = a.w + (0.16 + 1.8 / (b.x * b.x)) / 60.0;
    let angle = phase + b.z * 6.2831853;
    var p = vec3f(b.x * cos(angle), b.y * 0.15, b.x * sin(angle));
    if (b.w > 0.77) {
        // Dipole-like closed filaments: azimuth fixed, beads travel along each arch.
        let u = phase * 1.5 + b.y * 6.2831853;
        let shell = floor(b.x * 2.0) / 2.0;
        let r = 1.55 + shell * sin(u) * sin(u);
        let az = floor(b.z * 16.0) / 16.0 * 6.2831853 + b.y * 0.009;
        p = vec3f(r * sin(u) * cos(az),
            r * cos(u) * 1.12, r * sin(u) * sin(az));
    } else {
        p.y += 0.12 * sin(angle * 7.0 + b.x * 2.5) +
            0.22 * sin(angle * 3.0 + b.x * 1.7);
    }
    particles[i * 2u] = vec4f(p, phase);
}`;

const PARTICLES = CAMERA + `
struct V { @builtin(position) pos: vec4f, @location(0) uv: vec2f,
    @location(1) color: vec3f, @location(2) gain: f32 };
@vertex fn vs_main(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> V {
    let a = particles[i * 2u]; let b = particles[i * 2u + 1u];
    let corners = array<vec2f, 3>(vec2f(-1.0,-1.0),vec2f(3.0,-1.0),vec2f(-1.0,3.0));
    let uv = corners[v]; var clip = camera.viewProj * vec4f(a.xyz, 1.0);
    let size = (0.009 + 0.012 * b.w) * camera.proj[1][1];
    clip.x += uv.x * size / camera.aspect; clip.y += uv.y * size;
    let ray = a.xyz - camera.eye;
    let t = clamp(-dot(camera.eye, ray) / dot(ray, ray), 0.0, 1.0);
    if (length(camera.eye + t * ray) < 1.48 && t < 0.999) { clip = vec4f(2.0,2.0,2.0,1.0); }
    var o: V; o.pos = clip; o.uv = uv;
    let hot = vec3f(3.6, 0.64, 0.055); let cold = vec3f(0.025, 1.1, 2.8);
    o.color = mix(hot, cold, smoothstep(0.77, 0.8, b.w));
    if (b.w <= 0.77) { o.color = mix(vec3f(4.0,2.4,0.8), hot, smoothstep(1.8,5.8,b.x)); }
    o.gain = 0.055 + 0.10 * pow(0.5 + 0.5 * sin(a.w * 3.0 + b.z * 44.0), 8.0);
    return o;
}
@fragment fn fs_main(v: V) -> @location(0) vec4f {
    let r = dot(v.uv,v.uv); if (r > 1.0) { discard; }
    return vec4f(v.color * exp(-r * 4.5) * v.gain, 1.0);
}`;

const BACKDROP = CAMERA + `
struct V { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> V {
    let p = array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));
    var o: V; o.pos = vec4f(p[i],0.999,1); o.uv = p[i]; return o;
}
@fragment fn fs_main(v: V) -> @location(0) vec4f {
    let h = camera.invViewProj * vec4f(v.uv,1,1);
    let rd = normalize(h.xyz / h.w - camera.eye);
    let t = -dot(camera.eye,rd); let impact = length(camera.eye + t * rd);
    var col = vec3f(0.0006,0.001,0.004);
    let halo = exp(-pow((impact - 1.55) * 5.0,2.0));
    col += vec3f(1.3,0.32,0.04) * halo;
    col += vec3f(0.004,0.012,0.025) * exp(-impact * 0.25);
    if (impact < 1.48 && t > 0.0) { col = vec3f(0.0001,0.0002,0.0005); }
    // Sparse, stable screen-space star dust; no temporal random flicker.
    let cell = floor(v.pos.xy / 2.0);
    let noise = fract(sin(dot(cell,vec2f(12.9898,78.233))) * 43758.5453);
    if (noise > 0.9995 && impact > 2.0) { col += vec3f(0.12,0.18,0.3); }
    return vec4f(col + particles[0].xyz * 0.0,1);
}`;

export default async function magneticStar({ engine, width, height, settings = {}, film, log = () => {} }) {
    const count = settings.count ?? 1048576;
    if (!Number.isInteger(count) || count < 128 || count > 2097152) throw new Error('magnetic-star: count must be 128–2097152');
    const { device } = engine;
    const post = new PostChain(engine);
    post.setChain([new Bloom({ threshold: 0.7, knee: 0.5, intensity: 0.6, mips: 6 }), new Tonemap({ curve: 'aces', exposure: 1.25 })]);
    post.resize(width, height);
    const pass = engine.createPass({ name: 'Magnetic star / HDR', isMainScreenPass: false,
        ...post.sceneAttachments({ clearValue: { r: 0, g: 0, b: 0, a: 1 } }) });
    const background = engine.createBatch(pass, { shaderCode: BACKDROP, strideFloats: 8,
        maxInstances: 1, targetFormat: 'rgba16float', depthWriteEnabled: false });
    engine.updateBatchData(background, new Float32Array(8), 1);
    const batch = engine.createBatch(pass, { shaderCode: PARTICLES, strideFloats: 8,
        maxInstances: count, targetFormat: 'rgba16float', depthWriteEnabled: false,
        blend: { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' } } });
    const data = new Float32Array(count * 8);
    let seed = 271828;
    const rand = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
    for (let i = 0; i < count; i++) {
        const radius = 1.65 + Math.pow(rand(), 1.5) * 5.3;
        const height = rand() * 2 - 1, azimuth = rand(), kind = rand();
        data.set([0,0,0,rand()*Math.PI*2,radius,height,azimuth,kind], i*8);
    }
    engine.updateBatchData(batch, data, count);
    const evolve = engine.createComputeStage({ name: 'Magnetic star / particle orbits', shaderCode: EVOLVE,
        dispatchCount: [Math.ceil(count / 128),1,1] });
    engine.setComputePassBindGroup(evolve, [{ binding: 0, resource: { buffer: batch.storageBuffer } }]);
    // Initialize positions before the runner saves step zero.
    engine.render([evolve]); await device.queue.onSubmittedWorkDone();
    const camera = new Camera(47, width / height, 0.1, 100);
    const defaultView = { target: [0,0,0], distance: 21, azimuth: 0.2, polar: 1.12 };
    const track = film.track('plasma', batch.storageBuffer, { every: 6, stride: 32, fields:
        [0,1,2,3,4,5,6,7].map(word => ({ word, store: 'f32', interp: 'linear' })) });
    log(`Magnetic star: ${count.toLocaleString()} GPU particles, ${(data.byteLength/1048576).toFixed(0)} MiB persistent state; stylized, not physical.`);
    return {
        dt: 1/60, settings: { count }, tracks: [track], checkpoint: [{ name: 'plasma', buffer: batch.storageBuffer }],
        info: { particles: count, model: 'stylized accretion disk and dipole filaments',
            wgsl: { evolve: hash('sha256', EVOLVE), particles: hash('sha256', PARTICLES), backdrop: hash('sha256', BACKDROP) } },
        async step() { engine.render([evolve]); await device.queue.onSubmittedWorkDone(); },
        view(state, seconds) {
            const v = state ?? defaultView, s = Math.sin(v.polar);
            camera.updateView([v.target[0]+v.distance*s*Math.sin(v.azimuth), v.target[1]+v.distance*Math.cos(v.polar),
                v.target[2]+v.distance*s*Math.cos(v.azimuth)], v.target);
            camera.updateTime(seconds); engine.updateCamera(camera);
        },
        draw() { post.upload(); return [pass, ...post.getPassSequence()]; },
        destroy() { post.destroy(); engine.clearPasses(); },
    };
}