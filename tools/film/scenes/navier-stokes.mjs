// Periodic 2D vorticity/streamfunction solver. A visualization, not validated DNS.
import { hash } from 'node:crypto';
import { Camera } from '../../../dist/index.js';

export function resolveSettings(settings = {}) {
    const grid = settings.grid ?? 128;
    const iterations = settings.iterations ?? 120;
    const viscosity = settings.viscosity ?? 0.0002;
    if (!Number.isInteger(grid) || grid < 32 || grid > 256) throw new Error('navier-stokes: grid must be 32–256');
    if (!Number.isInteger(iterations) || iterations < 2 || iterations > 400 || iterations % 2) throw new Error('navier-stokes: iterations must be even, 2–400');
    if (!Number.isFinite(viscosity) || viscosity < 0 || viscosity * grid * grid / 120 > 0.24) throw new Error('navier-stokes: viscosity exceeds explicit diffusion stability bound');
    return { grid, iterations, viscosity };
}

export function initialState(n) {
    const data = new Float32Array(n * n * 4);
    // Double shear layer, derived from an exactly periodic streamfunction.
    // u = d(psi)/dy, v = -d(psi)/dx, omega = -laplacian(psi).
    const psi = (x, y) => -Math.log(Math.cosh(6 * Math.cos(2 * Math.PI * y))) / (12 * Math.PI)
        + 0.012 * (Math.cos(2 * Math.PI * x) + 0.35 * Math.cos(6 * Math.PI * x)) / (2 * Math.PI);
    const h = 1 / n;
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const px = x * h, py = y * h, p = psi(px, py), i = (y * n + x) * 4;
        data[i] = -(psi(px+h,py)+psi(px-h,py)+psi(px,py+h)+psi(px,py-h)-4*p)/(h*h);
        data[i+1] = 0.5 + 0.5 * Math.tanh(6 * Math.cos(2 * Math.PI * py));
        data[i+2] = p;
    }
    return data;
}

export function solverShaders({ grid: n, viscosity }) {
    const common = `
const N: i32 = ${n};
const H: f32 = ${1/n};
const DT: f32 = ${1/120};
@group(0) @binding(0) var<storage, read> src: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4f>;
fn at(p: vec2i) -> vec4f { let q = (p % vec2i(N) + vec2i(N)) % vec2i(N); return src[u32(q.y*N+q.x)]; }
`;
    return {
        poisson: common + `
@compute @workgroup_size(8,8) fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= u32(N) || id.y >= u32(N)) { return; }
    let p = vec2i(id.xy); let a = at(p);
    let z = (at(p+vec2i(1,0)).z + at(p-vec2i(1,0)).z + at(p+vec2i(0,1)).z + at(p-vec2i(0,1)).z + H*H*a.x)*0.25;
    // Weighted Jacobi removes the undamped checkerboard mode on a periodic grid.
    dst[id.y*u32(N)+id.x] = vec4f(a.xy, mix(a.z,z,0.8), 0);
}`,
        advect: common + `
fn velocity(p: vec2i) -> vec2f {
    return vec2f(at(p+vec2i(0,1)).z-at(p-vec2i(0,1)).z,
        at(p-vec2i(1,0)).z-at(p+vec2i(1,0)).z)/(2*H);
}
fn sampleField(p: vec2f) -> vec4f {
    let q = vec2i(floor(p)); let f = fract(p);
    return mix(mix(at(q),at(q+vec2i(1,0)),f.x),mix(at(q+vec2i(0,1)),at(q+vec2i(1,1)),f.x),f.y);
}
@compute @workgroup_size(8,8) fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= u32(N) || id.y >= u32(N)) { return; }
    let p = vec2i(id.xy); let a = at(p); let vel = velocity(p);
    let mid = vec2i(round(vec2f(p)-0.5*DT*vel/H));
    let back = vec2f(p)-DT*velocity(mid)/H;
    let b = sampleField(back);
    let lap = (at(p+vec2i(1,0)).x+at(p-vec2i(1,0)).x+at(p+vec2i(0,1)).x+at(p-vec2i(0,1)).x-4*a.x)/(H*H);
    dst[id.y*u32(N)+id.x] = vec4f(b.x + ${viscosity}*DT*lap,b.y,a.z,0);
}`,
    };
}

export function displayShader(n) {
    return `
@group(0) @binding(0) var<uniform> camera: array<vec4f,27>;
@group(0) @binding(1) var<storage,read> field: array<vec4f>;
struct V { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> V {
    let p = array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));
    var o: V; o.pos = vec4f(p[i],0.5,1); o.uv = p[i]*0.5+0.5; return o;
}
fn at(p: vec2i) -> vec4f { let q = (p % vec2i(${n})+vec2i(${n})) % vec2i(${n}); return field[u32(q.y*${n}+q.x)]; }
@fragment fn fs_main(v: V) -> @location(0) vec4f {
    let uv = vec2f(v.uv.x,1-v.uv.y);
    let p = uv*f32(${n}); let q = vec2i(floor(p)); let f = fract(p);
    let a = mix(mix(at(q),at(q+vec2i(1,0)),f.x),mix(at(q+vec2i(0,1)),at(q+vec2i(1,1)),f.x),f.y);
    let strength = 1-exp(-abs(a.x)*0.07);
    let cold = vec3f(0.06,0.72,0.96); let hot = vec3f(1.0,0.43,0.08);
    let tint = mix(cold,hot,smoothstep(-1.5,1.5,a.x));
    let contour = pow(0.5+0.5*cos(a.y*100.0),18.0);
    var color = vec3f(0.008,0.018,0.035) + tint*strength*(0.65+0.35*a.y);
    color += mix(cold,hot,a.y)*contour*0.18;
    let grid = pow(max(0.0,max(cos(uv.x*100.531),cos(uv.y*100.531))),80.0);
    color += vec3f(0.02,0.04,0.06)*grid;
    // Keep the engine camera binding active; it does not alter the physical field.
    return vec4f(color + camera[0].xyz*1e-20,1);
}`;
}

export default async function navierStokes({ engine, width, height, settings = {}, film, log = () => {} }) {
    const resolved = resolveSettings(settings), { grid, iterations } = resolved;
    const { device } = engine, shaders = solverShaders(resolved), display = displayShader(grid);
    const pass = engine.createPass({ name: 'Navier–Stokes / signed vorticity', isMainScreenPass: true });
    const batch = engine.createBatch(pass, { shaderCode: display, strideFloats: 4,
        maxInstances: grid*grid, depthWriteEnabled: false });
    // updateBatchData writes only instanceCount * stride bytes. Upload the entire
    // grid explicitly, but draw only one fullscreen triangle, not grid² triangles.
    engine.updateBatchData(batch, new Float32Array(4), 1);
    device.queue.writeBuffer(batch.storageBuffer, 0, initialState(grid));
    const state = batch.storageBuffer;
    const scratch = [0,1].map(i => device.createBuffer({ label: `fluid scratch ${i}`, size: state.size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }));
    const stage = (name, code, from, to) => {
        const s = engine.createComputeStage({ name, shaderCode: code, dispatchCount: [Math.ceil(grid/8),Math.ceil(grid/8),1] });
        engine.setComputePassBindGroup(s, [from,to].map((buffer,binding) => ({ binding, resource: { buffer } })));
        return s;
    };
    const solve = [stage('fluid / Jacobi A',shaders.poisson,scratch[0],scratch[1]),
        stage('fluid / Jacobi B',shaders.poisson,scratch[1],scratch[0])];
    const advect = stage('fluid / vorticity transport',shaders.advect,scratch[0],state);
    const sequence = Array.from({ length: iterations }, (_,i) => solve[i%2]);
    sequence.push(advect);
    const camera = new Camera(45,width/height,0.1,10);
    camera.updateView([0,0,2],[0,0,0]); engine.updateCamera(camera);
    log(`Navier–Stokes: ${grid}² periodic cells, ν=${resolved.viscosity}, dt=1/120; semi-Lagrangian, not conservation-exact.`);
    return {
        dt: 1/120, settings: resolved,
        tracks: [film.track('fluid',state,{ stride: 16, every: 1 })],
        checkpoint: [{ name: 'fluid', buffer: state }],
        info: { model: '2D periodic incompressible vorticity/streamfunction; double shear layer',
            limitations: 'Bilinear semi-Lagrangian numerical diffusion; finite-iteration Poisson solve; not DNS or a structure-preserving research implementation.',
            wgsl: Object.fromEntries(Object.entries({ ...shaders, display }).map(([k,v]) => [k,hash('sha256',v)])) },
        async step() {
            // Always restart scratch from canonical state: checkpoint/replay needs no JS ping-pong index.
            const encoder = device.createCommandEncoder();
            encoder.copyBufferToBuffer(state,0,scratch[0],0,state.size);
            device.queue.submit([encoder.finish()]);
            engine.render(sequence); await device.queue.onSubmittedWorkDone();
        },
        view(_state, seconds) { camera.updateTime(seconds); engine.updateCamera(camera); },
        draw() { return [pass]; },
        destroy() { scratch.forEach(b => b.destroy()); engine.clearPasses(); },
    };
}