// Small weakly-compressible SPH experiment; not validated CFD.
// Based on the poly6/spiky kernels of Mueller, Charypar & Gross (SCA 2003).
import { hash } from 'node:crypto';
import { Camera } from '../../../dist/index.js';
import { PostChain, Bloom, Tonemap, FXAA } from '../../../dist/post.js';

export const COUNT = 1491;
export const GRID = 56;
export const DT = 1 / 240;
export function initialLiquid() {
    const data = new Float32Array(COUNT * 8);
    let i = 0;
    const add = (x, y, z, vx, vy, vz) => {
        data.set([x, y, z, 0, vx, vy, vz, 0], i++ * 8);
    };
    // Concentric square-to-disk mapping preserves the particle count without
    // leaving square corners in the receiving pool. No particles overlap at the center.
    for (let z = 0; z < 20; z++) for (let x = 0; x < 20; x++) for (let y = 0; y < 2; y++) {
        const u = (x - 9.5) / 10, v = (z - 9.5) / 10;
        const horizontal = Math.abs(u) > Math.abs(v);
        const radius = 1.5 * (horizontal ? u : v);
        const angle = horizontal ? Math.PI / 4 * v / u : Math.PI / 2 - Math.PI / 4 * u / v;
        add(radius * Math.cos(angle), 0.11 + y * 0.14, radius * Math.sin(angle), 0, 0, 0);
    }
    // A rounded falling slug, slightly off-axis for an asymmetric splash.
    for (let z = -6; z <= 6; z++) for (let y = -6; y <= 6; y++) for (let x = -6; x <= 6; x++) {
        if (x*x + y*y + z*z > 29) continue;
        add(x * 0.14 - 0.15, y * 0.14 + 1.75, z * 0.14, 0.65, -2.7, 0.15);
    }
    if (i !== COUNT) throw new Error(`liquid-splash: expected ${COUNT} particles, got ${i}`);
    return data;
}

const COMMON = `
struct Particle { p: vec4f, v: vec4f };
const COUNT: u32 = ${COUNT}u;
const H: f32 = 0.28;
const MASS: f32 = 2.744;
const DT: f32 = ${DT};
@group(0) @binding(0) var<storage,read> src: array<Particle>;
@group(0) @binding(1) var<storage,read_write> dst: array<Particle>;
`;
export const DENSITY = COMMON + `
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x; if (i >= COUNT) { return; }
    var rho = 0.0;
    for (var j = 0u; j < COUNT; j++) {
        let d = src[i].p.xyz - src[j].p.xyz;
        let q = max(0.0, H*H-dot(d,d));
        rho += MASS * 315.0 / (64.0*3.14159265*pow(H,9.0)) * q*q*q;
    }
    dst[i] = src[i]; dst[i].p.w = max(rho, 100.0);
}`;
export const INTEGRATE = COMMON + `
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x; if (i >= COUNT) { return; }
    let a = src[i]; let rho = a.p.w;
    let pressure = 45.0 * max(rho - 1000.0, 0.0);
    var acc = vec3f(0,-9.81,0);
    for (var j = 0u; j < COUNT; j++) {
        if (i == j) { continue; }
        let b = src[j]; let d = a.p.xyz - b.p.xyz; let r = length(d);
        if (r < 0.0001 || r >= H) { continue; }
        let q = H-r; let k = 45.0/(3.14159265*pow(H,6.0));
        let pj = 45.0 * max(b.p.w - 1000.0, 0.0);
        acc += MASS*(pressure/(rho*rho)+pj/(b.p.w*b.p.w))*k*q*q*d/r;
        acc += 0.16*MASS/b.p.w*k*q*(b.v.xyz-a.v.xyz);
        // Mild cohesion is an artistic aid, not a calibrated surface-tension model.
        acc -= 0.12*MASS/b.p.w*k*q*q*d;
    }
    var v = a.v.xyz + DT*acc;
    var p = a.p.xyz + DT*v;
    // Cylindrical side wall: impacts cannot settle into a square footprint.
    let radius = length(p.xz);
    if (radius > 1.52) {
        let normal = p.xz / radius;
        let radialVelocity = dot(v.xz,normal);
        p = vec3f(normal.x*1.52,p.y,normal.y*1.52);
        if (radialVelocity > 0.0) {
            v -= vec3f(normal.x,0,normal.y)*(1.22*radialVelocity);
        }
    }
    if (p.y < 0.075) { p.y = 0.075; v.y = abs(v.y)*0.22; }
    if (p.y > 3.0) { p.y = 3.0; v.y = -abs(v.y)*0.22; }
    dst[i] = Particle(vec4f(p,rho),vec4f(v,0));
}`;

export const VOLUME = `
struct Particle { p: vec4f, v: vec4f };
@group(0) @binding(0) var<storage,read> particles: array<Particle>;
@group(0) @binding(1) var<storage,read_write> field: array<f32>;
@compute @workgroup_size(4,4,4) fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    if (any(id >= vec3u(${GRID}))) { return; }
    let p = vec3f(-1.7,-0.1,-1.7) + vec3f(id) * (3.4/${GRID-1}.0);
    var value = 0.0;
    for (var j = 0u; j < ${COUNT}u; j++) {
        let d = p-particles[j].p.xyz;
        let q = max(0.0,1.0-dot(d,d)/(0.235*0.235));
        value += q*q*q;
    }
    field[(id.z*${GRID}u+id.y)*${GRID}u+id.x] = value;
}`;

export const SURFACE = `
struct Camera {
    viewProj: mat4x4f, view: mat4x4f, proj: mat4x4f, invViewProj: mat4x4f,
    invView: mat4x4f, invProj: mat4x4f, eye: vec3f, time: f32,
    near: f32, far: f32, fov: f32, aspect: f32, pad: vec4f,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage,read> field: array<f32>;
struct Vertex { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> Vertex {
    let p = array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));
    var o: Vertex; o.pos=vec4f(p[i],0,1); o.uv=p[i]; return o;
}
fn at(p: vec3i) -> f32 {
    let q = clamp(p,vec3i(0),vec3i(${GRID-1}));
    return field[(q.z*${GRID}+q.y)*${GRID}+q.x];
}
fn density(p: vec3f) -> f32 {
    let u = (p-vec3f(-1.7,-0.1,-1.7))*(${GRID-1}.0/3.4);
    if (any(u<vec3f(0)) || any(u>vec3f(${GRID-1}))) { return 0.0; }
    let q=vec3i(floor(u)); let f=fract(u);
    return mix(mix(mix(at(q),at(q+vec3i(1,0,0)),f.x),
        mix(at(q+vec3i(0,1,0)),at(q+vec3i(1,1,0)),f.x),f.y),
        mix(mix(at(q+vec3i(0,0,1)),at(q+vec3i(1,0,1)),f.x),
        mix(at(q+vec3i(0,1,1)),at(q+vec3i(1,1,1)),f.x),f.y),f.z);
}
fn studio(d: vec3f) -> vec3f {
    var col = mix(vec3f(0.012,0.025,0.045),vec3f(0.15,0.23,0.30),smoothstep(-0.4,0.8,d.y));
    col += vec3f(3.8,4.2,4.6)*pow(max(0.0,dot(d,normalize(vec3f(-0.6,1,0.4)))),36.0);
    col += vec3f(1.8,0.95,0.4)*pow(max(0.0,dot(d,normalize(vec3f(1,0.35,-0.5)))),60.0);
    col += vec3f(0.4,1.3,1.7)*pow(max(0.0,dot(d,normalize(vec3f(-1,0.2,-1)))),30.0);
    return col;
}
@fragment fn fs_main(v: Vertex) -> @location(0) vec4f {
    let h = camera.invViewProj*vec4f(v.uv,1,1);
    let rd = normalize(h.xyz/h.w-camera.eye); let ro=camera.eye;
    var col = studio(rd)*0.25;
    let floorT = -ro.y/rd.y;
    if (floorT>0.0) {
        let floorP=ro+rd*floorT;
        col = vec3f(0.035,0.052,0.068)*(0.7+0.3*exp(-0.15*dot(floorP.xz,floorP.xz)));
        col *= 1.0-0.55*exp(-0.6*dot(floorP.xz,floorP.xz));
    }
    let t0=(vec3f(-1.7,-0.1,-1.7)-ro)/rd;
    let t1=(vec3f(1.7,3.3,1.7)-ro)/rd;
    let low=min(t0,t1); let high=max(t0,t1);
    var t=max(0.0,max(low.x,max(low.y,low.z)));
    let end=min(high.x,min(high.y,high.z));
    var hit=false;
    for (var i=0; i<440; i++) {
        if (t>end) { break; }
        if (density(ro+rd*t)>0.62) { hit=true; break; }
        t+=0.0125;
    }
    if (hit) {
        var a=max(0.0,t-0.0125); var b=t;
        for (var j=0; j<6; j++) {
            let mid=(a+b)*0.5;
            if (density(ro+rd*mid)>0.62) { b=mid; } else { a=mid; }
        }
        let p=ro+rd*b; let e=0.018;
        let n=normalize(vec3f(density(p-vec3f(e,0,0))-density(p+vec3f(e,0,0)),
            density(p-vec3f(0,e,0))-density(p+vec3f(0,e,0)),
            density(p-vec3f(0,0,e))-density(p+vec3f(0,0,e))));
        let fresnel=0.0204+0.9796*pow(1.0-max(0.0,dot(n,-rd)),5.0);
        let reflection=studio(reflect(rd,n));
        let transmission=vec3f(0.015,0.38,0.43)*(0.5+0.5*max(0.0,dot(n,normalize(vec3f(-1,2,1)))));
        col=mix(transmission,reflection,0.18+0.82*fresnel);
        col+=vec3f(1.2)*pow(max(0.0,dot(reflect(rd,n),normalize(vec3f(-0.6,1,0.4)))),140.0);
    }
    col=col/(vec3f(1)+col);
    return vec4f(pow(col,vec3f(1.0/2.2)),1);
}`;

// Refine each original volume element into eight smaller, mass-conserving elements.
export function refinedLiquid() {
    const source = initialLiquid(), result = new Float32Array(COUNT * 8 * 8);
    let i = 0;
    for (let p = 0; p < COUNT; p++) {
        for (const z of [-1, 1]) for (const y of [-1, 1]) for (const x of [-1, 1]) {
            result.set(source.subarray(p * 8, p * 8 + 8), i * 8);
            result[i * 8] += x * 0.035;
            result[i * 8 + 1] += y * 0.035;
            result[i * 8 + 2] += z * 0.035;
            i++;
        }
    }
    return result;
}

export function liquidQuality(settings = {}) {
    if (settings.quality !== undefined && !['preview', 'high'].includes(settings.quality)) {
        throw new Error('liquid-splash: quality must be preview or high');
    }
    const high = settings.quality === 'high';
    return { high, count: COUNT * (high ? 8 : 1), grid: high ? 144 : GRID, dt: DT / (high ? 2 : 1) };
}

// Non-temporal effects keep replay independent of previously rendered frames.
export function liquidPostEffects() {
    return [
        new Bloom({ threshold: 1.1, knee: 0.35, intensity: 0.12, mips: 4 }),
        new Tonemap({ curve: 'aces', exposure: 1.0 }),
        new FXAA({ subpixel: 0.4 }),
    ];
}

export default async function liquidSplash({ engine, width, height, film, settings = {}, log = () => {} }) {
    const { device } = engine;
    const { high, count, grid, dt } = liquidQuality(settings);
    const simulationShader = code => code.replace(`const COUNT: u32 = ${COUNT}u;`, `const COUNT: u32 = ${count}u;`)
        .replace('const H: f32 = 0.28;', `const H: f32 = ${high ? 0.14 : 0.28};`)
        .replace('const MASS: f32 = 2.744;', `const MASS: f32 = ${high ? 0.343 : 2.744};`)
        .replace(`const DT: f32 = ${DT};`, `const DT: f32 = ${dt};`);
    const densityCode = simulationShader(DENSITY), integrateCode = simulationShader(INTEGRATE);
    const volumeCode = VOLUME.replaceAll(String(GRID - 1), String(grid - 1)).replaceAll(String(GRID), String(grid))
        .replace(`${COUNT}u`, `${count}u`).replaceAll('0.235', high ? '0.15' : '0.235');
    let surfaceCode = SURFACE.replaceAll(String(GRID - 1), String(grid - 1)).replaceAll(String(GRID), String(grid));
    if (high) {
        surfaceCode = surfaceCode.replaceAll('0.0125', '0.004').replace('i<440', 'i<1500')
            .replace('let e=0.018', 'let e=0.012');
        surfaceCode = surfaceCode.replace('col=mix(transmission,reflection,0.18+0.82*fresnel);', `
        // Trace the refracted ray through the reconstructed liquid for optical thickness.
        let inside = refract(rd,n,1.0/1.333);
        var distance = 0.016;
        for (var k=0; k<360; k++) {
            if (density(p+inside*distance)<0.62) { break; }
            distance += 0.008;
        }
        let attenuation = exp(-vec3f(1.8,0.32,0.20)*distance);
        let behind = studio(refract(inside,-n,1.333))*0.7+vec3f(0.10,0.16,0.18);
        let transmitted = behind*attenuation+vec3f(0.015,0.28,0.32)*(vec3f(1)-attenuation);
        col=mix(transmitted,reflection,fresnel);`);
    }
    const usePost = settings.postProcess ?? high;
    if (typeof usePost !== 'boolean') throw new Error('liquid-splash: postProcess must be a boolean');
    let post;
    if (usePost) {
        surfaceCode = surfaceCode.replace('    col=col/(vec3f(1)+col);\n    return vec4f(pow(col,vec3f(1.0/2.2)),1);',
            '    return vec4f(col,1);');
        post = new PostChain(engine);
        post.resize(width,height);
        post.setChain(liquidPostEffects());
    }
    const makeBuffer = (size, label) => device.createBuffer({ label, size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const state = makeBuffer(count*32,'liquid canonical state');
    const scratch = makeBuffer(count*32,'liquid density scratch');
    device.queue.writeBuffer(state,0,high ? refinedLiquid() : initialLiquid());
    const pass=engine.createPass({ name:'Liquid / implicit free surface', isMainScreenPass:!post,
        ...(post ? post.sceneAttachments() : {}) });
    const batch=engine.createBatch(pass,{ shaderCode:surfaceCode, strideFloats:1,
        maxInstances:grid**3, depthWriteEnabled:false,
        ...(post ? { targetFormat:post.format } : {}) });
    engine.updateBatchData(batch,new Float32Array(1),1);
    const stage=(name,shaderCode,buffers,dispatchCount) => {
        const s=engine.createComputeStage({ name,shaderCode,dispatchCount });
        engine.setComputePassBindGroup(s,buffers.map((buffer,binding)=>({binding,resource:{buffer}})));
        return s;
    };
    const groups=[Math.ceil(count/64),1,1];
    const density=stage('liquid / density',densityCode,[state,scratch],groups);
    const integrate=stage('liquid / pressure viscosity gravity',integrateCode,[scratch,state],groups);
    const volumes = [];
    // Each high-detail slice is submitted and awaited separately to bound GPU work.
    for (let z = 0; z < grid; z += high ? 4 : grid) {
        const code = high ? volumeCode.replace('id: vec3u)', 'raw: vec3u)')
            .replace('    if (any(id', `    let id = raw + vec3u(0,0,${z});\n    if (any(id`) : volumeCode;
        volumes.push(stage('liquid / derived surface volume',code,[state,batch.storageBuffer],
            [Math.ceil(grid/4),Math.ceil(grid/4),high ? 1 : Math.ceil(grid/4)]));
    }
    const camera=new Camera(39,width/height,0.05,100);
    const defaultView={target:[0,high ? 0.25 : 0.65,0],distance:high ? 6.8 : 7.6,azimuth:0.62,polar:1.10};
    log(`Liquid: ${count} SPH particles; hidden ${grid}³ reconstruction; approximate visual simulation.`);
    return {
        dt, settings:{quality:high ? 'high' : 'preview',count,grid,postProcess:usePost},
        tracks:[film.track('liquid',state,{stride:32,every:1})],
        checkpoint:[{name:'liquid',buffer:state}],
        info:{model:'3D weakly-compressible SPH with implicit density surface',
            limitations:'Uncalibrated cohesion; closed cylindrical side wall; approximate single-interface refraction in high quality; no validated incompressibility.',
            reference:'Mueller, Charypar, Gross, Particle-Based Fluid Simulation for Interactive Applications, SCA 2003',
            wgsl:Object.fromEntries(Object.entries({DENSITY:densityCode,INTEGRATE:integrateCode,VOLUME:volumeCode,SURFACE:surfaceCode}).map(([k,v])=>[k,hash('sha256',v)]))},
        async step() { engine.render([density,integrate]); await device.queue.onSubmittedWorkDone(); },
        view(state,seconds) {
            const v=state??defaultView, s=Math.sin(v.polar);
            camera.updateView([v.target[0]+v.distance*s*Math.sin(v.azimuth),
                v.target[1]+v.distance*Math.cos(v.polar),v.target[2]+v.distance*s*Math.cos(v.azimuth)],v.target);
            camera.updateTime(seconds); engine.updateCamera(camera);
        },
        async prepareSurface() {
            if (high) for (const volume of volumes) { engine.render([volume]); await device.queue.onSubmittedWorkDone(); }
        },
        draw() {
            post?.upload();
            return [...(high ? [] : volumes),pass,...(post ? post.getPassSequence() : [])];
        },
        destroy() { state.destroy(); scratch.destroy(); post?.destroy(); engine.clearPasses(); },
    };
}