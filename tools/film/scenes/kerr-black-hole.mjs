// A relativistic Kerr black hole with gravitational lensing, Doppler-beamed spiral accretion,
// photon ring, tidal disruption stream, and helical Blandford-Znajek polar jets.
// 1,048,576 persistent GPU particles evolve deterministically; drawing never mutates state.
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
    let i = id.x;
    if (i * 2u >= arrayLength(&particles)) { return; }
    var a = particles[i * 2u];
    let b = particles[i * 2u + 1u];
    let dt = 1.0 / 60.0;

    // Population 1 & 2 (b.w < 0.80): Relativistic Keplerian accretion disk with MRI spiral waves.
    if (b.w < 0.80) {
        let r0 = b.x;
        // Kerr prograde angular velocity: rapid inner-ISCO differential rotation.
        let omega = 3.15 / (pow(r0, 1.38) + 0.22);
        let phase = a.w + omega * dt;
        let angle = phase + b.z * 6.2831853;

        // Multi-scale logarithmic spiral density waves (m=2 grand design + m=6 MRI filaments).
        let logR = log(max(r0, 1.55));
        let spiral2 = sin(angle * 2.0 - logR * 5.8 + phase * 0.30);
        let spiral6 = cos(angle * 6.0 - logR * 11.5 - phase * 0.25);
        let rMod = r0 * (1.0 + 0.055 * spiral2 + 0.028 * spiral6);

        // Inner plunging region near ISCO (r < 1.95) tightens toward the photon ring.
        let plunge = smoothstep(2.05, 1.56, r0);
        let r = mix(rMod, 1.56 + (r0 - 1.56) * 0.72, plunge);

        // Razor-thin relativistic equatorial disk so the front crossing leaves the shadow crisp.
        let flare = 0.022 * pow(max(r - 1.4, 0.08), 1.08);
        let warp = 0.045 * sin(angle - logR * 2.8) * smoothstep(2.4, 7.0, r);
        let y = b.y * flare + warp;

        let p = vec3f(r * cos(angle), y, r * sin(angle));
        particles[i * 2u] = vec4f(p, phase);
        return;
    }

    // Population 3 (0.80 <= b.w < 0.92): Braided Blandford-Znajek relativistic polar twin jets.
    if (b.w < 0.92) {
        let phase = a.w + (0.72 + 0.32 * abs(b.y)) * dt;
        let s = fract(phase + b.x * 0.17); // normalized distance [0, 1) along jet
        let signY = select(-1.0, 1.0, b.y >= 0.0);
        // Launch outside the photon sphere (y = 1.56 .. 11.8).
        let zJet = 1.56 + pow(s, 1.12) * 10.4;

        // 4 intertwined magnetic flux ropes with paraboloidal collimation and Alfvén waves.
        let strandId = floor(b.z * 4.0);
        let strand = strandId * 1.5707963;
        let coreOrSheath = fract(b.z * 4.0);
        let twist = zJet * 1.45 - phase * 9.0 + strand;
        let nozzle = 0.08 + 0.52 * pow(s, 0.58);
        let recollim = 1.0 + 0.28 * cos(zJet * 2.2 - phase * 12.0);
        let ropeOffset = nozzle * recollim * (0.35 + 0.65 * step(0.35, coreOrSheath));
        let ropeSpread = (fract(coreOrSheath * 7.3) - 0.5) * 0.11 * (0.25 + s);
        let finalR = max(0.02, ropeOffset + ropeSpread);
        let p = vec3f(finalR * cos(twist), signY * zJet, finalR * sin(twist));
        particles[i * 2u] = vec4f(p, phase);
        return;
    }

    // Population 4 (b.w >= 0.92): Tidal Disruption Event (TDE) infalling stellar stream.
    let phase = a.w + 0.42 * dt;
    let u = fract(phase * 0.40 + b.z);
    let rTde = mix(9.6, 1.72, pow(u, 0.74));
    let theta = u * 10.8 + b.x * 0.38 + phase * 0.65;
    let incl = (1.0 - smoothstep(0.12, 0.88, u)) * 0.58;
    let ribbonWidth = (0.06 + 0.18 * (1.0 - u)) * b.y;
    let p = vec3f(
        (rTde + ribbonWidth) * cos(theta),
        rTde * sin(theta) * sin(incl) + b.y * 0.05,
        (rTde + ribbonWidth) * sin(theta) * cos(incl)
    );
    particles[i * 2u] = vec4f(p, phase);
}`;

const PARTICLES = CAMERA + `
struct V {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec3f,
    @location(2) gain: f32,
};

@vertex fn vs_main(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> V {
    let a = particles[i * 2u];
    let b = particles[i * 2u + 1u];
    let corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    let uv = corners[v];

    let p = a.xyz;
    let r3d = max(length(p), 1e-3);

    // Orthonormal BH-centered camera frame: fwd points from camera through BH (0,0,0).
    let camDist = max(length(camera.eye), 1e-3);
    let fwd = -camera.eye / camDist;
    let worldUp = vec3f(0.0, 1.0, 0.0);
    let up = normalize(worldUp - fwd * dot(worldUp, fwd));
    let right = cross(fwd, up);

    // Coordinates of the particle relative to the black hole in the observer's frame:
    // x = horizontal, y = vertical, z = depth BEHIND the black hole (z < 0 is in front of BH).
    let x = dot(p, right);
    let y = dot(p, up);
    let z = dot(p, fwd);

    let rShadow = 1.52;
    var xApp = x;
    var yApp = y;
    var zApp = z;
    var lensMag = 1.0;

    if (b.w < 0.58) {
        // Primary image of the accretion disk:
        // Front half (z <= 0) crosses across the lower-middle of the black hole shadow.
        // Back half (z > 0) bends into the upper Einstein arch via the exact orthonormal invariant
        // yFold = sqrt(y^2 + z^2), which guarantees x^2 + yFold^2 == x^2 + y^2 + z^2 == r3d^2 at every camera angle!
        let fold = smoothstep(-0.18, 0.35, z);
        let zBehind = max(z, 0.0);
        let yRot = sqrt(y * y + zBehind * zBehind);
        let yFold = mix(y, yRot, fold);
        let bFlat = sqrt(x * x + yFold * yFold + 1e-5);
        let bRel = max(bFlat, rShadow * 1.004 + 0.94 * max(bFlat - rShadow, 0.0));
        let bLensed = mix(bFlat, bRel, fold);
        let scale = bLensed / bFlat;
        xApp = x * scale;
        yApp = yFold * scale;
        zApp = mix(z, 0.0, fold);
        lensMag = 1.0 + 0.60 * fold * exp(-max(bLensed - rShadow, 0.0) * 1.05);
    } else if (b.w < 0.80) {
        // Secondary relativistic image (n=1 lower Einstein arch beneath the black hole shadow):
        // Light rays emitted downward from the disk wrap around the bottom hemisphere in the z=0 lens plane.
        let yUnder = -sqrt(y * y + z * z * 0.85) - 0.12 * r3d;
        let bUnder = max(sqrt(x * x + yUnder * yUnder), 1e-4);
        let dirX = x / bUnder;
        let dirY = yUnder / bUnder;
        let radialNorm = clamp((r3d - 1.54) / 5.8, 0.0, 1.0);
        let wingBlend = smoothstep(0.62, 0.98, abs(dirX));
        let rCaustic = mix(
            rShadow * 1.006 + 0.65 * pow(radialNorm, 0.72),
            r3d * 0.96,
            wingBlend * 0.50
        );
        xApp = dirX * rCaustic;
        yApp = dirY * rCaustic;
        zApp = 0.0;
        lensMag = 0.95 * (1.0 - 0.30 * radialNorm);
    } else {
        // Jets and TDE stream: deflect rays that pass behind the black hole into the Einstein lens plane.
        if (z > -0.2) {
            let fold = smoothstep(-0.2, 0.8, z);
            let bFlat = max(sqrt(x * x + y * y), 1e-4);
            let bLensed = mix(bFlat, max(bFlat, rShadow * 1.015 + 0.90 * max(bFlat - rShadow, 0.0)), fold);
            let scale = bLensed / bFlat;
            xApp = x * scale;
            yApp = y * scale;
            zApp = mix(z, 0.0, fold);
        }
    }

    // Account for perspective between the particle's zApp depth and the z=0 black hole shadow plane.
    let worldPos = right * xApp + up * yApp + fwd * zApp;
    var clip = camera.viewProj * vec4f(worldPos, 1.0);

    let bProjected = sqrt(xApp * xApp + yApp * yApp) * (camDist / max(camDist + zApp, 1.0));
    if (bProjected < rShadow * 0.995 && (zApp >= -0.25 || b.w >= 0.80 || yApp > -0.02)) {
        clip = vec4f(2.0, 2.0, 2.0, 1.0);
    }
    let bScreen = bProjected;

    // Relativistic Doppler factor D = 1 / (gamma * (1 - beta_los)) from prograde Keplerian rotation.
    let angle = atan2(p.z, p.x);
    let velDir = vec3f(-sin(angle), 0.0, cos(angle));
    let toCam = -fwd;
    let rOrb = max(length(p.xz), 1.52);
    let beta = clamp(0.66 / sqrt(rOrb), 0.15, 0.58);
    let gamma = inverseSqrt(1.0 - beta * beta);
    let betaLos = beta * dot(velDir, toCam);
    let doppler = clamp(1.0 / (gamma * (1.0 - betaLos)), 0.45, 2.35);

    // Spiral wave contrast modulation so individual filaments stand out sharply.
    let logR = log(max(rOrb, 1.55));
    let filament = 0.58 + 0.42 * sin(angle * 2.0 - logR * 5.8 + a.w * 0.30) * cos(angle * 6.0 - logR * 11.5);

    var color = vec3f(1.0);
    var gain = 0.022;
    var sizeScale = 1.0;

    if (b.w < 0.80) {
        // Blackbody temperature gradient + relativistic Doppler frequency shift.
        let radialTemp = smoothstep(7.2, 1.58, b.x);
        let coreHot = vec3f(3.6, 2.75, 1.85);
        let midWarm = vec3f(2.85, 0.92, 0.16);
        let outerRed = vec3f(1.25, 0.14, 0.022);
        var baseCol = mix(outerRed, midWarm, smoothstep(0.0, 0.56, radialTemp));
        baseCol = mix(baseCol, coreHot, smoothstep(0.56, 1.0, radialTemp));

        let blueShift = vec3f(0.65, 1.65, 3.6);
        let redShift = vec3f(1.65, 0.16, 0.018);
        let shiftT = clamp((doppler - 1.0) * 0.88, -1.0, 1.0);
        if (shiftT >= 0.0) {
            color = mix(baseCol, baseCol * 0.58 + blueShift * 1.25, shiftT * 0.74);
        } else {
            color = mix(baseCol, redShift, (-shiftT) * 0.72);
        }

        let beaming = pow(doppler, 2.08);
        gain = (0.013 + 0.028 * radialTemp) * beaming * filament * lensMag;

        // Keep the thin front-crossing equatorial band crisp and semi-transparent against the black shadow.
        if (zApp < 0.0 && bScreen < rShadow * 1.02) {
            gain *= 0.52;
        }
    } else if (b.w < 0.92) {
        // Braided Blandford-Znajek jets: synchrotron cyan-blue core, violet sheath, and shock knots.
        let s = clamp((abs(p.y) - 1.62) / 10.2, 0.0, 1.0);
        let baseFade = smoothstep(0.0, 0.08, s);
        let knot = pow(0.5 + 0.5 * cos(abs(p.y) * 2.1 - a.w * 11.0), 5.0);
        let jetCore = vec3f(1.3, 2.4, 4.2);
        let jetSheath = vec3f(0.42, 0.85, 2.9);
        let jetTip = vec3f(1.45, 0.32, 2.25);
        color = mix(jetCore, jetSheath, smoothstep(0.0, 0.45, s));
        color = mix(color, jetTip, smoothstep(0.55, 1.0, s));
        color += vec3f(2.2, 2.6, 3.8) * knot * (1.0 - 0.45 * s);
        gain = baseFade * (0.020 * (1.0 - 0.65 * s) + 0.032 * knot) * (0.8 + 0.2 * doppler);
        sizeScale = 1.12;
    } else {
        // Tidal disruption stream: incandescent gold-cyan shocked stellar debris ribbon.
        let u = fract(a.w * 0.40 + b.z);
        let streamCol = mix(vec3f(0.42, 1.35, 3.0), vec3f(3.2, 1.45, 0.42), smoothstep(0.1, 0.85, u));
        let shock = pow(0.5 + 0.5 * sin(u * 26.0 - a.w * 6.0), 5.0);
        color = streamCol + vec3f(2.0, 1.6, 1.05) * shock;
        gain = 0.018 + 0.024 * shock;
        sizeScale = 1.08;
    }

    let size = (0.0082 + 0.0098 * fract(b.z * 17.3)) * sizeScale * camera.proj[1][1];
    clip.x += uv.x * size / camera.aspect;
    clip.y += uv.y * size;

    var o: V;
    o.pos = clip;
    o.uv = uv;
    o.color = color;
    o.gain = gain;
    return o;
}

@fragment fn fs_main(v: V) -> @location(0) vec4f {
    let r = dot(v.uv, v.uv);
    if (r > 1.0) { discard; }
    return vec4f(v.color * exp(-r * 4.2) * v.gain, 1.0);
}`;

const BACKDROP = CAMERA + `
struct V { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> V {
    let p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    var o: V;
    o.pos = vec4f(p[i], 0.999, 1.0);
    o.uv = p[i];
    return o;
}

fn hash3(p: vec3f) -> f32 {
    var q = fract(p * vec3f(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 33.33);
    return fract((q.x + q.y) * q.z);
}

@fragment fn fs_main(v: V) -> @location(0) vec4f {
    let h = camera.invViewProj * vec4f(v.uv, 1.0, 1.0);
    let rd = normalize(h.xyz / h.w - camera.eye);
    let t = -dot(camera.eye, rd);
    let closest = camera.eye + max(t, 0.0) * rd;
    let impact = length(closest);
    let rShadow = 1.52;

    // Deep space background with gravitationally lensed starfield around the Einstein ring.
    var col = vec3f(0.0003, 0.0006, 0.0018);
    let bDir = closest / max(impact, 1e-4);
    let deflection = 2.35 / max(impact * impact, 2.2);
    let lensedDir = normalize(rd - bDir * deflection);

    // 3D directional star grid (warps smoothly along the Einstein ring as camera moves).
    let starGrid = lensedDir * 195.0;
    let cell = floor(starGrid);
    let f = fract(starGrid) - 0.5;
    let rnd = hash3(cell);
    if (rnd > 0.992 && impact > rShadow * 1.01) {
        let d2 = dot(f, f);
        let starBright = exp(-d2 * 28.0) * smoothstep(rShadow * 1.01, rShadow * 1.25, impact);
        let mag = 1.0 + 1.8 * exp(-(impact - rShadow * 1.08) * 1.5);
        let starTint = mix(vec3f(1.0, 0.82, 0.65), vec3f(0.62, 0.85, 1.35), fract(rnd * 37.0));
        col += starTint * starBright * 0.45 * mag;
    }

    // Subtle deep-field relativistic nebula outside the shadow.
    let outsideShadow = smoothstep(rShadow * 0.99, rShadow * 1.15, impact);
    let jetAxis = abs(lensedDir.y);
    col += vec3f(0.004, 0.010, 0.028) * exp(-impact * 0.22) * outsideShadow;
    col += vec3f(0.008, 0.020, 0.060) * pow(jetAxis, 7.0) * exp(-impact * 0.18) * outsideShadow;

    // Razor-sharp Doppler-asymmetric relativistic photon ring at impact = rShadow (1.52),
    // aligned with the prograde rotation of the accretion disk (approaching on right).
    let spinSide = -dot(bDir, normalize(cross(vec3f(0.0, 1.0, 0.0), rd)));
    let ringDoppler = 1.0 + 0.65 * spinSide;
    let n1Ring = exp(-pow((impact - rShadow) * 34.0, 2.0)) * ringDoppler;
    let n0Corona = exp(-pow((impact - rShadow * 1.04) * 10.5, 2.0)) * (0.75 + 0.35 * spinSide) * outsideShadow;
    let ringColor = mix(vec3f(1.45, 0.44, 0.08), vec3f(1.5, 1.85, 2.45), clamp(0.5 + 0.5 * spinSide, 0.0, 1.0));
    col += ringColor * n1Ring * 1.25;
    col += vec3f(0.85, 0.30, 0.06) * n0Corona * 0.42;

    // Event horizon shadow: pure relativistic abyss inside rShadow = 1.52.
    if (impact < rShadow && t > 0.0) {
        let edge = smoothstep(rShadow * 0.975, rShadow, impact);
        col = mix(vec3f(0.0), col, edge);
    }

    return vec4f(col + particles[0].xyz * 0.0, 1.0);
}`;

export default async function kerrBlackHole({ engine, width, height, settings = {}, film, log = () => {} }) {
    const count = settings.count ?? 1048576;
    if (!Number.isInteger(count) || count < 128 || count > 2097152) {
        throw new Error('kerr-black-hole: count must be 128–2097152');
    }
    const { device } = engine;
    const post = new PostChain(engine);
    post.setChain([
        new Bloom({ threshold: 1.05, knee: 0.42, intensity: 0.28, mips: 5 }),
        new Tonemap({ curve: 'aces', exposure: 1.24 }),
    ]);
    post.resize(width, height);

    const pass = engine.createPass({
        name: 'Kerr Black Hole / HDR',
        isMainScreenPass: false,
        ...post.sceneAttachments({ clearValue: { r: 0, g: 0, b: 0, a: 1 } }),
    });

    const background = engine.createBatch(pass, {
        shaderCode: BACKDROP,
        strideFloats: 8,
        maxInstances: 1,
        targetFormat: 'rgba16float',
        depthWriteEnabled: false,
    });
    engine.updateBatchData(background, new Float32Array(8), 1);

    const batch = engine.createBatch(pass, {
        shaderCode: PARTICLES,
        strideFloats: 8,
        maxInstances: count,
        targetFormat: 'rgba16float',
        depthWriteEnabled: false,
        blend: {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
        },
    });

    const data = new Float32Array(count * 8);
    let seed = 314159265;
    const rand = () => {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        return seed / 4294967296;
    };

    for (let i = 0; i < count; i++) {
        // Concentrate disk particles near the inner relativistic ISCO (r = 1.58 .. 7.4).
        const radius = 1.58 + Math.pow(rand(), 1.55) * 5.85;
        const height = (rand() + rand() - 1.0); // triangular distribution peaked at equatorial plane
        const azimuth = rand();
        const kind = rand();
        data.set([0, 0, 0, rand() * Math.PI * 2, radius, height, azimuth, kind], i * 8);
    }
    engine.updateBatchData(batch, data, count);

    const evolve = engine.createComputeStage({
        name: 'Kerr Black Hole / geodesic & jet evolution',
        shaderCode: EVOLVE,
        dispatchCount: [Math.ceil(count / 128), 1, 1],
    });
    engine.setComputePassBindGroup(evolve, [{ binding: 0, resource: { buffer: batch.storageBuffer } }]);

    // Initialize positions before step zero is saved.
    engine.render([evolve]);
    await device.queue.onSubmittedWorkDone();

    const camera = new Camera(45, width / height, 0.1, 120);
    const defaultView = { target: [0, 0, 0], distance: 17.5, azimuth: -0.35, polar: 1.32 };

    const track = film.track('spacetime-plasma', batch.storageBuffer, {
        every: 6,
        stride: 32,
        fields: [0, 1, 2, 3, 4, 5, 6, 7].map(word => ({ word, store: 'f32', interp: 'linear' })),
    });

    log(`Kerr black hole: ${count.toLocaleString()} GPU particles, ${(data.byteLength / 1048576).toFixed(0)} MiB persistent state (gravitational lensing + Doppler beaming + BZ jets).`);

    return {
        dt: 1 / 60,
        settings: { count },
        tracks: [track],
        checkpoint: [{ name: 'spacetime-plasma', buffer: batch.storageBuffer }],
        info: {
            particles: count,
            model: 'Kerr black hole gravitational lensing, relativistic Doppler-beamed accretion, TDE stream, and Blandford-Znajek helical jets',
            wgsl: {
                evolve: hash('sha256', EVOLVE),
                particles: hash('sha256', PARTICLES),
                backdrop: hash('sha256', BACKDROP),
            },
        },
        async step() {
            engine.render([evolve]);
            await device.queue.onSubmittedWorkDone();
        },
        view(state, seconds) {
            const v = state ?? defaultView;
            const s = Math.sin(v.polar);
            camera.updateView(
                [
                    v.target[0] + v.distance * s * Math.sin(v.azimuth),
                    v.target[1] + v.distance * Math.cos(v.polar),
                    v.target[2] + v.distance * s * Math.cos(v.azimuth),
                ],
                v.target,
            );
            camera.updateTime(seconds);
            engine.updateCamera(camera);
        },
        draw() {
            post.upload();
            return [pass, ...post.getPassSequence()];
        },
        destroy() {
            post.destroy();
            engine.clearPasses();
        },
    };
}
