// WGSL generators for NightSky and the one-draw-call StarField.

import { DEG_TO_RAD, KASTEN_YOUNG } from './skyMath';
import type { ShaderSkyLayout } from './types';

export const NIGHT_SKY_FLOATS = 28;
export const NIGHT_OFF = {
    MOON_DIR: 0, MOON_COS: 3, SUN_DIR: 4, MOON_SIN: 7, MOON_RGB: 8,
    ILLUMINATION: 11, EARTHSHINE: 12, DOME: 16, AIRGLOW: 19,
    POLLUTION: 20, POLLUTION_STRENGTH: 23, GMST: 24, LATITUDE: 25,
    NIGHT_GAIN: 26, MILKY_WAY: 27,
} as const;

export function buildNightSkyWGSL(base: number): ShaderSkyLayout {
    const P = (o: number) => `skyParams[${base + o}]`;
    const O = NIGHT_OFF;
    return { wgslStructs: '', wgslLogic: /* wgsl */ `
fn sky_night_moon(dir: vec3<f32>) -> vec3<f32> {
    let moonTo = vec3<f32>(-${P(O.MOON_DIR)}, -${P(O.MOON_DIR + 1)}, -${P(O.MOON_DIR + 2)});
    let cosAng = clamp(dot(dir, moonTo), -1.0, 1.0);
    let moonCos = ${P(O.MOON_COS)};
    let discEdge = 0.00015;
    
    var moonRgb = vec3<f32>(0.0);
    if (cosAng > moonCos - discEdge) {
        let sinAng = sqrt(max(0.0, 1.0 - cosAng * cosAng));
        let tangent = (dir - moonTo * cosAng) / max(sinAng, 1.0e-6);
        let moonSin = max(${P(O.MOON_SIN)}, 1.0e-5);
        let r = clamp(sinAng / moonSin, 0.0, 1.0);
        let z = sqrt(max(0.0, 1.0 - r * r));
        let n = normalize(-moonTo * z + tangent * r);
        
        let p1 = sin(n.x * 5.2 + n.y * 3.8) * cos(n.z * 4.6 - n.y * 2.4);
        let p2 = sin(n.x * 11.0 - n.z * 8.5) * cos(n.y * 9.5 + n.x * 7.0) * 0.5;
        let mariaMask = smoothstep(-0.2, 0.55, p1 + p2);
        let lunarAlbedo = mix(vec3<f32>(1.0, 0.98, 0.94), vec3<f32>(0.70, 0.68, 0.64), mariaMask * 0.35);
        
        let baseRadiance = vec3<f32>(${P(O.MOON_RGB)}, ${P(O.MOON_RGB + 1)}, ${P(O.MOON_RGB + 2)});
        let radiance = max(baseRadiance, vec3<f32>(14.0, 13.8, 13.2));
        let surf = lunarAlbedo * radiance * (0.88 + 0.12 * z);
        let discMask = smoothstep(moonCos - discEdge, moonCos + discEdge, cosAng);
        moonRgb = surf * discMask;
    }
    
    let forwardGlow = max(cosAng, 0.0);
    let innerCorona = pow(forwardGlow, 1024.0) * 0.045;
    let outerCorona = pow(forwardGlow, 256.0) * 0.012;
    let wideHalo = pow(forwardGlow, 64.0) * 0.002;
    let corona = (innerCorona + outerCorona + wideHalo) * vec3<f32>(0.88, 0.94, 1.0);
    
    return moonRgb + corona;
}

fn sky_night_milky(dir: vec3<f32>) -> f32 {
    let c = cos(${P(O.GMST)}); let s = sin(${P(O.GMST)});
    let q = vec3<f32>(c * dir.x + s * dir.z, dir.y, -s * dir.x + c * dir.z);
    let galacticNormal = normalize(vec3<f32>(0.48, 0.75, 0.45));
    let band = exp(-pow(abs(dot(q, galacticNormal)) / 0.12, 1.4));
    let knots = 0.72 + 0.28 * sin(q.x * 31.0 + q.z * 17.0) * sin(q.y * 23.0 - q.z * 11.0);
    return max(band * knots, 0.0) * ${P(O.MILKY_WAY)};
}

fn sampleSky(dir: vec3<f32>) -> vec3<f32> {
    let up = max(dir.y, 0.0);
    let dome = vec3<f32>(${P(O.DOME)}, ${P(O.DOME + 1)}, ${P(O.DOME + 2)});
    let pollution = vec3<f32>(${P(O.POLLUTION)}, ${P(O.POLLUTION + 1)}, ${P(O.POLLUTION + 2)})
        * ${P(O.POLLUTION_STRENGTH)} * exp(-max(dir.y, 0.0) * 8.0);
    let airglow = vec3<f32>(0.22, 0.55, 0.34) * ${P(O.AIRGLOW)} * (0.35 + 0.65 * (1.0 - up));
    let milky = vec3<f32>(0.62, 0.72, 1.0) * sky_night_milky(dir);
    let ground = dome * 0.06;
    return max(mix(ground, dome + pollution + airglow + milky, step(0.0, dir.y))
        * ${P(O.NIGHT_GAIN)} + sky_night_moon(dir), vec3<f32>(0.0));
}

fn skySunRadiance(elevDeg: f32) -> vec3<f32> { return vec3<f32>(0.0); }
` };
}

export function buildStarWGSL(): string {
    const [ka, kb, kc] = KASTEN_YOUNG;
    return /* wgsl */ `
struct Camera {
    viewProj: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>, invViewProj: mat4x4<f32>,
    invView: mat4x4<f32>, invProj: mat4x4<f32>, eyePos: vec3<f32>, simTime: f32,
    zNear: f32, zFar: f32, fovRads: f32, aspect: f32, _padEnd: vec4<f32>,
};
struct Star { ra: f32, dec: f32, magnitude: f32, bv: f32 };
struct StarParams { gmst: f32, latitude: f32, longitude: f32, twinkle: f32,
    width: f32, height: f32, radiusPx: f32, intensity: f32 };
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> stars: array<Star>;
@group(1) @binding(0) var<uniform> starParams: StarParams;

struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) local: vec2<f32>,
    @location(1) color: vec3<f32>, @location(2) elevDeg: f32, @location(3) phase: f32 };

fn sky_star_color(bvIn: f32) -> vec3<f32> {
    let bv = clamp(bvIn, -0.4, 2.0);
    let t = 46.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62));
    var r = select(329.6987 * pow(max(t - 60.0, 1.0), -0.1332048), 255.0, t <= 66.0);
    var g = select(288.1222 * pow(max(t - 60.0, 1.0), -0.07551485), 99.4708 * log(t) - 161.1196, t <= 66.0);
    var b = select(select(138.5177 * log(max(t - 10.0, 1.0)) - 305.0448, 0.0, t <= 19.0), 255.0, t >= 66.0);
    let m = max(max(r, g), max(b, 1.0));
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(255.0)) / m;
}

@vertex fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
    let st = stars[ii];
    let h = starParams.gmst + starParams.longitude - st.ra;
    let cd = cos(st.dec); let sd = sin(st.dec); let cl = cos(starParams.latitude); let sl = sin(starParams.latitude);
    let world = vec3<f32>(-cd * sin(h), sl * sd + cl * cd * cos(h), cl * sd - sl * cd * cos(h));
    let viewDir = camera.view * vec4<f32>(world, 0.0);
    var centre = camera.proj * vec4<f32>(viewDir.xyz, 0.0);
    var corners = array<vec2<f32>, 3>(vec2<f32>(-1.732, -1.0), vec2<f32>(1.732, -1.0), vec2<f32>(0.0, 2.0));
    let local = corners[vi];
    let wPx = max(starParams.width, 320.0);
    let hPx = max(starParams.height, 240.0);
    let radius = max(starParams.radiusPx, 1.2);
    let ndcPx = vec2<f32>(2.0 / wPx, 2.0 / hPx) * radius;
    centre = vec4<f32>(centre.xy + local * ndcPx * centre.w, centre.z, centre.w);
    if (starParams.intensity <= 0.0001 || centre.w <= 0.0) {
        centre = vec4<f32>(2.0, 2.0, 1.0, 1.0);
    }
    var out: VSOut; out.pos = vec4<f32>(centre.xy, centre.w * 0.999998, centre.w); out.local = local;
    
    // Non-linear perceptual magnitude curve ensuring faint stars are clearly visible and bright stars stand out brilliantly
    let mag = clamp(st.magnitude, -1.5, 6.5);
    let magWeight = pow(max(0.0, (7.0 - mag) / 8.5), 1.6);
    let flux = (magWeight * 32.0 + 4.5) * starParams.intensity;
    
    // Smooth atmospheric horizon extinction curve: stars near the horizon fade seamlessly into atmospheric haze
    let horizonExtinction = smoothstep(-0.12, 0.03, world.y);
    out.color = sky_star_color(st.bv) * (flux * horizonExtinction);

    out.elevDeg = asin(clamp(world.y, -1.0, 1.0)) / ${DEG_TO_RAD};
    out.phase = fract(sin(st.ra * 12.9898 + st.dec * 78.233) * 43758.5453) * 6.28318530718;
    return out;
}

@fragment fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.local, in.local); if (r2 >= 1.0) { discard; }
    
    // Atmospheric scintillation: soothing natural celestial twinkle (gentle, organic middle ground)
    let h = max(in.elevDeg, 0.0);
    let elevationScale = 1.0 + 0.35 * exp(-h * 0.05);
    let t = camera.simTime;
    
    // Harmonious dual-frequency shimmer (smooth 3.8 Hz & 1.9 Hz)
    let wave1 = sin(t * 3.8 + in.phase * 3.14);
    let wave2 = sin(t * 1.9 - in.phase * 1.57);
    let shimmer = wave1 * 0.7 + wave2 * 0.3;
    
    // Balanced scintillation amplitude (soft, soothing twinkle without aggressive strobing)
    let amp = clamp(starParams.twinkle, 0.0, 1.0) * elevationScale * 0.38;
    let flicker = max(0.25, 1.0 + amp * shimmer);
    let profile = pow(max(1.0 - r2, 0.0), 1.5);
    
    return vec4<f32>(in.color * (flicker * profile), profile);
}
`;
}