// D2.1 / D2.2: Crystalline Bottle Macro Scene
// Features a high-resolution Point-Normal (PN-Triangles) subdivided glass bottle from the public assets,
// with spectral chromatic dispersion (3-wavelength Snell refraction), cut-crystal diamond fluting,
// analytic multi-light studio HDR reflections (Key softbox, dual Rim kickers, Top diffuser),
// Beer-Lambert physical liquid absorption (amber cognac core with meniscus), seamless cyclorama studio cove,
// and floating luminous crystal motes, processed through NullGraph's cinematic PostChain.

import { hash } from 'node:crypto';
import { readFileSync, existsSync } from 'node:fs';
import { Camera } from '../../../dist/index.js';
import {
    PostChain,
    Bloom,
    Halation,
    Tonemap,
    FXAA,
    Vignette,
    FilmGrain,
    ChromaticAberration
} from '../../../dist/post.js';
import { GLBParser } from '../../../dist/loaders.js';
import { SITE } from '../check.mjs';

// --- MATH HELPERS FOR PN-TRIANGLE SUBDIVISION ---
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function add3(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function sub3(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function scale3(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
function norm3(a) {
    const len = Math.hypot(a[0], a[1], a[2]) || 1;
    return [a[0] / len, a[1] / len, a[2] / len];
}

/**
 * High-precision Point-Normal (PN-Triangles) curved surface subdivision.
 * Converts low-poly polygonal geometry into a smooth, curved cubic Bézier surface with continuous normals.
 */
function subdividePNTriangles(vertices, indices, levels = 2) {
    let curVerts = vertices;
    let curIndices = indices;

    for (let lvl = 0; lvl < levels; lvl++) {
        const newVerts = [];
        const newIndices = [];

        for (let i = 0; i < curIndices.length; i += 3) {
            const i1 = curIndices[i], i2 = curIndices[i + 1], i3 = curIndices[i + 2];

            const p1 = [curVerts[i1 * 8], curVerts[i1 * 8 + 1], curVerts[i1 * 8 + 2]];
            const n1 = [curVerts[i1 * 8 + 3], curVerts[i1 * 8 + 4], curVerts[i1 * 8 + 5]];
            const uv1 = [curVerts[i1 * 8 + 6], curVerts[i1 * 8 + 7]];

            const p2 = [curVerts[i2 * 8], curVerts[i2 * 8 + 1], curVerts[i2 * 8 + 2]];
            const n2 = [curVerts[i2 * 8 + 3], curVerts[i2 * 8 + 4], curVerts[i2 * 8 + 5]];
            const uv2 = [curVerts[i2 * 8 + 6], curVerts[i2 * 8 + 7]];

            const p3 = [curVerts[i3 * 8], curVerts[i3 * 8 + 1], curVerts[i3 * 8 + 2]];
            const n3 = [curVerts[i3 * 8 + 3], curVerts[i3 * 8 + 4], curVerts[i3 * 8 + 5]];
            const uv3 = [curVerts[i3 * 8 + 6], curVerts[i3 * 8 + 7]];

            const w12 = dot3(sub3(p2, p1), n1);
            const w21 = dot3(sub3(p1, p2), n2);
            const w23 = dot3(sub3(p3, p2), n2);
            const w32 = dot3(sub3(p2, p3), n3);
            const w31 = dot3(sub3(p1, p3), n3);
            const w13 = dot3(sub3(p3, p1), n1);

            const b300 = p1, b030 = p2, b003 = p3;
            const b210 = scale3(add3(add3(scale3(p1, 2), p2), scale3(n1, -w12)), 1 / 3);
            const b120 = scale3(add3(add3(scale3(p2, 2), p1), scale3(n2, -w21)), 1 / 3);
            const b021 = scale3(add3(add3(scale3(p2, 2), p3), scale3(n2, -w23)), 1 / 3);
            const b012 = scale3(add3(add3(scale3(p3, 2), p2), scale3(n3, -w32)), 1 / 3);
            const b102 = scale3(add3(add3(scale3(p3, 2), p1), scale3(n3, -w31)), 1 / 3);
            const b201 = scale3(add3(add3(scale3(p1, 2), p3), scale3(n1, -w13)), 1 / 3);

            const E = scale3(add3(add3(add3(b210, b120), add3(b021, b012)), add3(b102, b201)), 1 / 6);
            const V = scale3(add3(add3(p1, p2), p3), 1 / 3);
            const b111 = add3(E, scale3(sub3(E, V), 0.5));

            function evalPN(u, v, w) {
                const u2 = u * u, v2 = v * v, w2 = w * w;
                const u3 = u2 * u, v3 = v2 * v, w3 = w2 * w;
                const pos = add3(
                    add3(add3(scale3(b300, u3), scale3(b030, v3)), scale3(b003, w3)),
                    add3(
                        add3(add3(scale3(b210, 3 * u2 * v), scale3(b120, 3 * u * v2)), add3(scale3(b201, 3 * u2 * w), scale3(b102, 3 * u * w2))),
                        add3(add3(scale3(b021, 3 * v2 * w), scale3(b012, 3 * v * w2)), scale3(b111, 6 * u * v * w))
                    )
                );
                const normVec = norm3(add3(add3(scale3(n1, u), scale3(n2, v)), scale3(n3, w)));
                const tex = [uv1[0] * u + uv2[0] * v + uv3[0] * w, uv1[1] * u + uv2[1] * v + uv3[1] * w];
                return [...pos, ...normVec, ...tex];
            }

            const pts = [
                evalPN(1, 0, 0),
                evalPN(0, 1, 0),
                evalPN(0, 0, 1),
                evalPN(0.5, 0.5, 0),
                evalPN(0, 0.5, 0.5),
                evalPN(0.5, 0, 0.5)
            ];
            const baseIdx = newVerts.length / 8;
            for (const pt of pts) newVerts.push(...pt);

            newIndices.push(
                baseIdx + 0, baseIdx + 3, baseIdx + 5,
                baseIdx + 3, baseIdx + 1, baseIdx + 4,
                baseIdx + 5, baseIdx + 4, baseIdx + 2,
                baseIdx + 3, baseIdx + 4, baseIdx + 5
            );
        }
        curVerts = new Float32Array(newVerts);
        curIndices = new Uint32Array(newIndices);
    }
    return { vertices: curVerts, indices: curIndices };
}

// --- SHADERS ---

const CAMERA_WGSL = `struct Camera {
    viewProj: mat4x4f, view: mat4x4f, proj: mat4x4f, invViewProj: mat4x4f,
    invView: mat4x4f, invProj: mat4x4f, eye: vec3f, time: f32,
    near: f32, far: f32, fov: f32, aspect: f32, pad: vec4f,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> instances: array<vec4f>;
`;

const BACKDROP_WGSL = CAMERA_WGSL + `
struct V {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex fn vs_main(@builtin(vertex_index) i: u32) -> V {
    let p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    var o: V;
    o.pos = vec4f(p[i], 0.9999, 1.0);
    o.uv = p[i];
    return o;
}

struct FragOut {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
};

@fragment fn fs_main(v: V) -> FragOut {
    let h = camera.invViewProj * vec4f(v.uv, 1.0, 1.0);
    let rd = normalize(h.xyz / h.w - camera.eye);
    let ro = camera.eye;
    
    // Intersection with obsidian pedestal floor at y = 0
    let hasFloorHit = (ro.y > 0.001) && (rd.y < -0.0001);
    let tFloor = -ro.y / rd.y;
    
    var outColor = vec3f(0.0);
    var outDepth = 1.0;
    
    // Studio cove background gradient with soft spotlight behind the bottle
    let ndc = v.uv;
    let distCenter = length(ndc - vec2f(0.0, 0.12));
    let coveDark = vec3f(0.010, 0.013, 0.020);
    let coveGlow = vec3f(0.14, 0.16, 0.22) * exp(-distCenter * 1.6);
    let coveAccent = vec3f(0.03, 0.06, 0.12) * exp(-pow(ndc.x - 0.65, 2.0) * 2.0);
    let backdropCol = coveDark + coveGlow + coveAccent;
    
    if (hasFloorHit && tFloor > 0.0) {
        let pFloor = ro + tFloor * rd;
        let clip = camera.viewProj * vec4f(pFloor, 1.0);
        outDepth = clamp(clip.z / clip.w, 0.0, 1.0);
        
        let dist = length(pFloor.xz);
        
        let floorNorm = vec3f(0.0, 1.0, 0.0);
        let reflDir = reflect(rd, floorNorm);
        
        // Studio softbox reflections on the obsidian pedestal
        let keyDir = normalize(vec3f(-0.7, 0.5, 0.65));
        let rimRDir = normalize(vec3f(0.85, 0.35, -0.6));
        let rimLDir = normalize(vec3f(-0.85, 0.35, -0.6));
        
        let specKey = pow(max(dot(reflDir, keyDir), 0.0), 32.0) * vec3f(1.2, 1.15, 1.05);
        let specRimR = pow(max(dot(reflDir, rimRDir), 0.0), 28.0) * vec3f(0.7, 1.0, 1.6);
        let specRimL = pow(max(dot(reflDir, rimLDir), 0.0), 28.0) * vec3f(0.5, 0.8, 1.4);
        
        // Soft contact shadow under the bottle (base radius ~ 0.182)
        let contactShadow = smoothstep(0.16, 0.36, dist);
        
        // Amber caustic light spread on the obsidian floor
        let causticRing = exp(-pow((dist - 0.26) * 5.5, 2.0)) * vec3f(0.9, 0.55, 0.15);
        let causticFringe = exp(-pow((dist - 0.38) * 8.0, 2.0)) * vec3f(0.3, 0.35, 0.65);
        let causticSpill = (causticRing + causticFringe) * 0.95;
        
        // Polished obsidian floor with smooth distance fade into the studio cove
        let obsidianBase = vec3f(0.010, 0.012, 0.018);
        let ndotv = max(-rd.y, 0.0);
        let fObsidian = 0.04 + 0.96 * pow(1.0 - ndotv, 5.0);
        
        let floorCol = (obsidianBase * (0.15 + 0.85 * contactShadow) + causticSpill + (specKey + specRimR + specRimL) * fObsidian);
        
        // Seamlessly blend floor into studio background at distance
        let horizonBlend = smoothstep(2.5, 6.0, tFloor);
        outColor = mix(floorCol, backdropCol, horizonBlend);
    } else {
        outColor = backdropCol;
        outDepth = 1.0;
    }
    
    var o: FragOut;
    o.color = vec4f(outColor + instances[0].xyz * 0.0, 1.0);
    o.depth = outDepth;
    return o;
}
`;

const BOTTLE_WGSL = CAMERA_WGSL + `
struct VertexIn {
    @location(0) pos: vec3f,
    @location(1) norm: vec3f,
    @location(2) uv: vec2f,
};

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) worldPos: vec3f,
    @location(1) worldNorm: vec3f,
    @location(2) uv: vec2f,
};

@vertex fn vs_main(in: VertexIn, @builtin(instance_index) inst: u32) -> VOut {
    let dummy = instances[inst];
    var o: VOut;
    o.pos = camera.viewProj * vec4f(in.pos + dummy.xyz * 0.0, 1.0);
    o.worldPos = in.pos + dummy.xyz * 0.0;
    o.worldNorm = normalize(in.norm);
    o.uv = in.uv;
    return o;
}

// Analytic studio environment reflections
fn studioEnvironment(dir: vec3f) -> vec3f {
    let keyDir = normalize(vec3f(-0.7, 0.45, 0.65));
    let rimRDir = normalize(vec3f(0.85, 0.35, -0.6));
    let rimLDir = normalize(vec3f(-0.85, 0.35, -0.6));
    let topDir = vec3f(0.0, 1.0, 0.0);
    
    let dotKey = max(dot(dir, keyDir), 0.0);
    let keySoftbox = pow(dotKey, 32.0) * vec3f(4.2, 4.0, 3.7) + pow(dotKey, 8.0) * 0.45 * vec3f(1.0, 0.98, 0.92);
    
    let dotRimR = max(dot(dir, rimRDir), 0.0);
    let rimRStrip = pow(dotRimR, 40.0) * vec3f(3.5, 4.8, 6.5);
    
    let dotRimL = max(dot(dir, rimLDir), 0.0);
    let rimLStrip = pow(dotRimL, 40.0) * vec3f(2.6, 3.6, 5.0);
    
    let dotTop = max(dot(dir, topDir), 0.0);
    let topDiffuser = pow(dotTop, 10.0) * vec3f(1.2, 1.25, 1.4);
    
    let ambient = vec3f(0.04, 0.045, 0.06);
    
    return keySoftbox + rimRStrip + rimLStrip + topDiffuser + ambient;
}

// Analytic backdrop sampler for refracted rays
fn sampleBackground(rd: vec3f) -> vec3f {
    let distCenter = length(rd.xy - vec2f(0.0, 0.12));
    let dark = vec3f(0.015, 0.018, 0.026);
    let glow = vec3f(0.35, 0.40, 0.55) * exp(-distCenter * 1.8);
    return dark + glow;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4f {
    let P = in.worldPos;
    let V = normalize(camera.eye - P);
    let h = P.y;
    let radius = length(P.xz);
    let angle = atan2(P.z, P.x);
    
    let tang = normalize(vec3f(-sin(angle), 0.0, cos(angle)));
    let bitang = normalize(cross(in.worldNorm, tang));
    
    // --- 1. Cut-Crystal Faceting & Fluting ---
    // Vertical prism flutes on main body (h from 0.08 to 0.62)
    let fluteCount = 20.0;
    let flutePhase = angle * fluteCount;
    let fluteShape = sin(flutePhase);
    let fluteFacet = sign(fluteShape) * pow(abs(fluteShape), 0.55);
    let fluteMask = smoothstep(0.07, 0.14, h) * smoothstep(0.63, 0.57, h);
    let flutePerturb = fluteFacet * 0.22 * fluteMask;
    
    // Diamond bevel facets around the shoulder (h from 0.57 to 0.74)
    let dAngle = angle * 16.0;
    let dHeight = (h - 0.57) * 55.0;
    let diamond = sin(dAngle + dHeight) * sin(dAngle - dHeight);
    let diamondFacet = sign(diamond) * pow(abs(diamond), 0.5);
    let shoulderMask = smoothstep(0.55, 0.61, h) * smoothstep(0.74, 0.69, h);
    let diamondPerturb = diamondFacet * 0.24 * shoulderMask;
    
    var N = normalize(in.worldNorm + tang * flutePerturb + bitang * diamondPerturb);
    
    // --- 2. Metal Collar (h between 0.742 and 0.812) ---
    let isGoldCollar = (h >= 0.742 && h < 0.812);
    if (isGoldCollar) {
        let lKeyDir = normalize(vec3f(-0.7, 0.45, 0.65));
        let lRimDir = normalize(vec3f(0.85, 0.35, -0.6));
        let ndotlKey = max(dot(in.worldNorm, lKeyDir), 0.0);
        
        let hKey = normalize(lKeyDir + V);
        let specKey = pow(max(dot(in.worldNorm, hKey), 0.0), 32.0);
        let hRim = normalize(lRimDir + V);
        let specRim = pow(max(dot(in.worldNorm, hRim), 0.0), 24.0);
        
        let goldAlbedo = vec3f(1.0, 0.78, 0.34);
        let refl = studioEnvironment(reflect(-V, in.worldNorm));
        
        let goldCol = goldAlbedo * (ndotlKey * 0.4 + 0.2) + goldAlbedo * refl * 0.85 + (specKey * vec3f(3.2, 2.8, 2.0) + specRim * vec3f(1.4, 2.2, 3.2)) * 0.7;
        return vec4f(goldCol, 1.0);
    }
    
    // --- 3. Crystal Fresnel & Fake Reflection ---
    let ndotv = clamp(dot(N, V), 0.0, 1.0);
    let F0 = 0.055;
    let F = F0 + (1.0 - F0) * pow(1.0 - ndotv, 5.0);
    
    let R = reflect(-V, N);
    let studioRefl = studioEnvironment(R);
    
    // Full-strength studio softbox specular highlights
    let lKeyDir = normalize(vec3f(-0.7, 0.45, 0.65));
    let lRimRDir = normalize(vec3f(0.85, 0.35, -0.6));
    let lRimLDir = normalize(vec3f(-0.85, 0.35, -0.6));
    
    let hKey = normalize(lKeyDir + V);
    let hRimR = normalize(lRimRDir + V);
    let hRimL = normalize(lRimLDir + V);
    
    // Crisp studio specular strips
    let specKey = (pow(max(dot(N, hKey), 0.0), 120.0) * 5.5 + pow(max(dot(N, hKey), 0.0), 20.0) * 0.5) * vec3f(1.0, 0.98, 0.92);
    let specRimR = pow(max(dot(N, hRimR), 0.0), 80.0) * 6.0 * vec3f(0.85, 1.1, 1.6);
    let specRimL = pow(max(dot(N, hRimL), 0.0), 80.0) * 4.8 * vec3f(0.9, 1.05, 1.4);
    
    // Prismatic facet glints
    let fluteGlint = pow(abs(fluteShape), 10.0) * pow(max(dot(N, hKey), 0.0), 24.0) * 2.8;
    let shoulderGlint = pow(abs(diamond), 8.0) * pow(max(dot(N, hRimR), 0.0), 24.0) * 2.4;
    
    let totalSpecular = studioRefl * F + (specKey + specRimR + specRimL + (fluteGlint + shoulderGlint) * vec3f(1.1, 1.1, 1.3)) * (F0 + 0.95 * pow(1.0 - ndotv, 3.0));
    
    // --- 4. Spectral Chromatic Refraction (Snell's Law) ---
    // Cauchy dispersion for lead crystal: Red n=1.51, Green n=1.55, Blue n=1.60
    var R_r = refract(-V, N, 1.0 / 1.51);
    var R_g = refract(-V, N, 1.0 / 1.55);
    var R_b = refract(-V, N, 1.0 / 1.60);
    
    if (dot(R_r, R_r) < 0.01) { R_r = reflect(-V, N); }
    if (dot(R_g, R_g) < 0.01) { R_g = reflect(-V, N); }
    if (dot(R_b, R_b) < 0.01) { R_b = reflect(-V, N); }
    
    // Sample refracted studio cove backdrop
    let bg_r = sampleBackground(R_r).r;
    let bg_g = sampleBackground(R_g).g;
    let bg_b = sampleBackground(R_b).b;
    var refractedLight = vec3f(bg_r, bg_g, bg_b);
    
    // --- 5. Physical Amber Liquid Absorption (Beer-Lambert Law) ---
    // The main body (h from 0.08 to 0.62) contains glowing amber liqueur
    let isBodySection = (h >= 0.075 && h <= 0.62);
    if (isBodySection) {
        // Path length through the liquid cylinder
        let chord = sqrt(max(0.0, 1.0 - pow(clamp(radius / 0.182, 0.0, 1.0), 2.0)));
        let liquidLevel = smoothstep(0.61, 0.58, h);
        let pathDist = max(chord * 0.32, 0.04) * liquidLevel;
        
        // Physical Beer-Lambert spectral absorption coefficients:
        // Blue is absorbed heavily, Green moderately, Red minimally
        let absorption = exp(-vec3f(0.55, 2.0, 6.8) * pathDist * 5.0);
        
        // Rich amber liquid color with internal light scattering
        let liquidScatter = vec3f(0.48, 0.28, 0.06) * (1.0 - exp(-pathDist * 4.0));
        
        refractedLight = refractedLight * absorption + liquidScatter;
        
        // Curved liquid meniscus line at top of liquid
        let meniscus = exp(-pow((h - 0.59) * 70.0, 2.0)) * 0.9 * vec3f(1.0, 0.85, 0.45);
        refractedLight += meniscus;
    } else if (h < 0.075) {
        // Thick clear crystal base with caustic prism glints
        let baseCaustic = exp(-pow((h - 0.035) * 40.0, 2.0)) * vec3f(0.95, 0.85, 0.70);
        let basePrism = pow(max(-R_g.y, 0.0), 3.0) * vec3f(0.4, 0.5, 0.7);
        refractedLight += baseCaustic + basePrism;
    } else {
        // Clear crystal neck & stopper above collar
        let clearGlint = pow(max(dot(reflect(-V, N), lKeyDir), 0.0), 16.0) * 0.4 * vec3f(1.0, 1.0, 1.0);
        refractedLight = refractedLight * 1.35 + clearGlint;
    }
    
    // Chromatic dispersion fringe enhancement along facet cuts
    let dispFringe = length(R_r - R_b) * 4.0;
    let spectralTint = vec3f(1.0 + dispFringe * 2.5, 1.0, 1.0 + dispFringe * 3.5);
    refractedLight *= spectralTint;
    
    // Final energy conservation blend between reflection and transmission
    let finalColor = totalSpecular + refractedLight * (1.0 - F);
    return vec4f(finalColor, 1.0);
}
`;

const EVOLVE_WGSL = `
@group(0) @binding(0) var<storage, read_write> particles: array<vec4f>;

@compute @workgroup_size(128) fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i * 2u >= arrayLength(&particles)) { return; }
    
    var posAge = particles[i * 2u];
    let seedParam = particles[i * 2u + 1u];
    
    let dt = 1.0 / 60.0;
    let t = posAge.w + dt;
    
    // Delicate floating crystal dust motes drifting gently around the bottle
    let speed = 0.03 + 0.02 * seedParam.x;
    let angle = seedParam.z * 6.2831853 + t * speed;
    let radius = 0.30 + seedParam.y * 0.40 + 0.03 * sin(t * 1.2 + seedParam.x * 15.0);
    let height = 0.10 + fract(seedParam.w + t * 0.015) * 0.82;
    
    let p = vec3f(radius * cos(angle), height, radius * sin(angle));
    particles[i * 2u] = vec4f(p, t);
}
`;

const DUST_WGSL = CAMERA_WGSL + `
struct V {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec3f,
};

@vertex fn vs_main(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> V {
    let a = instances[i * 2u];
    let b = instances[i * 2u + 1u];
    
    let corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    let uv = corners[v];
    
    var clip = camera.viewProj * vec4f(a.xyz, 1.0);
    let size = (0.0014 + 0.0022 * b.x) * camera.proj[1][1];
    clip.x += uv.x * size / camera.aspect;
    clip.y += uv.y * size;
    
    var o: V;
    o.pos = clip;
    o.uv = uv;
    
    // Subtle luxury dust sparkle: warm amber & cool diamond blue
    let warm = vec3f(1.4, 1.0, 0.4);
    let cool = vec3f(0.4, 0.8, 1.8);
    let baseColor = mix(warm, cool, b.y);
    let twinkle = 0.2 + 0.8 * pow(0.5 + 0.5 * sin(a.w * 3.5 + b.z * 40.0), 6.0);
    
    o.color = baseColor * twinkle * 0.45;
    return o;
}

@fragment fn fs_main(v: V) -> @location(0) vec4f {
    let r2 = dot(v.uv, v.uv);
    if (r2 > 1.0) { discard; }
    let alpha = exp(-r2 * 5.0) * 0.6;
    return vec4f(v.color * alpha, alpha);
}
`;

export default async function crystallineBottle({ engine, width, height, scale = 1, settings = {}, film, log = () => {} }) {
    const { device } = engine;
    const dustCount = settings.dustCount ?? 96;
    const subdivisionLevels = settings.subdivisionLevels ?? 3;

    // Load bottle geometry from public site assets
    const bottleFile = `${SITE}/public/3dAssetsNotIncluded/bottle.glb`;
    if (!existsSync(bottleFile)) {
        throw new Error(`crystalline-bottle: ${bottleFile} does not exist; check NG_SITE or starter-vite-ts checkout.`);
    }

    const rawGlb = readFileSync(bottleFile);
    const parsed = GLBParser.parse(rawGlb.buffer.slice(rawGlb.byteOffset, rawGlb.byteOffset + rawGlb.byteLength));
    const baseMesh = parsed.meshes?.[0];
    if (!baseMesh) {
        throw new Error('crystalline-bottle: could not parse mesh from bottle.glb');
    }

    log(`Loaded base bottle model: ${baseMesh.vertexCount} vertices, ${baseMesh.indices.length / 3} triangles.`);
    log(`Subdividing with PN-Triangles (Level ${subdivisionLevels})...`);

    const subdivided = subdividePNTriangles(baseMesh.vertices, baseMesh.indices, subdivisionLevels);
    log(`Subdivision complete: ${subdivided.vertices.length / 8} vertices, ${subdivided.indices.length / 3} smooth curved triangles.`);

    // NullGraph PostChain setup with calibrated cinematic thresholds
    const post = new PostChain(engine, { gbuffer: { depth: 'depth24plus' }, scale });
    post.setChain([
        new Bloom({ threshold: 1.05, knee: 0.35, intensity: 0.22, mips: 5 }),
        new Halation({ threshold: 1.15, knee: 0.35, intensity: 0.16, radius: 1.6, tint: [1.0, 0.72, 0.42] }),
        new Tonemap({ curve: 'aces', exposure: 1.15, whitePoint: 4.2 }),
        new FXAA(),
        new ChromaticAberration({ offset: 0.0012 }),
        new Vignette({ intensity: 0.26, softness: 0.70 }),
        new FilmGrain({ intensity: 0.020 }),
    ]);
    post.resize(width, height);

    // HDR Scene Render Pass
    const pass = engine.createPass({
        name: 'Crystalline Bottle / Studio HDR',
        isMainScreenPass: false,
        ...post.sceneAttachments({
            clearValue: { r: 0.010, g: 0.013, b: 0.020, a: 1.0 },
            depthClearValue: 1.0,
        }),
    });

    // 1. Studio Backdrop & Obsidian Floor Batch
    const background = engine.createBatch(pass, {
        label: 'Studio Stage & Obsidian Floor',
        shaderCode: BACKDROP_WGSL,
        strideFloats: 8,
        maxInstances: 1,
        targetFormat: 'rgba16float',
        depthWriteEnabled: true,
        depthCompare: 'always',
    });
    engine.updateBatchData(background, new Float32Array(8), 1);

    // 2. High-Res Subdivided Hero Bottle Mesh Batch
    const bottleBatch = engine.createBatch(pass, {
        label: 'Crystalline Bottle / Hero Mesh',
        shaderCode: BOTTLE_WGSL,
        strideFloats: 8,
        maxInstances: 1,
        targetFormat: 'rgba16float',
        depthWriteEnabled: true,
        depthCompare: 'less',
        vertexLayouts: [{
            arrayStride: 32,
            stepMode: 'vertex',
            attributes: [
                { shaderLocation: 0, offset: 0, format: 'float32x3' },
                { shaderLocation: 1, offset: 12, format: 'float32x3' },
                { shaderLocation: 2, offset: 24, format: 'float32x2' },
            ],
        }],
    });

    const vBuf = device.createBuffer({
        label: 'Crystalline Bottle / Subdivided Vertices',
        size: subdivided.vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(vBuf.getMappedRange()).set(subdivided.vertices);
    vBuf.unmap();

    const iBuf = device.createBuffer({
        label: 'Crystalline Bottle / Subdivided Indices',
        size: subdivided.indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(iBuf.getMappedRange()).set(subdivided.indices);
    iBuf.unmap();

    bottleBatch.vertexBuffer = vBuf;
    bottleBatch.indexBuffer = iBuf;
    bottleBatch.indexCount = subdivided.indices.length;
    bottleBatch.indexFormat = 'uint32';
    engine.updateBatchData(bottleBatch, new Float32Array(8), 1);

    // 3. Floating Luminous Crystal Dust Particles Batch
    const dustBatch = engine.createBatch(pass, {
        label: 'Floating Crystal Motes',
        shaderCode: DUST_WGSL,
        strideFloats: 8,
        maxInstances: dustCount,
        targetFormat: 'rgba16float',
        depthWriteEnabled: false,
        depthCompare: 'less',
        blend: {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
        },
    });

    const dustData = new Float32Array(dustCount * 8);
    let seed = 314159265;
    const rand = () => {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        return seed / 4294967296;
    };

    for (let i = 0; i < dustCount; i++) {
        const sx = rand();
        const sy = rand();
        const sz = rand();
        const sw = rand();
        dustData.set([0, 0, 0, 0.0, sx, sy, sz, sw], i * 8);
    }
    engine.updateBatchData(dustBatch, dustData, dustCount);

    const evolveStage = engine.createComputeStage({
        name: 'Crystal Dust / particle evolution',
        shaderCode: EVOLVE_WGSL,
        dispatchCount: [Math.ceil(dustCount / 128), 1, 1],
    });
    engine.setComputePassBindGroup(evolveStage, [{ binding: 0, resource: { buffer: dustBatch.storageBuffer } }]);

    // Initialize dust positions
    engine.render([evolveStage]);
    await device.queue.onSubmittedWorkDone();

    const camera = new Camera(38, width / height, 0.05, 50);
    const defaultView = { target: [0.0, 0.44, 0.0], distance: 1.58, azimuth: 0.38, polar: 1.44 };

    const track = film.track('crystal-dust', dustBatch.storageBuffer, {
        every: 6,
        stride: 32,
        fields: [0, 1, 2, 3, 4, 5, 6, 7].map(word => ({ word, store: 'f32', interp: 'linear' })),
    });

    log(`Crystalline bottle ready: ${subdivided.indices.length / 3} curved triangles, ${dustCount} floating dust motes.`);

    return {
        dt: 1 / 60,
        settings: { dustCount, subdivisionLevels },
        tracks: [track],
        checkpoint: [{ name: 'crystal-dust', buffer: dustBatch.storageBuffer }],
        info: {
            model: 'High-Res Subdivided Crystalline Bottle (Spectral Dispersion Refraction, Cut-Crystal Fluting, Amber Elixir Core, Studio Obsidian Floor, HDR Softbox Highlights)',
            wgsl: {
                backdrop: hash('sha256', BACKDROP_WGSL),
                bottle: hash('sha256', BOTTLE_WGSL),
                dust: hash('sha256', DUST_WGSL),
                evolve: hash('sha256', EVOLVE_WGSL),
            },
        },
        async step() {
            engine.render([evolveStage]);
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
            vBuf.destroy();
            iBuf.destroy();
            post.destroy();
            engine.clearPasses();
        },
    };
}
