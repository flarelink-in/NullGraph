// src/extras/environment/probeShaders.ts
//
// WGSL generators for Phase 2 (docs/lighting-probes-design.md): the volume lookup's group-2
// bindings (7–13, the slots Phase 1 §4.4 reserved), the re-emitted getIndirectDiffuse /
// getIndirectSpecular contract, the capture-atlas → SH-L1 projection kernel, and the
// reflection-probe tile → staging-cube assemble kernel. Each kernel is a complete self-contained
// module so it can be parse-validated offline and so `layout: 'auto'` sees exactly the bindings
// the body uses.
//
// The math mirrors probeMath.ts line-for-line; the two are kept in lockstep by
// test/environment-probes.test.mjs. NO BACKTICKS anywhere in this file — a stray one terminates
// the JS template literal silently and the rest of the shader parses as TypeScript
// (design-principles §13 rule 3; environment-ibl.test.mjs U10 pins it for Phase 1, and the
// probes test pins it here).

import { ENV_MATH_WGSL } from './envShaders';
import type { ShaderIndirectLayout } from './EnvironmentSystem';

// ---------------------------------------------------------------------------
// §4.3 / §4.4 — the group-2 bindings (7–13)
// ---------------------------------------------------------------------------

/**
 * The `wgslStructs` addition: bindings 7–13 of group 2, EXACTLY the budget Phase 1 §4.4 left.
 * The volume gets 4 texture bindings + 1 uniform (which is why SH is L1, not L2 — §4.2);
 * reflection probes get 1 cube-array texture + 1 storage buffer. The env sampler (binding 3) is
 * reused for every probe sample — a 6th binding would not fit.
 */
export function buildProbeStructsWGSL(group: number): string {
    const g = group;
    return /* wgsl */ `
struct ProbeParams {
    origin: vec4<f32>,      // .xyz world position of probe (0,0,0); .w = intensity
    invExtent: vec4<f32>,   // .xyz = 1 / (spacing * (dims-1)); .w = spacing
    dims: vec4<f32>,        // .xyz = grid dims as floats; .w = normalBias in texels
    fade: vec4<f32>,        // .x = border fade width (world units); .yzw reserved
};                          // 64 bytes

struct ReflProbe {
    center: vec4<f32>,      // .xyz proxy box centre (world); .w = blend distance
    halfExtent: vec4<f32>,  // .xyz proxy box half-extents; .w = influence radius
    capture: vec4<f32>,     // .xyz capture position; .w = layer index (NEGATIVE = never captured)
};                          // 48 bytes

struct ReflProbeBuffer {
    count: u32,
    _pad: vec3<u32>,
    probes: array<ReflProbe>,
};

@group(${g}) @binding(7)  var probeR: texture_3d<f32>;
@group(${g}) @binding(8)  var probeG: texture_3d<f32>;
@group(${g}) @binding(9)  var probeB: texture_3d<f32>;
@group(${g}) @binding(10) var probeValidity: texture_3d<f32>;
@group(${g}) @binding(11) var<uniform> probe: ProbeParams;
@group(${g}) @binding(12) var reflProbes: texture_cube_array<f32>;
@group(${g}) @binding(13) var<storage, read> reflProbeData: ReflProbeBuffer;
`;
}

// ---------------------------------------------------------------------------
// §5 / §8 — the re-emitted contract: getIndirectDiffuse + getIndirectSpecular
// ---------------------------------------------------------------------------

/**
 * The `wgslLogic` half of the Phase 2 `ShaderIndirectLayout` — the SAME two functions with
 * volume-sampling bodies (§1: Phase 2 is the release that uses the worldPos parameter). The
 * Phase 1 bodies, renamed getIndirectDiffuseGlobal / getIndirectSpecularGlobal by the decorator,
 * remain the fallback — every fallback in here blends TOWARD the global environment, never
 * toward black (§5, design principle: the failure mode of this system is "looks like Phase 1").
 */
export function buildProbeIndirectLogicWGSL(): string {
    return /* wgsl */ `
fn evalSHL1probe(r: vec4<f32>, g: vec4<f32>, b: vec4<f32>, N: vec3<f32>) -> vec3<f32> {
    // The cosine-convolution constants were folded into the stored coefficients at projection
    // time (§7.2), so the stored value IS irradiance/π and this dot is the whole evaluation.
    // max(): an L1 projection rings on surfaces facing away from the dominant bounce (§5, U4).
    let d = vec4<f32>(N, 1.0);
    return max(vec3<f32>(dot(r, d), dot(g, d), dot(b, d)), vec3<f32>(0.0));
}

fn volumeBorderWeight(uvw: vec3<f32>, fadeUV: f32) -> f32 {
    let edge = min(uvw, vec3<f32>(1.0) - uvw);
    return smoothstep(0.0, fadeUV, min(edge.x, min(edge.y, edge.z)));
}

fn parallaxCorrectProbe(p: ReflProbe, worldPos: vec3<f32>, R: vec3<f32>) -> vec3<f32> {
    // Intersect the reflection ray with the proxy box (slab method, furthest exit), then look up
    // the direction FROM THE CAPTURE POINT — p.center here is the sliding-reflection bug (§8.2).
    let invR = 1.0 / R;
    let first = (p.center.xyz + p.halfExtent.xyz - worldPos) * invR;
    let second = (p.center.xyz - p.halfExtent.xyz - worldPos) * invR;
    let furthest = max(first, second);
    let dist = min(min(furthest.x, furthest.y), furthest.z);
    let hit = worldPos + R * dist;
    return normalize(hit - p.capture.xyz);
}

fn reflInfluenceWeight(p: ReflProbe, worldPos: vec3<f32>) -> f32 {
    if (p.capture.w < 0.0) { return 0.0; }   // never captured → the global environment covers it
    let q = abs(worldPos - p.center.xyz) - p.halfExtent.xyz;
    let inside = -max(q.x, max(q.y, q.z));   // distance inside the nearest face (negative outside)
    if (inside < 0.0) { return 0.0; }
    if (length(worldPos - p.center.xyz) > p.halfExtent.w) { return 0.0; }   // influence radius
    return smoothstep(0.0, p.center.w, inside);   // 1 deep inside → 0 at the proxy boundary
}

fn getIndirectDiffuse(worldPos: vec3<f32>, N: vec3<f32>) -> vec3<f32> {
    // Anti-prune (Phase 1's reason): a diffuse-only material (Lambert/Toon) never reaches
    // getIndirectSpecular, so without this touch layout:'auto' would drop bindings 12/13 and the
    // combined group-2 bind group would fail validation on a real device.
    let _antiPrune = textureSampleLevel(reflProbes, envSampler, vec3<f32>(0.0, 1.0, 0.0), 0u, 0.0).r
                   + f32(reflProbeData.count);

    let global = getIndirectDiffuseGlobal(worldPos, N);
    // Normal bias: step off the surface toward the probe grid so a probe *behind* the wall is
    // not the nearest sample — the cheapest leak mitigation (§6.1).
    let biased = worldPos + N * (probe.dims.w * probe.invExtent.w);

    let uvw = (biased - probe.origin.xyz) * probe.invExtent.xyz;
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) {
        return global + vec3<f32>(_antiPrune * 0.0000001);   // outside the volume → Phase 1
    }
    // Hardware trilinear taps texel CENTRES: probe (i,j,k) sits at (i+0.5)/dims, not i/(dims-1)
    // — without this remap a probe-centre sample blends its neighbours (U2 pins it).
    let uvwSample = (uvw * (probe.dims.xyz - vec3<f32>(1.0)) + vec3<f32>(0.5)) / probe.dims.xyz;
    let r = textureSampleLevel(probeR, envSampler, uvwSample, 0.0);
    let g = textureSampleLevel(probeG, envSampler, uvwSample, 0.0);
    let b = textureSampleLevel(probeB, envSampler, uvwSample, 0.0);
    let valid = textureSampleLevel(probeValidity, envSampler, uvwSample, 0.0).r;

    let local = evalSHL1probe(r, g, b, N) * probe.origin.w;

    // Two blends, both toward Phase 1 and never toward black (§5): validity (how much of the
    // 8-probe neighbourhood was usable) × border (a smooth exit at the volume's edge).
    let border = volumeBorderWeight(uvw, probe.fade.x * probe.invExtent.x);
    return mix(global, local, valid * border) + vec3<f32>(_antiPrune * 0.0000001);
}

fn getIndirectSpecular(worldPos: vec3<f32>, N: vec3<f32>, V: vec3<f32>,
                       roughness: f32, F0: vec3<f32>) -> vec3<f32> {
    let R = reflect(-V, N);
    // The cube array's own mip count scales the roughness lobe (textureNumLevels, not
    // env.specularMip — the array may have fewer levels than the global cube).
    let mip = roughness * f32(textureNumLevels(reflProbes) - 1u);

    var acc = vec3<f32>(0.0);
    var w   = 0.0;
    let reflCount = reflProbeData.count;
    for (var i = 0u; i < reflCount && w < 1.0; i = i + 1u) {
        let p = reflProbeData.probes[i];
        let wi = min(reflInfluenceWeight(p, worldPos), 1.0 - w);   // total probe weight ≤ 1 (U7)
        if (wi <= 0.0) { continue; }
        acc += textureSampleLevel(reflProbes, envSampler, parallaxCorrectProbe(p, worldPos, R),
                                  u32(p.capture.w), mip).rgb * wi;
        w += wi;
    }
    // Whatever weight remains goes to Phase 1's global cube, shaded EXACTLY as Phase 1 shades it
    // (yaw + horizon fade + its own mip scale) — a scene with no probes is Phase 1 verbatim (§8.3).
    var Rg = rotateYEnv(R, -env.rotation.w);
    let horizon = mix(1.0, smoothstep(-0.1, 0.1, Rg.y), env.horizonFade);
    acc += textureSampleLevel(envSpecular, envSampler, Rg, roughness * env.specularMip).rgb * horizon * (1.0 - w);

    // Byte-for-byte Phase 1's split-sum recombination — what keeps the white furnace valid (§8.3).
    let dfg = textureSampleLevel(brdfLUT, envSampler, vec2<f32>(max(dot(N, V), 0.0), roughness), 0.0).rg;
    return acc * (F0 * dfg.x + dfg.y) * env.intensity;
}
`;
}

// ---------------------------------------------------------------------------
// The decorator (§9, §12.2) — Phase 2 = the Phase 1 contract, re-emitted against the volume
// ---------------------------------------------------------------------------

/** The markers the decorator guards — the exact definition strings Phase 1 emits. */
const DIFFUSE_DEF = 'fn getIndirectDiffuse(';
const SPECULAR_DEF = 'fn getIndirectSpecular(';

function countOccurrences(haystack: string, needle: string): number {
    let n = 0, i = 0;
    while ((i = haystack.indexOf(needle, i)) !== -1) { n++; i += needle.length; }
    return n;
}

/**
 * Compose the Phase 2 `ShaderIndirectLayout`: the environment's own structs/logic with the two
 * global functions renamed out of the way, then the probe bindings and the volume-sampling
 * bodies. Same discipline as `ShadowSystem.wrapLighting` (design-principles §4): ONE textual
 * rename per function, guarded by a marker check that throws the moment Phase 1's output stops
 * containing it — a decorator without that guard is a silent breakage waiting for a rename.
 */
export function buildProbeIndirectLayout(envLayout: ShaderIndirectLayout, group: number = 2): ShaderIndirectLayout {
    if (countOccurrences(envLayout.wgslLogic, DIFFUSE_DEF) !== 1 ||
        countOccurrences(envLayout.wgslLogic, SPECULAR_DEF) !== 1) {
        throw new Error(
            'ProbeVolume: EnvironmentSystem WGSL no longer defines getIndirectDiffuse()/' +
            'getIndirectSpecular() exactly once; the probe decorator is out of date with ' +
            'EnvironmentSystem.getShaderLayout().');
    }
    const renamed = envLayout.wgslLogic
        .replace(DIFFUSE_DEF, 'fn getIndirectDiffuseGlobal(')
        .replace(SPECULAR_DEF, 'fn getIndirectSpecularGlobal(');
    return {
        wgslStructs: envLayout.wgslStructs + '\n' + buildProbeStructsWGSL(group),
        wgslLogic: renamed + '\n' + buildProbeIndirectLogicWGSL(),
    };
}


// ---------------------------------------------------------------------------
// §7.2 The projection kernel — capture-atlas tiles → SH-L1 → the 3 volume textures
// ---------------------------------------------------------------------------

/**
 * One workgroup (64 threads) reduces one probe's 6 capture tiles (single workgroup: WGSL has no
 * atomic<f32>, and 6144 texels reduce comfortably — Phase 1 §7.2's shape and reason). Sky
 * substitution: a texel with cleared depth (1.0) hit nothing, so it reads the environment cube —
 * the capture's sky (§7.2); without it outdoor probes would lose the sky hemisphere. Validity:
 * texels closer than `misc.z` world units mark the probe enclosed (§7.2: > 60% ⇒ validity 0).
 *
 * Dispatch (1,1,1) per scheduled probe; `grid.xyz` is the probe's volume coordinate.
 */
export function buildProbeProjectionWGSL(opts: { faceSize: number }): string {
    const F = opts.faceSize;
    return /* wgsl */ `
struct ProbeProjectParams {
    tileOrigin: array<vec4<u32>, 6>,  // per-face tile pixel origin in the capture atlas
    grid: vec4<u32>,                  // .xyz = probe (i,j,k); .w = faceSize
    misc: vec4<f32>,                  // .x = near, .y = far, .z = enclosed distance, .w = sky intensity
};                                    // 128 bytes

@group(0) @binding(0) var captureColor: texture_2d<f32>;
@group(0) @binding(1) var captureDepth: texture_2d<f32>;
@group(0) @binding(2) var envSpecular: texture_cube<f32>;
@group(0) @binding(3) var envSampler: sampler;
@group(0) @binding(4) var<uniform> proj: ProbeProjectParams;
@group(0) @binding(5) var probeR: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var probeG: texture_storage_3d<rgba16float, write>;
@group(0) @binding(7) var probeB: texture_storage_3d<rgba16float, write>;
@group(0) @binding(8) var probeValidity: texture_storage_3d<rgba8unorm, write>;

${ENV_MATH_WGSL}

const FACE_SIZE: u32 = ${F}u;
const FACE_TEXELS: u32 = FACE_SIZE * FACE_SIZE;
const TEXELS: u32 = 6u * FACE_TEXELS;
// 1/(2π), 1/(4π): the folded cosine-convolution constants (§7.2) — mirrors probeMath.ts.
const INV_2PI: f32 = 0.15915494;
const INV_4PI: f32 = 0.07957747;

var<workgroup> partial: array<array<f32, 13>, 64>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    for (var k = 0u; k < 13u; k = k + 1u) { partial[t][k] = 0.0; }

    for (var texel = t; texel < TEXELS; texel = texel + 64u) {
        let face = texel / FACE_TEXELS;
        let rem = texel - face * FACE_TEXELS;
        let px = vec2<u32>(rem % FACE_SIZE, rem / FACE_SIZE);
        let uv = (vec2<f32>(px) + 0.5) / f32(FACE_SIZE);
        // A captured tile is a rendered viewport: NDC +y lands on texture row 0, so tile row y
        // sees cube-face v = 1 - uv.y — the flip that makes a rendered tile a true cube face.
        let dir = faceDirection(face, vec2<f32>(uv.x, 1.0 - uv.y));
        let sa = cubeTexelSolidAngle(uv, FACE_SIZE);
        let origin = proj.tileOrigin[face].xy;
        let ndcZ = textureLoad(captureDepth, vec2<i32>(origin + px), 0).x;

        var rgb: vec3<f32>;
        if (ndcZ >= 1.0) {
            // Nothing drawn along this ray: the sky. envSpecular mip 0 is the prefiltered source.
            rgb = textureSampleLevel(envSpecular, envSampler, dir, 0.0).rgb * proj.misc.w;
        } else {
            rgb = textureLoad(captureColor, vec2<i32>(origin + px), 0).rgb;
            let c = uv * 2.0 - 1.0;
            let rayLen = sqrt(1.0 + c.x * c.x + c.y * c.y);
            let linearZ = (proj.misc.x * proj.misc.y) / (proj.misc.y - ndcZ * (proj.misc.y - proj.misc.x));
            if (linearZ * rayLen < proj.misc.z) {
                partial[t][12] += 1.0;   // enclosed texel → drives validity (§7.2)
            }
        }
        partial[t][0] += rgb.r * dir.x * sa;
        partial[t][1] += rgb.r * dir.y * sa;
        partial[t][2] += rgb.r * dir.z * sa;
        partial[t][3] += rgb.g * dir.x * sa;
        partial[t][4] += rgb.g * dir.y * sa;
        partial[t][5] += rgb.g * dir.z * sa;
        partial[t][6] += rgb.b * dir.x * sa;
        partial[t][7] += rgb.b * dir.y * sa;
        partial[t][8] += rgb.b * dir.z * sa;
        partial[t][9]  += rgb.r * sa;
        partial[t][10] += rgb.g * sa;
        partial[t][11] += rgb.b * sa;
    }
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (t < stride) {
            for (var k = 0u; k < 13u; k = k + 1u) { partial[t][k] += partial[t + stride][k]; }
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        let coord = vec3<i32>(proj.grid.xyz);
        textureStore(probeR, coord, vec4<f32>(
            partial[0][0] * INV_2PI, partial[0][1] * INV_2PI, partial[0][2] * INV_2PI,
            partial[0][9] * INV_4PI));
        textureStore(probeG, coord, vec4<f32>(
            partial[0][3] * INV_2PI, partial[0][4] * INV_2PI, partial[0][5] * INV_2PI,
            partial[0][10] * INV_4PI));
        textureStore(probeB, coord, vec4<f32>(
            partial[0][6] * INV_2PI, partial[0][7] * INV_2PI, partial[0][8] * INV_2PI,
            partial[0][11] * INV_4PI));
        let enclosedFrac = partial[0][12] / f32(TEXELS);
        // More than 60% enclosed → the probe is inside geometry; validity 0 (§7.2).
        textureStore(probeValidity, coord,
            vec4<f32>(select(1.0 - enclosedFrac, 0.0, enclosedFrac > 0.6), 0.0, 0.0, 0.0));
    }
}
`;
}


// ---------------------------------------------------------------------------
// §8.1 The assemble kernel — reflection capture tiles → the staging cube
// ---------------------------------------------------------------------------

/**
 * Copy one reflection probe's 6 capture tiles (128² default) into the staging cube, applying the
 * same viewport y-flip as the projection kernel. Staging exists so the prefilter — Phase 1's GGX
 * kernel, reused unchanged — samples a true cube (§8.1); it runs next, reading this cube and
 * writing the probe's layers of the shared cube array. Dispatch (faceSize/8, faceSize/8, 6).
 */
export function buildReflAssembleWGSL(): string {
    return /* wgsl */ `
struct ReflAssembleParams {
    tileOrigin: array<vec4<u32>, 6>,  // per-face tile pixel origin in the capture atlas
    misc: vec4<f32>,                  // .x = sky intensity; .yzw reserved
};                                    // 112 bytes

@group(0) @binding(0) var captureColor: texture_2d<f32>;
@group(0) @binding(1) var captureDepth: texture_2d<f32>;
@group(0) @binding(2) var envSpecular: texture_cube<f32>;
@group(0) @binding(3) var envSampler: sampler;
@group(0) @binding(4) var<uniform> asmP: ReflAssembleParams;
@group(0) @binding(5) var dstCube: texture_storage_2d_array<rgba16float, write>;

${ENV_MATH_WGSL}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(dstCube);
    if (gid.x >= dim.x || gid.y >= dim.y) { return; }
    let origin = asmP.tileOrigin[gid.z].xy;
    // Cube texel (x, y) sees the direction a rendered viewport stored at (x, F-1-y) — NDC +y is
    // texture row 0, cube-face v = 1 (same flip as the SH projection, §7.1).
    let srcPx = vec2<i32>(i32(origin.x + gid.x), i32(origin.y + dim.y - 1u - gid.y));
    var rgb = textureLoad(captureColor, srcPx, 0).rgb;
    if (textureLoad(captureDepth, srcPx, 0).x >= 1.0) {
        // Nothing drawn along this ray: the sky — specular loses the reflection of the sky dome
        // without it (same substitution as the SH projection, §7.2).
        let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dim);
        rgb = textureSampleLevel(envSpecular, envSampler, faceDirection(gid.z, uv), 0.0).rgb * asmP.misc.x;
    }
    textureStore(dstCube, gid.xy, gid.z, vec4<f32>(rgb, 1.0));
}
`;
}

