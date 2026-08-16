// src/extras/environment/envShaders.ts
//
// WGSL generators for the EnvironmentSystem (docs/lighting-ibl-design.md). Every kernel is a
// complete, self-contained module so it can be parse-validated offline and so the engine's
// `layout: 'auto'` pipeline creation sees exactly the bindings the body uses.
//
// The math mirrors envMath.ts line-for-line; the two are kept in lockstep by
// test/environment-ibl.test.mjs (U1–U7), the same discipline as visibility's
// cullMath.ts ↔ visibilityShaders.ts pair.

import { ENV_PARAMS_FLOATS, SH_SOURCE_SIZE } from './envMath';

// ---------------------------------------------------------------------------
// Shared chunks
// ---------------------------------------------------------------------------

/**
 * Cube-face direction + equirect UV + true per-texel solid angle (§6.1, §7.2). The `switch` in
 * `faceDirection` is the WebGPU cube-face convention — one wrong sign mirrors the environment in
 * a way that looks plausible until a recognizable feature appears in a reflection (U1 pins it).
 */
export const ENV_MATH_WGSL = /* wgsl */ `
const PI: f32 = 3.14159265359;

fn faceDirection(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let c = uv * 2.0 - 1.0;                 // [-1,1]
    switch face {
        case 0u: { return normalize(vec3<f32>( 1.0, -c.y, -c.x)); }   // +X
        case 1u: { return normalize(vec3<f32>(-1.0, -c.y,  c.x)); }   // -X
        case 2u: { return normalize(vec3<f32>( c.x,  1.0,  c.y)); }   // +Y
        case 3u: { return normalize(vec3<f32>( c.x, -1.0, -c.y)); }   // -Y
        case 4u: { return normalize(vec3<f32>( c.x, -c.y,  1.0)); }   // +Z
        default: { return normalize(vec3<f32>(-c.x, -c.y, -1.0)); }   // -Z
    }
}

fn dirToEquirectUV(d: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(atan2(d.z, d.x) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI);
}

// The TRUE solid angle of a cube texel — corner texels subtend markedly less than centre ones,
// and a uniform 4π/384 weight tilts the irradiance field toward the corners (§7.2, U3).
// EXACT, via spherical excess: the midpoint form (4/size²)/(s²+t²+1)^1.5 converges only as
// O(1/size²) and over-counts by 0.38% at the 8×8 SH mip. Mirrors envMath.cubeTexelSolidAngle.
fn areaElement(x: f32, y: f32) -> f32 {
    return atan2(x * y, sqrt(x * x + y * y + 1.0));
}

fn cubeTexelSolidAngle(uv: vec2<f32>, size: u32) -> f32 {
    let inv = 1.0 / f32(size);
    let u = uv.x * 2.0 - 1.0;
    let v = uv.y * 2.0 - 1.0;
    let x0 = u - inv; let x1 = u + inv;
    let y0 = v - inv; let y1 = v + inv;
    return areaElement(x0, y0) - areaElement(x0, y1) - areaElement(x1, y0) + areaElement(x1, y1);
}

// The 9 SH-L2 basis values, order [1, y, z, x, xy, yz, 3z²−1, xz, x²−y²] — mirrors envMath.ts.
fn shBasisL2(d: vec3<f32>) -> array<f32, 9> {
    return array<f32, 9>(
        0.28209479,
        0.48860251 * d.y,
        0.48860251 * d.z,
        0.48860251 * d.x,
        1.09254843 * d.x * d.y,
        1.09254843 * d.y * d.z,
        0.31539157 * (3.0 * d.z * d.z - 1.0),
        1.09254843 * d.x * d.z,
        0.54627422 * (d.x * d.x - d.y * d.y),
    );
}

fn rotateY(v: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(c * v.x - s * v.z, v.y, s * v.x + c * v.z);
}
`;

/** Hammersley + GGX importance sampling, shared by the prefilter and the BRDF LUT (§6.3–6.4). */
export const ENV_SAMPLING_WGSL = /* wgsl */ `
fn radicalInverse(bitsIn: u32) -> f32 {
    var bits = bitsIn;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits >> 1u) & 0x55555555u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits >> 2u) & 0x33333333u);
    bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits >> 4u) & 0x0f0f0f0fu);
    bits = ((bits & 0x00ff00ffu) << 8u) | ((bits >> 8u) & 0x00ff00ffu);
    return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radicalInverse(i));
}

// TBN frame about N, then the GGX half-vector for roughness 'a' (already squared).
fn importanceSampleGGX(xi: vec2<f32>, N: vec3<f32>, a: f32) -> vec3<f32> {
    let phi = 2.0 * PI * xi.x;
    let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    let hTangent = vec3<f32>(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(N.y) > 0.999) { up = vec3<f32>(1.0, 0.0, 0.0); }
    let T = normalize(cross(up, N));
    let B = cross(N, T);
    return normalize(T * hTangent.x + B * hTangent.y + N * hTangent.z);
}

fn distributionGGXEnv(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}
`;

// ---------------------------------------------------------------------------
// Stage A — source cube from equirect or procedural sky (§6.1)
// ---------------------------------------------------------------------------

/**
 * Fill the source cube's mip 0. `procedural: false` samples an equirectangular HDR texture;
 * `true` evaluates an analytic gradient + sun disc, so the module works with NO asset at all.
 * Dispatch (size/8, size/8, 6); gid.z is the cube face.
 */
export function buildSourceCubeWGSL(opts: { procedural: boolean }): string {
    const body = opts.procedural ? /* wgsl */ `
struct SkyParams {
    zenith: vec3<f32>,
    sunCos: f32,
    horizon: vec3<f32>,
    sunIntensity: f32,
    ground: vec3<f32>,
    sunDir: vec3<f32>,          // TRAVEL direction (where the light goes)
};
@group(0) @binding(0) var<uniform> sky: SkyParams;
@group(0) @binding(2) var dstCube: texture_storage_2d_array<rgba16float, write>;

fn skyColor(dir: vec3<f32>) -> vec3<f32> {
    let t = clamp(dir.y, -1.0, 1.0);
    var c = mix(sky.horizon, sky.zenith, pow(max(t, 0.0), 0.6));
    c = mix(c, sky.ground, pow(max(-t, 0.0), 0.5));
    let sunAmt = smoothstep(sky.sunCos, sky.sunCos + 0.0003, dot(dir, -sky.sunDir));
    return c + sky.sunIntensity * sunAmt * vec3<f32>(1.0, 0.98, 0.95);
}
` : /* wgsl */ `
@group(0) @binding(0) var equirect: texture_2d<f32>;
@group(0) @binding(1) var srcSamp: sampler;
@group(0) @binding(2) var dstCube: texture_storage_2d_array<rgba16float, write>;
`;
    const sample = opts.procedural
        ? `textureStore(dstCube, gid.xy, gid.z, vec4<f32>(skyColor(dir), 1.0));`
        : `textureStore(dstCube, gid.xy, gid.z, vec4<f32>(textureSampleLevel(equirect, srcSamp, dirToEquirectUV(dir), 0.0).rgb, 1.0));`;

    return /* wgsl */ `
${ENV_MATH_WGSL}
${body}
@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Directly on the binding: WGSL has no constructible texture type, so a function cannot
    // return one — textureDimensions(someFn()) does not compile.
    let dim = textureDimensions(dstCube);
    if (gid.x >= dim.x || gid.y >= dim.y) { return; }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dim);
    let dir = faceDirection(gid.z, uv);
    ${sample}
}
`;
}

// ---------------------------------------------------------------------------
// Stage B — the source mip chain (§6.2): box average, the HiZPyramid pattern with max → mean
// ---------------------------------------------------------------------------

/**
 * One box-average mip of the source cube: reads mip L-1 as a cube, writes mip L as a 2d-array
 * storage view — structurally `buildHiZMipWGSL` (visibilityShaders.ts) with `max` → `mean`.
 * These mips exist for §6.3's firefly suppression and §7.2's 8×8 SH projection source.
 */
export function buildCubeMipWGSL(): string {
    return /* wgsl */ `
@group(0) @binding(0) var srcCube: texture_cube<f32>;
@group(0) @binding(1) var dstMip: texture_storage_2d_array<rgba16float, write>;

${ENV_MATH_WGSL}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dstDim = textureDimensions(dstMip);
    if (gid.x >= dstDim.x || gid.y >= dstDim.y) { return; }
    let srcDim = dstDim * 2u;
    let s = gid.xy * 2u;
    var sum = vec3<f32>(0.0);
    for (var dy = 0u; dy < 2u; dy = dy + 1u) {
        for (var dx = 0u; dx < 2u; dx = dx + 1u) {
            let c = min(s + vec2<u32>(dx, dy), srcDim - 1u);
            let uv = (vec2<f32>(c) + 0.5) / vec2<f32>(srcDim);
            sum += textureSampleLevel(srcCube, mipSampler, faceDirection(gid.z, uv), 0.0).rgb;
        }
    }
    textureStore(dstMip, gid.xy, gid.z, vec4<f32>(sum * 0.25, 1.0));
}

@group(0) @binding(2) var mipSampler: sampler;
`;
}

// ---------------------------------------------------------------------------
// Stage C — GGX prefilter, one stage per mip (§6.3)
// ---------------------------------------------------------------------------

/**
 * Convolve the source cube with a GGX lobe of roughness `level/(prefilterMips-1)` into
 * prefiltered mip `level`. Mip 0 short-circuits to a straight copy (a mirror — importance
 * sampling it would only add noise). The solid-angle mip selection (Karis) keeps a single
 * bright texel from surviving as a hard dot in a rough reflection; T2 measures the variance
 * with and without it.
 */
export function buildPrefilterWGSL(opts: { level: number; prefilterMips: number; samples: number; srcSize: number; arrayLayerOffset?: boolean }): string {
    const roughness = opts.prefilterMips > 1 ? opts.level / (opts.prefilterMips - 1) : 0;
    // Phase 2 (lighting-probes-design.md §8.1, §12.2): reflection probes prefilter INTO one cube
    // of the shared cube array. The destination layer offset arrives as a uniform so ONE set of
    // per-mip pipelines serves every probe — a pipeline-constant layer would recompile per probe.
    // Default (no offset) is byte-identical to the Phase 1 kernel.
    const layerDecl = opts.arrayLayerOffset
        ? '@group(0) @binding(3) var<uniform> reflLayerOffset: vec4<u32>;\n' : '';
    const storeLayer = opts.arrayLayerOffset ? 'gid.z + reflLayerOffset.x' : 'gid.z';
    if (roughness === 0) {
        // Mirror mip: a straight copy through the cube sampler.
        return /* wgsl */ `
@group(0) @binding(0) var srcCube: texture_cube<f32>;
@group(0) @binding(1) var srcSamp: sampler;
@group(0) @binding(2) var dstMip: texture_storage_2d_array<rgba16float, write>;
${layerDecl}
${ENV_MATH_WGSL}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dstDim = textureDimensions(dstMip);
    if (gid.x >= dstDim.x || gid.y >= dstDim.y) { return; }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dstDim);
    let dir = faceDirection(gid.z, uv);
    textureStore(dstMip, gid.xy, ${storeLayer}, vec4<f32>(textureSampleLevel(srcCube, srcSamp, dir, 0.0).rgb, 1.0));
}
`;
    }
    return /* wgsl */ `
@group(0) @binding(0) var srcCube: texture_cube<f32>;
@group(0) @binding(1) var srcSamp: sampler;
@group(0) @binding(2) var dstMip: texture_storage_2d_array<rgba16float, write>;
${layerDecl}
${ENV_MATH_WGSL}
${ENV_SAMPLING_WGSL}

const ROUGHNESS: f32 = ${roughness.toFixed(6)};
const SAMPLE_COUNT: u32 = ${opts.samples}u;
const SRC_SIZE: f32 = ${opts.srcSize.toFixed(1)};

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dstDim = textureDimensions(dstMip);
    if (gid.x >= dstDim.x || gid.y >= dstDim.y) { return; }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dstDim);
    let N = faceDirection(gid.z, uv);
    let V = N;                                   // the standard split-sum approximation

    var color = vec3<f32>(0.0);
    var weight = 0.0;
    for (var i = 0u; i < SAMPLE_COUNT; i = i + 1u) {
        let H = importanceSampleGGX(hammersley(i, SAMPLE_COUNT), N, ROUGHNESS);
        let L = normalize(2.0 * dot(V, H) * H - V);
        let NdotL = dot(N, L);
        if (NdotL <= 0.0) { continue; }

        // Firefly suppression (Karis): pick the source mip whose texel solid angle matches the
        // sample's, so one bright texel cannot survive as a hard dot in a rough reflection.
        let NdotH = max(dot(N, H), 0.0);
        let D = distributionGGXEnv(N, H, ROUGHNESS);
        let pdf = (D * NdotH / (4.0 * max(dot(V, H), 1e-4))) + 1e-4;
        let saTexel = 4.0 * PI / (6.0 * SRC_SIZE * SRC_SIZE);
        let saSample = 1.0 / (f32(SAMPLE_COUNT) * pdf);
        let mip = 0.5 * log2(saSample / saTexel);

        color += textureSampleLevel(srcCube, srcSamp, L, mip).rgb * NdotL;
        weight += NdotL;
    }
    textureStore(dstMip, gid.xy, ${storeLayer}, vec4<f32>(color / max(weight, 1e-4), 1.0));
}
`;
}

// ---------------------------------------------------------------------------
// Stage D — the BRDF LUT (§6.4). Scene-independent; baked once per device.
// ---------------------------------------------------------------------------

/**
 * The split-sum DFG integration over (NdotV, roughness) → (scale, bias), 256×256 `rg16float`.
 * Depends on nothing — not the environment, not the scene — so it is baked once; it is computed
 * rather than shipped so the package carries no binary asset. U7 pins the roughness-0 endpoint
 * (scale, bias) → (1, 0) and monotonicity in NdotV.
 */
export function buildBRDFLUTWGSL(opts: { samples?: number } = {}): string {
    const samples = opts.samples ?? 1024;
    return /* wgsl */ `
@group(0) @binding(0) var dstLUT: texture_storage_2d<rgba16float, write>;

${ENV_MATH_WGSL}
${ENV_SAMPLING_WGSL}

const SAMPLE_COUNT: u32 = ${samples}u;

fn geometrySchlickIBL(nDotV: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let k = (a * a) / 2.0;                       // IBL k, NOT the direct-lighting (r+1)²/8
    return nDotV / (nDotV * (1.0 - k) + k);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(dstLUT);
    if (gid.x >= dim.x || gid.y >= dim.y) { return; }
    let NdotV = (f32(gid.x) + 0.5) / f32(dim.x);
    let roughness = (f32(gid.y) + 0.5) / f32(dim.y);

    // Integration frame: N = (0,0,1), V in the xz-plane.
    let V = vec3<f32>(sqrt(max(0.0, 1.0 - NdotV * NdotV)), 0.0, NdotV);
    let N = vec3<f32>(0.0, 0.0, 1.0);

    var a = 0.0;
    var b = 0.0;
    for (var i = 0u; i < SAMPLE_COUNT; i = i + 1u) {
        let H = importanceSampleGGX(hammersley(i, SAMPLE_COUNT), N, roughness);
        let vh = max(dot(V, H), 0.0);
        let L = normalize(2.0 * vh * H - V);
        let nDotL = max(L.z, 0.0);
        let nDotH = max(H.z, 0.0);
        if (nDotL > 0.0) {
            let g = geometrySchlickIBL(NdotV, roughness) * geometrySchlickIBL(nDotL, roughness);
            let gVis = (g * vh) / (nDotH * NdotV);
            let fc = pow(1.0 - vh, 5.0);
            a += (1.0 - fc) * gVis;
            b += fc * gVis;
        }
    }
    textureStore(dstLUT, gid.xy, vec4<f32>(a / f32(SAMPLE_COUNT), b / f32(SAMPLE_COUNT), 0.0, 0.0));
}
`;
}

// ---------------------------------------------------------------------------
// SH projection (§7.2) — one workgroup over the source cube's 8×8 mip
// ---------------------------------------------------------------------------

/**
 * Project the source cube's 8×8 mip into SH-L2 irradiance, written into `EnvParams.shR/G/B`.
 * WGSL has no `atomic<f32>` (§3), so instead of a two-pass reduction this runs in ONE workgroup
 * of 64 threads over 6 faces × 64 texels = 384 samples, tree-reduced in workgroup memory.
 * The cosine-convolution constants (A0=π, A1=2π/3, A2=π/4) are folded in here, so the stored
 * value IS irradiance.
 */
export function buildSHProjectWGSL(): string {
    const S = SH_SOURCE_SIZE;
    return /* wgsl */ `
@group(0) @binding(0) var srcCube2DArray: texture_2d_array<f32>;
// 36 floats, NOT 27: this is written in the EXACT packed layout of EnvParams floats 0..35
// (shR[3] then shG[3] then shB[3], each a vec4 holding 3 coefficients — §7.3), so the result is
// moved into the uniform with one copyBufferToBuffer instead of a readback. envMath.packSHChannel
// is the CPU mirror of this indexing, and U4 pins the two together.
@group(0) @binding(1) var<storage, read_write> shOut: array<f32, 36>;

${ENV_MATH_WGSL}

const SH_MIP_SIZE: u32 = ${S}u;

var<workgroup> partialR: array<array<f32, 9>, 64>;
var<workgroup> partialG: array<array<f32, 9>, 64>;
var<workgroup> partialB: array<array<f32, 9>, 64>;

fn cosineWeight(k: u32) -> f32 {
    if (k == 0u) { return PI; }
    if (k < 4u) { return 2.0 * PI / 3.0; }
    return PI / 4.0;
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    for (var k = 0u; k < 9u; k = k + 1u) {
        partialR[t][k] = 0.0; partialG[t][k] = 0.0; partialB[t][k] = 0.0;
    }

    // 384 texels / 64 threads = 6 texels each — one per face, same (x,y).
    let xy = vec2<u32>(t % SH_MIP_SIZE, t / SH_MIP_SIZE);
    let uv = (vec2<f32>(xy) + 0.5) / f32(SH_MIP_SIZE);
    let dw = cubeTexelSolidAngle(uv, SH_MIP_SIZE);
    for (var face = 0u; face < 6u; face = face + 1u) {
        let dir = faceDirection(face, uv);
        let rgb = textureLoad(srcCube2DArray, xy, face, 0).rgb * dw;   // view IS the 8×8 mip
        let sh = shBasisL2(dir);
        for (var k = 0u; k < 9u; k = k + 1u) {
            let b = sh[k] * cosineWeight(k);
            partialR[t][k] += rgb.r * b;
            partialG[t][k] += rgb.g * b;
            partialB[t][k] += rgb.b * b;
        }
    }
    workgroupBarrier();

    // Tree reduction over 64 threads.
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (t < stride) {
            for (var k = 0u; k < 9u; k = k + 1u) {
                partialR[t][k] += partialR[t + stride][k];
                partialG[t][k] += partialG[t + stride][k];
                partialB[t][k] += partialB[t + stride][k];
            }
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        // Packed EnvParams layout: coefficient k of a channel lives at
        // channelBase + (k/3)*4 + (k%3) — the same index shCoeffR/G/B(k) reads back out.
        for (var k = 0u; k < 9u; k = k + 1u) {
            let slot = (k / 3u) * 4u + (k % 3u);
            shOut[0u  + slot] = partialR[0][k];
            shOut[12u + slot] = partialG[0][k];
            shOut[24u + slot] = partialB[0][k];
        }
    }
}
`;
}

// ---------------------------------------------------------------------------
// The indirect-lighting contract (§5) — the ShaderIndirectLayout Phase 1 emits
// ---------------------------------------------------------------------------

/**
 * The `wgslStructs` half of the Phase 1 `ShaderIndirectLayout`: the group-2 environment
 * bindings (3–6, alongside the lights at 0–2 — §4.4) plus the helpers the material fragments
 * call. Emitted into the same module as the material, so `PI` is NOT redeclared here (the PBR
 * chunk set owns it); the helpers use the literal.
 */
export function buildIndirectStructsWGSL(environmentGroup: number): string {
    const g = environmentGroup;
    return /* wgsl */ `
struct EnvParams {
    shR: array<vec4<f32>, 3>,   // SH-L2 red   — 9 coeffs packed 3×vec4 (§7.3)
    shG: array<vec4<f32>, 3>,   // green
    shB: array<vec4<f32>, 3>,   // blue
    rotation: vec4<f32>,        // .xyz reserved (cos/sin cache); .w = yaw radians
    intensity: f32,             // diffuse + specular multiplier
    specularMip: f32,           // prefilterMips - 1, as a float
    horizonFade: f32,           // occlusion of below-horizon specular; default 1.0
    _pad: f32,
};                              // ${ENV_PARAMS_FLOATS} floats = 176 bytes

@group(${g}) @binding(3) var envSampler: sampler;
@group(${g}) @binding(4) var envSpecular: texture_cube<f32>;
@group(${g}) @binding(5) var brdfLUT: texture_2d<f32>;
@group(${g}) @binding(6) var<uniform> env: EnvParams;

// Unpack coefficient k of the packed SH channel set (§7.3) — the order envMath.packSHChannel
// writes, pinned by U4.
fn shCoeffR(k: u32) -> f32 { return env.shR[k / 3u][k % 3u]; }
fn shCoeffG(k: u32) -> f32 { return env.shG[k / 3u][k % 3u]; }
fn shCoeffB(k: u32) -> f32 { return env.shB[k / 3u][k % 3u]; }

fn fresnelSchlickRoughness(cosTheta: f32, F0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return F0 + (max(vec3<f32>(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
`;
}

/**
 * The `wgslLogic` half: `getIndirectDiffuse` / `getIndirectSpecular`, the two functions every
 * implementation of the contract provides (§5.2). Both take `worldPos` even though Phase 1
 * ignores it — that parameter is what lets Phase 2 swap a probe volume in without touching a
 * single material.
 */
export function buildIndirectLogicWGSL(): string {
    return /* wgsl */ `
fn shBasisL2Env(d: vec3<f32>) -> array<f32, 9> {
    return array<f32, 9>(
        0.28209479,
        0.48860251 * d.y,
        0.48860251 * d.z,
        0.48860251 * d.x,
        1.09254843 * d.x * d.y,
        1.09254843 * d.y * d.z,
        0.31539157 * (3.0 * d.z * d.z - 1.0),
        1.09254843 * d.x * d.z,
        0.54627422 * (d.x * d.x - d.y * d.y),
    );
}

fn rotateYEnv(v: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(c * v.x - s * v.z, v.y, s * v.x + c * v.z);
}

/** Cosine-convolved irradiance arriving at 'worldPos' with normal 'N', divided by PI.
 *  Multiply by albedo (and kD) to get the diffuse indirect term. */
fn getIndirectDiffuse(worldPos: vec3<f32>, N: vec3<f32>) -> vec3<f32> {
    let _antiPrune = textureSampleLevel(brdfLUT, envSampler, vec2<f32>(0.0, 0.0), 0.0).r * 0.0000001;
    let _antiPruneCube = textureSampleLevel(envSpecular, envSampler, vec3<f32>(0.0, 1.0, 0.0), 0.0).r * 0.0000001;
    let n = rotateYEnv(N, -env.rotation.w);          // environment yaw
    let b = shBasisL2Env(n);
    var c = vec3<f32>(0.0);
    for (var k = 0u; k < 9u; k = k + 1u) {
        c += vec3<f32>(shCoeffR(k), shCoeffG(k), shCoeffB(k)) * b[k];
    }
    // Clamp: an L2 projection of a high-contrast environment rings, and the negative lobe reads
    // as black patches on surfaces facing away from the dominant light (§7.4, U5).
    return (max(c, vec3<f32>(0.0)) + vec3<f32>(_antiPrune + _antiPruneCube)) * env.intensity;
}

/** Pre-integrated specular radiance for the reflection lobe, already including the split-sum
 *  BRDF term — the caller multiplies by nothing but occlusion. */
fn getIndirectSpecular(worldPos: vec3<f32>, N: vec3<f32>, V: vec3<f32>,
                       roughness: f32, F0: vec3<f32>) -> vec3<f32> {
    // var, not let — the yaw rotation reassigns it, and WGSL let is immutable.
    var R = reflect(-V, N);
    R = rotateYEnv(R, -env.rotation.w);
    // Occlude reflections pointing under the horizon: a rough surface sampling below itself
    // would otherwise read the ground lobe of the cube at full strength (§4.3 horizonFade).
    let horizon = mix(1.0, smoothstep(-0.1, 0.1, R.y), env.horizonFade);
    let mip = roughness * env.specularMip;
    let prefiltered = textureSampleLevel(envSpecular, envSampler, R, mip).rgb * horizon;
    let dfg = textureSampleLevel(brdfLUT, envSampler, vec2<f32>(max(dot(N, V), 0.0), roughness), 0.0).rg;
    return prefiltered * (F0 * dfg.x + dfg.y) * env.intensity;
}
`;
}

// ---------------------------------------------------------------------------
// The skybox pass (§8)
// ---------------------------------------------------------------------------

/**
 * A fullscreen triangle drawn AFTER the opaque pass, reconstructing a view ray per pixel from
 * `camera.invViewProj` and sampling the prefiltered cube at `blurMip`. No vertex buffer —
 * `RenderPipeline` takes its `draw(3, instanceCount)` path. The batch's group-0 storage buffer
 * (bound unconditionally by `createBatch`) genuinely carries the sky params `[exposure, yaw,
 * blurMip, _]`, written through `updateBatchData` — so the binding is used, not an anti-prune
 * trick (§8.2).
 */
export function buildSkyWGSL(opts: { environmentGroup: number }): string {
    const g = opts.environmentGroup;
    return /* wgsl */ `
struct Camera {
    viewProj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    invViewProj: mat4x4<f32>,
    invView: mat4x4<f32>,
    invProj: mat4x4<f32>,
    eyePos: vec3<f32>,
    simTime: f32,
    zNear: f32,
    zFar: f32,
    fovRads: f32,
    aspect: f32,
    _padEnd: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;
// The batch storage buffer, honestly used (§8.2): [exposure, yaw, blurMip, _].
@group(0) @binding(1) var<storage, read> skyParams: array<f32>;

@group(${g}) @binding(3) var envSampler: sampler;
@group(${g}) @binding(4) var envSpecular: texture_cube<f32>;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var o: VSOut;
    o.pos = vec4<f32>(p[vi], 0.999999, 1.0);   // just inside the far plane, depth-tested less-equal
    o.ndc = p[vi];
    return o;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    // Reconstruct the world-space view ray through this pixel.
    let far = camera.invViewProj * vec4<f32>(in.ndc, 1.0, 1.0);
    let dir = normalize(far.xyz / far.w - camera.eyePos);

    let yaw = skyParams[1];
    let c = cos(yaw);
    let s = sin(yaw);
    let R = vec3<f32>(c * dir.x - s * dir.z, dir.y, s * dir.x + c * dir.z);

    let exposure = skyParams[0];
    let blurMip = skyParams[2];
    let color = textureSampleLevel(envSpecular, envSampler, R, blurMip).rgb * exposure;
    return vec4<f32>(color, 1.0);
}
`;
}
