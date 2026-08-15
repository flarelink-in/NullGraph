// src/extras/shadows/shadowShaders.ts
//
// WGSL generators for the ShadowSystem (docs/shadows-design.md §6–§8). Every module is a
// complete, self-contained unit so it can be parse-validated offline and so the engine's
// `layout: 'auto'` pipeline creation sees exactly the bindings the body uses.
//
// The receiver chunks (SHADOW_STRUCTS_WGSL / SHADOW_BINDINGS_WGSL / SHADOW_SAMPLING_WGSL) are
// emitted into the SAME module as the light bindings by the decorator (§8.1), so the shadowed
// `getIncidentLight` can read the light record directly and decide whether that light casts.

import { CAMERA_STRUCT_WGSL, INSTANCE_STRUCT_WGSL, CULL_CORE_WGSL } from '../visibility/visibilityShaders';
import { MAX_CASCADES } from './shadowMath';

// ---------------------------------------------------------------------------
// §4.2 — the receiver uniform + bindings
// ---------------------------------------------------------------------------

/** `ShadowCascade` / `ShadowData` — must match shadowMath.ts packShadowData() float-for-float. */
export const SHADOW_STRUCTS_WGSL = /* wgsl */ `
struct ShadowCascade {
    viewProj   : mat4x4<f32>,   // 64 B  world → cascade clip
    splitFar   : f32,           //       view-space depth this cascade covers up to
    texelWorld : f32,           //       world size of one shadow texel (bias + filter radius)
    _pad0      : f32,
    _pad1      : f32,           // 16 B
};                              // 80 B, 16-aligned

struct ShadowData {
    cascades      : array<ShadowCascade, ${MAX_CASCADES}>,
    lightDir      : vec4<f32>,   // .xyz normalized light travel direction, .w unused
    cascadeCount  : u32,
    pcfRadius     : u32,         // taps per side: 1 → 3×3, 2 → 5×5, 0 → hardware 2×2 only
    blendTexels   : f32,         // cascade cross-fade band, in split-distance units
    constantBias  : f32,
};
`;

/**
 * The shadow group bindings (§8.4). Group 3 by default: group 0 is engine-fixed, group 1 is the
 * material on the instance tier, group 2 is hardcoded into LightSystem's WGSL.
 */
export function buildShadowBindingsWGSL(group: number): string {
    return /* wgsl */ `
@group(${group}) @binding(0) var shadowMaps    : texture_depth_2d_array;
@group(${group}) @binding(1) var shadowSampler : sampler_comparison;
@group(${group}) @binding(2) var<uniform> shadow : ShadowData;
`;
}

// ---------------------------------------------------------------------------
// §8.3 — cascade selection, filtering, and the fade band
// ---------------------------------------------------------------------------

/**
 * The sampling functions. `textureSampleCompareLevel` — never the bare `textureSampleCompare`:
 * the latter takes implicit derivatives and is only valid in uniform control flow, and every
 * call here sits inside a cascade branch and a PCF loop (§8.3, pinned by U7).
 */
export const SHADOW_SAMPLING_WGSL = /* wgsl */ `
fn sampleCascade(k: u32, worldPos: vec3<f32>) -> f32 {
    let clip = shadow.cascades[k].viewProj * vec4<f32>(worldPos, 1.0);
    let ndc  = clip.xyz / clip.w;
    if (ndc.z > 1.0) { return 1.0; }                                   // beyond the cascade far plane
    let uv   = vec2<f32>(ndc.x, -ndc.y) * 0.5 + vec2<f32>(0.5);        // clip +Y up → texture +V down
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) { return 1.0; }

    let depthRef = ndc.z - shadow.constantBias;
    let texel = 1.0 / f32(textureDimensions(shadowMaps).x);
    let r     = i32(shadow.pcfRadius);
    if (r == 0) { return textureSampleCompareLevel(shadowMaps, shadowSampler, uv, k, depthRef); }

    var sum = 0.0;
    var n   = 0.0;
    for (var y = -r; y <= r; y = y + 1) {
        for (var x = -r; x <= r; x = x + 1) {
            let o = vec2<f32>(f32(x), f32(y)) * texel;
            sum = sum + textureSampleCompareLevel(shadowMaps, shadowSampler, uv + o, k, depthRef);
            n   = n + 1.0;
        }
    }
    return sum / n;
}

fn sampleShadow(worldPos: vec3<f32>) -> f32 {
    let viewDepth = -(camera.view * vec4<f32>(worldPos, 1.0)).z;

    var c = shadow.cascadeCount;                       // sentinel: beyond the last split
    for (var k = 0u; k < shadow.cascadeCount; k = k + 1u) {
        if (viewDepth < shadow.cascades[k].splitFar) { c = k; break; }
    }
    if (c >= shadow.cascadeCount) { return 1.0; }      // past shadowDistance → fully lit

    let lit = sampleCascade(c, worldPos);

    // Cross-fade into the next cascade over the last blendTexels of this one, so the
    // resolution change is a gradient rather than a visible seam.
    let edge = shadow.cascades[c].splitFar - shadow.blendTexels;
    if (c + 1u < shadow.cascadeCount && viewDepth > edge) {
        let t = (viewDepth - edge) / shadow.blendTexels;
        return mix(lit, sampleCascade(c + 1u, worldPos), clamp(t, 0.0, 1.0));
    }
    return lit;
}
`;

/**
 * Receiver-side normal offset for hand-written material fragments that have `N` in scope
 * (§7.3). The stock material bridge cannot use this (§8.2 — the lighting contract has no
 * normal, and recovering one with dpdx/dpdy inside the non-uniform light loop is illegal).
 */
export const SHADOW_BIASED_WGSL = /* wgsl */ `
fn shadowFactorBiased(worldPos: vec3<f32>, N: vec3<f32>, L: vec3<f32>) -> f32 {
    let viewDepth = -(camera.view * vec4<f32>(worldPos, 1.0)).z;
    var k = 0u;
    for (var i = 0u; i < shadow.cascadeCount; i = i + 1u) {
        if (viewDepth < shadow.cascades[i].splitFar) { k = i; break; }
    }
    let offset = N * (shadow.cascades[k].texelWorld * 1.5);
    return sampleShadow(worldPos + offset);
}
`;

// ---------------------------------------------------------------------------
// §8.1 — the decorator
// ---------------------------------------------------------------------------

/**
 * The shadowed `getIncidentLight` replacement. `accessor` reads the light record for loop index
 * `i` so the decorator can decide whether THAT light casts — `lights[i]` for the standard
 * technique, the cluster indirection for clustered-forward. Phase 1 shadows exactly one
 * directional light, identified by direction match (two directional lights pointing the same
 * way ARE the same light); Phase 2 replaces the dot product with an explicit per-light shadow
 * slot carried in the light record's free `spot_params.z`.
 */
export function buildShadowDecoratorWGSL(accessor: string): string {
    return /* wgsl */ `
fn getIncidentLight(i: u32, worldPos: vec3<f32>) -> IncidentLight {
    var inc = getIncidentLightUnshadowed(i, worldPos);
    let rec = ${accessor};
    if (rec.pos_type.w == 2.0 && dot(normalize(rec.dir_rad.xyz), shadow.lightDir.xyz) > 0.999) {
        inc.attenuation = inc.attenuation * sampleShadow(worldPos);
    }
    return inc;
}
`;
}

// ---------------------------------------------------------------------------
// §6.2 + §7 — the depth-only caster pass
// ---------------------------------------------------------------------------

export interface ShadowCasterShaderOptions {
    /** Bind group holding the ShadowData uniform. Default 3. */
    shadowGroupIndex?: number;
    /**
     * §6.3 zero-change fallback: emit a one-line fragment writing a constant to a throwaway
     * r8unorm target, for engines without `PipelineConfig.depthOnly`. ~1.5–2× pass cost.
     */
    colorFallback?: boolean;
}

/**
 * The depth-only caster shader. Vertex stage only (unless `colorFallback`): all bias lives here
 * (§7), because `PipelineConfig` exposes neither `depthBias` nor `cullMode` and the receiver
 * side has no normal in scope (§8.2).
 *
 * Group 0 is engine-fixed (camera@0 — the CASCADE's camera via `PipelineConfig.cameraBuffer`,
 * and the culled instances@1, which the vertex stage genuinely reads). Bias constants come from
 * a per-cascade `ShadowCasterParams` uniform at the shadow group binding 3 — one 16-byte buffer
 * per cascade, written by `ShadowSystem.update()`, so the caster's texelWorld is EXACT for the
 * cascade being rendered (a shared ShadowData cannot do this: the cascade index is a per-pass
 * property, and a uniform-array index would be a per-vertex one).
 */
export function buildShadowCasterWGSL(opts: ShadowCasterShaderOptions = {}): string {
    const group = opts.shadowGroupIndex ?? 3;
    return /* wgsl */ `
${CAMERA_STRUCT_WGSL}
${INSTANCE_STRUCT_WGSL}
${CULL_CORE_WGSL}

struct ShadowCasterParams {
    texelWorld         : f32,   // world size of one texel of THIS cascade
    normalOffsetTexels : f32,   // §7.2 — normal offset in texels
    constantBias       : f32,   // §7.1 — NDC-z bias
    _pad               : f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(${group}) @binding(3) var<uniform> casterParams: ShadowCasterParams;

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(instance_index) ii: u32,
) -> @builtin(position) vec4<f32> {
    let inst = instances[ii];
    let n = normalize(normalMatrixFromModel(inst.model) * normal);
    let world = (inst.model * vec4<f32>(pos, 1.0)).xyz;

    // §7.2 — normal offset on the caster, scaled by THIS cascade's world texel size so the bias
    // is resolution-correct: raise the shadow map to 4096 and acne does not return with a
    // different constant.
    let biasedWorld = world + n * (casterParams.texelWorld * casterParams.normalOffsetTexels);

    var clipPos = camera.viewProj * vec4<f32>(biasedWorld, 1.0);
    // §7.1 — constant bias in clip space, scaled by w so it is a plain NDC-z offset under the
    // orthographic cascade (w == 1). Pushes casters away from the light.
    clipPos.z = clipPos.z + casterParams.constantBias * clipPos.w;
    return clipPos;
}

${opts.colorFallback ? /* wgsl */ `
// §6.3 — the throwaway color target. Correct, and ships without PipelineConfig.depthOnly, at
// resolution² × cascades bytes of pointless bandwidth plus a full-rate fragment stage.
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
` : ''}
`;
}
