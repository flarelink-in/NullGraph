// src/extras/shadows/atlasShaders.ts
//
// WGSL generators for the Phase 2 shadow atlas (docs/shadows-atlas-design.md §4, §9). These
// replace Phase 1's fixed cascade-array sampling with a single atlas texture plus a runtime
// view/slot table, so any mix of directional/spot/point lights can cast without recompiling the
// receiver shader (the tables are `storage`, not `uniform` — §9.5).
//
// The receiver chunks are emitted into the SAME module as the light bindings by the decorator
// (§9.2), exactly as Phase 1's shadowShaders.ts did — only the middle changes: the direction
// match becomes a slot lookup, and the view choice becomes type-dependent.

import { CAMERA_STRUCT_WGSL, INSTANCE_STRUCT_WGSL } from '../visibility/visibilityShaders';
import { MAX_CASCADES } from './shadowMath';

// ---------------------------------------------------------------------------
// §4.1 / §4.2 — the receiver storage tables
// ---------------------------------------------------------------------------

/**
 * `ShadowView` / `ShadowSlot` — must match atlasMath.ts's packShadowView / packShadowSlot
 * float-for-float. `ShadowView` is what the receiver needs of a view; the CPU side additionally
 * keeps the full 432-byte Camera for the cull + depth draw (§4.1).
 */
export const ATLAS_STRUCTS_WGSL = /* wgsl */ `
struct ShadowView {
    viewProj   : mat4x4<f32>,   // 64 B  world → this view's clip
    rect       : vec4<f32>,     // 16 B  atlas UV transform: .xy = uv scale, .zw = uv offset
    texelWorld : f32,           //       world size of one texel at the view's reference depth
    splitFar   : f32,           //       cascades: view-space depth covered; punctual: light range
    kind       : u32,           //       0 = cascade, 1 = spot, 2 = cube face
    _pad       : u32,           // 16 B
};                              // 96 B, 16-aligned

struct ShadowSlot {
    firstView  : u32,           // index into the views array
    viewCount  : u32,           // cascades: N; spot: 1; point: 6
    kind       : u32,           // 0 = directional, 1 = spot, 2 = point
    lightRange : f32,           // punctual only: for the depth remap at sample time
};
`;

/**
 * The receiver uniform (§9.5). Smaller than Phase 1's `ShadowData`: the per-cascade matrices and
 * splits moved into the `shadowViews` storage table, leaving only the global filter/bias knobs
 * and the cascade split distances (for `selectCascade`).
 */
export const ATLAS_PARAMS_WGSL = /* wgsl */ `
struct ShadowParams {
    atlasSize     : f32,          // atlas edge in px — texel = filterTexels / atlasSize
    pcfRadius     : u32,          // taps per side: 1 → 3×3, 2 → 5×5, 0 → hardware 2×2 only
    constantBias  : f32,
    filterTexels  : f32,          // PCF kernel width in texels (drives the tap step)
    cascadeSplits : vec4<f32>,    // view-space split far distances, up to ${MAX_CASCADES}
    cascadeCount  : u32,
    blendTexels   : f32,          // cascade cross-fade band in distance units
    _pad0         : f32,
    _pad1         : f32,
};
`;

/**
 * The shadow group bindings (§9.5). Group 3 by default, as Phase 1. The two tables are `storage`
 * so view/slot counts are runtime values and the shader is not recompiled when a light starts
 * casting.
 */
export function buildAtlasBindingsWGSL(group: number): string {
    return /* wgsl */ `
@group(${group}) @binding(0) var shadowAtlas   : texture_depth_2d;
@group(${group}) @binding(1) var shadowSampler : sampler_comparison;
@group(${group}) @binding(2) var<uniform> shadow : ShadowParams;
@group(${group}) @binding(3) var<storage, read> shadowViews : array<ShadowView>;
@group(${group}) @binding(4) var<storage, read> shadowSlots : array<ShadowSlot>;
`;
}

// ---------------------------------------------------------------------------
// §9.2 — view selection + the atlas tap
// ---------------------------------------------------------------------------

/**
 * `selectCascade` / `cubeFace` / `sampleView`. `textureSampleCompareLevel` throughout — never the
 * bare `textureSampleCompare`, whose implicit derivatives are illegal inside the branch and PCF
 * loop (Phase 1 §8.3, pinned by U7).
 *
 * `cubeFace` is the WGSL twin of atlasMath.ts's `cubeFace` — U2 pins them to agree on the axes,
 * the diagonals, and 1000 random directions.
 */
export const ATLAS_SAMPLING_WGSL = /* wgsl */ `
// Pick the cascade whose split contains this view-space depth (directional lights only).
fn selectCascade(viewDepth: f32) -> u32 {
    for (var k = 0u; k < shadow.cascadeCount; k = k + 1u) {
        if (viewDepth < shadow.cascadeSplits[k]) { return k; }
    }
    return shadow.cascadeCount - 1u;
}

// Major-axis face select, matching the +X,-X,+Y,-Y,+Z,-Z build order (§9.2).
fn cubeFace(d: vec3<f32>) -> u32 {
    let a = abs(d);
    if (a.x >= a.y && a.x >= a.z) { return select(1u, 0u, d.x > 0.0); }
    if (a.y >= a.z)               { return select(3u, 2u, d.y > 0.0); }
    return select(5u, 4u, d.z > 0.0);
}

// The atlas tap (§9.3). Returns 1.0 (fully lit) for anything outside the view — behind a
// punctual view, past the far plane, or off the tile — so an evicted or out-of-range light
// fails open, never black (§5.2).
fn sampleView(vi: u32, worldPos: vec3<f32>) -> f32 {
    let v    = shadowViews[vi];
    let clip = v.viewProj * vec4<f32>(worldPos, 1.0);
    if (clip.w <= 0.0) { return 1.0; }                              // behind a punctual view
    let ndc = clip.xyz / clip.w;
    if (ndc.z < 0.0 || ndc.z > 1.0) { return 1.0; }                 // outside orthographic/perspective depth bounds
    var uv = vec2<f32>(ndc.x, -ndc.y) * 0.5 + vec2<f32>(0.5);       // clip +Y up → texture +V down
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) { return 1.0; }

    let texel    = shadow.filterTexels / shadow.atlasSize;             // in ATLAS uv units
    let depthRef = ndc.z - shadow.constantBias;
    let r        = i32(shadow.pcfRadius);

    var sum = 0.0;
    var n   = 0.0;
    for (var y = -r; y <= r; y = y + 1) {
        for (var x = -r; x <= r; x = x + 1) {
            // Offset in the VIEW's uv, then map into the atlas, then clamp INSIDE the tile so a
            // tap can never read a neighbouring light's depth (§9.4, pinned by U3).
            let local  = clamp(uv + vec2<f32>(f32(x), f32(y)) * (texel / v.rect.x),
                               vec2<f32>(0.0), vec2<f32>(1.0));
            let atlasUV = local * v.rect.xy + v.rect.zw;
            sum = sum + textureSampleCompareLevel(shadowAtlas, shadowSampler, atlasUV, depthRef);
            n   = n + 1.0;
        }
    }
    return sum / n;
}
`;

// ---------------------------------------------------------------------------
// §9.2 — the decorator: slot → view → tap
// ---------------------------------------------------------------------------

/**
 * The shadowed `getIncidentLight` replacement (§9.2). `accessor` reads the light record for loop
 * index `i` — `lights[i]` for the standard technique, the cluster indirection for
 * clustered-forward. The Phase 1 direction match is deleted; the shadow slot rides in
 * `spot_params.z` as `slot + 1`, with `0` meaning "this light does not cast" (§4.3).
 */
export function buildAtlasDecoratorWGSL(accessor: string): string {
    return /* wgsl */ `
fn getIncidentLight(i: u32, worldPos: vec3<f32>) -> IncidentLight {
    var inc = getIncidentLightUnshadowed(i, worldPos);
    let rec = ${accessor};                          // lights[i] | globalLights[lightIndices[...]]
    let packed = u32(rec.spot_params.z);            // 0 = no shadow, slot+1 otherwise
    if (packed == 0u) { return inc; }

    let slot = shadowSlots[packed - 1u];
    if (slot.kind == 0u) {                          // directional: pick cascade with smooth cross-fade
        let viewDepth = -(camera.view * vec4<f32>(worldPos, 1.0)).z;
        if (viewDepth >= shadow.cascadeSplits[shadow.cascadeCount - 1u]) {
            return inc;                             // Past shadow distance -> fully lit
        }
        let c = selectCascade(viewDepth);
        var shadowAtten = sampleView(slot.firstView + c, worldPos);

        // Smooth cascade cross-fading to eliminate split seam lines
        let edge = shadow.cascadeSplits[c] - shadow.blendTexels;
        if (c + 1u < shadow.cascadeCount && viewDepth > edge) {
            let nextAtten = sampleView(slot.firstView + c + 1u, worldPos);
            let t = clamp((viewDepth - edge) / shadow.blendTexels, 0.0, 1.0);
            shadowAtten = mix(shadowAtten, nextAtten, t);
        }
        inc.attenuation = inc.attenuation * shadowAtten;
        return inc;
    }

    var v = slot.firstView;
    if (slot.kind == 2u) {                          // point: pick the cube face
        v = v + cubeFace(worldPos - rec.pos_type.xyz);
    }                                               // spot: the single view
    inc.attenuation = inc.attenuation * sampleView(v, worldPos);
    return inc;
}
`;
}

// ---------------------------------------------------------------------------
// §8.2 — the per-tile clear
// ---------------------------------------------------------------------------

/**
 * One full-viewport triangle at z = 1.0, no vertex buffer (§8.2). Caching needs
 * `depthLoadOp: 'load'` on the atlas, so each dirty tile clears itself with this batch, drawn
 * first, scoped to the tile's viewport. `depthCompare: 'always'` overwrites whatever the tile
 * held.
 */
export const TILE_CLEAR_WGSL = /* wgsl */ `
${CAMERA_STRUCT_WGSL}

@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(1) var<storage, read> instances : array<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // A single full-screen triangle: (-1,-1) (3,-1) (-1,3), at the far plane (z = 1).
    _ = camera.simTime;
    _ = instances[0];
    var pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(pos[vi], 1.0, 1.0);
}
`;
