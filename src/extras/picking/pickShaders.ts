// src/extras/picking/pickShaders.ts
//
// Pick shaders (docs/picking-design.md §4, §6). Each writes one rgba32uint texel per fragment:
// (sourceTag, instanceIndex, bitcast<u32>(fragCoord.z), extra). The tag is baked into the module —
// every source has its own pipeline anyway — so the fragment stage reads no uniform.
//
// Explanations live in these TS comments rather than inside the WGSL template literals: text inside
// a template ships in the bundle, comments outside it do not.

import { CAMERA_STRUCT_WGSL, INSTANCE_STRUCT_WGSL } from '../visibility/visibilityShaders';

/**
 * The varyings every pick vertex stage returns, and the fragment that turns them into a texel.
 * `index` and `extra` are flat — integers must not be interpolated. Custom vertex shaders
 * (`Picker.addBatch(batch, { shaderCode })`) return `PickVaryings` and append this.
 */
export function buildPickFragmentWGSL(tag: number): string {
    if (!Number.isInteger(tag) || tag < 1) throw new Error(`pick shader: source tag ${tag} must be a positive integer.`);
    return /* wgsl */ `
struct PickVaryings {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) index: u32,
    @location(1) @interpolate(flat) extra: u32,
};

@fragment
fn fs_main(v: PickVaryings) -> @location(0) vec4<u32> {
    return vec4<u32>(${tag}u, v.index, bitcast<u32>(v.position.z), v.extra);
}
`;
}

/**
 * Instance tier (VisibilitySystem sources). The batch's storage buffer holds the COMPACTED visible
 * instances; `sourceIndex` is the app's own index into its InstanceBuffer (picking-design.md §5),
 * which is what a pick returns. `extra` carries the meshIndex.
 */
export function buildInstancePickWGSL(tag: number): string {
    return /* wgsl */ `
${CAMERA_STRUCT_WGSL}
${INSTANCE_STRUCT_WGSL}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;

${buildPickFragmentWGSL(tag)}

@vertex
fn vs_main(@location(0) pos: vec3<f32>, @builtin(instance_index) ii: u32) -> PickVaryings {
    let inst = instances[ii];
    var v: PickVaryings;
    v.position = camera.viewProj * (inst.model * vec4<f32>(pos, 1.0));
    v.index = inst.sourceIndex;
    v.extra = inst.meshIndex;
    return v;
}
`;
}

/**
 * Plain batches on the stock 14-float layout every lightweight and PBR material reads
 * (materials/CommonChunks.ts buildStandardVertexShader): `[_, pos.xyz, rot.xyzw, scale.xyz,
 * color.rgb]`. The transform must stay bit-for-bit the material's — scale, then rotate, then
 * translate — or the pick silhouette drifts off the drawn one. Index = the batch instance index.
 */
export function buildStandardPickWGSL(tag: number): string {
    return /* wgsl */ `
${CAMERA_STRUCT_WGSL}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> ecs: array<f32>;

fn rotateVector(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

${buildPickFragmentWGSL(tag)}

@vertex
fn vs_main(@location(0) localPos: vec3<f32>, @builtin(instance_index) ii: u32) -> PickVaryings {
    let base = ii * 14u;
    let p = vec3<f32>(ecs[base + 1u], ecs[base + 2u], ecs[base + 3u]);
    let q = vec4<f32>(ecs[base + 4u], ecs[base + 5u], ecs[base + 6u], ecs[base + 7u]);
    let s = vec3<f32>(ecs[base + 8u], ecs[base + 9u], ecs[base + 10u]);
    let world = rotateVector(localPos * s, q) + p;
    var v: PickVaryings;
    v.position = camera.viewProj * vec4<f32>(world, 1.0);
    v.index = ii;
    v.extra = 0u;
    return v;
}
`;
}
