// PBRShaderChunks.ts

import { MaterialBindings, PBR_MATH_WGSL } from './materialFragments';

export const PBRChunks = {
    // 1. Common Structs & Bindings (Always included).
    //    The MATERIAL half comes from materialFragments.ts — the single source of truth shared
    //    with `null-graph/visibility`, which reuses the same bindings at a different group index.
    CommonBindings: `
        const PI: f32 = 3.14159265359;

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
    _padEnd: vec4<f32>
};

        // Group 0: Engine Data
        @group(0) @binding(0) var<uniform> camera: Camera;
        @group(0) @binding(1) var<storage, read> ecs: array<f32>;

        // Group 1: Material Data
        ${MaterialBindings.pbr(1)}

        struct VertexOut {
            @builtin(position) clipPos: vec4<f32>,
            @location(0) worldPos: vec3<f32>,
            @location(1) worldNormal: vec3<f32>,
            @location(2) uv: vec2<f32>,
        };
    `,

    // 2. Skinning Bindings (Only included if mesh has bones)
    SkinningBindings: `
        struct Skeleton { boneMatrices: array<mat4x4<f32>, 70> };
        @group(2) @binding(0) var<uniform> skeleton: Skeleton;
    `,

    // 3. The complex PBR Math (Always included). Shared verbatim with the visibility path;
    //    `rotateVector` is appended here because only this path's ECS vertex shader needs it.
    PBRMath: `
        ${PBR_MATH_WGSL}

        fn rotateVector(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
            let t = 2.0 * cross(q.xyz, v);
            return v + q.w * t + cross(q.xyz, t);
        }
    `
};