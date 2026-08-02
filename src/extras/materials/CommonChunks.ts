// CommonChunks.ts
// Shared WGSL chunks for the lightweight material families (Basic, Emissive, Lambert, Matcap).
// Group 0 is always engine data (camera + raw ECS storage buffer) — zero scene graph,
// the vertex shader reads instance transforms straight out of the flat array.

import {ShaderConfig} from "./types";

export const CommonChunks = {
    // Camera uniform — must match the engine's camera buffer layout exactly
    CameraStruct: `
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
    `,

    // Group 0: Engine Data
    EngineBindings: `
        @group(0) @binding(0) var<uniform> camera: Camera;
        @group(0) @binding(1) var<storage, read> ecs: array<f32>;
    `,

    VertexOut: `
        struct VertexOut {
            @builtin(position) clipPos: vec4<f32>,
            @location(0) worldPos: vec3<f32>,
            @location(1) worldNormal: vec3<f32>,
            @location(2) uv: vec2<f32>,
            @location(3) ecsColor: vec3<f32>,
        }
    `,

    QuatMath: `
        // Quaternion rotation utility (needed for vertex shader)
        fn rotateVector(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
            let t = 2.0 * cross(q.xyz, v);
            return v + q.w * t + cross(q.xyz, t);
        }
    `,

    // Only included if mesh has bones
    SkinningBindings: `
        struct Skeleton { boneMatrices: array<mat4x4<f32>, 70> };
        @group(2) @binding(0) var<uniform> skeleton: Skeleton;
    `,
};

/**
 * The standard forward vertex shader shared by all lightweight materials.
 * Reads position/rotation/scale/color directly from the flat ECS storage buffer
 * (14 floats per instance) and optionally applies GPU skinning.
 */
export function buildStandardVertexShader(config: ShaderConfig): string {
    return `
    @vertex
    fn vs_main(
        @location(0) localPos: vec3<f32>,
        @location(1) normal: vec3<f32>,
        @location(2) uv: vec2<f32>,
        ${config.useSkinning ? `
        @location(3) joints: vec4<f32>,
        @location(4) weights: vec4<f32>,` : ''}
        @builtin(instance_index) iIdx: u32
    ) -> VertexOut {
        let base = iIdx * 14u;
        let ecsPos = vec3<f32>(ecs[base + 1u], ecs[base + 2u], ecs[base + 3u]);
        let ecsRot = vec4<f32>(ecs[base + 4u], ecs[base + 5u], ecs[base + 6u], ecs[base + 7u]);
        let ecsScale = vec3<f32>(ecs[base + 8u], ecs[base + 9u], ecs[base + 10u]);
        // Extract ECS Color
        let ecsColor = vec3<f32>(ecs[base + 11u], ecs[base + 12u], ecs[base + 13u]);

        ${config.useSkinning ? `
        // --- SKINNED TRANSFORM ---
        let jx = u32(joints.x); let jy = u32(joints.y);
        let jz = u32(joints.z); let jw = u32(joints.w);

        let skinMatrix =
            weights.x * skeleton.boneMatrices[jx] +
            weights.y * skeleton.boneMatrices[jy] +
            weights.z * skeleton.boneMatrices[jz] +
            weights.w * skeleton.boneMatrices[jw];

        let finalLocalPos = (skinMatrix * vec4<f32>(localPos, 1.0)).xyz;
        let finalNormal = normalize((skinMatrix * vec4<f32>(normal, 0.0)).xyz);
        ` : `
        // --- STATIC TRANSFORM ---
        let finalLocalPos = localPos;
        let finalNormal = normal;
        `}

        let scaledPos = finalLocalPos * ecsScale;
        let rotatedPos = rotateVector(scaledPos, ecsRot);
        let worldPosition = rotatedPos + ecsPos;
        let finalWorldNormal = rotateVector(finalNormal, ecsRot);

        var out: VertexOut;
        out.clipPos = camera.viewProj * vec4<f32>(worldPosition, 1.0);
        out.worldPos = worldPosition;
        out.worldNormal = finalWorldNormal;
        out.uv = uv;
        out.ecsColor = ecsColor; // Pass to fragment shader
        return out;
    }
    `;
}
