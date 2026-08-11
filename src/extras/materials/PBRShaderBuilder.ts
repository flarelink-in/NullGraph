// PBRShaderBuilder.ts
import { PBRChunks } from './PBRShaderChunks';
import { MaterialFragments } from './materialFragments';
import {ShaderConfig} from "./types";

export function buildPBRShader(config: ShaderConfig): string {

    // --- 1. Assemble the Vertex Shader dynamically ---
    const vertexShader = `
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
        return out;
    }
    `;

    // --- 2. Fragment stage — shared with null-graph/visibility ---
    const fragmentShader = MaterialFragments.pbr(config);

    // --- 3. Glue it all together ---
    // Notice how we safely inject the dynamic WGSL structs and logic
    // right before the fragment shader so they are available in scope.
    return `
        ${PBRChunks.CommonBindings}
        ${config.useSkinning ? PBRChunks.SkinningBindings : ''}
        ${PBRChunks.PBRMath}

        ${config.lighting ? config.lighting.wgslStructs : ''}
        ${config.lighting ? config.lighting.wgslLogic : ''}

        ${vertexShader}
        ${fragmentShader}
    `;
}