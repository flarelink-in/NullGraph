// BasicShaderBuilder.ts
// Unlit material — the cheapest possible fragment path. No lights, no normals,
// just baseColor * ECS instance color * (optional) albedo texture.

import {CommonChunks, buildStandardVertexShader} from "./CommonChunks";
import {ShaderConfig} from "./types";

const BasicBindings = `
    struct BasicMaterialParams {
        baseColor: vec4<f32>,        // 16 bytes (offset 0)
        uvTiling: vec2<f32>,         // 8 bytes  (offset 16)
        hasAlbedoTexture: f32,       // 4 bytes  (offset 24)
        _pad: f32,                   // 4 bytes  (offset 28)
    };
    // Total: 32 bytes

    // Group 1: Material Data
    @group(1) @binding(0) var samp: sampler;
    @group(1) @binding(1) var texAlbedo: texture_2d<f32>;
    @group(1) @binding(2) var<uniform> material: BasicMaterialParams;
`;

export function buildBasicShader(config: ShaderConfig): string {

    const fragmentShader = `
    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
        let instanceColor = vec4<f32>(in.ecsColor, 1.0);

        var color = material.baseColor * instanceColor;
        if (material.hasAlbedoTexture > 0.5) {
            color = textureSample(texAlbedo, samp, in.uv * material.uvTiling) * color;
        }

        return color;
    }
    `;

    return `
        ${CommonChunks.CameraStruct}
        ${BasicBindings}
        ${CommonChunks.EngineBindings}
        ${CommonChunks.VertexOut}
        ${CommonChunks.QuatMath}
        ${config.useSkinning ? CommonChunks.SkinningBindings : ''}

        ${buildStandardVertexShader(config)}
        ${fragmentShader}
    `;
}
