// MatcapShaderBuilder.ts
// Material capture — full "lit" appearance from a single texture lookup keyed
// on the view-space normal. Zero lights, zero BRDF math. This is the classic
// trick for making huge instance counts look shaded at unlit cost.

import {CommonChunks, buildStandardVertexShader} from "./CommonChunks";
import {ShaderConfig} from "./types";

const MatcapBindings = `
    struct MatcapMaterialParams {
        baseColor: vec4<f32>,        // 16 bytes (offset 0)
        hasAlbedoTexture: f32,       // 4 bytes  (offset 16)
        // Scalars, NOT vec3 — vec3 has 16-byte alignment in WGSL and would
        // push the struct size to 48 bytes, breaking the 32-byte buffer.
        _pad0: f32,                  // 4 bytes  (offset 20)
        _pad1: f32,                  // 4 bytes  (offset 24)
        _pad2: f32,                  // 4 bytes  (offset 28)
    };
    // Total: 32 bytes

    // Group 1: Material Data
    @group(1) @binding(0) var samp: sampler;
    @group(1) @binding(1) var texMatcap: texture_2d<f32>;
    @group(1) @binding(2) var texAlbedo: texture_2d<f32>;
    @group(1) @binding(3) var<uniform> material: MatcapMaterialParams;
`;

export function buildMatcapShader(config: ShaderConfig): string {

    const fragmentShader = `
    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
        let instanceColor = vec4<f32>(in.ecsColor, 1.0);

        // Project the normal into view space and remap to [0,1] UVs.
        // The matcap sphere texture encodes the entire lighting environment.
        let N = normalize(in.worldNormal);
        let viewNormal = normalize((camera.view * vec4<f32>(N, 0.0)).xyz);
        let matcapUV = vec2<f32>(viewNormal.x, -viewNormal.y) * 0.5 + 0.5;

        var color = textureSample(texMatcap, samp, matcapUV) * material.baseColor * instanceColor;
        if (material.hasAlbedoTexture > 0.5) {
            color = textureSample(texAlbedo, samp, in.uv) * color;
        }

        return color;
    }
    `;

    return `
        ${CommonChunks.CameraStruct}
        ${MatcapBindings}
        ${CommonChunks.EngineBindings}
        ${CommonChunks.VertexOut}
        ${CommonChunks.QuatMath}
        ${config.useSkinning ? CommonChunks.SkinningBindings : ''}

        ${buildStandardVertexShader(config)}
        ${fragmentShader}
    `;
}
