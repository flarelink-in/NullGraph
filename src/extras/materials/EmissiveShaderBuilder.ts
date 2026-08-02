// EmissiveShaderBuilder.ts
// Unlit + self-illumination. Emissive output can exceed 1.0 (HDR) so it feeds
// nicely into a Bloom offscreen pass. Optional pulse animation is driven by
// camera.simTime — zero CPU cost, zero per-frame uniform writes.

import {CommonChunks, buildStandardVertexShader} from "./CommonChunks";
import {ShaderConfig} from "./types";

const EmissiveBindings = `
    struct EmissiveMaterialParams {
        baseColor: vec4<f32>,        // 16 bytes (offset 0)
        emissiveColor: vec3<f32>,    // 12 bytes (offset 16)
        emissiveIntensity: f32, // 4 bytes  (offset 28)
        pulseSpeed: f32,             // 4 bytes  (offset 32)
        pulseAmount: f32,            // 4 bytes  (offset 36)
        hasAlbedoTexture: f32,       // 4 bytes  (offset 40)
        hasEmissiveMap: f32,         // 4 bytes  (offset 44)
    };
    // Total: 48 bytes

    // Group 1: Material Data
    @group(1) @binding(0) var samp: sampler;
    @group(1) @binding(1) var texAlbedo: texture_2d<f32>;
    @group(1) @binding(2) var texEmissive: texture_2d<f32>;
    @group(1) @binding(3) var<uniform> material: EmissiveMaterialParams;
`;

export function buildEmissiveShader(config: ShaderConfig): string {

    const fragmentShader = `
    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
        let instanceColor = vec4<f32>(in.ecsColor, 1.0);

        var baseColor = material.baseColor * instanceColor;
        if (material.hasAlbedoTexture > 0.5) {
            baseColor = textureSample(texAlbedo, samp, in.uv) * baseColor;
        }

        // Tint the glow by the ECS instance color too — the instance color is
        // the engine's per-instance variation channel, and a black baseColor
        // (the default) would otherwise swallow it entirely.
        var emissive = material.emissiveColor * in.ecsColor;
        if (material.hasEmissiveMap > 0.5) {
            emissive = emissive * textureSample(texEmissive, samp, in.uv).rgb;
        }

        // GPU-side pulse animation — pulseAmount of 0 disables it entirely
        let pulse = 1.0 + material.pulseAmount * sin(camera.simTime * material.pulseSpeed);

        let finalColor = baseColor.rgb + emissive * material.emissiveIntensity * pulse;
        return vec4<f32>(finalColor, baseColor.a);
    }
    `;

    return `
        ${CommonChunks.CameraStruct}
        ${EmissiveBindings}
        ${CommonChunks.EngineBindings}
        ${CommonChunks.VertexOut}
        ${CommonChunks.QuatMath}
        ${config.useSkinning ? CommonChunks.SkinningBindings : ''}

        ${buildStandardVertexShader(config)}
        ${fragmentShader}
    `;
}
