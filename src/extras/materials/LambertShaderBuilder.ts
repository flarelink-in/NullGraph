// LambertShaderBuilder.ts
// Cheap N·L diffuse — the middle ground between BasicMaterial (unlit) and
// StandardPBRMaterial (full BRDF). No specular, no normal maps, one texture
// fetch. Plugs into the same Clustered/Tiled LightSystem contract as PBR/Toon,
// which makes it the go-to material for crowds of thousands of lit instances.

import {CommonChunks, buildStandardVertexShader} from "./CommonChunks";
import {ShaderConfig} from "./types";

const LambertBindings = `
    struct LambertMaterialParams {
        baseColor: vec4<f32>,        // 16 bytes (offset 0)
        ambient: f32,                // 4 bytes  (offset 16)
        hasAlbedoTexture: f32,       // 4 bytes  (offset 20)
        _pad: vec2<f32>,             // 8 bytes  (offset 24)
    };
    // Total: 32 bytes

    // Group 1: Material Data
    @group(1) @binding(0) var samp: sampler;
    @group(1) @binding(1) var texAlbedo: texture_2d<f32>;
    @group(1) @binding(2) var<uniform> material: LambertMaterialParams;
`;

export function buildLambertShader(config: ShaderConfig): string {

    // Half-Lambert (Valve-style) wraps light around the terminator for a
    // softer look without any extra cost.
    const diffuseTerm = config.useHalfLambert ? `
            let wrapped = dot(N, L) * 0.5 + 0.5;
            let NdotL = wrapped * wrapped;
    ` : `
            let NdotL = max(dot(N, L), 0.0);
    `;

    const fragmentShader = `
    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
        let instanceColor = vec4<f32>(in.ecsColor, 1.0);

        var baseColor = material.baseColor * instanceColor;
        if (material.hasAlbedoTexture > 0.5) {
            baseColor = textureSample(texAlbedo, samp, in.uv) * baseColor;
        }

        let N = normalize(in.worldNormal);

        ${config.lighting ? `
        // --- DYNAMIC LIGHTING LOOP (Clustered/Tiled) ---
        let lightCount = getVisibleLightCount(in.worldPos);

        var finalColor = material.ambient * baseColor.rgb;

        for (var i = 0u; i < lightCount; i = i + 1u) {
            let incident = getIncidentLight(i, in.worldPos);
            let L = normalize(incident.direction);
            ${diffuseTerm}
            finalColor += baseColor.rgb * NdotL * incident.color * incident.attenuation;
        }
        ` : `
        // --- HARDCODED FALLBACK DIRECTIONAL LIGHT ---
        let L = normalize(vec3<f32>(1.0, 1.0, 0.5));
        let lightColor = vec3<f32>(1.0, 1.0, 1.0);
        ${diffuseTerm}

        var finalColor = material.ambient * baseColor.rgb;
        finalColor += baseColor.rgb * NdotL * lightColor;
        `}

        return vec4<f32>(finalColor, baseColor.a);
    }
    `;

    return `
        ${CommonChunks.CameraStruct}
        ${LambertBindings}
        ${CommonChunks.EngineBindings}
        ${CommonChunks.VertexOut}
        ${CommonChunks.QuatMath}
        ${config.useSkinning ? CommonChunks.SkinningBindings : ''}

        ${config.lighting ? config.lighting.wgslStructs : ''}
        ${config.lighting ? config.lighting.wgslLogic : ''}

        ${buildStandardVertexShader(config)}
        ${fragmentShader}
    `;
}
