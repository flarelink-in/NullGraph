// LambertShaderBuilder.ts
// Cheap N·L diffuse — the middle ground between BasicMaterial (unlit) and
// StandardPBRMaterial (full BRDF). No specular, no normal maps, one texture
// fetch. Plugs into the same Clustered/Tiled LightSystem contract as PBR/Toon,
// which makes it the go-to material for crowds of thousands of lit instances.

import {CommonChunks, buildStandardVertexShader} from "./CommonChunks";
import {MaterialBindings, MaterialFragments} from "./materialFragments";
import {ShaderConfig} from "./types";

// Bindings and fragment live in materialFragments.ts so `null-graph/visibility` can reuse the
// exact same stage with its own GPU-culled vertex shader. Group 1 is this path's convention.
const LambertBindings = MaterialBindings.lambert(1);

export function buildLambertShader(config: ShaderConfig): string {

    const fragmentShader = MaterialFragments.lambert(config);

    return `
        ${CommonChunks.CameraStruct}
        ${LambertBindings}
        ${CommonChunks.EngineBindings}
        ${CommonChunks.VertexOut}
        ${CommonChunks.QuatMath}
        ${config.useSkinning ? CommonChunks.SkinningBindings : ''}

        ${config.lighting ? config.lighting.wgslStructs : ''}
        ${config.lighting ? config.lighting.wgslLogic : ''}

        ${config.aerial ? config.aerial.wgslStructs : ''}
        ${config.aerial ? config.aerial.wgslLogic : ''}

        ${buildStandardVertexShader(config)}
        ${fragmentShader}
    `;
}
