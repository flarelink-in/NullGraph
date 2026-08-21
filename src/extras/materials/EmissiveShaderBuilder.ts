// EmissiveShaderBuilder.ts
// Unlit + self-illumination. Emissive output can exceed 1.0 (HDR) so it feeds
// nicely into a Bloom offscreen pass. Optional pulse animation is driven by
// camera.simTime — zero CPU cost, zero per-frame uniform writes.

import {CommonChunks, buildStandardVertexShader} from "./CommonChunks";
import {MaterialBindings, MaterialFragments} from "./materialFragments";
import {ShaderConfig} from "./types";

// Bindings and fragment live in materialFragments.ts so `null-graph/visibility` can reuse the
// exact same stage with its own GPU-culled vertex shader. Group 1 is this path's convention.
const EmissiveBindings = MaterialBindings.emissive(1);

export function buildEmissiveShader(config: ShaderConfig): string {

    const fragmentShader = MaterialFragments.emissive(config);

    return `
        ${CommonChunks.CameraStruct}
        ${EmissiveBindings}
        ${CommonChunks.EngineBindings}
        ${CommonChunks.VertexOut}
        ${CommonChunks.QuatMath}
        ${config.useSkinning ? CommonChunks.SkinningBindings : ''}

        ${config.aerial ? config.aerial.wgslStructs : ''}
        ${config.aerial ? config.aerial.wgslLogic : ''}

        ${buildStandardVertexShader(config)}
        ${fragmentShader}
    `;
}
