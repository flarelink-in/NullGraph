// MatcapShaderBuilder.ts
// Material capture — full "lit" appearance from a single texture lookup keyed
// on the view-space normal. Zero lights, zero BRDF math. This is the classic
// trick for making huge instance counts look shaded at unlit cost.

import {CommonChunks, buildStandardVertexShader} from "./CommonChunks";
import {MaterialBindings, MaterialFragments} from "./materialFragments";
import {ShaderConfig} from "./types";

// Bindings and fragment live in materialFragments.ts so `null-graph/visibility` can reuse the
// exact same stage with its own GPU-culled vertex shader. Group 1 is this path's convention.
const MatcapBindings = MaterialBindings.matcap(1);

export function buildMatcapShader(config: ShaderConfig): string {

    const fragmentShader = MaterialFragments.matcap(config);

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
