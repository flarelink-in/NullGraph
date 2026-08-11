// BasicShaderBuilder.ts
// Unlit material — the cheapest possible fragment path. No lights, no normals,
// just baseColor * ECS instance color * (optional) albedo texture.

import {CommonChunks, buildStandardVertexShader} from "./CommonChunks";
import {MaterialBindings, MaterialFragments} from "./materialFragments";
import {ShaderConfig} from "./types";

// Bindings and fragment live in materialFragments.ts so `null-graph/visibility` can reuse the
// exact same stage with its own GPU-culled vertex shader. Group 1 is this path's convention.
const BasicBindings = MaterialBindings.basic(1);

export function buildBasicShader(config: ShaderConfig): string {

    const fragmentShader = MaterialFragments.basic(config);

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
