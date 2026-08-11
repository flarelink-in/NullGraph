// materialFragments.ts
//
// THE SINGLE SOURCE OF TRUTH for every stock material's fragment stage: its group bindings, its
// helper math, and its `fs_main`. Both consumers compose from here, so a material can never
// disagree with itself:
//
//   1. The full-shader builders (`buildPBRShader`, `buildLambertShader`, …) — engine vertex stage
//      reading the 14-float ECS instance buffer, plus these chunks.
//   2. `null-graph/visibility` — its own GPU-culled vertex stage (instance transforms come from a
//      mat4 in the Instance record, not the ECS buffer), plus these SAME chunks.
//
// Everything here is parameterized by group index because the two consumers disagree about where
// the material lives: group 1 for a normal batch and for visibility's instance tier, but group 2+
// for visibility's meshlet tier (group 1 is the meshlet geometry there).
//
// The fragment bodies assume a `VertexOut` with `worldPos` / `worldNormal` / `uv` / `ecsColor`.
// `materials/CommonChunks.ts` and `visibility/visibilityShaders.ts` both emit exactly that struct —
// keep all three in lockstep.

import type { ShaderConfig, ShaderLightingLayout } from './types';

/** The stock material families. Each maps to one bindings + fragment pair below. */
export type MaterialKind = 'basic' | 'lambert' | 'emissive' | 'matcap' | 'toon' | 'pbr';

// ---------------------------------------------------------------------------
// Group bindings — parameterized by group index
// ---------------------------------------------------------------------------

export const MaterialBindings: Record<MaterialKind, (group: number) => string> = {
    basic: (g) => `
    struct BasicMaterialParams {
        baseColor: vec4<f32>,        // 16 bytes (offset 0)
        uvTiling: vec2<f32>,         // 8 bytes  (offset 16)
        hasAlbedoTexture: f32,       // 4 bytes  (offset 24)
        _pad: f32,                   // 4 bytes  (offset 28)
    };
    // Total: 32 bytes
    @group(${g}) @binding(0) var samp: sampler;
    @group(${g}) @binding(1) var texAlbedo: texture_2d<f32>;
    @group(${g}) @binding(2) var<uniform> material: BasicMaterialParams;
`,

    lambert: (g) => `
    struct LambertMaterialParams {
        baseColor: vec4<f32>,        // 16 bytes (offset 0)
        ambient: f32,                // 4 bytes  (offset 16)
        hasAlbedoTexture: f32,       // 4 bytes  (offset 20)
        _pad: vec2<f32>,             // 8 bytes  (offset 24)
    };
    // Total: 32 bytes
    @group(${g}) @binding(0) var samp: sampler;
    @group(${g}) @binding(1) var texAlbedo: texture_2d<f32>;
    @group(${g}) @binding(2) var<uniform> material: LambertMaterialParams;
`,

    emissive: (g) => `
    struct EmissiveMaterialParams {
        baseColor: vec4<f32>,        // 16 bytes (offset 0)
        emissiveColor: vec3<f32>,    // 12 bytes (offset 16)
        emissiveIntensity: f32,      // 4 bytes  (offset 28)
        pulseSpeed: f32,             // 4 bytes  (offset 32)
        pulseAmount: f32,            // 4 bytes  (offset 36)
        hasAlbedoTexture: f32,       // 4 bytes  (offset 40)
        hasEmissiveMap: f32,         // 4 bytes  (offset 44)
    };
    // Total: 48 bytes
    @group(${g}) @binding(0) var samp: sampler;
    @group(${g}) @binding(1) var texAlbedo: texture_2d<f32>;
    @group(${g}) @binding(2) var texEmissive: texture_2d<f32>;
    @group(${g}) @binding(3) var<uniform> material: EmissiveMaterialParams;
`,

    matcap: (g) => `
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
    @group(${g}) @binding(0) var samp: sampler;
    @group(${g}) @binding(1) var texMatcap: texture_2d<f32>;
    @group(${g}) @binding(2) var texAlbedo: texture_2d<f32>;
    @group(${g}) @binding(3) var<uniform> material: MatcapMaterialParams;
`,

    toon: (g) => `
    struct ToonMaterialParams {
        baseColor: vec4<f32>,        // 16 bytes (offset 0)
        rampSteps: f32,              // 4 bytes  (offset 16)
        rimLightPower: f32,          // 4 bytes  (offset 20)
        specularThreshold: f32,      // 4 bytes  (offset 24)
        outlineWidth: f32,           // 4 bytes  (offset 28)
        rimLightColor: vec3<f32>,    // 12 bytes + 4 padding (offset 32)
        outlineColor: vec3<f32>,     // 12 bytes + 4 padding (offset 48)
        outlineEnabled: f32,         // 4 bytes  (offset 64)
        hasAlbedoTexture: f32,       // 4 bytes  (offset 68)
        hasNormalTexture: f32,       // 4 bytes  (offset 72)
        padding: vec2<f32>,          // 8 bytes  (offset 76)
    };
    // Total: 84 bytes
    @group(${g}) @binding(0) var samp: sampler;
    @group(${g}) @binding(1) var texAlbedo: texture_2d<f32>;
    @group(${g}) @binding(2) var texNormal: texture_2d<f32>;
    @group(${g}) @binding(3) var<uniform> material: ToonMaterialParams;
`,

    pbr: (g) => `
    struct MaterialParams {
        baseColor: vec4<f32>,
        metallic: f32,
        roughness: f32,
        mapFormat: f32,
        padding: f32,
    };
    @group(${g}) @binding(0) var samp: sampler;
    @group(${g}) @binding(1) var texAlbedo: texture_2d<f32>;
    @group(${g}) @binding(2) var texNormal: texture_2d<f32>;
    @group(${g}) @binding(3) var texMRAO: texture_2d<f32>;
    @group(${g}) @binding(4) var<uniform> material: MaterialParams;
`,
};

// ---------------------------------------------------------------------------
// Helper math the fragment bodies call
// ---------------------------------------------------------------------------

/** `PI` — declared by the PBR/Toon chunk sets. Emitted only when the host module lacks it. */
export const PI_WGSL = `const PI: f32 = 3.14159265359;\n`;

/** Cook-Torrance BRDF + normal mapping. Requires `samp` / `texNormal` from the PBR bindings. */
export const PBR_MATH_WGSL = `
    fn DistributionGGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
        let a = roughness * roughness;
        let a2 = a * a;
        let NdotH = max(dot(N, H), 0.0);
        let NdotH2 = NdotH * NdotH;
        let num = a2;
        var denom = (NdotH2 * (a2 - 1.0) + 1.0);
        denom = PI * denom * denom;
        return num / denom;
    }

    fn GeometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
        let r = (roughness + 1.0);
        let k = (r * r) / 8.0;
        return NdotV / (NdotV * (1.0 - k) + k);
    }

    fn GeometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
        let NdotV = max(dot(N, V), 0.0);
        let NdotL = max(dot(N, L), 0.0);
        return GeometrySchlickGGX(NdotL, roughness) * GeometrySchlickGGX(NdotV, roughness);
    }

    fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
        return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    }

    fn getNormalFromMap(uv: vec2<f32>, worldPos: vec3<f32>, worldNormal: vec3<f32>) -> vec3<f32> {
        let tangentNormal = textureSample(texNormal, samp, uv).xyz * 2.0 - 1.0;
        let Q1 = dpdx(worldPos); let Q2 = dpdy(worldPos);
        let st1 = dpdx(uv);      let st2 = dpdy(uv);
        let N = normalize(worldNormal);
        let T = normalize(Q1 * st2.y - Q2 * st1.y);
        let B = -normalize(cross(N, T));
        let TBN = mat3x3<f32>(T, B, N);
        return normalize(TBN * tangentNormal);
    }
`;

/** Cel ramps, rim light, hard specular, normal mapping. Requires the Toon bindings. */
export const TOON_MATH_WGSL = `
    fn calcToonRamp(NdotL: f32, steps: f32) -> f32 {
        return ceil(NdotL * steps) / steps;
    }

    fn calcSmoothToonRamp(NdotL: f32, steps: f32, smoothness: f32) -> f32 {
        let ramp = NdotL * steps;
        let layer = floor(ramp);
        let nextLayer = min(layer + 1.0, steps);
        let frac = ramp - layer;
        let smoothedFrac = smoothstep(0.0, smoothness, frac);
        return mix(layer, nextLayer, smoothedFrac) / steps;
    }

    fn calcRimLight(normal: vec3<f32>, viewDir: vec3<f32>, color: vec3<f32>, power: f32) -> vec3<f32> {
        let NdotV = max(dot(normal, viewDir), 0.0);
        let rim = pow(1.0 - NdotV, power);
        return rim * color;
    }

    fn calcToonSpecular(normal: vec3<f32>, viewDir: vec3<f32>, lightDir: vec3<f32>,
                        threshold: f32, specularColor: vec3<f32>) -> vec3<f32> {
        let halfVec = normalize(lightDir + viewDir);
        let NdotH = max(dot(normal, halfVec), 0.0);
        let specIntensity = pow(NdotH, 32.0);
        return select(vec3(0.0), specularColor, specIntensity > threshold);
    }

    fn getToonNormal(uv: vec2<f32>, worldPos: vec3<f32>, worldNormal: vec3<f32>) -> vec3<f32> {
        if (material.hasNormalTexture < 0.5) {
            return normalize(worldNormal);
        }
        let tangentNormal = textureSample(texNormal, samp, uv).xyz * 2.0 - 1.0;
        let Q1 = dpdx(worldPos);
        let Q2 = dpdy(worldPos);
        let st1 = dpdx(uv);
        let st2 = dpdy(uv);
        let N = normalize(worldNormal);
        let T = normalize(Q1 * st2.y - Q2 * st1.y);
        let B = -normalize(cross(N, T));
        let TBN = mat3x3<f32>(T, B, N);
        return normalize(TBN * tangentNormal);
    }
`;

/** Extra math a kind's fragment needs beyond its bindings. */
export const MaterialHelpers: Record<MaterialKind, string> = {
    basic: '',
    lambert: '',
    emissive: '',
    matcap: '',
    toon: TOON_MATH_WGSL,
    pbr: PBR_MATH_WGSL,
};

/** Whether the kind's helpers/fragment reference `PI`. */
export const MaterialNeedsPI: Record<MaterialKind, boolean> = {
    basic: false, lambert: false, emissive: false, matcap: false, toon: false, pbr: true,
};

// ---------------------------------------------------------------------------
// Fragment stages
// ---------------------------------------------------------------------------

export const MaterialFragments: Record<MaterialKind, (config: ShaderConfig) => string> = {
    basic: () => `
    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
        let instanceColor = vec4<f32>(in.ecsColor, 1.0);

        var color = material.baseColor * instanceColor;
        if (material.hasAlbedoTexture > 0.5) {
            color = textureSample(texAlbedo, samp, in.uv * material.uvTiling) * color;
        }

        return color;
    }
    `,

    lambert: (config) => {
        // Half-Lambert (Valve-style) wraps light around the terminator for a
        // softer look without any extra cost.
        const diffuseTerm = config.useHalfLambert ? `
            let wrapped = dot(N, L) * 0.5 + 0.5;
            let NdotL = wrapped * wrapped;
    ` : `
            let NdotL = max(dot(N, L), 0.0);
    `;
        return `
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
    },

    emissive: () => `
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
    `,

    matcap: () => `
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
    `,

    toon: (config) => `
    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
        // Calculate view direction
        let viewDir = normalize(camera.eyePos - in.worldPos);
        let instanceColor = vec4<f32>(in.ecsColor, 1.0);

        ${config.lighting ? `
        // --- DYNAMIC LIGHTING LOOP (Toon Style) ---
        let lightCount = getVisibleLightCount(in.worldPos);

        var baseColor: vec4<f32>;
        if (material.hasAlbedoTexture > 0.5) {
            baseColor = textureSample(texAlbedo, samp, in.uv) * material.baseColor * instanceColor;
        } else {
            baseColor = material.baseColor * instanceColor;
        }

        let N = getToonNormal(in.uv, in.worldPos, in.worldNormal);

        var finalColor = vec3<f32>(0.0);
        let ambient = 0.15 * baseColor.rgb;
        finalColor += ambient;

        for (var i = 0u; i < lightCount; i = i + 1u) {
            let incident = getIncidentLight(i, in.worldPos);
            let L = normalize(incident.direction);

            let NdotL = max(dot(N, L), 0.0);

            ${config.useSmoothRamp ? `
            // Smooth toon ramp
            let toonShade = calcSmoothToonRamp(NdotL, material.rampSteps, 0.2);
            ` : `
            // Hard toon ramp
            let toonShade = calcToonRamp(NdotL, material.rampSteps);
            `}

            // Toon diffuse
            let diffuse = baseColor.rgb * toonShade * incident.color * incident.attenuation;
            finalColor += diffuse;

            // Toon specular (hard cutoff)
            let specular = calcToonSpecular(N, viewDir, L, material.specularThreshold, vec3(1.0));
            finalColor += specular * incident.color * incident.attenuation * 0.5;
        }

        // Rim light (calculated once, not per-light)
        let rimLight = calcRimLight(N, viewDir, material.rimLightColor, material.rimLightPower);
        finalColor += rimLight;

        ` : `
        // --- HARDCODED FALLBACK TOON LIGHTING ---
        var baseColor: vec4<f32>;
        if (material.hasAlbedoTexture > 0.5) {
            baseColor = textureSample(texAlbedo, samp, in.uv) * material.baseColor * instanceColor;
        } else {
            baseColor = material.baseColor * instanceColor;
        }

        let lightDir = normalize(vec3<f32>(1.0, 1.0, 0.5));
        let lightColor = vec3<f32>(3.0, 3.0, 3.0);

        let N = getToonNormal(in.uv, in.worldPos, in.worldNormal);
        let NdotL = max(dot(N, lightDir), 0.0);

        ${config.useSmoothRamp ? `
        let toonShade = calcSmoothToonRamp(NdotL, material.rampSteps, 0.2);
        ` : `
        let toonShade = calcToonRamp(NdotL, material.rampSteps);
        `}

        var finalColor = baseColor.rgb * toonShade * lightColor;

        let ambient = 0.15 * baseColor.rgb;
        finalColor += ambient;

        let specular = calcToonSpecular(N, viewDir, lightDir, material.specularThreshold, vec3(1.0));
        finalColor += specular * lightColor * 0.5;

        let rimLight = calcRimLight(N, viewDir, material.rimLightColor, material.rimLightPower);
        finalColor += rimLight;
        `}

        return vec4<f32>(finalColor, baseColor.a);
    }
    `,

    pbr: (config) => `
    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
       let albedo = textureSample(texAlbedo, samp, in.uv).rgb * material.baseColor.rgb;
       let packedData = textureSample(texMRAO, samp, in.uv);

        var ao = 1.0;
        var roughness = material.roughness;
        var metallic = material.metallic;

        if (material.mapFormat < 0.5) {
            ao = packedData.r; roughness *= packedData.g; metallic *= packedData.b;
        } else {
            metallic *= packedData.r; roughness *= packedData.g; ao = packedData.b;
        }

        let N = getNormalFromMap(in.uv, in.worldPos, in.worldNormal);
        let V = normalize(camera.eyePos - in.worldPos);

        var F0 = vec3<f32>(0.04);
        F0 = mix(F0, albedo, metallic);

        var Lo = vec3<f32>(0.0);

        ${config.lighting ? `
        // --- DYNAMIC LIGHTING LOOP ---
        let lightCount = getVisibleLightCount(in.worldPos);

        for (var i = 0u; i < lightCount; i = i + 1u) {

            // 1. Query the Light System for the physical light data
            let incident = getIncidentLight(i, in.worldPos);

            // 2. Surface Math (PBR)
            let L = incident.direction;
            let H = normalize(V + L);

            // Calculate Cook-Torrance BRDF components
            let NDF = DistributionGGX(N, H, roughness);
            let G   = GeometrySmith(N, V, L, roughness);
            let F   = fresnelSchlick(max(dot(H, V), 0.0), F0);

            let numerator = NDF * G * F;
            let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
            let specular = numerator / denominator;

            // Energy conservation: diffuse + specular cannot exceed 1.0
            let kS = F; // Specular contribution is exactly the Fresnel value
            var kD = vec3<f32>(1.0) - kS; // Diffuse is whatever energy is left over

            // Pure metals have no diffuse light (they absorb it), so we multiply by (1 - metallic)
            kD = kD * (1.0 - metallic);

            let NdotL = max(dot(N, L), 0.0);

            // 3. Apply the final calculated energy
            Lo += (kD * albedo / 3.14159265 + specular) * incident.color * incident.attenuation * NdotL;
        }
        ` : `
        // --- HARDCODED FALLBACK LIGHTING ---
        let lightDir = normalize(vec3<f32>(1.0, 1.0, 0.5));
        let lightColor = vec3<f32>(3.0, 3.0, 3.0);

        let L = lightDir;
        let H = normalize(V + L);
        let NDF = DistributionGGX(N, H, roughness);
        let G   = GeometrySmith(N, V, L, roughness);
        let F   = fresnelSchlick(max(dot(H, V), 0.0), F0);

        let specular = (NDF * G * F) / (4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001);
        let kS = F;
        var kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);

        Lo += (kD * albedo / PI + specular) * lightColor * max(dot(N, L), 0.0);
        `}

        let ambient = vec3<f32>(0.3) * albedo * ao;
        var finalColor = ambient + Lo;

        // Tonemapping
        finalColor = finalColor / (finalColor + vec3<f32>(1.0));
        finalColor = pow(finalColor, vec3<f32>(1.0 / 2.2));
        return vec4<f32>(finalColor, material.baseColor.a);
    }
    `,
};

// ---------------------------------------------------------------------------
// The composed pair
// ---------------------------------------------------------------------------

export interface MaterialFragmentOptions {
    /** Which stock material family. */
    kind: MaterialKind;
    /** Bind group the material's textures/uniform occupy. Default 1. */
    groupIndex?: number;
    /** Clustered/standard lighting layout from `LightSystem.getShaderLayout()`. */
    lighting?: ShaderLightingLayout;
    /** Lambert: wrap light around the terminator. */
    useHalfLambert?: boolean;
    /** Toon: soften the cel ramp edges. */
    useSmoothRamp?: boolean;
}

/**
 * Build a stock material's fragment stage as a **portable pair**: the WGSL declarations
 * (`bindings`) and the `fs_main` body (`fragment`). Together they drop into any host module that
 * provides a `camera` uniform and a `VertexOut` with `worldPos`/`worldNormal`/`uv`/`ecsColor`.
 *
 * This is what makes `null-graph/visibility` able to render a stock material without the caller
 * writing any WGSL — see `VisibilitySystem.createInstanceDrawBatch({ material })`.
 *
 * `bindings` already includes the lighting structs/logic when `lighting` is supplied, so the
 * fragment's `getVisibleLightCount` / `getIncidentLight` calls resolve.
 */
export function buildMaterialFragment(opts: MaterialFragmentOptions): {
    bindings: string;
    fragment: string;
} {
    const { kind, groupIndex = 1, lighting } = opts;
    if (!MaterialBindings[kind]) {
        throw new Error(`buildMaterialFragment: unknown material kind "${kind}".`);
    }

    const config: ShaderConfig = {
        useSkinning: false,
        lighting,
        useHalfLambert: opts.useHalfLambert,
        useSmoothRamp: opts.useSmoothRamp,
    };

    const bindings = [
        MaterialNeedsPI[kind] ? PI_WGSL : '',
        MaterialBindings[kind](groupIndex),
        MaterialHelpers[kind],
        lighting ? lighting.wgslStructs : '',
        lighting ? lighting.wgslLogic : '',
    ].join('\n');

    return { bindings, fragment: MaterialFragments[kind](config) };
}
