// PBRShaderChunks.ts

export const PBRChunks = {
    // 1. Common Structs & Bindings (Always included)
    CommonBindings: `
        const PI: f32 = 3.14159265359;

        struct Camera { 
            viewProj: mat4x4<f32>, 
            eyePos: vec3<f32>, 
            simTime: f32 
        };

        struct MaterialParams {
            baseColor: vec4<f32>,
            metallic: f32,
            roughness: f32,
            mapFormat: f32, 
            padding: f32,
        };

        // Group 0: Engine Data
        @group(0) @binding(0) var<uniform> camera: Camera;
        @group(0) @binding(1) var<storage, read> ecs: array<f32>;

        // Group 1: Material Data
        @group(1) @binding(0) var samp: sampler;
        @group(1) @binding(1) var texAlbedo: texture_2d<f32>;
        @group(1) @binding(2) var texNormal: texture_2d<f32>;
        @group(1) @binding(3) var texMRAO: texture_2d<f32>;
        @group(1) @binding(4) var<uniform> material: MaterialParams;

        struct VertexOut {
            @builtin(position) clipPos: vec4<f32>,
            @location(0) worldPos: vec3<f32>,
            @location(1) worldNormal: vec3<f32>,
            @location(2) uv: vec2<f32>,
        };
    `,

    // 2. Skinning Bindings (Only included if mesh has bones)
    SkinningBindings: `
        struct Skeleton { boneMatrices: array<mat4x4<f32>, 70> };
        @group(2) @binding(0) var<uniform> skeleton: Skeleton;
    `,

    // 3. The complex PBR Math (Always included)
    PBRMath: `
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
        
        fn rotateVector(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
        let t = 2.0 * cross(q.xyz, v);
        return v + q.w * t + cross(q.xyz, t);
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
    `
};