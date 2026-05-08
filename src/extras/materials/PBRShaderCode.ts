export const PBRShaderCode = `
    // ==========================================
    // BINDINGS & STRUCTS
    // ==========================================
    struct Camera { 
        viewProj: mat4x4<f32>, 
        eyePos: vec3<f32>, 
        simTime: f32 
    };
    
   struct MaterialParams {
        baseColor: vec4<f32>,
        metallic: f32,
        roughness: f32,
        mapFormat: f32, // 0.0 = ARM, 1.0 = MRAO
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

    const PI: f32 = 3.14159265359;

    // ==========================================
    // VERTEX SHADER
    // ==========================================
    @vertex
    fn vs_main(
        @location(0) localPos: vec3<f32>,   
        @location(1) normal: vec3<f32>,      
        @location(2) uv: vec2<f32>,          
        @builtin(instance_index) iIdx: u32
    ) -> VertexOut {
        let base = iIdx * 14u;
        let pos = vec3<f32>(ecs[base + 1u], ecs[base + 2u], ecs[base + 3u]);
        let scale = vec3<f32>(ecs[base + 8u], ecs[base + 9u], ecs[base + 10u]);

        // Basic transform (Assuming no rotation for this simple ECS)
        let worldPosition = (localPos * scale) + pos;

        var out: VertexOut;
        out.clipPos = camera.viewProj * vec4<f32>(worldPosition, 1.0);
        out.worldPos = worldPosition;
        out.worldNormal = normal; // In a real engine with rotation, multiply by Inverse Transpose Model Matrix
        out.uv = uv;
        return out;
    }

    // ==========================================
    // PBR MATH FUNCTIONS (Cook-Torrance BRDF)
    // ==========================================

    // 1. Normal Distribution (GGX) - How "spread out" the microfacets are
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

    // 2. Geometry Function (Schlick-GGX) - Self-shadowing of microfacets
    fn GeometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
        let r = (roughness + 1.0);
        let k = (r * r) / 8.0;

        let num = NdotV;
        let denom = NdotV * (1.0 - k) + k;
        return num / denom;
    }

    fn GeometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
        let NdotV = max(dot(N, V), 0.0);
        let NdotL = max(dot(N, L), 0.0);
        let ggx2 = GeometrySchlickGGX(NdotV, roughness);
        let ggx1 = GeometrySchlickGGX(NdotL, roughness);
        return ggx1 * ggx2;
    }

    // 3. Fresnel (Schlick) - Reflectivity based on viewing angle
    fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
        return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    }

    // ==========================================
    // TANGENT SPACE GENERATOR (The Secret Sauce)
    // ==========================================
    fn getNormalFromMap(uv: vec2<f32>, worldPos: vec3<f32>, worldNormal: vec3<f32>) -> vec3<f32> {
        let tangentNormal = textureSample(texNormal, samp, uv).xyz * 2.0 - 1.0;

        // Screen-space derivatives (calculates math based on neighboring pixels)
        let Q1 = dpdx(worldPos);
        let Q2 = dpdy(worldPos);
        let st1 = dpdx(uv);
        let st2 = dpdy(uv);

        let N = normalize(worldNormal);
        let T = normalize(Q1 * st2.y - Q2 * st1.y);
        let B = -normalize(cross(N, T)); // Gram-Schmidt process
        
        let TBN = mat3x3<f32>(T, B, N);
        return normalize(TBN * tangentNormal);
    }

    // ==========================================
    // FRAGMENT SHADER
    // ==========================================
    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
       let albedo = textureSample(texAlbedo, samp, in.uv).rgb * material.baseColor.rgb;
       let packedData = textureSample(texMRAO, samp, in.uv); 
        
        var ao = 1.0; 
        var roughness = material.roughness; 
        var metallic = material.metallic;
        
        if (material.mapFormat < 0.5) {
            ao = packedData.r;
            roughness *= packedData.g;
            metallic *= packedData.b;
        } else {
            metallic *= packedData.r;
            roughness *= packedData.g;
            ao = packedData.b;
        }

        // --- 2. Calculate Core Vectors ---
        let N = getNormalFromMap(in.uv, in.worldPos, in.worldNormal);
        let V = normalize(camera.eyePos - in.worldPos);
        
        // F0 represents the base reflectivity. Plastics/Dielectrics are usually 0.04.
        // Metals tint their reflections based on their albedo color.
        var F0 = vec3<f32>(0.04); 
        F0 = mix(F0, albedo, metallic);

        // --- 3. Light Setup (Single Directional Light for now) ---
        var Lo = vec3<f32>(0.0);
        let lightDir = normalize(vec3<f32>(1.0, 1.0, 0.5));
        let lightColor = vec3<f32>(3.0, 3.0, 3.0); // HDR Brightness
        
        let L = lightDir;
        let H = normalize(V + L); // Halfway vector
        let radiance = lightColor; // No distance attenuation for directional lights

        // --- 4. The Cook-Torrance BRDF Math ---
        let NDF = DistributionGGX(N, H, roughness);   
        let G   = GeometrySmith(N, V, L, roughness);      
        let F   = fresnelSchlick(max(dot(H, V), 0.0), F0);       

        let numerator    = NDF * G * F;
        let denominator  = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // +0.0001 prevents divide by zero
        let specular     = numerator / denominator;

        // Energy conservation: diffuse + specular cannot exceed 1.0
        let kS = F;
        var kD = vec3<f32>(1.0) - kS;
        kD *= 1.0 - metallic; // Pure metals have no diffuse light!

        let NdotL = max(dot(N, L), 0.0);
        
        // Add light contribution to outgoing radiance
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;

        // --- 5. Ambient Lighting (Using AO) ---
        let ambient = vec3<f32>(0.3) * albedo * ao;
        var finalColor = ambient + Lo;

        // --- 6. HDR Tonemapping (Reinhard) & Gamma Correction ---
        finalColor = finalColor / (finalColor + vec3<f32>(1.0));
        finalColor = pow(finalColor, vec3<f32>(1.0 / 2.2));

        return vec4<f32>(finalColor, material.baseColor.a);
    }
`;