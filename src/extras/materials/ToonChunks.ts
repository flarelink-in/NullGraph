// ToonShaderChunks.ts

export const ToonChunks = {
    // 1. Common Structs & Bindings (Always included)
    CommonBindings: `
        const PI: f32 = 3.14159265359;

        struct Camera {
            viewProj: mat4x4<f32>,      
            view: mat4x4<f32>,          
            proj: mat4x4<f32>,          
            invViewProj: mat4x4<f32>,   
            invView: mat4x4<f32>,       
            invProj: mat4x4<f32>,       
    
            eyePos: vec3<f32>,     
            simTime: f32,               
    
            zNear: f32,                 
            zFar: f32,                  
            fovRads: f32,               
            aspect: f32,                
            _padEnd: vec4<f32>          
        };

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

       
        // Group 0: Engine Data
        @group(0) @binding(0) var<uniform> camera: Camera;
        @group(0) @binding(1) var<storage, read> ecs: array<f32>;

        // Group 1: Material Data
        @group(1) @binding(0) var samp: sampler;
        @group(1) @binding(1) var texAlbedo: texture_2d<f32>;
        @group(1) @binding(2) var texNormal: texture_2d<f32>;
        @group(1) @binding(3) var<uniform> material: ToonMaterialParams;

        struct VertexOut {
    @builtin(position) clipPos: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) worldNormal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) ecsColor: vec3<f32>,
}
    `,

    // 2. Skinning Bindings (Only included if mesh has bones)
    SkinningBindings: `
        struct Skeleton { boneMatrices: array<mat4x4<f32>, 70> };
        @group(2) @binding(0) var<uniform> skeleton: Skeleton;
    `,

    ToonMath: `
    // Quaternion rotation utility (needed for vertex shader)
    fn rotateVector(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
        let t = 2.0 * cross(q.xyz, v);
        return v + q.w * t + cross(q.xyz, t);
    }

    // Core toon ramp function - creates the cel-shading bands
    fn calcToonRamp(NdotL: f32, steps: f32) -> f32 {
        return ceil(NdotL * steps) / steps;
    }

    // Smooth toon ramp (optional variation with softer edges)
    fn calcSmoothToonRamp(NdotL: f32, steps: f32, smoothness: f32) -> f32 {
        let ramp = NdotL * steps;
        let layer = floor(ramp);
        let nextLayer = min(layer + 1.0, steps);
        let frac = ramp - layer;
        let smoothedFrac = smoothstep(0.0, smoothness, frac);
        return mix(layer, nextLayer, smoothedFrac) / steps;
    }

    // Rim light calculation
    fn calcRimLight(
        normal: vec3<f32>, 
        viewDir: vec3<f32>, 
        color: vec3<f32>, 
        power: f32
    ) -> vec3<f32> {
        let NdotV = max(dot(normal, viewDir), 0.0);
        let rim = pow(1.0 - NdotV, power);
        return rim * color;
    }

    // Sharp specular highlight (binary on/off for toon look)
    fn calcToonSpecular(
        normal: vec3<f32>,
        viewDir: vec3<f32>,
        lightDir: vec3<f32>,
        threshold: f32,
        specularColor: vec3<f32>
    ) -> vec3<f32> {
        let halfVec = normalize(lightDir + viewDir);
        let NdotH = max(dot(normal, halfVec), 0.0);
        let specIntensity = pow(NdotH, 32.0); // Hard specular
        return select(vec3(0.0), specularColor, specIntensity > threshold);
    }

    // Get normal either from map or geometry
    fn getToonNormal(
        uv: vec2<f32>, 
        worldPos: vec3<f32>, 
        worldNormal: vec3<f32>
    ) -> vec3<f32> {
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

    // Shadow banding (posterization of shadow values)
    fn calcToonShadow(shadowFactor: f32, steps: f32) -> f32 {
        return ceil(shadowFactor * steps) / steps;
    }
`,

    OutlineMath: `
    // Push vertices along normal for outline effect (FIXED VERSION)
    fn calcOutlinePosition(
        clipPos: vec4<f32>, 
        normal: vec3<f32>, 
        width: f32
    ) -> vec4<f32> {
        // In clip space, offset vertices along normal direction
        let offset = normal.xy * width * clipPos.w;
        return vec4<f32>(
            clipPos.x + offset.x,
            clipPos.y + offset.y,
            clipPos.z,
            clipPos.w
        );
    }

    // Alternative: object-space outline extrusion
    fn calcObjectSpaceOutline(
        worldPos: vec3<f32>, 
        worldNormal: vec3<f32>, 
        width: f32,
        viewProj: mat4x4<f32>
    ) -> vec4<f32> {
        let extrudedPos = worldPos + worldNormal * width;
        return viewProj * vec4(extrudedPos, 1.0);
    }
`,

    // 5. Complete toon lighting function
    ToonLighting: `
        fn calculateToonLighting(
            worldPos: vec3<f32>,
            worldNormal: vec3<f32>,
            uv: vec2<f32>
        ) -> vec4<f32> {
            // Get base color from texture or uniform
            var baseColor: vec4<f32>;
            if (material.hasAlbedoTexture > 0.5) {
                baseColor = textureSample(texAlbedo, samp, uv) * material.baseColor;
            } else {
                baseColor = material.baseColor;
            }

            // Calculate view and light directions
            let viewDir = normalize(camera.eyePos - worldPos);
            let lightDir = normalize(-light.direction);
            
            // Get final normal
            let N = getToonNormal(uv, worldPos, worldNormal);
            
            // Diffuse toon shading
            let NdotL = max(dot(N, lightDir), 0.0);
            let toonDiffuse = calcToonRamp(NdotL, material.rampSteps);
            var finalColor = baseColor.rgb * toonDiffuse * light.color * light.intensity;

            // Ambient light (prevents completely dark shadows)
            let ambientStrength = 0.1;
            let ambient = ambientStrength * baseColor.rgb;
            finalColor += ambient;

            // Rim light
            let rimLight = calcRimLight(N, viewDir, material.rimLightColor, material.rimLightPower);
            finalColor += rimLight;

            // Specular highlight
            let specColor = vec3(1.0, 1.0, 1.0); // White specular
            let specular = calcToonSpecular(N, viewDir, lightDir, material.specularThreshold, specColor);
            finalColor += specular * light.intensity * 0.5;

            // Shadow banding (optional - if you have shadow maps)
            // let shadowFactor = getShadow(worldPos);
            // let bandedShadow = calcToonShadow(shadowFactor, material.rampSteps);
            // finalColor *= bandedShadow;

            return vec4(finalColor, baseColor.a);
        }
    `,

    // 6. Alternative: Multiple light support
    MultiLightToon: `
        struct PointLight {
            position: vec3<f32>,
            color: vec3<f32>,
            intensity: f32,
            range: f32,
        };

        @group(1) @binding(5) var<storage, read> pointLights: array<PointLight>;

        fn calculateMultiLightToon(
            worldPos: vec3<f32>,
            worldNormal: vec3<f32>,
            uv: vec2<f32>
        ) -> vec4<f32> {
            // Get base color
            var baseColor: vec4<f32>;
            if (material.hasAlbedoTexture > 0.5) {
                baseColor = textureSample(texAlbedo, samp, uv) * material.baseColor;
            } else {
                baseColor = material.baseColor;
            }

            let viewDir = normalize(camera.eyePos - worldPos);
            let N = getToonNormal(uv, worldPos, worldNormal);
            
            var finalColor = vec3(0.0);
            
            // Ambient
            let ambient = 0.1 * baseColor.rgb;
            finalColor += ambient;

            // Directional light
            let lightDir = normalize(-light.direction);
            let NdotL = max(dot(N, lightDir), 0.0);
            let toonDiffuse = calcToonRamp(NdotL, material.rampSteps);
            finalColor += baseColor.rgb * toonDiffuse * light.color * light.intensity;

            // Point lights
            for (var i = 0u; i < arrayLength(&pointLights); i++) {
                let pLight = pointLights[i];
                let lightVec = pLight.position - worldPos;
                let distance = length(lightVec);
                let attenuation = max(0.0, 1.0 - distance / pLight.range);
                
                if (attenuation > 0.0) {
                    let pointLightDir = normalize(lightVec);
                    let pNdotL = max(dot(N, pointLightDir), 0.0);
                    let pointToon = calcToonRamp(pNdotL, material.rampSteps);
                    finalColor += baseColor.rgb * pointToon * pLight.color * pLight.intensity * attenuation;
                }
            }

            // Rim light (based on view, not individual lights)
            let rimLight = calcRimLight(N, viewDir, material.rimLightColor, material.rimLightPower);
            finalColor += rimLight;

            return vec4(finalColor, baseColor.a);
        }
    `
};