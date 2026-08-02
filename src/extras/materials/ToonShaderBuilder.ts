import {ToonChunks} from "./ToonChunks";
import {ShaderConfig} from "./types";

export function buildToonShader(config: ShaderConfig): string {

    // --- 1. Assemble the Vertex Shader dynamically ---
    const vertexShader = `
    @vertex
    fn vs_main(
        @location(0) localPos: vec3<f32>,   
        @location(1) normal: vec3<f32>,      
        @location(2) uv: vec2<f32>,
        ${config.useSkinning ? `
        @location(3) joints: vec4<f32>,  
        @location(4) weights: vec4<f32>,` : ''}         
        @builtin(instance_index) iIdx: u32
    ) -> VertexOut {
        let base = iIdx * 14u;
        let ecsPos = vec3<f32>(ecs[base + 1u], ecs[base + 2u], ecs[base + 3u]);
        let ecsRot = vec4<f32>(ecs[base + 4u], ecs[base + 5u], ecs[base + 6u], ecs[base + 7u]);
        let ecsScale = vec3<f32>(ecs[base + 8u], ecs[base + 9u], ecs[base + 10u]);
        // Extract ECS Color
        let ecsColor = vec3<f32>(ecs[base + 11u], ecs[base + 12u], ecs[base + 13u]);

        ${config.useSkinning ? `
        // --- SKINNED TRANSFORM ---
        let jx = u32(joints.x); let jy = u32(joints.y);
        let jz = u32(joints.z); let jw = u32(joints.w);

        let skinMatrix = 
            weights.x * skeleton.boneMatrices[jx] +
            weights.y * skeleton.boneMatrices[jy] +
            weights.z * skeleton.boneMatrices[jz] +
            weights.w * skeleton.boneMatrices[jw];

        let finalLocalPos = (skinMatrix * vec4<f32>(localPos, 1.0)).xyz;
        let finalNormal = normalize((skinMatrix * vec4<f32>(normal, 0.0)).xyz);
        ` : `
        // --- STATIC TRANSFORM ---
        let finalLocalPos = localPos;
        let finalNormal = normal;
        `}

        let scaledPos = finalLocalPos * ecsScale;
        let rotatedPos = rotateVector(scaledPos, ecsRot);
        let worldPosition = rotatedPos + ecsPos;
        let finalWorldNormal = rotateVector(finalNormal, ecsRot);

        var out: VertexOut;
        out.clipPos = camera.viewProj * vec4<f32>(worldPosition, 1.0);
        out.worldPos = worldPosition;
        out.worldNormal = finalWorldNormal;
        out.uv = uv;
        out.ecsColor = ecsColor; // Pass to fragment shader
        return out;
    }
    `;

    // --- 2. Main Fragment Shader (CLEAN - No outlines) ---
    const fragmentShader = `
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
    `;

    // --- 3. Glue it all together ---
    return `
        ${ToonChunks.CommonBindings}
        ${config.useSkinning ? ToonChunks.SkinningBindings : ''}
        ${ToonChunks.ToonMath}

        ${config.lighting ? config.lighting.wgslStructs : ''}
        ${config.lighting ? config.lighting.wgslLogic : ''}

        ${vertexShader}
        ${fragmentShader}
    `;
}