// PBRShaderBuilder.ts
import { PBRChunks } from './PBRShaderChunks';
import {ShaderConfig} from "./types";

export function buildPBRShader(config: ShaderConfig): string {

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
        return out;
    }
    `;

    // --- 2. Assemble the Fragment Shader ---
    const fragmentShader = `
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
       // finalColor = vec3<f32>(1.0, 0.0, 0.0);
        return vec4<f32>(finalColor, material.baseColor.a);
    }
    `;

    // --- 3. Glue it all together ---
    // Notice how we safely inject the dynamic WGSL structs and logic
    // right before the fragment shader so they are available in scope.
    return `
        ${PBRChunks.CommonBindings}
        ${config.useSkinning ? PBRChunks.SkinningBindings : ''}
        ${PBRChunks.PBRMath}

        ${config.lighting ? config.lighting.wgslStructs : ''}
        ${config.lighting ? config.lighting.wgslLogic : ''}

        ${vertexShader}
        ${fragmentShader}
    `;
}