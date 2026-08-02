// LightSystem.ts
import {NullGraph, RenderBatch, ComputeStageNode, Camera} from '../../core';
import type {LightSystemConfig, ShaderLightingLayout} from './types';
import {getComputeShaderWGSL} from "./clusteredComputeShader";

export class LightSystem {
    public readonly requiresCompute: boolean;

    // Core Light Buffer (Always used)
    public lightStorageBuffer!: GPUBuffer;

    // Clustered Buffers (Only allocated if technique is clustered-forward)
    public clusterGridBuffer?: GPUBuffer;
    public lightIndexBuffer?: GPUBuffer;
    public globalCounterBuffer?: GPUBuffer;

    private stagingData: Float32Array | null = null;
    private isDirty: boolean = false;
    private totalLights: number;
    private totalClusters: number = 0;
    private maxIndices: number = 0;

    public static readonly FLOATS_PER_LIGHT = 16;

    constructor(private engine: NullGraph, public config: LightSystemConfig) {
        this.totalLights = config.capacity.point + config.capacity.directional + config.capacity.spot;
        this.requiresCompute = config.technique === 'clustered-forward' || config.technique === 'tiled-forward';

        this.initBuffers();
    }

    private initBuffers() {
        const device = this.engine.device;

        // 1. Core Light Buffer (64 bytes per light)
        this.lightStorageBuffer = device.createBuffer({
            label: 'NullGraph_LightData',
            size: this.totalLights * LightSystem.FLOATS_PER_LIGHT * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // 2. Clustered Buffers Allocation
        if (this.config.technique === 'clustered-forward' && this.config.clusterConfig) {
            const grid = this.config.clusterConfig.grid;
            this.totalClusters = grid[0] * grid[1] * grid[2];

            // Grid Buffer: 2 u32s per cluster (offset, count) = 8 bytes per cluster
            this.clusterGridBuffer = device.createBuffer({
                label: 'NullGraph_ClusterGrid',
                size: this.totalClusters * 8,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });

            // Index Buffer: Flat list of light IDs.
            this.maxIndices = this.totalClusters * this.config.clusterConfig.maxLightsPerTile;
            this.lightIndexBuffer = device.createBuffer({
                label: 'NullGraph_LightIndices',
                size: this.maxIndices * 4, // 1 u32 (4 bytes) per index
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });

            // Global Counter: 1 u32 atomic counter used by the compute shader
            this.globalCounterBuffer = device.createBuffer({
                label: 'NullGraph_GlobalLightCounter',
                size: 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
        }
    }

    /**
     * Inversion of Control: Attach our buffers to the Render Batch
     */
    public applyToBatch(batch: RenderBatch, bindGroupIndex: number = 2) {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: this.lightStorageBuffer } }
        ];

        // If clustered, bind the spatial grids to the SAME group for the fragment shader
        if (this.config.technique === 'clustered-forward') {
            entries.push({ binding: 1, resource: { buffer: this.clusterGridBuffer! } });
            entries.push({ binding: 2, resource: { buffer: this.lightIndexBuffer! } });
        }

        this.engine.setRenderPassBindGroup(batch, entries, bindGroupIndex);
    }

    /**
     * Accepts a raw Float32Array directly from the user or a Web Worker.
     * Enforces the Zero-Copy philosophy.
     */
    public setRawData(data: Float32Array) {
        this.stagingData = data;
        this.isDirty = true;
    }

    /**
     * Blasts the raw memory to VRAM. Should be called once per frame.
     */
    public upload() {
        if (!this.isDirty || !this.stagingData) return;

        this.engine.device.queue.writeBuffer(
            this.lightStorageBuffer,
            0,
            this.stagingData.buffer,
            this.stagingData.byteOffset,
            this.stagingData.byteLength
        );

        this.isDirty = false;
    }

    /**
     * Provides the WGSL string chunks to the dynamic shader builder.
     */
    public getShaderLayout(): ShaderLightingLayout | undefined {
        if (this.config.technique === 'standard') {
            return {
                wgslStructs: `
                    struct Light {
                        pos_type: vec4<f32>,     
                        col_int: vec4<f32>,      
                        dir_rad: vec4<f32>,      
                        spot_params: vec4<f32>   
                    };

                    struct IncidentLight {
                        color: vec3<f32>,
                        direction: vec3<f32>,
                        attenuation: f32,
                    };

                    @group(2) @binding(0) var<storage, read> lights: array<Light>;
                `,
                wgslLogic: `
                    fn getVisibleLightCount(worldPos: vec3<f32>) -> u32 {
                        return ${this.totalLights}u; // Standard loops over every light
                    }
                    fn getIncidentLight(index: u32, worldPos: vec3<f32>) -> IncidentLight {
                        let light = lights[index];
                        
                        let lightType = light.pos_type.w; 
                        let lightColor = light.col_int.xyz * light.col_int.w;
                        
                        var dir: vec3<f32>;
                        var atten = 1.0;

                        if (lightType == 2.0) { 
                            // DIRECTIONAL
                            dir = normalize(-light.dir_rad.xyz);
                        } else if (lightType == 3.0) {
                            // SPOT LIGHT
                            let lightPos = light.pos_type.xyz;
                            let spotDir = normalize(light.dir_rad.xyz);
                            let displacement = lightPos - worldPos;
                            let dist = length(displacement);
                            let lightDir = displacement / dist;
                            
                            let cosAngle = dot(-lightDir, spotDir);
                            let innerCos = light.spot_params.x;  // cos(innerConeAngle)
                            let outerCos = light.spot_params.y;  // cos(outerConeAngle)
                            
                            // Smooth spotlight falloff
                            let spotFactor = smoothstep(outerCos, innerCos, cosAngle);
                            
                            // Distance attenuation (same as point light)
                            let radius = light.dir_rad.w;
                            let distanceFalloff = clamp(1.0 - pow(dist / radius, 4.0), 0.0, 1.0);
                            atten = spotFactor * (distanceFalloff * distanceFalloff) / (dist * dist + 1.0);
                            
                            dir = lightDir;
                        } else { 
                            // POINT LIGHT (type == 1.0)
                            let displacement = light.pos_type.xyz - worldPos;
                            let dist = length(displacement);
                            let radius = light.dir_rad.w;
                            
                            dir = displacement / dist;
                            
                            let distanceFalloff = clamp(1.0 - pow(dist / radius, 4.0), 0.0, 1.0);
                            atten = (distanceFalloff * distanceFalloff) / (dist * dist + 1.0);
                        }
                        
                        return IncidentLight(lightColor, dir, atten);
                    }
                `
            };
        }
        if (this.config.technique === 'clustered-forward') {
            const grid = this.config.clusterConfig!.grid;

            return {
                wgslStructs: `
                    struct Light {
                        pos_type: vec4<f32>,     
                        col_int: vec4<f32>,      
                        dir_rad: vec4<f32>,      
                        spot_params: vec4<f32>   
                    };

                    struct IncidentLight {
                        color: vec3<f32>,
                        direction: vec3<f32>,
                        attenuation: f32,
                    };

                    struct Cluster {
                        start_offset: u32,
                        light_count: u32,
                    };

                    // Group 2 Bindings mapped from the applyToBatch method
                    @group(2) @binding(0) var<storage, read> globalLights: array<Light>;
                    @group(2) @binding(1) var<storage, read> clusterGrid: array<Cluster>;
                    @group(2) @binding(2) var<storage, read> lightIndices: array<u32>;
                `,
                wgslLogic: `
                    const GRID_X = ${grid[0]}u;
                    const GRID_Y = ${grid[1]}u;
                    const GRID_Z = ${grid[2]}u;

                    // 1. Math to find which 3D tile this pixel lives in
                    fn getClusterData(worldPos: vec3<f32>) -> Cluster {
                        let viewPos = camera.view * vec4<f32>(worldPos, 1.0);
                        let depth = -viewPos.z;

                        // Project to screen space [0, 1]
                        let clipPos = camera.proj * viewPos;
                        let ndc = clipPos.xy / clipPos.w;
                        let uv = ndc * 0.5 + vec2<f32>(0.5); 

                        // Find X and Y slices
                        let sliceX = u32(clamp(uv.x * f32(GRID_X), 0.0, f32(GRID_X) - 1.0));
                        let sliceY = u32(clamp(uv.y * f32(GRID_Y), 0.0, f32(GRID_Y) - 1.0));

                        // Find Z slice (Logarithmic)
                        let sliceZ_f = log(depth / camera.zNear) / log(camera.zFar / camera.zNear) * f32(GRID_Z);
                        let sliceZ = u32(clamp(sliceZ_f, 0.0, f32(GRID_Z) - 1.0));

                        let clusterIndex = sliceZ * (GRID_X * GRID_Y) + sliceY * GRID_X + sliceX;
                        return clusterGrid[clusterIndex];
                    }

                    // 2. Fulfill the ShaderBuilder Contract for the loop limit
                    fn getVisibleLightCount(worldPos: vec3<f32>) -> u32 {
                        return getClusterData(worldPos).light_count;
                    }

                    // 3. Fulfill the ShaderBuilder Contract for the Light Data
                    fn getIncidentLight(loopIndex: u32, worldPos: vec3<f32>) -> IncidentLight {
                        let cluster = getClusterData(worldPos);
                        
                        // Look up the REAL light index from the flat buffer using the cluster's offset
                        let realLightIndex = lightIndices[cluster.start_offset + loopIndex];
                        let light = globalLights[realLightIndex];
                        
                        let lightType = light.pos_type.w; 
                        let lightColor = light.col_int.xyz * light.col_int.w;
                        
                        var dir: vec3<f32>;
                        var atten = 1.0;

                        if (lightType == 2.0) { 
                            // DIRECTIONAL LIGHT
                            dir = normalize(-light.dir_rad.xyz);
                        } else if (lightType == 3.0) {
                            // SPOT LIGHT
                            let lightPos = light.pos_type.xyz;
                            let spotDir = normalize(light.dir_rad.xyz);
                            let displacement = lightPos - worldPos;
                            let dist = length(displacement);
                            let lightDir = displacement / dist;
                            
                            let cosAngle = dot(-lightDir, spotDir);
                            let innerCos = light.spot_params.x;  // cos(innerConeAngle)
                            let outerCos = light.spot_params.y;  // cos(outerConeAngle)
                            
                            // Smooth spotlight falloff
                            let spotFactor = smoothstep(outerCos, innerCos, cosAngle);
                            
                            // Distance attenuation
                            let radius = light.dir_rad.w;
                            let distanceFalloff = clamp(1.0 - pow(dist / radius, 4.0), 0.0, 1.0);
                            atten = spotFactor * (distanceFalloff * distanceFalloff) / (dist * dist + 1.0);
                            
                            dir = lightDir;
                        } else { 
                            // POINT LIGHT (type == 1.0)
                            let displacement = light.pos_type.xyz - worldPos;
                            let dist = length(displacement);
                            let radius = light.dir_rad.w;
                            
                            dir = displacement / dist;
                            
                            let distanceFalloff = clamp(1.0 - pow(dist / radius, 4.0), 0.0, 1.0);
                            atten = (distanceFalloff * distanceFalloff) / (dist * dist + 1.0);
                        }
                        
                        return IncidentLight(lightColor, dir, atten);
                    }
                `
            };
        }
        return undefined;
    }


    public getComputeStage(camera: Camera): ComputeStageNode {
        if (!this.requiresCompute || !this.config.clusterConfig) {
            throw new Error("This technique does not require a compute stage.");
        }

        const grid = this.config.clusterConfig.grid;
        // Dispatch enough workgroups to cover all clusters (64 threads per group)
        const dispatchX = Math.ceil(this.totalClusters / 64);

        const computeStage = this.engine.createComputeStage({
            name: 'Clustered Light Binning',
            shaderCode: this.getComputeShaderWGSL(),
            dispatchCount: [dispatchX, 1, 1],

            // CRITICAL: Reset the atomic counter and the grid to zero before binning
            preClearBuffers: [
                { buffer: this.globalCounterBuffer!, offset: 0, data: new Uint32Array([0]) }
                // Note: In a production engine, you also want a fast compute pass
                // to clear the ClusterGridBuffer to zero here.
            ],
        });

        // Wire up memory for the Compute Shader
        this.engine.setComputePassBindGroup(computeStage, [
            { binding: 0, resource: { buffer: this.engine.ctx.cameraUniformBuffer } },
            { binding: 1, resource: { buffer: this.lightStorageBuffer } },
            { binding: 2, resource: { buffer: this.clusterGridBuffer! } },
            { binding: 3, resource: { buffer: this.lightIndexBuffer! } },
            { binding: 4, resource: { buffer: this.globalCounterBuffer! } },
        ], 0); // Group 0 for compute

        return computeStage;
    }

    private getComputeShaderWGSL(): string {
        return getComputeShaderWGSL(this.config,this.totalLights)
    }

    public destroy() {
        this.lightStorageBuffer.destroy();
    }
}