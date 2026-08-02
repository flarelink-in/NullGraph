import {NullGraph, RenderBatch} from "../../core";

export interface ToonMaterialOptions {
    albedoMap?: GPUTextureView;        // Optional - for textured toon look
    normalMap?: GPUTextureView;        // Optional - rare in toon shading
    baseColor?: [number, number, number, number];
    rampSteps?: number;
    rimLightColor?: [number, number, number];
    rimLightPower?: number;
    specularThreshold?: number;
}

export class ToonMaterial {
    private engine: NullGraph;

    // Make textures optional/nullable
    public albedoMap: GPUTextureView | null = null;
    public normalMap: GPUTextureView | null = null;

    private uniformBuffer: GPUBuffer;
    private bindGroup: GPUBindGroup | null = null;

    public rampSteps: number;
    public rimLightColor: [number, number, number];
    public rimLightPower: number;
    public specularThreshold: number;

    // Track whether we're using textures
    private hasAlbedoTexture: boolean;
    private hasNormalTexture: boolean;

    constructor(engine: NullGraph, options: ToonMaterialOptions) {
        this.engine = engine;

        // Only assign textures if explicitly provided
        this.albedoMap = options.albedoMap || null;
        this.normalMap = options.normalMap || null;
        this.hasAlbedoTexture = !!options.albedoMap;
        this.hasNormalTexture = !!options.normalMap;

        // Toon parameters with sensible defaults
        const baseColor = options.baseColor || [1.0, 1.0, 1.0, 1.0];
        this.rampSteps = options.rampSteps ?? 3;
        this.rimLightColor = options.rimLightColor || [1.0, 1.0, 1.0];
        this.rimLightPower = options.rimLightPower ?? 2.0;
        this.specularThreshold = options.specularThreshold ?? 0.5;

        // Pack toon parameters into uniform buffer
        // Layout: vec4(16) + f32(4) + f32(4) + f32(4) + f32(4) + vec4(16) + vec4(16) + f32(4) + f32(4) + f32(4) + 8 padding = 80 bytes
        // The uniform data MUST be in this exact order:

        const uniformData = new Float32Array([
            ...baseColor,                    // vec4 (16 bytes)
            this.rampSteps,                  // f32 (4 bytes)
            this.rimLightPower,              // f32 (4 bytes)
            this.specularThreshold,          // f32 (4 bytes)
            0.0,                             // padding (4 bytes) - was outlineWidth
            ...this.rimLightColor, 0.0,      // vec3 + padding (16 bytes)
            0.0, 0.0, 0.0, 0.0,            // padding (16 bytes) - was outlineColor
            0.0,                             // padding (4 bytes) - was outlineEnabled
            this.hasAlbedoTexture ? 1.0 : 0.0, // f32 (4 bytes)
            this.hasNormalTexture ? 1.0 : 0.0, // f32 (4 bytes)
            0.0, 0.0                         // padding (8 bytes)
        ]);
// Total: 80 bytes

        this.uniformBuffer = this.engine.device.createBuffer({
            size: uniformData.byteLength,  // Should be 80 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.engine.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData as BufferSource);
    }

    public applyToBatch(batch: RenderBatch, groupIndex: number = 1): GPUBindGroup {
        // Build texture array - ALWAYS provide all 2 texture slots
        // Use fallback textures when actual textures are null
        const textures: GPUTextureView[] = [
            this.albedoMap || this.engine.textureManager.fallbackWhite,
            this.normalMap || this.engine.textureManager.fallbackNormal
        ];

        const extraUniforms: GPUBindGroupEntry[] = [
            { binding: 3, resource: { buffer: this.uniformBuffer } }
        ];

        this.bindGroup = this.engine.createTextureBindGroup(
            batch,
            textures,  // Always pass exactly 2 textures
            this.engine.textureManager.defaultSampler,
            extraUniforms,
            groupIndex
        );

        return this.bindGroup;
    }

    public updateUniforms(params: {
        baseColor?: [number, number, number, number];
        rampSteps?: number;
        rimLightColor?: [number, number, number];
        rimLightPower?: number;
        specularThreshold?: number;
    }) {
        if (params.rampSteps !== undefined) this.rampSteps = params.rampSteps;
        if (params.rimLightColor) this.rimLightColor = params.rimLightColor;
        if (params.rimLightPower !== undefined) this.rimLightPower = params.rimLightPower;
        if (params.specularThreshold !== undefined) this.specularThreshold = params.specularThreshold;

        const baseColor = params.baseColor || [1.0, 1.0, 1.0, 1.0];
        const uniformData = new Float32Array([
            ...baseColor,                    // offset 0:  vec4 (16 bytes)
            this.rampSteps,                  // offset 16: f32 (4 bytes)
            this.rimLightPower,              // offset 20: f32 (4 bytes)
            this.specularThreshold,          // offset 24: f32 (4 bytes)
            0.0,                             // offset 28: padding (4 bytes) - was outlineWidth
            ...this.rimLightColor, 0.0,      // offset 32: vec3 + padding (16 bytes)
            0.0, 0.0, 0.0, 0.0,            // offset 48: padding (16 bytes) - was outlineColor
            0.0,                             // offset 64: padding (4 bytes) - was outlineEnabled
            this.hasAlbedoTexture ? 1.0 : 0.0, // offset 68: f32 (4 bytes)
            this.hasNormalTexture ? 1.0 : 0.0, // offset 72: f32 (4 bytes)
            0.0, 0.0                         // offset 76: padding (8 bytes)
        ]);
        // Total: 84 bytes

        this.engine.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData as BufferSource);
    }

    public destroy() {
        this.uniformBuffer.destroy();
    }
}