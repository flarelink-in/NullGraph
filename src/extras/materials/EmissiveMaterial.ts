import {NullGraph, RenderBatch} from "../../core";

export interface EmissiveMaterialOptions {
    albedoMap?: GPUTextureView;        // Optional - base surface texture
    emissiveMap?: GPUTextureView;      // Optional - masks which pixels glow
    baseColor?: [number, number, number, number];
    emissiveColor?: [number, number, number];
    emissiveIntensity?: number;        // Can exceed 1.0 for HDR/Bloom pipelines
    pulseSpeed?: number;               // Radians per simTime second
    pulseAmount?: number;              // 0 = steady glow, 1 = full pulse
}

/**
 * Unlit self-illuminated material. Emissive values are added on top of the
 * base color, so intensities > 1.0 push into HDR range for a Bloom pass.
 * The pulse animation runs entirely on the GPU via camera.simTime — no
 * per-frame uniform writes needed.
 */
export class EmissiveMaterial {
    private engine: NullGraph;

    public albedoMap: GPUTextureView | null = null;
    public emissiveMap: GPUTextureView | null = null;

    private uniformBuffer: GPUBuffer;
    private bindGroup: GPUBindGroup | null = null;

    public baseColor: [number, number, number, number];
    public emissiveColor: [number, number, number];
    public emissiveIntensity: number;
    public pulseSpeed: number;
    public pulseAmount: number;

    private hasAlbedoTexture: boolean;
    private hasEmissiveMap: boolean;

    constructor(engine: NullGraph, options: EmissiveMaterialOptions = {}) {
        this.engine = engine;

        this.albedoMap = options.albedoMap || null;
        this.emissiveMap = options.emissiveMap || null;
        this.hasAlbedoTexture = !!options.albedoMap;
        this.hasEmissiveMap = !!options.emissiveMap;

        this.baseColor = options.baseColor || [0.0, 0.0, 0.0, 1.0];
        this.emissiveColor = options.emissiveColor || [1.0, 1.0, 1.0];
        this.emissiveIntensity = options.emissiveIntensity ?? 1.0;
        this.pulseSpeed = options.pulseSpeed ?? 2.0;
        this.pulseAmount = options.pulseAmount ?? 0.0;

        this.uniformBuffer = this.engine.device.createBuffer({
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.writeUniforms();
    }

    private writeUniforms() {
        // Layout must match EmissiveMaterialParams in EmissiveShaderBuilder (48 bytes)
        const uniformData = new Float32Array([
            ...this.baseColor,                 // offset 0:  vec4 (16 bytes)
            ...this.emissiveColor,             // offset 16: vec3 (12 bytes)
            this.emissiveIntensity,            // offset 28: f32 (4 bytes)
            this.pulseSpeed,                   // offset 32: f32 (4 bytes)
            this.pulseAmount,                  // offset 36: f32 (4 bytes)
            this.hasAlbedoTexture ? 1.0 : 0.0, // offset 40: f32 (4 bytes)
            this.hasEmissiveMap ? 1.0 : 0.0    // offset 44: f32 (4 bytes)
        ]);
        this.engine.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData as BufferSource);
    }

    public applyToBatch(batch: RenderBatch, groupIndex: number = 1): GPUBindGroup {
        // Always provide both texture slots — fall back when null
        const textures: GPUTextureView[] = [
            this.albedoMap || this.engine.textureManager.fallbackWhite,
            this.emissiveMap || this.engine.textureManager.fallbackWhite
        ];

        const extraUniforms: GPUBindGroupEntry[] = [
            { binding: 3, resource: { buffer: this.uniformBuffer } }
        ];

        this.bindGroup = this.engine.createTextureBindGroup(
            batch,
            textures,
            this.engine.textureManager.defaultSampler,
            extraUniforms,
            groupIndex
        );

        return this.bindGroup;
    }

    public updateUniforms(params: {
        baseColor?: [number, number, number, number];
        emissiveColor?: [number, number, number];
        emissiveIntensity?: number;
        pulseSpeed?: number;
        pulseAmount?: number;
    }) {
        if (params.baseColor) this.baseColor = params.baseColor;
        if (params.emissiveColor) this.emissiveColor = params.emissiveColor;
        if (params.emissiveIntensity !== undefined) this.emissiveIntensity = params.emissiveIntensity;
        if (params.pulseSpeed !== undefined) this.pulseSpeed = params.pulseSpeed;
        if (params.pulseAmount !== undefined) this.pulseAmount = params.pulseAmount;
        this.writeUniforms();
    }

    public destroy() {
        this.uniformBuffer.destroy();
    }
}
