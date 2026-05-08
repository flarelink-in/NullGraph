import {NullGraph, RenderBatch} from "../../core";

export interface PBRMaterialOptions {
    albedoMap?: GPUTextureView;
    normalMap?: GPUTextureView;
    packedMap?: GPUTextureView;
    packedMapFormat?: 'ARM' | 'MRAO';
    baseColor?: [number, number, number, number];
    metallicMultiplier?: number;
    roughnessMultiplier?: number;
}

export class StandardPBRMaterial {
    private engine: NullGraph;

    public albedoMap: GPUTextureView;
    public normalMap: GPUTextureView;
    public packedMap: GPUTextureView;

    private uniformBuffer: GPUBuffer;
    private bindGroup: GPUBindGroup | null = null;

    constructor(engine: NullGraph, options: PBRMaterialOptions) {
        this.engine = engine;

        // 1. Safe Fallbacks
        this.albedoMap = options.albedoMap || engine.textureManager.fallbackWhite;
        this.normalMap = options.normalMap || engine.textureManager.fallbackNormal;
        this.packedMap = options.packedMap || engine.textureManager.fallbackWhite;

        const baseColor = options.baseColor || [1.0, 1.0, 1.0, 1.0];
        const metallic = options.metallicMultiplier ?? 1.0;
        const roughness = options.roughnessMultiplier ?? 1.0;
        const formatFlag = options.packedMapFormat === 'MRAO' ? 1.0 : 0.0;

        const uniformData = new Float32Array([
            ...baseColor,
            metallic,
            roughness,
            formatFlag,
            0.0 // Padding
        ]);

        this.uniformBuffer = this.engine.device.createBuffer({
            size: uniformData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.engine.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData as BufferSource);
    }

    /**
     * Applies this material to a specific render batch.
     * This internally uses the engine's core attachTextureMaterial method!
     */
    public applyToBatch(batch: RenderBatch, groupIndex: number = 1): GPUBindGroup {
        // If we already built the bind group for this layout, we could cache it,
        // but since we are attaching to a specific batch, we rebuild/assign it.

        const extraUniforms: GPUBindGroupEntry[] = [
            { binding: 4, resource: { buffer: this.uniformBuffer } }
        ];

        // Call the core function!
        this.bindGroup = this.engine.attachTextureMaterial(
            batch,
            [this.albedoMap, this.normalMap, this.packedMap],
            this.engine.textureManager.defaultSampler,
            extraUniforms, // Pass our custom material parameters!
            groupIndex
        );

        return this.bindGroup;
    }

    public updateUniforms(baseColor: [number, number, number, number], metallic: number, roughness: number) {
        const uniformData = new Float32Array([
            ...baseColor, metallic, roughness, 0.0, 0.0
        ]);
        this.engine.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData as BufferSource);
    }

    public destroy() {
        this.uniformBuffer.destroy();
    }
}