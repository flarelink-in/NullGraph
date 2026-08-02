import {NullGraph, RenderBatch} from "../../core";

export interface BasicMaterialOptions {
    albedoMap?: GPUTextureView;        // Optional - flat color if omitted
    baseColor?: [number, number, number, number];
    uvTiling?: [number, number];       // Tile the albedo map (useful for large surfaces)
}

/**
 * The cheapest material in the library. Unlit: no lights, no normal reads.
 * Ideal for UI planes, skyboxes, debug geometry, or massive instance counts
 * where per-pixel lighting is a waste of GPU time.
 */
export class BasicMaterial {
    private engine: NullGraph;

    public albedoMap: GPUTextureView | null = null;

    private uniformBuffer: GPUBuffer;
    private bindGroup: GPUBindGroup | null = null;

    public baseColor: [number, number, number, number];
    public uvTiling: [number, number];

    private hasAlbedoTexture: boolean;

    constructor(engine: NullGraph, options: BasicMaterialOptions = {}) {
        this.engine = engine;

        this.albedoMap = options.albedoMap || null;
        this.hasAlbedoTexture = !!options.albedoMap;

        this.baseColor = options.baseColor || [1.0, 1.0, 1.0, 1.0];
        this.uvTiling = options.uvTiling || [1.0, 1.0];

        this.uniformBuffer = this.engine.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.writeUniforms();
    }

    private writeUniforms() {
        // Layout must match BasicMaterialParams in BasicShaderBuilder (32 bytes)
        const uniformData = new Float32Array([
            ...this.baseColor,                 // offset 0:  vec4 (16 bytes)
            ...this.uvTiling,                  // offset 16: vec2 (8 bytes)
            this.hasAlbedoTexture ? 1.0 : 0.0, // offset 24: f32 (4 bytes)
            0.0                                // offset 28: padding (4 bytes)
        ]);
        this.engine.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData as BufferSource);
    }

    public applyToBatch(batch: RenderBatch, groupIndex: number = 1): GPUBindGroup {
        const textures: GPUTextureView[] = [
            this.albedoMap || this.engine.textureManager.fallbackWhite
        ];

        const extraUniforms: GPUBindGroupEntry[] = [
            { binding: 2, resource: { buffer: this.uniformBuffer } }
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
        uvTiling?: [number, number];
    }) {
        if (params.baseColor) this.baseColor = params.baseColor;
        if (params.uvTiling) this.uvTiling = params.uvTiling;
        this.writeUniforms();
    }

    public destroy() {
        this.uniformBuffer.destroy();
    }
}
