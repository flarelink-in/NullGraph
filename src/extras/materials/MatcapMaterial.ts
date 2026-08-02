import {NullGraph, RenderBatch} from "../../core";

export interface MatcapMaterialOptions {
    matcapMap: GPUTextureView;         // The sphere-lit capture texture
    albedoMap?: GPUTextureView;        // Optional - multiplied under the matcap
    baseColor?: [number, number, number, number];
}

/**
 * Material capture shading: the entire lighting environment is baked into a
 * single sphere texture and looked up by view-space normal. Costs the same
 * as an unlit material but reads as fully lit — perfect for sculpt-style
 * viewers or massive crowds where real lights are too expensive.
 */
export class MatcapMaterial {
    private engine: NullGraph;

    public matcapMap: GPUTextureView;
    public albedoMap: GPUTextureView | null = null;

    private uniformBuffer: GPUBuffer;
    private bindGroup: GPUBindGroup | null = null;

    public baseColor: [number, number, number, number];

    private hasAlbedoTexture: boolean;

    constructor(engine: NullGraph, options: MatcapMaterialOptions) {
        this.engine = engine;

        this.matcapMap = options.matcapMap;
        this.albedoMap = options.albedoMap || null;
        this.hasAlbedoTexture = !!options.albedoMap;

        this.baseColor = options.baseColor || [1.0, 1.0, 1.0, 1.0];

        this.uniformBuffer = this.engine.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.writeUniforms();
    }

    private writeUniforms() {
        // Layout must match MatcapMaterialParams in MatcapShaderBuilder (32 bytes)
        const uniformData = new Float32Array([
            ...this.baseColor,                 // offset 0:  vec4 (16 bytes)
            this.hasAlbedoTexture ? 1.0 : 0.0, // offset 16: f32 (4 bytes)
            0.0, 0.0, 0.0                      // offset 20: padding (12 bytes)
        ]);
        this.engine.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData as BufferSource);
    }

    public applyToBatch(batch: RenderBatch, groupIndex: number = 1): GPUBindGroup {
        const textures: GPUTextureView[] = [
            this.matcapMap,
            this.albedoMap || this.engine.textureManager.fallbackWhite
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
    }) {
        if (params.baseColor) this.baseColor = params.baseColor;
        this.writeUniforms();
    }

    public destroy() {
        this.uniformBuffer.destroy();
    }
}
