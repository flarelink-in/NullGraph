export interface Material {
    applyToBatch(batch: RenderBatch, groupIndex?: number): GPUBindGroup;
    destroy(): void;
}

export interface PipelineConfig {
    shaderCode: string;
    strideFloats: number;
    maxInstances: number;
    vertexLayouts?: GPUVertexBufferLayout[];
    topology?: GPUPrimitiveTopology;
    targetFormats?: GPUTextureFormat[];
    material?:Material;

    // Optional properties for GPU-driven rendering
    isIndirect?: boolean;
    computeShaderCode?: string;
    sharedSourceBuffer?: GPUBuffer;
    extraBindGroup?: GPUBindGroup;
    depthWriteEnabled?: boolean;
    blend?: GPUBlendState;
    depthCompare?: GPUCompareFunction;

    targetFormat?: GPUTextureFormat;
}

export class RenderBatch {
    public pipeline!: GPURenderPipeline;
    public storageBuffer!: GPUBuffer;
    public bindGroup!: GPUBindGroup;
    public stride: number = 0;

    public vertexBuffer: GPUBuffer | null = null;
    public indexBuffer: GPUBuffer | null = null;
    public indexCount: number = 0;
    public indexFormat: GPUIndexFormat = 'uint16';

    public currentInstanceCount: number = 0;
    public maxInstanceCount?: number = 0;

    // NEW: Indirect & Compute Properties
    public isIndirect: boolean = false;
    public indirectBuffer?: GPUBuffer;
    public sourceStorageBuffer?: GPUBuffer;
    public computePipeline?: GPUComputePipeline;
    public computeBindGroup?: GPUBindGroup;
    public customBindGroups: { [groupIndex: number]: GPUBindGroup } = {};
    public computeCustomBindGroups: Record<number, GPUBindGroup> = {};


}

export interface EngineInitResult {
    success: boolean;
    enabledFeatures: ReadonlySet<string>;
    error?: string;
}