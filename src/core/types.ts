export interface PipelineConfig {
    shaderCode: string;
    strideFloats: number;
    maxInstances: number;
    vertexLayouts?: GPUVertexBufferLayout[];
    topology?: GPUPrimitiveTopology;
    targetFormats?: GPUTextureFormat[];

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
    public extraBindGroup?: GPUBindGroup;


}