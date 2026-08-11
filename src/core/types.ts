export interface Material {
    applyToBatch(batch: RenderBatch, groupIndex?: number): GPUBindGroup;
    destroy(): void;
}

// =========================================================
// RENDER PIPELINE CONFIGURATIONS
// =========================================================

export interface PipelineConfig {
    shaderCode: string;
    strideFloats: number;
    maxInstances: number;

    // Geometry & Layout
    vertexLayouts?: GPUVertexBufferLayout[];
    topology?: GPUPrimitiveTopology;

    // Targets & Blending
    targetFormats?: GPUTextureFormat[];
    targetFormat?: GPUTextureFormat; // Fallback for single target
    blend?: GPUBlendState;

    // Depth State
    depthWriteEnabled?: boolean;
    depthCompare?: GPUCompareFunction;

    // Extensions
    material?: Material;

    // GPU-Driven Rendering (Strictly visual properties now)
    isIndirect?: boolean;
    sharedSourceBuffer?: GPUBuffer; // External buffer containing instance data (populated by a ComputeStage)
}

export class RenderBatch {
    public pipeline!: GPURenderPipeline;
    public storageBuffer!: GPUBuffer;
    public bindGroup!: GPUBindGroup;
    public stride: number = 0;

    // Geometry
    public vertexBuffer: GPUBuffer | null = null;
    public indexBuffer: GPUBuffer | null = null;
    public indexCount: number = 0;
    public indexFormat: GPUIndexFormat = 'uint16';

    // Instancing
    public currentInstanceCount: number = 0;
    public maxInstanceCount: number = 0;

    // Indirect Drawing
    public isIndirect: boolean = false;
    public indirectBuffer?: GPUBuffer;
    public sourceStorageBuffer?: GPUBuffer;

    // Render-specific Custom Data
    public customBindGroups: { [groupIndex: number]: GPUBindGroup } = {};
}

// =========================================================
// GLOBAL ENGINE TYPES
// =========================================================

export interface EngineInitResult {
    success: boolean;
    enabledFeatures: ReadonlySet<string>;
    error?: string;
}

// =========================================================
// GPU CAPABILITY NEGOTIATION
// =========================================================

/**
 * A structured, queryable snapshot of the optional WebGPU features (and subgroup
 * sizes) that were successfully negotiated when the device was created. Downstream
 * systems read this instead of re-probing the adapter — e.g. the clustered
 * {@link LightSystem} checks `capabilities.subgroups` to pick a subgroup-accelerated
 * binning kernel over the universal atomic one.
 *
 * Booleans mirror the entries NullGraph opportunistically enables (see
 * `PROGRESSION_FEATURES` in `WebGPUContext`). Detection is deliberately broader than
 * current usage: `shaderF16`, `bgra8unormStorage`, and `dualSourceBlending` are surfaced
 * as groundwork so those features can be adopted later without re-plumbing negotiation.
 */
export interface GPUCapabilities {
    /** The features actually granted on the `GPUDevice`. */
    readonly features: ReadonlySet<GPUFeatureName>;

    /** Subgroup ops (`subgroupAdd`/`subgroupElect`/…, `enable subgroups;`) are available. */
    readonly subgroups: boolean;
    /** 16-bit floats (`enable f16;`) are available in WGSL. */
    readonly shaderF16: boolean;
    /** `bgra8unorm` is usable as a storage texture (`STORAGE_BINDING`). */
    readonly bgra8unormStorage: boolean;
    /** Dual-source blending (`@blend_src(0/1)`) is available. */
    readonly dualSourceBlending: boolean;

    /** Minimum hardware subgroup size; `0` when subgroups are unavailable/unreported. */
    readonly subgroupMinSize: number;
    /** Maximum hardware subgroup size; `0` when subgroups are unavailable/unreported. */
    readonly subgroupMaxSize: number;

    /** Convenience membership test against the enabled feature set. */
    has(feature: GPUFeatureName): boolean;
}

// =========================================================
// COMPUTE STAGE CONFIGURATIONS
// =========================================================

export interface BufferClearConfig {
    buffer: GPUBuffer;
    offset: number;
    data: Uint32Array | Float32Array;
}

export interface ComputeStageConfig {
    name: string;
    shaderCode: string;

    // Dispatch dimensions or a function for dynamic evaluation (e.g., resizing window)
    dispatchCount?: [number, number, number] | (() => [number, number, number]);

    // Explicit workgroup dimensions (optional, usually handled in WGSL)
    workgroupSize?: [number, number, number];

    // Declarative buffer resets executed exactly before this shader runs
    preClearBuffers?: BufferClearConfig[];
}

export class ComputeStageNode {
    public name: string;
    public computePipeline!: GPUComputePipeline;
    public customBindGroups: { [key: number]: GPUBindGroup } = {};
    public preClearBuffers: BufferClearConfig[];

    private _dispatchCount: [number, number, number] | (() => [number, number, number]);

    constructor(config: ComputeStageConfig) {
        this.name = config.name;
        this._dispatchCount = config.dispatchCount || [1, 1, 1];

        // Moved into the constructor where config is actively scoped
        this.preClearBuffers = config.preClearBuffers || [];
    }

    public get dispatchCount(): [number, number, number] {
        if (typeof this._dispatchCount === 'function') {
            return this._dispatchCount();
        }
        return this._dispatchCount;
    }
}