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

    /**
     * Name for the shader module and pipeline, quoted back by every WebGPU validation and
     * compile error that involves them. Defaults to `"<pass name> #<batch index>"`.
     */
    label?: string;

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

    /**
     * Depth-only rendering (shadows-design.md §12.1): when set, the pipeline omits the
     * `fragment` stage and all color targets — a fragment-less render pipeline is core WebGPU.
     * The pass it is created into must have zero color attachments (a depth-only
     * `RenderPassNode` needs only its `depthStencilAttachment` set).
     */
    depthOnly?: boolean;

    /**
     * The group-0 binding-0 camera uniform, defaulting to `ctx.cameraUniformBuffer`
     * (shadows-design.md §12.1). This is what makes "a cascade is a camera" true end-to-end —
     * and it is the enabling change for every future off-main-view render: shadows, reflection
     * probes, IBL capture, editor picking passes.
     */
    cameraBuffer?: GPUBuffer;

    /**
     * Per-batch viewport, in attachment pixels: `[x, y, width, height]`
     * (shadows-atlas-design.md §12.1). When set, the render pass scopes this batch's
     * rasterization to the rectangle via `setViewport` + `setScissorRect` — the change the
     * shadow atlas rests on (§6.2: a clip-space remap bleeds into neighbouring tiles because
     * clipping still happens against the full attachment). Inert when omitted.
     */
    viewport?: [x: number, y: number, width: number, height: number];

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

    /**
     * Per-batch viewport in attachment pixels (shadows-atlas-design.md §12.1). Copied from
     * `PipelineConfig.viewport` at creation; mutable so a system (the shadow atlas) can retarget
     * a batch when its tile moves without rebuilding the pipeline.
     */
    public viewport?: [x: number, y: number, width: number, height: number];

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

/**
 * Options for `NullGraph.init(canvas, options)` / `WebGPUContext.init`.
 *
 * Two of them change *where the engine runs* rather than how it renders
 * (docs/handoff.md, Cycle A):
 *
 * - **`device`** adopts a `GPUDevice` another library already created — ONNX Runtime Web's
 *   `await ort.env.webgpu.device` is the motivating case — so both libraries share buffers with
 *   zero copies. NullGraph never destroys a device it adopted.
 * - **`canvas = null`** renders the main-screen pass into an offscreen texture
 *   (`engine.screenTexture`) instead of a canvas, which is what runs the engine in Node (Dawn) and
 *   in tests. `width` and `height` are then required, because there is no canvas to size from.
 */
export interface EngineInitOptions {
    /** Optional features to request. Missing ones warn rather than fail (they are optional). */
    desiredFeatures?: GPUFeatureName[];
    /** Adapter preference. Refused together with `device` — an adopted device has no adapter request. */
    powerPreference?: 'low-power' | 'high-performance';
    /**
     * Opportunistically enable supported `PROGRESSION_FEATURES`. Default `true`. Refused together
     * with `device`: features are fixed when a device is created, so an adopted device's are
     * read from `device.features`, never negotiated.
     */
    autoNegotiate?: boolean;
    /** Render size in pixels. Defaults to the canvas size; required when `canvas` is `null`. */
    width?: number;
    height?: number;
    /**
     * Create the depth texture with `TEXTURE_BINDING` so it can be sampled
     * (environment-aerial-design.md §9.1). Default `false`, because the flag can cost
     * depth-compression fast paths every frame whether or not anything samples it.
     */
    samplableDepth?: boolean;
    /**
     * Adopt this device instead of requesting one. Capabilities are read from what the device
     * already has; nothing is negotiated, and NullGraph never calls `device.destroy()` on it.
     */
    device?: GPUDevice;
    /**
     * Main-screen color format. Defaults to `navigator.gpu.getPreferredCanvasFormat()` with a
     * canvas, and to `'rgba8unorm'` without one (byte order a PNG writer can take as-is).
     */
    format?: GPUTextureFormat;
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

    /**
     * Take the workgroup count from a **GPU buffer** instead of `dispatchCount`
     * (`dispatchWorkgroupsIndirect`). Three `u32` — `[x, y, z]` — at `indirectOffset`.
     *
     * This is what removes over-dispatch. `dispatchCount` can only be built from values the CPU
     * knows, so every stage whose real workload is a GPU-side count currently dispatches against a
     * *cap* and lets idle threads early-out — `VisibilitySystem` dispatches
     * `ceil(instanceCount / 64)` workgroups when the surviving count is a fraction of that, and a
     * broadphase pair list has no CPU-known bound at all. With this, a prior compute stage writes
     * the exact count and the dispatch is exact.
     *
     * The buffer needs `INDIRECT` usage, and almost always `STORAGE` too, since the stage that
     * writes it is a compute shader. `createComputeStage` refuses a buffer missing either
     * (`BufferManager.createDispatchIndirectBuffer` allocates a correct one).
     *
     * **The producing kernel must clamp.** Nothing on the CPU can validate a value the GPU wrote,
     * so a count above `device.limits.maxComputeWorkgroupsPerDimension` is the caller's to prevent
     * — see that helper's note.
     *
     * Core WebGPU. No feature flag, no detection.
     */
    indirectBuffer?: GPUBuffer;

    /** Byte offset of the `[x, y, z]` triple. Must be a multiple of 4. Default 0. */
    indirectOffset?: number;
}

export class ComputeStageNode {
    public name: string;
    public computePipeline!: GPUComputePipeline;
    public customBindGroups: { [key: number]: GPUBindGroup } = {};
    public preClearBuffers: BufferClearConfig[];

    /**
     * Set `false` to skip this stage — its `preClearBuffers` **and** its dispatch — for a frame,
     * without splicing it out of the sequence array. Mutable and checked per frame by
     * `RenderPipeline`, so a system that owns many stages can gate them cheaply.
     *
     * The shadow atlas uses this: a cached tile that is not being re-rendered should not pay for
     * its caster cull either (shadows-atlas-design.md §8). Defaults to `true`, so every existing
     * stage runs exactly as before.
     */
    public enabled: boolean = true;

    /**
     * Buffer holding `[x, y, z]` workgroup counts — see {@link ComputeStageConfig.indirectBuffer}.
     * Mutable, so a system can retarget a stage at a different count buffer without rebuilding the
     * pipeline (the same reason `RenderBatch.viewport` is mutable).
     */
    public indirectBuffer?: GPUBuffer;

    /** Byte offset into {@link indirectBuffer}. */
    public indirectOffset: number = 0;

    private _dispatchCount: [number, number, number] | (() => [number, number, number]);

    constructor(config: ComputeStageConfig) {
        this.name = config.name;
        this._dispatchCount = config.dispatchCount || [1, 1, 1];
        this.indirectBuffer = config.indirectBuffer;
        this.indirectOffset = config.indirectOffset ?? 0;

        // Moved into the constructor where config is actively scoped
        this.preClearBuffers = config.preClearBuffers || [];
    }

    /** True when this stage takes its workgroup count from a buffer rather than from the CPU. */
    public get isIndirect(): boolean {
        return this.indirectBuffer !== undefined;
    }

    public get dispatchCount(): [number, number, number] {
        if (typeof this._dispatchCount === 'function') {
            return this._dispatchCount();
        }
        return this._dispatchCount;
    }
}