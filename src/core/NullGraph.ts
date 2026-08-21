import { Camera } from './Camera';
import { BufferManager } from './BufferManager';
import { RenderPassConfig, RenderPassNode } from "./RenderPass";
import { EngineInitResult, PipelineConfig, RenderBatch, ComputeStageConfig, ComputeStageNode, GPUCapabilities } from "./types";
export * from "./types";

// Internal Modules
import { WebGPUContext } from './WebGPUContext';
import { BatchManager } from './BatchManager';
import { RenderPipeline } from './RenderPipeline';
import { TextureManager } from "./TextureManager";

export class NullGraph {
    // Public API Contracts
    public device!: GPUDevice;
    public bufferManager!: BufferManager;
    public textureManager!: TextureManager;

    // We expose the raw context as readonly so users can access core buffers
    // like ctx.cameraUniformBuffer for their custom compute stages.
    public readonly ctx = new WebGPUContext();

    private batchManager!: BatchManager;
    private renderPipeline!: RenderPipeline;

    // Legacy tracking arrays (Optional in DOD, but kept for memory management / fallback)
    private passes: RenderPassNode[] = [];
    private batches: RenderBatch[] = [];
    private computeStages: ComputeStageNode[] = [];

    public canvas!: HTMLCanvasElement | OffscreenCanvas;

    // Engine State
    public isDestroyed: boolean = false;

    public async init(
        canvas: HTMLCanvasElement | OffscreenCanvas,
        options: {
            desiredFeatures?: GPUFeatureName[],
            powerPreference?: 'low-power' | 'high-performance',
            width?: number,
            height?: number,
            /**
             * Create the depth texture with `TEXTURE_BINDING` so it can be sampled
             * (environment-aerial-design.md §9.1). Required by
             * `AerialPerspective.getApplyPass()` when it uses the engine's own depth buffer;
             * a scene that renders offscreen with its own depth attachment needs nothing here.
             * Default `false`, because the flag can cost depth-compression fast paths every
             * frame whether or not anything samples the texture.
             */
            samplableDepth?: boolean
        } = {},
    ): Promise<EngineInitResult> {
        try {
            this.canvas = canvas;

            // Pass powerPreference through to the WebGPU context
            const initInfo = await this.ctx.init(canvas, options);

            this.device = this.ctx.device;
            this.bufferManager = new BufferManager(this.device);
            this.batchManager = new BatchManager(this.ctx);
            this.renderPipeline = new RenderPipeline(this.ctx);
            this.textureManager = new TextureManager(this.device);

            this.isDestroyed = false;

            return {
                success: true,
                enabledFeatures: this.device.features
            };
        } catch (e) {
            console.error("NullGraph Initialization Failed:", e);
            return {
                success: false,
                enabledFeatures: new Set(),
                error: e instanceof Error ? e.message : "Unknown Error"
            };
        }
    }

    public resize(width: number, height: number): void {
        this.ctx.resize(width, height);
    }

    public get width(): number {
        return this.ctx.width;
    }

    public get height(): number {
        return this.ctx.height;
    }

    /**
     * The negotiated optional-feature snapshot (subgroups, shader-f16, storage formats,
     * dual-source blending, subgroup sizes). Valid only after a successful {@link init}.
     * See {@link GPUCapabilities}.
     */
    public get capabilities(): GPUCapabilities {
        return this.ctx.capabilities;
    }

    public clearPasses(): void {
        this.passes = [];
    }

    public clearBatches(): void {
        this.batches = [];
    }

    // =========================================================
    // PIPELINE NODE CREATION
    // =========================================================

    public createPass(config: RenderPassConfig): RenderPassNode {
        const pass = new RenderPassNode(config);
        this.passes.push(pass);
        return pass;
    }

    public createComputeStage(config: ComputeStageConfig): ComputeStageNode {
        const stage = this.batchManager.createComputeStage(config);
        this.computeStages.push(stage);
        return stage;
    }

    public createBatch(pass: RenderPassNode, config: PipelineConfig): RenderBatch {
        const batch = this.batchManager.createBatch(pass, config);
        this.batches.push(batch);
        return batch;
    }

    public clearBatch(pass: RenderPassNode, batch: RenderBatch): void {
        if (typeof this.batchManager.clearBatch === 'function') {
            this.batchManager.clearBatch(pass, batch);
        }

        const globalIndex = this.batches.indexOf(batch);
        if (globalIndex !== -1) {
            this.batches.splice(globalIndex, 1);
        }
    }

    // =========================================================
    // DATA & RESOURCE BINDING
    // =========================================================

    public setBatchGeometry(batch: RenderBatch, vertexBuffer: GPUBuffer, indexBuffer: GPUBuffer, indexCount: number, format: GPUIndexFormat = 'uint16'): void {
        this.batchManager.setBatchGeometry(batch, vertexBuffer, indexBuffer, indexCount, format);
    }

    public updateBatchData(batch: RenderBatch, rawData: Float32Array, instanceCount: number): void {
        this.batchManager.updateBatchData(batch, rawData, instanceCount);
    }

    public createTextureBindGroup(
        batch: RenderBatch,
        textureView: GPUTextureView | GPUTextureView[],
        sampler: GPUSampler,
        extraEntries: GPUBindGroupEntry[] = [],
        groupIndex: number = 1
    ): GPUBindGroup {
        return this.batchManager.createTextureBindGroup(batch, textureView, sampler, extraEntries, groupIndex);
    }

    /**
     * Binds custom data exclusively to a RenderBatch's graphics pipeline.
     */
    public setRenderPassBindGroup(batch: RenderBatch, entries: GPUBindGroupEntry[], groupIndex: number = 1): void {
        return this.batchManager.setRenderPassBindGroup(batch, entries, groupIndex);
    }

    /**
     * NEW: Binds custom data exclusively to a standalone ComputeStageNode.
     */
    public setComputePassBindGroup(stage: ComputeStageNode, entries: GPUBindGroupEntry[], groupIndex: number = 0): void {
        return this.batchManager.setComputePassBindGroup(stage, entries, groupIndex);
    }

    public updateCamera(camera: Camera): void {
        this.device.queue.writeBuffer(
            this.ctx.cameraUniformBuffer,
            0,
            camera.bufferData.buffer,
            camera.bufferData.byteOffset,
            camera.bufferData.byteLength // <--- USE THIS INSTEAD OF 20 * 4
        );
    }

    //----------


    // =========================================================
    // EXECUTION
    // =========================================================

    /**
     * Executes the WebGPU render loop.
     * @param pipelineSequence An explicit, linear array of compute and render stages.
     */
    public render(pipelineSequence?: (ComputeStageNode | RenderPassNode)[]): void {
        if (this.isDestroyed) return;

        // If the user provides an explicit pipeline array, use it (The Modern DOD Way)
        // If not, fallback to rendering the tracked passes (The Legacy Way)
        const sequenceToRender = pipelineSequence || this.passes;

        this.renderPipeline.render(sequenceToRender);
    }
}