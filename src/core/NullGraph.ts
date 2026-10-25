import { Camera } from './Camera';
import { BufferManager } from './BufferManager';
import { RenderPassConfig, RenderPassNode } from "./RenderPass";
import { EngineInitOptions, EngineInitResult, PipelineConfig, RenderBatch, ComputeStageConfig, ComputeStageNode, GPUCapabilities } from "./types";
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

    /** The canvas being presented to, or `null` when the engine renders headless. */
    public canvas: HTMLCanvasElement | OffscreenCanvas | null = null;

    // Engine State
    public isDestroyed: boolean = false;

    /**
     * Acquire (or adopt) a device and set up the main-screen target.
     *
     * @param canvas The canvas to present to, or `null` to render headless — the main-screen pass
     *   then draws into {@link screenTexture}, which is how the engine runs in Node (Dawn), in
     *   tests, and in render-to-PNG tooling. `null` needs `options.width` and `options.height`.
     * @param options See {@link EngineInitOptions}. `{ device }` adopts an existing device (e.g.
     *   ONNX Runtime Web's `await ort.env.webgpu.device`) so the two share buffers with no copy.
     *   `samplableDepth` is required by `AerialPerspective.getApplyPass()` when it uses the
     *   engine's own depth buffer.
     */
    public async init(
        canvas: HTMLCanvasElement | OffscreenCanvas | null,
        options: EngineInitOptions = {},
    ): Promise<EngineInitResult> {
        try {
            this.canvas = canvas;

            await this.ctx.init(canvas, options);

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

    /**
     * The offscreen texture the main-screen pass renders into when the engine was initialised
     * without a canvas (`init(null, …)`); `undefined` with a canvas. `RENDER_ATTACHMENT |
     * COPY_SRC`, in `engine.ctx.format` (`'rgba8unorm'` unless `options.format` said otherwise),
     * so a test can `copyTextureToBuffer` the finished frame.
     */
    public get screenTexture(): GPUTexture | undefined {
        return this.ctx.screenTexture;
    }

    /**
     * Release everything this engine created on the GPU, and stop rendering
     * (offline-films-browser-design.md §12):
     * - the buffers its batches allocated — never a `sharedSourceBuffer`, geometry or a camera buffer
     *   the caller passed in;
     * - the textures its texture manager made;
     * - the screen and depth targets, the camera uniform, and the canvas's configuration;
     * - the device, only if the engine requested it: an adopted device is its owner's.
     *
     * Passes and stages are forgotten, and `render()` does nothing afterwards. Extras own their own
     * resources (PostChain, LightSystem, …): destroy them first. Idempotent, and safe after an init that
     * failed part-way; every release is attempted, and failures are thrown together afterwards.
     */
    public destroy(): void {
        if (this.isDestroyed) return;
        this.isDestroyed = true;
        const errors: unknown[] = [];
        const attempt = (release: () => void) => { try { release(); } catch (e) { errors.push(e); } };
        attempt(() => this.batchManager?.destroy());
        attempt(() => this.textureManager?.destroy());
        attempt(() => this.ctx.destroy());
        this.passes = [];
        this.batches = [];
        this.computeStages = [];
        if (errors.length === 1) throw errors[0];
        if (errors.length > 1) throw new AggregateError(errors, `NullGraph.destroy: ${errors.length} releases failed.`);
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

    /**
     * The same stage as createComputeStage, compiled with createComputePipelineAsync: frames can go
     * on while it compiles. A shader that does not compile rejects with an error naming the stage.
     */
    public async createComputeStageAsync(config: ComputeStageConfig): Promise<ComputeStageNode> {
        const stage = await this.batchManager.createComputeStageAsync(config);
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