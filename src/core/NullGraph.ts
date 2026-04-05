import { Camera } from './Camera';
import { BufferManager } from './BufferManager';
import { RenderPassConfig, RenderPassNode } from "./RenderPass";
import { PipelineConfig, RenderBatch } from "./types";
export * from "./types"
// Internal Modules
import { WebGPUContext } from './WebGPUContext';
import { BatchManager } from './BatchManager';
import { RenderPipeline } from './RenderPipeline';

export class NullGraph {
    // Public API Contracts (Unchanged)
    public device!: GPUDevice;
    public bufferManager!: BufferManager;

    // Internal State
    private gpuCtx = new WebGPUContext();
    private batchManager!: BatchManager;
    private renderPipeline!: RenderPipeline;

    private passes: RenderPassNode[] = [];
    private batches: RenderBatch[] = [];

    public async init(canvas: HTMLCanvasElement): Promise<void> {
        await this.gpuCtx.init(canvas);

        // Expose the initialized device publicly
        this.device = this.gpuCtx.device;
        this.bufferManager = new BufferManager(this.device);

        // Initialize managers
        this.batchManager = new BatchManager(this.gpuCtx);
        this.renderPipeline = new RenderPipeline(this.gpuCtx);
    }

    public resize(width: number, height: number): void {
        this.gpuCtx.resize(width, height);
    }

    public clearPasses(): void {
        this.passes = [];
    }

    public createPass(config: RenderPassConfig): RenderPassNode {
        const pass = new RenderPassNode(config);
        this.passes.push(pass);
        return pass;
    }

    public clearBatches(): void {
        this.batches = [];
    }

    // --- DELEGATED BATCH METHODS ---

    public createBatch(pass: RenderPassNode, config: PipelineConfig): RenderBatch {
        return this.batchManager.createBatch(pass, config);
    }

    public setBatchGeometry(batch: RenderBatch, vertexBuffer: GPUBuffer, indexBuffer: GPUBuffer, indexCount: number, format: GPUIndexFormat = 'uint16'): void {
        this.batchManager.setBatchGeometry(batch, vertexBuffer, indexBuffer, indexCount, format);
    }

    public updateBatchData(batch: RenderBatch, rawData: Float32Array, instanceCount: number): void {
        this.batchManager.updateBatchData(batch, rawData, instanceCount);
    }

    public attachTextureMaterial(
        batch: RenderBatch,
        textureView: GPUTextureView | GPUTextureView[], // Allow array here
        sampler: GPUSampler
    ): GPUBindGroup {
        return this.batchManager.attachTextureMaterial(batch, textureView, sampler);
    }

    public attachCustomBindGroup(batch: RenderBatch, entries: GPUBindGroupEntry[]): GPUBindGroup {
        return this.batchManager.attachCustomBindGroup(batch, entries);
    }

    // --- DELEGATED CORE METHODS ---

    public updateCamera(camera: Camera): void {
        this.device.queue.writeBuffer(
            this.gpuCtx.cameraUniformBuffer, 0, camera.bufferData.buffer, camera.bufferData.byteOffset, 20 * 4
        );
    }

    public render(): void {
        this.renderPipeline.render(this.passes);
    }
}