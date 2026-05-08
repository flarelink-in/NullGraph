import { Camera } from './Camera';
import { BufferManager } from './BufferManager';
import { RenderPassConfig, RenderPassNode } from "./RenderPass";
import {EngineInitResult, PipelineConfig, RenderBatch} from "./types";
export * from "./types"
// Internal Modules
import { WebGPUContext } from './WebGPUContext';
import { BatchManager } from './BatchManager';
import { RenderPipeline } from './RenderPipeline';
import {TextureManager} from "./TextureManager";

export class NullGraph {
    // Public API Contracts (Unchanged)
    public device!: GPUDevice;
    public bufferManager!: BufferManager;

    // Internal State
    private gpuCtx = new WebGPUContext();
    private batchManager!: BatchManager;
    private renderPipeline!: RenderPipeline;
    public textureManager!: TextureManager;

    private passes: RenderPassNode[] = [];
    private batches: RenderBatch[] = [];

    public async init(
        canvas: HTMLCanvasElement,
        options: { desiredFeatures?: GPUFeatureName[] } = {}
    ): Promise<EngineInitResult> {
        try {
            // 1. Pass the desired features to the context
            const initInfo = await this.gpuCtx.init(canvas, options.desiredFeatures || []);

            // 2. Setup the public device and managers
            this.device = this.gpuCtx.device;
            this.bufferManager = new BufferManager(this.device);
            this.batchManager = new BatchManager(this.gpuCtx);
            this.renderPipeline = new RenderPipeline(this.gpuCtx);
            this.textureManager = new TextureManager(this.device);

            return {
                success: true,
                enabledFeatures: this.device.features // This is a native Set of strings
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

    public clearBatch(pass: RenderPassNode, batch: RenderBatch): void {
        // 1. Delegate to BatchManager for pipeline/resource cleanup
        if (typeof this.batchManager.clearBatch === 'function') {
            this.batchManager.clearBatch(pass, batch);
        }

        // 2. Remove from the local NullGraph tracking array
        const globalIndex = this.batches.indexOf(batch);
        if (globalIndex !== -1) {
            this.batches.splice(globalIndex, 1);
        }
    }

    public setBatchGeometry(batch: RenderBatch, vertexBuffer: GPUBuffer, indexBuffer: GPUBuffer, indexCount: number, format: GPUIndexFormat = 'uint16'): void {
        this.batchManager.setBatchGeometry(batch, vertexBuffer, indexBuffer, indexCount, format);
    }

    public updateBatchData(batch: RenderBatch, rawData: Float32Array, instanceCount: number): void {
        this.batchManager.updateBatchData(batch, rawData, instanceCount);
    }

    public attachTextureMaterial(
        batch: RenderBatch,
        textureView: GPUTextureView | GPUTextureView[],
        sampler: GPUSampler,
        extraEntries: GPUBindGroupEntry[] = [], // NEW: Allow appending buffers!
        groupIndex: number = 1
    ): GPUBindGroup {
        return this.batchManager.attachTextureMaterial(batch, textureView, sampler,extraEntries,groupIndex);
    }

    public attachCustomBindGroup(batch: RenderBatch, entries: GPUBindGroupEntry[], groupIndex: number = 1,
                                 target: 'render' | 'compute' | 'both' = 'render'): void {
        return this.batchManager.attachCustomBindGroup(batch, entries, groupIndex,target);
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