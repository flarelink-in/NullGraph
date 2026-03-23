// src/NullGraph.ts
import { Camera } from './Camera';
import { BufferManager } from './BufferManager';

export interface PipelineConfig {
    shaderCode: string;
    strideFloats: number;
    maxInstances: number;
    vertexLayouts?: GPUVertexBufferLayout[];
    topology?: GPUPrimitiveTopology;
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
}

export class NullGraph {
    public device!: GPUDevice;
    private context!: GPUCanvasContext;
    private format!: GPUTextureFormat;

    private cameraUniformBuffer!: GPUBuffer;
    private depthTexture!: GPUTexture;
    public bufferManager!: BufferManager;

    // NEW: The Render Queue!
    private batches: RenderBatch[] = [];

    public async init(canvas: HTMLCanvasElement) {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) throw new Error("WebGPU not supported!");
        this.device = await adapter.requestDevice();
        this.context = canvas.getContext('webgpu') as GPUCanvasContext;

        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format: this.format });

        this.cameraUniformBuffer = this.device.createBuffer({
            size: 16 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.setupDepthTexture(canvas.width, canvas.height);
        this.bufferManager = new BufferManager(this.device);
    }

    private setupDepthTexture(width: number, height: number) {
        if (this.depthTexture) this.depthTexture.destroy();
        this.depthTexture = this.device.createTexture({
            size: [width, height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }

    public resize(width: number, height: number) {
        this.setupDepthTexture(width, height);
    }

    public clearBatches() {
        this.batches = [];
    }

    public createBatch(config: PipelineConfig): RenderBatch {
        const batch = new RenderBatch();
        batch.stride = config.strideFloats;

        const shaderModule = this.device.createShaderModule({ code: config.shaderCode });

        batch.pipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: { module: shaderModule, entryPoint: 'vs_main', buffers: config.vertexLayouts },
            fragment: { module: shaderModule, entryPoint: 'fs_main', targets: [{ format: this.format }] },
            primitive: { topology: config.topology || 'triangle-list' },
            depthStencil: { depthWriteEnabled: true, depthCompare: 'less', format: 'depth24plus' }
        });

        batch.storageBuffer = this.device.createBuffer({
            size: config.maxInstances * config.strideFloats * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        batch.bindGroup = this.device.createBindGroup({
            layout: batch.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
                { binding: 1, resource: { buffer: batch.storageBuffer } } // Each batch gets its own ECS memory!
            ]
        });

        this.batches.push(batch); // Add to the render queue
        return batch;
    }

    public setBatchGeometry(batch: RenderBatch, vertexBuffer: GPUBuffer, indexBuffer: GPUBuffer, indexCount: number, format: GPUIndexFormat = 'uint16') {
        batch.vertexBuffer = vertexBuffer;
        batch.indexBuffer = indexBuffer;
        batch.indexCount = indexCount;
        batch.indexFormat = format;
    }

    public updateBatchData(batch: RenderBatch, rawData: Float32Array, instanceCount: number) {
        batch.currentInstanceCount = instanceCount;
        this.device.queue.writeBuffer(
            batch.storageBuffer,
            0,
            rawData.buffer,
            rawData.byteOffset,
            instanceCount * batch.stride * 4
        );
    }

    public updateCamera(camera: Camera) {
        this.device.queue.writeBuffer(
            this.cameraUniformBuffer, 0, camera.bufferData.buffer, camera.bufferData.byteOffset, 16 * 4
        );
    }

    public render() {
        if (this.batches.length === 0) return;

        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 },
                loadOp: 'clear', storeOp: 'store',
            }],
            depthStencilAttachment: {
                view: this.depthTexture.createView(),
                depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store',
            }
        });

        for (const batch of this.batches) {
            if (batch.currentInstanceCount === 0) continue;

            passEncoder.setPipeline(batch.pipeline);
            passEncoder.setBindGroup(0, batch.bindGroup);

            if (batch.vertexBuffer && batch.indexBuffer) {
                // 3D Object Path
                passEncoder.setVertexBuffer(0, batch.vertexBuffer);
                passEncoder.setIndexBuffer(batch.indexBuffer, batch.indexFormat);
                passEncoder.drawIndexed(batch.indexCount, batch.currentInstanceCount, 0, 0, 0);
            } else {
                // 2D Shader Triangle Path (For older demos)
                passEncoder.draw(3, batch.currentInstanceCount, 0, 0);
            }
        }

        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }
}