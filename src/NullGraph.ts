// src/NullGraph.ts
import { Camera } from './Camera';

export interface PipelineConfig {
    shaderCode: string;
    strideFloats: number;
    maxInstances: number;
    topology?: GPUPrimitiveTopology;
}

export class NullGraph {
    private device!: GPUDevice;
    private context!: GPUCanvasContext;
    private format!: GPUTextureFormat;

    // We now store pipeline data dynamically
    private pipeline!: GPURenderPipeline;
    private storageBuffer!: GPUBuffer;
    private cameraUniformBuffer!: GPUBuffer;
    private bindGroup!: GPUBindGroup;

    private currentStride = 0;
    private currentInstanceCount = 0;

    public async init(canvas: HTMLCanvasElement) {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) throw new Error("WebGPU not supported!");
        this.device = await adapter.requestDevice();
        this.context = canvas.getContext('webgpu') as GPUCanvasContext;

        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format: this.format });

        // Setup Camera Uniform (This is universal, so we keep it here)
        this.cameraUniformBuffer = this.device.createBuffer({
            size: 16 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    // NEW: The user tells the engine how to render their data
    public createPipeline(config: PipelineConfig) {
        this.currentStride = config.strideFloats;

        const shaderModule = this.device.createShaderModule({
            code: config.shaderCode
        });

        this.pipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: { module: shaderModule, entryPoint: 'vs_main' },
            fragment: { module: shaderModule, entryPoint: 'fs_main', targets: [{ format: this.format }] },
            primitive: { topology: config.topology || 'triangle-list' }
        });

        // Create the storage buffer based on user's max instances
        this.storageBuffer = this.device.createBuffer({
            size: config.maxInstances * config.strideFloats * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
                { binding: 1, resource: { buffer: this.storageBuffer } }
            ]
        });
    }

    public updateCamera(camera: Camera) {
        this.device.queue.writeBuffer(
            this.cameraUniformBuffer,
            0,
            camera.bufferData.buffer,
            camera.bufferData.byteOffset,
            16 * 4
        );
    }

    // NEW: Renamed to generic "Data" instead of "Entities"
    public updateData(rawData: Float32Array, instanceCount: number) {
        this.currentInstanceCount = instanceCount;
        this.device.queue.writeBuffer(
            this.storageBuffer,
            0,
            rawData.buffer,
            rawData.byteOffset,
            instanceCount * this.currentStride * 4
        );
    }

    public render() {
        if (this.currentInstanceCount === 0 || !this.pipeline) return;

        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }]
        });

        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, this.bindGroup);
        // Draw 3 vertices per instance
        passEncoder.draw(3, this.currentInstanceCount, 0, 0);
        passEncoder.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }
}