// src/NullGraph.ts
import { Camera } from './Camera';
import { BufferManager } from './BufferManager';
import {RenderPassConfig, RenderPassNode} from "./RenderPass";

export interface PipelineConfig {
    shaderCode: string;
    strideFloats: number;
    maxInstances: number;
    vertexLayouts?: GPUVertexBufferLayout[];
    topology?: GPUPrimitiveTopology;

    // NEW: Optional properties for GPU-driven rendering
    isIndirect?: boolean;
    computeShaderCode?: string;
    sharedSourceBuffer?: GPUBuffer;
    extraBindGroup?: GPUBindGroup;
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

    // NEW: Indirect & Compute Properties
    public isIndirect: boolean = false;
    public indirectBuffer?: GPUBuffer;
    public sourceStorageBuffer?: GPUBuffer;
    public computePipeline?: GPUComputePipeline;
    public computeBindGroup?: GPUBindGroup;
    public extraBindGroup?: GPUBindGroup;


}
/**
 * Core rendering engine that manages WebGPU state, pipelines, and render loops.
 * Supports both standard instance rendering and GPU-driven indirect rendering via compute shaders.
 */
export class NullGraph {
    public device!: GPUDevice;
    private context!: GPUCanvasContext;
    private format!: GPUTextureFormat;

    private cameraUniformBuffer!: GPUBuffer;
    private depthTexture!: GPUTexture;
    public bufferManager!: BufferManager;

    private batches: RenderBatch[] = [];
    private passes: RenderPassNode[] = [];
    /**
     * Bootstraps the WebGPU device, configures the canvas context,
     * and allocates global resources like the Camera uniform buffer and Depth texture.
     * @param canvas The HTML canvas element to render to.
     */
    public async init(canvas: HTMLCanvasElement) {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) throw new Error("WebGPU not supported!");
        this.device = await adapter.requestDevice();
        this.context = canvas.getContext('webgpu') as GPUCanvasContext;

        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format: this.format });

        this.cameraUniformBuffer = this.device.createBuffer({
            size: 20 * 4, // Upgraded size! (80 bytes)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.setupDepthTexture(canvas.width, canvas.height);
        this.bufferManager = new BufferManager(this.device);
    }

    /**
     * Recreates the depth texture to match the new canvas dimensions.
     * @private
     */
    private setupDepthTexture(width: number, height: number) {
        if (this.depthTexture) this.depthTexture.destroy();
        this.depthTexture = this.device.createTexture({
            size: [width, height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }

    /**
     * Call this when the browser window or canvas resizes to update internal textures.
     * @param width New canvas width in pixels.
     * @param height New canvas height in pixels.
     */
    public resize(width: number, height: number) {
        this.setupDepthTexture(width, height);
    }
    public clearPasses() {
        this.passes = [];
    }

    public createPass(config: RenderPassConfig): RenderPassNode {
        const pass = new RenderPassNode(config);
        this.passes.push(pass);
        return pass;
    }

    /**
     * Removes all current render batches from the pipeline.
     */
    public clearBatches() {
        this.batches = [];
    }

    /**
     * Creates a new rendering pipeline and allocates required GPU storage buffers.
     * If `isIndirect` is true, it also generates the compute pipeline and indirect buffers
     * required for GPU-driven drawing (culling/physics).
     * * @param config The pipeline configuration including shaders, strides, and topology.
     * @returns A RenderBatch object tracking the state and buffers for this specific draw call.
     */
    public createBatch(pass: RenderPassNode, config: PipelineConfig): RenderBatch {
        const batch = new RenderBatch();
        batch.stride = config.strideFloats;
        batch.isIndirect = config.isIndirect || false;
        batch.extraBindGroup = config.extraBindGroup;

        const shaderModule = this.device.createShaderModule({ code: config.shaderCode });

        // Standard Pipeline setup...
        // 1. Determine Color Format
        // In the future, you should add `colorFormat?: GPUTextureFormat` to RenderPassConfig
        // so offscreen passes can explicitly declare what format they are drawing to.
        // For now, we safely default to the canvas format.
        let targetFormat = pass.colorAttachments?.[0]?.view ? this.format : this.format;
        // If it's an offscreen pass with custom color attachments, use the first attachment's format
        if (!pass.isMainScreenPass && pass.colorAttachments.length > 0 && pass.colorAttachments[0].view) {
            // Note: In a robust engine, you'd pass the specific target format in PipelineConfig.
            // We default to the canvas format here for simplicity.
        }
        const expectsDepth = pass.depthStencilAttachment !== undefined || pass.isMainScreenPass;

        batch.pipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_main',
                buffers: config.vertexLayouts || []
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
                targets: [{ format: targetFormat }]
            },
            primitive: {
                topology: config.topology || 'triangle-list'
            },
            // Only inject depthStencil if the pass actually supports it
            ...(expectsDepth ? {
                depthStencil: {
                    depthWriteEnabled: true,
                    depthCompare: 'less',
                    format: 'depth24plus'
                }
            } : {})
        });

        // 3. Standard Storage Buffer
        batch.storageBuffer = this.device.createBuffer({
            size: config.maxInstances * config.strideFloats * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // 4. Compute/Indirect Architecture
        if (batch.isIndirect && config.computeShaderCode) {
            batch.indirectBuffer = this.device.createBuffer({
                size: 5 * 4,
                usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });

            batch.sourceStorageBuffer = config.sharedSourceBuffer || this.device.createBuffer({
                size: config.maxInstances * config.strideFloats * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            const computeModule = this.device.createShaderModule({ code: config.computeShaderCode });
            batch.computePipeline = this.device.createComputePipeline({
                layout: 'auto',
                compute: { module: computeModule, entryPoint: 'cs_main' }
            });

            batch.computeBindGroup = this.device.createBindGroup({
                layout: batch.computePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
                    { binding: 1, resource: { buffer: batch.sourceStorageBuffer } },
                    { binding: 2, resource: { buffer: batch.storageBuffer } },
                    { binding: 3, resource: { buffer: batch.indirectBuffer } }
                ]
            });
        }

        batch.bindGroup = this.device.createBindGroup({
            layout: batch.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
                { binding: 1, resource: { buffer: batch.storageBuffer } }
            ]
        });

        pass.addBatch(batch); // Automatically add to the requested pass
        return batch;
    }

    /**
     * Binds the actual 3D model geometry (vertices and indices) to a specific batch.
     * * @param batch The batch to apply the geometry to.
     * @param batch
     * @param vertexBuffer The GPUBuffer containing vertex data (positions, normals, uvs).
     * @param indexBuffer The GPUBuffer containing the draw order of vertices.
     * @param indexCount The total number of indices to draw.
     * @param format The format of the index buffer (defaults to 'uint16').
     */
    public setBatchGeometry(batch: RenderBatch, vertexBuffer: GPUBuffer, indexBuffer: GPUBuffer, indexCount: number, format: GPUIndexFormat = 'uint16') {
        batch.vertexBuffer = vertexBuffer;
        batch.indexBuffer = indexBuffer;
        batch.indexCount = indexCount;
        batch.indexFormat = format;
    }

    /**
     * Uploads instance data (like positions, colors, matrices) from the CPU to the GPU.
     * Routes data to the Source buffer for Compute passes, or directly to the Storage buffer for standard passes.
     * * @param batch The batch being updated.
     * @param batch
     * @param rawData A flat Float32Array containing all instance data combined.
     * @param instanceCount The number of active instances being drawn.
     */
    public updateBatchData(batch: RenderBatch, rawData: Float32Array, instanceCount: number) {
        batch.currentInstanceCount = instanceCount;

        // FAIL-SAFE: Route data to the Source buffer if Indirect, otherwise standard Storage.
        const targetBuffer = (batch.isIndirect && batch.sourceStorageBuffer)
            ? batch.sourceStorageBuffer
            : batch.storageBuffer;

        this.device.queue.writeBuffer(
            targetBuffer,
            0,
            rawData.buffer,
            rawData.byteOffset,
            instanceCount * batch.stride * 4
        );
    }

    /**
     * Updates the global camera uniform buffer. This is shared across all render batches.
     * @param camera The camera object containing the updated view/projection matrices.
     */
    public updateCamera(camera: Camera) {
        this.device.queue.writeBuffer(
            this.cameraUniformBuffer, 0, camera.bufferData.buffer, camera.bufferData.byteOffset, 20 * 4
        );
    }

    /**
     * Executes the main WebGPU command sequence.
     * Phase 1: Dispatches compute shaders for indirect batches to process data/culling.
     * Phase 2: Executes the render pass, routing each batch to Indirect, Indexed, or standard Draw calls.
     */
    public render() {
        if (this.passes.length === 0) return;

        const commandEncoder = this.device.createCommandEncoder();
        const zeroArray = new Uint32Array([0]);

        // Iterate through each pass sequentially
        for (const pass of this.passes) {

            // --- PHASE 1: COMPUTE (Indirect Batches in THIS Pass) ---
            let hasCompute = false;
            for (const batch of pass.batches) {
                if (batch.isIndirect && batch.indirectBuffer) {
                    this.device.queue.writeBuffer(batch.indirectBuffer, 4, zeroArray);
                    hasCompute = true;
                }
            }

            if (hasCompute) {
                const computePass = commandEncoder.beginComputePass();
                for (const batch of pass.batches) {
                    if (batch.isIndirect && batch.computePipeline && batch.computeBindGroup) {
                        computePass.setPipeline(batch.computePipeline);
                        computePass.setBindGroup(0, batch.computeBindGroup);
                        const workgroupCount = Math.ceil(batch.currentInstanceCount / 64);
                        if (workgroupCount > 0) computePass.dispatchWorkgroups(workgroupCount);
                    }
                }
                computePass.end();
            }

            // --- PHASE 2: RENDER PASS SETUP ---
            let colorAttachments = pass.colorAttachments;
            let depthStencilAttachment = pass.depthStencilAttachment;

            // Dynamically grab the canvas texture view if this is the main screen pass
            if (pass.isMainScreenPass) {
                colorAttachments = [{
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 },
                    loadOp: 'clear', storeOp: 'store',
                }];
                depthStencilAttachment = {
                    view: this.depthTexture.createView(),
                    depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store',
                };
            }

            const passEncoder = commandEncoder.beginRenderPass({
                colorAttachments: colorAttachments,
                ...(depthStencilAttachment ? { depthStencilAttachment } : {}) // Only add depth if it exists
            });

            // --- PHASE 3: DRAW BATCHES ---
            for (const batch of pass.batches) {
                if (!batch.isIndirect && batch.currentInstanceCount === 0) continue;

                passEncoder.setPipeline(batch.pipeline);
                passEncoder.setBindGroup(0, batch.bindGroup);
                if (batch.extraBindGroup) {
                    passEncoder.setBindGroup(1, batch.extraBindGroup);
                }

                if (batch.vertexBuffer) passEncoder.setVertexBuffer(0, batch.vertexBuffer);
                if (batch.indexBuffer) passEncoder.setIndexBuffer(batch.indexBuffer, batch.indexFormat);

                if (batch.isIndirect && batch.indirectBuffer) {
                    passEncoder.drawIndexedIndirect(batch.indirectBuffer, 0);
                } else if (batch.vertexBuffer && batch.indexBuffer) {
                    passEncoder.drawIndexed(batch.indexCount, batch.currentInstanceCount, 0, 0, 0);
                } else {
                    passEncoder.draw(3, batch.currentInstanceCount, 0, 0);
                }
            }

            passEncoder.end();
        }

        this.device.queue.submit([commandEncoder.finish()]);
    }
    public attachTextureMaterial(
        batch: RenderBatch,
        textureView: GPUTextureView,
        sampler: GPUSampler
    ): GPUBindGroup {
        // Create the bind group using the layout automatically inferred from your WGSL shader
        const bindGroup = this.device.createBindGroup({
            layout: batch.pipeline.getBindGroupLayout(1), // @group(1)
            entries: [
                { binding: 0, resource: textureView }, // @binding(0)
                { binding: 1, resource: sampler }      // @binding(1)
            ]
        });

        // Assign it to the batch so the render loop picks it up
        batch.extraBindGroup = bindGroup;

        return bindGroup;
    }

    public attachCustomBindGroup(
        batch: RenderBatch,
        entries: GPUBindGroupEntry[]
    ): GPUBindGroup {
        const bindGroup = this.device.createBindGroup({
            layout: batch.pipeline.getBindGroupLayout(1),
            entries: entries
        });

        batch.extraBindGroup = bindGroup;
        return bindGroup;
    }
}