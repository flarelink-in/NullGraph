import {RenderPassNode} from "./RenderPass";
import {Material, PipelineConfig, RenderBatch} from "./types";
import {WebGPUContext} from "./WebGPUContext";

export class BatchManager {
    constructor(private ctx: WebGPUContext) {}

    public createBatch(pass: RenderPassNode, config: PipelineConfig): RenderBatch {
        const batch = new RenderBatch();
        batch.stride = config.strideFloats;
        batch.isIndirect = config.isIndirect || false;

        const shaderModule = this.ctx.device.createShaderModule({ code: config.shaderCode });

        // --- MRT UPDATE: Support array of formats while preserving single format fallback ---
        const formats = config.targetFormats || [config.targetFormat || this.ctx.format];

        const colorTargets: GPUColorTargetState[] = formats.map(fmt => {
            const target: GPUColorTargetState = { format: fmt };
            // Apply blend state (if provided) to all targets
            if (config.blend) {
                target.blend = config.blend;
            }
            return target;
        });

        const expectsDepth = pass.depthStencilAttachment !== undefined || pass.isMainScreenPass;

        batch.pipeline = this.ctx.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_main',
                buffers: config.vertexLayouts || []
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
                targets: colorTargets // Now accepts the dynamically built array
            },
            primitive: {
                topology: config.topology || 'triangle-list'
            },
            ...(expectsDepth ? {
                depthStencil: {
                    depthWriteEnabled: config.depthWriteEnabled !== undefined ? config.depthWriteEnabled : true,
                    depthCompare: config.depthCompare || 'less',
                    format: 'depth24plus'
                }
            } : {})
        });

        batch.storageBuffer = this.ctx.device.createBuffer({
            size: config.maxInstances * config.strideFloats * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        if (batch.isIndirect && config.computeShaderCode) {
            batch.indirectBuffer = this.ctx.device.createBuffer({
                size: 5 * 4,
                usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });

            batch.sourceStorageBuffer = config.sharedSourceBuffer || this.ctx.device.createBuffer({
                size: config.maxInstances * config.strideFloats * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST| GPUBufferUsage.COPY_SRC,
            });
            const computeModule = this.ctx.device.createShaderModule({ code: config.computeShaderCode });
            batch.computePipeline = this.ctx.device.createComputePipeline({
                layout: 'auto',
                compute: { module: computeModule, entryPoint: 'cs_main' }
            });

            batch.computeBindGroup = this.ctx.device.createBindGroup({
                layout: batch.computePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.ctx.cameraUniformBuffer } },
                    { binding: 1, resource: { buffer: batch.sourceStorageBuffer } },
                    { binding: 2, resource: { buffer: batch.storageBuffer } },
                    { binding: 3, resource: { buffer: batch.indirectBuffer } }
                ]
            });
        }

        batch.bindGroup = this.ctx.device.createBindGroup({
            layout: batch.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.ctx.cameraUniformBuffer } },
                { binding: 1, resource: { buffer: batch.storageBuffer } }
            ]
        });
        batch.maxInstanceCount=config.maxInstances
        if (config.material) {
            batch.customBindGroups[1] = config.material.applyToBatch(batch);
        }

        pass.addBatch(batch);
        return batch;
    }

    /**
     * Removes a batch from a render pass and frees its dedicated GPU resources.
     */
    public clearBatch(pass: RenderPassNode, batch: RenderBatch): void {
        // 1. Remove the batch from the RenderPassNode
        if (typeof (pass as any).removeBatch === 'function') {
            (pass as any).removeBatch(batch);
        } else if (pass.batches) {
            // Fallback: manually splice it out if removeBatch doesn't exist
            const index = pass.batches.indexOf(batch);
            if (index !== -1) {
                pass.batches.splice(index, 1);
            }
        }

        // 2. Destroy dedicated GPU Buffers to free VRAM
        if (batch.storageBuffer) {
            batch.storageBuffer.destroy();
        }
        if (batch.indirectBuffer) {
            batch.indirectBuffer.destroy();
        }

        // Note: We deliberately DO NOT destroy batch.sourceStorageBuffer here
        // because it might be a shared buffer (config.sharedSourceBuffer).
        // If it was shared, destroying it would break other batches using it.
    }

    public setBatchGeometry(batch: RenderBatch, vertexBuffer: GPUBuffer, indexBuffer: GPUBuffer, indexCount: number, format: GPUIndexFormat = 'uint16') {
        batch.vertexBuffer = vertexBuffer;
        batch.indexBuffer = indexBuffer;
        batch.indexCount = indexCount;
        batch.indexFormat = format;
    }

    public updateBatchData(batch: RenderBatch, rawData: Float32Array, instanceCount: number) {
        batch.currentInstanceCount = instanceCount;
        const targetBuffer = (batch.isIndirect && batch.sourceStorageBuffer)
            ? batch.sourceStorageBuffer
            : batch.storageBuffer;

        this.ctx.device.queue.writeBuffer(
            targetBuffer, 0, rawData.buffer, rawData.byteOffset, instanceCount * batch.stride * 4
        );
    }

    // Inside your NullGraph core (or Context/Device manager)
    public attachTextureMaterial(
        batch: RenderBatch,
        textureView: GPUTextureView | GPUTextureView[],
        sampler: GPUSampler,
        extraEntries: GPUBindGroupEntry[] = [], // NEW: Allow appending buffers!
        groupIndex: number = 1
    ): GPUBindGroup {
        const views = Array.isArray(textureView) ? textureView : [textureView];

        // 1. Sampler is ALWAYS binding 0
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: sampler }
        ];

        // 2. Textures are 1, 2, 3...
        views.forEach((view, i) => {
            entries.push({ binding: i + 1, resource: view });
        });

        // 3. Append anything else (like Material Uniforms!)
        if (extraEntries.length > 0) {
            entries.push(...extraEntries);
        }

        const bindGroup = this.ctx.device.createBindGroup({
            layout: batch.pipeline.getBindGroupLayout(groupIndex),
            entries: entries
        });

        batch.customBindGroups[groupIndex] = bindGroup;
        return bindGroup;
    }
    public attachCustomBindGroup(
        batch: RenderBatch,
        entries: GPUBindGroupEntry[],
        groupIndex: number = 1,
        target: 'render' | 'compute' | 'both' = 'render' // NEW: Explicit targeting!
    ): void {

        // 1. Attach to Render Pipeline (if requested)
        if ((target === 'render' || target === 'both') && batch.pipeline) {
            batch.customBindGroups[groupIndex] = this.ctx.device.createBindGroup({
                layout: batch.pipeline.getBindGroupLayout(groupIndex),
                entries: entries
            });
        }

        // 2. Attach to Compute Pipeline (if requested)
        if ((target === 'compute' || target === 'both') && batch.computePipeline) {
            if (!batch.computeCustomBindGroups) batch.computeCustomBindGroups = {};

            batch.computeCustomBindGroups[groupIndex] = this.ctx.device.createBindGroup({
                layout: batch.computePipeline.getBindGroupLayout(groupIndex),
                entries: entries
            });
        }
    }
    /**
     * Attaches a generic Material (like StandardPBRMaterial) to a batch.
     */
    public attachMaterial(batch: RenderBatch, material: Material, groupIndex: number = 1): GPUBindGroup {
        // We pass the batch's compiled pipeline to the material so it can
        // dynamically match the WebGPU layout!
        const bindGroup = material.applyToBatch(batch);
        batch.customBindGroups[groupIndex] = bindGroup;
        return bindGroup;
    }
}