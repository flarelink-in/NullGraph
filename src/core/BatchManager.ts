import { RenderPassNode } from "./RenderPass";
import { PipelineConfig, RenderBatch } from "./types";
import { WebGPUContext } from "./WebGPUContext";

export class BatchManager {
    constructor(private ctx: WebGPUContext) {}

    public createBatch(pass: RenderPassNode, config: PipelineConfig): RenderBatch {
        const batch = new RenderBatch();
        batch.stride = config.strideFloats;
        batch.isIndirect = config.isIndirect || false;
        batch.extraBindGroup = config.extraBindGroup;

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

        pass.addBatch(batch);
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
        const targetBuffer = (batch.isIndirect && batch.sourceStorageBuffer)
            ? batch.sourceStorageBuffer
            : batch.storageBuffer;

        this.ctx.device.queue.writeBuffer(
            targetBuffer, 0, rawData.buffer, rawData.byteOffset, instanceCount * batch.stride * 4
        );
    }

    public attachTextureMaterial(batch: RenderBatch, textureView: GPUTextureView | GPUTextureView[], sampler: GPUSampler): GPUBindGroup {
        // --- MRT UPDATE: Allow array of textures for Deferred bind groups ---
        const views = Array.isArray(textureView) ? textureView : [textureView];

        const entries: GPUBindGroupEntry[] = views.map((view, i) => ({
            binding: i,
            resource: view
        }));

        // Add the sampler as the last binding
        entries.push({
            binding: views.length,
            resource: sampler
        });

        const bindGroup = this.ctx.device.createBindGroup({
            layout: batch.pipeline.getBindGroupLayout(1),
            entries: entries
        });

        batch.extraBindGroup = bindGroup;
        return bindGroup;
    }

    public attachCustomBindGroup(batch: RenderBatch, entries: GPUBindGroupEntry[]): GPUBindGroup {
        const bindGroup = this.ctx.device.createBindGroup({
            layout: batch.pipeline.getBindGroupLayout(1),
            entries: entries
        });
        batch.extraBindGroup = bindGroup;
        return bindGroup;
    }
}