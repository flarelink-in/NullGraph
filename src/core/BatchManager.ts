import { RenderPassNode } from "./RenderPass";
import { Material, PipelineConfig, RenderBatch, ComputeStageConfig, ComputeStageNode } from "./types";
import { WebGPUContext } from "./WebGPUContext";

export class BatchManager {
    constructor(private ctx: WebGPUContext) {}

    /**
     * Creates a pure rendering batch. No hidden compute execution.
     */
    public createBatch(pass: RenderPassNode, config: PipelineConfig): RenderBatch {
        const batch = new RenderBatch();
        batch.stride = config.strideFloats;
        batch.isIndirect = config.isIndirect || false;

        const shaderModule = this.ctx.device.createShaderModule({ code: config.shaderCode });
        const formats = config.targetFormats || [config.targetFormat || this.ctx.format];

        const colorTargets: GPUColorTargetState[] = formats.map(fmt => {
            const target: GPUColorTargetState = { format: fmt };
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
                targets: colorTargets
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

        // Allocate the standard instance storage buffer
        batch.storageBuffer = this.ctx.device.createBuffer({
            size: config.maxInstances * config.strideFloats * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        // INDIRECT BREAKING CHANGE: We only create the buffers. No compute pipelines attached here.
        if (batch.isIndirect) {
            batch.indirectBuffer = this.ctx.device.createBuffer({
                size: 5 * 4,
                usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });

            batch.sourceStorageBuffer = config.sharedSourceBuffer || this.ctx.device.createBuffer({
                size: config.maxInstances * config.strideFloats * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            });
        }

        // Render Bind Group (Group 0)
        batch.bindGroup = this.ctx.device.createBindGroup({
            layout: batch.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.ctx.cameraUniformBuffer } },
                { binding: 1, resource: { buffer: batch.storageBuffer } }
            ]
        });

        batch.maxInstanceCount = config.maxInstances;
        if (config.material) {
            batch.customBindGroups[1] = config.material.applyToBatch(batch);
        }

        pass.addBatch(batch);
        return batch;
    }

    /**
     * NEW: Explicitly create a separate, global compute execution stage.
     */
    public createComputeStage(config: ComputeStageConfig): ComputeStageNode {
        const stage = new ComputeStageNode(config);

        const computeModule = this.ctx.device.createShaderModule({ code: config.shaderCode });
        stage.computePipeline = this.ctx.device.createComputePipeline({
            layout: 'auto',
            compute: { module: computeModule, entryPoint: 'cs_main' }
        });

        return stage;
    }

    /**
     * NEW: Attaches a bind group to a standalone compute stage.
     */
    public setComputePassBindGroup(stage: ComputeStageNode, entries: GPUBindGroupEntry[], groupIndex: number = 0): void {
        stage.customBindGroups[groupIndex] = this.ctx.device.createBindGroup({
            layout: stage.computePipeline.getBindGroupLayout(groupIndex),
            entries: entries
        });
    }

    public clearBatch(pass: RenderPassNode, batch: RenderBatch): void {
        pass.removeBatch(batch);
        if (batch.storageBuffer) batch.storageBuffer.destroy();
        if (batch.indirectBuffer) batch.indirectBuffer.destroy();
    }

    public setBatchGeometry(batch: RenderBatch, vertexBuffer: GPUBuffer, indexBuffer: GPUBuffer, indexCount: number, format: GPUIndexFormat = 'uint16') {
        batch.vertexBuffer = vertexBuffer;
        batch.indexBuffer = indexBuffer;
        batch.indexCount = indexCount;
        batch.indexFormat = format;
    }

    public updateBatchData(batch: RenderBatch, rawData: Float32Array, instanceCount: number) {
        batch.currentInstanceCount = instanceCount;
        const targetBuffer = (batch.isIndirect && batch.sourceStorageBuffer) ? batch.sourceStorageBuffer : batch.storageBuffer;
        this.ctx.device.queue.writeBuffer(targetBuffer, 0, rawData.buffer, rawData.byteOffset, instanceCount * batch.stride * 4);
    }

    public createTextureBindGroup(batch: RenderBatch, textureView: GPUTextureView | GPUTextureView[], sampler: GPUSampler, extraEntries: GPUBindGroupEntry[] = [], groupIndex: number = 1): GPUBindGroup {
        const views = Array.isArray(textureView) ? textureView : [textureView];
        const entries: GPUBindGroupEntry[] = [{ binding: 0, resource: sampler }];
        views.forEach((view, i) => { entries.push({ binding: i + 1, resource: view }); });
        if (extraEntries.length > 0) entries.push(...extraEntries);

        const bindGroup = this.ctx.device.createBindGroup({
            layout: batch.pipeline.getBindGroupLayout(groupIndex),
            entries: entries
        });
        batch.customBindGroups[groupIndex] = bindGroup;
        return bindGroup;
    }

    public setRenderPassBindGroup(batch: RenderBatch, entries: GPUBindGroupEntry[], groupIndex: number = 1): void {
        if (batch.pipeline) {
            batch.customBindGroups[groupIndex] = this.ctx.device.createBindGroup({
                layout: batch.pipeline.getBindGroupLayout(groupIndex),
                entries: entries
            });
        }
    }

    public attachMaterial(batch: RenderBatch, material: Material, groupIndex: number = 1): GPUBindGroup {
        const bindGroup = material.applyToBatch(batch);
        batch.customBindGroups[groupIndex] = bindGroup;
        return bindGroup;
    }
}