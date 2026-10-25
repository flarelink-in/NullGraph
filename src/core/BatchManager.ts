import { RenderPassNode } from "./RenderPass";
import { Material, PipelineConfig, RenderBatch, ComputeStageConfig, ComputeStageNode } from "./types";
import { WebGPUContext } from "./WebGPUContext";

/** Bytes in a `dispatchWorkgroupsIndirect` argument triple: three `u32`. */
export const DISPATCH_INDIRECT_BYTES = 12;

/**
 * Refuse an indirect dispatch that cannot work, at stage creation, naming the offender.
 *
 * Every condition here fails the same silent way at runtime — the dispatch is dropped, the buffer
 * the kernel was supposed to fill stays at whatever it was, and the symptom appears in whatever
 * *reads* that buffer, one or two subsystems downstream. "Nothing is drawn" does not point at a
 * missing usage flag on a dispatch-args buffer.
 */
function validateIndirectDispatch(config: ComputeStageConfig): void {
    const buffer = config.indirectBuffer;
    if (!buffer) {
        if (config.indirectOffset !== undefined) {
            throw new Error(
                `ComputeStage "${config.name}": indirectOffset was set without an indirectBuffer. ` +
                `The offset addresses a buffer that does not exist; the stage would dispatch from ` +
                `dispatchCount and silently ignore it.`);
        }
        return;
    }

    const offset = config.indirectOffset ?? 0;
    if (!Number.isInteger(offset) || offset < 0 || offset % 4 !== 0) {
        throw new Error(
            `ComputeStage "${config.name}": indirectOffset ${offset} must be a non-negative ` +
            `multiple of 4 — the dispatch arguments are three u32.`);
    }
    if ((buffer.usage & GPUBufferUsage.INDIRECT) === 0) {
        throw new Error(
            `ComputeStage "${config.name}": indirectBuffer lacks GPUBufferUsage.INDIRECT, so the ` +
            `dispatch cannot read its workgroup count. Allocate it with ` +
            `bufferManager.createDispatchIndirectBuffer(), or add INDIRECT | STORAGE to its usage.`);
    }
    // Not fatal on its own — a caller may fill the triple with writeBuffer — but a stage taking its
    // count from a buffer no shader can write is almost always a wiring mistake, and saying so
    // costs nothing.
    if ((buffer.usage & GPUBufferUsage.STORAGE) === 0 && (buffer.usage & GPUBufferUsage.COPY_DST) === 0) {
        throw new Error(
            `ComputeStage "${config.name}": indirectBuffer has neither STORAGE nor COPY_DST usage, ` +
            `so nothing can ever write its workgroup count and the dispatch is permanently zero. ` +
            `Add STORAGE for a compute stage to fill it, or COPY_DST to write it from the CPU.`);
    }
    if (buffer.size < offset + DISPATCH_INDIRECT_BYTES) {
        throw new Error(
            `ComputeStage "${config.name}": indirectBuffer is ${buffer.size} bytes; reading three ` +
            `u32 at offset ${offset} needs ${offset + DISPATCH_INDIRECT_BYTES}.`);
    }
}

export class BatchManager {
    /**
     * The buffers this manager allocated for each live batch (offline-films-browser-design.md §12). A
     * batch's own instance, indirect and source buffers are listed; a `sharedSourceBuffer`, geometry
     * and a camera buffer are the caller's and never are, so clearing or destroying never frees them.
     */
    private owned = new Map<RenderBatch, GPUBuffer[]>();

    constructor(private ctx: WebGPUContext) {}

    /**
     * Creates a pure rendering batch. No hidden compute execution.
     */
    public createBatch(pass: RenderPassNode, config: PipelineConfig): RenderBatch {
        const batch = new RenderBatch();
        batch.stride = config.strideFloats;
        batch.isIndirect = config.isIndirect || false;
        // Per-batch viewport (shadows-atlas-design.md §12.1) — undefined for every existing batch.
        batch.viewport = config.viewport;

        // Viewport/scissor persist across draws within a render pass, so a full-pass batch that
        // follows a viewport-scoped one silently inherits its tile. RenderPipeline restores the
        // full viewport between them — but only if it knows the attachment size. Refuse the
        // combination it cannot fix, naming the pass, rather than rendering into the wrong rect.
        if (!pass.isMainScreenPass && !pass.size && pass.batches.length > 0) {
            const scoped = batch.viewport !== undefined;
            if (pass.batches.some(b => (b.viewport !== undefined) !== scoped)) {
                throw new Error(
                    `BatchManager: pass "${pass.name}" mixes viewport-scoped and full-pass batches ` +
                    `but has no size, so the full viewport cannot be restored between them ` +
                    `(viewport state persists across draws in a pass). Give the pass a ` +
                    `size: [width, height], or give every batch in it an explicit viewport.`);
            }
        }

        // Labels are free and are what a WebGPU error message quotes back: "[RenderPipeline
        // "scene #2"]" points at a batch, an unlabelled one points at nothing.
        const label = config.label ?? `${pass.name} #${pass.batches.length}`;
        const shaderModule = this.ctx.device.createShaderModule({ label, code: config.shaderCode });
        // depthOnly (shadows-design.md §12.1): no color targets at all — a fragment-less render
        // pipeline is core WebGPU, and a pipeline whose color-target count disagrees with the
        // pass is a validation error.
        const formats = config.depthOnly ? [] : (config.targetFormats || [config.targetFormat || this.ctx.format]);

        const colorTargets: GPUColorTargetState[] = formats.map(fmt => {
            const target: GPUColorTargetState = { format: fmt };
            if (config.blend) {
                target.blend = config.blend;
            }
            return target;
        });

        const expectsDepth = pass.depthStencilAttachment !== undefined || pass.isMainScreenPass;

        batch.pipeline = this.ctx.device.createRenderPipeline({
            label,
            layout: 'auto',
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_main',
                buffers: config.vertexLayouts || []
            },
            ...(config.depthOnly ? {} : {
                fragment: {
                    module: shaderModule,
                    entryPoint: 'fs_main',
                    targets: colorTargets
                }
            }),
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

        const owned = [batch.storageBuffer];
        this.owned.set(batch, owned);

        // INDIRECT BREAKING CHANGE: We only create the buffers. No compute pipelines attached here.
        if (batch.isIndirect) {
            batch.indirectBuffer = this.ctx.device.createBuffer({
                size: 5 * 4,
                usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            owned.push(batch.indirectBuffer);

            if (config.sharedSourceBuffer) {
                batch.sourceStorageBuffer = config.sharedSourceBuffer;
            } else {
                batch.sourceStorageBuffer = this.ctx.device.createBuffer({
                    size: config.maxInstances * config.strideFloats * 4,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                });
                owned.push(batch.sourceStorageBuffer);
            }
        }

        // Render Bind Group (Group 0). The camera buffer is overridable per batch
        // (shadows-design.md §12.1) — a shadow cascade IS a camera, and this is the line that
        // makes pointing the cull+draw chain at it possible.
        batch.bindGroup = this.ctx.device.createBindGroup({
            layout: batch.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: config.cameraBuffer || this.ctx.cameraUniformBuffer } },
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
        validateIndirectDispatch(config);
        const stage = new ComputeStageNode(config);

        // Labelled with the stage name so a WebGPU validation or compile error names the stage.
        const computeModule = this.ctx.device.createShaderModule({ label: config.name, code: config.shaderCode });
        stage.computePipeline = this.ctx.device.createComputePipeline({
            label: config.name,
            layout: 'auto',
            compute: { module: computeModule, entryPoint: 'cs_main' }
        });

        return stage;
    }

    /**
     * As createComputeStage, but the pipeline comes from createComputePipelineAsync: its compile
     * does not hold the caller (agents-ort-design.md §12.1). A shader that does not compile rejects,
     * naming the stage, with the device's message; it raises no uncaptured error.
     */
    public async createComputeStageAsync(config: ComputeStageConfig): Promise<ComputeStageNode> {
        validateIndirectDispatch(config);
        const stage = new ComputeStageNode(config), device = this.ctx.device;
        // The scope catches the module's own error, which says what in the WGSL is wrong; the
        // pipeline's rejection only says the module is invalid.
        device.pushErrorScope('validation');
        const module = device.createShaderModule({ label: config.name, code: config.shaderCode });
        const pipeline = device.createComputePipelineAsync({ label: config.name, layout: 'auto', compute: { module, entryPoint: 'cs_main' } });
        const scope = device.popErrorScope();
        try {
            stage.computePipeline = await pipeline;
        } catch (error) {
            throw new Error(`NullGraph: compute stage "${config.name}" did not compile: ${((await scope) ?? error as Error).message}`);
        }
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

    /** Remove the batch from its pass and destroy the buffers this manager allocated for it, and only those. */
    public clearBatch(pass: RenderPassNode, batch: RenderBatch): void {
        pass.removeBatch(batch);
        for (const buffer of this.owned.get(batch) ?? []) buffer.destroy();
        this.owned.delete(batch);
    }

    /**
     * Destroy every buffer this manager allocated, for the batches still live (engine teardown). Every
     * release is attempted; failures are thrown together afterwards.
     */
    public destroy(): void {
        const errors: unknown[] = [];
        for (const buffers of this.owned.values()) {
            for (const buffer of buffers) {
                try { buffer.destroy(); } catch (e) { errors.push(e); }
            }
        }
        this.owned.clear();
        if (errors.length === 1) throw errors[0];
        if (errors.length > 1) throw new AggregateError(errors, `BatchManager.destroy: ${errors.length} releases failed.`);
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