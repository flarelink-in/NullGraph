import { ComputeStageNode } from "./types";
import { RenderPassNode } from "./RenderPass";
import { WebGPUContext } from "./WebGPUContext";

export class RenderPipeline {
    constructor(private ctx: WebGPUContext) {}

    /**
     * Executes a clean sequence of compute stages and render passes linearly.
     */
    public render(pipelineSequence: (ComputeStageNode | RenderPassNode)[]) {
        if (pipelineSequence.length === 0) return;

        const commandEncoder = this.ctx.device.createCommandEncoder();
        let activeComputePass: GPUComputePassEncoder | null = null;

        for (const node of pipelineSequence) {



            // =========================================================
            // 1. PROCESS STANDALONE COMPUTE STAGE
            // =========================================================
            if (node instanceof ComputeStageNode) {

                // A disabled stage costs nothing this frame — no buffer clears, no dispatch.
                // Lets a system gate stages it owns without rebuilding the sequence array.
                if (!node.enabled) continue;

                if (node.preClearBuffers.length > 0) {
                    // If a compute pass is already open, we MUST close it to issue a queue command
                    if (activeComputePass) {
                        activeComputePass.end();
                        activeComputePass = null;
                    }

                    for (const clear of node.preClearBuffers) {
                        this.ctx.device.queue.writeBuffer(
                            clear.buffer,
                            clear.offset,
                            clear.data as BufferSource
                        );
                    }
                }
                // Batch multiple consecutive compute stages into one WebGPU pass
                if (!activeComputePass) {
                    activeComputePass = commandEncoder.beginComputePass();
                }

                activeComputePass.setPipeline(node.computePipeline);

                // Bind all custom data groups mapped to this compute shader
                for (const groupIndexStr in node.customBindGroups) {
                    const index = parseInt(groupIndexStr, 10);
                    activeComputePass.setBindGroup(index, node.customBindGroups[index]);
                }

                // Indirect: the workgroup count comes from a buffer a PRIOR stage wrote, so the
                // dispatch is exact instead of sized against a CPU-known cap. Dispatches within a
                // compute pass are ordered and a storage write is visible to the next dispatch's
                // indirect read, which is why this needs no explicit barrier — the same property
                // consecutive-stage batching already relies on.
                if (node.indirectBuffer) {
                    activeComputePass.dispatchWorkgroupsIndirect(node.indirectBuffer, node.indirectOffset);
                } else {
                    const counts = node.dispatchCount;
                    activeComputePass.dispatchWorkgroups(counts[0], counts[1], counts[2]);
                }
            }

                // =========================================================
                // 2. PROCESS RENDER PASS NODE (Offscreen or Onscreen)
            // =========================================================
            else if (node instanceof RenderPassNode) {
                // If a compute pass was active before this render pass, close it out
                if (activeComputePass) {
                    activeComputePass.end();
                    activeComputePass = null;
                }

                let colorAttachments = node.colorAttachments;
                let depthStencilAttachment = node.depthStencilAttachment;

                // Handle the main presentation swapchain fallback. The screen is the canvas's
                // current texture, or the offscreen screen texture when there is no canvas.
                if (node.isMainScreenPass) {
                    colorAttachments = [{
                        view: this.ctx.getScreenView(),
                        clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 },
                        loadOp: 'clear', storeOp: 'store',
                    }];
                    depthStencilAttachment = {
                        view: this.ctx.depthTexture.createView(),
                        depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store',
                    };
                }

                const passEncoder = commandEncoder.beginRenderPass({
                    colorAttachments: colorAttachments,
                    ...(depthStencilAttachment ? { depthStencilAttachment } : {})
                });

                // Viewport and scissor are PASS state, not per-draw state — once a batch narrows
                // them they stay narrowed for every following batch in this pass. `narrowed`
                // tracks that so a full-pass batch can restore them. `createBatch` guarantees the
                // size is knowable whenever a pass actually mixes the two (BatchManager).
                let narrowed = false;
                const passSize: [number, number] | undefined = node.isMainScreenPass
                    ? [this.ctx.width, this.ctx.height]
                    : (node.size ?? [this.ctx.width, this.ctx.height]);

                // Iterate through the geometric render batches assigned to this pass
                for (const batch of node.batches) {
                    // If it's not indirect and has no instances, bypass it entirely
                    if (!batch.isIndirect && batch.currentInstanceCount === 0) continue;

                    // If a per-batch viewport is suppressed / zero-area, skip drawing to avoid corrupting scissor/rasterizer state
                    if (batch.viewport && (batch.viewport[2] <= 0 || batch.viewport[3] <= 0)) continue;

                    passEncoder.setPipeline(batch.pipeline);
                    passEncoder.setBindGroup(0, batch.bindGroup);

                    // Per-batch viewport (shadows-atlas-design.md §12.1): scope rasterization to
                    // the batch's tile. setScissorRect is belt-and-braces — the viewport
                    // transform already guarantees containment — but it makes the invariant
                    // explicit and costs nothing.
                    if (batch.viewport) {
                        const [vx, vy, vw, vh] = batch.viewport;
                        passEncoder.setViewport(vx, vy, vw, vh, 0, 1);
                        passEncoder.setScissorRect(vx, vy, vw, vh);
                        narrowed = true;
                    } else if (narrowed && passSize) {
                        // Restore the full attachment, or this batch would silently inherit the
                        // previous batch's tile.
                        passEncoder.setViewport(0, 0, passSize[0], passSize[1], 0, 1);
                        passEncoder.setScissorRect(0, 0, passSize[0], passSize[1]);
                        narrowed = false;
                    }

                    if (batch.customBindGroups) {
                        for (const groupIndexStr in batch.customBindGroups) {
                            const index = parseInt(groupIndexStr, 10);
                            passEncoder.setBindGroup(index, batch.customBindGroups[index]);
                        }
                    }

                    if (batch.vertexBuffer) passEncoder.setVertexBuffer(0, batch.vertexBuffer);
                    if (batch.indexBuffer) passEncoder.setIndexBuffer(batch.indexBuffer, batch.indexFormat);

                    if (batch.isIndirect && batch.indirectBuffer) {
                        // The pipeline blindly trusts that a prior ComputeStageNode in the sequence array
                        // filled this indirectBuffer with data.
                        passEncoder.drawIndexedIndirect(batch.indirectBuffer, 0);
                    } else if (batch.vertexBuffer && batch.indexBuffer) {
                        passEncoder.drawIndexed(batch.indexCount, batch.currentInstanceCount, 0, 0, 0);
                    } else {
                        passEncoder.draw(3, batch.currentInstanceCount, 0, 0);
                    }
                }

                passEncoder.end();
            }
        }

        // Close out any lingering compute encoder allocations
        if (activeComputePass) {
            activeComputePass.end();
        }

        // Submit everything to the GPU queue in a single command buffer submission
        this.ctx.device.queue.submit([commandEncoder.finish()]);
    }
}