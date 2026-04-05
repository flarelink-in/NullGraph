import { RenderPassNode } from "./RenderPass";
import { WebGPUContext } from "./WebGPUContext";

export class RenderPipeline {
    constructor(private ctx: WebGPUContext) {}

    public render(passes: RenderPassNode[]) {
        if (passes.length === 0) return;

        const commandEncoder = this.ctx.device.createCommandEncoder();
        const zeroArray = new Uint32Array([0]);

        for (const pass of passes) {
            let hasCompute = false;
            for (const batch of pass.batches) {
                if (batch.isIndirect && batch.indirectBuffer) {
                    this.ctx.device.queue.writeBuffer(batch.indirectBuffer, 4, zeroArray);
                    hasCompute = true;
                }
            }

            if (hasCompute) {
                const computePass = commandEncoder.beginComputePass();
                for (const batch of pass.batches) {
                    if (batch.isIndirect && batch.computePipeline && batch.computeBindGroup) {
                        computePass.setPipeline(batch.computePipeline);
                        computePass.setBindGroup(0, batch.computeBindGroup);
                        let workgroupCount=0;

                            workgroupCount = Math.ceil(batch.currentInstanceCount / 64);

                        if (workgroupCount > 0) computePass.dispatchWorkgroups(workgroupCount);
                    }
                }
                computePass.end();
            }

            let colorAttachments = pass.colorAttachments;
            let depthStencilAttachment = pass.depthStencilAttachment;

            if (pass.isMainScreenPass) {
                colorAttachments = [{
                    view: this.ctx.context.getCurrentTexture().createView(),
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

        this.ctx.device.queue.submit([commandEncoder.finish()]);
    }
}