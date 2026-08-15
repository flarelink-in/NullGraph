import { RenderBatch } from "./types";

export interface RenderPassConfig {
    name: string;
    // Set to true if this pass should draw directly to the HTML Canvas
    isMainScreenPass: boolean;
    // Custom targets for offscreen rendering (G-Buffers, Post-Processing, Shadows)
    colorAttachments?: GPURenderPassColorAttachment[];
    // Custom depth target
    depthStencilAttachment?: GPURenderPassDepthStencilAttachment;
    /**
     * Attachment dimensions in pixels, `[width, height]`. Only needed when this pass mixes
     * viewport-scoped batches with full-pass ones: viewport/scissor state persists across draws
     * within a render pass, so `RenderPipeline` needs the attachment size to restore the full
     * viewport between them. A `GPUTextureView` does not expose its dimensions, hence the config.
     *
     * Omit it when every batch in the pass has a viewport, or none does — `createBatch` throws
     * naming this pass if it ever sees a mix without a size.
     */
    size?: [width: number, height: number];
}

export class RenderPassNode {
    public name: string;
    public isMainScreenPass: boolean;
    public batches: RenderBatch[] = [];

    public colorAttachments: GPURenderPassColorAttachment[];
    public depthStencilAttachment?: GPURenderPassDepthStencilAttachment;
    /** Attachment size in pixels — see {@link RenderPassConfig.size}. */
    public size?: [width: number, height: number];

    constructor(config: RenderPassConfig) {
        this.name = config.name;
        this.isMainScreenPass = config.isMainScreenPass;
        this.colorAttachments = config.colorAttachments || [];
        this.depthStencilAttachment = config.depthStencilAttachment;
        this.size = config.size;
    }

    public addBatch(batch: RenderBatch): void {
        this.batches.push(batch);
    }

    public removeBatch(batch: RenderBatch): void {
        const index = this.batches.indexOf(batch);
        if (index !== -1) {
            this.batches.splice(index, 1);
        }
    }
}