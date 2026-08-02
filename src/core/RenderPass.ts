import { RenderBatch } from "./types";

export interface RenderPassConfig {
    name: string;
    // Set to true if this pass should draw directly to the HTML Canvas
    isMainScreenPass: boolean;
    // Custom targets for offscreen rendering (G-Buffers, Post-Processing, Shadows)
    colorAttachments?: GPURenderPassColorAttachment[];
    // Custom depth target
    depthStencilAttachment?: GPURenderPassDepthStencilAttachment;
}

export class RenderPassNode {
    public name: string;
    public isMainScreenPass: boolean;
    public batches: RenderBatch[] = [];

    public colorAttachments: GPURenderPassColorAttachment[];
    public depthStencilAttachment?: GPURenderPassDepthStencilAttachment;

    constructor(config: RenderPassConfig) {
        this.name = config.name;
        this.isMainScreenPass = config.isMainScreenPass;
        this.colorAttachments = config.colorAttachments || [];
        this.depthStencilAttachment = config.depthStencilAttachment;
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