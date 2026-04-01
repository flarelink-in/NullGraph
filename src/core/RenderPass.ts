import {RenderBatch} from "./NullGraph";

export interface RenderPassConfig {
    name: string;
    // Set to true if this pass should draw directly to the HTML Canvas
    isMainScreenPass: boolean;
    // Custom targets for offscreen rendering (G-Buffers, Post-Processing)
    colorAttachments?: GPURenderPassColorAttachment[];
    // Custom depth target (e.g., for Shadow Maps)
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

    public addBatch(batch: RenderBatch) {
        this.batches.push(batch);
    }
}