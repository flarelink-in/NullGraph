// null-graph/geometry/Geometry.ts

import {VertexLayout} from "./VertexLayout";
import {NullGraph} from "../../core";

export class Geometry {
    public layout: VertexLayout;
    public vertices: Float32Array;
    public indices: Uint16Array | Uint32Array;

    // The WebGPU buffers (generated later)
    public vertexBuffer: GPUBuffer | null = null;
    public indexBuffer: GPUBuffer | null = null;

    constructor(layout: VertexLayout, vertices: Float32Array, indices: Uint16Array) {
        this.layout = layout;
        this.vertices = vertices;
        this.indices = indices;
    }

    // Automatically uses your BufferManager to upload data to the GPU
    public upload(engine: NullGraph) {
        this.vertexBuffer = engine.bufferManager.createVertexBuffer(this.vertices);
        this.indexBuffer = engine.bufferManager.createIndexBuffer(this.indices);
    }
}