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

    constructor(layout: VertexLayout, vertices: Float32Array, indices: Uint16Array | Uint32Array) {
        this.layout = layout;
        this.vertices = vertices;
        this.indices = indices;
    }

    /**
     * The GPU index format this geometry's index buffer must be drawn with. Some primitives
     * emit 32-bit indices (Cylinder/Cone/Capsule/Ring) and some 16-bit (Sphere/Cube/…), so
     * ALWAYS pass this to setBatchGeometry rather than relying on its 'uint16' default:
     *   engine.setBatchGeometry(batch, geom.vertexBuffer!, geom.indexBuffer!, geom.indices.length, geom.indexFormat);
     * Drawing a Uint32Array index buffer as 'uint16' misreads every index and covers only half
     * the buffer — the classic "spiky, half-missing" mesh.
     */
    public get indexFormat(): GPUIndexFormat {
        return this.indices instanceof Uint32Array ? 'uint32' : 'uint16';
    }

    // Automatically uses your BufferManager to upload data to the GPU
    public upload(engine: NullGraph) {
        this.vertexBuffer = engine.bufferManager.createVertexBuffer(this.vertices);
        this.indexBuffer = engine.bufferManager.createIndexBuffer(this.indices);
    }
    public destroy() {
        if (this.vertexBuffer) {
            this.vertexBuffer.destroy();
            this.vertexBuffer = null; // Prevent accidental reuse
        }

        if (this.indexBuffer) {
            this.indexBuffer.destroy();
            this.indexBuffer = null; // Prevent accidental reuse
        }
    }
}