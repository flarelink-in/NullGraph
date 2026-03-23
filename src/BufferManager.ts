// src/BufferManager.ts

export class BufferManager {
    private device: GPUDevice;

    constructor(device: GPUDevice) {
        this.device = device;
    }

    // Creates a VBO (Vertex Buffer Object)
    public createVertexBuffer(data: Float32Array): GPUBuffer {
        const buffer = this.device.createBuffer({
            size: data.byteLength,
            // Align memory to 4 bytes as required by WebGPU
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(buffer.getMappedRange()).set(data);
        buffer.unmap();
        return buffer;
    }

    // Creates an IBO (Index Buffer Object)
    public createIndexBuffer(data: Uint16Array | Uint32Array): GPUBuffer {
        const buffer = this.device.createBuffer({
            size: Math.ceil(data.byteLength / 4) * 4, // Ensure 4-byte alignment
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });

        if (data instanceof Uint16Array) {
            new Uint16Array(buffer.getMappedRange()).set(data);
        } else {
            new Uint32Array(buffer.getMappedRange()).set(data);
        }

        buffer.unmap();
        return buffer;
    }
}