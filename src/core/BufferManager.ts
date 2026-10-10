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

    /**
     * A `dispatchWorkgroupsIndirect` argument buffer — `count` triples of `[x, y, z]` u32,
     * writable by a compute stage and readable as dispatch arguments.
     *
     * Exists so the correct usage mask is the default path rather than something to remember:
     * `INDIRECT` alone cannot be filled, `STORAGE` alone cannot be dispatched from, and getting
     * either wrong fails as a dropped dispatch with no error and a symptom two subsystems away.
     * `COPY_DST` is included so `preClearBuffers` can reset the triple, which is the ordinary way
     * a counter-driven stage starts its frame.
     *
     * Initialized to `[0, 0, 0]` — a stage whose producer has not run yet dispatches nothing,
     * which is the safe direction. The alternative, uninitialized memory read as a workgroup
     * count, is a hang.
     *
     * **The producing kernel must clamp.** Nothing here can validate a value the GPU wrote, so
     * keeping each component at or below `device.limits.maxComputeWorkgroupsPerDimension` is the
     * kernel's job — `min(computed, cap)` in the shader that writes the triple.
     */
    public createDispatchIndirectBuffer(count: number = 1, label?: string): GPUBuffer {
        if (!Number.isInteger(count) || count < 1) {
            throw new Error(`BufferManager: createDispatchIndirectBuffer count ${count} must be a positive integer.`);
        }
        const buffer = this.device.createBuffer({
            label: label ?? 'NullGraph_DispatchIndirect',
            size: count * 12,
            usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(buffer.getMappedRange()).fill(0);
        buffer.unmap();
        return buffer;
    }
}