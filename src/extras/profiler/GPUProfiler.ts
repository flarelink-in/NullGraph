export class GPUProfiler {
    private device: GPUDevice;
    private querySet: GPUQuerySet;
    private resolveBuf: GPUBuffer;
    private readbackBuf: GPUBuffer;
    private isMapping = false;

    constructor(device: GPUDevice) {
        this.device = device;

        this.querySet = device.createQuerySet({ type: "timestamp", count: 2 });

        this.resolveBuf = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });

        this.readbackBuf = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
    }

    public begin() {
        if (this.isMapping) return;

        const enc = this.device.createCommandEncoder();

        // MODERN WEBGPU: Use a dummy compute pass to write the START timestamp
        const pass = enc.beginComputePass({
            timestampWrites: {
                querySet: this.querySet,
                beginningOfPassWriteIndex: 0 // Record time at index 0
            }
        });
        pass.end();

        this.device.queue.submit([enc.finish()]);
    }

    public end(onUpdate: (gpuTimeMs: number) => void) {
        if (this.isMapping) return;

        const enc = this.device.createCommandEncoder();

        // MODERN WEBGPU: Use a dummy compute pass to write the END timestamp
        const pass = enc.beginComputePass({
            timestampWrites: {
                querySet: this.querySet,
                endOfPassWriteIndex: 1 // Record time at index 1
            }
        });
        pass.end();

        // Resolve the queries to the GPU buffer
        enc.resolveQuerySet(this.querySet, 0, 2, this.resolveBuf, 0);

        // Copy to the CPU-readable buffer
        enc.copyBufferToBuffer(this.resolveBuf, 0, this.readbackBuf, 0, 16);

        this.device.queue.submit([enc.finish()]);

        // ==========================================
        // ASYNCHRONOUS NON-BLOCKING READ
        // ==========================================
        this.isMapping = true;

        this.readbackBuf.mapAsync(GPUMapMode.READ).then(() => {
            const times = new BigInt64Array(this.readbackBuf.getMappedRange());

            // Calculate duration in nanoseconds, then convert to milliseconds
            const durationNs = Number(times[1] - times[0]);
            const durationMs = durationNs / 1_000_000;

            this.readbackBuf.unmap();
            this.isMapping = false;

            if (durationMs > 0) {
                onUpdate(durationMs);
            }
        });
    }
}