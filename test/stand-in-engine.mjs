// A stand-in for an engine, for tests that run a guide's example without a GPU: buffers and uploads
// are inert, and what an example builds is recorded. `stages` holds every compute stage that was
// created; `buffers` every buffer, each with a `destroyed` flag.
globalThis.GPUBufferUsage ??= { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64 };

export function standIn() {
    const stages = [], buffers = [];
    const engine = {
        device: { limits: { maxStorageBufferBindingSize: 128 * 1024 * 1024, maxBufferSize: 256 * 1024 * 1024,
            maxComputeWorkgroupsPerDimension: 65535, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256,
            maxStorageBuffersPerShaderStage: 8 },
        createBuffer(o) {
            const buffer = { ...o, destroyed: false, destroy() { buffer.destroyed = true; } };
            buffers.push(buffer);
            return buffer;
        },
        queue: { writeBuffer() {} } },
        createComputeStage(o) { const s = { ...o, enabled: true, preClearBuffers: [] }; stages.push(s); return s; },
        setComputePassBindGroup() {},
        render() {},
    };
    return { engine, stages, buffers };
}
