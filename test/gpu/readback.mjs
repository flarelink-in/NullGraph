// GPU → CPU readback for tests and tooling. Legal in a test, forbidden in the frame loop
// (design-principles §13): every function here stalls on mapAsync by design.

/** Bytes per texel for the formats a screen or test target can have. */
const FORMATS = {
    'rgba8unorm': { bpp: 4, swizzle: false }, 'rgba8unorm-srgb': { bpp: 4, swizzle: false },
    'bgra8unorm': { bpp: 4, swizzle: true }, 'bgra8unorm-srgb': { bpp: 4, swizzle: true },
    'rgba16float': { bpp: 8, half: true }, 'rgba32float': { bpp: 16, float: true },
};

/** Copy a whole buffer (or a prefix) back to the CPU. Needs COPY_SRC on `buffer`. */
export async function readBuffer(device, buffer, size = buffer.size) {
    const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, staging, 0, size);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return out;   // ArrayBuffer — wrap in the view you need
}

/**
 * Read mip 0 of a 2D texture into tightly packed RGBA8 (top row first). Float formats are clamped
 * to [0, 1] — linear values, no tone mapping — which is right for "what did it draw", not for
 * judging HDR exposure. Needs COPY_SRC on the texture.
 */
export async function readTextureRGBA8(device, texture, width, height, format) {
    const f = FORMATS[format];
    if (!f) {
        throw new Error(`readTextureRGBA8: format '${format}' is not supported — use one of ` +
            `${Object.keys(FORMATS).join(', ')}.`);
    }
    // copyTextureToBuffer requires bytesPerRow to be a multiple of 256.
    const rowBytes = width * f.bpp;
    const bytesPerRow = Math.ceil(rowBytes / 256) * 256;
    const staging = device.createBuffer({
        size: bytesPerRow * height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture }, { buffer: staging, bytesPerRow }, [width, height]);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(staging.getMappedRange());

    const out = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) {
        const row = y * bytesPerRow;
        for (let x = 0; x < width; x++) {
            const o = (y * width + x) * 4;
            if (f.half || f.float) {
                const view = new DataView(src.buffer, src.byteOffset + row + x * f.bpp, f.bpp);
                for (let c = 0; c < 4; c++) {
                    const v = f.half ? halfToFloat(view.getUint16(c * 2, true)) : view.getFloat32(c * 4, true);
                    out[o + c] = Math.round(Math.min(1, Math.max(0, v)) * 255);
                }
            } else {
                const i = row + x * 4;
                out[o] = src[f.swizzle ? i + 2 : i];
                out[o + 1] = src[i + 1];
                out[o + 2] = src[f.swizzle ? i : i + 2];
                out[o + 3] = src[i + 3];
            }
        }
    }
    staging.unmap();
    staging.destroy();
    return out;
}

function halfToFloat(h) {
    const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (m / 1024);
    if (e === 31) return m ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + m / 1024);
}
