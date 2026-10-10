// Minimal PNG encoder (RGBA8, no dependencies) and the pixel statistics a render check needs.
//
// zlib does the compression and Node's own zlib.crc32 the checksums, so this is the whole format:
// signature, IHDR, one IDAT of filter-0 scanlines, IEND. Small on purpose — it exists so a render
// can be LOOKED AT (by a person or an agent) without pulling an image library into the repo.

import { crc32, deflateSync } from 'node:zlib';

const SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

function chunk(type, data) {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(body) >>> 0);
    return Buffer.concat([len, body, crc]);
}

/**
 * Encode tightly packed RGBA8 pixels (row-major, top row first) as a PNG.
 * @param {Uint8Array} rgba width*height*4 bytes
 */
export function encodePNG(rgba, width, height) {
    if (rgba.length !== width * height * 4) {
        throw new Error(`encodePNG: expected ${width * height * 4} bytes for ${width}x${height} RGBA, got ${rgba.length}`);
    }
    const ihdr = Buffer.alloc(13);
    ihdr.writeUInt32BE(width, 0);
    ihdr.writeUInt32BE(height, 4);
    ihdr[8] = 8;   // bit depth
    ihdr[9] = 6;   // colour type: RGBA
    ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;   // compression, filter, interlace

    const stride = width * 4;
    const raw = Buffer.alloc((stride + 1) * height);
    for (let y = 0; y < height; y++) {
        raw[y * (stride + 1)] = 0;   // filter: none
        raw.set(rgba.subarray(y * stride, (y + 1) * stride), y * (stride + 1) + 1);
    }
    return Buffer.concat([
        SIGNATURE,
        chunk('IHDR', ihdr),
        chunk('IDAT', deflateSync(raw, { level: 6 })),
        chunk('IEND', Buffer.alloc(0)),
    ]);
}

/**
 * What a "did anything render" gate needs: luminance spread and how much of the frame differs
 * from its own corner pixel (the clear colour, for any scene that does not cover the corners).
 * A uniform frame has stdDev 0 — the classic black-canvas failure looks exactly like that.
 */
export function pixelStats(rgba, width, height) {
    const n = width * height;
    let sum = 0, sumSq = 0, differs = 0;
    const r0 = rgba[0], g0 = rgba[1], b0 = rgba[2];
    for (let i = 0; i < n; i++) {
        const r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2];
        const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        sum += l; sumSq += l * l;
        if (Math.abs(r - r0) + Math.abs(g - g0) + Math.abs(b - b0) > 6) differs++;
    }
    const mean = sum / n;
    return {
        meanLuma: mean,
        stdDev: Math.sqrt(Math.max(0, sumSq / n - mean * mean)),
        coverage: differs / n,   // fraction of pixels visibly different from the corner
        corner: [r0, g0, b0],
    };
}
