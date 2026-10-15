// A minimal PNG reader for the anime-look spike: 8-bit, non-interlaced, RGB or RGBA. Node only.
import { inflateSync } from 'node:zlib';

/** { width, height, rgba: Uint8Array } of a PNG file's bytes. */
export function readPNG(bytes) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let at = 8, width = 0, height = 0, type = 0;
    const idat = [];
    while (at < bytes.length) {
        const length = view.getUint32(at), kind = String.fromCharCode(...bytes.subarray(at + 4, at + 8));
        const data = bytes.subarray(at + 8, at + 8 + length);
        if (kind === 'IHDR') {
            width = view.getUint32(at + 8);
            height = view.getUint32(at + 12);
            if (data[8] !== 8 || data[12] !== 0) throw new Error('readPNG: 8-bit, non-interlaced only');
            type = data[9];
        } else if (kind === 'IDAT') idat.push(data);
        else if (kind === 'IEND') break;
        at += 12 + length;
    }
    const channels = type === 6 ? 4 : type === 2 ? 3 : 0;
    if (!channels) throw new Error(`readPNG: colour type ${type} not read`);
    const raw = inflateSync(Buffer.concat(idat)), stride = width * channels;
    const pixels = new Uint8Array(height * stride);
    for (let y = 0; y < height; y++) {
        const filter = raw[y * (stride + 1)], row = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
        for (let x = 0; x < stride; x++) {
            const a = x >= channels ? pixels[y * stride + x - channels] : 0;
            const b = y > 0 ? pixels[(y - 1) * stride + x] : 0;
            const c = x >= channels && y > 0 ? pixels[(y - 1) * stride + x - channels] : 0;
            const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
            const predict = [0, a, b, (a + b) >> 1, pa <= pb && pa <= pc ? a : pb <= pc ? b : c][filter];
            pixels[y * stride + x] = (row[x] + predict) & 255;
        }
    }
    if (channels === 4) return { width, height, rgba: pixels };
    const rgba = new Uint8Array(width * height * 4);
    for (let i = 0; i < width * height; i++) rgba.set([...pixels.subarray(i * 3, i * 3 + 3), 255], i * 4);
    return { width, height, rgba };
}
