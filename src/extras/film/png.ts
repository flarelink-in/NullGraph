// D2.4 (offline-films-browser-design.md §3 "PNG", §10 B5): tightly packed RGBA8 as a PNG, in a page, a
// worker or Node, with nothing but the platform. CompressionStream('deflate') writes the zlib stream a
// PNG's IDAT holds; the chunk checksums are CRC-32. A 2D canvas would premultiply alpha and round every
// pixel's colour below alpha 255; this keeps the bytes as they are, which is what hashes and tests need.

import { fail, whole } from './util';

const SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

let table: Uint32Array | null = null;

/**
 * CRC-32, the polynomial of PNG and zlib, of `bytes`. Pass a previous result as `crc` to continue over
 * more bytes, as Node's `zlib.crc32(data, value)` does.
 */
export function crc32(bytes: Uint8Array, crc = 0): number {
    if (!table) {
        table = new Uint32Array(256);
        for (let n = 0; n < 256; n++) {
            let c = n;
            for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
            table[n] = c >>> 0;
        }
    }
    let c = ~crc >>> 0;
    for (let i = 0; i < bytes.length; i++) c = table[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
    return ~c >>> 0;
}

/** A chunk: length, type, data, and the CRC of type and data. */
function chunk(type: string, data: Uint8Array): Uint8Array {
    const out = new Uint8Array(12 + data.length);
    const view = new DataView(out.buffer);
    view.setUint32(0, data.length);
    for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
    out.set(data, 8);
    view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
    return out;
}

/**
 * Encode `rgba` (width × height × 4 bytes, row-major, top row first) as an 8-bit RGBA PNG: one IDAT of
 * unfiltered scanlines. The bytes come back exactly, alpha included.
 */
export async function encodePNG(rgba: Uint8Array, width: number, height: number): Promise<Uint8Array> {
    whole(width, 'encodePNG: width');
    whole(height, 'encodePNG: height');
    if (rgba.length !== width * height * 4) fail(`encodePNG: ${width}×${height} RGBA needs ${width * height * 4} bytes; got ${rgba.length}.`);
    const stride = width * 4;
    // Each scanline is its filter byte (0: none) and the row.
    const raw = new Uint8Array((stride + 1) * height);
    for (let y = 0; y < height; y++) raw.set(rgba.subarray(y * stride, (y + 1) * stride), y * (stride + 1) + 1);
    const deflated = new Response(new Blob([raw]).stream().pipeThrough(new CompressionStream('deflate')));
    const idat = new Uint8Array(await deflated.arrayBuffer());
    const ihdr = new Uint8Array(13);
    const view = new DataView(ihdr.buffer);
    view.setUint32(0, width);
    view.setUint32(4, height);
    ihdr[8] = 8; // bit depth
    ihdr[9] = 6; // colour type: RGBA; compression, filter and interlace stay 0
    const parts = [new Uint8Array(SIGNATURE), chunk('IHDR', ihdr), chunk('IDAT', idat), chunk('IEND', new Uint8Array(0))];
    const png = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    let at = 0;
    for (const p of parts) { png.set(p, at); at += p.length; }
    return png;
}
