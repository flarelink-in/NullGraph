// src/extras/text/FontAtlas.ts
//
// A font atlas on the GPU (docs/text-labels-design.md §4, §6): a distance-field texture, a glyph
// table the shaders read, and the CPU metrics layout needs. Three sources, one class:
//   fromMSDF   — msdf-atlas-gen JSON + its image (MSDF / MTSDF / SDF). The standard tool.
//   fromCanvas — any font the browser can draw, distance-transformed at runtime (SDF). Zero setup.
//   fromSDF    — raw distance bytes + glyph metrics. What the other two call; also the tests'.
//
// Core is imported as types only (see picking/Picker.ts for why).

import type { NullGraph } from '../../core';
import type { LayoutFont } from './layout';
import { computeSDF, encodeSDF, shelfPack } from './sdf';

export type DistanceType = 'msdf' | 'mtsdf' | 'sdf';

/** One glyph as described by an atlas. Bounds: `plane` in em (y up), `atlas` in atlas pixels. */
export interface AtlasGlyph {
    unicode: number;
    /** Pen advance, em. */
    advance: number;
    /** Quad extents around the pen position, em, y up — absent for whitespace. */
    plane?: { left: number; bottom: number; right: number; top: number };
    /** Where the glyph sits in the atlas image, pixels, measured from `yOrigin`. */
    atlas?: { left: number; bottom: number; right: number; top: number };
}

export interface AtlasDescription {
    type: DistanceType;
    /** Width of the encoded distance range, in atlas pixels. */
    distanceRange: number;
    /** Atlas pixels per em. */
    emSize: number;
    width: number;
    height: number;
    /** Which edge `atlas` bounds are measured from. */
    yOrigin: 'top' | 'bottom';
    lineHeight: number;
    ascender: number;
    descender: number;
    glyphs: AtlasGlyph[];
    kerning?: { left: number; right: number; advance: number }[];
}

/** Floats per glyph in the GPU table: uv (u at left, v at plane-bottom, u at right, v at plane-top), plane (l, b, r, t). */
export const ATLAS_GLYPH_FLOATS = 8;

const TYPES: DistanceType[] = ['msdf', 'mtsdf', 'sdf'];

/**
 * Read msdf-atlas-gen's JSON (`-json`). Metrics are normalised to em whatever `metrics.emSize` was.
 * `psdf` reads as `sdf` (a perpendicular SDF is still one channel); `hardmask`/`softmask` are refused.
 */
export function parseMSDFJson(json: any): AtlasDescription {
    const need = (v: any, what: string): any => {
        if (v === undefined || v === null) throw new Error(`FontAtlas: the atlas JSON has no ${what} — is it msdf-atlas-gen output (-json)?`);
        return v;
    };
    const atlas = need(json?.atlas, '"atlas" block');
    const metrics = need(json?.metrics, '"metrics" block');
    const rawType = String(need(atlas.type, 'atlas.type'));
    const type = (rawType === 'psdf' ? 'sdf' : rawType) as DistanceType;
    if (!TYPES.includes(type)) {
        throw new Error(`FontAtlas: atlas type "${rawType}" is not a distance field — regenerate with -type msdf, mtsdf or sdf.`);
    }
    const em = Number(metrics.emSize ?? 1);
    const s = 1 / em;
    const glyphs: AtlasGlyph[] = (need(json.glyphs, '"glyphs" array') as any[]).map((g) => ({
        unicode: g.unicode,
        advance: g.advance * s,
        plane: g.planeBounds && {
            left: g.planeBounds.left * s, bottom: g.planeBounds.bottom * s,
            right: g.planeBounds.right * s, top: g.planeBounds.top * s,
        },
        atlas: g.atlasBounds && { ...g.atlasBounds },
    }));
    return {
        type,
        distanceRange: Number(need(atlas.distanceRange, 'atlas.distanceRange')),
        emSize: Number(need(atlas.size, 'atlas.size (pixels per em)')),
        width: Number(need(atlas.width, 'atlas.width')),
        height: Number(need(atlas.height, 'atlas.height')),
        yOrigin: atlas.yOrigin === 'top' ? 'top' : 'bottom',
        lineHeight: Number(need(metrics.lineHeight, 'metrics.lineHeight')) * s,
        ascender: Number(need(metrics.ascender, 'metrics.ascender')) * s,
        descender: Number(need(metrics.descender, 'metrics.descender')) * s,
        glyphs,
        kerning: (json.kerning ?? []).map((k: any) => ({ left: k.unicode1, right: k.unicode2, advance: k.advance * s })),
    };
}

/**
 * The GPU glyph table and the CPU lookups, from a description. Texture v runs down from the top
 * row, so `yOrigin: 'bottom'` bounds are flipped here, once.
 */
export function buildGlyphTable(desc: AtlasDescription) {
    const n = desc.glyphs.length;
    const table = new Float32Array(n * ATLAS_GLYPH_FLOATS);
    const advances = new Float32Array(n);
    const shaped = new Uint8Array(n);
    const index = new Map<number, number>();
    desc.glyphs.forEach((g, i) => {
        index.set(g.unicode, i);
        advances[i] = g.advance;
        if (!g.plane || !g.atlas) return;
        const vAt = (y: number) => (desc.yOrigin === 'bottom' ? 1 - y / desc.height : y / desc.height);
        const o = i * ATLAS_GLYPH_FLOATS;
        table[o] = g.atlas.left / desc.width;
        table[o + 1] = vAt(g.atlas.bottom);
        table[o + 2] = g.atlas.right / desc.width;
        table[o + 3] = vAt(g.atlas.top);
        table[o + 4] = g.plane.left; table[o + 5] = g.plane.bottom;
        table[o + 6] = g.plane.right; table[o + 7] = g.plane.top;
        shaped[i] = g.plane.right > g.plane.left && g.plane.top > g.plane.bottom ? 1 : 0;
    });
    const kerning = new Map<number, number>();
    for (const k of desc.kerning ?? []) kerning.set(k.left * 0x110000 + k.right, k.advance);
    return { table, advances, shaped, index, kerning };
}

/** Options for {@link FontAtlas.fromCanvas}. */
export interface CanvasAtlasOptions {
    fontFamily: string;
    fontWeight?: string | number;
    fontStyle?: string;
    /** Characters to include. Default: printable ASCII. */
    chars?: string;
    /** Rasterisation size, pixels per em. Default 48. */
    emSizePx?: number;
    /** Encoded distance range, pixels. Larger allows wider halos. Default 8. */
    distanceRange?: number;
    /** Atlas width in pixels. Default 1024. */
    atlasWidth?: number;
}

const ASCII = Array.from({ length: 95 }, (_, i) => String.fromCharCode(32 + i)).join('');

export class FontAtlas implements LayoutFont {
    public readonly type: DistanceType;
    public readonly distanceRange: number;
    public readonly emSize: number;
    public readonly width: number;
    public readonly height: number;
    public readonly lineHeight: number;
    public readonly ascender: number;
    public readonly descender: number;
    public readonly glyphCount: number;

    /** The distance field: `rgba8unorm` for MSDF/MTSDF, `r8unorm` for SDF. */
    public readonly texture: GPUTexture;
    public readonly view: GPUTextureView;
    public readonly sampler: GPUSampler;
    /** Glyph table, {@link ATLAS_GLYPH_FLOATS} floats per glyph. */
    public readonly tableBuffer: GPUBuffer;

    private readonly index: Map<number, number>;
    private readonly advances: Float32Array;
    private readonly shaped: Uint8Array;
    private readonly kerningPairs: Map<number, number>;

    private constructor(engine: NullGraph, desc: AtlasDescription, upload: (texture: GPUTexture) => void) {
        if (!(desc.distanceRange > 0) || !(desc.emSize > 0)) {
            throw new Error(`FontAtlas: distanceRange (${desc.distanceRange}) and emSize (${desc.emSize}) must be positive.`);
        }
        this.type = desc.type;
        this.distanceRange = desc.distanceRange;
        this.emSize = desc.emSize;
        this.width = desc.width;
        this.height = desc.height;
        this.lineHeight = desc.lineHeight;
        this.ascender = desc.ascender;
        this.descender = desc.descender;
        this.glyphCount = desc.glyphs.length;

        const t = buildGlyphTable(desc);
        this.index = t.index; this.advances = t.advances; this.shaped = t.shaped; this.kerningPairs = t.kerning;

        const device = engine.device;
        this.texture = device.createTexture({
            label: 'NullGraph_FontAtlas',
            size: [desc.width, desc.height],
            format: desc.type === 'sdf' ? 'r8unorm' : 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        upload(this.texture);
        this.view = this.texture.createView();
        this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
        this.tableBuffer = device.createBuffer({
            label: 'NullGraph_FontAtlasGlyphs',
            size: Math.max(32, t.table.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.tableBuffer, 0, t.table);
    }

    // --- LayoutFont ---------------------------------------------------------------------------
    public glyphIndex(codepoint: number): number { return this.index.get(codepoint) ?? -1; }
    public advance(glyph: number): number { return this.advances[glyph]; }
    public kerning(left: number, right: number): number { return this.kerningPairs.get(left * 0x110000 + right) ?? 0; }
    public hasShape(glyph: number): boolean { return this.shaped[glyph] === 1; }

    /**
     * From raw distance bytes: `data` is width × height × (4 for MSDF/MTSDF, 1 for SDF), row-major,
     * top row first — exactly what the texture receives.
     */
    public static fromSDF(engine: NullGraph, desc: AtlasDescription & { data: Uint8Array }): FontAtlas {
        const channels = desc.type === 'sdf' ? 1 : 4;
        if (desc.data.length !== desc.width * desc.height * channels) {
            throw new Error(`FontAtlas: ${desc.type} data for ${desc.width}×${desc.height} needs ` +
                `${desc.width * desc.height * channels} bytes (${channels} per texel), got ${desc.data.length}.`);
        }
        return new FontAtlas(engine, desc, (texture) => {
            engine.device.queue.writeTexture({ texture }, desc.data as Uint8Array<ArrayBuffer>,
                { bytesPerRow: desc.width * channels, rowsPerImage: desc.height }, [desc.width, desc.height]);
        });
    }

    /**
     * From msdf-atlas-gen output: its JSON and its image. The image is distance DATA — decode it
     * without colour management (`createImageBitmap(blob, { colorSpaceConversion: 'none',
     * premultiplyAlpha: 'none' })`) or pass raw RGBA bytes as `{ data, width, height }`.
     */
    public static fromMSDF(
        engine: NullGraph, json: unknown,
        image: ImageBitmap | { data: Uint8Array | Uint8ClampedArray; width: number; height: number },
    ): FontAtlas {
        const desc = parseMSDFJson(json);
        if (image.width !== desc.width || image.height !== desc.height) {
            throw new Error(`FontAtlas: the image is ${image.width}×${image.height} but the JSON describes ${desc.width}×${desc.height}.`);
        }
        if ('data' in image) {
            const rgba = new Uint8Array(image.data.buffer, image.data.byteOffset, image.data.byteLength);
            if (desc.type === 'sdf') {
                const r = new Uint8Array(desc.width * desc.height);
                for (let i = 0; i < r.length; i++) r[i] = rgba[i * 4];
                return FontAtlas.fromSDF(engine, { ...desc, data: r });
            }
            return FontAtlas.fromSDF(engine, { ...desc, data: rgba });
        }
        if (desc.type === 'sdf') {
            throw new Error('FontAtlas: an SDF ImageBitmap cannot be copied into an r8unorm atlas — pass { data, width, height } (RGBA bytes) instead.');
        }
        return new FontAtlas(engine, desc, (texture) => {
            engine.device.queue.copyExternalImageToTexture({ source: image }, { texture }, [desc.width, desc.height]);
        });
    }

    /**
     * A single-channel SDF atlas for any font the browser can draw, built now: each glyph is
     * rasterised with Canvas 2D and distance-transformed ({@link computeSDF}). Browser and Worker
     * only (`OffscreenCanvas`). Kerning pairs are not available from Canvas, so layout uses none.
     */
    public static fromCanvas(engine: NullGraph, opts: CanvasAtlasOptions): FontAtlas {
        const Canvas = (globalThis as any).OffscreenCanvas;
        if (typeof Canvas !== 'function') {
            throw new Error('FontAtlas.fromCanvas needs OffscreenCanvas (browser or Worker). In Node, load a ' +
                'pre-generated atlas with FontAtlas.fromMSDF.');
        }
        const em = opts.emSizePx ?? 48;
        const range = opts.distanceRange ?? 8;
        const atlasWidth = opts.atlasWidth ?? 1024;
        const pad = Math.ceil(range / 2) + 1;
        const font = `${opts.fontStyle ?? 'normal'} ${opts.fontWeight ?? 'normal'} ${em}px ${opts.fontFamily}`;
        const chars = Array.from(new Set(Array.from(opts.chars ?? ASCII)));

        const probe = new Canvas(1, 1).getContext('2d');
        probe.font = font;
        const ref = probe.measureText('Hg');
        const ascent = ref.fontBoundingBoxAscent ?? ref.actualBoundingBoxAscent;
        const descent = ref.fontBoundingBoxDescent ?? ref.actualBoundingBoxDescent;

        type Raster = { cp: number; advance: number; w: number; h: number; ox: number; oy: number; sdf?: Float32Array };
        const rasters: Raster[] = [];
        for (const ch of chars) {
            const cp = ch.codePointAt(0)!;
            const m = probe.measureText(ch);
            const left = m.actualBoundingBoxLeft, right = m.actualBoundingBoxRight;
            const top = m.actualBoundingBoxAscent, bottom = m.actualBoundingBoxDescent;
            const r: Raster = { cp, advance: m.width / em, w: 0, h: 0, ox: 0, oy: 0 };
            if (right + left > 0 && top + bottom > 0) {
                r.ox = pad + Math.ceil(left);
                r.oy = pad + Math.ceil(top);
                r.w = r.ox + Math.ceil(right) + pad;
                r.h = r.oy + Math.ceil(bottom) + pad;
                const c = new Canvas(r.w, r.h).getContext('2d', { willReadFrequently: true });
                c.font = font;
                c.fillStyle = '#fff';
                c.textBaseline = 'alphabetic';
                c.fillText(ch, r.ox, r.oy);
                const rgba = c.getImageData(0, 0, r.w, r.h).data;
                const cov = new Float32Array(r.w * r.h);
                for (let i = 0; i < cov.length; i++) cov[i] = rgba[i * 4 + 3] / 255;
                r.sdf = computeSDF(cov, r.w, r.h);
            }
            rasters.push(r);
        }

        const shaped = rasters.filter(r => r.sdf);
        const { rects, height } = shelfPack(shaped.map(r => ({ w: r.w, h: r.h })), atlasWidth);
        const atlasHeight = Math.max(1, height);
        const data = new Uint8Array(atlasWidth * atlasHeight);
        const glyphs: AtlasGlyph[] = [];
        let s = 0;
        for (const r of rasters) {
            if (!r.sdf) { glyphs.push({ unicode: r.cp, advance: r.advance }); continue; }
            const rect = rects[s++];
            const bytes = encodeSDF(r.sdf, range, new Uint8Array(r.w * r.h));
            for (let y = 0; y < r.h; y++) data.set(bytes.subarray(y * r.w, (y + 1) * r.w), (rect.y + y) * atlasWidth + rect.x);
            glyphs.push({
                unicode: r.cp,
                advance: r.advance,
                plane: { left: -r.ox / em, bottom: -(r.h - r.oy) / em, right: (r.w - r.ox) / em, top: r.oy / em },
                atlas: { left: rect.x, bottom: rect.y + r.h, right: rect.x + r.w, top: rect.y },
            });
        }
        return FontAtlas.fromSDF(engine, {
            type: 'sdf', distanceRange: range, emSize: em, width: atlasWidth, height: atlasHeight, yOrigin: 'top',
            lineHeight: (ascent + descent) / em, ascender: ascent / em, descender: -descent / em,
            glyphs, data,
        });
    }

    public destroy(): void {
        this.texture.destroy();
        this.tableBuffer.destroy();
    }
}
