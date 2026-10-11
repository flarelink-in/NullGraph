// src/extras/text/TextSystem.ts
//
// Labels in the world (docs/text-labels-design.md): flat label + glyph records, a GPU cull that
// compacts visible glyphs, one indirect draw. Two ways to fill it:
//   setLabel(i, { text, position, … })   — lays the text out and writes both records
//   setRawData(labels, glyphs, glyphCount) — adopt buffers a Worker filled with layoutText
// Per frame: upload() (dirty ranges only) and splice getComputeStages() before the pass that holds
// the batch from createDrawBatch().
//
// Core is imported as types only (see picking/Picker.ts for why).

import type { NullGraph, RenderBatch, RenderPassNode, ComputeStageNode } from '../../core';
import { FontAtlas } from './FontAtlas';
import { layoutText, GLYPH_STRIDE, GLYPH_NONE, type LayoutOptions } from './layout';
import {
    buildTextCullWGSL, buildTextDrawWGSL, LABEL_FLOATS, TEXT_PARAMS_FLOATS, FOLLOW_NONE, LABEL_FLAG_HIDDEN,
} from './textShaders';

export interface TextSystemOptions {
    maxLabels: number;
    maxGlyphs: number;
    /**
     * Labels may follow positions in ANY GPU buffer — read on the GPU, so the CPU never touches
     * them. For a VisibilitySystem: `{ buffer: vis.instanceBuffer, strideFloats: 24, positionOffset: 12 }`.
     */
    follow?: { buffer: GPUBuffer; strideFloats: number; positionOffset: number };
    /** Compare each label's anchor against the scene's depth (hide behind closer geometry). Default true. */
    depthTest?: boolean;
    /** Labels whose anchor is farther from the eye than this are culled. Default: no limit. */
    maxDistance?: number;
    /** Camera uniform to cull and draw with. Default: the engine's. */
    camera?: GPUBuffer;
}

export type RGBA = [number, number, number, number];

export interface LabelOptions extends LayoutOptions {
    text: string;
    /** World anchor — or, with `follow`, an offset added to the followed position. Default [0,0,0]. */
    position?: [number, number, number];
    /** Index into the follow buffer. */
    follow?: number;
    /** Em size on screen, pixels. Default 16. */
    sizePx?: number;
    /** Default opaque white. Components 0..1, unpremultiplied. */
    color?: RGBA;
    /** Outline behind the text, `width` in screen pixels (clamped by the atlas's distance range). */
    halo?: { color: RGBA; width: number };
    /** Screen-space offset, pixels, +y up. */
    offsetPx?: [number, number];
    hidden?: boolean;
}

/** Pack RGBA (0..1) into the u32 the shader unpacks with unpack4x8unorm (R in the low byte). */
export function packRGBA8(c: ArrayLike<number>): number {
    const b = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255);
    return (b(c[0]) | (b(c[1]) << 8) | (b(c[2]) << 16) | (b(c[3]) << 24)) >>> 0;
}

/** Bytes per label / glyph record. */
const LABEL_BYTES = LABEL_FLOATS * 4;
const GLYPH_BYTES = GLYPH_STRIDE * 4;

export class TextSystem {
    public readonly maxLabels: number;
    public readonly maxGlyphs: number;
    /** Label records (CPU) — `LABEL_FLOATS` per label; f32 and u32 views over one buffer. */
    public labelsF32: Float32Array;
    public labelsU32: Uint32Array;
    /** Glyph records (CPU) — `GLYPH_STRIDE` per glyph; f32 and u32 views over one buffer. */
    public glyphsF32: Float32Array;
    public glyphsU32: Uint32Array;
    /** Glyph records the cull visits (the high-water mark of the glyph allocator). */
    public glyphCount = 0;

    public readonly labelBuffer: GPUBuffer;
    public readonly glyphBuffer: GPUBuffer;

    private readonly device: GPUDevice;
    private readonly paramsBuffer: GPUBuffer;
    private readonly followBuffer: GPUBuffer;
    private readonly ownsFollowBuffer: boolean;
    private readonly params = new Float32Array(TEXT_PARAMS_FLOATS);
    private readonly paramsU32 = new Uint32Array(this.params.buffer);
    private readonly camera: GPUBuffer;
    private readonly depthTest: boolean;
    private readonly maxDistance: number;
    private readonly followStride: number;
    private readonly followOffset: number;

    /** Per label: [glyphStart, glyphCount, glyphCapacity]. */
    private readonly ranges: Int32Array;
    private labelDirty: [number, number] = [Infinity, -1];
    private glyphDirty: [number, number] = [Infinity, -1];
    private paramsDirty = true;

    private batch: RenderBatch | null = null;
    private pass: RenderPassNode | null = null;
    private cull: ComputeStageNode | null = null;
    private quadIndexBuffer: GPUBuffer | null = null;

    constructor(private readonly engine: NullGraph, public readonly atlas: FontAtlas, options: TextSystemOptions) {
        const int = (v: number, what: string) => {
            if (!Number.isInteger(v) || v < 1) throw new Error(`TextSystem: ${what} ${v} must be a positive integer.`);
            return v;
        };
        this.maxLabels = int(options.maxLabels, 'maxLabels');
        this.maxGlyphs = int(options.maxGlyphs, 'maxGlyphs');
        this.device = engine.device;
        this.depthTest = options.depthTest ?? true;
        this.maxDistance = options.maxDistance ?? 3.0e38;
        this.camera = options.camera ?? engine.ctx.cameraUniformBuffer;

        if (options.follow) {
            const f = options.follow;
            if ((f.buffer.usage & GPUBufferUsage.STORAGE) === 0) {
                throw new Error('TextSystem: the follow buffer lacks GPUBufferUsage.STORAGE — the cull and draw read it as storage.');
            }
            if (!Number.isInteger(f.strideFloats) || f.strideFloats < 3 || !Number.isInteger(f.positionOffset) ||
                f.positionOffset < 0 || f.positionOffset + 3 > f.strideFloats) {
                throw new Error(`TextSystem: follow stride ${f.strideFloats} / positionOffset ${f.positionOffset} must ` +
                    `leave room for xyz inside each record.`);
            }
            this.followBuffer = f.buffer;
            this.followStride = f.strideFloats;
            this.followOffset = f.positionOffset;
            this.ownsFollowBuffer = false;
        } else {
            // Bound but never read (followStride 0) — one shader, no variant.
            this.followBuffer = this.device.createBuffer({ label: 'NullGraph_TextNoFollow', size: 16, usage: GPUBufferUsage.STORAGE });
            this.followStride = 0;
            this.followOffset = 0;
            this.ownsFollowBuffer = true;
        }

        const labels = new ArrayBuffer(this.maxLabels * LABEL_BYTES);
        this.labelsF32 = new Float32Array(labels);
        this.labelsU32 = new Uint32Array(labels);
        const glyphs = new ArrayBuffer(this.maxGlyphs * GLYPH_BYTES);
        this.glyphsF32 = new Float32Array(glyphs);
        this.glyphsU32 = new Uint32Array(glyphs);
        this.ranges = new Int32Array(this.maxLabels * 3);

        this.labelBuffer = this.device.createBuffer({ label: 'NullGraph_TextLabels', size: labels.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.glyphBuffer = this.device.createBuffer({ label: 'NullGraph_TextGlyphs', size: glyphs.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.paramsBuffer = this.device.createBuffer({ label: 'NullGraph_TextParams', size: TEXT_PARAMS_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }

    // -----------------------------------------------------------------------------------------
    // Filling
    // -----------------------------------------------------------------------------------------

    /**
     * Lay out and write label `index`. Re-setting a label reuses its glyph range when the new text
     * fits, and otherwise takes a fresh range at the end (the old one is cleared); plan `maxGlyphs`
     * with headroom for labels whose text grows.
     */
    public setLabel(index: number, o: LabelOptions): void {
        this.assertLabel(index);
        const size = o.sizePx ?? 16;
        if (!(size > 0)) throw new Error(`TextSystem: label ${index} sizePx ${size} must be positive.`);
        if (o.follow !== undefined && this.followStride === 0) {
            throw new Error(`TextSystem: label ${index} sets follow, but the system was created without a follow buffer.`);
        }

        // Measure into scratch first (no records written on failure), then place.
        const r = index * 3;
        const start = this.ranges[r], cap = this.ranges[r + 2];
        const need = this.measure(o);
        let at = start;
        if (need > cap) {
            this.clearGlyphs(start, cap);
            at = this.glyphCount;
            if (at + need > this.maxGlyphs) {
                throw new Error(`TextSystem: label ${index} needs ${need} glyphs but only ${this.maxGlyphs - at} of ` +
                    `${this.maxGlyphs} remain — raise maxGlyphs.`);
            }
            this.ranges[r] = at;
            this.ranges[r + 2] = need;
            this.glyphCount = at + need;
            this.paramsDirty = true;
        }
        const res = layoutText(this.atlas, o.text, index, o, this.glyphsU32, this.glyphsF32, at, at + Math.max(need, cap));
        this.clearGlyphs(at + res.glyphs, Math.max(this.ranges[r + 1], 0) - res.glyphs);
        this.ranges[r + 1] = res.glyphs;
        this.markGlyphs(at, Math.max(res.glyphs, this.ranges[r + 2]));

        this.writeLabelRecord(index, o, size);
    }

    /** Move a label without re-laying its text out. */
    public setLabelPosition(index: number, x: number, y: number, z: number): void {
        this.assertLabel(index);
        const o = index * LABEL_FLOATS;
        this.labelsF32[o] = x; this.labelsF32[o + 1] = y; this.labelsF32[o + 2] = z;
        this.markLabel(index);
    }

    /** Hide or show a label (its glyphs stay allocated). */
    public setLabelHidden(index: number, hidden: boolean): void {
        this.assertLabel(index);
        const o = index * LABEL_FLOATS + 10;
        this.labelsU32[o] = hidden ? this.labelsU32[o] | LABEL_FLAG_HIDDEN : this.labelsU32[o] & ~LABEL_FLAG_HIDDEN;
        this.markLabel(index);
    }

    /**
     * Adopt label and glyph records filled elsewhere (a Worker running layoutText). The arrays are
     * used by reference — call upload() after changing them. Glyph ranges from setLabel are reset.
     */
    public setRawData(labels: Float32Array, glyphs: Float32Array, glyphCount: number): void {
        if (labels.length > this.maxLabels * LABEL_FLOATS || glyphs.length > this.maxGlyphs * GLYPH_STRIDE) {
            throw new Error(`TextSystem: raw data exceeds capacity (labels ${labels.length / LABEL_FLOATS} / ${this.maxLabels}, ` +
                `glyphs ${glyphs.length / GLYPH_STRIDE} / ${this.maxGlyphs}).`);
        }
        if (!Number.isInteger(glyphCount) || glyphCount < 0 || glyphCount * GLYPH_STRIDE > glyphs.length) {
            throw new Error(`TextSystem: glyphCount ${glyphCount} does not fit the ${glyphs.length / GLYPH_STRIDE} glyph records supplied.`);
        }
        this.labelsF32 = labels; this.labelsU32 = new Uint32Array(labels.buffer, labels.byteOffset, labels.length);
        this.glyphsF32 = glyphs; this.glyphsU32 = new Uint32Array(glyphs.buffer, glyphs.byteOffset, glyphs.length);
        this.glyphCount = glyphCount;
        this.ranges.fill(0);
        this.labelDirty = [0, labels.length / LABEL_FLOATS - 1];
        this.glyphDirty = [0, glyphCount - 1];
        this.paramsDirty = true;
    }

    /** Write what changed since the last upload: dirty label/glyph ranges and the params. */
    public upload(): void {
        const q = this.device.queue;
        if (this.labelDirty[1] >= this.labelDirty[0]) {
            const [a, b] = this.labelDirty;
            q.writeBuffer(this.labelBuffer, a * LABEL_BYTES, this.labelsF32.buffer, this.labelsF32.byteOffset + a * LABEL_BYTES, (b - a + 1) * LABEL_BYTES);
            this.labelDirty = [Infinity, -1];
        }
        if (this.glyphDirty[1] >= this.glyphDirty[0]) {
            const [a, b] = this.glyphDirty;
            q.writeBuffer(this.glyphBuffer, a * GLYPH_BYTES, this.glyphsF32.buffer, this.glyphsF32.byteOffset + a * GLYPH_BYTES, (b - a + 1) * GLYPH_BYTES);
            this.glyphDirty = [Infinity, -1];
        }
        const w = this.engine.width, h = this.engine.height;
        if (this.paramsDirty || this.params[0] !== w || this.params[1] !== h) {
            this.params[0] = w; this.params[1] = h;
            this.params[2] = this.atlas.distanceRange;
            this.params[3] = this.atlas.emSize;
            this.params[4] = this.maxDistance;
            this.paramsU32[5] = this.glyphCount;
            this.paramsU32[6] = this.followStride;
            this.paramsU32[7] = this.followOffset;
            q.writeBuffer(this.paramsBuffer, 0, this.params);
            this.paramsDirty = false;
        }
    }

    // -----------------------------------------------------------------------------------------
    // Frame graph
    // -----------------------------------------------------------------------------------------

    /**
     * The one draw batch: premultiplied alpha, no depth write, depth-tested at each label's anchor
     * when `depthTest` (pass a pass with the scene's depth attachment), drawn by one
     * drawIndexedIndirect whatever the label count. Put it LAST in its pass.
     */
    public createDrawBatch(pass: RenderPassNode, opts: { targetFormat?: GPUTextureFormat } = {}): RenderBatch {
        if (this.batch) throw new Error('TextSystem: createDrawBatch() was already called — one batch per system (it owns one cull).');
        const batch = this.engine.createBatch(pass, {
            label: 'text labels',
            shaderCode: buildTextDrawWGSL(this.atlas.type),
            strideFloats: 1,
            maxInstances: this.maxGlyphs,
            isIndirect: true,
            sharedSourceBuffer: this.glyphBuffer,
            vertexLayouts: [],
            targetFormat: opts.targetFormat,
            blend: {
                color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
            depthWriteEnabled: false,
            depthCompare: this.depthTest ? 'less-equal' : 'always',
            cameraBuffer: this.camera,               // the same camera the cull projects with
        });
        this.quadIndexBuffer = this.engine.bufferManager.createIndexBuffer(new Uint16Array([0, 1, 2, 2, 1, 3]));
        batch.indexBuffer = this.quadIndexBuffer;
        batch.indexCount = 6;
        batch.indexFormat = 'uint16';
        this.engine.setRenderPassBindGroup(batch, [
            { binding: 0, resource: this.atlas.sampler },
            { binding: 1, resource: this.atlas.view },
            { binding: 2, resource: { buffer: this.glyphBuffer } },
            { binding: 3, resource: { buffer: this.labelBuffer } },
            { binding: 4, resource: { buffer: this.atlas.tableBuffer } },
            { binding: 5, resource: { buffer: this.followBuffer } },
            { binding: 6, resource: { buffer: this.paramsBuffer } },
        ], 1);

        const cull = this.engine.createComputeStage({
            name: 'Text Cull',
            shaderCode: buildTextCullWGSL(),
            dispatchCount: () => [Math.ceil(this.glyphCount / 64), 1, 1],
            // [indexCount = 6, instanceCount = 0 (the cull counts), firstIndex, baseVertex, firstInstance]
            preClearBuffers: [{ buffer: batch.indirectBuffer!, offset: 0, data: new Uint32Array([6, 0, 0, 0, 0]) }],
        });
        this.engine.setComputePassBindGroup(cull, [
            { binding: 0, resource: { buffer: this.camera } },
            { binding: 1, resource: { buffer: this.glyphBuffer } },
            { binding: 2, resource: { buffer: this.labelBuffer } },
            { binding: 3, resource: { buffer: this.atlas.tableBuffer } },
            { binding: 4, resource: { buffer: this.followBuffer } },
            { binding: 5, resource: { buffer: this.paramsBuffer } },
            { binding: 6, resource: { buffer: batch.storageBuffer } },
            { binding: 7, resource: { buffer: batch.indirectBuffer! } },
        ], 0);
        this.batch = batch;
        this.pass = pass;
        this.cull = cull;
        return batch;
    }

    /** `[cull]` — splice before the pass holding the batch. */
    public getComputeStages(): ComputeStageNode[] {
        if (!this.cull) throw new Error('TextSystem: getComputeStages() before createDrawBatch() — the cull writes into that batch.');
        return [this.cull];
    }

    /**
     * Release the GPU resources and take the batch out of its pass — a batch left behind would bind
     * these destroyed buffers, and every later frame would be refused at submit. Drop the stages
     * from getComputeStages() from your sequence too.
     */
    public destroy(): void {
        if (this.batch && this.pass) this.engine.clearBatch(this.pass, this.batch);
        this.pass = null;
        this.labelBuffer.destroy();
        this.glyphBuffer.destroy();
        this.paramsBuffer.destroy();
        this.quadIndexBuffer?.destroy();
        if (this.ownsFollowBuffer) this.followBuffer.destroy();
    }

    // -----------------------------------------------------------------------------------------

    private writeLabelRecord(index: number, o: LabelOptions, size: number): void {
        const f = this.labelsF32, u = this.labelsU32, b = index * LABEL_FLOATS;
        const p = o.position ?? [0, 0, 0];
        f[b] = p[0]; f[b + 1] = p[1]; f[b + 2] = p[2];
        f[b + 3] = size;
        u[b + 4] = packRGBA8(o.color ?? [1, 1, 1, 1]);
        u[b + 5] = packRGBA8(o.halo?.color ?? [0, 0, 0, 0]);
        f[b + 6] = o.halo ? Math.max(0, o.halo.width) : 0;
        u[b + 7] = o.follow ?? FOLLOW_NONE;
        f[b + 8] = o.offsetPx?.[0] ?? 0; f[b + 9] = o.offsetPx?.[1] ?? 0;
        u[b + 10] = o.hidden ? LABEL_FLAG_HIDDEN : 0;
        u[b + 11] = 0;
        this.markLabel(index);
    }

    /** Glyph records `text` will need (layout into a throwaway buffer sized generously). */
    private measure(o: LabelOptions): number {
        let n = 0;
        for (const ch of o.text) if (ch !== ' ' && ch !== '\n') n++;
        const scratch = new ArrayBuffer(Math.max(1, n) * GLYPH_BYTES);
        return layoutText(this.atlas, o.text, 0, o, new Uint32Array(scratch), new Float32Array(scratch), 0).glyphs;
    }

    private clearGlyphs(start: number, count: number): void {
        for (let k = 0; k < count; k++) this.glyphsU32[(start + k) * GLYPH_STRIDE + 1] = GLYPH_NONE;
        if (count > 0) this.markGlyphs(start, count);
    }

    private markLabel(i: number): void {
        this.labelDirty[0] = Math.min(this.labelDirty[0], i);
        this.labelDirty[1] = Math.max(this.labelDirty[1], i);
    }

    private markGlyphs(start: number, count: number): void {
        if (count <= 0) return;
        this.glyphDirty[0] = Math.min(this.glyphDirty[0], start);
        this.glyphDirty[1] = Math.max(this.glyphDirty[1], start + count - 1);
    }

    private assertLabel(i: number): void {
        if (!Number.isInteger(i) || i < 0 || i >= this.maxLabels) {
            throw new Error(`TextSystem: label index ${i} is outside [0, ${this.maxLabels}) — raise maxLabels.`);
        }
    }
}
