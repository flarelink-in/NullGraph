// src/extras/post/ParamArena.ts
import type { ParamSchema, ParamType } from './types';

/** One 256-byte slice per effect: minUniformBufferOffsetAlignment on essentially
 *  all hardware, required because each pass binds its effect's slice via
 *  { buffer, offset, size }. */
export const SLICE_BYTES = 256;
export const SLICE_FLOATS = 64;

interface TypeInfo { size: number; align: number; wgsl: string; }

/** WGSL uniform address-space layout rules. vec3f aligning to 16 (not 12) is
 *  the classic silent-corruption bug — this table is the single source of truth. */
const TYPE_INFO: Record<ParamType, TypeInfo> = {
    f32:   { size: 4,  align: 4,  wgsl: 'f32' },
    i32:   { size: 4,  align: 4,  wgsl: 'i32' },
    u32:   { size: 4,  align: 4,  wgsl: 'u32' },
    vec2f: { size: 8,  align: 8,  wgsl: 'vec2f' },
    vec3f: { size: 12, align: 16, wgsl: 'vec3f' },
    vec4f: { size: 16, align: 16, wgsl: 'vec4f' },
};

export interface FieldLayoutResult {
    /** Byte offset of each user field, in schema order. */
    byteOffsets: number[];
    /** Struct size, rounded up to a multiple of the largest member alignment (min 16). */
    sizeBytes: number;
    /** Complete WGSL struct declaration with explicit _padN members. */
    wgsl: string;
    /** First member name — used by the composer to force a static binding use. */
    touchField: string;
}

/**
 * THE one offset-computing function. The WGSL emitter and the JS accessor
 * generator both call this; if they could disagree, they eventually would,
 * and the symptom is silently misread uniforms.
 */
export function layoutFields(fields: ParamSchema, structName: string = 'Params'): FieldLayoutResult {
    const lines: string[] = [];
    const byteOffsets: number[] = [];
    let offset = 0;
    let padN = 0;
    let maxAlign = 4;

    // WGSL forbids empty structs; a param-less effect gets one reserved vec4f.
    const empty = fields.length === 0;
    const list: ParamSchema = empty
        ? [{ name: '_reserved', type: 'vec4f', default: [0, 0, 0, 0] }]
        : fields;

    for (const field of list) {
        const info = TYPE_INFO[field.type];
        if (!info) throw new Error(`ParamArena: unknown param type "${field.type}" on field "${field.name}".`);
        maxAlign = Math.max(maxAlign, info.align);
        const aligned = Math.ceil(offset / info.align) * info.align;
        for (let p = offset; p < aligned; p += 4) {
            lines.push(`    _pad${padN++}: f32,`);
        }
        byteOffsets.push(aligned);
        lines.push(`    ${field.name}: ${info.wgsl},`);
        offset = aligned + info.size;
    }

    const structAlign = Math.max(maxAlign, 16);
    const sizeBytes = Math.ceil(offset / structAlign) * structAlign;
    for (let p = offset; p < sizeBytes; p += 4) {
        lines.push(`    _pad${padN++}: f32,`);
    }

    return {
        byteOffsets: empty ? [] : byteOffsets,
        sizeBytes,
        wgsl: `struct ${structName} {\n${lines.join('\n')}\n};`,
        touchField: list[0].name,
    };
}

export interface ArenaEntry {
    name: string;
    schema: ParamSchema;
    layout: FieldLayoutResult;
    sliceByteOffset: number;
    sliceBytes: number;
}

/**
 * One GPUBuffer (UNIFORM | COPY_DST) plus one Float32Array staging view.
 * One padded slice per EFFECT — not per pass — so the worker-facing layout
 * stays compact. upload() writes the whole arena unconditionally: at a few
 * kilobytes, one writeBuffer is cheaper than tracking what changed.
 */
export class ParamArena {
    public readonly entries: ArenaEntry[] = [];
    public readonly buffer: GPUBuffer;
    private staging: Float32Array;

    constructor(device: GPUDevice, effects: { name: string; schema: ParamSchema }[]) {
        const seen = new Map<string, number>();
        let byteCursor = 0;

        for (const e of effects) {
            const count = seen.get(e.name) ?? 0;
            seen.set(e.name, count + 1);
            const layout = layoutFields(e.schema);
            // An effect larger than one slice gets ceil(size/256) consecutive
            // slices; two effects never share a slice.
            const sliceBytes = Math.max(1, Math.ceil(layout.sizeBytes / SLICE_BYTES)) * SLICE_BYTES;
            this.entries.push({
                name: count === 0 ? e.name : `${e.name}#${count + 1}`,
                schema: e.schema,
                layout,
                sliceByteOffset: byteCursor,
                sliceBytes,
            });
            byteCursor += sliceBytes;
        }

        this.staging = new Float32Array(byteCursor / 4);
        this.buffer = device.createBuffer({
            label: 'PostChain_ParamArena',
            size: byteCursor,
            // Numeric literals (per the WebGPU spec) rather than GPUBufferUsage.*
            // so the arena can be unit-tested outside a WebGPU environment.
            // UNIFORM = 0x40, COPY_DST = 0x08.
            usage: 0x40 | 0x08,
        });
    }

    public get floatCount(): number {
        return this.staging.length;
    }

    /** The staging subarray for effect i. Plain index writes land here. */
    public slotFor(i: number): Float32Array {
        const e = this.entries[i];
        return this.staging.subarray(e.sliceByteOffset / 4, (e.sliceByteOffset + e.sliceBytes) / 4);
    }

    /** Adopt a worker-computed arena by reference. No copy. */
    public adopt(data: Float32Array): void {
        if (data.length !== this.staging.length) {
            throw new Error(`PostChain: setRawParams expected ${this.staging.length} floats, received ${data.length}.`);
        }
        this.staging = data;
    }

    public upload(queue: GPUQueue): void {
        queue.writeBuffer(this.buffer, 0, this.staging.buffer, this.staging.byteOffset, this.staging.byteLength);
    }

    public describeLayout(): Record<string, { offsetFloats: number; floats: number; fields: string[] }> {
        const out: Record<string, { offsetFloats: number; floats: number; fields: string[] }> = {};
        for (const e of this.entries) {
            out[e.name] = {
                offsetFloats: e.sliceByteOffset / 4,
                floats: e.sliceBytes / 4,
                fields: e.schema.map(f => f.name),
            };
        }
        return out;
    }

    public destroy(): void {
        this.buffer.destroy();
    }
}
