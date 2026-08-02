// src/extras/post/PostEffectBase.ts
import type { Effect, DeclareContext, EffectDeclaration } from './Effect';
import type { ParamSchema, ParamType } from './types';
import { layoutFields } from './ParamArena';

/**
 * Shared param plumbing for concrete effects. An effect owns NO GPU objects —
 * it is a param cursor plus a pass declaration. Field offsets come from the
 * same layoutFields() the WGSL emitter uses, so JS and shader can never
 * disagree about where a value lives.
 *
 * Values set before bind() (constructor options) are held in `pending` and
 * flushed on the FIRST bind only — a later re-bind (setRawParams adoption)
 * must not stomp worker-written data.
 */
export abstract class PostEffectBase implements Effect {
    public abstract readonly name: string;
    public enabled: boolean = true;

    protected schema: ParamSchema = [];
    private floatOffsets: Record<string, number> = {};
    private fieldTypes: Record<string, ParamType> = {};
    private slotF32: Float32Array | null = null;
    private slotU32: Uint32Array | null = null;
    private pending: Map<string, number | number[]> | null = null;

    protected initParams(schema: ParamSchema, overrides: Record<string, number | number[] | undefined> = {}): void {
        this.schema = schema;
        const layout = layoutFields(schema);
        this.pending = new Map();
        schema.forEach((f, i) => {
            this.floatOffsets[f.name] = layout.byteOffsets[i] / 4;
            this.fieldTypes[f.name] = f.type;
            const v = overrides[f.name] !== undefined ? overrides[f.name]! : f.default;
            this.pending!.set(f.name, v);
        });
    }

    public bind(slot: Float32Array): void {
        this.slotF32 = slot;
        this.slotU32 = new Uint32Array(slot.buffer, slot.byteOffset, slot.length);
        if (this.pending) {
            for (const [name, v] of this.pending) this.writeField(name, v);
            this.pending = null;
        }
    }

    protected getParam(name: string): number {
        if (this.slotF32) {
            const off = this.floatOffsets[name];
            const t = this.fieldTypes[name];
            return (t === 'u32' || t === 'i32') ? this.slotU32![off] : this.slotF32[off];
        }
        return this.pending?.get(name) as number;
    }

    protected setParam(name: string, v: number | number[]): void {
        if (this.slotF32) {
            this.writeField(name, v);
        } else {
            this.pending?.set(name, v);
        }
    }

    private writeField(name: string, v: number | number[]): void {
        const off = this.floatOffsets[name];
        if (off === undefined) return;
        if (Array.isArray(v)) {
            for (let i = 0; i < v.length; i++) this.slotF32![off + i] = v[i];
            return;
        }
        const t = this.fieldTypes[name];
        if (t === 'u32' || t === 'i32') {
            this.slotU32![off] = v;
        } else {
            this.slotF32![off] = v;
        }
    }

    public abstract declare(ctx: DeclareContext): EffectDeclaration;
}
