// src/extras/post/CustomEffect.ts
import type { DeclareContext, EffectDeclaration } from './Effect';
import type { ParamSchema, ResourceRef, OutputRef } from './types';
import { PostEffectBase } from './PostEffectBase';

export interface CustomEffectConfig {
    name: string;
    /** WGSL fragment body. Must define: fn effect(uv: vec2f) -> vec4f
     *  Available symbols: params.<field>, camera.*, passInfo.resolution,
     *  passInfo.texelSize, sampleInputN(uv), loadInputN(px). */
    code: string;
    params?: ParamSchema;
    /** Default ['chain']. 'depth' requires gbuffer.depth on the chain. */
    inputs?: ResourceRef[];
    /** Default 'chain'. */
    output?: OutputRef;
}

/**
 * Bring-your-own-WGSL as a first-class path: a single fullscreen pass with a
 * generated param struct. Each schema field becomes a live property on the
 * instance (myEffect.strength = 0.5), plus get()/set() for dynamic names.
 * Multi-pass custom effects should subclass PostEffectBase instead.
 */
export class CustomEffect extends PostEffectBase {
    public readonly name: string;
    private readonly cfg: CustomEffectConfig;

    // Index signature so generated properties are usable without casts.
    [key: string]: any;

    constructor(cfg: CustomEffectConfig) {
        super();
        this.name = cfg.name;
        this.cfg = cfg;
        const schema = cfg.params ?? [];
        this.initParams(schema, {});
        for (const f of schema) {
            if (f.name in this) {
                throw new Error(`CustomEffect "${cfg.name}": param name "${f.name}" collides with a built-in member.`);
            }
            Object.defineProperty(this, f.name, {
                get: () => this.getParam(f.name),
                set: (v: number | number[]) => this.setParam(f.name, v),
            });
        }
    }

    public get(name: string): number {
        return this.getParam(name);
    }

    public set(name: string, v: number | number[]): void {
        this.setParam(name, v);
    }

    public declare(ctx: DeclareContext): EffectDeclaration {
        ctx.pass({
            name: this.name,
            shaderCode: this.cfg.code,
            inputs: this.cfg.inputs ?? ['chain'],
            output: this.cfg.output ?? 'chain',
        });
        return { params: this.schema };
    }
}
