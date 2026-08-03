// src/extras/post/FilmGrain.ts
import type { DeclareContext, EffectDeclaration } from './Effect';
import type { ParamSchema } from './types';
import { PostEffectBase } from './PostEffectBase';

export interface FilmGrainOptions {
    /** Grain amplitude added to the image (0 = off). */
    intensity?: number;
    /** Grain cell size in pixels; 1 = per-pixel, 2+ = coarser film stock. */
    size?: number;
    /** 0 = monochrome grain, 1 = fully independent RGB noise (color negative). */
    colorAmount?: number;
    /** How much the grain fades on bright pixels (0 = uniform, 1 = shadows only). */
    luminanceResponse?: number;
    /** Extra offset added to camera.time; lets two chains decorrelate. */
    seed?: number;
}

const SCHEMA: ParamSchema = [
    { name: 'intensity', type: 'f32', default: 0.06 },
    { name: 'size', type: 'f32', default: 1.0 },
    { name: 'colorAmount', type: 'f32', default: 0.25 },
    { name: 'luminanceResponse', type: 'f32', default: 0.7 },
    { name: 'seed', type: 'f32', default: 0.0 },
];

/** One pass, full resolution. PCG-hashed noise (integer hash — fract(sin)
 *  breaks up on half-precision GPUs), triangular-distributed for a filmic
 *  look, animated from camera.time so it flickers with no caller work. */
export class FilmGrain extends PostEffectBase {
    public readonly name = 'FilmGrain';

    constructor(opts: FilmGrainOptions = {}) {
        super();
        this.initParams(SCHEMA, {
            intensity: opts.intensity,
            size: opts.size,
            colorAmount: opts.colorAmount,
            luminanceResponse: opts.luminanceResponse,
            seed: opts.seed,
        });
    }

    public get intensity(): number { return this.getParam('intensity'); }
    public set intensity(v: number) { this.setParam('intensity', v); }
    public get size(): number { return this.getParam('size'); }
    public set size(v: number) { this.setParam('size', v); }
    public get colorAmount(): number { return this.getParam('colorAmount'); }
    public set colorAmount(v: number) { this.setParam('colorAmount', v); }
    public get luminanceResponse(): number { return this.getParam('luminanceResponse'); }
    public set luminanceResponse(v: number) { this.setParam('luminanceResponse', v); }
    public get seed(): number { return this.getParam('seed'); }
    public set seed(v: number) { this.setParam('seed', v); }

    public declare(ctx: DeclareContext): EffectDeclaration {
        ctx.pass({
            name: 'FilmGrain',
            shaderCode: `fn grainPcg3d(v0: vec3u) -> vec3u {
    var v = v0 * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v = v ^ (v >> vec3u(16u));
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}
fn grainRand(cell: vec2f, t: f32) -> vec3f {
    let u = grainPcg3d(vec3u(bitcast<u32>(cell.x), bitcast<u32>(cell.y), bitcast<u32>(t)));
    return vec3f(u) * (1.0 / 4294967296.0);
}
fn effect(uv: vec2f) -> vec4f {
    let src = sampleInput0(uv);
    let cell = floor(uv * passInfo.resolution / max(params.size, 1.0));
    let t = camera.time + params.seed;
    // Average of two uniform draws -> triangular distribution: film grain
    // clusters near zero, raw white noise reads as digital static.
    let r1 = grainRand(cell, t);
    let r2 = grainRand(cell + vec2f(97.0, 31.0), t + 61.7);
    let tri = (r1 + r2) * 0.5 - vec3f(0.5);
    let n = mix(vec3f(tri.x), tri, clamp(params.colorAmount, 0.0, 1.0));
    let luma = dot(src.rgb, vec3f(0.2126, 0.7152, 0.0722));
    let response = mix(1.0, clamp(1.0 - luma, 0.0, 1.0), params.luminanceResponse);
    return vec4f(src.rgb + n * params.intensity * response, src.a);
}`,
            inputs: ['chain'],
            output: 'chain',
        });
        return { params: this.schema };
    }
}
