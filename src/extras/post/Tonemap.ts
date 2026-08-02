// src/extras/post/Tonemap.ts
import type { DeclareContext, EffectDeclaration } from './Effect';
import type { ParamSchema } from './types';
import { PostEffectBase } from './PostEffectBase';

/** Constructor-time constant selecting a WGSL branch — not a runtime param.
 *  A uniform would force all curves into every shader for no benefit. */
export type TonemapCurve = 'aces' | 'reinhard' | 'agx' | 'none';

export interface TonemapOptions {
    curve?: TonemapCurve;
    exposure?: number;
    whitePoint?: number;
}

const SCHEMA: ParamSchema = [
    { name: 'exposure', type: 'f32', default: 1.0 },
    { name: 'whitePoint', type: 'f32', default: 4.0 },
];

const CURVES: Record<TonemapCurve, string> = {
    // Narkowicz ACES fit.
    aces: `fn tonemapCurve(x: vec3f) -> vec3f {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}`,
    // Extended Reinhard with white point.
    reinhard: `fn tonemapCurve(x: vec3f) -> vec3f {
    let wp = max(params.whitePoint, 1e-3);
    return clamp(x * (1.0 + x / (wp * wp)) / (1.0 + x), vec3f(0.0), vec3f(1.0));
}`,
    // Minimal AgX (Wrensch polynomial approximation), returning linear.
    agx: `fn agxContrast(x: vec3f) -> vec3f {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
fn tonemapCurve(c: vec3f) -> vec3f {
    let agx_mat = mat3x3f(
        0.842479062253094, 0.0423282422610123, 0.0423756549057051,
        0.0784335999999992, 0.878468636469772, 0.0784336,
        0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    let agx_inv = mat3x3f(
        1.19687900512017, -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
        -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var val = agx_mat * max(c, vec3f(1e-10));
    val = clamp(log2(val), vec3f(min_ev), vec3f(max_ev));
    val = (val - min_ev) / (max_ev - min_ev);
    val = agxContrast(val);
    return pow(max(agx_inv * val, vec3f(0.0)), vec3f(2.2));
}`,
    none: `fn tonemapCurve(x: vec3f) -> vec3f { return x; }`,
};

/** One pass, full resolution. The chain — not this effect — appends the
 *  linear->sRGB encode when this is the final on-screen pass. */
export class Tonemap extends PostEffectBase {
    public readonly name = 'Tonemap';
    public readonly curve: TonemapCurve;

    constructor(opts: TonemapOptions = {}) {
        super();
        this.curve = opts.curve ?? 'aces';
        this.initParams(SCHEMA, { exposure: opts.exposure, whitePoint: opts.whitePoint });
    }

    public get exposure(): number { return this.getParam('exposure'); }
    public set exposure(v: number) { this.setParam('exposure', v); }
    public get whitePoint(): number { return this.getParam('whitePoint'); }
    public set whitePoint(v: number) { this.setParam('whitePoint', v); }

    public declare(ctx: DeclareContext): EffectDeclaration {
        ctx.pass({
            name: 'Tonemap',
            shaderCode: `${CURVES[this.curve]}
fn effect(uv: vec2f) -> vec4f {
    let src = sampleInput0(uv);
    return vec4f(tonemapCurve(src.rgb * params.exposure), src.a);
}`,
            inputs: ['chain'],
            output: 'chain',
        });
        return { params: this.schema };
    }
}
