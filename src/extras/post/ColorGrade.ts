// src/extras/post/ColorGrade.ts
import type { DeclareContext, EffectDeclaration } from './Effect';
import type { ParamSchema } from './types';
import { PostEffectBase } from './PostEffectBase';

export interface ColorGradeOptions {
    /** Linear multiplier applied first. */
    exposure?: number;
    /** Log-space contrast around mid-grey (0.18). 1 = neutral. */
    contrast?: number;
    /** 0 = greyscale, 1 = neutral, >1 = boosted. */
    saturation?: number;
    /** Warm/cool shift, roughly -1..1. Positive = warmer. */
    temperature?: number;
    /** Green/magenta shift, roughly -1..1. Positive = magenta. */
    tint?: number;
    /** Hue rotation in radians around the grey axis. */
    hueShift?: number;
    /** ASC-CDL-style pedestal added after gain. */
    lift?: [number, number, number];
    /** Per-channel power (applied as 1/gamma). */
    gamma?: [number, number, number];
    /** Per-channel slope. */
    gain?: [number, number, number];
}

const SCHEMA: ParamSchema = [
    { name: 'exposure', type: 'f32', default: 1.0 },
    { name: 'contrast', type: 'f32', default: 1.0 },
    { name: 'saturation', type: 'f32', default: 1.0 },
    { name: 'temperature', type: 'f32', default: 0.0 },
    { name: 'tint', type: 'f32', default: 0.0 },
    { name: 'hueShift', type: 'f32', default: 0.0 },
    { name: 'lift', type: 'vec3f', default: [0, 0, 0] },
    { name: 'gamma', type: 'vec3f', default: [1, 1, 1] },
    { name: 'gain', type: 'vec3f', default: [1, 1, 1] },
];

/**
 * One pass, full resolution, parametric (no LUT — pass inputs are chain
 * resources and pooled targets only, so an external LUT texture cannot be
 * bound; a LUT variant needs a composer extension first). Order: exposure,
 * white balance, log contrast, hue, saturation, then lift/gamma/gain.
 * Operates in linear; typically placed before Tonemap.
 */
export class ColorGrade extends PostEffectBase {
    public readonly name = 'ColorGrade';

    constructor(opts: ColorGradeOptions = {}) {
        super();
        this.initParams(SCHEMA, {
            exposure: opts.exposure,
            contrast: opts.contrast,
            saturation: opts.saturation,
            temperature: opts.temperature,
            tint: opts.tint,
            hueShift: opts.hueShift,
            lift: opts.lift,
            gamma: opts.gamma,
            gain: opts.gain,
        });
    }

    public get exposure(): number { return this.getParam('exposure'); }
    public set exposure(v: number) { this.setParam('exposure', v); }
    public get contrast(): number { return this.getParam('contrast'); }
    public set contrast(v: number) { this.setParam('contrast', v); }
    public get saturation(): number { return this.getParam('saturation'); }
    public set saturation(v: number) { this.setParam('saturation', v); }
    public get temperature(): number { return this.getParam('temperature'); }
    public set temperature(v: number) { this.setParam('temperature', v); }
    public get tint(): number { return this.getParam('tint'); }
    public set tint(v: number) { this.setParam('tint', v); }
    public get hueShift(): number { return this.getParam('hueShift'); }
    public set hueShift(v: number) { this.setParam('hueShift', v); }
    public setLift(rgb: [number, number, number]): void { this.setParam('lift', rgb); }
    public setGamma(rgb: [number, number, number]): void { this.setParam('gamma', rgb); }
    public setGain(rgb: [number, number, number]): void { this.setParam('gain', rgb); }

    public declare(ctx: DeclareContext): EffectDeclaration {
        ctx.pass({
            name: 'ColorGrade',
            shaderCode: `fn cgHueRotate(c: vec3f, a: f32) -> vec3f {
    // Rodrigues rotation around the (1,1,1)/sqrt(3) grey axis.
    let k = vec3f(0.57735027);
    return c * cos(a) + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - cos(a));
}
fn effect(uv: vec2f) -> vec4f {
    let src = sampleInput0(uv);
    var c = src.rgb * params.exposure;
    let t = params.temperature * 0.1;
    let g = params.tint * 0.1;
    c *= vec3f(1.0 + t - g * 0.5, 1.0 + g, 1.0 - t - g * 0.5);
    // pow(c/0.18, k) * 0.18 == contrast in log space pivoted at mid-grey.
    c = pow(max(c, vec3f(1e-6)) / 0.18, vec3f(params.contrast)) * 0.18;
    c = cgHueRotate(c, params.hueShift);
    let luma = dot(c, vec3f(0.2126, 0.7152, 0.0722));
    c = max(mix(vec3f(luma), c, params.saturation), vec3f(0.0));
    c = max(c * params.gain + params.lift, vec3f(0.0));
    c = pow(c, 1.0 / max(params.gamma, vec3f(1e-3)));
    return vec4f(c, src.a);
}`,
            inputs: ['chain'],
            output: 'chain',
        });
        return { params: this.schema };
    }
}
