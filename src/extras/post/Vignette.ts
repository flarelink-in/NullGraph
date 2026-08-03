// src/extras/post/Vignette.ts
import type { DeclareContext, EffectDeclaration } from './Effect';
import type { ParamSchema } from './types';
import { PostEffectBase } from './PostEffectBase';

export interface VignetteOptions {
    /** How far from center the darkening begins (0..1). */
    radius?: number;
    /** Width of the falloff band; larger = softer edge. */
    softness?: number;
    /** Blend strength toward `color` at the corners (0 = off, 1 = full). */
    intensity?: number;
    /** Ellipse roundness: 1 = circular, 0 = follows aspect ratio. */
    roundness?: number;
    /** Corner color in the chain's working space (linear). Default black. */
    color?: [number, number, number];
}

const SCHEMA: ParamSchema = [
    { name: 'radius', type: 'f32', default: 0.75 },
    { name: 'softness', type: 'f32', default: 0.45 },
    { name: 'intensity', type: 'f32', default: 0.5 },
    { name: 'roundness', type: 'f32', default: 1.0 },
    { name: 'color', type: 'vec3f', default: [0, 0, 0] },
];

/** One pass, full resolution. Smooth radial blend toward a corner color,
 *  with the mask dithered — a slow vignette gradient is the classic way to
 *  produce visible banding in an 8-bit swapchain. */
export class Vignette extends PostEffectBase {
    public readonly name = 'Vignette';

    constructor(opts: VignetteOptions = {}) {
        super();
        this.initParams(SCHEMA, {
            radius: opts.radius,
            softness: opts.softness,
            intensity: opts.intensity,
            roundness: opts.roundness,
            color: opts.color,
        });
    }

    public get radius(): number { return this.getParam('radius'); }
    public set radius(v: number) { this.setParam('radius', v); }
    public get softness(): number { return this.getParam('softness'); }
    public set softness(v: number) { this.setParam('softness', v); }
    public get intensity(): number { return this.getParam('intensity'); }
    public set intensity(v: number) { this.setParam('intensity', v); }
    public get roundness(): number { return this.getParam('roundness'); }
    public set roundness(v: number) { this.setParam('roundness', v); }
    public setColor(rgb: [number, number, number]): void { this.setParam('color', rgb); }

    public declare(ctx: DeclareContext): EffectDeclaration {
        ctx.pass({
            name: 'Vignette',
            shaderCode: `fn effect(uv: vec2f) -> vec4f {
    let src = sampleInput0(uv);
    let aspect = passInfo.texelSize.y / max(passInfo.texelSize.x, 1e-6);
    var d = uv - vec2f(0.5);
    d.x *= mix(aspect, 1.0, clamp(params.roundness, 0.0, 1.0));
    let dist = length(d) * 2.0;
    var fade = smoothstep(params.radius, params.radius + params.softness, dist) * params.intensity;
    // Interleaved gradient noise dither on the mask, +/- half an 8-bit step.
    let px = uv * passInfo.resolution;
    let ign = fract(52.9829189 * fract(dot(px, vec2f(0.06711056, 0.00583715))));
    fade = clamp(fade + (ign - 0.5) / 255.0, 0.0, 1.0);
    return vec4f(mix(src.rgb, params.color, fade), src.a);
}`,
            inputs: ['chain'],
            output: 'chain',
        });
        return { params: this.schema };
    }
}
