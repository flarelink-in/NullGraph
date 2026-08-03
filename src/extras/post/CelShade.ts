// src/extras/post/CelShade.ts
import type { DeclareContext, EffectDeclaration } from './Effect';
import type { ParamSchema } from './types';
import { PostEffectBase } from './PostEffectBase';

export interface CelShadeOptions {
    /** Number of luminance bands. 3-4 reads as comic, 5-6 as anime. */
    bands?: number;
    /** Width of the transition between bands, as a fraction of one band.
     *  0 = hard posterize steps, ~0.1 = slightly painted edges. */
    softness?: number;
    /** Saturation boost after banding; flat shading reads best a bit vivid. */
    saturation?: number;
}

const SCHEMA: ParamSchema = [
    { name: 'bands', type: 'f32', default: 4.0 },
    { name: 'softness', type: 'f32', default: 0.08 },
    { name: 'saturation', type: 'f32', default: 1.2 },
];

/**
 * One pass, full resolution. Quantizes luminance into flat bands while
 * keeping hue — the toon "stepped lighting" look — then boosts saturation.
 * Place AFTER Tonemap: banding display-referred values gives evenly spaced
 * steps; banding raw HDR crushes everything into the bottom band.
 * Pairs with Outline for the full anime/comic look.
 */
export class CelShade extends PostEffectBase {
    public readonly name = 'CelShade';

    constructor(opts: CelShadeOptions = {}) {
        super();
        this.initParams(SCHEMA, {
            bands: opts.bands,
            softness: opts.softness,
            saturation: opts.saturation,
        });
    }

    public get bands(): number { return this.getParam('bands'); }
    public set bands(v: number) { this.setParam('bands', Math.max(1, v)); }
    public get softness(): number { return this.getParam('softness'); }
    public set softness(v: number) { this.setParam('softness', v); }
    public get saturation(): number { return this.getParam('saturation'); }
    public set saturation(v: number) { this.setParam('saturation', v); }

    public declare(ctx: DeclareContext): EffectDeclaration {
        ctx.pass({
            name: 'CelShade',
            shaderCode: `fn effect(uv: vec2f) -> vec4f {
    let src = sampleInput0(uv);
    let l = dot(src.rgb, vec3f(0.2126, 0.7152, 0.0722));
    let bands = max(params.bands, 1.0);
    // Each band flattens to its center; smoothstep at the top of the band
    // gives an optional painted transition instead of a hard posterize step.
    let x = l * bands;
    let cell = floor(x);
    let soft = clamp(params.softness, 1e-3, 0.49);
    let stepUp = smoothstep(1.0 - soft, 1.0, fract(x));
    let lq = clamp((cell + 0.5 + stepUp) / bands, 0.0, 1.0);
    var rgb = src.rgb * (lq / max(l, 1e-4));
    let l2 = dot(rgb, vec3f(0.2126, 0.7152, 0.0722));
    rgb = max(mix(vec3f(l2), rgb, params.saturation), vec3f(0.0));
    return vec4f(rgb, src.a);
}`,
            inputs: ['chain'],
            output: 'chain',
        });
        return { params: this.schema };
    }
}
