// src/extras/post/ChromaticAberration.ts
import type { DeclareContext, EffectDeclaration } from './Effect';
import type { ParamSchema } from './types';
import { PostEffectBase } from './PostEffectBase';

export interface ChromaticAberrationOptions {
    /** Maximum RGB channel offset at the screen edge, in UV units. */
    strength?: number;
    /** Radial falloff exponent; higher keeps the center cleaner. */
    falloff?: number;
}

/** Compile-time tap count along the dispersion axis. 12 taps at full strength
 *  still resolve as a smooth rainbow rather than three discrete ghosts. */
const SAMPLES = 12;

const SCHEMA: ParamSchema = [
    { name: 'strength', type: 'f32', default: 0.006 },
    { name: 'falloff', type: 'f32', default: 2.0 },
];

/** One pass, full resolution. Spectral fringing: many taps along the radial
 *  axis, each weighted by a wavelength ramp (red outermost, blue innermost),
 *  so strong settings smear into a continuous rainbow instead of splitting
 *  into three hard copies of the image. */
export class ChromaticAberration extends PostEffectBase {
    public readonly name = 'ChromaticAberration';

    constructor(opts: ChromaticAberrationOptions = {}) {
        super();
        this.initParams(SCHEMA, { strength: opts.strength, falloff: opts.falloff });
    }

    public get strength(): number { return this.getParam('strength'); }
    public set strength(v: number) { this.setParam('strength', v); }
    public get falloff(): number { return this.getParam('falloff'); }
    public set falloff(v: number) { this.setParam('falloff', v); }

    public declare(ctx: DeclareContext): EffectDeclaration {
        ctx.pass({
            name: 'ChromaticAberration',
            // Triangular RGB bumps at t = 0 / 0.5 / 1 form a partition of unity,
            // so the per-channel weight sums stay equal and no channel gains energy.
            shaderCode: `fn caSpectrum(t: f32) -> vec3f {
    return clamp(vec3f(
        1.0 - abs(t * 2.0),
        1.0 - abs(t * 2.0 - 1.0),
        1.0 - abs(t * 2.0 - 2.0)
    ), vec3f(0.0), vec3f(1.0));
}
fn effect(uv: vec2f) -> vec4f {
    let dir = uv - vec2f(0.5);
    let dist = length(dir) * 2.0;
    let amount = params.strength * pow(clamp(dist, 0.0, 1.0), max(params.falloff, 0.0));
    var acc = vec3f(0.0);
    var wsum = vec3f(0.0);
    for (var i = 0; i < ${SAMPLES}; i = i + 1) {
        let t = f32(i) / f32(${SAMPLES} - 1);
        let w = caSpectrum(t);
        let s = sampleInput0(uv + dir * amount * (0.5 - t) * 2.0);
        acc += s.rgb * w;
        wsum += w;
    }
    return vec4f(acc / max(wsum, vec3f(1e-5)), sampleInput0(uv).a);
}`,
            inputs: ['chain'],
            output: 'chain',
        });
        return { params: this.schema };
    }
}
