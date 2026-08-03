// src/extras/post/Kuwahara.ts
import type { DeclareContext, EffectDeclaration } from './Effect';
import type { ParamSchema } from './types';
import { PostEffectBase } from './PostEffectBase';

export interface KuwaharaOptions {
    /** Sector radius in pixels. Runtime-adjustable, clamped to MAX_RADIUS. */
    radius?: number;
}

/** Compile-time loop bound. 8 is 4 x 81 = 324 taps worst case — the runtime
 *  radius param exists precisely so that is opt-in, not the default. */
const MAX_RADIUS = 8;

const SCHEMA: ParamSchema = [
    { name: 'radius', type: 'u32', default: 4 },
];

/**
 * One pass, full resolution. Classic 4-sector Kuwahara: for each pixel take
 * the four (r+1)^2 quadrant windows that share it as a corner, and output the
 * mean of whichever has the lowest luminance variance. Variance never crosses
 * an edge, so edges stay crisp while surfaces flatten into painterly strokes.
 */
export class Kuwahara extends PostEffectBase {
    public readonly name = 'Kuwahara';

    constructor(opts: KuwaharaOptions = {}) {
        super();
        this.initParams(SCHEMA, { radius: opts.radius });
    }

    public get radius(): number { return this.getParam('radius'); }
    public set radius(v: number) { this.setParam('radius', Math.max(1, Math.min(MAX_RADIUS, Math.floor(v)))); }

    public declare(ctx: DeclareContext): EffectDeclaration {
        ctx.pass({
            name: 'Kuwahara',
            shaderCode: `fn kLuma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
fn effect(uv: vec2f) -> vec4f {
    let r = i32(clamp(params.radius, 1u, ${MAX_RADIUS}u));
    let center = sampleInput0(uv);
    var bestMean = center.rgb;
    var bestVar = 1e30;
    for (var q = 0; q < 4; q = q + 1) {
        let sgn = vec2f(f32(select(-1, 1, (q & 1) == 1)), f32(select(-1, 1, (q & 2) == 2)));
        var mean = vec3f(0.0);
        var lSum = 0.0;
        var lSq = 0.0;
        for (var y = 0; y <= r; y = y + 1) {
            for (var x = 0; x <= r; x = x + 1) {
                let c = sampleInput0(uv + vec2f(f32(x), f32(y)) * sgn * passInfo.texelSize).rgb;
                let l = kLuma(c);
                mean += c;
                lSum += l;
                lSq += l * l;
            }
        }
        let n = f32((r + 1) * (r + 1));
        mean /= n;
        let v = lSq / n - (lSum / n) * (lSum / n);
        if (v < bestVar) { bestVar = v; bestMean = mean; }
    }
    return vec4f(bestMean, center.a);
}`,
            inputs: ['chain'],
            output: 'chain',
        });
        return { params: this.schema };
    }
}
