// src/extras/post/Bloom.ts
import type { DeclareContext, EffectDeclaration } from './Effect';
import type { ParamSchema, TargetHandle } from './types';
import { PostEffectBase } from './PostEffectBase';

export interface BloomOptions {
    threshold?: number;
    knee?: number;
    intensity?: number;
    /** Pyramid depth. Clamped so the smallest mip stays >= 8 px per side. */
    mips?: number;
}

const SCHEMA: ParamSchema = [
    { name: 'threshold', type: 'f32', default: 1.15 },
    { name: 'knee', type: 'f32', default: 0.5 },
    { name: 'intensity', type: 'f32', default: 0.7 },
];

// Soft-knee prefilter (Jimenez / Unity-style quadratic threshold).
const PREFILTER = `fn effect(uv: vec2f) -> vec4f {
    let raw = sampleInput0(uv).rgb;
    let c = clamp(raw, vec3f(0.0), vec3f(65000.0));
    let br = max(c.r, max(c.g, c.b));
    let knee = params.threshold * params.knee + 1e-5;
    var soft = clamp(br - params.threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 1e-5);
    let w = max(soft, br - params.threshold) / max(br, 1e-5);
    return vec4f(c * max(w, 0.0), 1.0);
}`;

// 13-tap Jimenez downsample. The source is 2x this target's resolution, so
// source texel size = passInfo.texelSize * 0.5.
const DOWNSAMPLE = `fn effect(uv: vec2f) -> vec4f {
    let st = passInfo.texelSize * 0.5;
    let a = sampleInput0(uv + st * vec2f(-2.0, -2.0)).rgb;
    let b = sampleInput0(uv + st * vec2f( 0.0, -2.0)).rgb;
    let c = sampleInput0(uv + st * vec2f( 2.0, -2.0)).rgb;
    let d = sampleInput0(uv + st * vec2f(-1.0, -1.0)).rgb;
    let e = sampleInput0(uv + st * vec2f( 1.0, -1.0)).rgb;
    let f = sampleInput0(uv + st * vec2f(-2.0,  0.0)).rgb;
    let g = sampleInput0(uv).rgb;
    let h = sampleInput0(uv + st * vec2f( 2.0,  0.0)).rgb;
    let i = sampleInput0(uv + st * vec2f(-1.0,  1.0)).rgb;
    let j = sampleInput0(uv + st * vec2f( 1.0,  1.0)).rgb;
    let k = sampleInput0(uv + st * vec2f(-2.0,  2.0)).rgb;
    let l = sampleInput0(uv + st * vec2f( 0.0,  2.0)).rgb;
    let m = sampleInput0(uv + st * vec2f( 2.0,  2.0)).rgb;
    var col = (d + e + i + j) * 0.125;
    col += (a + b + f + g) * 0.03125;
    col += (b + c + g + h) * 0.03125;
    col += (f + g + k + l) * 0.03125;
    col += (g + h + l + m) * 0.03125;
    return vec4f(col, 1.0);
}`;

// 9-tap tent upsample of the smaller mip (input 0), added onto the same-level
// downsample (input 1). Smaller-mip texel size = passInfo.texelSize * 2.0.
const UPSAMPLE = `fn effect(uv: vec2f) -> vec4f {
    let st = passInfo.texelSize * 2.0;
    var s = sampleInput0(uv + st * vec2f(-1.0, -1.0)).rgb;
    s += sampleInput0(uv + st * vec2f( 0.0, -1.0)).rgb * 2.0;
    s += sampleInput0(uv + st * vec2f( 1.0, -1.0)).rgb;
    s += sampleInput0(uv + st * vec2f(-1.0,  0.0)).rgb * 2.0;
    s += sampleInput0(uv).rgb * 4.0;
    s += sampleInput0(uv + st * vec2f( 1.0,  0.0)).rgb * 2.0;
    s += sampleInput0(uv + st * vec2f(-1.0,  1.0)).rgb;
    s += sampleInput0(uv + st * vec2f( 0.0,  1.0)).rgb * 2.0;
    s += sampleInput0(uv + st * vec2f( 1.0,  1.0)).rgb;
    return vec4f(sampleInput1(uv).rgb + s / 16.0, 1.0);
}`;

const COMPOSITE = `fn effect(uv: vec2f) -> vec4f {
    let scene = sampleInput0(uv);
    let bloom = sampleInput1(uv).rgb;
    return vec4f(scene.rgb + bloom * params.intensity, scene.a);
}`;

/**
 * 2*mips passes: prefilter, (mips-1) downs, (mips-1) ups, composite.
 * Each mip is a separate logical target — not a mip level of one texture —
 * which is what lets the pool reason about them.
 */
export class Bloom extends PostEffectBase {
    public readonly name = 'Bloom';
    private mipsRequested: number;

    constructor(opts: BloomOptions = {}) {
        super();
        this.mipsRequested = opts.mips ?? 6;
        this.initParams(SCHEMA, {
            threshold: opts.threshold,
            knee: opts.knee,
            intensity: opts.intensity,
        });
    }

    public get threshold(): number { return this.getParam('threshold'); }
    public set threshold(v: number) { this.setParam('threshold', v); }
    public get knee(): number { return this.getParam('knee'); }
    public set knee(v: number) { this.setParam('knee', v); }
    public get intensity(): number { return this.getParam('intensity'); }
    public set intensity(v: number) { this.setParam('intensity', v); }

    public declare(ctx: DeclareContext): EffectDeclaration {
        // Mip i lives at scale 0.5^(i+1); clamp so the smallest stays >= 8 px
        // per side (a 3x2 mip produces visible pulsing).
        const maxMips = Math.max(1, Math.floor(Math.log2(Math.min(ctx.width, ctx.height) / 8)));
        const mips = Math.min(Math.max(1, Math.floor(this.mipsRequested)), maxMips);

        const B: TargetHandle[] = [];
        const U: TargetHandle[] = [];
        for (let i = 0; i < mips; i++) {
            B.push(ctx.target({ scale: Math.pow(0.5, i + 1) }, `bloomB${i}`));
        }
        for (let i = 0; i < mips - 1; i++) {
            U.push(ctx.target({ scale: Math.pow(0.5, i + 1) }, `bloomU${i}`));
        }

        ctx.pass({ name: 'Bloom prefilter', shaderCode: PREFILTER, inputs: ['chain'], output: B[0] });
        for (let i = 1; i < mips; i++) {
            ctx.pass({ name: `Bloom down ${i}`, shaderCode: DOWNSAMPLE, inputs: [B[i - 1]], output: B[i] });
        }

        let lowest: TargetHandle = B[mips - 1];
        for (let i = mips - 2; i >= 0; i--) {
            ctx.pass({ name: `Bloom up ${i}`, shaderCode: UPSAMPLE, inputs: [lowest, B[i]], output: U[i] });
            lowest = U[i];
        }

        ctx.pass({ name: 'Bloom composite', shaderCode: COMPOSITE, inputs: ['chain', lowest], output: 'chain' });
        return { params: this.schema };
    }
}
