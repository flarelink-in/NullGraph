// src/extras/post/LensFlare.ts
import type { DeclareContext, EffectDeclaration } from './Effect';
import type { ParamSchema } from './types';
import { PostEffectBase } from './PostEffectBase';

export interface LensFlareOptions {
    /** Luminance above which light contributes to flares. */
    threshold?: number;
    /** Ghost count along the flip vector. Compile-time, like Bloom's mips. */
    ghosts?: number;
    /** Spacing of the ghost chain (fraction of the center vector). */
    ghostDispersal?: number;
    /** Radius of the circular halo, in UV units. */
    haloWidth?: number;
    /** Radial RGB shift on each ghost sample, in UV units. */
    chromaShift?: number;
    /** Strength of the flares added back to the image. */
    intensity?: number;
}

const SCHEMA: ParamSchema = [
    { name: 'threshold', type: 'f32', default: 1.2 },
    { name: 'ghostDispersal', type: 'f32', default: 0.3 },
    { name: 'haloWidth', type: 'f32', default: 0.25 },
    { name: 'chromaShift', type: 'f32', default: 0.01 },
    { name: 'intensity', type: 'f32', default: 0.15 },
];

const EXTRACT = `fn effect(uv: vec2f) -> vec4f {
    let c = sampleInput0(uv).rgb;
    return vec4f(max(c - vec3f(params.threshold), vec3f(0.0)), 1.0);
}`;

// John Chapman-style pseudo flares: ghosts are samples of the flipped image
// marching through screen center; the halo is a fixed-radius ring of the same.
function featuresWGSL(ghosts: number): string {
    return `fn lfChroma(uv: vec2f, dir: vec2f) -> vec3f {
    let d = dir * params.chromaShift;
    return vec3f(sampleInput0(uv - d).r, sampleInput0(uv).g, sampleInput0(uv + d).b);
}
fn lfWeight(uv: vec2f) -> f32 {
    // Fade features toward the screen edge so ghosts pop smoothly, not at a border.
    return pow(max(1.0 - length(vec2f(0.5) - uv) / 0.7071, 0.0), 5.0);
}
fn effect(uv: vec2f) -> vec4f {
    let fuv = vec2f(1.0) - uv;
    let ghostVec = (vec2f(0.5) - fuv) * params.ghostDispersal;
    let dir = normalize(ghostVec + vec2f(1e-6, 0.0));
    var result = vec3f(0.0);
    for (var i = 0; i < ${ghosts}; i = i + 1) {
        let suv = fract(fuv + ghostVec * f32(i));
        result += lfChroma(suv, dir) * lfWeight(suv);
    }
    let huv = fract(fuv + dir * params.haloWidth);
    result += lfChroma(huv, dir) * lfWeight(huv);
    return vec4f(result, 1.0);
}`;
}

// Fixed 1.5-texel gaussian; flare features are low-frequency, one soft pass
// per axis is enough.
function blurWGSL(axis: 'x' | 'y'): string {
    const dir = axis === 'x' ? 'vec2f(passInfo.texelSize.x, 0.0)' : 'vec2f(0.0, passInfo.texelSize.y)';
    return `fn effect(uv: vec2f) -> vec4f {
    let step = ${dir} * 1.5;
    var w = array<f32, 5>(0.227027, 0.194595, 0.121622, 0.054054, 0.016216);
    var c = sampleInput0(uv).rgb * w[0];
    for (var i = 1; i < 5; i = i + 1) {
        let o = step * f32(i);
        c += sampleInput0(uv + o).rgb * w[i];
        c += sampleInput0(uv - o).rgb * w[i];
    }
    return vec4f(c, 1.0);
}`;
}

const COMPOSITE = `fn effect(uv: vec2f) -> vec4f {
    let scene = sampleInput0(uv);
    return vec4f(scene.rgb + sampleInput1(uv).rgb * params.intensity, scene.a);
}`;

/**
 * Five passes: threshold extract (half res), ghost/halo features (quarter),
 * separable blur (quarter), additive composite. No starburst or lens-dirt
 * modulation — both need an external texture, which pass inputs cannot bind.
 */
export class LensFlare extends PostEffectBase {
    public readonly name = 'LensFlare';
    private ghostCount: number;

    constructor(opts: LensFlareOptions = {}) {
        super();
        this.ghostCount = Math.max(1, Math.min(8, Math.floor(opts.ghosts ?? 4)));
        this.initParams(SCHEMA, {
            threshold: opts.threshold,
            ghostDispersal: opts.ghostDispersal,
            haloWidth: opts.haloWidth,
            chromaShift: opts.chromaShift,
            intensity: opts.intensity,
        });
    }

    public get threshold(): number { return this.getParam('threshold'); }
    public set threshold(v: number) { this.setParam('threshold', v); }
    public get ghostDispersal(): number { return this.getParam('ghostDispersal'); }
    public set ghostDispersal(v: number) { this.setParam('ghostDispersal', v); }
    public get haloWidth(): number { return this.getParam('haloWidth'); }
    public set haloWidth(v: number) { this.setParam('haloWidth', v); }
    public get chromaShift(): number { return this.getParam('chromaShift'); }
    public set chromaShift(v: number) { this.setParam('chromaShift', v); }
    public get intensity(): number { return this.getParam('intensity'); }
    public set intensity(v: number) { this.setParam('intensity', v); }

    public declare(ctx: DeclareContext): EffectDeclaration {
        const half = ctx.target({ scale: 0.5 }, 'flareExtract');
        const qA = ctx.target({ scale: 0.25 }, 'flareA');
        const qB = ctx.target({ scale: 0.25 }, 'flareB');
        const qC = ctx.target({ scale: 0.25 }, 'flareC');

        ctx.pass({ name: 'Flare extract', shaderCode: EXTRACT, inputs: ['chain'], output: half });
        ctx.pass({ name: 'Flare features', shaderCode: featuresWGSL(this.ghostCount), inputs: [half], output: qA });
        ctx.pass({ name: 'Flare blur H', shaderCode: blurWGSL('x'), inputs: [qA], output: qB });
        ctx.pass({ name: 'Flare blur V', shaderCode: blurWGSL('y'), inputs: [qB], output: qC });
        ctx.pass({ name: 'Flare composite', shaderCode: COMPOSITE, inputs: ['chain', qC], output: 'chain' });
        return { params: this.schema };
    }
}
