// src/extras/post/Halation.ts
import type { DeclareContext, EffectDeclaration } from './Effect';
import type { ParamSchema } from './types';
import { PostEffectBase } from './PostEffectBase';

export interface HalationOptions {
    /** Luminance above which light starts scattering. */
    threshold?: number;
    /** Soft-knee width as a fraction of threshold (same shape as Bloom's). */
    knee?: number;
    /** Strength of the scattered glow added back to the image. */
    intensity?: number;
    /** Blur spread multiplier; 1 = ~9 quarter-res texels per axis. */
    radius?: number;
    /** Scatter color. Default is the orange-red of film emulsion halation. */
    tint?: [number, number, number];
}

const SCHEMA: ParamSchema = [
    { name: 'threshold', type: 'f32', default: 1.0 },
    { name: 'knee', type: 'f32', default: 0.5 },
    { name: 'intensity', type: 'f32', default: 0.6 },
    { name: 'radius', type: 'f32', default: 1.5 },
    { name: 'tint', type: 'vec3f', default: [1.0, 0.3, 0.12] },
];

// Same soft-knee prefilter as Bloom, so the two effects respond to the same
// highlights consistently when stacked.
const EXTRACT = `fn effect(uv: vec2f) -> vec4f {
    let raw = sampleInput0(uv).rgb;
    let c = clamp(raw, vec3f(0.0), vec3f(65000.0));
    let br = max(c.r, max(c.g, c.b));
    let knee = params.threshold * params.knee + 1e-5;
    var soft = clamp(br - params.threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 1e-5);
    let w = max(soft, br - params.threshold) / max(br, 1e-5);
    return vec4f(c * max(w, 0.0), 1.0);
}`;

// Source is 2x this target's resolution, so source texel = texelSize * 0.5.
const DOWN = `fn effect(uv: vec2f) -> vec4f {
    let st = passInfo.texelSize * 0.5;
    var c = sampleInput0(uv + st * vec2f(-1.0, -1.0)).rgb;
    c += sampleInput0(uv + st * vec2f( 1.0, -1.0)).rgb;
    c += sampleInput0(uv + st * vec2f(-1.0,  1.0)).rgb;
    c += sampleInput0(uv + st * vec2f( 1.0,  1.0)).rgb;
    return vec4f(c * 0.25, 1.0);
}`;

// Separable 9-tap gaussian; params.radius stretches the footprint.
function blurWGSL(axis: 'x' | 'y'): string {
    const dir = axis === 'x' ? 'vec2f(passInfo.texelSize.x, 0.0)' : 'vec2f(0.0, passInfo.texelSize.y)';
    return `fn effect(uv: vec2f) -> vec4f {
    let step = ${dir} * params.radius;
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
    let halo = sampleInput1(uv).rgb;
    return vec4f(scene.rgb + halo * params.tint * params.intensity, scene.a);
}`;

/**
 * Film-emulsion halation: bright light scatters through the film base and
 * re-exposes the red-sensitive layer, putting a warm glow around highlights.
 * Five passes: extract (half res), down (quarter), gaussian H, gaussian V,
 * additive tinted composite. Additive (not screen) because the chain is HDR
 * here — screen blending assumes [0,1] and crushes pre-tonemap values.
 */
export class Halation extends PostEffectBase {
    public readonly name = 'Halation';

    constructor(opts: HalationOptions = {}) {
        super();
        this.initParams(SCHEMA, {
            threshold: opts.threshold,
            knee: opts.knee,
            intensity: opts.intensity,
            radius: opts.radius,
            tint: opts.tint,
        });
    }

    public get threshold(): number { return this.getParam('threshold'); }
    public set threshold(v: number) { this.setParam('threshold', v); }
    public get knee(): number { return this.getParam('knee'); }
    public set knee(v: number) { this.setParam('knee', v); }
    public get intensity(): number { return this.getParam('intensity'); }
    public set intensity(v: number) { this.setParam('intensity', v); }
    public get radius(): number { return this.getParam('radius'); }
    public set radius(v: number) { this.setParam('radius', v); }
    public setTint(rgb: [number, number, number]): void { this.setParam('tint', rgb); }

    public declare(ctx: DeclareContext): EffectDeclaration {
        const half = ctx.target({ scale: 0.5 }, 'halationHalf');
        const qA = ctx.target({ scale: 0.25 }, 'halationA');
        const qB = ctx.target({ scale: 0.25 }, 'halationB');
        const qC = ctx.target({ scale: 0.25 }, 'halationC');

        ctx.pass({ name: 'Halation extract', shaderCode: EXTRACT, inputs: ['chain'], output: half });
        ctx.pass({ name: 'Halation down', shaderCode: DOWN, inputs: [half], output: qA });
        ctx.pass({ name: 'Halation blur H', shaderCode: blurWGSL('x'), inputs: [qA], output: qB });
        ctx.pass({ name: 'Halation blur V', shaderCode: blurWGSL('y'), inputs: [qB], output: qC });
        ctx.pass({ name: 'Halation composite', shaderCode: COMPOSITE, inputs: ['chain', qC], output: 'chain' });
        return { params: this.schema };
    }
}
