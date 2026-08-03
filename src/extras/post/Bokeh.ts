// src/extras/post/Bokeh.ts
import type { DeclareContext, EffectDeclaration } from './Effect';
import type { ParamSchema } from './types';
import { PostEffectBase } from './PostEffectBase';

export interface BokehOptions {
    /** Distance to the focus plane, in world units along the view axis. */
    focusDistance?: number;
    /** Distance from the focus plane at which blur reaches maximum. */
    focusRange?: number;
    /** Maximum blur radius, in half-res pixels. */
    maxRadius?: number;
    /** Extra weight for HDR highlights (>1 luminance), which is what makes
     *  bright points bloom into visible discs instead of averaging away. */
    highlight?: number;
}

/** Compile-time gather tap count. 48 golden-angle taps cover a disc without
 *  the ring artifacts a concentric kernel shows at this budget. */
const TAPS = 48;

const SCHEMA: ParamSchema = [
    { name: 'focusDistance', type: 'f32', default: 10.0 },
    { name: 'focusRange', type: 'f32', default: 6.0 },
    { name: 'maxRadius', type: 'f32', default: 12.0 },
    { name: 'highlight', type: 'f32', default: 1.0 },
];

/** Golden-angle spiral over the unit disc, emitted as a WGSL const array. */
function buildDiscWGSL(): string {
    const rows: string[] = [];
    for (let i = 0; i < TAPS; i++) {
        const r = Math.sqrt((i + 0.5) / TAPS);
        const th = i * 2.399963229728653;
        rows.push(`    vec2f(${(r * Math.cos(th)).toFixed(6)}, ${(r * Math.sin(th)).toFixed(6)})`);
    }
    return `const BOKEH_DISC = array<vec2f, ${TAPS}>(\n${rows.join(',\n')}\n);`;
}

// Note (1.0 - uv.y): undoing the composer's Y flip, same as SSAO.
const COC_WGSL = `fn viewDist(uv: vec2f, depth: f32) -> f32 {
    let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, min(depth, 0.99999), 1.0);
    let v = camera.invProj * ndc;
    return -(v.z / v.w);
}
fn cocAt(uv: vec2f, depth: f32) -> f32 {
    let d = viewDist(uv, depth);
    return clamp(abs(d - params.focusDistance) / max(params.focusRange, 1e-3), 0.0, 1.0);
}`;

// Half res: scene color in rgb, normalized circle of confusion in alpha.
const PREPARE_DEPTH = `${COC_WGSL}
fn effect(uv: vec2f) -> vec4f {
    return vec4f(sampleInput0(uv).rgb, cocAt(uv, sampleInput1(uv)));
}`;

// No depth buffer: constant CoC — a uniform full-screen bokeh blur.
const PREPARE_FLAT = `fn effect(uv: vec2f) -> vec4f {
    return vec4f(sampleInput0(uv).rgb, 1.0);
}`;

// Scatter-as-gather: each tap contributes only if its own CoC disc is large
// enough to reach this pixel, which stops sharp foreground detail from being
// smeared into a blurred background. No early-out branch — a return before
// the loop would make the textureSamples non-uniform and fail validation.
const GATHER = `${buildDiscWGSL()}
fn effect(uv: vec2f) -> vec4f {
    let center = sampleInput0(uv);
    let radiusPx = center.a * params.maxRadius;
    var acc = center.rgb * 1e-3;
    var wsum = 1e-3;
    for (var i = 0; i < ${TAPS}; i = i + 1) {
        let tapR = length(BOKEH_DISC[i]) * radiusPx;
        let s = sampleInput0(uv + BOKEH_DISC[i] * radiusPx * passInfo.texelSize);
        var w = clamp(s.a * params.maxRadius - tapR + 1.0, 0.0, 1.0);
        let luma = dot(s.rgb, vec3f(0.2126, 0.7152, 0.0722));
        w *= 1.0 + params.highlight * max(luma - 1.0, 0.0);
        acc += s.rgb * w;
        wsum += w;
    }
    return vec4f(acc / wsum, center.a);
}`;

// Full res: recompute CoC from full-res depth for a crisp focus edge, but let
// the half-res CoC (blur.a) widen it so out-of-focus areas don't halo.
const COMPOSITE_DEPTH = `${COC_WGSL}
fn effect(uv: vec2f) -> vec4f {
    let sharp = sampleInput0(uv);
    let blur = sampleInput1(uv);
    let coc = max(cocAt(uv, sampleInput2(uv)), blur.a * 0.5);
    return vec4f(mix(sharp.rgb, blur.rgb, smoothstep(0.05, 0.5, coc)), sharp.a);
}`;

const COMPOSITE_FLAT = `fn effect(uv: vec2f) -> vec4f {
    let sharp = sampleInput0(uv);
    let blur = sampleInput1(uv);
    return vec4f(mix(sharp.rgb, blur.rgb, smoothstep(0.05, 0.5, blur.a)), sharp.a);
}`;

/**
 * Three passes: prepare (half res color + CoC), disc gather (half res),
 * composite (full res). With gbuffer.depth the CoC comes from view-space
 * distance to the focus plane; without it the effect degrades to a uniform
 * full-screen bokeh blur rather than throwing. v1 is single-layer — near-field
 * blur does not spread over in-focus geometry; a separate near layer is v2.
 */
export class Bokeh extends PostEffectBase {
    public readonly name = 'Bokeh';

    constructor(opts: BokehOptions = {}) {
        super();
        this.initParams(SCHEMA, {
            focusDistance: opts.focusDistance,
            focusRange: opts.focusRange,
            maxRadius: opts.maxRadius,
            highlight: opts.highlight,
        });
    }

    public get focusDistance(): number { return this.getParam('focusDistance'); }
    public set focusDistance(v: number) { this.setParam('focusDistance', v); }
    public get focusRange(): number { return this.getParam('focusRange'); }
    public set focusRange(v: number) { this.setParam('focusRange', v); }
    public get maxRadius(): number { return this.getParam('maxRadius'); }
    public set maxRadius(v: number) { this.setParam('maxRadius', v); }
    public get highlight(): number { return this.getParam('highlight'); }
    public set highlight(v: number) { this.setParam('highlight', v); }

    public declare(ctx: DeclareContext): EffectDeclaration {
        const hasDepth = ctx.has('depth');
        const prepared = ctx.target({ scale: 0.5, format: 'rgba16float' }, 'bokehPrepared');
        const gathered = ctx.target({ scale: 0.5, format: 'rgba16float' }, 'bokehGathered');

        ctx.pass({
            name: 'Bokeh prepare',
            shaderCode: hasDepth ? PREPARE_DEPTH : PREPARE_FLAT,
            inputs: hasDepth ? ['chain', 'depth'] : ['chain'],
            output: prepared,
        });
        ctx.pass({ name: 'Bokeh gather', shaderCode: GATHER, inputs: [prepared], output: gathered });
        ctx.pass({
            name: 'Bokeh composite',
            shaderCode: hasDepth ? COMPOSITE_DEPTH : COMPOSITE_FLAT,
            inputs: hasDepth ? ['chain', gathered, 'depth'] : ['chain', gathered],
            output: 'chain',
        });
        return { params: this.schema };
    }
}
