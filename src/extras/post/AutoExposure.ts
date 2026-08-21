// src/extras/post/AutoExposure.ts
import type { DeclareContext, EffectDeclaration } from './Effect';
import type { ParamSchema } from './types';
import { PostEffectBase } from './PostEffectBase';

export interface AutoExposureOptions {
    /** Target mid-grey the average luminance is mapped to. */
    key?: number;
    /** Per-frame retention of the previous exposure (0 = instant, 0.99 = very
     *  slow). Frame-rate dependent by design — the chain has no dt and the
     *  16-bit history texture cannot hold an absolute timestamp accurately. */
    smoothing?: number;
    /** Exposure clamp floor — stops night scenes being pulled to daylight. */
    minExposure?: number;
    /** Exposure clamp ceiling. */
    maxExposure?: number;
}

const SCHEMA: ParamSchema = [
    { name: 'key', type: 'f32', default: 0.18 },
    { name: 'smoothing', type: 'f32', default: 0.95 },
    { name: 'minExposure', type: 'f32', default: 0.1 },
    { name: 'maxExposure', type: 'f32', default: 8.0 },
];

const LUMA_WGSL = `fn aeLogLuma(c: vec3f) -> f32 {
    return log2(max(dot(c, vec3f(0.2126, 0.7152, 0.0722)), 1e-5));
}`;

// Log-average (geometric mean) metering: arithmetic mean lets one bright
// window dominate an interior scene.
const MEASURE = `${LUMA_WGSL}
fn effect(uv: vec2f) -> vec4f {
    let st = passInfo.texelSize * 0.25;
    var s = aeLogLuma(sampleInput0(uv + st * vec2f(-1.0, -1.0)).rgb);
    s += aeLogLuma(sampleInput0(uv + st * vec2f( 1.0, -1.0)).rgb);
    s += aeLogLuma(sampleInput0(uv + st * vec2f(-1.0,  1.0)).rgb);
    s += aeLogLuma(sampleInput0(uv + st * vec2f( 1.0,  1.0)).rgb);
    return vec4f(s * 0.25, 0.0, 0.0, 1.0);
}`;

// Box-reduce the log values; averaging logs keeps the geometric mean.
const REDUCE = `fn effect(uv: vec2f) -> vec4f {
    let st = passInfo.texelSize * 0.5;
    var s = sampleInput0(uv + st * vec2f(-1.0, -1.0)).r;
    s += sampleInput0(uv + st * vec2f( 1.0, -1.0)).r;
    s += sampleInput0(uv + st * vec2f(-1.0,  1.0)).r;
    s += sampleInput0(uv + st * vec2f( 1.0,  1.0)).r;
    return vec4f(s * 0.25, 0.0, 0.0, 1.0);
}`;

// 1x1 history target: r = adapted exposure, g = last measured average
// luminance (kept for debugging). Input 1 is this target's PREVIOUS frame —
// the pass both reads and writes it, which is what selects historyView.
const ADAPT = `fn effect(uv: vec2f) -> vec4f {
    var avg = 0.0;
    for (var y = 0; y < 4; y = y + 1) {
        for (var x = 0; x < 6; x = x + 1) {
            avg += sampleInput0((vec2f(f32(x), f32(y)) + vec2f(0.5)) / vec2f(6.0, 4.0)).r;
        }
    }
    let avgLum = exp2(avg / 24.0);
    let targetExp = clamp(params.key / max(avgLum, 1e-5), params.minExposure, params.maxExposure);
    let prev = sampleInput1(vec2f(0.5)).r;
    // Frame 0 reads the pool's zero-clear: jump straight to target.
    var adapted = mix(targetExp, prev, clamp(params.smoothing, 0.0, 0.999));
    if (prev <= 0.0) { adapted = targetExp; }
    return vec4f(adapted, avgLum, 0.0, 1.0);
}`;

const APPLY = `fn effect(uv: vec2f) -> vec4f {
    let c = sampleInput0(uv);
    let e = sampleInput1(vec2f(0.5)).r;
    return vec4f(c.rgb * e, c.a);
}`;

/**
 * Five passes: log-luma measure (1/8), two box reductions (1/32, 1/128),
 * a 1x1 adapt pass with a double-buffered history texture for temporal
 * smoothing, and the multiply. Whole-frame metering in v1 (no center
 * weighting). Place before Tonemap so the curve receives normalized input;
 * if a manual Tonemap.exposure is also set the two multiply.
 */
export class AutoExposure extends PostEffectBase {
    public readonly name = 'AutoExposure';

    constructor(opts: AutoExposureOptions = {}) {
        super();
        this.initParams(SCHEMA, {
            key: opts.key,
            smoothing: opts.smoothing,
            minExposure: opts.minExposure,
            maxExposure: opts.maxExposure,
        });
    }

    public get key(): number { return this.getParam('key'); }
    public set key(v: number) { this.setParam('key', v); }
    public get smoothing(): number { return this.getParam('smoothing'); }
    public set smoothing(v: number) { this.setParam('smoothing', v); }
    public get minExposure(): number { return this.getParam('minExposure'); }
    public set minExposure(v: number) { this.setParam('minExposure', v); }
    public get maxExposure(): number { return this.getParam('maxExposure'); }
    public set maxExposure(v: number) { this.setParam('maxExposure', v); }

    public declare(ctx: DeclareContext): EffectDeclaration {
        const l0 = ctx.target({ scale: 1 / 8, format: 'r16float' }, 'aeLuma');
        const l1 = ctx.target({ scale: 1 / 32, format: 'r16float' }, 'aeReduce1');
        const l2 = ctx.target({ scale: 1 / 128, format: 'r16float' }, 'aeReduce2');
        // scale ~0 floors to 1x1 in the pool. rg16float: exposure + debug lum.
        const state = ctx.target({ scale: 1e-6, format: 'rg16float', history: 2 }, 'aeState');

        ctx.pass({ name: 'AE measure', shaderCode: MEASURE, inputs: ['chain'], output: l0 });
        ctx.pass({ name: 'AE reduce 1', shaderCode: REDUCE, inputs: [l0], output: l1 });
        ctx.pass({ name: 'AE reduce 2', shaderCode: REDUCE, inputs: [l1], output: l2 });
        ctx.pass({ name: 'AE adapt', shaderCode: ADAPT, inputs: [l2, state], output: state });
        ctx.pass({ name: 'AE apply', shaderCode: APPLY, inputs: ['chain', state], output: 'chain' });
        return { params: this.schema };
    }
}
