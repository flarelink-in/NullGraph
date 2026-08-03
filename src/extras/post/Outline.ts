// src/extras/post/Outline.ts
import type { DeclareContext, EffectDeclaration } from './Effect';
import type { ParamSchema } from './types';
import { PostEffectBase } from './PostEffectBase';

export interface OutlineOptions {
    /** Line width in pixels (Sobel sample spread). */
    thickness?: number;
    /** Response to depth discontinuities (silhouettes, object borders). */
    depthSensitivity?: number;
    /** Response to luminance discontinuities (interior detail lines). */
    lumaSensitivity?: number;
    /** Ink strength; 1 = fully opaque lines. */
    strength?: number;
    /** Ink color. Default black. */
    color?: [number, number, number];
}

const SCHEMA: ParamSchema = [
    { name: 'thickness', type: 'f32', default: 1.5 },
    { name: 'depthSensitivity', type: 'f32', default: 8.0 },
    { name: 'lumaSensitivity', type: 'f32', default: 0.6 },
    { name: 'strength', type: 'f32', default: 1.0 },
    { name: 'color', type: 'vec3f', default: [0, 0, 0] },
];

// Sobel on linearized depth, normalized by the center distance so a 10 cm
// step draws the same line at 2 m and at 20 m, plus Sobel on luma for
// interior lines depth cannot see (texture borders, shading creases).
const WITH_DEPTH = `fn olLinDepth(d: f32) -> f32 {
    return camera.near * camera.far / (camera.far - d * (camera.far - camera.near));
}
fn olLuma(uv: vec2f) -> f32 { return dot(sampleInput0(uv).rgb, vec3f(0.2126, 0.7152, 0.0722)); }
fn effect(uv: vec2f) -> vec4f {
    let c = sampleInput0(uv);
    let o = passInfo.texelSize * params.thickness;

    let d00 = olLinDepth(sampleInput1(uv + vec2f(-o.x, -o.y)));
    let d10 = olLinDepth(sampleInput1(uv + vec2f( 0.0, -o.y)));
    let d20 = olLinDepth(sampleInput1(uv + vec2f( o.x, -o.y)));
    let d01 = olLinDepth(sampleInput1(uv + vec2f(-o.x,  0.0)));
    let dc  = olLinDepth(sampleInput1(uv));
    let d21 = olLinDepth(sampleInput1(uv + vec2f( o.x,  0.0)));
    let d02 = olLinDepth(sampleInput1(uv + vec2f(-o.x,  o.y)));
    let d12 = olLinDepth(sampleInput1(uv + vec2f( 0.0,  o.y)));
    let d22 = olLinDepth(sampleInput1(uv + vec2f( o.x,  o.y)));
    let dgx = (d20 + 2.0 * d21 + d22) - (d00 + 2.0 * d01 + d02);
    let dgy = (d02 + 2.0 * d12 + d22) - (d00 + 2.0 * d10 + d20);
    let depthEdge = length(vec2f(dgx, dgy)) / max(dc, 1e-3) * params.depthSensitivity;

    let l00 = olLuma(uv + vec2f(-o.x, -o.y));
    let l10 = olLuma(uv + vec2f( 0.0, -o.y));
    let l20 = olLuma(uv + vec2f( o.x, -o.y));
    let l01 = olLuma(uv + vec2f(-o.x,  0.0));
    let l21 = olLuma(uv + vec2f( o.x,  0.0));
    let l02 = olLuma(uv + vec2f(-o.x,  o.y));
    let l12 = olLuma(uv + vec2f( 0.0,  o.y));
    let l22 = olLuma(uv + vec2f( o.x,  o.y));
    let lgx = (l20 + 2.0 * l21 + l22) - (l00 + 2.0 * l01 + l02);
    let lgy = (l02 + 2.0 * l12 + l22) - (l00 + 2.0 * l10 + l20);
    let lumaEdge = length(vec2f(lgx, lgy)) * params.lumaSensitivity;

    let edge = clamp(max(depthEdge, lumaEdge), 0.0, 1.0);
    return vec4f(mix(c.rgb, params.color, edge * clamp(params.strength, 0.0, 1.0)), c.a);
}`;

// No depth buffer: luma edges only. Silhouettes against similar-luminance
// backgrounds will not outline — depth is what makes this effect solid.
const LUMA_ONLY = `fn olLuma(uv: vec2f) -> f32 { return dot(sampleInput0(uv).rgb, vec3f(0.2126, 0.7152, 0.0722)); }
fn effect(uv: vec2f) -> vec4f {
    let c = sampleInput0(uv);
    let o = passInfo.texelSize * params.thickness;
    let l00 = olLuma(uv + vec2f(-o.x, -o.y));
    let l10 = olLuma(uv + vec2f( 0.0, -o.y));
    let l20 = olLuma(uv + vec2f( o.x, -o.y));
    let l01 = olLuma(uv + vec2f(-o.x,  0.0));
    let l21 = olLuma(uv + vec2f( o.x,  0.0));
    let l02 = olLuma(uv + vec2f(-o.x,  o.y));
    let l12 = olLuma(uv + vec2f( 0.0,  o.y));
    let l22 = olLuma(uv + vec2f( o.x,  o.y));
    let lgx = (l20 + 2.0 * l21 + l22) - (l00 + 2.0 * l01 + l02);
    let lgy = (l02 + 2.0 * l12 + l22) - (l00 + 2.0 * l10 + l20);
    let edge = clamp(length(vec2f(lgx, lgy)) * params.lumaSensitivity, 0.0, 1.0);
    return vec4f(mix(c.rgb, params.color, edge * clamp(params.strength, 0.0, 1.0)), c.a);
}`;

/**
 * One pass, full resolution. Ink lines from a Sobel over linearized depth
 * (silhouettes) and luminance (interior detail); falls back to luma-only
 * without gbuffer.depth. Place AFTER Tonemap and CelShade so lines are drawn
 * on the final flat colors, and BEFORE FXAA so the line edges get smoothed.
 */
export class Outline extends PostEffectBase {
    public readonly name = 'Outline';

    constructor(opts: OutlineOptions = {}) {
        super();
        this.initParams(SCHEMA, {
            thickness: opts.thickness,
            depthSensitivity: opts.depthSensitivity,
            lumaSensitivity: opts.lumaSensitivity,
            strength: opts.strength,
            color: opts.color,
        });
    }

    public get thickness(): number { return this.getParam('thickness'); }
    public set thickness(v: number) { this.setParam('thickness', v); }
    public get depthSensitivity(): number { return this.getParam('depthSensitivity'); }
    public set depthSensitivity(v: number) { this.setParam('depthSensitivity', v); }
    public get lumaSensitivity(): number { return this.getParam('lumaSensitivity'); }
    public set lumaSensitivity(v: number) { this.setParam('lumaSensitivity', v); }
    public get strength(): number { return this.getParam('strength'); }
    public set strength(v: number) { this.setParam('strength', v); }
    public setColor(rgb: [number, number, number]): void { this.setParam('color', rgb); }

    public declare(ctx: DeclareContext): EffectDeclaration {
        const hasDepth = ctx.has('depth');
        ctx.pass({
            name: 'Outline',
            shaderCode: hasDepth ? WITH_DEPTH : LUMA_ONLY,
            inputs: hasDepth ? ['chain', 'depth'] : ['chain'],
            output: 'chain',
        });
        return { params: this.schema };
    }
}
