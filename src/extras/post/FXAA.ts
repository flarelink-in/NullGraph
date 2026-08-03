// src/extras/post/FXAA.ts
import type { DeclareContext, EffectDeclaration } from './Effect';
import type { ParamSchema } from './types';
import { PostEffectBase } from './PostEffectBase';

export interface FXAAOptions {
    /** Relative edge-contrast threshold; lower = more edges processed. */
    edgeThreshold?: number;
    /** Absolute floor below which contrast is ignored (dark-area skip). */
    edgeThresholdMin?: number;
    /** Sub-pixel aliasing removal amount (0 = off, 1 = maximum softening). */
    subpixel?: number;
}

const SCHEMA: ParamSchema = [
    { name: 'edgeThreshold', type: 'f32', default: 0.125 },
    { name: 'edgeThresholdMin', type: 'f32', default: 0.0312 },
    { name: 'subpixel', type: 'f32', default: 0.75 },
];

/**
 * One pass, full resolution. FXAA 3.11 quality preset (12-step edge walk).
 * Place AFTER Tonemap: the luma estimate assumes display-referred values, and
 * on HDR input the edge detector misfires on highlights.
 *
 * All taps go through textureSampleLevel on the composer-declared tex0 —
 * textureSample's implicit derivatives would be illegal inside the early-out
 * and the edge-walk loop (non-uniform control flow).
 */
export class FXAA extends PostEffectBase {
    public readonly name = 'FXAA';

    constructor(opts: FXAAOptions = {}) {
        super();
        this.initParams(SCHEMA, {
            edgeThreshold: opts.edgeThreshold,
            edgeThresholdMin: opts.edgeThresholdMin,
            subpixel: opts.subpixel,
        });
    }

    public get edgeThreshold(): number { return this.getParam('edgeThreshold'); }
    public set edgeThreshold(v: number) { this.setParam('edgeThreshold', v); }
    public get edgeThresholdMin(): number { return this.getParam('edgeThresholdMin'); }
    public set edgeThresholdMin(v: number) { this.setParam('edgeThresholdMin', v); }
    public get subpixel(): number { return this.getParam('subpixel'); }
    public set subpixel(v: number) { this.setParam('subpixel', v); }

    public declare(ctx: DeclareContext): EffectDeclaration {
        ctx.pass({
            name: 'FXAA',
            shaderCode: `fn fxaaTex(uv: vec2f) -> vec3f { return textureSampleLevel(tex0, sampLinear, uv, 0.0).rgb; }
// sqrt approximates the perceptual (gamma) luma FXAA was tuned for.
fn fxaaLuma(rgb: vec3f) -> f32 { return sqrt(dot(rgb, vec3f(0.299, 0.587, 0.114))); }
fn effect(uv: vec2f) -> vec4f {
    let ts = passInfo.texelSize;
    let center = textureSampleLevel(tex0, sampLinear, uv, 0.0);
    let lumaCenter = fxaaLuma(center.rgb);
    let lumaDown  = fxaaLuma(fxaaTex(uv + vec2f( 0.0, -ts.y)));
    let lumaUp    = fxaaLuma(fxaaTex(uv + vec2f( 0.0,  ts.y)));
    let lumaLeft  = fxaaLuma(fxaaTex(uv + vec2f(-ts.x,  0.0)));
    let lumaRight = fxaaLuma(fxaaTex(uv + vec2f( ts.x,  0.0)));

    let lumaMin = min(lumaCenter, min(min(lumaDown, lumaUp), min(lumaLeft, lumaRight)));
    let lumaMax = max(lumaCenter, max(max(lumaDown, lumaUp), max(lumaLeft, lumaRight)));
    let lumaRange = lumaMax - lumaMin;
    if (lumaRange < max(params.edgeThresholdMin, lumaMax * params.edgeThreshold)) {
        return center;
    }

    let lumaDownLeft  = fxaaLuma(fxaaTex(uv + vec2f(-ts.x, -ts.y)));
    let lumaUpRight   = fxaaLuma(fxaaTex(uv + vec2f( ts.x,  ts.y)));
    let lumaUpLeft    = fxaaLuma(fxaaTex(uv + vec2f(-ts.x,  ts.y)));
    let lumaDownRight = fxaaLuma(fxaaTex(uv + vec2f( ts.x, -ts.y)));

    let lumaDownUp = lumaDown + lumaUp;
    let lumaLeftRight = lumaLeft + lumaRight;
    let lumaLeftCorners = lumaDownLeft + lumaUpLeft;
    let lumaDownCorners = lumaDownLeft + lumaDownRight;
    let lumaRightCorners = lumaDownRight + lumaUpRight;
    let lumaUpCorners = lumaUpRight + lumaUpLeft;

    let edgeHorizontal = abs(-2.0 * lumaLeft + lumaLeftCorners)
        + abs(-2.0 * lumaCenter + lumaDownUp) * 2.0
        + abs(-2.0 * lumaRight + lumaRightCorners);
    let edgeVertical = abs(-2.0 * lumaUp + lumaUpCorners)
        + abs(-2.0 * lumaCenter + lumaLeftRight) * 2.0
        + abs(-2.0 * lumaDown + lumaDownCorners);
    let isHorizontal = edgeHorizontal >= edgeVertical;

    let luma1 = select(lumaLeft, lumaDown, isHorizontal);
    let luma2 = select(lumaRight, lumaUp, isHorizontal);
    let gradient1 = luma1 - lumaCenter;
    let gradient2 = luma2 - lumaCenter;
    let is1Steepest = abs(gradient1) >= abs(gradient2);
    let gradientScaled = 0.25 * max(abs(gradient1), abs(gradient2));

    var stepLength = select(ts.x, ts.y, isHorizontal);
    var lumaLocalAverage = 0.0;
    if (is1Steepest) {
        stepLength = -stepLength;
        lumaLocalAverage = 0.5 * (luma1 + lumaCenter);
    } else {
        lumaLocalAverage = 0.5 * (luma2 + lumaCenter);
    }
    var currentUv = uv;
    if (isHorizontal) {
        currentUv.y += stepLength * 0.5;
    } else {
        currentUv.x += stepLength * 0.5;
    }

    let offset = select(vec2f(0.0, ts.y), vec2f(ts.x, 0.0), isHorizontal);
    var uv1 = currentUv - offset;
    var uv2 = currentUv + offset;
    var lumaEnd1 = fxaaLuma(fxaaTex(uv1)) - lumaLocalAverage;
    var lumaEnd2 = fxaaLuma(fxaaTex(uv2)) - lumaLocalAverage;
    var reached1 = abs(lumaEnd1) >= gradientScaled;
    var reached2 = abs(lumaEnd2) >= gradientScaled;
    if (!reached1) { uv1 -= offset; }
    if (!reached2) { uv2 += offset; }

    var q = array<f32, 12>(1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0);
    if (!(reached1 && reached2)) {
        for (var i = 2; i < 12; i = i + 1) {
            if (!reached1) { lumaEnd1 = fxaaLuma(fxaaTex(uv1)) - lumaLocalAverage; }
            if (!reached2) { lumaEnd2 = fxaaLuma(fxaaTex(uv2)) - lumaLocalAverage; }
            reached1 = abs(lumaEnd1) >= gradientScaled;
            reached2 = abs(lumaEnd2) >= gradientScaled;
            if (reached1 && reached2) { break; }
            if (!reached1) { uv1 -= offset * q[i]; }
            if (!reached2) { uv2 += offset * q[i]; }
        }
    }

    let distance1 = select(uv.y - uv1.y, uv.x - uv1.x, isHorizontal);
    let distance2 = select(uv2.y - uv.y, uv2.x - uv.x, isHorizontal);
    let isDirection1 = distance1 < distance2;
    let distanceFinal = min(distance1, distance2);
    let edgeThickness = distance1 + distance2;

    let isLumaCenterSmaller = lumaCenter < lumaLocalAverage;
    let correctVariation1 = (lumaEnd1 < 0.0) != isLumaCenterSmaller;
    let correctVariation2 = (lumaEnd2 < 0.0) != isLumaCenterSmaller;
    let correctVariation = select(correctVariation2, correctVariation1, isDirection1);
    var finalOffset = select(0.0, -distanceFinal / edgeThickness + 0.5, correctVariation);

    let lumaAverage = (1.0 / 12.0) * (2.0 * (lumaDownUp + lumaLeftRight) + lumaLeftCorners + lumaRightCorners);
    let subPixelOffset1 = clamp(abs(lumaAverage - lumaCenter) / lumaRange, 0.0, 1.0);
    let subPixelOffset2 = (-2.0 * subPixelOffset1 + 3.0) * subPixelOffset1 * subPixelOffset1;
    finalOffset = max(finalOffset, subPixelOffset2 * subPixelOffset2 * params.subpixel);

    var finalUv = uv;
    if (isHorizontal) {
        finalUv.y += finalOffset * stepLength;
    } else {
        finalUv.x += finalOffset * stepLength;
    }
    return vec4f(fxaaTex(finalUv), center.a);
}`,
            inputs: ['chain'],
            output: 'chain',
        });
        return { params: this.schema };
    }
}
