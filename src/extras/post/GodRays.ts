// src/extras/post/GodRays.ts
import type { DeclareContext, EffectDeclaration } from './Effect';
import type { ParamSchema } from './types';
import { PostEffectBase } from './PostEffectBase';

export interface GodRaysOptions {
  /** Screen-space origin of light rays in UV space [0..1, 0..1]. Default [0.5, 0.5]. */
  lightPosition?: [number, number];
  /** Marching sample count. Default 32. */
  samples?: number;
  /** Ray reach spread multiplier. Default 1.0. */
  density?: number;
  /** Brightness weight per radial step. Default 0.22. */
  weight?: number;
  /** Radial falloff decay per step. Default 0.965. */
  decay?: number;
  /** Minimum luminance to cast rays. Default 0.15. */
  threshold?: number;
  /** Overall intensity multiplier. Default 1.0. */
  exposure?: number;
  /** Color tint multiplier for the scattered rays. Default [1.0, 1.0, 1.0]. */
  tint?: [number, number, number];
}

const SCHEMA: ParamSchema = [
  { name: 'lightPosition', type: 'vec2f', default: [0.5, 0.5] },
  { name: 'density', type: 'f32', default: 1.0 },
  { name: 'weight', type: 'f32', default: 0.22 },
  { name: 'decay', type: 'f32', default: 0.965 },
  { name: 'threshold', type: 'f32', default: 0.15 },
  { name: 'exposure', type: 'f32', default: 1.0 },
  { name: 'tint', type: 'vec3f', default: [1.0, 1.0, 1.0] },
];

function godRaysWGSL(samples: number): string {
  return `fn effect(uv: vec2f) -> vec4f {
    let center = params.lightPosition;
    let base = sampleInput0(uv);
    let NUM_SAMPLES = ${samples};
    let density = params.density;
    let weight = params.weight;
    let decay = params.decay;
    let exposure = params.exposure;
    let threshold = params.threshold;

    var deltaTexCoord = (uv - center) * (1.0 / f32(NUM_SAMPLES) * density);
    var illumCoord = uv;
    var illuminationDecay = 1.0;
    var godRays = vec3f(0.0);

    for (var i = 0; i < NUM_SAMPLES; i = i + 1) {
        illumCoord -= deltaTexCoord;
        var samp = sampleInput0(illumCoord).rgb;
        let luma = max(samp.r, max(samp.g, samp.b));
        samp = samp * smoothstep(threshold, threshold * 2.5 + 0.1, luma);
        godRays += samp * illuminationDecay * weight;
        illuminationDecay *= decay;
    }

    let tintedRays = godRays * params.tint * exposure;
    return vec4f(base.rgb + tintedRays, base.a);
}`;
}

/**
 * Volumetric God Rays (Radial Crepuscular Light Scattering):
 * Marches along radial screen-space vectors from the light center,
 * accumulating decayed luminance to simulate atmospheric light shafts.
 */
export class GodRays extends PostEffectBase {
  public readonly name = 'GodRays';
  private readonly samples: number;

  constructor(opts: GodRaysOptions = {}) {
    super();
    this.samples = Math.max(8, Math.min(64, opts.samples ?? 32));
    this.initParams(SCHEMA, {
      lightPosition: opts.lightPosition,
      density: opts.density,
      weight: opts.weight,
      decay: opts.decay,
      threshold: opts.threshold,
      exposure: opts.exposure,
      tint: opts.tint,
    });
  }

  public get density(): number { return this.getParam('density'); }
  public set density(v: number) { this.setParam('density', v); }
  public get weight(): number { return this.getParam('weight'); }
  public set weight(v: number) { this.setParam('weight', v); }
  public get decay(): number { return this.getParam('decay'); }
  public set decay(v: number) { this.setParam('decay', v); }
  public get threshold(): number { return this.getParam('threshold'); }
  public set threshold(v: number) { this.setParam('threshold', v); }
  public get exposure(): number { return this.getParam('exposure'); }
  public set exposure(v: number) { this.setParam('exposure', v); }
  public setLightPosition(xy: [number, number]): void { this.setParam('lightPosition', xy); }
  public setTint(rgb: [number, number, number]): void { this.setParam('tint', rgb); }

  public declare(ctx: DeclareContext): EffectDeclaration {
    ctx.pass({
      name: 'GodRays',
      shaderCode: godRaysWGSL(this.samples),
      inputs: ['chain'],
      output: 'chain',
    });
    return { params: this.schema };
  }
}
