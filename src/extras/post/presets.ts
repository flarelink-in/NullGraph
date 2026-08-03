// src/extras/post/presets.ts
//
// Implements docs/post-presets.md. Two tiers, named so nobody is surprised
// by a recompile mid-frame:
//   Presets.*          -> constructor option objects. Full range, needs setChain()/compile().
//   Presets.*.runtime  -> the runtime-only subset, safe to Object.assign per-frame.
//
// Chain presets are FACTORIES, never shared instances: PostChain rejects a
// duplicate instance, and each effect holds a live subarray of the param arena.
//
// Stage order used by every chain is the canonical EFFECT_ORDER in
// chainOrder.ts (see the doc's §2 for the reasoning): HDR effects before
// Tonemap; stylization on display-referred values after; FXAA after Outline so
// ink edges get smoothed; grain last so it stays crisp.

import { Bloom, type BloomOptions } from './Bloom';
import { SSAO, type SSAOOptions } from './SSAO';
import { Tonemap, type TonemapOptions } from './Tonemap';
import { FXAA } from './FXAA';
import { ColorGrade } from './ColorGrade';
import { Vignette } from './Vignette';
import { FilmGrain } from './FilmGrain';
import { Halation } from './Halation';
import { ChromaticAberration } from './ChromaticAberration';
import { Kuwahara } from './Kuwahara';
import { CelShade } from './CelShade';
import { Outline } from './Outline';
import type { Effect } from './Effect';

/** Runtime-safe subsets: applying these never triggers a recompile. */
type BloomRuntime = Pick<Required<BloomOptions>, 'threshold' | 'knee' | 'intensity'>;
type SSAORuntime = Pick<Required<SSAOOptions>, 'radius' | 'bias' | 'intensity' | 'samples'>;
type TonemapRuntime = Pick<Required<TonemapOptions>, 'exposure' | 'whitePoint'>;

type Preset<Opts, Runtime> = Readonly<Opts> & { readonly runtime: Readonly<Runtime> };

function preset<Opts extends object, R extends object>(opts: Opts, runtime: R): Preset<Opts, R> {
    return Object.freeze({ ...opts, runtime: Object.freeze(runtime) }) as Preset<Opts, R>;
}

// ---------------------------------------------------------------- Tonemap

export const TonemapPresets = {
    /** Default. Film-like rolloff. */
    neutral: preset<TonemapOptions, TonemapRuntime>(
        { curve: 'aces', exposure: 1.0, whitePoint: 4.0 }, { exposure: 1.0, whitePoint: 4.0 }),
    /** Brighter, wider highlight latitude. */
    filmic: preset<TonemapOptions, TonemapRuntime>(
        { curve: 'aces', exposure: 1.15, whitePoint: 6.0 }, { exposure: 1.15, whitePoint: 6.0 }),
    /** Best hue retention at extremes. Archviz, product. */
    natural: preset<TonemapOptions, TonemapRuntime>(
        { curve: 'agx', exposure: 1.0, whitePoint: 4.0 }, { exposure: 1.0, whitePoint: 4.0 }),
    /** Low contrast, never clips. */
    soft: preset<TonemapOptions, TonemapRuntime>(
        { curve: 'reinhard', exposure: 1.0, whitePoint: 2.5 }, { exposure: 1.0, whitePoint: 2.5 }),
    /** Clamp only. The only preset that preserves authored colors exactly. */
    faithful: preset<TonemapOptions, TonemapRuntime>(
        { curve: 'none', exposure: 1.0, whitePoint: 1.0 }, { exposure: 1.0, whitePoint: 1.0 }),
} as const;

// ------------------------------------------------------------------ Bloom
// `threshold` is LINEAR HDR, pre-tonemap. Values > 1.0 mean "brighter than
// white" — a scene with no emissives or lights above 1.0 will not bloom.
// High `mips` presets clamp by resolution (smallest mip >= 8 px), so the
// glow is wider at 4K than at 720p.

export const BloomPresets = {
    subtle: preset<BloomOptions, BloomRuntime>(
        { threshold: 1.40, knee: 0.30, intensity: 0.25, mips: 5 },
        { threshold: 1.40, knee: 0.30, intensity: 0.25 }),
    natural: preset<BloomOptions, BloomRuntime>(
        { threshold: 1.15, knee: 0.50, intensity: 0.50, mips: 6 },
        { threshold: 1.15, knee: 0.50, intensity: 0.50 }),
    cinematic: preset<BloomOptions, BloomRuntime>(
        { threshold: 1.00, knee: 0.60, intensity: 0.80, mips: 7 },
        { threshold: 1.00, knee: 0.60, intensity: 0.80 }),
    dreamy: preset<BloomOptions, BloomRuntime>(
        { threshold: 0.60, knee: 0.80, intensity: 1.00, mips: 7 },
        { threshold: 0.60, knee: 0.80, intensity: 1.00 }),
    neon: preset<BloomOptions, BloomRuntime>(
        { threshold: 0.90, knee: 0.20, intensity: 1.40, mips: 7 },
        { threshold: 0.90, knee: 0.20, intensity: 1.40 }),
} as const;

// ------------------------------------------------------------------- SSAO
// `radius` and `bias` are VIEW-SPACE WORLD UNITS. These values assume a
// ~1-unit-per-meter scene. Use forSceneScale() for other conventions.

export const SSAOPresets = {
    performance: preset<SSAOOptions, SSAORuntime>(
        { radius: 0.50, bias: 0.025, intensity: 0.80, samples: 8 },
        { radius: 0.50, bias: 0.025, intensity: 0.80, samples: 8 }),
    balanced: preset<SSAOOptions, SSAORuntime>(
        { radius: 0.60, bias: 0.020, intensity: 0.90, samples: 16 },
        { radius: 0.60, bias: 0.020, intensity: 0.90, samples: 16 }),
    quality: preset<SSAOOptions, SSAORuntime>(
        { radius: 0.80, bias: 0.020, intensity: 1.00, samples: 32 },
        { radius: 0.80, bias: 0.020, intensity: 1.00, samples: 32 }),
    contact: preset<SSAOOptions, SSAORuntime>(
        { radius: 0.25, bias: 0.015, intensity: 1.20, samples: 16 },
        { radius: 0.25, bias: 0.015, intensity: 1.20, samples: 16 }),
    ambient: preset<SSAOOptions, SSAORuntime>(
        { radius: 1.50, bias: 0.035, intensity: 0.70, samples: 24 },
        { radius: 1.50, bias: 0.035, intensity: 0.70, samples: 24 }),
} as const;

/** Adapt a radius-bearing SSAO preset from meters to the scene's unit scale. */
export function forSceneScale(preset: SSAOOptions, unitsPerMeter: number): SSAOOptions {
    return {
        ...preset,
        radius: (preset.radius ?? 0.6) * unitsPerMeter,
        bias: (preset.bias ?? 0.02) * unitsPerMeter,
    };
}

// ----------------------------------------------------------------- Chains
// Complete looks. SSAO and (full-quality) Outline need the chain built with
// gbuffer.depth; Outline degrades to luma-only edges without it, SSAO throws.

export const ChainPresets = {
    /** Dataviz / UI. Authored colors survive to the display unchanged. */
    faithful: (): Effect[] => [new Tonemap(TonemapPresets.faithful)],

    /** Correct HDR->display and nothing else. */
    neutral: (): Effect[] => [new Tonemap(TonemapPresets.neutral)],

    product: (): Effect[] => [
        new SSAO(SSAOPresets.contact),
        new Bloom(BloomPresets.subtle),
        new Tonemap(TonemapPresets.natural)],

    archviz: (): Effect[] => [
        new SSAO(SSAOPresets.ambient),
        new Bloom(BloomPresets.subtle),
        new Tonemap({ ...TonemapPresets.natural, exposure: 1.1 })],

    cinematic: (): Effect[] => [
        new SSAO(SSAOPresets.balanced),
        new Bloom(BloomPresets.cinematic),
        new Tonemap(TonemapPresets.filmic)],

    /** No SSAO: contact shadows fight the haze. */
    ethereal: (): Effect[] => [
        new Bloom(BloomPresets.dreamy),
        new Tonemap(TonemapPresets.soft)],

    synthwave: (): Effect[] => [
        new Bloom(BloomPresets.neon),
        new Tonemap({ ...TonemapPresets.filmic, exposure: 1.2 })],

    performance: (): Effect[] => [
        new SSAO(SSAOPresets.performance),
        new Bloom({ ...BloomPresets.subtle, mips: 4 }),
        new Tonemap(TonemapPresets.neutral)],

    /** Toon shading with ink outlines. Best with gbuffer.depth for silhouettes. */
    anime: (): Effect[] => [
        new Bloom(BloomPresets.natural),
        new Tonemap(TonemapPresets.filmic),
        new CelShade({ bands: 5, softness: 0.06, saturation: 1.2 }),
        new Outline({ thickness: 1.5, strength: 0.8 }),
        new FXAA()],

    /** Harder bands, thicker ink, punchier color than `anime`. */
    comic: (): Effect[] => [
        new Tonemap(TonemapPresets.filmic),
        new CelShade({ bands: 3, softness: 0.02, saturation: 1.35 }),
        new Outline({ thickness: 2.5, strength: 1.0 }),
        new FXAA(),
        new Vignette({ intensity: 0.35 })],

    /** Kuwahara brushwork over a restrained base. */
    painterly: (): Effect[] => [
        new Bloom(BloomPresets.subtle),
        new Tonemap(TonemapPresets.natural),
        new Kuwahara({ radius: 5 }),
        new Vignette({ intensity: 0.3, softness: 0.6 })],

    /** 35mm print: halation glow, gate weave-free grain, gentle fringing. */
    analog: (): Effect[] => [
        new Halation({ intensity: 0.5 }),
        new Bloom(BloomPresets.subtle),
        new Tonemap(TonemapPresets.neutral),
        new FXAA(),
        new ChromaticAberration({ strength: 0.003 }),
        new Vignette({ intensity: 0.45, softness: 0.5 }),
        new FilmGrain({ intensity: 0.05, size: 1.5 })],

    /** Near-monochrome, hard contrast, heavy corners, coarse mono grain. */
    noir: (): Effect[] => [
        new ColorGrade({ saturation: 0.08, contrast: 1.35 }),
        new Tonemap(TonemapPresets.filmic),
        new Vignette({ intensity: 0.6, softness: 0.55, radius: 0.6 }),
        new FilmGrain({ intensity: 0.08, size: 1.5, colorAmount: 0.0 })],
} as const;

export type ChainPresetName = keyof typeof ChainPresets;
