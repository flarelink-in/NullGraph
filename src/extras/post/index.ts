// null-graph/post — post-processing chain
//
// Explicit sequence, pre-allocated memory, no per-frame garbage, and a param
// layout a Web Worker can drive. See docs/post-processing-design.md and
// docs/transient-pool.md.

export * from './types';
export * from './Effect';
export { TransientPool } from './TransientPool';
export type { PoolReport, TargetInfo } from './TransientPool';
export { ParamArena, layoutFields, SLICE_BYTES, SLICE_FLOATS } from './ParamArena';
export type { FieldLayoutResult, ArenaEntry } from './ParamArena';
export { composePassShader, PASS_HEADER_FLOATS } from './ShaderComposer';
export type { ComposeOptions, ComposedShader, ComposedBinding, ComposerInput } from './ShaderComposer';
export { PostEffectBase } from './PostEffectBase';
export { PostChain } from './PostChain';
export type { PostChainConfig } from './PostChain';
export { Tonemap } from './Tonemap';
export type { TonemapOptions, TonemapCurve } from './Tonemap';
export { Bloom } from './Bloom';
export type { BloomOptions } from './Bloom';
export { SSAO } from './SSAO';
export type { SSAOOptions } from './SSAO';
export { Vignette } from './Vignette';
export type { VignetteOptions } from './Vignette';
export { ChromaticAberration } from './ChromaticAberration';
export type { ChromaticAberrationOptions } from './ChromaticAberration';
export { FilmGrain } from './FilmGrain';
export type { FilmGrainOptions } from './FilmGrain';
export { Halation } from './Halation';
export type { HalationOptions } from './Halation';
export { Kuwahara } from './Kuwahara';
export type { KuwaharaOptions } from './Kuwahara';
export { Bokeh } from './Bokeh';
export type { BokehOptions } from './Bokeh';
export { FXAA } from './FXAA';
export type { FXAAOptions } from './FXAA';
export { ColorGrade } from './ColorGrade';
export type { ColorGradeOptions } from './ColorGrade';
export { AutoExposure } from './AutoExposure';
export type { AutoExposureOptions } from './AutoExposure';
export { LensFlare } from './LensFlare';
export type { LensFlareOptions } from './LensFlare';
export { CelShade } from './CelShade';
export type { CelShadeOptions } from './CelShade';
export { Outline } from './Outline';
export type { OutlineOptions } from './Outline';
export {
    TonemapPresets, BloomPresets, SSAOPresets, ChainPresets, forSceneScale,
} from './presets';
export type { ChainPresetName } from './presets';
export { EFFECT_ORDER, orderChain } from './chainOrder';
export { CustomEffect } from './CustomEffect';
export type { CustomEffectConfig } from './CustomEffect';
