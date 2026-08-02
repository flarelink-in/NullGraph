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
export { CustomEffect } from './CustomEffect';
export type { CustomEffectConfig } from './CustomEffect';
