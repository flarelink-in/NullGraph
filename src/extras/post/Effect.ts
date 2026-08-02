// src/extras/post/Effect.ts
import type { ParamSchema, ResourceRef, OutputRef, TargetHandle, TargetSpec } from './types';

/** Handed to Effect.declare(). The only way an effect can request resources. */
export interface DeclareContext {
    /** Internal target, e.g. one bloom mip. Aliased aggressively by the pool. */
    target(spec: TargetSpec, debugName: string): TargetHandle;
    /** Emit one GPU pass. Call N times for a multi-pass effect; order is call order. */
    pass(req: PassRequest): void;
    /** True if the chain was built with gbuffer.depth / .velocity. Lets an effect
     *  degrade gracefully instead of throwing. */
    has(resource: 'depth' | 'velocity'): boolean;
    readonly width: number;
    readonly height: number;
}

export interface PassRequest {
    name: string;
    /** WGSL fragment. The chain PREPENDS: the generated Params struct, its
     *  @group(0) bindings, a fullscreen-triangle vertex entry, and the input
     *  texture/sampler bindings in `inputs` order. Author writes only the fragment. */
    shaderCode: string;
    inputs: ResourceRef[];
    output: OutputRef;
}

export interface EffectDeclaration {
    params: ParamSchema;
}

/**
 * An effect owns NO GPU objects. It is a param cursor plus a pass declaration.
 * This makes ownership single (the chain), makes effects movable between chains,
 * and makes `new Bloom(...)` before `new PostChain(...)` legal.
 */
export interface Effect {
    readonly name: string;
    /** Read at compile time only. Changing it requires setChain()/compile(). */
    enabled: boolean;
    /** Called during compile. Must be pure and deterministic given the same ctx. */
    declare(ctx: DeclareContext): EffectDeclaration;
    /** Called after arena layout is fixed. The effect stores this view and all
     *  its accessors write straight into it. No copies, no dirty flags. */
    bind(slot: Float32Array): void;
}