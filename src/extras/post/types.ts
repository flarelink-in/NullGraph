// src/extras/post/types.d.ts

export type PostFormat =
    | 'rgba16float' | 'rgba8unorm' | 'rg11b10ufloat'
    | 'r8unorm' | 'r16float' | 'rg16float'
    | 'depth24plus' | 'depth32float';
export type ParamType = 'f32' | 'i32' | 'u32' | 'vec2f' | 'vec3f' | 'vec4f';

/** Ordered field list. Order IS the WGSL struct order — the chain generates the
 *  struct declaration and injects it, so a mismatch is impossible by construction.
 *  Generator must apply WGSL uniform-address-space alignment (vec3f aligns to 16). */
export interface ParamField {
    name: string;
    type: ParamType;
    default: number | number[];
}
export type ParamSchema = ParamField[];

/** A pooled render target. Opaque handle; the real GPUTexture is not resolved
 *  until TransientPool.resolve(). Never hold a GPUTextureView across a resize. */
export interface TargetHandle {
    readonly id: number;
    readonly debugName: string;
}

export interface TargetSpec {
    /** Resolution multiplier vs. the chain's internal size. 0.5 = half res. */
    scale?: number;
    format?: PostFormat;
    /** Extra usage bits ORed onto the default
     *  (RENDER_ATTACHMENT | TEXTURE_BINDING). Default 0. */
    extraUsage?: GPUTextureUsageFlags;
    /** Frames this target must survive. 1 = transient (default, aliasable).
     *  2 = double-buffered history (TAA, temporal denoise). Never aliased. */
    history?: 1 | 2;
}

/** Well-known chain-level resources an effect may read. */
export type ChainResource =
    | 'chain'    // output of the previous effect (or the scene color for effect 0)
    | 'scene'    // the original scene color, always available
    | 'depth'    // requires gbuffer.depth; throws at compile if absent
    | 'velocity'; // requires gbuffer.velocity; throws at compile if absent

export type ResourceRef = ChainResource | TargetHandle;

/** Where a pass writes. 'chain' advances the chain cursor for the next effect. */
export type OutputRef = 'chain' | TargetHandle;