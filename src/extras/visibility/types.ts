// src/extras/visibility/types.ts
//
// Configuration types for the VisibilitySystem (docs/visibility-design.md).

/** Hi-Z occlusion configuration (design doc §5). Requires a samplable depth source. */
export interface HiZConfig {
    /**
     * A depth texture view with `TEXTURE_BINDING` usage that the pyramid is built from each
     * frame. This is LAST frame's depth (single-phase, §5.2). `ctx.depthTexture` is
     * RENDER_ATTACHMENT-only and cannot be used — supply a depth pre-pass target or the
     * PostChain g-buffer depth instead.
     */
    depthSource: GPUTextureView;
    /** Screen size the depth source was rendered at. Defaults to the engine canvas size. */
    width?: number;
    height?: number;
}

export interface VisibilityConfig {
    /** Maximum number of instances across the whole scene (sizes the instance + work buffers). */
    maxInstances: number;
    /**
     * Maximum meshlets per instance LOD, across all registered meshes (sizes the meshlet-cull
     * dispatch's Y dimension). Required only for the meshlet tier.
     */
    maxMeshletsPerInstance?: number;
    /**
     * Maximum meshlets visible in a single frame per meshlet-tier draw batch (sizes the
     * compacted `{instanceId, meshletId}` output). Required only for the meshlet tier.
     */
    maxVisibleMeshletInstances?: number;
    /**
     * Hi-Z occlusion. Omit to run frustum + cone culling only (no depth source needed).
     * When provided, occlusion is applied on top.
     */
    hiZ?: HiZConfig;
    /**
     * Screen-space LOD error threshold in pixels (design doc §8). A mesh coarsens to the
     * cheapest LOD whose projected error stays under this. Default 1.0.
     */
    lodErrorPx?: number;
    /**
     * Accuracy mode. `'single-phase'` (default) builds Hi-Z from last frame's depth — one frame
     * of latency, possible single-frame pop-in on fast disocclusion. `'two-phase'` re-culls the
     * phase-1 rejects against this frame's fresh depth (§9). Two-phase is reserved and currently
     * throws — ship single-phase first (build order §13 step 7).
     */
    mode?: 'single-phase' | 'two-phase';
}

/**
 * A stock material instance (`StandardPBRMaterial`, `LambertMaterial`, …) paired with its family
 * name. Passing this to a draw batch wires the material's WGSL **and** its bind group
 * automatically — no hand-written shader chunks.
 *
 * The `kind` is explicit rather than sniffed from the class so the visibility module never has to
 * import the material classes (keeping `null-graph/visibility` independent of
 * `null-graph/materials` at runtime).
 */
export interface VisibilityMaterial {
    /** Which stock family this instance belongs to. */
    kind: 'basic' | 'lambert' | 'emissive' | 'matcap' | 'toon' | 'pbr';
    /** The material instance. Its `applyToBatch(batch, group)` binds the textures/uniform. */
    instance: { applyToBatch(batch: any, groupIndex?: number): unknown };
    /** `LightSystem.getShaderLayout()` — enables the material's clustered/tiled lighting loop. */
    lighting?: { wgslStructs: string; wgslLogic: string };
    /** Lambert only: wrap light around the terminator. */
    useHalfLambert?: boolean;
    /** Toon only: soften the cel ramp edges. */
    useSmoothRamp?: boolean;
}

/** Options for a single instance-tier draw batch (one per unique mesh). */
export interface InstanceDrawOptions {
    /**
     * Render with a stock material — the one-liner path. Generates the material's bindings and
     * fragment stage and binds its resources, so no `fragmentBody`/`extraBindings`/`materialEntries`
     * are needed. Ignored if `fragmentBody` is also supplied (explicit WGSL wins).
     */
    material?: VisibilityMaterial;
    /** Fragment shader body override (receives `VertexOut`, returns `vec4<f32>`). */
    fragmentBody?: string;
    /** Vertex attribute declarations matching the mega vertex layout. Defaults to StandardLayout. */
    vertexAttributesWGSL?: string;
    /** GPUVertexBufferLayout[] for the mega vertex buffer. Defaults to StandardLayout's descriptor. */
    vertexLayouts?: GPUVertexBufferLayout[];
    /** Extra group-1+ bindings the fragment needs (a material), emitted verbatim. */
    extraBindings?: string;
    /** Color target format. Defaults to the engine's preferred canvas format. */
    targetFormat?: GPUTextureFormat;
    /** Extra group-1+ bind-group entries (a material's), bound via setRenderPassBindGroup. */
    materialEntries?: GPUBindGroupEntry[];
    /** Group index the materialEntries bind to. Default 1. */
    materialGroupIndex?: number;
}

/** Options for a meshlet-tier draw batch (one per material). */
export interface MeshletDrawOptions {
    /**
     * Render with a stock material — the one-liner path. On this tier the material is bound at
     * `materialGroupIndex` (default 2, since group 1 holds the meshlet geometry), and the generated
     * WGSL uses that same index automatically.
     */
    material?: VisibilityMaterial;
    fragmentBody?: string;
    extraBindings?: string;
    materialEntries?: GPUBindGroupEntry[];
    materialGroupIndex?: number;
    targetFormat?: GPUTextureFormat;
}
