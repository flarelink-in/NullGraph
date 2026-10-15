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
    /**
     * `EnvironmentSystem.getShaderLayout()` (lighting-ibl-design.md §12.2 item 3) — enables the
     * material's image-based indirect term, the same one-liner way `lighting` enables clustered
     * light. PBR gets the full split-sum; Lambert/Toon get the diffuse half.
     */
    indirect?: { wgslStructs: string; wgslLogic: string };
    /**
     * `AerialPerspective.getShaderLayout()` or `HeightFog.getShaderLayout()`
     * (environment-aerial-design.md §5.3) — fogs the fragment at its own distance, the same
     * one-liner way `lighting` enables clustered light.
     *
     * For TRANSPARENT batches. Opaque geometry is covered once per pixel by the fullscreen apply
     * pass, and a batch carrying this layout inside a pass that apply covers is refused at build
     * time by `AerialPerspective.registerBatch` (§5.4).
     */
    aerial?: { wgslStructs: string; wgslLogic: string };
    /** Lambert only: wrap light around the terminator. */
    useHalfLambert?: boolean;
    /** Toon only: soften the cel ramp edges. */
    useSmoothRamp?: boolean;
}

/** Model-space deformation for instance-tier draws; resources remain caller-owned. */
export interface InstanceVertexHook {
    /** Defines fn deform(pos: vec3<f32>, normal: vec3<f32>, vi: u32, inst: Instance) -> Deformed.
     * Deformed (provided by visibility) has position and normal, both vec3<f32>. */
    wgsl: string;
    /** Binding declarations at `group`, emitted verbatim (no automatic group rewriting). */
    bindings: string;
    entries: GPUBindGroupEntry[];
    /** Dedicated bind group; default 2. Choose another group when lighting occupies 2. */
    group?: number;
}

/**
 * The on-screen sizes, in pixels, at which an instance batch draws a row: the diameter of its
 * bounding sphere, measured as the meshlet tier's LOD measures it (from the sphere's nearest
 * distance, the vertical field of view and the cull's screen height). A row is drawn while
 * `minPixels <= size < maxPixels`; a bound left out is open. Two batches of one mesh index with
 * adjoining bands draw each row exactly once, as a near mesh or a far one.
 */
export interface InstanceLodBand {
    minPixels?: number;
    maxPixels?: number;
}

/** Options for an instance-tier draw batch (one per unique mesh). */
export interface InstanceDrawOptions {
    /**
     * Camera uniform for BOTH the cull stage and the draw (shadows-design.md §9.3). Default:
     * `engine.ctx.cameraUniformBuffer`. A shadow cascade passes its own 432-byte Camera buffer
     * here — the cascade IS a camera, so nothing downstream learns what a shadow is.
     */
    camera?: GPUBuffer;
    /**
     * Override the system-wide Hi-Z setting for this batch (shadows-design.md §9.3). A shadow
     * view has no pyramid, so caster culls pass `occlusion: false`. Default: the system-wide
     * setting (`config.hiZ !== undefined`).
     */
    occlusion?: boolean;
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
    /** Runs before the instance model transform. Not compatible with a full shaderCode override. */
    vertexHook?: InstanceVertexHook;
    /**
     * Level of detail by size on screen: this batch draws its mesh index's rows only while they
     * are this large. Give a near mesh `{ minPixels: 12 }` and a far one (a cube) `{ maxPixels: 12 }`
     * under the same mesh index, and each row is drawn by exactly one of them. Checked in the
     * batch's own cull, one comparison per row; without it the cull is unchanged.
     */
    lod?: InstanceLodBand;
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
    /**
     * Full shader-module override (shadows-design.md §6.2). Replaces the generated draw WGSL
     * entirely — the shadow caster pass uses it to swap in the depth-only vertex shader while
     * reusing the whole cull + indirect-draw chain. Must still define `vs_main` and consume
     * group 0 (camera@0, instances@1).
     */
    shaderCode?: string;
    /** Forwarded to `PipelineConfig.depthOnly` (§12.1) — no fragment stage, no color targets. */
    depthOnly?: boolean;
    /**
     * Forwarded to `PipelineConfig.viewport` (shadows-atlas-design.md §12.1) — scopes the draw
     * to a tile of the pass's attachment. The shadow atlas sets it to the view's tile rect.
     */
    viewport?: [x: number, y: number, width: number, height: number];
    /**
     * The cull stage's Hi-Z UV remap (shadows-atlas-design.md §7.2): `[scaleX, scaleY,
     * offsetX, offsetY]` in the pyramid's UV space. A shadow view passes its tile's atlas rect
     * so the occlusion test samples only that tile; the main view leaves the default
     * `[1, 1, 0, 0]` (the identity — bit-for-bit the v1.0.5 behavior, §10 U5).
     */
    hizRect?: [number, number, number, number];
    /**
     * The screen size the cull stage's Hi-Z footprint/LOD math uses (shadows-atlas-design.md
     * §7.2: "screenSize for a shadow view is the tile size"). Default: the Hi-Z source size,
     * then the engine canvas.
     */
    screenSize?: [number, number];
}

/** Options for a meshlet-tier draw batch (one per material). */
export interface MeshletDrawOptions {
    /**
     * Camera uniform for BOTH the cull stage and the draw (shadows-design.md §9.3). Default:
     * `engine.ctx.cameraUniformBuffer`.
     */
    camera?: GPUBuffer;
    /**
     * Override the system-wide Hi-Z setting for this batch (shadows-design.md §9.3). Default:
     * the system-wide setting (`config.hiZ !== undefined`).
     */
    occlusion?: boolean;
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
