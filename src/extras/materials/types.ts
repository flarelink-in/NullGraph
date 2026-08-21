
export interface ShaderLightingLayout {
    wgslStructs: string;
    wgslLogic: string;
}

/**
 * The indirect-lighting contract (lighting-ibl-design.md §5.1) — a deliberate sibling of
 * {@link ShaderLightingLayout}, identical in shape. An implementation (Phase 1: the global
 * environment; Phase 2: a probe volume) provides two functions in `wgslLogic`:
 *
 *   fn getIndirectDiffuse(worldPos: vec3<f32>, N: vec3<f32>) -> vec3<f32>;
 *   fn getIndirectSpecular(worldPos: vec3<f32>, N: vec3<f32>, V: vec3<f32>,
 *                          roughness: f32, F0: vec3<f32>) -> vec3<f32>;
 *
 * Both take `worldPos` even though Phase 1 ignores it — that parameter is what lets Phase 2
 * swap a probe volume in without touching a single material (lighting-ibl-design.md §5.2).
 */
export interface ShaderIndirectLayout {
    wgslStructs: string;   // bindings + helper declarations
    wgslLogic: string;     // getIndirectDiffuse / getIndirectSpecular
}

/**
 * The aerial-perspective contract (environment-aerial-design.md §5.3) — the FOURTH injected
 * layout, the same shape as {@link ShaderLightingLayout} and {@link ShaderIndirectLayout}.
 * `wgslLogic` must define one entry point:
 *
 *   fn applyAerialAt(color: vec3<f32>, worldPos: vec3<f32>) -> vec3<f32>;
 *
 * Two implementations satisfy it and a material never learns which it got: the froxel volume
 * (`AerialPerspective.getShaderLayout()`) and the closed-form exponential height fog
 * (`HeightFog.getShaderLayout()`). That interchangeability is the point — the cheap analytic path
 * ships first and the volume replaces it without touching a single material.
 *
 * **This layout is for TRANSPARENT passes.** Opaque geometry gets the fullscreen apply pass
 * instead: once per pixel rather than once per fragment, runtime-toggleable, and able to do
 * depth-aware upsampling because it owns the depth buffer (§5.1). A batch carrying this layout
 * inside a pass covered by that apply is fogged twice, and `AerialPerspective.registerBatch`
 * refuses it at build time (§5.4).
 */
export interface ShaderAerialLayout {
    wgslStructs: string;   // the volume/uniform binding declarations
    wgslLogic: string;     // applyAerialAt, plus the shared composition helpers
}

export interface ShaderConfig {
    useSkinning: boolean;
    // Make lighting entirely optional
    lighting?: ShaderLightingLayout;
    /**
     * Indirect (image-based / probe) lighting layout from
     * `EnvironmentSystem.getShaderLayout()` (lighting-ibl-design.md §5.3). Optional — without
     * it the PBR/Lambert/Toon fragments keep their hardcoded ambient terms, byte-for-byte.
     */
    indirect?: ShaderIndirectLayout;
    /**
     * Aerial-perspective layout (environment-aerial-design.md §5.3). Optional — without it every
     * fragment body is byte-for-byte what it was before Phase 5.
     */
    aerial?: ShaderAerialLayout;
    // Toon-specific options
    useSmoothRamp?: boolean;
    // Lambert-specific options
    useHalfLambert?: boolean;
}
