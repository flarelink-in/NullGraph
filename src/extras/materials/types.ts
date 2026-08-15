
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
    // Toon-specific options
    useSmoothRamp?: boolean;
    // Lambert-specific options
    useHalfLambert?: boolean;
}
