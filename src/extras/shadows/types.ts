// src/extras/shadows/types.ts
//
// Configuration types for the ShadowSystem (docs/shadows-design.md §9).

import type { VisibilitySystem } from '../visibility/VisibilitySystem';

export interface ShadowConfig {
    /** Number of cascades, 1–4 (MAX_CASCADES sizes the ShadowData uniform). Default 4. */
    cascades?: number;
    /** Shadow map resolution (square, power of two). Default 2048. */
    resolution?: number;
    /**
     * Phase 2 (shadows-atlas-design.md): the atlas edge in px (square, power of two). When set,
     * the system renders every light into ONE atlas texture partitioned by importance, instead
     * of Phase 1's per-cascade array. Default undefined → Phase 1 behavior.
     */
    atlasSize?: number;
    /** Phase 2: largest tile a single view may request. Default 2048. */
    maxTileSize?: number;
    /** Phase 2: hard cap on shadow views (cascades + spots + 6×points). Default 64. */
    maxViews?: number;
    /** Phase 2: hard cap on shadow-casting lights. Default 64. */
    maxSlots?: number;
    /**
     * Phase 2 §8.1: when a view re-renders. `'every-frame'` (default) is always correct;
     * `'on-demand'` renders only dirty views; `'round-robin'` additionally refreshes
     * `refreshPerFrame` clean views per frame.
     */
    updateMode?: 'every-frame' | 'on-demand' | 'round-robin';
    /** Phase 2 §8.1: clean views refreshed per frame in `'round-robin'` mode. Default 2. */
    refreshPerFrame?: number;
    /**
     * Phase 2 §9.1: the cube-face fov margin, in filter texels. Must cover `pcfRadius` or face
     * seams appear (§11 errors when they disagree). Default = pcfRadius.
     */
    cubeFovMarginTexels?: number;
    /**
     * Phase 2 §7.2: build a Hi-Z pyramid over the atlas and occlusion-cull casters on the light
     * view. Reuses `HiZPyramid` unchanged against last frame's atlas. Default false.
     */
    lightViewOcclusion?: boolean;
    /**
     * World-space distance from the camera beyond which shadows stop — NOT camera.far. A
     * 1000-unit far plane with 4 cascades gives a useless first cascade; shadows stop here
     * (default 200) and lighting falls back to unshadowed beyond it (§5.1).
     */
    shadowDistance?: number;
    /** Practical-split blend: 0 = uniform, 1 = logarithmic. Default 0.6 (§5.1). */
    lambda?: number;
    /**
     * How far behind each cascade the ortho near plane is pushed along the light direction, so
     * off-screen casters still land in the map (§5.4). Default = shadowDistance. Supply the
     * scene's extent along the light when you know it.
     */
    casterExtrusion?: number;
    /** Constant NDC-z bias applied on the caster (§7.1). Default 0.0015. */
    constantBias?: number;
    /**
     * Caster normal offset in shadow texels (§7.2) — resolution-correct acne control. Trades
     * acne for peter-panning; §11 throws when it exceeds 2% of a cascade's extent. Default 1.5.
     */
    normalOffsetTexels?: number;
    /** PCF taps per side: 0 → hardware 2×2 only, 1 → 3×3 (default), 2 → 5×5 (§8.3). */
    pcfRadius?: number;
    /** Cascade cross-fade band, in split-distance units (§8.3). Default 4.0. */
    blendTexels?: number;
    /** Casters render this many LODs coarser than the shadow view would pick (§5.5). Default 1. */
    lodBias?: number;
    /**
     * The VisibilitySystem whose instances are the casters (§9.3). Required — Phase 1 has no
     * CPU caster path.
     */
    visibility: VisibilitySystem;
    /**
     * Bind group the shadow resources occupy in the receiver material (§8.4). Default 3: group
     * 0 is engine-fixed, 1 is the material, 2 is hardcoded into LightSystem's WGSL. Must be ≥ 3.
     */
    shadowGroupIndex?: number;
    /**
     * Escape hatch for the §6.3 fallback: when the engine lacks `PipelineConfig.depthOnly`,
     * render with a throwaway r8unorm color target instead of throwing. ~1.5–2× pass cost.
     */
    allowColorFallback?: boolean;
}
