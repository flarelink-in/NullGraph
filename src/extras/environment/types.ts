// src/extras/environment/types.ts
//
// Configuration types for the EnvironmentSystem (docs/lighting-ibl-design.md) — Phase 1:
// image-based lighting. Phase 2 (probes) is additive and lives in the same subpath.

/** A procedural sky — the zero-asset environment source (design doc §6.1). */
export interface ProceduralSkyConfig {
    /** Linear HDR color at the zenith (straight up). */
    zenith: [number, number, number];
    /** Linear HDR color at the horizon. */
    horizon: [number, number, number];
    /** Linear HDR color at/below the nadir. */
    ground: [number, number, number];
    /** The sun's TRAVEL direction (where the light goes); normalized internally. */
    sunDir: [number, number, number];
    /** Sun disc intensity (linear HDR). 0 disables the disc. */
    sunIntensity: number;
    /** cos(angular radius) of the sun disc. Default ~0.9995 (≈1.8°). */
    sunCos?: number;
}

export interface EnvironmentConfig {
    /** Source cube face size in pixels. Power of two ≥ 8. Default 512. */
    sourceSize?: number;
    /**
     * Mip levels in the prefiltered specular cube; mip L holds roughness L/(prefilterMips-1).
     * Must not exceed log2(sourceSize)+1. Default 6.
     */
    prefilterMips?: number;
    /** GGX importance-sample count per prefiltered texel. Default 128. */
    prefilterSamples?: number;
    /** BRDF integration LUT resolution (square). Default 256. */
    lutSize?: number;
    /** Diffuse + specular multiplier. Default 1.0. */
    intensity?: number;
    /**
     * Occlusion of below-horizon specular reflections, 0..1 (design doc §4.3). 1.0 (default)
     * fades reflections pointing under the horizon to the horizon color; 0 disables the fade.
     */
    horizonFade?: number;
    /**
     * `'once'` (default) retires the bake stages after the first frame they run.
     * `'amortized'` keeps them and rebuilds ONE prefilter mip per frame (plus the SH
     * projection), so an animated sky fully refreshes in `prefilterMips` frames (§6.5).
     * Requires `keepSourceCube: true`.
     */
    updateMode?: 'once' | 'amortized';
    /**
     * Retain the source cube after the bake. Required by `updateMode: 'amortized'` and by a
     * skybox that samples the unfiltered source; otherwise it is released (§4.2).
     */
    keepSourceCube?: boolean;
    /**
     * The bind group indirect lighting shares with the lights (§3: core WebGPU guarantees 4
     * bind groups and all four are spent). Must equal the LightSystem's group. Default 2.
     */
    environmentGroup?: number;
}

/** Runtime sky parameters, written through the sky batch's group-0 storage buffer (§8.2). */
export interface SkyParams {
    /** Exposure multiplier on the sampled environment. Default 1.0. */
    exposure?: number;
    /** Environment yaw in radians (rotates the sampled direction). Default 0. */
    yaw?: number;
    /** Prefiltered-cube mip the sky samples — 0 is sharp, higher is a blurred backdrop. */
    blurMip?: number;
}
