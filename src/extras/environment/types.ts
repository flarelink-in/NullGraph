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

// ---------------------------------------------------------------------------
// Phase 2 — irradiance volumes & probes (docs/lighting-probes-design.md)
// ---------------------------------------------------------------------------

import type { EnvironmentSystem } from './EnvironmentSystem';
import type { VisibilitySystem } from '../visibility/VisibilitySystem';
import type { ShadowSystem } from '../shadows/ShadowSystem';
import type { LightSystem } from '../lights/LightSystem';
import type { VisibilityMaterial } from '../visibility/types';

export interface ProbeVolumeConfig {
    /** World-space position of probe (0,0,0) — the volume's near corner. */
    origin: [number, number, number];
    /** World-space distance between neighbouring probes. */
    spacing: number;
    /** Probe counts per axis. dims[0]×dims[1]×dims[2] ≤ maxProbes. */
    dims: [number, number, number];
    /** Capture cube-face resolution in px. Four SH coefficients need no more (§7.1). Default 32. */
    captureFaceSize?: number;
    /** Probes recaptured per frame (amortization, §7.4). Default 16. */
    probesPerFrame?: number;
    /** Capture atlas edge in px (holds probesPerFrame×6 tiles + the reflection band). Default 1024. */
    captureAtlasSize?: number;
    /** Normal bias in probe-grid texels — the cheapest leak mitigation (§6.1). Default 0.5. */
    normalBiasTexels?: number;
    /** Border fade width in world units — a smooth exit at the volume's edge (§5). Default 4. */
    borderFade?: number;
    /** Indirect intensity multiplier on the volume lookup. Default 1.0. */
    intensity?: number;
    /** Hard probe cap — a 3D texture allocation beyond this is refused (§11). Default 65536. */
    maxProbes?: number;
    /** Capture near plane. Default spacing × 0.05. */
    captureNear?: number;
    /** Capture far plane. Default 2 × the volume diagonal. */
    captureFar?: number;
    /** Scheduler near-priority weight (§7.4): a probe at the camera refreshes this many times
     *  more often than one past the reference distance. Default 16. */
    distanceBias?: number;
    /** SH band. Only 1 is supported — band 2 needs 7 texture bindings where 4 exist (§4.2). */
    shBands?: 1 | 2;
    /** Phase 1 environment — the fallback outside the volume and the capture's ambient (§5, §7.3). */
    environment: EnvironmentSystem;
    /** GPU caster culling for captures; probe capture has no CPU draw path. */
    visibility: VisibilitySystem;
    /** Captures see shadows; without them GI is flat and wrong (§9). */
    shadows?: ShadowSystem;
    /** Accept shadowless capture (§11). Default false. */
    allowShadowlessCapture?: boolean;
    /** Frames between reflection-probe refreshes (one probe per interval). Default 2. */
    reflectionRefreshInterval?: number;
    /** Atlas band reserved for reflection captures, in px. A ReflectionProbes attached through
     *  {@link ProbeVolume.setReflectionProbes} must match this face size. Default 128. */
    reflectionFaceSize?: number;
}

export interface ReflectionProbesConfig {
    /** Cube count in the array, allocated once at maxProbes × 6 layers. Default 16. */
    maxProbes?: number;
    /** Capture/prefilter face size in px — specular carries high frequencies (§8.1). Default 128. */
    faceSize?: number;
    /** GGX prefiltered mip levels per cube. Default 6. */
    prefilterMips?: number;
    /** GGX importance samples per texel. Default 64. */
    prefilterSamples?: number;
}

/** One reflection probe registration ({@link ReflectionProbes.add}). */
export interface ReflProbeConfig {
    /** Proxy box centre (world). */
    center: [number, number, number];
    /** Proxy box half-extents (world). Zero on any axis is rejected (§11). */
    halfExtent: [number, number, number];
    /** Where the cube is captured from. Default: centre — but see §8.2: parallax reprojects
     *  FROM the capture point, so off-centre capture is handled exactly. */
    capture?: [number, number, number];
    /** Blend distance inside the proxy boundary: weight fades 1 → 0 over this band. Default 1. */
    blend?: number;
    /** Influence radius from the centre; outside it the weight is 0. Default: |halfExtent|. */
    radius?: number;
    /** Debug name, used in error messages. */
    name?: string;
}

/** Options for {@link ProbeVolume.registerCaptureMesh}. */
export interface CaptureMeshOptions {
    material: VisibilityMaterial;
    /** The LightSystem whose entries fill group-2 bindings 0–1(–4) of the capture materials'
     *  combined bind group. Pass the same system (through shadows.wrapLighting for the layout)
     *  that lights the main scene pass. */
    lights: LightSystem;
}
