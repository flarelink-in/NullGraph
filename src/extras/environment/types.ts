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
    /**
     * Prefiltered-cube mip the sky samples — 0 is sharp, higher is a blurred backdrop.
     * **Read only in `cheapSky: 1` mode**; under the default per-pixel evaluation there is no
     * mip chain to select and this value is ignored.
     */
    blurMip?: number;
    /**
     * Where the skybox gets its colour (environment-sky-design.md §4.5, §6.2):
     * `0` (default) evaluates the source's `sampleSky` per pixel — the sky you see is the same
     * function the bake is lit by, at pixel resolution rather than the prefiltered cube's
     * 0.70°/texel; `1` samples the prefiltered cube at `blurMip` instead, which is the
     * pre-Phase-1 behaviour and the right answer for a deliberately blurred backdrop.
     */
    cheapSky?: number;
}

// ---------------------------------------------------------------------------
// Phase 1 sky sources (docs/environment-sky-design.md §4.1–4.2)
// ---------------------------------------------------------------------------

/**
 * The `{ wgslStructs, wgslLogic }` pair a sky source contributes to the TWO consumers of the sky —
 * the source-cube bake kernel and the skybox fragment. Sibling of `ShaderLightingLayout` and
 * `ShaderIndirectLayout` (design-principles §4): same shape, same injection story.
 *
 * `wgslLogic` MUST define both entry points of the seam (§4.2):
 * `fn sampleSky(dir: vec3<f32>) -> vec3<f32>` and `fn skySunRadiance(elevDeg: f32) -> vec3<f32>`.
 */
export interface ShaderSkyLayout {
    /** Binding declarations + structs the source needs, if any (empty string otherwise). */
    wgslStructs: string;
    /** `fn sampleSky(...)` and `fn skySunRadiance(...)` plus any `sky_`-prefixed helpers. */
    wgslLogic: string;
}

/**
 * A sky the EnvironmentSystem can bake and display — data, not a code path
 * (environment-sky-design.md §1, principle 2). Adding a new model is a new implementation of
 * this interface; the engine never branches on the source type.
 *
 * Two consumers share ONE evaluator: `getShaderLayout(base)` is injected into both the bake
 * kernel and the skybox fragment, so the sky you are lit by and the sky you see are the same
 * function by construction (principle 1, pinned by U2).
 *
 * **Params discipline (§4.5).** The sky batch's storage buffer carries a 4-float header
 * (`[exposure, yaw, blurMip, cheapSky]`, `SKY_PARAM_FLOATS`) followed by the source's own
 * `paramFloats` floats. Both `getShaderLayout(base)` and `writeParams(out, base)` receive the
 * base offset (= 4) and address `skyParams[base + i]`. `writeParams` must not allocate
 * (pinned by U10). If `paramFloats` is 0 the source must not read `skyParams` at all.
 */
export interface SkySource {
    /** Debug name, used verbatim in the build-time errors (§9). */
    readonly name: string;
    /** Number of floats this source writes into the sky params buffer after the 4-float header. */
    readonly paramFloats: number;
    /**
     * True when the source's skybox evaluation reads the **source cube** (EquirectSky, §5.2).
     * Such a source requires `keepSourceCube: true`; the mismatch is a named build-time error (§9).
     */
    readonly needsSourceCube?: boolean;
    /**
     * The evaluator layout for the SKYBOX site (and for the bake too, unless
     * `getBakeShaderLayout` is provided). `base` is the source's float offset in `skyParams`.
     */
    getShaderLayout(base: number): ShaderSkyLayout;
    /**
     * Optional bake-site override: a source backed by the source cube cannot sample the cube it
     * is writing, so its bake facet samples the input asset instead (EquirectSky reads the 2D
     * equirect). Analytic sources omit this and share one body across both consumers.
     */
    getBakeShaderLayout?(base: number): ShaderSkyLayout;
    /**
     * Optional extra bind-group entries for the Stage-A bake kernel (group 0, bindings ≥ 3) —
     * e.g. EquirectSky's equirect texture + sampler.
     */
    getBakeBindEntries?(env: EnvironmentSystem): GPUBindGroupEntry[];
    /**
     * Optional extra bind-group entries for the skybox's group 2, in the RESERVED range 14–17
     * (§4.3) — e.g. EquirectSky's source cube + sampler.
     */
    getSkyboxBindEntries?(env: EnvironmentSystem): GPUBindGroupEntry[];
    /** Serialize the source's params into `out` starting at `base`. Must not allocate (U10). */
    writeParams(out: Float32Array, base: number): void;
    /**
     * Move the sun (TRAVEL direction). Writes source state only — no rebuild; the params are
     * re-uploaded on the next `setSkyParams`. Present on sun-bearing sources; Phase 2's clock
     * calls it every frame the sun moves (environment-timeofday-design.md §11.2).
     */
    setSunDirection?(dir: ArrayLike<number>): void;
    /** Phase 3 seam; ignored by every Phase 1 source (environment-timeofday-design.md §11.2). */
    setMoonDirection?(dir: ArrayLike<number>): void;
    /** Phase 3's allocation-free handoff for phase, sidereal rotation, and twilight state. */
    setCelestialState?(state: CelestialState): void;
}


// ---------------------------------------------------------------------------
// Phase 2 — irradiance volumes & probes (docs/lighting-probes-design.md)
// ---------------------------------------------------------------------------

import type { EnvironmentSystem } from './EnvironmentSystem';
import type { CelestialState } from './TimeOfDay';
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
