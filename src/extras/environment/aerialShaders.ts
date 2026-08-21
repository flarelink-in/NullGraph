// src/extras/environment/aerialShaders.ts
//
// Phase 5 — the froxel volume kernel, the fullscreen apply pass, and the injected layout
// (docs/environment-aerial-design.md §4, §5).
//
// THREE consumers, ONE application contract. `buildAerialApplyFunctionsWGSL` is emitted verbatim
// into both the fullscreen pass and the injected material layout — the composition invariant from
// environment-sky-design.md §8 U2, restated here as U6: the two must contain byte-identical
// bodies, or `color * T + inScatter` can drift between the surface that fogs opaque geometry and
// the one that fogs glass.
//
// Nothing in this file may contain a backtick, including inside a WGSL comment: these are
// JavaScript template literals, and a backtick silently ends the string and hands the rest of the
// shader to the TypeScript parser (design-principles §13, rule 3). U12 asserts it.

import type { AtmosphereConfig } from './atmosphereMath';
import { ATMOSPHERE_MATH_WGSL, buildAtmosphereConstantsWGSL } from './atmosphereShaders';
import { type ResolvedAerialConfig, resolveAerialConfig } from './mediaMath';
import type { ShaderAerialLayout } from '../materials/types';

/** Group-2 binding for the froxel volume. 0-13 are lights/env/probes; 14-17 are sky sources. */
export const AERIAL_VOLUME_BINDING = 18;
/** Group-2 binding for the froxel volume's sampler. */
export const AERIAL_SAMPLER_BINDING = 19;

/** Floats in the apply pass's batch storage buffer: [enabled, intensity, tolerance, _pad]. */
export const AERIAL_APPLY_PARAM_FLOATS = 4;

function n(x: number): string { const s = String(x); return /[.e]/i.test(s) ? s : `${s}.0`; }

// ---------------------------------------------------------------------------
// The W axis (§4.2) — pure math, no bindings, emitted into all three consumers
// ---------------------------------------------------------------------------

/**
 * `aerialW` (distance to normalized slice coordinate) and `aerialSliceDistance` (its inverse at a
 * texel centre). The half-texel offset in `aerialSliceDistance` is what makes the two exact
 * inverses under the sampler: a lookup at `aerialW(d)` lands on the texel that integrated to `d`.
 * Getting it wrong shifts the entire fog one slice toward or away from the camera, which reads as
 * a density error rather than as an indexing bug (U4).
 */
export function buildAerialAxisWGSL(config: ResolvedAerialConfig): string {
    const near = config.nearDistance;
    const max = config.maxDistance;
    // `viewDistance`, not `distance`: WGSL has a builtin of that name and shadowing it in a
    // generated module is a trap for whoever extends this function next.
    const forward = config.distribution === 'log'
        ? `    return clamp(log(max(viewDistance, AERIAL_NEAR) / AERIAL_NEAR) / log(AERIAL_MAX / AERIAL_NEAR), 0.0, 1.0);`
        : `    return clamp((max(viewDistance, AERIAL_NEAR) - AERIAL_NEAR) / (AERIAL_MAX - AERIAL_NEAR), 0.0, 1.0);`;
    const inverse = config.distribution === 'log'
        ? `    return AERIAL_NEAR * pow(AERIAL_MAX / AERIAL_NEAR, w);`
        : `    return AERIAL_NEAR + (AERIAL_MAX - AERIAL_NEAR) * w;`;
    return /* wgsl */ `
const AERIAL_NEAR: f32 = ${n(near)};
const AERIAL_MAX: f32 = ${n(max)};
const AERIAL_SLICES: f32 = ${n(config.dims[2])};
const AERIAL_UNITS_PER_KM: f32 = ${n(config.unitsPerKm)};
const AERIAL_GROUND_LEVEL: f32 = ${n(config.groundLevel)};
fn aerialW(viewDistance: f32) -> f32 {
${forward}
}
fn aerialDistanceAtW(w: f32) -> f32 {
${inverse}
}
fn aerialSliceDistance(z: u32) -> f32 {
    return aerialDistanceAtW((f32(z) + 0.5) / AERIAL_SLICES);
}
`;
}

// ---------------------------------------------------------------------------
// The application contract (§4.1) — THE shared block, byte-identical in both consumers (U6)
// ---------------------------------------------------------------------------

/**
 * `composeAerial` / `applyAerial` / `applyAerialAt`, referencing `aerialVolume` and
 * `aerialSampler` which each consumer declares at its own group and binding.
 *
 * `color * T + inScatter` lives in exactly ONE function, `composeAerial`, because that order is
 * the single most common bug in a fog implementation: reversed, distant objects glow instead of
 * fading and it looks superficially like "too much fog" (§4.1, U5). The bilateral path in the
 * fullscreen pass resolves a different `ap` but composes through this same function, so there is
 * no second place for the order to be wrong.
 *
 * Takes no config beyond the axis block so that the emitted text cannot vary between consumers —
 * U6 compares it verbatim.
 */
export const AERIAL_APPLY_FUNCTIONS_WGSL = /* wgsl */ `
fn aerialUVW(uv: vec2<f32>, viewDistance: f32) -> vec3<f32> {
    return vec3<f32>(uv, aerialW(viewDistance));
}
fn composeAerial(color: vec3<f32>, ap: vec4<f32>) -> vec3<f32> {
    return color * ap.a + ap.rgb;
}
fn applyAerial(color: vec3<f32>, uvw: vec3<f32>) -> vec3<f32> {
    return composeAerial(color, textureSampleLevel(aerialVolume, aerialSampler, uvw, 0.0));
}
fn applyAerialAt(color: vec3<f32>, worldPos: vec3<f32>) -> vec3<f32> {
    let clip = camera.viewProj * vec4<f32>(worldPos, 1.0);
    let ndc = clip.xy / max(clip.w, 1.0e-6);
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return applyAerial(color, aerialUVW(uv, length(worldPos - camera.eyePos)));
}
`;

// ---------------------------------------------------------------------------
// §4 — the froxel volume kernel
// ---------------------------------------------------------------------------

/**
 * The medium, evaluated at a height in km. The atmosphere rows are compile-time constants (the
 * medium does not change at runtime); the fog row is uniform-driven so it can be animated, and it
 * is emitted UNCONDITIONALLY — zero coefficients disable it for the cost of three multiply-adds,
 * which is cheaper than a pipeline rebuild and removes the "toggling fog recompiles" footgun.
 *
 * Both functions sum coefficients BEFORE the exponential. That summation is the whole of §6.2:
 * two systems each integrating their own medium and multiplying the results applies extinction
 * twice, and the result is dimmer and bluer than either alone.
 */
function mediaWGSL(hasAtmosphere: boolean): string {
    const atmosphereExtinction = hasAtmosphere
        ? `    total += ATM_RAYLEIGH * atm_densityR(h) + vec3<f32>(ATM_MIE_EXTINCT * atm_densityM(h)) + ATM_OZONE * atm_densityO(h);`
        : '';
    const atmosphereScatter = hasAtmosphere
        ? `    total += ATM_RAYLEIGH * atm_densityR(h) * atm_phaseR(cosTheta)
             + vec3<f32>(ATM_MIE_SCATTER * atm_densityM(h) * atm_phaseM(cosTheta));`
        : '';
    return /* wgsl */ `
fn aerial_fogDensity(h: f32) -> f32 {
    return exp(-(h - params.fogReferenceHeight) / params.fogScaleHeight);
}
fn aerial_phaseFog(cosTheta: f32) -> f32 {
    let g2 = params.fogG * params.fogG;
    return 3.0 * (1.0 - g2) * (1.0 + cosTheta * cosTheta)
        / (8.0 * 3.14159265359 * (2.0 + g2) * pow(max(1.0 + g2 - 2.0 * params.fogG * cosTheta, 1.0e-4), 1.5));
}
fn aerial_extinction(h: f32) -> vec3<f32> {
    var total = vec3<f32>(0.0);
${atmosphereExtinction}
    total += (params.fogScattering + params.fogAbsorption) * aerial_fogDensity(h);
    return total;
}
fn aerial_phaseScattering(h: f32, cosTheta: f32) -> vec3<f32> {
    var total = vec3<f32>(0.0);
${atmosphereScatter}
    total += params.fogScattering * aerial_fogDensity(h) * aerial_phaseFog(cosTheta);
    return total;
}
`;
}

/**
 * 64 bytes / 16 floats. Only genuinely runtime state lives here: the sun moves and the fog is
 * animated. `unitsPerKm` and `groundLevel` are baked into the axis block as constants instead —
 * two sources for one number is a drift surface, and these two never change after construction.
 */
const AERIAL_PARAMS_WGSL = /* wgsl */ `
struct AerialParams {
    sunDir: vec3<f32>,
    sunIntensity: f32,
    fogScattering: vec3<f32>,
    fogG: f32,
    fogAbsorption: vec3<f32>,
    fogScaleHeight: f32,
    fogReferenceHeight: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};
`;

/** Floats in {@link AERIAL_PARAMS_WGSL}. The CPU staging array is sized from this. */
export const AERIAL_PARAM_FLOATS = 16;

const AERIAL_CAMERA_WGSL = /* wgsl */ `
struct Camera {
    viewProj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    invViewProj: mat4x4<f32>,
    invView: mat4x4<f32>,
    invProj: mat4x4<f32>,
    eyePos: vec3<f32>,
    simTime: f32,
    zNear: f32,
    zFar: f32,
    fovRads: f32,
    aspect: f32,
    _padEnd: vec4<f32>,
};
`;

export interface AerialVolumeShaderOptions {
    config: ResolvedAerialConfig;
    /**
     * Phase 4's medium. Omit for a fog-only volume: the Rayleigh/Mie/ozone rows AND the
     * transmittance-LUT bindings both disappear, because a fog-only volume genuinely has nothing
     * to look up — sunlight reaches it unattenuated by definition.
     */
    atmosphere?: AtmosphereConfig | null;
}

/**
 * The froxel volume build kernel — ONE THREAD PER (x, y), looping over z.
 *
 * This shape is the §4.3 cost argument, not an optimization detail. Each froxel continues the
 * previous slice's accumulation instead of restarting from the camera, so the whole 32x32x32
 * volume costs one ray per lateral froxel with 32 accumulation steps: ~32,768 samples against
 * ~62,000,000 for a per-pixel march at 1080p. A version that restarts per slice renders the same
 * image and costs 32x more, which is why U3 pins slice composition rather than pixel values.
 */
export function buildAerialVolumeWGSL(options: AerialVolumeShaderOptions): string {
    const c = options.config;
    const hasAtmosphere = !!options.atmosphere;
    const [nx, ny, nz] = c.dims;

    const atmosphereBlock = hasAtmosphere
        ? `${buildAtmosphereConstantsWGSL(options.atmosphere!)}${ATMOSPHERE_MATH_WGSL}
@group(0) @binding(2) var trans: texture_2d<f32>;
@group(0) @binding(3) var lutSampler: sampler;`
        : `const ATM_GROUND: f32 = 0.0;`;

    // Sun transmittance to the sample point. With an atmosphere this is Phase 4's LUT, with the
    // planet-occlusion test the sky-view kernel uses; without one there is nothing to attenuate.
    const sunTransmittance = hasAtmosphere
        ? `        let sunVisible = select(1.0, 0.0, atm_sphereDistance(ATM_GROUND, r, muS) > 1.0e-6);
        let sunT = textureSampleLevel(trans, lutSampler, atm_transUv(h, muS), 0.0).rgb * sunVisible;`
        : `        let sunT = vec3<f32>(1.0);`;

    return /* wgsl */ `
${AERIAL_CAMERA_WGSL}${AERIAL_PARAMS_WGSL}${buildAerialAxisWGSL(c)}${atmosphereBlock}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> params: AerialParams;
@group(0) @binding(4) var dst: texture_storage_3d<rgba16float, write>;
${mediaWGSL(hasAtmosphere)}
@compute @workgroup_size(4, 4)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= ${nx}u || id.y >= ${ny}u) { return; }

    // The froxel centre's world-space view ray, reconstructed exactly the way the skybox builds
    // its own (envShaders): unproject the far plane and subtract the eye.
    let uv = (vec2<f32>(f32(id.x), f32(id.y)) + 0.5) / vec2<f32>(${n(nx)}, ${n(ny)});
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let farPoint = camera.invViewProj * vec4<f32>(ndc, 1.0, 1.0);
    let dir = normalize(farPoint.xyz / farPoint.w - camera.eyePos);

    // Camera altitude in km, in the tangent frame Phase 4's sky-view kernel also uses: the eye
    // sits at (0, groundRadius + altitude, 0) and world +Y is up.
    let altitude = (camera.eyePos.y - AERIAL_GROUND_LEVEL) / AERIAL_UNITS_PER_KM;
    let r0 = ATM_GROUND + altitude;
    let sunTo = -normalize(params.sunDir);
    let cosTheta = dot(dir, sunTo);
    let toKm = 1.0 / AERIAL_UNITS_PER_KM;

    var L = vec3<f32>(0.0);
    var T = vec3<f32>(1.0);
    var previous = AERIAL_NEAR * toKm;

    for (var z = 0u; z < ${nz}u; z = z + 1u) {
        let sliceDistKm = aerialSliceDistance(z) * toKm;
        let ds = (sliceDistKm - previous) / ${n(c.stepsPerSlice)};
        for (var k = 0u; k < ${c.stepsPerSlice}u; k = k + 1u) {
            let s = previous + (f32(k) + 0.5) * ds;
            let p = vec3<f32>(dir.x * s, r0 + dir.y * s, dir.z * s);
            let r = max(length(p), 1.0e-6);
            let h = r - ATM_GROUND;
            let up = p / r;
            let muS = dot(sunTo, up);
${sunTransmittance}
            let dsMetres = ds * 1000.0;
            // In-scatter uses the transmittance ENTERING the step, then the step attenuates.
            // Swapping the two biases the whole volume brighter with no other symptom.
            L += T * sunT * params.sunIntensity * aerial_phaseScattering(h, cosTheta) * dsMetres;
            T *= exp(-aerial_extinction(h) * dsMetres);
        }
        // The alpha channel is the MEAN of the three transmittances (§4.1). The chromatic part of
        // extinction survives in the in-scatter, which is blue; this scalar is the approximation
        // the four-channel packing forces, stated rather than discovered.
        textureStore(dst, vec3<i32>(i32(id.x), i32(id.y), i32(z)),
            vec4<f32>(max(L, vec3<f32>(0.0)), dot(T, vec3<f32>(0.33333333))));
        previous = sliceDistKm;
    }
}
`;
}

// ---------------------------------------------------------------------------
// §5.1 — the primary surface: one fullscreen pass over opaque
// ---------------------------------------------------------------------------

/**
 * The fullscreen apply pass. Reads depth, reconstructs view distance, samples the volume, writes
 * `color * T + inScatter`.
 *
 * Chosen over the injected layout for opaque geometry because of correctness-per-pixel (overdraw
 * makes the per-fragment path apply and discard), runtime toggleability with no pipeline rebuild,
 * and depth-aware upsampling — NOT because the injected layout does not fit in group 2. It does
 * fit, with room (§5.2), and saying so plainly is what makes this decision revisitable.
 */
export function buildAerialApplyWGSL(config: ResolvedAerialConfig): string {
    const [nx, ny] = config.dims;

    // The bilateral resolve. Each lateral tap samples the volume at ITS OWN view distance, so on
    // a continuous surface the five taps are five correct estimates and averaging them removes
    // froxel-grid quantization; across a silhouette the taps that disagree about distance are
    // rejected, which is what stops the far surface's fog bleeding into the near surface's pixels
    // (§4.3's characteristic artefact, bounded by T3). The injected path cannot do this: it sees
    // one fragment and has no neighbourhood.
    const resolve = config.depthAwareUpsampling ? /* wgsl */ `
fn aerialTap(uv: vec2<f32>, offset: vec2<f32>, centreDistance: f32, dims: vec2<f32>,
             accum: ptr<function, vec4<f32>>, weight: ptr<function, f32>) {
    let tapUv = clamp(uv + offset, vec2<f32>(0.0), vec2<f32>(1.0));
    let tapDepth = textureLoad(sceneDepth, vec2<i32>(tapUv * dims), 0);
    let tapDistance = aerialViewDistance(tapUv, tapDepth);
    let agrees = tapDepth < 1.0
        && abs(tapDistance - centreDistance) <= aerialParams[2] * max(centreDistance, AERIAL_NEAR);
    let w = select(0.0, 1.0, agrees);
    *accum += w * textureSampleLevel(aerialVolume, aerialSampler, aerialUVW(tapUv, tapDistance), 0.0);
    *weight += w;
}
fn aerialResolve(uv: vec2<f32>, centreDistance: f32, dims: vec2<f32>) -> vec4<f32> {
    var accum = textureSampleLevel(aerialVolume, aerialSampler, aerialUVW(uv, centreDistance), 0.0);
    var weight = 1.0;
    // One froxel in UV, which is just 1/dims of the lateral froxel count. Named froxelStep
    // rather than step because WGSL has a builtin of that name.
    let froxelStep = vec2<f32>(1.0 / ${n(nx)}, 1.0 / ${n(ny)});
    aerialTap(uv, vec2<f32>( froxelStep.x, 0.0), centreDistance, dims, &accum, &weight);
    aerialTap(uv, vec2<f32>(-froxelStep.x, 0.0), centreDistance, dims, &accum, &weight);
    aerialTap(uv, vec2<f32>(0.0,  froxelStep.y), centreDistance, dims, &accum, &weight);
    aerialTap(uv, vec2<f32>(0.0, -froxelStep.y), centreDistance, dims, &accum, &weight);
    return accum / weight;
}
` : /* wgsl */ `
fn aerialResolve(uv: vec2<f32>, centreDistance: f32, dims: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(aerialVolume, aerialSampler, aerialUVW(uv, centreDistance), 0.0);
}
`;

    return /* wgsl */ `
${AERIAL_CAMERA_WGSL}${buildAerialAxisWGSL(config)}
@group(0) @binding(0) var<uniform> camera: Camera;
// The batch storage buffer, honestly used: [enabled, intensity, depthTolerance, _pad]. Writing
// through it is what makes the pass runtime-toggleable with no pipeline recompilation (§5.1).
@group(0) @binding(1) var<storage, read> aerialParams: array<f32>;
@group(1) @binding(0) var sceneColor: texture_2d<f32>;
@group(1) @binding(1) var sceneDepth: texture_depth_2d;
@group(1) @binding(2) var aerialVolume: texture_3d<f32>;
@group(1) @binding(3) var aerialSampler: sampler;
${AERIAL_APPLY_FUNCTIONS_WGSL}
fn aerialViewDistance(uv: vec2<f32>, depth: f32) -> f32 {
    // Reconstruct the view-space position and take its LENGTH, not its z: the volume's W axis is
    // radial distance along the view ray, which is what the froxel march accumulates over. Using
    // linear z here would under-fog the screen edges by 1/cos of the field of view.
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth);
    let viewPos = camera.invProj * vec4<f32>(ndc, 1.0);
    return length(viewPos.xyz / viewPos.w);
}
${resolve}
struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var o: VSOut;
    o.pos = vec4<f32>(p[vi], 0.0, 1.0);
    o.ndc = p[vi];
    return o;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let uv = vec2<f32>(in.ndc.x * 0.5 + 0.5, 0.5 - in.ndc.y * 0.5);
    let dims = vec2<f32>(textureDimensions(sceneDepth));
    let texel = vec2<i32>(uv * dims);
    let base = textureLoad(sceneColor, texel, 0);
    let depth = textureLoad(sceneDepth, texel, 0);

    // Background pixels are the sky, and the sky already carries the whole atmospheric column
    // from Phase 4's sky-view LUT. Fogging them again double-counts the same medium. This early
    // return is what T2 asserts: toggling the pass changes no pixel at the far plane.
    if (depth >= 1.0 || aerialParams[0] < 0.5) { return base; }

    let viewDistance = aerialViewDistance(uv, depth);
    let ap = aerialResolve(uv, viewDistance, dims);
    let scaled = vec4<f32>(ap.rgb * aerialParams[1], ap.a);
    return vec4<f32>(composeAerial(base.rgb, scaled), base.a);
}
`;
}

// ---------------------------------------------------------------------------
// §5.3 — the secondary surface: the injected layout, for transparents
// ---------------------------------------------------------------------------

/**
 * The froxel-backed `ShaderAerialLayout` — the fourth injected layout, the same shape as
 * `lighting`, `indirect` and `sky`.
 *
 * It emits the SAME `AERIAL_APPLY_FUNCTIONS_WGSL` as the fullscreen pass. That is the whole point
 * of §5.4: two paths, one system. The sets of pixels they visit are disjoint by construction (the
 * fullscreen pass runs before the transparent pass and the transparent pass's fragments are not
 * in the depth buffer it read), and U8 asserts it anyway, because "not possible by construction"
 * is a claim about code that changes.
 */
export function buildAerialLayout(config: ResolvedAerialConfig): ShaderAerialLayout {
    const g = config.aerialGroup;
    return {
        wgslStructs: /* wgsl */ `
@group(${g}) @binding(${AERIAL_VOLUME_BINDING}) var aerialVolume: texture_3d<f32>;
@group(${g}) @binding(${AERIAL_SAMPLER_BINDING}) var aerialSampler: sampler;`,
        wgslLogic: `${buildAerialAxisWGSL(config)}${AERIAL_APPLY_FUNCTIONS_WGSL}`,
    };
}

// ---------------------------------------------------------------------------
// §6.2 / §11 step 1 — the analytic single-medium path (the v1.2 exponential height fog)
// ---------------------------------------------------------------------------

/**
 * The closed-form exponential height fog: no volume, no depth texture, no compute stage. It is a
 * drop-in for {@link buildAerialLayout} at the material site — both define `applyAerialAt`, so a
 * material never learns which one it got (design-principles §4).
 *
 * Scattering and absorption share one density profile, so the single-scattering albedo is constant
 * along the ray and the in-scatter collapses to `E * p * albedo * (1 - T)` — the familiar
 * `fogColor * (1 - T)`. The one approximation is that the sun's own transmittance is held constant
 * along the ray; that is exactly what makes it cheap, and U9 pins it against the froxel march.
 */
export function buildHeightFogLayout(config: ResolvedAerialConfig): ShaderAerialLayout {
    const g = config.aerialGroup;
    return {
        wgslStructs: /* wgsl */ `
struct HeightFogParams {
    extinction: vec3<f32>,
    scaleHeight: f32,
    albedo: vec3<f32>,
    referenceHeight: f32,
    sunIrradiance: vec3<f32>,
    g: f32,
    sunDir: vec3<f32>,
    unitsPerKm: f32,
    groundLevel: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};
@group(${g}) @binding(${AERIAL_VOLUME_BINDING}) var<uniform> heightFog: HeightFogParams;`,
        wgslLogic: /* wgsl */ `
fn heightFog_phase(cosTheta: f32) -> f32 {
    let g2 = heightFog.g * heightFog.g;
    return 3.0 * (1.0 - g2) * (1.0 + cosTheta * cosTheta)
        / (8.0 * 3.14159265359 * (2.0 + g2) * pow(max(1.0 + g2 - 2.0 * heightFog.g * cosTheta, 1.0e-4), 1.5));
}
fn heightFog_transmittance(altitude: f32, dirY: f32, distanceKm: f32) -> vec3<f32> {
    let H = heightFog.scaleHeight;
    let base = exp(-(altitude - heightFog.referenceHeight) / H);
    // The k -> 0 limit of (H/k)(1 - exp(-kD/H)) is D. A horizontal ray through ground fog is not
    // a degenerate case to guard against; it is the most common configuration there is.
    let x = dirY * distanceKm / H;
    let lengthKm = select((H / dirY) * (1.0 - exp(-x)), distanceKm, abs(x) < 1.0e-4);
    return exp(-heightFog.extinction * base * lengthKm * 1000.0);
}
fn composeAerial(color: vec3<f32>, ap: vec4<f32>) -> vec3<f32> {
    return color * ap.a + ap.rgb;
}
fn applyAerialAt(color: vec3<f32>, worldPos: vec3<f32>) -> vec3<f32> {
    let delta = worldPos - camera.eyePos;
    let distanceKm = length(delta) / heightFog.unitsPerKm;
    let dir = normalize(delta);
    let altitude = (camera.eyePos.y - heightFog.groundLevel) / heightFog.unitsPerKm;
    let T = heightFog_transmittance(altitude, dir.y, distanceKm);
    let cosTheta = dot(dir, -normalize(heightFog.sunDir));
    let inScatter = heightFog.sunIrradiance * heightFog_phase(cosTheta) * heightFog.albedo
                  * (vec3<f32>(1.0) - T);
    return composeAerial(color, vec4<f32>(inScatter, dot(T, vec3<f32>(0.33333333))));
}
`,
    };
}

/** Resolve-and-build convenience for callers holding a raw config. */
export function buildAerialLayoutFromConfig(config = {}): ShaderAerialLayout {
    return buildAerialLayout(resolveAerialConfig(config));
}
