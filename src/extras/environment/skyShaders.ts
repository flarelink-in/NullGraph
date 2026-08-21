// src/extras/environment/skyShaders.ts
//
// WGSL generators for the sky sources (docs/environment-sky-design.md §4–§6). Each builder
// returns the `{ wgslStructs, wgslLogic }` pair the seam injects into BOTH consumers — the
// source-cube bake kernel (buildSourceCubeWGSL) and the skybox fragment (buildSkyWGSL) — so the
// sky you are lit by and the sky you see are the same function by construction (design §1,
// principle 1; pinned by U2).
//
// The math mirrors skyMath.ts line-for-line; every shared constant (Kasten–Young, XYZ→RGB) is
// interpolated FROM skyMath so the CPU mirror and the shader cannot disagree about a number.
//
// Constraints honoured here (design §4.2, checked by U7/U8):
//   - no `PI` declaration (the ENV_MATH_WGSL preamble owns it in the bake),
//   - helper names are `sky_`-prefixed except the two entry points,
//   - no backtick anywhere (it would terminate the template literal carrying it).

import {
    DEG_TO_RAD, KASTEN_YOUNG, XYZ_TO_RGB,
    GRADIENT_OFF, PREETHAM_OFF,
} from './skyMath';
import type { ShaderSkyLayout } from './types';

/** Full-precision float literal (String() round-trips doubles exactly enough for f32 inputs). */
function num(n: number): string {
    const s = String(n);
    return s.includes('.') || s.includes('e') || s.includes('n') ? s : `${s}.0`;
}

/** Signed term for a sum: `- 1.5372 * Y` rather than `+ -1.5372 * Y`. */
function term(n: number, varName: string): string {
    return n < 0 ? `- ${num(-n)} * ${varName}` : `+ ${num(n)} * ${varName}`;
}

/** The skySunRadiance body the extinction-driven sources share (Kasten–Young, skyMath-owned). */
function sunRadianceWGSL(betaR: string, betaG: string, betaB: string): string {
    const [a, b, c] = KASTEN_YOUNG;
    return /* wgsl */ `
fn skySunRadiance(elevDeg: f32) -> vec3<f32> {
    let h = max(elevDeg, -2.0);
    let am = 1.0 / (sin(h * ${num(DEG_TO_RAD)}) + ${num(a)} * pow(h + ${num(b)}, -${num(c)}));
    return vec3<f32>(exp(-${betaR} * am), exp(-${betaG} * am), exp(-${betaB} * am));
}
`;
}

// ---------------------------------------------------------------------------
// GradientSky (§5.1) — the pre-refactor skyColor, moved behind the seam numerically unchanged
// ---------------------------------------------------------------------------

/**
 * The gradient layout. `wgslLogic` is byte-identical in the bake kernel and the skybox (U2):
 * both consumers read the same `skyParams` floats at the same offsets.
 */
export function buildGradientSkyWGSL(base: number): ShaderSkyLayout {
    const P = (off: number) => `skyParams[${base + off}]`;
    const O = GRADIENT_OFF;
    const wgslLogic = /* wgsl */ `
// The pre-refactor skyColor body (environment-sky-design.md §5.1, U1 pins it numerically inert):
// three-colour gradient plus a sun disc, transcribed into the seam unchanged.
fn sampleSky(dir: vec3<f32>) -> vec3<f32> {
    let t = clamp(dir.y, -1.0, 1.0);
    var c = mix(vec3<f32>(${P(O.HORIZON)}, ${P(O.HORIZON + 1)}, ${P(O.HORIZON + 2)}),
                vec3<f32>(${P(O.ZENITH)}, ${P(O.ZENITH + 1)}, ${P(O.ZENITH + 2)}),
                pow(max(t, 0.0), 0.6));
    c = mix(c, vec3<f32>(${P(O.GROUND)}, ${P(O.GROUND + 1)}, ${P(O.GROUND + 2)}), pow(max(-t, 0.0), 0.5));
    let sunTo = vec3<f32>(-${P(O.SUN_DIR)}, -${P(O.SUN_DIR + 1)}, -${P(O.SUN_DIR + 2)});
    let sunAmt = smoothstep(${P(O.SUN_COS)}, ${P(O.SUN_COS)} + 0.0003, dot(dir, sunTo));
    return c + ${P(O.SUN_INTENSITY)} * sunAmt * skySunRadiance(0.0);
}

// The authored disc tint — constant by construction: the gradient sky has no extinction model,
// so the disc on screen and the directional light that matches it agree at every elevation.
fn skySunRadiance(elevDeg: f32) -> vec3<f32> {
    return vec3<f32>(1.0, 0.98, 0.95);
}
`;
    return { wgslStructs: '', wgslLogic };
}


// ---------------------------------------------------------------------------
// PreethamSky (§5.3) — analytic daylight: Perez distribution + zenith terms + extinction disc
// ---------------------------------------------------------------------------

/**
 * The Preetham layout. The A–E coefficients, zenith terms, and extinction betas arrive through
 * `skyParams` (derived on the CPU by skyMath.ts), so this body is a fixed transcription — the
 * same text goes into the bake kernel and the skybox fragment (U2).
 */
export function buildPreethamSkyWGSL(base: number): ShaderSkyLayout {
    const P = (off: number) => `skyParams[${base + off}]`;
    const O = PREETHAM_OFF;
    const M = XYZ_TO_RGB;
    // One Perez factor with per-channel coefficients, cosθ clamped exactly as the CPU mirror
    // does (skyMath.perezF): B/cosθ diverges at the horizon; the fitted B < 0 makes the true
    // limit 0, so the clamp errs toward slightly more horizon glow (design-principles §3).
    const perezFactor = (p: number) =>
        `(1.0 + ${P(O.PEREZ + p)} * exp(${P(O.PEREZ + p + 1)} / ct)) * ` +
        `(1.0 + ${P(O.PEREZ + p + 2)} * exp(${P(O.PEREZ + p + 3)} * gamma) ` +
        `+ ${P(O.PEREZ + p + 4)} * cg * cg)`;
    const wgslLogic = /* wgsl */ `
// Preetham et al. 1999. Its failure mode is documented, not hidden (§5.3): below ~5° solar
// elevation the fit degrades and the per-channel factors can dip negative near the horizon at
// high turbidity — the final max() keeps the result finite and non-negative (U5).

fn sky_perez(cosTheta: f32, gamma: f32) -> vec3<f32> {
    let ct = max(cosTheta, 0.02);
    let cg = cos(gamma);
    let FY = ${perezFactor(0)};
    let Fx = ${perezFactor(5)};
    let Fy = ${perezFactor(10)};
    // Zenith terms pre-divided by F(0, thetaSun) on the CPU, so no per-texel division here.
    let Y = ${P(O.ZENITH_OVER_F0)} * FY;
    let x = ${P(O.ZENITH_OVER_F0 + 1)} * Fx;
    let y = ${P(O.ZENITH_OVER_F0 + 2)} * Fy;
    let yy = max(y, 1.0e-4);
    let X = x * Y / yy;
    let Z = (1.0 - x - y) * Y / yy;
    return vec3<f32>(
        ${num(M[0])} * X ${term(M[1], 'Y')} ${term(M[2], 'Z')},
        ${num(M[3])} * X ${term(M[4], 'Y')} ${term(M[5], 'Z')},
        ${num(M[6])} * X ${term(M[7], 'Y')} ${term(M[8], 'Z')});
}

fn sampleSky(dir: vec3<f32>) -> vec3<f32> {
    let sunTo = vec3<f32>(-${P(O.SUN_DIR)}, -${P(O.SUN_DIR + 1)}, -${P(O.SUN_DIR + 2)});
    let cosGamma = clamp(dot(dir, sunTo), -1.0, 1.0);
    let gamma = acos(cosGamma);
    var rgb = sky_perez(max(dir.y, 0.02), gamma);
    if (dir.y < 0.02) {
        // Below the horizon: ease toward the ground — albedo x the horizon sky light, so the
        // blend is continuous at dir.y = 0.02 (mirrors the gradient's ground mix).
        let t = clamp((0.02 - dir.y) / 0.1, 0.0, 1.0);
        let ground = vec3<f32>(${P(O.GROUND)}, ${P(O.GROUND + 1)}, ${P(O.GROUND + 2)}) * rgb;
        rgb = mix(rgb, ground, t);
    }
    // The TRUE sun disc (§1): angular radius from sunCos, colour from skySunRadiance — the same
    // function Phase 2's light record mirrors, so disc and light cannot drift apart.
    let discAmt = smoothstep(${P(O.SUN_COS)} - ${P(O.DISC_EDGE)}, ${P(O.SUN_COS)} + ${P(O.DISC_EDGE)}, cosGamma);
    rgb = rgb + discAmt * ${P(O.DISC_INTENSITY)} * skySunRadiance(degrees(asin(clamp(sunTo.y, -1.0, 1.0))));
    return max(rgb, vec3<f32>(0.0));
}
${sunRadianceWGSL(P(O.BETA), P(O.BETA + 1), P(O.BETA + 2))}`;
    return { wgslStructs: '', wgslLogic };
}

// ---------------------------------------------------------------------------
// EquirectSky (§5.2) — the imported environment: two facets, one source
// ---------------------------------------------------------------------------

/**
 * The SKYBOX facet: sampleSky reads the source cube at mip 0 — sourceSize² (512² by default,
 * 0.176°/texel), a 4× angular improvement over the prefiltered cube the skybox used to read.
 * Bindings 14/15 of the sky batch's group 2 — the reserved source range (§4.3).
 */
export function buildEquirectSkyWGSL(): ShaderSkyLayout {
    const wgslStructs = /* wgsl */ `
@group(2) @binding(14) var skySrcCube: texture_cube<f32>;
@group(2) @binding(15) var skySrcSampler: sampler;
`;
    const wgslLogic = /* wgsl */ `
// The source cube at mip 0 (§5.2) — which is why this source requires keepSourceCube: true.
fn sampleSky(dir: vec3<f32>) -> vec3<f32> {
    return textureSampleLevel(skySrcCube, skySrcSampler, dir, 0.0).rgb;
}

// A photographed sky carries no analytic sun; white is the neutral placeholder (§4.2).
fn skySunRadiance(elevDeg: f32) -> vec3<f32> {
    return vec3<f32>(1.0, 1.0, 1.0);
}
`;
    return { wgslStructs, wgslLogic };
}

/**
 * The BAKE facet: the skybox reads the source cube, but the bake WRITES it — so the bake facet
 * samples the 2D equirect asset instead (dirToEquirectUV from ENV_MATH_WGSL, which the bake
 * kernel preamble provides). This is the ONE sanctioned divergence between a source's two
 * facets; the analytic sources share a single body across both sites.
 */
export function buildEquirectSkyBakeWGSL(): ShaderSkyLayout {
    const wgslStructs = /* wgsl */ `
@group(0) @binding(3) var skyEquirect: texture_2d<f32>;
@group(0) @binding(4) var skyEquirectSampler: sampler;
`;
    const wgslLogic = /* wgsl */ `
fn sampleSky(dir: vec3<f32>) -> vec3<f32> {
    return textureSampleLevel(skyEquirect, skyEquirectSampler, dirToEquirectUV(dir), 0.0).rgb;
}

fn skySunRadiance(elevDeg: f32) -> vec3<f32> {
    return vec3<f32>(1.0, 1.0, 1.0);
}
`;
    return { wgslStructs, wgslLogic };
}
