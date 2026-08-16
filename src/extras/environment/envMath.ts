// src/extras/environment/envMath.ts
//
// Pure, GPU-free environment math — the CPU mirror of the WGSL kernels in envShaders.ts
// (docs/lighting-ibl-design.md §6–§7). Everything here runs in a plain unit test with no
// device, so the cube-face convention, the solid-angle weights, the SH projection, and the
// EnvParams packing are pinned by numbers, not by eyeballing a render (design-principles §13).
// It is also worker-safe: plain numbers and typed arrays in, typed arrays out.
//
// The WGSL mirrors live in envShaders.ts (`ENV_MATH_WGSL`); the two are kept in lockstep by
// test/environment-ibl.test.mjs (U1–U7).

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Floats in the `EnvParams` uniform: 3×12 SH + 4 rotation + 4 scalars (design doc §4.3). */
export const ENV_PARAMS_FLOATS = 44;

/** The mip of the source cube the SH projection reads: the 8×8 level (§7.2). */
export const SH_SOURCE_SIZE = 8;

/** Cosine-convolution constants for irradiance from radiance SH (§7.2): A0, A1, A2. */
export const SH_COSINE_L2 = new Float64Array([Math.PI, (2 * Math.PI) / 3, Math.PI / 4]);

// ---------------------------------------------------------------------------
// Cube faces & equirect (§6.1) — mirrors faceDirection / dirToEquirectUV in WGSL
// ---------------------------------------------------------------------------

/**
 * World direction of the texel at normalized `(u, v)` on cube `face`, in the WebGPU cube-face
 * convention. Getting one sign wrong mirrors the environment in a way that looks plausible
 * until a recognizable feature appears in a reflection — U1 pins all six faces.
 */
export function faceDirection(face: number, u: number, v: number, out: Float32Array | Float64Array | number[] = [0, 0, 0]): typeof out {
    const cx = u * 2 - 1;
    const cy = v * 2 - 1;
    let x: number, y: number, z: number;
    switch (face) {
        case 0: x = 1; y = -cy; z = -cx; break;    // +X
        case 1: x = -1; y = -cy; z = cx; break;    // -X
        case 2: x = cx; y = 1; z = cy; break;      // +Y
        case 3: x = cx; y = -1; z = -cy; break;    // -Y
        case 4: x = cx; y = -cy; z = 1; break;     // +Z
        default: x = -cx; y = -cy; z = -1; break;  // -Z
    }
    const l = Math.hypot(x, y, z);
    out[0] = x / l; out[1] = y / l; out[2] = z / l;
    return out;
}

/** Equirectangular UV for a world direction (the inverse the equirect bake samples). */
export function dirToEquirectUV(d: ArrayLike<number>, out: Float32Array | number[] = [0, 0]): typeof out {
    out[0] = Math.atan2(d[2], d[0]) / (2 * Math.PI) + 0.5;
    out[1] = Math.acos(Math.min(1, Math.max(-1, d[1]))) / Math.PI;
    return out;
}

/**
 * The TRUE solid angle of cube texel `(x, y)` on a face of `size` pixels (§7.2). A cube's
 * corner texels subtend markedly less than its centre ones; a uniform 4π/384 weight tilts the
 * whole irradiance field toward the corners — U3 catches that by checking the weights sum to 4π.
 */
export function cubeTexelSolidAngle(x: number, y: number, size: number): number {
    // EXACT, via spherical excess — not the midpoint approximation `(4/size²)/(s²+t²+1)^1.5`.
    // That approximation converges only as O(1/size²): at the 8×8 mip the SH projection actually
    // uses, it over-counts total sphere area by 0.38%, which lands directly in the irradiance as
    // an energy error. The exact form sums to 4π at any size (U3 asserts 1e-4).
    const inv = 1 / size;
    const u = (x + 0.5) * 2 * inv - 1;
    const v = (y + 0.5) * 2 * inv - 1;
    const x0 = u - inv, x1 = u + inv;
    const y0 = v - inv, y1 = v + inv;
    return areaElement(x0, y0) - areaElement(x0, y1) - areaElement(x1, y0) + areaElement(x1, y1);
}

/** The signed spherical area of the [0,x]×[0,y] rectangle on the unit cube face. */
function areaElement(x: number, y: number): number {
    return Math.atan2(x * y, Math.sqrt(x * x + y * y + 1));
}

// ---------------------------------------------------------------------------
// Spherical harmonics — L2 projection & evaluation (§7)
// ---------------------------------------------------------------------------

/**
 * The 9 SH-L2 basis values for a unit direction, in the order the whole module uses:
 * [1, y, z, x, xy, yz, 3z²−1, xz, x²−y²] with the L2 constants folded in.
 * Mirrors `shBasisL2` in WGSL.
 */
export function shBasisL2(d: ArrayLike<number>, out: Float32Array | Float64Array | number[] = new Array(9)): typeof out {
    const x = d[0], y = d[1], z = d[2];
    out[0] = 0.28209479177387814;                    // 1/2 · √(1/π)
    out[1] = 0.4886025119029199 * y;                 // √(3/4π) · y
    out[2] = 0.4886025119029199 * z;
    out[3] = 0.4886025119029199 * x;
    out[4] = 1.0925484305920792 * x * y;             // ½√(15/π) · xy
    out[5] = 1.0925484305920792 * y * z;
    out[6] = 0.31539156525352005 * (3 * z * z - 1);  // ¼√(5/π) · (3z²−1)
    out[7] = 1.0925484305920792 * x * z;
    out[8] = 0.5462742152960396 * (x * x - y * y);   // ¼√(15/π) · (x²−y²)
    return out;
}

/** The cosine-convolution weight for coefficient k (band 0/1/2 → A0/A1/A2). */
function cosineWeight(k: number): number {
    return k === 0 ? SH_COSINE_L2[0] : k < 4 ? SH_COSINE_L2[1] : SH_COSINE_L2[2];
}

/**
 * Project a cube face's worth of radiance samples into SH-L2 irradiance coefficients — the CPU
 * mirror of the single-workgroup projection kernel (§7.2). `cube` is 6 faces × `size`² texels of
 * RGBA, face-major. Output is 9 vec3 coefficients (27 floats), cosine constants folded in, so
 * the stored value IS irradiance and a debug readout is directly interpretable.
 */
export function projectSHL2FromCube(
    cube: Float32Array | Float64Array,
    size: number,
    out: Float32Array = new Float32Array(27),
): Float32Array {
    const basis = new Float64Array(9);
    const dir = new Float64Array(3);
    out.fill(0);
    for (let face = 0; face < 6; face++) {
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const w = cubeTexelSolidAngle(x, y, size);
                faceDirection(face, (x + 0.5) / size, (y + 0.5) / size, dir);
                shBasisL2(dir, basis);
                const t = (face * size * size + y * size + x) * 4;
                for (let k = 0; k < 9; k++) {
                    const b = basis[k] * w * cosineWeight(k);
                    out[k * 3 + 0] += cube[t + 0] * b;
                    out[k * 3 + 1] += cube[t + 1] * b;
                    out[k * 3 + 2] += cube[t + 2] * b;
                }
            }
        }
    }
    return out;
}

/**
 * Evaluate the L2 irradiance in direction `dir` — RAW, so it can ring negative for a
 * high-contrast environment (§7.4). `getIndirectDiffuseCPU` is the clamped mirror the shader
 * actually runs.
 */
export function evalSHL2(coeffs: ArrayLike<number>, dir: ArrayLike<number>, out: Float32Array | number[] = [0, 0, 0]): typeof out {
    const basis = shBasisL2(dir, _basisScratch);
    out[0] = 0; out[1] = 0; out[2] = 0;
    for (let k = 0; k < 9; k++) {
        out[0] += coeffs[k * 3 + 0] * basis[k];
        out[1] += coeffs[k * 3 + 1] * basis[k];
        out[2] += coeffs[k * 3 + 2] * basis[k];
    }
    return out;
}

const _basisScratch = new Float64Array(9);

/** Rotate a vector about +Y by `angle` radians — mirrors `rotateY` in WGSL. */
export function rotateY(v: ArrayLike<number>, angle: number, out: Float32Array | Float64Array | number[] = [0, 0, 0]): typeof out {
    const c = Math.cos(angle), s = Math.sin(angle);
    const x = v[0], z = v[2];
    out[0] = c * x - s * z;
    out[1] = v[1];
    out[2] = s * x + c * z;
    return out;
}

/**
 * The CPU mirror of `getIndirectDiffuse` (§7.4): rotate by the environment yaw, evaluate,
 * clamp the L2 ringing, scale by intensity. The `max(·, 0)` is not cosmetic — an unclamped L2
 * lobe reads as black patches on surfaces facing away from the dominant light (U5).
 */
export function getIndirectDiffuseCPU(
    coeffs: ArrayLike<number>, yaw: number, intensity: number,
    N: ArrayLike<number>, out: Float32Array | number[] = [0, 0, 0],
): typeof out {
    const n = rotateY(N, -yaw, _rotScratch);
    evalSHL2(coeffs, n, out);
    out[0] = Math.max(out[0], 0) * intensity;
    out[1] = Math.max(out[1], 0) * intensity;
    out[2] = Math.max(out[2], 0) * intensity;
    return out;
}

const _rotScratch = new Float64Array(3);

// ---------------------------------------------------------------------------
// EnvParams packing (§4.3, §7.3)
// ---------------------------------------------------------------------------

/**
 * Pack one channel's 9 SH coefficients into 3 vec4 slots (3-per-vec4, the twelfth slot unused).
 * WGSL uniform arrays stride to 16 bytes per element, so `array<vec3,9>` would cost 144 bytes
 * and read awkwardly; this costs 48 and unpacks with three loads. U4 pins the order against the
 * generated WGSL's `shCoeff(k)` — a transposed coefficient produces plausible-looking light
 * coming from the wrong hemisphere.
 */
export function packSHChannel(coeffs: ArrayLike<number>, channel: number, out: Float32Array, offsetVec4: number): void {
    const o = offsetVec4 * 4;
    for (let v = 0; v < 3; v++) {
        for (let c = 0; c < 3; c++) {
            out[o + v * 4 + c] = coeffs[(v * 3 + c) * 3 + channel];
        }
        out[o + v * 4 + 3] = 0;
    }
}

/**
 * Pack the whole `EnvParams` uniform (44 floats, 176 bytes — §4.3):
 * shR@0..11, shG@12..23, shB@24..35, rotation@36..39 (0,0,0,yaw), then
 * intensity@40, specularMip@41, horizonFade@42, pad@43.
 */
export function packEnvParams(
    coeffs: ArrayLike<number>,
    yaw: number, intensity: number, specularMip: number, horizonFade: number,
    out: Float32Array = new Float32Array(ENV_PARAMS_FLOATS),
): Float32Array {
    packSHChannel(coeffs, 0, out, 0);
    packSHChannel(coeffs, 1, out, 3);
    packSHChannel(coeffs, 2, out, 6);
    out[36] = 0; out[37] = 0; out[38] = 0; out[39] = yaw;
    out[40] = intensity;
    out[41] = specularMip;
    out[42] = horizonFade;
    out[43] = 0;
    return out;
}

// ---------------------------------------------------------------------------
// BRDF LUT — the CPU mirror of the split-sum DFG integration (§6.4), for U7
// ---------------------------------------------------------------------------

/** Low-discrepancy Hammersley sequence point (radical inverse base 2). */
export function hammersley(i: number, n: number): [number, number] {
    let bits = i >>> 0;
    bits = ((bits << 16) | (bits >>> 16)) >>> 0;
    bits = (((bits & 0x55555555) << 1) | ((bits >>> 1) & 0x55555555)) >>> 0;
    bits = (((bits & 0x33333333) << 2) | ((bits >>> 2) & 0x33333333)) >>> 0;
    bits = (((bits & 0x0f0f0f0f) << 4) | ((bits >>> 4) & 0x0f0f0f0f)) >>> 0;
    bits = (((bits & 0x00ff00ff) << 8) | ((bits >>> 8) & 0x00ff00ff)) >>> 0;
    return [i / n, bits * 2.3283064365386963e-10];
}

/** GGX importance-sampled half-vector for roughness `a` (already squared) about N=(0,0,1). */
export function importanceSampleGGX(xi: [number, number], a: number): [number, number, number] {
    const phi = 2 * Math.PI * xi[0];
    const cosTheta = Math.sqrt((1 - xi[1]) / (1 + (a * a - 1) * xi[1]));
    const sinTheta = Math.sqrt(Math.max(0, 1 - cosTheta * cosTheta));
    return [Math.cos(phi) * sinTheta, Math.sin(phi) * sinTheta, cosTheta];
}

function geometrySchlickGGX_IBL(nDotV: number, roughness: number): number {
    const a = roughness * roughness;
    const k = (a * a) / 2;
    return nDotV / (nDotV * (1 - k) + k);
}

function geometrySmithIBL(nDotV: number, nDotL: number, roughness: number): number {
    return geometrySchlickGGX_IBL(nDotV, roughness) * geometrySchlickGGX_IBL(nDotL, roughness);
}

/**
 * Integrate the split-sum DFG term at `(NdotV, roughness)` — the exact loop the LUT bake runs
 * in WGSL (§6.4), with N = V = (0,0,1) in the integration frame. Returns `[scale, bias]`.
 */
export function integrateBRDFCPU(NdotV: number, roughness: number, samples = 1024): [number, number] {
    const sinV = Math.sqrt(Math.max(0, 1 - NdotV * NdotV));
    const V: [number, number, number] = [sinV, 0, NdotV];
    let a = 0, b = 0;
    for (let i = 0; i < samples; i++) {
        const xi = hammersley(i, samples);
        const H = importanceSampleGGX(xi, roughness);
        // L = 2·dot(V,H)·H − V
        const vh = V[0] * H[0] + V[1] * H[1] + V[2] * H[2];
        const L = [2 * vh * H[0] - V[0], 2 * vh * H[1] - V[1], 2 * vh * H[2] - V[2]];
        const nDotL = Math.max(L[2], 0);
        const nDotH = Math.max(H[2], 0);
        const vDotH = Math.max(vh, 0);
        if (nDotL > 0) {
            const g = geometrySmithIBL(NdotV, nDotL, roughness);
            const gVis = (g * vDotH) / (nDotH * NdotV);
            const fc = Math.pow(1 - vDotH, 5);
            a += (1 - fc) * gVis;
            b += fc * gVis;
        }
    }
    return [a / samples, b / samples];
}

// ---------------------------------------------------------------------------
// Config validation (§11) — pure, so the error messages are pinned headless
// ---------------------------------------------------------------------------

/**
 * Validate the environment config. Every failure names the offender and the way out
 * (design-principles §2) — a silently clamped value here would surface as a black scene or a
 * validation error deep in the bake, attributed to the wrong subsystem.
 */
export function validateEnvironmentConfig(cfg: {
    sourceSize: number;
    prefilterMips: number;
    environmentGroup: number;
    updateMode: 'once' | 'amortized';
    keepSourceCube: boolean;
    format?: string;
}): void {
    const isPow2 = (n: number) => Number.isInteger(n) && n >= 8 && (n & (n - 1)) === 0;
    if (!isPow2(cfg.sourceSize)) {
        throw new Error(
            `EnvironmentSystem: sourceSize ${cfg.sourceSize} must be a power of two ≥ 8; the mip ` +
            `chain and the SH projection mip assume it.`);
    }
    const maxMips = Math.log2(cfg.sourceSize) + 1;
    if (cfg.prefilterMips > maxMips) {
        throw new Error(
            `EnvironmentSystem: prefilterMips ${cfg.prefilterMips} exceeds log2(sourceSize)+1 = ${maxMips}.`);
    }
    if (cfg.environmentGroup !== 2) {
        throw new Error(
            `EnvironmentSystem: environmentGroup ${cfg.environmentGroup} must equal the LightSystem's ` +
            `group (2); core WebGPU guarantees only 4 bind groups and all four are in use ` +
            `(lighting-ibl-design.md §3).`);
    }
    if (cfg.format === 'rgba32float') {
        throw new Error(
            `EnvironmentSystem: format "rgba32float" needs the float32-filterable feature and cannot ` +
            `be linearly sampled on the baseline device; use rgba16float.`);
    }
    if (cfg.updateMode === 'amortized' && !cfg.keepSourceCube) {
        throw new Error(
            `EnvironmentSystem: updateMode "amortized" requires keepSourceCube: true; the prefilter ` +
            `reads the source every refresh.`);
    }
}
