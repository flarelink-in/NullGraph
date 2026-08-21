// src/extras/environment/SkySource.ts
//
// The three sky sources Phase 1 ships (docs/environment-sky-design.md §5), covering the three
// ways a sky exists: authored (GradientSky), photographed (EquirectSky), computed (PreethamSky).
// A source is data, not a code path (design principle 2): the EnvironmentSystem injects a
// source's layout into the bake kernel and the skybox alike and never branches on its type.
//
// Writing a new source is the whole extension story (§7): implement SkySource, emit WGSL that
// reads skyParams[base + i], and serialize params with an allocation-free writeParams.

import type { EnvironmentSystem } from './EnvironmentSystem';
import type { ProceduralSkyConfig, ShaderSkyLayout, SkySource } from './types';
import {
    DEG_TO_RAD, GRADIENT_SKY_FLOATS, PREETHAM_SKY_FLOATS, PREETHAM_OFF,
    gradientFromConfig, gradientWriteParams, normalize3,
    preethamBeta, preethamPerezCoeffs, preethamZenith, preethamF0, preethamDiscEdge,
} from './skyMath';
import {
    buildGradientSkyWGSL, buildPreethamSkyWGSL, buildEquirectSkyWGSL, buildEquirectSkyBakeWGSL,
} from './skyShaders';

// ---------------------------------------------------------------------------
// §5.1 GradientSky — the zero-asset default (the pre-refactor sky, numerically unchanged)
// ---------------------------------------------------------------------------

/**
 * The shipped default and the recommended source for stylized work: a three-colour gradient
 * plus a sun disc, ~20 flops. Its `skySunRadiance` is the constant authored disc tint, so the
 * disc on screen and the directional light that matches it agree at every elevation.
 */
export class GradientSky implements SkySource {
    public readonly name = 'GradientSky';
    public readonly paramFloats = GRADIENT_SKY_FLOATS;

    public readonly zenith: [number, number, number];
    public readonly horizon: [number, number, number];
    public readonly ground: [number, number, number];
    public readonly sunDir: [number, number, number];
    public readonly sunIntensity: number;
    public readonly sunCos: number;

    constructor(cfg: ProceduralSkyConfig) {
        const s = gradientFromConfig(cfg);
        this.zenith = s.zenith;
        this.horizon = s.horizon;
        this.ground = s.ground;
        this.sunIntensity = s.sunIntensity;
        this.sunCos = s.sunCos;
        const l = Math.hypot(s.sunDir[0], s.sunDir[1], s.sunDir[2]) || 1;
        this.sunDir = [s.sunDir[0] / l, s.sunDir[1] / l, s.sunDir[2] / l];
    }

    /** Move the sun (TRAVEL direction). Params-only; no rebuild (design §7). */
    public setSunDirection(dir: ArrayLike<number>): void {
        const l = Math.hypot(dir[0], dir[1], dir[2]) || 1;
        this.sunDir[0] = dir[0] / l;
        this.sunDir[1] = dir[1] / l;
        this.sunDir[2] = dir[2] / l;
    }

    public getShaderLayout(base: number): ShaderSkyLayout {
        return buildGradientSkyWGSL(base);
    }

    public writeParams(out: Float32Array, base: number): void {
        gradientWriteParams(this, out, base);
    }
}

// ---------------------------------------------------------------------------
// §5.3 PreethamSky — analytic daylight (turbidity-driven Perez model)
// ---------------------------------------------------------------------------

export interface PreethamSkyConfig {
    /** Atmospheric turbidity: 1.8 = pristine, 10 = urban haze. Default 2.8. */
    turbidity?: number;
    /** Below-horizon albedo. Default [0.15, 0.14, 0.12]. */
    groundAlbedo?: [number, number, number];
    /** The sun's TRAVEL direction (where the light goes). Default [-0.4, -1.0, -0.35]. */
    sunDir?: [number, number, number];
    /** Sun disc angular RADIUS in degrees. Default 0.266 — the TRUE sun (§1). */
    sunAngularRadius?: number;
    /**
     * Authored disc radiance in scene units (§4.4). Not the physical ~46,000: a physical disc
     * at physical radiance would over-count irradiance in the 8×8 SH projection, whose texels
     * are 11.25° wide. Default 40.
     */
    sunDiscIntensity?: number;
}

/**
 * Preetham et al. 1999 analytic daylight: five distribution coefficients (A–E) per channel,
 * driven by turbidity, plus absolute zenith terms — the sky's gradient, horizon brightening,
 * and circumsolar glow are DERIVED rather than authored, which is what lets Phase 2 drive it
 * with one number (design §5.3). The A–E coefficients and zenith terms are derived on the CPU
 * at set-time and ride in the params array, so the per-texel WGSL body stays ~40 flops (§6.3).
 * Changing turbidity or the sun direction rewrites params only — no pipeline rebuild.
 */
export class PreethamSky implements SkySource {
    public readonly name = 'PreethamSky';
    public readonly paramFloats = PREETHAM_SKY_FLOATS;

    private turbidity: number;
    private readonly groundAlbedo: [number, number, number];
    private readonly sunDir = new Float32Array(3);   // TRAVEL direction, normalized
    private readonly sunCos: number;
    private readonly discEdge: number;
    private readonly discIntensity: number;

    // Derived state — recomputed when turbidity or the sun moves; writeParams is pure copies.
    private readonly beta = new Float32Array(3);
    private readonly zenithOverF0 = new Float32Array(3);
    private readonly perez = new Float32Array(15);
    private readonly scratch3 = new Float32Array(3);
    private readonly scratchF0 = new Float32Array(3);

    constructor(cfg: PreethamSkyConfig = {}) {
        this.turbidity = cfg.turbidity ?? 2.8;
        const ga = cfg.groundAlbedo ?? [0.15, 0.14, 0.12];
        this.groundAlbedo = [ga[0], ga[1], ga[2]];
        this.discIntensity = cfg.sunDiscIntensity ?? 40.0;
        const radiusRad = (cfg.sunAngularRadius ?? 0.266) * DEG_TO_RAD;
        this.sunCos = Math.cos(radiusRad);
        this.discEdge = preethamDiscEdge(this.sunCos);
        normalize3(cfg.sunDir ?? [-0.4, -1.0, -0.35], this.sunDir);
        this.recompute();
    }

    /** Move the sun (TRAVEL direction). Params-only; no rebuild (design §7). */
    public setSunDirection(dir: ArrayLike<number>): void {
        normalize3(dir, this.sunDir);
        this.recompute();
    }

    /** Change turbidity. Params-only; no rebuild — but DO invalidate the bake (Phase 2 §6.4). */
    public setTurbidity(turbidity: number): void {
        this.turbidity = turbidity;
        this.recompute();
    }

    /** Current turbidity — introspection for tests and Phase 2's re-bake gate. */
    public get currentTurbidity(): number { return this.turbidity; }

    private recompute(): void {
        preethamBeta(this.turbidity, this.beta);
        preethamPerezCoeffs(this.turbidity, this.perez);
        // Sun zenith angle from the TRAVEL direction: sunPos = -sunDir, thetaS = acos(sunPos.y).
        const thetaS = Math.acos(Math.min(1.0, Math.max(-1.0, -this.sunDir[1])));
        preethamZenith(this.turbidity, thetaS, this.scratch3);
        preethamF0(this.perez, thetaS, this.scratchF0);
        for (let i = 0; i < 3; i++) {
            this.zenithOverF0[i] = this.scratch3[i] / this.scratchF0[i];
        }
    }

    public getShaderLayout(base: number): ShaderSkyLayout {
        return buildPreethamSkyWGSL(base);
    }

    public writeParams(out: Float32Array, base: number): void {
        const O = PREETHAM_OFF;
        out[base + O.SUN_DIR] = this.sunDir[0];
        out[base + O.SUN_DIR + 1] = this.sunDir[1];
        out[base + O.SUN_DIR + 2] = this.sunDir[2];
        out[base + O.SUN_COS] = this.sunCos;
        out[base + O.DISC_EDGE] = this.discEdge;
        out[base + O.DISC_INTENSITY] = this.discIntensity;
        out[base + O.GROUND] = this.groundAlbedo[0];
        out[base + O.GROUND + 1] = this.groundAlbedo[1];
        out[base + O.GROUND + 2] = this.groundAlbedo[2];
        for (let i = 0; i < 3; i++) out[base + O.BETA + i] = this.beta[i];
        for (let i = 0; i < 3; i++) out[base + O.ZENITH_OVER_F0 + i] = this.zenithOverF0[i];
        for (let i = 0; i < 15; i++) out[base + O.PEREZ + i] = this.perez[i];
    }
}

// ---------------------------------------------------------------------------
// §5.2 EquirectSky — the imported environment (bake reads the 2D asset, skybox reads the cube)
// ---------------------------------------------------------------------------

/**
 * A photographed sky. The bake facet converts the 2D equirect into the source cube; the skybox
 * facet then samples that cube at mip 0 — 0.176°/texel at the default sourceSize, 4× sharper
 * than the prefiltered cube the skybox used to read. `needsSourceCube` is what makes
 * `keepSourceCube: true` mandatory; the mismatch is a named build-time error (§9, U9).
 */
export class EquirectSky implements SkySource {
    public readonly name = 'EquirectSky';
    public readonly paramFloats = 0;
    public readonly needsSourceCube = true;

    private readonly equirectView: GPUTextureView;

    constructor(opts: { equirect: GPUTextureView }) {
        this.equirectView = opts.equirect;
    }

    public getShaderLayout(_base: number): ShaderSkyLayout {
        return buildEquirectSkyWGSL();
    }

    /** The bake cannot sample the cube it writes, so this facet reads the 2D asset instead. */
    public getBakeShaderLayout(_base: number): ShaderSkyLayout {
        return buildEquirectSkyBakeWGSL();
    }

    public getBakeBindEntries(env: EnvironmentSystem): GPUBindGroupEntry[] {
        return [
            { binding: 3, resource: this.equirectView },
            { binding: 4, resource: env.sampler },
        ];
    }

    public getSkyboxBindEntries(env: EnvironmentSystem): GPUBindGroupEntry[] {
        return [
            { binding: 14, resource: env.sourceView },
            { binding: 15, resource: env.sampler },
        ];
    }

    public writeParams(_out: Float32Array, _base: number): void {
        // No scalar params — the source IS the texture.
    }
}

