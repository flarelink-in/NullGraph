import type { CelestialState } from './TimeOfDay';
import type { ShaderSkyLayout, SkySource } from './types';
import { DEG_TO_RAD } from './skyMath';
import { moonRadiance, NIGHT_MOON_RADIUS_RAD } from './nightMath';
import { buildNightSkyWGSL, NIGHT_OFF, NIGHT_SKY_FLOATS } from './nightShaders';

export interface NightSkyConfig {
    nightGain?: number;
    moonAngularRadius?: number;
    moonColor?: [number, number, number];
    earthshine?: [number, number, number];
    domeColor?: [number, number, number];
    airglow?: number;
    lightPollutionColor?: [number, number, number];
    lightPollution?: number;
    milkyWay?: number;
    latitudeDeg?: number;
    fixedGmstRad?: number;
}

/** Resolved night dome and lunar disc. Unresolved stars are deliberately a separate StarField. */
export class NightSky implements SkySource {
    public readonly name = 'NightSky';
    public readonly paramFloats = NIGHT_SKY_FLOATS;

    private readonly moonDir = new Float32Array([0, -1, 0]);
    private readonly sunDir = new Float32Array([0, 1, 0]);
    private readonly moonColor: [number, number, number];
    private readonly earthshine: [number, number, number];
    private readonly dome: [number, number, number];
    private readonly pollution: [number, number, number];
    private readonly moonRadius: number;
    private illumination = 0;
    private elevationDeg = 45;
    private phase = 0;
    private gmst: number;
    private readonly latitude: number;
    private readonly gain: number;
    private readonly airglow: number;
    private readonly pollutionStrength: number;
    private readonly milkyWay: number;

    constructor(cfg: NightSkyConfig = {}) {
        this.gain = cfg.nightGain ?? 1;
        if (this.gain <= 0) {
            throw new Error('NightSky: nightGain must be > 0; 0 renders an unlit night, which is what omitting NightSky does (design §1).');
        }
        this.moonRadius = (cfg.moonAngularRadius ?? NIGHT_MOON_RADIUS_RAD / DEG_TO_RAD) * DEG_TO_RAD;
        this.moonColor = cfg.moonColor ?? [1.0, 0.92, 0.78];
        this.earthshine = cfg.earthshine ?? [0.00012, 0.00016, 0.00022];
        this.dome = cfg.domeColor ?? [0.00018, 0.00027, 0.00055];
        this.airglow = cfg.airglow ?? 0.00012;
        this.pollution = cfg.lightPollutionColor ?? [1.0, 0.34, 0.08];
        this.pollutionStrength = cfg.lightPollution ?? 0;
        this.milkyWay = cfg.milkyWay ?? 0.0006;
        this.latitude = (cfg.latitudeDeg ?? 45) * DEG_TO_RAD;
        this.gmst = cfg.fixedGmstRad ?? 0;
    }

    public setSunDirection(dir: ArrayLike<number>): void { this.copyNormal(dir, this.sunDir); }
    public setMoonDirection(dir: ArrayLike<number>): void { this.copyNormal(dir, this.moonDir); }

    public setCelestialState(state: CelestialState): void {
        this.setSunDirection(state.sunDir);
        this.setMoonDirection(state.moonDir);
        this.illumination = state.moonIllumination;
        this.elevationDeg = state.moonElevationDeg;
        this.phase = state.moonPhase;
        this.gmst = state.gmstRad;
    }

    private copyNormal(dir: ArrayLike<number>, out: Float32Array): void {
        const l = Math.hypot(dir[0], dir[1], dir[2]) || 1;
        out[0] = dir[0] / l; out[1] = dir[1] / l; out[2] = dir[2] / l;
    }

    public getShaderLayout(base: number): ShaderSkyLayout { return buildNightSkyWGSL(base); }

    public writeParams(out: Float32Array, base: number): void {
        const O = NIGHT_OFF;
        out.set(this.moonDir, base + O.MOON_DIR);
        out[base + O.MOON_COS] = Math.cos(this.moonRadius);
        out.set(this.sunDir, base + O.SUN_DIR);
        out[base + O.MOON_SIN] = Math.sin(this.moonRadius);
        const phaseAngle = Math.acos(Math.min(1, Math.max(-1, 2 * this.illumination - 1)));
        const radiance = moonRadiance(this.illumination, this.elevationDeg, phaseAngle, this.gain);
        out[base + O.MOON_RGB] = this.moonColor[0] * radiance;
        out[base + O.MOON_RGB + 1] = this.moonColor[1] * radiance;
        out[base + O.MOON_RGB + 2] = this.moonColor[2] * radiance;
        out[base + O.ILLUMINATION] = this.illumination;
        out.set(this.earthshine, base + O.EARTHSHINE);
        out.set(this.dome, base + O.DOME);
        out[base + O.AIRGLOW] = this.airglow;
        out.set(this.pollution, base + O.POLLUTION);
        out[base + O.POLLUTION_STRENGTH] = this.pollutionStrength;
        out[base + O.GMST] = this.gmst;
        out[base + O.LATITUDE] = this.latitude;
        out[base + O.NIGHT_GAIN] = this.gain;
        out[base + O.MILKY_WAY] = this.milkyWay;
    }
}