import type { EnvironmentSystem } from './EnvironmentSystem';
import type { CelestialState } from './TimeOfDay';
import type { ShaderSkyLayout, SkySource } from './types';
import { preethamDiscEdge } from './skyMath';
import { Atmosphere } from './Atmosphere';
import { buildAtmosphereBakeLayout, buildAtmosphereSkyLayout } from './atmosphereShaders';

export interface AtmosphereSkyConfig { sunAngularRadius?: number; sunDiscIntensity?: number; }

/** Drop-in SkySource backed by an Atmosphere's per-frame sky-view LUT. */
export class AtmosphereSky implements SkySource {
    public readonly name = 'AtmosphereSky';
    public readonly paramFloats = 10;
    private readonly sunCos: number;
    private readonly discEdge: number;
    private readonly discIntensity: number;
    private readonly uv = new Float32Array(2);

    constructor(private readonly atmosphere: Atmosphere, config: AtmosphereSkyConfig = {}) {
        const radius = (config.sunAngularRadius ?? 0.266) * Math.PI / 180;
        this.sunCos = Math.cos(radius); this.discEdge = preethamDiscEdge(this.sunCos);
        this.discIntensity = config.sunDiscIntensity ?? 20;
    }

    public setSunDirection(dir: ArrayLike<number>): void { this.atmosphere.setSunDirection(dir); }
    public setCelestialState(state: CelestialState): void { this.setSunDirection(state.sunDir); }
    public getShaderLayout(base: number): ShaderSkyLayout { return buildAtmosphereSkyLayout(base, 2); }
    public getBakeShaderLayout(base: number): ShaderSkyLayout { return buildAtmosphereBakeLayout(base); }
    public getBakeBindEntries(_env: EnvironmentSystem): GPUBindGroupEntry[] {
        return [{ binding: 3, resource: this.atmosphere.transmittanceView },
            { binding: 4, resource: this.atmosphere.skyView }, { binding: 5, resource: this.atmosphere.sampler }];
    }
    public getSkyboxBindEntries(_env: EnvironmentSystem): GPUBindGroupEntry[] {
        return [{ binding: 14, resource: this.atmosphere.transmittanceView },
            { binding: 15, resource: this.atmosphere.skyView }, { binding: 16, resource: this.atmosphere.sampler }];
    }
    public writeParams(out: Float32Array, base: number): void {
        const sun = this.atmosphere.sunDirection;
        out[base] = sun[0]; out[base + 1] = sun[1]; out[base + 2] = sun[2];
        out[base + 3] = this.sunCos; out[base + 4] = this.discEdge; out[base + 5] = this.discIntensity;
        out[base + 6] = -sun[1] >= 0 ? 1 : 0; out[base + 7] = 0;
        this.atmosphere.sunTransmittanceUv(this.uv);
        out[base + 8] = this.uv[0]; out[base + 9] = this.uv[1];
    }
}