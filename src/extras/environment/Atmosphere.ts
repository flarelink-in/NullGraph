import type { NullGraph, ComputeStageNode } from '../../core';
import { type AtmosphereConfig, type AtmosphereMedium, resolveAtmosphereConfig, transmittanceUv } from './atmosphereMath';
import { buildMultipleScatteringWGSL, buildSkyViewWGSL, buildTransmittanceWGSL } from './atmosphereShaders';

/** Owns the physical-atmosphere LUTs and their ordered compute stages. */
export class Atmosphere {
    public readonly config: Required<AtmosphereConfig>;
    private readonly device: GPUDevice;
    private readonly transmittanceTexture: GPUTexture;
    private readonly multipleScatteringTexture: GPUTexture;
    private readonly skyViewTexture: GPUTexture;
    private readonly transmittanceTextureView: GPUTextureView;
    private readonly multipleScatteringTextureView: GPUTextureView;
    private readonly skyViewTextureView: GPUTextureView;
    private readonly lutSampler: GPUSampler;
    private readonly skyViewParamsBuffer: GPUBuffer;
    private readonly skyViewParams = new Float32Array(4);
    private readonly initialStages: ComputeStageNode[];
    private readonly frameStages: ComputeStageNode[];
    private mediumStagesIssued = false;
    private frame = 0;
    private readonly sunDir = new Float32Array([0, -1, 0]);
    private cameraAltitude = 0;

    constructor(engine: NullGraph, config: AtmosphereConfig = {}) {
        this.device = engine.device;
        this.config = resolveAtmosphereConfig(config);
        const c = this.config;
        const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
        this.transmittanceTexture = this.device.createTexture({
            label: 'NullGraph_AtmosphereTransmittance', size: c.transmittanceSize, format: 'rgba16float', usage,
        });
        this.multipleScatteringTexture = this.device.createTexture({
            label: 'NullGraph_AtmosphereMultipleScattering', size: c.multipleScatteringSize, format: 'rgba16float', usage,
        });
        this.skyViewTexture = this.device.createTexture({
            label: 'NullGraph_AtmosphereSkyView', size: c.skyViewSize, format: 'rgba16float', usage,
        });
        this.transmittanceTextureView = this.transmittanceTexture.createView();
        this.multipleScatteringTextureView = this.multipleScatteringTexture.createView();
        this.skyViewTextureView = this.skyViewTexture.createView();
        this.lutSampler = this.device.createSampler({
            label: 'NullGraph_AtmosphereSampler', magFilter: 'linear', minFilter: 'linear',
            addressModeU: 'repeat', addressModeV: 'clamp-to-edge',
        });
        this.skyViewParamsBuffer = this.device.createBuffer({
            label: 'NullGraph_AtmosphereSkyViewParams', size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const transmittanceStage = engine.createComputeStage({
            name: 'Atmosphere: transmittance LUT', shaderCode: buildTransmittanceWGSL(c),
            dispatchCount: [c.transmittanceSize[0] / 8, c.transmittanceSize[1] / 8, 1],
        });
        engine.setComputePassBindGroup(transmittanceStage, [{ binding: 0, resource: this.transmittanceTextureView }]);

        const multipleScatteringStage = engine.createComputeStage({
            name: 'Atmosphere: multiple-scattering LUT', shaderCode: buildMultipleScatteringWGSL(c),
            dispatchCount: [c.multipleScatteringSize[0], c.multipleScatteringSize[1], 1],
        });
        engine.setComputePassBindGroup(multipleScatteringStage, [
            { binding: 0, resource: this.transmittanceTextureView },
            { binding: 1, resource: this.lutSampler },
            { binding: 2, resource: this.multipleScatteringTextureView },
        ]);

        const skyViewStage = engine.createComputeStage({
            name: 'Atmosphere: sky-view LUT', shaderCode: buildSkyViewWGSL(c),
            dispatchCount: [c.skyViewSize[0] / 8, c.skyViewSize[1] / 4, 1],
        });
        engine.setComputePassBindGroup(skyViewStage, [
            { binding: 0, resource: { buffer: this.skyViewParamsBuffer } },
            { binding: 1, resource: this.transmittanceTextureView },
            { binding: 2, resource: this.multipleScatteringTextureView },
            { binding: 3, resource: this.lutSampler },
            { binding: 4, resource: this.skyViewTextureView },
        ]);
        this.initialStages = [transmittanceStage, multipleScatteringStage, skyViewStage];
        this.frameStages = [skyViewStage];
    }

    public setSunDirection(dir: ArrayLike<number>): void {
        const length = Math.hypot(dir[0], dir[1], dir[2]) || 1;
        this.sunDir[0] = dir[0] / length; this.sunDir[1] = dir[1] / length; this.sunDir[2] = dir[2] / length;
    }

    public setCameraAltitude(heightKm: number): void {
        const thickness = this.config.atmosphereTopRadius - this.config.groundRadius;
        if (!Number.isFinite(heightKm) || heightKm < 0) {
            throw new Error(`Atmosphere: camera altitude ${heightKm} km must be finite and non-negative.`);
        }
        if (heightKm > thickness) {
            throw new Error(`Atmosphere: camera at ${heightKm} km is above atmosphereTopRadius ${this.config.atmosphereTopRadius} km; the sky-view LUT is undefined outside the medium. Clamp the camera or raise the top.`);
        }
        this.cameraAltitude = heightKm;
    }

    public getStages(): ComputeStageNode[] {
        this.skyViewParams.set(this.sunDir, 0); this.skyViewParams[3] = this.cameraAltitude;
        this.device.queue.writeBuffer(this.skyViewParamsBuffer, 0, this.skyViewParams as Float32Array<ArrayBuffer>);
        const runSkyView = this.frame++ % this.config.skyViewInterval === 0;
        if (!this.mediumStagesIssued) { this.mediumStagesIssued = true; return this.initialStages; }
        return runSkyView ? this.frameStages : EMPTY_STAGES;
    }

    public get transmittanceView(): GPUTextureView { return this.transmittanceTextureView; }
    public get multipleScatteringView(): GPUTextureView { return this.multipleScatteringTextureView; }
    public get skyView(): GPUTextureView { return this.skyViewTextureView; }
    public get sampler(): GPUSampler { return this.lutSampler; }
    public get medium(): AtmosphereMedium { return this.config; }
    public get sunDirection(): Float32Array { return this.sunDir; }
    public sunTransmittanceUv(out: Float32Array): Float32Array {
        return transmittanceUv(0, Math.min(1, Math.max(-1, -this.sunDir[1])), this.config, out);
    }

    public destroy(): void {
        this.transmittanceTexture.destroy(); this.multipleScatteringTexture.destroy();
        this.skyViewTexture.destroy(); this.skyViewParamsBuffer.destroy();
    }
}

const EMPTY_STAGES: ComputeStageNode[] = [];