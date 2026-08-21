import type { NullGraph, RenderBatch, RenderPassNode } from '../../core';
import type { EnvironmentSystem } from './EnvironmentSystem';
import type { CelestialState } from './TimeOfDay';
import { DEG_TO_RAD } from './skyMath';
import { proceduralStarCatalogue } from './nightMath';
import { buildStarWGSL } from './nightShaders';

export interface StarFieldConfig {
    maxStars?: number; stars?: Float32Array; latitudeDeg?: number; longitudeDeg?: number;
    fixedGmstRad?: number; twinkle?: number; radiusPx?: number; intensity?: number;
}

/** One vertex-buffer-less, additive instanced draw for the unresolved star catalogue. */
export class StarField {
    private readonly engine: NullGraph;
    private readonly maxStars: number;
    private readonly params = new Float32Array(8);
    private stars: Float32Array;
    private batch: RenderBatch | null = null;
    private paramsBuffer: GPUBuffer | null = null;
    private dirty = true;

    constructor(engine: NullGraph, cfg: StarFieldConfig = {}) {
        this.engine = engine;
        this.maxStars = cfg.maxStars ?? 9100;
        this.stars = cfg.stars ?? proceduralStarCatalogue(this.maxStars);
        this.validate(this.stars);
        this.params[0] = cfg.fixedGmstRad ?? 0;
        this.params[1] = (cfg.latitudeDeg ?? 45) * DEG_TO_RAD;
        this.params[2] = (cfg.longitudeDeg ?? 0) * DEG_TO_RAD;
        this.params[3] = cfg.twinkle ?? 0.18;
        this.params[6] = cfg.radiusPx ?? 1.5;
        this.params[7] = cfg.intensity ?? 0.035;
    }

    private validate(data: Float32Array): void {
        if (data.length % 4 !== 0) throw new Error(`StarField: star records are 4 floats (ra, dec, magnitude, bv); got ${data.length}, which is not a multiple of 4.`);
        const n = data.length / 4;
        if (n > this.maxStars) throw new Error(`StarField: ${n} stars exceeds maxInstances ${this.maxStars} fixed at batch creation. Pass maxStars at construction.`);
    }

    /** Adopt by reference; upload() observes later caller mutations. */
    public setStars(data: Float32Array): void { this.validate(data); this.stars = data; this.dirty = true; }
    public setCelestialState(state: CelestialState): void { this.params[0] = state.gmstRad; }
    public setParams(p: { twinkle?: number; radiusPx?: number; intensity?: number; latitudeDeg?: number; longitudeDeg?: number }): void {
        if (p.twinkle !== undefined) this.params[3] = p.twinkle;
        if (p.radiusPx !== undefined) this.params[6] = p.radiusPx;
        if (p.intensity !== undefined) this.params[7] = p.intensity;
        if (p.latitudeDeg !== undefined) this.params[1] = p.latitudeDeg * DEG_TO_RAD;
        if (p.longitudeDeg !== undefined) this.params[2] = p.longitudeDeg * DEG_TO_RAD;
    }

    public attach(environment: EnvironmentSystem, target?: Parameters<EnvironmentSystem['getSkyPass']>[0]): RenderBatch {
        const pass = target ? environment.getSkyPass(target) : environment.getCreatedSkyPass();
        return this.attachToPass(pass, environment.skyTargetFormat);
    }

    public attachToPass(pass: RenderPassNode, targetFormat: GPUTextureFormat = 'rgba16float'): RenderBatch {
        if (this.batch) return this.batch;
        this.params[4] = pass.size?.[0] ?? this.engine.width;
        this.params[5] = pass.size?.[1] ?? this.engine.height;
        this.paramsBuffer = this.engine.device.createBuffer({
            label: 'NullGraph_StarParams', size: this.params.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.batch = this.engine.createBatch(pass, {
            shaderCode: buildStarWGSL(), strideFloats: 4, maxInstances: this.maxStars,
            targetFormat,
            blend: {
                color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
                alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            },
            depthWriteEnabled: false, depthCompare: 'less-equal',
        });
        this.engine.setRenderPassBindGroup(this.batch, [{ binding: 0, resource: { buffer: this.paramsBuffer } }], 1);
        this.upload();
        return this.batch;
    }

    public upload(width?: number, height?: number): void {
        if (!this.batch || !this.paramsBuffer) return;
        const w = width ?? (this.params[4] > 0 ? this.params[4] : this.engine.width);
        const h = height ?? (this.params[5] > 0 ? this.params[5] : this.engine.height);
        this.params[4] = w > 0 ? w : 1920;
        this.params[5] = h > 0 ? h : 1080;
        if (this.dirty) {
            this.engine.updateBatchData(this.batch, this.stars, this.stars.length / 4);
            this.dirty = false;
        }
        this.engine.device.queue.writeBuffer(this.paramsBuffer, 0, this.params as Float32Array<ArrayBuffer>);
    }
}