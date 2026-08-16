// src/extras/environment/ReflectionProbes.ts
//
// ReflectionProbes (docs/lighting-probes-design.md §8): local parallax-corrected specular.
// `maxProbes` cubes live in ONE `texture_cube_array` (§3: per-probe cube bindings would not fit
// the group-2 budget), each prefiltered with Phase 1's GGX kernel at `baseArrayLayer =
// layerIndex * 6` (§12.2 — one uniform, no second kernel).
//
// Capture is the volume's pass pointed at this system's band of the shared atlas; the assemble
// stage below gathers the tiles into a staging cube (applying the viewport y-flip and the sky
// substitution) and the prefilter chain turns it into the probe's GGX mip chain. Refresh is
// amortized — the ProbeVolume schedules at most one probe per `reflectionRefreshInterval` frames.
//
// Conservative degradation: an added-but-never-captured probe packs a NEGATIVE layer, the blend
// loop skips it, and its weight goes to the global environment — a standalone ReflectionProbes
// (no volume attached) behaves exactly as Phase 1 (§5).

import { NullGraph, ComputeStageNode } from '../../core';
import { buildPrefilterWGSL } from './envShaders';
import { buildReflAssembleWGSL } from './probeShaders';
import { REFL_PROBE_FLOATS, REFL_PROBE_HEADER_FLOATS, packReflProbe } from './probeMath';
import type { ReflectionProbesConfig, ReflProbeConfig } from './types';

interface ReflProbeRecord {
    center: [number, number, number];
    halfExtent: [number, number, number];
    capture: [number, number, number];
    blend: number;
    radius: number;
    name: string;
    /** The probe's cube-array base layer (layer × 6): its stable identity, assigned at add. */
    layer: number;
    /** False until its first capture completes → skipped by the blend loop (negative packed w). */
    captured: boolean;
    /** Requested for the next scheduled refresh. */
    invalidated: boolean;
    /** Sort key (§8.3: smaller proxy volumes are the more specific statement — they win). */
    proxyVolume: number;
}

/** GPU-side inputs the assemble stage needs, handed over by the ProbeVolume that owns the pass. */
interface CaptureSource {
    colorView: GPUTextureView;
    depthView: GPUTextureView;
    envSpecularView: GPUTextureView;
    envSampler: GPUSampler;
    tileOrigins: [number, number][];   // 6 faces, in atlas pixels
}

export class ReflectionProbes {
    private readonly engine: NullGraph;
    private readonly device: GPUDevice;
    public readonly maxProbes: number;
    public readonly faceSize: number;
    public readonly prefilterMips: number;

    // --- GPU resources, allocated once ---
    private readonly cubeArray: GPUTexture;
    private readonly cubeArrayView: GPUTextureView;        // 'cube-array', all mips — materials sample this
    private readonly staging: GPUTexture;                  // one cube, 1 mip — the prefilter's source
    private readonly stagingCubeView: GPUTextureView;      // 'cube'
    private readonly stagingArrayView: GPUTextureView;     // '2d-array' — the assemble stage's store target
    private readonly dataBuffer: GPUBuffer;                // header + maxProbes × 48 bytes
    private readonly layerParams: GPUBuffer;               // 16 B — base layer offset for the prefilter
    private readonly assembleParams: GPUBuffer;            // 112 B — tile origins + sky intensity
    private readonly sampler: GPUSampler;

    // --- stages ---
    private assembleStage: ComputeStageNode;
    private readonly prefilterStages: ComputeStageNode[] = [];
    private captureSource: CaptureSource | null = null;

    // --- CPU state ---
    private readonly records: ReflProbeRecord[] = [];      // SORTED by proxyVolume ascending
    private readonly stagingF32: Float32Array;
    private readonly stagingU32: Uint32Array;
    private readonly paramsF32 = new Float32Array(28);     // 112 bytes: 24 u32 + 4 f32
    private readonly paramsU32 = new Uint32Array(this.paramsF32.buffer);
    private readonly layerF32 = new Float32Array(4);
    private readonly layerU32 = new Uint32Array(this.layerF32.buffer);
    private refreshCursor = 0;

    constructor(engine: NullGraph, config: ReflectionProbesConfig = {}) {
        this.engine = engine;
        this.device = engine.device;
        this.maxProbes = config.maxProbes ?? 16;
        this.faceSize = config.faceSize ?? 128;
        this.prefilterMips = config.prefilterMips ?? 6;
        const samples = config.prefilterSamples ?? 64;
        const maxMips = Math.log2(this.faceSize) + 1;
        if (this.prefilterMips > maxMips) {
            throw new Error(
                `ReflectionProbes: prefilterMips ${this.prefilterMips} exceeds log2(faceSize)+1 = ${maxMips}.`);
        }

        // The cube array every material samples (group 2, binding 12).
        this.cubeArray = this.device.createTexture({
            label: 'NullGraph_ReflProbes',
            size: [this.faceSize, this.faceSize, this.maxProbes * 6],
            format: 'rgba16float',
            mipLevelCount: this.prefilterMips,
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.cubeArrayView = this.cubeArray.createView({ label: 'NullGraph_ReflProbesView', dimension: 'cube-array' });

        // The capture's staging cube — the prefilter's source. Written by the assemble stage,
        // sampled by Phase 1's GGX kernel; both usages on one allocation, no copy pass.
        this.staging = this.device.createTexture({
            label: 'NullGraph_ReflStaging',
            size: [this.faceSize, this.faceSize, 6],
            format: 'rgba16float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.stagingCubeView = this.staging.createView({ label: 'NullGraph_ReflStagingCube', dimension: 'cube' });
        this.stagingArrayView = this.staging.createView({ label: 'NullGraph_ReflStagingArray', dimension: '2d-array' });

        this.dataBuffer = this.device.createBuffer({
            label: 'NullGraph_ReflProbeData',
            size: (REFL_PROBE_HEADER_FLOATS + this.maxProbes * REFL_PROBE_FLOATS) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.layerParams = this.device.createBuffer({
            label: 'NullGraph_ReflLayerParams', size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.assembleParams = this.device.createBuffer({
            label: 'NullGraph_ReflAssembleParams', size: 112,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.sampler = this.device.createSampler({
            label: 'NullGraph_ReflSampler',
            magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
        });

        this.stagingF32 = new Float32Array(REFL_PROBE_HEADER_FLOATS + this.maxProbes * REFL_PROBE_FLOATS);
        this.stagingU32 = new Uint32Array(this.stagingF32.buffer);
        this.device.queue.writeBuffer(this.dataBuffer, 0, this.stagingF32 as Float32Array<ArrayBuffer>);   // count = 0

        // The prefilter chain: one stage per mip, each writing this probe's 6 layers through the
        // layer-offset uniform (§12.2 — one set of pipelines for every probe).
        for (let m = 0; m < this.prefilterMips; m++) {
            const stage = engine.createComputeStage({
                name: `NullGraph_ReflPrefilter_mip${m}`,
                shaderCode: buildPrefilterWGSL({
                    level: m, prefilterMips: this.prefilterMips, samples,
                    srcSize: this.faceSize, arrayLayerOffset: true,
                }),
                dispatchCount: [
                    Math.max(1, Math.ceil((this.faceSize >> m) / 8)),
                    Math.max(1, Math.ceil((this.faceSize >> m) / 8)),
                    6,
                ],
            });
            stage.enabled = false;
            engine.setComputePassBindGroup(stage, [
                { binding: 0, resource: this.stagingCubeView },
                { binding: 1, resource: this.sampler },
                {
                    binding: 2,
                    resource: this.cubeArray.createView({
                        dimension: '2d-array', baseMipLevel: m, mipLevelCount: 1,
                    }),
                },
                { binding: 3, resource: { buffer: this.layerParams } },
            ], 0);
            this.prefilterStages.push(stage);
        }

        // The assemble stage's bind group is built by bindCaptureSource (it reads the volume's
        // atlas, which does not exist until a ProbeVolume attaches).
        this.assembleStage = engine.createComputeStage({
            name: 'NullGraph_ReflAssemble',
            shaderCode: buildReflAssembleWGSL(),
            dispatchCount: [
                Math.ceil(this.faceSize / 8), Math.ceil(this.faceSize / 8), 6,
            ],
        });
        this.assembleStage.enabled = false;
    }


    // ---------------------------------------------------------------------------
    // Registration (build-time)
    // ---------------------------------------------------------------------------

    /**
     * Register one probe. Returns its LAYER — the stable handle for {@link invalidate}. Records
     * are kept sorted by proxy volume ascending (§8.3: a small probe is a more specific statement
     * about a place, so it claims weight first); layers never move.
     */
    public add(cfg: ReflProbeConfig): number {
        if (this.records.length >= this.maxProbes) {
            throw new Error(
                `ReflectionProbes: probe ${this.records.length} exceeds maxProbes ${this.maxProbes}; ` +
                `the cube array is allocated once at ${this.maxProbes} × 6 layers.`);
        }
        for (let a = 0; a < 3; a++) {
            if (cfg.halfExtent[a] === 0) {
                throw new Error(
                    `ReflectionProbes: probe "${cfg.name ?? 'unnamed'}" has a zero half-extent on axis ${a}; ` +
                    `parallax correction divides by the reflection ray and would produce NaN.`);
            }
        }
        const layer = this.records.length;
        this.records.push({
            center: [...cfg.center],
            halfExtent: [...cfg.halfExtent],
            capture: cfg.capture ? [...cfg.capture] : [...cfg.center],
            blend: cfg.blend ?? 1.0,
            radius: cfg.radius ?? Math.hypot(cfg.halfExtent[0], cfg.halfExtent[1], cfg.halfExtent[2]),
            name: cfg.name ?? `probe-${layer}`,
            layer,
            captured: false,
            invalidated: true,                    // capture me first (§7.4's "never captured" input)
            proxyVolume: 8 * cfg.halfExtent[0] * cfg.halfExtent[1] * cfg.halfExtent[2],
        });
        this.records.sort((a, b) => a.proxyVolume - b.proxyVolume);
        this.uploadRecords();
        return layer;
    }

    /** Sort order may have changed; repack everything (adds are build-time, so this is fine). */
    private uploadRecords(): void {
        this.stagingU32[0] = this.records.length;
        for (let i = 0; i < this.records.length; i++) {
            const r = this.records[i];
            packReflProbe(this.stagingF32, REFL_PROBE_HEADER_FLOATS + i * REFL_PROBE_FLOATS,
                r.center, r.blend, r.halfExtent, r.radius, r.capture, r.layer, r.captured);
        }
        this.device.queue.writeBuffer(this.dataBuffer, 0, this.stagingF32 as Float32Array<ArrayBuffer>);
    }


    /** Repack + re-upload just one record (its captured flag flipped). */
    private uploadRecord(rec: ReflProbeRecord): void {
        const i = this.records.indexOf(rec);
        packReflProbe(this.stagingF32, REFL_PROBE_HEADER_FLOATS + i * REFL_PROBE_FLOATS,
            rec.center, rec.blend, rec.halfExtent, rec.radius, rec.capture, rec.layer, rec.captured);
        this.device.queue.writeBuffer(this.dataBuffer,
            (REFL_PROBE_HEADER_FLOATS + i * REFL_PROBE_FLOATS) * 4,
            this.stagingF32 as Float32Array<ArrayBuffer>, REFL_PROBE_HEADER_FLOATS + i * REFL_PROBE_FLOATS, REFL_PROBE_FLOATS);
    }

    private recordOf(layer: number): ReflProbeRecord {
        const r = this.records.find(r => r.layer === layer);
        if (!r) throw new Error(`ReflectionProbes: no probe with layer ${layer}.`);
        return r;
    }

    /** Mark a probe for recapture (the scene inside its proxy changed). */
    public invalidate(layer: number): void {
        this.recordOf(layer).invalidated = true;
    }

    // ---------------------------------------------------------------------------
    // Capture plumbing — driven by the ProbeVolume
    // ---------------------------------------------------------------------------

    /** The ProbeVolume calls this once when it attaches: where the 6 capture faces land in ITS
     *  atlas. Builds the assemble stage's bind group against those views. */
    public bindCaptureSource(src: CaptureSource): void {
        this.captureSource = src;
        for (let f = 0; f < 6; f++) {
            this.paramsU32[f * 4] = src.tileOrigins[f][0];
            this.paramsU32[f * 4 + 1] = src.tileOrigins[f][1];
        }
        this.engine.setComputePassBindGroup(this.assembleStage, [
            { binding: 0, resource: src.colorView },
            { binding: 1, resource: src.depthView },
            { binding: 2, resource: src.envSpecularView },
            { binding: 3, resource: src.envSampler },
            { binding: 4, resource: { buffer: this.assembleParams } },
            { binding: 5, resource: this.stagingArrayView },
        ], 0);
    }

    /** Which probe to refresh next: invalidated first, else round-robin. -1 = nothing to refresh. */
    public pickRefresh(): number {
        if (this.records.length === 0) return -1;
        for (const r of this.records) if (r.invalidated) return r.layer;
        const rec = this.records[this.refreshCursor % this.records.length];
        this.refreshCursor++;
        return rec.layer;
    }

    /** The world position a refresh of `layer` is captured from (§4.4: ≠ centre in general). */
    public capturePositionOf(layer: number): [number, number, number] {
        return this.recordOf(layer).capture;
    }

    /**
     * Enable the refresh pipeline for `layer` this frame (or disable it with -1): writes the
     * prefilter's base-layer uniform and the assemble stage's sky intensity, and gates every
     * stage — the scheduler's "whatever suppresses the draw suppresses the compute" mirror.
     */
    public primeRefresh(layer: number, skyIntensity: number): void {
        const active = layer >= 0;
        if (active) {
            this.layerU32[0] = this.recordOf(layer).layer * 6;
            this.device.queue.writeBuffer(this.layerParams, 0, this.layerF32);
            this.paramsF32[24] = skyIntensity;
            this.device.queue.writeBuffer(this.assembleParams, 0, this.paramsF32);
        }
        this.assembleStage.enabled = active && this.captureSource !== null;
        for (const s of this.prefilterStages) s.enabled = active;
    }

    /** Call after a refresh has been SCHEDULED: the probe's cube is valid from this frame's
     *  submission onward, so the blend loop may let it in (the packed layer flips positive). */
    public markCaptured(layer: number): void {
        const r = this.recordOf(layer);
        r.captured = true;
        r.invalidated = false;
        this.uploadRecord(r);
    }


    // ---------------------------------------------------------------------------
    // Read-side surface
    // ---------------------------------------------------------------------------

    /** The probe count the blend loop reads (also the header of {@link dataBuffer}). */
    public get probeCount(): number { return this.records.length; }
    /** The materials' cube-array view (group 2, binding 12). */
    public get cubeView(): GPUTextureView { return this.cubeArrayView; }
    /** The sorted `ReflProbe` records (group 2, binding 13). */
    public get probeDataBuffer(): GPUBuffer { return this.dataBuffer; }

    /** The refresh stages — splice AFTER the capture pass and the volume's projection stages
     *  (assemble → GGX mips). Gated by {@link primeRefresh}; all disabled when idle. */
    public getPrefilterStages(): ComputeStageNode[] {
        return [this.assembleStage, ...this.prefilterStages];
    }

    /** Readback-free proxy record access for the volume's `invalidate(bounds)`. */
    public intersects(layer: number, min: ArrayLike<number>, max: ArrayLike<number>): boolean {
        const r = this.recordOf(layer);
        for (let a = 0; a < 3; a++) {
            if (r.center[a] + r.halfExtent[a] < min[a] || r.center[a] - r.halfExtent[a] > max[a]) return false;
        }
        return true;
    }

    /** Every registered record's layer — for a volume invalidating by bounds. */
    public layers(): number[] {
        return this.records.map(r => r.layer);
    }

    public destroy(): void {
        this.cubeArray.destroy();
        this.staging.destroy();
        this.dataBuffer.destroy();
        this.layerParams.destroy();
        this.assembleParams.destroy();
    }
}

