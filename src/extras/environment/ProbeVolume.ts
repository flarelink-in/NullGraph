// src/extras/environment/ProbeVolume.ts
//
// ProbeVolume (docs/lighting-probes-design.md): Phase 2 indirect light — an SH-L1 irradiance
// volume, hardware-trilinearly blended, fed by GPU probe capture amortized across frames.
//
// The capture path is the shadow atlas architecture pointed at a color target (§7.1): six 90°
// `perspectiveZO` views per scheduled probe — `cubeFaceView` from the shadows module — one
// rgba16float atlas of tiles, per-view camera buffers, per-batch viewports, GPU caster culling
// through the VisibilitySystem, and `ComputeStageNode.enabled` gating draw AND cull in the same
// place (design-principles §9). The projection stages turn capture tiles into SH-L1 texels; the
// capture pass shades with the FULL material set, reading the volume as it stood at the start of
// the frame — which is what makes multi-bounce GI fall out for free (§7.3).
//
// The materials never know: `getShaderLayout()` re-emits Phase 1's two contract functions
// against the volume (design-principles §4 — decorate, don't fork).

import { NullGraph, RenderBatch, ComputeStageNode, RenderPassNode } from '../../core';
import type { EnvironmentSystem, ShaderIndirectLayout } from './EnvironmentSystem';
import type { ShadowSystem } from '../shadows/ShadowSystem';
import type { VisibilitySystem } from '../visibility/VisibilitySystem';
import type { LightSystem } from '../lights/LightSystem';
import type { VisibilityMaterial } from '../visibility/types';
import { cubeFaceView, createBuiltView, type BuiltView } from '../shadows/atlasMath';
import { buildProbeIndirectLayout, buildProbeProjectionWGSL } from './probeShaders';
import {
    PROBE_PARAMS_FLOATS, packProbeParams, validateProbeVolumeConfig,
    createProbeScheduler, scheduleProbes, probePosition, probeCoord,
    ENCLOSED_DIST_SPACING,
    type ProbeScheduler,
} from './probeMath';
import { ReflectionProbes } from './ReflectionProbes';
import type { ProbeVolumeConfig } from './types';

/** One capture slot: the 6 cube-face views of ONE probe scheduled this frame (§7.1). */
interface CaptureSlot {
    /** Per-face tile rects in atlas pixels — fixed at construction (uniform tiles need no allocator). */
    tileRects: [number, number, number, number][];
    /** One 432-byte camera buffer per face — "a probe cube face is a Camera" (§3). */
    cameraBuffers: GPUBuffer[];
    /** Per-face bake staging for cubeFaceView. */
    views: BuiltView[];
    /** [face][mesh] indirect draw batches. */
    batches: RenderBatch[][];
    /** [face][mesh] cull stages feeding the batches — gated with the viewports, same loop (§9). */
    cullStages: (ComputeStageNode | null)[][];
}

interface CaptureMeshRecord {
    meshName: string;
    meshIndex: number;
    material: VisibilityMaterial;
    lights: LightSystem;
}

type RequiredProbeConfig = Required<Omit<ProbeVolumeConfig, 'environment' | 'visibility' | 'shadows'>>;

/** The viewport that rasterizes nothing — the shadow atlas's suppression idiom (§8.3 there). */
const ZERO_VIEWPORT: [number, number, number, number] = [0, 0, 0, 0];
const EMPTY_STAGES: ComputeStageNode[] = [];

export class ProbeVolume {
    private readonly engine: NullGraph;
    private readonly device: GPUDevice;
    private readonly cfg: RequiredProbeConfig;
    private readonly env: EnvironmentSystem;
    private readonly vis: VisibilitySystem;
    private readonly shadows: ShadowSystem | null;
    public readonly probeCount: number;

    // --- the volume (§4.2) ---
    private readonly probeRTexture: GPUTexture;
    private readonly probeGTexture: GPUTexture;
    private readonly probeBTexture: GPUTexture;
    private readonly validityTexture: GPUTexture;
    private readonly probeRView: GPUTextureView;
    private readonly probeGView: GPUTextureView;
    private readonly probeBView: GPUTextureView;
    private readonly validityView: GPUTextureView;
    private readonly probeParamsBuffer: GPUBuffer;


    // --- capture (§7.1) ---
    private readonly captureAtlas: GPUTexture;
    private readonly captureColorView: GPUTextureView;      // render attachment
    private readonly captureColorSampleView: GPUTextureView;
    private readonly captureDepth: GPUTexture;
    private readonly captureDepthView: GPUTextureView;      // render attachment
    private readonly captureDepthSampleView: GPUTextureView;
    private readonly capturePass: RenderPassNode;
    private readonly slots: CaptureSlot[] = [];
    private readonly projStages: ComputeStageNode[] = [];
    private readonly projParamsBuffers: GPUBuffer[] = [];

    // --- scheduler (§7.4) ---
    private readonly scheduler: ProbeScheduler;
    private readonly probePositions: Float32Array;          // 3 per probe
    private readonly invDistRef: number;
    private frame: number = 0;
    private scheduledCount: number = 0;
    private readonly lastCamPos: [number, number, number];

    // --- capture meshes / reflection band ---
    private readonly captureMeshes: CaptureMeshRecord[] = [];
    private readonly captureCullStages: ComputeStageNode[] = [];
    private refl: ReflectionProbes | null = null;
    private reflTileRects: [number, number, number, number][] = [];
    private reflCameraBuffers: GPUBuffer[] = [];
    private reflViews: BuiltView[] = [];
    private reflBatches: RenderBatch[][] = [];
    private reflCullStages: (ComputeStageNode | null)[][] = [];

    // --- placeholders for bindings 12/13 when no ReflectionProbes is attached ---
    private readonly placeholderCube: GPUTexture;
    private readonly placeholderCubeView: GPUTextureView;
    private readonly placeholderDataBuffer: GPUBuffer;

    // --- the contract + group-2 tracking (same pattern as EnvironmentSystem) ---
    private layout: ShaderIndirectLayout | null = null;
    private readonly boundBatches: Set<RenderBatch> = new Set();

    // --- scratch (persistent; the frame path allocates nothing) ---
    private readonly projParamsF32 = new Float32Array(32);     // 128 bytes
    private readonly projParamsU32 = new Uint32Array(this.projParamsF32.buffer);
    private readonly _coord: [number, number, number] = [0, 0, 0];
    private readonly _pos: [number, number, number] = [0, 0, 0];

    constructor(engine: NullGraph, config: ProbeVolumeConfig) {
        this.engine = engine;
        this.device = engine.device;
        const span = [
            config.spacing * (config.dims[0] - 1),
            config.spacing * (config.dims[1] - 1),
            config.spacing * (config.dims[2] - 1),
        ];
        const diagonal = Math.hypot(span[0], span[1], span[2]);
        this.cfg = {
            origin: config.origin,
            spacing: config.spacing,
            dims: config.dims,
            captureFaceSize: config.captureFaceSize ?? 32,
            probesPerFrame: config.probesPerFrame ?? 16,
            captureAtlasSize: config.captureAtlasSize ?? 1024,
            normalBiasTexels: config.normalBiasTexels ?? 0.5,
            borderFade: config.borderFade ?? 4,
            intensity: config.intensity ?? 1.0,
            maxProbes: config.maxProbes ?? 65536,
            captureNear: config.captureNear ?? config.spacing * 0.05,
            captureFar: config.captureFar ?? 2 * diagonal,
            distanceBias: config.distanceBias ?? 16,
            shBands: config.shBands ?? 1,
            allowShadowlessCapture: config.allowShadowlessCapture ?? false,
            reflectionRefreshInterval: config.reflectionRefreshInterval ?? 2,
            reflectionFaceSize: config.reflectionFaceSize ?? 128,
        };
        validateProbeVolumeConfig({ ...this.cfg, ...config });
        this.env = config.environment;
        this.vis = config.visibility;
        this.shadows = config.shadows ?? null;
        this.probeCount = this.cfg.dims[0] * this.cfg.dims[1] * this.cfg.dims[2];
        this.invDistRef = 4 / diagonal;   // the near weight fades out at ¼ of the diagonal
        this.lastCamPos = [
            this.cfg.origin[0] + span[0] / 2,
            this.cfg.origin[1] + span[1] / 2,
            this.cfg.origin[2] + span[2] / 2,
        ];

        // --- the volume textures (§4.2): SH-L1, one channel per 3D texture + validity ---
        const mkVolume = (format: 'rgba16float' | 'rgba8unorm', label: string) => this.device.createTexture({
            label,
            size: [this.cfg.dims[0], this.cfg.dims[1], this.cfg.dims[2]],
            dimension: '3d',
            format,
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.probeRTexture = mkVolume('rgba16float', 'NullGraph_ProbeR');
        this.probeGTexture = mkVolume('rgba16float', 'NullGraph_ProbeG');
        this.probeBTexture = mkVolume('rgba16float', 'NullGraph_ProbeB');
        this.validityTexture = mkVolume('rgba8unorm', 'NullGraph_ProbeValidity');
        this.probeRView = this.probeRTexture.createView();
        this.probeGView = this.probeGTexture.createView();
        this.probeBView = this.probeBTexture.createView();
        this.validityView = this.validityTexture.createView();

        // Conservative degradation starts HERE (§5 principle): a probe that was never captured
        // reads validity 0 → the global environment, never black. And the SH textures are zeroed
        // too, because mix(env, NaN, 0) is still NaN — uninitialized storage is not an option.
        const [dx, dy, dz] = this.cfg.dims;
        const q = this.device.queue;
        const bprSH = Math.ceil((dx * 8) / 256) * 256;    // writeTexture rows are 256-aligned
        q.writeTexture({ texture: this.probeRTexture }, new Uint8Array(bprSH * dy * dz),
            { bytesPerRow: bprSH, rowsPerImage: dy }, [dx, dy, dz]);
        q.writeTexture({ texture: this.probeGTexture }, new Uint8Array(bprSH * dy * dz),
            { bytesPerRow: bprSH, rowsPerImage: dy }, [dx, dy, dz]);
        q.writeTexture({ texture: this.probeBTexture }, new Uint8Array(bprSH * dy * dz),
            { bytesPerRow: bprSH, rowsPerImage: dy }, [dx, dy, dz]);
        const bprV = Math.ceil((dx * 4) / 256) * 256;
        q.writeTexture({ texture: this.validityTexture }, new Uint8Array(bprV * dy * dz),
            { bytesPerRow: bprV, rowsPerImage: dy }, [dx, dy, dz]);

        this.probeParamsBuffer = this.device.createBuffer({
            label: 'NullGraph_ProbeParams', size: PROBE_PARAMS_FLOATS * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        q.writeBuffer(this.probeParamsBuffer, 0, packProbeParams(this.cfg) as Float32Array<ArrayBuffer>);

        // --- the capture target: the shadow atlas, in color (§7.1) ---
        this.captureAtlas = this.device.createTexture({
            label: 'NullGraph_ProbeCaptureAtlas',
            size: [this.cfg.captureAtlasSize, this.cfg.captureAtlasSize],
            format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.captureColorView = this.captureAtlas.createView();
        this.captureColorSampleView = this.captureColorView;
        this.captureDepth = this.device.createTexture({
            label: 'NullGraph_ProbeCaptureDepth',
            size: [this.cfg.captureAtlasSize, this.cfg.captureAtlasSize],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.captureDepthView = this.captureDepth.createView();
        this.captureDepthSampleView = this.captureDepth.createView({ aspect: 'depth-only' });


        this.capturePass = engine.createPass({
            name: 'NullGraph_ProbeCapture',
            isMainScreenPass: false,
            colorAttachments: [{
                view: this.captureColorView,
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
            depthStencilAttachment: {
                view: this.captureDepthView,
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',   // the projection stage reads it for validity (§7.2)
            },
        });

        // --- capture slots: probesPerFrame × 6 faces, tiles fixed by a static grid (§7.1:
        //  uniform-size tiles need no allocator; the shadow atlas's quad-tree solves a problem
        //  this pass does not have) ---
        const F = this.cfg.captureFaceSize;
        const tilesPerRow = Math.floor(this.cfg.captureAtlasSize / F);
        for (let s = 0; s < this.cfg.probesPerFrame; s++) {
            const slot: CaptureSlot = { tileRects: [], cameraBuffers: [], views: [], batches: [], cullStages: [] };
            for (let f = 0; f < 6; f++) {
                const tileIndex = s * 6 + f;
                slot.tileRects.push([
                    (tileIndex % tilesPerRow) * F,
                    Math.floor(tileIndex / tilesPerRow) * F,
                    F, F,
                ]);
                slot.cameraBuffers.push(this.device.createBuffer({
                    label: `NullGraph_ProbeCam_s${s}f${f}`, size: 432,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                }));
                slot.views.push(createBuiltView());
                slot.batches.push([]);
                slot.cullStages.push([]);
            }
            this.slots.push(slot);
        }

        // --- projection stages: one per slot, gated by the scheduler (§7.2) ---
        for (let s = 0; s < this.cfg.probesPerFrame; s++) {
            const stage = engine.createComputeStage({
                name: `NullGraph_ProbeProject_${s}`,
                shaderCode: buildProbeProjectionWGSL({ faceSize: F }),
                dispatchCount: [1, 1, 1],
            });
            stage.enabled = false;
            const params = this.device.createBuffer({
                label: `NullGraph_ProbeProjParams_${s}`, size: 128,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            engine.setComputePassBindGroup(stage, [
                { binding: 0, resource: this.captureColorSampleView },
                { binding: 1, resource: this.captureDepthSampleView },
                { binding: 2, resource: this.env.specularView },
                { binding: 3, resource: this.env.sampler },
                { binding: 4, resource: { buffer: params } },
                { binding: 5, resource: this.probeRView },
                { binding: 6, resource: this.probeGView },
                { binding: 7, resource: this.probeBView },
                { binding: 8, resource: this.validityView },
            ], 0);
            this.projStages.push(stage);
            this.projParamsBuffers.push(params);
        }

        // --- scheduler + probe world positions (§7.4) ---
        this.scheduler = createProbeScheduler(this.probeCount, this.cfg.probesPerFrame);
        this.probePositions = new Float32Array(this.probeCount * 3);
        for (let i = 0; i < this.probeCount; i++) {
            probeCoord(this.cfg.dims, i, this._coord);
            probePosition(this.cfg.origin, this.cfg.spacing, this._coord[0], this._coord[1], this._coord[2], this._pos);
            this.probePositions[i * 3] = this._pos[0];
            this.probePositions[i * 3 + 1] = this._pos[1];
            this.probePositions[i * 3 + 2] = this._pos[2];
        }

        // --- placeholders keep bindings 12/13 valid when no ReflectionProbes is attached ---
        this.placeholderCube = this.device.createTexture({
            label: 'NullGraph_ReflPlaceholder',
            size: [1, 1, 6], format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING,
        });
        this.placeholderCubeView = this.placeholderCube.createView({ dimension: 'cube-array' });
        this.placeholderDataBuffer = this.device.createBuffer({
            label: 'NullGraph_ReflDataPlaceholder', size: 64,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        q.writeBuffer(this.placeholderDataBuffer, 0, new Float32Array(16));   // count = 0
    }


    // ---------------------------------------------------------------------------
    // The contract (§9) — getShaderLayout + the ONE combined group-2 bind group
    // ---------------------------------------------------------------------------

    /**
     * The Phase 2 `ShaderIndirectLayout` — the SAME two functions Phase 1 emits, reimplemented
     * against the volume. `_refl` is accepted for API parity (§9): the layout is identical with
     * or without reflection probes (bindings 12/13 are always declared; placeholders are bound
     * when absent), so passing the system early or late changes nothing.
     */
    public getShaderLayout(_refl?: ReflectionProbes | null): ShaderIndirectLayout {
        if (!this.layout) {
            this.layout = buildProbeIndirectLayout(this.env.getShaderLayout(), 2);
        }
        return this.layout;
    }

    /**
     * THE COMBINED group-2 bind group: lights (0–2) + environment (3–6) + probes (7–13).
     * Escalates Phase 1's argument (§9): group 2 now has THREE owners and exactly one of them
     * may build the bind group. `env` defaults to config.environment, `refl` to the attached
     * ReflectionProbes (or placeholders).
     */
    public applyToBatch(
        batch: RenderBatch, lights: LightSystem,
        env: EnvironmentSystem = this.env, refl: ReflectionProbes | null = this.refl,
    ): void {
        const g = 2;
        if (batch.customBindGroups[g] && !this.boundBatches.has(batch)) {
            throw new Error(
                `ProbeVolume: batch already has a group-2 bind group; group 2 has three owners ` +
                `(lights, environment, probes) and only ProbeVolume.applyToBatch(batch, lights, env, refl) ` +
                `may build it.`);
        }
        this.engine.setRenderPassBindGroup(batch, [
            ...lights.getBindGroupEntries(),
            { binding: 3, resource: env.sampler },
            { binding: 4, resource: env.specularView },
            { binding: 5, resource: env.brdfView },
            { binding: 6, resource: { buffer: env.paramsBuffer } },
            { binding: 7, resource: this.probeRView },
            { binding: 8, resource: this.probeGView },
            { binding: 9, resource: this.probeBView },
            { binding: 10, resource: this.validityView },
            { binding: 11, resource: { buffer: this.probeParamsBuffer } },
            { binding: 12, resource: refl ? refl.cubeView : this.placeholderCubeView },
            { binding: 13, resource: { buffer: refl ? refl.probeDataBuffer : this.placeholderDataBuffer } },
        ], g);
        this.boundBatches.add(batch);
    }

    // ---------------------------------------------------------------------------
    // Reflection probes (§8)
    // ---------------------------------------------------------------------------

    /**
     * Attach a ReflectionProbes: reserves the atlas band (its face size must match
     * `reflectionFaceSize`), hands it the capture source, and creates the 6-view capture slots.
     * Capture batches for meshes registered SO FAR are created; meshes registered LATER get
     * them automatically (registration is idempotent, so order does not matter — the same rule
     * the shadow atlas's slot machinery follows, design-principles §10).
     */
    public setReflectionProbes(refl: ReflectionProbes): void {
        if (refl.faceSize !== this.cfg.reflectionFaceSize) {
            throw new Error(
                `ProbeVolume: reflection face size ${refl.faceSize} does not match the reserved atlas ` +
                `band of ${this.cfg.reflectionFaceSize}px (config.reflectionFaceSize).`);
        }
        this.refl = refl;
        const RF = refl.faceSize;
        const tilesPerRow = Math.floor(this.cfg.captureAtlasSize / RF);
        const bandRows = Math.ceil(6 / tilesPerRow);
        const bandTop = this.cfg.captureAtlasSize - bandRows * RF;   // reflection band: top of the atlas
        for (let f = 0; f < 6; f++) {
            this.reflTileRects.push([(f % tilesPerRow) * RF, bandTop + Math.floor(f / tilesPerRow) * RF, RF, RF]);
            this.reflCameraBuffers.push(this.device.createBuffer({
                label: `NullGraph_ReflCam_f${f}`, size: 432,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }));
            this.reflViews.push(createBuiltView());
            this.reflBatches.push([]);
            this.reflCullStages.push([]);
        }
        refl.bindCaptureSource({
            colorView: this.captureColorSampleView,
            depthView: this.captureDepthSampleView,
            envSpecularView: this.env.specularView,
            envSampler: this.env.sampler,
            tileOrigins: this.reflTileRects.map(r => [r[0], r[1]] as [number, number]),
        });
        for (const m of this.captureMeshes) this.createReflBatches(m);
    }


    // ---------------------------------------------------------------------------
    // Capture meshes (§7.1) — "batches = views × casterMeshes"
    // ---------------------------------------------------------------------------

    /**
     * Register one mesh for capture: `probesPerFrame × 6` indirect batches (plus 6 reflection
     * batches when a ReflectionProbes is attached), all in the single capture pass, GPU-culled
     * per view. The material shades the scene's FULL lit response — captures are lit (direct +
     * shadowed + the previous frame's volume), which is what multi-bounce is made of (§7.3).
     * `material.indirect` is auto-filled with THIS volume's layout when omitted.
     */
    public registerCaptureMesh(
        meshName: string, meshIndex: number,
        material: VisibilityMaterial, lights: LightSystem,
    ): void {
        if (!material.lighting) {
            throw new Error(
                `ProbeVolume: capture mesh "${meshName}" has no material.lighting — the capture shades ` +
                `the scene's lit response, so pass lighting: shadows.wrapLighting(lights) ` +
                `(captures must see shadows, §9).`);
        }
        const m: VisibilityMaterial = { ...material, indirect: material.indirect ?? this.getShaderLayout() };
        const record: CaptureMeshRecord = { meshName, meshIndex, material: m, lights };
        this.captureMeshes.push(record);

        const F = this.cfg.captureFaceSize;
        for (const slot of this.slots) {
            for (let f = 0; f < 6; f++) {
                const batch = this.vis.createInstanceDrawBatch(this.capturePass, meshName, meshIndex, {
                    material: m,
                    camera: slot.cameraBuffers[f],
                    occlusion: false,                          // §7.1: culling is frustum-only
                    viewport: slot.tileRects[f],
                    screenSize: [F, F],
                    targetFormat: 'rgba16float',
                });
                slot.batches[f].push(batch);
                const cull = this.vis.getCullStage(batch);
                slot.cullStages[f].push(cull);
                if (cull) this.captureCullStages.push(cull);
                // A suppressed tile suppresses its draw AND its cull in one place (design-§9):
                // slots start idle, so start them suppressed.
                batch.viewport = ZERO_VIEWPORT;
                if (cull) cull.enabled = false;
                this.applyToBatch(batch, lights);
                if (this.shadows) this.shadows.applyToBatch(batch);
            }
        }
        if (this.refl) this.createReflBatches(record);
    }

    /** The 6 reflection-band batches for one mesh (called from registerCaptureMesh and
     *  setReflectionProbes — registration is idempotent). */
    private createReflBatches(record: CaptureMeshRecord): void {
        const refl = this.refl!;
        const RF = refl.faceSize;
        for (let f = 0; f < 6; f++) {
            const batch = this.vis.createInstanceDrawBatch(this.capturePass, record.meshName, record.meshIndex, {
                material: record.material,
                camera: this.reflCameraBuffers[f],
                occlusion: false,
                viewport: this.reflTileRects[f],
                screenSize: [RF, RF],
                targetFormat: 'rgba16float',
            });
            this.reflBatches[f].push(batch);
            const cull = this.vis.getCullStage(batch);
            this.reflCullStages[f].push(cull);
            if (cull) this.captureCullStages.push(cull);
            batch.viewport = ZERO_VIEWPORT;
            if (cull) cull.enabled = false;
            this.applyToBatch(batch, record.lights);
            if (this.shadows) this.shadows.applyToBatch(batch);
        }
    }


    // ---------------------------------------------------------------------------
    // Per-frame (§7.3/§7.4): schedule → bake views → gate draws+culls+projections
    // ---------------------------------------------------------------------------

    /**
     * Schedule this frame's probes and bake their capture views. Call ONCE per frame, after
     * `vis.update()`, before `engine.render`. No allocation (constraint #2).
     */
    public update(cameraPos: ArrayLike<number>): void {
        this.frame++;
        this.lastCamPos[0] = cameraPos[0];
        this.lastCamPos[1] = cameraPos[1];
        this.lastCamPos[2] = cameraPos[2];
        const F = this.cfg.captureFaceSize;
        const q = this.device.queue;

        const n = scheduleProbes(this.scheduler, this.probePositions, this.probeCount,
            cameraPos, this.frame, this.cfg.distanceBias, this.invDistRef);
        this.scheduledCount = n;

        for (let s = 0; s < this.slots.length; s++) {
            const slot = this.slots[s];
            const active = s < n;
            if (active) {
                const idx = this.scheduler.picks[s];
                probeCoord(this.cfg.dims, idx, this._coord);
                probePosition(this.cfg.origin, this.cfg.spacing,
                    this._coord[0], this._coord[1], this._coord[2], this._pos);
                for (let f = 0; f < 6; f++) {
                    const v = cubeFaceView(this._pos, f, this.cfg.captureFar, this.cfg.captureNear,
                        F, 0, 0, slot.views[f]);
                    // One 432-byte Camera block per face — "a probe cube face is a Camera" (§3).
                    q.writeBuffer(slot.cameraBuffers[f], 0, v.viewData.buffer, v.viewData.byteOffset, 432);
                }
                const u = this.projParamsU32, p = this.projParamsF32;
                for (let f = 0; f < 6; f++) {
                    u[f * 4] = slot.tileRects[f][0];
                    u[f * 4 + 1] = slot.tileRects[f][1];
                }
                u[24] = this._coord[0]; u[25] = this._coord[1]; u[26] = this._coord[2]; u[27] = F;
                p[28] = this.cfg.captureNear;
                p[29] = this.cfg.captureFar;
                p[30] = this.cfg.spacing * ENCLOSED_DIST_SPACING;
                p[31] = this.env.isBaked ? this.env.intensity : 0;
                q.writeBuffer(this.projParamsBuffers[s], 0, p as Float32Array<ArrayBuffer>, 0, 32);
            }
            for (let f = 0; f < 6; f++) {
                for (let mi = 0; mi < slot.batches[f].length; mi++) {
                    slot.batches[f][mi].viewport = active ? slot.tileRects[f] : ZERO_VIEWPORT;
                    const cull = slot.cullStages[f][mi];
                    if (cull) cull.enabled = active;
                }
            }
            this.projStages[s].enabled = active;
        }

        this.updateReflectionBand();
    }

    /** One reflection refresh per `reflectionRefreshInterval` frames, plus its 6 face views. */
    private updateReflectionBand(): void {
        const refl = this.refl;
        if (!refl) return;
        let layer = -1;
        if (this.frame % this.cfg.reflectionRefreshInterval === 0) layer = refl.pickRefresh();
        if (layer >= 0) {
            const pos = refl.capturePositionOf(layer);
            for (let f = 0; f < 6; f++) {
                const v = cubeFaceView(pos, f, this.cfg.captureFar, this.cfg.captureNear,
                    refl.faceSize, 0, 0, this.reflViews[f]);
                this.device.queue.writeBuffer(this.reflCameraBuffers[f], 0,
                    v.viewData.buffer, v.viewData.byteOffset, 432);
            }
            refl.markCaptured(layer);
        }
        refl.primeRefresh(layer, this.env.isBaked ? this.env.intensity : 0);
        const active = layer >= 0;
        for (let f = 0; f < 6; f++) {
            for (let mi = 0; mi < this.reflBatches[f].length; mi++) {
                this.reflBatches[f][mi].viewport = active ? this.reflTileRects[f] : ZERO_VIEWPORT;
                const cull = this.reflCullStages[f][mi];
                if (cull) cull.enabled = active;
            }
        }
    }


    // ---------------------------------------------------------------------------
    // Invalidation + prewarm (§7.4)
    // ---------------------------------------------------------------------------

    /** How many probes still need a (re)capture — drives `prewarm`. */
    public warmPending(): number {
        let n = 0;
        for (let i = 0; i < this.probeCount; i++) {
            if (this.scheduler.lastCaptureFrame[i] < 0 || this.scheduler.invalidated[i] !== 0) n++;
        }
        return n;
    }

    /**
     * Dynamically update GI volume intensity multiplier (0.0 disables indirect GI volume).
     */
    public setIntensity(intensity: number): void {
        this.cfg.intensity = intensity;
        this.device.queue.writeBuffer(this.probeParamsBuffer, 0, packProbeParams(this.cfg) as Float32Array<ArrayBuffer>);
    }

    /**
     * The caller moved a light or opened a door (§7.4): mark every probe inside `bounds` for
     * priority recapture, plus every reflection probe whose proxy box intersects.
     */
    public invalidate(bounds: { min: ArrayLike<number>; max: ArrayLike<number> }): void {
        const { origin, spacing, dims } = this.cfg;
        const volMin = origin;
        const volMax = [origin[0] + spacing * (dims[0] - 1), origin[1] + spacing * (dims[1] - 1), origin[2] + spacing * (dims[2] - 1)];
        for (let a = 0; a < 3; a++) {
            if (bounds.max[a] < volMin[a] || bounds.min[a] > volMax[a]) {
                throw new Error(
                    `ProbeVolume: invalidate() bounds do not intersect the volume; nothing was marked dirty.`);
            }
        }
        for (let i = 0; i < this.probeCount; i++) {
            const x = this.probePositions[i * 3], y = this.probePositions[i * 3 + 1], z = this.probePositions[i * 3 + 2];
            if (x >= bounds.min[0] && x <= bounds.max[0] &&
                y >= bounds.min[1] && y <= bounds.max[1] &&
                z >= bounds.min[2] && z <= bounds.max[2]) {
                this.scheduler.invalidated[i] = 1;
            }
        }
        if (this.refl) {
            for (const layer of this.refl.layers()) {
                if (this.refl.intersects(layer, bounds.min, bounds.max)) this.refl.invalidate(layer);
            }
        }
    }

    /**
     * Fill the whole volume (and any attached reflection probes) synchronously over as many
     * internal frames as it takes (§7.4) — for editors and load screens. Drives its own minimal
     * pipeline (the capture culls + the capture pass + the projections), so call it AFTER the
     * visibility system has instance data (`vis.update()` at least once). `onProgress(done, total)`
     * is called once per internal frame so a loading bar is possible. Each iteration shades
     * against the previous iteration's volume, so prewarm also converges several bounce steps.
     */
    public async prewarm(onProgress?: (done: number, total: number) => void): Promise<void> {
        const total = this.probeCount;
        // The load path may allocate this scratch per iteration — constraint #2 guards the frame loop.
        while (this.warmPending() > 0) {
            this.update(this.lastCamPos);
            const seq: (ComputeStageNode | RenderPassNode)[] = [];
            for (const s of this.captureCullStages) seq.push(s);
            seq.push(this.capturePass);
            for (const s of this.projStages) seq.push(s);
            if (this.refl) for (const s of this.refl.getPrefilterStages()) seq.push(s);
            this.engine.render(seq);
            onProgress?.(total - this.warmPending(), total);
            await this.device.queue.onSubmittedWorkDone();
        }
    }

    // ---------------------------------------------------------------------------
    // Frame hand-off (§2)
    // ---------------------------------------------------------------------------

    /**
     * Capture caster culls are VisibilitySystem stages and arrive via `vis.getComputeStages()`,
     * exactly like the shadow atlas's caster culls — splicing them in twice would double-dispatch
     * the same compaction. Design-principles §9 is still honoured: ProbeVolume gates them (with
     * the viewports) through `getCullStage(batch)`. This getter therefore returns an empty array;
     * it exists so the §2 pipeline shape reads the same on every subsystem.
     */
    public getCaptureStages(): ComputeStageNode[] {
        return EMPTY_STAGES;
    }

    /** THE capture pass (§7.1): every active slot's tiles + the reflection band, one submission. */
    public getCapturePass(): RenderPassNode {
        return this.capturePass;
    }

    /** The projection stages — gated per slot by `update()`. Splice AFTER the capture pass. */
    public getProjectStages(): ComputeStageNode[] {
        return this.projStages;
    }

    // ---------------------------------------------------------------------------
    // Introspection
    // ---------------------------------------------------------------------------

    /** How many probes were scheduled by the most recent `update()` (≤ probesPerFrame). */
    public get scheduledThisFrame(): number { return this.scheduledCount; }
    /** The pick list of the most recent `update()` (first {@link scheduledThisFrame} entries). */
    public get scheduledPicks(): Int32Array { return this.scheduler.picks; }
    /** probesPerFrame × 6 capture slots — the capture cost cap (T6). */
    public get captureSlotCount(): number { return this.slots.length * 6; }

    public destroy(): void {
        this.probeRTexture.destroy();
        this.probeGTexture.destroy();
        this.probeBTexture.destroy();
        this.validityTexture.destroy();
        this.probeParamsBuffer.destroy();
        this.captureAtlas.destroy();
        this.captureDepth.destroy();
        for (const slot of this.slots) for (const b of slot.cameraBuffers) b.destroy();
        for (const b of this.projParamsBuffers) b.destroy();
        for (const b of this.reflCameraBuffers) b.destroy();
        this.placeholderCube.destroy();
        this.placeholderDataBuffer.destroy();
    }
}

