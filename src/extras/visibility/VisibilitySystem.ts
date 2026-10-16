// src/extras/visibility/VisibilitySystem.ts
//
// The VisibilitySystem (docs/visibility-design.md): GPU-driven culling that decides, on the GPU,
// what to draw — and issues the draws indirectly so the CPU never iterates the scene. It is a
// batteries-included wrapper over the engine's existing indirect-draw primitive
// (`createBatch({ isIndirect: true })` + `drawIndexedIndirect`, core/BatchManager.ts:60-70,
// core/RenderPipeline.ts:107-110), adding correct frustum extraction, Hi-Z occlusion, meshlet
// cone culling, and LOD selection on top.
//
// Two tiers share one substrate (the Hi-Z pyramid, the camera, the indirect-draw mechanism):
//   - instance tier (§7.1): one indirect batch per unique mesh, coarse per-object culling.
//   - meshlet tier (§7.2): one indirect batch per material, fine per-cluster culling + LOD.
//
// Per frame the CPU only uploads instance data, clears a few counters, and splices
// `getComputeStages()` ahead of the scene pass — no per-frame allocation, no scene-graph walk.

import { NullGraph, RenderBatch, ComputeStageNode, RenderPassNode } from '../../core';
import { StandardLayout } from '../geometry/VertexLayout';
import type { MegabufferBuilder } from '../geometry/MegaBufferBuilder';
import type { MeshletPack } from '../geometry/MeshletBuilder';
import { HiZPyramid } from './HiZPyramid';
import {
    buildInstanceCullWGSL,
    buildInstanceScatterWGSL,
    INSTANCE_CULL_GROUP,
    buildMeshletInstanceCullWGSL,
    buildMeshletCullWGSL,
    buildInstanceDrawWGSL,
    buildMeshletDrawWGSL,
} from './visibilityShaders';
import { INSTANCE_FLOATS } from './cullMath';
import { buildMaterialFragment } from '../materials/materialFragments';
import type { VisibilityConfig, InstanceDrawOptions, MeshletDrawOptions, VisibilityMaterial } from './types';

/** Triangles in one meshlet × 3 corners = the meshlet draw's static index count (124 × 3). */
const MESHLET_DRAW_INDEX_COUNT = 124 * 3;

interface CullStageRecord {
    /**
     * In order. The instance tier: cull, then scatter (design doc §14: survivors in source order).
     * The meshlet tier: its one stage, still compacting with atomicAdd (§14.4, phase 2).
     */
    stages: ComputeStageNode[];
    paramsBuffer: GPUBuffer;
    /** The instance tier's scratch: visibility bits and block counts. */
    scratch?: GPUBuffer[];
    /** meshIndex filter for instance-tier culls; unused (0) for meshlet-tier stages. */
    meshIndex: number;
    /** The draw batch these stages feed — the key for {@link VisibilitySystem.getCullStages}. */
    batch?: RenderBatch;
    /** The pass `batch` was created into — what {@link VisibilitySystem.removeDrawBatch} detaches it from. */
    pass?: RenderPassNode;
    /**
     * Hi-Z UV remap for this stage (shadows-atlas-design.md §7.2). Default `[1,1,0,0]` — the
     * identity, which leaves the main view bit-for-bit unchanged (§10 U5).
     */
    hizRect?: [number, number, number, number];
    /**
     * Screen size for the Hi-Z footprint/LOD math (§7.2: a shadow view uses its tile size).
     * Default: the Hi-Z source size, then the engine canvas.
     */
    screenSize?: [number, number];
}

export class VisibilitySystem {
    private readonly engine: NullGraph;
    private readonly config: VisibilityConfig;
    private readonly device: GPUDevice;
    private readonly occlusion: boolean;
    private readonly lodErrorPx: number;

    // --- shared inputs ---
    /**
     * The instance records on the GPU (`INSTANCE_FLOATS` per instance, STORAGE). Public read-only
     * so other GPU consumers can read the same data with no copy — labels following instances
     * (`TextSystem` `follow: { buffer, strideFloats: 24, positionOffset: 12 }`), a simulation's
     * compute stage. Written by `upload()` from the `setInstances()` array — or, after
     * `setInstanceCount()`, only by the GPU and `queue.writeBuffer`.
     */
    public readonly instanceBuffer: GPUBuffer;
    private instanceStaging: Float32Array | null = null;
    private instanceCount: number = 0;
    private gpuRows = false;

    // --- geometry (setGeometry) ---
    private mega: MegabufferBuilder | null = null;
    private megaVertexBuffer: GPUBuffer | null = null;
    private megaIndexBuffer: GPUBuffer | null = null;
    private megaIndexFormat: GPUIndexFormat = 'uint16';
    private meshletBuffers: {
        meshlets: GPUBuffer;
        meshletVertices: GPUBuffer;
        meshletTriangles: GPUBuffer;
        meshLODs: GPUBuffer;
        meshTable: GPUBuffer;
    } | null = null;
    private megaStrideFloats: number = 0;

    // --- Hi-Z ---
    private hiZ: HiZPyramid | null = null;

    // --- meshlet-tier shared work buffers ---
    private visibleInstancesBuffer: GPUBuffer | null = null;
    private countersBuffer: GPUBuffer | null = null;
    private meshletInstanceCull: CullStageRecord | null = null;
    private staticIndexBuffer: GPUBuffer | null = null;

    // --- registered cull stages (built lazily by getComputeStages) ---
    private instanceTierCulls: CullStageRecord[] = [];
    private meshletTierCulls: CullStageRecord[] = [];
    private meshletParamsBuffer: GPUBuffer | null = null;
    private computeChain: ComputeStageNode[] | null = null;

    // params staging: 12 slots → 48 bytes (shadows-atlas-design.md §7.2 grew CullParams by
    // hizRect). See CULL_PARAMS_WGSL.
    private readonly paramsFloat = new Float32Array(12);
    private readonly paramsU32 = new Uint32Array(this.paramsFloat.buffer);

    constructor(engine: NullGraph, config: VisibilityConfig) {
        this.engine = engine;
        this.device = engine.device;
        this.config = config;

        if (config.mode === 'two-phase') {
            // Reserved (design doc §9, build order §13 step 7). Fail loud rather than ship an
            // untested second cull+draw — single-phase is the proven baseline.
            throw new Error('VisibilitySystem: mode "two-phase" is reserved for a later build; use "single-phase".');
        }
        if (!config.maxInstances || config.maxInstances < 1) {
            throw new Error('VisibilitySystem: config.maxInstances must be at least 1.');
        }
        // The instance cull dispatches one workgroup of INSTANCE_CULL_GROUP per that many instances.
        const groups = Math.ceil(config.maxInstances / INSTANCE_CULL_GROUP);
        const maxGroups = engine.device.limits?.maxComputeWorkgroupsPerDimension ?? 65535;
        if (groups > maxGroups) {
            throw new Error(`VisibilitySystem: config.maxInstances ${config.maxInstances} needs ${groups} cull workgroups; this device ` +
                `dispatches at most ${maxGroups} (${maxGroups * INSTANCE_CULL_GROUP} instances).`);
        }

        this.occlusion = config.hiZ !== undefined;
        this.lodErrorPx = config.lodErrorPx ?? 1.0;

        // The shared source instance buffer (STORAGE). Written once per frame in upload().
        this.instanceBuffer = this.device.createBuffer({
            label: 'NullGraph_VisInstances',
            size: config.maxInstances * INSTANCE_FLOATS * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        if (this.occlusion) {
            this.hiZ = new HiZPyramid(engine);
            const hiZCfg = config.hiZ!;
            this.hiZ.configure(
                hiZCfg.depthSource,
                hiZCfg.width ?? engine.width,
                hiZCfg.height ?? engine.height,
            );
        }
    }

    // ---------------------------------------------------------------------------
    // Geometry
    // ---------------------------------------------------------------------------

    /**
     * Upload the shared geometry. `mega` provides the unified vertex/index pool and per-mesh
     * draw offsets (instance tier). `pack` (optional) provides the meshlet buffers (meshlet
     * tier). One mega vertex buffer backs both tiers (geometry doc §2.1).
     */
    public setGeometry(mega: MegabufferBuilder, pack?: MeshletPack): void {
        this.mega = mega;
        const built = mega.build();
        this.megaStrideFloats = mega.vertexStrideFloats;
        this.megaIndexFormat = built.indexFormat;

        // VERTEX (instance-tier draw) | STORAGE (meshlet-tier vertex pulling).
        this.megaVertexBuffer = this.createMappedBuffer(
            'NullGraph_VisMegaVerts', built.megaVertices,
            GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.megaIndexBuffer = this.createMappedBuffer(
            'NullGraph_VisMegaIndices', built.megaIndices,
            GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST);

        if (pack) {
            this.meshletBuffers = {
                meshlets: this.createMappedBuffer('NullGraph_VisMeshlets', pack.meshlets, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
                meshletVertices: this.createMappedBuffer('NullGraph_VisMeshletVerts', pack.meshletVertices, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
                meshletTriangles: this.createMappedBuffer('NullGraph_VisMeshletTris', pack.meshletTriangles, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
                meshLODs: this.createMappedBuffer('NullGraph_VisMeshLODs', pack.meshLODs, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
                meshTable: this.createMappedBuffer('NullGraph_VisMeshTable', pack.meshTable, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
            };
        }
        this.computeChain = null; // geometry changed; rebuild the chain on next getComputeStages
    }

    // ---------------------------------------------------------------------------
    // Instances (zero-copy)
    // ---------------------------------------------------------------------------

    /**
     * Adopt the caller's instance array by reference (the LightSystem.setRawData contract). A
     * Web Worker may fill `data` (see {@link InstanceBuffer}); nothing is copied until upload().
     */
    public setInstances(data: Float32Array, count: number): void {
        if (count > this.config.maxInstances) {
            throw new Error(`VisibilitySystem: setInstances received ${count} instances, exceeding maxInstances ${this.config.maxInstances}.`);
        }
        this.instanceStaging = data;
        this.instanceCount = count;
        this.gpuRows = false;
    }

    /**
     * Rows written on the GPU — an agent system's instance stage, a simulation. Sets the count the
     * culls dispatch over and drops any adopted array, so `upload()` writes only the cull params
     * and never overwrites a row. Rows that share the system and never move are written once with
     * `device.queue.writeBuffer(vis.instanceBuffer, …)`. `setInstances()` switches back.
     */
    public setInstanceCount(count: number): void {
        if (!Number.isInteger(count) || count < 0) {
            throw new Error(`VisibilitySystem: setInstanceCount needs a whole number of instances ≥ 0; got ${count}.`);
        }
        if (count > this.config.maxInstances) {
            throw new Error(`VisibilitySystem: setInstanceCount received ${count} instances, exceeding maxInstances ${this.config.maxInstances}.`);
        }
        this.instanceStaging = null;
        this.instanceCount = count;
        this.gpuRows = true;
    }

    /** `'gpu'` after `setInstanceCount()`: nothing on the CPU writes the rows. `'cpu'` otherwise. */
    public get instanceSource(): 'cpu' | 'gpu' {
        return this.gpuRows ? 'gpu' : 'cpu';
    }

    /** Blast the instance array to VRAM and refresh every cull stage's params. Once per frame. */
    public upload(): void {
        if (this.instanceStaging) {
            this.device.queue.writeBuffer(
                this.instanceBuffer, 0,
                this.instanceStaging.buffer, this.instanceStaging.byteOffset,
                this.instanceCount * INSTANCE_FLOATS * 4,
            );
        }
        // Refresh the per-stage CullParams (instanceCount + screen size change per frame/resize).
        const records: CullStageRecord[] = [...this.instanceTierCulls, ...this.meshletTierCulls];
        if (this.meshletInstanceCull) records.push(this.meshletInstanceCull);
        for (const rec of records) this.writeParams(rec);
    }

    // ---------------------------------------------------------------------------
    // Draw batches
    // ---------------------------------------------------------------------------

    /**
     * Instance tier: register one indirect draw batch for a unique mesh. Instances whose
     * `meshIndex` matches are frustum/occlusion-culled on the GPU and drawn with a single
     * `drawIndexedIndirect`, independent of instance count (roadmap §8.1).
     */
    public createInstanceDrawBatch(
        pass: RenderPassNode,
        meshName: string,
        meshIndex: number,
        options: InstanceDrawOptions = {},
    ): RenderBatch {
        if (!this.mega || !this.megaVertexBuffer || !this.megaIndexBuffer) {
            throw new Error('VisibilitySystem: setGeometry(mega) must be called before createInstanceDrawBatch().');
        }
        const offset = this.mega.getOffset(meshName);
        const materialGroup = options.materialGroupIndex ?? 1;
        const wgsl = this.resolveMaterialWGSL(
            options.material, options.fragmentBody, options.extraBindings, materialGroup);
        const hook = options.vertexHook, hookGroup = hook?.group ?? 2;
        const occupied = new Set<number>([0]);
        if (hook) {
            const refuse = (why: string): never => { throw new Error(`VisibilitySystem: vertexHook ${why}`); };
            if (options.shaderCode !== undefined) refuse('cannot be combined with shaderCode; use the generated vertex stage.');
            if (!Number.isInteger(hookGroup) || hookGroup < 1 || hookGroup >= (this.device.limits.maxBindGroups ?? 4)) {
                refuse('group must be a whole number above 0 and below device.limits.maxBindGroups.');
            }
            const groups = (code: string) => [...code.matchAll(/@group\(\s*(\d+)\s*\)/g)].map(m => Number(m[1]));
            for (const group of groups(`${wgsl.extraBindings ?? ''}\n${wgsl.fragmentBody ?? ''}`)) occupied.add(group);
            if (options.material || options.materialEntries) occupied.add(materialGroup);
            if (occupied.has(hookGroup)) refuse(`group ${hookGroup} is already used by the material, lighting or extra bindings; choose a dedicated group.`);
            if (groups(`${hook.bindings}\n${hook.wgsl}`).some(g => g !== hookGroup)) {
                refuse(`bindings must declare only group ${hookGroup}; group numbers are not rewritten.`);
            }
        }
        const lod = options.lod;
        if (lod) {
            const bound = (v: number | undefined) => v === undefined || (typeof v === 'number' && Number.isFinite(v) && v >= 0);
            const { minPixels, maxPixels } = lod;
            if (!bound(minPixels) || !bound(maxPixels) || (minPixels === undefined && maxPixels === undefined) ||
                (minPixels !== undefined && maxPixels !== undefined && minPixels >= maxPixels)) {
                throw new Error(`VisibilitySystem: lod needs minPixels or maxPixels, each a finite number ≥ 0, ` +
                    `and minPixels below maxPixels; got ${JSON.stringify(lod)}.`);
            }
        }
        // §9.3 (shadows-design.md): a shadow cascade points this batch at its own camera and
        // runs without Hi-Z. Both default to today's behavior.
        const camera = options.camera ?? this.engine.ctx.cameraUniformBuffer;
        const occlusion = options.occlusion ?? this.occlusion;

        const batch = this.engine.createBatch(pass, {
            shaderCode: options.shaderCode ?? buildInstanceDrawWGSL({ ...options, ...wgsl }),
            strideFloats: INSTANCE_FLOATS,
            maxInstances: this.config.maxInstances,
            isIndirect: true,
            sharedSourceBuffer: this.instanceBuffer, // avoid a redundant auto-allocation
            vertexLayouts: options.vertexLayouts ?? StandardLayout.getWebGPUDescriptor(),
            targetFormat: options.targetFormat,
            depthOnly: options.depthOnly,
            depthWriteEnabled: true,
            depthCompare: 'less',
            cameraBuffer: camera,
            // §12.1 (shadows-atlas-design.md): a shadow view scopes its draw to its atlas tile.
            viewport: options.viewport,
        });

        this.engine.setBatchGeometry(batch, this.megaVertexBuffer, this.megaIndexBuffer, offset.indexCount, this.megaIndexFormat);
        // Static draw args: [indexCount, instanceCount(=0, GPU-fills), firstIndex, baseVertex, 0].
        this.device.queue.writeBuffer(batch.indirectBuffer!, 0,
            new Uint32Array([offset.indexCount, 0, offset.firstIndex, offset.baseVertex, 0]));

        // A stock material binds itself (textures + uniform) via its own applyToBatch.
        if (options.material) {
            options.material.instance.applyToBatch(batch, materialGroup);
        }
        if (options.materialEntries) {
            this.engine.setRenderPassBindGroup(batch, options.materialEntries, materialGroup);
        }

        if (hook) {
            // Auto-layout pipelines require empty intervening bind groups to be set as well.
            for (let group = 1; group < hookGroup; group++) {
                if (!occupied.has(group)) this.engine.setRenderPassBindGroup(batch, [], group);
            }
            this.engine.setRenderPassBindGroup(batch, hook.entries, hookGroup);
        }

        // Design doc §14: cull (bits + block counts), then scatter (offsets, survivors in source order,
        // the draw count). No global atomic: the draw order is the same every run.
        const paramsBuffer = this.createParamsBuffer();
        const blocks = Math.ceil(this.config.maxInstances / INSTANCE_CULL_GROUP);
        const scratchBuffer = (label: string, words: number) => this.device.createBuffer({
            label: `NullGraph_VisCull${label}`, size: Math.max(16, words * 4), usage: GPUBufferUsage.STORAGE,
        });
        const bits = scratchBuffer('Bits', blocks * 8);
        const counts = scratchBuffer('BlockCounts', blocks);
        const cull = this.engine.createComputeStage({
            name: `Instance Cull: ${meshName}`,
            shaderCode: buildInstanceCullWGSL({ occlusion, lod }),
            dispatchCount: () => [Math.ceil(this.instanceCount / INSTANCE_CULL_GROUP), 1, 1],
        });
        this.engine.setComputePassBindGroup(cull, [
            { binding: 0, resource: { buffer: camera } },
            { binding: 1, resource: { buffer: this.instanceBuffer } },
            { binding: 2, resource: { buffer: bits } },
            { binding: 3, resource: { buffer: counts } },
            { binding: 4, resource: { buffer: paramsBuffer } },
            ...(occlusion ? [{ binding: 5, resource: this.hiZ!.view }] : []),
        ], 0);
        // At least one workgroup: with no instances, it writes the draw count 0.
        const scatter = this.engine.createComputeStage({
            name: `Instance Cull Scatter: ${meshName}`,
            shaderCode: buildInstanceScatterWGSL(),
            dispatchCount: () => [Math.max(1, Math.ceil(this.instanceCount / INSTANCE_CULL_GROUP)), 1, 1],
        });
        this.engine.setComputePassBindGroup(scatter, [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: { buffer: this.instanceBuffer } },
            { binding: 2, resource: { buffer: bits } },
            { binding: 3, resource: { buffer: counts } },
            { binding: 4, resource: { buffer: batch.storageBuffer } },
            { binding: 5, resource: { buffer: batch.indirectBuffer! } },
        ], 0);

        const record: CullStageRecord = {
            stages: [cull, scatter], paramsBuffer, scratch: [bits, counts], meshIndex, batch, pass,
            hizRect: options.hizRect,
            screenSize: options.screenSize,
        };
        this.instanceTierCulls.push(record);
        // Seed the params NOW, not at the next upload(). A params buffer starts zeroed, and a
        // zero instanceCount culls everything in silently — so a batch created after this
        // frame's upload() drew nothing, with no error. Found by the first Node/Dawn run of
        // test/shadows-gpu (its renderScene creates the batch after upload).
        this.writeParams(record);
        this.computeChain = null;
        return batch;
    }

    /**
     * Meshlet tier: register one indirect draw batch covering every meshlet of every instance
     * using a material. Culled per-cluster (cone + frustum + Hi-Z) with LOD, drawn by
     * meshlet-as-instance vertex pulling over a static index buffer (design doc §7.3).
     */
    public createMeshletDrawBatch(pass: RenderPassNode, options: MeshletDrawOptions = {}): RenderBatch {
        if (!this.meshletBuffers || !this.megaVertexBuffer) {
            throw new Error('VisibilitySystem: meshlet tier requires setGeometry(mega, pack) before createMeshletDrawBatch().');
        }
        const maxPerInstance = this.config.maxMeshletsPerInstance;
        const maxVisible = this.config.maxVisibleMeshletInstances;
        if (!maxPerInstance || !maxVisible) {
            throw new Error('VisibilitySystem: meshlet tier requires config.maxMeshletsPerInstance and config.maxVisibleMeshletInstances.');
        }

        // Group 1 holds the meshlet geometry on this tier, so materials start at group 2.
        const materialGroup = options.materialGroupIndex ?? 2;
        const wgsl = this.resolveMaterialWGSL(
            options.material, options.fragmentBody, options.extraBindings, materialGroup);
        // §9.3 (shadows-design.md): per-batch camera + occlusion overrides, defaulting to
        // today's behavior.
        const camera = options.camera ?? this.engine.ctx.cameraUniformBuffer;
        const occlusion = options.occlusion ?? this.occlusion;

        const batch = this.engine.createBatch(pass, {
            shaderCode: buildMeshletDrawWGSL({ strideFloats: this.megaStrideFloats, ...options, ...wgsl }),
            strideFloats: 2,                       // one vec2<u32> per visible meshlet entry
            maxInstances: maxVisible,
            isIndirect: true,
            sharedSourceBuffer: this.instanceBuffer,
            vertexLayouts: [],                     // vertex pulling — no vertex attributes
            targetFormat: options.targetFormat,
            depthWriteEnabled: true,
            depthCompare: 'less',
            cameraBuffer: camera,
        });

        // Static index buffer [0..371]; mega vertex buffer bound but unread (design doc §12 note 2).
        const staticIndex = this.getStaticIndexBuffer();
        this.engine.setBatchGeometry(batch, this.megaVertexBuffer, staticIndex, MESHLET_DRAW_INDEX_COUNT, 'uint16');
        this.device.queue.writeBuffer(batch.indirectBuffer!, 0,
            new Uint32Array([MESHLET_DRAW_INDEX_COUNT, 0, 0, 0, 0]));

        // Group 1: read-only geometry + instance data (design doc §7.3).
        this.engine.setRenderPassBindGroup(batch, [
            { binding: 0, resource: { buffer: this.meshletBuffers.meshlets } },
            { binding: 1, resource: { buffer: this.meshletBuffers.meshletVertices } },
            { binding: 2, resource: { buffer: this.meshletBuffers.meshletTriangles } },
            { binding: 3, resource: { buffer: this.megaVertexBuffer } },
            { binding: 4, resource: { buffer: this.instanceBuffer } },
        ], 1);
        if (options.material) {
            options.material.instance.applyToBatch(batch, materialGroup);
        }
        if (options.materialEntries) {
            this.engine.setRenderPassBindGroup(batch, options.materialEntries, materialGroup);
        }

        const paramsBuffer = this.createParamsBuffer();
        const stage = this.engine.createComputeStage({
            name: 'Meshlet Cull',
            shaderCode: buildMeshletCullWGSL({ occlusion, cone: true }),
            dispatchCount: [
                Math.ceil(this.config.maxInstances / 8),
                Math.ceil(maxPerInstance / 8),
                1,
            ],
            preClearBuffers: [{ buffer: batch.indirectBuffer!, offset: 4, data: new Uint32Array([0]) }],
        });
        // `params` is bound only when occlusion is on, matching the shader (layout:'auto' prunes
        // the unused binding otherwise and bind-group creation would throw).
        this.engine.setComputePassBindGroup(stage, [
            { binding: 0, resource: { buffer: camera } },
            { binding: 1, resource: { buffer: this.instanceBuffer } },
            { binding: 2, resource: { buffer: this.getVisibleInstancesBuffer() } },
            { binding: 3, resource: { buffer: this.meshletBuffers.meshlets } },
            { binding: 4, resource: { buffer: this.getCountersBuffer() } },
            { binding: 5, resource: { buffer: batch.storageBuffer } },
            { binding: 6, resource: { buffer: batch.indirectBuffer! } },
            ...(occlusion ? [{ binding: 7, resource: { buffer: paramsBuffer } }, { binding: 8, resource: this.hiZ!.view }] : []),
        ], 0);


        const record: CullStageRecord = { stages: [stage], paramsBuffer, meshIndex: 0, batch, pass };
        this.meshletTierCulls.push(record);
        // Seeded at registration for the same reason as the instance tier — see there.
        this.writeParams(record);
        this.computeChain = null;
        return batch;
    }

    // ---------------------------------------------------------------------------
    // Compute chain
    // ---------------------------------------------------------------------------

    /**
     * The ordered compute stages to splice into the render sequence BEFORE the scene pass:
     * `[...hiZBuild, ...instance culls, ...their scatters, meshletInstanceCull,
     * ...meshletCulls]`. Built once and
     * cached; the camera is read from the shared engine uniform, so no per-call camera is needed.
     */
    /**
     * The compute stages that feed `batch`, in order, or `[]` if this system did not create it.
     * The instance tier's are two: cull, then scatter (design doc §14: survivors in source
     * order); the meshlet tier's, one.
     *
     * They still ride this system's compute chain. The handles exist so that an owner can gate them
     * with `enabled = false` when it knows the draw is suppressed this frame (the shadow atlas does
     * that for cached tiles, shadows-atlas-design.md §8), or run them in a sequence of its own (the
     * Picker, ProbeVolume). Gate or run all of them, in this order.
     */
    public getCullStages(batch: RenderBatch): ComputeStageNode[] {
        for (const rec of [...this.instanceTierCulls, ...this.meshletTierCulls]) if (rec.batch === batch) return [...rec.stages];
        return [];
    }

    /**
     * Unregister a draw batch this system created: its cull stage leaves the compute chain, the
     * stage's params buffer is released, and the batch is detached from its pass with its own
     * draw buffers destroyed. Returns `false` (and does nothing) for a batch this system did not
     * create.
     *
     * Required by anything that shares this system and can be torn down before it. A cull stage
     * that outlives its owner keeps binding the owner's destroyed buffers (a shadow cascade's
     * camera, say), and from then on EVERY frame that runs this chain is refused at submit —
     * "used in submit while destroyed", the whole command buffer dropped. `ShadowSystem.destroy`
     * calls this for each of its caster batches.
     */
    public removeDrawBatch(batch: RenderBatch): boolean {
        for (const list of [this.instanceTierCulls, this.meshletTierCulls]) {
            const i = list.findIndex(r => r.batch === batch);
            if (i < 0) continue;
            const [rec] = list.splice(i, 1);
            rec.paramsBuffer.destroy();
            rec.scratch?.forEach(b => b.destroy());
            if (rec.pass) this.engine.clearBatch(rec.pass, batch);
            this.computeChain = null;
            return true;
        }
        return false;
    }

    public getComputeStages(): ComputeStageNode[] {
        if (this.computeChain) return this.computeChain;

        if (this.instanceTierCulls.length === 0 && this.meshletTierCulls.length === 0) {
            throw new Error('VisibilitySystem: getComputeStages() called before any draw batch was created.');
        }

        const chain: ComputeStageNode[] = [];
        if (this.hiZ) chain.push(...this.hiZ.getBuildStages());
        // Every batch's cull, then every scatter: each batch keeps its own order, and no dispatch
        // waits on the one just before it.
        for (let pass = 0; pass < 2; pass++) chain.push(...this.instanceTierCulls.map(r => r.stages[pass]));
        if (this.meshletTierCulls.length > 0) {
            chain.push(...this.getMeshletInstanceCull().stages);
            chain.push(...this.meshletTierCulls.flatMap(r => r.stages));
        }

        this.computeChain = chain;
        return chain;
    }

    // ---------------------------------------------------------------------------
    // Internals
    // ---------------------------------------------------------------------------

    /**
     * Resolve the `material` shorthand into the WGSL pair the draw shader needs. Explicit
     * `fragmentBody` always wins — a caller who hand-wrote a shader gets exactly that shader.
     */
    private resolveMaterialWGSL(
        material: VisibilityMaterial | undefined,
        explicitFragment: string | undefined,
        explicitBindings: string | undefined,
        groupIndex: number,
    ): { fragmentBody?: string; extraBindings?: string } {
        if (!material || explicitFragment) {
            return { fragmentBody: explicitFragment, extraBindings: explicitBindings };
        }
        const { bindings, fragment } = buildMaterialFragment({
            kind: material.kind,
            groupIndex,
            lighting: material.lighting,
            indirect: material.indirect,
            aerial: material.aerial,
            useHalfLambert: material.useHalfLambert,
            useSmoothRamp: material.useSmoothRamp,
        });
        // Any caller-supplied bindings are appended, so a material can still be combined with
        // extra custom resources at higher binding numbers.
        return { fragmentBody: fragment, extraBindings: bindings + (explicitBindings ?? '') };
    }

    /** The shared meshlet-tier coarse instance-cull stage (LOD + visibleInstances), built lazily. */
    private getMeshletInstanceCull(): CullStageRecord {
        if (this.meshletInstanceCull) return this.meshletInstanceCull;
        if (!this.meshletBuffers) {
            throw new Error('VisibilitySystem: meshlet tier requires setGeometry(mega, pack).');
        }
        const paramsBuffer = this.createParamsBuffer();
        const stage = this.engine.createComputeStage({
            name: 'Meshlet Instance Cull',
            shaderCode: buildMeshletInstanceCullWGSL({ occlusion: this.occlusion }),
            dispatchCount: () => [Math.ceil(this.instanceCount / 64), 1, 1],
            preClearBuffers: [{ buffer: this.getCountersBuffer(), offset: 0, data: new Uint32Array([0]) }],
        });
        this.engine.setComputePassBindGroup(stage, [
            { binding: 0, resource: { buffer: this.engine.ctx.cameraUniformBuffer } },
            { binding: 1, resource: { buffer: this.instanceBuffer } },
            { binding: 2, resource: { buffer: this.getVisibleInstancesBuffer() } },
            { binding: 3, resource: { buffer: this.getCountersBuffer() } },
            { binding: 4, resource: { buffer: this.meshletBuffers.meshTable } },
            { binding: 5, resource: { buffer: this.meshletBuffers.meshLODs } },
            { binding: 6, resource: { buffer: paramsBuffer } },
            ...(this.occlusion ? [{ binding: 7, resource: this.hiZ!.view }] : []),
        ], 0);
        this.meshletInstanceCull = { stages: [stage], paramsBuffer, meshIndex: 0 };
        return this.meshletInstanceCull;
    }

    private getVisibleInstancesBuffer(): GPUBuffer {
        if (!this.visibleInstancesBuffer) {
            this.visibleInstancesBuffer = this.device.createBuffer({
                label: 'NullGraph_VisVisibleInstances',
                size: this.config.maxInstances * 16, // VisibleInstance = 4 u32
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
        }
        return this.visibleInstancesBuffer;
    }

    private getCountersBuffer(): GPUBuffer {
        if (!this.countersBuffer) {
            this.countersBuffer = this.device.createBuffer({
                label: 'NullGraph_VisCounters',
                size: 16, // counters[0] = visibleInstanceCount; rest reserved
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
        }
        return this.countersBuffer;
    }

    private getStaticIndexBuffer(): GPUBuffer {
        if (!this.staticIndexBuffer) {
            const indices = new Uint16Array(MESHLET_DRAW_INDEX_COUNT);
            for (let i = 0; i < MESHLET_DRAW_INDEX_COUNT; i++) indices[i] = i;
            this.staticIndexBuffer = this.createMappedBuffer(
                'NullGraph_VisStaticIndex', indices, GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST);
        }
        return this.staticIndexBuffer;
    }

    private createParamsBuffer(): GPUBuffer {
        return this.device.createBuffer({
            label: 'NullGraph_VisCullParams',
            size: 48, // CullParams grew hizRect (shadows-atlas-design.md §7.2): 32 → 48 bytes.
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    /** Fill one stage's CullParams uniform (48 bytes) from current frame state. */
    private writeParams(rec: CullStageRecord): void {
        const u = this.paramsU32;
        const f = this.paramsFloat;
        u[0] = this.instanceCount;
        u[1] = rec.meshIndex;
        u[2] = this.hiZ ? this.hiZ.maxLevel : 0;
        u[3] = 0; // flags: reserved (occlusion is baked into the shader variant)
        // §7.2 — a shadow view's screen size is its TILE size, so the footprint→mip selection
        // stays correct; the main view uses the Hi-Z source / canvas size.
        f[4] = rec.screenSize?.[0] ?? this.config.hiZ?.width ?? this.engine.width;
        f[5] = rec.screenSize?.[1] ?? this.config.hiZ?.height ?? this.engine.height;
        f[6] = this.lodErrorPx;
        f[7] = 0;
        // §7.2 — the Hi-Z UV remap. Identity (1,1,0,0) for the main view; a shadow view passes
        // its tile's atlas rect so the occlusion test samples only that tile.
        const hr = rec.hizRect;
        f[8] = hr?.[0] ?? 1;
        f[9] = hr?.[1] ?? 1;
        f[10] = hr?.[2] ?? 0;
        f[11] = hr?.[3] ?? 0;
        this.device.queue.writeBuffer(rec.paramsBuffer, 0, this.paramsFloat);
    }

    private createMappedBuffer(label: string, data: ArrayBuffer | ArrayBufferView, usage: GPUBufferUsageFlags): GPUBuffer {
        const byteLength = (data as ArrayBufferView).byteLength ?? (data as ArrayBuffer).byteLength;
        const buffer = this.device.createBuffer({
            label,
            size: Math.ceil(byteLength / 4) * 4,
            usage,
            mappedAtCreation: true,
        });
        const range = buffer.getMappedRange();
        if (data instanceof ArrayBuffer) {
            new Uint8Array(range).set(new Uint8Array(data));
        } else {
            new Uint8Array(range).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
        }
        buffer.unmap();
        return buffer;
    }

    public resize(width: number, height: number): void {
        if (this.hiZ) {
            const hiZCfg = this.config.hiZ!;
            this.hiZ.configure(hiZCfg.depthSource, hiZCfg.width ?? width, hiZCfg.height ?? height);
        }
        this.computeChain = null; // Hi-Z stages were rebuilt
    }

    public destroy(): void {
        this.instanceBuffer.destroy();
        this.megaVertexBuffer?.destroy();
        this.megaIndexBuffer?.destroy();
        if (this.meshletBuffers) {
            Object.values(this.meshletBuffers).forEach(b => b.destroy());
        }
        this.visibleInstancesBuffer?.destroy();
        this.countersBuffer?.destroy();
        this.staticIndexBuffer?.destroy();
        this.hiZ?.destroy();
        for (const rec of [...this.instanceTierCulls, ...this.meshletTierCulls]) {
            rec.paramsBuffer.destroy();
            rec.scratch?.forEach(b => b.destroy());
        }
        this.meshletInstanceCull?.paramsBuffer.destroy();
        this.meshletParamsBuffer?.destroy();
        this.computeChain = null;
    }
}
