// src/extras/shadows/ShadowSystem.ts
//
// The ShadowSystem: Phase 1 cascaded shadow maps (docs/shadows-design.md) and Phase 2's atlas,
// punctual lights, and GPU caster culling (docs/shadows-atlas-design.md).
//
// Phase 1 (no `atlasSize`): one directional light, N cascades, a depth24plus 2D array, one layer
// per cascade. Phase 2 (`atlasSize` set): every light type casts into ONE atlas texture
// partitioned by importance; tiles that did not change are not re-rendered; casters hidden
// behind other casters are culled on the GPU.
//
// The unifying move (§1): a `ShadowView` — a `Camera` plus a render target — becomes a
// `ShadowView` plus an atlas rectangle. A cascade, a spot cone, and one of six cube faces differ
// only in how their matrix is built and how big a tile they get. Every downstream mechanism —
// culling, drawing, sampling, caching — is written once against `ShadowView`.
//
// Per frame: setLight / registerLight → update(camera) → splice getComputeStages() +
// getPasses()/getAtlasPass() ahead of the scene pass. No allocation in the frame loop.

import { NullGraph, RenderBatch, ComputeStageNode, RenderPassNode, Camera } from '../../core';
import type { LightSystem } from '../lights/LightSystem';
import type { ShaderLightingLayout } from '../lights/types';
import { StandardLayout } from '../geometry/VertexLayout';
import { HiZPyramid } from '../visibility/HiZPyramid';
import {
    SHADOW_DATA_FLOATS, SHADOW_CASTER_PARAMS_FLOATS,
    validateShadowConfig, buildCascades, packShadowData, checkNormalOffset,
    createCascadeResults,
    type CascadeResult, type ShadowCameraState,
} from './shadowMath';
import {
    buildShadowBindingsWGSL, buildShadowDecoratorWGSL, buildShadowCasterWGSL,
    SHADOW_STRUCTS_WGSL, SHADOW_SAMPLING_WGSL, SHADOW_BIASED_WGSL,
} from './shadowShaders';
import {
    SHADOW_VIEW_RECORD_FLOATS, SHADOW_SLOT_RECORD_FLOATS,
    SHADOW_SLOT_KIND_DIRECTIONAL, SHADOW_SLOT_KIND_SPOT, SHADOW_SLOT_KIND_POINT,
    spotView, cubeFaceView, packSlotIndex, MAX_SHADOW_SLOT,
    importance, resolveTileSize, TILE_LADDER, createBuiltView,
    type BuiltView, type ShadowViewRequest, type ImportanceCameraState,
} from './atlasMath';
import {
    ATLAS_STRUCTS_WGSL, ATLAS_PARAMS_WGSL, ATLAS_SAMPLING_WGSL,
    buildAtlasBindingsWGSL, buildAtlasDecoratorWGSL, TILE_CLEAR_WGSL,
} from './atlasShaders';
import { ShadowAtlas } from './ShadowAtlas';
import type { ShadowConfig } from './types';

/** One registered caster mesh: `cascades` batches, one per cascade layer's depth pass (Phase 1),
 *  or `maxViews` batches — one per view SLOT — in the single atlas pass (Phase 2). */
interface CasterRecord {
    meshName: string;
    meshIndex: number;
    batches: RenderBatch[];
}

/**
 * Phase 2 per-view-slot caster state. Batches are created once per SLOT (not per resident view)
 * at registration time, because tiles are allocated later in `update()`. Each frame `update()`
 * points the slot's batches at the current tile's viewport and zeroes the draw's instance count
 * when the slot is empty or its view is clean (§8 caching).
 */
interface ViewSlotDraw {
    /** The caster batches for this slot, one per registered caster mesh. */
    batches: RenderBatch[];
    /**
     * The VisibilitySystem cull stage feeding each batch, index-aligned with `batches`. Gated
     * with the viewport every frame: a cached tile that rasterizes nothing must not pay for its
     * caster cull either (§8) — the viewport suppresses rasterization, not the compute.
     */
    cullStages: (ComputeStageNode | null)[];
    /** The tile-clear batch for this slot (§8.2), drawn first when the view is dirty. */
    clearBatch: RenderBatch | null;
}

/** A registered shadow-casting light (Phase 2 §4.2/§4.3). */
interface LightRecord {
    name: string;
    kind: number;               // SHADOW_SLOT_KIND_*
    slot: number;               // index into the slots table; written to spot_params.z as slot+1
    firstView: number;          // index into the views array
    viewCount: number;          // cascades: N; spot: 1; point: 6
    // Punctual state (spot/point):
    position: [number, number, number];
    direction: [number, number, number];
    range: number;
    outerCos: number;           // spot only
    userImportance: number;
    /** The packed view matrices from the last update — dirty detection compares against these. */
    lastViewData: Float32Array | null;
}

export class ShadowSystem {
    private readonly engine: NullGraph;
    private readonly device: GPUDevice;
    private readonly config: Required<Omit<ShadowConfig, 'visibility' | 'allowColorFallback' | 'atlasSize' | 'maxTileSize' | 'maxViews' | 'maxSlots' | 'updateMode' | 'refreshPerFrame' | 'cubeFovMarginTexels' | 'lightViewOcclusion'>> & {
        visibility?: ShadowConfig['visibility'];
        allowColorFallback?: boolean;
        atlasSize?: number;
        maxTileSize?: number;
        maxViews?: number;
        maxSlots?: number;
        updateMode?: 'every-frame' | 'on-demand' | 'round-robin';
        refreshPerFrame?: number;
        cubeFovMarginTexels?: number;
        lightViewOcclusion?: boolean;
    };

    // --- light state ---
    private lightDir: [number, number, number] = [0, -1, 0];

    // --- Phase 1 GPU resources (cascade array path; unused when atlasSize is set) ---
    private readonly cascadeArray: GPUTexture | null = null;
    private readonly layerViews: GPUTextureView[] = [];
    private samplingViewP1!: GPUTextureView;
    private comparisonSamplerP1!: GPUSampler;
    private readonly cascadeBuffers: GPUBuffer[] = [];
    private readonly casterParamsBuffers: GPUBuffer[] = [];
    private shadowDataBuffer!: GPUBuffer;

    // --- Phase 2 GPU resources (atlas path) ---
    private atlas: ShadowAtlas | null = null;
    private atlasPass: RenderPassNode | null = null;
    private shadowParamsBuffer: GPUBuffer | null = null;
    private lightHiZ: HiZPyramid | null = null;
    /**
     * The atlas must start cleared to 1.0 (far) on the first frame — under `depthLoadOp: 'load'`
     * (the caching modes) a clean tile would otherwise read uninitialized memory. The pass's
     * loadOp is read at beginRenderPass each frame, so update() flips it to 'load' after frame 1.
     */
    private atlasClearedOnce: boolean = false;

    // --- per-frame staging (persistent — no allocation in update) ---
    private readonly shadowDataF32 = new Float32Array(SHADOW_DATA_FLOATS);
    private readonly casterParamsF32 = new Float32Array(SHADOW_CASTER_PARAMS_FLOATS);
    private cascades: CascadeResult[] | null = null;

    /**
     * The reusable cascade fit target. `buildCascades` overwrites these in place every frame, so
     * `fittedCascades` exposes live data rather than a snapshot — see its doc comment.
     */
    private readonly cascadeResults: CascadeResult[];
    /** The reusable punctual view target — one at a time, since updateAtlas consumes it inline. */
    private readonly scratchView: BuiltView = createBuiltView();
    /** The ShadowParams staging pair (u32 view aliases the same 12 words). */
    private readonly shadowParamsF32 = new Float32Array(12);
    private readonly shadowParamsU32 = new Uint32Array(this.shadowParamsF32.buffer);
    /**
     * This frame's camera state for the importance heuristic. Persistent, and — critically —
     * shared by BOTH tile sizing and eviction, so the two can never rank views differently
     * (see `importanceOf`).
     */
    private readonly camState: { eye: [number, number, number]; near: number; screenH: number } =
        { eye: [0, 0, 0], near: 0.1, screenH: 1 };
    /** Reusable importance request — filled per call rather than an object literal per view. */
    private readonly impRequest: ShadowViewRequest & { position: [number, number, number] } =
        { kind: 'cascade', cascadeIndex: 0, position: [0, 0, 0], range: 1, userImportance: 1 };

    // Phase 2 staging: maxViews × 108 camera floats + the per-view metadata.
    private viewDataF32: Float32Array | null = null;
    private viewMeta: { rect: [number, number, number, number]; texelWorld: number; splitFar: number; kind: number }[] = [];
    private slotMeta: { firstView: number; viewCount: number; kind: number; lightRange: number }[] = [];
    private frame: number = 0;
    private roundRobinCursor: number = 0;

    // --- registered casters + lights ---
    private readonly casters: CasterRecord[] = [];
    private readonly lights: LightRecord[] = [];
    private readonly passes: RenderPassNode[] = [];   // Phase 1 per-cascade passes
    /**
     * Phase 2 per-view-slot draw state, indexed by view slot (0..maxViews). Created lazily on
     * the first `registerCaster` (which must precede the first `update`). Each frame `update()`
     * retargets every slot's batches to the current tile and suppresses clean/empty slots (§8).
     */
    private viewSlots: ViewSlotDraw[] = [];
    /** The caster meshes registered so far — used to build a slot's batches on first use. */
    private casterMeshes: { meshName: string; meshIndex: number }[] = [];
    /**
     * How many view slots have had their batches built. Slots are materialized up to the number
     * of views the REGISTERED LIGHTS actually own, not up to `maxViews` — building all `maxViews`
     * cost a render pipeline + a cull stage + a params buffer per (slot × caster mesh) at startup,
     * most of them for view slots no light will ever occupy.
     */
    private materializedSlots: number = 0;

    constructor(engine: NullGraph, config: ShadowConfig) {
        this.engine = engine;
        this.device = engine.device;

        const cascades = config.cascades ?? 4;
        const resolution = config.resolution ?? 2048;
        validateShadowConfig(cascades, resolution);

        const shadowGroupIndex = config.shadowGroupIndex ?? 3;
        if (shadowGroupIndex <= 2) {
            throw new Error(
                `ShadowSystem: shadowGroupIndex ${shadowGroupIndex} collides with LightSystem's ` +
                `hardcoded @group(2); use 3 or higher.`);
        }

        const shadowDistance = config.shadowDistance ?? 200;
        this.cascadeResults = createCascadeResults(cascades);
        this.config = {
            cascades,
            resolution,
            shadowDistance,
            lambda: config.lambda ?? 0.6,
            casterExtrusion: config.casterExtrusion ?? shadowDistance,
            constantBias: config.constantBias ?? 0.0015,
            normalOffsetTexels: config.normalOffsetTexels ?? 1.5,
            pcfRadius: config.pcfRadius ?? 1,
            blendTexels: config.blendTexels ?? 4.0,
            lodBias: config.lodBias ?? 1,
            shadowGroupIndex,
            visibility: config.visibility,
            allowColorFallback: config.allowColorFallback,
            atlasSize: config.atlasSize,
            maxTileSize: config.maxTileSize,
            maxViews: config.maxViews,
            maxSlots: config.maxSlots,
            updateMode: config.updateMode,
            refreshPerFrame: config.refreshPerFrame,
            cubeFovMarginTexels: config.cubeFovMarginTexels,
            lightViewOcclusion: config.lightViewOcclusion,
        };

        if (this.config.atlasSize !== undefined) {
            // ----- Phase 2: the atlas path -----
            this.atlas = new ShadowAtlas(engine, {
                atlasSize: this.config.atlasSize,
                maxTileSize: this.config.maxTileSize,
                maxViews: this.config.maxViews,
                maxSlots: this.config.maxSlots,
            });
            this.viewDataF32 = new Float32Array(this.atlas.maxViews * 108);
            for (let i = 0; i < this.atlas.maxViews; i++) {
                this.viewMeta.push({ rect: [0, 0, 0, 0], texelWorld: 0, splitFar: 0, kind: 0 });
                this.casterParamsBuffers.push(this.device.createBuffer({
                    label: `NullGraph_ShadowAtlasCasterParams${i}`,
                    size: SHADOW_CASTER_PARAMS_FLOATS * 4,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                }));
            }
            this.shadowParamsBuffer = this.device.createBuffer({
                label: 'NullGraph_ShadowParams',
                size: 48, // ShadowParams: 4 + 4 + 4 + 4 + 16 + 4 + 4×3 = 48 bytes
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            // §7.2 — the light-view Hi-Z over the atlas. Reused unchanged; built from last
            // frame's atlas, so a one-frame-stale shadow is the worst case (and less visible
            // than one-frame-stale geometry).
            if (this.config.lightViewOcclusion) {
                this.lightHiZ = new HiZPyramid(engine);
                this.lightHiZ.configure(this.atlas.samplingView, this.atlas.atlasSize, this.atlas.atlasSize);
            }
            // §9.1 — the cube fov margin must cover the filter kernel, or face seams appear.
            const margin = this.config.cubeFovMarginTexels ?? this.config.pcfRadius;
            if (margin < this.config.pcfRadius) {
                throw new Error(
                    `ShadowSystem: filterTexels ${this.config.pcfRadius} exceeds the cube-face fov ` +
                    `margin (${margin} texels); raise cubeFovMarginTexels or lower pcfRadius — ` +
                    `face seams will appear.`);
            }
        } else {
            // ----- Phase 1: the cascade-array path (unchanged) -----
            this.cascadeArray = this.device.createTexture({
                label: 'NullGraph_ShadowCascades',
                size: [resolution, resolution, cascades],
                format: 'depth24plus',
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            });
            for (let i = 0; i < cascades; i++) {
                this.layerViews.push(this.cascadeArray.createView({
                    dimension: '2d', baseArrayLayer: i, arrayLayerCount: 1,
                }));
            }
            this.samplingViewP1 = this.cascadeArray.createView({ dimension: '2d-array' });
            this.comparisonSamplerP1 = this.device.createSampler({
                label: 'NullGraph_ShadowSampler',
                compare: 'less', magFilter: 'linear', minFilter: 'linear',
            });
            for (let i = 0; i < cascades; i++) {
                this.cascadeBuffers.push(this.device.createBuffer({
                    label: `NullGraph_ShadowCamera${i}`,
                    size: 432,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                }));
                this.casterParamsBuffers.push(this.device.createBuffer({
                    label: `NullGraph_ShadowCasterParams${i}`,
                    size: SHADOW_CASTER_PARAMS_FLOATS * 4,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                }));
            }
            this.shadowDataBuffer = this.device.createBuffer({
                label: 'NullGraph_ShadowData',
                size: SHADOW_DATA_FLOATS * 4,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
        }
    }

    /** True when the system is in the Phase 2 atlas path. */
    public get isAtlas(): boolean { return this.atlas !== null; }

    // ---------------------------------------------------------------------------
    // Per frame (§9.2)
    // ---------------------------------------------------------------------------

    /** Set the sun's TRAVEL direction (where the light goes). Normalized internally. */
    public setLight(direction: ArrayLike<number>): void {
        const l = Math.hypot(direction[0], direction[1], direction[2]) || 1;
        this.lightDir = [direction[0] / l, direction[1] / l, direction[2] / l];
    }

    /**
     * Phase 2 §4.3 — register a shadow-casting light. Returns the shadow slot; the caller writes
     * `slot + 1` into the light record's `spot_params.z` (float 14 of the 16-float record) so the
     * receiver can find this light's views. `0` in that field means "does not cast."
     *
     * ```ts
     * const slot = shadows.registerLight({ kind: 'point', position, range: 30 });
     * lightData[lightIndex * 16 + 14] = slot + 1;   // spot_params.z
     * ```
     */
    public registerLight(request: {
        kind: 'directional' | 'spot' | 'point';
        name?: string;
        position?: ArrayLike<number>;
        direction?: ArrayLike<number>;
        range?: number;
        outerCos?: number;
        importance?: number;
    }): number {
        if (!this.atlas) {
            throw new Error('ShadowSystem: registerLight requires the Phase 2 atlas path (set config.atlasSize).');
        }
        const slot = this.atlas.allocateSlot();
        if (slot < 0 || slot > MAX_SHADOW_SLOT) {
            throw new Error(
                `ShadowSystem: shadow slot ${slot} exceeds ${MAX_SHADOW_SLOT}; ` +
                `spot_params.z encodes slot+1 as a float.`);
        }
        const kind = request.kind === 'directional' ? SHADOW_SLOT_KIND_DIRECTIONAL
            : request.kind === 'spot' ? SHADOW_SLOT_KIND_SPOT
            : SHADOW_SLOT_KIND_POINT;
        const viewCount = kind === SHADOW_SLOT_KIND_DIRECTIONAL ? this.config.cascades
            : kind === SHADOW_SLOT_KIND_SPOT ? 1 : 6;
        const firstView = this.lights.reduce((n, l) => n + l.viewCount, 0);
        if (firstView + viewCount > this.atlas.maxViews) {
            throw new Error(
                `ShadowAtlas: light "${request.name ?? request.kind}" requests ${viewCount} views, ` +
                `exceeding maxViews ${this.atlas.maxViews}; raise maxViews or stop shadowing this light.`);
        }
        const rec: LightRecord = {
            name: request.name ?? request.kind,
            kind, slot, firstView, viewCount,
            position: [request.position?.[0] ?? 0, request.position?.[1] ?? 0, request.position?.[2] ?? 0],
            direction: [request.direction?.[0] ?? 0, request.direction?.[1] ?? -1, request.direction?.[2] ?? 0],
            range: request.range ?? 30,
            outerCos: request.outerCos ?? Math.cos(Math.PI / 6),
            userImportance: request.importance ?? 1.0,
            lastViewData: null,
        };
        this.lights.push(rec);
        this.slotMeta.push({ firstView, viewCount, kind, lightRange: rec.range });
        // Build the draw state for the slots this light just claimed (a no-op until a caster is
        // registered). Doing it here — rather than for all `maxViews` up front, or lazily in
        // update() — keeps pipeline compilation at setup time and proportional to the lights that
        // actually exist.
        this.ensureViewSlots(firstView + viewCount);
        return slot;
    }

    /**
     * Phase 2 — update a registered light's dynamic parameters (position, direction, range, cone).
     * Call this per-frame when a shadow-casting light moves so its atlas shadow cameras follow.
     */
    public updateLight(slot: number, update: {
        position?: ArrayLike<number>;
        direction?: ArrayLike<number>;
        range?: number;
        outerCos?: number;
        importance?: number;
    }): void {
        const rec = this.lights.find(l => l.slot === slot);
        if (!rec) return;
        if (update.position) {
            rec.position = [update.position[0], update.position[1], update.position[2]];
        }
        if (update.direction) {
            const d = update.direction;
            const l = Math.hypot(d[0], d[1], d[2]) || 1;
            rec.direction = [d[0] / l, d[1] / l, d[2] / l];
        }
        if (update.range !== undefined) {
            rec.range = update.range;
            const sm = this.slotMeta[slot];
            if (sm) sm.lightRange = update.range;
        }
        if (update.outerCos !== undefined) rec.outerCos = update.outerCos;
        if (update.importance !== undefined) rec.userImportance = update.importance;
    }

    /**
     * Phase 2 §8.1 — declare a light's casters moved. The light's views re-render this frame.
     * Dirtiness is declared, not inferred: NullGraph has no scene graph, so it cannot know a
     * caster moved. The default `updateMode: 'every-frame'` makes this a no-op (always correct).
     */
    public invalidate(slot: number): void {
        const rec = this.lights.find(l => l.slot === slot);
        if (!rec) {
            throw new Error(
                `ShadowSystem: invalidate(${slot}) — no such shadow slot; registerLight() returns the valid range.`);
        }
        if (!this.atlas) return;
        for (let v = rec.firstView; v < rec.firstView + rec.viewCount; v++) {
            this.atlas.markDirty(v);
        }
    }

    /**
     * Splits, sphere fit, snap, extrusion — then writes the cascade camera buffers, the caster
     * params, and the ShadowData uniform (Phase 1), OR builds every light's views, allocates
     * tiles, and writes the atlas tables (Phase 2). No allocation either way.
     */
    public update(camera: Camera): void {
        if (this.atlas) {
            this.updateAtlas(camera);
            return;
        }
        // ----- Phase 1 path -----
        const results = buildCascades(this.fitCameraState(camera), this.lightDir, this.config,
            this.cascadeResults);
        checkNormalOffset(results, this.config.normalOffsetTexels);
        const q = this.device.queue;
        for (let k = 0; k < results.length; k++) {
            q.writeBuffer(this.cascadeBuffers[k], 0, results[k].viewData as Float32Array<ArrayBuffer>);
            const p = this.casterParamsF32;
            p[0] = results[k].texelWorld;
            p[1] = this.config.normalOffsetTexels;
            p[2] = this.config.constantBias;
            p[3] = 0;
            q.writeBuffer(this.casterParamsBuffers[k], 0, p as Float32Array<ArrayBuffer>);
        }
        packShadowData(results, this.lightDir, this.config.pcfRadius, this.config.blendTexels,
            this.config.constantBias, this.shadowDataF32);
        q.writeBuffer(this.shadowDataBuffer, 0, this.shadowDataF32 as Float32Array<ArrayBuffer>);
        this.cascades = results;
    }

    /** Phase 2 update: build every light's views, allocate/evict tiles, write the tables. */
    private updateAtlas(camera: Camera): void {
        const atlas = this.atlas!;
        this.frame++;
        // This frame's camera state, written into the persistent record BEFORE any tile is sized
        // or evicted — `importanceOf` reads it, so sizing and eviction share one camera.
        this.camState.eye[0] = camera.bufferData[96];
        this.camState.eye[1] = camera.bufferData[97];
        this.camState.eye[2] = camera.bufferData[98];
        this.camState.near = camera.near;
        this.camState.screenH = this.engine.height;
        const updateMode = this.config.updateMode ?? 'every-frame';
        const refreshPerFrame = this.config.refreshPerFrame ?? 2;

        // Build each light's views and ensure each has a tile.
        for (const light of this.lights) {
            // Directional lights: build the cascades and pack them into the view slots.
            if (light.kind === SHADOW_SLOT_KIND_DIRECTIONAL) {
                const results = buildCascades(this.fitCameraState(camera), this.lightDir,
                    this.config, this.cascadeResults);
                for (let k = 0; k < results.length; k++) {
                    const viewIndex = light.firstView + k;
                    this.viewDataF32!.set(results[k].viewData, viewIndex * 108);
                    // Mutate the persistent meta record rather than replacing it — the object
                    // literal was a per-cascade, per-frame allocation.
                    const meta = this.viewMeta[viewIndex];
                    meta.rect = this.ensureTile(viewIndex, this.config.resolution, true, light.name);
                    meta.texelWorld = results[k].texelWorld;
                    meta.splitFar = results[k].splitFar;
                    meta.kind = 0;
                    const p = this.casterParamsF32;
                    p[0] = results[k].texelWorld;
                    p[1] = this.config.normalOffsetTexels;
                    p[2] = this.config.constantBias;
                    this.device.queue.writeBuffer(this.casterParamsBuffers[viewIndex], 0, p as Float32Array<ArrayBuffer>);
                    // Cascades follow the camera → always dirty (§8.1).
                    atlas.markDirty(viewIndex);
                }
                continue;
            }

            // Punctual lights: build, pack, and dirty-check each view in ONE pass through the
            // reusable scratch view. The original collected a BuiltView[] first, which allocated
            // an array per light plus a 108-float array per view, every frame.
            if (!light.lastViewData || light.lastViewData.length !== light.viewCount * 108) {
                light.lastViewData = new Float32Array(light.viewCount * 108);   // once, per light
            }
            for (let v = 0; v < light.viewCount; v++) {
                const viewIndex = light.firstView + v;
                const size = this.tileSizeFor(light, viewIndex);
                const bv = light.kind === SHADOW_SLOT_KIND_SPOT
                    ? spotView(light.position, light.direction, light.outerCos, light.range,
                        camera.near, size, camera.bufferData[99], this.scratchView)
                    : cubeFaceView(light.position, v, light.range, camera.near, size,
                        this.config.cubeFovMarginTexels ?? this.config.pcfRadius,
                        camera.bufferData[99], this.scratchView);

                const rect = this.ensureTile(viewIndex, size, false, light.name);
                this.viewDataF32!.set(bv.viewData, viewIndex * 108);
                const meta = this.viewMeta[viewIndex];
                meta.rect = rect;
                meta.texelWorld = bv.texelWorld;
                meta.splitFar = bv.splitFar;
                meta.kind = bv.kind;

                const p = this.casterParamsF32;
                p[0] = bv.texelWorld;
                p[1] = this.config.normalOffsetTexels;
                p[2] = this.config.constantBias;
                this.device.queue.writeBuffer(this.casterParamsBuffers[viewIndex], 0, p as Float32Array<ArrayBuffer>);

                // Dirty detection: the view's matrix changed (light moved) since last frame.
                // Compared by index against the stash — `subarray` was a per-view allocation.
                const stash = light.lastViewData;
                if (updateMode !== 'every-frame') {
                    for (let i = 0; i < 16; i++) {
                        if (stash[v * 108 + i] !== bv.viewData[i]) { atlas.markDirty(viewIndex); break; }
                    }
                }
                // Stash this frame's matrices for next frame's comparison.
                stash.set(bv.viewData, v * 108);
            }
        }

        // Round-robin: refresh a few clean views per frame (§8.1) — bounds the damage from a
        // missed invalidate() to a few frames instead of forever.
        if (updateMode === 'round-robin') {
            let refreshed = 0;
            for (let n = 0; n < atlas.maxViews && refreshed < refreshPerFrame; n++) {
                const viewIndex = (this.roundRobinCursor + n) % atlas.maxViews;
                const rec = atlas.getView(viewIndex);
                if (rec && !rec.dirty) { atlas.markDirty(viewIndex); refreshed++; }
            }
            this.roundRobinCursor = (this.roundRobinCursor + refreshPerFrame) % atlas.maxViews;
        }

        // Write the tables + the ShadowParams uniform.
        atlas.writeTables(this.viewDataF32!, this.viewMeta, this.slotMeta);
        this.writeShadowParams();

        // §6.1 / §8 — point every view slot's batches at its current tile and suppress the
        // clean/empty slots. Runs after all tiles are allocated so the viewports are current.
        this.retargetViewSlots();

        // §8.2 — after frame 1's full-atlas 'clear' has been submitted, flip the caching modes to
        // 'load' so clean tiles are preserved. Gated on frame > 1 so the flip lands AFTER the
        // first render (update runs before engine.render within a frame).
        if (this.frame > 1) this.settleAtlasLoadOp();
    }

    /**
     * The plain-data camera snapshot the cascade fitter consumes — a persistent record, refilled
     * per call rather than an object literal per frame.
     */
    private readonly fitState: ShadowCameraState & { simTime: number } =
        { invView: [], fovRads: 0, aspect: 1, near: 0.1, simTime: 0 };

    private fitCameraState(camera: Camera): ShadowCameraState {
        const s = this.fitState;
        s.invView = camera.inverseViewMatrix;
        s.fovRads = camera.bufferData[102];
        s.aspect = camera.bufferData[103];
        s.near = camera.near;
        s.simTime = camera.bufferData[99];
        return s;
    }

    /**
     * A view's importance (§5.1), evaluated against **this frame's camera** (`camState`).
     *
     * Both tile sizing and eviction go through here. That is the point: sizing a tile from the
     * live camera while ranking eviction from a different one makes the two disagree, and the
     * symptom is that a full atlas evicts the tiles nearest the viewer.
     */
    private importanceOf(light: LightRecord, viewIndex: number): number {
        const r = this.impRequest;
        r.kind = light.kind === SHADOW_SLOT_KIND_DIRECTIONAL ? 'cascade'
            : (light.kind === SHADOW_SLOT_KIND_SPOT ? 'spot' : 'point');
        r.cascadeIndex = viewIndex - light.firstView;
        r.position[0] = light.position[0];
        r.position[1] = light.position[1];
        r.position[2] = light.position[2];
        r.range = light.range;
        r.userImportance = light.userImportance;
        return importance(r, this.camState as ImportanceCameraState);
    }

    /** The tile size a view wants this frame, with hysteresis (§5.1). */
    private tileSizeFor(light: LightRecord, viewIndex: number): number {
        const atlas = this.atlas!;
        const existing = atlas.getView(viewIndex);
        const imp = this.importanceOf(light, viewIndex);
        if (existing) {
            const { size, changed } = resolveTileSize(existing.size, imp, undefined, atlas.maxTileSize);
            if (changed) {
                // A resize is a full re-render and a visible quality step (§5.1) — free the old
                // tile, allocate the new, and mark dirty.
                atlas.freeView(viewIndex);
                return size;
            }
            return existing.size;
        }
        return Math.min(TILE_LADDER[0], atlas.maxTileSize);
    }

    /** Ensure a view has a tile of `size`; allocate (evicting if needed) when missing. Returns
     *  the atlas UV rect. Cascades are pinned (§5.3 rule 2). */
    private ensureTile(viewIndex: number, size: number, pinned: boolean, lightName: string): [number, number, number, number] {
        const atlas = this.atlas!;
        const existing = atlas.getView(viewIndex);
        if (existing && existing.size === size) return existing.rect;
        if (existing) atlas.freeView(viewIndex);

        let rec = atlas.allocateView(viewIndex, size, pinned, this.frame);
        // Evict least-important non-pinned views until the tile fits (§5.2).
        while (!rec) {
            const evicted = atlas.evictLeastImportant(
                (vi) => this.importanceOfView(vi),
                (vi) => this.nameOfView(vi));
            if (evicted < 0) {
                const stats = atlas.report();
                throw new Error(
                    `ShadowAtlas: cannot place a ${size}px tile — ${stats.freePx}px² free but ` +
                    `fragmented across ${stats.freeRegions} regions; lower maxTileSize or raise atlasSize.`);
            }
            rec = atlas.allocateView(viewIndex, size, pinned, this.frame);
        }
        return rec.rect;
    }

    /**
     * Importance of a resident view, for eviction (§5.2). Shares `importanceOf` — and therefore
     * this frame's real camera — with tile sizing.
     */
    private importanceOfView(viewIndex: number): number {
        for (const light of this.lights) {
            if (viewIndex >= light.firstView && viewIndex < light.firstView + light.viewCount) {
                return this.importanceOf(light, viewIndex);
            }
        }
        return 0;
    }

    /** The light name owning a view, for the eviction report (§5.2). */
    private nameOfView(viewIndex: number): string {
        for (const light of this.lights) {
            if (viewIndex >= light.firstView && viewIndex < light.firstView + light.viewCount) {
                return light.name;
            }
        }
        return `<view ${viewIndex}>`;
    }

    // ---------------------------------------------------------------------------
    // Phase 2 per-view-slot draw state (§6.1, §8)
    // ---------------------------------------------------------------------------

    /** Total view slots the registered lights actually occupy. */
    private get usedViewSlots(): number {
        let n = 0;
        for (const l of this.lights) n = Math.max(n, l.firstView + l.viewCount);
        return n;
    }

    /**
     * Materialize view-slot draw state up to `count` slots. Each slot gets a tile-clear batch
     * (§8.2) plus one caster batch per registered mesh. Idempotent and monotonic, so it is safe
     * to call from both `registerLight` and `registerCaster` in either order — whichever runs
     * second fills in what the first could not know.
     */
    private ensureViewSlots(count: number): void {
        // No casters means nothing draws into a tile, so a slot has no batches to build — not
        // even a tile-clear, which exists only to make room for casters. A lights-only system
        // therefore touches neither the atlas pass nor the pipeline cache.
        if (!this.atlas || this.casterMeshes.length === 0) return;
        const target = Math.min(count, this.atlas.maxViews);
        if (target <= this.materializedSlots) return;
        const pass = this.getAtlasPass();
        for (let v = this.materializedSlots; v < target; v++) {
            // §8.2 — the per-tile clear: one full-viewport triangle at z=1.0, drawn first,
            // scoped to the tile's viewport. Only drawn when the view is dirty (update() zeroes
            // its instance count otherwise). Not indirect — a plain 3-vertex draw.
            const clearBatch = this.engine.createBatch(pass, {
                shaderCode: TILE_CLEAR_WGSL,
                strideFloats: 1,
                maxInstances: 1,
                isIndirect: false,
                vertexLayouts: [],
                depthOnly: true,
                depthWriteEnabled: true,
                depthCompare: 'always',           // overwrite whatever the tile held
                cameraBuffer: this.atlas.cameraBuffers[v],
                viewport: [0, 0, 1, 1],           // retargeted every frame in update()
            });
            clearBatch.currentInstanceCount = 0;   // suppressed until a tile is dirty
            const slot: ViewSlotDraw = { batches: [], cullStages: [], clearBatch };
            this.viewSlots.push(slot);
            // Caster batches for every mesh registered so far. A mesh registered later appends
            // to each existing slot (see registerCaster), so the two orders converge.
            if (this.config.visibility) {
                for (const m of this.casterMeshes) {
                    const b = this.createSlotCasterBatch(v, m.meshName, m.meshIndex);
                    slot.batches.push(b);
                    slot.cullStages.push(this.config.visibility.getCullStage(b));
                }
            }
        }
        this.materializedSlots = target;
    }

    /** One caster batch for view slot `v` — created at registration, retargeted every frame. */
    private createSlotCasterBatch(v: number, meshName: string, meshIndex: number): RenderBatch {
        const vis = this.config.visibility!;
        const pass = this.getAtlasPass();
        const batch = vis.createInstanceDrawBatch(pass, meshName, meshIndex, {
            camera: this.atlas!.cameraBuffers[v],
            occlusion: this.config.lightViewOcclusion ?? false,
            shaderCode: buildShadowCasterWGSL({
                shadowGroupIndex: this.config.shadowGroupIndex,
                colorFallback: this.config.allowColorFallback,
            }),
            vertexLayouts: StandardLayout.getWebGPUDescriptor(),
            depthOnly: !this.config.allowColorFallback,
            targetFormat: this.config.allowColorFallback ? 'r8unorm' : undefined,
            viewport: [0, 0, 1, 1],               // retargeted every frame in update()
        });
        this.engine.setRenderPassBindGroup(batch, [
            { binding: 3, resource: { buffer: this.casterParamsBuffers[v] } },
        ], this.config.shadowGroupIndex);
        batch.currentInstanceCount = 0;           // suppressed until the slot has a dirty view
        return batch;
    }

    /**
     * Per frame (atlas path): point every view slot's batches at the slot's current tile, and
     * suppress the slots whose view is empty or clean (§8 caching). A dirty view first draws its
     * tile-clear batch, then its casters. `every-frame` mode marks everything dirty, so nothing
     * is ever suppressed — the always-correct default (§8.1).
     */
    private retargetViewSlots(): void {
        const atlas = this.atlas!;
        if (this.viewSlots.length === 0) return;
        const updateMode = this.config.updateMode ?? 'every-frame';
        for (let v = 0; v < this.viewSlots.length; v++) {
            const slot = this.viewSlots[v];
            const rec = atlas.getView(v);
            const active = rec !== null && (updateMode === 'every-frame' || rec.dirty);
            // An inactive slot (no resident view, or a clean cached one) draws nothing. The
            // caster batches are indirect — their instance count is refilled by the cull stage
            // every frame, so we can't suppress them by zeroing drawArgs (the cull would win the
            // race). A zero-area viewport rasterizes nothing regardless of instance count, which
            // is the clean suppression that doesn't fight the GPU-side cull.
            const vp: [number, number, number, number] = (rec && active) ? rec.viewport : [0, 0, 0, 0];
            for (const batch of slot.batches) {
                batch.viewport = vp;
            }
            // ...and gate the culls themselves. The viewport stops rasterization but not the
            // compute: without this, a fully cached atlas still dispatched one cull per
            // (slot × caster mesh) per frame, each over the whole instance buffer — the bulk of
            // the cost caching exists to remove (§8.3).
            for (const stage of slot.cullStages) {
                if (stage) stage.enabled = active;
            }
            if (slot.clearBatch) {
                // The tile-clear is a plain (non-indirect) 3-vertex draw, so it obeys
                // currentInstanceCount directly — no cull stage feeds it.
                slot.clearBatch.viewport = (rec && active) ? rec.viewport : [0, 0, 0, 0];
                slot.clearBatch.currentInstanceCount = active ? 1 : 0;
            }
            if (rec && active) atlas.markClean(v, this.frame);
        }
    }

    /** Write the ShadowParams uniform (atlas path). */
    private writeShadowParams(): void {
        const atlas = this.atlas!;
        const p = this.shadowParamsF32;
        const u = this.shadowParamsU32;
        p[0] = atlas.atlasSize;
        u[1] = this.config.pcfRadius;
        p[2] = this.config.constantBias;
        p[3] = 1.0;                   // 1 texel step per PCF tap (pixel-accurate alignment for 3x3 & 5x5)
        // Cascade splits for selectCascade (directional lights only).
        const dirLight = this.lights.find(l => l.kind === SHADOW_SLOT_KIND_DIRECTIONAL);
        if (dirLight) {
            for (let k = 0; k < this.config.cascades; k++) {
                p[4 + k] = this.viewMeta[dirLight.firstView + k]?.splitFar ?? 0;
            }
        }
        u[8] = this.config.cascades;
        p[9] = this.config.blendTexels ?? 6.0;
        this.device.queue.writeBuffer(this.shadowParamsBuffer!, 0, p as Float32Array<ArrayBuffer>);
    }

    /**
     * The worker hand-off (§9.2): adopt a cascade array computed off-thread. Phase 1 only.
     */
    public setRawCascades(views: Float32Array, shadowData: Float32Array): void {
        const n = this.config.cascades;
        if (views.length < n * 108 || shadowData.length < SHADOW_DATA_FLOATS) {
            throw new Error(
                `ShadowSystem: setRawCascades expects ${n * 108} view floats and ` +
                `${SHADOW_DATA_FLOATS} ShadowData floats (got ${views.length} / ${shadowData.length}).`);
        }
        const q = this.device.queue;
        for (let k = 0; k < n; k++) {
            q.writeBuffer(this.cascadeBuffers[k], 0, views.buffer, views.byteOffset + k * 432, 432);
        }
        q.writeBuffer(this.shadowDataBuffer, 0, shadowData.buffer, shadowData.byteOffset, SHADOW_DATA_FLOATS * 4);
        this.cascades = [];
    }

    // ---------------------------------------------------------------------------
    // Casters (§9.3)
    // ---------------------------------------------------------------------------

    /**
     * Register one caster mesh. Phase 1: `cascades` depth-only indirect batches, one per cascade
     * layer. Phase 2: one depth-only indirect batch per resident view, all in the single atlas
     * pass, each scoped to its tile's viewport (§6.1).
     */
    public registerCaster(meshName: string, meshIndex: number): void {
        const vis = this.config.visibility;
        if (!vis) {
            throw new Error('ShadowSystem: registerCaster requires config.visibility; Phase 1 has no CPU caster path.');
        }
        const record: CasterRecord = { meshName, meshIndex, batches: [] };

        if (this.atlas) {
            // ----- Phase 2: one batch per USED view slot, in the single atlas pass -----
            // Tiles are allocated in update(), which runs AFTER registration — so batches are
            // created per SLOT, and update() retargets each slot's viewport and suppresses
            // clean/empty slots every frame (§8 caching).
            this.casterMeshes.push({ meshName, meshIndex });
            // Slots already materialized by earlier registerLight calls need this new mesh...
            for (let v = 0; v < this.viewSlots.length; v++) {
                const batch = this.createSlotCasterBatch(v, meshName, meshIndex);
                this.viewSlots[v].batches.push(batch);
                this.viewSlots[v].cullStages.push(vis.getCullStage(batch));
                record.batches.push(batch);
            }
            // ...and any lights registered so far but not yet materialized (caster-first order)
            // pick it up here, since casterMeshes now contains it.
            const before = this.viewSlots.length;
            this.ensureViewSlots(this.usedViewSlots);
            for (let v = before; v < this.viewSlots.length; v++) {
                record.batches.push(...this.viewSlots[v].batches);
            }
        } else {
            // ----- Phase 1: one batch per cascade layer -----
            for (let k = 0; k < this.config.cascades; k++) {
                const pass = this.getOrCreatePass(k);
                const batch = vis.createInstanceDrawBatch(pass, meshName, meshIndex, {
                    camera: this.cascadeBuffers[k],
                    occlusion: false,
                    shaderCode: buildShadowCasterWGSL({
                        shadowGroupIndex: this.config.shadowGroupIndex,
                        colorFallback: this.config.allowColorFallback,
                    }),
                    vertexLayouts: StandardLayout.getWebGPUDescriptor(),
                    depthOnly: !this.config.allowColorFallback,
                    targetFormat: this.config.allowColorFallback ? 'r8unorm' : undefined,
                });
                this.engine.setRenderPassBindGroup(batch, [
                    { binding: 3, resource: { buffer: this.casterParamsBuffers[k] } },
                ], this.config.shadowGroupIndex);
                record.batches.push(batch);
            }
        }
        this.casters.push(record);
    }

    // ---------------------------------------------------------------------------
    // Frame graph (§2)
    // ---------------------------------------------------------------------------

    /**
     * The per-view caster cull stages. Phase 1: empty by design (the caster culls ride the
     * VisibilitySystem's own chain). Phase 2: the light-view Hi-Z build stages, when enabled —
     * the per-view culls themselves still ride the VisibilitySystem's chain.
     */
    public getComputeStages(): ComputeStageNode[] {
        if (this.lightHiZ) return this.lightHiZ.getBuildStages();
        return [];
    }

    /** Phase 1: the N depth-only render passes, one per cascade layer. Call after `update()`. */
    public getPasses(): RenderPassNode[] {
        if (this.atlas) {
            throw new Error('ShadowSystem: getPasses() is the Phase 1 path; use getAtlasPass() in the atlas path.');
        }
        if (!this.cascades) {
            throw new Error('ShadowSystem: getPasses() called before update(camera); cascade matrices are unset.');
        }
        return this.passes;
    }

    /** Phase 2: the ONE atlas pass holding every view's draws (§6.1). Call after `update()`. */
    public getAtlasPass(): RenderPassNode {
        if (!this.atlas) {
            throw new Error('ShadowSystem: getAtlasPass() requires the Phase 2 atlas path (set config.atlasSize).');
        }
        if (this.atlasPass) return this.atlasPass;
        const updateMode = this.config.updateMode ?? 'every-frame';
        this.atlasPass = this.engine.createPass({
            name: 'Shadow Atlas',
            isMainScreenPass: false,
            // Every batch here is viewport-scoped, so no restore is needed — but declaring the
            // size keeps that true if a caller ever adds a full-atlas batch of their own.
            size: [this.atlas.atlasSize, this.atlas.atlasSize],
            // Caching (§8.2): 'load' preserves clean tiles; dirty tiles clear themselves with a
            // tile-clear batch. 'every-frame' mode clears the whole atlas once. The caching modes
            // still start with 'clear' on frame 1 — a clean tile must read far-plane depth, not
            // uninitialized memory; update() flips it to 'load' after the first clear.
            depthStencilAttachment: {
                view: this.atlas.samplingView,
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        });
        return this.atlasPass;
    }

    /**
     * Flip the atlas pass to `depthLoadOp: 'load'` after the first frame's full clear, so the
     * caching modes preserve clean tiles (§8.2). Called from update() every frame; a no-op once
     * the flip has happened. `every-frame` mode stays on 'clear' forever (it never caches).
     */
    private settleAtlasLoadOp(): void {
        if (this.atlasClearedOnce || !this.atlasPass?.depthStencilAttachment) return;
        this.atlasClearedOnce = true;
        const updateMode = this.config.updateMode ?? 'every-frame';
        if (updateMode !== 'every-frame') {
            this.atlasPass.depthStencilAttachment.depthLoadOp = 'load';
        }
    }

    // ---------------------------------------------------------------------------
    // Receiver wiring (§8 Phase 1 / §9 Phase 2)
    // ---------------------------------------------------------------------------

    /**
     * Decorate a LightSystem's shader layout so every stock material gains shadows with zero
     * edits. Phase 1 folds a cascade lookup into `IncidentLight.attenuation`; Phase 2 replaces
     * the direction match with a slot lookup (§9.2) — `spot_params.z` carries `slot + 1`.
     */
    public wrapLighting(lights: LightSystem): ShaderLightingLayout {
        const inner = lights.getShaderLayout();
        if (!inner) {
            throw new Error(
                `ShadowSystem: wrapLighting requires a LightSystem with technique "standard" or ` +
                `"clustered-forward"; "${lights.config.technique}" returns no shader layout.`);
        }
        const marker = 'fn getIncidentLight(';
        if (!inner.wgslLogic.includes(marker)) {
            throw new Error(
                'ShadowSystem: LightSystem WGSL no longer defines getIncidentLight(); the shadow ' +
                'decorator is out of date with LightSystem.getShaderLayout().');
        }
        const accessor = lights.config.technique === 'clustered-forward'
            ? 'globalLights[lightIndices[getClusterData(worldPos).start_offset + i]]'
            : 'lights[i]';

        if (this.atlas) {
            // ----- Phase 2: slot → view → tap -----
            return {
                wgslStructs: inner.wgslStructs + ATLAS_STRUCTS_WGSL + ATLAS_PARAMS_WGSL
                    + buildAtlasBindingsWGSL(this.config.shadowGroupIndex),
                wgslLogic: inner.wgslLogic.replace(marker, 'fn getIncidentLightUnshadowed(')
                    + ATLAS_SAMPLING_WGSL + buildAtlasDecoratorWGSL(accessor),
            };
        }
        // ----- Phase 1: cascade array + direction match -----
        return {
            wgslStructs: inner.wgslStructs + SHADOW_STRUCTS_WGSL
                + buildShadowBindingsWGSL(this.config.shadowGroupIndex),
            wgslLogic: inner.wgslLogic.replace(marker, 'fn getIncidentLightUnshadowed(')
                + SHADOW_SAMPLING_WGSL + SHADOW_BIASED_WGSL + buildShadowDecoratorWGSL(accessor),
        };
    }

    /**
     * Bind the shadow resources onto a receiver batch at the shadow group. Phase 1 binds the
     * cascade array + ShadowData; Phase 2 binds the atlas + the view/slot tables (§9.5).
     */
    public applyToBatch(batch: RenderBatch): void {
        if (this.atlas) {
            this.engine.setRenderPassBindGroup(batch, [
                { binding: 0, resource: this.atlas.samplingView },
                { binding: 1, resource: this.atlas.comparisonSampler },
                { binding: 2, resource: { buffer: this.shadowParamsBuffer! } },
                { binding: 3, resource: { buffer: this.atlas.viewsBuffer } },
                { binding: 4, resource: { buffer: this.atlas.slotsBuffer } },
            ], this.config.shadowGroupIndex);
            return;
        }
        this.engine.setRenderPassBindGroup(batch, [
            { binding: 0, resource: this.samplingViewP1 },
            { binding: 1, resource: this.comparisonSamplerP1 },
            { binding: 2, resource: { buffer: this.shadowDataBuffer } },
        ], this.config.shadowGroupIndex);
    }

    // ---------------------------------------------------------------------------
    // Internals
    // ---------------------------------------------------------------------------

    /** The depth-only pass for cascade layer `k`, created on first use (Phase 1 §6.1). */
    private getOrCreatePass(k: number): RenderPassNode {
        if (this.passes[k]) return this.passes[k];
        const pass = this.engine.createPass({
            name: `Shadow Cascade ${k}`,
            isMainScreenPass: false,
            size: [this.config.resolution, this.config.resolution],
            depthStencilAttachment: {
                view: this.layerViews[k],
                depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store',
            },
        });
        this.passes[k] = pass;
        return pass;
    }

    /** The atlas's sampling view (texture_depth_2d) — for custom receiver wiring (Phase 2). */
    public get shadowMapView(): GPUTextureView {
        return this.atlas ? this.atlas.samplingView : this.samplingViewP1;
    }
    /** The comparison sampler (hardware 2×2 PCF). */
    public get sampler(): GPUSampler {
        return this.atlas ? this.atlas.comparisonSampler : this.comparisonSamplerP1;
    }
    /** The ShadowData uniform buffer (Phase 1) / ShadowParams uniform buffer (Phase 2). */
    public get dataBuffer(): GPUBuffer {
        return this.atlas ? this.shadowParamsBuffer! : this.shadowDataBuffer;
    }
    /** The fitted cascades from the last `update()` — null until the first call (Phase 1). */
    public get fittedCascades(): readonly CascadeResult[] | null { return this.cascades; }
    /** The atlas (Phase 2). Null in the Phase 1 path. */
    public get shadowAtlas(): ShadowAtlas | null { return this.atlas; }

    public destroy(): void {
        this.cascadeArray?.destroy();
        for (const b of this.cascadeBuffers) b.destroy();
        for (const b of this.casterParamsBuffers) b.destroy();
        this.shadowDataBuffer?.destroy();
        this.atlas?.destroy();
        this.shadowParamsBuffer?.destroy();
        this.lightHiZ?.destroy();
        this.cascades = null;
    }
}
