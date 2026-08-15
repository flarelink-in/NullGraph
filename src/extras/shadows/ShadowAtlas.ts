// src/extras/shadows/ShadowAtlas.ts
//
// The ShadowAtlas (docs/shadows-atlas-design.md §4, §5, §8): the single fixed depth allocation
// the engine partitions by importance, plus the view/slot/camera tables the receiver reads.
//
// One `depth24plus` atlas texture, allocated once. Tiles are power-of-two squares handed out by
// a quad-tree free-list (atlasMath.ts). A view keeps its tile across frames unless resized or
// evicted (§5.3), so a cached tile stays valid. Eviction fails OPEN — an evicted view's light
// samples as fully lit, never black (§5.2) — and is reported through `report()`, never silent.
//
// Nothing here is allocated per frame; the per-frame writes are `maxViews × 432` bytes of camera
// data and one `views`/`slots` refresh, all from persistent typed arrays.

import { NullGraph } from '../../core';
import {
    QuadTreeAllocator, TILE_LADDER, SHADOW_VIEW_RECORD_FLOATS, SHADOW_SLOT_RECORD_FLOATS,
    packShadowView, packShadowSlot,
} from './atlasMath';

/** One resident view's atlas bookkeeping (§4.1: "allocation bookkeeping — tile index,
 *  resolution, dirty, last-used frame"). */
export interface AtlasViewRecord {
    /** Index into the views array / camera buffers. */
    viewIndex: number;
    /** Tile top-left in atlas pixels. */
    x: number;
    y: number;
    /** Tile edge in px (power of two). */
    size: number;
    /** The atlas UV transform written to the receiver table: `[scaleX, scaleY, offsetX, offsetY]`. */
    rect: [number, number, number, number];
    /** The per-batch viewport `[x, y, w, h]` in atlas pixels (§12.1). */
    viewport: [number, number, number, number];
    /** Re-render this frame? (§8.1 — dirtiness is declared, not inferred.) */
    dirty: boolean;
    /** The frame this view was last rendered (for round-robin + eviction LRU tie-breaks). */
    lastUsedFrame: number;
    /** Cascades are pinned (§5.3 rule 2): never evicted, allocated first. */
    pinned: boolean;
}

export interface ShadowAtlasConfig {
    /** Atlas edge in px (square, power of two). Default 4096 (§4.4: ≈32 MB at depth24plus). */
    atlasSize?: number;
    /** Largest tile a single view may request. Default 2048 (the top of the ladder). */
    maxTileSize?: number;
    /** Hard cap on views (cascades + spots + 6×points). Default 64 (§4.4). */
    maxViews?: number;
    /** Hard cap on shadow-casting lights. Default 64. */
    maxSlots?: number;
}

export class ShadowAtlas {
    private readonly device: GPUDevice;
    public readonly atlasSize: number;
    public readonly maxTileSize: number;
    public readonly maxViews: number;
    public readonly maxSlots: number;

    // --- GPU resources, allocated once (§4.4) ---
    public readonly texture: GPUTexture;
    public readonly samplingView: GPUTextureView;
    public readonly comparisonSampler: GPUSampler;
    public readonly viewsBuffer: GPUBuffer;
    public readonly slotsBuffer: GPUBuffer;
    public readonly cameraBuffers: GPUBuffer[] = [];

    // --- CPU state ---
    private readonly allocator: QuadTreeAllocator;
    private readonly views: (AtlasViewRecord | null)[] = [];
    private viewCount: number = 0;
    private slotCount: number = 0;

    // --- per-frame staging (persistent — no allocation in the frame loop) ---
    private readonly viewsF32: Float32Array;
    private readonly slotsF32: Float32Array;

    // --- profiler surface (§5.2: eviction is reported, never silent) ---
    private evictionCount: number = 0;
    private lastEvictedName: string = '';

    constructor(engine: NullGraph, config: ShadowAtlasConfig = {}) {
        this.device = engine.device;
        this.atlasSize = config.atlasSize ?? 4096;
        this.maxTileSize = config.maxTileSize ?? TILE_LADDER[0];
        this.maxViews = config.maxViews ?? 64;
        this.maxSlots = config.maxSlots ?? 64;

        if (this.maxTileSize > this.atlasSize) {
            throw new Error(
                `ShadowAtlas: maxTileSize ${this.maxTileSize} exceeds atlasSize ${this.atlasSize}; ` +
                `raise atlasSize or lower maxTileSize.`);
        }

        this.allocator = new QuadTreeAllocator(this.atlasSize, TILE_LADDER[TILE_LADDER.length - 1]);

        // §4.4 — the one fixed allocation. depth24plus is samplable + comparison-filterable.
        this.texture = this.device.createTexture({
            label: 'NullGraph_ShadowAtlas',
            size: [this.atlasSize, this.atlasSize],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.samplingView = this.texture.createView();
        this.comparisonSampler = this.device.createSampler({
            label: 'NullGraph_ShadowAtlasSampler',
            compare: 'less', magFilter: 'linear', minFilter: 'linear',
        });

        this.viewsBuffer = this.device.createBuffer({
            label: 'NullGraph_ShadowViews',
            size: SHADOW_VIEW_RECORD_FLOATS * 4 * this.maxViews,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.slotsBuffer = this.device.createBuffer({
            label: 'NullGraph_ShadowSlots',
            size: SHADOW_SLOT_RECORD_FLOATS * 4 * this.maxSlots,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        for (let i = 0; i < this.maxViews; i++) {
            this.cameraBuffers.push(this.device.createBuffer({
                label: `NullGraph_ShadowViewCamera${i}`,
                size: 432,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }));
            this.views.push(null);
        }

        this.viewsF32 = new Float32Array(SHADOW_VIEW_RECORD_FLOATS * this.maxViews);
        this.slotsF32 = new Float32Array(SHADOW_SLOT_RECORD_FLOATS * this.maxSlots);
    }

    // ---------------------------------------------------------------------------
    // Views — allocation, eviction, dirty tracking
    // ---------------------------------------------------------------------------

    /**
     * Allocate a view's tile. Cascades pass `pinned: true` (§5.3 rule 2 — the sun is not allowed
     * to lose its shadow because forty torches got close to the camera). Returns the record, or
     * `null` when the atlas cannot place the tile (the caller then evicts or errors — §5.2, §11).
     */
    public allocateView(viewIndex: number, size: number, pinned: boolean, frame: number): AtlasViewRecord | null {
        if (viewIndex >= this.maxViews) return null;
        const placed = this.allocator.alloc(size);
        if (!placed) return null;
        const [x, y] = placed;
        const s = size / this.atlasSize;
        const rec: AtlasViewRecord = {
            viewIndex, x, y, size,
            rect: [s, s, x / this.atlasSize, y / this.atlasSize],
            viewport: [x, y, size, size],
            dirty: true,           // a newly-allocated tile is always rendered (§8.1)
            lastUsedFrame: frame,
            pinned,
        };
        this.views[viewIndex] = rec;
        this.viewCount++;
        return rec;
    }

    /** Free a view's tile and mark the slot empty. */
    public freeView(viewIndex: number): void {
        const rec = this.views[viewIndex];
        if (!rec) return;
        this.allocator.free(rec.x, rec.y, rec.size);
        this.views[viewIndex] = null;
        this.viewCount--;
    }

    /** The view record, or null if that index is not resident. */
    public getView(viewIndex: number): AtlasViewRecord | null {
        return this.views[viewIndex];
    }

    /** Mark a view dirty — it re-renders this frame (§8.1). */
    public markDirty(viewIndex: number): void {
        const rec = this.views[viewIndex];
        if (rec) rec.dirty = true;
    }

    /** Mark a view clean after its tile has been rendered. */
    public markClean(viewIndex: number, frame: number): void {
        const rec = this.views[viewIndex];
        if (rec) { rec.dirty = false; rec.lastUsedFrame = frame; }
    }

    /**
     * Evict the least-important non-pinned resident view (§5.2). Cascades are never evicted
     * (§5.3 rule 2). Returns the freed view index, or -1 when nothing is evictable. The caller
     * reports the eviction through `report()` — it is never silent.
     */
    public evictLeastImportant(importanceOf: (viewIndex: number) => number, nameOf: (viewIndex: number) => string): number {
        let worst = -1;
        let worstImp = Infinity;
        let worstFrame = Infinity;
        for (let i = 0; i < this.maxViews; i++) {
            const rec = this.views[i];
            if (!rec || rec.pinned) continue;
            const imp = importanceOf(i);
            // Least importance wins; ties break to the least-recently-used.
            if (imp < worstImp || (imp === worstImp && rec.lastUsedFrame < worstFrame)) {
                worst = i; worstImp = imp; worstFrame = rec.lastUsedFrame;
            }
        }
        if (worst >= 0) {
            this.lastEvictedName = nameOf(worst);
            this.evictionCount++;
            this.freeView(worst);
        }
        return worst;
    }

    /** The number of resident views. */
    public get residentViews(): number { return this.viewCount; }

    /** Allocate a shadow slot. Returns the slot index, or -1 when slots are exhausted. */
    public allocateSlot(): number {
        if (this.slotCount >= this.maxSlots) return -1;
        return this.slotCount++;
    }

    /** The number of live shadow slots. */
    public get residentSlots(): number { return this.slotCount; }

    // ---------------------------------------------------------------------------
    // Per-frame table refresh
    // ---------------------------------------------------------------------------

    /**
     * Write the views + slots storage buffers from the current records. `viewData` is a
     * `maxViews × 108` Float32Array of packed Camera blocks; `slots` is the packed slot records.
     * Called once per frame after all views are built — no allocation, straight from staging.
     */
    public writeTables(
        viewData: Float32Array,
        viewMeta: { rect: [number, number, number, number]; texelWorld: number; splitFar: number; kind: number }[],
        slots: { firstView: number; viewCount: number; kind: number; lightRange: number }[],
    ): void {
        const q = this.device.queue;
        // Cameras: one 432-byte write per resident view.
        for (let i = 0; i < this.maxViews; i++) {
            if (!this.views[i]) continue;
            q.writeBuffer(this.cameraBuffers[i], 0, viewData.buffer,
                viewData.byteOffset + i * 432, 432);
        }
        // Views table.
        for (let i = 0; i < this.maxViews; i++) {
            const rec = this.views[i];
            const o = i * SHADOW_VIEW_RECORD_FLOATS;
            if (!rec) {
                // A non-resident view reads as rect=(0,0,0,0), which sampleView treats as fully
                // lit (§5.2 — eviction fails open). Zero the whole record.
                this.viewsF32.fill(0, o, o + SHADOW_VIEW_RECORD_FLOATS);
                continue;
            }
            const meta = viewMeta[i];
            // The whole staging array plus a float offset — `subarray` would allocate a view per
            // resident view per frame (roadmap constraint #2).
            packShadowView(this.viewsF32, o, viewData,
                rec.rect, meta.texelWorld, meta.splitFar, meta.kind, i * 108);
        }
        q.writeBuffer(this.viewsBuffer, 0, this.viewsF32 as Float32Array<ArrayBuffer>);
        // Slots table.
        for (let i = 0; i < slots.length; i++) {
            packShadowSlot(this.slotsF32, i * SHADOW_SLOT_RECORD_FLOATS,
                slots[i].firstView, slots[i].viewCount, slots[i].kind, slots[i].lightRange);
        }
        q.writeBuffer(this.slotsBuffer, 0, this.slotsF32 as Float32Array<ArrayBuffer>);
    }

    // ---------------------------------------------------------------------------
    // Profiler surface (§5.2)
    // ---------------------------------------------------------------------------

    /** Eviction + residency stats for the profiler. Eviction is reported, never silent. */
    public report(): { residentViews: number; residentSlots: number; evictions: number; lastEvicted: string; freePx: number; freeRegions: number } {
        const stats = this.allocator.freeStats();
        return {
            residentViews: this.viewCount,
            residentSlots: this.slotCount,
            evictions: this.evictionCount,
            lastEvicted: this.lastEvictedName,
            freePx: stats.freePx,
            freeRegions: stats.regions,
        };
    }

    /** The allocator, exposed for the U1 overlap test. */
    public get quadTree(): QuadTreeAllocator { return this.allocator; }

    public destroy(): void {
        this.texture.destroy();
        this.viewsBuffer.destroy();
        this.slotsBuffer.destroy();
        for (const b of this.cameraBuffers) b.destroy();
    }
}
