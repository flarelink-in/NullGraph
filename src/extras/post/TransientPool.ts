// src/extras/post/TransientPool.ts
import type { TargetHandle, TargetSpec } from './types';

/**
 * Lifetime-analysis target pool. This is the load-bearing piece: get it right
 * and everything above it is bookkeeping.
 *
 * Two-phase, like a FrameGraph setup pass:
 *   1. alloc()/markWrite()/markRead() during declare — records intent, allocates nothing
 *   2. resolve() once all passes are known — computes lifetimes, aliases, creates textures
 *
 * "Aliasing" in WebGPU means exactly one thing: two logical targets are
 * assigned the same GPUTexture object. There are no heaps, no placed
 * resources, no barriers — descriptors must match byte-identically, and
 * contents are not preserved across an alias boundary.
 */

// Numeric literals (per the WebGPU spec) rather than GPUTextureUsage.* so the
// module can be imported — and the allocator unit-tested — outside a WebGPU
// environment. TEXTURE_BINDING = 0x04, RENDER_ATTACHMENT = 0x10.
const DEFAULT_USAGE: GPUTextureUsageFlags = 0x10 | 0x04;

/** depth24plus has an implementation-defined layout; 4 is the accepted estimate. */
const BYTES_PER_PIXEL: Record<string, number> = {
    'rgba16float': 8,
    'rgba8unorm': 4,
    'rgba8unorm-srgb': 4,
    'rg16float': 4,
    'r8unorm': 1,
    'rg11b10ufloat': 4,
    'depth24plus': 4,
    'depth32float': 4,
    'r16float': 2,
};

function bytesPerPixel(format: GPUTextureFormat): number {
    return BYTES_PER_PIXEL[format] ?? 4;
}

function isDepthFormat(format: GPUTextureFormat): boolean {
    return format.startsWith('depth');
}

interface InternalHandle extends TargetHandle {
    /** Pool generation stamp — detects stale handles after resize()/reset(). */
    readonly gen: number;
}

interface LogicalTarget {
    id: number;
    debugName: string;
    width: number;
    height: number;
    format: GPUTextureFormat;
    usage: GPUTextureUsageFlags;
    history: 1 | 2;
    firstTouch: number;
    lastTouch: number;
    written: boolean;
    firstWrite: number;
    physicalIndex: number;
}

interface PhysicalTexture {
    key: string;
    texture: GPUTexture;
    view: GPUTextureView;
    maxLastTouch: number;
    byteSize: number;
    tenants: number;
}

interface HistoryPair {
    targetId: number;
    a: { texture: GPUTexture; view: GPUTextureView };
    b: { texture: GPUTexture; view: GPUTextureView };
    parity: number;
    byteSize: number;
}

interface PassMeta {
    index: number;
    name: string;
    inputs: number[];
    outputs: number[];
    loadOp: GPULoadOp;
}

export interface PoolReport {
    physicalTextures: number;
    logicalTargets: number;
    bytes: number;
    bytesWithoutAliasing: number;
}

export interface TargetInfo {
    width: number;
    height: number;
    format: GPUTextureFormat;
    usage: GPUTextureUsageFlags;
}

export class TransientPool {
    private phase: 'declare' | 'resolved' = 'declare';
    private generation = 0;

    private targets: LogicalTarget[] = [];
    private physical: PhysicalTexture[] = [];
    private historyPairs = new Map<number, HistoryPair>();
    private passMeta = new Map<number, PassMeta>();

    constructor(
        private device: GPUDevice,
        private baseWidth: number,
        private baseHeight: number,
        private defaultFormat: GPUTextureFormat,
    ) {}

    // =========================================================
    // DECLARE phase
    // =========================================================

    public alloc(spec: TargetSpec, debugName: string): TargetHandle {
        this.assertPhase('declare', 'alloc');
        const scale = spec.scale ?? 1.0;
        // floor + clamp-to-1: a bloom pyramid must yield the same size for the
        // same scale regardless of parity, and must never reach 0.
        const width = Math.max(1, Math.floor(this.baseWidth * scale));
        const height = Math.max(1, Math.floor(this.baseHeight * scale));

        const target: LogicalTarget = {
            id: this.targets.length,
            debugName,
            width,
            height,
            format: (spec.format as GPUTextureFormat) ?? this.defaultFormat,
            usage: DEFAULT_USAGE | (spec.extraUsage ?? 0),
            history: spec.history ?? 1,
            firstTouch: -1,
            lastTouch: -1,
            written: false,
            firstWrite: -1,
            physicalIndex: -1,
        };
        this.targets.push(target);

        const handle: InternalHandle = { id: target.id, debugName, gen: this.generation };
        return handle;
    }

    public markWrite(h: TargetHandle, passIndex: number, info?: { passName?: string; loadOp?: GPULoadOp }): void {
        this.assertPhase('declare', 'markWrite');
        const t = this.getTarget(h);
        if (!t.written || passIndex < t.firstWrite) t.firstWrite = passIndex;
        t.written = true;
        t.firstTouch = t.firstTouch < 0 ? passIndex : Math.min(t.firstTouch, passIndex);
        t.lastTouch = Math.max(t.lastTouch, passIndex);

        const meta = this.meta(passIndex, info?.passName);
        meta.outputs.push(t.id);
        if (info?.loadOp) meta.loadOp = info.loadOp;
    }

    public markRead(h: TargetHandle, passIndex: number, info?: { passName?: string }): void {
        this.assertPhase('declare', 'markRead');
        const t = this.getTarget(h);
        t.firstTouch = t.firstTouch < 0 ? passIndex : Math.min(t.firstTouch, passIndex);
        t.lastTouch = Math.max(t.lastTouch, passIndex);

        this.meta(passIndex, info?.passName).inputs.push(t.id);
    }

    // =========================================================
    // Transition
    // =========================================================

    public resolve(): void {
        this.assertPhase('declare', 'resolve');

        // ---- V2 / V1 ----
        for (const t of this.targets) {
            if (t.firstTouch < 0) {
                throw new Error(`PoolError: target "${t.debugName}" was allocated but never used.`);
            }
            if (!t.written || t.firstWrite !== t.firstTouch) {
                throw new Error(`PoolError: target "${t.debugName}" is read at pass ${t.firstTouch} before it is written.`);
            }
        }

        // ---- History targets: two physical textures each, never aliased ----
        for (const t of this.targets) {
            if (t.history !== 2) continue;
            const make = () => {
                const texture = this.device.createTexture({
                    label: `Pool_${t.debugName}_history`,
                    size: [t.width, t.height],
                    format: t.format,
                    usage: t.usage,
                });
                return { texture, view: texture.createView() };
            };
            this.historyPairs.set(t.id, {
                targetId: t.id,
                a: make(),
                b: make(),
                parity: 0,
                byteSize: t.width * t.height * bytesPerPixel(t.format),
            });
        }
        // Frame 0 must read zeroes, not uninitialised VRAM.
        if (this.historyPairs.size > 0) {
            const enc = this.device.createCommandEncoder();
            for (const pair of this.historyPairs.values()) {
                const fmt = this.targets[pair.targetId].format;
                for (const side of [pair.a, pair.b]) {
                    if (isDepthFormat(fmt)) {
                        enc.beginRenderPass({
                            colorAttachments: [],
                            depthStencilAttachment: {
                                view: side.view,
                                depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store',
                            },
                        }).end();
                    } else {
                        enc.beginRenderPass({
                            colorAttachments: [{
                                view: side.view,
                                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                                loadOp: 'clear', storeOp: 'store',
                            }],
                        }).end();
                    }
                }
            }
            this.device.queue.submit([enc.finish()]);
        }

        // ---- Transient targets: linear-scan assignment ----
        const transient = this.targets
            .filter(t => t.history === 1)
            .sort((a, b) => (a.firstTouch - b.firstTouch) || (a.id - b.id));

        for (const t of transient) {
            const key = `${t.width}x${t.height}|${t.format}|${t.usage}|1|2d`;
            // THE PREDICATE: strictly less-than. With <=, a pass that reads B0
            // and writes U0 at the same index would alias them, and a single
            // pass would sample the texture it is rendering into.
            const p = this.physical.find(p => p.key === key && p.maxLastTouch < t.firstTouch);
            if (p) {
                t.physicalIndex = this.physical.indexOf(p);
                p.maxLastTouch = t.lastTouch;
                p.tenants++;
            } else {
                const texture = this.device.createTexture({
                    label: `Pool_${t.debugName}`,
                    size: [t.width, t.height],
                    format: t.format,
                    usage: t.usage,
                });
                this.physical.push({
                    key,
                    texture,
                    view: texture.createView(),
                    maxLastTouch: t.lastTouch,
                    byteSize: t.width * t.height * bytesPerPixel(t.format),
                    tenants: 1,
                });
                t.physicalIndex = this.physical.length - 1;
            }
        }

        // ---- V4 ----
        for (const t of this.targets) {
            if (t.history === 2 && t.physicalIndex !== -1) {
                throw new Error(`PoolError: internal assertion — history target "${t.debugName}" entered the shared pool.`);
            }
        }

        // ---- V3: no pass samples and renders to the same physical texture ----
        for (const meta of this.passMeta.values()) {
            for (const outId of meta.outputs) {
                const out = this.targets[outId];
                if (out.history === 2) continue; // view/historyView are distinct textures
                for (const inId of meta.inputs) {
                    const inp = this.targets[inId];
                    if (inp.history === 2) continue;
                    if (inp.physicalIndex === out.physicalIndex) {
                        throw new Error(
                            `PoolError: pass ${meta.index} "${meta.name}" would sample and render to the same ` +
                            `physical texture (targets "${inp.debugName}" and "${out.debugName}").`,
                        );
                    }
                }
            }
        }

        // ---- V5: loadOp 'load' requires a single-tenant target ----
        for (const meta of this.passMeta.values()) {
            if (meta.loadOp !== 'load') continue;
            for (const outId of meta.outputs) {
                const out = this.targets[outId];
                if (out.history === 2) continue;
                if (this.physical[out.physicalIndex].tenants > 1) {
                    throw new Error(
                        `PoolError: pass ${meta.index} uses loadOp 'load' on aliased target "${out.debugName}"; ` +
                        `contents are not preserved across aliasing.`,
                    );
                }
            }
        }

        this.phase = 'resolved';
    }

    // =========================================================
    // RESOLVED phase
    // =========================================================

    public view(h: TargetHandle): GPUTextureView {
        this.assertPhase('resolved', 'view');
        const t = this.getTarget(h);
        if (t.history === 2) {
            const pair = this.historyPairs.get(t.id)!;
            return pair.parity === 0 ? pair.a.view : pair.b.view;
        }
        return this.physical[t.physicalIndex].view;
    }

    public historyView(h: TargetHandle): GPUTextureView {
        this.assertPhase('resolved', 'historyView');
        const t = this.getTarget(h);
        if (t.history !== 2) {
            throw new Error(`PoolError: historyView() on target "${t.debugName}", which was not declared history: 2.`);
        }
        const pair = this.historyPairs.get(t.id)!;
        return pair.parity === 0 ? pair.b.view : pair.a.view;
    }

    public advanceFrame(): void {
        this.assertPhase('resolved', 'advanceFrame');
        for (const pair of this.historyPairs.values()) pair.parity ^= 1;
    }

    /** Dimensions/format of a logical target. Legal in either phase. */
    public info(h: TargetHandle): TargetInfo {
        const t = this.getTarget(h);
        return { width: t.width, height: t.height, format: t.format, usage: t.usage };
    }

    public report(): PoolReport {
        this.assertPhase('resolved', 'report');
        let bytes = 0;
        for (const p of this.physical) bytes += p.byteSize;
        for (const pair of this.historyPairs.values()) bytes += pair.byteSize * 2;

        let naive = 0;
        for (const t of this.targets) {
            naive += t.width * t.height * bytesPerPixel(t.format) * (t.history === 2 ? 2 : 1);
        }
        return {
            physicalTextures: this.physical.length + this.historyPairs.size * 2,
            logicalTargets: this.targets.length,
            bytes,
            bytesWithoutAliasing: naive,
        };
    }

    // =========================================================
    // Teardown / rebuild
    // =========================================================

    /** Destroys all textures and invalidates every handle. Back to DECLARE. */
    public resize(width: number, height: number): void {
        this.baseWidth = width;
        this.baseHeight = height;
        this.reset();
    }

    public reset(): void {
        for (const p of this.physical) p.texture.destroy();
        for (const pair of this.historyPairs.values()) {
            pair.a.texture.destroy();
            pair.b.texture.destroy();
        }
        this.physical = [];
        this.historyPairs.clear();
        this.targets = [];
        this.passMeta.clear();
        this.generation++;
        this.phase = 'declare';
    }

    public destroy(): void {
        this.reset();
    }

    // =========================================================
    // Internals
    // =========================================================

    private assertPhase(expected: 'declare' | 'resolved', method: string): void {
        if (this.phase !== expected) {
            throw new Error(`PoolError: ${method}() is only legal in the '${expected}' phase (current: '${this.phase}').`);
        }
    }

    private getTarget(h: TargetHandle): LogicalTarget {
        const ih = h as InternalHandle;
        if (ih.gen !== this.generation || ih.id < 0 || ih.id >= this.targets.length) {
            throw new Error(`PoolError: stale handle "${h.debugName}" — invalidated by resize()/reset().`);
        }
        return this.targets[ih.id];
    }

    private meta(passIndex: number, name?: string): PassMeta {
        let m = this.passMeta.get(passIndex);
        if (!m) {
            m = { index: passIndex, name: name ?? `pass_${passIndex}`, inputs: [], outputs: [], loadOp: 'clear' };
            this.passMeta.set(passIndex, m);
        } else if (name) {
            m.name = name;
        }
        return m;
    }
}
