export interface Brain {
    readonly kind: 'stage' | 'async';
    readonly wgsl?: string;
    readonly buffers?: readonly GPUBuffer[];
    readonly dims?: { obs: number; act: number };
    /** Optional resource-owner check, performed before creating a consuming stage. */
    assertDevice?(device: GPUDevice): void;
    /**
     * Async brains (step 9): called by setBrain before the brain's pass is built. The brain's runs
     * read `obs` and write a buffer of its own, which its `wgsl` (a think pass) and `buffers` then
     * copy into `act`. It may read `act` here, to start from the actions agents have; never write it.
     */
    bind?(io: { obs: GPUBuffer; act: GPUBuffer; capacity: number; obsDim: number; actDim: number }): void;
    /** Async brains: one run over the bound buffers. Resolves once its GPU work is queued. */
    run?(): Promise<void>;
    destroy(): void;
}

/** Ignore comments when checking the required entry signature; the device validates full WGSL. */
export function agentSignatureMatches(wgsl: unknown, signature: RegExp): boolean {
    return typeof wgsl === 'string' && signature.test(wgsl.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, ''));
}

export function assertThinkSignature(wgsl: string): void {
    if (!agentSignatureMatches(wgsl, /\bfn\s+think\s*\(\s*i\s*:\s*u32\s*,\s*species\s*:\s*u32\s*,\s*tick\s*:\s*u32\s*\)\s*\{/)) {
        throw new Error('RuleBrain: the WGSL must define fn think(i: u32, species: u32, tick: u32).');
    }
}

/** Pure WGSL rules; owns no GPU resources. Brain and auxiliary buffers remain caller-owned. */
export class RuleBrain implements Brain {
    public readonly kind = 'stage';
    constructor(public readonly wgsl: string) { assertThinkSignature(wgsl); }
    public destroy(): void { /* No resources to release. */ }
}

/** PCG-style hash twin. The upper 24 bits convert exactly to f32 and stay in [0, 1). */
export function rand01(i: number, tick: number, stream: number): number {
    const seed = (i ^ Math.imul(tick, 0x9e3779b9) ^ Math.imul(stream, 0x85ebca6b)) >>> 0;
    const state = (Math.imul(seed, 747796405) + 2891336453) >>> 0;
    const word = Math.imul((state >>> ((state >>> 28) + 4)) ^ state, 277803737) >>> 0;
    return (((word >>> 22) ^ word) >>> 8) / 16777216;
}