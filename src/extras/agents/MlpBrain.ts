import type { NullGraph } from '../../core';
import type { Brain } from './RuleBrain';
import { packMlpWeights, type MlpNetwork } from './mlpMath';
import { buildMlpThinkWGSL } from './mlpShaders';

/** Synchronous dense inference. Caller owns this brain; detach/destroy its systems before destroy(). */
export class MlpBrain implements Brain {
    public readonly kind = 'stage';
    public readonly dims: { readonly obs: number; readonly act: number };
    public readonly wgsl: string;
    public readonly buffers: readonly GPUBuffer[];
    private readonly shape: readonly { input: number; output: number; activation: string }[];
    private destroyed = false;

    constructor(private readonly engine: NullGraph, network: MlpNetwork) {
        const packed = packMlpWeights(network);
        const limits = engine.device.limits;
        if (packed.byteLength > Math.min(limits.maxBufferSize, limits.maxStorageBufferBindingSize)) {
            throw new Error('MlpBrain: weights exceed device buffer limits.');
        }
        this.shape = network.map(({ input, output, activation }) => ({ input, output, activation }));
        this.dims = Object.freeze({ obs: network[0].input, act: network[network.length - 1].output });
        this.wgsl = buildMlpThinkWGSL(network);
        const buffer = engine.device.createBuffer({ label: 'MlpBrain weights', size: packed.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        try { engine.device.queue.writeBuffer(buffer, 0, packed as Float32Array<ArrayBuffer>); }
        catch (error) { buffer.destroy(); throw error; }
        this.buffers = Object.freeze([buffer]);
    }

    /** Checked by AgentSystem before creating a consuming stage. */
    public assertDevice(device: GPUDevice): void {
        if (this.destroyed) throw new Error('MlpBrain: cannot use after destroy().');
        if (device !== this.engine.device) throw new Error('MlpBrain: target system must use the same GPU device.');
    }

    /** Entire model is validated/copied before the single upload; shape and activations stay fixed. */
    public setWeights(network: MlpNetwork): void {
        this.assertDevice(this.engine.device);
        const packed = packMlpWeights(network);
        if (network.length !== this.shape.length || network.some((layer, i) =>
            layer.input !== this.shape[i].input || layer.output !== this.shape[i].output ||
            layer.activation !== this.shape[i].activation)) {
            throw new Error('MlpBrain: setWeights requires the same shape and activations; construct a new brain.');
        }
        this.engine.device.queue.writeBuffer(this.buffers[0], 0, packed as Float32Array<ArrayBuffer>);
    }

    public destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        this.buffers[0].destroy();
    }
}