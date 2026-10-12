import type { NullGraph } from '../../core';
import type { AgentDefinition } from './agentLayout';
import type { Brain } from './RuleBrain';
import { mlpLayout, packMlpWeights, type MlpNetwork } from './mlpMath';
import { buildMlpThinkWGSL } from './mlpShaders';

function fail(message: string): never { throw new Error(`MlpBrain: ${message}`); }

/** The first layer where two networks differ in width or activation, or -1 if they match. */
function shapeDiffers(a: MlpNetwork, b: readonly { input: number; output: number; activation: string }[]): number {
    for (let k = 0; k < Math.max(a.length, b.length); k++) {
        if (!a[k] || !b[k] || a[k].input !== b[k].input || a[k].output !== b[k].output || a[k].activation !== b[k].activation) return k;
    }
    return -1;
}

/** Synchronous dense inference. Caller owns this brain; detach/destroy its systems before destroy(). */
export class MlpBrain implements Brain {
    public readonly kind = 'stage';
    public readonly dims: { readonly obs: number; readonly act: number };
    public readonly wgsl: string;
    public readonly buffers: readonly GPUBuffer[];
    private readonly shape: readonly { input: number; output: number; activation: string }[];
    /** Species names in row order for a per-species brain; null for one network. */
    private readonly rows: readonly string[] | null;
    private destroyed = false;

    /**
     * One network per species of `definition`, all the same shape, as rows of one weight table:
     * each agent reads its species' row in the same kernel (agents-species-brains-design.md §5).
     */
    public static bySpecies(engine: NullGraph, definition: AgentDefinition,
        networks: Readonly<Record<string, MlpNetwork>>): MlpBrain {
        const species = [...definition.species].sort((a, b) => a.index - b.index).map(s => s.name);
        for (const name of Object.keys(networks)) if (!species.includes(name)) fail(`"${name}" is not a declared species.`);
        return new MlpBrain(engine, networks[species[0]], { species, networks: species.map(name => networks[name]) });
    }

    /** `rows` is for `MlpBrain.bySpecies`; `network` is then its first row. */
    constructor(private readonly engine: NullGraph, network: MlpNetwork,
        rows?: { readonly species: readonly string[]; readonly networks: readonly (MlpNetwork | undefined)[] }) {
        const networks = rows ? rows.species.map((name, r) =>
            rows.networks[r] ?? fail(`no network for species "${name}"; give one per species.`)) : [network];
        const layout = mlpLayout(networks[0]);
        networks.forEach((net, r) => {
            const k = shapeDiffers(net, networks[0]);
            if (k >= 0) fail(`the network for "${rows!.species[r]}" differs in shape from "${rows!.species[0]}" at layer ${k}; give it its own brain: setBrain(brain, { species: '${rows!.species[r]}' }).`);
        });
        const table = new Float32Array(layout.floats * networks.length);
        networks.forEach((net, r) => packMlpWeights(net, table.subarray(r * layout.floats, (r + 1) * layout.floats)));
        const limits = engine.device.limits;
        if (table.byteLength > Math.min(limits.maxBufferSize, limits.maxStorageBufferBindingSize)) {
            fail('weights exceed device buffer limits.');
        }
        this.rows = rows ? [...rows.species] : null;
        this.shape = networks[0].map(({ input, output, activation }) => ({ input, output, activation }));
        this.dims = Object.freeze({ obs: networks[0][0].input, act: networks[0][networks[0].length - 1].output });
        this.wgsl = buildMlpThinkWGSL(networks[0], this.rows ?? undefined);
        const buffer = engine.device.createBuffer({ label: 'MlpBrain weights', size: table.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        try { engine.device.queue.writeBuffer(buffer, 0, table); }
        catch (error) { buffer.destroy(); throw error; }
        this.buffers = Object.freeze([buffer]);
    }

    /** Checked by AgentSystem before creating a consuming stage. */
    public assertDevice(device: GPUDevice): void {
        if (this.destroyed) fail('cannot use after destroy().');
        if (device !== this.engine.device) fail('target system must use the same GPU device.');
    }

    /**
     * The entire model is validated and copied before one upload; shape and activations stay fixed.
     * A per-species brain updates one row: `setWeights(network, { species })`.
     */
    public setWeights(network: MlpNetwork, options?: { readonly species?: string }): void {
        this.assertDevice(this.engine.device);
        const name = options?.species;
        if (!this.rows && name !== undefined) fail('setWeights({ species }) needs a brain from MlpBrain.bySpecies.');
        if (this.rows && name === undefined) fail('this brain holds one network per species; call setWeights(network, { species }).');
        const row = this.rows ? this.rows.indexOf(name!) : 0;
        if (row < 0) fail(`"${name}" is not a declared species.`);
        const packed = packMlpWeights(network);
        if (shapeDiffers(network, this.shape) >= 0) {
            fail('setWeights requires the same shape and activations; construct a new brain.');
        }
        this.engine.device.queue.writeBuffer(this.buffers[0], row * packed.byteLength, packed as Float32Array<ArrayBuffer>);
    }

    public destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        this.buffers[0].destroy();
    }
}
