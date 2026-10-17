import type { NullGraph } from '../../core';
import type { AgentDefinition } from './agentLayout';
import type { Brain } from './RuleBrain';
import { mlpLayout, packMlpWeights, type MlpNetwork } from './mlpMath';
import { buildMlpThinkWGSL } from './mlpShaders';

function fail(message: string): never { throw new Error(`MlpBrain: ${message}`); }

/** The first layer where two networks differ in width or activation, or -1 if they match. */
function shapeDiffers(a: MlpNetwork, b: readonly { input: number; output: number; activation: string }[]): number {
    if (!Array.isArray(a)) return 0;
    for (let k = 0; k < Math.max(a.length, b.length); k++) {
        if (!a[k] || !b[k] || a[k].input !== b[k].input || a[k].output !== b[k].output || a[k].activation !== b[k].activation) return k;
    }
    return -1;
}

/** Synchronous dense inference. Caller owns this brain; detach/destroy its systems before destroy(). */
export class MlpBrain implements Brain {
    public readonly kind = 'stage';
    public readonly cooperative?: boolean;
    public readonly rowCount: number;
    public readonly dims: { readonly obs: number; readonly act: number };
    public readonly wgsl: string;
    public readonly buffers: readonly GPUBuffer[];
    private readonly shape: readonly { input: number; output: number; activation: string }[];
    private readonly floats: number;
    /** Species names in row order for a per-species brain, 'genome' for per-agent rows, null for one network. */
    private readonly rows: readonly string[] | 'genome' | null;
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

    /**
     * N networks of one shape as the rows of one weight table; each agent reads the row its
     * record's `genome` word names, staged in workgroup memory when a 64-agent block shares a row
     * (agents-genome-rows-design.md §5, §6).
     */
    public static byGenome(engine: NullGraph, networks: readonly MlpNetwork[]): MlpBrain {
        if (!Array.isArray(networks) || networks.length < 1) fail('byGenome needs at least one network.');
        return new MlpBrain(engine, networks[0], { networks });
    }

    /** `rows` is for `MlpBrain.bySpecies` and `MlpBrain.byGenome`; `network` is then its first row. */
    constructor(private readonly engine: NullGraph, network: MlpNetwork,
        rows?: { readonly species?: readonly string[]; readonly networks: readonly (MlpNetwork | undefined)[] }) {
        const bySpecies = rows?.species;
        const networks = bySpecies ? bySpecies.map((name, r) =>
            rows.networks[r] ?? fail(`no network for species "${name}"; give one per species.`))
            : rows ? rows.networks as readonly MlpNetwork[] : [network];
        const layout = mlpLayout(networks[0]);
        networks.forEach((net, r) => {
            const k = shapeDiffers(net, networks[0]);
            if (k >= 0) fail(bySpecies
                ? `the network for "${bySpecies[r]}" differs in shape from "${bySpecies[0]}" at layer ${k}; give it its own brain: setBrain(brain, { species: '${bySpecies[r]}' }).`
                : `the network for row ${r} differs in shape from row 0 at layer ${k}; give it its own brain.`);
        });
        const table = new Float32Array(layout.floats * networks.length);
        networks.forEach((net, r) => packMlpWeights(net, table.subarray(r * layout.floats, (r + 1) * layout.floats)));
        const limits = engine.device.limits;
        if (table.byteLength > Math.min(limits.maxBufferSize, limits.maxStorageBufferBindingSize)) {
            fail('weights exceed device buffer limits.');
        }
        this.rows = bySpecies ? [...bySpecies] : rows ? 'genome' : null;
        this.rowCount = networks.length;
        this.floats = layout.floats;
        if (this.rows === 'genome') this.cooperative = true;
        this.shape = networks[0].map(({ input, output, activation }) => ({ input, output, activation }));
        this.dims = Object.freeze({ obs: networks[0][0].input, act: networks[0][networks[0].length - 1].output });
        this.wgsl = buildMlpThinkWGSL(networks[0], bySpecies ? this.rows as readonly string[] : rows ? networks.length : undefined);
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
     * A per-species brain updates one row with `{ species }`; a `byGenome` brain with `{ row }`.
     */
    public setWeights(network: MlpNetwork, options?: { readonly species?: string; readonly row?: number }): void {
        this.assertDevice(this.engine.device);
        const name = options?.species, r = options?.row;
        if (this.rows !== 'genome' && r !== undefined) fail('setWeights({ row }) needs a brain from MlpBrain.byGenome.');
        if (!Array.isArray(this.rows) && name !== undefined) fail('setWeights({ species }) needs a brain from MlpBrain.bySpecies.');
        if (Array.isArray(this.rows) && name === undefined) fail('this brain holds one network per species; call setWeights(network, { species }).');
        if (this.rows === 'genome' && r === undefined) fail('this brain holds one network per row; call setWeights(network, { row }).');
        const row = Array.isArray(this.rows) ? this.rows.indexOf(name!) : (r ?? 0);
        if (Array.isArray(this.rows) && row < 0) fail(`"${name}" is not a declared species.`);
        if (this.rows === 'genome' && (!Number.isInteger(row) || row < 0 || row >= this.rowCount)) {
            fail(`row ${row} is outside this brain's ${this.rowCount} rows.`);
        }
        const packed = packMlpWeights(network);
        if (shapeDiffers(network, this.shape) >= 0) {
            fail('setWeights requires the same shape and activations; construct a new brain.');
        }
        this.engine.device.queue.writeBuffer(this.buffers[0], row * packed.byteLength, packed as Float32Array<ArrayBuffer>);
    }

    /**
     * Uploads one or more rows packed back to back (each `mlpLayout(network).floats` floats, as
     * `packMlpWeights` writes), starting at `first` (default 0), in one `writeBuffer`.
     */
    public setRows(packed: Float32Array, options?: { readonly first?: number }): void {
        this.assertDevice(this.engine.device);
        if (this.rows !== 'genome') fail('setRows needs a brain from MlpBrain.byGenome.');
        if (!(packed instanceof Float32Array) || packed.length === 0 || packed.length % this.floats !== 0) {
            fail(`setRows needs whole rows of ${this.floats} floats; got ${(packed as { length?: unknown })?.length}.`);
        }
        const first = options?.first ?? 0, count = packed.length / this.floats;
        if (!Number.isInteger(first) || first < 0 || first + count > this.rowCount) {
            const out = !Number.isInteger(first) || first < 0 || first >= this.rowCount ? first : first + count - 1;
            fail(`row ${out} is outside this brain's ${this.rowCount} rows.`);
        }
        for (let k = 0; k < packed.length; k++) {
            if (!Number.isFinite(packed[k])) fail(`packed weights[${k}] must be finite.`);
        }
        this.engine.device.queue.writeBuffer(this.buffers[0], first * this.floats * 4, packed as Float32Array<ArrayBuffer>);
    }

    public destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        this.buffers[0].destroy();
    }
}
