import type { Brain } from '../agents/RuleBrain';
import type { OrtModel, OrtTensorInfo } from './OrtRuntime';

export interface OrtBrainOptions {
    /** The observations' input, when the model has several. */
    readonly input?: string;
    /** The actions' output, when the model has several. */
    readonly output?: string;
}

function fail(message: string): never { throw new Error(`OrtBrain: ${message}`); }

// The think pass of an async brain (agents-ort-design.md §4): copy this agent's row of the brain's
// results into act. AgentSystem gates it, as every think pass, to the brain's species and cadence.
const COPY = `@group(1) @binding(0) var<storage, read> ortResults: array<f32>;
fn think(i: u32, species: u32, tick: u32) {
for (var k = 0u; k < ACT_DIM; k++) { setAct(i, k, ortResults[i * ACT_DIM + k]); }
setReason(i, 0u);
}`;

/**
 * Step 9, phase 9.3 (agents-ort-design.md §7): an ONNX model, run by ONNX Runtime, as the brain of
 * one or more species. Its runs read the system's observations and write a results buffer of its
 * own, `[capacity × actDim]` f32; its think pass copies that into the actions. Interactive frames
 * act on the run that landed between them, one frame late; `agents.step()` waits for a run.
 *
 * The model is the app's, from `runtime.load`: destroy it after this brain. One brain serves one
 * system: its buffers are bound once.
 */
export class OrtBrain implements Brain {
    public readonly kind = 'async';
    public readonly wgsl = COPY;
    // Fields without a value of their own are declared only, so that the bundle holds no code for them.
    /** `{ obs, act }`, when the model gives both widths as numbers. */
    declare public readonly dims?: { readonly obs: number; readonly act: number };
    declare private readonly input: OrtTensorInfo;
    declare private readonly output: OrtTensorInfo;
    declare private results?: GPUBuffer;
    declare private bound?: { obs: GPUBuffer; act: GPUBuffer };
    declare private pending?: Promise<unknown>;
    private destroyed = false;

    constructor(public readonly model: OrtModel, options: OrtBrainOptions = {}) {
        const pick = (kind: string, list: readonly OrtTensorInfo[], name?: string) => name !== undefined
            ? list.find(t => t.name === name) ?? fail(`the model has no ${kind} "${name}"; it has ${list.map(t => `"${t.name}"`).join(', ')}.`)
            : list.length === 1 ? list[0]
                : fail(`the model has ${list.length} ${kind}s (${list.map(t => `"${t.name}"`).join(', ')}); name the ${kind === 'input' ? 'observations' : 'actions'} with { ${kind}: '<name>' }.`);
        this.input = pick('input', model.inputs, options.input);
        this.output = pick('output', model.outputs, options.output);
        for (const [kind, t] of [['input', this.input], ['output', this.output]] as const) {
            if (t.type !== 'float32' || t.dims.length !== 2) fail(`${kind} "${t.name}" is [${t.dims}] ${t.type}; expected [N, ${kind === 'input' ? 'obsDim' : 'actDim'}] float32.`);
        }
        const [obs, act] = [this.input.dims[1], this.output.dims[1]];
        if (typeof obs === 'number' && typeof act === 'number' && obs > 0 && act > 0) this.dims = { obs, act };
    }

    /** The results buffer, once bound: the think pass's one group-1 buffer. */
    public get buffers(): readonly GPUBuffer[] | undefined { return this.results && [this.results]; }

    /** Checked by AgentSystem before the brain is bound: ONNX Runtime runs only on its own device. */
    public assertDevice(device: GPUDevice): void {
        if (this.destroyed) fail('used after destroy().');
        if (device !== this.model.device) {
            fail('this system\'s engine has another device. ONNX Runtime 1.29 runs only on a device it made: init the engine with { device: runtime.device }.');
        }
    }

    /**
     * Called by `setBrain`: binds the model to the system's observations and to a results buffer,
     * which starts as a copy of the actions, so that a species keeps acting as it did until the
     * first run lands. Again for the same system, it copies the actions again.
     */
    public bind(io: { obs: GPUBuffer; act: GPUBuffer; capacity: number; obsDim: number; actDim: number }): void {
        if (this.destroyed) fail('used after destroy().');
        const device = this.model.device, bytes = io.capacity * io.actDim * 4;
        const seed = (results: GPUBuffer) => {
            const encoder = device.createCommandEncoder();
            encoder.copyBufferToBuffer(io.act, 0, results, 0, bytes);
            device.queue.submit([encoder.finish()]);
        };
        if (this.bound) {
            if (this.bound.obs !== io.obs || this.bound.act !== io.act) fail('bound to another system\'s buffers; buffers are fixed for life. Load the model again for this system.');
            return seed(this.results!);
        }
        for (const [kind, t] of [['input', this.input], ['output', this.output]] as const) {
            const n = t.dims[0];
            if (typeof n === 'number' && n >= 0 && n !== io.capacity) {
                fail(`${kind} "${t.name}" is [${t.dims}]: its first axis is fixed at ${n}, and the system has ${io.capacity} agents. Export it with a symbolic first axis (PyTorch: dynamic_shapes, or dynamic_axes={'${t.name}': {0: 'N'}}).`);
            }
        }
        // Rounded up to 16 B, as ONNX Runtime writes 16 B at a time (§6).
        const results = device.createBuffer({ label: 'OrtBrain results', size: Math.ceil(bytes / 16) * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        try {
            seed(results);
            this.model.bind({ inputs: { [this.input.name]: { buffer: io.obs, dims: [io.capacity, io.obsDim] } },
                outputs: { [this.output.name]: { buffer: results, dims: [io.capacity, io.actDim] } } });
        } catch (error) {
            results.destroy();
            throw error;
        }
        this.results = results;
        this.bound = { obs: io.obs, act: io.act };
    }

    /** One run over the bound buffers; AgentSystem starts it. */
    public async run(): Promise<void> {
        if (this.destroyed) fail('used after destroy().');
        await (this.pending = this.model.run());
    }

    /** Releases the results buffer, after a run in flight. The model stays the app's. */
    public destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        const release = () => this.results?.destroy();
        if (this.pending) void this.pending.then(release, release);
        else release();
    }
}
