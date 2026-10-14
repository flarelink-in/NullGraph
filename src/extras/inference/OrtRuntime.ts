/**
 * Phase 9.1 (agents-ort-design.md §5, §6): ONNX Runtime's WebGPU provider on the device it makes,
 * over GPU buffers bound once. ONNX Runtime 1.29 runs on no device the app made (spike S5), and
 * destroys its own with its last session, so the runtime holds a session of its own for life: the
 * anchor. The engine adopts the device: `engine.init(canvas, { device: runtime.device })`.
 *
 * The app installs and imports onnxruntime-web, an optional peer dependency, and passes its
 * namespace in. Nothing here imports it, not even its types: `OrtLike` declares what is used.
 */

/** The members of onnxruntime-web's namespace that `OrtRuntime` uses: `import * as ort from 'onnxruntime-web/webgpu'`. */
export interface OrtLike {
    readonly InferenceSession: { create(model: Uint8Array, options?: object): Promise<OrtSessionLike> };
    readonly Tensor: { fromGpuBuffer(buffer: GPUBuffer, options: { dataType: 'float32'; dims: readonly number[] }): unknown };
    readonly env: { readonly webgpu: { adapter?: unknown; readonly device?: unknown } };
}

/** The members of an ONNX Runtime session that `OrtModel` uses. */
export interface OrtSessionLike {
    readonly inputMetadata: readonly OrtValueMetadataLike[];
    readonly outputMetadata: readonly OrtValueMetadataLike[];
    run(feeds: object, fetches: object): Promise<unknown>;
    release(): Promise<void>;
}

export interface OrtValueMetadataLike {
    readonly name: string;
    readonly isTensor: boolean;
    readonly type?: string;
    readonly shape?: readonly (number | string)[];
}

export interface OrtRuntimeOptions {
    /** The adapter ONNX Runtime makes its device from. Without it, ONNX Runtime asks for its own. */
    readonly adapter?: GPUAdapter;
}

/** A file of weights stored outside the .onnx file: `path` as the file names it; `data` its bytes, or a URL. */
export interface OrtExternalData {
    readonly path: string;
    readonly data: Uint8Array | ArrayBuffer | string;
}

export interface OrtModelOptions {
    /**
     * ONNX Runtime's graph capture (agents-ort-design.md §9): on, unless `false`, or unless the
     * model's weights are in a second file (`externalData`). Under capture ONNX Runtime 1.29 can
     * fail such a model from its second run; `true` asks for capture all the same.
     */
    readonly capture?: boolean;
    /** Weights stored outside the .onnx file, passed to ONNX Runtime as they are (§6.1). */
    readonly externalData?: readonly OrtExternalData[];
}

/** An input or output, as the model declares it. */
export interface OrtTensorInfo {
    readonly name: string;
    /** 'float32', 'int64', …; 'not a tensor' for a sequence or a map. */
    readonly type: string;
    /** Each axis: its size, or the name of a free axis. Empty when the file gives no shape. */
    readonly dims: readonly (number | string)[];
}

/** A GPU buffer for one input or output, and the tensor's shape in it. */
export interface OrtBinding {
    readonly buffer: GPUBuffer;
    readonly dims: readonly number[];
}

function fail(who: string, message: string): never { throw new Error(`${who}: ${message}`); }
const firstLine = (error: unknown) => String((error as Error)?.message ?? error).split('\n')[0];
// y = x + x on one float32: the anchor's model, 66 bytes of ONNX. It never runs; while its session
// lives, ONNX Runtime keeps its device.
const ANCHOR = 'CAhCAhANOjoKDgoBeAoBeBIBeSIDQWRkEgZhbmNob3JaDwoBeBIKCggIARIECgIIAWIPCgF5EgoKCAgBEgQKAggB';

/** ONNX Runtime's device, kept alive by the anchor session, and the models loaded on it. */
export class OrtRuntime {
    private readonly models = new Set<OrtModel>();
    private destroyed = false;

    private constructor(private readonly ort: OrtLike, public readonly device: GPUDevice, private readonly anchor: OrtSessionLike) {}

    /**
     * Starts ONNX Runtime's WebGPU provider: its WASM is loaded and its device made (0.5–0.7 s in
     * spike S4). Init the engine on `runtime.device` afterwards.
     */
    public static async create(ort: OrtLike, options: OrtRuntimeOptions = {}): Promise<OrtRuntime> {
        if (typeof ort?.InferenceSession?.create !== 'function' || typeof ort.Tensor?.fromGpuBuffer !== 'function' || !ort.env?.webgpu) {
            fail('OrtRuntime', 'that is not onnxruntime-web (no InferenceSession, Tensor.fromGpuBuffer or env.webgpu); pass the namespace of import * as ort from \'onnxruntime-web/webgpu\'.');
        }
        if (options.adapter) ort.env.webgpu.adapter = options.adapter;
        let anchor: OrtSessionLike, device: GPUDevice | undefined;
        try {
            anchor = await ort.InferenceSession.create(Uint8Array.from(atob(ANCHOR), c => c.charCodeAt(0)),
                { executionProviders: [{ name: 'webgpu' }] });
            device = await (ort.env.webgpu.device as Promise<GPUDevice | undefined>);
        } catch (error) {
            return fail('OrtRuntime', `ONNX Runtime could not start its WebGPU provider: ${firstLine(error)}.`);
        }
        if (!device) {
            await anchor.release();
            fail('OrtRuntime', 'ONNX Runtime could not start its WebGPU provider: it made no device.');
        }
        return new OrtRuntime(ort, device, anchor);
    }

    /**
     * Creates a session for the model on the runtime's device: the WebGPU provider, outputs kept on
     * the GPU, graph capture unless `capture: false` or weights in a second file. Bind its buffers
     * with `model.bind`.
     */
    public async load(model: Uint8Array | ArrayBuffer, options: OrtModelOptions = {}): Promise<OrtModel> {
        this.assertAlive();
        const bytes = model instanceof Uint8Array ? model : model instanceof ArrayBuffer ? new Uint8Array(model)
            : fail('OrtRuntime', 'expected the model\'s bytes, as a Uint8Array or an ArrayBuffer.');
        // Decided with the user, 2026-10-14 (agents-ort-design.md §14, item 5).
        const capture = options.capture ?? !options.externalData?.length;
        let session: OrtSessionLike;
        try {
            // An object, never the string 'webgpu': given a string, ONNX Runtime captures halfway (spike S2).
            session = await this.ort.InferenceSession.create(bytes, { executionProviders: [{ name: 'webgpu' }],
                preferredOutputLocation: 'gpu-buffer', enableGraphCapture: capture,
                ...(options.externalData ? { externalData: options.externalData } : {}) });
        } catch (error) {
            return fail('OrtRuntime', `ONNX Runtime could not load the model: ${firstLine(error)}.`);
        }
        if (this.destroyed || !session.inputMetadata || !session.outputMetadata) {
            await session.release();
            this.assertAlive();
            fail('OrtRuntime', 'this ONNX Runtime reports no input or output metadata; onnxruntime-web 1.29 or later is needed.');
        }
        const loaded: OrtModel = new OrtModel(this.ort, session, this.device, capture, () => this.models.delete(loaded));
        this.models.add(loaded);
        return loaded;
    }

    /** Destroys every model still loaded, then releases the anchor: the device goes with it. Destroy the engine first. */
    public async destroy(): Promise<void> {
        if (this.destroyed) return;
        this.destroyed = true;
        await Promise.all([...this.models].map(model => model.destroy()));
        await this.anchor.release();
    }

    private assertAlive(): void {
        if (this.destroyed) fail('OrtRuntime', 'used after destroy().');
    }
}

const describe = (m: OrtValueMetadataLike): OrtTensorInfo =>
    ({ name: m.name, type: m.isTensor ? m.type ?? 'unknown' : 'not a tensor', dims: m.isTensor ? [...m.shape ?? []] : [] });

/** One ONNX Runtime session, bound once to GPU buffers, run over them. */
export class OrtModel {
    declare public readonly inputs: readonly OrtTensorInfo[];
    declare public readonly outputs: readonly OrtTensorInfo[];
    declare private feeds?: Record<string, unknown>;
    declare private fetches?: Record<string, unknown>;
    declare private pending?: Promise<unknown>;
    private destroyed = false;

    /** @internal Made by `OrtRuntime.load`. */
    constructor(private readonly ort: OrtLike, private readonly session: OrtSessionLike,
        /** The runtime's device: every buffer bound must belong to it. */
        public readonly device: GPUDevice,
        /** Whether ONNX Runtime captures the session's GPU work. */
        public readonly capture: boolean,
        private readonly forget: () => void) {
        this.inputs = session.inputMetadata.map(describe);
        this.outputs = session.outputMetadata.map(describe);
    }

    /**
     * Gives every input and every output its GPU buffer, once: graph capture binds a session's
     * buffers on its first run and ignores others later (spike S2). Each tensor must be float32, of
     * the model's rank, with its fixed axes, and the same size for a named free axis everywhere.
     */
    public bind(buffers: { readonly inputs: Readonly<Record<string, OrtBinding>>; readonly outputs: Readonly<Record<string, OrtBinding>> }): void {
        this.assertAlive();
        if (this.feeds) fail('OrtModel', 'bound already; buffers are fixed for life, because graph capture binds them once. Load the model again for other buffers.');
        const symbols = new Map<string, [number, string]>();
        const tensors = (kind: string, declared: readonly OrtTensorInfo[], given: Readonly<Record<string, OrtBinding>>) => {
            for (const name of Object.keys(given)) {
                if (!declared.some(t => t.name === name)) fail('OrtModel', `"${name}" is not an ${kind} of this model; it has ${declared.map(t => `"${t.name}"`).join(', ')}.`);
            }
            const out: Record<string, unknown> = {};
            for (const t of declared) {
                const binding = Object.hasOwn(given, t.name) ? given[t.name] : fail('OrtModel', `${kind} "${t.name}" has no buffer; every input and output is bound to one.`);
                if (t.type !== 'float32') fail('OrtModel', `${kind} "${t.name}" is ${t.type}; only float32 tensors can be bound.`);
                const dims = [...binding.dims];
                const shape = () => fail('OrtModel', `${kind} "${t.name}" is [${t.dims}]; it was given [${dims}].`);
                if (!dims.every(d => Number.isInteger(d) && d > 0) || (t.dims.length && t.dims.length !== dims.length)) shape();
                t.dims.forEach((d, k) => {
                    if (typeof d === 'number') { if (d >= 0 && d !== dims[k]) shape(); }
                    else if (d) {
                        const seen = symbols.get(d);
                        if (seen && seen[0] !== dims[k]) fail('OrtModel', `the axis "${d}" is ${seen[0]} in "${seen[1]}" and ${dims[k]} in "${t.name}".`);
                        symbols.set(d, seen ?? [dims[k], t.name]);
                    }
                });
                // ONNX Runtime's kernels read and write 16 B at a time: past the values, up to the next
                // 16 B. On NVIDIA a write that ends beyond the buffer is dropped whole (phase 9.2).
                const bytes = Math.ceil(dims.reduce((a, d) => a * d, 4) / 16) * 16, { buffer } = binding;
                const storage = (buffer.usage & GPUBufferUsage.STORAGE) !== 0;
                if (buffer.size < bytes || !storage) {
                    fail('OrtModel', `"${t.name}" needs ${bytes} B with STORAGE usage: its values, rounded up to 16 B, as ONNX Runtime reads and writes 16 B at a time; its buffer has ${buffer.size} B${storage ? '' : ', without STORAGE'}.`);
                }
                out[t.name] = this.ort.Tensor.fromGpuBuffer(buffer, { dataType: 'float32', dims });
            }
            return out;
        };
        const feeds = tensors('input', this.inputs, buffers.inputs);
        this.fetches = tensors('output', this.outputs, buffers.outputs);
        this.feeds = feeds;
    }

    /**
     * Runs the session over the bound buffers. Resolves once ONNX Runtime has queued its GPU work,
     * not when the GPU is done; the queue orders what is submitted after it. One run at a time.
     */
    public async run(): Promise<void> {
        this.assertAlive();
        if (!this.feeds) fail('OrtModel', 'run() before bind().');
        if (this.pending) fail('OrtModel', 'run() while a run is in flight; await it first.');
        const pending = this.session.run(this.feeds, this.fetches!);
        this.pending = pending;
        try {
            await pending;
        } finally {
            this.pending = undefined;
        }
    }

    /** Releases the session, after a run in flight. The buffers stay the app's. */
    public async destroy(): Promise<void> {
        if (this.destroyed) return;
        this.destroyed = true;
        this.forget();
        await this.pending?.catch(() => undefined);
        await this.session.release();
    }

    private assertAlive(): void {
        if (this.destroyed) fail('OrtModel', 'used after destroy().');
    }
}
