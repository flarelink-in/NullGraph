import type { EngineInitOptions, GPUCapabilities } from './types';

/**
 * The curated set of forward-looking WebGPU features NullGraph opportunistically enables
 * when `autoNegotiate` is on (the default). Every entry is checked against the adapter's
 * support list before being requested, so listing one here can never make init fail — it
 * only ever *enables* an optional capability, surfaced afterward via {@link GPUCapabilities}.
 */
export const PROGRESSION_FEATURES: readonly GPUFeatureName[] = [
    'subgroups',
    'shader-f16',
    'bgra8unorm-storage',
    'dual-source-blending',
];

/**
 * The limits NullGraph raises to the adapter's own values when `autoNegotiate` is on. A device
 * otherwise gets WebGPU's defaults whatever the GPU offers (8 storage buffers per stage and
 * 128 MB bindings, where current desktop GPUs report 16 and ~2 GB). Curated rather than "every
 * limit", as the spec advises, because better limits can cost performance on some
 * implementations; each entry has a consumer:
 * - `maxStorageBuffersPerShaderStage`: stage brains and app stages beyond the default's 3 spare
 *   bindings (agents);
 * - `maxStorageBufferBindingSize`, `maxBufferSize`: populations and instance rows past 128 MB.
 * Modules size against `device.limits` and refuse above it, so the baseline keeps working.
 */
export const PROGRESSION_LIMITS: readonly (keyof GPUSupportedLimits)[] = [
    'maxStorageBuffersPerShaderStage',
    'maxStorageBufferBindingSize',
    'maxBufferSize',
];

/**
 * Main-screen format when there is no canvas (`init(null, …)`). `rgba8unorm` because it is
 * core-renderable everywhere and its bytes are already in PNG order — `bgra8unorm`, the usual
 * canvas preference, would need a swizzle in every readback.
 */
export const HEADLESS_SCREEN_FORMAT: GPUTextureFormat = 'rgba8unorm';

export class WebGPUContext {
    public device!: GPUDevice;
    /**
     * The canvas context — `undefined` when the engine was initialised without a canvas, in which
     * case the main-screen pass renders into {@link screenTexture} instead. Read the frame's
     * target through {@link getScreenView}, which covers both.
     */
    public context?: GPUCanvasContext;
    public format!: GPUTextureFormat;
    public depthTexture!: GPUTexture;
    public cameraUniformBuffer!: GPUBuffer;

    /**
     * The offscreen main-screen target when there is no canvas; `undefined` otherwise. Usage is
     * `RENDER_ATTACHMENT | COPY_SRC`, so a test or a render-to-PNG tool can read the frame back —
     * readback itself stays out of the engine (it is legal in a test, forbidden in the frame loop).
     */
    public screenTexture?: GPUTexture;
    private screenView?: GPUTextureView;

    /**
     * `false` when the device was adopted through `init(…, { device })`. NullGraph never destroys
     * a device it did not create — the owner (ONNX Runtime, another renderer, a test harness)
     * still has work on it.
     */
    public ownsDevice: boolean = true;

    /** True after {@link destroy}, until the next {@link init}. */
    public isDestroyed: boolean = false;

    /**
     * Whether {@link depthTexture} carries `TEXTURE_BINDING` and can therefore be sampled
     * (environment-aerial-design.md §9.1). Opt in at {@link init} with `{ samplableDepth: true }`.
     *
     * Off by default, and the default is the point: adding `TEXTURE_BINDING` can disable some
     * drivers' depth-compression fast paths for that texture, and it costs that on EVERY frame
     * whether or not anything samples it. Paying at construction for what you use beats paying
     * per frame for what you might.
     *
     * Three consumers want it, which is why it is one flag and not three: aerial perspective's
     * apply pass, `HiZPyramid` (which allocates a separate r32float pyramid today precisely
     * because this is unsampleable), and a depth pre-pass.
     */
    public depthSamplable: boolean = false;

    /** Negotiated optional-feature snapshot, populated during {@link init}. */
    public capabilities!: GPUCapabilities;

    public width: number = 1;
    public height: number = 1;

    /** True when the main-screen pass renders into {@link screenTexture} rather than a canvas. */
    public get isHeadless(): boolean {
        return this.context === undefined;
    }

    /**
     * Initializes the WebGPU context with fallback-safe feature negotiation.
     *
     * @param canvas The canvas to present to, or `null` to render the main-screen pass into an
     *   offscreen {@link screenTexture} (Node/Dawn, tests, render-to-PNG). `null` needs
     *   `options.width` and `options.height`.
     * @param options See {@link EngineInitOptions}. `options.device` adopts an existing device.
     */
    public async init(
        canvas: HTMLCanvasElement | OffscreenCanvas | null,
        options: EngineInitOptions = {}
    ): Promise<void> {
        validateInitOptions(canvas, options);
        this.isDestroyed = false;

        if (options.device) {
            this.adoptDevice(options.device, options.desiredFeatures);
        } else {
            await this.requestDevice(options);
        }

        this.cameraUniformBuffer = this.device.createBuffer({
            label: 'Global Camera Uniform',
            size: 432,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.width = options.width ?? canvas?.width ?? 1;
        this.height = options.height ?? canvas?.height ?? 1;

        // Stored, not just applied: resize() recreates the texture, and a flag that survived only
        // until the first window resize would fail exactly once, later, in someone else's session.
        this.depthSamplable = options.samplableDepth ?? false;

        if (canvas) {
            this.context = canvas.getContext('webgpu') as GPUCanvasContext;
            this.format = options.format ?? navigator.gpu.getPreferredCanvasFormat();

            this.context.configure({
                device: this.device,
                format: this.format,
                alphaMode: 'premultiplied'
            });

            // The canvas's own size, not options.width/height: the depth attachment has to match
            // the texture getCurrentTexture() returns, and that is sized by the canvas element.
            this.setupDepthTexture(canvas.width, canvas.height);
        } else {
            this.format = options.format ?? HEADLESS_SCREEN_FORMAT;
            this.setupScreenTexture(this.width, this.height);
            this.setupDepthTexture(this.width, this.height);
        }
    }

    /**
     * The view the main-screen pass renders into this frame: the canvas's current texture, or the
     * offscreen {@link screenTexture} when there is no canvas. The offscreen view is created once
     * per size, so the headless frame path allocates nothing here.
     */
    public getScreenView(): GPUTextureView {
        if (this.context) return this.context.getCurrentTexture().createView();
        return this.screenView!;
    }

    public resize(width: number, height: number): void {
        // Only resize if the device exists (avoids errors during early resize events), and never after
        // destroy(): new targets would outlive the context that is meant to own them.
        if (this.device && !this.isDestroyed) {
            this.width = width;
            this.height = height;
            if (this.isHeadless) this.setupScreenTexture(width, height);
            this.setupDepthTexture(width, height);
        }
    }

    /**
     * Release what this context created: the screen and depth targets, the camera uniform, and the
     * canvas's configuration. An adopted device is never destroyed — its owner still has work on it;
     * a device this context requested is. Safe after an init that failed part-way, and idempotent.
     * Every release is attempted; failures are thrown together afterwards.
     *
     * Destroying an owned device rejects every read still pending on it (`AbortError`): await
     * `device.queue.onSubmittedWorkDone()` first if something is reading back.
     */
    public destroy(): void {
        if (this.isDestroyed) return;
        this.isDestroyed = true;
        const errors: unknown[] = [];
        const attempt = (release: () => void) => { try { release(); } catch (e) { errors.push(e); } };
        attempt(() => this.screenTexture?.destroy());
        attempt(() => this.depthTexture?.destroy());
        attempt(() => this.cameraUniformBuffer?.destroy());
        attempt(() => this.context?.unconfigure());
        if (this.ownsDevice) attempt(() => this.device?.destroy());
        this.screenTexture = undefined;
        this.screenView = undefined;
        if (errors.length === 1) throw errors[0];
        if (errors.length > 1) throw new AggregateError(errors, `WebGPUContext.destroy: ${errors.length} releases failed.`);
    }

    /** Request an adapter and a device, negotiating optional features. The canvas-era path, unchanged. */
    private async requestDevice(options: EngineInitOptions): Promise<void> {
        // `globalThis.navigator`, not `navigator`: Node before 21 has no navigator at all, and a
        // bare reference would throw a ReferenceError instead of this message.
        const gpu = globalThis.navigator?.gpu;
        // Node: install a WebGPU implementation as navigator.gpu first — test/gpu/node-gpu.mjs
        // does it with Dawn — or adopt a device. The message names both ways out, briefly: every
        // byte of a thrown string ships in the bundle.
        if (!gpu) {
            throw new Error("WebGPU unavailable: navigator.gpu is missing. Install one (Node: Dawn) or pass init(…, { device }).");
        }

        // --- ADAPTER REQUEST WITH POWER PREFERENCE ---
        // Try high-performance first, fall back gracefully
        let adapter: GPUAdapter | null = null;
        const powerPref = options.powerPreference ?? 'high-performance';

        try {
            // Request with power preference hint
            adapter = await gpu.requestAdapter({
                powerPreference: powerPref
            });

            // If no adapter with preference, try any adapter
            if (!adapter) {
                console.warn(
                    `NullGraph: No adapter found with powerPreference "${powerPref}". ` +
                    `Falling back to default.`
                );
                adapter = await gpu.requestAdapter();
            }
        } catch (e) {
            // If requestAdapter itself throws, try without options
            console.warn(
                `NullGraph: Adapter request failed with powerPreference. ` +
                `Falling back to default. Error:`, e
            );
            adapter = await gpu.requestAdapter();
        }

        if (!adapter) {
            throw new Error("No appropriate GPU adapter found.");
        }

        // Log which GPU we got for debugging. adapter.info is a structured object;
        // interpolating it directly yields "[object Object]", so read its fields.
        const info = adapter.info;
        console.log(`NullGraph: Using GPU — ${describeAdapter(info)}`);

        // --- FEATURE NEGOTIATION ---
        // Union the caller's explicit desiredFeatures with the modern "progression" set the
        // adapter actually supports. Every entry is guarded by adapter.features.has(), so
        // requestDevice() can never fail on a missing feature. Auto-negotiation is purely
        // additive — it only ever ENABLES optional capabilities, surfaced via `capabilities`.
        const requested = new Set<GPUFeatureName>();

        if (options.desiredFeatures) {
            for (const feature of options.desiredFeatures) {
                if (adapter.features.has(feature)) {
                    requested.add(feature);
                } else {
                    console.warn(
                        `NullGraph: Optional feature "${feature}" is not supported by this device.`
                    );
                }
            }
        }

        // Limits are raised to what this adapter supports, so the request can never fail on them.
        const requiredLimits: Record<string, number> = {};
        if (options.autoNegotiate ?? true) {
            for (const feature of PROGRESSION_FEATURES) {
                if (adapter.features.has(feature)) requested.add(feature);
            }
            for (const limit of PROGRESSION_LIMITS) {
                const value = adapter.limits[limit];
                if (typeof value === 'number') requiredLimits[limit] = value;
            }
        }

        this.device = await adapter.requestDevice({
            requiredFeatures: [...requested],
            requiredLimits,
        });
        this.ownsDevice = true;

        // Snapshot what the device actually granted so downstream systems (e.g. the
        // clustered LightSystem) can select modern code paths without re-probing.
        this.capabilities = buildCapabilities(this.device, info);
    }

    /**
     * Adopt a device another library created (ONNX Runtime Web, a test harness, another
     * renderer). Nothing is negotiated — features were fixed when its owner created it — so the
     * capability snapshot is simply what the device has. The baseline path needs no optional
     * feature, so whatever the owner requested is enough to render; it only decides which
     * accelerated paths light up.
     */
    private adoptDevice(device: GPUDevice, desiredFeatures?: GPUFeatureName[]): void {
        this.device = device;
        this.ownsDevice = false;

        // Same non-fatal policy as the request path: a desired feature is optional by definition.
        // Features are fixed at device creation, so only the owner can add one.
        if (desiredFeatures) {
            for (const feature of desiredFeatures) {
                if (!device.features.has(feature)) {
                    console.warn(`NullGraph: Optional feature "${feature}" is not enabled on the adopted device — its owner must request it.`);
                }
            }
        }

        // GPUDevice.adapterInfo is how an adopter learns the subgroup sizes it would otherwise
        // read off the adapter. Where a browser does not expose it yet, subgroup sizes report 0 —
        // the same "unreported" value buildCapabilities already uses.
        const info = (device as { adapterInfo?: GPUAdapterInfo }).adapterInfo;
        console.log(`NullGraph: Adopted an existing GPU device — ${describeAdapter(info)}`);
        this.capabilities = buildCapabilities(device, info);
    }

    private setupScreenTexture(width: number, height: number): void {
        if (this.screenTexture) this.screenTexture.destroy();
        this.screenTexture = this.device.createTexture({
            label: 'NullGraph_Screen',
            size: [Math.max(1, width), Math.max(1, height)],
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        this.screenView = this.screenTexture.createView();
    }

    private setupDepthTexture(width: number, height: number): void {
        if (this.depthTexture) this.depthTexture.destroy();

        // Depth texture dimensions must be at least 1x1
        const dWidth = Math.max(1, width);
        const dHeight = Math.max(1, height);

        // Sampling `depth24plus` needs a `texture_depth_2d` binding with a NON-FILTERING sampler,
        // or `textureLoad`. Both are core WebGPU — no feature flag, no format change.
        this.depthTexture = this.device.createTexture({
            size: [dWidth, dHeight],
            format: 'depth24plus',
            usage: this.depthSamplable
                ? GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
                : GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }
}

/**
 * Refuse option combinations that cannot mean what the caller thinks, naming the option and the
 * way out (design-principles §2). Every one of these would otherwise be silently ignored.
 */
function validateInitOptions(
    canvas: HTMLCanvasElement | OffscreenCanvas | null,
    options: EngineInitOptions
): void {
    // Kept short on purpose — thrown strings ship in the bundle. The reasoning lives here:
    // powerPreference only steers an adapter request, and an adopted device's owner already made
    // it; features are fixed at device creation, so autoNegotiate cannot apply to an adopted one
    // (engine.capabilities reports what it has); without a canvas nothing gives a screen size.
    if (options.device) {
        if (options.powerPreference !== undefined) {
            throw new Error('NullGraph.init: powerPreference does nothing on an adopted device — set it where the device is requested.');
        }
        if (options.autoNegotiate !== undefined) {
            throw new Error('NullGraph.init: autoNegotiate does nothing on an adopted device — its features are fixed; read engine.capabilities.');
        }
    }

    if (!canvas) {
        const { width, height } = options;
        if (width === undefined || height === undefined) {
            throw new Error('NullGraph.init: headless init (canvas = null) needs options.width and options.height.');
        }
        if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
            throw new Error(
                `NullGraph.init: headless screen size must be positive integers, got ` +
                `${width} x ${height}.`);
        }
    }
}

function describeAdapter(info: GPUAdapterInfo | undefined): string {
    if (!info) return 'unknown adapter';
    return [info.vendor, info.architecture, info.description].filter(Boolean).join(' ') || 'unknown adapter';
}

/**
 * Distills the raw device feature set and adapter subgroup sizes into the structured,
 * queryable {@link GPUCapabilities} snapshot consumed across the engine.
 */
function buildCapabilities(device: GPUDevice, info: GPUAdapterInfo | undefined): GPUCapabilities {
    const enabled = new Set<GPUFeatureName>();
    for (const f of device.features) enabled.add(f as GPUFeatureName);

    const subgroups = enabled.has('subgroups');
    return {
        features: enabled,
        subgroups,
        shaderF16: enabled.has('shader-f16'),
        bgra8unormStorage: enabled.has('bgra8unorm-storage'),
        dualSourceBlending: enabled.has('dual-source-blending'),
        // subgroupMinSize/MaxSize live on GPUAdapterInfo and stay optional until every
        // browser reports them; treat "subgroups off" or "unreported" as 0.
        subgroupMinSize: subgroups ? (info?.subgroupMinSize ?? 0) : 0,
        subgroupMaxSize: subgroups ? (info?.subgroupMaxSize ?? 0) : 0,
        has: (feature) => enabled.has(feature),
    };
}
