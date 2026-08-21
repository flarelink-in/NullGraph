import type { GPUCapabilities } from './types';

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

export class WebGPUContext {
    public device!: GPUDevice;
    public context!: GPUCanvasContext;
    public format!: GPUTextureFormat;
    public depthTexture!: GPUTexture;
    public cameraUniformBuffer!: GPUBuffer;

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

    /**
     * Initializes the WebGPU context with fallback-safe feature negotiation.
     * @param canvas
     * @param options
     */
    public async init(
        canvas: HTMLCanvasElement | OffscreenCanvas,
        options: {
            desiredFeatures?: GPUFeatureName[],
            powerPreference?: 'low-power' | 'high-performance',
            /** Opportunistically enable supported {@link PROGRESSION_FEATURES}. Default `true`. */
            autoNegotiate?: boolean,
            width?: number,
            height?: number,
            /**
             * Create the depth texture with `TEXTURE_BINDING` so it can be sampled — see
             * {@link depthSamplable}. Required by `AerialPerspective.getApplyPass()` on the
             * main-screen path. Default `false`.
             */
            samplableDepth?: boolean
        } = {}
    ): Promise<void> {
        if (!navigator.gpu) {
            throw new Error("WebGPU is not supported on this browser.");
        }

        // --- ADAPTER REQUEST WITH POWER PREFERENCE ---
        // Try high-performance first, fall back gracefully
        let adapter: GPUAdapter | null = null;
        const powerPref = options.powerPreference ?? 'high-performance';

        try {
            // Request with power preference hint
            adapter = await navigator.gpu.requestAdapter({
                powerPreference: powerPref
            });

            // If no adapter with preference, try any adapter
            if (!adapter) {
                console.warn(
                    `NullGraph: No adapter found with powerPreference "${powerPref}". ` +
                    `Falling back to default.`
                );
                adapter = await navigator.gpu.requestAdapter();
            }
        } catch (e) {
            // If requestAdapter itself throws, try without options
            console.warn(
                `NullGraph: Adapter request failed with powerPreference. ` +
                `Falling back to default. Error:`, e
            );
            adapter = await navigator.gpu.requestAdapter();
        }

        if (!adapter) {
            throw new Error("No appropriate GPU adapter found.");
        }

        // Log which GPU we got for debugging. adapter.info is a structured object;
        // interpolating it directly yields "[object Object]", so read its fields.
        const info = adapter.info;
        console.log(
            `NullGraph: Using GPU — ${[info.vendor, info.architecture, info.description]
                .filter(Boolean).join(' ') || 'unknown adapter'}`
        );

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

        if (options.autoNegotiate ?? true) {
            for (const feature of PROGRESSION_FEATURES) {
                if (adapter.features.has(feature)) requested.add(feature);
            }
        }

        this.device = await adapter.requestDevice({
            requiredFeatures: [...requested]
        });

        // Snapshot what the device actually granted so downstream systems (e.g. the
        // clustered LightSystem) can select modern code paths without re-probing.
        this.capabilities = buildCapabilities(this.device, info);

        this.context = canvas.getContext('webgpu') as GPUCanvasContext;
        this.format = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: 'premultiplied'
        });

        this.cameraUniformBuffer = this.device.createBuffer({
            label: 'Global Camera Uniform',
            size: 432,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.width = options.width ?? canvas.width ?? 1;
        this.height = options.height ?? canvas.height ?? 1;

        // Stored, not just applied: resize() recreates the texture, and a flag that survived only
        // until the first window resize would fail exactly once, later, in someone else's session.
        this.depthSamplable = options.samplableDepth ?? false;

        this.setupDepthTexture(canvas.width, canvas.height);
    }

    public resize(width: number, height: number): void {
        // Only resize if the device exists (avoids errors during early resize events)
        if (this.device) {
            this.width = width;
            this.height = height;
            this.setupDepthTexture(width, height);
        }
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
 * Distills the raw device feature set and adapter subgroup sizes into the structured,
 * queryable {@link GPUCapabilities} snapshot consumed across the engine.
 */
function buildCapabilities(device: GPUDevice, info: GPUAdapterInfo): GPUCapabilities {
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
        subgroupMinSize: subgroups ? (info.subgroupMinSize ?? 0) : 0,
        subgroupMaxSize: subgroups ? (info.subgroupMaxSize ?? 0) : 0,
        has: (feature) => enabled.has(feature),
    };
}