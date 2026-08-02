export class WebGPUContext {
    public device!: GPUDevice;
    public context!: GPUCanvasContext;
    public format!: GPUTextureFormat;
    public depthTexture!: GPUTexture;
    public cameraUniformBuffer!: GPUBuffer;

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
            width?: number,
            height?: number
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

        // Log which GPU we got for debugging
        const adapterInfo = adapter.info;
        console.log(
            `NullGraph: Using GPU - ${adapterInfo} `
        );

        // --- FEATURE NEGOTIATION ---
        // Only request features that the physical hardware actually supports.
        const featuresToEnable: GPUFeatureName[] = [];
        if (options.desiredFeatures !== undefined) {
            for (const feature of options.desiredFeatures) {
                if (adapter.features.has(feature)) {
                    featuresToEnable.push(feature);
                } else {
                    console.warn(
                        `NullGraph: Optional feature "${feature}" is not supported by this device.`
                    );
                }
            }
        }

        this.device = await adapter.requestDevice({
            requiredFeatures: featuresToEnable
        });

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

        this.depthTexture = this.device.createTexture({
            size: [dWidth, dHeight],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }
}