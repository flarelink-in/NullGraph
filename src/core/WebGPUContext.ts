export class WebGPUContext {
    public device!: GPUDevice;
    public context!: GPUCanvasContext;
    public format!: GPUTextureFormat;
    public depthTexture!: GPUTexture;
    public cameraUniformBuffer!: GPUBuffer;

    /**
     * Initializes the WebGPU context with fallback-safe feature negotiation.
     * @param desiredFeatures - Features you'd like (e.g., ['timestamp-query'])
     */
    public async init(canvas: HTMLCanvasElement, desiredFeatures: GPUFeatureName[] = []): Promise<void> {
        if (!navigator.gpu) {
            throw new Error("WebGPU is not supported on this browser.");
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error("No appropriate GPU adapter found.");
        }

        // --- FEATURE NEGOTIATION ---
        // Only request features that the physical hardware actually supports.
        const featuresToEnable: GPUFeatureName[] = [];
        for (const feature of desiredFeatures) {
            if (adapter.features.has(feature)) {
                featuresToEnable.push(feature);
            } else {
                console.warn(`NullGraph: Optional feature "${feature}" is not supported by this device.`);
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
            alphaMode: 'premultiplied' // Good practice for blending with UI overlays
        });

        this.cameraUniformBuffer = this.device.createBuffer({
            size: 20 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.setupDepthTexture(canvas.width, canvas.height);
    }

    public resize(width: number, height: number): void {
        // Only resize if the device exists (avoids errors during early resize events)
        if (this.device) {
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