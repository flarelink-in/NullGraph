export class WebGPUContext {
    public device!: GPUDevice;
    public context!: GPUCanvasContext;
    public format!: GPUTextureFormat;
    public depthTexture!: GPUTexture;
    public cameraUniformBuffer!: GPUBuffer;

    public async init(canvas: HTMLCanvasElement): Promise<void> {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) throw new Error("WebGPU not supported!");
        this.device = await adapter.requestDevice();
        this.context = canvas.getContext('webgpu') as GPUCanvasContext;

        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format: this.format });

        this.cameraUniformBuffer = this.device.createBuffer({
            size: 20 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.setupDepthTexture(canvas.width, canvas.height);
    }

    public resize(width: number, height: number): void {
        this.setupDepthTexture(width, height);
    }

    private setupDepthTexture(width: number, height: number): void {
        if (this.depthTexture) this.depthTexture.destroy();
        this.depthTexture = this.device.createTexture({
            size: [width, height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }
}