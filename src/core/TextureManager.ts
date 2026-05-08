export interface TextureLoadOptions {
    /** * IMPORTANT: True for Albedo/Color maps. False for Normal/Metallic/Roughness maps.
     * If true, the GPU will automatically linearize the colors when sampling.
     */
    sRGB?: boolean;
    /** Defaults to linear filtering and repeat addressing */
    samplerDescriptor?: GPUSamplerDescriptor;
}

export class TextureManager {
    private device: GPUDevice;

    // Cache to prevent loading the exact same texture into VRAM twice
    private textureCache: Map<string, GPUTexture> = new Map();
    private samplerCache: Map<string, GPUSampler> = new Map();
    private solidTextures: Map<string, GPUTextureView> = new Map();

    // A default sampler we can reuse for most standard 3D models
    public defaultSampler: GPUSampler;

    constructor(device: GPUDevice) {
        this.device = device;

        this.defaultSampler = this.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            addressModeU: 'repeat',
            addressModeV: 'repeat',
        });
    }



    /**
     * Fetches an image, decodes it, uploads it to the GPU, and returns a View.
     */
    public async load(url: string, options: TextureLoadOptions = {}): Promise<GPUTextureView> {
        const isSRGB = options.sRGB ?? false;

        // 1. Check Cache
        const cacheKey = `${url}_${isSRGB ? 'srgb' : 'linear'}`;
        if (this.textureCache.has(cacheKey)) {
            return this.textureCache.get(cacheKey)!.createView();
        }

        try {
            // 2. Fetch the Image
            const response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const blob = await response.blob();

            // 3. Decode the Image (with a safety net for weird PNGs)
            let imageBitmap: ImageBitmap;
            try {
                // Try strict WebGPU color space mapping first
                imageBitmap = await createImageBitmap(blob, {
                    colorSpaceConversion: isSRGB ? 'default' : 'none'
                });
            } catch (decodeError) {
                // Safety Net: If the browser's strict decoder chokes on a dummy 1x1 PNG,
                // try to decode it normally without color space rules.
                imageBitmap = await createImageBitmap(blob);
            }

            // 4. Allocate VRAM
            const format: GPUTextureFormat = isSRGB ? 'rgba8unorm-srgb' : 'rgba8unorm';
            const texture = this.device.createTexture({
                size: [imageBitmap.width, imageBitmap.height, 1],
                format: format,
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
            });

            // 5. Upload Image to GPU
            this.device.queue.copyExternalImageToTexture(
                { source: imageBitmap },
                { texture: texture },
                [imageBitmap.width, imageBitmap.height]
            );

            // 6. Cache and Return View
            this.textureCache.set(cacheKey, texture);
            return texture.createView();

        } catch (error) {
            console.warn(`[TextureManager] Failed to load/decode texture. Using fallback. URL: ${url}`, error);

            // Return a safe 1x1 pixel fallback texture so the shader can still run.
            // If it's an sRGB texture, it's usually Albedo (White).
            // If it's not, it's usually Normal/Data.
            return isSRGB ? this.fallbackWhite : this.fallbackWhite;
        }
    }
    /**
     * Gets a custom sampler, caching it based on its stringified properties.
     */
    public getSampler(descriptor: GPUSamplerDescriptor): GPUSampler {
        const key = JSON.stringify(descriptor);
        if (this.samplerCache.has(key)) {
            return this.samplerCache.get(key)!;
        }
        const sampler = this.device.createSampler(descriptor);
        this.samplerCache.set(key, sampler);
        return sampler;
    }

    /**
     * Clears all textures from VRAM. Call this when destroying a scene.
     */
    public clearCache(): void {
        this.textureCache.forEach(texture => texture.destroy());
        this.textureCache.clear();
        this.samplerCache.clear();
    }

    /** Generates a 1x1 solid color texture on the GPU */
    public getSolidTexture(r: number, g: number, b: number, a: number = 255): GPUTextureView {
        const key = `${r}_${g}_${b}_${a}`;
        if (this.solidTextures.has(key)) return this.solidTextures.get(key)!;

        const tex = this.device.createTexture({
            size: [1, 1, 1], format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        });

        this.device.queue.writeTexture(
            { texture: tex }, new Uint8Array([r, g, b, a]),
            { bytesPerRow: 4 }, [1, 1, 1]
        );

        const view = tex.createView();
        this.solidTextures.set(key, view);
        return view;
    }

    public get fallbackWhite() { return this.getSolidTexture(255, 255, 255, 255); }
    public get fallbackNormal() { return this.getSolidTexture(128, 128, 255, 255); }
}