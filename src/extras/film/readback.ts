// D2.4 (offline-films-browser-design.md §4.3 B): frames read back from a headless engine's screen texture.
// Each grab copies the texture into one of three staging buffers and starts mapping it; a take waits for the
// oldest and hands it over. One frame stays in flight, so frame j maps while frame j + 1 steps and draws. The
// bytes are exact, which is why hashes, PNGs and the tests read frames this way. A port of the Node tool's
// FrameCapture (tools/film/capture.mjs), with Uint8Array and a reusable output buffer.

import { fail, whole } from './util';

interface Grab {
    readonly frame: number;
    readonly buffer: GPUBuffer;
    readonly mapped: Promise<void>;
}

export class FrameReadback {
    readonly width: number;
    readonly height: number;
    private readonly device: GPUDevice;
    private readonly texture: GPUTexture;
    /** copyTextureToBuffer's rows are 256-byte aligned; the frame's own rows are width × 4. */
    private readonly paddedRow: number;
    private readonly staging: GPUBuffer[];
    private inFlight: Grab[] = [];
    private next = 0;
    private pool: Uint8Array | null = null;
    private destroyed = false;

    constructor(device: GPUDevice, texture: GPUTexture, width: number, height: number) {
        this.width = whole(width, 'readback width');
        this.height = whole(height, 'readback height');
        if (texture.format !== 'rgba8unorm' && texture.format !== 'rgba8unorm-srgb') {
            fail(`readback reads rgba8unorm frames, top row first; the screen texture is ${texture.format} (a headless engine's default is rgba8unorm).`);
        }
        this.device = device;
        this.texture = texture;
        this.paddedRow = Math.ceil(this.width * 4 / 256) * 256;
        this.staging = [];
        try {
            for (let i = 0; i < 3; i++) this.staging.push(device.createBuffer({
                label: 'film readback', size: this.paddedRow * this.height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            }));
        } catch (error) {
            for (const buffer of this.staging) {
                try { buffer.destroy(); } catch { /* attempt every release; preserve the allocation error */ }
            }
            throw error;
        }
    }

    /** Frames grabbed and not yet taken. */
    get pending(): number {
        return this.inFlight.length;
    }

    /** Copy the texture as the queue reaches this point, and start mapping it. Take one before grabbing a third. */
    grab(frame: number): void {
        if (this.destroyed) fail('readback: grab() after destroy().');
        if (this.inFlight.length >= this.staging.length - 1) fail('readback: take() a frame before grabbing two more.');
        const buffer = this.staging[this.next++ % this.staging.length];
        const encoder = this.device.createCommandEncoder({ label: 'film readback' });
        encoder.copyTextureToBuffer({ texture: this.texture }, { buffer, bytesPerRow: this.paddedRow }, [this.width, this.height]);
        this.device.queue.submit([encoder.finish()]);
        const mapped = buffer.mapAsync(GPUMapMode.READ);
        // Mapping may fail before the next take (device loss, or a workload failure while the
        // previous frame is in flight). Observe immediately, but retain the rejection for take.
        void mapped.catch(() => {});
        this.inFlight.push({ frame, buffer, mapped });
    }

    /**
     * The oldest grab's pixels, tightly packed RGBA8, top row first. Written into `out` when given (at least
     * width × height × 4 bytes; nothing is allocated, design-principles §5), else into a buffer this readback
     * reuses, which the next take() overwrites.
     */
    async take(out?: Uint8Array): Promise<{ readonly frame: number; readonly bytes: Uint8Array }> {
        const size = this.width * this.height * 4;
        if (out && out.length < size) fail(`readback: take(out) needs ${size} bytes for ${this.width}×${this.height} RGBA; got ${out.length}.`);
        const bytes = out ?? (this.pool ??= new Uint8Array(size));
        const grab = this.oldest();
        await grab.mapped;
        try {
            const src = new Uint8Array(grab.buffer.getMappedRange());
            const row = this.width * 4;
            if (this.paddedRow === row) bytes.set(src.subarray(0, size));
            else for (let y = 0; y < this.height; y++) bytes.set(src.subarray(y * this.paddedRow, y * this.paddedRow + row), y * row);
        } finally {
            grab.buffer.unmap();
        }
        return { frame: grab.frame, bytes: out ? bytes : bytes.subarray(0, size) };
    }

    /**
     * The oldest grab as a WebCodecs VideoFrame, `'RGBX'` so the alpha byte is ignored (a scene that leaves
     * alpha below 1 encodes opaque, as ffmpeg's yuv420p path does). Built straight from the mapped range, so
     * the bytes are copied once, by the VideoFrame. The caller owns the frame.
     */
    async takeVideoFrame(timestampUs: number, durationUs: number): Promise<{ readonly frame: number; readonly videoFrame: VideoFrame }> {
        const grab = this.oldest();
        await grab.mapped;
        try {
            const videoFrame = new VideoFrame(grab.buffer.getMappedRange(), {
                format: 'RGBX', codedWidth: this.width, codedHeight: this.height,
                layout: [{ offset: 0, stride: this.paddedRow }], timestamp: timestampUs, duration: durationUs,
            });
            return { frame: grab.frame, videoFrame };
        } finally {
            grab.buffer.unmap();
        }
    }

    /** Release the staging buffers. Reads still pending reject when their buffer goes; those rejections are swallowed. */
    destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        for (const grab of this.inFlight) grab.mapped.catch(() => {});
        this.inFlight = [];
        for (const buffer of this.staging) buffer.destroy();
        this.pool = null;
    }

    private oldest(): Grab {
        if (this.destroyed) fail('readback: take after destroy().');
        const grab = this.inFlight.shift();
        if (!grab) fail('readback: nothing to take; grab() a frame first.');
        return grab;
    }
}
