// D2.1 (docs/offline-films-design.md §4.4): the frame engine.render() drew, read back. Each grab copies the
// screen texture into one of three staging buffers and starts mapping it; take() waits for the oldest
// and hands over its bytes, tightly packed RGBA8, top row first. Grab frame j, then take frame j - 1 while
// the GPU works: one frame stays in flight.

export class FrameCapture {
    constructor(engine, width, height) {
        const format = engine.ctx.format;
        if (format !== 'rgba8unorm' && format !== 'rgba8unorm-srgb') {
            throw new Error(`film: the screen is ${format}; films read rgba8unorm (Node's headless screen).`);
        }
        this.engine = engine;
        this.width = width;
        this.height = height;
        this.row = width * 4;
        this.paddedRow = Math.ceil(this.row / 256) * 256;               // copyTextureToBuffer's row alignment
        this.staging = [0, 1, 2].map(() => engine.device.createBuffer({ label: 'film capture',
            size: this.paddedRow * height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
        this.inFlight = [];
        this.next = 0;
    }

    /** Copies the screen as the queue reaches this point, and starts mapping it. */
    grab(frame) {
        if (this.inFlight.length >= this.staging.length - 1) throw new Error('film: take() a frame before grabbing two more.');
        const buffer = this.staging[this.next++ % this.staging.length];
        const encoder = this.engine.device.createCommandEncoder({ label: 'film capture' });
        encoder.copyTextureToBuffer({ texture: this.engine.screenTexture }, { buffer, bytesPerRow: this.paddedRow }, [this.width, this.height]);
        this.engine.device.queue.submit([encoder.finish()]);
        this.inFlight.push({ frame, buffer, mapped: buffer.mapAsync(GPUMapMode.READ) });
    }

    get pending() { return this.inFlight.length; }

    /** The oldest grabbed frame: { frame, bytes } (a Buffer it owns). */
    async take() {
        const { frame, buffer, mapped } = this.inFlight.shift();
        await mapped;
        const src = new Uint8Array(buffer.getMappedRange());
        let bytes;
        if (this.paddedRow === this.row) bytes = Buffer.from(src);
        else {
            bytes = Buffer.allocUnsafe(this.row * this.height);
            for (let y = 0; y < this.height; y++) bytes.set(src.subarray(y * this.paddedRow, y * this.paddedRow + this.row), y * this.row);
        }
        buffer.unmap();
        return { frame, bytes };
    }

    destroy() {
        for (const b of this.staging) b.destroy();
    }
}
