// Stand-ins for null-graph/film's runner tests (test/film-browser.test.mjs, phase F2), so recordFilm,
// FilmSession, FrameReadback and the video sink run whole in Node, with no GPU and no encoder:
// - filmDevice(): a GPUDevice that records what is made and destroyed, keeps error scopes and listeners
//   as WebGPU does, maps and unmaps buffers as WebGPU does (a mapped range detaches on unmap), and paints
//   every read-back or captured frame from `device.screen`, which the stand-in workload sets as it draws;
// - VideoFrame and OffscreenCanvas on globalThis, recording every frame made and each close;
// - fakeMediabunny(): mediabunny's output classes, recording samples, packets and the track's options;
// - counterWorkload(): a workload whose frames show its step count, so a test reads which step each
//   frame was drawn at, and which records every build, reset, step, draw and dispose.
// Nothing here renders. A frame's pixel (x, y) is paint(steps, x, y), whatever the render sequence.

globalThis.GPUBufferUsage ??= {
    MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512,
};
globalThis.GPUTextureUsage ??= { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 };
globalThis.GPUMapMode ??= { READ: 1, WRITE: 2 };

export class GPUValidationError { constructor(message) { this.message = message; } }
export class GPUOutOfMemoryError { constructor(message) { this.message = message; } }

export const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

/** A stand-in frame's pixel at (x, y), drawn at step count `steps`: the count in R and G, x and y in B and A. */
export const paint = (steps, x, y) => [steps & 255, (steps >> 8) & 255, x & 255, y & 255];
/** The step count a stand-in frame was drawn at, read from its first pixel. */
export const shown = bytes => bytes[0] | (bytes[1] << 8);

export function pixels(steps, width, height) {
    const out = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) out.set(paint(steps, x, y), (y * width + x) * 4);
    return out;
}

/**
 * A device for the film's engine to adopt. `fenceMs(n)` and `mapMs()` say how long the nth
 * onSubmittedWorkDone and every mapAsync take; `lose()` resolves `lost`; `raise(kind, message)` raises an
 * error as an operation would, into the innermost scope that takes it, else as an uncapturederror event.
 */
export function filmDevice({ adapterInfo = { vendor: 'stand-in', architecture: 'test', description: 'the film stand-in device' } } = {}) {
    const log = {
        buffers: [], textures: [], submits: 0, copies: 0, fences: 0, scopes: [], listeners: 0,
        errors: [], configured: [],
    };
    const screen = { steps: 0 };
    const stack = [];
    let resolveLost;
    const target = new EventTarget();
    const device = {
        label: 'film stand-in',
        features: new Set(),
        limits: { maxTextureDimension2D: 8192, maxBufferSize: 1 << 30, maxStorageBufferBindingSize: 1 << 27 },
        adapterInfo,
        screen,
        log,
        lost: new Promise(resolve => { resolveLost = resolve; }),
        fenceMs: () => 0,
        mapMs: () => 0,
        lose(reason = 'unknown', message = 'The stand-in GPU went away.') { resolveLost({ reason, message }); },
        raise(kind, message) {
            const error = kind === 'out-of-memory' ? new GPUOutOfMemoryError(message) : new GPUValidationError(message);
            log.errors.push(`${kind}: ${message}`);
            for (let i = stack.length - 1; i >= 0; i--) {
                if (stack[i].filter === kind) {
                    stack[i].errors.push(error);
                    return;
                }
            }
            const event = new Event('uncapturederror');
            event.error = error;
            target.dispatchEvent(event);
        },
        addEventListener(type, listener, options) {
            if (type === 'uncapturederror') log.listeners++;
            target.addEventListener(type, listener, options);
        },
        removeEventListener(type, listener, options) {
            if (type === 'uncapturederror') log.listeners--;
            target.removeEventListener(type, listener, options);
        },
        pushErrorScope(filter) {
            stack.push({ filter, errors: [] });
            log.scopes.push(`push ${filter}`);
        },
        async popErrorScope() {
            const scope = stack.pop();
            if (!scope) throw new DOMException('popErrorScope() with no error scope pushed.', 'OperationError');
            log.scopes.push(`pop ${scope.filter}`);
            return scope.errors[0] ?? null;
        },
        scopeDepth: () => stack.length,
        createBuffer(descriptor) {
            const buffer = standInBuffer(device, descriptor);
            log.buffers.push(buffer);
            return buffer;
        },
        createTexture(descriptor) {
            const texture = {
                ...descriptor, destroyed: 0,
                createView: () => ({ texture }),
                destroy() { texture.destroyed++; },
            };
            log.textures.push(texture);
            return texture;
        },
        createSampler: () => ({}),
        createShaderModule: () => ({}),
        createRenderPipeline: () => ({ getBindGroupLayout: () => ({}) }),
        createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
        createBindGroup: () => ({}),
        createBindGroupLayout: () => ({}),
        createPipelineLayout: () => ({}),
        createCommandEncoder() {
            const commands = [];
            const pass = {
                setPipeline() {}, setBindGroup() {}, setViewport() {}, setScissorRect() {}, setVertexBuffer() {}, setIndexBuffer() {},
                draw() {}, drawIndexed() {}, drawIndexedIndirect() {}, dispatchWorkgroups() {}, dispatchWorkgroupsIndirect() {}, end() {},
            };
            return {
                copyTextureToBuffer(source, destination, size) { commands.push({ source, destination, size }); },
                beginRenderPass: () => pass,
                beginComputePass: () => pass,
                finish: () => ({ commands }),
            };
        },
        queue: {
            writeBuffer() {},
            writeTexture() {},
            submit(commandBuffers) {
                log.submits++;
                for (const { commands = [] } of commandBuffers) for (const command of commands) copyFrame(device, command);
            },
            async onSubmittedWorkDone() {
                const ms = device.fenceMs(++log.fences);
                if (ms) await sleep(ms);
                else await new Promise(resolve => setImmediate(resolve));
            },
        },
        destroy() { log.deviceDestroyed = true; },
    };
    return device;
}

/** copyTextureToBuffer, run at submit: the frame as the workload last drew it, rows at bytesPerRow, padding 0xEE. */
function copyFrame(device, { destination, size: [width, height] }) {
    const buffer = destination.buffer;
    device.log.copies++;
    if (buffer.mapState !== 'unmapped') {
        device.raise('validation', `Buffer "${buffer.label}" used in a submit while ${buffer.mapState}.`);
        return;
    }
    buffer.data.fill(0xee);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) buffer.data.set(paint(device.screen.steps, x, y), y * destination.bytesPerRow + x * 4);
    }
}

/** A GPUBuffer: mapAsync resolves after `device.mapMs()`, rejects with an AbortError if unmapped or destroyed first. */
function standInBuffer(device, descriptor) {
    let state = 'unmapped';
    let range = null;
    let cancel = null;
    const buffer = {
        ...descriptor,
        data: new Uint8Array(descriptor.size),
        destroyed: 0,
        maps: 0,
        get mapState() { return state; },
        mapAsync() {
            if (state !== 'unmapped') return Promise.reject(new DOMException(`Buffer "${descriptor.label}" is already ${state}.`, 'OperationError'));
            buffer.maps++;
            state = 'pending';
            return new Promise((resolve, reject) => {
                cancel = reject;
                setTimeout(() => {
                    if (state !== 'pending') return;
                    state = 'mapped';
                    resolve();
                }, device.mapMs());
            });
        },
        getMappedRange() {
            if (state !== 'mapped') throw new DOMException(`getMappedRange() on buffer "${descriptor.label}", which is ${state}.`, 'OperationError');
            return (range ??= buffer.data.slice().buffer);
        },
        unmap() {
            if (state === 'pending') cancel(new DOMException('The mapping was aborted: unmap() or destroy() came first.', 'AbortError'));
            if (range) structuredClone(range, { transfer: [range] }); // detached, as WebGPU detaches it
            range = null;
            state = 'unmapped';
        },
        destroy() {
            buffer.destroyed++;
            buffer.unmap();
        },
    };
    return buffer;
}

/** OffscreenCanvas with a WebGPU context that only configures; `getContext` returns null when `noWebGPU`. */
export class FakeOffscreenCanvas {
    static noWebGPU = false;
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.context = null;
    }

    getContext(kind) {
        if (kind !== 'webgpu' || FakeOffscreenCanvas.noWebGPU) return null;
        return (this.context ??= {
            canvas: this,
            configuration: null,
            configure(configuration) { this.configuration = configuration; configuration.device.log.configured.push(configuration); },
            unconfigure() { this.configuration = null; },
            getCurrentTexture() {
                if (!this.configuration) throw new DOMException('getCurrentTexture() on an unconfigured context.', 'InvalidStateError');
                const texture = { format: this.configuration.format, createView: () => ({ texture }), destroy() {} };
                return texture;
            },
        });
    }
}

/**
 * WebCodecs' VideoFrame, from a stand-in canvas (its device's screen) or from bytes, copied at
 * construction with `layout`'s stride as the real one copies them; a detached buffer throws, as there.
 * `FakeVideoFrame.made` holds every frame; `closes` counts each close.
 */
export class FakeVideoFrame {
    static made = [];
    /** Set to a message to make construction from a canvas throw it. */
    static refuseCanvas = null;

    constructor(source, init = {}) {
        if (typeof init.timestamp !== 'number') throw new TypeError("Failed to construct 'VideoFrame': required member timestamp is undefined.");
        if (source instanceof FakeOffscreenCanvas) {
            const device = source.context?.configuration?.device;
            if (!device) throw new DOMException('The canvas has no configured WebGPU context.', 'InvalidStateError');
            if (FakeVideoFrame.refuseCanvas) throw new DOMException(FakeVideoFrame.refuseCanvas, 'InvalidStateError');
            this.format = 'BGRA';
            this.codedWidth = source.width;
            this.codedHeight = source.height;
            this.data = pixels(device.screen.steps, source.width, source.height);
        } else {
            const { format, codedWidth, codedHeight, layout } = init;
            const bytes = ArrayBuffer.isView(source) ? new Uint8Array(source.buffer, source.byteOffset, source.byteLength) : new Uint8Array(source);
            const row = codedWidth * 4;
            const stride = layout?.[0]?.stride ?? row;
            const offset = layout?.[0]?.offset ?? 0;
            if (bytes.length < offset + stride * (codedHeight - 1) + row) throw new TypeError(`VideoFrame: ${bytes.length} bytes are too few for ${codedWidth}×${codedHeight}.`);
            this.format = format;
            this.codedWidth = codedWidth;
            this.codedHeight = codedHeight;
            this.data = new Uint8Array(row * codedHeight);
            for (let y = 0; y < codedHeight; y++) this.data.set(bytes.subarray(offset + y * stride, offset + y * stride + row), y * row);
        }
        this.timestamp = init.timestamp;
        this.duration = init.duration ?? null;
        this.closes = 0;
        FakeVideoFrame.made.push(this);
    }

    /** One tightly packed plane of 4-byte pixels, as `data` holds it. */
    allocationSize() {
        if (this.closes) throw new DOMException('allocationSize() on a closed VideoFrame.', 'InvalidStateError');
        return this.data.length;
    }

    async copyTo(destination) {
        if (this.closes) throw new DOMException('copyTo() on a closed VideoFrame.', 'InvalidStateError');
        new Uint8Array(destination.buffer ?? destination, destination.byteOffset ?? 0).set(this.data);
        return [{ offset: 0, stride: this.codedWidth * 4 }];
    }

    close() {
        this.closes++;
    }
}

// ---------------------------------------------------------------------------------------------------
// F3: OPFS, Web Locks and the storage estimate
// ---------------------------------------------------------------------------------------------------

/** A chunk as a FileSystemWritableFileStream takes it: bytes, a string, or { type: 'write' | 'truncate', … }. */
function chunkOf(chunk, at) {
    if (typeof chunk === 'string') return { position: at, bytes: new TextEncoder().encode(chunk) };
    if (ArrayBuffer.isView(chunk)) return { position: at, bytes: new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength) };
    if (chunk instanceof ArrayBuffer) return { position: at, bytes: new Uint8Array(chunk) };
    if (chunk?.type === 'truncate') return { truncate: chunk.size };
    if (chunk?.type === 'write') return chunkOf(chunk.data, chunk.position ?? at);
    throw new TypeError('The stand-in writable takes bytes, a string or a { type } chunk.');
}

/**
 * OPFS as storage.ts uses it. A file's writable is a real WritableStream: its writes go to a swap copy that
 * close() commits and abort() drops, as Chrome's .crswap does; a write past `quotaBytes` (committed bytes and
 * swap growth together) rejects with QuotaExceededError and errors the stream. A folder holding an open
 * writable cannot be removed (NoModificationAllowedError, as in Chrome). `refuse(name, errorName)` makes that
 * file's createWritable reject; `failWrite(name, n)` makes its nth whole-file write (a manifest) reject;
 * `fs.onWritable(name)`, when set, runs as a writable is opened.
 */
export function fakeOPFS({ quotaBytes = Infinity } = {}) {
    const fs = { used: 0, quotaBytes, refused: new Map(), failWrites: new Map(), writes: new Map(), onWritable: null };
    const sizeOf = node => (node.kind === 'file' ? node.data.length : [...node.children.values()].reduce((n, c) => n + sizeOf(c), 0));
    const openIn = node => (node.kind === 'file' ? node.open > 0 : [...node.children.values()].some(openIn));
    const file = name => {
        const node = {
            kind: 'file', name, data: new Uint8Array(0), open: 0,
            async getFile() { return new Blob([node.data]); },
            async createWritable() {
                const refused = fs.refused.get(name);
                if (refused) throw new DOMException(`The stand-in OPFS refused a writable for "${name}".`, refused);
                fs.onWritable?.(name);
                node.open++;
                let swap = new Uint8Array(64);
                let size = 0;
                let settled = false;
                const settle = () => { if (!settled) { settled = true; node.open--; } };
                const stream = new WritableStream({
                    write(raw) {
                        const c = chunkOf(raw, size);
                        if (c.truncate !== undefined) { size = Math.min(size, c.truncate); return; }
                        const end = c.position + c.bytes.length;
                        if (fs.used + Math.max(0, end - node.data.length) > fs.quotaBytes) {
                            // The stream errors, and Chrome drops its swap file with it: the file is free again.
                            settle();
                            throw new DOMException('The operation failed because it would cause the application to exceed its storage quota.', 'QuotaExceededError');
                        }
                        if (end > swap.length) { const grown = new Uint8Array(Math.max(end, swap.length * 2)); grown.set(swap.subarray(0, size)); swap = grown; }
                        swap.set(c.bytes, c.position);
                        size = Math.max(size, end);
                    },
                    close() {
                        const n = (fs.writes.get(name) ?? 0) + 1;
                        fs.writes.set(name, n);
                        if (fs.failWrites.get(name) === n) {
                            settle();
                            throw new DOMException(`The stand-in OPFS failed to commit "${name}".`, 'InvalidStateError');
                        }
                        fs.used += size - node.data.length;
                        node.data = swap.slice(0, size);
                        settle();
                    },
                    abort() { settle(); },
                });
                // FileSystemWritableFileStream's own write(): a writer taken for the one write.
                stream.write = async chunk => {
                    const writer = stream.getWriter();
                    try { await writer.write(chunk); } finally { writer.releaseLock(); }
                };
                return stream;
            },
        };
        return node;
    };
    const dir = name => {
        const node = {
            kind: 'directory', name, children: new Map(),
            async getDirectoryHandle(child, { create = false } = {}) {
                let c = node.children.get(child);
                if (c && c.kind !== 'directory') throw new DOMException(`"${child}" is a file.`, 'TypeMismatchError');
                if (!c) {
                    if (!create) throw new DOMException(`No folder "${child}".`, 'NotFoundError');
                    c = dir(child);
                    node.children.set(child, c);
                }
                return c;
            },
            async getFileHandle(child, { create = false } = {}) {
                let c = node.children.get(child);
                if (c && c.kind !== 'file') throw new DOMException(`"${child}" is a folder.`, 'TypeMismatchError');
                if (!c) {
                    if (!create) throw new DOMException(`No file "${child}".`, 'NotFoundError');
                    c = file(child);
                    node.children.set(child, c);
                }
                return c;
            },
            async removeEntry(child, { recursive = false } = {}) {
                const c = node.children.get(child);
                if (!c) throw new DOMException(`No entry "${child}".`, 'NotFoundError');
                if (c.kind === 'directory' && c.children.size && !recursive) throw new DOMException(`"${child}" is not empty.`, 'InvalidModificationError');
                if (openIn(c)) throw new DOMException('An attempt was made to modify an object where modifications are not allowed.', 'NoModificationAllowedError');
                fs.used -= sizeOf(c);
                node.children.delete(child);
            },
            async *entries() { for (const entry of [...node.children]) yield entry; },
            async *keys() { for (const key of [...node.children.keys()]) yield key; },
        };
        return node;
    };
    const root = dir('');
    return {
        root, fs,
        refuse(name, errorName = 'NotAllowedError') { fs.refused.set(name, errorName); },
        failWrite(name, n) { fs.failWrites.set(name, n); },
        /** The tree under `path` (folder names joined by '/') as { name: size | subtree }. */
        async tree(node = root) {
            const out = {};
            for (const [name, c] of node.children) out[name] = c.kind === 'file' ? c.data.length : await this.tree(c);
            return out;
        },
        async folder(...names) {
            let node = root;
            for (const name of names) node = node.children.get(name);
            return node ?? null;
        },
    };
}

/**
 * Web Locks: exclusive locks, `ifAvailable`, and `query()`. `holdElsewhere(name)` holds a lock as another tab
 * would, and returns its release: what closing that tab does.
 */
export function fakeLocks() {
    const held = new Set();
    const waiting = [];
    const release = name => {
        held.delete(name);
        const i = waiting.findIndex(w => w.name === name);
        if (i >= 0) waiting.splice(i, 1)[0].wake();
    };
    return {
        held,
        async request(name, options, callback) {
            if (typeof options === 'function') { callback = options; options = {}; }
            if (held.has(name)) {
                if (options.ifAvailable) return callback(null);
                await new Promise(wake => waiting.push({ name, wake }));
            }
            held.add(name);
            try {
                return await callback({ name, mode: options.mode ?? 'exclusive' });
            } finally {
                release(name);
            }
        },
        async query() { return { held: [...held].map(name => ({ name, mode: 'exclusive' })), pending: waiting.map(w => ({ name: w.name, mode: 'exclusive' })) }; },
        holdElsewhere(name) {
            held.add(name);
            return () => release(name);
        },
    };
}

/** Put OPFS, Web Locks and a storage estimate on the stand-in navigator; `locks: null` leaves Web Locks out. */
export function installStorage({ opfs = fakeOPFS(), locks = fakeLocks(), quota = 1e10 } = {}) {
    const estimate = { quota, usage: 0 };
    globalThis.navigator.storage = {
        getDirectory: async () => opfs.root,
        estimate: async () => ({ quota: estimate.quota, usage: opfs.fs.used + estimate.usage }),
    };
    if (locks) globalThis.navigator.locks = locks;
    return { opfs, locks, estimate };
}

/** Install the stand-in VideoFrame and OffscreenCanvas (and navigator.gpu's preferred format); returns an uninstall. */
export function installWebCodecs({ canvas = true } = {}) {
    const before = { VideoFrame: globalThis.VideoFrame, OffscreenCanvas: globalThis.OffscreenCanvas, navigator: Object.getOwnPropertyDescriptor(globalThis, 'navigator') };
    FakeVideoFrame.made = [];
    FakeVideoFrame.refuseCanvas = null;
    FakeOffscreenCanvas.noWebGPU = false;
    globalThis.VideoFrame = FakeVideoFrame;
    if (canvas) globalThis.OffscreenCanvas = FakeOffscreenCanvas;
    else delete globalThis.OffscreenCanvas;
    Object.defineProperty(globalThis, 'navigator', {
        configurable: true, writable: true,
        value: { userAgent: 'film stand-in', gpu: { getPreferredCanvasFormat: () => 'bgra8unorm' } },
    });
    return () => {
        globalThis.VideoFrame = before.VideoFrame;
        if (before.OffscreenCanvas) globalThis.OffscreenCanvas = before.OffscreenCanvas;
        else delete globalThis.OffscreenCanvas;
        if (before.navigator) Object.defineProperty(globalThis, 'navigator', before.navigator);
    };
}

/**
 * mediabunny's output side. `failAddAt`: the add of that frame rejects (the encoder fails). `failWrapAt`:
 * wrapping that frame in a VideoSample throws. `failFinalize`: finalize rejects before its target is finalized,
 * and closes the target in its finally, as 1.61.3's does (an encoder failing at its flush). `packetBytes`: each
 * frame's packet size. `addMs`: each add's wait (backpressure). `encodable`: what getEncodableVideoCodecs finds.
 * `hangAddAt` / `hangFinalize`: that add, or finalize, never returns, as with an encoder lost with Chrome's GPU
 * process (G9-crash), and cancelling the output does not release it; `onHang` runs as the hang begins.
 * `releaseOnCancel`: cancelling the output ends an add's `addMs` wait at once, the add then rejecting, as a
 * closed VideoEncoder releases mediabunny's wait for its queue and refuses the frame.
 */
export function fakeMediabunny({
    failAddAt = -1, failWrapAt = -1, failFinalize = false, packetBytes = 1000, addMs = 0, encodable = ['avc', 'vp9', 'av1'],
    hangAddAt = -1, hangFinalize = false, onHang = () => {}, releaseOnCancel = false,
} = {}) {
    const log = { sources: [], outputs: [], samples: [], wraps: 0 };
    class Quality { constructor(options) { this.bitrate = options.bitrate; } }
    class Mp4OutputFormat {
        constructor(options = {}) { this.options = options; }
        getSupportedVideoCodecs() { return ['avc', 'hevc', 'vp9', 'av1']; }
    }
    class WebMOutputFormat {
        constructor(options = {}) { this.options = options; }
        getSupportedVideoCodecs() { return ['vp8', 'vp9', 'av1']; }
    }
    class BufferTarget { buffer = null; }
    class VideoSample {
        constructor(frame) {
            if (log.wraps++ === failWrapAt) throw new TypeError('The stand-in VideoSample refused this frame.');
            if (!(frame instanceof FakeVideoFrame) || frame.closes) throw new TypeError('VideoSample needs an open VideoFrame.');
            this.frame = frame;
            this.timestamp = frame.timestamp / 1e6;
            this.closes = 0;
            log.samples.push(this);
        }

        close() {
            this.closes++;
            this.frame.close();
        }
    }
    class VideoSampleSource {
        constructor(config) {
            this.config = config;
            this.added = [];
            log.sources.push(this);
        }

        async add(sample) {
            if (sample.closes) throw new TypeError('A closed VideoSample was added.');
            const j = this.added.length;
            const { frame } = sample;
            this.added.push({ timestamp: frame.timestamp, duration: frame.duration, format: frame.format, steps: shown(frame.data), data: frame.data.slice(0, 64) });
            if (j === 0) {
                this.config.onEncoderConfig({ codec: 'avc1.640028', width: frame.codedWidth, height: frame.codedHeight, hardwareAcceleration: 'no-preference' });
            }
            if (addMs && releaseOnCancel) {
                await Promise.race([sleep(addMs), this.output.canceled]);
                if (this.output.state === 'canceled') throw new DOMException("Failed to execute 'encode' on 'VideoEncoder': Cannot call 'encode' on a closed codec.", 'InvalidStateError');
            } else if (addMs) await sleep(addMs);
            if (j === hangAddAt) {
                onHang();
                await new Promise(() => {});
            }
            if (j >= failAddAt && failAddAt >= 0) throw new Error('EncodingError: the stand-in encoder failed.');
            // A write the stream refused surfaces at the next add, as mediabunny's muxer error does (1.61.3).
            if (this.output?.writeError) throw this.output.writeError;
            this.config.onEncodedPacket({ byteLength: packetBytes, type: j % 120 === 0 ? 'key' : 'delta' });
            this.output?.writePacket(packetBytes);
        }
    }
    /** mediabunny's StreamTarget: positioned writes to `writable`, closed on finalize and on cancel (1.61.3). */
    class StreamTarget {
        constructor(writable, options = {}) {
            if (!(writable instanceof WritableStream)) throw new TypeError('StreamTarget requires a WritableStream instance.');
            this.writable = writable;
            this.options = options;
        }
    }
    class Output {
        constructor({ format, target }) {
            this.format = format;
            this.target = target;
            this.state = 'pending';
            this.tracks = [];
            this.canceled = new Promise(resolve => { this.onCanceled = resolve; });
            this.writer = null;
            this.offset = 0;
            this.writeError = null;
            log.outputs.push(this);
        }

        addVideoTrack(source, options) {
            source.output = this;
            this.tracks.push({ source, options });
        }
        async start() {
            if (this.target instanceof StreamTarget) this.writer = this.target.writable.getWriter();
            this.state = 'started';
        }
        /** A packet's bytes, written as the StreamTarget would, its error kept for the next add. */
        writePacket(bytes) {
            if (!this.writer) return;
            const position = this.offset;
            this.offset += bytes;
            void this.writer.write({ type: 'write', position, data: new Uint8Array(bytes) }).catch(error => { this.writeError ??= error; });
        }
        async finalize() {
            this.state = 'finalizing';
            if (hangFinalize) {
                onHang();
                await new Promise(() => {});
            }
            let closed = false;
            try {
                // As an encoder failing at its flush: before the target is finalized, its stream still open.
                if (failFinalize) throw new Error('The stand-in muxer could not finalize.');
                const added = this.tracks[0]?.source.added ?? [];
                const summary = new TextEncoder().encode(JSON.stringify(added.map(f => [f.timestamp, f.duration, f.steps])));
                if (this.writer) {
                    // The file holds the summary alone, as the memory target's buffer does: written over the packets.
                    await this.writer.ready;
                    if (this.writeError) throw this.writeError;
                    await this.writer.write({ type: 'write', position: 0, data: summary });
                    await this.writer.write({ type: 'truncate', size: summary.length });
                    closed = true;
                    await this.writer.close();
                } else {
                    this.target.buffer = summary.buffer;
                }
                this.state = 'finalized';
            } catch (error) {
                this.state = 'canceled';
                throw error;
            } finally {
                // As 1.61.3's finally: a target not finalized is closed whatever happened, which on a file's
                // writable would commit what it holds.
                if (this.writer && !closed) await this.writer.close().catch(() => {});
            }
        }

        cancel() {
            // As mediabunny 1.61.3: a second cancel returns the first; an output finalizing or finalized is not cancelled.
            if (this.state === 'canceled') return this.cancelling ?? Promise.resolve();
            if (this.state === 'finalizing' || this.state === 'finalized') return Promise.resolve();
            this.state = 'canceled';
            this.onCanceled();
            // And it closes its StreamTarget's writer, which on a file's writable would commit it. On a stream a
            // refused write has errored, that close fails, and so does the cancel.
            return this.cancelling = (async () => { if (this.writer) await this.writer.close(); })();
        }
    }
    return {
        Quality, Mp4OutputFormat, WebMOutputFormat, BufferTarget, StreamTarget, Output, VideoSample, VideoSampleSource,
        async getEncodableVideoCodecs(codecs) { return encodable.filter(c => codecs.includes(c)); },
        log,
    };
}

/** What a stand-in film's file holds: [timestamp µs, duration µs, steps shown] for each frame. */
export async function frames(blob) {
    return JSON.parse(new TextDecoder().decode(await blob.arrayBuffer()));
}

/**
 * A clock for performance.now(), so the guard's measurements are exact: a stand-in step or draw costs
 * what the test says, at once. `restore()` puts the real clock back.
 */
export function fakeClock() {
    let t = 1000;
    const own = Object.getOwnPropertyDescriptor(performance, 'now');
    Object.defineProperty(performance, 'now', { configurable: true, writable: true, value: () => t });
    return {
        advance(ms) { t += ms; },
        restore() {
            if (own) Object.defineProperty(performance, 'now', own);
            else delete performance.now;
        },
    };
}

/**
 * A workload whose frames show its step count since placement. Construction places it (steps = 0) and
 * makes one GPU buffer, registered with `defer` before anything else can fail. Options:
 * `reset` (default true) gives it reset(); `dt`; `stepMs(n)` / `drawWork(n)` add time to the nth step or
 * draw (on `clock` when given, else as real sleeps: a step sleeps, a draw makes the next fence wait);
 * `fail` names the point or points that throw ('construct', 'reset', 'step:<n>', 'draw:<n>',
 * 'prepare:<frame>', 'dispose'); `onStep(n)` runs at each step.
 */
export function counterWorkload(device, { reset = true, dt = 1 / 60, stepMs = () => 0, drawWork = () => 0, fail = [], onStep = null, clock = null, info } = {}) {
    const log = { builds: 0, resets: 0, disposals: 0, steps: 0, draws: [], prepares: 0, deferred: 0, buffers: [], engines: [], signals: [] };
    const points = [fail].flat();
    const thrown = point => { if (points.includes(point)) throw new Error(`the stand-in workload failed at ${point}`); };
    const create = async ({ engine, width, height, signal, defer }) => {
        log.builds++;
        log.engines.push(engine);
        log.signals.push(signal);
        const buffer = engine.device.createBuffer({ label: `workload ${log.builds}`, size: 16, usage: GPUBufferUsage.STORAGE });
        log.buffers.push(buffer);
        defer(() => { log.deferred++; buffer.destroy(); });
        thrown('construct');
        let steps = 0;
        let disposed = false;
        const workload = {
            dt,
            info,
            async step() {
                const n = ++log.steps;
                thrown(`step:${n}`);
                onStep?.(n);
                const ms = stepMs(n);
                if (ms && clock) clock.advance(ms);
                else if (ms) await sleep(ms);
                steps++;
            },
            prepare(frame) {
                log.prepares++;
                thrown(`prepare:${frame.index}`);
            },
            draw(frame) {
                const n = log.draws.length + 1;
                thrown(`draw:${n}`);
                log.draws.push({ index: frame.index, steps, seconds: frame.seconds, view: frame.view, width, height });
                device.screen.steps = steps;
                const ms = drawWork(n);
                if (ms && clock) clock.advance(ms);
                else if (ms) {
                    const before = device.fenceMs;
                    device.fenceMs = k => { device.fenceMs = before; return ms + before(k); };
                }
                return [];
            },
            dispose() {
                if (disposed) return;
                disposed = true;
                log.disposals++;
                buffer.destroy();
                thrown('dispose');
            },
        };
        if (reset) {
            workload.reset = () => {
                log.resets++;
                thrown('reset');
                steps = 0;
            };
        }
        return workload;
    };
    return { create, log };
}
