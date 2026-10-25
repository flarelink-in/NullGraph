// D2.4 (offline-films-browser-design.md §11, Q4): mediabunny is an optional peer dependency. Nothing here
// imports it statically, nor its types (they would pull @types/dom-webcodecs into every consumer's build);
// `MediabunnyLike` declares the members null-graph/film uses, as OrtLike does for onnxruntime-web. The
// namespace is imported on first use, or an app or a test passes one in. The pure helpers (planFilm,
// cameraPath, encodePNG, …) never load it, so Node's film tool imports them without it installed.

/** A mediabunny `Quality`: an explicit bitrate here (§4.4: an object, never a bare number). */
export type MediabunnyQuality = object;

/** An output format: which video codecs it can hold. */
export interface MediabunnyOutputFormatLike {
    getSupportedVideoCodecs(): string[];
}

/** A `VideoSample` over a WebCodecs `VideoFrame`; closing it closes the frame. */
export interface MediabunnyVideoSampleLike {
    close(): void;
}

/** An encoded packet, as `onEncodedPacket` sees it. */
export interface MediabunnyEncodedPacketLike {
    readonly byteLength: number;
    readonly type: 'key' | 'delta';
}

/**
 * A `VideoSampleSource`. `add` resolves once the encoder and the writer can take more (backpressure); it
 * does not close the sample (1.61.3, `closeSample = false`), so the sink does.
 */
export interface MediabunnyVideoSourceLike {
    add(sample: MediabunnyVideoSampleLike): Promise<void>;
}

export interface MediabunnyVideoSourceConfig {
    readonly codec: string;
    readonly quality: MediabunnyQuality;
    readonly keyFrameInterval: number;
    readonly latencyMode: 'quality';
    readonly alpha: 'discard';
    readonly hardwareAcceleration: 'no-preference';
    /** Called once per candidate config before its support check: the last call is the one in use. */
    readonly onEncoderConfig: (config: VideoEncoderConfig) => void;
    readonly onEncodedPacket: (packet: MediabunnyEncodedPacketLike) => void;
}

export interface MediabunnyOutputLike {
    readonly state: string;
    addVideoTrack(source: MediabunnyVideoSourceLike, metadata: { readonly frameRate: number; readonly maximumPacketCount: number }): void;
    start(): Promise<void>;
    finalize(): Promise<void>;
    cancel(): Promise<void>;
}

/** A `BufferTarget`: the whole file in memory, `buffer` set once the output is finalized. */
export interface MediabunnyBufferTargetLike {
    readonly buffer: ArrayBuffer | null;
}

/**
 * A `StreamTarget` over a WritableStream: positioned writes, `{ type: 'write', position, data }`, in 16 MiB
 * chunks when `chunked` (at most two held at once, 1.61.3), and the stream closed on finalize and on cancel.
 */
export type MediabunnyStreamTargetLike = object;

/** The members of `import * as mediabunny from 'mediabunny'` (1.61.x) that null-graph/film uses. */
export interface MediabunnyLike {
    readonly Quality: new (options: { readonly bitrate: number }) => MediabunnyQuality;
    readonly Mp4OutputFormat: new (options?: { readonly fastStart?: false | 'in-memory' | 'reserve' }) => MediabunnyOutputFormatLike;
    readonly WebMOutputFormat: new (options?: object) => MediabunnyOutputFormatLike;
    readonly BufferTarget: new () => MediabunnyBufferTargetLike;
    readonly StreamTarget: new (writable: WritableStream, options?: { readonly chunked?: boolean; readonly chunkSize?: number }) => MediabunnyStreamTargetLike;
    readonly Output: new (options: { readonly format: MediabunnyOutputFormatLike; readonly target: MediabunnyBufferTargetLike | MediabunnyStreamTargetLike }) => MediabunnyOutputLike;
    readonly VideoSample: new (frame: VideoFrame) => MediabunnyVideoSampleLike;
    readonly VideoSampleSource: new (config: MediabunnyVideoSourceConfig) => MediabunnyVideoSourceLike;
    /**
     * The codecs of `codecs` this browser's WebCodecs can encode at this size, rate and quality, in their
     * order. It takes no hardware preference (1.61.3): the sink's default, `no-preference`, is what it asks.
     */
    getEncodableVideoCodecs(codecs?: string[], options?: {
        readonly width?: number;
        readonly height?: number;
        readonly quality?: MediabunnyQuality;
        readonly frameRate?: number;
    }): Promise<string[]>;
}

/** The namespace passed in, or mediabunny imported now; a named refusal when it is not installed. */
export async function loadMediabunny(injected?: MediabunnyLike): Promise<MediabunnyLike> {
    if (injected) return injected;
    try {
        // A literal specifier, so the app's bundler finds the package; modules.d.ts keeps TypeScript from
        // reading mediabunny's own declarations.
        return (await import('mediabunny')) as unknown as MediabunnyLike;
    } catch {
        throw new Error('film: mediabunny is not installed; install mediabunny to encode video in the browser.');
    }
}
