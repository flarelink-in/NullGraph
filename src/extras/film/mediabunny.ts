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

/** The members of `import * as mediabunny from 'mediabunny'` (1.61.x) that null-graph/film uses. */
export interface MediabunnyLike {
    readonly Quality: new (options: { readonly bitrate: number }) => MediabunnyQuality;
    readonly Mp4OutputFormat: new (options?: object) => MediabunnyOutputFormatLike;
    readonly WebMOutputFormat: new (options?: object) => MediabunnyOutputFormatLike;
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
        // A literal specifier, so the app's bundler finds the package; mediabunny-module.d.ts keeps
        // TypeScript from reading mediabunny's own declarations.
        return (await import('mediabunny')) as unknown as MediabunnyLike;
    } catch {
        throw new Error('film: mediabunny is not installed; install mediabunny to encode video in the browser.');
    }
}
