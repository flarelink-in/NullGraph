// D2.4 (offline-films-browser-design.md §3, §5): what this browser can do for a film of this size, before
// anything is built. A cheap probe: it draws no frame, writes and deletes no file, and asks for no
// permission. API presence and a supported encoder configuration are evidence, not a guarantee: an
// encoder can still fail at run time, and OPFS can refuse a write. Operational smoke tests are separate.

import { loadMediabunny, type MediabunnyLike } from './mediabunny';
import { readOutput, type FilmCodec, type FilmContainer } from './output';
import { bitrateFor } from './plan';
import { fail, whole } from './util';

/** The globals the probe reads; `globalThis` unless a test passes a stand-in. */
export interface FilmProbeScope {
    readonly navigator?: {
        readonly gpu?: unknown;
        readonly storage?: {
            readonly getDirectory?: unknown;
            estimate?(): Promise<{ readonly quota?: number; readonly usage?: number }>;
        };
    };
    readonly VideoFrame?: unknown;
    readonly VideoEncoder?: unknown;
    readonly OffscreenCanvas?: unknown;
    readonly Worker?: unknown;
    readonly showSaveFilePicker?: unknown;
}

export interface FilmProbeOptions {
    readonly width: number;
    readonly height: number;
    readonly fps: number;
    readonly container?: FilmContainer;
    /** In preference order; default the container's (Q7: H.264 alone for MP4). */
    readonly codecs?: readonly FilmCodec[];
    /** Bits a second; default `bitrateFor(width, height, fps)`, as the sink will use. */
    readonly bitrate?: number;
    /** The app's device. WebGPU then counts as present whatever `navigator.gpu` says. */
    readonly device?: GPUDevice;
    /** mediabunny's namespace, when the app has it already; imported otherwise. */
    readonly mediabunny?: MediabunnyLike;
    readonly scope?: FilmProbeScope;
}

export interface FilmSupport {
    /** A source, an encodable codec and a target exist: configuration-supported, not an end-to-end guarantee. */
    readonly ok: boolean;
    /** Encodable at this size, rate and bitrate, held by the container, in the app's order. */
    readonly codecs: FilmCodec[];
    readonly sources: {
        /** An OffscreenCanvas with a WebGPU context, and VideoFrame (Q1's default source for video). */
        readonly canvas: { readonly api: boolean; readonly operational: 'untested' };
        /** WebGPU and VideoFrame: frames read back, exact bytes. */
        readonly readback: { readonly api: boolean };
    };
    readonly targets: {
        readonly memory: true;
        readonly opfs: { readonly api: boolean; readonly writable: 'untested' };
        readonly picker: { readonly api: boolean };
    };
    /** Dedicated workers with OffscreenCanvas exist; whether the adapter and encoder work there is untested. */
    readonly worker: boolean;
    /** OPFS's own estimate, in bytes, or null when the browser gives none. */
    readonly storage: { readonly quota: number; readonly usage: number; readonly available: number } | null;
    /** Plain words for each thing missing. */
    readonly reasons: string[];
}

/** Smaller sizes tried when the asked one has no encoder, so the refusal names only sizes it probed. */
const LADDER: readonly (readonly [number, number])[] = [[3840, 2160], [2560, 1440], [1920, 1080], [1280, 720], [854, 480]];

function hasWebGPUCanvas(OffscreenCanvasClass: unknown): boolean {
    if (typeof OffscreenCanvasClass !== 'function') return false;
    try {
        const canvas = new (OffscreenCanvasClass as new (w: number, h: number) => { getContext(kind: string): unknown })(2, 2);
        return canvas.getContext('webgpu') != null;
    } catch {
        return false;
    }
}

export async function probeFilm(options: FilmProbeOptions): Promise<FilmSupport> {
    const width = whole(options.width, 'probe: width');
    const height = whole(options.height, 'probe: height');
    if (width % 2 || height % 2) fail(`width and height must be even for video encoding (yuv420p); got ${width}×${height}.`);
    const fps = whole(options.fps, 'probe: fps');
    const output = readOutput({ container: options.container, codecs: options.codecs, bitrate: options.bitrate });
    const scope = options.scope ?? (globalThis as FilmProbeScope);
    const reasons: string[] = [];

    const gpu = Boolean(options.device ?? scope.navigator?.gpu);
    if (!gpu) reasons.push('film: WebGPU is not available in this browser.');
    const videoFrame = typeof scope.VideoFrame === 'function';
    if (!videoFrame) reasons.push('film: WebCodecs VideoFrame is not available in this browser.');
    const webgpuCanvas = hasWebGPUCanvas(scope.OffscreenCanvas);
    if (gpu && videoFrame && !webgpuCanvas) reasons.push('film: no OffscreenCanvas with a WebGPU context here; frames would be read back.');
    const canvas = gpu && videoFrame && webgpuCanvas;
    const readback = gpu && videoFrame;

    const codecs = await encodableCodecs(options.mediabunny, scope, width, height, fps, output.container, output.codecs, output.bitrate, reasons);

    const storageManager = scope.navigator?.storage;
    const opfs = typeof storageManager?.getDirectory === 'function';
    if (!opfs) reasons.push('film: OPFS (navigator.storage.getDirectory) is not available here; a film can only be kept in memory, within its budget.');
    let storage: FilmSupport['storage'] = null;
    if (typeof storageManager?.estimate === 'function') {
        try {
            const { quota, usage } = await storageManager.estimate();
            if (typeof quota === 'number' && typeof usage === 'number') storage = { quota, usage, available: Math.max(0, quota - usage) };
        } catch { /* no estimate: the storage check then refuses nothing */ }
    }

    return {
        ok: gpu && (canvas || readback) && codecs.length > 0,
        codecs,
        sources: { canvas: { api: canvas, operational: 'untested' }, readback: { api: readback } },
        targets: { memory: true, opfs: { api: opfs, writable: 'untested' }, picker: { api: typeof scope.showSaveFilePicker === 'function' } },
        worker: typeof scope.Worker === 'function' && typeof scope.OffscreenCanvas === 'function',
        storage,
        reasons,
    };
}

/** The requested codecs this browser can encode for this film, in the app's order; reasons for the rest. */
async function encodableCodecs(
    injected: MediabunnyLike | undefined, scope: FilmProbeScope, width: number, height: number, fps: number,
    container: FilmContainer, requested: readonly FilmCodec[], bitrate: number | undefined, reasons: string[],
): Promise<FilmCodec[]> {
    if (typeof scope.VideoEncoder !== 'function') {
        reasons.push('film: WebCodecs VideoEncoder is not available in this browser.');
        return [];
    }
    let mb: MediabunnyLike;
    try {
        mb = await loadMediabunny(injected);
    } catch (error) {
        reasons.push((error as Error).message);
        return [];
    }
    return chooseCodecs(mb, width, height, fps, container, requested, bitrate, reasons);
}

/**
 * Of the app's codecs, those the container holds and WebCodecs can encode at this size, rate and bitrate,
 * in the app's order. The probe and the video sink choose the same way. When none is left, `reasons` gets a
 * refusal naming the smaller sizes, from a fixed ladder, that were probed and do have an encoder (§8).
 */
export async function chooseCodecs(
    mb: MediabunnyLike, width: number, height: number, fps: number,
    container: FilmContainer, requested: readonly FilmCodec[], bitrate: number | undefined, reasons: string[],
): Promise<FilmCodec[]> {
    const held = (container === 'mp4' ? new mb.Mp4OutputFormat() : new mb.WebMOutputFormat()).getSupportedVideoCodecs();
    const candidates = requested.filter(c => held.includes(c));
    for (const c of requested) if (!held.includes(c)) reasons.push(`film: this mediabunny's ${container} cannot hold ${c}.`);
    if (!candidates.length) return [];
    const encodable = (w: number, h: number) => mb.getEncodableVideoCodecs([...candidates], {
        width: w, height: h, quality: new mb.Quality({ bitrate: bitrate ?? bitrateFor(w, h, fps) }), frameRate: fps,
    });
    const found = await encodable(width, height);
    const codecs = candidates.filter(c => found.includes(c));
    if (!codecs.length) {
        const sizes: string[] = [];
        for (const [w, h] of LADDER) {
            const [lw, lh] = height > width ? [h, w] : [w, h];
            if (lw * lh < width * height && (await encodable(lw, lh)).length) sizes.push(`${lw}×${lh}`);
        }
        reasons.push(`film: no WebCodecs encoder for ${width}×${height} (${candidates.join(', ')}); supported sizes on this browser: ${sizes.join(', ') || 'none of those probed'}.`);
    }
    return codecs;
}
