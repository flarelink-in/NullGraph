// D2.4 (offline-films-browser-design.md §4.4, §4.5, §8): what a film is written as and where, checked before
// anything is built, and the storage check. Pure: which codecs this browser can encode is the probe's
// question (probe.ts); this file refuses only what no browser could do.

import { fail, isObject, whole } from './util';

export type FilmContainer = 'mp4' | 'webm';
export type FilmCodec = 'avc' | 'hevc' | 'vp8' | 'vp9' | 'av1';

/** The codecs each container can hold; an app's list is refused for any other. */
export const CONTAINER_CODECS: Readonly<Record<FilmContainer, readonly FilmCodec[]>> = {
    mp4: ['avc', 'hevc', 'vp9', 'av1'],
    webm: ['vp8', 'vp9', 'av1'],
};

/**
 * Q7 (decided 2026-10-25, as recommended): MP4 is H.264 alone unless the app opts into a fallback list,
 * because VP9 or AV1 in MP4 do not play everywhere H.264 does. WebM tries VP9, then AV1.
 */
export const DEFAULT_CODECS: Readonly<Record<FilmContainer, readonly FilmCodec[]>> = {
    mp4: ['avc'],
    webm: ['vp9', 'av1'],
};

/** Q3 (decided 2026-10-25): the in-memory fallback's encoded-payload budget, unless the app sets one. */
export const DEFAULT_MEMORY_BUDGET = 256 * 1024 * 1024;

/**
 * Where the file goes (§4.5): `'auto'` prefers writable OPFS and falls back to memory only within the
 * budget; `{ file }` is a handle the app obtained from the save picker in a user gesture.
 */
export type FilmTarget = 'auto' | 'memory' | 'opfs' | { readonly file: FileSystemFileHandle };

export interface FilmOutputOptions {
    readonly container?: FilmContainer;
    /** In preference order; default `DEFAULT_CODECS[container]`. */
    readonly codecs?: readonly FilmCodec[];
    /** Bits a second; default `bitrateFor(width, height, fps)`. */
    readonly bitrate?: number;
    readonly target?: FilmTarget;
    /** The download name: metadata, never an internal path. Default `film.<container>`. */
    readonly name?: string;
    readonly memoryBudgetBytes?: number;
}

/** The output options, checked and completed. `bitrate` stays undefined until the film's size is known. */
export interface FilmOutput {
    readonly container: FilmContainer;
    readonly codecs: readonly FilmCodec[];
    readonly bitrate: number | undefined;
    readonly target: FilmTarget;
    readonly name: string;
    readonly memoryBudgetBytes: number;
}

/** The app's `output` option, checked and completed. */
export function readOutput(output: unknown = {}): FilmOutput {
    if (!isObject(output)) fail('output must be an object.');
    const container = output.container ?? 'mp4';
    if (container !== 'mp4' && container !== 'webm') fail(`output.container must be "mp4" or "webm"; got ${JSON.stringify(container)}.`);
    const codecs = output.codecs ?? DEFAULT_CODECS[container];
    if (!Array.isArray(codecs) || codecs.length === 0) fail('output.codecs must be a non-empty list, in preference order.');
    for (const codec of codecs) {
        if (!CONTAINER_CODECS[container].includes(codec)) {
            fail(`output.codecs: ${container} cannot hold ${JSON.stringify(codec)}; it holds ${CONTAINER_CODECS[container].join(', ')}.`);
        }
    }
    const bitrate = output.bitrate === undefined ? undefined : whole(output.bitrate, 'output.bitrate (bits a second)');
    const target = output.target ?? 'auto';
    const isFile = isObject(target) && isObject(target.file) && typeof (target.file as { createWritable?: unknown }).createWritable === 'function';
    if (target !== 'auto' && target !== 'memory' && target !== 'opfs' && !isFile) {
        fail('output.target must be "auto", "memory", "opfs" or { file } with a FileSystemFileHandle from the save picker.');
    }
    const name = output.name ?? `film.${container}`;
    if (typeof name !== 'string' || !name.trim()) fail('output.name must be a non-empty file name.');
    const memoryBudgetBytes = whole(output.memoryBudgetBytes ?? DEFAULT_MEMORY_BUDGET, 'output.memoryBudgetBytes');
    return { container, codecs: [...codecs], bitrate, target: target as FilmTarget, name, memoryBudgetBytes };
}

/** `navigator.storage.estimate()`'s answer, or a test's. */
export interface StorageEstimateLike {
    readonly quota?: number;
    readonly usage?: number;
}

/**
 * Compare the film's predicted size (predictBytes) with what OPFS reports free, as the Node tool checks
 * free disk. An estimate, not a reservation: quota, eviction and the encoder's real output can still
 * change the outcome, so a write can fail later anyway. Without quota figures nothing is refused.
 */
export function checkStorage(neededBytes: number, estimate: StorageEstimateLike): { readonly needed: number; readonly available: number | null } {
    const needed = whole(neededBytes, 'the predicted size in bytes');
    const { quota, usage } = estimate;
    if (typeof quota !== 'number' || typeof usage !== 'number') return { needed, available: null };
    const available = Math.max(0, quota - usage);
    if (needed > available) {
        fail(`not enough storage in OPFS — ${Math.ceil(needed / 1e6)} MB needed, ${Math.floor(available / 1e6)} MB available.`);
    }
    return { needed, available };
}
