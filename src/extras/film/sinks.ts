// D2.4 (offline-films-browser-design.md §4.4, §4.5, §4.6): where frames go. The video sink wraps each
// VideoFrame in a mediabunny VideoSample and adds it to a VideoSampleSource, whose promise carries the
// encoder's and the writer's backpressure, and it closes the sample whatever happens: VideoSampleSource.add
// does not close samples (1.61.3), and unclosed VideoFrames stall the encoder. The muxer writes MP4 (H.264
// unless the app opts in, Q7) or WebM into memory, within a payload budget (Q3), or through a FileStreamGate
// to OPFS or a picked file, MP4 with its index reserved at the front (phase F3). The PNG sink writes each
// frame's exact bytes as a file of its own in OPFS. The hash sink digests read-back bytes, for the
// determinism tests (§2, P1).

import type { MediabunnyLike, MediabunnyVideoSampleLike } from './mediabunny';
import { CONTAINER_TYPES, type FilmCodec, type FilmOutput } from './output';
import { encodePNG } from './png';
import { predictBytes } from './plan';
import { chooseCodecs } from './probe';
import { frameFileName, type FileStreamGate } from './storage';
import { clause, fail, sentence } from './util';

export interface VideoSinkOptions {
    readonly mediabunny: MediabunnyLike;
    readonly width: number;
    readonly height: number;
    readonly fps: number;
    /** J: the track declares it, and no more are taken. */
    readonly frames: number;
    readonly output: FilmOutput;
    /** Bits a second, resolved (`output.bitrate ?? bitrateFor(…)`). */
    readonly bitrate: number;
    /** The file the muxer writes through (OPFS, or a picked file); absent, the film stays in memory. */
    readonly gate?: FileStreamGate | null;
}

/** A film's frames in, its file out: the video sink, or the PNG sink. */
export interface VideoSink {
    readonly codec: FilmCodec;
    readonly mimeType: string;
    /** The encoder config mediabunny settled on; null until the first frame configures the encoder (and for PNG). */
    readonly encoderConfig: VideoEncoderConfig | null;
    /** Frames encoded (a video's packets, a sequence's files). */
    readonly packets: number;
    readonly keyPackets: number;
    /** Encoded payload so far, in bytes: what the memory budget limits. */
    readonly bytes: number;
    /**
     * Takes ownership of `frame` (§4.6): it is closed whether the add succeeds or fails, and closed by the
     * sink even when wrapping it fails. Resolves when the encoder and the writer can take more.
     */
    add(frame: VideoFrame): Promise<void>;
    /** Flush the encoder and finalize the file: the film in memory as a Blob, or null once its file is committed. */
    finish(): Promise<Blob | null>;
    /** Stop and discard: nothing it wrote is kept (a file's writable is aborted). Idempotent. */
    cancel(): Promise<void>;
}

const mb2 = (bytes: number) => (bytes / 1e6).toFixed(1);
const errorWords = (error: unknown) => (error instanceof Error ? error.message : String(error));
const isQuota = (error: unknown) => (error as { name?: unknown } | null)?.name === 'QuotaExceededError';

/** §8: a write the file refused, in plain words, at the frame where the sink learned of it. */
function writeFailed(at: number, error: unknown, where: string): Error {
    const words = `${(error as { name?: string } | null)?.name ?? 'Error'}: ${errorWords(error)}`;
    return new Error(isQuota(error)
        ? `film: ${where} ran out of room at frame ${at} (${clause(words)}); free some storage, or film shorter or at a lower bitrate.`
        : `film: writing the film to ${where} failed at frame ${at}: ${sentence(words)}`, { cause: error });
}

export async function createVideoSink(o: VideoSinkOptions): Promise<VideoSink> {
    const { mediabunny: mb, width, height, fps, frames, output, bitrate } = o;
    const gate = o.gate ?? null;
    const where = output.target === 'opfs' || output.target === 'auto' ? 'OPFS' : 'the chosen file';
    const seconds = frames / fps;
    // Q3: memory only within the payload budget, refused before frame 0 when the prediction is over it.
    if (!gate) {
        const predicted = predictBytes(bitrate, seconds);
        if (predicted > output.memoryBudgetBytes) {
            fail(`a ${seconds} s film at ${(bitrate / 1e6).toFixed(1)} Mbit/s is expected to take ${mb2(predicted)} MB, over the in-memory budget of ` +
                `${mb2(output.memoryBudgetBytes)} MB; raise output.memoryBudgetBytes, shorten the film, or film to OPFS.`);
        }
    }
    const reasons: string[] = [];
    const [codec] = await chooseCodecs(mb, width, height, fps, output.container, output.codecs, bitrate, reasons);
    if (!codec) fail((reasons.at(-1) ?? `no WebCodecs encoder for ${width}×${height}.`).replace(/^film: /, ''));

    const configs: VideoEncoderConfig[] = [];
    let packets = 0, keyPackets = 0, bytes = 0;
    let overBudget: Error | null = null;
    const source = new mb.VideoSampleSource({
        codec, quality: new mb.Quality({ bitrate }), keyFrameInterval: 2, latencyMode: 'quality', alpha: 'discard',
        hardwareAcceleration: 'no-preference',
        onEncoderConfig: config => { configs.push(config); },
        onEncodedPacket: packet => {
            packets++;
            bytes += packet.byteLength;
            if (packet.type === 'key') keyPackets++;
            if (!gate && bytes > output.memoryBudgetBytes && !overBudget) {
                overBudget = new Error(`film: the film passed its in-memory budget of ${mb2(output.memoryBudgetBytes)} MB at ${packets} frames ` +
                    `(${mb2(bytes)} MB encoded); raise output.memoryBudgetBytes, shorten the film, or film to OPFS.`);
            }
        },
    });
    const memory = gate ? null : new mb.BufferTarget();
    // A file's MP4 reserves its index (moov) at the front for J packets, so the file plays as it loads (§4.5).
    const mp4 = output.container === 'mp4';
    const format = mp4 ? new mb.Mp4OutputFormat({ fastStart: gate ? 'reserve' : 'in-memory' }) : new mb.WebMOutputFormat();
    const muxer = new mb.Output({ format, target: gate ? new mb.StreamTarget(gate.stream, { chunked: true }) : memory! });
    try {
        muxer.addVideoTrack(source, { frameRate: fps, maximumPacketCount: frames });
        await muxer.start();
    } catch (error) {
        // Until this function returns, no session ledger owns the muxer. Unwind partial setup
        // here, preserving the initialization error even if cancellation also fails.
        gate?.discard();
        try { await muxer.cancel(); } catch { /* primary error wins */ }
        await gate?.abort();
        throw error;
    }
    const mimeType = CONTAINER_TYPES[output.container].mimeType;
    let ended = false;
    let added = 0;
    // §8: the encoder's own words, with the frame at which the sink learned of the failure (WebCodecs
    // reports errors late, through mediabunny's next add or finalize); a write the file refused, in its own.
    const failed = (at: number, error: unknown) => (gate?.writeError
        ? writeFailed(at, gate.writeError, where)
        : new Error(`film: video encoder failed at frame ${at}: ${sentence(errorWords(error))}`, { cause: error }));

    return {
        codec, mimeType,
        get encoderConfig() { return configs.at(-1) ?? null; },
        get packets() { return packets; },
        get keyPackets() { return keyPackets; },
        get bytes() { return bytes; },
        async add(frame) {
            let sample: MediabunnyVideoSampleLike;
            try {
                if (ended) fail('the video sink has finished; it takes no more frames.');
                if (overBudget) throw overBudget;
                sample = new mb.VideoSample(frame);
            } catch (error) {
                frame.close();
                throw error;
            }
            const at = added++;
            try {
                await source.add(sample);
            } catch (error) {
                throw failed(at, error);
            } finally {
                sample.close();
            }
            if (overBudget) throw overBudget;
        },
        async finish() {
            ended = true;
            if (overBudget) throw overBudget;
            try {
                await muxer.finalize();
            } catch (error) {
                throw failed(added, error);
            }
            // WebCodecs may emit its remaining packets only during flush/finalize. The last
            // add's check cannot see them; refuse before allocating the result Blob as well.
            if (overBudget) throw overBudget;
            if (gate) {
                // The muxer finalized and closed its stream: only now is the file committed. A finalize that fails
                // closes the stream too (1.61.3's finally), and the session then aborts the file instead.
                try {
                    await gate.commit();
                } catch (error) {
                    if ((error as Error | null)?.message?.startsWith('film: ')) throw error;
                    throw writeFailed(added, error, where);
                }
                return null;
            }
            if (!memory?.buffer) fail('the muxer finalized no file.');
            return new Blob([memory.buffer], { type: mimeType });
        },
        async cancel() {
            ended = true;
            // The gate first, so that mediabunny's cancel, which closes its writer (1.61.3), aborts the file at once.
            gate?.discard();
            try {
                if (muxer.state !== 'finalized' && muxer.state !== 'canceled') await muxer.cancel();
            } catch (error) {
                // A write the file refused errored the stream, so mediabunny's cancel cannot close it: that write
                // already ended the film (its error is the film's), and nothing is left to cancel.
                if (!gate?.writeError) throw error;
            } finally {
                // An output already finalizing cannot be cancelled (1.61.3): abort the file directly.
                await gate?.abort();
            }
        },
    };
}

export interface PngSinkOptions {
    /** The export's frames folder in OPFS (storage.ts). */
    readonly frames: FileSystemDirectoryHandle;
    readonly width: number;
    readonly height: number;
}

/**
 * A PNG sequence (§4.4): each frame's exact RGBA8 bytes as frame-000000.png, frame-000001.png, … in OPFS,
 * one file committed a frame. Frames must be 8-bit RGB(A) VideoFrames at the film's size; recordFilm hands
 * it read-back frames, whose bytes are exact. Each file is committed as it is written: a discard removes the
 * export's folder, frames and all (storage.ts).
 */
export function createPngSink(o: PngSinkOptions): VideoSink {
    const { frames, width, height } = o;
    let written = 0, bytes = 0;
    let ended = false;
    let scratch: Uint8Array | null = null;
    const rgba = new Uint8Array(width * height * 4);
    return {
        codec: 'png', mimeType: 'image/png', encoderConfig: null,
        get packets() { return written; },
        get keyPackets() { return written; },
        get bytes() { return bytes; },
        async add(frame) {
            const at = written;
            try {
                if (ended) fail('the PNG sink has finished; it takes no more frames.');
                const format = frame.format;
                if (format !== 'RGBA' && format !== 'RGBX' && format !== 'BGRA' && format !== 'BGRX') {
                    fail(`a PNG frame must be 8-bit RGB with or without alpha; frame ${at} is ${format}.`);
                }
                if (frame.codedWidth !== width || frame.codedHeight !== height) {
                    fail(`frame ${at} is ${frame.codedWidth}×${frame.codedHeight}; the film is ${width}×${height}.`);
                }
                const size = frame.allocationSize();
                if (!scratch || scratch.length < size) scratch = new Uint8Array(size);
                const [plane] = await frame.copyTo(scratch);
                const bgr = format.startsWith('B');
                const opaque = format.endsWith('X');
                for (let y = 0; y < height; y++) {
                    for (let x = 0, s = plane.offset + y * plane.stride, d = y * width * 4; x < width; x++, s += 4, d += 4) {
                        rgba[d] = scratch[s + (bgr ? 2 : 0)];
                        rgba[d + 1] = scratch[s + 1];
                        rgba[d + 2] = scratch[s + (bgr ? 0 : 2)];
                        rgba[d + 3] = opaque ? 255 : scratch[s + 3];
                    }
                }
                const png = await encodePNG(rgba, width, height);
                // The file's entry and its writable can be refused too (no room for the swap file, say): §8's words.
                let writable: FileSystemWritableFileStream | null = null;
                try {
                    writable = await (await frames.getFileHandle(frameFileName(at), { create: true })).createWritable();
                    await writable.write(png as Uint8Array<ArrayBuffer>);
                    await writable.close();
                } catch (error) {
                    try { await writable?.abort(); } catch { /* the write's own error stands */ }
                    throw writeFailed(at, error, 'OPFS');
                }
                written++;
                bytes += png.byteLength;
            } finally {
                frame.close();
            }
        },
        async finish() {
            ended = true;
            return null;
        },
        async cancel() {
            ended = true;
        },
    };
}

/** SHA-256 of `bytes`, in hex: the hash sink's digest of an exact read-back frame. */
export async function sha256(bytes: Uint8Array): Promise<string> {
    const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes as Uint8Array<ArrayBuffer>));
    return Array.from(digest, b => b.toString(16).padStart(2, '0')).join('');
}
