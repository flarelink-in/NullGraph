// D2.4 (offline-films-browser-design.md §4.4, §4.5, §4.6): where frames go. The video sink wraps each
// VideoFrame in a mediabunny VideoSample and adds it to a VideoSampleSource, whose promise carries the
// encoder's and the writer's backpressure, and it closes the sample whatever happens: VideoSampleSource.add
// does not close samples (1.61.3), and unclosed VideoFrames stall the encoder. The muxer writes MP4 (H.264
// unless the app opts in, Q7) or WebM into memory, within a payload budget (Q3); OPFS and a picked file are
// phase F3. The hash sink digests read-back bytes, for the determinism tests (§2, P1).

import type { MediabunnyLike, MediabunnyVideoSampleLike } from './mediabunny';
import type { FilmCodec, FilmOutput } from './output';
import { predictBytes } from './plan';
import { chooseCodecs } from './probe';
import { fail, sentence } from './util';

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
}

export interface VideoSink {
    readonly codec: FilmCodec;
    readonly mimeType: string;
    /** The encoder config mediabunny settled on; null until the first frame configures the encoder. */
    readonly encoderConfig: VideoEncoderConfig | null;
    readonly packets: number;
    readonly keyPackets: number;
    /** Encoded payload so far, in bytes: what the memory budget limits. */
    readonly bytes: number;
    /**
     * Takes ownership of `frame` (§4.6): it is closed whether the add succeeds or fails, and closed by the
     * sink even when wrapping it fails. Resolves when the encoder and the writer can take more.
     */
    add(frame: VideoFrame): Promise<void>;
    /** Flush the encoder and finalize the file. */
    finish(): Promise<Blob>;
    /** Stop and discard; idempotent. */
    cancel(): Promise<void>;
}

const mb2 = (bytes: number) => (bytes / 1e6).toFixed(1);

export async function createVideoSink(o: VideoSinkOptions): Promise<VideoSink> {
    const { mediabunny: mb, width, height, fps, frames, output, bitrate } = o;
    if (output.target !== 'memory' && output.target !== 'auto') {
        fail('OPFS and picked-file targets arrive in phase F3; record to output.target "memory" (or "auto") for now.');
    }
    // Q3: memory only within the payload budget, refused before frame 0 when the prediction is over it.
    const seconds = frames / fps;
    const predicted = predictBytes(bitrate, seconds);
    if (predicted > output.memoryBudgetBytes) {
        fail(`a ${seconds} s film at ${(bitrate / 1e6).toFixed(1)} Mbit/s is expected to take ${mb2(predicted)} MB, over the in-memory budget of ` +
            `${mb2(output.memoryBudgetBytes)} MB; raise output.memoryBudgetBytes or shorten the film (OPFS streaming is phase F3).`);
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
            if (bytes > output.memoryBudgetBytes && !overBudget) {
                overBudget = new Error(`film: the film passed its in-memory budget of ${mb2(output.memoryBudgetBytes)} MB at ${packets} frames ` +
                    `(${mb2(bytes)} MB encoded); raise output.memoryBudgetBytes or shorten the film.`);
            }
        },
    });
    const target = new mb.BufferTarget();
    const format = output.container === 'mp4' ? new mb.Mp4OutputFormat({ fastStart: 'in-memory' }) : new mb.WebMOutputFormat();
    const muxer = new mb.Output({ format, target });
    try {
        muxer.addVideoTrack(source, { frameRate: fps, maximumPacketCount: frames });
        await muxer.start();
    } catch (error) {
        // Until this function returns, no session ledger owns the muxer. Unwind partial setup
        // here, preserving the initialization error even if cancellation also fails.
        try { await muxer.cancel(); } catch { /* primary error wins */ }
        throw error;
    }
    const mimeType = output.container === 'mp4' ? 'video/mp4' : 'video/webm';
    let ended = false;
    let added = 0;
    // §8: the encoder's own words, with the frame at which the sink learned of the failure (WebCodecs
    // reports errors late, through mediabunny's next add or finalize).
    const encoderFailed = (at: number, error: unknown) => new Error(
        `film: video encoder failed at frame ${at}: ${sentence(error instanceof Error ? error.message : String(error))}`, { cause: error });

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
                throw encoderFailed(at, error);
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
                throw encoderFailed(added, error);
            }
            // WebCodecs may emit its remaining packets only during flush/finalize. The last
            // add's check cannot see them; refuse before allocating the result Blob as well.
            if (overBudget) throw overBudget;
            if (!target.buffer) fail('the muxer finalized no file.');
            return new Blob([target.buffer], { type: mimeType });
        },
        async cancel() {
            ended = true;
            if (muxer.state !== 'finalized' && muxer.state !== 'canceled') await muxer.cancel();
        },
    };
}

/** SHA-256 of `bytes`, in hex: the hash sink's digest of an exact read-back frame. */
export async function sha256(bytes: Uint8Array): Promise<string> {
    const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes as Uint8Array<ArrayBuffer>));
    return Array.from(digest, b => b.toString(16).padStart(2, '0')).join('');
}
