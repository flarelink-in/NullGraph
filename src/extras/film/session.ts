// D2.4 (offline-films-browser-design.md §4.9): the export session. Its lifecycle's states and transitions,
// the ledger that unwinds what a session acquired (last-first, attempting every release), and FilmSession:
// the frames-in, file-out contract an app drives with its own loop and recordFilm drives for it.

import { loadMediabunny, type MediabunnyLike } from './mediabunny';
import { readOutput, type FilmCodec, type FilmOutput, type FilmOutputOptions } from './output';
import { bitrateFor, frameStamp, planFilm, readFilm, type FilmSettings } from './plan';
import { createVideoSink, type VideoSink } from './sinks';
import { fail } from './util';

/**
 * A session's states, in the order a film passes through them. Every path ends in `disposing`, so
 * teardown always runs before the operation settles, then in one terminal state.
 */
export type FilmState =
    | 'probing' | 'initializing' | 'calibrating' | 'resetting' | 'recording'
    | 'stopping' | 'finalizing' | 'disposing'
    | 'completed' | 'partial' | 'cancelled' | 'failed';

/**
 * The legal next states. A failure or a cancel goes to `disposing` from any working state, except while
 * recording: `stopping` first settles the samples and read-backs in flight. A finished film goes from
 * `recording` straight to `finalizing`; a kept partial goes through `stopping`. recordFilm calibrates and
 * resets before recording; an app driving its own loop goes from `initializing` to `recording`.
 */
export const FILM_TRANSITIONS: Readonly<Record<FilmState, readonly FilmState[]>> = {
    probing: ['initializing', 'disposing'],
    initializing: ['calibrating', 'recording', 'disposing'],
    calibrating: ['resetting', 'disposing'],
    resetting: ['recording', 'disposing'],
    recording: ['finalizing', 'stopping'],
    stopping: ['finalizing', 'disposing'],
    finalizing: ['disposing'],
    disposing: ['completed', 'partial', 'cancelled', 'failed'],
    completed: [],
    partial: [],
    cancelled: [],
    failed: [],
};

export const isTerminal = (state: FilmState): boolean => FILM_TRANSITIONS[state].length === 0;

/** A session's state, and when it entered each one. Refuses an illegal transition by name. */
export class FilmLifecycle {
    private current: FilmState = 'probing';
    /** Each state entered, with `performance.now()` (or `Date.now()` where there is none) at entry. */
    readonly history: { readonly state: FilmState; readonly at: number }[] = [{ state: 'probing', at: now() }];

    get state(): FilmState {
        return this.current;
    }

    to(next: FilmState): void {
        const legal = FILM_TRANSITIONS[this.current];
        if (!legal.includes(next)) {
            const list = legal.length > 1 ? `${legal.slice(0, -1).join(', ')} or ${legal.at(-1)}` : legal[0] ?? 'nothing (it has ended)';
            fail(`a session cannot go from ${this.current} to ${next}; from ${this.current} it goes to ${list}.`);
        }
        this.current = next;
        this.history.push({ state: next, at: now() });
    }
}

const now = () => (globalThis.performance ? performance.now() : Date.now());

/** A release that failed: which one, and what it threw. */
export interface CleanupError {
    readonly name: string;
    readonly error: unknown;
}

/**
 * What a session acquired, released last-first. Register each resource as soon as it exists, before the
 * next operation that can fail, so a half-built session unwinds too (§4.6). `unwind` attempts every
 * release whatever the others do and returns the failures: they are secondary entries in the report,
 * never a replacement for the error that ended the film. A second `unwind` does nothing. A release
 * registered after the unwind runs at once, so nothing acquired late is leaked.
 */
export class CleanupLedger {
    private entries: { readonly name: string; readonly release: () => void | Promise<void> }[] = [];
    private unwound = false;
    private late: Promise<void>[] = [];
    /** Failures of releases that ran at registration, after the unwind. */
    readonly lateErrors: CleanupError[] = [];

    get size(): number {
        return this.entries.length;
    }

    defer(name: string, release: () => void | Promise<void>): void {
        if (!this.unwound) {
            this.entries.push({ name, release });
            return;
        }
        this.late.push((async () => {
            try { await release(); } catch (error) { this.lateErrors.push({ name, error }); }
        })());
    }

    async unwind(): Promise<CleanupError[]> {
        if (this.unwound) {
            await Promise.all(this.late);
            return [];
        }
        this.unwound = true;
        const errors: CleanupError[] = [];
        for (const { name, release } of this.entries.reverse()) {
            try { await release(); } catch (error) { errors.push({ name, error }); }
        }
        this.entries = [];
        return errors;
    }
}

// ---------------------------------------------------------------------------------------------------
// FilmSession: frames in, a file out
// ---------------------------------------------------------------------------------------------------

/** What a session made, and how it ended (§4.8). recordFilm's report extends it. */
export interface FilmSessionReport {
    readonly exportId: string;
    /** The terminal state once the session has ended; the current state before. */
    readonly status: FilmState;
    readonly film: FilmSettings;
    readonly output: {
        readonly container: FilmOutput['container'];
        readonly codecs: readonly FilmCodec[];
        readonly codec: FilmCodec | null;
        /** Asked of the encoder, bits a second. */
        readonly bitrate: number;
        readonly target: 'memory';
        readonly memoryBudgetBytes: number;
        readonly name: string;
    } | null;
    /** The config mediabunny settled on: what was asked of WebCodecs, not proof of a hardware encoder. */
    readonly encoderConfig: VideoEncoderConfig | null;
    readonly frames: { readonly planned: number; readonly accepted: number; readonly encoded: number; readonly keyFrames: number };
    readonly bytes: number;
    /** Encoded bits a second over the frames encoded. Spike S1 found encoders far from the asked rate (Q9, parked). */
    readonly achievedBitrate: number | null;
    readonly timing: { readonly addWaitMs: number; readonly finalizeMs: number; readonly quantization: string };
    readonly states: readonly { readonly state: FilmState; readonly at: number }[];
    readonly cancelled: boolean;
    /** What ended the film, when something did. */
    readonly failure: string | null;
    /** Releases that failed during teardown: secondary, never the failure itself. */
    readonly cleanupErrors: readonly string[];
}

export interface FilmResult<R extends FilmSessionReport = FilmSessionReport> {
    /** `'partial'`: a film explicitly kept incomplete, finalized and playable, `report.frames` saying how much. */
    readonly status: 'completed' | 'partial';
    /** The file in memory: a Blob of `video/mp4` or `video/webm`. Null for a session with no video output. */
    readonly file: Blob | null;
    readonly name: string | null;
    readonly codec: FilmCodec | null;
    /** Frames in the file. */
    readonly frames: number;
    readonly report: R;
    /** Release the result's backing storage: nothing for a file in memory (the app revokes its own URLs). Idempotent. */
    dispose(): Promise<void>;
}

export interface FilmSessionOptions {
    readonly film: { readonly width: number; readonly height: number; readonly fps: number; readonly seconds: number; readonly warmupSeconds?: number };
    /** The video file; `false` for a session that encodes nothing (a runner that only hashes or measures). */
    readonly output?: FilmOutputOptions | false;
    /** mediabunny's namespace, when the app has it already; imported otherwise. */
    readonly mediabunny?: MediabunnyLike;
    /** Aborting cancels the session: everything is discarded. */
    readonly signal?: AbortSignal;
}

const abortError = () => new DOMException('Export cancelled', 'AbortError');
const isAbort = (error: unknown) => (error as { name?: unknown } | null)?.name === 'AbortError';
const describe = (error: unknown) => (error instanceof Error ? error.message : String(error));
const round = (ms: number) => Math.round(ms * 10) / 10;

/**
 * Frames in, a file out (§4.9). `add` takes each frame's ownership on invocation, requires exactly the
 * planned timestamp T(j) (catching a frame given in seconds, or one skipped), and serializes. `finish`
 * needs all J frames; `stop()` keeps a finalized, playable partial (or discards one with no frames);
 * `cancel()` discards. The first terminal call wins and later ones return its outcome; teardown runs
 * before any of them settles.
 */
export class FilmSession {
    readonly exportId: string;
    readonly film: FilmSettings;
    /** J: the film's frames at film.fps. */
    readonly frames: number;
    readonly output: FilmOutput | null;
    readonly lifecycle = new FilmLifecycle();
    /** What the session, and a runner driving it, acquired: released last-first at teardown. */
    readonly ledger = new CleanupLedger();
    private sink: VideoSink | null = null;
    private readonly bitrate: number;
    private accepted = 0;
    private counted = 0;
    private chain: Promise<void> = Promise.resolve();
    private failure: unknown = null;
    private cancelled = false;
    private sinkFailed = false;
    private terminal: Promise<FilmResult | null> | null = null;
    private addWaitMs = 0;
    private finalizeMs = 0;
    private cleanupErrors: CleanupError[] = [];
    /** A discard has cancelled the encoder: what adds in flight then throw is its consequence, not the failure. */
    private discarding = false;
    /** Set by abandon(): every wait on the encoder or muxer, in flight or to come, fails with it. */
    private abandoned: unknown = null;
    /** Interrupts the one wait on the encoder or muxer in flight: adds are serialized, and end() waits on them first. */
    private interrupt: ((error: unknown) => void) | null = null;

    private constructor(film: FilmSettings, frames: number, output: FilmOutput | null, bitrate: number) {
        this.exportId = globalThis.crypto?.randomUUID?.() ?? `film-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
        this.film = film;
        this.frames = frames;
        this.output = output;
        this.bitrate = bitrate;
    }

    /** Check the options, make the encoder and muxer, and return the session, ready for frames. */
    static async open(options: FilmSessionOptions): Promise<FilmSession> {
        const film = readFilm(options.film);
        const frames = planFilm({ fps: film.fps, seconds: film.seconds }, { dt: 1 / film.fps }).frames;
        const output = options.output === false ? null : readOutput(options.output ?? {});
        const session = new FilmSession(film, frames, output, output?.bitrate ?? bitrateFor(film.width, film.height, film.fps));
        session.lifecycle.to('initializing');
        try {
            if (options.signal?.aborted) throw abortError();
            if (output) {
                const mediabunny = await loadMediabunny(options.mediabunny);
                const sink = await createVideoSink({ mediabunny, width: film.width, height: film.height, fps: film.fps, frames, output, bitrate: session.bitrate });
                session.sink = sink;
                session.ledger.defer('video encoder', () => sink.cancel());
            }
            const signal = options.signal;
            if (signal) {
                // Setup yields while loading/probing/starting the encoder. Abort events are not
                // replayed for listeners installed after that asynchronous work.
                if (signal.aborted) throw abortError();
                const onAbort = () => { void session.cancel(); };
                signal.addEventListener('abort', onAbort, { once: true });
                session.ledger.defer('abort listener', () => signal.removeEventListener('abort', onAbort));
            }
        } catch (error) {
            await session.end('discard', error, isAbort(error));
            throw error;
        }
        return session;
    }

    get state(): FilmState {
        return this.lifecycle.state;
    }

    /** Frames taken so far. */
    get added(): number {
        return this.sink ? this.accepted : this.counted;
    }

    get codec(): FilmCodec | null {
        return this.sink?.codec ?? null;
    }

    /** Encoded payload so far, in bytes. */
    get bytes(): number {
        return this.sink?.bytes ?? 0;
    }

    /** For a runner that calibrates and resets before frames (recordFilm): move the lifecycle on. */
    advance(state: FilmState): void {
        this.lifecycle.to(state);
    }

    /** For a session with no video output: count a frame the runner produced (hashed, or only measured). */
    noteFrame(): void {
        this.counted++;
    }

    add(frame: VideoFrame): Promise<void> {
        if (this.terminal) {
            frame.close();
            return Promise.reject(this.cancelled ? abortError() : new Error(`film: the session has ended (${this.lifecycle.state}); it takes no more frames.`));
        }
        const run = async () => {
            // Queued before a cancel, run after it: closed, never encoded, so the discard is not kept waiting.
            if (this.cancelled || this.abandoned) {
                frame.close();
                throw this.abandoned ?? abortError();
            }
            if (this.lifecycle.state === 'initializing') this.lifecycle.to('recording');
            const state = this.lifecycle.state;
            if (state !== 'recording' && state !== 'stopping') {
                frame.close();
                fail(`the session is ${state}; frames start in recording.`);
            }
            if (this.failure) {
                frame.close();
                throw this.failure;
            }
            const j = this.accepted;
            if (j >= this.frames) {
                frame.close();
                fail(`the film has all its ${this.frames} frames; finish() it.`);
            }
            const want = frameStamp(j, this.film.fps).timestampUs;
            if (frame.timestamp !== want) {
                frame.close();
                fail(`frame ${j} must be stamped ${want} µs (T(j) = round(j × 10⁶ / ${this.film.fps})); got ${frame.timestamp}.`);
            }
            this.accepted++;
            const t = now();
            try {
                if (this.sink) await this.waitOn(this.sink.add(frame)); else frame.close();
            } catch (error) {
                this.sinkFailed = true;
                if (!this.discarding) this.failure ??= error;
                throw error;
            } finally {
                this.addWaitMs += now() - t;
            }
        };
        const done = this.chain.then(run);
        this.chain = done.catch(() => {});
        return done;
    }

    /** All J frames are in: flush, finalize, tear down, and return the film. */
    async finish(): Promise<FilmResult> {
        if (!this.terminal) {
            await this.chain;
            if (this.failure) {
                await this.end('discard');
                throw this.failure;
            }
            if (this.sink && this.accepted < this.frames) {
                fail(`finish() after ${this.accepted} of ${this.frames} frames; add the rest, or stop({ keepPartial: true }) to keep a partial film.`);
            }
        }
        const result = await this.end('finish');
        if (!result) throw this.failure ?? (this.cancelled ? abortError() : new Error('film: the film was not finalized.'));
        return result;
    }

    /** Stop taking frames; keep a finalized partial film if there is one (`keepPartial`, the default), else discard. */
    stop(options: { readonly keepPartial?: boolean } = {}): Promise<FilmResult | null> {
        return this.end(options.keepPartial === false ? 'discard' : 'keep');
    }

    /** Discard everything. Resolves once torn down. */
    async cancel(): Promise<void> {
        await this.end('discard', null, true);
    }

    /**
     * End with a runner's `error`: discard, or keep a finalizable partial when `keepPartial`. `cancelled`
     * (default: `error` is an AbortError) says the app asked for the end; a cancel always discards.
     */
    abort(error: unknown, options: { readonly keepPartial?: boolean; readonly cancelled?: boolean } = {}): Promise<FilmResult | null> {
        return this.end(options.keepPartial ? 'keep' : 'discard', error, options.cancelled ?? isAbort(error));
    }

    /**
     * Stop waiting on an encoder that will not answer, and discard. A lost GPU process takes Chrome's hardware
     * encoder with it, and an add or a finalize waiting on it then never returns (G9-crash); recordFilm calls
     * this when its device is lost and the encoder is still silent. The wait in flight, and every one after
     * it, fails with `error`, even inside a finish() already under way; the session then tears down as a
     * failure. What mediabunny still holds of the dead encoder is left to the garbage collector.
     */
    abandon(error: unknown): Promise<FilmResult | null> {
        // Tearing down or ended: nothing waits on the encoder any more, and the outcome is already decided.
        if (this.lifecycle.state === 'disposing' || isTerminal(this.lifecycle.state)) return this.terminal ?? Promise.resolve(null);
        this.abandoned ??= error;
        this.failure ??= error;
        this.interrupt?.(this.abandoned);
        return this.end('discard', error);
    }

    /** `work` (a wait on the encoder or muxer), unless abandon() comes first. */
    private waitOn<T>(work: Promise<T>): Promise<T> {
        if (this.abandoned) {
            work.catch(() => {});
            return Promise.reject(this.abandoned);
        }
        return new Promise<T>((resolve, reject) => {
            this.interrupt = reject;
            work.then(resolve, reject).finally(() => { if (this.interrupt === reject) this.interrupt = null; });
        });
    }

    report(): FilmSessionReport {
        const sink = this.sink;
        const encodedSeconds = sink?.packets ? sink.packets / this.film.fps : 0;
        const output = this.output;
        return {
            exportId: this.exportId,
            status: this.lifecycle.state,
            film: this.film,
            output: output && {
                container: output.container, codecs: output.codecs, codec: sink?.codec ?? null, bitrate: this.bitrate,
                target: 'memory', memoryBudgetBytes: output.memoryBudgetBytes, name: output.name,
            },
            encoderConfig: sink?.encoderConfig ? JSON.parse(JSON.stringify(sink.encoderConfig)) as VideoEncoderConfig : null,
            frames: { planned: this.frames, accepted: this.added, encoded: sink?.packets ?? 0, keyFrames: sink?.keyPackets ?? 0 },
            bytes: sink?.bytes ?? 0,
            achievedBitrate: sink && encodedSeconds ? Math.round(sink.bytes * 8 / encodedSeconds) : null,
            timing: { addWaitMs: round(this.addWaitMs), finalizeMs: round(this.finalizeMs), quantization: 'T(j) = round(j × 10⁶ / fps) µs; D(j) = T(j + 1) − T(j)' },
            states: [...this.lifecycle.history],
            cancelled: this.cancelled,
            failure: this.failure && !(this.cancelled && isAbort(this.failure)) ? describe(this.failure) : null,
            cleanupErrors: this.cleanupErrors.map(e => `${e.name}: ${describe(e.error)}`),
        };
    }

    private end(intent: 'finish' | 'keep' | 'discard', error: unknown = null, cancelled = false): Promise<FilmResult | null> {
        if (this.terminal) return this.terminal;
        this.terminal = (async () => {
            if (error) this.failure ??= error;
            if (cancelled) this.cancelled = true;
            const early: CleanupError[] = [];
            // A discard need not wait for the encoder: cancelling it first releases an add still waiting on it,
            // which an encoder lost with the GPU process would hold for ever (G9-crash).
            if (this.sink && (intent === 'discard' || this.cancelled)) {
                this.discarding = true;
                try { await this.sink.cancel(); } catch (cancelError) { early.push({ name: 'video encoder', error: cancelError }); }
            }
            // An unrecoverable error overrides success; a user's cancel always discards.
            await this.chain; // the frames already handed over settle: encoded, or closed
            // Queued adds can enter recording or fail while we wait. Decide from their settled
            // outcome, not the state at the instant stop/finish was called.
            const goal = this.cancelled || this.sinkFailed || (intent === 'finish' && this.failure) ? 'discard' : intent;
            if (this.lifecycle.state === 'recording' && goal !== 'finish') this.lifecycle.to('stopping');
            let file: Blob | null = null;
            let status: 'completed' | 'partial' | null = null;
            const working = this.lifecycle.state === 'recording' || this.lifecycle.state === 'stopping';
            const finalize = working && (goal === 'finish' || (goal === 'keep' && this.added > 0 && !this.sinkFailed));
            if (finalize) {
                this.lifecycle.to('finalizing');
                const t = now();
                try {
                    file = this.sink ? await this.waitOn(this.sink.finish()) : null;
                    status = goal === 'finish' ? 'completed' : 'partial';
                } catch (finalizeError) {
                    this.failure ??= finalizeError;
                } finally {
                    this.finalizeMs = now() - t;
                }
            }
            this.lifecycle.to('disposing');
            this.cleanupErrors = [...early, ...await this.ledger.unwind()];
            // Nothing kept: cancelled when the app asked for the end (or stopped before any frame), else failed.
            this.lifecycle.to(status ?? (this.failure && !this.cancelled ? 'failed' : 'cancelled'));
            if (!status) return null;
            const report = this.report();
            return {
                status, file, name: this.output?.name ?? null, codec: this.sink?.codec ?? null, frames: this.added, report,
                dispose: async () => {},
            };
        })();
        return this.terminal;
    }
}

/** Open a session for an app that draws and captures frames itself (§4.9, §6). */
export function createFilmSession(options: FilmSessionOptions): Promise<FilmSession> {
    return FilmSession.open(options);
}
