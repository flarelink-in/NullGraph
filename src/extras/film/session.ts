// D2.4 (offline-films-browser-design.md §4.5, §4.9): the export session. Its lifecycle's states and transitions,
// the ledger that unwinds what a session acquired (last-first, attempting every release), and FilmSession:
// the frames-in, file-out contract an app drives with its own loop and recordFilm drives for it. A session
// chooses where its file goes before frame 0 (memory, OPFS or a picked file) and never changes it; in storage
// it commits the film only once finalized, and otherwise leaves nothing behind.

import { loadMediabunny, type MediabunnyLike } from './mediabunny';
import { CONTAINER_TYPES, checkStorage, readOutput, type FilmCodec, type FilmOutput, type FilmOutputOptions } from './output';
import { bitrateFor, frameStamp, planFilm, predictBytes, readFilm, type FilmSettings } from './plan';
import { createPngSink, createVideoSink, type VideoSink } from './sinks';
import { frameFileName, openOpfsExport, openPickedFile, type FilmStorage } from './storage';
import { clause, fail } from './util';

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

/** Where a session's file went (§4.5): memory, OPFS, or the file the user picked. */
export type FilmTargetKind = 'memory' | 'opfs' | 'file';

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
        /** Asked of the encoder, bits a second (none for a PNG sequence). */
        readonly bitrate: number | null;
        /** Where the file went, chosen before frame 0 and never changed after. */
        readonly target: FilmTargetKind | null;
        /** Why: asked for, OPFS writable here, or memory because OPFS was not (with OPFS's reason). */
        readonly targetReason: string | null;
        readonly memoryBudgetBytes: number;
        readonly name: string;
    } | null;
    /** The film's file in OPFS or the picked file; null in memory. */
    readonly storage: {
        readonly path: string | null;
        /** The film's Web Lock was held while it was written (false where Web Locks are missing, or for a picked file). */
        readonly locked: boolean;
        /** The payload was committed and, in OPFS, its manifest marked complete or partial. */
        readonly committed: boolean;
        /** The storage check before frame 0: bytes predicted, and OPFS's free bytes (null when it gave none). */
        readonly needed: number | null;
        readonly available: number | null;
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
    /**
     * The video: a Blob in memory, or a File in OPFS or the picked file (read from disk, so it holds no memory).
     * Null for a PNG sequence (see `sequence`) and for a session with no output.
     */
    readonly file: Blob | null;
    readonly name: string | null;
    readonly codec: FilmCodec | null;
    /** Where the file is; null for a session with no output. */
    readonly target: FilmTargetKind | null;
    /** The film's folder in OPFS is null-graph-films/<exportId>/, as listFilms names it; null elsewhere. */
    readonly exportId: string | null;
    /** A PNG sequence's frames folder in OPFS and its files, frame-000000.png first; null for a video. */
    readonly sequence: { readonly directory: FileSystemDirectoryHandle; readonly names: readonly string[] } | null;
    /** Frames in the file. */
    readonly frames: number;
    readonly report: R;
    /**
     * Release the result's backing storage: the film's folder in OPFS (the File then reads no more); nothing for
     * a film in memory (the app revokes its own URLs) or for a picked file, which is the user's. Idempotent.
     */
    dispose(): Promise<void>;
}

export interface FilmSessionOptions {
    readonly film: { readonly width: number; readonly height: number; readonly fps: number; readonly seconds: number; readonly warmupSeconds?: number };
    /** The file; `false` for a session that encodes nothing (a runner that only hashes or measures). */
    readonly output?: FilmOutputOptions | false;
    /** mediabunny's namespace, when the app has it already; imported otherwise. */
    readonly mediabunny?: MediabunnyLike;
    /** Aborting cancels the session: everything is discarded. */
    readonly signal?: AbortSignal;
    /** What the manifest's report should carry beside the session's own (recordFilm's source, guard and timings). */
    readonly annotate?: () => Record<string, unknown>;
}

/** Where a session's file goes, resolved before frame 0 and never changed after (§4.5). */
interface ResolvedTarget {
    readonly kind: FilmTargetKind;
    readonly reason: string;
    readonly storage: FilmStorage | null;
    readonly needed: number | null;
    readonly available: number | null;
}

const mb2 = (bytes: number) => (bytes / 1e6).toFixed(1);
const errorWords = (error: unknown) => (error instanceof Error ? error.message : String(error)).replace(/^film: /, '');

/**
 * The target (§4.5): a picked file or memory when asked for; OPFS when asked for, or by `auto` wherever OPFS
 * is there, has room for the film's predicted bytes and opens. Without OPFS, `auto` keeps a video in memory
 * only within its budget, saying why, and otherwise refuses with both reasons: never a silent escalation.
 */
async function openTarget(output: FilmOutput, exportId: string, predicted: number): Promise<ResolvedTarget> {
    const wanted = output.target;
    const { mimeType, extension } = CONTAINER_TYPES[output.container];
    if (typeof wanted === 'object') {
        return { kind: 'file', reason: 'asked for', storage: await openPickedFile(wanted.file, exportId), needed: null, available: null };
    }
    if (wanted === 'memory') return { kind: 'memory', reason: 'asked for', storage: null, needed: null, available: null };
    let why: string;
    let available: number | null = null;
    try {
        const storage = (globalThis as { navigator?: Navigator }).navigator?.storage;
        if (typeof storage?.getDirectory !== 'function') fail('OPFS (navigator.storage.getDirectory) is not available here.');
        // An estimate that fails is no estimate: the check then refuses nothing, as the probe's does.
        if (typeof storage.estimate === 'function') available = checkStorage(predicted, await storage.estimate().catch(() => ({}))).available;
        let opened: FilmStorage;
        try {
            opened = await openOpfsExport({
                exportId, name: output.name, container: output.container, mimeType, extension, payload: output.container === 'png' ? 'png' : 'video',
            });
        } catch (error) {
            if ((error as Error | null)?.message?.startsWith('film: ')) throw error;
            fail(`OPFS could not be written here (${(error as { name?: string } | null)?.name ?? 'Error'}: ${clause(errorWords(error))}).`);
        }
        return { kind: 'opfs', reason: wanted === 'opfs' ? 'asked for' : 'OPFS is writable here (Q3)', storage: opened, needed: predicted, available };
    } catch (error) {
        if (wanted === 'opfs') throw error;
        why = clause(errorWords(error));
    }
    if (output.container === 'png') fail(`a PNG sequence is written to OPFS, and ${why}.`);
    if (predicted > output.memoryBudgetBytes) {
        fail(`${why}; and in memory the film's predicted ${mb2(predicted)} MB is over its budget of ${mb2(output.memoryBudgetBytes)} MB ` +
            '(raise output.memoryBudgetBytes, or shorten the film).');
    }
    return { kind: 'memory', reason: `memory, within its ${mb2(output.memoryBudgetBytes)} MB budget, because ${why}`, storage: null, needed: predicted, available };
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
    /** Where the file goes; null for a session with no output. */
    private target: ResolvedTarget | null = null;
    private annotate: (() => Record<string, unknown>) | null = null;

    private constructor(film: FilmSettings, frames: number, output: FilmOutput | null, bitrate: number) {
        this.exportId = globalThis.crypto?.randomUUID?.() ?? `film-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
        this.film = film;
        this.frames = frames;
        this.output = output;
        this.bitrate = bitrate;
    }

    /** Check the options, choose where the file goes, make the encoder and muxer, and return the session, ready for frames. */
    static async open(options: FilmSessionOptions): Promise<FilmSession> {
        const film = readFilm(options.film);
        const frames = planFilm({ fps: film.fps, seconds: film.seconds }, { dt: 1 / film.fps }).frames;
        const output = options.output === false ? null : readOutput(options.output ?? {});
        const session = new FilmSession(film, frames, output, output?.bitrate ?? bitrateFor(film.width, film.height, film.fps));
        session.annotate = options.annotate ?? null;
        session.lifecycle.to('initializing');
        try {
            if (options.signal?.aborted) throw abortError();
            if (output) {
                const png = output.container === 'png';
                // mediabunny first: a missing package should refuse before anything is written to storage.
                const mediabunny = png ? null : await loadMediabunny(options.mediabunny);
                // A PNG holds at most its raw scanlines (a filter byte and the row each) and a little: deflate's worst
                // case and the chunks. The bound the storage check uses.
                const predicted = png
                    ? Math.ceil((film.width * 4 + 1) * film.height * 1.001 + 1024) * frames
                    : predictBytes(session.bitrate, film.seconds);
                const target = await openTarget(output, session.exportId, predicted);
                session.target = target;
                const storage = target.storage;
                if (storage) {
                    // Registered before the sink, so released after it: the encoder's writable is aborted first, then
                    // the folder of a film never committed is removed, then its lock let go.
                    session.ledger.defer('film storage', async () => {
                        try {
                            if (!storage.committed) await storage.discard();
                        } finally {
                            storage.release();
                        }
                    });
                }
                const sink = png
                    ? createPngSink({ frames: storage!.frames!, width: film.width, height: film.height })
                    : await createVideoSink({
                        mediabunny: mediabunny!, width: film.width, height: film.height, fps: film.fps, frames, output, bitrate: session.bitrate,
                        gate: storage?.gate ?? null,
                    });
                session.sink = sink;
                session.ledger.defer(png ? 'PNG writer' : 'video encoder', () => sink.cancel());
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
        const png = output?.container === 'png';
        const target = this.target;
        const storage = target?.storage ?? null;
        return {
            exportId: this.exportId,
            status: this.lifecycle.state,
            film: this.film,
            output: output && {
                container: output.container, codecs: output.codecs, codec: sink?.codec ?? null, bitrate: png ? null : this.bitrate,
                target: target?.kind ?? null, targetReason: target?.reason ?? null, memoryBudgetBytes: output.memoryBudgetBytes, name: output.name,
            },
            storage: target && target.kind !== 'memory' ? {
                path: storage?.path ?? null, locked: storage?.locked ?? false, committed: storage?.committed ?? false,
                needed: target.needed, available: target.available,
            } : null,
            encoderConfig: sink?.encoderConfig ? JSON.parse(JSON.stringify(sink.encoderConfig)) as VideoEncoderConfig : null,
            frames: { planned: this.frames, accepted: this.added, encoded: sink?.packets ?? 0, keyFrames: sink?.keyPackets ?? 0 },
            bytes: sink?.bytes ?? 0,
            achievedBitrate: sink && !png && encodedSeconds ? Math.round(sink.bytes * 8 / encodedSeconds) : null,
            timing: { addWaitMs: round(this.addWaitMs), finalizeMs: round(this.finalizeMs), quantization: 'T(j) = round(j × 10⁶ / fps) µs; D(j) = T(j + 1) − T(j)' },
            states: [...this.lifecycle.history],
            cancelled: this.cancelled,
            failure: this.failure && !(this.cancelled && isAbort(this.failure)) ? describe(this.failure) : null,
            cleanupErrors: this.cleanupErrors.map(e => `${e.name}: ${describe(e.error)}`),
        };
    }

    /** The report a committed film's manifest keeps: the session's own, with what its runner adds (§4.8). */
    private manifestReport(): Record<string, unknown> {
        let extra: Record<string, unknown> = {};
        try { extra = this.annotate?.() ?? {}; } catch { /* the manifest keeps the session's own report */ }
        return { ...this.report(), ...extra };
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
                try { await this.sink.cancel(); } catch (cancelError) { early.push({ name: this.sink.codec === 'png' ? 'PNG writer' : 'video encoder', error: cancelError }); }
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
            const storage = this.target?.storage ?? null;
            if (finalize) {
                this.lifecycle.to('finalizing');
                const t = now();
                try {
                    file = this.sink ? await this.waitOn(this.sink.finish()) : null;
                    const kept = goal === 'finish' ? 'completed' : 'partial';
                    // The payload is committed (the muxer closed its file, or every frame was written): mark the
                    // film so in its manifest, and take the file from disk (§4.5).
                    if (storage) file = await storage.commit(kept, { frames: this.added, bytes: storage.gate?.bytes ?? this.sink?.bytes ?? 0, report: this.manifestReport() });
                    status = kept;
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
            return filmResult(status, file, this.output?.name ?? null, this.sink?.codec ?? null, this.added, this.report(), this.target?.kind ?? null, this.exportId, storage);
        })();
        return this.terminal;
    }
}

/**
 * A session's result, made outside the session so that what it holds is only what it gives: a result kept by
 * the app must not keep the session, and with it the sink and the muxer (an in-memory film's whole buffer).
 */
function filmResult(
    status: FilmResult['status'], file: Blob | null, name: string | null, codec: FilmCodec | null, frames: number, report: FilmSessionReport,
    target: FilmTargetKind | null, exportId: string, storage: FilmStorage | null,
): FilmResult {
    const directory = storage?.frames ?? null;
    return {
        status, file, name, codec, frames, report, target,
        exportId: storage?.kind === 'opfs' ? exportId : null,
        sequence: directory ? { directory, names: Array.from({ length: frames }, (_, j) => frameFileName(j)) } : null,
        dispose: async () => { await storage?.dispose(); },
    };
}

/** Open a session for an app that draws and captures frames itself (§4.9, §6). */
export function createFilmSession(options: FilmSessionOptions): Promise<FilmSession> {
    return FilmSession.open(options);
}
