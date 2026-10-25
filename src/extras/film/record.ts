// D2.4 (offline-films-browser-design.md §2, §4.1, §4.6–§4.9, §6): recordFilm, the managed runner, and snapshot.
// recordFilm builds an engine of its own on the app's device (never the app's canvas), asks the app for a
// workload, calibrates, resets, warms up, and films frame j at s·(j + 1) steps from the frame number, never
// the clock: I_j = R(S_((j+1)·s), C(j / r), Q). Each frame is drawn and captured in one task, handed to the
// session with backpressure, and the runner yields a real task now and then so the page stays responsive.
// Every resource goes into the session's ledger as it is made, so a film that fails anywhere unwinds whole.

import type { NullGraph } from '../../core';
import type { CameraView } from './camera';
import {
    BATCH_FRAMES, CONGESTED_BATCH_MS, calibrateFilm, checkOperation, readGuardLimits,
    type FilmCalibration, type FilmGuardLimits,
} from './guard';
import type { MediabunnyLike } from './mediabunny';
import { readOutput, type FilmOutputOptions } from './output';
import { encodePNG } from './png';
import { frameStamp, planFilm, readFilm, type FilmPlan } from './plan';
import { FrameReadback } from './readback';
import { CleanupLedger, FilmSession, type FilmResult, type FilmSessionOptions, type FilmSessionReport, type FilmState } from './session';
import { sha256 } from './sinks';
import { clause, fail, positive, sentence, watchFreezes, whole, yieldTask } from './util';
import type { CreateFilmWorkload, FilmFrame, FilmWorkload } from './workload';

type CoreModule = typeof import('../../core');

/** The package's own core, imported at run time (modules.d.ts), so a film's engine is the core the app uses. */
async function loadCore(): Promise<CoreModule> {
    return (await import('null-graph')) as unknown as CoreModule;
}

export type FilmSourceChoice = 'auto' | 'canvas' | 'readback';
/** `'none'`: frames are drawn and measured, never captured (no video and no hashes). */
export type FilmSource = 'canvas' | 'readback' | 'none';

export interface RecordFilmOptions {
    /** The app's device. The film's engine adopts it and never destroys it. Pause the preview first (§4.6). */
    readonly device: GPUDevice;
    readonly createWorkload: CreateFilmWorkload;
    readonly film: FilmSessionOptions['film'];
    /** A camera path (`cameraPath(keyframes)`) or any `(seconds) => view`; omitted, the workload's own view holds. */
    readonly camera?: ((seconds: number) => CameraView) | null;
    /** Q1: `'auto'` films from an offscreen canvas where that works, and reads frames back otherwise. */
    readonly source?: FilmSourceChoice;
    /** The video file; `false` makes no file (a run that only hashes or measures). */
    readonly output?: FilmOutputOptions | false;
    /** SHA-256 of every frame's exact bytes, into the report (P1). Frames are then read back. */
    readonly hashes?: boolean;
    /** Aborting cancels: everything is discarded, and recordFilm rejects with an AbortError. */
    readonly signal?: AbortSignal;
    /** `'keep-if-finalizable'` keeps a finalized partial film after a failure that is not a cancel. */
    readonly partial?: 'discard' | 'keep-if-finalizable';
    readonly onProgress?: (progress: FilmProgress) => void;
    /** Called with the report on success, cancellation and failure; what it throws never changes the outcome. */
    readonly onReport?: (report: FilmReport) => void;
    readonly mediabunny?: MediabunnyLike;
    /** The guard's limits, which may only be lowered (§4.7). */
    readonly guard?: FilmGuardLimits;
    /** Milliseconds of work between task yields (§4.9). Default 8. */
    readonly yieldEveryMs?: number;
    /** Where a function-valued camera or workload came from: JSON cannot carry the functions themselves (§4.8). */
    readonly provenance?: Record<string, unknown>;
}

export interface FilmProgress {
    readonly phase: FilmState;
    /** Frames made so far, out of `frames`. */
    readonly frame: number;
    readonly frames: number;
    /** Since recording began; 0 before. */
    readonly elapsedMs: number;
    /** Null until a frame has been made. */
    readonly msPerFrame: number | null;
    readonly etaMs: number | null;
    /** Encoded so far. */
    readonly bytes: number;
    /** The page is hidden, and the browser may throttle the film. Film time never follows the wall clock. */
    readonly background: boolean;
}

export interface FilmReport extends FilmSessionReport {
    /** Null when the film ended before a source was chosen. */
    readonly source: FilmSource | null;
    readonly sourceReason: string | null;
    readonly browser: string | null;
    readonly adapter: { readonly vendor: string; readonly architecture: string; readonly description: string } | null;
    readonly plan: Pick<FilmPlan, 'frames' | 'stepsPerFrame' | 'warmupSteps' | 'dt'> | null;
    readonly workload: Record<string, unknown> | null;
    readonly provenance: Record<string, unknown> | null;
    readonly guard: {
        readonly calibration: FilmCalibration | null;
        readonly fencing: 'every 8 frames' | 'every operation';
        /** Where a congested 8-frame batch switched fencing to every operation: the first frame so fenced, or the warm-up. */
        readonly switchedAt: number | 'warm-up' | null;
        /** The slowest step and draw, measured only while every operation was fenced (§4.7). */
        readonly worstStepMs: number | null;
        readonly worstDrawMs: number | null;
        /** Operations fenced while the page froze: their time was the freeze's, so they were not judged (§4.9). */
        readonly unjudged: number;
    };
    readonly timings: {
        readonly calibrationMs: number;
        readonly warmupMs: number;
        readonly recordingMs: number;
        readonly msPerFrame: number | null;
    };
    /** One per frame, when `hashes` was set. */
    readonly hashes: readonly string[] | null;
    /** WebGPU errors on the device while the film ran: any user of the device can raise one (§4.6). */
    readonly gpuErrors: readonly string[];
    readonly deviceLost: string | null;
    /** What onProgress threw: reported, never the film's outcome. */
    readonly observerErrors: readonly string[];
}

const now = () => performance.now();
const abortError = () => new DOMException('Export cancelled', 'AbortError');
const isAbort = (error: unknown) => (error as { name?: unknown } | null)?.name === 'AbortError';
const describe = (error: unknown) => (error instanceof Error ? error.message : String(error));
const round = (ms: number) => Math.round(ms * 10) / 10;

/** The kind of a WebGPU error, by its class (built-in class names survive minification). */
function gpuErrorKind(error: unknown): string {
    const name = (error as { constructor?: { name?: string } } | null)?.constructor?.name;
    return name === 'GPUOutOfMemoryError' ? 'out-of-memory' : name === 'GPUInternalError' ? 'internal' : 'validation';
}

/** One film or snapshot at a time on a device (§4.6): error scopes and listeners are device-wide. */
const turns = new WeakMap<GPUDevice, Promise<void>>();

/** A device's loss, as the films on it see it. */
interface LossWatch {
    info: GPUDeviceLostInfo | null;
    /** Called once, when the device is lost: each film adds one while it runs and removes it at teardown. */
    readonly listeners: Set<() => void>;
}

/**
 * Each device's loss, watched once however many films it makes. A reaction to `device.lost` cannot be
 * removed, so a reaction per film would keep that film's scope, its engine included, alive for as long as
 * the app's device lives (G1 caught this). This one holds a record of its own and nothing else.
 */
const losses = new WeakMap<GPUDevice, LossWatch>();

function watchLoss(device: GPUDevice): LossWatch {
    let watch = losses.get(device);
    if (!watch) {
        const record: LossWatch = { info: null, listeners: new Set() };
        void device.lost.then(info => {
            record.info = info;
            for (const listener of [...record.listeners]) listener();
        });
        losses.set(device, record);
        watch = record;
    }
    return watch;
}

/** A lost device's reason and message, as the report and §8's words carry them. */
const lostWords = (info: GPUDeviceLostInfo) => (info.message ? `${info.reason}: ${clause(info.message)}` : String(info.reason));

/** How long a failed film waits for `device.lost` to settle before deciding the failure was not a loss. */
const LOSS_SETTLE_MS = 100;

/**
 * How long a film whose device was lost still waits on its encoder. A working encoder takes an add in
 * milliseconds (spike S1: 7–20 ms on the hardware path); one lost with Chrome's GPU process never answers.
 */
export const LOST_ENCODER_GRACE_MS = 2000;

/** Wait for the device's previous film to finish tearing down; abortable. Resolves to the release. */
async function takeTurn(device: GPUDevice, signal: AbortSignal): Promise<() => void> {
    const previous = turns.get(device) ?? Promise.resolve();
    let release!: () => void;
    const mine = new Promise<void>(done => { release = done; });
    const turn = previous.then(() => mine);
    turns.set(device, turn);
    // A cancelled waiter releases its own link, not its predecessor's ownership. Keep the tail
    // registered until the entire chain settles, otherwise a new caller can bypass a running film.
    void turn.then(() => {
        if (turns.get(device) === turn) turns.delete(device);
    });
    const leave = () => {
        release();
    };
    let onAbort = () => {};
    try {
        await Promise.race([previous, new Promise<never>((_, reject) => {
            onAbort = () => reject(abortError());
            signal.addEventListener('abort', onAbort, { once: true });
            if (signal.aborted) onAbort();
        })]);
    } catch (error) {
        leave();
        throw error;
    } finally {
        signal.removeEventListener('abort', onAbort);
    }
    return leave;
}

/**
 * The workload, built with a ledger of its own: if construction throws, what it registered is released
 * (last-first); once it succeeds, `dispose()` owns everything and the registrations are dropped (§4.6).
 * A workload that is not one (no step, draw or dispose, or no dt) is refused and released.
 */
async function buildWorkload(create: CreateFilmWorkload, engine: NullGraph, width: number, height: number, signal: AbortSignal): Promise<FilmWorkload> {
    const partial = new CleanupLedger();
    let workload: FilmWorkload;
    try {
        workload = await create({ engine, width, height, signal, defer: release => partial.defer('workload construction', release) });
    } catch (error) {
        await partial.unwind();
        throw error;
    }
    const shaped = !!workload && typeof workload.step === 'function' && typeof workload.draw === 'function' && typeof workload.dispose === 'function';
    try {
        if (!shaped) fail('createWorkload must resolve to a workload with step(), draw(frame) and dispose() (§4.6).');
        positive(workload.dt, 'the workload\'s dt');
    } catch (error) {
        if (shaped) await Promise.resolve().then(() => workload.dispose()).catch(() => {});
        else await partial.unwind();
        throw error;
    }
    return workload;
}

/**
 * Why WebGPU frames cannot be taken from an unattached OffscreenCanvas here, or null when they can: a
 * context configured as the engine configures it, cleared, and a VideoFrame taken from it, inside a
 * validation scope of its own. A reason selects readback before frame 0 (§4.9); a film never changes
 * its source midway.
 */
async function canvasFails(device: GPUDevice, width: number, height: number): Promise<string | null> {
    if (typeof OffscreenCanvas !== 'function' || typeof VideoFrame !== 'function') return 'there is no OffscreenCanvas or VideoFrame here';
    let why: string | null = null;
    device.pushErrorScope('validation');
    try {
        const canvas = new OffscreenCanvas(width, height);
        const context = canvas.getContext('webgpu');
        if (!context) {
            why = 'an OffscreenCanvas here has no WebGPU context';
        } else {
            context.configure({ device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'premultiplied' });
            try {
                const encoder = device.createCommandEncoder({ label: 'film canvas check' });
                encoder.beginRenderPass({
                    colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
                }).end();
                device.queue.submit([encoder.finish()]);
                new VideoFrame(canvas, { timestamp: 0 }).close();
            } finally {
                context.unconfigure();
            }
        }
    } catch (error) {
        why = `a VideoFrame could not be taken from an OffscreenCanvas here (${clause(describe(error))})`;
    }
    const error = await device.popErrorScope();
    return why ?? (error ? `the canvas check raised a GPU validation error (${clause(error.message)})` : null);
}

/** `exact`: why the film needs exact bytes (hashes, a PNG sequence), or null when a video alone is made. */
async function chooseSource(
    device: GPUDevice, wanted: FilmSourceChoice, exact: string | null, video: boolean, width: number, height: number,
): Promise<{ readonly source: FilmSource; readonly reason: string }> {
    if (!video && !exact) return { source: 'none', reason: 'no video and no hashes: frames are drawn and measured, never captured' };
    if (exact) return { source: 'readback', reason: exact };
    if (wanted === 'readback') return { source: 'readback', reason: 'asked for' };
    const why = await canvasFails(device, width, height);
    if (!why) return { source: 'canvas', reason: wanted === 'canvas' ? 'asked for' : 'an offscreen canvas works here (Q1)' };
    if (wanted === 'canvas') fail(`source "canvas" cannot be used here: ${why}.`);
    return { source: 'readback', reason: `canvas frames are unavailable (${why}); frames are read back` };
}

/**
 * Film a workload (§6). Resolves to the film (`status: 'completed'`, or `'partial'` when `partial` allowed
 * keeping one after a failure, with `report.failure` saying what ended it). Rejects with an AbortError on
 * cancel, and otherwise with the error that ended the film; `onReport` gets the report either way.
 * Refusals before the session opens (options, mediabunny, no encoder) reject without a report.
 */
export async function recordFilm(o: RecordFilmOptions): Promise<FilmResult<FilmReport>> {
    // Probing: whatever needs no GPU is checked before anything is made.
    if (!o || typeof o !== 'object') fail('recordFilm needs its options.');
    const { device } = o;
    if (!device || typeof device.createBuffer !== 'function') fail('recordFilm: device must be the app\'s GPUDevice; the film\'s engine adopts it.');
    if (typeof o.createWorkload !== 'function') fail('recordFilm: createWorkload must be a function that resolves to the workload (§4.6).');
    const film = readFilm(o.film);
    const output = o.output === false ? null : readOutput(o.output ?? {});
    // A PNG sequence is frames, each the read-back bytes exactly; a video is what the encoder makes of them.
    const png = output?.container === 'png';
    const video = !!output && !png;
    const camera = o.camera ?? null;
    if (camera !== null && typeof camera !== 'function') fail('camera must be (seconds) => view, such as cameraPath(keyframes), or omitted.');
    const keepPartial = o.partial ?? 'discard';
    if (keepPartial !== 'discard' && keepPartial !== 'keep-if-finalizable') fail(`partial must be "discard" or "keep-if-finalizable"; got ${JSON.stringify(keepPartial)}.`);
    const limits = readGuardLimits(o.guard);
    const hashes = o.hashes === true;
    const wanted = o.source ?? 'auto';
    if (wanted !== 'auto' && wanted !== 'canvas' && wanted !== 'readback') fail(`source must be "auto", "canvas" or "readback"; got ${JSON.stringify(wanted)}.`);
    if (hashes && wanted === 'canvas') fail('hashes need exact bytes, and canvas frames are not exact (§4.3): use source "readback" or "auto".');
    if (png && wanted === 'canvas') fail('a PNG sequence needs exact bytes, and canvas frames are not exact (§4.3): use source "readback" or "auto".');
    const exact = hashes ? 'hashes need exact bytes' : png ? 'PNG frames need exact bytes' : null;
    const yieldEvery = o.yieldEveryMs ?? 8;
    if (!(yieldEvery >= 0)) fail(`yieldEveryMs must be a number of milliseconds, 0 or more; got ${JSON.stringify(o.yieldEveryMs)}.`);
    const signal = o.signal ?? new AbortController().signal;
    const checkAbort = () => { if (signal.aborted) throw abortError(); };
    checkAbort();

    const leave = await takeTurn(device, signal);
    let session: FilmSession;
    try {
        checkAbort();
        // The manifest of a film committed to OPFS keeps the runner's side of the report too (§4.8).
        session = await FilmSession.open({ film: o.film, output: output ? o.output ?? {} : false, mediabunny: o.mediabunny, annotate: () => annotation() });
    } catch (error) {
        leave();
        throw error;
    }
    // Registered before anything on the device, so released after all of it.
    session.ledger.defer('device turn', leave);
    // §4.7: a lost GPU process can take the hardware encoder with it, and a wait on it then never returns
    // (G9-crash). Once the device is lost, a film still waiting LOST_ENCODER_GRACE_MS later gives the encoder up.
    const loss = watchLoss(device);
    let giveUp: ReturnType<typeof setTimeout> | null = null;
    const onLoss = () => {
        giveUp = setTimeout(() => { void session.abandon(new Error('film: GPU device lost, and the video encoder stopped answering')); }, LOST_ENCODER_GRACE_MS);
    };
    loss.listeners.add(onLoss);
    session.ledger.defer('device-loss watch', () => {
        loss.listeners.delete(onLoss);
        if (giveUp) clearTimeout(giveUp);
    });

    let source: FilmSource | null = null;
    let sourceReason: string | null = null;
    const gpuErrors: { readonly kind: string; readonly message: string }[] = [];
    const observerErrors: string[] = [];
    const hashList: string[] = [];
    let lost: string | null = null;
    let making = -1;
    let plan: FilmPlan | null = null;
    let workloadInfo: Record<string, unknown> | null = null;
    let calibration: FilmCalibration | null = null;
    let perOperation = false;
    let switchedAt: number | 'warm-up' | null = null;
    let worstStepMs: number | null = null;
    let worstDrawMs: number | null = null;
    let unjudged = 0;
    const timings = { calibrationMs: 0, warmupMs: 0, recordingMs: 0, msPerFrame: null as number | null };
    let recordingStart = 0;
    let result: FilmResult | null = null;
    let primary: unknown = null;
    let scopes = 0;

    const progress = (phase: FilmState, frame: number) => {
        if (!o.onProgress) return;
        const elapsedMs = recordingStart ? now() - recordingStart : 0;
        const msPerFrame = frame ? elapsedMs / frame : null;
        try {
            o.onProgress({
                phase, frame, frames: session.frames, elapsedMs, msPerFrame,
                etaMs: msPerFrame === null ? null : msPerFrame * (session.frames - frame), bytes: session.bytes,
                background: typeof document !== 'undefined' && document.visibilityState === 'hidden',
            });
        } catch (error) {
            observerErrors.push(describe(error));
        }
    };
    /** The runner's side of the report (§4.8): the final report's, and what a committed film's manifest keeps. */
    const annotation = (): Omit<FilmReport, keyof FilmSessionReport> => {
        const info = (device as { adapterInfo?: GPUAdapterInfo }).adapterInfo;
        return {
            source, sourceReason,
            browser: typeof navigator !== 'undefined' ? navigator.userAgent ?? null : null,
            adapter: info ? { vendor: info.vendor, architecture: info.architecture, description: info.description } : null,
            plan: plan && { frames: plan.frames, stepsPerFrame: plan.stepsPerFrame, warmupSteps: plan.warmupSteps, dt: plan.dt },
            workload: workloadInfo,
            provenance: o.provenance ?? null,
            guard: {
                calibration, fencing: perOperation ? 'every operation' : 'every 8 frames', switchedAt,
                worstStepMs: worstStepMs === null ? null : round(worstStepMs), worstDrawMs: worstDrawMs === null ? null : round(worstDrawMs),
                unjudged,
            },
            timings: { ...timings },
            hashes: hashes ? [...hashList] : null,
            gpuErrors: gpuErrors.map(e => `${e.kind}: ${e.message}`),
            deviceLost: lost,
            observerErrors: [...observerErrors],
        };
    };
    // §4.7: an uncaptured error or a lost device stops the film before its next submission. A loss is the
    // film's from the moment the film starts watching until it has torn down.
    const noteLoss = () => { if (!lost && loss.info) lost = lostWords(loss.info); };
    const checkHealth = () => {
        noteLoss();
        if (lost) fail('GPU device lost'); // replaced below by the full message, once the partial's fate is known
        if (gpuErrors.length) fail(`GPU ${gpuErrors[0].kind} error while building or filming the scene: ${sentence(gpuErrors[0].message)}`);
    };

    try {
        // Initializing.
        const core = await loadCore();
        checkAbort();
        const onError = (event: Event) => {
            const error = (event as GPUUncapturedErrorEvent).error;
            gpuErrors.push({ kind: gpuErrorKind(error), message: error.message });
        };
        device.addEventListener('uncapturederror', onError);
        session.ledger.defer('uncaptured-error listener', () => device.removeEventListener('uncapturederror', onError));
        // §4.9: a hidden page films at its usual pace, but a frozen one runs no task, so an operation or a batch
        // measured across a freeze timed the freeze. It is not judged, and the film goes on (G9-background).
        const freezes = watchFreezes();
        session.ledger.defer('page lifecycle listeners', () => freezes.stop());
        checkHealth();
        ({ source, reason: sourceReason } = await chooseSource(device, wanted, exact, video, film.width, film.height));
        checkAbort();

        // Construction and calibration run inside error scopes (§4.6), popped below or in the finally.
        device.pushErrorScope('out-of-memory');
        scopes = 1;
        device.pushErrorScope('validation');
        scopes = 2;
        const engine = new core.NullGraph();
        session.ledger.defer('engine', () => engine.destroy());
        const canvas = source === 'canvas' ? new OffscreenCanvas(film.width, film.height) : null;
        const init = await engine.init(canvas, canvas ? { device } : { device, width: film.width, height: film.height });
        if (!init.success) fail(`the film's engine did not start: ${sentence(init.error ?? 'unknown error')}`);
        checkAbort();
        let workload: FilmWorkload | null = await buildWorkload(o.createWorkload, engine, film.width, film.height, signal);
        session.ledger.defer('workload', async () => {
            const owned = workload;
            workload = null;
            await owned?.dispose();
        });
        workloadInfo = workload.info ?? null;
        plan = planFilm(film, { dt: workload.dt });
        const readback = source === 'readback' ? new FrameReadback(device, engine.screenTexture as GPUTexture, film.width, film.height) : null;
        if (readback) session.ledger.defer('readback', () => readback.destroy());
        const frameAt = (j: number): FilmFrame => {
            const stamp = frameStamp(j, film.fps);
            return { ...stamp, view: camera ? camera(stamp.seconds) : null };
        };

        // Calibrating.
        session.advance('calibrating');
        progress('calibrating', 0);
        checkAbort();
        let t = now();
        calibration = await calibrateFilm(engine, workload, {
            width: film.width, height: film.height, stepsPerFrame: plan.stepsPerFrame, frame: frameAt(0), limits, signal,
        });
        timings.calibrationMs = round(now() - t);
        scopes--;
        const validation = await device.popErrorScope();
        scopes--;
        const outOfMemory = await device.popErrorScope();
        if (outOfMemory) gpuErrors.push({ kind: 'out-of-memory', message: outOfMemory.message });
        if (validation) gpuErrors.push({ kind: 'validation', message: validation.message });
        checkHealth();
        perOperation = calibration.fenceEvery === 1;
        checkAbort();

        // Resetting: calibration stepped and drew; back to the exact initial state, then W warm-up steps.
        session.advance('resetting');
        progress('resetting', 0);
        checkAbort();
        if (workload.reset) {
            await workload.reset();
        } else {
            const old = workload;
            workload = null;
            await old.dispose();
            workload = await buildWorkload(o.createWorkload, engine, film.width, film.height, signal);
        }
        checkAbort();
        const active: FilmWorkload = workload;
        const maxCalibratedStep = Math.max(0, ...calibration.stepMs);
        const maxCalibratedDraw = Math.max(0, ...calibration.drawMs);
        let recheck: 'step' | 'draw' | null = null;
        let lastFence = now();
        let fencedFreezes = freezes.count;
        let lastYield = now();
        /**
         * One operation fenced and timed, judged against the stop unless the page froze while it was measured (§4.9)
         * or it is a single outlier whose calibrated operation was under a quarter of the stop (Q10: measured once more).
         */
        const judge = (kind: 'step' | 'draw', ms: number, at: number | 'warm-up', before: number) => {
            if (freezes.count !== before) {
                unjudged++;
                return;
            }
            if (kind === 'step') worstStepMs = Math.max(worstStepMs ?? 0, ms);
            else worstDrawMs = Math.max(worstDrawMs ?? 0, ms);
            if (ms <= limits.stopMs) {
                if (recheck === kind) recheck = null;
                return;
            }
            const calibrated = kind === 'step' ? maxCalibratedStep : maxCalibratedDraw;
            if (recheck === null && calibrated < limits.stopMs / 4) {
                recheck = kind;
                return;
            }
            checkOperation(ms, at, limits.stopMs);
        };
        const step = async (at: number | 'warm-up') => {
            checkAbort();
            checkHealth();
            if (!perOperation) {
                await active.step();
                return;
            }
            const before = freezes.count;
            const s = now();
            await active.step();
            await device.queue.onSubmittedWorkDone();
            judge('step', now() - s, at, before);
        };
        // §4.7: an 8-frame batch slower than CONGESTED_BATCH_MS fences every operation from then on.
        const batchFence = async (next: number | 'warm-up') => {
            await device.queue.onSubmittedWorkDone();
            const ms = now() - lastFence;
            lastFence = now();
            const frozen = freezes.count !== fencedFreezes;
            fencedFreezes = freezes.count;
            if (ms > CONGESTED_BATCH_MS && !perOperation && !frozen) {
                perOperation = true;
                switchedAt = next;
            }
        };
        const maybeYield = async () => {
            if (now() - lastYield < yieldEvery) return;
            await yieldTask();
            lastYield = now();
            checkAbort();
        };
        t = now();
        for (let i = 0; i < plan.warmupSteps; i++) {
            await step('warm-up');
            if (!perOperation && (i + 1) % BATCH_FRAMES === 0) await batchFence('warm-up');
            await maybeYield();
        }
        await device.queue.onSubmittedWorkDone();
        timings.warmupMs = round(now() - t);
        checkAbort();

        // Recording. Readback keeps one frame in flight: frame j − 1 is handed over after frame j is grabbed.
        session.advance('recording');
        let handed = 0;
        const handOver = async (reader: FrameReadback) => {
            const stamp = frameStamp(handed, film.fps);
            if (video && !hashes) {
                const { videoFrame } = await reader.takeVideoFrame(stamp.timestampUs, stamp.durationUs);
                handed++;
                await session.add(videoFrame);
                return;
            }
            const { frame, bytes } = await reader.take();
            handed++;
            if (hashes) hashList[frame] = await sha256(bytes);
            if (!output) {
                session.noteFrame();
                return;
            }
            // A video ignores alpha ('RGBX', as ffmpeg's yuv420p does); a PNG keeps every byte ('RGBA').
            await session.add(new VideoFrame(bytes, {
                format: png ? 'RGBA' : 'RGBX', codedWidth: film.width, codedHeight: film.height, timestamp: stamp.timestampUs, duration: stamp.durationUs,
            }));
        };
        recordingStart = now();
        lastFence = recordingStart;
        fencedFreezes = freezes.count;
        for (let j = 0; j < plan.frames; j++) {
            making = j;
            for (let s = 0; s < plan.stepsPerFrame; s++) await step(j);
            const frame = frameAt(j);
            await active.prepare?.(frame);
            checkAbort();
            checkHealth();
            // Render and capture in one task, with nothing between them (§4.3).
            const before = freezes.count;
            const drawStart = now();
            engine.render(active.draw(frame));
            const captured = canvas ? new VideoFrame(canvas, { timestamp: frame.timestampUs, duration: frame.durationUs }) : null;
            try {
                readback?.grab(j);
                if (perOperation) {
                    await device.queue.onSubmittedWorkDone();
                    judge('draw', now() - drawStart, j, before);
                }
            } catch (error) {
                captured?.close();
                throw error;
            }
            if (captured) await session.add(captured);
            else if (readback) { if (readback.pending > 1) await handOver(readback); }
            else session.noteFrame();
            if (!perOperation && (j + 1) % BATCH_FRAMES === 0) await batchFence(j + 1);
            progress('recording', j + 1);
            await maybeYield();
        }
        while (readback?.pending) await handOver(readback);
        await device.queue.onSubmittedWorkDone();
        checkHealth();
        timings.recordingMs = round(now() - recordingStart);
        timings.msPerFrame = round(timings.recordingMs / plan.frames);

        // Finalizing, then disposing: the session tears everything down before it returns. A cancel before
        // finish() is asked for wins; one after joins it (§4.9: the first terminal request wins).
        progress('finalizing', plan.frames);
        checkAbort();
        result = await session.finish();
    } catch (error) {
        primary = error;
    } finally {
        // Leave the device's error-scope stack as it was found, whatever threw; what the scopes held is reported.
        for (; scopes > 0; scopes--) {
            const kind = scopes === 2 ? 'validation' : 'out-of-memory';
            const error = await device.popErrorScope().catch(() => null);
            if (error) gpuErrors.push({ kind, message: error.message });
        }
    }
    if (!result) {
        if (recordingStart && !timings.recordingMs) timings.recordingMs = round(now() - recordingStart);
        const cancelled = signal.aborted && isAbort(primary);
        // A lost device can fail a platform operation (mapAsync rejects, say) before device.lost settles (G9 saw
        // it on readback): give the loss a moment, so the film reports it rather than its symptom. Polled, not
        // raced against device.lost, which would leave a reaction on it for every failed film; timed, since a
        // hidden page's timers fire at most once a second; and counted as well, so a clock that stands still
        // (a test's) cannot hold it for ever.
        noteLoss();
        const symptom = typeof DOMException === 'function' && primary instanceof DOMException;
        for (let i = 0, until = now() + LOSS_SETTLE_MS; symptom && !cancelled && !lost && now() < until && i < LOSS_SETTLE_MS / 10; i++) {
            await new Promise(resolve => setTimeout(resolve, 10));
            noteLoss();
        }
        // A lost device explains whatever failed after it (mapAsync rejects, say); an AbortError the app
        // did not ask for is a failure, not a cancel.
        const lostCause = !cancelled && lost ? primary : null;
        if (lostCause) primary = new Error('film: GPU device lost');
        else if (!cancelled && isAbort(primary)) primary = new Error(`film: ${sentence(describe(primary))}`, { cause: primary });
        result = await session.abort(primary, { keepPartial: keepPartial === 'keep-if-finalizable' && !cancelled, cancelled });
        if (lostCause) {
            primary = new Error(`film: GPU device lost ${making < 0 ? 'before frame 0' : `at frame ${making}`} (${lost}); ` +
                `partial file ${result ? 'kept' : 'discarded'}.`, { cause: lostCause });
        }
    }
    const sessionReport = session.report();
    const report: FilmReport = {
        ...sessionReport,
        failure: primary && !isAbort(primary) ? describe(primary) : sessionReport.failure,
        ...annotation(),
    };
    try { o.onReport?.(report); } catch { /* isolated (§4.8): the report's observer never changes the film's outcome */ }
    if (result) return { ...result, report };
    settleStacks(primary);
    throw primary;
}

/**
 * Read an error's stack, and its causes', before it leaves the runner. Until a stack is first read, V8 keeps
 * each frame's function, and the scope it closes over (the film's engine among them), alive for as long as
 * the app keeps the error (G1 and G5 caught this after a guard stop).
 */
function settleStacks(error: unknown): void {
    for (let e = error, depth = 0; e && typeof e === 'object' && depth < 8; e = (e as { cause?: unknown }).cause, depth++) {
        try { void (e as { stack?: unknown }).stack; } catch { /* a stack getter that throws keeps nothing */ }
    }
}

export interface SnapshotOptions {
    readonly device: GPUDevice;
    readonly createWorkload: CreateFilmWorkload;
    readonly width: number;
    readonly height: number;
    /** Film time the frame shows; default 0. The workload is not stepped. */
    readonly seconds?: number;
    readonly view?: CameraView | null;
    readonly signal?: AbortSignal;
}

export interface FilmSnapshot {
    /** The frame as an exact RGBA8 PNG (§3). */
    readonly png: Uint8Array;
    /** The frame's bytes, tightly packed RGBA8, top row first. */
    readonly rgba: Uint8Array;
    readonly width: number;
    readonly height: number;
    readonly limitations: readonly string[];
}

/**
 * One frame, drawn on an engine of its own (never the app's canvas, never resized) and read back exactly
 * (§6). The workload is built, drawn once at `seconds` with `view`, and disposed: nothing is stepped, so a
 * simulation shows its initial state unless `createWorkload` builds a later one.
 */
export async function snapshot(o: SnapshotOptions): Promise<FilmSnapshot> {
    if (!o || typeof o !== 'object') fail('snapshot needs its options.');
    const { device } = o;
    if (!device || typeof device.createBuffer !== 'function') fail('snapshot: device must be the app\'s GPUDevice.');
    if (typeof o.createWorkload !== 'function') fail('snapshot: createWorkload must be a function that resolves to the workload (§4.6).');
    // A PNG has no yuv420p rule: any whole size.
    const width = whole(o.width, 'snapshot width');
    const height = whole(o.height, 'snapshot height');
    const seconds = o.seconds ?? 0;
    if (!(Number.isFinite(seconds) && seconds >= 0)) fail(`snapshot seconds must be a time, 0 or more; got ${JSON.stringify(o.seconds)}.`);
    const signal = o.signal ?? new AbortController().signal;
    if (signal.aborted) throw abortError();
    const ledger = new CleanupLedger();
    ledger.defer('device turn', await takeTurn(device, signal));
    let primary: unknown = null;
    let shot: FilmSnapshot | null = null;
    let scopes = 0;
    try {
        const core = await loadCore();
        if (signal.aborted) throw abortError();
        device.pushErrorScope('out-of-memory');
        scopes = 1;
        device.pushErrorScope('validation');
        scopes = 2;
        const engine = new core.NullGraph();
        ledger.defer('engine', () => engine.destroy());
        const init = await engine.init(null, { device, width, height });
        if (!init.success) fail(`the snapshot's engine did not start: ${sentence(init.error ?? 'unknown error')}`);
        if (signal.aborted) throw abortError();
        const workload = await buildWorkload(o.createWorkload, engine, width, height, signal);
        ledger.defer('workload', () => workload.dispose());
        if (signal.aborted) throw abortError();
        const readback = new FrameReadback(device, engine.screenTexture as GPUTexture, width, height);
        ledger.defer('readback', () => readback.destroy());
        const frame: FilmFrame = { index: 0, seconds, timestampUs: Math.round(seconds * 1e6), durationUs: 0, view: o.view ?? null };
        await workload.prepare?.(frame);
        if (signal.aborted) throw abortError();
        engine.render(workload.draw(frame));
        readback.grab(0);
        const { bytes } = await readback.take(new Uint8Array(width * height * 4));
        scopes--;
        const validation = await device.popErrorScope();
        scopes--;
        const outOfMemory = await device.popErrorScope();
        const error = outOfMemory ?? validation;
        if (error) fail(`GPU ${outOfMemory ? 'out-of-memory' : 'validation'} error while building or drawing the scene: ${sentence(error.message)}`);
        const png = await encodePNG(bytes, width, height);
        if (signal.aborted) throw abortError();
        shot = {
            png, rgba: bytes, width, height,
            limitations: ['The draw was not measured against the guard: its first submission compiles pipelines, an unverified first-submission risk (§4.7).'],
        };
    } catch (error) {
        primary = error;
    } finally {
        for (; scopes > 0; scopes--) await device.popErrorScope().catch(() => null);
    }
    const cleanup = await ledger.unwind();
    if (primary) {
        settleStacks(primary);
        throw primary;
    }
    if (cleanup.length) fail(`the snapshot was taken, but releasing ${cleanup.map(c => c.name).join(', ')} failed: ${sentence(describe(cleanup[0].error))}`);
    return shot as FilmSnapshot;
}
