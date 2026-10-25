// D2.4 (offline-films-browser-design.md §4.7): the guard. Best-effort risk reduction, not reset prevention:
// an operation measured above 1,000 ms is refused before the film starts, and the film stops when a measured
// operation passes 1,500 ms. Times are queue-completion wall milliseconds (submit to onSubmittedWorkDone), not
// GPU timestamps, and a timer cannot interrupt work already submitted. The thresholds are the Node tool's
// (tools/film/guard.mjs): a conservative policy, not a universal browser or operating-system boundary.

import type { NullGraph } from '../../core';
import type { FilmFrame, FilmWorkload } from './workload';
import { fail, watchFreezes } from './util';

/** Refuse a film when one calibrated operation takes longer. */
export const FILM_LIMIT_MS = 1000;
/** Stop a film when one measured operation takes longer. */
export const FILM_STOP_MS = 1500;
/** Above this many ms a frame, every step and draw is fenced and timed on its own (§4.7). */
export const PER_OPERATION_MS = 150;
/** Otherwise the queue is fenced every BATCH_FRAMES frames, and a batch slower than this switches to per-operation fencing. */
export const BATCH_FRAMES = 8;
export const CONGESTED_BATCH_MS = 1200;

export interface FilmGuardLimits {
    /** At most FILM_LIMIT_MS: limits may only be lowered (G5 lowers them to test the guard). */
    readonly limitMs?: number;
    /** At most FILM_STOP_MS. */
    readonly stopMs?: number;
}

/** The limits, checked: positive, and never above the defaults. */
export function readGuardLimits(limits: FilmGuardLimits = {}): { readonly limitMs: number; readonly stopMs: number } {
    const limitMs = limits.limitMs ?? FILM_LIMIT_MS;
    const stopMs = limits.stopMs ?? FILM_STOP_MS;
    if (!(limitMs > 0 && limitMs <= FILM_LIMIT_MS && stopMs > 0 && stopMs <= FILM_STOP_MS)) {
        fail(`guard limits may only be lowered: limitMs ≤ ${FILM_LIMIT_MS} and stopMs ≤ ${FILM_STOP_MS}; got ${limitMs} and ${stopMs}.`);
    }
    return { limitMs, stopMs };
}

export interface FilmCalibration {
    /** The first step and draw, pipeline compilation included: recorded, not judged (an unverified first-submission risk). */
    readonly firstMs: number;
    readonly stepMs: readonly number[];
    readonly drawMs: readonly number[];
    /** stepsPerFrame × the slowest step + the slowest draw. */
    readonly msPerFrame: number;
    /** 8: fence every 8 frames. 1: fence and time every step and draw. */
    readonly fenceEvery: 1 | 8;
    readonly limitMs: number;
    readonly stopMs: number;
    readonly timing: 'queue-completion wall ms';
    /** What this evidence does not cover, in plain words, for the report. */
    readonly limitations: readonly string[];
}

export interface CalibrateFilmOptions {
    readonly width: number;
    readonly height: number;
    readonly stepsPerFrame: number;
    /** The frame drawn while calibrating (frame 0, usually). */
    readonly frame: FilmFrame;
    readonly limits?: FilmGuardLimits;
    /** Measured rounds of one step and one draw after the first. Default 3. */
    readonly rounds?: number;
    readonly signal?: AbortSignal;
}

const now = () => performance.now();
const checkSignal = (signal?: AbortSignal) => { if (signal?.aborted) throw new DOMException('Export cancelled', 'AbortError'); };

/** One operation, fenced: submit, then wait for the queue. */
async function timed(device: GPUDevice, run: () => void | Promise<void>): Promise<number> {
    const t = now();
    await run();
    await device.queue.onSubmittedWorkDone();
    return now() - t;
}

/**
 * Measure the workload's steps and draws at the film's size, before frame 0. It first waits out whatever the
 * app last submitted, so a preview's frame is never charged to the film. The first step and draw compile
 * pipelines (seconds on a cold driver cache), so they are recorded and not judged; then each of `rounds`
 * steps and draws is fenced, and one over `limitMs` refuses the film. Calibration steps the workload: reset
 * it, or build it again, before the film (§4.7).
 */
export async function calibrateFilm(engine: NullGraph, workload: FilmWorkload, o: CalibrateFilmOptions): Promise<FilmCalibration> {
    const { limitMs, stopMs } = readGuardLimits(o.limits);
    const device = engine.device;
    const rounds = o.rounds ?? 3;
    await device.queue.onSubmittedWorkDone();
    checkSignal(o.signal);
    const draw = async () => {
        await workload.prepare?.(o.frame);
        return timed(device, () => engine.render(workload.draw(o.frame)));
    };
    const t = now();
    await workload.step();
    await workload.prepare?.(o.frame);
    engine.render(workload.draw(o.frame));
    await device.queue.onSubmittedWorkDone();
    const firstMs = now() - t;
    const refuse = (kind: string, ms: number) => fail(
        `a ${kind} at ${o.width}×${o.height} took ${Math.round(ms).toLocaleString('en-US')} ms, over the ${limitMs.toLocaleString('en-US')} ms ` +
        'per-submission limit (Windows resets a GPU busy for ~2 s). Try a smaller size, fewer particles, or a faster GPU.');
    const stepMs: number[] = [];
    const drawMs: number[] = [];
    // A round measured across a page freeze timed the freeze (§4.9): it is not judged, and is measured again.
    const freezes = watchFreezes();
    let frozenRounds = 0;
    try {
        for (let i = 0; i < rounds; i++) {
            checkSignal(o.signal);
            const before = freezes.count;
            const s = await timed(device, () => workload.step());
            if (s > limitMs && freezes.count === before) refuse('step', s);
            checkSignal(o.signal);
            const d = await draw();
            if (d > limitMs && freezes.count === before) refuse('draw', d);
            if (freezes.count !== before && frozenRounds++ < 3) {
                i--;
                continue;
            }
            stepMs.push(s);
            drawMs.push(d);
        }
    } finally {
        freezes.stop();
    }
    const msPerFrame = o.stepsPerFrame * Math.max(0, ...stepMs) + Math.max(0, ...drawMs);
    return {
        firstMs, stepMs, drawMs, msPerFrame, fenceEvery: msPerFrame > PER_OPERATION_MS ? 1 : 8, limitMs, stopMs,
        timing: 'queue-completion wall ms',
        limitations: [
            'The first step and draw (pipeline compilation included) ran before any measurement: an unverified first-submission risk.',
            'Times are queue-completion wall milliseconds, not GPU timestamps; another user of the device or a busy compositor can lengthen them.',
            'Without a reduced-size probe, a first full-size operation cannot be predicted, only measured.',
            ...(frozenRounds ? [`The page was frozen during ${frozenRounds} round(s), measured again${frozenRounds > 3 ? ' until the limit of 3, then kept unjudged' : ''}.`] : []),
        ],
    };
}

/**
 * The runtime stop (§8, Q10): throws when a measured operation exceeds `stopMs` and is not excused as a single
 * outlier after a fast calibration (`< stopMs / 4`). `at` is the frame being made, or `'warm-up'` before frame 0.
 */
export function checkOperation(ms: number, at: number | 'warm-up', stopMs: number): void {
    if (ms > stopMs) {
        fail(`submission ${at === 'warm-up' ? 'during the warm-up' : `at frame ${at}`} took ${Math.round(ms).toLocaleString('en-US')} ms, ` +
            `over the ${stopMs.toLocaleString('en-US')} ms safety stop (a GPU stalled by other work can cause this too; try again on an idle GPU).`);
    }
}
