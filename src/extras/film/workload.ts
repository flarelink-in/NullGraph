// D2.4 (offline-films-browser-design.md §4.1, §4.6): what recordFilm asks of an application's workload. A
// workload may be a simulation, a replay, a procedural renderer or a scene adapter; the runner requires no
// scene graph, seed scheme or camera model. It builds on the film's own engine, which adopts the app's device,
// and owns only what it creates.

import type { ComputeStageNode, NullGraph, RenderPassNode } from '../../core';
import type { CameraView } from './camera';

/** What `engine.render()` takes: compute stages and render passes, in order. */
export type RenderSequence = (ComputeStageNode | RenderPassNode)[];

/** One frame of the film, as the workload sees it. */
export interface FilmFrame {
    readonly index: number;
    /** `index / fps`: film time, never wall time. */
    readonly seconds: number;
    readonly timestampUs: number;
    readonly durationUs: number;
    /** The camera path's view at `seconds`, or null when the film has no camera path (the workload's own view holds). */
    readonly view: CameraView | null;
}

export interface FilmWorkloadContext {
    /** The film's engine, on the app's device; destroyed by the runner after `dispose()`. */
    readonly engine: NullGraph;
    readonly width: number;
    readonly height: number;
    readonly signal: AbortSignal;
    /**
     * Register a release for something just made, before the next thing that can fail: if construction
     * throws, these run (last-first). Once construction succeeds they are dropped, and `dispose()` owns
     * the workload's resources, so nothing is released twice.
     */
    defer(cleanup: () => void | Promise<void>): void;
}

export interface FilmWorkload {
    /** The fixed step, in seconds; a frame is a whole number of steps (§2). A stateless workload steps as a no-op. */
    readonly dt: number;
    /**
     * Put the workload back in its exact initial state, placement included, without the warm-up. Called
     * after calibration, which steps and draws. Without it the runner disposes the workload and builds a
     * new one.
     */
    reset?(): void | Promise<void>;
    /** One step: one bounded GPU submission (or a few), never a fence the runner has to wait out. */
    step(): void | Promise<void>;
    /** Optional bounded work before a frame is drawn (derived reconstruction, a replay's read). */
    prepare?(frame: FilmFrame): void | Promise<void>;
    /**
     * The frame's render sequence, synchronously, applying `frame.view` if there is one. It must not submit
     * or yield: the runner renders it and captures the frame in the same task (§4.3).
     */
    draw(frame: FilmFrame): RenderSequence;
    /** Release what the workload made (never what it borrowed). Idempotent. */
    dispose(): void | Promise<void>;
    /** Anything the report should carry about the workload (settings, a seed, versions). */
    readonly info?: Record<string, unknown>;
}

export type CreateFilmWorkload = (context: FilmWorkloadContext) => Promise<FilmWorkload>;
