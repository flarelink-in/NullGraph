// D2.4 (offline-films-browser-design.md §4.9): the export session's skeleton. Two pieces F2 builds on:
// the lifecycle's states and the transitions between them, and the ledger that unwinds what a session
// acquired, in reverse order, attempting every release. Pure.

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
 * `recording` straight to `finalizing`; a kept partial goes through `stopping`.
 */
export const FILM_TRANSITIONS: Readonly<Record<FilmState, readonly FilmState[]>> = {
    probing: ['initializing', 'disposing'],
    initializing: ['calibrating', 'disposing'],
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
        if (!FILM_TRANSITIONS[this.current].includes(next)) {
            fail(`a session cannot go from ${this.current} to ${next}; from ${this.current} it goes to ${FILM_TRANSITIONS[this.current].join(' or ') || 'nothing (it has ended)'}.`);
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
