// Shared by null-graph/film's pure modules. Every refusal starts "film: " and names the offender, the
// number and the way out (design-principles.md §2); the wording is offline-films-browser-design.md §8's,
// and where a check also exists in the Node tool (tools/film/manifest.mjs) the words are the same.

/** Throw a named refusal. */
export function fail(message: string): never {
    throw new Error(`film: ${message}`);
}

export const isObject = (value: unknown): value is Record<string, unknown> =>
    value !== null && typeof value === 'object' && !Array.isArray(value);

/** `value` when it is a whole number above zero, else a refusal naming `what`. */
export function whole(value: unknown, what: string): number {
    if (typeof value === 'number' && Number.isInteger(value) && value > 0) return value;
    return fail(`${what} must be a whole number above zero; got ${JSON.stringify(value)}.`);
}

/** `value` when it is a finite number above zero, else a refusal naming `what`. */
export function positive(value: unknown, what: string): number {
    if (typeof value === 'number' && Number.isFinite(value) && value > 0) return value;
    return fail(`${what} must be a number above zero; got ${JSON.stringify(value)}.`);
}

/** `text` without its closing full stop, for quoting another API's own words inside a message. */
export const clause = (text: string): string => text.replace(/[\s.]+$/, '');

/** `text` as one sentence ending in a full stop. */
export const sentence = (text: string): string => `${clause(text)}.`;

/**
 * The page's freezes (Page Lifecycle's freeze and resume events) while this watches. A frozen page runs no
 * task, so an operation measured across a freeze timed the freeze, not the GPU (§4.9, G9-background).
 * Where there is no document (a worker, Node) there is nothing to watch, and the count stays 0.
 */
export function watchFreezes(): { readonly count: number; stop(): void } {
    const doc = (globalThis as { document?: Pick<Document, 'addEventListener' | 'removeEventListener'> }).document;
    let count = 0;
    if (typeof doc?.addEventListener !== 'function') return { get count() { return count; }, stop() {} };
    const note = () => { count++; };
    doc.addEventListener('freeze', note);
    doc.addEventListener('resume', note);
    return {
        get count() { return count; },
        stop() {
            doc.removeEventListener('freeze', note);
            doc.removeEventListener('resume', note);
        },
    };
}

/**
 * A real task boundary, not a chain of resolved promises: the browser may render, present and take
 * input here (offline-films-browser-design.md §4.9). A MessageChannel message, as spike S1 measured,
 * not a timer, which a hidden page throttles to once a second. A channel of its own each time, closed
 * on arrival, so overlapping yields never steal each other's message and Node is not kept alive.
 */
export function yieldTask(): Promise<void> {
    if (typeof MessageChannel !== 'function') return new Promise(resolve => setTimeout(resolve, 0));
    return new Promise(resolve => {
        const channel = new MessageChannel();
        channel.port1.onmessage = () => {
            channel.port1.close();
            channel.port2.close();
            resolve();
        };
        channel.port2.postMessage(0);
    });
}
