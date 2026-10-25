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
