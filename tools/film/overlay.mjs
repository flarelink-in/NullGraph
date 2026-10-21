// O.0 (docs/film-overlays-design.md §2, §3): the text pass over a finished film. This file is the pure part: ffmpeg
// `drawtext` filters from a list of overlay items. The film draws no text under Node, so text is burned after encoding.
// Pure: no node: imports, so the headless tests check every refusal.

const fail = message => { throw new Error(`film: overlays: ${message}`); };
const number = value => typeof value === 'number' && Number.isFinite(value);
const place = value => number(value) || (typeof value === 'string' && value !== '');

/**
 * Text as ffmpeg reads it. A filter graph is read twice, the graph (where [],; and quotes are special) and then each
 * filter's options (where : and quotes are), so a value is escaped for the options first and then for the graph.
 */
export const filterValue = text => String(text).replace(/[\\:']/g, c => `\\${c}`).replace(/[\\'[\],;]/g, c => `\\${c}`);

/** An expression for the graph: its commas separate filter options, so they are escaped; nothing else needs it. */
const expression = text => text.replace(/,/g, '\\,');

const FIELDS = ['text', 'x', 'y', 'size', 'from', 'to', 'fade'];

/**
 * The overlay list of a film file, checked: a copy of each item with only the fields it gave. `seconds` is the film's
 * length, the default end of an item. An item that ends after the film is allowed: a shorter preview (`--seconds`)
 * of the same film file then simply never shows it.
 */
export function readOverlays(list, seconds = Infinity) {
    if (!Array.isArray(list)) fail('overlays must be a list of items.');
    return list.map((item, i) => {
        const at = `overlay ${i}`;
        if (item === null || typeof item !== 'object' || Array.isArray(item)) fail(`${at} must be an object.`);
        for (const key of Object.keys(item)) if (!FIELDS.includes(key)) fail(`${at}: unknown field "${key}" (use ${FIELDS.join(', ')}).`);
        if (typeof item.text !== 'string' || item.text === '') fail(`${at}: text must be a non-empty string.`);
        if (item.size !== undefined && !(number(item.size) && item.size > 0)) fail(`${at}: size must be a number above zero.`);
        for (const axis of ['x', 'y']) if (item[axis] !== undefined && !place(item[axis])) fail(`${at}: ${axis} must be a number or an ffmpeg expression.`);
        for (const edge of ['from', 'to']) if (item[edge] !== undefined && !(number(item[edge]) && item[edge] >= 0)) fail(`${at}: ${edge} must be 0 or more.`);
        const from = item.from ?? 0, to = item.to ?? seconds;
        if (!(from < to)) fail(`${at}: from must be below to (${from} and ${to}).`);
        if (item.fade !== undefined && !(number(item.fade) && item.fade >= 0)) fail(`${at}: fade must be 0 or more.`);
        if (item.fade > 0 && !Number.isFinite(to)) fail(`${at}: fade needs the film's length or an end (to).`);
        if ((item.fade ?? 0) > (to - from) / 2) fail(`${at}: fade (${item.fade} s) must be at most half the item's range (${to - from} s).`);
        return { ...item };
    });
}

/**
 * One `drawtext` filter per overlay item, in order. `width`, `height` and `seconds` are the film's; `font` is a font file.
 * An item is `{ text, x, y, size, from, to, fade }` (design §2). `x` and `y` are pixels or ffmpeg expressions
 * (`w-text_w-40`); `from` and `to` are seconds in the finished film, defaulting to all of it.
 */
export function overlayFilters({ overlays, width, height, seconds = Infinity, font }) {
    const size = Math.round(height / 30), pad = Math.round(size * 0.6);
    return readOverlays(overlays, seconds).map(item => {
        const from = item.from ?? 0, to = item.to ?? seconds, fade = item.fade ?? 0;
        let filter = `drawtext=fontfile=${filterValue(font)}:text=${filterValue(item.text)}:expansion=none:x=${item.x ?? pad}:y=${item.y ?? pad}`
            + `:fontsize=${item.size ?? size}:fontcolor=white:borderw=3:bordercolor=black@0.75`;
        if (item.from !== undefined || item.to !== undefined) {
            filter += `:enable=${expression(Number.isFinite(to) ? `between(t,${from},${to})` : `gte(t,${from})`)}`;
        }
        if (fade > 0) filter += `:alpha=${expression(`max(0,min(1,min((t-${from})/${fade},(${to}-t)/${fade})))`)}`;
        return filter;
    });
}
