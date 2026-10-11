// src/extras/text/layout.ts
//
// Text layout (docs/text-labels-design.md §4): a string → glyph records in a flat buffer. Pure and
// GPU-free, so a Worker can lay out labels and hand the buffer over. Per-codepoint with kerning
// pairs; no shaping (non-goal §1).
//
// Glyph record, 16 bytes: u32 label, u32 glyph, f32 penX, f32 penY — the glyph origin inside the
// label in em units, y up, alignment applied. Whitespace advances the pen but emits no record.

/** u32 / f32 slots per glyph record. */
export const GLYPH_STRIDE = 4;
/** A record whose glyph is NONE is skipped by the cull — how a shortened label clears its tail. */
export const GLYPH_NONE = 0xFFFFFFFF;

/** The atlas surface layout needs — `FontAtlas` implements it; tests can fake it. */
export interface LayoutFont {
    /** Glyph index for a codepoint, or -1. */
    glyphIndex(codepoint: number): number;
    /** Pen advance of a glyph, em. */
    advance(glyph: number): number;
    /** Kerning between two codepoints, em (usually ≤ 0). */
    kerning(left: number, right: number): number;
    /** Baseline-to-baseline distance, em. */
    readonly lineHeight: number;
    /** Above the baseline, em (positive). */
    readonly ascender: number;
    /** Below the baseline, em (negative). */
    readonly descender: number;
    /** Whether a glyph draws anything (whitespace does not). */
    hasShape(glyph: number): boolean;
}

export interface LayoutOptions {
    /** Horizontal alignment of each line about the anchor. Default 'left'. */
    align?: 'left' | 'center' | 'right';
    /** What sits on the anchor vertically. Default 'alphabetic' (the first line's baseline). */
    baseline?: 'top' | 'middle' | 'alphabetic' | 'bottom';
    /** Extra space between glyphs, em. Default 0. */
    letterSpacing?: number;
    /** Line height as a multiple of the font's. Default 1. */
    lineHeight?: number;
    /** Wrap at spaces so no line is wider than this, em. Default: no wrapping. */
    maxWidth?: number;
}

export interface LayoutResult {
    /** Records written. */
    glyphs: number;
    /** Widest line, em. */
    width: number;
    /** Ascender of the first line to descender of the last, em. */
    height: number;
    lines: number;
    /** Codepoints the font has no glyph for (and no '?' fallback). */
    missing: number;
}

const QUESTION = 0x3F, SPACE = 0x20;

interface Placed { glyph: number; x: number }

/**
 * Lay `text` out for label `label`, writing records from glyph slot `offset`. `u32` and `f32`
 * must view the same memory. Throws, naming the text and the numbers, when the records would run
 * past `capacity` (default: the end of the views) — nothing is written in that case.
 */
export function layoutText(
    font: LayoutFont, text: string, label: number, opts: LayoutOptions,
    u32: Uint32Array, f32: Float32Array, offset: number, capacity: number = Math.floor(u32.length / GLYPH_STRIDE),
): LayoutResult {
    const spacing = opts.letterSpacing ?? 0;
    const lineH = font.lineHeight * (opts.lineHeight ?? 1);
    const maxWidth = opts.maxWidth ?? Infinity;
    const fallback = font.glyphIndex(QUESTION);
    const space = font.glyphIndex(SPACE);
    const spaceAdvance = space >= 0 ? font.advance(space) : 0.25;
    let missing = 0;

    // Resolve codepoints once: glyph index per codepoint (fallback applied), -2 for line breaks.
    const resolved: { cp: number; glyph: number }[] = [];
    for (const ch of text) {
        const cp = ch.codePointAt(0)!;
        if (cp === 0x0A) { resolved.push({ cp, glyph: -2 }); continue; }
        if (cp === SPACE) { resolved.push({ cp, glyph: space }); continue; }
        let g = font.glyphIndex(cp);
        if (g < 0) { missing++; g = fallback; }
        if (g < 0) continue;
        resolved.push({ cp, glyph: g });
    }

    // Break into lines: explicit newlines, then greedy word wrap at spaces.
    const lines: Placed[][] = [];
    const widths: number[] = [];
    let line: Placed[] = [], pen = 0, prevCp = -1;
    let wordStart = 0, penAtWordStart = 0;   // index into `line`, pen before the word

    const flush = () => {
        // Trim trailing letter spacing / spaces from the measured width.
        let w = 0;
        for (const p of line) if (font.hasShape(p.glyph)) w = Math.max(w, p.x + font.advance(p.glyph));
        lines.push(line); widths.push(w);
        line = []; pen = 0; prevCp = -1; wordStart = 0; penAtWordStart = 0;
    };

    for (const r of resolved) {
        if (r.glyph === -2) { flush(); continue; }
        if (r.cp === SPACE) {
            pen += spaceAdvance + spacing;
            prevCp = r.cp;
            wordStart = line.length; penAtWordStart = pen;
            continue;
        }
        if (prevCp >= 0) pen += font.kerning(prevCp, r.cp);
        const x = pen;
        pen += font.advance(r.glyph) + spacing;
        line.push({ glyph: r.glyph, x });
        prevCp = r.cp;
        // Wrap: the current word overflows and something precedes it on this line.
        if (pen - spacing > maxWidth && wordStart > 0) {
            const word = line.splice(wordStart);
            const shift = penAtWordStart;
            flush();
            for (const p of word) line.push({ glyph: p.glyph, x: p.x - shift });
            pen = line.length ? line[line.length - 1].x + font.advance(line[line.length - 1].glyph) + spacing : 0;
            prevCp = r.cp;
        }
    }
    flush();

    let count = 0;
    for (const l of lines) for (const p of l) if (font.hasShape(p.glyph)) count++;
    if (offset < 0 || offset + count > capacity) {
        const preview = text.length > 24 ? `${text.slice(0, 24)}…` : text;
        throw new Error(`layoutText: "${preview}" needs ${count} glyph records at offset ${offset}, ` +
            `but capacity is ${capacity} — raise maxGlyphs, or reuse freed ranges.`);
    }

    const nLines = lines.length;
    const height = font.ascender - font.descender + (nLines - 1) * lineH;
    const baseline = opts.baseline ?? 'alphabetic';
    // y of the FIRST line's baseline, relative to the anchor.
    const firstBaseline =
        baseline === 'alphabetic' ? 0 :
        baseline === 'top' ? -font.ascender :
        baseline === 'bottom' ? (nLines - 1) * lineH - font.descender :
        /* middle */ (nLines - 1) * lineH / 2 - (font.ascender + font.descender) / 2;

    let k = offset;
    for (let li = 0; li < nLines; li++) {
        const w = widths[li];
        const dx = opts.align === 'center' ? -w / 2 : opts.align === 'right' ? -w : 0;
        const y = firstBaseline - li * lineH;
        for (const p of lines[li]) {
            if (!font.hasShape(p.glyph)) continue;
            const o = k * GLYPH_STRIDE;
            u32[o] = label;
            u32[o + 1] = p.glyph;
            f32[o + 2] = p.x + dx;
            f32[o + 3] = y;
            k++;
        }
    }

    return { glyphs: count, width: Math.max(0, ...widths), height, lines: nLines, missing };
}
