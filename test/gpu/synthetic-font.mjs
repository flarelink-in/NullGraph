// A synthetic font with EXACT shapes — boxes, a bar, a ring — and analytic distance fields.
//
// Real fonts make pixel assertions approximate ("roughly an A here"); these glyphs are rectangles
// whose screen footprint the tests compute to the pixel, and whose fields are analytic rather than
// rasterised, so every expectation is a number. No font file, no licence, no download.
//
// Returns an AtlasDescription + data for FontAtlas.fromSDF, plus `shapes`: each glyph's filled
// region in em (the quad is larger — it carries the distance range's padding).

const EM = 32;            // atlas pixels per em
const RANGE = 8;          // distance range, atlas pixels (±4 px)

/** Filled region per glyph, em, y up. A ring is an outer box minus an inner one. */
export const SHAPES = {
    A: { advance: 0.7, outer: [0.1, 0.0, 0.6, 0.7] },
    B: { advance: 0.6, outer: [0.0, 0.0, 0.5, 0.5] },
    I: { advance: 0.55, outer: [0.2, 0.0, 0.35, 0.7] },
    O: { advance: 0.7, outer: [0.05, 0.0, 0.65, 0.7], inner: [0.2, 0.15, 0.5, 0.55] },
    '?': { advance: 0.6, outer: [0.2, 0.3, 0.4, 0.5] },
};
export const KERN_AB = -0.1;
export const SPACE_ADVANCE = 0.3;
export const METRICS = { lineHeight: 1.2, ascender: 0.8, descender: -0.2 };

/** Signed distance to a box, em, POSITIVE INSIDE. */
function boxSD(x, y, [l, b, r, t]) {
    const dx = Math.max(l - x, x - r), dy = Math.max(b - y, y - t);
    const outside = Math.hypot(Math.max(dx, 0), Math.max(dy, 0));
    const inside = Math.min(Math.max(dx, dy), 0);
    return -(outside + inside);
}

function shapeSD(x, y, s) {
    const d = boxSD(x, y, s.outer);
    return s.inner ? Math.min(d, -boxSD(x, y, s.inner)) : d;
}

/**
 * @param {{ type?: 'sdf' | 'msdf' | 'mtsdf', omitFallback?: boolean }} opts
 *   `msdf` / `mtsdf` store the same field in R, G, B (and A), so the median path is exercised.
 */
export function syntheticFont({ type = 'sdf', omitFallback = false } = {}) {
    const pad = RANGE / 2 / EM;                       // em of padding around each shape
    const entries = Object.entries(SHAPES).filter(([ch]) => !(omitFallback && ch === '?'));
    const cells = entries.map(([ch, s]) => {
        const [l, b, r, t] = s.outer;
        // Whole-texel cells, and plane bounds that span EXACTLY those texels — otherwise the quad
        // would stretch the field by the rounding.
        const w = Math.ceil((r - l + 2 * pad) * EM), h = Math.ceil((t - b + 2 * pad) * EM);
        const plane = { left: l - pad, bottom: b - pad, right: l - pad + w / EM, top: b - pad + h / EM };
        return { ch, s, plane, w, h };
    });
    const width = 256;
    let x = 1;
    const height = Math.max(...cells.map(c => c.h)) + 2;
    const channels = type === 'sdf' ? 1 : 4;
    const data = new Uint8Array(width * height * channels);
    const glyphs = [{ unicode: 32, advance: SPACE_ADVANCE }];
    for (const c of cells) {
        for (let py = 0; py < c.h; py++) {
            for (let px = 0; px < c.w; px++) {
                // texel centre → em (row 0 is the cell's top edge)
                const ex = c.plane.left + (px + 0.5) / EM;
                const ey = c.plane.top - (py + 0.5) / EM;
                const v = Math.round(Math.min(1, Math.max(0, 0.5 + shapeSD(ex, ey, c.s) * EM / RANGE)) * 255);
                const o = ((1 + py) * width + x + px) * channels;
                for (let k = 0; k < channels; k++) data[o + k] = v;
            }
        }
        glyphs.push({
            unicode: c.ch.codePointAt(0), advance: c.s.advance, plane: c.plane,
            atlas: { left: x, bottom: 1 + c.h, right: x + c.w, top: 1 },
        });
        x += c.w + 1;
    }
    return {
        type, distanceRange: RANGE, emSize: EM, width, height, yOrigin: 'top', ...METRICS,
        glyphs, kerning: [{ left: 65, right: 66, advance: KERN_AB }], data,
    };
}
