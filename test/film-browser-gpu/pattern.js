// F2's browser G tests (docs/offline-films-browser-design.md §10): the test pattern, shared by the page that
// draws it (workloads.js) and run.mjs, which reads the decoded video back. Three bands, top row first:
//   1. eight 100% colour bars, white to black, with a red marker square near the white bar's top-left
//      corner: a flip in either direction moves it;
//   2. eight patches, alpha 1, below 1 and 0: S1's premultiplied and additive glows and a clear patch among
//      them (§4.4 "Colour");
//   3. the frame index in BITS bits, white for a 1, most significant bit first: a decoded frame says which
//      frame it is, so a dropped, doubled or reordered frame is caught (G3).
// Every colour is whole bytes / 255 and every edge is integer pixel arithmetic, the same in WGSL and here, so a
// GPU writes exactly patternFrame()'s bytes.

export const BARS = [
    { name: 'white', rgba: [255, 255, 255, 255] },
    { name: 'yellow', rgba: [255, 255, 0, 255] },
    { name: 'cyan', rgba: [0, 255, 255, 255] },
    { name: 'green', rgba: [0, 255, 0, 255] },
    { name: 'magenta', rgba: [255, 0, 255, 255] },
    { name: 'red', rgba: [255, 0, 0, 255] },
    { name: 'blue', rgba: [0, 0, 255, 255] },
    { name: 'black', rgba: [0, 0, 0, 255] },
];

export const PATCHES = [
    { name: 'grey 50%', rgba: [128, 128, 128, 255] },
    { name: 'orange a=1', rgba: [204, 102, 51, 255] },
    { name: 'premultiplied a=0.5', rgba: [102, 51, 26, 128] },
    { name: 'additive a=0', rgba: [204, 102, 51, 0] },
    { name: 'clear a=0', rgba: [0, 0, 0, 0] },
    { name: 'grey 25%', rgba: [64, 64, 64, 255] },
    { name: 'grey 75%', rgba: [191, 191, 191, 255] },
    { name: 'teal', rgba: [51, 153, 102, 255] },
];

export const MARKER = { name: 'marker', rgba: [255, 0, 0, 255] };
export const BITS = 16;

/** The marker square: x in [w/32, w/32 + w/16), y in [h/24, h/24 + h/12), integer division as in the shader. */
const marker = (width, height) => {
    const x0 = Math.floor(width / 32), y0 = Math.floor(height / 24);
    return [x0, x0 + Math.floor(width / 16), y0, y0 + Math.floor(height / 12)];
};

/** Pixel (x, y) of frame `frame`, RGBA8, as the shader writes it. */
export function patternPixel(x, y, width, height, frame) {
    const band = Math.min(Math.floor(y * 3 / height), 2);
    if (band === 0) {
        const [x0, x1, y0, y1] = marker(width, height);
        if (x >= x0 && x < x1 && y >= y0 && y < y1) return MARKER.rgba;
        return BARS[Math.min(Math.floor(x * 8 / width), 7)].rgba;
    }
    if (band === 1) return PATCHES[Math.min(Math.floor(x * 8 / width), 7)].rgba;
    const bit = Math.min(Math.floor(x * BITS / width), BITS - 1);
    return (frame >> (BITS - 1 - bit)) & 1 ? BARS[0].rgba : BARS[7].rgba;
}

/** Frame `frame` as tightly packed RGBA8, top row first. */
export function patternFrame(width, height, frame) {
    const out = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) out.set(patternPixel(x, y, width, height, frame), (y * width + x) * 4);
    return out;
}

/**
 * Where run.mjs samples a decoded frame: the centre of every bar and patch and of the marker, each far from
 * a 4:2:0 chroma edge, with what an opaque video should hold there (the RGB; the alpha byte is dropped).
 */
export function patternSamples(width, height) {
    const cx = i => Math.floor((i + 0.5) * width / 8);
    const cy = band => Math.floor((band + 0.5) * height / 3);
    const [x0, x1, y0, y1] = marker(width, height);
    return [
        ...BARS.map((b, i) => ({ name: b.name, x: cx(i), y: cy(0), rgb: b.rgba.slice(0, 3) })),
        ...PATCHES.map((p, i) => ({ name: p.name, x: cx(i), y: cy(1), rgb: p.rgba.slice(0, 3), alpha: p.rgba[3] })),
        { name: MARKER.name, x: Math.floor((x0 + x1) / 2), y: Math.floor((y0 + y1) / 2), rgb: MARKER.rgba.slice(0, 3) },
    ];
}

/** The bar code's band, for ffmpeg's crop: x, y, width, height. Each of its BITS columns is one bit. */
export const barcodeBand = (width, height) => ({ x: 0, y: Math.ceil(height * 2 / 3), width, height: height - Math.ceil(height * 2 / 3) });

/**
 * The pattern's shader, for a batch in a main-screen pass: a full-screen triangle. Group 0 is the batch's own
 * (binding 0 the engine's camera uniform, binding 1 the batch's storage, here [frame, width, height, 0]); the
 * camera is read only so that the automatic layout keeps it.
 */
export const PATTERN_WGSL = /* wgsl */`
@group(0) @binding(0) var<uniform> camera: vec4f;
@group(0) @binding(1) var<storage, read> info: array<vec4f>;

struct V { @builtin(position) pos: vec4f };

@vertex fn vs_main(@builtin(vertex_index) i: u32) -> V {
    var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    var o: V;
    o.pos = vec4f(p[i], 0.0, 1.0);
    return o;
}

fn byteColour(c: vec4u) -> vec4f { return vec4f(c) / 255.0; }

@fragment fn fs_main(v: V) -> @location(0) vec4f {
    if (camera.w < -1.0e30) { discard; }
    let frame = u32(info[0].x);
    let w = u32(info[0].y);
    let h = u32(info[0].z);
    let x = u32(v.pos.x);
    let y = u32(v.pos.y);
    let band = min(y * 3u / h, 2u);
    let column = min(x * 8u / w, 7u);
    if (band == 0u) {
        let x0 = w / 32u;
        let y0 = h / 24u;
        if (x >= x0 && x < x0 + w / 16u && y >= y0 && y < y0 + h / 12u) { return byteColour(vec4u(${MARKER.rgba.join('u, ')}u)); }
        var bars = array<vec4u, 8>(${BARS.map(b => `vec4u(${b.rgba.join('u, ')}u)`).join(', ')});
        return byteColour(bars[column]);
    }
    if (band == 1u) {
        var patches = array<vec4u, 8>(${PATCHES.map(p => `vec4u(${p.rgba.join('u, ')}u)`).join(', ')});
        return byteColour(patches[column]);
    }
    let bit = min(x * ${BITS}u / w, ${BITS - 1}u);
    if (((frame >> (${BITS - 1}u - bit)) & 1u) == 1u) { return byteColour(vec4u(255u, 255u, 255u, 255u)); }
    return byteColour(vec4u(0u, 0u, 0u, 255u));
}`;
