// Spike S1's alpha pattern (docs/offline-films-browser-design.md §4.4 "Colour", Q1): 4 × 2 patches, top row
// first, shared by the page (which draws it) and run.mjs (which reads the decoded video back). The expected
// bytes are the unorm8 of what the shader writes; an opaque video should keep the RGB and drop the alpha,
// as the readback's 'RGBX' frames and ffmpeg's yuv420p do.

export const PATCHES = [
    { name: 'red', rgba: [255, 0, 0, 255] },
    { name: 'green', rgba: [0, 255, 0, 255] },
    { name: 'blue', rgba: [0, 0, 255, 255] },
    { name: 'grey', rgba: [128, 128, 128, 255] },
    { name: 'orange a=1', rgba: [204, 102, 51, 255] },
    { name: 'premultiplied a=0.5', rgba: [102, 51, 26, 128] },
    { name: 'additive a=0', rgba: [204, 102, 51, 0] },
    { name: 'clear a=0', rgba: [0, 0, 0, 0] },
];

export const PATTERN_WGSL = /* wgsl */`
@group(0) @binding(0) var<uniform> dims: vec4f;
struct V { @builtin(position) pos: vec4f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> V {
    var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    var o: V;
    o.pos = vec4f(p[i], 0.0, 1.0);
    return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f {
    var colors = array<vec4f, 8>(
        vec4f(1.0, 0.0, 0.0, 1.0), vec4f(0.0, 1.0, 0.0, 1.0), vec4f(0.0, 0.0, 1.0, 1.0), vec4f(0.5, 0.5, 0.5, 1.0),
        vec4f(0.8, 0.4, 0.2, 1.0), vec4f(0.4, 0.2, 0.1, 0.5), vec4f(0.8, 0.4, 0.2, 0.0), vec4f(0.0, 0.0, 0.0, 0.0));
    let col = min(u32(v.pos.x / dims.x * 4.0), 3u);
    let row = min(u32(v.pos.y / dims.y * 2.0), 1u);
    return colors[row * 4u + col];
}`;

/** The pixel at the centre of patch i, far from the 4:2:0 chroma edges. */
export const patchCentre = (i, width, height) => [Math.floor(((i % 4) + 0.5) * width / 4), Math.floor((Math.floor(i / 4) + 0.5) * height / 2)];
