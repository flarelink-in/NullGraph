// src/extras/text/textShaders.ts
//
// Label cull and draw shaders (docs/text-labels-design.md §5). The two stages share one function
// — anchorAndSnap — so the cull and the draw cannot disagree about where a glyph lands: the cull
// rejects exactly the quads the draw would have put off screen.
//
// Records (byte layouts mirror TextSystem's CPU views):
//   Glyph       16 B  { label, glyph, pen.xy }
//   Label       48 B  { pos.xyz, sizePx, color, halo, haloPx, follow, offsetPx.xy, flags, _pad }
//   AtlasGlyph  32 B  { uv (uLeft, vBottom, uRight, vTop), plane (l, b, r, t) }
//   TextParams  32 B  { viewport.xy, distanceRange, emSize, maxDistance, glyphCount, followStride, followOffset }
//
// Explanations stay in TS comments: text inside a WGSL template ships in the bundle.

import type { DistanceType } from './FontAtlas';
import { CAMERA_STRUCT_WGSL } from '../visibility/visibilityShaders';

/** Floats per label record. */
export const LABEL_FLOATS = 12;
/** Floats in the params uniform. */
export const TEXT_PARAMS_FLOATS = 8;
/** `Label.follow` value meaning "not following". */
export const FOLLOW_NONE = 0xFFFFFFFF;
/** `Label.flags` bit 0 — the label is hidden (culled, not drawn). */
export const LABEL_FLAG_HIDDEN = 1;

const RECORDS_WGSL = /* wgsl */ `
struct Glyph { label: u32, glyph: u32, pen: vec2<f32> };
struct Label {
    pos: vec3<f32>, sizePx: f32,
    color: u32, halo: u32, haloPx: f32, follow: u32,
    offsetPx: vec2<f32>, flags: u32, _pad: u32,
};
struct AtlasGlyph { uv: vec4<f32>, plane: vec4<f32> };
struct TextParams {
    viewport: vec2<f32>, distanceRange: f32, emSize: f32,
    maxDistance: f32, glyphCount: u32, followStride: u32, followOffset: u32,
};
`;

// The anchor (followed position + offset, or the label's own position) and its pixel-snapped
// screen position. Snapping to whole pixels keeps small text from shimmering as it moves.
const ANCHOR_WGSL = /* wgsl */ `
fn anchorWorld(L: Label) -> vec3<f32> {
    if (params.followStride > 0u && L.follow != 0xFFFFFFFFu) {
        let b = L.follow * params.followStride + params.followOffset;
        return vec3<f32>(follow[b], follow[b + 1u], follow[b + 2u]) + L.pos;
    }
    return L.pos;
}

fn snappedPixel(clip: vec4<f32>) -> vec2<f32> {
    let ndc = clip.xy / clip.w;
    return floor((ndc * vec2<f32>(0.5, -0.5) + 0.5) * params.viewport + 0.5);
}

fn cornerPixel(anchorPx: vec2<f32>, L: Label, G: Glyph, corner: vec2<f32>) -> vec2<f32> {
    let off = L.offsetPx + (G.pen + corner) * L.sizePx;
    return anchorPx + vec2<f32>(off.x, -off.y);
}
`;

/** Bindings shared by the cull (group 0) and the draw (group 1), numbered from `first`. */
function dataBindings(group: number, first: number): string {
    return /* wgsl */ `
@group(${group}) @binding(${first}) var<storage, read> glyphs: array<Glyph>;
@group(${group}) @binding(${first + 1}) var<storage, read> labels: array<Label>;
@group(${group}) @binding(${first + 2}) var<storage, read> atlasGlyphs: array<AtlasGlyph>;
@group(${group}) @binding(${first + 3}) var<storage, read> follow: array<f32>;
@group(${group}) @binding(${first + 4}) var<uniform> params: TextParams;
`;
}

/**
 * The cull: one thread per glyph record. Group 0: 0 camera, 1 glyphs, 2 labels, 3 atlasGlyphs,
 * 4 follow, 5 params, 6 visible (out), 7 drawArgs (out).
 */
export function buildTextCullWGSL(): string {
    return /* wgsl */ `
${CAMERA_STRUCT_WGSL}
${RECORDS_WGSL}
@group(0) @binding(0) var<uniform> camera: Camera;
${dataBindings(0, 1)}
@group(0) @binding(6) var<storage, read_write> visible: array<u32>;
@group(0) @binding(7) var<storage, read_write> drawArgs: array<atomic<u32>, 5>;
${ANCHOR_WGSL}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.glyphCount) { return; }
    let G = glyphs[i];
    if (G.glyph == 0xFFFFFFFFu) { return; }
    let L = labels[G.label];
    if ((L.flags & 1u) != 0u) { return; }
    let world = anchorWorld(L);
    if (distance(world, camera.eyePos) > params.maxDistance) { return; }
    let clip = camera.viewProj * vec4<f32>(world, 1.0);
    if (clip.w <= 0.0 || clip.z < 0.0 || clip.z > clip.w) { return; }
    let A = atlasGlyphs[G.glyph];
    let anchorPx = snappedPixel(clip);
    let a = cornerPixel(anchorPx, L, G, A.plane.xy);
    let b = cornerPixel(anchorPx, L, G, A.plane.zw);
    let lo = min(a, b);
    let hi = max(a, b);
    if (hi.x <= 0.0 || hi.y <= 0.0 || lo.x >= params.viewport.x || lo.y >= params.viewport.y) { return; }
    let slot = atomicAdd(&drawArgs[1], 1u);
    visible[slot] = i;
}
`;
}

/**
 * The draw: vertex pulling over one quad per visible glyph. Group 0: 0 camera, 1 visible (the
 * batch's own storage — the cull's output). Group 1: 0 sampler, 1 atlas, 2.. the data bindings.
 */
export function buildTextDrawWGSL(type: DistanceType): string {
    // MSDF / MTSDF: the median of RGB is the distance; MTSDF's alpha is a true SDF, which survives
    // the wide offsets a halo asks for. SDF: one channel.
    const fill = type === 'sdf' ? 's.r' : 'max(min(s.r, s.g), min(max(s.r, s.g), s.b))';
    const halo = type === 'mtsdf' ? 's.a' : fill;
    return /* wgsl */ `
${CAMERA_STRUCT_WGSL}
${RECORDS_WGSL}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> visible: array<u32>;
@group(1) @binding(0) var atlasSampler: sampler;
@group(1) @binding(1) var atlasTex: texture_2d<f32>;
${dataBindings(1, 2)}
${ANCHOR_WGSL}

struct TextOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) color: u32,
    @location(2) @interpolate(flat) halo: u32,
    @location(3) @interpolate(flat) pxRange: f32,
    @location(4) @interpolate(flat) haloPx: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> TextOut {
    let G = glyphs[visible[ii]];
    let L = labels[G.label];
    let A = atlasGlyphs[G.glyph];
    let clip = camera.viewProj * vec4<f32>(anchorWorld(L), 1.0);
    let right = (vi & 1u) == 1u;
    let top = (vi & 2u) == 2u;
    let corner = vec2<f32>(select(A.plane.x, A.plane.z, right), select(A.plane.y, A.plane.w, top));
    let px = cornerPixel(snappedPixel(clip), L, G, corner);
    let ndc = (px / params.viewport - 0.5) * vec2<f32>(2.0, -2.0);
    var o: TextOut;
    o.position = vec4<f32>(ndc * clip.w, clip.z, clip.w);
    o.uv = vec2<f32>(select(A.uv.x, A.uv.z, right), select(A.uv.y, A.uv.w, top));
    o.color = L.color;
    o.halo = L.halo;
    o.pxRange = params.distanceRange * L.sizePx / params.emSize;
    o.haloPx = min(L.haloPx, 0.5 * o.pxRange - 0.5);
    return o;
}

@fragment
fn fs_main(v: TextOut) -> @location(0) vec4<f32> {
    let s = textureSample(atlasTex, atlasSampler, v.uv);
    let fillD = (${fill} - 0.5) * v.pxRange;
    let haloD = (${halo} - 0.5) * v.pxRange;
    let fa = clamp(fillD + 0.5, 0.0, 1.0);
    let ha = select(0.0, clamp(haloD + v.haloPx + 0.5, 0.0, 1.0), v.haloPx > 0.0);
    let fc = unpack4x8unorm(v.color);
    let hc = unpack4x8unorm(v.halo);
    let f = fc.a * fa;
    let h = hc.a * ha * (1.0 - f);
    return vec4<f32>(fc.rgb * f + hc.rgb * h, f + h);
}
`;
}
