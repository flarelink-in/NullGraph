# Text API Usage

`null-graph/text` draws labels in the world — names over units, values over data points, signs on a
map — at object scale: **hundreds of thousands of labels in one draw call**, GPU-culled, crisp at
any size, fillable from a Worker, and able to follow positions only the GPU knows. Design:
[`text-labels-design.md`](../text-labels-design.md).

Labels are **screen-space**: a constant size in pixels, anchored to a world point, optionally
depth-tested at that point. UI and paragraphs stay in the DOM.

## 1. A font atlas

```ts
import { FontAtlas } from 'null-graph/text';

// Zero setup, browser or Worker: any font the browser can draw, distance-transformed now.
const atlas = FontAtlas.fromCanvas(engine, { fontFamily: 'Inter, sans-serif', fontWeight: 600 });

// Best quality: msdf-atlas-gen output (MSDF / MTSDF / SDF), e.g.
//   msdf-atlas-gen -font Inter.ttf -type msdf -size 48 -pxrange 6 -imageout inter.png -json inter.json
const img = await createImageBitmap(await (await fetch('inter.png')).blob(),
    { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });   // the image is DATA, not colour
const msdf = FontAtlas.fromMSDF(engine, await (await fetch('inter.json')).json(), img);
```

`fromCanvas` needs `OffscreenCanvas`; in Node, load a pre-generated atlas with `fromMSDF` (pass raw
RGBA bytes as `{ data, width, height }`). The module ships no font data.

## 2. Labels

```ts
import { TextSystem } from 'null-graph/text';

const text = new TextSystem(engine, atlas, {
    maxLabels: 50_000,
    maxGlyphs: 500_000,
    depthTest: true,          // hide labels behind closer geometry (compared at the anchor)
    maxDistance: 400,         // cull labels farther than this from the eye
});
const batch = text.createDrawBatch(scenePass);          // put it LAST in its pass

text.setLabel(0, {
    text: 'Base Camp',
    position: [12, 4, -30],
    sizePx: 18,
    color: [1, 1, 1, 1],
    halo: { color: [0, 0, 0, 0.8], width: 2 },          // readable over anything
    offsetPx: [0, 12],                                  // nudge up 12 px on screen
    align: 'center', baseline: 'bottom',
});

// Per frame:
text.upload();                                          // dirty ranges only
engine.render([...vis.getComputeStages(), ...text.getComputeStages(), scenePass]);
```

Cheap updates without re-layout: `setLabelPosition(i, x, y, z)`, `setLabelHidden(i, hidden)`.
Layout options: `align` (`left`/`center`/`right`), `baseline` (`top`/`middle`/`alphabetic`/`bottom`),
`letterSpacing` and `maxWidth` (em, word wrap at spaces), `lineHeight` (multiple), `\n` for lines.

## 3. Labels that follow the GPU

Point labels at positions in any GPU buffer — the cull and draw read them there, so moving 50,000
agents costs the CPU nothing:

```ts
const text = new TextSystem(engine, atlas, {
    maxLabels: 50_000, maxGlyphs: 400_000,
    follow: { buffer: vis.instanceBuffer, strideFloats: 24, positionOffset: 12 },   // model translation
});
text.setLabel(i, { text: `unit ${i}`, follow: i, position: [0, 2.2, 0] });        // position = offset
```

## 4. From a Worker

`layoutText` is pure. Lay out in a Worker into flat records, transfer, adopt:

```ts
// worker
import { layoutText } from 'null-graph/text';
const n = layoutText(font, 'Hello', labelIndex, { align: 'center' }, glyphsU32, glyphsF32, offset).glyphs;

// main thread
text.setRawData(labelsF32, glyphsF32, glyphCount);     // adopted by reference; then text.upload()
```

`font` is anything implementing `LayoutFont` (glyph indices, advances, kerning, metrics) — a
`FontAtlas`, or a plain object built from the same atlas JSON. Records: label 12 floats
(`LABEL_FLOATS`), glyph 4 (`GLYPH_STRIDE`) — layouts in the design doc §4.

## 5. Notes

- **Scripts:** per-codepoint layout with kerning — Latin, Cyrillic, Greek, digits, CJK ideographs.
  No shaping, ligatures or right-to-left (Arabic, Devanagari… need a shaper).
- **Halos** are limited by the atlas's distance range: at most `distanceRange/2 · sizePx/emSize`
  pixels wide. Use a larger `distanceRange` (or an MTSDF atlas) for wide outlines.
- **Glyph capacity:** re-setting a label reuses its glyph range if the text fits, else takes a new
  range at the end. Give `maxGlyphs` headroom for labels whose text grows.
- One draw batch per `TextSystem` (it owns one cull). For a second view, use a second system.
- **Draw order inside a system is undefined.** The cull compacts visible glyphs with atomics, so
  overlapping labels blend in whatever order its threads finished. For labels that must read over
  the rest (a selection's details, a waypoint), give them a second small system with
  `depthTest: false`, created after the first so its batch draws later. Both systems can share
  one atlas and one follow buffer.
- **The depth test uses the anchor's depth across the whole quad.** A label drawn below a
  ground-level anchor (negative `offsetPx` y) sits over nearer ground and is hidden. Raise the
  anchor, offset upward, or use a `depthTest: false` system.
- **`destroy()`** releases the buffers and takes the batch out of its pass. Drop
  `getComputeStages()` from your sequence too.
- **Not yet:** decluttering (overlap removal), per-glyph colour, picking labels, world-size labels.

Executable reference: `test/text-gpu` (X1–X9 — position to the pixel, constant size, GPU cull = CPU
mirror, follow with zero CPU writes, depth test, halo, antialiasing, 600k glyphs in one draw).
