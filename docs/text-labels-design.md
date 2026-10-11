# Text Labels — Design (Cycle B)

**Module:** `null-graph/text` · **Status:** ✍ → building · **Companion:** [`picking-design.md`](./picking-design.md)

## 1. Purpose

Labels in the 3D world — names over units, values over data points, signs on a map — at the scale
NullGraph renders objects: **hundreds of thousands of labels, one draw call**, GPU-culled, crisp at
any size, fillable from a Worker, and able to **follow GPU-driven positions with zero CPU work** (a
label over each of 50,000 agents whose positions only the GPU knows).

**Most important pre-existing fact:** the instance tier's pattern — a compute stage culls records
into a compacted list and writes the indirect instance count, one `drawIndexedIndirect` draws the
survivors (`visibilityShaders.ts:259-293`, `RenderPipeline.ts` indirect branch) — fits glyphs
exactly. A glyph is an instance whose "mesh" is one quad.

**Non-goals (v1):**
- **CSS / UI layout in WebGPU.** UI stays DOM; HTML-in-Canvas is a detect-later item (roadmap §6).
- **Complex scripts**: shaping, ligatures, bidi/RTL, combining marks. Layout is per-codepoint with
  kerning pairs — correct for Latin, Cyrillic, Greek, digits and CJK ideographs (no shaping needed),
  wrong for Arabic, Devanagari and similar.
- **Label collision / decluttering** (map-style placement). v2; it is a compute pass over the same
  records.
- **A bundled font.** The module ships no font data; the app brings an atlas (§6).
- Rich text within one label (per-glyph colour), text on paths, curved text.

**Principles:** draw calls do not scale with label count; the frame path allocates nothing; labels
are flat records a Worker can fill (`layoutText` is pure); baseline core WebGPU; fail loud.

## 2. Architecture

```
app / worker:  layoutText(atlas, "Hello", opts, glyphsU32, glyphsF32, offset)   (pure, CPU, on change)
               text.setLabel(i, {...}) or write the label record directly
per frame:     text.upload()                                                     (dirty ranges only)
engine.render([ ...vis.getComputeStages(), text.cull, scenePass(… text batch last …) ])

 glyph buffer ─┐        ┌──────────── cull (compute) ────────────┐      ┌─ draw (1 indirect) ─┐
 label buffer ─┼──────▶ │ per glyph: anchor → clip, behind/far?   │ ───▶ │ quad per visible     │
 follow buffer ┘        │ pixel-snapped quad off-screen? → append │      │ glyph, MSDF/SDF AA,  │
 (optional, any         └─────────────────────────────────────────┘      │ halo, premultiplied  │
  GPU buffer)                                                             └──────────────────────┘
```

## 3. Integration facts

| Fact | Where |
|---|---|
| Indirect batches draw `drawIndexedIndirect` from args a compute stage writes; vertex pulling needs only an index buffer | `RenderPipeline.ts` (indirect branch; `setVertexBuffer` is skipped when absent) |
| An indirect batch can reuse an external buffer instead of allocating its source copy | `PipelineConfig.sharedSourceBuffer`, `BatchManager.ts` |
| Group 0 binding 1 is the batch's own storage buffer — here, the compacted visible-glyph list | `BatchManager.createBatch` |
| A compute stage can reset a counter before it runs | `ComputeStageConfig.preClearBuffers` |
| Blend state and depth state are per batch | `PipelineConfig.blend / depthWriteEnabled / depthCompare` |
| Camera block: viewProj 0, eye 96, …; viewport size is not in it | `src/core/Camera.ts` — so the viewport goes in the text params |
| Thrown strings and WGSL comments ship in the bundle; TS comments do not | design note from Cycle A (`docs/handoff.md`) |

## 4. Data model

**Glyph record** (16 B) — one per glyph instance, written by `layoutText`:
`{ label: u32, glyph: u32, pen: vec2<f32> }` — which label, which atlas glyph, and the glyph origin
inside the label in **em** units (y up), alignment already applied.

**Label record** (48 B):

| Field | Type | Meaning |
|---|---|---|
| `pos` | vec3 f32 | world anchor — or, when following, an offset added to the followed position |
| `sizePx` | f32 | em size on screen, in pixels (constant pixel size, like map labels) |
| `color`, `halo` | u32 ×2 | RGBA8, unpremultiplied |
| `haloPx` | f32 | halo width in screen pixels (clamped by the atlas's distance range) |
| `follow` | u32 | index into the follow buffer, or `0xFFFFFFFF` |
| `offsetPx` | vec2 f32 | screen-space offset (nameplate above a head) |
| `flags`, `_pad` | u32 ×2 | bit 0: hidden |

**Atlas glyph table** (32 B per glyph, GPU): `{ uv: vec4 (u0, v0, u1, v1), plane: vec4 (l, b, r, t) em }`.
Advances and kerning stay CPU-side — only layout needs them.

**Follow** — optional: any GPU buffer, a stride in floats and the offset of `xyz` within a record.
For a `VisibilitySystem`: its instance buffer, stride 24, offset 12 (the model matrix's translation).

## 5. Mechanism

**Cull (one thread per glyph).** Read glyph → label → anchor (followed position + offset, or `pos`).
Reject hidden labels, anchors behind the camera (`w ≤ 0`), anchors beyond `maxDistance`. Compute the
glyph's screen rectangle — anchor projected and **snapped to the pixel grid** (so small text does not
shimmer as it moves), plus `offsetPx + (pen + plane) · sizePx` — and reject it if it misses the
viewport. Survivors append their glyph index; `drawArgs[1]` counts them.

**Draw.** Vertex pulling over a 4-vertex quad (`[0,1,2, 2,1,3]`): the same anchor and snap, the
corner from `vertex_index`, the offset applied in NDC scaled by `clip.w` so the size stays constant
in pixels; `z` is the anchor's, so `depthTest` compares the label's anchor against the scene.
Fragment: distance = `median(r,g,b)` (MSDF / MTSDF) or `r` (SDF); `screenPxRange = distanceRange ·
sizePx / emSize` is constant per label, so no derivatives are needed. Fill coverage
`clamp(d·screenPxRange + 0.5)`, halo coverage the same shifted by `haloPx`; output premultiplied.

**Depth test** is a system option: on (labels hide behind closer geometry, compared at the anchor) or
off (always on top). Labels never write depth. Draw the text batch last in its pass.

## 6. Font atlases

| Source | Type | Where it runs | Use |
|---|---|---|---|
| `FontAtlas.fromMSDF(engine, json, image)` | MSDF / MTSDF / SDF | anywhere | msdf-atlas-gen output (`-json`, any `-type`): the standard tool, best quality |
| `FontAtlas.fromCanvas(engine, { fontFamily, chars, … })` | SDF | browser + Worker (`OffscreenCanvas`) | zero setup: any installed or loaded web font, any script the browser draws |
| `FontAtlas.fromSDF(engine, { data, glyphs, … })` | any | anywhere | the low-level constructor both of the above call — and the tests' synthetic fonts |

The canvas path rasterizes each glyph at `emSizePx`, then runs an exact Euclidean distance transform
(Felzenszwalb & Huttenlocher) inside and outside the shape. It is a pure function (`computeSDF`),
pinned headless against brute force.

## 10. Test suite

**Headless (`npm run test:text`):** layout (advances, kerning, letter spacing, alignment, baselines,
wrapping, missing glyphs, capacity); MSDF JSON → table (y origin, UVs); the distance transform
against brute force; label packing; generated WGSL structure; system wiring through a mock device.

**GPU (`test/text-gpu`):** X1 a label lands at its projected anchor at its size (±1 px); X2 constant
pixel size at two depths; X3 GPU cull count = CPU mirror (behind, off-screen, too far, hidden); X4
labels follow a GPU-written buffer with zero CPU writes; X5 depth test on hides, off shows; X6 the
halo rings the fill; X7 edge antialiasing spans ~1 px; X8 600k glyphs in exactly one draw; X9 (Chrome
only) a canvas-generated atlas renders an "O" with a hole.

## 11. Errors

Named at the call site: label/glyph capacity exceeded (with counts); `follow` used without a follow
buffer; non-positive `sizePx`; an atlas JSON missing fields or of an unknown `type`; `fromCanvas`
without `OffscreenCanvas` (the way out: `fromMSDF`); a glyph table larger than the atlas allows.

## 12. Engine touchpoints

None. The module uses public engine surface only (`createBatch`, `createComputeStage`, bind groups,
`sharedSourceBuffer`, `preClearBuffers`).

## 13. Build order

1. `layoutText`, atlas parsing, `computeSDF` + headless tests. 2. `TextSystem` (buffers, cull,
draw) + mock-device wiring tests. 3. GPU suite with synthetic fonts. 4. `fromCanvas` + its browser-only
check. 5. API doc, render-CLI scene, handoff.

**Deferred:** decluttering, rich text, shaping, picking labels, world-size (non-pixel) labels, SDF
effects beyond the halo (glow, shadow), atlas generation in Node.
