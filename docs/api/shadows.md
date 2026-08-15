# Shadows API Usage

The `null-graph/shadows` module is the shadow runtime. **Phase 1** is the cascaded shadow map:
one directional light (a sun that occludes), N cascades fitted to the view, depth-only caster
passes, GPU caster culling, and a filtered receiver-side lookup that drops into every existing
material without touching `LightSystem` or the material builders. **Phase 2** turns that into a
*system*: every light type casts, shadow memory is one fixed atlas the engine partitions by
importance, tiles that did not change are not re-rendered, and casters hidden behind other casters
are culled on the GPU.

The load-bearing idea (design §1): **a shadow map is the same draw the main pass already makes,
from a different matrix.** A cascade *is* a `Camera` — an orthographic one — so caster culling is
the existing [`VisibilitySystem`](./visibility.md) cull chain pointed at a different camera
uniform, and the depth pass is the engine's batch primitive with `depthOnly`. Phase 2 generalizes
that `Camera` into a `ShadowView` — a `Camera` plus an atlas rectangle — so a cascade, a spot
cone, and one of six cube faces differ only in how their matrix is built and how big a tile they
get.

- **Phase 1 architecture:** [`shadows-design.md`](../shadows-design.md)
- **Phase 2 architecture (atlas, punctual lights, caching):** [`shadows-atlas-design.md`](../shadows-atlas-design.md)
- **Buffer layouts, WGSL contracts, the material bridge:** [`shadows-shader-reference.md`](./shadows-shader-reference.md)

```ts
import { ShadowSystem } from 'null-graph/shadows';
import { VisibilitySystem, InstanceBuffer } from 'null-graph/visibility';
import { MegabufferBuilder, StandardLayout } from 'null-graph/geometry';
import { LightSystem } from 'null-graph/lights';
```

---

## Do you need this?

| Scene | Use |
|---|---|
| Open world, one dominant sun | **Phase 1** (below) or **Phase 2** with one directional light. |
| Indoor / many local lights | **Phase 2** — spot and point shadows live in the atlas. |
| A mix: a sun plus torches/lamps | **Phase 2** — one atlas, every light type casts. |
| A small diorama, one light | Maybe. A single cascade (`cascades: 1`) is the cheap path. |

Phase 1 is deliberately **one light, N cascades**. Phase 2 is the *system*: many lights, one
atlas, tile caching, occlusion-culled casters. Phase 2 is **additive** — a Phase 1 config (no
`atlasSize`) runs the Phase 1 path unchanged; setting `atlasSize` switches to the atlas.

## What you wire up

```
VisibilitySystem  ──►  the casters (its instances are what occlude)
ShadowSystem      ──►  fits cascades, renders depth, decorates the lighting
LightSystem       ──►  the directional light the shadow matches
materials         ──►  unchanged — they consume incident.attenuation, which the shadow folds into
```

---

## 1. Build geometry + visibility (the casters)

Shadows reuse the visibility instance buffer as the caster set, so build that first (see
[`visibility.md`](./visibility.md)):

```ts
const mega = new MegabufferBuilder(8);
mega.addMesh('rock', rockVerts, rockIndices);
mega.addMesh('ground', groundVerts, groundIndices);

const vis = new VisibilitySystem(engine, { maxInstances: 100_000 });
vis.setGeometry(mega);
```

## 2. Create the shadow system

```ts
const shadows = new ShadowSystem(engine, {
    cascades: 4,
    resolution: 2048,
    shadowDistance: 200,        // NOT camera.far — shadows stop here
    lambda: 0.6,                // split blend: 0 = uniform, 1 = logarithmic
    casterExtrusion: 200,       // how far behind the cascade the near plane reaches
    constantBias: 0.0015,
    normalOffsetTexels: 1.5,
    pcfRadius: 1,               // 1 → 3×3 taps of hardware PCF
    blendTexels: 4.0,           // cascade cross-fade band
    lodBias: 1,                 // casters render one LOD coarser
    visibility: vis,            // required — Phase 1 has no CPU caster path
    shadowGroupIndex: 3,        // bind group for the shadow resources
});
```

| Option | Default | Notes |
|---|---|---|
| `cascades` | `4` | 1–4. `MAX_CASCADES` sizes the `ShadowData` uniform. |
| `resolution` | `2048` | Power of two — texel snapping assumes it. |
| `shadowDistance` | `200` | Bounds the cascade set. Beyond it, lighting is unshadowed. |
| `lambda` | `0.6` | Practical-split blend (§5.1). |
| `casterExtrusion` | `shadowDistance` | §5.4. Supply the scene's extent along the light if you know it. |
| `constantBias` | `0.0015` | NDC-z bias on the caster (§7.1). |
| `normalOffsetTexels` | `1.5` | Caster normal offset in texels (§7.2). §11 throws if it swamps a cascade. |
| `pcfRadius` | `1` | 0 → hardware 2×2 only, 1 → 3×3, 2 → 5×5. |
| `blendTexels` | `4.0` | Cascade cross-fade band, in split-distance units. |
| `lodBias` | `1` | Coarsens caster LOD selection (§5.5). |
| `visibility` | — | **Required.** The `VisibilitySystem` whose instances are the casters. |
| `shadowGroupIndex` | `3` | Must be ≥ 3 (group 2 is `LightSystem`'s). |
| `allowColorFallback` | `false` | §6.3 escape hatch if `depthOnly` is unavailable. ~1.5–2× pass cost. |

## 3. Register casters

One call per caster mesh. Internally this is `cascades` depth-only indirect batches — one per
cascade layer — each pointed at its cascade's camera with Hi-Z off:

```ts
shadows.registerCaster('rock', rockMeshIndex);
shadows.registerCaster('ground', groundMeshIndex);
```

Draw-call count is `cascades × casterMeshes`, **independent of instance count** — the same
property the main pass has. 4 cascades × 6 caster meshes with 200 000 instances is 24 draws.

## 4. Decorate the lighting + bind the receiver

`wrapLighting` returns a layout of the same shape as `LightSystem.getShaderLayout()`, whose
`getIncidentLight` calls the original and folds the shadow term into `attenuation`. Lambert, Toon,
and PBR all consume `incident.attenuation`, so all three gain shadows with zero edits:

```ts
const lights = new LightSystem(engine, { technique: 'standard', capacity: { directional: 1, point: 0, spot: 0 } });
const shadowedLighting = shadows.wrapLighting(lights);

const batch = vis.createInstanceDrawBatch(scenePass, 'rock', rockMeshIndex, {
    vertexLayouts: StandardLayout.getWebGPUDescriptor(),
    material: { kind: 'pbr', instance: rockMat, lighting: shadowedLighting },
});
lights.applyToBatch(batch, 2);        // the light buffers at group 2
shadows.applyToBatch(batch);          // shadowMaps + sampler + ShadowData at group 3
```

Phase 1 shadows **exactly one directional light**, identified by direction match (two directional
lights pointing the same way *are* the same light). Set the sun direction on both systems:

```ts
shadows.setLight([-0.5, -1.0, -0.3]);   // travel direction; normalized internally
// …and the matching directional light's dir_rad in the LightSystem data.
```

## 5. Render

`update(camera)` fits the cascades and writes the buffers; then splice the caster culls and the
depth passes ahead of the scene pass:

```ts
function frame() {
    vis.setInstances(instances.data, instances.count);

    shadows.setLight(sunDir);
    shadows.update(camera);             // splits, sphere fit, snap, write N cascade buffers + ShadowData
    vis.upload();

    engine.render([
        ...vis.getComputeStages(),      // main-view culls AND the per-cascade caster culls
        ...shadows.getComputeStages(),  // empty by design — caster culls ride vis's chain
        ...shadows.getPasses(),         // N depth-only passes, one per cascade layer
        scenePass,                      // materials sample the cascade array
        ...post.getPassSequence(),
    ]);
}
```

`update()` writes `cascades × 432 + cascades × 16 + 352` bytes from persistent typed arrays — no
allocation. It takes only matrices and scalars, so a worker can compute the cascade array and hand
it over with `shadows.setRawCascades(viewF32, shadowDataF32)` (the `LightSystem.setRawData`
contract).

---

# Phase 2 — the atlas, punctual lights, and caching

Set `atlasSize` and the system switches from the per-cascade array to **one atlas texture** that
every light type shares. The unifying move (atlas-design §1): a `ShadowView` — a `Camera` plus a
render target — becomes a `ShadowView` **plus an atlas rectangle**. A cascade, a spot cone, and
one of six cube faces differ only in how their matrix is built and how big a tile they get.

## P2.1 Create the atlas system

```ts
const shadows = new ShadowSystem(engine, {
    atlasSize: 4096,            // the ONE fixed allocation — every light shares it
    maxTileSize: 2048,          // largest tile a single view may request
    maxViews: 64,               // cascades + spots + 6×points
    maxSlots: 64,               // shadow-casting lights
    cascades: 4,                // for any directional light
    shadowDistance: 200,
    constantBias: 0.0015,
    pcfRadius: 1,
    updateMode: 'round-robin',  // 'every-frame' | 'on-demand' | 'round-robin'
    refreshPerFrame: 2,         // clean views refreshed per frame in round-robin
    lightViewOcclusion: true,   // Hi-Z on the light view (atlas-design §7.2)
    visibility: vis,
    shadowGroupIndex: 3,
});
```

| Phase 2 option | Default | Notes |
|---|---|---|
| `atlasSize` | — | **Setting it selects the atlas path.** Power of two. 4096² ≈ 32 MB. |
| `maxTileSize` | `2048` | Top of the tile ladder (`2048, 1024, 512, 256, 128`). |
| `maxViews` | `64` | Hard cap on views. Exceeding it throws the §11 error naming the light. |
| `maxSlots` | `64` | Hard cap on shadow-casting lights. |
| `updateMode` | `'every-frame'` | §8.1. `'every-frame'` is always correct; `'on-demand'` renders only dirty views; `'round-robin'` also refreshes `refreshPerFrame` clean views/frame. |
| `refreshPerFrame` | `2` | Bounds the damage from a missed `invalidate()` to a few frames. |
| `cubeFovMarginTexels` | `pcfRadius` | The cube-face fov margin. §11 throws when it's smaller than `pcfRadius` (face seams). |
| `lightViewOcclusion` | `false` | §7.2 — Hi-Z occlusion-cull casters on the light view, reusing `HiZPyramid` over last frame's atlas. |

## P2.2 Register lights

`registerLight` returns the **shadow slot**; write `slot + 1` into the light record's
`spot_params.z` (float 14 of the 16-float record) so the receiver can find the light's views. `0`
in that field means "does not cast" (atlas-design §4.3):

```ts
const sunSlot  = shadows.registerLight({ kind: 'directional', name: 'sun' });
const lampSlot = shadows.registerLight({ kind: 'point', name: 'lamp', position: [2, 3, 1], range: 30 });
const spotSlot = shadows.registerLight({ kind: 'spot',  name: 'spot', position: [0, 5, 0],
                                         direction: [0, -1, 0], range: 20, outerCos: Math.cos(0.6) });

lightData[sunLightIndex  * 16 + 14] = sunSlot + 1;   // spot_params.z
lightData[lampLightIndex * 16 + 14] = lampSlot + 1;
lightData[spotLightIndex * 16 + 14] = spotSlot + 1;
```

A directional light owns `cascades` consecutive views, a spot one, a point exactly six in
`+X,−X,+Y,−Y,+Z,−Z` order (so face selection is arithmetic). The Phase 1 direction match is
**deleted** in the atlas path — the slot is explicit, so two identical directional lights no
longer collide.

## P2.3 Register casters + render

Same calls as Phase 1; the atlas path puts every view's draws into **one** pass:

```ts
shadows.registerCaster('rock', rockMeshIndex);

function frame() {
    vis.setInstances(instances.data, instances.count);
    shadows.setLight(sunDir);           // the directional light's travel direction
    shadows.update(camera);             // builds every view, allocates/evicts tiles, writes tables
    vis.upload();

    engine.render([
        ...vis.getComputeStages(),      // main-view culls AND the per-view caster culls
        ...shadows.getComputeStages(),  // the light-view Hi-Z build, when lightViewOcclusion is on
        shadows.getAtlasPass(),         // ONE depth-only pass, one viewport-scoped batch per (view × mesh)
        scenePass,
        ...post.getPassSequence(),
    ]);
}
```

Draw-call count is `views × casterMeshes` in one pass, independent of instance count — and the
caching below removes the views whose tiles did not change.

## P2.4 Caching — not re-rendering what did not change

NullGraph has no scene graph, so it cannot know a caster moved — **dirtiness is declared, not
inferred** (atlas-design §8.1). The default `updateMode: 'every-frame'` re-renders everything
(always correct). To cache:

```ts
const shadows = new ShadowSystem(engine, { atlasSize: 4096, updateMode: 'on-demand', /* … */ });

// When a caster that light `lampSlot` sees moves:
shadows.invalidate(lampSlot);         // that light's views re-render this frame
```

A view is re-rendered when its matrix changed (the light moved), its tile was resized/evicted/
newly allocated, the caller called `invalidate(slot)`, or `updateMode` is `'every-frame'`.
`'round-robin'` additionally refreshes `refreshPerFrame` clean views per frame — the pragmatic
setting for a world where some casters are "mostly static."

## P2.5 Eviction and the budget

The atlas is allocated once; when demand exceeds it, the system **evicts the least-important
resident views** (never cascades — the sun is not allowed to lose its shadow because forty torches
got close to the camera). An evicted view's light samples as **fully lit** — never black — and the
eviction is reported through `shadows.shadowAtlas.report()`, never silent. Exceeding `maxViews`
is a hard error naming the light.

---

## Errors (fail-loud, at build/compile time)

| Condition | Behavior |
|---|---|
| `cascades` outside 1–4 | throws — `MAX_CASCADES` sizes the `ShadowData` uniform |
| `resolution` not a power of two | throws — texel snapping assumes it |
| `shadowGroupIndex` ≤ 2 | throws — collides with `LightSystem`'s `@group(2)` |
| `registerCaster` without `visibility` | throws — Phase 1 has no CPU caster path |
| `getPasses()` before `update(camera)` | throws — cascade matrices unset |
| `wrapLighting` on a technique with no layout | throws — needs `standard` or `clustered-forward` |
| `getIncidentLight` marker moved in `LightSystem` | throws — the decorator is out of date |
| normal offset > 2% of a cascade's extent | throws — contact shadows would detach |
| **Phase 2:** more views than `maxViews` | throws — names the light; raise `maxViews` or stop shadowing it |
| **Phase 2:** a single view larger than the atlas | throws — raise `atlasSize` or lower `maxTileSize` |
| **Phase 2:** atlas full but fragmented | throws — `freePx`² free across `k` regions; lower `maxTileSize` or raise `atlasSize` |
| **Phase 2:** shadow slot > 254 | throws — `spot_params.z` encodes `slot+1` as a float |
| **Phase 2:** `cubeFovMarginTexels` < `pcfRadius` | throws — face seams will appear |
| **Phase 2:** `invalidate(n)` on an unregistered slot | throws — `registerLight()` returns the valid range |
| **Phase 2:** `registerLight` without `atlasSize` | throws — punctual lights need the atlas path |

## Notes & limits

- **Phase 1: one directional light.** Spot/point shadows need the Phase 2 atlas (`atlasSize`).
- **Phase 1: no Hi-Z on the light view.** Caster culling is frustum-only (Phase 2 adds occlusion
  via `lightViewOcclusion`).
- **Fixed-radius PCF only.** PCSS / VSM / ESM / MSM are out of scope for both phases (the atlas
  makes them *possible*; picking one is a separate quality decision).
- **All bias lives on the caster.** `PipelineConfig` exposes no `depthBias`/`cullMode`, and the
  receiver has no normal in scope (§8.2) — so the constant and normal offsets are applied in the
  depth-only vertex shader. This is why Phase 1 needs no rasterizer-state change.
- **Conservative always beats clever.** A caster wrongly culled is a missing shadow in the *lit*
  image. Every cascade fit errs toward including geometry (§5.3 sphere fit, §5.4 extrusion), and
  the atlas-wide Hi-Z is a *max* reduction, so neighbour bleed causes under-culling, never
  over-culling (atlas-design §7.2).
- **The atlas needs per-batch viewport support** (atlas-design §12.1) — `PipelineConfig.viewport`,
  six lines in `RenderPipeline`, inert for every existing batch. The zero-change alternative (a
  clip-space tile remap) **bleeds** into neighbouring tiles and is not the baseline (§6.2).
- **Testing.** The cascade math, the atlas allocator, the view builders, the cube face select, the
  tap clamp, the slot packing, and every generated WGSL module are covered headless by
  `npm run test:shadows`. The GPU integration scenes (design §10 T1–T7, atlas-design §10 T1–T6)
  live in `test/shadows-gpu/` and need a browser/WebGPU harness — serve the repo root and open
  `/test/shadows-gpu/`.
