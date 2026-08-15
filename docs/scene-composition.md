# Composing a Scene with NullGraph — Guide

**Status:** guide (not an implementation spec). Reflects the shipped API plus, where marked, the
v1.0.5 designs ([`visibility-design.md`](./visibility-design.md),
[`geometry-meshlets-design.md`](./geometry-meshlets-design.md)).
**Audience:** someone with an arbitrary scene "X" asking *how do I actually put this on screen,
and how much of it is boilerplate I can default?*

---

## 1. The one honest claim

> **The *shape* of composing a scene is generic. The *content* is not — and by design, cannot be a
> preset.**

Every NullGraph scene — a single product shot, a 100 000-instance world, a stylized short — is
assembled the *same way*: you build an ordered array of GPU stages and hand it to `render()`. That
structure is 100% generic; §2 is the same for every scene you will ever make.

What is **not** generic, and where you should not expect a preset to exist: the scene's *content* —
which meshes, where, what material, which lights. NullGraph is deliberately **zero scene graph**
(`roadmap.md` §0). There is no `Scene` object, no node tree, no traversal, no "load a glTF and it
just renders." You own your data and its layout. That is the cost of the zero-copy, worker-drivable,
no-GC design — and it is why "compose any scene" is a *recipe*, not a *function*.

**Where presets genuinely live and should be leaned on:** the post-processing **look**
(`ChainPresets`, shipped, §5) and effect **order** (`orderChain`, shipped). Those are real, they
default well, and §5 tells you which to pick. There is no "scene preset," and per the engine's
philosophy there should not be one — but the parts that *can* be defaulted, are.

If that trade is not what you want, §9 sketches an optional convenience layer — clearly marked as a
**proposal, not shipped** — that would encode the generic backbone so you write less array-plumbing.

---

## 2. The invariant: one frame is one ordered sequence

`engine.render(sequence)` executes a linear array of `ComputeStageNode | RenderPassNode`
(`NullGraph.ts:192`). **Composing a scene is composing that array.** Its canonical order is generic:

```
render([
   ── compute that produces data a later pass reads ──
   lightBinning,          // clustered/tiled LightSystem.getComputeStage()   (if many lights)
   ...visibility,         // Hi-Z build + instance/meshlet cull              (v1.0.5, for scale)

   ── render passes, offscreen before on-screen ──
   scenePass,             // your geometry draws (or the indirect cull output)
   ...post,               // PostChain.getPassSequence()  — ends on the main-screen pass
]);
```

Three ordering rules, and they hold for *every* scene:

1. **Compute before the pass that consumes it.** Light binning and visibility culling write buffers
   the scene pass reads, so they precede it (`RenderPipeline.ts:65-70` gives the barrier for free).
2. **Offscreen passes before the main-screen pass.** The main-screen pass substitutes the swapchain
   (`RenderPipeline.ts:72-82`); anything feeding it (scene → post pyramid) comes first.
3. **Post is last.** `PostChain` owns the linear→display encode on its final pass
   (`post-processing-design.md` §10).

Only the *contents* of each bracket change between scenes. That is the whole generic backbone.

---

## 3. The recipe, built up from minimal

### 3.0 The absolute minimum (one object) — this already works today

Straight from [`quick-start.md`](./quick-start.md), condensed to the skeleton:

```ts
const engine = new NullGraph();
await engine.init(canvas);
const camera = new Camera(75, canvas.width / canvas.height, 0.1, 1000);

const pass = engine.createPass({ name: 'Main', isMainScreenPass: true });

const geom = Primitives.createCube(StandardLayout, 2, 2, 2);
geom.upload(engine);

const material = new StandardPBRMaterial(engine, {           // fallback textures = sane default
  albedoMap: engine.textureManager.fallbackWhite,
  normalMap: engine.textureManager.fallbackNormal,
  packedMap: engine.textureManager.fallbackWhite, packedMapFormat: 'ARM',
  baseColor: [0.1, 0.5, 0.9, 1.0],
});
const batch = engine.createBatch(pass, {
  shaderCode: buildPBRShader({ useSkinning: false }),
  strideFloats: 14, maxInstances: 1,
  vertexLayouts: geom.layout.getWebGPUDescriptor(),
});
material.applyToBatch(batch);
engine.setBatchGeometry(batch, geom.vertexBuffer, geom.indexBuffer, geom.indices.length);
engine.updateBatchData(batch, instanceFloats /* 14 */, 1);

function frame() {
  camera.updateView(eye, target); engine.updateCamera(camera);
  engine.render([pass]);                      // ← the sequence, here just one pass
  requestAnimationFrame(frame);
}
```

Everything below **adds brackets to that `render([...])` array.** Nothing about the skeleton
changes.

### 3.1 Add lighting

```ts
const lights = new LightSystem(engine, {
  technique: 'clustered-forward',                 // 'standard' for a handful of lights
  capacity: { point: 1024, directional: 4, spot: 128 },
  clusterConfig: { grid: [16, 9, 24], maxLightsPerTile: 128 },
});
// materials read the light layout the system emits:
const shader = buildPBRShader({ useSkinning: false, lighting: lights.getShaderLayout() });
// bind the light buffers to the batch (group 2):
lights.applyToBatch(batch, 2);

// per frame: a worker (or you) fills a Float32Array; adopt by reference, upload once
lights.setRawData(lightArray); lights.upload();

// clustered/tiled techniques contribute a binning compute stage BEFORE the scene pass:
const binning = lights.getComputeStage(camera);
engine.render([binning, pass]);
```

`technique: 'standard'` needs no compute stage (it loops all lights in the shader) — pick it for
small scenes; pick `clustered-forward` when light count is large (`LightSystem.ts:381`).

### 3.2 Add post-processing — where presets do the work

```ts
import { PostChain, ChainPresets } from 'null-graph/post';

const post = new PostChain(engine, { format: 'rgba16float', gbuffer: { depth: 'depth24plus' }, outputToScreen: true });
const scenePass = engine.createPass({ name: 'Scene', isMainScreenPass: false, ...post.sceneAttachments() });

post.setChain(ChainPresets.cinematic());          // ← a shipped preset. Full look, one line.

const sequence = [binning, scenePass, ...post.getPassSequence()];
function frame() {
  /* update lights, camera */
  post.upload();
  engine.render(sequence);
}
```

Note the scene pass flips to `isMainScreenPass: false` — the *post* chain now owns the final
on-screen pass. This is the composition changing shape, but only by rebracketing the same array.

### 3.3 Add scale — visibility & meshlets (v1.0.5, designed not yet shipped)

For thousands+ of objects, replace hand-managed batches with GPU culling
([`visibility-design.md`](./visibility-design.md)). The array gains a cull bracket; the draw
becomes indirect:

```ts
// build unified geometry once (§ geometry-meshlets-design.md §2.1):
const mega = new MegabufferBuilder(8);  /* addMesh × N */
const meshlets = new MeshletBuilder(mega, { positionOffsetFloats: 0, normalOffsetFloats: 3 });
/* meshlets.addMesh × N */; const pack = meshlets.build();

const vis = new VisibilitySystem(engine, { maxInstances: 100_000, /* ... */ });
vis.setGeometry(pack);
const worldBatch = vis.createDrawBatch(scenePass, { shaderCode: shader, /* material bindings */ });

function frame() {
  vis.setInstances(instanceArray);                 // zero-copy, worker-filled
  vis.upload();
  engine.render([ binning, ...vis.getComputeStages(camera), scenePass, ...post.getPassSequence() ]);
}
```

Same backbone, one more compute bracket. Below the scale threshold, skip this entirely — plain
batches (§3.0) are correct and simpler.

---

## 4. Generic vs per-scene, at a glance

| Concern | Generic (reuse verbatim) | Per-scene (you provide) |
|---|---|---|
| Frame structure | the `render([...])` order (§2) | — |
| Init / camera / resize | `init`, `Camera`, `resize` | fov, near/far |
| Geometry *loading* | `Primitives`, `GLBParser`, `Megabuffer/Meshlet` builders | the actual meshes, transforms |
| Instance data | the upload mechanism (`updateBatchData` / `setInstances`) | the values (positions, colors) |
| Lighting *machinery* | `LightSystem`, binning stage | light positions/colors/count |
| Material *machinery* | `buildPBRShader`, fallback textures | textures, PBR params |
| Post **look** | **`ChainPresets` (§5) — default heavily** | optional per-effect tweaks |
| World transforms | — | **you compute them** (no scene graph, §8) |

---

## 5. Presets and sane defaults — the honest inventory

What ships that you *can* lean on, and the recommended default for each decision:

**Post look — `null-graph/post` (rich, shipped).** Full looks in one call:

| Preset | Use |
|---|---|
| `ChainPresets.neutral` / `faithful` | correct HDR→display / authored colors survive (dataviz, UI) — **safe default** |
| `ChainPresets.product` / `archviz` | contact shadows + subtle bloom, hue-faithful tonemap |
| `ChainPresets.cinematic` | balanced SSAO + wide bloom + filmic — **good default for "looks good"** |
| `ChainPresets.anime` / `comic` | cel shading + ink outlines |
| `ChainPresets.analog` / `noir` / `synthwave` | film / mono / neon stylization |

Plus building blocks: `TonemapPresets`, `BloomPresets`, `SSAOPresets`, `forSceneScale(preset,
unitsPerMeter)` to rescale SSAO to non-metric scenes, and `orderChain(effects)` which sorts any
effect set into the canonical `EFFECT_ORDER` (`chainOrder.ts`) — so you can throw effects in any
order and get the correct pipeline. **Lean on these heavily; they are the one part that defaults
well.**

**Decisions with a good default (not "presets", but defaultable):**

| Decision | Default | Switch when |
|---|---|---|
| Light technique | `standard` | → `clustered-forward` past ~a few dozen lights |
| Visibility tier | none (plain batches) | → instance tier past ~thousands of objects; → meshlet tier for large per-mesh triangle counts |
| Camera controller | `OrbitalCameraState` | Fly / Follow / Path per interaction model |
| Material textures | `textureManager.fallback*` | real textures when authored |

---

## 6. Three archetypes, same backbone

Only the fill differs; the `render([...])` order (§2) is identical.

**A — Hero / product object.** One (or few) objects, maximum fidelity.
`render([ scenePass, ...post ])` with `ChainPresets.product`, `technique:'standard'`, no
visibility. This is §3.0 + §3.2.

**B — Massive instanced world.** Foliage/rocks/debris, tens of thousands of instances of few
meshes. `render([ binning, ...vis.getComputeStages, scenePass, ...post ])` with
`clustered-forward` lights and the **instance tier** (few meshes ⇒ few draws, `visibility` §7.1),
`ChainPresets.cinematic`. This is §3.1 + §3.2 + §3.3.

**C — Stylized / NPR short.** Moderate geometry, look is the point. §3.0 backbone +
`ChainPresets.anime` or `comic` (both want `gbuffer.depth` for silhouettes). Presets carry ~all of
the distinctiveness.

---

## 7. Decision guide

| If your scene has… | reach for… |
|---|---|
| a few objects, static lights | §3.0 + `ChainPresets`, `technique:'standard'` |
| many dynamic lights | `clustered-forward` + `getComputeStage` (§3.1) |
| thousands+ of objects | instance-tier `VisibilitySystem` (§3.3) |
| huge meshes (100k+ tris each), lots off-screen/back-facing | meshlet tier + `MeshletBuilder` |
| a specific film/anime/product look | the matching `ChainPresets.*` |
| worker-driven simulation | fill the `Float32Array`s in the worker; main thread only uploads |

---

## 8. Honest limits — the non-generic parts

Do not expect these to be handled for you:

- **No transform hierarchy.** There is no parenting/scene graph. Parent-child world matrices are
  *your* computation (do it on the CPU or in a worker, write final matrices into instance data).
  This is the deliberate cost of zero-copy DOD.
- **No automatic resource lifetime.** You `destroy()` what you create. Recompiling a `PostChain`
  currently leaks pass nodes (`post-processing-design.md` §3.2) — an acknowledged rough edge.
- **Visibility & meshlets are v1.0.5 — designed, not shipped.** Until then, cull/batch on the CPU.
  §3.3 is forward-looking; §3.0–3.2 work today.
- **Instance stride is material-defined.** The 14-float PBR default (position, quaternion, scale,
  color — see quick-start) is a convention of `buildPBRShader`, not a law. Custom materials define
  their own stride, and your instance `Float32Array` must match it.
- **Not yet in the engine:** shadows (v1.1), deferred/visibility-buffer shading (v1.1), TAA
  (v2.0). Composition today is forward-rendered. See `roadmap.md`.

None of these break the §2 backbone — they are content/feature gaps, not structural ones. That is
exactly why the claim in §1 holds: the shape generalizes; the content is yours.

---

## 9. Proposed (NOT shipped): a sequence-assembly helper

This section is a **design proposal**, offered because you asked whether composition could be more
turnkey. It does not exist; decide whether it is worth building.

The one piece of genuine boilerplate above is hand-ordering the `render([...])` array (§2) and
keeping the compute-before-pass / offscreen-before-onscreen rules straight. A thin, optional builder
could encode *only that ordering* without reintroducing a scene graph:

```ts
// PROPOSED — not implemented.
const frame = new FrameSequence(engine)
  .lights(lights)         // inserts binning compute if the technique needs it
  .visibility(vis)        // inserts Hi-Z + cull compute, before the scene pass
  .scene(scenePass)
  .post(post);            // appends getPassSequence(), asserts it ends on the main-screen pass

engine.render(frame.build());   // returns the same flat array you'd hand-write
```

It would **only** assemble and validate the canonical order (§2) — no retained state, no traversal,
no per-frame allocation, `build()` returns the identical flat array. That keeps it inside the
engine's philosophy (`roadmap.md` §0: "produce nodes to splice into the array; do not introduce a
retained scene"). The trade-off is one more abstraction to learn versus writing the array
explicitly, which the shipped examples favor. **Flagged, not assumed** — say the word if you want it
promoted to a real spec.
