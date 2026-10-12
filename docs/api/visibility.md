# Visibility API Usage

The `null-graph/visibility` module is the GPU-driven culling runtime. It decides **on the GPU**
what to draw — frustum, meshlet normal-cone (backface cluster), and Hi-Z occlusion culling, plus
LOD selection — and issues the draws indirectly, so the CPU never iterates the scene. Draw-call
count becomes proportional to material/mesh count, not instance count.

It is a batteries-included wrapper over the engine's existing indirect-draw primitive
(`createBatch({ isIndirect: true })` + `drawIndexedIndirect`), consuming the meshlet buffers
produced by [`MeshletBuilder`](./geometry.md#meshletbuilder-v105).

- **Architecture:** [`visibility-design.md`](../visibility-design.md)
- **Buffer layouts, WGSL contracts, material bridge:** [`visibility-shader-reference.md`](./visibility-shader-reference.md)

```ts
import {
    VisibilitySystem, InstanceBuffer, LOD_AUTO,
    INSTANCE_FLAG_NO_OCCLUSION, INSTANCE_FLAG_NO_CONE_CULL,
} from 'null-graph/visibility';
import { MegabufferBuilder, MeshletBuilder } from 'null-graph/geometry';
```

---

## Do you need this?

| Scene | Use |
|---|---|
| A few dozen objects | **Not this.** Plain `engine.createBatch` is simpler and just as fast. |
| Thousands of instances of a few meshes | **Instance tier** — the big win, and it needs no meshlets. |
| Huge meshes (100k+ tris) mostly off-screen or back-facing | **Meshlet tier** — sub-object culling. |
| Dense occlusion (interiors, cities) | Add **Hi-Z** on either tier. |

Visibility trades setup complexity for scene-size independence. Below a few hundred objects that
trade is not worth it.

## Two tiers

| Tier | Granularity | Draw calls | Needs meshlets? |
|---|---|---|---|
| **Instance** | per-object | one per unique mesh | no — `setGeometry(mega)` alone |
| **Meshlet** | per-cluster (≤64v/≤124t) | one per material | yes — `setGeometry(mega, pack)` |

Both share one mega vertex pool, one Hi-Z pyramid, and the indirect-draw mechanism. Pick per draw
batch; you can mix them in one scene.

---

## 1. Build geometry

`MeshletBuilder` layers on `MegabufferBuilder` — one unified vertex pool backs both tiers.

```ts
const mega = new MegabufferBuilder(8);                  // 8 floats/vertex: pos(3) normal(3) uv(2)
const meshlets = new MeshletBuilder(mega, { positionOffsetFloats: 0, normalOffsetFloats: 3 });

const rock = meshlets.addMesh('rock', { vertices: rockVerts, indices: rockIndices });
const tree = meshlets.addMesh('tree', {
    vertices: treeVerts,
    lods: [                                             // optional discrete LOD chain
        { indices: treeLOD0, error: 0.0  },
        { indices: treeLOD1, error: 0.05 },             // world-space geometric error
        { indices: treeLOD2, error: 0.20 },
    ],
});

const pack = meshlets.build();                          // omit if you only want the instance tier
```

For the instance tier alone, a plain `MegabufferBuilder` (no `MeshletBuilder`) is enough.

## 2. Create the system

```ts
const vis = new VisibilitySystem(engine, {
    maxInstances: 100_000,                  // sizes the instance + work buffers

    // meshlet tier only:
    maxMeshletsPerInstance: 256,            // max meshlets in any one LOD
    maxVisibleMeshletInstances: 2_000_000,  // per-frame visible-cluster budget

    // optional Hi-Z occlusion — needs a SAMPLABLE depth view (not ctx.depthTexture):
    // hiZ: { depthSource: myDepthView },

    lodErrorPx: 1.0,                        // screen-space error budget (default 1.0)
});

vis.setGeometry(mega, pack);                // pack optional → instance tier only
```

| Option | Default | Notes |
|---|---|---|
| `maxInstances` | — | Required, ≥ 1. |
| `maxMeshletsPerInstance` | — | Required for the meshlet tier. |
| `maxVisibleMeshletInstances` | — | Required for the meshlet tier. Too small ⇒ flicker; see the debug table in the shader reference. |
| `hiZ` | off | Omit for frustum + cone culling only. |
| `lodErrorPx` | `1.0` | Larger ⇒ coarser LODs sooner. |
| `mode` | `'single-phase'` | `'two-phase'` is reserved and **throws**. |

## 3. Fill instances (zero-copy, worker-fillable)

Instances are 96 bytes each (24 floats): a column-major model matrix, an object-space bounding
sphere, and the mesh reference. Adopt the array by reference — nothing is copied until `upload()`.

```ts
const instances = new InstanceBuffer(100_000);
instances.write(
    i,
    modelMat4,                  // column-major 16 floats (gl-matrix mat4)
    [cx, cy, cz, radius],       // object-space bounding sphere
    rock.meshIndex,
    LOD_AUTO,                   // or a relative level: 0 = finest
    0,                          // flags
);
```

`flags` bits: `INSTANCE_FLAG_NO_OCCLUSION` (skip Hi-Z for this instance — useful for the object the
camera is inside), `INSTANCE_FLAG_NO_CONE_CULL` (skip backface cluster rejection — for two-sided
foliage).

A Web Worker can fill the raw arrays directly; the exact byte layout is in the
[shader reference §1.1](./visibility-shader-reference.md#11-instance--96-bytes-24-floats).

**The bounding sphere must enclose the mesh.** A too-small radius culls geometry that is actually
visible — the most common source of "things pop out at the screen edge."

### Rows written on the GPU

When a compute stage writes the rows itself (an agent system, a particle simulation), there is no
CPU array to adopt. Set the count instead:

```ts
vis.setInstanceCount(n);   // the culls dispatch over n rows; upload() now writes only cull params
// Rows that never move can share the system: write them once, nothing overwrites them.
device.queue.writeBuffer(vis.instanceBuffer, 0, props.data);
```

`vis.instanceSource` reads `'gpu'` after `setInstanceCount()` and `'cpu'` otherwise. A later
`setInstances(data, count)` adopts an array again, and `upload()` resumes copying it every frame.
Splice the stage that writes the rows **before** `vis.getComputeStages()`, so the culls read this
frame's rows.

## 4. Register draw batches

One per unique mesh (instance tier) or per material (meshlet tier).

```ts
// Instance tier — one draw call for all rocks, regardless of count:
const rockBatch = vis.createInstanceDrawBatch(scenePass, 'rock', rock.meshIndex, {
    vertexLayouts: StandardLayout.getWebGPUDescriptor(),
});

// Meshlet tier — one draw call covering every meshlet of every instance:
const meshletBatch = vis.createMeshletDrawBatch(scenePass, { /* material options */ });
```

Both accept `fragmentBody`, `extraBindings`, `materialEntries`, `materialGroupIndex`, and
`targetFormat`. The default fragment is a neutral hemispheric shade — good for bring-up, not for
real rendering. See §6.

> **`fragmentBody` is a whole entry point, not a body.** Supply the complete function —
> `@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> { … }` — where `VertexOut` carries
> `clipPos`, `worldPos`, `worldNormal`, `uv` and `ecsColor`. A bare `return …;` is a WGSL parse
> error ("statement found outside of function body").

**Which instance am I?** The instance cull copies survivors into a compacted list, and writes each
one's index in your `InstanceBuffer` into the record's last field — `sourceIndex` in WGSL,
`INSTANCE_SOURCE_INDEX_OFFSET` (= 23) on the CPU. A custom instance-tier shader reads
`instances[instance_index].sourceIndex` to look up per-instance data the record does not hold;
picking uses it to report your index. **`vis.instanceBuffer`** is public (read-only): the GPU copy
of your records, for other GPU consumers — e.g. labels that follow instances
(`TextSystem` `follow: { buffer: vis.instanceBuffer, strideFloats: 24, positionOffset: 12 }`).

A batch's parameters are written when it is registered, so a batch created after this frame's
`upload()` still draws on its first frame. To tear one down — a system you destroy while the
`VisibilitySystem` it shares lives on — call `vis.removeDrawBatch(batch)`: its cull stage leaves
the chain, its params buffer is released, and the batch leaves its pass. (A cull stage that outlives
its owner keeps binding the owner's destroyed buffers, and every later frame is refused at submit;
`ShadowSystem.destroy()` does this for its casters.)

## 5. Render

```ts
function frame() {
    // (a worker may have written into instances.data)
    vis.setInstances(instances.data, instances.count);
    vis.upload();

    engine.render([
        lightBinning,                 // if using clustered lights
        ...vis.getComputeStages(),    // Hi-Z build + culls — MUST precede the scene pass
        scenePass,
        ...post.getPassSequence(),
    ]);
}
```

`getComputeStages()` returns a cached array and reads the shared camera uniform, so no per-call
camera is needed. Call `vis.resize(w, h)` alongside `engine.resize`, and `vis.destroy()` on
teardown.

## 6. Materials — the one-liner

Pass a stock material and you are done. Its WGSL (bindings + fragment) is generated and its
textures/uniform are bound automatically — **no shader code to write**:

```ts
import { StandardPBRMaterial } from 'null-graph/materials';

const rockMat = new StandardPBRMaterial(engine, { /* maps, baseColor, … */ });

const batch = vis.createInstanceDrawBatch(scenePass, 'rock', rock.meshIndex, {
    vertexLayouts: StandardLayout.getWebGPUDescriptor(),
    material: { kind: 'pbr', instance: rockMat },
});
```

With clustered lighting — the material's full lighting loop, still one line:

```ts
const batch = vis.createInstanceDrawBatch(scenePass, 'rock', rock.meshIndex, {
    vertexLayouts: StandardLayout.getWebGPUDescriptor(),
    material: { kind: 'pbr', instance: rockMat, lighting: lights.getShaderLayout() },
});
lights.applyToBatch(batch, 2);   // bind the light buffers at group 2
```

`kind` is one of `'basic' | 'lambert' | 'emissive' | 'matcap' | 'toon' | 'pbr'` and must match the
material instance's family. It is explicit rather than sniffed from the class so
`null-graph/visibility` never has to import `null-graph/materials` at runtime.

| Field | Purpose |
|---|---|
| `kind` | which stock family (selects the WGSL) |
| `instance` | the material object; its `applyToBatch` binds the resources |
| `lighting` | `lights.getShaderLayout()` — enables the material's clustered/tiled loop |
| `useHalfLambert` | Lambert only — wrap light around the terminator |
| `useSmoothRamp` | Toon only — soften the cel ramp |

**Groups are handled for you.** Instance tier binds the material at group 1; meshlet tier at
group 2 (group 1 holds the meshlet geometry there), and the generated WGSL uses the matching index.
Override with `materialGroupIndex` — e.g. `3` on the meshlet tier if you also want lighting at 2.

Works identically on the meshlet tier:

```ts
vis.createMeshletDrawBatch(scenePass, { material: { kind: 'lambert', instance: rockMat } });
```

**Custom shaders still win.** Supplying `fragmentBody` overrides `material` entirely, so a
hand-written shader is never second-guessed. See
[shader reference §4](./visibility-shader-reference.md#4-using-a-built-in-material-with-a-culled-batch)
for the manual route and the exact WGSL contract.

Why this works: the visibility `VertexOut` is **field-identical** to the material contract
(`worldPos`/`worldNormal`/`uv`/`ecsColor`), and both modules compose their fragment stage from the
same `materials/materialFragments.ts` — so a material can never disagree with itself. Guarded by
`npm run test:visibility`.

## 7. Hi-Z occlusion

Omit `hiZ` to run frustum + cone culling only. To add occlusion, supply a **samplable** depth
texture (`TEXTURE_BINDING`) that the pyramid is rebuilt from each frame.

- `ctx.depthTexture` is `RENDER_ATTACHMENT`-only and **cannot** be used. Supply a depth pre-pass
  target or the `PostChain` g-buffer depth.
- The pyramid is built from **last frame's** depth (single-phase), so a fast camera cut can briefly
  disocclude geometry for one frame. Frame 0 — and the first frame after a resize — occludes
  nothing: a depth footprint of exactly 0 (a never-rendered, zero-initialized texture) is treated as
  "no history". Pinned on a GPU by `test/visibility-gpu` V3a; before Cycle A it culled *everything*.
- Per-instance opt-out: `INSTANCE_FLAG_NO_OCCLUSION`.

## 8. Post-Processing Presets Integration

`VisibilitySystem` draw passes integrate directly with `PostChain` and stylized post-processing presets (such as `ChainPresets.anime()`).

```ts
import { VisibilitySystem } from 'null-graph/visibility';
import { PostChain, ChainPresets } from 'null-graph/post';

// 1. Setup PostChain with G-Buffer depth enabled for outline/silhouette rendering
const post = new PostChain(engine, {
  format: 'rgba16float',
  gbuffer: { depth: 'depth24plus' },
  outputToScreen: true,
});
post.setChain(ChainPresets.anime());

// 2. Main scene pass consumes post scene attachments
const scenePass = engine.createPass({
  name: 'Visibility Main Pass',
  isMainScreenPass: false,
  ...post.sceneAttachments({ clearValue: { r: 0.01, g: 0.012, b: 0.02, a: 1.0 } }),
});

// 3. Create draw batch matching the PostChain format
vis.createMeshletDrawBatch(scenePass, {
  targetFormat: post.format,
  fragmentBody: `...`,
});

// 4. Return compute stages, main pass, and post chain sequence in update()
return [
  ...vis.getComputeStages(),
  scenePass,
  ...post.getPassSequence(),
];
```

## 9. Errors (fail-loud, at build/compile time)

| Condition | Behavior |
|---|---|
| `mode: 'two-phase'` | throws (reserved for a later build) |
| `maxInstances < 1` | throws |
| `setInstances` count > `maxInstances` | throws |
| `setInstanceCount` count > `maxInstances`, negative or not a whole number | throws |
| instance-tier batch before `setGeometry(mega)` | throws |
| meshlet tier without `setGeometry(mega, pack)` | throws |
| meshlet tier without `maxMeshletsPerInstance` / `maxVisibleMeshletInstances` | throws |
| `getComputeStages()` before any draw batch | throws |

## 10. Notes & limits

- **Two-phase mode is reserved.** Single-phase is the proven baseline (design §13 build order).
- **No CPU readback.** Nothing stalls the pipeline to read a GPU counter; dispatch sizes come from
  CPU-known caps with idle threads early-outing.
- **Meshlet-tier group 2 collision.** The material defaults to group 2 there, which is also
  `lights.applyToBatch(batch, 2)`. Use `materialGroupIndex: 3` if you need both.
- **`maxVisibleMeshletInstances` is not bounds-checked on the GPU.** Under-sizing it overflows the
  compaction buffer. Budget generously.
- **Testing.** The pure cull math (frustum, sphere, cone, Hi-Z level, LOD) and every generated WGSL
  module are covered headless by `npm run test:visibility`. On a real device, `test/visibility-gpu`
  (run by `npm run test:gpu` / `test:gpu:browser`) pins one draw per mesh at any instance count, the
  GPU cull against the CPU mirror, and Hi-Z occlusion; the meshlet-tier scenes are not written yet.
