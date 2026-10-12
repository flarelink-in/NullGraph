# Picking API Usage

`null-graph/picking` answers "what is under this pixel?" — which source, which instance, its depth
and its world position — for GPU-culled `VisibilitySystem` instances and plain batches. Design:
[`picking-design.md`](../picking-design.md).

A pick is **its own small render, off the frame path**: a camera zoomed onto the clicked pixels,
culled and drawn through the paths your scene already uses, then a few texels read back. A picker
that is not picking costs nothing.

## 1. Set up

```ts
import { Picker } from 'null-graph/picking';

const picker = new Picker(engine, { maxRadius: 4 });   // largest radius a pick may ask for (≤ 7)

// GPU-culled instances: a hit's `index` is the instance's index in YOUR InstanceBuffer.
const rocks = picker.addInstances(vis, 'rock', rockMeshIndex, { label: 'rocks' });

// A plain batch on the stock 14-float layout (every stock material): `index` = batch instance index.
const props = picker.addBatch(propBatch, { label: 'props', vertexLayouts: geom.layout.getWebGPUDescriptor() });
```

Nothing changes in your frame — keep rendering `[...vis.getComputeStages(), scenePass, …]`. The pick
batches' cull stages ride the visibility chain **disabled**, and the pick pass is never in your
sequence.

## 2. Pick

```ts
canvas.addEventListener('pointerdown', async (e) => {
    const dpr = canvas.width / canvas.clientWidth;               // render-target pixels, not CSS px
    const hit = await picker.pick(e.offsetX * dpr, e.offsetY * dpr, camera, { radius: 3 });
    if (!hit) return;
    hit.source;      // rocks | props — the handle add*() returned
    hit.index;       // your instance index
    hit.meshIndex;   // instance sources
    hit.position;    // world-space hit point, [x, y, z]
    hit.depth;       // clip depth in [0, 1]
    hit.pixel;       // the pixel that was hit (≠ the click when radius > 0)
});
```

- `camera` must be the camera the frame was drawn with; its state is captured at the call.
- `radius` accepts a hit up to that many pixels away — the nearest wins (thin lines, small points).
- Picks are serialized and each resolves after one GPU round trip (~0.5 ms on a desktop GPU).

## 3. Custom vertex layouts

For a plain batch whose shader reads a different instance layout, supply the pick vertex stage:

```ts
import { buildPickFragmentWGSL } from 'null-graph/picking';

picker.addBatch(myBatch, {
    shaderCode: (tag) => `
        ${MY_CAMERA_AND_BINDINGS}                     // group 0: camera (binding 0) + your storage (binding 1)
        ${buildPickFragmentWGSL(tag)}                  // declares PickVaryings and fs_main
        @vertex fn vs_main(@location(0) pos: vec3<f32>, @builtin(instance_index) i: u32) -> PickVaryings {
            var v: PickVaryings;
            v.position = camera.viewProj * myTransform(pos, i);
            v.index = i;       // what the hit reports
            v.extra = 0u;
            return v;
        }`,
});
```

## 4. Notes

- **Instance records carry their source index.** The instance cull writes each survivor's index
  in your buffer into the record's last field (`sourceIndex`, `INSTANCE_SOURCE_INDEX_OFFSET`), so
  any custom instance-tier shader can ask "which of the app's instances am I?" — picking is one use.
- **Not pickable yet:** the meshlet tier, skinned meshes (a pick without the skeleton picks the bind
  pose), text labels.
- `engine.render()` with **no** sequence renders every pass the engine tracks, the pick pass
  included; its batches sit at zero instances between picks, so it costs a clear of a 9×9 target.
- `picker.destroy()` unregisters every source from its VisibilitySystem. Picks still queued when it runs
  resolve `null`, and so does a pick whose readback was in flight; neither touches the released
  resources.

## 5. Errors

| Condition | Message |
|---|---|
| `pick()` with no sources | `Picker: nothing is pickable — add sources with addInstances() or addBatch().` |
| pixel outside the target | `Picker: pixel (x, y) is outside the W×H target — pass render-target pixels (CSS px × devicePixelRatio).` |
| `radius > maxRadius` | `Picker: radius r exceeds maxRadius R set at construction …` |
| indirect plain batch | `Picker: batch "…" is indirect — … pick it through its VisibilitySystem (addInstances).` |
| system from another engine | `Picker: that VisibilitySystem belongs to a different engine.` |

Executable reference: `test/picking-gpu` (P1–P8, every expectation checked against a CPU ray cast).
