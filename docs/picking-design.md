# Picking — Design (Cycle B)

**Module:** `null-graph/picking` · **Status:** ✍ → building · **Companion:** [`text-labels-design.md`](./text-labels-design.md)

## 1. Purpose

Answer "what is under this pixel?" for GPU-driven scenes: the user clicks, NullGraph returns **which
source and which instance** was hit, its depth and its world position. Instances are the app's own
indices — the same index it wrote into its `InstanceBuffer` or batch storage — so the app maps a hit
to its entity with no lookup table on the engine side.

**The single most important pre-existing fact:** a per-batch camera override already exists, and its
doc comment names this exact use — *"the enabling change for every future off-main-view render:
shadows, reflection probes, IBL capture, editor picking passes"* (`src/core/types.ts:46-48`). A pick is
"one more view": a camera whose frustum covers only the clicked pixels, culled and drawn by the
machinery shadows already use.

**Non-goals (v1):**
- Meshlet-tier picking (its draw path pulls `{instanceId, meshletId}` pairs; v2).
- Skinned meshes (a pick shader without the skeleton picks the bind pose; documented, not refused —
  a batch does not record that it is skinned).
- Picking text labels (v2, once labels exist and a use case asks).
- Rectangle / lasso selection, and picking every frame (hover) — both are "a bigger pick region",
  and the region size is a constant here; revisit with a use case.
- CPU ray casts against bounds. Cheap, but blind to mesh shape and occlusion; apps can do it themselves
  from their own arrays.

**Principles:** zero cost when not picking (no pass, no dispatch); no readback in the frame loop — a
pick is its own submission, off the frame path; baseline core WebGPU only; fail loud at build time.

## 2. Architecture

A pick does not join the app's frame. It is a separate, immediate `engine.render()` of a tiny pass:

```
picker.pick(x, y, camera)
  ├─ write the PICK CAMERA (main camera with its projection zoomed onto the (2R+1)² pixels around x,y)
  ├─ enable the pick sources' cull stages ─┐
  ├─ engine.render([...pickCulls, pickPass]) │  (instance sources: frustum-culled against the pick
  ├─ disable them again ─────────────────────┘   camera — a few pixels wide, so almost nothing survives)
  ├─ copyTextureToBuffer(pick target → staging), submit
  └─ await mapAsync → nearest hit within `radius` of the centre → PickHit | null
```

```
         ┌──────────── app frame (unchanged) ────────────┐
         │ [...vis.getComputeStages(), scenePass, ...post] │   pick culls ride the chain DISABLED
         └─────────────────────────────────────────────────┘
pick() ─▶ [pickCull_A, pickCull_B, pickPass] ─▶ copy ─▶ map ─▶ hit
           rgba32uint (2R+1)² target: (sourceTag, index, depthBits, meshIndex)
```

## 3. Integration facts

| Fact | Where |
|---|---|
| A batch can render through any camera buffer (group 0 binding 0) | `PipelineConfig.cameraBuffer`, `src/core/types.ts:46-48` |
| Instance-tier draws accept a camera, an occlusion switch, a custom shader and a colour format | `VisibilitySystem.createInstanceDrawBatch`, `VisibilitySystem.ts:213-229` |
| A batch's cull stage can be gated without rebuilding the chain | `getCullStage(batch)` (`VisibilitySystem.ts:399`) + `ComputeStageNode.enabled` (`types.ts:257`) |
| Disabled stages cost nothing (no clears, no dispatch) | `RenderPipeline.render`, the `if (!node.enabled) continue` branch |
| `engine.render()` encodes and submits on its own — callable outside the app's frame | `RenderPipeline.ts:14`, `:171` |
| **The instance cull copies whole records and drops the source index** | `visibilityShaders.ts:291` — `outInstances[slot] = inst` |
| The `Instance` record's last field is unused padding | `INSTANCE_STRUCT_WGSL` (`_pad`), `cullMath.ts:32` `INSTANCE_PAD_OFFSET = 23` |
| Stock plain batches read 14 floats per instance: `[_, px,py,pz, qx,qy,qz,qw, sx,sy,sz, r,g,b]` | `materials/CommonChunks.ts:78` |
| Camera uniform: viewProj 0, view 16, proj 32, invViewProj 48, invView 64, invProj 80, eye 96, near/far/fov/aspect 100–103 | `src/core/Camera.ts` |
| `rgba32uint` is a core render-attachment format; integer targets cannot blend (none needed) | WebGPU core |

## 4. Data model

**Pick target** — `rgba32uint`, `(2R+1)²` texels, `R = maxRadius` (default 4 → 9×9, 1.3 KB), plus a
`depth24plus` attachment of the same size. Per texel:

| Channel | Meaning |
|---|---|
| `r` | source tag — 1-based index of the registered source; `0` = nothing |
| `g` | instance index — the `InstanceBuffer` index (instance sources) or the batch instance index (batch sources) |
| `b` | `bitcast<u32>(fragCoord.z)` — clip depth in [0, 1] |
| `a` | `meshIndex` (instance sources), `0` otherwise |

**Pick camera** — 108 floats, the main camera's layout with four matrices replaced. With `s = W/(2R+1)`,
`t = H/(2R+1)` and the clicked pixel's centre in NDC `(cx, cy)`:

```
M = | s  0  0  -s·cx |       proj'        = M · proj         invProj'        = invProj · M⁻¹
    | 0  t  0  -t·cy |       viewProj'    = M · viewProj     invViewProj'    = invViewProj · M⁻¹
    | 0  0  1    0   |       (view, invView, eye, near/far/fov/aspect unchanged — depth is untouched,
    | 0  0  0    1   |        so z-test and the returned depth match the main view)
```

**Staging** — one `MAP_READ` buffer, `256 × (2R+1)` bytes (rows padded to 256), reused by every pick.

## 5. The one engine touchpoint: instance records carry their source index

The instance cull writes survivors into a compacted list, so a draw shader sees "visible instance
#k", never "instance #i of the app's buffer". Picking needs `i`. So does any custom shader that wants
per-instance data the record does not hold.

**Change:** in the compacted copy, the cull writes the source index into the record's padding field,
renamed `_pad` → `sourceIndex`. The source buffer's field stays 0 (the app never writes it).
`INSTANCE_SOURCE_INDEX_OFFSET` (= 23) is exported beside the old `INSTANCE_PAD_OFFSET`.

**Zero-change fallback, and why it is worse:** a pick-only cull variant that writes indices into a
separate buffer — a fork of the cull kernel (design-principles §4) that every future cull change must
be mirrored into, for one u32 the record already has room for.

## 6. Sources

- **`addInstances(vis, meshName, meshIndex)`** — a pick batch inside the picker's pass, created through
  `vis.createInstanceDrawBatch` with `camera: pickCamera`, `occlusion: false` (the Hi-Z pyramid is the
  main view's), the pick shader and `targetFormat: 'rgba32uint'`. Its cull stage is the visibility
  system's own, disabled except during a pick.
- **`addBatch(batch)`** — a pick batch sharing the source's geometry **and its instance storage
  buffer** (group 0 is rebuilt to bind the source's buffer — no second upload). Default shader: the
  stock 14-float layout. Custom layouts pass `shaderCode: (tag) => wgsl`. Its instance count is copied
  from the source at pick time. Indirect batches are refused (their count lives on the GPU).

## 7. Resolving a hit

Search texels within `radius` (Euclidean, ≤ R) of the centre, inside the screen; nearest non-empty
texel wins, ties broken by smaller depth. Position: unproject the hit pixel's centre and depth through
the MAIN camera's `invViewProj`. Picks are serialized (one staging buffer); a second `pick()` waits for
the first.

## 10. Test suite

**Headless (`npm run test:picking`):** pick-matrix maths (the clicked pixel maps to NDC (0,0); a pixel
R+1 away maps outside [-1,1]; depth row untouched); world-position round trip; hit selection
(nearest, tie by depth, radius, screen clipping); every refusal in §11; generated WGSL structure.

**GPU (`test/picking-gpu`, run by `npm run test:gpu`):**
P1 a pick at an instance's projected centre returns that instance's `InstanceBuffer` index, across many
instances; P2 an empty pixel returns `null`; P3 overlapping instances return the front one; P4 the
returned world position matches the analytic surface; P5 a stock-material plain batch is pickable;
P6 radius: a thin object is missed at radius 0 and hit at radius 3; P7 the app's frame is unchanged
(same draw count, pick culls disabled); P8 concurrent picks all resolve correctly.

## 11. Errors

| Condition | Message (abridged) |
|---|---|
| `pick()` with no sources | `Picker: nothing is pickable — add sources with addInstances() or addBatch().` |
| pixel outside the render target | `Picker: pixel (x, y) is outside the W×H target — pass render-target pixels (CSS px × devicePixelRatio).` |
| `radius > maxRadius` | `Picker: radius r exceeds maxRadius R set at construction.` |
| `addBatch` on an indirect batch | `Picker: batch "…" is indirect — its instance count lives on the GPU; pick it through its VisibilitySystem.` |
| `addInstances` with a system on another engine | `Picker: that VisibilitySystem belongs to a different engine.` |

## 12. Engine touchpoints

| Touchpoint | Required? | Fallback | Cost of fallback |
|---|---|---|---|
| `sourceIndex` in compacted instance records (§5) | **required** | fork the cull kernel | two kernels to keep in sync |
| new subpath `null-graph/picking` | required | — | — |
| core | **none** | — | — |

## 13. Build order

1. `pickMath.ts` (pure) + headless tests. 2. The §5 touchpoint, visibility tests green. 3. Shaders +
`Picker` + headless wiring tests through a mock device. 4. GPU suite. 5. API doc, handoff.

**Deferred:** meshlet tier, skinned meshes, labels, hover/rect selection, GPU-side "pick → highlight"
without readback.
