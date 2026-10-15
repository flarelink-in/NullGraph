# NullGraph Agents — skinned models for GPU agents (D1, phase 5) — Technical Design

**Parent:** [`agents-animated-models-design.md`](./agents-animated-models-design.md) §6 ("Toward
skinned models") and §10, row 5. Cycle D, step D1, phase 5 (the user's "10.5").

**Asked by the user** (2026-10-15, after phase 4 was committed): "you can start with 10.5".

**Status:** decided with the user, 2026-10-15 (§9: the four recommendations). Phase 5.0 next.

---

## 1. Purpose

**Thousands of GPU agents drawn as skinned low-poly models — the user's Blender animals — each
animated from its own simulation state with no CPU work per agent per frame.** The same as the
rigid path of phases 1–3, with smooth bending at the joints instead of rigid parts.

### The single most important pre-existing facts

1. **The rigid path works end to end.** `bakeRigidAnimation` writes one part per vertex
   (`partOfVertex`) and a 3×4 transform per part per frame; the demo's hook fetches one part from
   two clips, two frames each (four 3×4 fetches per vertex). Phase 0: the hook costs 1.2–1.5× the
   rest pose; welded cows draw 4K / 16K / 64K in 3.7 / 8.1 / 28.5 ms on the AMD iGPU.
2. **A glTF skin** gives each vertex up to four joints and weights (`JOINTS_0`, `WEIGHTS_0`; more in
   `JOINTS_1`), and each joint an inverse bind matrix. A joint's skinning matrix is its world
   transform times its inverse bind matrix; the skinned mesh's own node transform is ignored.
3. **`GLBParser` already reads skins, partly.** A skinned mesh keeps its vertices in mesh space,
   16 floats each (position, normal, uv, four joints as floats, four weights). Only the first skin is
   read. Weights stored as normalized integers (allowed by glTF; not Blender's or Mixamo's default)
   come out as raw integers, which is wrong today.
4. **`Animator` and `SkeletonManager`** evaluate one skin on the CPU into a uniform of at most 70
   bones: one character at a time, the opposite of a crowd.
5. **The vertex hook reads per-vertex data by `vertex_index`** from storage buffers (phase 2), so
   joints and weights need no new vertex attributes: `StandardLayout` and the megabuffer stay as they
   are.
6. **Test models available now** (the site's `public/`, never copied into this repository): two
   Mixamo dancers (56 joints, 17,754 triangles, one clip, float weights, byte joints) and a butterfly
   (2 joints, STEP keys). **No rigged animal exists yet;** the user will make them in Blender.

### Non-goals

- More than four influences per vertex (`JOINTS_1`): refused by name.
- Dual-quaternion skinning, morph targets, IK, root motion.
- Changing `Animator` or `SkeletonManager`.
- A skinned animal in the demo before the user's model exists.

## 2. Architecture

```
GLB (skinned) ──GLBParser.parse──► mesh (mesh space, joints, weights) + skin + nodes + clips
               └─ bakeSkinnedAnimation(glb, { fps, clips }) ──► mesh (bind pose), 4 joints + 4 weights
                                                               per vertex, a joint-matrix table, clips
the app's vertex hook ── BAKED_POSE_WGSL (sample a clip, blend 4 joints) ──► posed vertex
```

## 3. The bake (loaders)

- **`bakeSkinnedAnimation(glb, options)`**, the same options as the rigid bake (`fps`, `clips` with
  `loop`, `weld`).
- **Influences:** every skin joint becomes one, its matrix per frame `W_t(joint) · IBM(joint)`.
  Unskinned meshes in the same file (a hat parented to a head bone, say) become one influence each
  with weight 1, `W_t(node) · W_rest(node)⁻¹` as in the rigid bake, so one call takes the whole model.
- **Per vertex:** four influence indices and four weights, normalized to sum 1; unused slots weight 0.
- **Returns** `{ vertices, indices, influencesOfVertex: Uint32Array (4 per vertex),
  weightsOfVertex: Float32Array (4 per vertex), influences, influenceNodes, transforms (3×4 per
  influence per frame), frames, clips, bounds }`: the rigid result's shape, with four influences
  where it had one part. Bounds cover every vertex in every frame, blended as the shader blends.
- **Welding** merges vertices that share position, uv, influences and weights.
- **Refused by name:** two skins, `JOINTS_1`, weights that are all zero, an influence count the
  table cannot index.
- **Checked on the CPU** against `Animator`'s matrices (an independent path already in the library)
  and against direct glTF evaluation written in the tests, at frames and between them.

## 4. Sampling in WGSL (loaders, a helper)

`BAKED_POSE_WGSL`: the shader half of the bake, so an app's hook stops repeating the demo's 25
lines. It declares nothing; it expects storage arrays named in its header (the table, the
influences and weights) and gives:

- `fn bakedClipPose(clip: vec2<u32>, phase: f32, influence: u32) -> mat3x4` — one influence at a
  phase of a clip (`firstFrame`, `frames`), its two nearest frames blended;
- `fn bakedVertexPose(clip, phase, vi) -> mat3x4` — the four influences of vertex `vi`, weighted.

The demo's hook then chooses clips and phases from the agent's state and mixes two poses. The
rigid result can be sampled by the same helper with one influence and weight 1.

## 5. The demo

Nothing changes until the user's skinned animals exist. Then Models mode loads them in place of
the Kenney animals, and its hook calls `bakedVertexPose`. The far cubes, fences and field stay.

## 6. Costs to measure first (phase 5.0)

A skinned stand-in built from the welded cow in code: its parts become joints, and vertices near a
part boundary take two influences. Drawn as the demo's agents at 4K–64K, rigid against skinned
(four influences, f32 weights) against skinned with packed `unorm8` weights, in Chrome on the AMD
iGPU and under Node on NVIDIA, as phase 0 measured. Decides the weights' precision and whether the
rigid path keeps its own one-fetch hook.

## 7. Validation

- **Headless:** the bake against `Animator` and against direct glTF evaluation on an in-memory
  skinned fixture (U-tests), and on the Mixamo dancers and the butterfly when present; the
  helper's WGSL parses; every refusal fires.
- **GPU:** a fixture posed by the hook and the helper equals the CPU-posed mesh, pixel for pixel
  within a tolerance, as V5 does for rigid parts.
- **Size:** the loaders entry grows by the bake and the helper; set from the build and reported.

## 8. Build order

| Phase | Builds | Checked by |
|---|---|---|
| 5.0 | Spike: the cost of four influences against one, at 4K–64K | Costs, recorded here |
| 5.1 | `bakeSkinnedAnimation`; `GLBParser` reads normalized weights; tests; guide | Headless tests |
| 5.2 | `BAKED_POSE_WGSL`; a GPU pose test; guide | Headless + GPU test |
| 5.3 | The user's animals in Models mode (when they exist) | Drives, screenshots |

## 9. Decisions (the user, 2026-10-15)

1. **Two formats:** a new `bakeSkinnedAnimation`; `bakeRigidAnimation` and its one-fetch hook stay.
2. **`BAKED_POSE_WGSL`, the sampling helper, is public API in loaders.**
3. **Test models:** the Mixamo dancers, the butterfly and in-memory fixtures; the demo keeps the
   Kenney animals until the user's Blender animals exist.
4. **Measure first:** phase 5.0, the skinned stand-in made from the cow.

### The questions as asked

1. **One format or two?** `bakeSkinnedAnimation` returns four influences per vertex; unskinned
   meshes in the file become weight-1 influences. `bakeRigidAnimation` stays as it is, with its
   one-fetch hook. *Recommended.* The alternative is one `bakeAnimation` for both, rigid models
   paying four weights per vertex and a branch in the shader.
2. **A WGSL sampling helper as public API** (`BAKED_POSE_WGSL`, §4)? *Recommended:* every app with
   baked animation needs the same sampling, and the demo's hook shrinks.
3. **Test models:** the Mixamo dancers, the butterfly and in-memory fixtures now, and the demo
   unchanged until your animals exist. *Recommended,* unless a rigged animal is ready.
4. **Measure first (5.0)?** A skinned stand-in made from the cow, before the format is fixed.
   *Recommended:* it decides the weights' precision and whether rigid keeps its own hook.

## Phase 5.0 results — 2026-10-15: what skinning costs

**Where:** `test/spikes/skinned-models/` (`npm run spike:skinned-models`; in Chrome, `npm run drive --
--serve . "test/spikes/skinned-models/index.html" --wait-for "skinned-models done" --timeout 900`).
The demo's cow and fox, welded and baked by the library's `bakeRigidAnimation`, scaled as the demo
scales them; rows written once as phase 0 wrote them (one in 16 a fox, random headings, the whole
field in view); one frame R = 3 times between two timestamps, culls included. A skinned stand-in is
made from each model: every vertex lists four influences (its part, the body, two more), and the
top 15 cm of each leg blends into the body (32 of the cow's 418 vertices, 34 of the fox's 423).
Measured beside a second idea the design had not planned: a **palette**, a compute pass that blends
each agent's part matrices once per frame, so that the hook reads matrices instead of sampling clips.

| Mode | What a vertex does |
|---|---|
| rigid | the demo's hook: one part, two clips × two frames sampled (12 vec4 reads) |
| skin4 | four influences, each sampled as rigid samples one (48 vec4 reads) |
| skin4b | skin4, influences of weight 0 skipped |
| skin4u8 | skin4 with the weights packed as four 8-bit unorms |
| pal1 | the palette, one matrix read (rigid) |
| pal4 | the palette, four weighted matrices (skinned) |

**Draw cost, ms per frame** (palette pass, culls and draw; 1280 × 720; median of 24 frames):

| Agents | Host | rigid | skin4 | skin4b | skin4u8 | pal1 | pal4 |
|---|---|---|---|---|---|---|---|
| 4,096 | Node · NVIDIA | 2.25 | 5.05 | 2.47 | 5.05 | 2.03 | 2.18 |
| 16,384 | Node · NVIDIA | 8.74 | 19.88 | 9.52 | 19.86 | 7.80 | 8.52 |
| 65,536 | Node · NVIDIA | 34.36 | 78.67 | 37.55 | 78.69 | 30.61 | 33.69 |
| 4,096 | Node · AMD iGPU | 5.94 | 16.82 | 8.45 | 16.93 | 4.57 | 4.59 |
| 16,384 | Node · AMD iGPU | 22.39 | 25.45 | 13.37 | 25.67 | 7.06 | 7.25 |
| 4,096 | Chrome · AMD iGPU | 5.73 | 15.78 | 8.56 | 15.92 | 5.19 | 5.20 |
| 16,384 | Chrome · AMD iGPU | 11.34 | 25.71 | 14.56 | 22.76 | 8.37 | 7.84 |

At 65,536 on the AMD iGPU the spike's own guard skipped the count (skin4's cost predicted over 400 ms
a render). The 16K rigid figure under Node on AMD reads high against Chrome's and phase 0's; the
iGPU's clocks move between runs.

**Checked by image** (close-ups of every mode, 921,600 pixels): pal1 differs from rigid in 2 pixels,
so the palette poses exactly as the hook does; skin4, skin4b and pal4 differ from rigid in the same
1,235 pixels, the blended leg tops, so the three skinned paths agree; skin4u8 in 1,153.

**What it shows:**
1. **Sampling four influences per vertex costs 2.2–2.8× the rigid hook**, everywhere.
2. **Packing the weights into 8 bits saves nothing:** the cost is the pose reads, not the weights.
3. **Skipping zero weights brings skinning near rigid** (1.1× on NVIDIA, 1.3–1.4× on AMD) for models
   whose vertices mostly have one influence; a model weighted smoothly everywhere would not gain.
4. **The palette makes skinning cost what rigid costs today, and rigid cheaper:** pal4 is 0.69–0.98×
   the rigid hook, pal1 0.73–0.90× (0.32× in the noisy Node/AMD 16K run). A model's vertices share
   their agent's matrices; the palette pass computes each once per agent instead of once per vertex.
5. **The palette's price is memory:** agents × influences × 48 B. The cow's six parts at 64K agents
   is 18.9 MB; a Blender animal of 20 bones at 256K agents would be 252 MB, past the 128 MB a storage
   binding allows by default.

**What it changes in the plan (questions for the user, below):** the palette replaces per-vertex
sampling, for skinned models and, as the numbers allow, for rigid ones. Its memory is bounded by
computing it only for the agents drawn as models (the near LOD band), over the model batch's visible
list, instead of for every agent.

### Questions after 5.0

1. **Adopt the palette** (a compute pass per frame writes each drawn agent's influence matrices; the
   hook reads one or four)? *Recommended:* skinned at rigid's cost, rigid cheaper.
2. **One palette slot per visible model, not per agent?** The pass runs over the model batch's
   compacted visible list (its storage buffer and its count, which an app can bind already) and the
   hook reads the slot by `instance_index`. Memory follows what is on screen, which the LOD band
   bounds. *Recommended* over one slot per agent (simple, but past the binding limit at large counts).
3. **Move the demo's rigid Models onto the palette now** (faster), or keep its hook until the user's
   skinned animals arrive?
4. **The library's share:** WGSL helpers (sample a clip into the palette; read one or four matrices in
   the hook) as decided in §9, or also a small class that allocates the palette and builds its pass
   for a batch?
