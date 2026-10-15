# NullGraph Agents — animated models for GPU agents ("draw as models") — Technical Design

**Parent:** [`work/plan.md`](./work/plan.md), Cycle D ("animated agents"), step 1, decided by the
user on 2026-10-15 (§11). It builds on C1 step 4
([`agents-gpu-loop-design.md`](./agents-gpu-loop-design.md): agents written into
`VisibilitySystem` rows) and on the Herd & Wolves demo of step 10
([`agents-live-switch-design.md`](./agents-live-switch-design.md)).

**Asked by the user** (2026-10-15): "lets start with the 4th one" — the plumbing for animated
models: a vertex hook in visibility and a bake step in loaders. Then, in the demo, "one more draw
option next to draw as darts", and the grid ground replaced by "an artsy toon shaded green field",
first with the fox and pig models of the site's `3dAssetsNotIncluded`. The user will model the
real animals in Blender later; this lays the plumbing. Phase 2's creatures will need the same
rendering backbone.

**Status:** decided with the user, 2026-10-15 (§11): Cycle D, step 1. Phase 0 measured (results
at the end); phase 1 committed (`f538ffe`), phase 2 built and validated, uncommitted. Phase 3 next.

---

## 1. Purpose

**Thousands of GPU agents, each drawn as an animated low-poly model whose animation follows its
own simulation state, with no CPU work per agent per frame.** A wolf that runs plays its run
cycle at its own phase; one resting after a catch plays its eat cycle; slow prey walk.

Delivered in three layers:
1. **Library, `null-graph/loaders`:** bake a GLB's animation clips into one small table of part
   (or bone) transforms per frame.
2. **Library, `null-graph/visibility`:** a vertex hook on instance draw batches, so an app can
   move each vertex before the transform, reading its own buffers.
3. **Demo:** "Draw agents as: Models" next to darts and arrows; a toon-shaded field instead of
   the grid; the post chain's `anime` preset as an option.

### The single most important pre-existing facts

1. **The test models are not skinned.** Each animal is 5–6 rigid parts (body, head or tail,
   four legs) moved by node transforms. Fox: 568 triangles, 6 parts, 8 clips (`static`, `idle`
   1 s, `walk` 0.5 s, `run` 0.25 s, `eat` 0.5 s, `dance`, two gestures), one `colormap` texture.
   Pig 424, deer 760, dog 490, cow 578 triangles, the same clips.
2. **A rigid part is a bone with weight 1.** So one baked table of transforms per part per frame
   serves these animals, and the user's skinned Blender models later (bones, up to 4 weights).
3. **The visible rows already say which agent they are.** The cull copies each surviving row
   with `sourceIndex` = its original row, and for agents row = `firstInstance + agent index`. A
   vertex hook can therefore read the agent's own record: speed, age, resting timer, species.
4. **Instance batches have a fixed vertex stage** (`buildInstanceDrawWGSL`): position, normal and
   uv through the instance's model matrix. Only the fragment is pluggable (`fragmentBody`,
   `material`, `extraBindings`), or the whole shader (`shaderCode`).
5. **`GLBParser` flattens a GLB:** every mesh is baked into model space at its rest transform,
   and which node a mesh came from is lost. Animation channels are kept (`GLBAnimation`, node
   targets). `Animator` evaluates skins only.
6. **The toon look already exists:** `ToonMaterial` on culled batches (`material: { kind:
   'toon' }`), and `ChainPresets.anime()` in `null-graph/post` (bloom, filmic tonemap, 5-band cel
   shading, ink outline from depth, FXAA). The Axion Engine demo has neither: static GLBs, one
   shared `colormap.png`, simple lighting.
7. **LOD chains are index lists over one mesh's vertices** (`lods: [{ indices, error }]`); the
   repository has no mesh simplifier, so lower LODs come from the modeller.

### Non-goals

- Skinned GLBs in the first build (§6 keeps the format ready for them; phase 5 builds them).
- Blending more than two clips; inverse kinematics; root motion (the simulation moves agents).
- A mesh simplifier. Impostors.
- The video (the user's question 3, answered separately).
- Changing the darts or arrows modes.

## 2. Architecture

```
GLB (fox) ──GLBParser.load──► meshes + node of each + clips
                └─ bakeRigidAnimation(glb, { fps: 30, clips }) ──► one mesh (part per vertex),
                                                                     a transform table, a clip table
VisibilitySystem.createInstanceDrawBatch(pass, 'fox', mesh, {
    vertexHook: { wgsl, bindings } })      ◄── the app's WGSL: pick clip/phase from the agent record
       vs_main: deformed = hook(pos, normal, vertexIndex, instance) → model matrix → camera
agents.toInstances(vis)  (unchanged)       rows → cull → visible list with sourceIndex
```

## 3. The bake (loaders)

- **`GLBParser` records each mesh's node** (a new optional field, `node`), and keeps the node
  hierarchy it already reads. Nothing else in its output changes.
- **`bakeRigidAnimation(glb, options)`** (new, in `null-graph/loaders`):
  - merges the meshes into one, and writes a **part index per vertex** (the mesh's node);
  - samples each asked clip at `fps` (default 30) over its length: for each frame and part, the
    matrix `W_t(part) · W_rest(part)⁻¹` (the part's world transform at time t, over its rest
    transform, since the vertices are already at rest in model space);
  - returns `{ vertices, indices, partOfVertex: Uint32Array, transforms: Float32Array (3×4 per
    part per frame), clips: [{ name, firstFrame, frames, seconds, loop }] }`.
- **Size:** fox, 6 parts × 4 clips (idle, walk, run, eat) at 30 fps = 6 × 68 frames × 48 B =
  20 KB. Uploaded once, shared by every agent of that model.
- **Checked on the CPU** against direct glTF evaluation at frame times and between frames.

## 4. The vertex hook (visibility)

A new option on `createInstanceDrawBatch` (instance tier only):

```ts
vertexHook?: {
    /** fn deform(pos: vec3<f32>, normal: vec3<f32>, vi: u32, inst: Instance) -> Deformed */
    wgsl: string;
    /** WGSL declarations of its bindings, at the group named by `group` (default 2). */
    bindings: string;
    entries: GPUBindGroupEntry[];
    group?: number;
};
```

- `vs_main` calls `deform` before the model matrix when the hook is given; without it, the
  generated WGSL is byte for byte today's (a test).
- **The part index needs no new vertex attribute:** `vi` (`@builtin(vertex_index)`) is the
  absolute index into the mega vertex buffer, including the batch's base vertex, so the hook
  reads `partOf[vi]` from a storage buffer. The mega layout, `StandardLayout`, stays as it is.
- **Picking and culling stay on the rest pose:** the bounding sphere must cover every pose; the
  bake reports the largest radius over all frames.
- **Shadows:** a cascade's batch of the same mesh needs the same hook, or shadows show the rest
  pose. Noted, not built first.

## 5. Animation from the simulation (the demo's hook)

The demo's WGSL, not the library's:
- **Clip:** from the agent's speed over its species' top speed: below 0.15 idle, then walk, then
  run above 0.7, **cross-faded** between walk and run over that range (the two poses' transforms
  blended), so a speed change never pops. A resting wolf (`state.w > 0`) plays **eat**.
- **Phase:** each clip at its own rate from the agent's age, plus a per-agent offset (a hash of
  its index), so a herd never steps in unison. A clip that starts again restarts from the age:
  the cross-fade hides the jump between walk and run; idle↔walk is a soft pop at worst (§11,
  question 4).
- **Cost:** two transform fetches per vertex at most (two clips), one record read per vertex
  (cached across a model's vertices by the GPU, not by us).

## 6. Toward skinned models (the user's Blender animals)

The same table holds bone matrices; the hook does four weighted fetches instead of one. Needs
`JOINTS_0`/`WEIGHTS_0` per vertex — as attributes, or, like the part index, in a storage buffer
read by `vi`. `bakeSkinnedAnimation` reuses `Animator`'s sampling. Phase 5, once the user has a
skinned model to test with.

## 7. The demo

- **Options:** "Draw agents as": Darts | Models | Arrows. Models: wolves as the fox, prey as the
  cow (until the user's own animals), subject to this GPU's measured draw safety guard,
  without a machine-specific count cap, with the
  reason on screen. Hover, click and the inspector work as with darts (rest-pose picking).
- **The field:** in Models mode, a ground plane with a toon-shaded grass colour (a gentle noise
  of two greens and soft bands), and grass patches (`patch-grass-foliage.glb`) scattered as
  static visibility rows, culled and LOD-free. The grid overlay toggle stays.
- **Look:** a toggle for `ChainPresets.anime()`; models lit with the toon ramp and the GLB's
  `colormap.png`.
- **The overlay** gains the draw cost of the models (a timing group around the draw is not
  possible inside the frame's render pass; the page widget's GPU time shows the frame).

## 8. Costs to measure first (phase 0)

Draw of 4K / 16K / 64K foxes with the hook, against darts, in Chrome on the AMD iGPU and under
Node on NVIDIA; the bake's time; memory. Darts today: one row and one 96-byte visible-list row
per agent per batch; models add nothing per agent beyond that.

## 9. Validation

- **Headless:** the bake against glTF evaluation (U-tests); the hook's WGSL parses, and no-hook
  WGSL is unchanged; `GLBParser`'s other output unchanged on the fixture GLBs.
- **GPU:** a fox mesh baked and drawn with the hook at a fixed frame equals the CPU-posed mesh
  (rendered and compared, or read back through a compute copy).
- **Site:** drives at each count, screenshots read by eye; the GPU-reset guard as everywhere.

## 10. Build order

| Phase | Builds | Checked by |
|---|---|---|
| 0 | Spike: fox baked by a script, drawn with a hacked shader at 4K–64K | Costs (§8), recorded here |
| 1 ✅ | `GLBParser` node per mesh; `bakeRigidAnimation`; tests; guide | Headless tests; results below (2026-10-15, uncommitted) |
| 2 ✅ | The vertex hook; tests; guide | Headless + GPU test; results below (2026-10-15, uncommitted) |
| 3 ✅ | The demo: Models mode, cow prey and fox wolves, clips from simulation state | Site typecheck, preparation/shader check and Chrome/AMD drives, 2026-10-15 |
| 4 | The field, grass, toon, `anime` preset toggle; docs page | Drives, screenshots |
| 5 | Skinned models (when the user has one) | Tests, drive |

## 11. Decisions (the user, 2026-10-15)

1. **A new cycle, D: animated agents,** with Phase 2 after it.
2. **Public library API:** `vertexHook` in visibility, `bakeRigidAnimation` in loaders.
3. **Stand-ins:** a herd animal for the prey (deer or cow) rather than the pig; the fox for the
   wolves. Phase 0 compares deer and cow.
4. **Phase continuity: the smooth way,** one accumulated phase per agent (a float the simulation
   advances), not phases from age.
5. **The field:** explained to the user, not answered yet. Default until then: Models mode only;
   darts and arrows keep the grid floor, whose lines match the simulation's 8 m cells.

### The questions as asked

1. **Where it lives in the plan:** a new cycle (D: "animated agents"), with Phase 2 after it?
2. **Library API:** add `vertexHook` to visibility and `bakeRigidAnimation` to loaders (public
   API; both entries grow by an estimated 0.5–1 KB min+gzip)? The alternative is demo-only code
   with a full custom `shaderCode`, which Phase 2 would then have to redo.
3. **Stand-ins:** fox for the wolves and pig for the prey (the user named them); deer or cow
   look closer to a herd.
4. **Phase continuity:** age-based phases (simple, a possible soft pop idle↔walk), or an
   accumulated phase per agent written by an app step stage (smooth, one float per agent)?
5. **The field:** only in Models mode (darts and arrows keep the grid), or in every mode?

## Phase 0 results — 2026-10-15: what animated models cost

**Where:** `test/spikes/animated-models/` (`npm run spike:animated-models`; in Chrome, `npm run
drive -- --serve . "/test/spikes/animated-models/index.html" --wait-for "animated-models done"`).
`bake.mjs` bakes a GLB's rigid clips into part transforms (§3) and can weld its vertices; `run.js`
writes the demo's rows once (1 agent in 16 a fox, the rest deer or cow, random headings, on the
demo's field) and draws one frame R = 3 times between two timestamps, culls included, with the
camera over the whole field as the demo opens: every agent in view, the worst case. The hook is
prototyped as a whole vertex stage (`shaderCode`) over visibility's exported WGSL. The models are
Kenney assets, copied from the test site into `models/`: 3D models are never committed
(`.gitignore` ignores `*.glb`, `*.gltf`, `*.fbx`, `*.obj`, `*.blend`, the user's rule).

**The bake:** fox, deer, cow each in 6–23 ms; 4 clips (idle 30, walk 15, run 8, eat 15 frames at
30 fps); tables of 16–20 KB per model. The `static` clip's first frame equals the rest pose within
2.4e-7. Read by eye: posed and rest-pose close-ups differ (heads, legs), and no part comes apart.

**Draw cost, ms per frame** (culls + draw, 1280 × 720; median of 24 frames):

| Agents | Host | Darts | Deer, rest pose | Deer, posed | Cow, posed | Deer, posed, welded | Cow, posed, welded |
|---|---|---|---|---|---|---|---|
| 4,096 | Chrome · AMD iGPU | 1.6 | 10.8 | 13.3 | 7.2 | 6.6 | 3.7 |
| 16,384 | Chrome · AMD iGPU | 1.3 | 15.8 | 20.1 | 12.2 | 11.0 | 8.1 |
| 65,536 | Chrome · AMD iGPU | 3.6 | 59.9 | 80.3 | 44.3 | 39.2 | 28.5 |
| 4,096 | Node · NVIDIA | 0.4 | 2.7 | 4.1 | 2.6 | 2.5 | 1.9 |
| 16,384 | Node · NVIDIA | 1.0 | 10.4 | 15.8 | 10.1 | 9.5 | 7.3 |
| 65,536 | Node · NVIDIA | 3.2 | 41.0 | 62.7 | 39.7 | 37.3 | 28.6 |

Node on the AMD iGPU read within 10% of Chrome on it. (Node/Dawn took the NVIDIA GPU when no
adapter was named, on this day.)

**What it shows:**
1. **The approach works:** part by vertex index, the agent by `sourceIndex`, two clips blended by
   speed, two frames each — the hook costs 1.2–1.5× the rest pose.
2. **The draw is vertex-bound.** The models split every vertex per face for flat normals (deer:
   1,442 vertices for 760 triangles). **Welded** by position, uv and part (deer 524, fox 423, cow
   418), with each face's normal found per pixel from the derivatives of its world position, the
   look is the same (read by eye) and the frame costs 0.45–0.60× as much.
3. **The cow is the cheaper herd animal:** welded, 0.55–0.77× the deer.
4. **Budget on the AMD iGPU, welded cows:** 3.7 ms at 4K and 8.1 ms at 16K, beside a step of about
   4 and 8 ms (step 10's measurements): 60 fps at 4K, about 50–60 at 16K. At 64K the step's
   neighbour queries (60 ms) already dominate.
5. **Far agents are the next lever:** from the demo's opening view most agents are a few pixels.
   A lower LOD per model (the modeller's; the repository has no simplifier) would cut the vertex
   work there.

**What phase 0 changes in the plan:**
- **§3, the bake welds** vertices by position, uv and part by default, and **the toon fragment
  finds face normals per pixel** (an option, so smooth-shaded models keep theirs). Stock
  `ToonMaterial` reads `in.worldNormal`; Models mode uses its own fragment, or the material gains a
  faceted switch (phase 4 decides).
- **§7, no fixed `HERD_MODEL_MAX`** (the user's rule for site demos: no cap tuned to this machine's
  GPU). Models are offered wherever darts are; the GPU-reset guard predicts the draw from the
  frames measured, as the steps are predicted.
- **The prey: the cow,** by cost and by look (decision 3 asked phase 0 to compare).

**Not run:** shadows, post-processing, picking with posed models, the colormap texture (flat
colours here), LOD.

## Phase 1 results — 2026-10-15: rigid animation bake

**Built by Opus, completed and validated by Cline; committed as `f538ffe`.** Parser node metadata and
`bakeRigidAnimation` are exported from loaders. The bake samples LINEAR, STEP and CUBICSPLINE,
supports looping and endpoint-inclusive one-shots, welds exact position/uv/part matches by
default, and reports bounds over sampled poses and convex blends (not unsampled glTF motion).
`MegabufferBuilder.addMesh` accepts Uint32 indices (type-only); MeshletBuilder's narrowing cast
is removed. The loader guide's actual preparation example type-checks, bundles and runs on an
in-memory GLB. No model files were added.

**Validation:** development build including TypeScript, IDE build and diff whitespace checks
pass. `npm run test:loaders` passes **8/8** with external corpus options enabled. U37–U40 use
an in-memory fixture and independent CPU evaluator. The optional animal test reads ignored
local models. The compatibility test compares **193 GLBs** recursively under the site's
`public/` with Opus's saved pre-change loaders build: all previous fields and typed-array values
agree. Blob URLs are normalized by creation order; layout callbacks are compared by their
returned descriptors rather than function identity.

Fox/deer/cow frame poses agree with glTF within **1.6e-7**, static/rest within **2.4e-7**.
Welding yields **423 / 524 / 418** vertices. Four-clip tables are **19,584 / 16,320 / 19,584 B**.
Local bakes took **7–37 ms**, varying with test concurrency. At 30 fps, blending between samples
deviates up to **36.7 mm** for run, **4.8 mm** for walk. Higher fps trades memory for accuracy.

All existing headless scripts pass: core, profiler, cameras, agent-kit, post, geometry,
visibility, shadows, picking, text, agents (83 Node tests), inference (24), environment.
Node/Dawn GPU regressions pass: visibility **6/6**, core **8/8**, agents **137/137**, zero GPU
errors. Visibility/core used NVIDIA (core also tests adoption on AMD); agents used AMD.

**Size:** esbuild browser ESM, minified, `gl-matrix` external, default gzip: parser-only
**3,207 B**, bake-only **3,261 B**, full entry **6,463 B** min+gzip. The saved old full entry
rebundled identically is **4,196 B**: **+2,267 B**, above the 0.5–1 KB estimate. Regression
ceilings are **3,300 / 3,400 / 6,700 B** respectively; they record measured cost, not a
user-approved revision of the estimate. Parser-only imports carry no bake; bake-only imports
carry no parser; an unused entry leaves no code (all tested).

**Known limitations, unchanged:** parser rest normals use the node matrix, not inverse
transpose (wrong under uneven scale, including the cow's head). Material colour factors and
vertex colours are not preserved. The bake combines meshes into one material draw; morph
weights are ignored and skins refused. Welded flat-shaded models need face-normal reconstruction
in the fragment shader. These constraints are explicit in the guide.

**Reproduce corpus comparison:** set `NG_GLB_COMPAT_DIR` to an external GLB directory and
`NG_GLB_OLD_LOADER` to a previous ESM loaders build, then run `npm run test:loaders`. Without
both, that check explicitly skips; without local animal models their check also skips. The old
build used here is in local Claude temp session `20c36f59-ab5e-4974-8dfb-d4d710b7ad75`,
`scratchpad/old/loaders.js`.

**Not run:** browser GPU regressions, posed-model GPU parity, demo drives, animated shadows or
post-processing. **Next: phase 2**, the public vertex hook, its GPU pose test and guide.

## Phase 2 results — 2026-10-15: public vertex hook

**Built and validated, uncommitted.** `InstanceDrawOptions.vertexHook` uses the §4 contract,
exported as `InstanceVertexHook`. Visibility provides `Deformed { position, normal }`, invokes
`deform(pos, normal, vi, inst)` in model space before the instance transform, then applies the
instance's inverse-transpose normal matrix. No vertex attributes, row layouts, cull kernels or
per-frame CPU work were added. Without a hook the default and custom-attribute/fragment WGSL
are byte-identical to D1.1 (SHA-256 snapshots captured before editing).

**Binding composition:** a dedicated group defaults to 2; the caller's WGSL declares the same
literal group. Rejects group 0, fractional/out-of-device groups, numeric material/lighting/extra
binding group collisions, mismatched hook group declarations and `shaderCode` plus hook,
before creating a batch. Unoccupied intervening groups are bound empty (needed by auto-layout
pipelines). Hook buffers remain caller-owned. Stock material at 1 plus clustered lighting at 2
can use a hook at 3; group attributes are not rewritten. This is instance-tier only. Animated
shadows and posed picking are not automatically wired.

**Headless:** the two hook tests were observed failing before implementation, then green.
`test/visibility-hook.test.mjs` now has **5/5** tests: no-hook identity, WGSL parse/transform order,
binding/refusal checks, all six stock material families, and the guide's actual example compiled
against declarations, bundled and executed on a stand-in engine. Existing visibility, loaders,
shadows, picking and agents suites pass; loaders' external compatibility test explicitly skips
without options (the phase 1 run checked all 193 files).

**Real GPU:** V5 in the visibility suite builds a two-part GLB entirely in memory, bakes it,
and renders two source rows at different frames through the public hook. Its reference applies
the fixture's glTF rotations/translations directly on the CPU, without reading the bake's
transforms. Padding geometry ensures a nonzero baseVertex; sparse source rows (1 and 3) catch
visible-index/sourceIndex confusion. An animated parent, normal-coloured fragment, and a
non-uniform instance transform exercise placement and normals. **1,874 covered pixels, zero
pixels differ by >2/255**; the allowed edge tolerance is eight pixels. Drawing without the hook
differs at **2,843 pixels**, demonstrating the comparison detects missing deformation.

The complete visibility GPU suite is **8/8, zero GPU errors**, independently on **Node/NVIDIA,
Node/AMD, Chrome/AMD**. No model assets enter git. Guide: `docs/api/visibility.md`; the loaders
guide now points to the implemented hook rather than calling it a future API.

**Size:** identical esbuild browser ESM/minify settings, `gl-matrix` external, default gzip;
full visibility entry **13,220 → 13,747 B min+gzip (+527 B)**, within the 0.5–1 KB estimate.
Baseline loaded the two changed runtime source files from `f538ffe` through an esbuild plugin;
the core bundle remains unchanged. Development/TypeScript and IDE builds pass.

**Next: phase 3**, Models mode in Herd & Wolves, fox/cow clips driven from simulation state
and an accumulated per-agent phase. No site edits, yalc push, commit of phase 2 or deployment
in this session. Field/toon/post remain phase 4; skinning waits for phase 5.

### Phase 3 integration — 2026-10-15

Cow prey and fox wolves implemented in the local site. Each step advances one wrapped phase float per agent; the public hook blends idle/walk/run and selects eat while resting. Model geometry and pose translations are scaled together for matching rest-pose picking. The absent external animal palette is replaced by four UV swatches sampled from the installed site palette; no assets added. Larger model draws use conservative queue-completion wall-time measurements on this GPU, not a fixed count cap. The site can refuse 64K/1M without submitting the predicted unsafe render. Optional `node test/spikes/animated-models/check-site.mjs` validates the actual site asset selection, scale, clip order and generated shaders. Chrome/AMD drives and site typecheck pass; field/grass/toon/post are not implemented here.
