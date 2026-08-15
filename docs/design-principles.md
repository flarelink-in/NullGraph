# NullGraph — Design Principles

**Status:** normative. A module that violates one of these is wrong even if it renders.
**Scope:** how the roadmap's five constraints ([`roadmap.md`](./roadmap.md) §0) become code, and the
shape every design document in `docs/` follows.

The roadmap says *what* NullGraph is: data-oriented, zero scene graph, zero-copy, detect-don't-
polyfill, fail-loud. This document is the other half — the practices that make those hold up under
a real implementation, each one written down because getting it wrong cost something here.

Every rule below is followed by the case that produced it. That is deliberate: a principle without
its scar is a slogan, and the next reader needs to know what it is defending against.

---

## 1. Verify against the tree, and cite the line

Every design document has an **"Integration facts (verified against this repository)"** table:
a claim, the file and line it came from, and the consequence for the design. Nothing enters that
table from memory.

**Why.** The specs in this repo are load-bearing — they are read as the source of truth when the
code is being written weeks later. A plan built on an assumed API rots silently; a plan built on
cited lines fails loudly the moment the citation stops matching.

**The case.** The shadows design opened by repeating the roadmap's claim that CSM caster culling is
"just the `VisibilitySystem` run per cascade — free." Reading `BatchManager.createBatch` showed
group 0 binding 0 hard-wired to the one global camera buffer, so *nothing* could point the cull at
a different matrix. The claim was almost true, and the gap was a one-line engine change. Written
from memory, that gap would have surfaced during implementation instead of during design.

**Corollary — line numbers drift.** They moved across nine documents the day `RenderPipeline` and
`BatchManager` grew three fields. Prefer citing a *distinctive symbol* (`createBatch`'s group-0
bind group) over a bare line where the prose allows, and re-run a reference sweep after any core
edit. A stale citation is worse than none: it reads as verified.

---

## 2. Fail loud, at build time, naming the offender

No silent fallbacks. A module that would render the wrong thing refuses to compile, and the error
names the object, the number, and the way out.

```
ShadowAtlas: cannot place a 512px tile — 262144px² free but fragmented across 9 regions;
lower maxTileSize or raise atlasSize.
```

**Why.** The failure modes in a renderer are visual, and visual failures are attributed to the
wrong subsystem. "The shadow is missing" reads as a shadow bug when it is an allocator bug. An
error message is a shortcut past a debugging session.

**The case.** `BatchManager` now refuses a pass that mixes viewport-scoped and full-pass batches
without a declared size, because `RenderPipeline` genuinely cannot restore the viewport without
one. The alternative — render into the wrong rectangle — is a bug that would be found by eye,
weeks later, in a screenshot.

**Where the rule bends.** Errors belong at build/compile time, not in the frame loop. If a
condition can only be detected per frame, either hoist the check to construction or design the
condition out. Throwing at 60 Hz is not failing loud; it is failing continuously.

---

## 3. Conservative always beats clever — and "conservative" has a direction

Every approximation must err toward doing *more* work, never toward dropping something the user
would see. State the direction explicitly in the design, because it is not always obvious which
way is safe.

| Subsystem | Safe direction | Why |
|---|---|---|
| Visibility culling | draw something invisible | a dropped object is a hole; an extra one is wasted lanes |
| Cascade fitting | include off-screen casters (§5.4 extrusion) | a culled caster is a *missing shadow*, one indirection from its cause |
| Atlas eviction | fall back to fully lit | an evicted light reading black is a worse artifact than a missing shadow |
| Probe volumes | fall back to the global environment | never to black — "looks like Phase 1" beats "looks broken" |

**The case worth knowing.** Atlas-wide Hi-Z pools texels from neighbouring tiles at coarse mips,
which sounds like a correctness problem. It is not, and the reason is the direction: the pyramid is
a **max** reduction and the test is `ndcNear > occ`, so contamination only ever *raises* `occ`,
which only ever culls *less*. Under-culling costs draws; over-culling deletes shadows. Naming the
direction turned an apparent blocker into a free simplification.

---

## 4. Decorate the contract; don't fork it

When a new subsystem needs to change how an existing one behaves, it wraps that system's generated
WGSL rather than editing or duplicating it.

**The case.** `LightSystem.getShaderLayout()` emits `getIncidentLight(i, worldPos)`, and Lambert,
Toon, and PBR all consume its `IncidentLight.attenuation`. `ShadowSystem.wrapLighting(lights)`
renames that function to `getIncidentLightUnshadowed` and defines a new one that multiplies the
shadow term into `attenuation`. Three material families gained shadows with **zero** edits to any
material file, and `LightSystem` never learned that shadows exist.

**The discipline that makes it safe.** The rename is one deterministic textual replacement, guarded
by a marker check that throws if `LightSystem`'s output ever stops containing it, and pinned by a
headless test against the real `LightSystem` for both techniques. A decorator without that guard is
a silent breakage waiting for someone to rename a function.

**Design the seam before you need it.** `getIndirectDiffuse(worldPos, N)` takes `worldPos` in
lighting Phase 1 even though Phase 1 ignores it, precisely so Phase 2 can swap a probe volume in
without touching a material. A contract that omits the parameter it will need later forces a
rewrite at exactly the point where the code has the most users.

---

## 5. `out` parameters, not allocation — and the default still allocates

Anything on the per-frame path takes an optional trailing `out` and writes into it. The default
allocates, so one-shot callers and tests keep the simple signature.

```ts
export function fitCascade(cam, lightDir, nearD, farD, config,
                           out: CascadeResult = createCascadeResult()): CascadeResult
```

**Why both halves.** Constraint #2 forbids per-frame allocation. But a pooled-only API is hostile to
tests, which want independent results they can compare. Optional `out` satisfies both, and it is
already the idiom here (`splitFrustumCorners`, `packShadowData`).

**The case.** The shipped `ShadowSystem.update()` allocated 12 `Float32Array`s and 1 `Uint32Array`
per frame — 4 cascade `viewData`, 1 spot, 6 cube faces, 1 `packShadowData` output — plus every
gl-matrix temporary. Adding `out` parameters and hoisting the temporaries to module scope took it
to **0 and 0**, measured by patching the global typed-array constructors, with all 29 existing
tests untouched because every `out` is optional.

**Measure it, don't assert it.** Heap-delta and GC-event probes both gave misleading answers here
(one measured retention, not allocation; the other used a `PerformanceObserver` entry type that is
a no-op in Node 24). Counting constructor calls is unambiguous. If a performance claim goes in a
document, the number behind it should come from something that can be re-run.

---

## 6. Pool aggressively, but never let scratch escape into a return value

Module-scope scratch makes a function non-reentrant. That is fine for a single-threaded frame path
(and a Worker gets its own module instance) — but the object a caller *holds* must be theirs.

**The failure this prevents.** If `fitCascade`'s returned `viewData` were the module scratch, the
caller's cascade 0 would silently mutate when cascade 1 was fitted. Every value would be plausible
and wrong. Test U14 holds a result, computes a deliberately different one, and asserts the first is
byte-identical afterward.

**The related trap.** A cached `Uint32Array` view over a `Float32Array` must inherit `byteOffset`
**and** `length`, not just the buffer — Node's typed-array pool hands out non-zero offsets, and a
whole-buffer view writes integer fields to the wrong words with no error anywhere. U14b pins it
with a guard region before the view.

---

## 7. Shared, finite GPU resources need a declared budget

Core WebGPU guarantees **four** bind groups. NullGraph spends all four: 0 engine, 1 material,
2 lights, 3 shadows. There is no fifth on the baseline device.

**The consequence, made explicit.** Every subsequent fragment-stage subsystem joins an existing
group, and `lighting-ibl-design.md` §4.4 is where the whole allocation is written down — binding by
binding, plus per-stage counts against `maxStorageBuffersPerShaderStage` (7 of 8),
sampled textures (11 of 16), samplers, and uniforms.

**Why a table and not a convention.** The budget decided real design outcomes, not just bookkeeping:
irradiance volumes store **SH-L1** rather than L2 because L2 needs 7 texture bindings where 4 exist;
reflection probes are one `texture_cube_array` because per-probe cubes do not fit; DDGI visibility —
the correct fix for light leaking — is deferred to v2.0 because it needs two bindings nobody has.
A subsystem that allocates bindings without updating that table is spending someone else's budget.

**Watch for latent collisions.** `CommonChunks.SkinningBindings` and `PBRChunks.SkinningBindings`
both declare `@group(2) @binding(0)` — the same group `LightSystem` hardcodes — so CPU skinning and
clustered lighting cannot coexist in one shader today. Found by reading, not by a bug report.

---

## 8. Know which state is per-draw and which is per-pass

WebGPU render-pass state persists across draws. Anything a batch sets, the next batch inherits.

**The case.** Per-batch `viewport` was added for the shadow atlas, set only when a batch declared
one. Viewport and scissor are pass state, so a full-pass batch following a tiled one silently
rendered into the previous batch's tile. Unreachable in the shipped configuration — every batch in
the atlas pass has a viewport — and a trap for the first person to add a full-pass batch there.

**The rule.** When adding per-batch state that the encoder holds at pass scope, you owe three
things: a way to restore the default, the information needed to restore it (a `GPUTextureView` does
not expose its dimensions, hence `RenderPassConfig.size`), and a loud error for the case you cannot
restore.

---

## 9. Whatever suppresses a draw must suppress its compute, in the same place

**The case.** The shadow atlas suppresses a cached tile by giving its batches a zero-area viewport.
That stops rasterization and nothing else — the caster *cull* is a `ComputeStageNode` that kept
dispatching over the entire instance buffer for a tile drawing nothing. At 52 views × 6 meshes ×
100 000 instances that is ~600 000 wasted workgroups per frame, more than the draws the caching was
introduced to remove.

**The rule.** GPU-driven rendering splits one logical operation across a compute stage and a draw.
Gating one is not gating the operation. Set both in the same function so they cannot drift, and
write the test that fails when only one is set (U12).

`ComputeStageNode.enabled` exists for this, and `VisibilitySystem.getCullStage(batch)` is how an
owner reaches the stage feeding a draw it owns.

---

## 10. Size structures by demand, not by capacity

`maxViews` is a bound, not a workload. Building per-slot draw state for all of it cost a render
pipeline, a cull stage, and a params buffer per (slot × caster mesh) — 384 pipeline compilations at
startup for a 64-view atlas with 6 caster meshes, most for slots no light could occupy.

**The rule.** Allocate GPU *memory* against the cap (it must not move), but build *pipelines and
stages* against what is actually registered, materialised from whichever call can first know the
count, idempotently, so registration order does not matter.

---

## 11. Everything expensive happens at build or compile time

Per-frame cost is uploads, a few buffer clears, dispatches, and draws. Meshletization, prefiltering,
pipeline creation, and bind-group assembly happen once.

**Corollary — amortize instead of skipping.** When a structure is too expensive to rebuild every
frame and too stale to build once, refresh a bounded slice per frame: one prefilter mip
(`updateMode: 'amortized'`), `refreshPerFrame` shadow tiles, `probesPerFrame` probes. Round-robin
also bounds the damage from a missed invalidation to a few frames rather than forever — which
matters in an engine with no scene graph, where **dirtiness is declared by the caller, not
inferred**.

---

## 12. Detect, don't polyfill — and don't stub what doesn't exist

The baseline path runs on core WebGPU with no feature flags. An optional feature may *accelerate*
(subgroup-binned lights, `multi-draw-indirect`), never gate.

**Do not pre-build for an API that has not shipped.** WebGPU has no ray-tracing extension, so
NullGraph has no BLAS/TLAS types. Stubbing them would encode guesses about an API that does not
exist and would have to be deleted. Detect and adopt when the platform ships it.

**Choose formats by what core guarantees.** `rgba16float` is a core storage format *and* filterable;
`rgba32float` needs `float32-filterable` and silently loses linear sampling without it. That single
fact decided every texture format in the lighting specs.

---

## 13. Tests pin numbers, and a regression test must be seen to fail

Assert counts, tolerances, and byte layouts — never "it looks right". Split by what a device is
needed for:

- **Headless (`npm run test:*`)** — all pure math, every generated WGSL module (parsed, entry points
  present, the binding-usage invariant `layout:'auto'` relies on), and any system logic reachable
  through a mock device.
- **GPU harness (`test/*-gpu/`)** — anything needing a real device, asserted numerically with one
  readback. Readback is legal in a test and forbidden in the frame loop.

**Two rules that carry most of the value.**

1. **A regression test must be observed failing on the old behaviour.** Every fix in this repo was
   verified by temporarily restoring the bug and watching the test go red. A test that has never
   failed is a test that might be asserting nothing.
2. **Write the test that catches the *inverted* version.** A flipped comparison in a cull or a
   shadow tap produces a plausible image — the visibility spec's `ndcNear > occ` and the shadow
   spec's `textureSampleCompareLevel` rule both have tests whose stated purpose is catching the
   flip, because the eye will not.

Prefer a test that names its own failure diagnosis: *"if the wall disappears and the quad draws,
`ndcNear > occ` is flipped."*

---

## 14. The shape of a design document

Every spec in `docs/` follows the same order, so a reader knows where to look:

| § | Section | Contains |
|---|---|---|
| 1 | Purpose | what it does, the single most important pre-existing fact, **non-goals**, design principles |
| 2 | Architecture | the `engine.render([...])` sequence, then a box diagram |
| 3 | Integration facts | the verified table (§1 above) |
| 4 | Data model | every GPU struct with byte counts and alignment |
| 5–9 | Mechanism | the actual design, one concern per section |
| 10 | Test suite | headless unit tests, then GPU integration tests, each with its expected number |
| 11 | Errors | the table of build-time errors, verbatim |
| 12 | Engine touchpoints | **required** vs **optional**, each with its zero-change fallback and that fallback's cost |
| 13 | Build order | steps that each render correctly, then **Deferred** |

Two sections earn their place by being uncomfortable to write:

- **Non-goals**, because a spec that promises everything commits to nothing, and the boundary is
  what makes the next release plannable.
- **Engine touchpoints**, because "this needs a core change" is the sentence a module author is
  most tempted to soften. State the change, state the zero-change fallback, and state honestly why
  the fallback is worse.

**Phasing.** When a module is large, split it into two documents where Phase 1 ships a complete
vertical slice and Phase 2 generalizes it. Phase 1 defines the contract; Phase 2 reimplements it.
If Phase 2 finds itself changing Phase 1's callers, the contract was drawn in the wrong place.

---

## 15. Review checklist

Before a module is called done:

- [ ] Every claim in §3 cites a file and line that currently says that.
- [ ] Non-goals are written down.
- [ ] Every failure path names the offender and the way out; none of them run per frame.
- [ ] Each approximation states its safe direction, and a test pins the inverted version.
- [ ] The frame path allocates nothing — measured, not assumed.
- [ ] No pooled scratch escapes into a returned object.
- [ ] New GPU bindings are recorded in the budget table (`lighting-ibl-design.md` §4.4).
- [ ] Per-batch state that the encoder holds at pass scope can be restored, or is refused.
- [ ] Anything that suppresses a draw suppresses its compute in the same function.
- [ ] Pipelines and stages are built against registered demand, not against caps.
- [ ] The baseline path uses no optional WebGPU feature.
- [ ] Every regression test was observed failing before it passed.
- [ ] Engine touchpoints list required vs optional, with fallbacks and their costs.
