# NullGraph Visibility — Technical Design

**Target:** `src/extras/visibility/VisibilitySystem.ts` (+ WGSL, types)
**Package:** `null-graph/visibility` (new subpath export)
**Status:** `v1.0.5` design, implementation-ready.
**Companion:** [`geometry-meshlets-design.md`](./geometry-meshlets-design.md) — the data this system
culls. Read it first.
**Roadmap:** [`roadmap.md`](./roadmap.md) §5.2, §4.

---

## 1. Purpose

Decide, on the GPU, what to draw — and issue the draws indirectly so the CPU never iterates the
scene. Cull by frustum, by meshlet normal cone (backface clusters), and by Hi-Z occlusion
(hidden geometry), select LOD, and write the indirect draw arguments the engine already knows how
to render.

The premise of NullGraph is "massive web worlds." That is only reachable if visible-set
determination is **independent of scene size on the CPU**. Today the CPU must decide per object.
After v1.0.5, the CPU uploads instance data once, clears a few counters, dispatches a compute
chain, and issues a fixed handful of draws; the GPU does the rest.

This document is the runtime. It consumes the flat meshlet buffers from
[`geometry-meshlets-design.md`](./geometry-meshlets-design.md) and produces indirect draws. It
does **not** define geometry, materials, or shading — it hands a compacted visible set to a batch
whose fragment shader is the caller's (or a material's).

### The single most important pre-existing fact

**NullGraph already implements GPU-driven instance culling.** `createBatch({ isIndirect: true })`
allocates a 5-`u32` `indirectBuffer` and a `sourceStorageBuffer`
(`src/core/BatchManager.ts:82-91`); a compute stage writes the visible count into
`indirectBuffer[1]` and packs survivors; `RenderPipeline.ts:140` draws it with
`drawIndexedIndirect`. This exact loop is documented in
[`api/core.md`](./api/core.md#indirect-drawing-gpu-compute-culling) and
[`deprecated-api.md`](./deprecated-api.md) §2.

**`VisibilitySystem` is therefore not a new engine capability — it is a correct, batteries-included
wrapper** over that primitive, adding the three things the manual example lacks: proper frustum
plane extraction, Hi-Z occlusion, and meshlet/LOD granularity. No core-engine change is required
for either tier (§12).

### Non-goals for v1.0.5

- **Deferred / visibility-buffer *shading*.** Writing triangle-IDs and shading in a fullscreen
  pass is v1.1. v1.0.5 culls and draws forward, with the caller's material.
- **A persistent GPU scene.** Instances are re-uploaded (or adopted zero-copy) each frame; there
  is no GPU-resident retained structure yet.
- **CPU readback of counts.** Nothing stalls the pipeline to read a GPU counter. Dispatch sizes
  come from CPU-known caps with idle threads early-outing (§8), exactly as the clustered binner
  dispatches over all clusters (`LightSystem.ts:387`).

### Design principles

1. **Wrap the existing primitive; don't fork it.** The indirect batch, `indirectBuffer`,
   `sourceStorageBuffer`, and `preClearBuffers` reset are used as documented. If a manual user and
   `VisibilitySystem` disagree, the manual docs win and this is a bug.
2. **Two tiers, same substrate.** Instance-tier (coarse, per-object) and meshlet-tier (fine,
   per-cluster) share the Hi-Z pyramid, the camera, and the indirect-draw mechanism. A caller
   picks a tier per material; the plumbing is identical.
3. **Conservative always beats clever.** Every approximation (loose spheres, projected AABBs,
   center-as-apex cones) errs toward *drawing something invisible*, never toward *dropping
   something visible*. A dropped visible object is a hole the user sees; an extra drawn object is
   a few wasted lanes.
4. **Detect, don't require.** Baseline is core WebGPU. `capabilities.subgroups` accelerates
   stream compaction; `multi-draw-indirect`, when present, removes the meshlet draw's idle lanes.
   Neither is on the baseline path.

---

## 2. Architecture

Per frame, `VisibilitySystem` contributes a compute chain that runs **before** the scene pass:

```
   engine.render([
      ...vis.getComputeStages(camera),   ┌─ Hi-Z build (N mips, from LAST frame's depth)
                                         ├─ instance cull  → visible instances + LOD
                                         └─ meshlet cull   → visible (instance,meshlet) list
      scenePass,        ← draws vis.batches with drawIndexedIndirect (writes this frame's depth)
      ...post.getPassSequence(),
   ]);
```

```
        ┌──────────────────────────────────────────────────────────────┐
        │  VisibilitySystem            user-facing surface              │
        │  setGeometry / setInstances / createDrawBatch / getComputeStages │
        └───┬───────────┬───────────────┬────────────────┬─────────────┘
            │           │               │                │
      ┌─────▼────┐ ┌────▼──────┐ ┌──────▼───────┐ ┌──────▼──────────┐
      │  HiZ     │ │ InstanceCull│ MeshletCull  │ │ Indirect batches │
      │ pyramid  │ │ (ComputeNode)│ (ComputeNode)│ │ (per material)   │
      └────┬─────┘ └────┬───────┘ └──────┬───────┘ └──────┬──────────┘
           │            │                │                │
        r32float    indirectBuffer[1]  storageBuffer   drawIndexedIndirect
        mip chain   += atomic count    = visible list   (RenderPipeline.ts:140)
```

Every box maps to an existing engine primitive: `createComputeStage` for the compute nodes
(`NullGraph.ts:112`), `createBatch({isIndirect})` for the draws, and the meshlet buffers are plain
storage bindings.

---

## 3. Integration facts (verified against this repository)

Load-bearing. Each was read from source, not assumed.

| Fact | Source |
|---|---|
| `isIndirect` batch allocates `indirectBuffer` = **5 × u32 = 20 B**, usage `INDIRECT \| STORAGE \| COPY_DST`. That is exactly `[indexCount, instanceCount, firstIndex, baseVertex, firstInstance]`. | `src/core/BatchManager.ts:83-86` |
| `sourceStorageBuffer = config.sharedSourceBuffer \|\| <auto>`. Pass your own to avoid a redundant allocation. | `src/core/BatchManager.ts:88-91` |
| The draw's **group 0** is always `[camera@0, batch.storageBuffer@1]`, created at `createBatch`. `layout:'auto'` prunes unused bindings, so the draw shader **must** use both. | `src/core/BatchManager.ts:94-102` |
| For an indirect batch, `updateBatchData` writes to **`sourceStorageBuffer`**, not `storageBuffer`. So `storageBuffer` is free to be the compute's compacted **output** that the draw reads at `@group(0)@binding(1)`. | `src/core/BatchManager.ts:154` |
| The draw is `drawIndexedIndirect(batch.indirectBuffer, 0)`; index & vertex buffers come from `setBatchGeometry`. An indirect batch is **exempt** from the `instanceCount===0` skip. | `RenderPipeline.ts:105,107-110` |
| Compute stages run before the render pass when sequenced first; the compute pass is closed before the render pass begins, giving the compute→draw memory barrier for free. | `RenderPipeline.ts:65-70` |
| `preClearBuffers` does a `queue.writeBuffer` before the stage — used to zero `indirectBuffer[1]` (offset 4) and the atomic cursors each frame, exactly as the clustered binner zeroes its counter. | `RenderPipeline.ts:30-44`, `LightSystem.ts:396` |
| Dispatch count may be a `() => [x,y,z]` re-evaluated per frame, but only from **CPU-known** values. GPU counts require §12. | `src/core/types.ts:156-161` |
| `ctx.depthTexture` is `RENDER_ATTACHMENT`-only — **not samplable**. Hi-Z must read a depth target created with `TEXTURE_BINDING` (a depth pre-pass, or the `PostChain` g-buffer depth). | `WebGPUContext.ts:126`, `post-processing-design.md` §5 |
| Camera uniform is 108 floats: `viewProj@0, view@16, proj@32, invViewProj@48, invProj@80, eye@96, near@100, far@101`. Frustum planes come from `viewProj`; Hi-Z projection from `view`/`proj`. | `Camera.ts`, `post-processing-design.md` §3 |

---

## 4. Data model

Three GPU inputs (uploaded once, or per-frame zero-copy for instances), three GPU work buffers.

### 4.1 Instances — the per-frame input (zero-copy)

```wgsl
struct Instance {
    model        : mat4x4<f32>,   // 64 B  object → world
    boundsCenter : vec4<f32>,     // 16 B  object-space sphere: .xyz center, .w radius
    meshIndex    : u32,           //       index into meshTable (geometry doc §5.4)
    lodOverride  : u32,           //       0xFFFFFFFF = auto-select (§8)
    flags        : u32,           //       bit0: disable occlusion cull; bit1: disable cone cull
    _pad         : u32,           // 16 B
};                                // 96 B total, 16-aligned
```

96 bytes, four-plus-`vec4` layout, so a Web Worker fills a `Float32Array` and the system adopts it
by reference — the `LightSystem.setRawData` contract (`LightSystem.ts:92`). `boundsCenter` is
copied from the referenced mesh at instance-build time; storing it per-instance avoids a
mesh-table indirection in the hot cull loop.

### 4.2 Geometry — uploaded once (from `MeshletPack`)

`megaVertices` (VERTEX | STORAGE), `meshlets`, `meshletVertices`, `meshletTriangles`, `meshLODs`,
`meshTable` — all `STORAGE`, byte layout per geometry doc §5. `setGeometry(pack)` uploads them and
caches the `GPUBuffer`s.

### 4.3 Work buffers — allocated once at compile

| Buffer | Purpose | Sizing |
|---|---|---|
| Hi-Z pyramid | `r32float` texture, `⌈log2(max(w,h))⌉+1` mips, `TEXTURE_BINDING \| STORAGE_BINDING` | screen |
| `visibleInstances` | instance-cull output: `{ instanceId, lodMeshletOffset, lodMeshletCount }` + atomic count | `maxInstances` |
| batch `storageBuffer` (per draw) | meshlet-cull output: compacted `{ instanceId, meshletId }` list | `maxVisibleMeshletInstances` |
| batch `indirectBuffer` (per draw) | the 5×u32 draw args | 20 B (engine-allocated) |

Nothing here is allocated per frame. `visibleInstances` and the atomic counters are the only
things written during culling; they are reset by `preClearBuffers`.

---

## 5. Hi-Z pyramid

A mip chain where texel at level *L* holds the **maximum** (farthest) depth of the 2×2 region
below it. Max, not min, because the depth buffer stores the *nearest* surface per pixel (clear
1.0, `depthCompare:'less'`, `BatchManager.ts:69`), and an object is occluded only when its nearest
point is behind the farthest of those near surfaces over its footprint (§5.2).

### 5.1 Build

`⌈log2(max(w,h))⌉` compute nodes (created once, not per frame). All read/write one `r32float`
texture, one mip per node:

- **mip 0** reads the app's samplable **depth texture** (`texture_depth_2d`, `textureLoad(...,0)`)
  over 2×2 and writes the max to pyramid mip 0.
- **mip L (L>0)** reads pyramid mip L-1 as `texture_2d<f32>` (`textureLoad`, unfilterable) over
  2×2, writes max to mip L.

```wgsl
@group(0) @binding(0) var srcDepth : texture_depth_2d;                 // mip-0 node only
@group(0) @binding(0) var srcMip   : texture_2d<f32>;                  // mip-L node (L>0)
@group(0) @binding(1) var dstMip   : texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dst = gid.xy;
    let dim = textureDimensions(dstMip);
    if (dst.x >= dim.x || dst.y >= dim.y) { return; }
    let s = dst * 2u;
    let d0 = load(s + vec2u(0,0));   // load() = textureLoad(srcDepth,·,0) or textureLoad(srcMip,·,0)
    let d1 = load(s + vec2u(1,0));
    let d2 = load(s + vec2u(0,1));
    let d3 = load(s + vec2u(1,1));
    textureStore(dstMip, dst, vec4f(max(max(d0,d1), max(d2,d3)), 0,0,0));
}
```

Successive dispatches in one compute pass are ordered with a memory barrier between them under
WebGPU's sync model, so the N nodes may batch into one pass (they carry no `preClearBuffers`, so
the engine does batch them, `RenderPipeline.ts:46`). Odd dimensions floor on the way down; the
max-reduction stays conservative because a dropped column can only make the stored occluder depth
*smaller* (less occlusion), never larger.

`r32float` is core WebGPU as both a storage texture (`write`) and a sampled `texture_2d<f32>`
(unfilterable `textureLoad`). No optional feature is used.

### 5.2 One-frame latency (single-phase, the baseline)

The pyramid is built from **last frame's** depth, because this frame's depth does not exist until
the scene pass, which runs after culling. Consequences:

- One depth texture, persistent across frames: written at the end of frame N, read at the start of
  N+1. The read (compute) precedes the write (render) within N+1 in sequence order, so it is the
  *previous* contents that are read — correct, and synchronized by the pass boundary.
- Frame 0: the pyramid is cleared to 1.0 (far) → nothing occludes → everything draws. Correct
  degenerate start.
- Fast camera cuts can briefly disocclude geometry that last-frame Hi-Z still thinks is hidden →
  one frame of pop-in. The two-phase mode (§9) removes this at the cost of a second cull+draw.

---

## 6. The cull tests

Shared WGSL, used by both tiers. All operate on a **world-space** bounding sphere.

### 6.1 Frustum (Gribb–Hartmann, from `viewProj`)

```wgsl
// viewProj is column-major (gl-matrix / WGSL). row(i) = (m[0][i], m[1][i], m[2][i], m[3][i]).
fn frustumPlanes(vp: mat4x4<f32>) -> array<vec4f, 6> {
    let r0 = vec4f(vp[0][0], vp[1][0], vp[2][0], vp[3][0]);
    let r1 = vec4f(vp[0][1], vp[1][1], vp[2][1], vp[3][1]);
    let r2 = vec4f(vp[0][2], vp[1][2], vp[2][2], vp[3][2]);
    let r3 = vec4f(vp[0][3], vp[1][3], vp[2][3], vp[3][3]);
    var p = array<vec4f,6>(r3+r0, r3-r0, r3+r1, r3-r1, r3+r2, r3-r2); // L,R,B,T,N,F
    for (var i=0u; i<6u; i++) { p[i] = p[i] / length(p[i].xyz); }      // normalize for radius test
    return p;
}
fn sphereInFrustum(p: array<vec4f,6>, c: vec3f, r: f32) -> bool {
    for (var i=0u; i<6u; i++) { if (dot(p[i].xyz, c) + p[i].w < -r) { return false; } }
    return true;
}
```

### 6.2 World-space sphere from an instance

```wgsl
fn worldSphere(inst: Instance) -> vec4f {
    let c = (inst.model * vec4f(inst.boundsCenter.xyz, 1.0)).xyz;
    // conservative radius under (possibly non-uniform) scale: largest column length
    let sx = length(inst.model[0].xyz);
    let sy = length(inst.model[1].xyz);
    let sz = length(inst.model[2].xyz);
    return vec4f(c, inst.boundsCenter.w * max(sx, max(sy, sz)));
}
```

### 6.3 Meshlet normal cone (meshlet tier only)

Consumes the cone from geometry doc §6.2. `axis` transforms by the normal matrix
(inverse-transpose of `model`) and is **renormalized**; `apex` by `model`. The sign convention and
the `cutoff == 2.0` "never cull" sentinel are the geometry doc's — reproduced here because getting
`>=` / axis-vs-`d` wrong silently culls the *visible* meshlets (geometry doc §6.2, "three ways
this goes wrong"):

```wgsl
fn coneBackfacing(axisW: vec3f, cutoff: f32, apexW: vec3f, eye: vec3f) -> bool {
    if (cutoff >= 2.0) { return false; }               // sentinel: cone too wide to ever cull
    let d = normalize(apexW - eye);                     // camera → meshlet
    return dot(normalize(axisW), d) >= cutoff;          // all triangles face away
}
```

### 6.4 Hi-Z occlusion

Conservative projected-bounds test. Rejects only when the sphere is provably behind the pyramid.

```wgsl
fn occluded(c: vec3f, r: f32, hiz: texture_2d<f32>, cam: Camera, screen: vec2f) -> bool {
    let vc = (cam.view * vec4f(c, 1.0)).xyz;            // view space, forward = -Z
    let zView = -vc.z;
    if (zView - r < cam.near) { return false; }         // crosses near plane → keep (visible)

    // screen-space AABB of the sphere (conservative: project center ± r on x/y)
    let p  = cam.proj;
    let cx = (p * vec4f(vc, 1.0));
    let ex = (p * vec4f(vc + vec3f(r,0,0), 1.0));
    let ey = (p * vec4f(vc + vec3f(0,r,0), 1.0));
    let ndcC = cx.xy / cx.w;
    let rx = abs(ex.x/ex.w - ndcC.x);
    let ry = abs(ey.y/ey.w - ndcC.y);
    let uvMin = (vec2f(ndcC.x - rx, -(ndcC.y + ry)) * 0.5) + 0.5; // note Y flip
    let uvMax = (vec2f(ndcC.x + rx, -(ndcC.y - ry)) * 0.5) + 0.5;

    let sizePx = max(uvMax.x - uvMin.x, uvMax.y - uvMin.y) * max(screen.x, screen.y);
    let level  = clamp(u32(ceil(log2(max(sizePx, 1.0)))), 0u, u32(cam.hizMaxLevel));
    let dim    = vec2f(textureDimensions(hiz, level));

    // nearest depth of the sphere, in NDC (project the closest point along -Z)
    let near   = p * vec4f(vc.xy, vc.z + r, 1.0);
    let ndcNear= near.z / near.w;

    // sample the 2x2 texels covering the footprint at `level`, take the MAX occluder depth
    let t0 = vec2u(clamp(uvMin, vec2f(0), vec2f(1)) * dim);
    let t1 = vec2u(clamp(uvMax, vec2f(0), vec2f(1)) * dim);
    var occ = 0.0;
    occ = max(occ, textureLoad(hiz, t0, level).r);
    occ = max(occ, textureLoad(hiz, vec2u(t1.x, t0.y), level).r);
    occ = max(occ, textureLoad(hiz, vec2u(t0.x, t1.y), level).r);
    occ = max(occ, textureLoad(hiz, t1, level).r);

    return ndcNear > occ;   // sphere's nearest point is behind the farthest occluder → occluded
}
```

`ndcNear > occ` is the whole test, and its direction is the analogue of `transient-pool.md`'s
`<` predicate: flip it and every visible object is culled instead of every hidden one. §10 T4
pins an exact visible-triangle count so a flipped comparison is caught by a number, not by eye.

---

## 7. The two tiers

### 7.1 Instance tier — the zero-change baseline

Coarse, per-object. This is the documented primitive (`api/core.md`) with Hi-Z and mesh bucketing
added. One indirect batch **per unique mesh** (instances of that mesh share its index range):

- **Input:** the full `Instance[]` in `sourceStorageBuffer` (written by `updateBatchData`, or a
  worker array adopted zero-copy).
- **Cull compute:** one thread per instance. `sphereInFrustum && !occluded` → append the
  instance's shading data to the mesh bucket's `storageBuffer` at `atomicAdd(&args.instanceCount, 1)`,
  which doubles as the compaction slot.
- **Draw:** `drawIndexedIndirect` with `[mesh.indexCount, instanceCount, mesh.firstIndex,
  mesh.baseVertex, 0]`. Vertex buffer = mega; index buffer = mega indices; per-instance transform
  read from `storageBuffer[instance_index]`.

Draw-call count = number of distinct meshes drawn, **independent of instance count**. For worlds
that are many instances of few meshes (foliage, rocks, debris) this is the whole win and it needs
**no engine change and no meshlets.** Ship it first.

### 7.2 Meshlet tier — sub-object granularity

Fine, per-cluster. One indirect batch **per material**, covering every meshlet of every instance
using that material. Drawn by **meshlet-as-instance programmable vertex pulling**, which fits
`drawIndexedIndirect` with a *static* index buffer — so it, too, needs no engine change (§7.3).

Two compute passes:

1. **Instance cull** (as §7.1, coarse) → compacted `visibleInstances` with each survivor's chosen
   LOD range `[lodMeshletOffset, lodMeshletCount)` (§8) and an atomic `visibleInstanceCount`.
2. **Meshlet cull.** Dispatched over `maxInstances × MAX_MESHLETS_PER_INSTANCE` (CPU-known caps);
   thread `(i, j)` early-outs if `i >= visibleInstanceCount` or `j >= inst.lodMeshletCount`. Else
   it tests meshlet `lodMeshletOffset + j`: `!coneBackfacing && sphereInFrustum && !occluded`
   (each on the meshlet's own sphere/cone, transformed by the instance model). Survivors are
   appended to the batch `storageBuffer` as `{ instanceId, meshletId }` at
   `atomicAdd(&args.instanceCount, 1)`.

The over-dispatch (idle lanes past `visibleInstanceCount` / `lodMeshletCount`) is the price of not
reading a GPU count back to the CPU. It is bounded and cheap; §12 removes it with indirect
dispatch when that lands.

### 7.3 The meshlet draw — vertex pulling with a static index buffer

The draw is one `drawIndexedIndirect` with `indirectBuffer = [124*3, visibleMeshletCount, 0,0,0]`
and a **static index buffer** `[0, 1, 2, …, 371]` (`uint16`, uploaded once). For an *indexed* draw
`@builtin(vertex_index)` is the fetched index value, so it runs `0..371` while
`@builtin(instance_index)` runs `0..visibleMeshletCount-1`. Each draw-instance is one visible
meshlet:

```wgsl
// group 0 is fixed by createBatch: camera@0, batch.storageBuffer@1 (both must be used — see note).
@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(1) var<storage, read> visible : array<vec2<u32>>;  // = batch.storageBuffer: {instanceId, meshletId}

// group 1 (setRenderPassBindGroup): the read-only geometry + instance transforms
@group(1) @binding(0) var<storage, read> meshlets        : array<Meshlet>;
@group(1) @binding(1) var<storage, read> meshletVertices : array<u32>;
@group(1) @binding(2) var<storage, read> meshletTriangles: array<u32>;
@group(1) @binding(3) var<storage, read> megaVertices    : array<f32>;
@group(1) @binding(4) var<storage, read> instances       : array<Instance>;

@vertex
fn vs_main(@builtin(instance_index) mi: u32, @builtin(vertex_index) vi: u32) -> VSOut {
    let entry   = visible[mi];
    let inst    = instances[entry.x];
    let m       = meshlets[entry.y];
    let localTri= vi / 3u;
    if (localTri >= m.offsets.w) {                       // past this meshlet's triangle count
        var o: VSOut; o.pos = vec4f(-2.0, -2.0, -2.0, 1.0); return o;  // outside clip → discarded
    }
    let corner  = vi % 3u;
    let packed  = meshletTriangles[m.offsets.y + localTri];
    let local   = (packed >> (corner * 8u)) & 0xFFu;
    let gv      = meshletVertices[m.offsets.x + local];
    let pos     = vec3f(megaVertices[gv*STRIDE+0u], megaVertices[gv*STRIDE+1u], megaVertices[gv*STRIDE+2u]);
    var o: VSOut;
    o.pos = camera.viewProj * inst.model * vec4f(pos, 1.0);
    // ... pull normal/uv similarly, pass instanceId for material lookup ...
    return o;
}
```

**The `@group(0)@binding(1)` requirement.** `createBatch` binds `batch.storageBuffer` there
unconditionally and `layout:'auto'` prunes it if unused (`BatchManager.ts:94-102`), so the shader
must use it. We *make it the visible list* — genuinely needed — by having the meshlet-cull compute
write compacted entries into `batch.storageBuffer`. `updateBatchData` is not used on this batch
(for indirect it targets `sourceStorageBuffer`, `BatchManager.ts:154`), so `storageBuffer` is free
for this. Set `strideFloats: 2` (one `vec2<u32>` per entry) and `maxInstances:
maxVisibleMeshletInstances` so the engine sizes `storageBuffer` correctly, and pass
`sharedSourceBuffer: <the instance buffer>` so the auto `sourceStorageBuffer` allocation
(`BatchManager.ts:88`) is not wasted.

**Idle lanes.** A half-full meshlet still spends `372` vertex invocations; the extra ones take the
degenerate branch and are clipped. That is the cost of no mesh shaders on WebGPU. When
`capabilities.has('multi-draw-indirect')`, a v1.1 path replaces the static index buffer with one
real indexed draw per meshlet and removes the waste (roadmap-modules.md §3); the baseline stays universal.

---

## 8. LOD selection

In the instance-cull compute, per surviving instance, using the mesh's `MeshLOD.error` (geometry
doc §8) and the camera:

```wgsl
fn selectLod(mesh: MeshEntry, wc: vec3f, r: f32, cam: Camera) -> u32 {
    if (inst.lodOverride != 0xFFFFFFFFu) { return inst.lodOverride; }
    let dist = max(length(wc - cam.eye) - r, cam.near);
    let pxPerUnit = cam.screenH / (2.0 * dist * tan(cam.fovY * 0.5));
    var chosen = mesh.lodOffset;                          // finest by default
    for (var k=0u; k<mesh.lodCount; k++) {
        let lod = meshLODs[mesh.lodOffset + k];
        if (lod.error * pxPerUnit <= cam.lodErrorPx) { chosen = mesh.lodOffset + k; } // coarsen while under budget
        else { break; }
    }
    return chosen;
}
```

`lodErrorPx` defaults to 1.0 (one pixel of geometric error). A single-LOD mesh has `error==0` at
level 0, which always passes → it never LODs, the correct degenerate behavior. Selection lives in
the cull, not on the CPU, so a Worker filling instances never computes distances.

---

## 9. Two-phase temporal culling (accuracy mode, optional within v1.0.5)

Single-phase (§5.2) can pop in disoccluded geometry for one frame. Two-phase removes it, the
Ubisoft/Nanite method, at the cost of a second cull+draw:

1. **Phase-1 cull** against **last frame's** Hi-Z → draw survivors (`scenePass`, writes depth).
2. **Rebuild Hi-Z** from the depth just written (§5.1 again, on this frame's depth).
3. **Phase-2 cull** re-tests only the instances/meshlets that phase-1 *occlusion-rejected* (a
   per-object "rejected" bit written in phase 1) against the **new** Hi-Z → draw the false
   negatives into the same targets with `loadOp:'load'`.

Cost: one extra Hi-Z build, one extra cull dispatch, one extra indirect draw per material, and a
second small work buffer for the phase-1 rejects. No new engine capability — it is the same
primitives sequenced twice. Ship single-phase first (§13); add two-phase behind
`config.mode:'two-phase'` once single-phase is proven.

---

## 10. Test suite

Culling tests need a GPU, but the math is unit-testable in isolation and the integration cases
assert exact counts read back once (readback is legal in a test, forbidden in the frame loop).

**Pure-logic unit tests (no GPU):**

- U1 — `frustumPlanes` on a known `viewProj` yields six planes matching a CPU reference to 1e-5.
- U2 — `sphereInFrustum`: a sphere straddling each plane is classified in/out correctly;
  a sphere fully outside one plane by more than its radius is rejected.
- U3 — the Hi-Z `level` selection: a footprint of `sizePx` picks `ceil(log2(sizePx))`, clamped.
- U4 — cone sign: reuse geometry doc T5's cube meshlets; a camera in front of the +Z face does
  **not** cull it, a camera behind it (looking at its back) **does**. Flip the `>=` and U4 fails.

**Integration tests (GPU, one readback each):**

### T1 — instance frustum
1000 instances, 500 inside the frustum, 500 behind the camera. Occlusion disabled.
**Expect:** `indirectBuffer[1] === 500`, and the 500 packed instances are exactly the in-frustum
set.

### T2 — draw-call independence
10 000 instances of **one** mesh, all visible. **Expect:** exactly **one** `drawIndexedIndirect`,
`instanceCount === 10 000`. Raise to 100 000 → still one draw call. This is the headline property
(roadmap §8.1 gate 1).

### T3 — meshlet cone culling
One sphere mesh (meshletized), camera outside. **Expect:** roughly half its meshlets survive cone
culling (`visibleMeshletCount ≈ 0.5 × totalMeshlets`, within a tolerance), and the rendered image
is identical to drawing all meshlets (back meshlets contribute nothing). Disable cone culling via
`flags` → count returns to `totalMeshlets`.

### T4 — Hi-Z occlusion (catches the §6.4 flip)
A large occluder wall at z=10 fully covering a 100-triangle quad at z=20. **Expect:** the quad's
meshlet is culled → its triangles do not appear in the visible count. Move the quad to z=5 (in
front) → it reappears. **Diagnostic:** if the *wall* disappears and the *quad* draws, `ndcNear > occ`
is flipped (§6.4).

### T5 — LOD selection
One mesh, three LODs (`error` 0, 0.1, 0.4). At a distance where `0.1·pxPerUnit ≤ 1 < 0.4·pxPerUnit`,
**expect** LOD 1 selected (`visibleInstances[k].lodMeshletOffset` in LOD-1's range). Push the
camera far enough that `0.4·pxPerUnit ≤ 1` → LOD 2.

### T6 — first-frame safety
Frame 0 with a never-written Hi-Z. **Expect:** nothing is occlusion-culled (`occ==1.0` everywhere),
i.e. identical output to occlusion disabled. No black frame.

### T7 — zero-copy, zero-alloc
Drive `setInstances` from a Worker-filled view (transferable `ArrayBuffer` or
`SharedArrayBuffer` — the contract is the same either way); run 100 frames.
**Expect:** no per-frame allocation on the main thread (instrument `report()`), and the frame path
is upload + clears + dispatch chain + draws only.

---

## 11. Errors

Build/compile time, naming the offender. No silent fallback.

| Condition | Message |
|---|---|
| meshlet tier without `setGeometry` | `VisibilitySystem: meshlet tier requires setGeometry(pack) before createDrawBatch().` |
| instance count exceeds `maxInstances` | `VisibilitySystem: setInstances received <n> instances, exceeding maxInstances <m>.` |
| no samplable depth for Hi-Z | `VisibilitySystem: hiZ.enabled requires a depthSource with TEXTURE_BINDING; ctx.depthTexture is RENDER_ATTACHMENT-only.` |
| `MAX_MESHLETS_PER_INSTANCE` exceeded by a mesh | `VisibilitySystem: mesh "<name>" LOD <k> has <n> meshlets, exceeding maxMeshletsPerInstance <m>; raise the cap.` |
| draw batch before instances set | `VisibilitySystem: getComputeStages() called before setInstances().` |
| visible-meshlet overflow (validation build only) | `VisibilitySystem: visible meshlet count <n> exceeded maxVisibleMeshletInstances <m>; culling wrote past the buffer.` |

The last one deserves an atomic-clamped counter in a validation build: without it, an
underestimated `maxVisibleMeshletInstances` writes past `storageBuffer` and corrupts the draw
silently — the visibility analogue of `transient-pool.md`'s over-aliasing.

---

## 12. Engine touchpoints

**None are required.** Both tiers run on the engine as it exists. Two are *optional* efficiency
additions, filed here so they are not rediscovered later:

1. **Indirect dispatch** (`dispatchWorkgroupsIndirect`). Today `RenderPipeline` dispatches a
   CPU-known count (`RenderPipeline.ts:58`). The meshlet cull (§7.2) and two-phase (§9) over-dispatch
   to cover a GPU-side count. Adding an optional `dispatchIndirect?: GPUBuffer` to
   `ComputeStageConfig`, and calling `dispatchWorkgroupsIndirect` when present, lets the instance
   cull write phase-2's exact dispatch size. **Additive, opt-in, correct-without-it.**
2. **`setBatchIndexBuffer(batch, buffer, count, format)`.** The meshlet draw wants an index buffer
   without a meaningful vertex buffer. `setBatchGeometry` sets both (`BatchManager.ts:145-150`);
   the baseline works by binding the mega buffer as an unused vertex buffer (legal — a bound-but-
   unread vertex buffer is valid), but a dedicated setter is tidier. **Cosmetic.**

Everything load-bearing — atomics, `INDIRECT|STORAGE` buffers, `r32float` storage textures,
`drawIndexedIndirect`, the compute→render barrier — is core WebGPU and already reachable.

---

## 13. Build order

Each step renders correctly at its end.

1. **Frustum plane / sphere math (§6.1–6.2)** — pure, GPU-free. Assert U1, U2.
2. **Instance tier, frustum only (§7.1)** — wrap the documented indirect primitive with correct
   plane extraction and mesh bucketing. Assert T1, T2. This is a complete, shippable GPU culler
   with **zero engine changes** and no meshlets. Everything after is additive.
3. **Hi-Z build (§5)** — the pyramid, occlusion disabled in the test until T6/T4 pass. Assert U3,
   T6.
4. **Occlusion test (§6.4)** wired into the instance cull. Assert T4 on whole objects. Get
   `ndcNear > occ` right here.
5. **Meshlet tier (§7.2–7.3)** — cone + per-meshlet frustum/Hi-Z, the static-index-buffer draw.
   Assert U4, T3. Needs geometry doc steps 1–3.
6. **LOD (§8)** — selection in the instance cull. Assert T5.
7. **Two-phase (§9)** — behind `mode:'two-phase'`. Only after single-phase is proven, so any pop
   regression is isolated to this step.

Step 2 alone justifies the release: draw calls decoupled from instance count, on today's engine.
Steps 3–7 tighten it toward "massive worlds."

### Deferred

Two-phase as default (needs profiling to justify the second draw); `multi-draw-indirect` fast path
for the meshlet draw (v1.1, detect-only); the visibility-buffer *shading* model (v1.1); persistent
GPU-side instance residency and streaming, which is the on-ramp to virtual geometry (v2.0,
geometry doc §13).

---

## 14. A stable draw order (2026-10-16)

**Asked by the user** (2026-10-16), after Cycle D's offline films (D2.1) found it: "change the cull
to fixed draw orders; that might solve other problems with cull too".

### 14.1 The problem

**Every survivor of the cull claims its slot with `atomicAdd`** (`visibilityShaders.ts:313`; the
meshlet tier at `:393` and `:473`). Which thread claims first changes from frame to frame and from
run to run, so the order of the compacted list, and with it the draw order, does too. That causes
three things:

1. **Depth ties change their winner.** Two instances with faces at the same depth (Herd & Wolves'
   far cubes, all at one height; overlapping herds) show one, then the other. Two offline films of
   the same simulation differed in up to half their frames (D2.1, F1: at most 0.53% of a frame; the
   anime chain's depth ink widens each tie). In the live demo, the same thing is a flicker.
2. **Every survivor contends on one counter.** At 1M instances that is a million atomics on one
   word per batch.
3. **The dispatch is one thread per instance in groups of 64**, so at most 65,535 × 64 = 4,194,240
   instances can be culled per draw.

### 14.2 The design: survivors in source order

The instance tier's cull becomes two dispatches per batch. Neither uses a global atomic.

| Pass | Dispatch | Does |
|---|---|---|
| **cull** | ⌈N / 256⌉ groups of 256 | The same tests as before: mesh, frustum, LOD band, Hi-Z. Each group records its 256 results as eight 32-bit words of visibility bits (built in workgroup memory with `atomicOr`, which commutes), and their total in `blockCounts[group]`. |
| **scatter** | max(1, ⌈N / 256⌉) groups of 256 | Each group first finds its offset: the survivors of every block before it. Its 256 lanes sum `blockCounts` in parallel, then reduce the sums in workgroup memory. A visible instance's slot is that offset plus its rank in the group: the set bits of the words before its own, plus those below its bit in its own (`countOneBits`). It copies the instance there with `sourceIndex = i`, as the old cull did. The last group writes the total into `drawArgs[1]`, the indirect draw's instance count, as a plain store. |

Survivors land in **ascending source index**, every frame and every run, and on every GPU.

**Two passes, not three.** The first build had a separate scan pass: one group turning the counts
into offsets. Each extra dependent dispatch costs time at small counts, and this scene culls eight
batches. So the scatter now sums the earlier blocks' counts itself. That is cheap: 4,096 counts at
1M instances, 16 a lane.

**Data, per batch:** the bits (N / 8 bytes) and `blockCounts` (N / 256 words): 144 KB at 1M
instances. N is the system's `maxInstances`.

**Limits:**
- groups of 256 are the core default limit (`maxComputeInvocationsPerWorkgroup`);
- the cap moves from 4.19M to 65,535 × 256 = 16.8M instances per draw;
- a larger `maxInstances` is refused by name, from the device's own limit.

**No stale count.** With no instances, the cull dispatches nothing. The scatter's single group then
writes 0. The `preClearBuffers` write of `drawArgs[1]` is no longer needed, and goes.

### 14.3 The API change

**`getCullStage(batch): ComputeStageNode | null` becomes `getCullStages(batch): ComputeStageNode[]`**
(the batch's cull and scatter, in order; empty for a batch this system did not create).

- **Why it must change.** Picker and ProbeVolume put a batch's cull into sequences of their own
  (the pick camera, the probe faces); ShadowSystem and ProbeVolume gate it with `enabled`. All of
  them now need every stage.
- **The fix in each:** gate all of the stages, and splice all of them, in order.
- **No alias,** by the repository's rule for breaking changes (`deprecated-api.md` §9).
- **Outside the library,** nothing calls it: the test site and Axion were searched.

`buildInstanceCullWGSL(opts)` keeps its name and options, and now builds the cull pass.
`buildInstanceScatterWGSL()` and `INSTANCE_CULL_GROUP` are new.

**The chain:** every batch's cull, then every batch's scatter. Each batch keeps its own order, and
no dispatch waits on the one just before it.

### 14.4 Phases

1. **The instance tier** (the films, Herd & Wolves, shadows, probes and picking all draw through it).
   Built 2026-10-16; results in §14.6.
2. **The meshlet tier,** later. Its two passes take the same compaction:
   - the coarse pass over instances;
   - the meshlet pass over (visible instance, meshlet) pairs, ordered by slot, then meshlet.

   Until then the meshlet tier keeps `atomicAdd`, which this section records.

### 14.5 Not changed

- The tests, LOD bands, Hi-Z, the `Instance` record, the draw shaders and the bind groups of the
  draw.
- The text module's own label cull (`buildTextCullWGSL`), which still appends with `atomicAdd`.
  Its labels draw with depth off, where order decides which label covers which. That is a question
  for the text module, not taken up here.

### 14.6 Results — 2026-10-16 (phase 1, the instance tier)

**Built:**
- `buildInstanceCullWGSL` (the cull, bits and block counts), `buildInstanceScatterWGSL` and
  `INSTANCE_CULL_GROUP` (`visibilityShaders.ts`);
- two stages a batch, their scratch released with the batch, the cap check
  (`VisibilitySystem.ts`);
- `getCullStages`, and its callers updated (ShadowSystem, ProbeVolume, Picker);
- `deprecated-api.md` §9, `api/visibility.md`, design-principles §9.

**Correctness:**
- **V7, new:** read back the compacted lists of 13,000 instances. Both are exactly the CPU mirror's
  survivors in ascending order (2,872 spheres, 966 cubes), and two runs write the same bytes.
  - Watched failing against the old cull, which listed them from 1536 in one run and differently in
    the next.
  - Passes on Node/NVIDIA, Node/AMD and Chrome/AMD.
- **The other suites that cull,** with zero WebGPU errors:
  - on Node/NVIDIA and on Node/AMD: visibility 11/11, shadows 14/14, picking 8/8, probes 6/6,
    loaders 12/12, agents 137/137, film 9/9, the culling render scene;
  - in Chrome/AMD: shadows, visibility and picking.
- **Headless:** visibility, picking, shadows, environment, text, loaders, agents and film all pass.
  The tests that pinned the old shape say what they meant:
  - U41/U42: the shader's hashes, and where the band sits, now before survival;
  - W2: the source index is written by the scatter;
  - L3, K1, the lifecycle test: two stages a batch, and the scratch released with it;
  - the shadow and probe stand-ins: `getCullStages`.
- **The film that found it:** two films of 4,096 agents, flashes off, are now **equal frame for
  frame, byte for byte** on both GPUs (F1 tightened from "under 1%"). The flashes keep their own
  `atomicAdd`, in the scenario's respawn stage.

**Cost:** D2.0's draw case, the Models look at 1080p on the GTX 1650. GPU ms per frame, culls
included, the mean of the three views; the old and new builds were alternated twice:

| Agents | Old (atomicAdd) | New (two passes) | Change |
|---|---|---|---|
| 16,384 | 3.42 | 3.96 | +0.54 ms (+16%) |
| 65,536 | 3.38 | 3.44 | +0.07 ms (+2%) |
| 262,144 | 5.07 | 5.09 | +0.02 ms |
| 1,048,576 | 11.15 | 12.52 | +1.37 ms (+12%) |

- **At 1M the scatter's re-read is the cost.** It reads each survivor's 96-byte record again, about
  a million in a whole-field view, where the old cull wrote it in the pass that read it.
- **At 16K it is mostly the second dispatch per batch,** and the first count in each run is also
  the noisiest.
- **The first build, with a separate scan pass,** cost about +0.4–0.8 ms at 16K–64K; two passes took
  most of that back.
- **Per whole frame,** step included, the change is at most about +8% (16K) and +0.2% (1M).
- **The draw itself exceeds the "about 10%" line at 16K and 1M,** so the numbers go to the user
  before this is committed (§14.4's rule).
