# NullGraph Meshlet Geometry — Technical Design

**Target:** `src/extras/geometry/MeshletBuilder.ts` (+ types)
**Package:** `null-graph/geometry`
**Status:** `v1.0.5` design, implementation-ready.
**Companion:** [`visibility-design.md`](./visibility-design.md) — the runtime that culls and draws
what this document lays out. Read that second.
**Roadmap:** [`roadmap.md`](./roadmap.md) §5.1.

---

## 1. Purpose

Partition a mesh into **meshlets** — small, self-contained clusters of ≤64 vertices and ≤124
triangles — each carrying a bounding sphere and a normal cone. Pack the result into flat GPU
storage buffers with a stable, documented byte layout.

Meshlets are the unit of sub-object culling. Today NullGraph can only cull at the granularity of
a whole draw (i.e. not at all, on the GPU). With meshlets, a 500k-triangle mesh that is 90%
off-screen or back-facing costs roughly 10% of its triangles, decided on the GPU, with no CPU
per-cluster work. This document produces the data; [`visibility-design.md`](./visibility-design.md)
consumes it.

This is a **build-time** module. It runs when geometry is authored or loaded, not per frame. Its
entire per-frame cost is zero — the buffers it produces are uploaded once and read by the cull
compute stage thereafter.

### Non-goals for v1.0.5

- **Mesh simplification.** The builder meshletizes the index arrays it is *given*. Generating
  LODs by edge-collapse/quadric simplification is a separate, large algorithm — v1.1. v1.0.5 LOD
  is "bring your own decimated index buffers; we meshletize and index each level" (§8).
- **Vertex compression / quantization.** Meshlet vertices index the existing `f32` mega vertex
  buffer. Quantized (e.g. `snorm16`/oct-normal) vertices are v1.1.
- **Virtual geometry.** The Nanite-style DAG of cluster groups with continuous LOD and runtime
  streaming is v2.0 (§13). v1.0.5 meshlets are flat and discrete, deliberately.
- **The culling and drawing.** Owned by the companion doc. This module knows nothing about the
  camera, Hi-Z, or indirect draws.

### Design principles

1. **One source of truth for vertices.** Meshlet vertex references are *global* indices into a
   single mega vertex buffer, the same convention `MegabufferBuilder` already uses
   (`baseVertex`). `MeshletBuilder` is *layered on* `MegabufferBuilder` — it consumes that mega
   vertex pool rather than building its own, so the two can never disagree (§2.1).
2. **Flat, aligned, boring buffers.** Every output is a typed array with a fixed stride and no
   pointers. The GPU reads structs whose size is a multiple of 16 bytes (§5). No packing cleverness
   that the cull shader has to reverse.
3. **Deterministic.** The same mesh in produces byte-identical buffers out, so builds are
   diffable and the test suite (§10) can assert exact counts.
4. **No dependencies.** NullGraph depends only on `gl-matrix`. The meshletizer is a ~150-line
   greedy pass, not a bound library. (`meshoptimizer` is referenced for provenance, not linked.)

---

## 2. What a meshlet is

A meshlet is three spans plus two bounds:

```
             mega vertex buffer (f32, interleaved, shared by all meshes)
             ┌──────────────────────────────────────────────────────────┐
             │  v0  v1  v2  v3  v4  ...                    ... vN         │
             └───▲─────▲──────▲───────────────────────────────────────────┘
                 │     │      │        meshletVertices[vertexOffset + local]
   meshletVertices (u32): [ ... 17, 42, 8, 91, ... ]   ← global indices, ≤64 per meshlet
                            └─ local index 0,1,2,3 within this meshlet ─┘

   meshletTriangles (u32): [ (2,0,1) , (2,1,3) , ... ]  ← 3 LOCAL indices packed per u32, ≤124
                             each byte < vertexCount

   Meshlet descriptor: boundingSphere + normalCone + the two offsets and two counts.
```

To draw or cull meshlet `m`, everything needed is in its descriptor:

- `boundsCenter.xyz`, `boundsCenter.w = radius` — object-space sphere, for frustum + Hi-Z.
- `coneApex`, `coneAxis.xyz`, `coneAxis.w = cutoff` — for backface cluster culling.
- `vertexOffset, vertexCount` — its span in `meshletVertices`.
- `triangleOffset, triangleCount` — its span in `meshletTriangles`.

Reconstructing triangle `t` of meshlet `m`:

```
tri   = meshletTriangles[m.triangleOffset + t]
l0    =  tri        & 0xFF          // local vertex index, < m.vertexCount
l1    = (tri >>  8) & 0xFF
l2    = (tri >> 16) & 0xFF
g0    = meshletVertices[m.vertexOffset + l0]   // global vertex index
...    position = megaVertices[g0 * strideFloats + posOffset ...]
```

The high byte (`tri >> 24`) is reserved (0 in v1.0.5). It is where a per-triangle material or
double-sided flag will live later; the decode already masks it off so nothing changes when it is
populated.

### 2.1 MegabufferBuilder vs MeshletBuilder — who owns what

`MegabufferBuilder` (ships today) and `MeshletBuilder` (this doc) sit at **different levels of the
hierarchy** and are easy to confuse. They are meant to be **chained**, not chosen between:
`scene → instances → meshes → meshlets → triangles → vertices`. Megabuffer owns the mesh/vertex
level; meshlets own the sub-mesh cluster level one step finer.

| | `MegabufferBuilder` | `MeshletBuilder` |
|---|---|---|
| Level | mesh / vertex | sub-mesh cluster |
| Owns | the unified **vertex + index** pool for every mesh, plus per-mesh `{indexCount, firstIndex, baseVertex}` | the **partition** of each mesh's triangles into meshlets, plus bounds + cones |
| Vertices | **stores** them | **references** them (global indices into the megabuffer) |
| Feeds | instance-tier draws — `drawIndexedIndirect` over a mesh's index range (visibility §7.1) | meshlet-tier draws — programmable vertex pulling (visibility §7.3) |

`MeshletBuilder` **consumes a `MegabufferBuilder`**; it does not build its own vertex pool. One
mega vertex buffer therefore backs *both* culling tiers: the instance tier reads the megabuffer's
per-mesh offsets, the meshlet tier reads the meshlet buffers, and both pull from the same
vertices. That shared pool is the point — it is what lets one scene mix coarse instance-culled
meshes and fine meshlet-culled meshes against a single allocation. If you never enable the meshlet
tier, `MegabufferBuilder` alone (+ instance-tier visibility) is a complete, meshlet-free path.

---

## 3. Integration facts (verified against this repository)

Load-bearing. The buffer layout is designed around these, not around a generic engine.

| Fact | Source |
|---|---|
| `MegabufferBuilder.addMesh` already returns `{ indexCount, firstIndex, baseVertex, meshID }` and auto-increments `meshID` "for the Compute Shader". The meshlet builder reuses this offset model. | `src/extras/geometry/MegaBufferBuilder.ts:3-9,27-47` |
| The mega index buffer promotes to `Uint32Array` past 65 535 vertices. Meshlet **global** vertex indices are therefore `u32`. | `src/extras/geometry/MegaBufferBuilder.ts:50-58` |
| `BufferManager.createVertexBuffer` / `createIndexBuffer` upload a typed array to a `GPUBuffer`. Meshlet buffers are uploaded the same way but with `STORAGE` usage (they are read by compute, not bound as vertex/index at cull time). | `src/extras/geometry/Geometry.ts:22-25` |
| A batch's vertex + index buffers are attached via `setBatchGeometry(batch, vbo, ibo, indexCount, format)`. The **mega vertex buffer** is bound here; the index buffer the meshlet path uses is the *compacted* one the cull stage writes (companion doc §7), not `meshletTriangles`. | `src/core/NullGraph.ts:139` |
| An `INDEX \| STORAGE` buffer is legal, which is what lets the cull compute write the index buffer the draw consumes. Meshlet source buffers themselves are plain `STORAGE`. | WebGPU core |

Nothing here requires an engine change. The builder produces typed arrays; the app uploads them
with the existing `BufferManager`.

---

## 4. Build algorithm

Greedy, index-order, single pass. Deterministic and O(triangles).

### 4.1 The meshletizer

```
inputs:  indices (u32[], length 3T), maxVertices=64, maxTriangles=124
state:   current meshlet:  localVerts: u32[]  (global indices, ≤64)
                           localMap:   Map<globalIndex, localIndex>
                           tris:       u32[]   (packed local, ≤124)

for each triangle (a, b, c) in index order:
    # how many of a,b,c are NOT already local to the current meshlet?
    newVerts = count of {a,b,c} not in localMap
    if  localVerts.length + newVerts  > maxVertices
     or tris.length + 1              > maxTriangles:
        flushMeshlet()            # finalize current, start empty
    la = localIndexOf(a)          # inserts into localMap/localVerts if absent
    lb = localIndexOf(b)
    lc = localIndexOf(c)
    tris.push( la | (lb<<8) | (lc<<16) )

flushMeshlet()                    # finalize the tail
```

`flushMeshlet()` appends `localVerts` to the global `meshletVertices`, `tris` to the global
`meshletTriangles`, computes bounds/cone (§6), and emits one descriptor (§5) with the running
offsets. Then it clears `localVerts`, `localMap`, `tris`.

**Why index-order and not spatial.** Index-order greedy is deterministic, dependency-free, and
correct. Its only weakness is cluster *shape* — clusters can be stringy if the input index order
is incoherent, which weakens cone and Hi-Z culling but never makes it wrong. A spatial
improvement (grow toward the nearest unclaimed adjacent triangle) is a drop-in replacement for
this loop and is noted as optional in §12. Ship index-order first; it is the correctness
baseline every later optimization is diffed against.

### 4.2 The vertex-count guard is the subtle line

```
if (localVerts.length + newVerts > maxVertices ...)      // CORRECT: count only NEW verts
if (localVerts.length + 3        > maxVertices ...)       // WRONG: over-flushes, tiny meshlets
```

A triangle usually shares 1–2 vertices with the meshlet being grown. Counting all three as new
flushes far too eagerly and produces meshlets averaging ~40 vertices instead of ~64 — a ~1.5×
inflation in meshlet count, and therefore in cull-shader threads and descriptor memory. Count
`newVerts` against `localMap`, not a constant. This is the single line most likely to be written
wrong; §10 T3 pins the exact meshlet count that detects it.

---

## 5. GPU buffer layout

Four buffers per build. All are uploaded once as `STORAGE` (the mega vertex buffer is also the
draw-time vertex buffer, so it additionally carries `VERTEX`).

### 5.1 `meshlets` — descriptor array, **64 bytes each, 16-aligned**

```wgsl
struct Meshlet {
    boundsCenter : vec4<f32>,   // .xyz object-space center,  .w = radius
    coneApex     : vec4<f32>,   // .xyz object-space apex,    .w unused (0)
    coneAxis     : vec4<f32>,   // .xyz normalized axis,      .w = cutoff  (§6.2)
    offsets      : vec4<u32>,   // .x vertexOffset .y triangleOffset .z vertexCount .w triangleCount
};
@group(0) @binding(_) var<storage, read> meshlets : array<Meshlet>;
```

64 bytes, every member 16-aligned, no implicit padding. The layout mirrors the light layout's
"round, aligned, four vec4s" discipline (`LightSystem.ts:121`, 64 bytes/light) so the same
worker-fills-a-Float32Array story applies (§9).

### 5.2 `meshletVertices` — `array<u32>`

Flat. Meshlet `m` owns `[offsets.x, offsets.x + offsets.z)`. Each entry is a global index into
`megaVertices`.

### 5.3 `meshletTriangles` — `array<u32>`

Flat. Meshlet `m` owns `[offsets.y, offsets.y + offsets.w)`. Each entry packs three local vertex
indices (§2). One u32 per triangle — not bit-packed to 3 bytes. The 25% memory saving from tight
3-byte packing is not worth the misaligned reads in the draw path; triangle data is small.

### 5.4 `meshTable` + `meshLODs` — the mesh → meshlet map

A "mesh" (one logical model, e.g. `"rock"`) owns a contiguous run of LOD levels, each of which
owns a contiguous run of meshlets:

```wgsl
struct MeshLOD {
    meshletOffset : u32,   // first meshlet index for this LOD
    meshletCount  : u32,
    error         : f32,   // world-space geometric error of this level (§8)
    _pad          : f32,
};
struct MeshEntry {        // one per logical mesh
    lodOffset : u32,       // first MeshLOD index
    lodCount  : u32,
    _pad0     : u32,
    _pad1     : u32,
};
@group(0) @binding(_) var<storage, read> meshTable : array<MeshEntry>;   // 16 B each
@group(0) @binding(_) var<storage, read> meshLODs  : array<MeshLOD>;     // 16 B each
```

An **instance** (companion doc §4) references a mesh by its `MeshEntry` index. LOD selection
(companion doc §8) picks one `MeshLOD` from that mesh's range and dispatches its meshlets.

### 5.5 Byte accounting (for `report()`)

| Buffer | Stride | Count |
|---|---|---|
| `meshlets` | 64 B | number of meshlets |
| `meshletVertices` | 4 B | Σ vertexCount |
| `meshletTriangles` | 4 B | Σ triangleCount = total triangles |
| `meshLODs` | 16 B | Σ LOD levels |
| `meshTable` | 16 B | number of logical meshes |
| `megaVertices` | strideFloats·4 B | total unique vertices |

`meshletTriangles` length equals the input triangle count exactly — meshletization partitions
triangles, it never duplicates them. `meshletVertices` length exceeds unique vertex count because
boundary vertices are re-listed in each meshlet that touches them (the "overdraw" of vertex
references, typically 1.05–1.15×). Both figures are in `report()` so regressions are visible.

---

## 6. Bounds and cone

Computed per meshlet in `flushMeshlet()` over the meshlet's own vertices/faces, in object space.

### 6.1 Bounding sphere

v1 uses AABB-center + farthest-vertex radius — cheap, deterministic, slightly loose:

```
c = 0.5 * (min(positions) + max(positions))
r = max over vertices of length(position - c)
```

Looseness costs a little culling efficiency, never correctness (a looser sphere over-draws, never
over-culls). Ritter's or Welzl's minimal sphere is a drop-in refinement (§12).

**Instance transform.** The sphere is object-space. The cull shader transforms it: center by the
model matrix, radius by the **largest** absolute scale factor of that matrix. Under non-uniform
scale, using the max scale keeps the sphere conservative. This is the companion doc's job; it is
called out here because it is why the radius is stored separately from the center rather than
pre-baked.

### 6.2 Normal cone — and the sign that will be wrong

The cone lets the shader reject a meshlet whose every triangle faces away from the camera.

Build:

```
axis   = normalize( Σ_faces  faceNormal_i * area_i )      // area-weighted average
minCos = min over faces of dot(faceNormal_i, axis)         // tightest opening
```

All face normals lie within a cone of half-angle `α = acos(minCos)` about `axis`. The meshlet is
entirely back-facing — and therefore cullable — when even its most camera-facing triangle is
back-facing. Working that through:

```
cutoff = (minCos <= 0) ? 2.0 : sqrt(1 - minCos*minCos)     //  = sin(α); 2.0 = "never cull"
```

Stored as `coneAxis.w`. `apex` is the sphere center in v1 (a true apex solve is a §12 refinement;
center-as-apex is conservative — it can only under-cull).

The cull test the companion shader runs (world space, `d` = direction from camera to apex):

```wgsl
let d = normalize(apexWorld - cameraPos);
let backfacing = dot(axisWorld, d) >= cutoff;      // >= , and axis vs d, not -d
```

**The three ways this goes wrong, all silent:**
1. `<=` instead of `>=` — inverts it: culls exactly the front-facing meshlets. Looks like
   catastrophic holes.
2. Forgetting the `minCos <= 0 → 2.0` sentinel — a meshlet spanning >90° gets a bogus small
   cutoff and is wrongly culled from some angles. Reads as flickering facets.
3. Not renormalizing `axisWorld` after the normal-matrix transform under non-uniform scale —
   the cutoff comparison drifts. Reads as "culling looks slightly too aggressive near scaled
   objects."

Because these are silent and view-dependent, §10 T5 tests the cone against a hand-built cube
meshlet with known face normals rather than trusting eyeballing.

**Non-uniform scale.** The cone axis transforms by the inverse-transpose (normal matrix), not the
model matrix, then must be renormalized. For uniform scale the two coincide. v1.0.5 documents the
normal-matrix path in the companion shader; a builder-side flag `assumeUniformScale` (default
false) is not needed because the transform is entirely runtime.

---

## 7. Public API

```ts
import { MegabufferBuilder, MeshletBuilder } from 'null-graph/geometry';

// 1. The megabuffer owns the unified vertex/index pool and the stride. Every mesh (and every LOD
//    level) is appended to it; it also serves the instance tier via getOffset() (visibility §7.1).
const mega = new MegabufferBuilder(8);   // 8 floats/vertex: pos(3) normal(3) uv(2)

// 2. MeshletBuilder layers on top of that same megabuffer — it does not own vertices (§2.1).
const builder = new MeshletBuilder(mega, {
  positionOffsetFloats: 0,
  normalOffsetFloats: 3,
  maxVertices: 64,            // clamp: 1..255 (local index is 8-bit)
  maxTriangles: 124,          // clamp: 1..124  (fits the companion draw expansion)
});

// addMesh appends geometry to `mega` AND meshletizes it. One mesh, one LOD:
const rock = builder.addMesh('rock', {
  vertices: rockVerts,        // Float32Array, interleaved at mega's stride
  indices:  rockIndices,      // Uint32Array | Uint16Array, one LOD
});

// One mesh, explicit LOD chain (decimated index arrays you already have):
const tree = builder.addMesh('tree', {
  vertices: treeVerts,
  lods: [
    { indices: treeLOD0, error: 0.0  },   // full detail
    { indices: treeLOD1, error: 0.05 },   // world-space geometric error, meters
    { indices: treeLOD2, error: 0.20 },
  ],
});

const packed = builder.build();
```

`addMesh` returns a `MeshHandle { meshIndex: number; name: string }`. `meshIndex` is what an
instance stores to reference this mesh (companion doc §4).

`build()` returns:

```ts
interface MeshletPack {
  megaVertices:     Float32Array;   // upload: VERTEX | STORAGE
  meshlets:         ArrayBuffer;    // 64 B stride  — upload: STORAGE
  meshletVertices:  Uint32Array;    //              — upload: STORAGE
  meshletTriangles: Uint32Array;    //              — upload: STORAGE
  meshLODs:         Float32Array;   // 16 B stride  — upload: STORAGE (read as MeshLOD)
  meshTable:        Uint32Array;    // 16 B stride  — upload: STORAGE (read as MeshEntry)
  report: MeshletReport;
}

interface MeshletReport {
  meshes: number; lods: number; meshlets: number;
  triangles: number; vertexRefs: number; uniqueVertices: number;
  vertexReuse: number;             // vertexRefs / uniqueVertices, want < 1.2
  avgVerticesPerMeshlet: number;   // want > 55 with 64-cap (see §4.2)
  bytes: number;                   // sum of all buffers
}
```

`build()` also finalizes the shared `mega`: `pack.megaVertices` **is** the megabuffer's built
vertex array, and `mega.getOffset(name)` stays valid afterward so the **instance tier** can draw
that same geometry by its `{indexCount, firstIndex, baseVertex}` without touching meshlets (§2.1).

Uploading is deliberately the app's job, one call each through `BufferManager`, because usage
flags differ per buffer and the builder must not depend on a live `GPUDevice` (it must run in a
Worker or a build script). This mirrors `Geometry.upload(engine)` staying separate from geometry
construction (`Geometry.ts:22`).

---

## 8. LOD

v1.0.5 LOD is **discrete and provided**: the caller supplies pre-decimated index arrays per
level; the builder meshletizes each independently and records its range in `meshLODs` with a
world-space `error` (the geometric deviation of that level from LOD0, in the mesh's units).

Selection happens in the cull compute (companion doc §8), not here, but the metric is defined by
what this module stores:

```
projectedError(instance, lod) = lod.error * (screenHeightPx / (2 * distance * tan(fovY/2)))
select the coarsest lod whose projectedError <= errorThresholdPx   (default 1.0 px)
```

Storing `error` in world units (not a precomputed pixel value) is deliberate: the same packed
buffers must serve any resolution and FOV, resolved at runtime from the camera. A mesh with a
single LOD gets one `MeshLOD` with `error = 0`, which always passes — i.e. it never LODs, which
is the correct degenerate behavior.

Mesh simplification to *generate* those levels is out of scope (§1). The hooks are shaped so that
a v1.1 simplifier just becomes another way to fill `lods[]`.

---

## 9. Zero-copy / worker story

The whole builder runs without a `GPUDevice`, so it runs in a Web Worker or a Node build step.
Its outputs are transferable typed arrays. Two supported flows:

- **Bake at build time.** Run `MeshletBuilder` in a Node script, write the `MeshletPack` typed
  arrays to a `.bin`, ship it, `fetch` + `createBuffer` at load. Zero client build cost.
- **Bake in a Worker.** Run in a Worker, `postMessage(pack, [transfer list])` the buffers to the
  main thread by transfer (no copy), upload. Matches the engine-wide "a Worker fills the array,
  the main thread uploads it" pattern (`LightSystem.setRawData`, `LightSystem.ts:92`).

Per-frame cost of this module: **zero.** Everything it produces is immutable after `build()`.
Instances that reference these meshes change per frame, but that is instance data (companion doc),
not meshlet data.

---

## 10. Test suite

Acceptance criteria. Pure logic — no GPU — so all of these run in a plain unit test. Caps are
`maxVertices=64, maxTriangles=124` unless stated.

**Property assertions (every build must satisfy):**

- P1 — every meshlet has `vertexCount ≤ 64` and `triangleCount ≤ 124`.
- P2 — every packed local index in `meshletTriangles` is `< its meshlet's vertexCount`.
- P3 — `Σ triangleCount === inputTriangleCount` (partition, no loss, no duplication).
- P4 — decoding every meshlet's triangles back to global indices reproduces the input triangle
  set exactly (as a multiset).
- P5 — determinism: building the same input twice yields byte-identical buffers.

**Exact cases:**

### T1 — single triangle
One triangle. **Expect:** 1 meshlet, `vertexCount 3`, `triangleCount 1`,
`meshletVertices.length 3`, `meshletTriangles.length 1`, packed value decodes to local `(0,1,2)`.

### T2 — fits exactly
A fan of 62 triangles sharing a center, using 64 unique vertices. **Expect:** 1 meshlet
(`triangleCount 62 ≤ 124`, `vertexCount 64 ≤ 64`). Adding one more triangle that introduces a
65th vertex **must** produce 2 meshlets — the vertex cap, not the triangle cap, forces the split.

### T3 — the `newVerts` guard (catches §4.2)
An indexed grid plane, 16×16 quads → 512 triangles, 289 vertices, supplied in row-major triangle
order. **Expect (index-order greedy, correct guard):** a meshlet count the test pins exactly
(compute it once from the reference implementation and freeze it). **Diagnostic:** if the count is
~1.4–1.6× that value, the guard counted 3 new vertices per triangle instead of only the unshared
ones (§4.2). This is the meshlet-geometry analogue of the `<=` bug in `transient-pool.md` T2.

### T4 — triangle cap dominates
A mesh where triangles are added such that vertex reuse is total (a triangle strip: each new
triangle adds ≤1 new vertex). With a 124-triangle cap and 64-vertex cap, the **triangle** cap is
reached first. **Expect:** every full meshlet has exactly `triangleCount 124`, and
`vertexCount ≤ 64`.

### T5 — normal cone (catches §6.2)
Six meshlets, each the two triangles of one face of a unit cube, with known face normals
(±X, ±Y, ±Z). **Expect:** each meshlet's `coneAxis.xyz` equals its face normal (to 1e-5) and
`cutoff === 0` (a flat cluster, `minCos === 1`, `sin(α) = 0`). A meshlet built from two adjacent
cube faces (normals 90° apart) has `minCos = cos(45°)` and `cutoff = sin(45°) ≈ 0.7071`. A meshlet
spanning three mutually-perpendicular faces has `minCos ≤ 0` → `cutoff === 2.0` (never cull).

### T6 — bounding sphere contains everything
For every meshlet, every one of its vertices satisfies
`length(pos - boundsCenter.xyz) ≤ boundsCenter.w + 1e-4`. (Containment is the only correctness
property; tightness is a quality metric, reported not asserted.)

### T7 — LOD table wiring
A mesh added with three LOD index arrays. **Expect:** its `MeshEntry.lodCount === 3`, the three
`MeshLOD` ranges are contiguous and non-overlapping, `Σ MeshLOD.meshletCount` equals the mesh's
total meshlets, and the `error` values are stored in the given order.

---

## 11. Errors

All at build time, all naming the offending mesh. No silent clamping that changes topology.

| Condition | Message |
|---|---|
| `maxTriangles > 124` | `MeshletBuilder: maxTriangles <n> exceeds 124; the companion draw path expands 124 triangles per meshlet.` |
| `maxVertices > 255` | `MeshletBuilder: maxVertices <n> exceeds 255; local indices are 8-bit.` |
| index out of vertex range | `MeshletBuilder: mesh "<name>" index <i> references vertex <v>, out of range (<count> vertices).` |
| index array length not ÷3 | `MeshletBuilder: mesh "<name>" index array length <n> is not a multiple of 3.` |
| degenerate LOD (0 triangles) | `MeshletBuilder: mesh "<name>" LOD <k> has no triangles.` |
| duplicate mesh name | `MeshletBuilder: mesh name "<name>" already added.` |

Clamping `maxTriangles` silently to 124 would desync the builder from the draw expansion and
surface as dropped triangles at the tail of large meshlets — so it throws instead.

---

## 12. Build order

Each step ends shippable.

1. **Meshletizer core (§4)** — pure partitioning, no bounds. Assert P1–P5 and T1–T4. This is the
   correctness baseline; get the `newVerts` guard (§4.2) right here or T3 fails.
2. **Buffer packing (§5)** — emit the four buffers + `report()`. Assert byte strides and
   `report().triangles === inputTriangles`.
3. **Bounds + cone (§6)** — add sphere and normal cone. Assert T5, T6. Do the cube fixture; do not
   trust visual inspection of the cone sign.
4. **LOD table (§8)** — multi-LOD `addMesh`, `meshTable`/`meshLODs`. Assert T7.
5. **Worker/bake path (§9)** — verify the pack survives `postMessage` transfer and a Node bake
   round-trips byte-identically (P5 across process boundary).

Steps 1–2 are enough for the companion `VisibilitySystem` to begin instance-tier work (it does
not need bounds until meshlet-tier culling). Hand it those two first.

### Deferred to v1.1+

Spatial (adjacency-guided) meshlet growth for tighter cones; minimal bounding spheres
(Ritter/Welzl); true cone apex solve; per-triangle material byte in `tri >> 24`; vertex
quantization; a mesh *simplifier* to generate LOD chains.

### Deferred to v2.0 — virtual geometry

The step from "flat discrete meshlets" to Nanite-class virtual geometry is a **DAG of meshlet
groups**: adjacent meshlets are grouped, each group is simplified and re-split into fewer meshlets
forming the next LOD, and edges between levels record the simplification error. Runtime picks a
*cut* through the DAG per view so LOD varies continuously across one mesh, and only the meshlets on
the cut are streamed in. That needs the simplifier (v1.1), a group/partition pass, a streaming
residency system, and per-meshlet-group error bounds — none of which the v1.0.5 flat layout
precludes. `MeshEntry`/`MeshLOD` become a DAG node table; the meshlet descriptor already carries
the per-cluster error slot needed to evaluate a cut. It is called out here so v1.0.5's layout
choices stay forward-compatible, not because any of it ships in v1.0.5.
