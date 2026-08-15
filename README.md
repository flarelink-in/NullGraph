# NullGraph

> Zero scene graph. Zero copy. Infinite scale.

<div align="center">

[![npm version](https://img.shields.io/npm/v/null-graph?style=for-the-badge&color=33e6cc&labelColor=0a0a0a)](https://www.npmjs.com/package/null-graph)
[![npm downloads](https://img.shields.io/npm/dm/null-graph?style=for-the-badge&color=ff6b6b&labelColor=0a0a0a)](https://www.npmjs.com/package/null-graph)
[![bundle size](https://img.shields.io/bundlephobia/minzip/null-graph?style=for-the-badge&color=fcc419&labelColor=0a0a0a)](https://bundlephobia.com/package/null-graph)
[![license](https://img.shields.io/github/license/Vikas593-cloud/NullGraph-Test-Engine?style=for-the-badge&color=51cf66&labelColor=0a0a0a)](https://github.com/Vikas593-cloud/NullGraph-Test-Engine/blob/main/LICENSE)
[![stars](https://img.shields.io/github/stars/Vikas593-cloud/NullGraph?style=for-the-badge&color=ff922b&labelColor=0a0a0a)](https://github.com/Vikas593-cloud/NullGraph/stargazers)

</div>

A Data-Oriented WebGPU rendering framework for massive web worlds.

NullGraph is a brutalist, high-performance rendering library designed specifically for Web Workers and Data-Oriented Design (DOD).

It completely abandons the traditional Object-Oriented Scene Graph (`Root -> Node -> Mesh -> Geometry`) in favor of mapping raw, contiguous `ArrayBuffers` directly to WebGPU Storage Buffers.

If you are building an MMO, a voxel engine, or a multiverse with tens of thousands of dynamic entities, NullGraph keeps rendering off your main thread and out of the garbage collector.

---


# Why NullGraph?

Traditional WebGL frameworks (like Three.js or Babylon.js) are built for ease of use, heavily relying on the `new` keyword, dynamic memory, and Garbage Collection. When scaling up to massive open worlds, this OOP overhead causes main-thread stuttering and shader compilation lag.

NullGraph solves this by doing less:

- **Zero Scene Graph:** No `.traverse()`, no `.updateMatrixWorld()`. The GPU reads your flat array directly.

- **Zero-Copy Streaming:** Calculate your ECS layout in a Web Worker, pass the `Float32Array` to the main thread, and blast it straight to VRAM.

- **Render Queues (Batches):** Render thousands of unique object types simultaneously with minimal GPU state changes.

- **No GC Spikes:** Memory is pre-allocated. No runtime object creation or destruction.

- **Compute-Driven Indirect Drawing:** Offload culling entirely to the GPU. NullGraph supports WebGPU Compute Shaders that dynamically build `IndirectDrawArgs`, resulting in zero CPU overhead for visibility checks.

- **GPU-Driven Visibility (v1.0.5):** Batteries-included culling on the GPU — frustum, Hi-Z occlusion, meshlet normal-cone (backface cluster) rejection, and screen-space-error LOD. Draw-call count becomes proportional to **material count, not object count**, so 100,000 instances still cost a handful of draws.

- **Shadows That Inherit the Culling:** A shadow cascade is just another camera, so caster culling reuses the same GPU cull chain rather than duplicating it — 4 cascades over 100,000 instances is 4 indirect draws per caster mesh. Cached atlas tiles skip both their draw *and* their cull.

- **Multi-Pass Architecture:** Seamlessly chain offscreen render passes into screen-space post-processing pipelines (Bloom, CRT, HUD effects) by attaching textures directly to subsequent batches.

---

# Installation and Setup

NullGraph is distributed as a modular ESM package. To maintain its "Zero-Copy" philosophy, it requires `gl-matrix` as a peer dependency to ensure your application and the engine share the same math structures.

## 1. Install via NPM

```bash
# Install the core engine
npm install null-graph

# Install required peer dependencies
npm install gl-matrix

# Recommended: Install WebGPU types for IDE autocomplete
npm install @webgpu/types --save-dev
```

---

## 2. Module Architecture

NullGraph uses Subpath Exports to keep your production bundles lean. You only pay for the features you import.

### `null-graph`
The core engine: device and capability negotiation, buffer/texture management, and the linear
executor. You hand `render()` a flat array of `ComputeStageNode | RenderPassNode` and it walks it
in order — consecutive compute stages batch into one WebGPU pass, and a render pass closes it,
which is where the compute→draw memory barrier comes from for free. Camera state is one 432-byte
uniform (view/proj/their inverses + eye, near/far/fov/aspect) shared by every batch, and indirect
draw is wired through: a batch flagged `isIndirect` is drawn with `drawIndexedIndirect` from a
buffer a compute stage filled.

### `null-graph/geometry`
13 primitive generators, declarative vertex layouts, and the two-layer buffer story:
`MegabufferBuilder` packs every mesh into one vertex + index pool and records per-mesh
`{indexCount, firstIndex, baseVertex}`; `MeshletBuilder` consumes that pool and partitions each
mesh into ≤64-vertex / ≤124-triangle clusters, each with a bounding sphere and a normal cone, plus
discrete LOD chains. Meshlets store **global** indices into the mega pool, so one vertex allocation
backs both the coarse and fine draw paths.

### `null-graph/visibility`
GPU-driven culling. Two tiers over the same substrate: instance-tier (one indirect draw per unique
mesh, independent of instance count) and meshlet-tier (per-cluster, drawn by vertex pulling over a
static index buffer, since WebGPU has no mesh shaders). A compute chain builds an `r32float` Hi-Z
max-reduction pyramid from last frame's depth, then culls by frustum, normal cone, and occlusion,
selects LOD by screen-space error, and compacts survivors into the indirect draw args. Nothing is
read back to the CPU — dispatch sizes come from CPU-known caps with idle threads early-outing. The
cull math has a CPU mirror (`cullMath.ts`) pinned by headless tests.

### `null-graph/shadows`
Cascaded shadow maps with GPU caster culling. A cascade fills the *same* 108-float camera layout
the cull and draw shaders already consume, so caster culling is the visibility chain pointed at a
different uniform rather than new machinery. Cascades are fitted with a rotation-invariant bounding
sphere and snapped to whole shadow texels (a box fit shimmers on camera yaw; an unsnapped centre
crawls on translation), and the ortho near plane is extruded so off-screen casters still land in
the map. Bias is applied on the caster in the depth-only vertex shader. Receiver-side it
**decorates** the light system's WGSL — wrapping `getIncidentLight` and folding the shadow term
into `attenuation` — so Lambert, Toon and PBR all gain shadows with no material edits.

### `null-graph/loaders`
`GLBParser` with resource unpacking, `SkeletonManager`, and an `Animator` for keyframe sampling and
interpolation, feeding the vertex-shader skinning path in the material builders.

### `null-graph/materials`
Basic, Lambert, Toon, Emissive, Matcap, and Cook-Torrance PBR. Each family is a **portable pair** —
WGSL bindings plus an `fs_main` body, parameterized by bind-group index — from one source of truth,
so the same material drops into a normal batch or a GPU-culled visibility batch unchanged. Lighting
arrives as an injected WGSL layout (`getVisibleLightCount` / `getIncidentLight`), which is what lets
the light system swap clustered for standard without any material knowing.

### `null-graph/lights`
Data-oriented lighting: 7 proxy types writing into a flat 16-float-per-light array a Worker can
fill, adopted by reference. Three techniques — `standard`, `tiled-forward`, and compute-driven
`clustered-forward`, which bins lights into a 3D froxel grid (logarithmic depth slices) in a
compute stage, using subgroup ops when the device negotiated them and an atomic baseline otherwise.

### `null-graph/post`
An explicit, pre-allocated post chain. `TransientPool` aliases render targets by lifetime analysis
so a 12-effect chain reuses a handful of textures, with `history: 2` double-buffering reserved for
temporal effects. `ParamArena` keeps every effect's uniforms in one arena with zero per-frame
allocation, and `ShaderComposer` generates each pass's WGSL — 16 effects including Bloom, SSAO,
Bokeh, FXAA, and tonemapping.

### `null-graph/cameras`
Orbital, Fly, Follow, and Path controllers as pure state objects — spherical-coordinate math with
no DOM dependency — plus optional event proxies, so the same controller runs on the main thread or
inside a Worker.

### `null-graph/debug-ui` & `null-graph/profiler`
Real-time telemetry widgets and WebGPU timestamp-query GPU profiling.

---
# The Architecture Demo Suite

## Play the Live Demo
[nullgraph.dev](https://nullgraph.dev/)

## Github Source Code [v1.0.0]
[NullGraph-Test-Engine](https://github.com/Vikas593-cloud/NullGraph-Test-Engine)

---
# Documentation

For comprehensive guides and API references, please check our documentation:

### Guides
- [Quick Start Guide](./docs/quick-start.md) — get a lit cube on screen
- [Scene Composition](./docs/scene-composition.md) — how to assemble any scene, end to end
- [Engine Roadmap](./docs/roadmap.md) — versioned module plan and what ships when
- [Design Principles](./docs/design-principles.md) — the rules every module follows, and the case behind each
- [Deprecated & Breaking API Changes](./docs/deprecated-api.md) — migration guide

### Technical Designs
Implementation-ready specs. Each states its non-goals, cites the engine facts it depends on, and
lists the acceptance tests up front.

- [Meshlet Geometry](./docs/geometry-meshlets-design.md) — clusters, bounds, normal cones, LOD chains
- [Visibility](./docs/visibility-design.md) — two-tier GPU culling, Hi-Z, indirect draw
- [Shadows · Phase 1](./docs/shadows-design.md) — cascade fitting, caster bias, the lighting decorator
- [Shadows · Phase 2](./docs/shadows-atlas-design.md) — atlas, punctual lights, tile caching, light-view Hi-Z
- [Lighting · Phase 1](./docs/lighting-ibl-design.md) — IBL: prefiltered env, BRDF LUT, SH irradiance
- [Lighting · Phase 2](./docs/lighting-probes-design.md) — irradiance volumes, probe capture, reflection probes
- [Post-Processing](./docs/post-processing-design.md) — how the `PostChain` is architected
- [Post-Processing Presets](./docs/post-presets.md) — ready-made Bloom / SSAO / Tonemap bundles
- [Transient Pool](./docs/transient-pool.md) — render-target recycling internals

### API References
- [Core (`null-graph`)](./docs/api/core.md) · [Core Reference](./docs/api/core-reference.md)
- [Cameras (`null-graph/cameras`)](./docs/api/cameras.md)
- [Debug UI (`null-graph/debug-ui`)](./docs/api/debug-ui.md)
- [Geometry (`null-graph/geometry`)](./docs/api/geometry.md)
- [Lights (`null-graph/lights`)](./docs/api/lights.md) · [Shader & Compute Reference](./docs/api/lights-shader-reference.md)
- [Loaders (`null-graph/loaders`)](./docs/api/loaders.md)
- [Materials (`null-graph/materials`)](./docs/api/materials.md)
- [Post-Processing (`null-graph/post`)](./docs/api/post.md)
- [Profiler (`null-graph/profiler`)](./docs/api/profiler.md)
- [Shadows (`null-graph/shadows`)](./docs/api/shadows.md) · [Shader & Buffer Reference](./docs/api/shadows-shader-reference.md)
- [Visibility (`null-graph/visibility`)](./docs/api/visibility.md) · [Shader & Buffer Reference](./docs/api/visibility-shader-reference.md)
## Roadmap

NullGraph is the high-performance rendering backbone for the Axion Engine.

---

### Core Architecture

- [x] Multi-Object Render Queue / Batching

- [x] Depth / Z-Buffer Integration (Proper 3D occlusion)

- [x] VBO/IBO Geometry Buffer Manager

- [x] Multi-Pass Rendering & Texture Attachments

- [x] GPU Compute Frustum Culling & Indirect Drawing

- [x] Geometry Builder & `null-graph/geometry` extras

- [x] Megabuffer (unified vertex/index pool) & Meshlet Builder (bounds, cones, LOD)

- [x] GPU-Driven Visibility — two-tier culling (instance + meshlet), Hi-Z occlusion, screen-space-error LOD

- [ ] Deferred / Visibility-Buffer Shading

- [ ] Virtual Geometry (Nanite-style cluster-LOD DAG)

---

### Materials & Assets

- [x] Physically Based Rendering (Cook-Torrance BRDF)

- [x] Integrated PBR Material System (Albedo, Normal, ARM maps)

- [x] Native GLB/GLTF Parsing & Resource Unpacking

- [x] Alpha Blending & Additive Transparency States

---

### Animation & Logic

- [x] Hardware-Accelerated Skeletal Animation (GPU Skinning)

- [x] Animation Timeline & Keyframe Interpolation (Animator)

- [ ] Morph Targets / Shape Keys

- [ ] GPU-Driven Particle Systems (Compute-based)

---

### Lighting & Post-Processing

- [x] Dynamic Light System (Point / Directional / Spot) via Zero-Copy proxies

- [x] Real-time Light Culling — Tiled (Forward+) & Compute-Driven Clustered Forward

- [x] Post-Processing Pipeline (Bloom, SSAO, Tonemap, Custom Effects)

- [x] Directional Shadows / Cascaded Shadow Maps (CSM) — sphere-fit cascades, texel snapping, GPU caster culling, hardware PCF

- [x] Shadow Atlas — punctual (spot + point) shadows, importance-driven tiles, per-tile caching, light-view Hi-Z

- [ ] Image-Based Lighting (IBL) & Environment Mapping — [specced](./docs/lighting-ibl-design.md)

- [ ] Irradiance Volumes & Reflection Probes — [specced](./docs/lighting-probes-design.md)
---

## Showcase

<div align="center">

### Architecture Demos
|                         AoS                         |                         SoA                         |                         AoSoA                         |
|:---------------------------------------------------:|:---------------------------------------------------:|:-----------------------------------------------------:|
| <img src="./showcases-images/AoS.png" width="350"/> | <img src="./showcases-images/SoA.png" width="350"/> | <img src="./showcases-images/AoSoA.png" width="350"/> |

### GPU Compute & Post-Processing
|                        GPU Culling                         |                        Space Fleet                         | CRT Effect |
|:----------------------------------------------------------:|:----------------------------------------------------------:|:----------:|
| <img src="./showcases-images/GPUCulling.png" width="350"/> | <img src="./showcases-images/SpaceFleet.png" width="350"/> |<img src="./showcases-images/crt.png" width="350"/>

### PBR Materials & Animation
|                         Rusty Metal                         |                            Skeletal Animation                             |                         Morphogenesis                         |
|:-----------------------------------------------------------:|:-------------------------------------------------------------------------:|:-------------------------------------------------------------:|
| <img src="./showcases-images/rusty_metal.png" width="350"/> | <img src="./showcases-images/animation-post-processing.png" width="350"/> | <img src="./showcases-images/morphogenesis.png" width="350"/> |

</div>

---

## 📦 Package Stats

<div align="center">

| Metric | Value |
|--------|-------|
| **Weekly Downloads** | ![npm](https://img.shields.io/npm/dm/null-graph?style=flat-square&color=33e6cc) |
| **Version** | ![npm](https://img.shields.io/npm/v/null-graph?style=flat-square&color=ff6b6b) |
| **License** | ![license](https://img.shields.io/npm/l/null-graph?style=flat-square&color=51cf66) |
| **Minified + GZip** | ![size](https://img.shields.io/bundlephobia/minzip/null-graph?style=flat-square&color=fcc419) |

</div>

---

## 🔗 Related Repositories

| Repository                                                                           | Description |
|--------------------------------------------------------------------------------------|-------------|
| [**NullGraph Test Engine**](https://github.com/Vikas593-cloud/NullGraph-Test-Engine) | Interactive demo suite & documentation hub |
| [**Axion Engine**](https://axion-engine.web.app)                                     | Full game engine built on NullGraph |

<p align="center">
  <sub>Built with 🔥 by the NullGraph</sub>
</p>