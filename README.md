# NullGraph

A data-oriented WebGPU rendering library with explicit render and compute scheduling, flat
instance data, and optional GPU-driven rendering and simulation modules.

[![npm version](https://img.shields.io/npm/v/null-graph)](https://www.npmjs.com/package/null-graph)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

NullGraph has no retained scene graph. Applications supply geometry, buffers, batches and an
ordered sequence of compute stages and render passes. This makes resource ownership and execution
order explicit, and lets GPU simulation output feed rendering without a per-frame CPU readback.
It can run on the main thread, in a Web Worker with an `OffscreenCanvas`, or headlessly with a
WebGPU implementation.

## Installation and getting started

```sh
npm install null-graph
```

Import the core and the modules your application needs:

```ts
import { NullGraph, Camera } from 'null-graph';
import { Primitives, StandardLayout } from 'null-graph/geometry';
import { StandardPBRMaterial } from 'null-graph/materials';
```

- [Application guide and runnable examples](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/agent/guide.md)
- [Core setup and usage](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/core.md)
- [Core API and buffer reference](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/core-reference.md)
- [Documentation index for coding assistants](https://github.com/Vikas593-cloud/NullGraph/blob/main/llms.txt)

The application guide includes complete examples and a render-to-PNG workflow. Its tooling runs
from a repository checkout; the npm package ships the library's `dist/` files, not the repository
runners. Branch-head documentation may describe changes newer than your installed package;
match documentation to the release or commit you use.

## Rendering model

- **Flat data rather than scene traversal.** Instance transforms and attributes live in typed
  arrays or GPU buffers. Applications manage hierarchy and gameplay state themselves.
- **Explicit scheduling.** Submit a linear sequence of compute stages and render passes, including
  multipass rendering, custom WGSL, and direct or indirect draws.
- **Batched geometry and instances.** Geometry pools, instance buffers and GPU visibility reduce
  per-object submission work. Draw count still depends on meshes, materials and passes.
- **GPU-resident integration.** Compute stages can write buffers that later rendering stages use.
  Buffers shared on the same device can avoid CPU readback and re-upload. Uploading a CPU typed
  array still transfers data to the GPU; it is not a zero-copy CPU-to-VRAM operation.
- **Worker and headless support.** Use `OffscreenCanvas` for worker rendering, or render to an
  offscreen texture without a canvas. An existing `GPUDevice` can be adopted for interoperability.
- **Reusable frame resources.** Buffer-backed APIs and prebuilt stage sequences support low-
  allocation frame loops. Applications remain responsible for their own allocations and uploads.

## Modules and features

Modules are separate package subpaths so applications can import only what they use. Bundle size
depends on the imported features and the application's bundler.

| Import | Implemented features | Documentation |
|---|---|---|
| `null-graph` | Device initialization/adoption, cameras, buffer and texture management, batches, compute stages, render passes, indirect drawing and headless rendering | [Usage](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/core.md) · [Reference](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/core-reference.md) |
| `null-graph/geometry` | Vertex layouts, primitives, geometry builders, dynamic geometry, unified vertex/index pools, meshlets and LOD-chain construction | [Geometry](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/geometry.md) |
| `null-graph/materials` | PBR, Lambert, Toon, Basic, Emissive and Matcap materials; composable WGSL shader builders | [Materials](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/materials.md) |
| `null-graph/loaders` | GLB parsing, animation timeline evaluation and skeletal data management for GPU vertex skinning | [Loaders and animation](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/loaders.md) |
| `null-graph/visibility` | GPU instance and meshlet culling, frustum and Hi-Z occlusion tests, normal-cone rejection, screen-space-error LOD and indirect draw generation | [Visibility](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/visibility.md) · [Shader reference](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/visibility-shader-reference.md) |
| `null-graph/lights` | Buffer-backed lights with standard, tiled-forward and clustered-forward lighting | [Lights](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/lights.md) · [Shader reference](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/lights-shader-reference.md) |
| `null-graph/shadows` | Cascaded directional shadows, punctual-light shadow atlas, cached tiles and caster culling integrated with visibility | [Shadows](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/shadows.md) · [Shader reference](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/shadows-shader-reference.md) |
| `null-graph/environment` | Image-based diffuse/specular lighting, environment prefiltering, BRDF LUTs, irradiance volumes, GPU probe capture and parallax-corrected reflection probes | [Environment](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/environment.md) · [Probes](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/probes.md) |
| `null-graph/post` | Explicit post-processing chains, reusable transient render targets, uniform arenas and effects including bloom, SSAO, depth of field, FXAA and tone mapping | [Post-processing](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/post.md) · [Presets](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/post-presets.md) |
| `null-graph/picking` | On-demand GPU picking for supported batches and visibility instances, returning source instance index, depth and world position | [Picking](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/picking.md) |
| `null-graph/text` | GPU-culled SDF/MSDF labels, halos, depth modes, atlas generation/loading and labels following GPU-buffer positions | [Text](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/text.md) |
| `null-graph/agents` | GPU fixed-step sense → think → act simulation, spatial neighbour queries, WGSL rule policies, small dense-network inference, custom sensors/actions, inspection and instance output | [Agents](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/agents.md) · [Spatial grid](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/agent-grid.md) |
| `null-graph/cameras` | Orbit, fly, follow and path controllers with DOM-independent state and optional input proxies | [Cameras](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/cameras.md) |
| `null-graph/debug-ui` | Debug views and telemetry, including grid and agent visualization | [Debug UI](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/debug-ui.md) |
| `null-graph/profiler` | WebGPU timestamp-query profiling on devices that support it | [Profiler](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/profiler.md) |

### Lighting and environmental effects

Direct lights, shadows and indirect lighting connect to materials through shader-layout contracts.
Probe volumes support iterative bounce-light capture, while reflection probes supply local
specular lighting. These are probe-based approximations, not ray-traced or guaranteed leak-free GI.

Additional environment APIs are documented by feature rather than implementation order:

- [Physical atmosphere](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/atmosphere.md)
- [Aerial perspective and fog](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/aerial-perspective.md)
- [Time of day](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/time-of-day.md)
- [Night rendering](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/night-sky.md)
- [Environment shader and buffer layouts](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/environment-shader-reference.md)
- [Probe shader and buffer layouts](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/probes-shader-reference.md)

### GPU agents

`AgentSystem` keeps live simulation state on the GPU and exposes explicit frame and lockstep
scheduling. `RuleBrain` runs application WGSL; `MlpBrain` evaluates a supplied dense network.
Neither policy is restricted to predator/prey or flocking behaviour. Applications define their
observations, action semantics and additional simulation stages.

The built-in movement and neighbour-query model is currently planar (XZ). Neural inference is
limited to small dense networks; adding a network does not train it. Training tooling, general
ONNX import and an ONNX Runtime brain adapter are not currently part of the agents API.
Neighbour scans have a configurable per-cell budget: crowded cells can omit candidates, and
overflow-free operation is required for the documented same-device deterministic replay guarantee.
See the [agents guide](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/api/agents.md)
for supported shapes, buffer ownership, inspection and integration examples.

## Requirements and boundaries

- **WebGPU is required.** There is no WebGL fallback. Optional GPU features are negotiated against
  the device; performance and available resource limits vary by adapter and host.
- **This is a rendering library, not a complete game engine.** It does not provide a retained
  scene hierarchy, gameplay ECS, general rigid-body physics or automatic asset-to-scene assembly.
- **Readbacks are explicit.** Picking, inspection and frame capture involve GPU-to-CPU transfers;
  GPU-resident frame paths need not perform those operations every frame.
- **Text layout is not a shaping engine.** Complex-script shaping and bidirectional layout require
  additional application tooling.
- **Scale is workload-dependent.** Geometry, passes, lights, shadows, query density, uploads and
  GPU limits all affect throughput. There is no universal entity-count or frame-time guarantee.

Planned capabilities are kept separate from the feature list in the
[engine roadmap](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/roadmap.md).

## Examples and validation

- [Interactive demos](https://nullgraph.dev/)
- [Demo source](https://github.com/Vikas593-cloud/NullGraph-Test-Engine)
- [Application examples](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/agent/guide.md)
- [Validation tools and workflow](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/agent/validation.md)

From a repository checkout:

```sh
npm install
npm run build:lib
npm run test:core
npm run test:agents
npm run test:agent-kit
```

Real-GPU harnesses run through Node/Dawn and headless Chrome:

```sh
npm run test:gpu
npm run test:gpu:browser
npm run render -- test/gpu/scenes/quick-start.mjs --out cube.png
```

The GPU harnesses check numerical results and report uncaptured WebGPU errors. A passing run
applies to the tested host and workload, not every GPU or application. Read the validation guide
for setup, output reports and image inspection.

### Screenshots

| GPU culling | PBR materials | Skeletal animation |
|---|---|---|
| ![GPU culling demo](./showcases-images/GPUCulling.png) | ![PBR material demo](./showcases-images/rusty_metal.png) | ![Skeletal animation demo](./showcases-images/animation-post-processing.png) |

## Contributing and license

Maintainers and coding agents can start with
[Working on NullGraph](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/work/README.md).
Architecture notes are in the
[design principles](https://github.com/Vikas593-cloud/NullGraph/blob/main/docs/design-principles.md).

NullGraph is licensed under the [MIT license](./LICENSE).
