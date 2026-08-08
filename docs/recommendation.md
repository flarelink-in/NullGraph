# WebGPU Update Recommendation

This document outlines the recommendations for updating the `@webgpu/types` package in `null-graph` from `^0.1.69` to the latest version. It covers the benefits, which modules we can enhance with the new features, and potential breaking changes to watch out for.

---

## Implementation Status (2026-08-07)

This document has been actioned. Summary of what shipped and what was deliberately deferred, in
keeping with null-graph's "stay in sync with WebGPU progression, without legacy/fallback bloat" stance:

| Item | Status | Notes |
|---|---|---|
| `@webgpu/types` bump | ✅ Done | `^0.1.69` → `^0.1.71`. Feature-name types were already present at 0.1.69, so this is low-risk. |
| `wgsl_reflect` | ✅ Removed | It was a **dead devDependency** — imported nowhere in `src/`, `build.js`, or tests. Removing it dissolves the compatibility concern in §3 outright. |
| Feature negotiation + capability registry | ✅ Done | `WebGPUContext` auto-negotiates `PROGRESSION_FEATURES`; results exposed as `engine.capabilities`. See [Core API](./api/core.md#feature-negotiation--capabilities). |
| Subgroup-accelerated clustered binning | ✅ Done | `clusteredComputeShader.ts` emits a subgroup-coalesced kernel when `capabilities.subgroups` is set; the atomic kernel remains the universal baseline. |
| Float16 light layout | ⏸️ Deferred | Capability **detected** (`capabilities.shaderF16`). Not adopted: `shader-f16` isn't universal, so an f16 storage layout means a feature-gated *second* code path across all 7 proxies and every light WGSL struct — dual-path bloat for a marginal win on a small buffer. Adopt when native `shader-f16` is ubiquitous. |
| Read-write storage-texture post | ⏸️ Deferred | Capability **detected** (`capabilities.bgra8unormStorage`). The PostChain is fragment-based; a compute rewrite is a separate effort. Plumbing is ready — `TransientPool.alloc` already accepts `extraUsage` for `STORAGE_BINDING`. |
| Dual-source blending in `buildPBRShader` | ⏸️ Deferred | Capability **detected** (`capabilities.dualSourceBlending`). Niche (OIT/advanced transparency); no consuming use case yet. |

The remainder of this document is the original analysis, retained for context.

---

## 1. Benefits of Updating

WebGPU and WGSL have been evolving rapidly. Updating to the latest types unlocks several advanced GPU features that align perfectly with `null-graph`'s Data-Oriented, compute-heavy architecture:

- **Subgroup Operations (`subgroups`, `subgroups-f16`)**: Subgroups allow invocations within a compute workgroup to share data instantly without the overhead of `workgroup` shared memory or atomics. This is a game-changer for the clustered/tiled forward culling compute stages, allowing much faster light binning.
- **Float16 Capabilities (`shader-f16`)**: Reduces memory footprint and bandwidth. If applied to our light proxies, we could halve the size of the 16-float light layout (from 64 bytes to 32 bytes), drastically reducing the `lightStorageBuffer` upload time per frame.
- **Read-Write Storage Textures**: Modern WebGPU supports `read_write` access on storage textures. This would simplify compute-based post-processing pipelines (like Bloom or auto-exposure) that currently require ping-ponging between read-only and write-only textures.
- **Dual-Source Blending (`dual-source-blending`)**: Enables advanced transparency effects (like Order-Independent Transparency) which could be exposed via the `buildPBRShader` material builder.

## 2. Modules We Can Edit

To take full advantage of these new features, we can update the following modules:

### `src/core/WebGPUContext.ts`
- **Feature Negotiation**: Update `init()` to negotiate for the new optional features. We can probe the adapter for `'subgroups'`, `'shader-f16'`, and `'bgra8unorm-storage'` and automatically enable them if available.
- **Canvas Configuration**: Ensure the `GPUCanvasContext.configure` options align with the latest spec.

### `src/extras/lights/`
- **Compute Binning (`clusteredComputeShader.ts`)**: Rewrite the compute kernel to use WGSL subgroup operations (`subgroupAdd`, `subgroupBallot`) instead of relying purely on the `globalCounter` atomic bumping. This will significantly reduce atomic contention when thousands of lights map to the same cluster.
- **Light Proxies (`light-proxies.ts` & others)**: If we adopt `shader-f16`, the proxies could be refactored to wrap a `Float16Array` (or a `Uint16Array` with math utilities) instead of a `Float32Array`. 

### `src/core/RenderPipeline.ts` & `src/extras/post/`
- Implement new passes that utilize read-write storage textures for post-processing, avoiding the creation of intermediate texture views.
- Extend `RenderPassNode` to optionally handle `dual-source-blending` attachment setups.

### `package.json`
- Bump `@webgpu/types` to `latest`.
- We may also need to update `wgsl_reflect` to ensure it can parse the new WGSL syntax like `enable subgroups;`.

## 3. Potential Breaking Changes

Updating the types and implementing these features may introduce the following breaking changes or regressions:

> [!WARNING]
> **WGSL Syntax Updates**
> Newer `@webgpu/types` often come alongside stricter WGSL compiler implementations in browsers. We may see new shader compilation errors for previously ignored syntax (e.g., implicit type conversions, strict scoping).

> [!CAUTION]
> **Float16 Array Support in JS**
> If we transition our Light proxies to use 16-bit floats to save bandwidth, JavaScript's native `Float16Array` is a relatively new addition (ECMAScript 2024). We may need to provide a polyfill or use `DataView` for broader browser compatibility, changing the `target` array type in `BaseLightCursor`.

> [!IMPORTANT]
> **`wgsl_reflect` Compatibility** — ✅ **Resolved by removal.** `wgsl_reflect` was a dead
> devDependency (no runtime reflection happens anywhere in `src/`), so it was deleted rather than
> upgraded. Bind-group layouts come from `pipeline.getBindGroupLayout()` (WebGPU's own reflection),
> which parses `enable subgroups;` shaders natively — there is nothing left to keep compatible.

> [!NOTE]
> **Device Limits and Fallbacks**
> Some features like subgroups are not guaranteed on all hardware (especially older mobile GPUs). Any compute stage leveraging them must have a fallback implementation using standard workgroup shared memory.
