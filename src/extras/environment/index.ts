// src/extras/environment/index.ts
//
// `null-graph/environment` — Phase 1 image-based lighting (docs/lighting-ibl-design.md) plus
// Phase 2 irradiance volumes, GPU probe capture, and reflection probes
// (docs/lighting-probes-design.md) — additive, same subpath.

export * from './types';
export * from './envMath';
export * from './envShaders';
export * from './EnvironmentSystem';
export * from './probeMath';
export * from './probeShaders';
export * from './ReflectionProbes';
export * from './ProbeVolume';
