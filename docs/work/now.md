# Now — 2026-10-15

## Position

- **Cycle C, C2 complete.** Steps 6–10 committed; step 5's scaling study remains deferred.
- **Cycle D, step D1: animated agents.** Design/phase 0 committed `edecdf4`; phase 1 (rigid bake,
  parser node metadata, tests and guide) committed **`f538ffe`** at the user's request.
- **Phase 2 built and validated, uncommitted.** Public instance-tier `vertexHook`, model-space
  deformation, dedicated bind-group validation, unchanged no-hook WGSL, guide and GPU pose test.
  [Design/results](../agents-animated-models-design.md). User called these phases “10.1/10.2”;
  they are D1.1/D1.2, not the already-completed live-switch phases.
- **Next: phase 3**, Models mode in the site: cow prey, fox wolves; animation from simulation
  state and one accumulated phase per agent. No machine-specific count caps. Field/toon/post
  remain phase 4; field scope defaults to Models only (design §11, item 5 still open).
  Skinned models wait for the user's model (phase 5).

## Validation and limitations

- Development build + TypeScript and IDE build pass. Visibility, loaders, shadows, picking,
  agents headless regressions pass; five new hook checks include all stock material families
  and the actual guide example compiled, bundled and run.
- Visibility GPU **8/8** on Node/NVIDIA, Node/AMD and Chrome/AMD, zero GPU errors.
  New baked-rig comparison: 1,874 covered pixels, zero differ by >2/255; no-hook control differs
  at 2,843 pixels. Exercises baseVertex, sparse sourceIndex, animated parent and normals.
- Full visibility entry **13,220 → 13,747 B min+gzip (+527 B)**. Core bundle unchanged.
- Phase 1 validated 193 site GLBs against the old parser. Parser normal/colour limitations
  unchanged; at 30 fps blended run poses deviate up to 36.7 mm between samples.
- Hook does not propagate automatically to shadow casters or posed picking. Group 2 conflicts
  with clustered lighting: choose a dedicated group (normally 3), matching WGSL declarations.

## Repository state

- **null-graph:** `master`; D1.1 committed, D1.2 uncommitted. `dist/` development build;
  tracked core dist files unchanged. `package-lock.json` ignored. No push.
- **Site:** `D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts`, untouched this session.
  Installed library remains step 10; no yalc push or deployment. Restart Vite after refresh.
- Optional GLB compatibility test needs `NG_GLB_COMPAT_DIR` and `NG_GLB_OLD_LOADER`; old build
  in local Claude temp session `20c36f59-ab5e-4974-8dfb-d4d710b7ad75`, `scratchpad/old/loaders.js`.
  Models stay local/ignored; never commit them.
