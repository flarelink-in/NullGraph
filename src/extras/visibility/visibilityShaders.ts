// src/extras/visibility/visibilityShaders.ts
//
// WGSL generators for the VisibilitySystem (docs/visibility-design.md). Every kernel is a
// complete, self-contained module so it can be parse-validated offline (wgsl_reflect) and so the
// engine's `layout: 'auto'` pipeline creation sees exactly the bindings the body uses.
//
// The cull math mirrors cullMath.ts line-for-line; the two are kept in lockstep by the shared
// test suite (test/visibility-cull.test.mjs), which pins the JS math and parses this WGSL.

// ---------------------------------------------------------------------------
// Shared chunks
// ---------------------------------------------------------------------------

/** Camera uniform — must match core/Camera.ts (108 floats) and materials/CommonChunks.ts. */
export const CAMERA_STRUCT_WGSL = /* wgsl */ `
struct Camera {
    viewProj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    invViewProj: mat4x4<f32>,
    invView: mat4x4<f32>,
    invProj: mat4x4<f32>,
    eyePos: vec3<f32>,
    simTime: f32,
    zNear: f32,
    zFar: f32,
    fovRads: f32,
    aspect: f32,
    _padEnd: vec4<f32>
};
`;

/** Instance record — 96 bytes, matches cullMath.ts INSTANCE_* layout (design doc §4.1). */
export const INSTANCE_STRUCT_WGSL = /* wgsl */ `
struct Instance {
    model: mat4x4<f32>,
    boundsCenter: vec4<f32>,
    meshIndex: u32,
    lodOverride: u32,
    flags: u32,
    _pad: u32,
};
`;

/** Meshlet descriptor — 64 bytes, matches geometry-meshlets-design.md §5.1. */
export const MESHLET_STRUCT_WGSL = /* wgsl */ `
struct Meshlet {
    boundsCenter: vec4<f32>,
    coneApex: vec4<f32>,
    coneAxis: vec4<f32>,
    offsets: vec4<u32>,
};
`;

/** MeshEntry / MeshLOD — the mesh → LOD → meshlet map (geometry doc §5.4). */
export const MESH_TABLE_STRUCT_WGSL = /* wgsl */ `
struct MeshEntry {
    lodOffset: u32,
    lodCount: u32,
    _pad0: u32,
    _pad1: u32,
};
struct MeshLOD {
    meshletOffset: u32,
    meshletCount: u32,
    error: f32,
    _pad: f32,
};
`;

/**
 * Per-draw cull parameters uniform (48 bytes — shadows-atlas-design.md §7.2 grew it from 32).
 * `hizRect` remaps the occlusion test's UVs into a sub-rectangle of the Hi-Z pyramid so a
 * shadow view samples only its own atlas tile; the main view passes the identity (1,1,0,0),
 * which leaves the v1.0.5 kernel bit-for-bit unchanged (§10 U5).
 */
export const CULL_PARAMS_WGSL = /* wgsl */ `
struct CullParams {
    instanceCount: u32,
    meshIndex: u32,
    hizMaxLevel: u32,
    flags: u32,
    screenSize: vec2<f32>,
    lodErrorPx: f32,
    _pad: f32,
    hizRect: vec4<f32>,
};
`;

/** Frustum + sphere + cone math (design doc §6.1–§6.3). No textures, safe everywhere. */
export const CULL_CORE_WGSL = /* wgsl */ `
fn frustumPlanes(vp: mat4x4<f32>) -> array<vec4<f32>, 6> {
    let r0 = vec4<f32>(vp[0][0], vp[1][0], vp[2][0], vp[3][0]);
    let r1 = vec4<f32>(vp[0][1], vp[1][1], vp[2][1], vp[3][1]);
    let r2 = vec4<f32>(vp[0][2], vp[1][2], vp[2][2], vp[3][2]);
    let r3 = vec4<f32>(vp[0][3], vp[1][3], vp[2][3], vp[3][3]);
    // ZO clip volume (WebGPU): the near plane is r2 alone (clip z >= 0), NOT r3 + r2
    // (clip z >= -w, the OpenGL form). See cullMath.ts convention 2.
    var p = array<vec4<f32>, 6>(r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2);
    for (var i = 0u; i < 6u; i = i + 1u) {
        p[i] = p[i] / length(p[i].xyz);
    }
    return p;
}

fn sphereInFrustum(p: array<vec4<f32>, 6>, c: vec3<f32>, r: f32) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        if (dot(p[i].xyz, c) + p[i].w < -r) { return false; }
    }
    return true;
}

fn worldSphereFromModel(model: mat4x4<f32>, boundsCenter: vec4<f32>) -> vec4<f32> {
    let c = (model * vec4<f32>(boundsCenter.xyz, 1.0)).xyz;
    let sx = length(model[0].xyz);
    let sy = length(model[1].xyz);
    let sz = length(model[2].xyz);
    return vec4<f32>(c, boundsCenter.w * max(sx, max(sy, sz)));
}

// Normal matrix = inverse-transpose of the upper 3x3. The inverse's ROWS are cross(col)/det,
// which are exactly the normal matrix's COLUMNS, so we build it directly (design doc §6.2).
fn normalMatrixFromModel(model: mat4x4<f32>) -> mat3x3<f32> {
    let c0 = model[0].xyz;
    let c1 = model[1].xyz;
    let c2 = model[2].xyz;
    let invDet = 1.0 / dot(c0, cross(c1, c2));
    return mat3x3<f32>(cross(c1, c2) * invDet, cross(c2, c0) * invDet, cross(c0, c1) * invDet);
}

fn coneIsBackfacing(axisW: vec3<f32>, cutoff: f32, apexW: vec3<f32>, eye: vec3<f32>) -> bool {
    if (cutoff >= 2.0) { return false; }
    let d = normalize(apexW - eye);
    return dot(normalize(axisW), d) >= cutoff;
}
`;

/**
 * Hi-Z occlusion test (design doc §6.4). Only emitted when occlusion is enabled, because it
 * references the pyramid texture binding. Conservative: rejects only when the sphere's nearest
 * point is provably behind the farthest occluder over its footprint. The near-plane guard keeps
 * anything crossing the near plane visible.
 *
 * ZERO_IS_NO_HISTORY — a footprint whose max depth is 0 means "no depth history yet", not "a wall
 * at the near plane". The pyramid and a never-rendered depth source both start zeroed (WebGPU
 * zero-initializes), and a REAL max-reduced depth is 0 only if geometry touches the near plane
 * across the whole footprint. Treating zero as occluding made frame 0 — and the first frame after
 * every resize — cull EVERYTHING, the opposite of visibility-design §5.2 ("frame 0 occludes
 * nothing"). Found by test/visibility-gpu V3a (Cycle A, docs/handoff.md). Unknown means draw.
 * (Explained here rather than in the WGSL: comments inside the template ship in the bundle.)
 */
export const HIZ_OCCLUSION_WGSL = /* wgsl */ `
fn occludedByHiZ(c: vec3<f32>, r: f32, cam: Camera, screenSize: vec2<f32>, hizMaxLevel: u32, hiz: texture_2d<f32>, hizRect: vec4<f32>) -> bool {
    let vc = (cam.view * vec4<f32>(c, 1.0)).xyz;
    let zView = -vc.z;
    if (zView - r < cam.zNear) { return false; }

    let p = cam.proj;
    let cx = p * vec4<f32>(vc, 1.0);
    let ex = p * vec4<f32>(vc + vec3<f32>(r, 0.0, 0.0), 1.0);
    let ey = p * vec4<f32>(vc + vec3<f32>(0.0, r, 0.0), 1.0);
    let ndcC = cx.xy / cx.w;
    let rx = abs(ex.x / ex.w - ndcC.x);
    let ry = abs(ey.y / ey.w - ndcC.y);
    // NDC -> uv, undoing the clip-space Y flip (clip +Y up, texture +V down).
    let uvMin = (vec2<f32>(ndcC.x - rx, -(ndcC.y + ry)) * 0.5) + vec2<f32>(0.5);
    let uvMax = (vec2<f32>(ndcC.x + rx, -(ndcC.y - ry)) * 0.5) + vec2<f32>(0.5);

    // shadows-atlas-design.md §7.2 — remap into the view's atlas tile before sampling. The
    // pyramid's texel space is the whole atlas; a shadow view owns only hizRect of it. The
    // identity (1,1,0,0) leaves the main view bit-for-bit unchanged (§10 U5).
    let uvMinA = uvMin * hizRect.xy + hizRect.zw;
    let uvMaxA = uvMax * hizRect.xy + hizRect.zw;

    let sizePx = max(uvMax.x - uvMin.x, uvMax.y - uvMin.y) * max(screenSize.x, screenSize.y);
    let level = clamp(u32(ceil(log2(max(sizePx, 1.0)))), 0u, hizMaxLevel);
    let dimU = textureDimensions(hiz, level);
    let dim = vec2<f32>(dimU);

    // Nearest depth of the sphere along view -Z. No remap: the depth buffer stores this same
    // projection's raw NDC z, so clipZ/clipW compares like-for-like.
    let nearPt = p * vec4<f32>(vc.xy, vc.z + r, 1.0);
    let ndcNear = nearPt.z / nearPt.w;

    let t0 = min(vec2<u32>(clamp(uvMin, vec2<f32>(0.0), vec2<f32>(1.0)) * dim), dimU - 1u);
    let t1 = min(vec2<u32>(clamp(uvMax, vec2<f32>(0.0), vec2<f32>(1.0)) * dim), dimU - 1u);
    var occ = 0.0;
    occ = max(occ, textureLoad(hiz, t0, level).r);
    occ = max(occ, textureLoad(hiz, vec2<u32>(t1.x, t0.y), level).r);
    occ = max(occ, textureLoad(hiz, vec2<u32>(t0.x, t1.y), level).r);
    occ = max(occ, textureLoad(hiz, t1, level).r);

    if (occ <= 0.0) { return false; } // zero = no depth history: draw (see ZERO_IS_NO_HISTORY)
    return ndcNear > occ;
}
`;

// ---------------------------------------------------------------------------
// Hi-Z pyramid build (design doc §5.1)
// ---------------------------------------------------------------------------

/**
 * Compute kernel that builds one Hi-Z mip. `fromDepth` selects the mip-0 variant (reads the
 * app's samplable depth texture as `texture_depth_2d`); otherwise it reads the previous pyramid
 * mip as `texture_2d<f32>`. Both write a single `r32float` storage mip, taking the MAX of the
 * 2×2 source texels (max because depth stores the NEAREST surface; see §5). Source reads are
 * clamped so odd source dimensions stay conservative.
 */
export function buildHiZMipWGSL(fromDepth: boolean): string {
    const srcName = fromDepth ? 'srcDepth' : 'srcMip';
    const srcDecl = fromDepth
        ? `@group(0) @binding(0) var srcDepth: texture_depth_2d;`
        : `@group(0) @binding(0) var srcMip: texture_2d<f32>;`;
    // texture_depth_2d loads a bare f32; texture_2d<f32> loads a vec4 we reduce via .r.
    const load = (coord: string): string =>
        fromDepth ? `textureLoad(srcDepth, ${coord}, 0)` : `textureLoad(srcMip, ${coord}, 0).r`;
    return /* wgsl */ `
${srcDecl}
@group(0) @binding(1) var dstMip: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst = gid.xy;
    let dstDim = textureDimensions(dstMip);
    if (dst.x >= dstDim.x || dst.y >= dstDim.y) { return; }
    let s = dst * 2u;
    let srcDim = textureDimensions(${srcName});
    let c00 = min(s + vec2<u32>(0u, 0u), srcDim - 1u);
    let c10 = min(s + vec2<u32>(1u, 0u), srcDim - 1u);
    let c01 = min(s + vec2<u32>(0u, 1u), srcDim - 1u);
    let c11 = min(s + vec2<u32>(1u, 1u), srcDim - 1u);
    let d0 = ${load('c00')};
    let d1 = ${load('c10')};
    let d2 = ${load('c01')};
    let d3 = ${load('c11')};
    textureStore(dstMip, dst, vec4<f32>(max(max(d0, d1), max(d2, d3)), 0.0, 0.0, 0.0));
}
`;
}


// ---------------------------------------------------------------------------
// Instance tier (design doc §7.1)
// ---------------------------------------------------------------------------

export interface CullShaderOptions {
    /** Include the Hi-Z occlusion test + pyramid binding. Default false. */
    occlusion?: boolean;
}

/**
 * Instance-tier cull: one thread per instance, routed to this batch's mesh by `params.meshIndex`.
 * Survivors are compacted into the batch's storage buffer and counted by an atomic bump of
 * `drawArgs[1]` (the indirect draw's instanceCount). Frustum always; Hi-Z occlusion when enabled.
 *
 * Bindings (group 0): 0 camera, 1 source instances, 2 out instances, 3 drawArgs, 4 params,
 * [5 hiz when occlusion].
 */
export function buildInstanceCullWGSL(opts: CullShaderOptions = {}): string {
    const occlusion = opts.occlusion ?? false;
    return /* wgsl */ `
${CAMERA_STRUCT_WGSL}
${INSTANCE_STRUCT_WGSL}
${CULL_PARAMS_WGSL}
${CULL_CORE_WGSL}
${occlusion ? HIZ_OCCLUSION_WGSL : ''}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<storage, read_write> outInstances: array<Instance>;
@group(0) @binding(3) var<storage, read_write> drawArgs: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: CullParams;
${occlusion ? '@group(0) @binding(5) var hiz: texture_2d<f32>;' : ''}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.instanceCount) { return; }
    let inst = instances[i];
    if (inst.meshIndex != params.meshIndex) { return; }

    let ws = worldSphereFromModel(inst.model, inst.boundsCenter);
    let planes = frustumPlanes(camera.viewProj);
    if (!sphereInFrustum(planes, ws.xyz, ws.w)) { return; }

    ${occlusion ? `if ((inst.flags & 1u) == 0u) {
        if (occludedByHiZ(ws.xyz, ws.w, camera, params.screenSize, params.hizMaxLevel, hiz, params.hizRect)) { return; }
    }` : ''}

    let slot = atomicAdd(&drawArgs[1], 1u);
    outInstances[slot] = inst;
}
`;
}

// ---------------------------------------------------------------------------
// Meshlet tier (design doc §7.2) — two compute passes
// ---------------------------------------------------------------------------

/**
 * Meshlet-tier pass 1: coarse instance cull + LOD selection. Survivors are written to
 * `visibleInstances` as `{instanceId, lodMeshletOffset, lodMeshletCount}`, counted by
 * `counters[0]`. LOD is chosen from the mesh's MeshLOD.error (design doc §8); `lodOverride`
 * (relative level) short-circuits when set.
 *
 * Bindings (group 0): 0 camera, 1 instances, 2 visibleInstances, 3 counters, 4 meshTable,
 * 5 meshLODs, 6 params, [7 hiz when occlusion].
 */
export function buildMeshletInstanceCullWGSL(opts: CullShaderOptions = {}): string {
    const occlusion = opts.occlusion ?? false;
    return /* wgsl */ `
${CAMERA_STRUCT_WGSL}
${INSTANCE_STRUCT_WGSL}
${MESH_TABLE_STRUCT_WGSL}
${CULL_PARAMS_WGSL}
${CULL_CORE_WGSL}
${occlusion ? HIZ_OCCLUSION_WGSL : ''}

struct VisibleInstance {
    instanceId: u32,
    lodMeshletOffset: u32,
    lodMeshletCount: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<storage, read_write> visibleInstances: array<VisibleInstance>;
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> meshTable: array<MeshEntry>;
@group(0) @binding(5) var<storage, read> meshLODs: array<MeshLOD>;
@group(0) @binding(6) var<uniform> params: CullParams;
${occlusion ? '@group(0) @binding(7) var hiz: texture_2d<f32>;' : ''}

fn selectLodIndex(inst: Instance, mesh: MeshEntry, wc: vec3<f32>, r: f32) -> u32 {
    if (inst.lodOverride != 0xFFFFFFFFu) {
        return mesh.lodOffset + min(inst.lodOverride, mesh.lodCount - 1u);
    }
    let dist = max(length(wc - camera.eyePos) - r, camera.zNear);
    let pxPerUnit = params.screenSize.y / (2.0 * dist * tan(camera.fovRads * 0.5));
    var chosen = mesh.lodOffset;
    for (var k = 0u; k < mesh.lodCount; k = k + 1u) {
        let lod = meshLODs[mesh.lodOffset + k];
        if (lod.error * pxPerUnit <= params.lodErrorPx) { chosen = mesh.lodOffset + k; }
        else { break; }
    }
    return chosen;
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.instanceCount) { return; }
    let inst = instances[i];

    let ws = worldSphereFromModel(inst.model, inst.boundsCenter);
    let planes = frustumPlanes(camera.viewProj);
    if (!sphereInFrustum(planes, ws.xyz, ws.w)) { return; }

    ${occlusion ? `if ((inst.flags & 1u) == 0u) {
        if (occludedByHiZ(ws.xyz, ws.w, camera, params.screenSize, params.hizMaxLevel, hiz, params.hizRect)) { return; }
    }` : ''}

    let mesh = meshTable[inst.meshIndex];
    let lodIndex = selectLodIndex(inst, mesh, ws.xyz, ws.w);
    let lod = meshLODs[lodIndex];

    let slot = atomicAdd(&counters[0], 1u);
    visibleInstances[slot] = VisibleInstance(i, lod.meshletOffset, lod.meshletCount, 0u);
}
`;
}

export interface MeshletCullShaderOptions extends CullShaderOptions {
    /** Include the meshlet normal-cone (backface cluster) test. Default true. */
    cone?: boolean;
}

/**
 * Meshlet-tier pass 2: per-meshlet cull + compaction. Dispatched as a 2D grid
 * (visibleInstanceSlot × meshletWithinLod); idle lanes early-out past `counters[0]` /
 * `lodMeshletCount`. Survivors are written as `{instanceId, meshletId}` into the draw batch's
 * storage buffer and counted by `drawArgs[1]`. Cone + frustum + Hi-Z each on the meshlet's own
 * bounds, transformed by the instance model.
 *
 * Bindings (group 0): 0 camera, 1 instances, 2 visibleInstances, 3 meshlets, 4 counters,
 * 5 outMeshlets (batch storage), 6 drawArgs (batch indirect), 7 params, [8 hiz when occlusion].
 */
export function buildMeshletCullWGSL(opts: MeshletCullShaderOptions = {}): string {
    const occlusion = opts.occlusion ?? false;
    const cone = opts.cone ?? true;
    // `params` is only referenced by the occlusion path, so it (and its struct) are declared
    // only when occlusion is on — `layout: 'auto'` prunes an unused binding and the engine's
    // bind-group creation would then throw on the extra entry.
    return /* wgsl */ `
${CAMERA_STRUCT_WGSL}
${INSTANCE_STRUCT_WGSL}
${MESHLET_STRUCT_WGSL}
${occlusion ? CULL_PARAMS_WGSL : ''}
${CULL_CORE_WGSL}
${occlusion ? HIZ_OCCLUSION_WGSL : ''}

struct VisibleInstance {
    instanceId: u32,
    lodMeshletOffset: u32,
    lodMeshletCount: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<storage, read> visibleInstances: array<VisibleInstance>;
@group(0) @binding(3) var<storage, read> meshlets: array<Meshlet>;
@group(0) @binding(4) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> outMeshlets: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read_write> drawArgs: array<atomic<u32>>;
${occlusion ? '@group(0) @binding(7) var<uniform> params: CullParams;' : ''}
${occlusion ? '@group(0) @binding(8) var hiz: texture_2d<f32>;' : ''}


@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    let j = gid.y;
    let visibleCount = atomicLoad(&counters[0]);
    if (slot >= visibleCount) { return; }
    let vi = visibleInstances[slot];
    if (j >= vi.lodMeshletCount) { return; }

    let meshletIndex = vi.lodMeshletOffset + j;
    let m = meshlets[meshletIndex];
    let inst = instances[vi.instanceId];

    ${cone ? `if ((inst.flags & 2u) == 0u) {
        let apexW = (inst.model * vec4<f32>(m.coneApex.xyz, 1.0)).xyz;
        let axisW = normalMatrixFromModel(inst.model) * m.coneAxis.xyz;
        if (coneIsBackfacing(axisW, m.coneAxis.w, apexW, camera.eyePos)) { return; }
    }` : ''}

    let ws = worldSphereFromModel(inst.model, m.boundsCenter);
    let planes = frustumPlanes(camera.viewProj);
    if (!sphereInFrustum(planes, ws.xyz, ws.w)) { return; }

    ${occlusion ? `if ((inst.flags & 1u) == 0u) {
        if (occludedByHiZ(ws.xyz, ws.w, camera, params.screenSize, params.hizMaxLevel, hiz, params.hizRect)) { return; }
    }` : ''}

    let outSlot = atomicAdd(&drawArgs[1], 1u);
    outMeshlets[outSlot] = vec2<u32>(vi.instanceId, meshletIndex);
}
`;
}

// ---------------------------------------------------------------------------
// Draw shaders
// ---------------------------------------------------------------------------

export interface DrawShaderOptions {
    /** Fragment shader body. Receives a `VertexOut` (`in`) and returns `vec4<f32>`. */
    fragmentBody?: string;
    /** Extra group-1+ bindings the fragment needs (e.g. a material). Emitted verbatim. */
    extraBindings?: string;
}

const DEFAULT_FRAGMENT = /* wgsl */ `
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.worldNormal);
    let hemi = 0.5 + 0.5 * n.y;
    let baseColor = in.uv.x * 0.0 + vec3<f32>(0.7, 0.7, 0.75); // neutral; materials override
    let lit = baseColor * (0.25 + 0.75 * hemi);
    return vec4<f32>(lit, 1.0);
}
`;

/**
 * Field-identical to `materials/CommonChunks.ts` `VertexOut` — deliberately so. A built-in
 * material's fragment shader (`buildLambertShader` et al.) reads `in.worldNormal` / `in.ecsColor`,
 * so matching the field NAMES and locations is what lets a stock material fragment drop into a
 * visibility draw batch unchanged (api/visibility-shader-reference.md §4). Renaming a field here
 * silently breaks every material bridge, so keep the two structs in lockstep.
 *
 * `ecsColor` is the per-instance tint. The visibility `Instance` record carries no color channel
 * (design doc §4.1), so it is filled with white — `material.baseColor * instanceColor` then
 * reduces to `material.baseColor`, which is the correct no-tint behavior.
 */
const VERTEX_OUT_WGSL = /* wgsl */ `
struct VertexOut {
    @builtin(position) clipPos: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) worldNormal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) ecsColor: vec3<f32>,
};
`;

/**
 * Instance-tier draw. A standard indexed-instanced draw (drawn via `drawIndexedIndirect`):
 * vertices/indices come from the mega buffers bound by `setBatchGeometry`, and the per-instance
 * transform is read from the compacted survivors in the batch's storage buffer
 * (`@builtin(instance_index)`). The default vertex attributes match `StandardLayout`
 * (position@0, normal@1, uv@2); supply `vertexAttributesWGSL` to match a different layout.
 */
export function buildInstanceDrawWGSL(opts: DrawShaderOptions & { vertexAttributesWGSL?: string } = {}): string {
    const attrs = opts.vertexAttributesWGSL ?? `
        @location(0) pos: vec3<f32>,
        @location(1) normal: vec3<f32>,
        @location(2) uv: vec2<f32>,`;
    return /* wgsl */ `
${CAMERA_STRUCT_WGSL}
${INSTANCE_STRUCT_WGSL}
${CULL_CORE_WGSL}
${VERTEX_OUT_WGSL}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
${opts.extraBindings ?? ''}

@vertex
fn vs_main(${attrs}
    @builtin(instance_index) ii: u32,
) -> VertexOut {
    let inst = instances[ii];
    let world = inst.model * vec4<f32>(pos, 1.0);
    var out: VertexOut;
    out.clipPos = camera.viewProj * world;
    out.worldPos = world.xyz;
    out.worldNormal = normalize(normalMatrixFromModel(inst.model) * normal);
    out.uv = uv;
    out.ecsColor = vec3<f32>(1.0);   // no per-instance tint; see VERTEX_OUT_WGSL
    return out;
}

${opts.fragmentBody ?? DEFAULT_FRAGMENT}
`;
}

export interface MeshletDrawShaderOptions extends DrawShaderOptions {
    /** Mega vertex stride in floats (from the MegabufferBuilder). Required. */
    strideFloats: number;
    /** Float offset of position within a vertex. Default 0. */
    positionOffsetFloats?: number;
    /** Float offset of normal within a vertex. Default 3. */
    normalOffsetFloats?: number;
    /** Float offset of uv within a vertex. Default 6. */
    uvOffsetFloats?: number;
}

/**
 * Meshlet-tier draw: meshlet-as-instance programmable vertex pulling (design doc §7.3). Drawn
 * with a static index buffer `[0..371]` and `drawIndexedIndirect([372, visibleMeshletCount, ...])`.
 * `@builtin(instance_index)` selects the visible meshlet; `@builtin(vertex_index)` (the fetched
 * index, 0..371) selects the corner within it. Vertices past the meshlet's triangle count emit a
 * degenerate, clipped position.
 *
 * Group 0 is fixed by the engine (camera@0, batch.storageBuffer@1 = the visible list). Group 1 is
 * the read-only geometry + instance data bound via `setRenderPassBindGroup`.
 */
export function buildMeshletDrawWGSL(opts: MeshletDrawShaderOptions): string {
    const stride = opts.strideFloats;
    const pos = opts.positionOffsetFloats ?? 0;
    const nrm = opts.normalOffsetFloats ?? 3;
    const uv = opts.uvOffsetFloats ?? 6;
    return /* wgsl */ `
${CAMERA_STRUCT_WGSL}
${INSTANCE_STRUCT_WGSL}
${MESHLET_STRUCT_WGSL}
${CULL_CORE_WGSL}
${VERTEX_OUT_WGSL}

const MEGA_STRIDE: u32 = ${stride}u;
const POS_OFF: u32 = ${pos}u;
const NRM_OFF: u32 = ${nrm}u;
const UV_OFF: u32 = ${uv}u;

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> visible: array<vec2<u32>>;

@group(1) @binding(0) var<storage, read> meshlets: array<Meshlet>;
@group(1) @binding(1) var<storage, read> meshletVertices: array<u32>;
@group(1) @binding(2) var<storage, read> meshletTriangles: array<u32>;
@group(1) @binding(3) var<storage, read> megaVertices: array<f32>;
@group(1) @binding(4) var<storage, read> instances: array<Instance>;
${opts.extraBindings ?? ''}

@vertex
fn vs_main(@builtin(instance_index) mi: u32, @builtin(vertex_index) vi: u32) -> VertexOut {
    let entry = visible[mi];
    let inst = instances[entry.x];
    let m = meshlets[entry.y];
    var out: VertexOut;
    let localTri = vi / 3u;
    if (localTri >= m.offsets.w) {
        out.clipPos = vec4<f32>(-2.0, -2.0, -2.0, 1.0);
        out.worldPos = vec3<f32>(0.0);
        out.worldNormal = vec3<f32>(0.0, 1.0, 0.0);
        out.uv = vec2<f32>(0.0);
        out.ecsColor = vec3<f32>(1.0);
        return out;
    }
    let corner = vi % 3u;
    let packed = meshletTriangles[m.offsets.y + localTri];
    let local = (packed >> (corner * 8u)) & 0xFFu;
    let gv = meshletVertices[m.offsets.x + local];
    let base = gv * MEGA_STRIDE;
    let pos = vec3<f32>(megaVertices[base + POS_OFF], megaVertices[base + POS_OFF + 1u], megaVertices[base + POS_OFF + 2u]);
    let nrm = vec3<f32>(megaVertices[base + NRM_OFF], megaVertices[base + NRM_OFF + 1u], megaVertices[base + NRM_OFF + 2u]);
    let uv = vec2<f32>(megaVertices[base + UV_OFF], megaVertices[base + UV_OFF + 1u]);

    let world = inst.model * vec4<f32>(pos, 1.0);
    out.clipPos = camera.viewProj * world;
    out.worldPos = world.xyz;
    out.worldNormal = normalize(normalMatrixFromModel(inst.model) * nrm);
    out.uv = uv;
    out.ecsColor = vec3<f32>(1.0);   // no per-instance tint; see VERTEX_OUT_WGSL
    return out;
}

${opts.fragmentBody ?? DEFAULT_FRAGMENT}
`;
}
