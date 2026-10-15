// Caller-owned bindings and scheduling: these snippets allocate nothing and declare no resources.
// Include BAKED_POSE_WGSL once, then the sampling/write snippets in compute, or the read snippet
// in a vertex hook. All offsets/strides are in vec4 rows, except influence counts and slot indices.

/** Shared row-major affine pose and math. No bindings. Normals use the linear part, not its inverse transpose. */
export const BAKED_POSE_WGSL = /* wgsl */ `
struct BakedPose { r0: vec4<f32>, r1: vec4<f32>, r2: vec4<f32> };
fn bakedMix(a: BakedPose, b: BakedPose, weight: f32) -> BakedPose {
    return BakedPose(mix(a.r0, b.r0, weight), mix(a.r1, b.r1, weight), mix(a.r2, b.r2, weight));
}
fn bakedPosition(pose: BakedPose, position: vec3<f32>) -> vec3<f32> {
    let p = vec4<f32>(position, 1.0);
    return vec3<f32>(dot(pose.r0, p), dot(pose.r1, p), dot(pose.r2, p));
}
fn bakedNormal(pose: BakedPose, normal: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(pose.r0.xyz, normal), dot(pose.r1.xyz, normal), dot(pose.r2.xyz, normal));
}
`;

/** Requires BAKED_POSE_WGSL and `bakedTransforms: array<vec4<f32>>` (storage read).
 * tableBase is a vec4-row offset; clip is (firstFrame, frames). Finite phase is measured in cycles.
 * Looping wraps (including negative phases); non-looping clamps, including its final sample.
 * Caller guarantees frames >= 1, influence < influences and valid table bounds. */
export const BAKED_SAMPLE_WGSL = /* wgsl */ `
fn bakedFramePose(tableBase: u32, influences: u32, frame: u32, influence: u32) -> BakedPose {
    let base = tableBase + (frame * influences + influence) * 3u;
    return BakedPose(bakedTransforms[base], bakedTransforms[base + 1u], bakedTransforms[base + 2u]);
}
fn bakedClipPose(tableBase: u32, influences: u32, clip: vec2<u32>, phase: f32, looping: bool, influence: u32) -> BakedPose {
    var frame = clamp(phase, 0.0, 1.0) * f32(clip.y - 1u);
    if (looping) { frame = fract(phase) * f32(clip.y); }
    let a = min(u32(frame), clip.y - 1u);
    var b = min(a + 1u, clip.y - 1u);
    if (looping) { b = (a + 1u) % clip.y; }
    return bakedMix(bakedFramePose(tableBase, influences, clip.x + a, influence),
        bakedFramePose(tableBase, influences, clip.x + b, influence), fract(frame));
}
`;

/** Requires BAKED_POSE_WGSL and `bakedPalette: array<vec4<f32>>` (storage read_write).
 * paletteBase is a vec4-row offset, stride is the number of influences reserved per visible slot.
 * App dispatches after culling, guards slot against the batch's GPU count, and samples/blends once
 * per (visible slot, influence), using the compacted instance's sourceIndex to choose its clips. */
export const BAKED_PALETTE_WRITE_WGSL = /* wgsl */ `
fn bakedWritePalette(paletteBase: u32, stride: u32, slot: u32, influence: u32, pose: BakedPose) {
    let base = paletteBase + (slot * stride + influence) * 3u;
    bakedPalette[base] = pose.r0;
    bakedPalette[base + 1u] = pose.r1;
    bakedPalette[base + 2u] = pose.r2;
}
`;

/** Requires BAKED_POSE_WGSL and `bakedPalette: array<vec4<f32>>` (storage read).
 * Read by compacted instance_index, NOT sourceIndex. Four indices and f32 weights come from the
 * skinned bake; the rigid path calls bakedPalettePose with its one part. Zero weights skip reads.
 * Caller guarantees slot/influence bounds, normalized weights, and palette stride >= influences. */
export const BAKED_PALETTE_READ_WGSL = /* wgsl */ `
fn bakedPalettePose(paletteBase: u32, stride: u32, slot: u32, influence: u32) -> BakedPose {
    let base = paletteBase + (slot * stride + influence) * 3u;
    return BakedPose(bakedPalette[base], bakedPalette[base + 1u], bakedPalette[base + 2u]);
}
fn bakedSkinPose(paletteBase: u32, stride: u32, slot: u32, joints: vec4<u32>, weights: vec4<f32>) -> BakedPose {
    var pose = BakedPose(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    for (var k = 0u; k < 4u; k++) {
        if (weights[k] > 0.0) {
            let p = bakedPalettePose(paletteBase, stride, slot, joints[k]);
            pose.r0 += p.r0 * weights[k];
            pose.r1 += p.r1 * weights[k];
            pose.r2 += p.r2 * weights[k];
        }
    }
    return pose;
}
`;