// Optional evaluation instrumentation, never a trainer input. No Node imports.
// A sample owns one tile and reads stable records after containment; catches are binned before
// respawn. No atomics on floats, cross-workgroup reductions, or in-loop CPU mappings.
export const DIAGNOSTICS_VERSION = 'coevo-diagnostics-v1';
export const DIAGNOSTIC_FIELDS = Object.freeze(['preyAlive', 'wolfAlive', 'preyWall', 'wolfWall',
    'preySpeedSum', 'wolfSpeedSum', 'wolfRest', 'nearestPreySum', 'nearestPreyCount',
    'overlapPairs', 'penetrationSum', 'maxPenetration', 'pairCount', 'reserved0', 'reserved1', 'reserved2']);
const fail = text => { throw new Error(`coevolve: diagnostics: ${text}`); };

export function diagnosticLayout(lay, { steps, stride, wallDistance = 1 } = {}) {
    if (!Number.isSafeInteger(steps) || steps < 1 || steps > 0xffffffff) fail('steps must be a positive u32');
    if (!Number.isSafeInteger(stride) || stride < 1 || stride > 0xffffffff) fail('stride must be a positive u32');
    if (lay.prey * steps >= 2 ** 32) fail('catch histogram could overflow u32');
    if (!(Number.isFinite(wallDistance) && wallDistance >= 0 && wallDistance < lay.L / 2)) fail('invalid wall distance');
    const samples = Math.ceil(steps / stride), words = samples * lay.arenas * 16;
    if (!Number.isSafeInteger(words) || words * 8 > 64 * 1024 * 1024) fail('sample and catch buffers exceed 64 MiB');
    return { version: DIAGNOSTICS_VERSION, steps, stride, wallDistance, samples, words,
        fields: [...DIAGNOSTIC_FIELDS], catchGrid: [4, 4],
        phase: 'state after containment; catches before respawn',
        sampling: 'state at zero-based tick multiples of stride; catches on every step, grouped by floor(tick/stride)' };
}

const constants = (lay, d) => `
const DIAG_STRIDE = ${d.stride}u;
const DIAG_STEPS = ${d.steps}u;
const DIAG_TILES = ${lay.arenas}u;
fn diagLo(tile: u32) -> vec2<f32> {
    return vec2<f32>(${Number(lay.origin[0]).toFixed(9)}, ${Number(lay.origin[1]).toFixed(9)}) +
        vec2<f32>(f32(tile % ${lay.cols}u), f32(tile / ${lay.cols}u)) * ${Number(lay.pitch).toFixed(9)};
}
fn diagIndex(tick: u32, tile: u32) -> u32 { return (tick / DIAG_STRIDE * DIAG_TILES + tile) * 16u; }
`;

export function diagnosticCatchWGSL(lay, d) {
    return `${constants(lay, d)}
@group(1) @binding(0) var<storage, read_write> catchLocations: array<atomic<u32>>;
fn step(i: u32, tick: u32) {
    let a = agents[i];
    if (tick >= DIAG_STEPS || a.ids.x != SPECIES_PREY || (a.ids.y & 257u) != 257u) { return; }
    let uv = clamp((a.posRadius.xz - diagLo(a.ids.w)) / ${Number(lay.L).toFixed(9)}, vec2<f32>(0.0), vec2<f32>(0.999999));
    let cell = vec2<u32>(uv * 4.0);
    atomicAdd(&catchLocations[diagIndex(tick, a.ids.w) + cell.y * 4u + cell.x], 1u);
}`;
}

export function diagnosticStateWGSL(lay, d) {
    return `${constants(lay, d)}
@group(1) @binding(0) var<storage, read_write> diagnosticSamples: array<f32>;
fn step(i: u32, tick: u32) {
    if (tick >= DIAG_STEPS || tick % DIAG_STRIDE != 0u) { return; }
    let tile = agents[i].ids.w;
    if (tile >= DIAG_TILES) { return; }
    // The app wrapper skips dead records. Elect the first living body, not a possibly dead wolf.
    for (var slot = 0u; slot < ${lay.perArena}u; slot++) {
        let index = select(${lay.wolfTotal}u + tile * ${lay.prey}u + slot - ${lay.wolves}u, tile * ${lay.wolves}u + slot, slot < ${lay.wolves}u);
        if (index == i) { break; }
        if ((agents[index].ids.y & 1u) != 0u && agents[index].ids.w == tile) { return; }
    }
    let base = diagIndex(tick, tile);
    var sums: array<f32, 16>;
    let lo = diagLo(tile);
    for (var slot = 0u; slot < ${lay.perArena}u; slot++) {
        let index = select(${lay.wolfTotal}u + tile * ${lay.prey}u + slot - ${lay.wolves}u, tile * ${lay.wolves}u + slot, slot < ${lay.wolves}u);
        let a = agents[index];
        if ((a.ids.y & 1u) == 0u || a.ids.w != tile) { continue; }
        let wolf = slot < ${lay.wolves}u;
        let s = select(0u, 1u, wolf);
        sums[s] += 1.0;
        let p = a.posRadius.xz - lo;
        let wall = min(min(p.x, p.y), min(${Number(lay.L).toFixed(9)} - p.x, ${Number(lay.L).toFixed(9)} - p.y)) - a.posRadius.w;
        if (wall <= ${Number(d.wallDistance).toFixed(9)}) { sums[2u + s] += 1.0; }
        sums[4u + s] += length(a.velHeading.xz);
        if (wolf && a.state.w > 0.0) { sums[6] += 1.0; }
        var nearest = 1e30;
        for (var otherSlot = 0u; otherSlot < ${lay.perArena}u; otherSlot++) {
            let j = select(${lay.wolfTotal}u + tile * ${lay.prey}u + otherSlot - ${lay.wolves}u, tile * ${lay.wolves}u + otherSlot, otherSlot < ${lay.wolves}u);
            let b = agents[j];
            if (j == index || (b.ids.y & 1u) == 0u || b.ids.w != tile) { continue; }
            let dist = distance(a.posRadius.xz, b.posRadius.xz);
            if (wolf && otherSlot >= ${lay.wolves}u) { nearest = min(nearest, dist); }
            if (j > index) {
                sums[12] += 1.0;
                let penetration = a.posRadius.w + b.posRadius.w - dist;
                if (penetration > 0.0) {
                    sums[9] += 1.0; sums[10] += penetration; sums[11] = max(sums[11], penetration);
                }
            }
        }
        if (wolf && nearest < 1e30) { sums[7] += nearest; sums[8] += 1.0; }
    }
    for (var k = 0u; k < 16u; k++) { diagnosticSamples[base + k] = sums[k]; }
}`;
}

/** Raw samples remain exported so summaries do not hide episode/time denominators. */
export function decodeDiagnostics(lay, d, sampleWords, catches) {
    if (sampleWords.length !== d.words || catches.length !== d.words) fail('incomplete readback');
    const values = new Float32Array(sampleWords.buffer, sampleWords.byteOffset, sampleWords.length);
    if (!values.every(Number.isFinite)) fail('nonfinite sample');
    const ratio = (a, b) => b ? a / b : null;
    const episodes = Array.from({ length: lay.arenas }, (_, tile) => {
        const history = Array.from({ length: d.samples }, (_, sample) => {
            const at = (sample * lay.arenas + tile) * 16, raw = Array.from(values.subarray(at, at + 16));
            return { tick: sample * d.stride, raw, catchLocations: Array.from(catches.subarray(at, at + 16)),
                preyWallShare: ratio(raw[2], raw[0]), wolfWallShare: ratio(raw[3], raw[1]),
                preyMeanSpeed: ratio(raw[4], raw[0]), wolfMeanSpeed: ratio(raw[5], raw[1]),
                wolfRestFraction: ratio(raw[6], raw[1]), nearestPreyMean: ratio(raw[7], raw[8]),
                overlapShare: ratio(raw[9], raw[12]), meanPenetration: ratio(raw[10], raw[9]), maxPenetration: raw[11] };
        });
        return { tile, history, catches: history.reduce((sum, h) => sum + h.catchLocations.reduce((a, b) => a + b, 0), 0) };
    });
    return { ...d, episodes };
}