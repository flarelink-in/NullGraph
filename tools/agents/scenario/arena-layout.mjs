// Step 14.1: explicit geometry, not measured training presets. No GPU allocations here.
const fail = message => { throw new Error(`coevolve: ${message}`); };
export function arenaLayout({ arenas, wolves, prey, L, gap, cols, contacts = null }) {
    for (const [name, value] of Object.entries({ arenas, wolves, prey, cols }))
        if (!Number.isSafeInteger(value) || value < 1) fail(`${name} ${value} must be a positive safe integer`);
    if (cols > arenas) fail(`cols ${cols} exceeds arenas ${arenas}`);
    for (const [name, value] of Object.entries({ L, gap }))
        if (!Number.isFinite(value) || value <= 0) fail(`${name} ${value} must be positive and finite`);
    if (prey % 64 || arenas * wolves % 64) fail('prey must be a multiple of 64 and arenas*wolves a multiple of 64');
    if (L <= 2) fail(`L ${L} leaves no spawn inset`);
    const J = contacts?.iterations ?? 0, push = contacts?.maxPush ?? 0;
    if (contacts && (!Number.isInteger(J) || J < 1 || J > 8 || !Number.isFinite(push) || push <= 0))
        fail('contacts require iterations 1..8 and positive finite maxPush');
    const requiredGap = Math.max(8, 1.2 + 2 * J * push);
    if (gap <= requiredGap) fail(`gap ${gap} must exceed ${requiredGap}`);
    const perArena = wolves + prey, count = arenas * perArena, wolfTotal = arenas * wolves;
    if (!Number.isSafeInteger(count) || count >= 2 ** 32 || count * 16 >= 2 ** 32) fail('record/index product overflows u32');
    const rows = Math.ceil(arenas / cols), pitch = L + gap;
    const spanX = cols * pitch - gap, spanZ = rows * pitch - gap;
    // The f32 shader must still distinguish an inset body from a tile edge.
    if (![spanX, spanZ, pitch].every(x => Number.isFinite(x) && Math.abs(Math.fround(x + 0.1) - Math.fround(x)) > 0))
        fail('geometry loses body-scale precision in f32');
    return Object.freeze({ arenas, wolves, prey, L, gap, cols, rows, pitch, perArena, count, wolfTotal,
        spanX, spanZ, origin: Object.freeze([-spanX / 2, -spanZ / 2]) });
}

export function catchFlags(records, lay, margin) {
    if (!Number.isFinite(margin) || margin < 0) fail(`catch margin ${margin} must be finite and nonnegative`);
    if (!(records instanceof Float32Array) || records.length !== lay.count * 16) fail('catch oracle needs complete records');
    const words = new Uint32Array(records.buffer, records.byteOffset, records.length), result = new Uint32Array(lay.count);
    for (let i = lay.wolfTotal; i < lay.count; i++) {
        if (!(words[i * 16 + 13] & 1)) continue;
        const tile = words[i * 16 + 15];
        for (let j = tile * lay.wolves; j < (tile + 1) * lay.wolves; j++) {
            if (!(words[j * 16 + 13] & 1) || records[j * 16 + 11] > 0 || words[j * 16 + 15] !== tile) continue;
            const reach = Math.fround(Math.fround(records[i * 16 + 3] + records[j * 16 + 3]) + Math.fround(margin));
            if (Math.hypot(records[i * 16] - records[j * 16], records[i * 16 + 2] - records[j * 16 + 2]) < reach) {
                result[i] = 256; result[j] = 512;
            }
        }
    }
    return result;
}

// Dedicated read-only snapshot dispatch; no peer flag read/write races. No pre-act query lists.
export function exhaustiveCatchWGSL(lay, margin) {
    if (!Number.isFinite(margin) || margin < 0) fail(`catch margin ${margin} must be finite and nonnegative`);
    return `@group(1) @binding(0) var<storage, read> catchSnapshot: array<Agent>;
fn step(i: u32, tick: u32) {
    let me = catchSnapshot[i];
    if ((me.ids.y & 1u) == 0u) { return; }
    let wolf = me.ids.x == SPECIES_WOLF;
    if (wolf && me.state.w > 0.0) { return; }
    let tile = me.ids.w;
    let first = select(tile * ${lay.wolves}u, ${lay.wolfTotal}u + tile * ${lay.prey}u, wolf);
    let last = first + select(${lay.wolves}u, ${lay.prey}u, wolf);
    for (var j = first; j < last; j++) {
        let other = catchSnapshot[j];
        if ((other.ids.y & 1u) == 0u || other.ids.w != tile) { continue; }
        if (!wolf && other.state.w > 0.0) { continue; }
        if (distance(me.posRadius.xz, other.posRadius.xz) < me.posRadius.w + other.posRadius.w + ${Number(margin).toFixed(9)}) {
            agents[i].ids.y |= select(256u, 512u, wolf);
            break;
        }
    }
}`;
}