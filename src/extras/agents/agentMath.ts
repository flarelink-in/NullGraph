// src/extras/agents/agentMath.ts
//
// CPU mirrors of the agent kernels (docs/agents-design.md §6–§7) and the fixed-step accumulator
// (§5.2). The mirrors are the DEFINITIONS the GPU stages are tested against, so they favour the
// obvious form: brute-force neighbour search, one agent at a time. They are not on any frame path.
//
// The local frame used by every sensor and by steering (design §7.1):
//     forward f = ( sin h, cos h )   — heading 0 faces +Z
//     side    s = ( cos h, −sin h )  — the direction f turns toward when h increases
// so a positive `side` reading means "a positive turn steers toward it". With Y up that side is the
// agent's left. Sensors and steering MUST share this frame; U5's steer-toward test fails if either
// flips.

import {
    AGENT_FLOATS, AGENT_RADIUS_OFFSET, AGENT_VELOCITY_OFFSET, AGENT_HEADING_OFFSET, AGENT_ENERGY_OFFSET,
    AGENT_AGE_OFFSET, AGENT_SPEED_SCALE_OFFSET, AGENT_SPECIES_OFFSET, AGENT_FLAGS_OFFSET, AGENT_FLAG_ALIVE,
    AGENT_FLAG_FROZEN, MAX_NEAREST_K, NEAREST_WIDTH, SELF_WIDTH, BOUNDS_WIDTH, type AgentDefinition,
} from './agentLayout';

/** The two views every mirror reads — an AgentRecords, or any adopted array pair. */
export interface RecordViews {
    readonly data: Float32Array;
    readonly u32: Uint32Array;
}

const TWO_PI = Math.PI * 2;
const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);

/** Wrap an angle to [-π, π). The WGSL twin uses the same floor form. */
export function wrapAngle(h: number): number {
    return h - TWO_PI * Math.floor((h + Math.PI) / TWO_PI);
}

/** A world XZ offset in an agent's local frame: out[o] = side, out[o + 1] = forward. */
export function toLocal(dx: number, dz: number, heading: number, out: { [k: number]: number }, o = 0): void {
    const sn = Math.sin(heading), cs = Math.cos(heading);
    out[o] = dx * cs - dz * sn;
    out[o + 1] = dx * sn + dz * cs;
}

// Module scratch: the frame path never calls these, but tests measure them allocation-free, and a
// returned array is never this scratch (design-principles §6).
const _d2 = new Float64Array(MAX_NEAREST_K);

/**
 * The definition of a neighbour query (design §6.2–6.3): up to `k` LIVE agents other than `i`,
 * of a species in `mask`, whose XZ distance is ≤ `radius`, nearest first, ties broken by the
 * LOWER index. Writes their indices to `out`; returns how many were found.
 */
export function nearestNeighbours(
    def: AgentDefinition, rec: RecordViews, i: number, mask: number, radius: number, k: number,
    out: { [n: number]: number },
): number {
    const d = rec.data, u = rec.u32, b = i * AGENT_FLOATS;
    const px = d[b], pz = d[b + 2], r2 = radius * radius;
    let found = 0;
    for (let j = 0; j < def.capacity; j++) {
        if (j === i) continue;
        const bj = j * AGENT_FLOATS;
        if ((u[bj + AGENT_FLAGS_OFFSET] & AGENT_FLAG_ALIVE) === 0) continue;
        if (((mask >>> u[bj + AGENT_SPECIES_OFFSET]) & 1) === 0) continue;
        const dx = d[bj] - px, dz = d[bj + 2] - pz;
        const dist2 = dx * dx + dz * dz;
        if (dist2 > r2) continue;
        // j ascends, so an equal distance already held belongs to a lower index and stays ahead.
        if (found === k && dist2 >= _d2[k - 1]) continue;
        let p = found < k ? found++ : k - 1;
        while (p > 0 && _d2[p - 1] > dist2) {
            _d2[p] = _d2[p - 1];
            out[p] = out[p - 1];
            p--;
        }
        _d2[p] = dist2;
        out[p] = j;
    }
    return found;
}

const _nearIdx = new Int32Array(MAX_NEAREST_K);

/**
 * One agent's observation row, exactly as the sense stage writes it (design §7.1). A dead agent's
 * row is zeros. Every feature is in the agent's local frame and clamped to [-1, 1].
 */
export function senseAgent(def: AgentDefinition, rec: RecordViews, i: number,
    out: Float32Array = new Float32Array(def.obsDim), o = 0): Float32Array {
    const d = rec.data, u = rec.u32, b = i * AGENT_FLOATS;
    if ((u[b + AGENT_FLAGS_OFFSET] & AGENT_FLAG_ALIVE) === 0) {
        out.fill(0, o, o + def.obsDim);
        return out;
    }
    const sp = def.species[u[b + AGENT_SPECIES_OFFSET]];
    const px = d[b], pz = d[b + 2];
    const vx = d[b + AGENT_VELOCITY_OFFSET], vz = d[b + AGENT_VELOCITY_OFFSET + 2];
    const h = d[b + AGENT_HEADING_OFFSET];
    const sn = Math.sin(h), cs = Math.cos(h);
    const r = sp.senseRadius;
    const velNorm = 2 * sp.maxSpeed;

    let at = o;
    let nearest = 0;
    for (const s of def.sensors) {
        if (s.kind === 'self') {
            const max = sp.maxSpeed * d[b + AGENT_SPEED_SCALE_OFFSET];
            out[at] = clamp(Math.hypot(vx, vz) / max, -1, 1);
            out[at + 1] = clamp(d[b + AGENT_ENERGY_OFFSET], -1, 1);
            at += SELF_WIDTH;
        } else if (s.kind === 'nearest') {
            const mask = def.nearestMasks[nearest++];
            if (s.observe === false) continue;
            const found = nearestNeighbours(def, rec, i, mask, Math.min(r, s.radius ?? def.cellSize), s.k, _nearIdx);
            for (let rank = 0; rank < s.k; rank++, at += NEAREST_WIDTH) {
                if (rank >= found) {
                    out.fill(0, at, at + NEAREST_WIDTH);
                    continue;
                }
                const bn = _nearIdx[rank] * AGENT_FLOATS;
                const dx = d[bn] - px, dz = d[bn + 2] - pz;
                const dvx = d[bn + AGENT_VELOCITY_OFFSET] - vx, dvz = d[bn + AGENT_VELOCITY_OFFSET + 2] - vz;
                out[at] = clamp((dx * cs - dz * sn) / r, -1, 1);
                out[at + 1] = clamp((dx * sn + dz * cs) / r, -1, 1);
                out[at + 2] = clamp((dvx * cs - dvz * sn) / velNorm, -1, 1);
                out[at + 3] = clamp((dvx * sn + dvz * cs) / velNorm, -1, 1);
                out[at + 4] = 1;
            }
        } else if (s.kind === 'bounds') {
            // The nearest wall, ties resolved in the order −X, +X, −Z, +Z. Its outward normal is the
            // direction toward it from inside; magnitude 1 at the wall falling to 0 at senseRadius.
            const w = def.world;
            let dist = px - w.minX, nx = -1, nz = 0;
            if (w.maxX - px < dist) { dist = w.maxX - px; nx = 1; nz = 0; }
            if (pz - w.minZ < dist) { dist = pz - w.minZ; nx = 0; nz = -1; }
            if (w.maxZ - pz < dist) { dist = w.maxZ - pz; nx = 0; nz = 1; }
            const mag = Math.max(0, 1 - Math.max(dist, 0) / r);
            out[at] = (nx * cs - nz * sn) * mag;
            out[at + 1] = (nx * sn + nz * cs) * mag;
            at += BOUNDS_WIDTH;
        } else {
            if (!s.mirror) throw new Error(`AgentSystem: custom sensor "${s.name}" has no CPU mirror; pass mirror to use it on the CPU.`);
            s.mirror(d, u, i, out, at, def);
            at += s.width;
        }
    }
    return out;
}

/** Every agent's row: `[capacity, obsDim]`, the layout the obs buffer and ONNX use. */
export function senseAll(def: AgentDefinition, rec: RecordViews,
    out: Float32Array = new Float32Array(def.capacity * def.obsDim)): Float32Array {
    for (let i = 0; i < def.capacity; i++) senseAgent(def, rec, i, out, i * def.obsDim);
    return out;
}

/**
 * One fixed step of one agent under `steering()` (design §7.2), in place:
 *  - a row with any non-finite value is discarded: no turn, target speed 0 — a broken brain stops
 *    agents, it never moves or corrupts them;
 *  - turn ∈ [-1, 1] × maxTurn; throttle ∈ [-1, 1] → target speed ∈ [0, maxSpeed × speedScale],
 *    approached at `accel`;
 *  - velocity always lies along the heading; the centre stays inside the world inset by the radius,
 *    and a wall reflects the heading when the agent was moving into it.
 * Dead agents are untouched; frozen agents only age, with velocity zero.
 */
export function actAgent(def: AgentDefinition, rec: RecordViews, i: number, act: ArrayLike<number>, a = 0): void {
    const d = rec.data, u = rec.u32, b = i * AGENT_FLOATS;
    const flags = u[b + AGENT_FLAGS_OFFSET];
    if ((flags & AGENT_FLAG_ALIVE) === 0) return;
    const dt = def.dt;
    d[b + AGENT_AGE_OFFSET] += dt;
    if (flags & AGENT_FLAG_FROZEN) {
        d[b + AGENT_VELOCITY_OFFSET] = 0;
        d[b + AGENT_VELOCITY_OFFSET + 1] = 0;
        d[b + AGENT_VELOCITY_OFFSET + 2] = 0;
        return;
    }
    if (def.actions.kind === 'custom') {
        if (!def.actions.mirror) throw new Error(`AgentSystem: custom actions "${def.actions.name}" have no CPU mirror; pass mirror to use them on the CPU.`);
        def.actions.mirror(d, u, i, act, a, dt);
        return;
    }
    const sp = def.species[u[b + AGENT_SPECIES_OFFSET]];
    const a0 = act[a], a1 = act[a + 1];
    let turn = 0, target = 0;
    if (Number.isFinite(a0) && Number.isFinite(a1)) {
        turn = clamp(a0, -1, 1);
        target = (clamp(a1, -1, 1) + 1) * 0.5 * sp.maxSpeed * d[b + AGENT_SPEED_SCALE_OFFSET];
    }

    let h = wrapAngle(d[b + AGENT_HEADING_OFFSET] + turn * sp.maxTurn * dt);
    const dv = sp.accel * dt;
    let speed = Math.hypot(d[b + AGENT_VELOCITY_OFFSET], d[b + AGENT_VELOCITY_OFFSET + 2]);
    speed += clamp(target - speed, -dv, dv);
    let fx = Math.sin(h), fz = Math.cos(h);
    let x = d[b] + fx * speed * dt;
    let z = d[b + 2] + fz * speed * dt;

    const w = def.world, rad = d[b + AGENT_RADIUS_OFFSET];
    let loX = w.minX + rad, hiX = w.maxX - rad, loZ = w.minZ + rad, hiZ = w.maxZ - rad;
    if (loX > hiX) loX = hiX = (w.minX + w.maxX) / 2;
    if (loZ > hiZ) loZ = hiZ = (w.minZ + w.maxZ) / 2;
    if (x < loX) { x = loX; if (fx < 0) h = wrapAngle(-h); }
    else if (x > hiX) { x = hiX; if (fx > 0) h = wrapAngle(-h); }
    fz = Math.cos(h);
    if (z < loZ) { z = loZ; if (fz < 0) h = wrapAngle(Math.PI - h); }
    else if (z > hiZ) { z = hiZ; if (fz > 0) h = wrapAngle(Math.PI - h); }
    fx = Math.sin(h); fz = Math.cos(h);

    d[b] = x;
    d[b + 2] = z;
    d[b + AGENT_VELOCITY_OFFSET] = fx * speed;
    d[b + AGENT_VELOCITY_OFFSET + 1] = 0;
    d[b + AGENT_VELOCITY_OFFSET + 2] = fz * speed;
    d[b + AGENT_HEADING_OFFSET] = h;
}

/** One step of every agent, reading `act` as `[capacity, actDim]`. Sense first, then this — as the GPU does. */
export function stepAll(def: AgentDefinition, rec: RecordViews, act: Float32Array): void {
    for (let i = 0; i < def.capacity; i++) actAgent(def, rec, i, act, i * def.actDim);
}

/**
 * The fixed-step accumulator (design §5.2; physics P1 §5.1). Returns how many steps this frame
 * runs. One deliberate difference from P1's pseudocode: excess time is dropped — and the frame
 * counted as clamped — only when DEBT REMAINS after `maxSteps` steps. P1 drops the remainder
 * whenever it reaches maxSteps, which discards up to a step of legitimate time and counts a clamp
 * that did not happen. Never throws: a bad frame time counts as zero (design-principles §2).
 */
export class FixedStep {
    public accumulator = 0;
    /** Frames whose excess time was dropped — a number to read, not an exception (P1 §5.2). */
    public clampedFrames = 0;
    /** Steps the last advance() ran. */
    public steps = 0;
    public totalSteps = 0;

    constructor(
        public readonly dt = 1 / 60,
        public readonly maxSteps = 3,
        public readonly maxFrameDelta = 0.25,
    ) {
        if (!(dt > 0) || !Number.isInteger(maxSteps) || maxSteps < 1 || !(maxFrameDelta >= dt)) {
            throw new Error(`AgentSystem: FixedStep needs dt > 0, an integer maxSteps ≥ 1 and maxFrameDelta ≥ dt.`);
        }
    }

    public advance(frameSeconds: number): number {
        const t = frameSeconds > 0 ? Math.min(frameSeconds, this.maxFrameDelta) : 0;
        this.accumulator += t;
        let n = 0;
        while (this.accumulator >= this.dt && n < this.maxSteps) {
            this.accumulator -= this.dt;
            n++;
        }
        if (this.accumulator >= this.dt) {
            this.accumulator = 0;
            this.clampedFrames++;
        }
        this.steps = n;
        this.totalSteps += n;
        return n;
    }

    /** Fraction of a step left over — for render interpolation, which Phase 1 does not do. */
    public get alpha(): number {
        return this.accumulator / this.dt;
    }
}
