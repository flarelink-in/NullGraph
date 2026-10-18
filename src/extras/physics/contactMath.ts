// src/extras/physics/contactMath.ts
//
// Pure CPU definitions and mirror for Phase 2 step 13 (DiscContacts on XZ plane).
// Tests and offline tools only: no GPU dependencies.

import { pointCell, buildPointGrid, type PointGridLayout } from '../spatial/gridMath';
import { gridWindow } from '../spatial/gridWindow';

export interface DiscLayout {
    capacity: number;
    /** Floats per record, a multiple of 4. Floats 0–3 are x, y, z and radius. */
    stride: number;
    /** A u32 flags word: bodies without `mask` set are skipped (dead). */
    alive?: { offset: number; mask: number };
    /** Bodies with `mask` set push others and never move. */
    fixed?: { offset: number; mask: number };
    /** Floats offset..offset+2 are a velocity (offset+0: vx, offset+2: vz); the inward XZ normal part is removed (β), vy untouched. */
    velocity?: { offset: number };
}

export interface DiscContactsOptions extends Partial<DiscLayout> {
    world: { min: [number, number]; max: [number, number] };
    maxRadius: number;
    cellSize?: number;
    clamp?: boolean;
    dt?: number;
    iterations?: number;
    relaxation?: number;
    compliance?: number;
    maxPush?: number;
    maxScanPerCell?: number;
    mass?: 'area' | 'equal';
    damping?: number;
}

export interface ResolvedDiscContacts {
    layout: DiscLayout;
    world: { minX: number; minZ: number; maxX: number; maxZ: number };
    grid: { cellsX: number; cellsZ: number; cellTotal: number };
    cellSize: number;
    maxRadius: number;
    clamp: boolean;
    dt: number;
    iterations: number;
    relaxation: number;
    compliance: number;
    tildeCompliance: number;
    maxPush: number;
    maxScanPerCell: number;
    mass: 'area' | 'equal';
    damping: number;
}

export const DEFAULT_ITERATIONS = 2;
export const DEFAULT_RELAXATION = 1.75;
export const DEFAULT_MAX_PUSH = 0.25;
export const DEFAULT_MAX_SCAN = 32;
export const DEFAULT_DT = 1 / 60;
export const DEFAULT_DAMPING = 1;
export const DEFAULT_MASS: 'area' | 'equal' = 'area';
export const MAX_CELLS = 1_048_575;

/** Deterministic tie-breaking unit normal when two bodies have coincident centres (d <= 1e-6). */
export function tieBreak(i: number, j: number): [number, number] {
    const lo = Math.min(i, j) >>> 0;
    const hi = Math.max(i, j) >>> 0;
    let h = (Math.imul(lo, 0x9e3779b9) ^ Math.imul((hi + 0x7f4a7c15) >>> 0, 0x85ebca6b)) >>> 0;
    h = Math.imul(h ^ (h >>> 16), 0x7feb352d) >>> 0;
    h = Math.imul(h ^ (h >>> 15), 0x846ca68b) >>> 0;
    h = (h ^ (h >>> 16)) >>> 0;
    const a = (h >>> 8) * (6.283185307179586 / 16777216.0);
    const cosA = Math.cos(a), sinA = Math.sin(a);
    return i === lo ? [cosA, sinA] : [-cosA, -sinA];
}

/** Validate layout and options, resolving defaults according to Spike S9. */
export function validateDiscContacts(layout: DiscLayout, options: DiscContactsOptions): ResolvedDiscContacts {
    const stride = options.stride ?? layout.stride;
    if (!Number.isInteger(stride) || stride <= 0 || stride % 4 !== 0) {
        throw new Error(`DiscContacts: stride ${stride} must be a positive multiple of 4.`);
    }

    const capacity = options.capacity ?? layout.capacity;
    if (!Number.isInteger(capacity) || capacity <= 0) {
        throw new Error(`DiscContacts: capacity ${capacity} must be a positive integer.`);
    }

    const alive = options.alive ?? layout.alive;
    if (alive !== undefined) {
        const o = alive.offset;
        if (!Number.isInteger(o) || o < 4 || o >= stride) {
            throw new Error(`DiscContacts: alive offset ${o} must be a word of the record after its first four (4–${stride - 1}).`);
        }
        const m = alive.mask;
        if (!Number.isInteger(m) || m <= 0 || m > 0xffffffff) {
            throw new Error(`DiscContacts: alive mask ${m} must be a non-zero u32.`);
        }
    }

    const fixed = options.fixed ?? layout.fixed;
    if (fixed !== undefined) {
        const o = fixed.offset;
        if (!Number.isInteger(o) || o < 4 || o >= stride) {
            throw new Error(`DiscContacts: fixed offset ${o} must be a word of the record after its first four (4–${stride - 1}).`);
        }
        const m = fixed.mask;
        if (!Number.isInteger(m) || m <= 0 || m > 0xffffffff) {
            throw new Error(`DiscContacts: fixed mask ${m} must be a non-zero u32.`);
        }
    }

    const velocity = options.velocity ?? layout.velocity;
    if (velocity !== undefined) {
        const o = velocity.offset;
        if (!Number.isInteger(o) || o < 4 || o > stride - 3) {
            throw new Error(`DiscContacts: velocity offset ${o} must leave room for 3 floats (4–${stride - 3}).`);
        }
    }

    if (!Number.isFinite(options.maxRadius) || options.maxRadius <= 0) {
        throw new Error(`DiscContacts: maxRadius ${options.maxRadius} must be positive.`);
    }

    const maxPush = options.maxPush ?? DEFAULT_MAX_PUSH;
    if (!Number.isFinite(maxPush) || maxPush <= 0) {
        throw new Error(`DiscContacts: maxPush ${maxPush} must be positive.`);
    }

    const dt = options.dt ?? DEFAULT_DT;
    if (!Number.isFinite(dt) || dt <= 0) {
        throw new Error(`DiscContacts: dt ${dt} must be positive.`);
    }

    const iterations = options.iterations ?? DEFAULT_ITERATIONS;
    if (!Number.isInteger(iterations) || iterations < 1 || iterations > 8) {
        throw new Error(`DiscContacts: iterations ${iterations} must be an integer from 1 to 8.`);
    }

    const relaxation = options.relaxation ?? DEFAULT_RELAXATION;
    if (!Number.isFinite(relaxation) || relaxation <= 0 || relaxation >= 2) {
        throw new Error(`DiscContacts: relaxation ${relaxation} must be above 0 and below 2.`);
    }

    const compliance = options.compliance ?? 0;
    if (!Number.isFinite(compliance) || compliance < 0) {
        throw new Error(`DiscContacts: compliance ${compliance} must be at least 0.`);
    }

    const damping = options.damping ?? DEFAULT_DAMPING;
    if (!Number.isFinite(damping) || damping < 0 || damping > 1) {
        throw new Error(`DiscContacts: damping ${damping} must be from 0 to 1.`);
    }

    const mass = options.mass ?? DEFAULT_MASS;
    if (mass !== 'area' && mass !== 'equal') {
        throw new Error(`DiscContacts: mass "${mass}" must be "area" or "equal".`);
    }

    const maxScanPerCell = options.maxScanPerCell ?? DEFAULT_MAX_SCAN;
    if (!Number.isInteger(maxScanPerCell) || maxScanPerCell <= 0) {
        throw new Error(`DiscContacts: maxScanPerCell ${maxScanPerCell} must be a positive integer.`);
    }

    const w = options.world;
    if (!w || !w.min || !w.max || w.min.length < 2 || w.max.length < 2 ||
        !Number.isFinite(w.min[0]) || !Number.isFinite(w.min[1]) ||
        !Number.isFinite(w.max[0]) || !Number.isFinite(w.max[1]) ||
        w.min[0] >= w.max[0] || w.min[1] >= w.max[1]) {
        throw new Error('DiscContacts: world must specify finite min < max bounds.');
    }
    const minX = w.min[0], minZ = w.min[1], maxX = w.max[0], maxZ = w.max[1];

    const bound = 2 * options.maxRadius + (iterations - 1) * maxPush;
    const cellSize = options.cellSize ?? bound;
    if (!Number.isFinite(cellSize) || cellSize < bound) {
        throw new Error(`DiscContacts: cellSize ${cellSize} is below 2 × maxRadius + (iterations − 1) × maxPush = ${bound}; contacts would be missed.`);
    }

    const cellsX = Math.ceil((maxX - minX) / cellSize);
    const cellsZ = Math.ceil((maxZ - minZ) / cellSize);
    const cellTotal = cellsX * cellsZ;
    if (cellTotal > MAX_CELLS) {
        throw new Error(`DiscContacts: the grid needs ${cellTotal} cells; raise cellSize or shrink the world.`);
    }

    const tildeCompliance = compliance / (dt * dt);

    return {
        layout: { capacity, stride, alive, fixed, velocity },
        world: { minX, minZ, maxX, maxZ },
        grid: { cellsX, cellsZ, cellTotal },
        cellSize,
        maxRadius: options.maxRadius,
        clamp: options.clamp ?? false,
        dt,
        iterations,
        relaxation,
        compliance,
        tildeCompliance,
        maxPush,
        maxScanPerCell,
        mass,
        damping,
    };
}

/**
 * CPU mirror of DiscContacts: runs one step's contact passes over records.
 * Uses PointGrid's CPU mirror (buildPointGrid) and follows exact WGSL dispatch order.
 */
export function resolveDiscContacts(
    layout: DiscLayout,
    options: DiscContactsOptions,
    records: Float32Array,
    out?: Float32Array,
    stepTick = 0,
): {
    records: Float32Array;
    counters: { bodiesOverMaxRadius: number; overflow: number };
    contactCounts: Uint32Array;
} {
    const resolved = validateDiscContacts(layout, options);
    const totalFloats = resolved.layout.capacity * resolved.layout.stride;
    if (records.length < totalFloats) {
        throw new Error('DiscContacts: records must have STORAGE usage and hold capacity × stride floats.');
    }

    let target: Float32Array;
    if (out) {
        if (out.length < totalFloats) {
            throw new Error('DiscContacts: out buffer too small.');
        }
        out.set(records.subarray(0, totalFloats));
        target = out;
    } else {
        target = records;
    }

    const stride = resolved.layout.stride;
    const aliveOffset = resolved.layout.alive?.offset;
    const aliveMask = resolved.layout.alive?.mask ?? 0;
    const fixedOffset = resolved.layout.fixed?.offset;
    const fixedMask = resolved.layout.fixed?.mask ?? 0;
    const velOffset = resolved.layout.velocity?.offset;

    const u32View = new Uint32Array(target.buffer, target.byteOffset, target.length);

    const isAlive = (i: number): boolean => {
        if (aliveOffset === undefined) return true;
        return (u32View[i * stride + aliveOffset] & aliveMask) !== 0;
    };

    const isFixed = (i: number): boolean => {
        if (fixedOffset === undefined) return false;
        return (u32View[i * stride + fixedOffset] & fixedMask) !== 0;
    };

    const inverseMass = (i: number, r: number): number => {
        if (isFixed(i)) return 0;
        if (resolved.mass === 'equal') return 1;
        const radius = Math.max(r, 1e-4);
        return (resolved.maxRadius * resolved.maxRadius) / (radius * radius);
    };

    // PointGrid built from x^0 positions
    const gridLayout: PointGridLayout = {
        world: { minX: resolved.world.minX, minZ: resolved.world.minZ },
        cellSize: resolved.cellSize,
        grid: resolved.grid,
    };
    const grid = buildPointGrid(
        gridLayout, target, stride, resolved.layout.capacity,
        isAlive, resolved.maxScanPerCell,
    );

    const capacity = resolved.layout.capacity;
    const cellsX = resolved.grid.cellsX;
    const cellsZ = resolved.grid.cellsZ;
    const deltasX = new Float32Array(capacity);
    const deltasZ = new Float32Array(capacity);
    const contactCounts = new Uint32Array(capacity);

    for (let k = 0; k < resolved.iterations; k++) {
        deltasX.fill(0);
        deltasZ.fill(0);
        contactCounts.fill(0);
        const windowSeed = (stepTick * resolved.iterations + k) >>> 0;

        // Gather pass
        for (let i = 0; i < capacity; i++) {
            if (!isAlive(i)) continue;
            const xi = target[i * stride];
            const zi = target[i * stride + 2];
            const ri = target[i * stride + 3];
            const wi = inverseMass(i, ri);
            if (wi === 0) continue;

            const cell = pointCell(gridLayout, xi, zi);
            const cx = cell % cellsX;
            const cz = Math.floor(cell / cellsX);

            let sumX = 0, sumZ = 0, count = 0;

            const zMin = Math.max(0, cz - 1), zMax = Math.min(cellsZ - 1, cz + 1);
            const xMin = Math.max(0, cx - 1), xMax = Math.min(cellsX - 1, cx + 1);

            for (let z = zMin; z <= zMax; z++) {
                for (let x = xMin; x <= xMax; x++) {
                    const c = z * cellsX + x;
                    const first = grid.cellOffsets[c];
                    const n = grid.cellOffsets[c + 1] - first;
                    let at = first;
                    if (n > resolved.maxScanPerCell) {
                        at += gridWindow(c, windowSeed) % n;
                    }
                    const left = Math.min(n, resolved.maxScanPerCell);
                    for (let s = 0; s < left; s++) {
                        const j = grid.cellBodies[at];
                        at++;
                        if (at === first + n) at = first;
                        if (j === i) continue;
                        if (!isAlive(j)) continue;

                        const xj = target[j * stride];
                        const zj = target[j * stride + 2];
                        const rj = target[j * stride + 3];

                        const dx = xi - xj;
                        const dz = zi - zj;
                        const reach = ri + rj;
                        const d2 = dx * dx + dz * dz;
                        if (d2 >= reach * reach) continue;

                        const wj = inverseMass(j, rj);
                        if (wi + wj === 0) continue;

                        const dist = Math.sqrt(d2);
                        let nx: number, nz: number;
                        if (dist > 1e-6) {
                            nx = dx / dist;
                            nz = dz / dist;
                        } else {
                            [nx, nz] = tieBreak(i, j);
                        }
                        const penetration = reach - dist;
                        const share = wi / (wi + wj + resolved.tildeCompliance);
                        const mag = penetration * share;
                        sumX += nx * mag;
                        sumZ += nz * mag;
                        count++;
                    }
                }
            }
            if (count > 0) {
                deltasX[i] = sumX;
                deltasZ[i] = sumZ;
                contactCounts[i] = count;
            }
        }

        // Apply pass
        for (let i = 0; i < capacity; i++) {
            const count = contactCounts[i];
            if (count === 0) continue;

            const scale = resolved.relaxation / Math.sqrt(count);
            let ux = deltasX[i] * scale;
            let uz = deltasZ[i] * scale;
            let len = Math.hypot(ux, uz);
            if (len > resolved.maxPush) {
                const pushScale = resolved.maxPush / len;
                ux *= pushScale;
                uz *= pushScale;
                len = resolved.maxPush;
            }

            let xzX = target[i * stride] + ux;
            let xzZ = target[i * stride + 2] + uz;

            if (resolved.clamp) {
                const r = target[i * stride + 3];
                const midX = (resolved.world.minX + resolved.world.maxX) * 0.5;
                const midZ = (resolved.world.minZ + resolved.world.maxZ) * 0.5;
                const loX = Math.min(resolved.world.minX + r, midX);
                const hiX = Math.max(resolved.world.maxX - r, midX);
                const loZ = Math.min(resolved.world.minZ + r, midZ);
                const hiZ = Math.max(resolved.world.maxZ - r, midZ);
                xzX = Math.max(loX, Math.min(hiX, xzX));
                xzZ = Math.max(loZ, Math.min(hiZ, xzZ));
            }

            target[i * stride] = xzX;
            target[i * stride + 2] = xzZ;

            if (velOffset !== undefined && resolved.damping > 0 && len > 1e-9) {
                const vo = i * stride + velOffset;
                const vx = target[vo];
                const vz = target[vo + 2];
                const nx = ux / len;
                const nz = uz / len;
                const vn = vx * nx + vz * nz;
                if (vn < 0) {
                    target[vo] = vx - resolved.damping * vn * nx;
                    target[vo + 2] = vz - resolved.damping * vn * nz;
                }
            }
        }
    }

    let bodiesOverMaxRadius = 0;
    for (let i = 0; i < capacity; i++) {
        if (isAlive(i) && target[i * stride + 3] > resolved.maxRadius) {
            bodiesOverMaxRadius++;
        }
    }

    return {
        records: target,
        counters: {
            bodiesOverMaxRadius,
            overflow: grid.overflow,
        },
        contactCounts,
    };
}
