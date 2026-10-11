// src/extras/agents/agentRecords.ts
//
// The CPU staging area for agent records (docs/agents-design.md §5.1). spawn() and kill() write
// records and widen one dirty range; the system uploads that range and nothing else. After upload
// the GPU owns the record — this array is where spawns are staged, not a mirror of the simulation.

import {
    AGENT_FLOATS, AGENT_POSITION_OFFSET, AGENT_RADIUS_OFFSET, AGENT_VELOCITY_OFFSET, AGENT_HEADING_OFFSET,
    AGENT_ENERGY_OFFSET, AGENT_AGE_OFFSET, AGENT_SPEED_SCALE_OFFSET, AGENT_USER0_OFFSET, AGENT_SPECIES_OFFSET,
    AGENT_FLAGS_OFFSET, AGENT_GENOME_OFFSET, AGENT_USER1_OFFSET, AGENT_FLAG_ALIVE, AGENT_FLAG_FROZEN,
    AGENT_FLAG_APP_SHIFT, type AgentDefinition,
} from './agentLayout';
import { wrapAngle } from './agentMath';

export interface AgentInit {
    /** World position [x, y, z]; y is carried, not integrated (design §1 non-goals). */
    position: ArrayLike<number>;
    /** Species name or index. */
    species: string | number;
    /** Yaw, radians, 0 = +Z. Default 0. */
    heading?: number;
    /** Initial speed along the heading, m/s, clamped to the agent's max (maxSpeed × speedScale). Default 0. */
    speed?: number;
    /** Body radius, m. Default: the species' radius. */
    radius?: number;
    /** Default 1. */
    energy?: number;
    /** Default 0. */
    age?: number;
    /** Multiplier on the species' max speed. Default 1. */
    speedScale?: number;
    /** Phase 2 weight-table row. Default 0. */
    genome?: number;
    /** App flag bits, stored in bits 8–31. Default 0. */
    appFlags?: number;
    /** Held in place: no movement, velocity zero. Default false. */
    frozen?: boolean;
    user0?: number;
    user1?: number;
}

export class AgentRecords {
    public readonly capacity: number;
    /** Float view of the records — adopt it, fill it from a Worker, or write through spawn(). */
    public readonly data: Float32Array;
    /** Integer view over the same memory: same byteOffset and length (design-principles §6). */
    public readonly u32: Uint32Array;

    private first = Infinity;
    private last = -1;

    /**
     * `data`, when given, is adopted by reference (the setRawData contract) and its first
     * `count` records are marked dirty; otherwise a zeroed array is allocated — every agent dead.
     */
    constructor(private readonly def: AgentDefinition, data?: Float32Array, count = 0) {
        this.capacity = def.capacity;
        const need = def.capacity * AGENT_FLOATS;
        if (data && data.length < need) {
            throw new Error(`AgentSystem: adopted data holds ${data.length} floats; capacity ${def.capacity} needs ${need}.`);
        }
        this.data = data ?? new Float32Array(need);
        this.u32 = new Uint32Array(this.data.buffer, this.data.byteOffset, this.data.length);
        if (data && count > 0) this.markDirty(0, Math.min(count, def.capacity) - 1);
    }

    /** Write a live agent's record. */
    public spawn(i: number, init: AgentInit): void {
        this.assertIndex(i);
        const s = typeof init.species === 'number' ? init.species : this.def.speciesIndex(init.species);
        const species = this.def.species[s];
        if (!species || !Number.isInteger(s)) throw new Error(`AgentSystem: species index ${init.species} is not declared.`);
        const p = init.position;
        const heading = init.heading ?? 0;
        const speed = init.speed ?? 0;
        const radius = init.radius ?? species.radius;
        const speedScale = init.speedScale ?? 1;
        if (!(Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2]) && Number.isFinite(heading) && Number.isFinite(speed))) {
            throw new Error(`AgentSystem: agent ${i} has a non-finite position, heading or speed.`);
        }
        if (!(radius > 0 && speedScale > 0)) throw new Error(`AgentSystem: agent ${i} radius and speedScale must be positive.`);
        if (speed < 0) throw new Error(`AgentSystem: agent ${i} speed ${speed} is negative; turn the heading instead.`);
        const appFlags = init.appFlags ?? 0;
        if (!(Number.isInteger(appFlags) && appFlags >= 0 && appFlags < 1 << (32 - AGENT_FLAG_APP_SHIFT))) {
            throw new Error(`AgentSystem: agent ${i} appFlags ${appFlags} must fit in ${32 - AGENT_FLAG_APP_SHIFT} bits.`);
        }

        const d = this.data, u = this.u32, b = i * AGENT_FLOATS;
        const h = wrapAngle(heading);
        const frozen = init.frozen === true;
        // Never faster than the agent's own max — the invariant steering then keeps (design §7.2).
        const v = frozen ? 0 : Math.min(speed, species.maxSpeed * speedScale);
        d[b + AGENT_POSITION_OFFSET] = p[0];
        d[b + AGENT_POSITION_OFFSET + 1] = p[1];
        d[b + AGENT_POSITION_OFFSET + 2] = p[2];
        d[b + AGENT_RADIUS_OFFSET] = radius;
        d[b + AGENT_VELOCITY_OFFSET] = Math.sin(h) * v;   // velocity always lies along the heading
        d[b + AGENT_VELOCITY_OFFSET + 1] = 0;
        d[b + AGENT_VELOCITY_OFFSET + 2] = Math.cos(h) * v;
        d[b + AGENT_HEADING_OFFSET] = h;
        d[b + AGENT_ENERGY_OFFSET] = init.energy ?? 1;
        d[b + AGENT_AGE_OFFSET] = init.age ?? 0;
        d[b + AGENT_SPEED_SCALE_OFFSET] = speedScale;
        d[b + AGENT_USER0_OFFSET] = init.user0 ?? 0;
        u[b + AGENT_SPECIES_OFFSET] = s;
        u[b + AGENT_FLAGS_OFFSET] = (AGENT_FLAG_ALIVE | (frozen ? AGENT_FLAG_FROZEN : 0) | (appFlags << AGENT_FLAG_APP_SHIFT)) >>> 0;
        u[b + AGENT_GENOME_OFFSET] = init.genome ?? 0;
        u[b + AGENT_USER1_OFFSET] = init.user1 ?? 0;
        this.markDirty(i, i);
    }

    /** Clear the alive bit; the record stays, and every stage skips it. */
    public kill(i: number): void {
        this.assertIndex(i);
        const k = i * AGENT_FLOATS + AGENT_FLAGS_OFFSET;
        this.u32[k] = (this.u32[k] & ~AGENT_FLAG_ALIVE) >>> 0;
        this.markDirty(i, i);
    }

    public isAlive(i: number): boolean {
        return (this.u32[i * AGENT_FLOATS + AGENT_FLAGS_OFFSET] & AGENT_FLAG_ALIVE) !== 0;
    }

    /** First and last dirty record, or -1 / -1 when nothing changed since clearDirty(). */
    public get dirtyFirst(): number { return this.last < 0 ? -1 : this.first; }
    public get dirtyLast(): number { return this.last; }

    public markDirty(first: number, last: number): void {
        if (first < this.first) this.first = first;
        if (last > this.last) this.last = last;
    }

    public clearDirty(): void {
        this.first = Infinity;
        this.last = -1;
    }

    private assertIndex(i: number): void {
        if (!(Number.isInteger(i) && i >= 0 && i < this.capacity)) {
            throw new Error(`AgentSystem: index ${i} is outside capacity ${this.capacity}.`);
        }
    }
}
