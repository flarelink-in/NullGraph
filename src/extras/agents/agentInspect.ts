// src/extras/agents/agentInspect.ts
//
// Step 4's inspector (docs/agents-gpu-loop-design.md §9.6): one agent's record, observation row,
// action row, reason, neighbour indices and the system's counters, copied by `AgentSystem.inspect`
// into one staging buffer off the frame path, and decoded here. Pure: no device — U20 checks the
// decoder against a staging buffer built by hand.

import {
    AGENT_BYTES, AGENT_POSITION_OFFSET, AGENT_RADIUS_OFFSET, AGENT_VELOCITY_OFFSET, AGENT_HEADING_OFFSET,
    AGENT_ENERGY_OFFSET, AGENT_AGE_OFFSET, AGENT_SPEED_SCALE_OFFSET, AGENT_USER0_OFFSET, AGENT_SPECIES_OFFSET,
    AGENT_FLAGS_OFFSET, AGENT_GENOME_OFFSET, AGENT_USER1_OFFSET, AGENT_FLAG_ALIVE, AGENT_FLAG_FROZEN,
    type AgentDefinition,
} from './agentLayout';

/** A neighbour slot the query left empty. */
const NO_AGENT = 0xFFFFFFFF;

/** One agent as the GPU had it: what it senses (obs), what it chose (act), why, and who it saw. */
export interface AgentSnapshot {
    index: number;
    alive: boolean;
    frozen: boolean;
    species: string;
    position: [number, number, number];
    heading: number;
    /** m/s, along the heading. */
    speed: number;
    radius: number;
    energy: number;
    /** Seconds alive. */
    age: number;
    speedScale: number;
    /** The record's app words: user0 (f32, e.g. a timer), user1 and genome (u32), flags (u32, app bits 8–31). */
    user0: number;
    user1: number;
    genome: number;
    flags: number;
    /** The observation row, named by the definition's obsNames ("wolf[0].forward"). */
    obs: { name: string; value: number }[];
    /** The action row, named by actNames ("turn", "throttle"). */
    act: { name: string; value: number }[];
    /** What the brain passed to setReason(), 0 if nothing. */
    reason: number;
    /** Per `nearest` sensor, the agent indices it found, nearest first (empty slots dropped). */
    neighbours: { sensor: string; indices: number[] }[];
    /** System counters: steps run, steps whose grid skipped entries, and how many were skipped. */
    tick: number;
    overflowSteps: number;
    overflowTotal: number;
}

/** Byte offsets of each part in the staging buffer, for one layout. */
export interface AgentSnapshotLayout {
    record: number;
    obs: number;
    act: number;
    reason: number;
    neighbours: number;
    state: number;
    /** Neighbour slots per agent (the sum of every `nearest` sensor's k). */
    neighbourStride: number;
    /** Bytes in all: 204 for Herd & Wolves (24 obs floats, 2 actions, 4 neighbour slots). */
    size: number;
}

export function agentSnapshotLayout(def: AgentDefinition): AgentSnapshotLayout {
    const neighbourStride = def.sensors.reduce((n, s) => n + (s.kind === 'nearest' ? s.k : 0), 0);
    const record = 0;
    const obs = record + AGENT_BYTES;
    const act = obs + def.obsDim * 4;
    const reason = act + def.actDim * 4;
    const neighbours = reason + 4;
    const state = neighbours + neighbourStride * 4;
    return { record, obs, act, reason, neighbours, state, neighbourStride, size: state + 16 };
}

/** Decode a staging buffer laid out by `agentSnapshotLayout(def)` for agent `index`. */
export function decodeAgentSnapshot(def: AgentDefinition, index: number, data: ArrayBuffer,
    layout: AgentSnapshotLayout = agentSnapshotLayout(def)): AgentSnapshot {
    const f = new Float32Array(data), u = new Uint32Array(data);
    const r = layout.record / 4, o = layout.obs / 4, a = layout.act / 4, n = layout.neighbours / 4, s = layout.state / 4;
    const flags = u[r + AGENT_FLAGS_OFFSET];
    const speciesIndex = u[r + AGENT_SPECIES_OFFSET];
    const neighbours: { sensor: string; indices: number[] }[] = [];
    let base = n;
    for (const sensor of def.sensors) {
        if (sensor.kind !== 'nearest') continue;
        const indices: number[] = [];
        for (let k = 0; k < sensor.k; k++) if (u[base + k] !== NO_AGENT) indices.push(u[base + k]);
        neighbours.push({ sensor: sensor.name, indices });
        base += sensor.k;
    }
    const v = r + AGENT_VELOCITY_OFFSET;
    return {
        index,
        alive: (flags & AGENT_FLAG_ALIVE) !== 0,
        frozen: (flags & AGENT_FLAG_FROZEN) !== 0,
        species: def.species[speciesIndex]?.name ?? `#${speciesIndex}`,
        position: [f[r + AGENT_POSITION_OFFSET], f[r + AGENT_POSITION_OFFSET + 1], f[r + AGENT_POSITION_OFFSET + 2]],
        heading: f[r + AGENT_HEADING_OFFSET],
        speed: Math.hypot(f[v], f[v + 1], f[v + 2]),
        radius: f[r + AGENT_RADIUS_OFFSET],
        energy: f[r + AGENT_ENERGY_OFFSET],
        age: f[r + AGENT_AGE_OFFSET],
        speedScale: f[r + AGENT_SPEED_SCALE_OFFSET],
        user0: f[r + AGENT_USER0_OFFSET],
        user1: u[r + AGENT_USER1_OFFSET],
        genome: u[r + AGENT_GENOME_OFFSET],
        flags,
        obs: def.obsNames.map((name, k) => ({ name, value: f[o + k] })),
        act: def.actNames.map((name, k) => ({ name, value: f[a + k] })),
        reason: u[layout.reason / 4],
        neighbours,
        tick: u[s],
        overflowSteps: u[s + 1],
        overflowTotal: u[s + 2],
    };
}
