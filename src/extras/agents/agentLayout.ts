// src/extras/agents/agentLayout.ts
//
// The agent record (docs/agents-design.md §4.1), the species table (§4.2), and the observation and
// action layouts derived from declared sensors and actions (§7). Pure: no device, no core import.
//
// Everything a later stage needs to agree on lives here, once: byte offsets, the WGSL structs, the
// named observation offsets rule brains read, and the slot names the inspector shows. The sense
// kernel, its CPU mirror (agentMath.ts) and any brain are all generated from one AgentDefinition,
// so a layout cannot drift between them (design §1, principle 3).

// ---------------------------------------------------------------------------------------------
// The record — 64 bytes, four vec4. Floats 0–3 match physics P1's body record, so one binning
// kernel serves agents and bodies (design §3 fact 9).
// ---------------------------------------------------------------------------------------------

export const AGENT_FLOATS = 16;
export const AGENT_BYTES = AGENT_FLOATS * 4;
export const AGENT_POSITION_OFFSET = 0;     // xyz, world
export const AGENT_RADIUS_OFFSET = 3;       // body radius, world
export const AGENT_VELOCITY_OFFSET = 4;     // xyz, world; always along the heading (no sideslip)
export const AGENT_HEADING_OFFSET = 7;      // yaw, radians, 0 = +Z, wrapped to [-π, π)
export const AGENT_ENERGY_OFFSET = 8;
export const AGENT_AGE_OFFSET = 9;          // seconds alive
export const AGENT_SPEED_SCALE_OFFSET = 10; // multiplier on the species' max speed
export const AGENT_USER0_OFFSET = 11;
export const AGENT_SPECIES_OFFSET = 12;     // u32
export const AGENT_FLAGS_OFFSET = 13;       // u32
export const AGENT_GENOME_OFFSET = 14;      // u32, Phase 2
export const AGENT_USER1_OFFSET = 15;       // u32

export const AGENT_FLAG_ALIVE = 1;
export const AGENT_FLAG_FROZEN = 2;
/** Flag bits 8–31 belong to the app. */
export const AGENT_FLAG_APP_SHIFT = 8;

/** Species table row: 8 words (32 bytes) — a uniform array stride that is a multiple of 16. */
export const SPECIES_FLOATS = 8;
export const MAX_SPECIES = 16;
export const MAX_NEAREST_K = 8;
export const MAX_OBS_DIM = 256;
export const MAX_ACT_DIM = 16;
/** A meshIndex no batch draws — a hidden instance (the Cycle B demo's trick). */
export const HIDDEN_MESH = 0xFFFF;

// Comments stay out of these templates: thrown strings and template text ship in the bundle.
export const AGENT_STRUCT_WGSL = /* wgsl */ `
struct Agent {
    posRadius: vec4<f32>,
    velHeading: vec4<f32>,
    state: vec4<f32>,
    ids: vec4<u32>,
};
struct Species {
    maxSpeed: f32,
    maxTurn: f32,
    accel: f32,
    senseRadius: f32,
    meshIndex: u32,
    hoverMesh: u32,
    selectedMesh: u32,
    boundsRadius: f32,
};
`;

// ---------------------------------------------------------------------------------------------
// Sensors and actions — declared, never hand-counted (design §7)
// ---------------------------------------------------------------------------------------------

/** Width of each built-in sensor block, and the field offsets inside it. */
export const SELF_WIDTH = 2;      // speed / own max speed, energy
export const NEAREST_WIDTH = 5;   // side, forward (Δpos / senseRadius), Δvel side, forward (/ 2·maxSpeed), present
export const BOUNDS_WIDTH = 2;    // side, forward: the nearest wall's direction × (1 − distance / senseRadius)
export const STEERING_WIDTH = 2;  // turn, throttle

/** CPU stand-in for a custom sensor's WGSL — fills `out[o .. o+width)` for agent `i`. */
export type CustomSensorMirror = (records: Float32Array, u32: Uint32Array, i: number, out: Float32Array, o: number) => void;
/** CPU stand-in for custom actions — integrates agent `i` in place from `act[a .. a+width)`. */
export type CustomActionMirror = (records: Float32Array, u32: Uint32Array, i: number, act: ArrayLike<number>, a: number, dt: number) => void;

export type SensorSpec =
    | { readonly kind: 'self'; readonly name: string }
    | { readonly kind: 'nearest'; readonly name: string; readonly species: readonly string[] | null; readonly k: number }
    | { readonly kind: 'bounds'; readonly name: string }
    | { readonly kind: 'custom'; readonly name: string; readonly width: number; readonly wgsl: string; readonly mirror?: CustomSensorMirror };

export type ActionSpec =
    | { readonly kind: 'steering' }
    | { readonly kind: 'custom'; readonly name: string; readonly width: number; readonly wgsl: string; readonly mirror?: CustomActionMirror };

/**
 * Sensor declarations. A namespace rather than bare exports: a function exported as `self` would
 * shadow the worker global of the same name in any module that imported it.
 */
export const Sensors = {
    /** The agent's own speed (relative to its max) and energy — 2 floats. */
    self(o: { name?: string } = {}): SensorSpec {
        return { kind: 'self', name: o.name ?? 'self' };
    },
    /** The `k` nearest live agents of the given species within the observer's senseRadius — 5 floats each, nearest first. */
    nearest(o: { species?: string | readonly string[]; k?: number; name?: string } = {}): SensorSpec {
        const species = o.species === undefined ? null : typeof o.species === 'string' ? [o.species] : [...o.species];
        return { kind: 'nearest', name: o.name ?? (species ? species.join('_') : 'any'), species, k: o.k ?? 1 };
    },
    /** How close, and which way, the nearest world wall is — 2 floats; zero beyond senseRadius. */
    bounds(o: { name?: string } = {}): SensorSpec {
        return { kind: 'bounds', name: o.name ?? 'bounds' };
    },
    /** App WGSL writing `width` floats. Give a `mirror` to use it in CPU tests and dataset recording. */
    custom(o: { name: string; width: number; wgsl: string; mirror?: CustomSensorMirror }): SensorSpec {
        return { kind: 'custom', name: o.name, width: o.width, wgsl: o.wgsl, mirror: o.mirror };
    },
};

export const Actions = {
    /** `act[0]` turn ∈ [-1, 1] × maxTurn; `act[1]` throttle ∈ [-1, 1] → target speed ∈ [0, maxSpeed·speedScale]. */
    steering(): ActionSpec {
        return { kind: 'steering' };
    },
    /** App WGSL replacing the integrator. */
    custom(o: { name: string; width: number; wgsl: string; mirror?: CustomActionMirror }): ActionSpec {
        return { kind: 'custom', name: o.name, width: o.width, wgsl: o.wgsl, mirror: o.mirror };
    },
};

// ---------------------------------------------------------------------------------------------
// The definition
// ---------------------------------------------------------------------------------------------

export interface SpeciesOptions {
    name: string;
    /** m/s */ maxSpeed: number;
    /** rad/s */ maxTurn: number;
    /** m/s² */ accel: number;
    /** m — must not exceed the cell size (design §6.2) */ senseRadius: number;
    /** Body radius, m — the record's default at spawn. */ radius: number;
    /** Instance-tier mesh indices, used by toInstances. Default: hidden. */
    mesh?: number;
    hoverMesh?: number;
    selectedMesh?: number;
    /** Object-space bounding radius of the mesh, used by toInstances. */
    boundsRadius?: number;
}

export interface AgentOptions {
    capacity: number;
    /** The XZ rectangle agents live in. */
    world: { min: readonly [number, number]; max: readonly [number, number] };
    /** Grid cell edge, m; ≥ every species' senseRadius. */
    cellSize: number;
    species: readonly SpeciesOptions[];
    sensors: readonly SensorSpec[];
    actions?: ActionSpec;
    /** Fixed step, s. Default 1/60. */
    dt?: number;
    /** Steps one frame may run before excess time is dropped. Default 3. */
    maxStepsPerFrame?: number;
    /** The tab-switch guard, s. Default 0.25. */
    maxFrameDelta?: number;
    /** Neighbours scanned per cell before the rest are dropped and counted (design §6.2). Default 64. */
    maxScanPerCell?: number;
    /** Default 262,144 (512 × 512). */
    maxCells?: number;
}

export interface ResolvedSpecies {
    readonly index: number;
    readonly name: string;
    readonly maxSpeed: number;
    readonly maxTurn: number;
    readonly accel: number;
    readonly senseRadius: number;
    readonly radius: number;
    readonly mesh: number;
    readonly hoverMesh: number;
    readonly selectedMesh: number;
    readonly boundsRadius: number;
}

/** One block of the observation row: a whole sensor, or one neighbour of a `nearest` sensor. */
export interface ObsSlot {
    readonly sensor: SensorSpec;
    /** Neighbour rank for `nearest` (0 = closest); 0 otherwise. */
    readonly rank: number;
    readonly offset: number;
    readonly width: number;
    /** The WGSL constant naming `offset`, e.g. `OBS_WOLF_0`. */
    readonly constant: string;
}

export interface AgentDefinition {
    readonly capacity: number;
    readonly world: { readonly minX: number; readonly minZ: number; readonly maxX: number; readonly maxZ: number };
    readonly cellSize: number;
    readonly grid: { readonly cellsX: number; readonly cellsZ: number; readonly cellTotal: number };
    readonly species: readonly ResolvedSpecies[];
    readonly sensors: readonly SensorSpec[];
    readonly actions: ActionSpec;
    readonly obsDim: number;
    readonly actDim: number;
    readonly obsSlots: readonly ObsSlot[];
    /** One name per observation float, for the inspector ("wolf[0].forward"). */
    readonly obsNames: readonly string[];
    readonly actNames: readonly string[];
    /** Every named offset, as emitted into WGSL by constantsWGSL(). */
    readonly constants: Readonly<Record<string, number>>;
    /** Species bitmask per `nearest` sensor, in sensor order (all bits for "any species"). */
    readonly nearestMasks: readonly number[];
    readonly dt: number;
    readonly maxStepsPerFrame: number;
    readonly maxFrameDelta: number;
    readonly maxScanPerCell: number;
    speciesIndex(name: string): number;
}

const IDENT = /^[A-Za-z][A-Za-z0-9_]*$/;
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const positiveInt = (v: unknown): v is number => Number.isInteger(v) && (v as number) >= 1;

function fail(message: string): never {
    throw new Error(`AgentSystem: ${message}`);
}

/**
 * Validate the declarations and derive everything the stages agree on: the grid, the packed
 * species, the observation and action dimensions, named offsets and slot names. Every refusal
 * happens here, at construction — never per frame (design-principles §2).
 */
export function defineAgents(o: AgentOptions): AgentDefinition {
    if (!positiveInt(o.capacity)) fail(`capacity ${o.capacity} must be a positive integer.`);

    const [minX, minZ] = o.world?.min ?? [NaN, NaN];
    const [maxX, maxZ] = o.world?.max ?? [NaN, NaN];
    if (![minX, minZ, maxX, maxZ].every(finite) || !(minX < maxX && minZ < maxZ)) {
        fail('world min must be finite and below max on both axes.');
    }
    if (!(finite(o.cellSize) && o.cellSize > 0)) fail(`cellSize ${o.cellSize} must be positive.`);
    const cellsX = Math.ceil((maxX - minX) / o.cellSize);
    const cellsZ = Math.ceil((maxZ - minZ) / o.cellSize);
    const maxCells = o.maxCells ?? 262_144;
    if (cellsX * cellsZ > maxCells) {
        fail(`${cellsX}×${cellsZ} cells exceeds maxCells ${maxCells}; raise cellSize or maxCells.`);
    }

    const dt = o.dt ?? 1 / 60;
    const maxStepsPerFrame = o.maxStepsPerFrame ?? 3;
    const maxFrameDelta = o.maxFrameDelta ?? 0.25;
    const maxScanPerCell = o.maxScanPerCell ?? 64;
    if (!(finite(dt) && dt > 0)) fail(`dt ${dt} must be positive.`);
    if (!positiveInt(maxStepsPerFrame)) fail(`maxStepsPerFrame ${maxStepsPerFrame} must be a positive integer.`);
    if (!(finite(maxFrameDelta) && maxFrameDelta >= dt)) fail(`maxFrameDelta ${maxFrameDelta} must be at least dt ${dt}.`);
    if (!positiveInt(maxScanPerCell)) fail(`maxScanPerCell ${maxScanPerCell} must be a positive integer.`);

    // --- species ---
    const list = o.species ?? [];
    if (list.length < 1 || list.length > MAX_SPECIES) fail(`declare 1–${MAX_SPECIES} species; got ${list.length}.`);
    const byName = new Map<string, number>();
    const species: ResolvedSpecies[] = list.map((s, index) => {
        if (typeof s.name !== 'string' || !IDENT.test(s.name)) fail(`species name "${s.name}" must be an identifier (letters, digits, _).`);
        if (byName.has(s.name)) fail(`species "${s.name}" is declared twice.`);
        byName.set(s.name, index);
        for (const key of ['maxSpeed', 'accel', 'senseRadius', 'radius'] as const) {
            if (!(finite(s[key]) && s[key] > 0)) fail(`species "${s.name}" ${key} ${s[key]} must be positive.`);
        }
        if (!(finite(s.maxTurn) && s.maxTurn >= 0)) fail(`species "${s.name}" maxTurn ${s.maxTurn} must be ≥ 0.`);
        if (s.senseRadius > o.cellSize) {
            fail(`species "${s.name}" senses ${s.senseRadius} m but cells are ${o.cellSize} m — a neighbour two cells ` +
                `away would be missed. Raise cellSize to ≥ ${s.senseRadius}.`);
        }
        const mesh = s.mesh ?? HIDDEN_MESH;
        return {
            index, name: s.name, maxSpeed: s.maxSpeed, maxTurn: s.maxTurn, accel: s.accel,
            senseRadius: s.senseRadius, radius: s.radius,
            mesh, hoverMesh: s.hoverMesh ?? mesh, selectedMesh: s.selectedMesh ?? mesh,
            boundsRadius: s.boundsRadius ?? 0,
        };
    });
    const speciesIndex = (name: string): number => {
        const i = byName.get(name);
        if (i === undefined) fail(`species "${name}" is not declared.`);
        return i;
    };
    const allSpecies = (1 << species.length) - 1;

    // --- observation layout ---
    const sensors = o.sensors ?? [];
    if (sensors.length < 1) fail('declare at least one sensor — a brain needs something to read.');
    const obsSlots: ObsSlot[] = [];
    const obsNames: string[] = [];
    const constants: Record<string, number> = {};
    const nearestMasks: number[] = [];
    const sensorNames = new Set<string>();
    let obsDim = 0;
    const constant = (name: string, value: number) => {
        if (name in constants) fail(`two declarations both produce the WGSL constant ${name}; give one a different name.`);
        constants[name] = value;
    };
    for (const s of sensors) {
        if (typeof s.name !== 'string' || !IDENT.test(s.name)) fail(`sensor name "${s.name}" must be an identifier (letters, digits, _).`);
        if (sensorNames.has(s.name)) fail(`two sensors are both named "${s.name}"; pass a distinct name.`);
        sensorNames.add(s.name);
        const upper = s.name.toUpperCase();
        if (s.kind === 'nearest') {
            if (!(Number.isInteger(s.k) && s.k >= 1 && s.k <= MAX_NEAREST_K)) fail(`sensor "${s.name}" k ${s.k} must be 1–${MAX_NEAREST_K}.`);
            nearestMasks.push(s.species ? s.species.reduce((m, n) => m | (1 << speciesIndex(n)), 0) : allSpecies);
            for (let r = 0; r < s.k; r++) {
                obsSlots.push({ sensor: s, rank: r, offset: obsDim, width: NEAREST_WIDTH, constant: `OBS_${upper}_${r}` });
                constant(`OBS_${upper}_${r}`, obsDim);
                for (const f of ['side', 'forward', 'velSide', 'velForward', 'present']) obsNames.push(`${s.name}[${r}].${f}`);
                obsDim += NEAREST_WIDTH;
            }
            continue;
        }
        let width: number;
        let fields: string[];
        if (s.kind === 'self') { width = SELF_WIDTH; fields = ['speed', 'energy']; }
        else if (s.kind === 'bounds') { width = BOUNDS_WIDTH; fields = ['side', 'forward']; }
        else {
            if (!positiveInt(s.width)) fail(`sensor "${s.name}" width ${s.width} must be a positive integer.`);
            if (typeof s.wgsl !== 'string') fail(`sensor "${s.name}" needs its wgsl.`);
            width = s.width;
            fields = Array.from({ length: width }, (_, k) => String(k));
        }
        obsSlots.push({ sensor: s, rank: 0, offset: obsDim, width, constant: `OBS_${upper}` });
        constant(`OBS_${upper}`, obsDim);
        for (const f of fields) obsNames.push(`${s.name}.${f}`);
        obsDim += width;
    }
    if (obsDim > MAX_OBS_DIM) fail(`the sensors produce ${obsDim} observation floats; the limit is ${MAX_OBS_DIM}.`);

    // --- action layout ---
    const actions: ActionSpec = o.actions ?? Actions.steering();
    let actDim: number;
    let actNames: string[];
    if (actions.kind === 'steering') {
        actDim = STEERING_WIDTH;
        actNames = ['turn', 'throttle'];
    } else {
        if (!(Number.isInteger(actions.width) && actions.width >= 1 && actions.width <= MAX_ACT_DIM)) {
            fail(`actions "${actions.name}" width ${actions.width} must be 1–${MAX_ACT_DIM}.`);
        }
        if (typeof actions.wgsl !== 'string') fail(`actions "${actions.name}" needs its wgsl.`);
        actDim = actions.width;
        actNames = Array.from({ length: actDim }, (_, k) => `${actions.name}.${k}`);
    }

    constant('OBS_DIM', obsDim);
    constant('ACT_DIM', actDim);
    // Field offsets inside the built-in blocks — shared by every sensor of a kind.
    Object.assign(constants, {
        SELF_SPEED: 0, SELF_ENERGY: 1,
        NEAR_SIDE: 0, NEAR_FORWARD: 1, NEAR_VEL_SIDE: 2, NEAR_VEL_FORWARD: 3, NEAR_PRESENT: 4,
        BOUNDS_SIDE: 0, BOUNDS_FORWARD: 1,
    });
    if (actions.kind === 'steering') Object.assign(constants, { ACT_TURN: 0, ACT_THROTTLE: 1 });

    return {
        capacity: o.capacity,
        world: { minX, minZ, maxX, maxZ },
        cellSize: o.cellSize,
        grid: { cellsX, cellsZ, cellTotal: cellsX * cellsZ },
        species, sensors: [...sensors], actions,
        obsDim, actDim, obsSlots, obsNames, actNames, constants, nearestMasks,
        dt, maxStepsPerFrame, maxFrameDelta, maxScanPerCell,
        speciesIndex,
    };
}

/** The species table as uploaded: MAX_SPECIES rows of SPECIES_FLOATS words; u32 fields as raw bits. */
export function packSpecies(def: AgentDefinition, out: Float32Array = new Float32Array(MAX_SPECIES * SPECIES_FLOATS)): Float32Array {
    if (out.length < MAX_SPECIES * SPECIES_FLOATS) fail(`packSpecies needs ${MAX_SPECIES * SPECIES_FLOATS} floats; got ${out.length}.`);
    const u = new Uint32Array(out.buffer, out.byteOffset, out.length);
    out.fill(0, 0, MAX_SPECIES * SPECIES_FLOATS);
    for (const s of def.species) {
        const b = s.index * SPECIES_FLOATS;
        out[b] = s.maxSpeed; out[b + 1] = s.maxTurn; out[b + 2] = s.accel; out[b + 3] = s.senseRadius;
        u[b + 4] = s.mesh; u[b + 5] = s.hoverMesh; u[b + 6] = s.selectedMesh;
        out[b + 7] = s.boundsRadius;
    }
    return out;
}

/** Every named offset as WGSL `const` declarations, for rule brains and generated kernels. */
export function constantsWGSL(def: AgentDefinition): string {
    return Object.entries(def.constants).map(([k, v]) => `const ${k} = ${v}u;`).join('\n') + '\n';
}

/** Refuse a brain or model whose dimensions differ from the declared layout. */
export function assertBrainDims(def: AgentDefinition, obsDim: number, actDim: number): void {
    if (obsDim !== def.obsDim || actDim !== def.actDim) {
        fail(`the brain maps ${obsDim}→${actDim}; the layout is ${def.obsDim}→${def.actDim}.`);
    }
}
