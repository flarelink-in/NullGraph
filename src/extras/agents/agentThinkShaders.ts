import { AGENT_STRUCT_WGSL, constantsWGSL, type AgentDefinition } from './agentLayout';
import { AGENT_STATE_WGSL } from './agentSystemShaders';
import { buildAgentThinkPredicateWGSL } from './agentSenseShaders';
import { agentSignatureMatches, assertThinkSignature } from './RuleBrain';

export const AGENT_RANDOM_WGSL = `
fn rand01(i: u32, tick: u32, stream: u32) -> f32 {
    let seed = i ^ (tick * 0x9e3779b9u) ^ (stream * 0x85ebca6bu);
    let hashState = seed * 747796405u + 2891336453u;
    let word = ((hashState >> ((hashState >> 28u) + 4u)) ^ hashState) * 277803737u;
    return f32(((word >> 22u) ^ word) >> 8u) * (1.0 / 16777216.0);
}`;

function speciesConstants(def: AgentDefinition): string {
    const names = new Set<string>();
    return def.species.map(s => {
        const name = `SPECIES_${s.name.toUpperCase()}`;
        if (names.has(name) || Object.hasOwn(def.constants, name)) {
            throw new Error(`AgentSystem: generated constant "${name}" is declared twice.`);
        }
        names.add(name);
        return `const ${name} = ${s.index}u;`;
    }).join('\n');
}

export function buildAgentThinkWGSL(def: AgentDefinition, wgsl: string, thinkEvery = 1): string {
    assertThinkSignature(wgsl);
    return `${AGENT_STRUCT_WGSL}${AGENT_STATE_WGSL}${constantsWGSL(def)}
${speciesConstants(def)}
@group(0) @binding(0) var<storage, read> agents: array<Agent>;
@group(0) @binding(1) var<storage, read> observations: array<f32>;
@group(0) @binding(2) var<storage, read> state: AgentState;
@group(0) @binding(3) var<storage, read_write> act: array<f32>;
@group(0) @binding(4) var<storage, read_write> reason: array<u32>;
@group(0) @binding(5) var<uniform> species: array<Species, 16>;
${buildAgentThinkPredicateWGSL(thinkEvery)}
${AGENT_RANDOM_WGSL}
fn obs(i: u32, k: u32) -> f32 { return observations[i * OBS_DIM + k]; }
fn setAct(i: u32, k: u32, v: f32) { act[i * ACT_DIM + k] = v; }
fn setReason(i: u32, r: u32) { reason[i] = r; }
${wgsl}
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= ${def.capacity}u) { return; }
    let a = agents[i];
    if (!thinksNow(i, a)) { return; }
    _ = arrayLength(&observations);
    _ = arrayLength(&act);
    _ = arrayLength(&reason);
    _ = species[a.ids.x].maxSpeed;
    think(i, a.ids.x, state.tick);
}`;
}

export function buildAgentStepWGSL(def: AgentDefinition, name: string, wgsl: string): string {
    if (!agentSignatureMatches(wgsl, /\bfn\s+step\s*\(\s*i\s*:\s*u32\s*,\s*tick\s*:\s*u32\s*\)\s*\{/)) {
        throw new Error(`AgentSystem: step stage "${name}" must define fn step(i: u32, tick: u32).`);
    }
    let stride = 0;
    const offsets: string[] = [];
    for (const sensor of def.sensors) {
        if (sensor.kind !== 'nearest') continue;
        offsets.push(`const NEIGHBOURS_${sensor.name.toUpperCase()} = ${stride}u;`);
        stride += sensor.k;
    }
    const w = def.world;
    return `${AGENT_STRUCT_WGSL}${AGENT_STATE_WGSL}${constantsWGSL(def)}
${speciesConstants(def)}
const DT = ${def.dt};
const NO_AGENT = 0xffffffffu;
const NEIGHBOURS_PER_AGENT = ${stride}u;
${offsets.join('\n')}
@group(0) @binding(0) var<storage, read> neighbours: array<u32>;
@group(0) @binding(1) var<storage, read> obs: array<f32>;
@group(0) @binding(2) var<storage, read> act: array<f32>;
@group(0) @binding(3) var<storage, read> state: AgentState;
@group(0) @binding(4) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(5) var<uniform> species: array<Species, 16>;
${AGENT_RANDOM_WGSL}
fn randomPointInWorld(i: u32, tick: u32) -> vec2<f32> {
    let radius = agents[i].posRadius.w;
    let lo = min(vec2<f32>(${w.minX}, ${w.minZ}) + radius, vec2<f32>(${(w.minX + w.maxX) / 2}, ${(w.minZ + w.maxZ) / 2}));
    let hi = max(vec2<f32>(${w.maxX}, ${w.maxZ}) - radius, vec2<f32>(${(w.minX + w.maxX) / 2}, ${(w.minZ + w.maxZ) / 2}));
    return lo + (hi - lo) * vec2<f32>(rand01(i, tick, 0u), rand01(i, tick, 1u));
}
fn respawn(i: u32, xz: vec2<f32>, heading: f32) {
    agents[i].posRadius.x = xz.x;
    agents[i].posRadius.z = xz.y;
    agents[i].velHeading = vec4<f32>(0.0, 0.0, 0.0, heading - 6.283185307179586 * floor((heading + 3.141592653589793) / 6.283185307179586));
    agents[i].state.y = 0.0;
}
${wgsl}
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= ${def.capacity}u) { return; }
    if ((agents[i].ids.y & 1u) == 0u) { return; }
    _ = arrayLength(&neighbours);
    _ = arrayLength(&obs);
    _ = arrayLength(&act);
    _ = species[agents[i].ids.x].maxSpeed;
    step(i, state.tick);
}`;
}