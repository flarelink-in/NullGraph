import { AGENT_STRUCT_WGSL, constantsWGSL, type AgentDefinition } from './agentLayout';
import { AGENT_STATE_WGSL } from './agentSystemShaders';

/** Shared by sense and the forthcoming think stage. Newborns never reuse a slot's old row. */
export function buildAgentThinkPredicateWGSL(thinkEvery: number): string {
    if (!(Number.isInteger(thinkEvery) && thinkEvery > 0 && thinkEvery <= 0xffffffff)) {
        throw new Error(`AgentSystem: thinkEvery ${thinkEvery} must be a positive integer.`);
    }
    return `const THINK_EVERY = ${thinkEvery}u;
fn thinksNow(i: u32, a: Agent) -> bool {
    return (a.ids.y & 1u) != 0u && (a.state.y == 0.0 || (state.tick + i / 64u) % THINK_EVERY == 0u);
}`;
}

/** One observation row per invocation, in exactly the declaration order of senseAgent(). */
export function buildAgentSenseWGSL(def: AgentDefinition, thinkEvery = 1): string {
    const predicate = buildAgentThinkPredicateWGSL(thinkEvery);
    const stride = def.sensors.reduce((n, s) => n + (s.kind === 'nearest' ? s.k : 0), 0);
    const custom: string[] = [], blocks: string[] = [];
    let offset = 0, base = 0;
    for (const sensor of def.sensors) {
        if (sensor.kind === 'self') {
            blocks.push(`obs[row + ${offset}u] = clamp(length(a.velHeading.xz) / (sp.maxSpeed * a.state.z), -1.0, 1.0);
    obs[row + ${offset + 1}u] = clamp(a.state.x, -1.0, 1.0);`);
            offset += 2;
        } else if (sensor.kind === 'nearest') {
            blocks.push(`for (var rank = 0u; rank < ${sensor.k}u; rank++) {
        let at = row + ${offset}u + rank * 5u;
        for (var field = 0u; field < 5u; field++) { obs[at + field] = 0.0; }
        let j = neighbours[i * ${stride}u + ${base}u + rank];
        if (j == 0xffffffffu) { continue; }
        let other = agents[j];
        let delta = other.posRadius.xz - a.posRadius.xz;
        if (dot(delta, delta) > sp.senseRadius * sp.senseRadius) { continue; }
        let dv = other.velHeading.xz - a.velHeading.xz;
        obs[at] = clamp(dot(delta, side) / sp.senseRadius, -1.0, 1.0);
        obs[at + 1u] = clamp(dot(delta, forward) / sp.senseRadius, -1.0, 1.0);
        obs[at + 2u] = clamp(dot(dv, side) / (2.0 * sp.maxSpeed), -1.0, 1.0);
        obs[at + 3u] = clamp(dot(dv, forward) / (2.0 * sp.maxSpeed), -1.0, 1.0);
        obs[at + 4u] = 1.0;
    }`);
            offset += sensor.k * 5;
            base += sensor.k;
        } else if (sensor.kind === 'bounds') {
            const w = def.world;
            blocks.push(`{
        let p = a.posRadius.xz;
        var distance = p.x - ${w.minX};
        var normal = vec2<f32>(-1.0, 0.0);
        if (${w.maxX} - p.x < distance) { distance = ${w.maxX} - p.x; normal = vec2<f32>(1.0, 0.0); }
        if (p.y - ${w.minZ} < distance) { distance = p.y - ${w.minZ}; normal = vec2<f32>(0.0, -1.0); }
        if (${w.maxZ} - p.y < distance) { distance = ${w.maxZ} - p.y; normal = vec2<f32>(0.0, 1.0); }
        let magnitude = max(0.0, 1.0 - max(distance, 0.0) / sp.senseRadius);
        obs[row + ${offset}u] = dot(normal, side) * magnitude;
        obs[row + ${offset + 1}u] = dot(normal, forward) * magnitude;
    }`);
            offset += 2;
        } else {
            // Comments cannot satisfy the signature check. Full WGSL validation belongs to the device.
            const text = sensor.wgsl.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '');
            const signature = new RegExp(`\\bfn\\s+${sensor.name}\\s*\\(\\s*i\\s*:\\s*u32\\s*,\\s*a\\s*:\\s*Agent\\s*\\)\\s*->\\s*array\\s*<\\s*f32\\s*,\\s*${sensor.width}u?\\s*>\\s*\\{`);
            if (!signature.test(text)) {
                throw new Error(`AgentSystem: custom sensor "${sensor.name}" must define fn ${sensor.name}(i: u32, a: Agent) -> array<f32, ${sensor.width}>.`);
            }
            custom.push(sensor.wgsl);
            blocks.push(`{
        let values = ${sensor.name}(i, a);
        for (var field = 0u; field < ${sensor.width}u; field++) { obs[row + ${offset}u + field] = values[field]; }
    }`);
            offset += sensor.width;
        }
    }
    return `${AGENT_STRUCT_WGSL}${AGENT_STATE_WGSL}${constantsWGSL(def)}
@group(0) @binding(0) var<storage, read> agents: array<Agent>;
@group(0) @binding(1) var<storage, read> neighbours: array<u32>;
@group(0) @binding(2) var<storage, read> state: AgentState;
@group(0) @binding(3) var<storage, read_write> obs: array<f32>;
@group(0) @binding(4) var<uniform> species: array<Species, 16>;
${predicate}
${custom.join('\n')}
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= ${def.capacity}u) { return; }
    let a = agents[i];
    if (!thinksNow(i, a)) { return; }
    // Preserve auto-layout bindings without reading another invocation's observation row.
    _ = arrayLength(&neighbours);
    _ = arrayLength(&obs);
    let sp = species[a.ids.x];
    _ = sp.maxSpeed;
    let sn = sin(a.velHeading.w);
    let cs = cos(a.velHeading.w);
    let side = vec2<f32>(cs, -sn);
    let forward = vec2<f32>(sn, cs);
    let row = i * OBS_DIM;
    ${blocks.join('\n    ')}
}`;
}