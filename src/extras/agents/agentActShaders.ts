import { AGENT_STRUCT_WGSL, constantsWGSL, type AgentDefinition } from './agentLayout';

/** In-place CPU actAgent twin: age, frozen, sanitize, steer, integrate, X then Z reflection. */
export function buildAgentActWGSL(def: AgentDefinition): string {
    let custom = '', body: string;
    if (def.actions.kind === 'custom') {
        const action = def.actions;
        const text = action.wgsl.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '');
        if (!new RegExp(`\\bfn\\s+${action.name}\\s*\\(\\s*i\\s*:\\s*u32\\s*\\)\\s*\\{`).test(text)) {
            throw new Error(`AgentSystem: custom actions "${action.name}" must define fn ${action.name}(i: u32).`);
        }
        custom = action.wgsl;
        body = `agents[i] = a;
    ${action.name}(i);`;
    } else {
        const w = def.world;
        body = `let sp = species[a.ids.x];
    let a0 = act[i * ACT_DIM];
    let a1 = act[i * ACT_DIM + 1u];
    var turn = 0.0;
    var targetSpeed = 0.0;
    if ((bitcast<u32>(a0) & 0x7f800000u) != 0x7f800000u &&
        (bitcast<u32>(a1) & 0x7f800000u) != 0x7f800000u) {
        turn = clamp(a0, -1.0, 1.0);
        targetSpeed = (clamp(a1, -1.0, 1.0) + 1.0) * 0.5 * sp.maxSpeed * a.state.z;
    }
    var h = wrapAngle(a.velHeading.w + turn * sp.maxTurn * DT);
    let dv = sp.accel * DT;
    var speed = length(a.velHeading.xz);
    speed += clamp(targetSpeed - speed, -dv, dv);
    var fx = sin(h);
    var fz = cos(h);
    var x = a.posRadius.x + fx * speed * DT;
    var z = a.posRadius.z + fz * speed * DT;
    let radius = a.posRadius.w;
    var loX = ${w.minX} + radius;
    var hiX = ${w.maxX} - radius;
    var loZ = ${w.minZ} + radius;
    var hiZ = ${w.maxZ} - radius;
    if (loX > hiX) { loX = ${(w.minX + w.maxX) / 2}; hiX = loX; }
    if (loZ > hiZ) { loZ = ${(w.minZ + w.maxZ) / 2}; hiZ = loZ; }
    if (x < loX) { x = loX; if (fx < 0.0) { h = wrapAngle(-h); } }
    else if (x > hiX) { x = hiX; if (fx > 0.0) { h = wrapAngle(-h); } }
    fz = cos(h);
    if (z < loZ) { z = loZ; if (fz < 0.0) { h = wrapAngle(PI - h); } }
    else if (z > hiZ) { z = hiZ; if (fz > 0.0) { h = wrapAngle(PI - h); } }
    fx = sin(h); fz = cos(h);
    a.posRadius.x = x;
    a.posRadius.z = z;
    a.velHeading = vec4<f32>(fx * speed, 0.0, fz * speed, h);
    agents[i] = a;`;
    }
    return `${AGENT_STRUCT_WGSL}${constantsWGSL(def)}
const DT = ${def.dt};
const PI = 3.141592653589793;
const TWO_PI = 6.283185307179586;
@group(0) @binding(0) var<storage, read> act: array<f32>;
@group(0) @binding(1) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(2) var<uniform> species: array<Species, 16>;
fn wrapAngle(h: f32) -> f32 { return h - TWO_PI * floor((h + PI) / TWO_PI); }
${custom}
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= ${def.capacity}u) { return; }
    var a = agents[i];
    if ((a.ids.y & 1u) == 0u) { return; }
    _ = arrayLength(&act);
    _ = species[a.ids.x].maxSpeed;
    a.state.y += DT;
    if ((a.ids.y & 2u) != 0u) {
        a.velHeading = vec4<f32>(0.0, 0.0, 0.0, a.velHeading.w);
        agents[i] = a;
        return;
    }
    ${body}
}`;
}