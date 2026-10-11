// Bounded nearest query: one invocation per agent, distance/index insertion ordering.
import { pointCellWGSL } from '../spatial/gridShaders';
import { AGENT_STRUCT_WGSL, MAX_NEAREST_K, type AgentDefinition } from './agentLayout';

/** Output is k indices per agent, padded with 0xffffffff. Dead query agents match the CPU mirror.
 * Atomic scatter order is unspecified: over-budget query subsets may vary between builds. */
export function buildAgentGridQueryWGSL(def: AgentDefinition, mask: number, radius: number, k: number,
    options: { stride?: number; base?: number } = {}): string {
    const stride = options.stride ?? k, base = options.base ?? 0;
    if (!(Number.isFinite(radius) && radius >= 0 && radius <= def.cellSize &&
        Number.isInteger(k) && k >= 1 && k <= MAX_NEAREST_K &&
        Number.isInteger(mask) && mask >= 0 && mask <= 0xffffffff)) {
        throw new Error('AgentSystem: grid query radius, k or species mask is invalid.');
    }
    if (!(Number.isSafeInteger(stride) && Number.isSafeInteger(base) && base >= 0 && stride >= base + k)) {
        throw new Error('AgentSystem: grid query stride and base must fit the sensor slice.');
    }
    return `${AGENT_STRUCT_WGSL}
@group(0) @binding(0) var<storage, read> agents: array<Agent>;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read> bodies: array<u32>;
@group(0) @binding(4) var<storage, read_write> result: array<u32>;
${pointCellWGSL(def)}
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= ${def.capacity}u) { return; }
    let pos = agents[i].posRadius.xyz;
    let cell = pointCell(pos);
    let cx = i32(cell % ${def.grid.cellsX}u);
    let cz = i32(cell / ${def.grid.cellsX}u);
    var indices: array<u32, ${k}>;
    var distances: array<f32, ${k}>;
    var found = 0u;
    for (var n = 0u; n < ${k}u; n++) { indices[n] = 0xffffffffu; }
    for (var z = max(0, cz - 1); z <= min(${def.grid.cellsZ - 1}, cz + 1); z++) {
        for (var x = max(0, cx - 1); x <= min(${def.grid.cellsX - 1}, cx + 1); x++) {
            let c = u32(z) * ${def.grid.cellsX}u + u32(x);
            let end = offsets[c] + min(counts[c], ${def.maxScanPerCell}u);
            for (var at = offsets[c]; at < end; at++) {
                let j = bodies[at];
                if (j == i || (agents[j].ids.y & 1u) == 0u ||
                    ((${mask}u >> agents[j].ids.x) & 1u) == 0u) { continue; }
                let delta = agents[j].posRadius.xz - pos.xz;
                let d2 = dot(delta, delta);
                if (d2 > ${radius * radius}) { continue; }
                var p = 0u;
                while (p < found) {
                    if (distances[p] > d2 || (distances[p] == d2 && indices[p] > j)) { break; }
                    p++;
                }
                if (p == ${k}u) { continue; }
                var n = min(found, ${k - 1}u);
                while (n > p) { distances[n] = distances[n - 1u]; indices[n] = indices[n - 1u]; n--; }
                distances[p] = d2;
                indices[p] = j;
                found = min(${k}u, found + 1u);
            }
        }
    }
    for (var n = 0u; n < ${k}u; n++) { result[i * ${stride}u + ${base ? `${base}u + ` : ''}n] = indices[n]; }
}`;
}