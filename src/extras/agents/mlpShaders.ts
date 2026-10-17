import { mlpLayout, type MlpLayer, type MlpLayerLayout, type MlpNetwork } from './mlpMath';
import { speciesConstantName } from './agentLayout';

const TILE_QUADS = 512;          // 8 KB of workgroup memory per tile, half the default 16 KB limit (S6)

function rowLoop(layer: MlpLayer, packed: MlpLayerLayout, index: number, r0: number, r1: number,
    base: string, weight: string): string {
    const quads = packed.inputStride / 4;
    let code = `    for (var r = ${r0}u; r < ${r1}u; r++) {
        var z = mlpWeights[${base}${packed.biasOffset / 4}u + r / 4u][r % 4u];
        for (var k = 0u; k < ${quads}u; k++) {
            z += dot(${weight}, x${index}[k]);
        }
`;
    if (layer.activation === 'relu') code += '        z = max(z, 0.0);\n';
    else if (layer.activation === 'tanh') code += '        z = tanh(clamp(z, -15.0, 15.0));\n';
    else if (layer.activation === 'sigmoid') code += `        let e = exp(-abs(z));
        z = select(e / (1.0 + e), 1.0 / (1.0 + e), z >= 0.0);
`;
    return `${code}        x${index + 1}[r / 4u][r % 4u] = z;
    }
`;
}

/**
 * Think contribution only: AgentSystem supplies observations, actions and the shared gating entry.
 * - Without `rows`: one network, phase 6's code byte for byte.
 * - `rows` as species names (`MlpBrain.bySpecies`, agents-species-brains-design.md §5): each agent
 *   reads its species' row, chosen by the system's species constants.
 * - `rows` as a row count (`MlpBrain.byGenome`, agents-genome-rows-design.md §6): a cooperative
 *   `thinkBlock` that stages the 64-agent block's row in workgroup memory a tile at a time and falls
 *   back to the agent's own row in storage when its `genome` word differs; an out-of-range genome
 *   writes NaN actions without reading outside the table.
 */
export function buildMlpThinkWGSL(network: MlpNetwork, rows?: readonly string[] | number): string {
    const layout = mlpLayout(network);
    if (typeof rows === 'number') {
        if (!(Number.isInteger(rows) && rows >= 1)) throw new Error('MlpBrain: byGenome needs at least one network.');
        const rowQuads = layout.floats / 4;
        let code = `@group(1) @binding(0) var<storage, read> mlpWeights: array<vec4<f32>>;
var<workgroup> stagedRow: u32;
var<workgroup> mlpTile: array<vec4<f32>, ${TILE_QUADS}>;
fn thinkBlock(i: u32, species: u32, tick: u32, thinking: bool, lane: u32) {
    if (lane == 0u) { stagedRow = min(agents[i].ids.z, ${rows - 1}u) * ${rowQuads}u; }
    let rowBase = workgroupUniformLoad(&stagedRow);
    var inTable = false;
    var myBase = 0u;
    var x0: array<vec4<f32>, ${Math.ceil(network[0].input / 4)}>;
    if (thinking) {
        let g = agents[i].ids.z;
        if (g < ${rows}u) {
            inTable = true;
            myBase = g * ${rowQuads}u;
            for (var k = 0u; k < ${network[0].input}u; k++) {
                x0[k / 4u][k % 4u] = obs(i, k);
            }
        }
    }
`;
        network.forEach((layer, index) => {
            const packed = layout.layers[index], quads = packed.inputStride / 4;
            const tileRows = Math.min(layer.output, Math.floor(TILE_QUADS / quads));
            code += `    var x${index + 1}: array<vec4<f32>, ${packed.outputStride / 4}>;\n`;
            for (let r0 = 0; r0 < layer.output; r0 += tileRows) {
                const r1 = Math.min(layer.output, r0 + tileRows);
                code += `    workgroupBarrier();
    for (var q = lane; q < ${(r1 - r0) * quads}u; q += 64u) {
        mlpTile[q] = mlpWeights[rowBase + ${packed.weightOffset / 4 + r0 * quads}u + q];
    }
    workgroupBarrier();
    if (inTable) {
        if (myBase == rowBase) {
${rowLoop(layer, packed, index, r0, r1, 'rowBase + ', `mlpTile[(r - ${r0}u) * ${quads}u + k]`)}        } else {
${rowLoop(layer, packed, index, r0, r1, 'myBase + ', `mlpWeights[myBase + ${packed.weightOffset / 4}u + r * ${quads}u + k]`)}        }
    }
`;
            }
        });
        return `${code}    if (thinking) {
        var nan = 0x7fc00000u;
        for (var k = 0u; k < ${network[network.length - 1].output}u; k++) {
            if (inTable) { setAct(i, k, x${network.length}[k / 4u][k % 4u]); }
            else { setAct(i, k, bitcast<f32>(nan)); }
        }
        setReason(i, 0u);
    }
}`;
    }
    const at = rows ? 'row + ' : '';
    let code = `@group(1) @binding(0) var<storage, read> mlpWeights: array<vec4<f32>>;
fn think(i: u32, species: u32, tick: u32) {
`;
    if (rows) {
        code += `    var row = 0u;
    switch species {
${rows.map((name, r) => `        case ${speciesConstantName(name)}: { row = ${r * layout.floats / 4}u; }`).join('\n')}
        default: {}
    }
`;
    }
    code += `    var x0: array<vec4<f32>, ${Math.ceil(network[0].input / 4)}>;
    for (var k = 0u; k < ${network[0].input}u; k++) {
        x0[k / 4u][k % 4u] = obs(i, k);
    }
`;
    network.forEach((layer, index) => {
        const packed = layout.layers[index], quads = packed.inputStride / 4;
        code += `    var x${index + 1}: array<vec4<f32>, ${packed.outputStride / 4}>;\n` +
            rowLoop(layer, packed, index, 0, layer.output, at, `mlpWeights[${at}${packed.weightOffset / 4}u + r * ${quads}u + k]`);
    });
    return `${code}    for (var k = 0u; k < ${network[network.length - 1].output}u; k++) {
        setAct(i, k, x${network.length}[k / 4u][k % 4u]);
    }
    setReason(i, 0u);
}`;
}
