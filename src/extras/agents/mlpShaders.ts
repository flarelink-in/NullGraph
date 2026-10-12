import { mlpLayout, type MlpNetwork } from './mlpMath';

/** Think contribution only: AgentSystem supplies observations, actions and the shared gating entry. */
export function buildMlpThinkWGSL(network: MlpNetwork): string {
    const layout = mlpLayout(network);
    let code = `@group(1) @binding(0) var<storage, read> mlpWeights: array<vec4<f32>>;
fn think(i: u32, species: u32, tick: u32) {
    var x0: array<vec4<f32>, ${Math.ceil(network[0].input / 4)}>;
    for (var k = 0u; k < ${network[0].input}u; k++) {
        x0[k / 4u][k % 4u] = obs(i, k);
    }
`;
    network.forEach((layer, index) => {
        const packed = layout.layers[index], next = index + 1;
        const quads = packed.inputStride / 4;
        code += `    var x${next}: array<vec4<f32>, ${packed.outputStride / 4}>;
    for (var r = 0u; r < ${layer.output}u; r++) {
        var z = mlpWeights[${packed.biasOffset / 4}u + r / 4u][r % 4u];
        for (var k = 0u; k < ${quads}u; k++) {
            z += dot(mlpWeights[${packed.weightOffset / 4}u + r * ${quads}u + k], x${index}[k]);
        }
`;
        if (layer.activation === 'relu') code += '        z = max(z, 0.0);\n';
        else if (layer.activation === 'tanh') code += '        z = tanh(clamp(z, -15.0, 15.0));\n';
        else if (layer.activation === 'sigmoid') code += `        let e = exp(-abs(z));
        z = select(e / (1.0 + e), 1.0 / (1.0 + e), z >= 0.0);
`;
        code += `        x${next}[r / 4u][r % 4u] = z;
    }
`;
    });
    return `${code}    for (var k = 0u; k < ${network[network.length - 1].output}u; k++) {
        setAct(i, k, x${network.length}[k / 4u][k % 4u]);
    }
    setReason(i, 0u);
}`;
}