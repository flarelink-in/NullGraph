// Step 7, part 2 (docs/agents-cloning-design.md §4.2, §9): a trained network's file, read back.
//
// train.mjs writes <species>.mlp.json. Its `layers` are an MlpNetwork with plain arrays; its
// obsNames and actNames say what each input and output means. Equal widths do not show that a
// feature means what the network expects; equal names do, so they are checked here.
//
// Plain JS, no node: imports: an app loads a network the same way.

export const MODEL_FORMAT = 'null-graph/mlp';
export const MODEL_VERSION = 1;

const quoted = name => (name === undefined ? 'nothing' : `"${name}"`);

/**
 * The layers of a parsed <species>.mlp.json as an MlpNetwork, for `new MlpBrain(engine, network)`.
 * Refuses a file whose obsNames or actNames are not `definition`'s, in order. `file` names it in errors.
 */
export function networkFromModel(model, definition, file) {
    if (model?.format !== MODEL_FORMAT || model.version !== MODEL_VERSION) {
        throw new Error(`model: ${file} is "${model?.format}" version ${model?.version}; this reader takes "${MODEL_FORMAT}" version ${MODEL_VERSION}.`);
    }
    for (const key of ['obsNames', 'actNames']) {
        const mine = model[key] ?? [], theirs = definition[key];
        for (let k = 0; k < Math.max(mine.length, theirs.length); k++) {
            if (mine[k] !== theirs[k]) throw new Error(`model: ${file} ${key}[${k}] is ${quoted(mine[k])}; the definition has ${quoted(theirs[k])}.`);
        }
    }
    return model.layers.map(l => ({ input: l.input, output: l.output, activation: l.activation,
        weights: Float32Array.from(l.weights), biases: Float32Array.from(l.biases) }));
}
