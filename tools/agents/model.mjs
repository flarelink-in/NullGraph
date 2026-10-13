// Step 7, part 2 (docs/agents-cloning-design.md §4.2, §9): a trained network's file.
//
// <species>.mlp.json holds one network. Its `layers` are an MlpNetwork with plain arrays; its
// obsNames and actNames say what each input and output means. Equal widths do not show that a
// feature means what the network expects; equal names do, so they are checked on reading.
//
// Plain JS, no node: imports: an app loads a network the same way.

export const MODEL_FORMAT = 'null-graph/mlp';
export const MODEL_VERSION = 1;

const quoted = name => (name === undefined ? 'nothing' : `"${name}"`);
/** An f32 as a JSON number that reads back to the same f32: 9 significant digits. */
const nine = value => (value === 0 ? 0 : Number(value.toPrecision(9)));
const words = network => network.flatMap(l => [...new Uint32Array(l.weights.buffer, l.weights.byteOffset, l.weights.length),
    ...new Uint32Array(l.biases.buffer, l.biases.byteOffset, l.biases.length)]);
const layersOf = model => model.layers.map(l => ({ input: l.input, output: l.output, activation: l.activation,
    weights: Float32Array.from(l.weights), biases: Float32Array.from(l.biases) }));

/**
 * The text of a <species>.mlp.json. `network` is an MlpNetwork; `rest` (trainedOn, training,
 * measured) is stored as given, after the layers. The same input gives the same bytes.
 * Refuses a network that would not read back bit for bit: JSON has no -0 and no non-finite number.
 */
export function modelText({ species, obsNames, actNames, network, ...rest }) {
    const text = JSON.stringify({
        format: MODEL_FORMAT, version: MODEL_VERSION, species, obsNames, actNames,
        layers: network.map(l => ({ input: l.input, output: l.output, activation: l.activation,
            weights: Array.from(l.weights, nine), biases: Array.from(l.biases, nine) })),
        ...rest,
    }, null, 1) + '\n';
    const before = words(network), after = words(layersOf(JSON.parse(text)));
    if (before.length !== after.length || before.some((w, k) => w !== after[k])) {
        throw new Error(`model: the network for "${species}" does not read back bit for bit; it holds a -0 or a value that is not finite.`);
    }
    return text;
}

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
    return layersOf(model);
}
