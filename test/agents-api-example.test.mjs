import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { resolve, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';
import { AgentSystem, MlpBrain, RuleBrain, defineAgents, mlpLayout, packMlpWeights, Sensors, buildAgentThinkWGSL, buildMlpThinkWGSL,
    agentSnapshotLayout, buildPointGrid, decodeAgentSnapshot, gridNearestNeighbours, senseAgent } from '../dist/agents.js';
import { herdDefinition } from '../tools/agents/scenario/herd-wolves.mjs';
import { standIn } from './stand-in-engine.mjs';   // runs an example's setup without a GPU

const root = fileURLToPath(new URL('../', import.meta.url));

test('API guide: every example type-checks, bundles and generates a valid think module', async () => {
    const guide = readFileSync(join(root, 'docs/api/agents.md'), 'utf8');
    const examples = [...guide.matchAll(/```ts\r?\n([\s\S]*?)```/g)].map(match => match[1]);
    assert.equal(examples.length, 7, 'validate every TypeScript example in the guide');
    const dir = mkdtempSync(join(root, 'test/.agents-api-'));
    try {
        const files = examples.map((source, n) => {
            writeFileSync(join(dir, `example${n}.ts`), source);
            return `example${n}.ts`;
        });
        writeFileSync(join(dir, 'tsconfig.json'), JSON.stringify({
            compilerOptions: { target: 'ES2022', module: 'ESNext', moduleResolution: 'Bundler',
                strict: true, noEmit: true, skipLibCheck: true, types: ['@webgpu/types'],
                lib: ['ES2022', 'DOM', 'DOM.Iterable'] },
            files,
        }));
        try {
            execFileSync(process.execPath, [resolve(root, 'node_modules/typescript/bin/tsc'),
                '--project', join(dir, 'tsconfig.json')], { cwd: root, stdio: 'pipe' });
        } catch (error) {
            throw new Error(`API example TypeScript failed:\n${error.stdout}\n${error.stderr}`);
        }
        // Use the actual declarations and brains from the guide, not second handwritten examples.
        const definitionOf = source => {
            const declaration = source.match(/const definition = defineAgents\(([\s\S]*?)\);/);
            assert.ok(declaration, 'every example declares its definition');
            return Function('defineAgents', 'Sensors', `return defineAgents(${declaration[1]});`)(defineAgents, Sensors);
        };
        let rules = 0, networks = 0, perSpecies = 0, byGenome = 0, trained = 0, switching = 0, lists = 0;
        for (const [n, source] of examples.entries()) {
            const result = await build({ stdin: { contents: source, loader: 'ts', resolveDir: root },
                bundle: true, platform: 'browser', format: 'esm', write: false, logLevel: 'silent' });
            assert.ok(result.outputFiles[0].contents.length > 0);
            const loads = /interface TrainedNetwork/.test(source), prepares = /prepareBrain\(/.test(source);
            // The neighbour-list example's definition holds WGSL, which definitionOf cannot cut out.
            const queries = /Sensors\.query\(/.test(source);
            const definition = loads || prepares ? herdDefinition() : queries ? undefined : definitionOf(source);
            const brain = source.match(/new RuleBrain\(`([\s\S]*?)`\)/);
            let wgsl;
            if (prepares) {
                // The switching example (step 9, phase 9.5): prepare a network and a rule brain on a
                // stand-in engine. Each compiles once, asynchronously; switching compiles nothing.
                switching++;
                const module = join(dir, `example${n}.mjs`);
                writeFileSync(module, result.outputFiles[0].text);
                const { preparedSwitch } = await import(pathToFileURL(module).href);
                const { engine, stages } = standIn(), agents = new AgentSystem(engine, definition);
                agents.setBrain(new RuleBrain('fn think(i: u32, species: u32, tick: u32) {}'));
                const network = new MlpBrain(engine, [{ input: definition.obsDim, output: definition.actDim, activation: 'tanh',
                    weights: new Float32Array(definition.obsDim * definition.actDim), biases: new Float32Array(definition.actDim) }]);
                const rules = new RuleBrain('fn think(i: u32, species: u32, tick: u32) { setReason(i, 9u); }');
                const before = stages.length;
                const { use } = await preparedSwitch(agents, 'wolf', [network, rules]);
                const prepared = stages.slice(before);
                assert.deepEqual(prepared.map(s => s.async === true), [true, true], 'two passes, each compiled asynchronously');
                const thinks = () => {
                    agents.advance(definition.dt);
                    return agents.getComputeStages().filter(s => s.name === 'AgentSystem think');
                };
                for (const [k, next] of [network, rules, network].entries()) {
                    use(next);
                    assert.equal(stages.length, before + 2, `switch ${k + 1} compiled`);
                    assert.ok(thinks().includes(prepared[next === network ? 0 : 1]), `switch ${k + 1} does not run the prepared pass`);
                }
                wgsl = prepared[0].shaderCode;
                agents.destroy();
                network.destroy();
            } else if (loads) {
                // The trained-network example: load the two committed files as it says, on a stand-in engine.
                trained++;
                const module = join(dir, `example${n}.mjs`);
                writeFileSync(module, result.outputFiles[0].text);
                const { useTrainedNetwork } = await import(pathToFileURL(module).href);
                const { engine, stages } = standIn(), agents = new AgentSystem(engine, definition, { thinkEvery: 3 });
                const files = ['prey', 'wolf'].map(s => JSON.parse(readFileSync(join(root, `tools/agents/models/herd-wolves/${s}.mlp.json`), 'utf8')));
                const brains = files.map(file => useTrainedNetwork(engine, agents, file));
                agents.advance(definition.dt);
                const thinks = agents.getComputeStages().filter(s => s.name === 'AgentSystem think');
                assert.equal(thinks.length, 2, 'one think pass per species');
                assert.deepEqual(brains.map(b => b.dims), [{ obs: 24, act: 2 }, { obs: 24, act: 2 }]);
                const built = stages.length, renamed = { ...files[0], obsNames: [...files[0].obsNames].reverse() };
                assert.throws(() => useTrainedNetwork(engine, agents, renamed), /prey\.mlp\.json was trained on other observations or actions\./);
                assert.throws(() => useTrainedNetwork(engine, agents, { ...files[0], actNames: ['turn'] }), /trained on other observations or actions/);
                assert.equal(stages.length, built, 'a refused file builds nothing');
                wgsl = thinks[0].shaderCode;
                agents.destroy();
                for (const b of brains) b.destroy();
            } else if (/MlpBrain\.byGenome\(/.test(source)) {
                // The per-agent network example (step 11, phase 11.4): 4 rows in one weight table,
                // agents spawned in 64-agent blocks, and single-row / multi-row updates without a compile.
                byGenome++;
                const module = join(dir, `example${n}.mjs`);
                writeFileSync(module, result.outputFiles[0].text);
                const { createByGenomeDemo, rowForGain } = await import(pathToFileURL(module).href);
                const { engine, stages } = standIn(), writes = [];
                engine.device.queue.writeBuffer = (buffer, offset, data, dataOffset = 0, size = data.length - dataOffset) => {
                    if (data instanceof Float32Array) writes.push({ buffer, offset, data: data.slice(dataOffset, dataOffset + size) });
                };
                const demo = createByGenomeDemo(engine);
                const floats = mlpLayout(rowForGain(1)).floats, weightsBuf = demo.brain.buffers[0];
                assert.equal(demo.brain.rowCount, 4);
                assert.equal(demo.brain.cooperative, true);
                const tableWrites = () => writes.filter(w => w.buffer === weightsBuf);
                assert.equal(tableWrites().length, 1, 'construction uploaded the 4-row table once');
                for (let r = 0; r < 4; r++) {
                    assert.deepEqual(tableWrites()[0].data.subarray(r * floats, (r + 1) * floats), packMlpWeights(rowForGain(r + 1)));
                }
                const words = new Uint32Array(demo.agents.records.data.buffer, demo.agents.records.data.byteOffset, demo.agents.capacity * 16);
                for (let i = 0; i < demo.agents.capacity; i++) {
                    assert.equal(words[i * 16 + 14], Math.floor(i / 64), `agent ${i} is in block ${Math.floor(i / 64)}`);
                }
                demo.agents.advance(demo.agents.definition.dt);
                const [thinkPass, ...extra] = demo.agents.getComputeStages().filter(s => s.name === 'AgentSystem think');
                assert.ok(thinkPass && !extra.length && thinkPass.shaderCode.includes('fn thinkBlock('), 'one cooperative think pass');
                const built = stages.length;
                demo.setRowGain(2, 5);
                assert.equal(stages.length, built, 'updating one row compiles nothing');
                assert.equal(tableWrites().length, 2);
                assert.equal(tableWrites()[1].offset, 2 * floats * 4);
                assert.deepEqual(tableWrites()[1].data, packMlpWeights(rowForGain(5)));
                demo.setPairGains(1, 6, 7);
                assert.equal(stages.length, built, 'updating a range of rows compiles nothing');
                assert.equal(tableWrites().length, 3);
                assert.equal(tableWrites()[2].offset, 1 * floats * 4);
                assert.deepEqual(tableWrites()[2].data.subarray(0, floats), packMlpWeights(rowForGain(6)));
                assert.deepEqual(tableWrites()[2].data.subarray(floats), packMlpWeights(rowForGain(7)));
                wgsl = thinkPass.shaderCode;
                demo.destroy();
                assert.equal(weightsBuf.destroyed, true);
            } else if (/setBrain\(\w+, \{ species: /.test(source)) {
                // The per-species example: run its setup on a stand-in engine and check what its
                // comments promise about passes and compiles. Its think modules are the stages it built.
                perSpecies++;
                const module = join(dir, `example${n}.mjs`);
                writeFileSync(module, result.outputFiles[0].text);
                const { createHerd } = await import(pathToFileURL(module).href);
                const { engine, stages } = standIn(), herd = createHerd(engine);
                const thinks = () => {
                    herd.agents.advance(herd.agents.definition.dt);
                    return herd.agents.getComputeStages().filter(s => s.name === 'AgentSystem think');
                };
                const [rulesPass, networkPass, ...more] = thinks();
                assert.ok(rulesPass && networkPass && !more.length, 'rules for every species, then the network for wolves: two passes');
                const built = stages.length;
                herd.wolvesUseRules();
                assert.equal(stages.length, built, 'moving wolves to the rules compiles nothing');
                assert.deepEqual(thinks(), [rulesPass], 'the network serves nobody and leaves the chain');
                herd.wolvesUseNetwork();
                assert.equal(stages.length, built + 1, 'giving wolves to the network again compiles its pass again');
                assert.equal(stages.at(-1).shaderCode, networkPass.shaderCode, 'with the same shader text');
                assert.equal(thinks().length, 2);
                herd.destroy();
                assert.equal(new WgslReflect(rulesPass.shaderCode).entry.compute[0].name, 'cs_main');
                wgsl = networkPass.shaderCode;
            } else if (queries) {
                // The neighbour-list example (step 12, phase 12.3), on a stand-in engine: its list adds
                // no observation; its sensor, step stage and brain name only declared constants; its
                // mirror repeats the grid's search through the definition it is given.
                lists++;
                const module = join(dir, `example${n}.mjs`);
                writeFileSync(module, result.outputFiles[0].text);
                const { createCrowd } = await import(pathToFileURL(module).href);
                const { engine, stages } = standIn(), extraBuffers = new Map();
                engine.setComputePassBindGroup = (stage, entries, group) => {
                    if (group === 1) extraBuffers.set(stage.name, entries.map(entry => entry.resource.buffer));
                };
                const crowd = createCrowd(engine), def = crowd.agents.definition;
                assert.equal(def.obsDim, 5, 'the list adds no observation floats');
                assert.deepEqual(def.obsNames, ['self.speed', 'self.energy', 'bounds.side', 'bounds.forward', 'crowd.0']);
                assert.equal(crowd.agents.neighbourStride, 8);
                const code = name => stages.find(s => s.name === `AgentSystem ${name}`).shaderCode;
                assert.ok(code('query contacts').includes('if (d2 > 1.44)') && code('query contacts').includes('result[i * 8u + n]'),
                    'one list of 8 within 1.2 m');
                assert.deepEqual(extraBuffers.get('AgentSystem step touching'), [crowd.touching], 'the step stage binds the app buffer');
                crowd.agents.advance(def.dt);
                const [think] = crowd.agents.getComputeStages().filter(s => s.name === 'AgentSystem think');
                for (const shader of [code('sense'), code('step touching'), think.shaderCode]) {
                    new WgslReflect(shader);
                    // The parser accepts a name nothing declares; a device would refuse the module.
                    for (const [name] of shader.matchAll(/\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b/g)) {
                        assert.ok(shader.includes(`const ${name} = `), `${name} is used but never declared`);
                    }
                }
                const rec = crowd.agents.records, crowdAt = def.constants.OBS_CROWD, listed = [];
                const grid = buildPointGrid(def, rec.data, 16, def.capacity, i => rec.isAlive(i), def.maxScanPerCell);
                for (let i = 0; i < def.capacity; i++) {
                    const filled = gridNearestNeighbours(def, rec, grid, i, def.nearestMasks[0], 1.2, 8, listed, 0);
                    assert.equal(senseAgent(def, rec, i)[crowdAt], filled / 8, `agent ${i}'s crowd`);
                }
                assert.deepEqual([0, 1, 33].map(i => senseAgent(def, rec, i)[crowdAt]), [0.25, 0.375, 0.5],
                    'a corner, an edge and the inside, as the spawn comment says');
                const layout = agentSnapshotLayout(def), bytes = new ArrayBuffer(layout.size), words = new Uint32Array(bytes);
                words.fill(0xffffffff, layout.neighbours / 4, layout.state / 4);
                words.set([1, 32], layout.neighbours / 4);
                assert.deepEqual(decodeAgentSnapshot(def, 0, bytes, layout).neighbours, [{ sensor: 'contacts', indices: [1, 32] }]);
                wgsl = think.shaderCode;
                crowd.destroy();
                assert.equal(crowd.touching.destroyed, true);
            } else if (brain) {
                rules++;
                wgsl = buildAgentThinkWGSL(definition, brain[1], 2);
            } else {
                // The network example: run its bundle and check the numbers its comment promises.
                assert.match(source, /new MlpBrain\(/, `example ${n} installs neither a RuleBrain nor an MlpBrain`);
                networks++;
                const module = join(dir, `example${n}.mjs`);
                writeFileSync(module, result.outputFiles[0].text);
                const { wallAvoider, sample } = await import(pathToFileURL(module).href);
                assert.equal(wallAvoider[0].input, definition.obsDim, 'the network reads every observation');
                assert.equal(wallAvoider[wallAvoider.length - 1].output, definition.actDim, 'and writes every action');
                assert.ok(Math.abs(sample[0] - Math.tanh(-1.5)) < 1e-6 && Math.abs(sample[0] - -0.905) < 5e-4, `turn ${sample[0]}`);
                assert.ok(Math.abs(sample[1] - Math.tanh(1)) < 1e-6 && Math.abs(sample[1] - 0.762) < 5e-4, `throttle ${sample[1]}`);
                wgsl = buildAgentThinkWGSL(definition, buildMlpThinkWGSL(wallAvoider), 1);
            }
            assert.equal(new WgslReflect(wgsl).entry.compute[0].name, 'cs_main');
        }
        assert.deepEqual([rules, networks, perSpecies, byGenome, trained, switching, lists], [1, 1, 1, 1, 1, 1, 1],
            'one rule, one network, one per-species, one byGenome, one trained-network, one switching and one neighbour-list example');
        console.log('API examples: TypeScript, browser bundles, the network example\'s numbers, the per-species example\'s passes, ' +
            'the byGenome example\'s blocks and row updates, the switching example\'s prepared passes, the trained-network example\'s loading, ' +
            'the neighbour-list example\'s list, constants and mirror, and generated WGSL passed.');
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});
