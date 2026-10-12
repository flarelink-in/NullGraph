import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { resolve, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';
import { defineAgents, Sensors, buildAgentThinkWGSL, buildMlpThinkWGSL } from '../dist/agents.js';

const root = fileURLToPath(new URL('../', import.meta.url));

globalThis.GPUBufferUsage ??= { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64 };
// Runs an example's setup without a GPU: buffers and uploads are inert; built stages are recorded.
function standIn() {
    const stages = [];
    const engine = {
        device: { limits: { maxStorageBufferBindingSize: 128 * 1024 * 1024, maxBufferSize: 256 * 1024 * 1024,
            maxComputeWorkgroupsPerDimension: 65535, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256,
            maxStorageBuffersPerShaderStage: 8 },
        createBuffer(o) { return { ...o, destroy() {} }; },
        queue: { writeBuffer() {} } },
        createComputeStage(o) { const s = { ...o, enabled: true, preClearBuffers: [] }; stages.push(s); return s; },
        setComputePassBindGroup() {},
        render() {},
    };
    return { engine, stages };
}

test('API guide: every example type-checks, bundles and generates a valid think module', async () => {
    const guide = readFileSync(join(root, 'docs/api/agents.md'), 'utf8');
    const examples = [...guide.matchAll(/```ts\r?\n([\s\S]*?)```/g)].map(match => match[1]);
    assert.equal(examples.length, 3, 'validate every TypeScript example in the guide');
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
        let rules = 0, networks = 0, perSpecies = 0;
        for (const [n, source] of examples.entries()) {
            const result = await build({ stdin: { contents: source, loader: 'ts', resolveDir: root },
                bundle: true, platform: 'browser', format: 'esm', write: false, logLevel: 'silent' });
            assert.ok(result.outputFiles[0].contents.length > 0);
            const definition = definitionOf(source);
            const brain = source.match(/new RuleBrain\(`([\s\S]*?)`\)/);
            let wgsl;
            if (/setBrain\(\w+, \{ species: /.test(source)) {
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
        assert.deepEqual([rules, networks, perSpecies], [1, 1, 1], 'one rule example, one network example, one per-species example');
        console.log('API examples: TypeScript, browser bundles, the network example\'s numbers, the per-species example\'s passes ' +
            'and generated WGSL passed.');
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});
