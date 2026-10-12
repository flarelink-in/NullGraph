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

test('API guide: every example type-checks, bundles and generates a valid think module', async () => {
    const guide = readFileSync(join(root, 'docs/api/agents.md'), 'utf8');
    const examples = [...guide.matchAll(/```ts\r?\n([\s\S]*?)```/g)].map(match => match[1]);
    assert.equal(examples.length, 2, 'validate every TypeScript example in the guide');
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
        let rules = 0, networks = 0;
        for (const [n, source] of examples.entries()) {
            const result = await build({ stdin: { contents: source, loader: 'ts', resolveDir: root },
                bundle: true, platform: 'browser', format: 'esm', write: false, logLevel: 'silent' });
            assert.ok(result.outputFiles[0].contents.length > 0);
            const definition = definitionOf(source);
            const brain = source.match(/new RuleBrain\(`([\s\S]*?)`\)/);
            let wgsl;
            if (brain) {
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
        assert.deepEqual([rules, networks], [1, 1], 'one rule example and one network example');
        console.log('API examples: TypeScript, browser bundles, the network example\'s numbers and generated WGSL passed.');
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});
