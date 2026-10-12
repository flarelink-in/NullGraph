import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';
import { defineAgents, Sensors, buildAgentThinkWGSL } from '../dist/agents.js';

const root = fileURLToPath(new URL('../', import.meta.url));

test('API guide: minimal example type-checks, bundles and generates a valid think module', async () => {
    const guide = readFileSync(join(root, 'docs/api/agents.md'), 'utf8');
    const examples = [...guide.matchAll(/```ts\r?\n([\s\S]*?)```/g)].map(match => match[1]);
    assert.equal(examples.length, 1, 'validate every TypeScript example in the guide');
    const source = examples[0];
    const dir = mkdtempSync(join(root, 'test/.agents-api-'));
    try {
        writeFileSync(join(dir, 'example.ts'), source);
        writeFileSync(join(dir, 'tsconfig.json'), JSON.stringify({
            compilerOptions: { target: 'ES2022', module: 'ESNext', moduleResolution: 'Bundler',
                strict: true, noEmit: true, skipLibCheck: true, types: ['@webgpu/types'],
                lib: ['ES2022', 'DOM', 'DOM.Iterable'] },
            files: ['example.ts'],
        }));
        try {
            execFileSync(process.execPath, [resolve(root, 'node_modules/typescript/bin/tsc'),
                '--project', join(dir, 'tsconfig.json')], { cwd: root, stdio: 'pipe' });
        } catch (error) {
            throw new Error(`API example TypeScript failed:\n${error.stdout}\n${error.stderr}`);
        }
        const result = await build({ stdin: { contents: source, loader: 'ts', resolveDir: root },
            bundle: true, platform: 'browser', format: 'esm', write: false, logLevel: 'silent' });
        assert.ok(result.outputFiles[0].contents.length > 0);
        // Use the actual declaration and brain from the guide, not a second handwritten example.
        const declaration = source.match(/const definition = defineAgents\(([\s\S]*?)\);/);
        const brain = source.match(/new RuleBrain\(`([\s\S]*?)`\)/);
        assert.ok(declaration && brain, 'the example must include a definition and rule brain');
        const definition = Function('defineAgents', 'Sensors', `return defineAgents(${declaration[1]});`)(defineAgents, Sensors);
        const module = new WgslReflect(buildAgentThinkWGSL(definition, brain[1], 2));
        assert.equal(module.entry.compute[0].name, 'cs_main');
        console.log('API example: TypeScript, browser bundle and generated brain WGSL passed.');
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});