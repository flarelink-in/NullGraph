import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { gzipSync } from 'node:zlib';
import { build } from 'esbuild';
import { glb, box } from './glb-fixtures/build.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));

test('rigid bake size lines and tree-shaking: parser-only apps carry no bake', async () => {
    const entry = join(root, 'src/extras/loaders/index.ts').replaceAll('\\', '/');
    const bundle = async contents => (await build({ stdin: { contents, loader: 'ts', resolveDir: root },
        bundle: true, minify: true, format: 'esm', platform: 'browser', external: ['gl-matrix'],
        write: false, logLevel: 'silent' })).outputFiles[0];
    const parser = await bundle(`export { GLBParser } from '${entry}';`);
    const bake = await bundle(`export { bakeRigidAnimation } from '${entry}';`);
    const all = await bundle(`export * from '${entry}';`);
    for (const [name, output, ceiling] of [['parser', parser, 3300], ['bake', bake, 3400], ['entry', all, 6700]]) {
        const bytes = gzipSync(output.contents).length;
        assert.ok(bytes <= ceiling, `${name}: ${bytes} B min+gzip exceeds ${ceiling} B`);
        console.log(`loaders ${name}: ${output.contents.length} B min, ${bytes} B min+gzip (line ${ceiling} B)`);
    }
    assert.ok(!parser.text.includes('bakeRigidAnimation:'), 'parser-only apps carry the bake');
    assert.ok(bake.text.includes('bakeRigidAnimation:') && !bake.text.includes('Invalid GLB.'), 'the bake imports no parser');
    assert.equal((await bundle(`import '${entry}';`)).text.trim(), '', 'unused entry leaves no code');
});

test('loaders guide: rigid preparation example type-checks, bundles and runs on an in-memory GLB', async () => {
    const guide = readFileSync(join(root, 'docs/api/loaders.md'), 'utf8');
    const examples = [...guide.matchAll(/```ts\r?\n([\s\S]*?)```/g)].map(m => m[1]);
    const sources = examples.filter(s => s.includes('export async function prepareRigidModel'));
    assert.equal(sources.length, 1, 'check the actual rigid preparation example');
    const dir = mkdtempSync(join(root, 'test/.loaders-api-'));
    try {
        writeFileSync(join(dir, 'example.ts'), sources[0]);
        writeFileSync(join(dir, 'tsconfig.json'), JSON.stringify({ compilerOptions: {
            target: 'ES2022', module: 'ESNext', moduleResolution: 'Bundler', strict: true,
            noEmit: true, skipLibCheck: true, types: ['@webgpu/types'], lib: ['ES2022', 'DOM', 'DOM.Iterable'],
        }, files: ['example.ts'] }));
        try {
            execFileSync(process.execPath, [join(root, 'node_modules/typescript/bin/tsc'), '--project', join(dir, 'tsconfig.json')], { cwd: root, stdio: 'pipe' });
        } catch (error) { throw new Error(`Loaders example TypeScript failed:\n${error.stdout}\n${error.stderr}`); }
        const output = await build({ stdin: { contents: sources[0], loader: 'ts', resolveDir: root },
            bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent' });
        const module = join(dir, 'example.mjs');
        writeFileSync(module, output.outputFiles[0].text);
        const { prepareRigidModel } = await import(pathToFileURL(module).href);
        const bytes = glb({ nodes: [{ mesh: 0 }], meshes: [{ primitives: [box()] }],
            animations: ['idle', 'walk', 'run', 'eat'].map(name => ({ name,
                samplers: [{ times: [0, 1], values: [0, 0, 0, 2, 0, 0] }],
                channels: [{ sampler: 0, node: 0, path: 'translation' }] })) });
        const result = await prepareRigidModel(`data:model/gltf-binary;base64,${Buffer.from(bytes).toString('base64')}`);
        assert.equal(result.animation.vertexCount, 8);
        assert.deepEqual(result.animation.clips.map(c => [c.name, c.frames, c.loop]),
            [['idle', 30, true], ['walk', 30, true], ['run', 30, true], ['eat', 31, false]]);
        assert.deepEqual(result.partOfVertex, result.animation.partOfVertex);
        assert.deepEqual([...result.geometry.megaIndices], [...result.animation.indices]);
        assert.equal(result.bounds[3], result.animation.bounds.radius);
        assert.ok(result.bounds[3] > 2.5);
    } finally { rmSync(dir, { recursive: true, force: true }); }
});