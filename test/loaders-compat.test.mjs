// Optional external-corpus regression: no model files or old implementation enter the repository.
// NG_GLB_COMPAT_DIR: directory searched recursively; NG_GLB_OLD_LOADER: previous ESM loaders build.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { GLBParser } from '../dist/loaders.js';

const directory = process.env.NG_GLB_COMPAT_DIR, oldLoader = process.env.NG_GLB_OLD_LOADER;
test('GLBParser preserves previous output across an external GLB corpus', {
    skip: directory && oldLoader ? false : 'set NG_GLB_COMPAT_DIR and NG_GLB_OLD_LOADER to compare an external corpus',
}, async () => {
    const { GLBParser: Previous } = await import(pathToFileURL(oldLoader).href);
    const filesIn = dir => readdirSync(dir, { withFileTypes: true }).flatMap(e =>
        e.isDirectory() ? filesIn(join(dir, e.name)) : e.name.endsWith('.glb') ? [join(dir, e.name)] : []);
    const files = filesIn(directory);
    assert.ok(files.length > 0, 'the compatibility corpus is not empty');
    const previousShape = data => ({ ...Object.fromEntries(Object.entries(data).filter(([k]) => k !== 'nodes')),
        meshes: data.meshes?.map(mesh => ({
            ...Object.fromEntries(Object.entries(mesh).filter(([k]) => k !== 'node' && k !== 'getWebGPULayout')),
            // Function identities differ between builds; compare the descriptors they produce.
            ...(mesh.getWebGPULayout ? { layout: mesh.getWebGPULayout() } : {}),
        })) ?? data.meshes });
    let urls = 0;
    const original = URL.createObjectURL;
    // Embedded image URLs are opaque, fresh on every parse; normalize them by creation order.
    URL.createObjectURL = () => `blob:compat-${urls++}`;
    try {
        for (const file of files) {
            const bytes = readFileSync(file);
            const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
            urls = 0;
            const old = Previous.parse(buffer.slice(0));
            urls = 0;
            const current = GLBParser.parse(buffer.slice(0));
            assert.deepEqual(previousShape(current), previousShape(old), file);
            for (const mesh of current.meshes ?? []) assert.ok(current.nodes?.[mesh.node], `${file}: mesh node missing`);
        }
    } finally { URL.createObjectURL = original; }
    console.log(`GLBParser compatibility: ${files.length} GLBs unchanged apart from node metadata`);
});