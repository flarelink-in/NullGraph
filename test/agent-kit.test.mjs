import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { RENDER_SCENES } from './gpu/suites.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const read = path => readFileSync(resolve(root, path), 'utf8').replace(/\r\n/g, '\n');
const docs = ['llms.txt', 'docs/agent/llms.txt', 'docs/agent/guide.md', 'docs/agent/validation.md'];

test('agent entry documents have valid local links', () => {
    let count = 0;
    for (const doc of docs) {
        for (const match of read(doc).matchAll(/\[[^\]]+\]\(([^)]+)\)/g)) {
            const target = match[1].split('#')[0];
            if (!target || /^https?:/.test(target)) continue;
            assert.ok(existsSync(resolve(root, dirname(doc), target)), `${doc}: missing ${target}`);
            count++;
        }
    }
    assert.ok(count >= 20, 'routing documents should actually link to the kit and references');
});

test('manifest and entry documents identify version 1.1.1 without shipping the kit', () => {
    const pkg = JSON.parse(read('package.json'));
    assert.equal(pkg.version, '1.1.1');
    assert.deepEqual(pkg.files, ['dist']);
    for (const doc of docs.slice(0, 3)) assert.ok(read(doc).includes(pkg.version), doc);
});

test('guide scenes are exact executable copies and registered for GPU smoke tests', async () => {
    const guide = read('docs/agent/guide.md');
    const matches = [...guide.matchAll(/<!-- scene: ([^ ]+) -->\n```js\n([\s\S]*?)\n```/g)];
    assert.equal(matches.length, 2);
    for (const [, path, snippet] of matches) {
        assert.equal(snippet.trim(), read(path).trim(), `example drift: ${path}`);
        assert.ok(RENDER_SCENES.some(scene => scene.file === path), `${path} is not registered`);
        const module = await import(new URL(`../${path}`, import.meta.url));
        assert.equal(typeof module.default, 'function', path);
    }
});

test('guide stays under the approximate 50k-token ceiling', () => {
    // Deliberately conservative character budget (~3 chars/token), not a tokenizer guarantee.
    assert.ok(read('docs/agent/guide.md').length <= 150_000);
});