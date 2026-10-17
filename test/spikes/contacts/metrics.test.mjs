import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

// Exercise the spike's actual pure measurement function without starting a GPU harness.
const source = readFileSync(new URL('./run.js', import.meta.url), 'utf8');
const body = source.slice(source.indexOf('function measureOverlap('), source.indexOf('async function overlapHistory('));
const measure = new Function(`${body}; return measureOverlap;`)();
test('S8: X-sorted exact overlap and nearest-prey sweep matches all-pairs oracle', () => {
    const count = 128, f = new Float32Array(count * 16), u = new Uint32Array(f.buffer);
    for (let i = 0; i < count; i++) {
        f[i * 16] = Math.sin(i * 19.3) * 4;
        f[i * 16 + 2] = Math.cos(i * 7.1) * 4;
        f[i * 16 + 3] = i < 16 ? 0.6 : 0.4;
    }
    let worstM = 0, worstPct = 0, over = 0, sum = 0;
    for (let i = 0; i < count; i++) {
        let nearest = Infinity, overlap = false;
        for (let j = 0; j < count; j++) {
            if (i === j) continue;
            const d = Math.hypot(f[i * 16] - f[j * 16], f[i * 16 + 2] - f[j * 16 + 2]);
            const r = f[i * 16 + 3] + f[j * 16 + 3];
            const penetration = Math.max(0, r - d);
            worstM = Math.max(worstM, penetration);
            worstPct = Math.max(worstPct, 100 * penetration / r);
            overlap ||= penetration / r > 0.05;
            if (j >= 16) nearest = Math.min(nearest, d);
        }
        if (overlap) over++;
        if (i >= 16) sum += nearest;
    }
    const actual = measure(u, { wolves: 16 });
    const expected = { worstM, worstPct, over5Pct: 100 * over / count, nearestPreyM: sum / (count - 16) };
    for (const key of Object.keys(expected)) assert.ok(Math.abs(actual[key] - expected[key]) < 1e-12, key);
});