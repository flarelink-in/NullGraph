// Acceptance tests for TransientPool, taken directly from docs/transient-pool.md §10.
// Runs in plain Node against the built bundle with a mocked GPUDevice:
//   npm run build:lib && node test/post-pool.test.mjs
import { TransientPool } from '../dist/post.js';

// ---- mock device (the pool only touches these entry points) ----
let textureCounter = 0;
function mockDevice() {
    return {
        createTexture: () => {
            const view = { id: textureCounter++ };
            return { createView: () => view, destroy() {} };
        },
        createCommandEncoder: () => ({
            beginRenderPass: () => ({ end() {} }),
            finish: () => ({}),
        }),
        queue: { submit() {} },
    };
}

let failures = 0;
function check(name, fn) {
    try {
        fn();
        console.log(`  ok    ${name}`);
    } catch (e) {
        failures++;
        console.log(`  FAIL  ${name}\n        ${e.message}`);
    }
}
function assertEq(actual, expected, what) {
    if (actual !== expected) throw new Error(`${what}: expected ${expected}, got ${actual}`);
}
function assertNear(actual, expected, tolerance, what) {
    if (Math.abs(actual - expected) > tolerance) throw new Error(`${what}: expected ~${expected}, got ${actual}`);
}
function assertThrows(fn, what) {
    try { fn(); } catch { return; }
    throw new Error(`${what}: expected a throw`);
}

const MB = 1024 * 1024;
function pool() { return new TransientPool(mockDevice(), 1920, 1080, 'rgba16float'); }

// T1 — basic disjoint reuse
check('T1: disjoint full-res targets share one physical texture', () => {
    const p = pool();
    const A = p.alloc({ scale: 1 }, 'A');
    const B = p.alloc({ scale: 1 }, 'B');
    p.markWrite(A, 0); p.markRead(A, 3);
    p.markWrite(B, 5); p.markRead(B, 6);
    p.resolve();
    const r = p.report();
    assertEq(r.physicalTextures, 1, 'physicalTextures');
    assertEq(r.logicalTargets, 2, 'logicalTargets');
});

// T2 — adjacency must NOT alias (catches the <= bug)
check('T2: adjacent intervals stay separate (strict < predicate)', () => {
    const p = pool();
    const A = p.alloc({ scale: 1 }, 'A');
    const B = p.alloc({ scale: 1 }, 'B');
    p.markWrite(A, 0); p.markRead(A, 3);
    p.markWrite(B, 3); p.markRead(B, 6);
    p.resolve();
    assertEq(p.report().physicalTextures, 2, 'physicalTextures');
});

// T3 — descriptor mismatch blocks reuse
check('T3: format mismatch blocks reuse', () => {
    const p = pool();
    const A = p.alloc({ scale: 1, format: 'rgba16float' }, 'A');
    const B = p.alloc({ scale: 1, format: 'rgba8unorm' }, 'B');
    p.markWrite(A, 0); p.markRead(A, 1);
    p.markWrite(B, 3); p.markRead(B, 4);
    p.resolve();
    assertEq(p.report().physicalTextures, 2, 'physicalTextures');
});
check('T3b: extraUsage mismatch blocks reuse', () => {
    const p = pool();
    const A = p.alloc({ scale: 1 }, 'A');
    const B = p.alloc({ scale: 1, extraUsage: 0x08 /* STORAGE_BINDING */ }, 'B');
    p.markWrite(A, 0); p.markRead(A, 1);
    p.markWrite(B, 3); p.markRead(B, 4);
    p.resolve();
    assertEq(p.report().physicalTextures, 2, 'physicalTextures');
});

// T4 — history exclusion
check('T4: history targets are double-buffered and never aliased', () => {
    const p = pool();
    const A = p.alloc({ scale: 1, history: 2 }, 'A');
    const B = p.alloc({ scale: 1, history: 1 }, 'B');
    p.markWrite(A, 0); p.markRead(A, 1);
    p.markWrite(B, 3); p.markRead(B, 4);
    p.resolve();
    assertEq(p.report().physicalTextures, 3, 'physicalTextures');
    assertThrows(() => p.historyView(B), 'historyView on history:1');
    const before = p.historyView(A);
    p.advanceFrame();
    if (p.view(A) !== before) throw new Error('advanceFrame did not swap the double buffer');
});

// T4b — the AutoExposure adapt pattern: one pass reads its own history target
// (previous frame's side) while writing the current side. Read and write are
// marked at the SAME pass index, which must satisfy V1, and view()/historyView()
// must be distinct textures so the pass never samples its own attachment.
check('T4b: same-pass read+write of a history target resolves', () => {
    const p = pool();
    const S = p.alloc({ scale: 1e-6, format: 'rg16float', history: 2 }, 'aeState');
    p.markRead(S, 2); p.markWrite(S, 2);
    p.markRead(S, 3);
    p.resolve();
    const info = p.info(S);
    assertEq(info.width, 1, 'width floors to 1');
    assertEq(info.height, 1, 'height floors to 1');
    assertEq(info.history, 2, 'info exposes history');
    if (p.view(S) === p.historyView(S)) throw new Error('view and historyView are the same texture');
});

// T5 — read before write
check('T5: read-before-write throws V1 naming the target', () => {
    const p = pool();
    const A = p.alloc({ scale: 1 }, 'A');
    p.markRead(A, 2);
    p.markWrite(A, 4);
    try {
        p.resolve();
    } catch (e) {
        if (!e.message.includes('"A"') || !e.message.includes('before it is written')) {
            throw new Error(`wrong error: ${e.message}`);
        }
        return;
    }
    throw new Error('expected resolve() to throw');
});

// T6 — SSAO + Bloom(4 mips) + Tonemap, the v1 integration case
check('T6: v1 chain aliases exactly sceneColor <-> chain_2', () => {
    const p = pool();
    const sceneColor = p.alloc({ scale: 1 }, 'sceneColor');
    const sceneDepth = p.alloc({ scale: 1, format: 'depth24plus' }, 'sceneDepth');
    const AO_raw = p.alloc({ scale: 0.5, format: 'r8unorm' }, 'AO_raw');
    const AO = p.alloc({ scale: 0.5, format: 'r8unorm' }, 'AO');
    const chain_1 = p.alloc({ scale: 1 }, 'chain_1');
    const B0 = p.alloc({ scale: 0.5 }, 'B0');
    const B1 = p.alloc({ scale: 0.25 }, 'B1');
    const B2 = p.alloc({ scale: 0.125 }, 'B2');
    const B3 = p.alloc({ scale: 0.0625 }, 'B3');
    const U2 = p.alloc({ scale: 0.125 }, 'U2');
    const U1 = p.alloc({ scale: 0.25 }, 'U1');
    const U0 = p.alloc({ scale: 0.5 }, 'U0');
    const chain_2 = p.alloc({ scale: 1 }, 'chain_2');

    p.markWrite(sceneColor, 0); p.markWrite(sceneDepth, 0);          // 0 Scene
    p.markRead(sceneDepth, 1); p.markWrite(AO_raw, 1);               // 1 SSAO occlude
    p.markRead(AO_raw, 2); p.markWrite(AO, 2);                       // 2 SSAO blur
    p.markRead(sceneColor, 3); p.markRead(AO, 3); p.markWrite(chain_1, 3); // 3 SSAO apply
    p.markRead(chain_1, 4); p.markWrite(B0, 4);                      // 4 prefilter
    p.markRead(B0, 5); p.markWrite(B1, 5);                           // 5 down
    p.markRead(B1, 6); p.markWrite(B2, 6);                           // 6 down
    p.markRead(B2, 7); p.markWrite(B3, 7);                           // 7 down
    p.markRead(B3, 8); p.markRead(B2, 8); p.markWrite(U2, 8);        // 8 up
    p.markRead(U2, 9); p.markRead(B1, 9); p.markWrite(U1, 9);        // 9 up
    p.markRead(U1, 10); p.markRead(B0, 10); p.markWrite(U0, 10);     // 10 up
    p.markRead(chain_1, 11); p.markRead(U0, 11); p.markWrite(chain_2, 11); // 11 composite
    p.markRead(chain_2, 12);                                         // 12 tonemap -> screen
    p.resolve();

    const r = p.report();
    assertEq(r.logicalTargets, 13, 'logicalTargets');
    assertEq(r.physicalTextures, 12, 'physicalTextures');
    assertNear(r.bytesWithoutAliasing / MB, 66.8, 0.5, 'bytesWithoutAliasing (MiB)');
    assertNear((r.bytesWithoutAliasing - r.bytes) / MB, 1920 * 1080 * 8 / MB, 0.01, 'aliasing saves one full-res target');
    // Assert the alias pair explicitly, not just the count.
    if (p.view(sceneColor) !== p.view(chain_2)) throw new Error('sceneColor and chain_2 must share a physical texture');
    if (p.view(sceneColor) === p.view(chain_1)) throw new Error('sceneColor and chain_1 must NOT share');
});

// T7 — ping-pong emergence
check('T7: a tail of full-res passes resolves to 2 physical textures', () => {
    const p = pool();
    const names = ['sceneColor', 'chain_1', 'chain_2', 'chain_3', 'chain_4'];
    const spans = [[0, 3], [3, 11], [11, 12], [12, 13], [13, 14]];
    const handles = names.map(n => p.alloc({ scale: 1 }, n));
    handles.forEach((h, i) => {
        p.markWrite(h, spans[i][0]);
        p.markRead(h, spans[i][1]);
    });
    p.resolve();
    assertEq(p.report().physicalTextures, 2, 'physicalTextures');
});

// T8 — resize invalidation
check('T8: resize returns to declare phase and invalidates handles', () => {
    const p = pool();
    const A = p.alloc({ scale: 1 }, 'A');
    p.markWrite(A, 0); p.markRead(A, 1);
    p.resolve();
    p.resize(1280, 720);
    assertThrows(() => p.view(A), 'view() on a pre-resize handle');
    assertThrows(() => p.report(), 'report() before the next resolve');
    const B = p.alloc({ scale: 1 }, 'B'); // declare phase again
    p.markWrite(B, 0);
    p.resolve();
    p.view(B);
});

// V3 — the independent confirmation of the predicate
check('V3: a pass sampling and rendering the same physical texture throws', () => {
    const p = pool();
    const A = p.alloc({ scale: 1 }, 'A');
    p.markWrite(A, 0);
    p.markRead(A, 1, { passName: 'bad' });
    p.markWrite(A, 1, { passName: 'bad' });
    assertThrows(() => p.resolve(), 'V3');
});

console.log(failures === 0 ? '\nAll pool acceptance tests passed.' : `\n${failures} test(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
