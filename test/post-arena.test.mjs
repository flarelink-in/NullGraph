// Unit tests for ParamArena / layoutFields (docs/post-processing-design.md §6,
// §13 step 1): vec3f alignment, 256-byte slice offsets, and the effect
// accessor path (including u32 fields, which must be written as integer bits).
//   npm run build:lib && node test/post-arena.test.mjs
import { layoutFields, ParamArena, Bloom, SSAO } from '../dist/post.js';

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
    const a = JSON.stringify(actual);
    const b = JSON.stringify(expected);
    if (a !== b) throw new Error(`${what}: expected ${b}, got ${a}`);
}

const mockDevice = { createBuffer: () => ({ destroy() {} }) };

check('layoutFields: vec3f aligns to 16, not 12', () => {
    const l = layoutFields([
        { name: 'a', type: 'f32', default: 0 },
        { name: 'b', type: 'vec3f', default: [0, 0, 0] },
        { name: 'c', type: 'f32', default: 0 },
    ]);
    assertEq(l.byteOffsets, [0, 16, 28], 'byteOffsets');
    assertEq(l.sizeBytes, 32, 'sizeBytes');
    if (!l.wgsl.includes('_pad0: f32')) throw new Error('expected explicit padding members');
});

check('layoutFields: vec2f then vec3f', () => {
    const l = layoutFields([
        { name: 'a', type: 'vec2f', default: [0, 0] },
        { name: 'b', type: 'vec3f', default: [0, 0, 0] },
    ]);
    assertEq(l.byteOffsets, [0, 16], 'byteOffsets');
    assertEq(l.sizeBytes, 32, 'sizeBytes');
});

check('layoutFields: empty schema emits a non-empty struct', () => {
    const l = layoutFields([]);
    assertEq(l.byteOffsets, [], 'byteOffsets');
    assertEq(l.sizeBytes, 16, 'sizeBytes');
    assertEq(l.touchField, '_reserved', 'touchField');
});

check('ParamArena: 256-byte slices, one per effect', () => {
    const arena = new ParamArena(mockDevice, [
        { name: 'SSAO', schema: [{ name: 'radius', type: 'f32', default: 0.6 }] },
        { name: 'Bloom', schema: [{ name: 'threshold', type: 'f32', default: 1.15 }] },
        { name: 'Tonemap', schema: [{ name: 'exposure', type: 'f32', default: 1.0 }] },
    ]);
    assertEq(arena.floatCount, 192, 'floatCount');
    const layout = arena.describeLayout();
    assertEq(layout.SSAO.offsetFloats, 0, 'SSAO offset');
    assertEq(layout.Bloom.offsetFloats, 64, 'Bloom offset');
    assertEq(layout.Tonemap.offsetFloats, 128, 'Tonemap offset');
    assertEq(layout.Tonemap.fields, ['exposure'], 'Tonemap fields');
});

check('ParamArena: adopt() rejects a wrong-length array with the doc message', () => {
    const arena = new ParamArena(mockDevice, [{ name: 'X', schema: [] }]);
    try {
        arena.adopt(new Float32Array(3));
    } catch (e) {
        assertEq(e.message, 'PostChain: setRawParams expected 64 floats, received 3.', 'message');
        return;
    }
    throw new Error('expected a throw');
});

check('Effect accessors: constructor values flush on first bind only', () => {
    const bloom = new Bloom({ threshold: 2.0 });
    if (bloom.threshold !== 2.0) throw new Error('pre-bind read failed');
    const arena = new ParamArena(mockDevice, [{ name: 'Bloom', schema: bloom.declare(fakeCtx()).params }]);
    bloom.bind(arena.slotFor(0));
    const slot = arena.slotFor(0);
    assertEq(slot[0], 2.0, 'threshold@0');
    assertEq(slot[1], 0.5, 'knee@1 default');
    bloom.intensity = 0.25;
    assertEq(slot[2], 0.25, 'intensity@2 live write');
    // Re-bind (setRawParams path) must NOT re-flush constructor values.
    const worker = new Float32Array(arena.floatCount);
    worker[0] = 9.0;
    arena.adopt(worker);
    bloom.bind(arena.slotFor(0));
    assertEq(bloom.threshold, 9.0, 'threshold reads adopted data');
});

check('Effect accessors: u32 params are written as integer bits', () => {
    const ssao = new SSAO({ samples: 24 });
    const arena = new ParamArena(mockDevice, [{ name: 'SSAO', schema: ssao.declare(fakeCtx()).params }]);
    ssao.bind(arena.slotFor(0));
    const slot = arena.slotFor(0);
    const u32 = new Uint32Array(slot.buffer, slot.byteOffset, slot.length);
    // schema: radius@0, bias@1, intensity@2, samples@3
    assertEq(u32[3], 24, 'samples stored as u32 bits');
    assertEq(ssao.samples, 24, 'samples reads back as an integer');
});

function fakeCtx() {
    let id = 0;
    return {
        target: (_spec, debugName) => ({ id: id++, debugName }),
        pass: () => {},
        has: () => true,
        width: 1920,
        height: 1080,
    };
}

console.log(failures === 0 ? '\nAll arena tests passed.' : `\n${failures} test(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
