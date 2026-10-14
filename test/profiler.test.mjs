// Headless GPUProfiler tests (null-graph/profiler), on a mock device whose read the test settles.
//
// The questions: what end() reports, and what happens when its read fails. A read fails when the
// device is destroyed while it is pending, as a worker that remakes its engine on another device
// does (step 10's device switch). The real-device half is C8 in test/core-gpu/run.js.

globalThis.GPUBufferUsage = globalThis.GPUBufferUsage ?? {
    MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32,
    UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512,
};
globalThis.GPUMapMode = globalThis.GPUMapMode ?? { READ: 1, WRITE: 2 };

const { GPUProfiler } = await import('../dist/profiler.js');

let failures = 0;
async function check(name, fn) {
    try { await fn(); console.log(`  ok    ${name}`); }
    catch (e) { failures++; console.log(`  FAIL  ${name}\n        ${e.message}`); }
}
function assert(v, m) { if (!v) throw new Error(m); }

// Every unhandled rejection, kept for the check that is running.
const unhandled = [];
process.on('unhandledRejection', (reason) => unhandled.push(reason));
const settle = () => new Promise((done) => setTimeout(done, 10));

/** A device whose readback buffer's mapAsync returns a promise the test settles: `reads`. */
function mockDevice() {
    const reads = [];
    const device = {
        times: new BigInt64Array(2),
        createQuerySet: () => ({}),
        createBuffer: (d) => ({
            ...d,
            mapAsync: () => new Promise((resolve, reject) => reads.push({ resolve, reject })),
            getMappedRange: () => device.times.slice().buffer,
            unmap() {},
        }),
        createCommandEncoder: () => ({
            beginComputePass: () => ({ end() {} }),
            resolveQuerySet() {}, copyBufferToBuffer() {},
            finish: () => ({}),
        }),
        queue: { submit() {} },
    };
    return { device, reads };
}

await check('end() reports the time between its two timestamps, in milliseconds', async () => {
    const { device, reads } = mockDevice();
    const profiler = new GPUProfiler(device);
    const got = [];
    profiler.begin();
    profiler.end((ms) => got.push(ms));
    device.times = BigInt64Array.of(1_000_000n, 3_500_000n);
    reads[0].resolve();
    await settle();
    assert(got.length === 1 && got[0] === 2.5, `onUpdate got [${got}], expected [2.5]`);
});

await check('a read that fails is dropped: nothing thrown, onUpdate not called, the next end() reads', async () => {
    unhandled.length = 0;
    const { device, reads } = mockDevice();
    const profiler = new GPUProfiler(device);
    let updates = 0;
    profiler.begin();
    profiler.end(() => updates++);
    // What Chrome raised when the worker destroyed the device during the read (phase 10.0).
    reads[0].reject(new DOMException('Device is destroyed.', 'AbortError'));
    await settle();
    assert(unhandled.length === 0, `unhandled rejection: ${unhandled.map(String).join('; ')}`);
    assert(updates === 0, `onUpdate called ${updates} times`);
    profiler.begin();
    profiler.end(() => updates++);
    assert(reads.length === 2, `the next end() started ${reads.length - 1} reads, expected 1`);
});

await check('an error thrown by onUpdate still surfaces', async () => {
    unhandled.length = 0;
    const { device, reads } = mockDevice();
    const profiler = new GPUProfiler(device);
    profiler.begin();
    profiler.end(() => { throw new Error('the app’s own error'); });
    device.times = BigInt64Array.of(0n, 1_000_000n);
    reads[0].resolve();
    await settle();
    assert(unhandled.length === 1 && /the app’s own error/.test(unhandled[0]?.message),
        `expected the app's error as one unhandled rejection, got [${unhandled.map(String).join('; ')}]`);
});

if (failures) {
    console.log(`\n${failures} profiler test(s) failed.`);
    process.exit(1);
}
console.log('\nAll profiler headless tests passed.');
