// Headless core execution-model tests — indirect dispatch (architecture-tree.md §1).
//
// Everything here runs through a mock device that RECORDS the encoder calls, because the property
// under test is "which command did the pipeline issue", and that is exactly the thing a rendered
// image cannot tell you: a dropped dispatch and a dispatch that ran but computed zero look
// identical downstream.

// Node has no WebGPU globals; the usage masks are read by the validation under test, so they must
// be the real bit values, not placeholders.
globalThis.GPUBufferUsage = globalThis.GPUBufferUsage ?? {
    MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32,
    UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512,
};
globalThis.GPUTextureUsage = globalThis.GPUTextureUsage ?? {
    COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16,
};

const { BatchManager, BufferManager, RenderPipeline } = await import('../dist/index.js');

let failures = 0;
function check(name, fn) {
    try { fn(); console.log(`  ok    ${name}`); }
    catch (e) { failures++; console.log(`  FAIL  ${name}\n        ${e.message}`); }
}
function assert(v, m) { if (!v) throw new Error(m); }
function throws(fn, pattern, label) {
    let message = '';
    try { fn(); } catch (e) { message = e.message; }
    if (!pattern.test(message)) throw new Error(`${label}: expected ${pattern}, got "${message}"`);
    return message;
}

const U = globalThis.GPUBufferUsage;

/** A buffer stub carrying the two fields the validation actually reads. */
function buf(usage, size = 64) {
    return { usage, size, destroy() {} };
}

/** Records every call the RenderPipeline makes on a compute pass. */
function recordingCtx() {
    const calls = [];
    const computePass = {
        setPipeline: () => calls.push(['setPipeline']),
        setBindGroup: (i) => calls.push(['setBindGroup', i]),
        dispatchWorkgroups: (x, y, z) => calls.push(['dispatch', x, y, z]),
        dispatchWorkgroupsIndirect: (b, o) => calls.push(['dispatchIndirect', b, o]),
        end: () => calls.push(['endComputePass']),
    };
    const encoder = {
        beginComputePass: () => { calls.push(['beginComputePass']); return computePass; },
        beginRenderPass: () => {
            calls.push(['beginRenderPass']);
            return {
                setPipeline() {}, setBindGroup() {}, setViewport() {}, setScissorRect() {},
                setVertexBuffer() {}, setIndexBuffer() {}, drawIndexed() {}, draw() {},
                drawIndexedIndirect() {}, end: () => calls.push(['endRenderPass']),
            };
        },
        finish: () => ({}),
    };
    const ctx = {
        device: {
            createCommandEncoder: () => encoder,
            createShaderModule: () => ({}),
            createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
            createBindGroup: () => ({}),
            createBuffer: (d) => ({ ...d, getMappedRange: () => new ArrayBuffer(d.size), unmap() {}, destroy() {} }),
            queue: { writeBuffer: (...a) => calls.push(['writeBuffer', ...a.slice(0, 2)]), submit() {} },
        },
        width: 1920, height: 1080,
        cameraUniformBuffer: {},
    };
    return { ctx, calls };
}

// ---------------------------------------------------------------------------

check('U1: an indirect stage issues dispatchWorkgroupsIndirect, not dispatchWorkgroups', () => {
    const { ctx, calls } = recordingCtx();
    const manager = new BatchManager(ctx);
    const args = buf(U.INDIRECT | U.STORAGE);
    const stage = manager.createComputeStage({
        name: 'cull', shaderCode: '@compute @workgroup_size(1) fn cs_main() {}',
        dispatchCount: [999, 1, 1],          // present, and must be ignored
        indirectBuffer: args, indirectOffset: 12,
    });

    new RenderPipeline(ctx).render([stage]);

    const indirect = calls.filter(c => c[0] === 'dispatchIndirect');
    const direct = calls.filter(c => c[0] === 'dispatch');
    assert(indirect.length === 1, `expected one indirect dispatch, got ${indirect.length}`);
    assert(direct.length === 0, `dispatchCount was still used: ${JSON.stringify(direct)}`);
    assert(indirect[0][1] === args, 'the wrong buffer was passed');
    assert(indirect[0][2] === 12, `offset ${indirect[0][2]} was not forwarded`);
});

check('U2: a direct stage is unchanged — dispatchWorkgroups, byte-for-byte the old path', () => {
    const { ctx, calls } = recordingCtx();
    const manager = new BatchManager(ctx);
    const stage = manager.createComputeStage({
        name: 'bin', shaderCode: '@compute @workgroup_size(1) fn cs_main() {}',
        dispatchCount: [8, 4, 2],
    });
    assert(stage.isIndirect === false, 'a stage with no indirectBuffer reports isIndirect');

    new RenderPipeline(ctx).render([stage]);

    const direct = calls.filter(c => c[0] === 'dispatch');
    assert(direct.length === 1, `expected one direct dispatch, got ${direct.length}`);
    assert(direct[0][1] === 8 && direct[0][2] === 4 && direct[0][3] === 2,
        `dispatch counts were ${direct[0].slice(1)}`);
    assert(calls.every(c => c[0] !== 'dispatchIndirect'), 'a direct stage went indirect');
});

check('U3: the callback form of dispatchCount still evaluates per frame', () => {
    const { ctx, calls } = recordingCtx();
    const manager = new BatchManager(ctx);
    let n = 4;
    const stage = manager.createComputeStage({
        name: 'dynamic', shaderCode: '@compute @workgroup_size(1) fn cs_main() {}',
        dispatchCount: () => [n, 1, 1],
    });
    const pipeline = new RenderPipeline(ctx);
    pipeline.render([stage]);
    n = 9;
    pipeline.render([stage]);
    const counts = calls.filter(c => c[0] === 'dispatch').map(c => c[1]);
    assert(counts.length === 2 && counts[0] === 4 && counts[1] === 9,
        `expected [4, 9], got ${JSON.stringify(counts)}`);
});

check('U4: enabled=false skips an indirect dispatch AND its clears, same as a direct one', () => {
    // design-principles §9: whatever suppresses a draw must suppress its compute. An indirect
    // stage that ignored `enabled` would keep dispatching from a stale count buffer.
    const { ctx, calls } = recordingCtx();
    const manager = new BatchManager(ctx);
    const args = buf(U.INDIRECT | U.STORAGE);
    const stage = manager.createComputeStage({
        name: 'gated', shaderCode: '@compute @workgroup_size(1) fn cs_main() {}',
        indirectBuffer: args,
        preClearBuffers: [{ buffer: args, offset: 0, data: new Uint32Array([0, 1, 1]) }],
    });
    stage.enabled = false;
    new RenderPipeline(ctx).render([stage]);
    assert(calls.every(c => c[0] !== 'dispatchIndirect'), 'a disabled stage still dispatched');
    assert(calls.every(c => c[0] !== 'writeBuffer'), 'a disabled stage still ran its clears');
});

check('U5: indirectBuffer is mutable — a stage can be retargeted without a rebuild', () => {
    const { ctx, calls } = recordingCtx();
    const manager = new BatchManager(ctx);
    const first = buf(U.INDIRECT | U.STORAGE);
    const second = buf(U.INDIRECT | U.STORAGE);
    const stage = manager.createComputeStage({
        name: 'retarget', shaderCode: '@compute @workgroup_size(1) fn cs_main() {}',
        indirectBuffer: first,
    });
    const pipeline = new RenderPipeline(ctx);
    pipeline.render([stage]);
    stage.indirectBuffer = second;
    stage.indirectOffset = 24;
    pipeline.render([stage]);
    const issued = calls.filter(c => c[0] === 'dispatchIndirect');
    assert(issued[0][1] === first && issued[1][1] === second, 'retargeting did not take effect');
    assert(issued[1][2] === 24, `offset ${issued[1][2]} was not updated`);
});

check('U6: consecutive stages still share one compute pass when one is indirect', () => {
    const { ctx, calls } = recordingCtx();
    const manager = new BatchManager(ctx);
    const code = '@compute @workgroup_size(1) fn cs_main() {}';
    const a = manager.createComputeStage({ name: 'a', shaderCode: code, dispatchCount: [1, 1, 1] });
    const b = manager.createComputeStage({
        name: 'b', shaderCode: code, indirectBuffer: buf(U.INDIRECT | U.STORAGE),
    });
    new RenderPipeline(ctx).render([a, b]);
    const begins = calls.filter(c => c[0] === 'beginComputePass').length;
    assert(begins === 1, `expected one compute pass, got ${begins} — the pass split between stages`);
});

check('errors: every indirect-dispatch refusal fires, names the stage, and states the way out', () => {
    const { ctx } = recordingCtx();
    const manager = new BatchManager(ctx);
    const code = '@compute @workgroup_size(1) fn cs_main() {}';
    const make = (extra) => () => manager.createComputeStage({ name: 'offender', shaderCode: code, ...extra });

    // No INDIRECT usage — the dispatch cannot read its count.
    let m = throws(make({ indirectBuffer: buf(U.STORAGE) }), /lacks GPUBufferUsage\.INDIRECT/, 'missing INDIRECT');
    assert(/offender/.test(m), `the message does not name the stage: ${m}`);
    assert(/createDispatchIndirectBuffer/.test(m), `no way out given: ${m}`);

    // No STORAGE and no COPY_DST — nothing can ever write the count, so it is permanently zero.
    m = throws(make({ indirectBuffer: buf(U.INDIRECT) }), /neither STORAGE nor COPY_DST/, 'unwritable');
    assert(/permanently zero/.test(m), `the message does not say what goes wrong: ${m}`);

    // COPY_DST alone is legal — a caller may write the triple from the CPU.
    make({ indirectBuffer: buf(U.INDIRECT | U.COPY_DST) })();

    throws(make({ indirectBuffer: buf(U.INDIRECT | U.STORAGE), indirectOffset: 6 }),
        /multiple of 4/, 'misaligned offset');
    throws(make({ indirectBuffer: buf(U.INDIRECT | U.STORAGE), indirectOffset: -4 }),
        /non-negative/, 'negative offset');
    throws(make({ indirectBuffer: buf(U.INDIRECT | U.STORAGE, 16), indirectOffset: 8 }),
        /is 16 bytes; reading three u32 at offset 8 needs 20/, 'buffer too small');
    throws(make({ indirectOffset: 12 }), /indirectOffset was set without an indirectBuffer/,
        'offset without buffer');

    // And the ordinary case must NOT throw, or the validation is just refusing everything.
    make({ indirectBuffer: buf(U.INDIRECT | U.STORAGE), indirectOffset: 0 })();
    make({ dispatchCount: [1, 1, 1] })();
});

check('createDispatchIndirectBuffer allocates a usable buffer, zeroed', () => {
    const { ctx } = recordingCtx();
    const buffers = new BufferManager(ctx.device);
    const b = buffers.createDispatchIndirectBuffer(3);
    assert(b.size === 36, `expected 36 bytes for 3 triples, got ${b.size}`);
    assert((b.usage & U.INDIRECT) !== 0, 'missing INDIRECT');
    assert((b.usage & U.STORAGE) !== 0, 'missing STORAGE');
    assert((b.usage & U.COPY_DST) !== 0, 'missing COPY_DST — preClearBuffers could not reset it');

    // The buffer it produces must satisfy the validation it exists to satisfy.
    new BatchManager(ctx).createComputeStage({
        name: 'consumer', shaderCode: '@compute @workgroup_size(1) fn cs_main() {}',
        indirectBuffer: b, indirectOffset: 24,
    });

    throws(() => buffers.createDispatchIndirectBuffer(0), /must be a positive integer/, 'zero count');
    throws(() => buffers.createDispatchIndirectBuffer(1.5), /must be a positive integer/, 'fractional count');
});

console.log(failures === 0 ? '\nAll core compute tests passed.' : `\n${failures} test(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
