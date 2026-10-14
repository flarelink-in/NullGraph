// Spike S5: does ONNX Runtime run on a device the app made? See docs/agents-design.md §13 (C0) and
// §9.4, and docs/agents-ort-design.md.
//
//   Node + Dawn:   npm run spike:s5          one kind of device per process (settings below)
//   Chrome:        npm run drive -- --serve . "/test/spikes/ort/index.html?spike=s5&device=engine" --wait-for "S5 done" --timeout 120
//
// Settings (NG_SPIKE_DEVICE and NG_SPIKE_BUNDLE under Node, ?device= and &bundle= in a browser):
//   device   engine           the engine's device, as NullGraph.init() makes it (the default)
//            engine-defaults  the engine's device with WebGPU's default limits (autoNegotiate: false)
//            like-ort         a device asked for as ONNX Runtime asks for its own: the adapter's limits
//            ort              ONNX Runtime's own device: the control, which must work
//   bundle   webgpu           the WebGPU provider compiled into the WASM (the default)
//            all              the older provider written in JavaScript; it does not load under Node/Dawn
//
// ONNX Runtime offers two ways to hand it a device, one per provider, and both are tried:
//   - the "device" option of the webgpu provider (the new provider reads it);
//   - ort.env.webgpu.device, set before the first session (the typings say the older provider reads it).
// The result is a finding, not a pass or a failure: the checks here only hold what the measurement
// needs, that the WGSL kernel runs on the device and that the device is still alive at the end.
import { NullGraph } from '../../../dist/index.js';
import { harnessEnv } from '../../gpu/harness-env.mjs';
import { LAYERS, describe, fmt, inBrowser, loadOrt, mlpForward, onnxOf, random, readF32, setting, tanhNetwork, wgslKernel } from './common.mjs';

const { log, check, fatal, finish } = harnessEnv('__ORT_SPIKE_RESULTS__');
const kind = setting('device', 'engine'), bundle = setting('bundle', 'webgpu');
const N = 20000, ROWS = 256, [OBS, ACT] = [LAYERS[0], LAYERS[LAYERS.length - 1]];

const { ort } = await loadOrt({ bundle });
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
let device = null;
if (kind === 'engine' || kind === 'engine-defaults') {
    const engine = new NullGraph();
    const init = await engine.init(null, { width: 8, height: 8, autoNegotiate: kind === 'engine' });
    if (!init.success) { fatal(init.error); throw new Error(init.error); }
    device = engine.device;
} else if (kind === 'like-ort') {
    const limits = {};
    for (const name of ['maxComputeWorkgroupStorageSize', 'maxComputeWorkgroupsPerDimension', 'maxStorageBufferBindingSize', 'maxBufferSize',
        'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX', 'maxComputeWorkgroupSizeY', 'maxComputeWorkgroupSizeZ']) limits[name] = adapter.limits[name];
    device = await adapter.requestDevice({ requiredLimits: limits, requiredFeatures: adapter.features.has('timestamp-query') ? ['timestamp-query'] : [] });
} else {
    ort.env.webgpu.adapter = adapter;
}
const ours = device !== null;
let setter = '';
if (ours) {
    try { ort.env.webgpu.device = device; setter = 'accepted'; } catch (e) { setter = `refused: ${String(e.message).slice(0, 80)}`; }
}
const provider = bundle === 'webgpu' ? { name: 'webgpu', ...(ours ? { device } : {}) } : 'webgpu';
const network = tanhNetwork(), bytes = onnxOf(network);
const result = { spike: 'S5', host: inBrowser ? 'Chrome' : 'Node/Dawn', bundle, device: kind, setter, captures: {} };

// Progress lines: on an app-made device ONNX Runtime has stopped answering, and they say where.
for (const capture of [false, true]) {
    const slot = result.captures[capture ? 'on' : 'off'] = {};
    let session;
    log(`   … graph capture ${capture ? 'on' : 'off'}: creating a session`);
    try {
        session = await ort.InferenceSession.create(bytes, { executionProviders: [provider], preferredOutputLocation: 'gpu-buffer', enableGraphCapture: capture });
        slot.session = 'created';
        log('   … created');
    } catch (e) {
        slot.session = `failed: ${String(e.message ?? e).split('\n')[0].slice(0, 160)}`;
        continue;
    }
    // ONNX Runtime's own device does not outlive its last session: the control gets a new one per session.
    const theirs = await ort.env.webgpu.device;
    if (!ours) device = theirs;
    slot.reportsOurDevice = theirs === device;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const obsData = Float32Array.from({ length: N * OBS }, () => random() * 2 - 1);
    const obs = device.createBuffer({ size: N * OBS * 4, usage }), act = device.createBuffer({ size: N * ACT * 4, usage });
    device.queue.writeBuffer(obs, 0, obsData);
    device.queue.writeBuffer(act, 0, new Float32Array(N * ACT).fill(-7));
    const feeds = { obs: ort.Tensor.fromGpuBuffer(obs, { dataType: 'float32', dims: [N, OBS] }) };
    const fetches = { act: ort.Tensor.fromGpuBuffer(act, { dataType: 'float32', dims: [N, ACT] }) };
    slot.runs = [];
    for (let r = 0; r < 3; r++) {
        try {
            log(`   … run ${r + 1}`);
            await session.run(feeds, fetches);
            await device.queue.onSubmittedWorkDone();
            const got = await readF32(device, act, ROWS * ACT);
            let worst = 0;
            for (let i = 0; i < ROWS; i++) mlpForward(network, obsData.subarray(i * OBS, (i + 1) * OBS)).forEach((v, j) => { worst = Math.max(worst, Math.abs(v - got[i * ACT + j])); });
            slot.runs.push(worst <= 1e-4 ? 'right' : got[0] === -7 ? 'our buffer was not written' : `wrong by ${worst.toExponential(1)}`);
        } catch (e) {
            slot.runs.push(`failed: ${String(e.message ?? e).split('\n')[0].slice(0, 140)}`);
        }
    }
    try { await session.release(); } catch (e) { slot.release = `failed: ${String(e.message).slice(0, 80)}`; }
}
if (!device) { fatal('ONNX Runtime made no device'); throw new Error('no device'); }

// What the measurement needs: the device itself is sound. The WGSL kernel runs on it and gives the
// right actions, before and after ONNX Runtime was through with it.
const kernel = wgslKernel(network);
let wgsl, alive;
try {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const obsData = Float32Array.from({ length: ROWS * OBS }, () => random() * 2 - 1);
    const obs = device.createBuffer({ size: ROWS * OBS * 4, usage }), act = device.createBuffer({ size: ROWS * ACT * 4, usage });
    const weights = device.createBuffer({ size: kernel.packed.byteLength, usage });
    device.queue.writeBuffer(obs, 0, obsData);
    device.queue.writeBuffer(weights, 0, kernel.packed);
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code: kernel.code }), entryPoint: 'cs_main' } });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: obs } }, { binding: 1, resource: { buffer: act } }, { binding: 2, resource: { buffer: weights } }] }));
    pass.dispatchWorkgroups(Math.ceil(ROWS / 64));
    pass.end();
    device.queue.submit([encoder.finish()]);
    const got = await readF32(device, act, ROWS * ACT);
    let worst = 0;
    for (let i = 0; i < ROWS; i++) mlpForward(network, obsData.subarray(i * OBS, (i + 1) * OBS)).forEach((v, j) => { worst = Math.max(worst, Math.abs(v - got[i * ACT + j])); });
    wgsl = worst;
    alive = true;
} catch (e) {
    alive = false;
    wgsl = `${e?.name ?? 'error'}: ${String(e?.message ?? e).slice(0, 100)}`;
}
result.adapter = describe(device);
result.deviceAfterwards = alive ? 'alive' : `gone (${wgsl})`;
const L = device.limits;
log(`\n== S5 · ${result.host} · ${result.adapter} · ONNX Runtime Web ${ort.env.versions?.web ?? ''}, the "${bundle}" bundle`);
log(`   the device: ${kind === 'ort' ? 'ONNX Runtime\'s own (the control)' : kind === 'engine' ? 'the engine\'s' : kind === 'engine-defaults' ? 'the engine\'s, with WebGPU\'s default limits' : 'the app\'s, asked for as ONNX Runtime asks for its own'}`);
log(`   limits: maxStorageBufferBindingSize ${fmt(L.maxStorageBufferBindingSize)}, maxComputeInvocationsPerWorkgroup ${L.maxComputeInvocationsPerWorkgroup}, ` +
    `maxComputeWorkgroupStorageSize ${fmt(L.maxComputeWorkgroupStorageSize)}; features: ${[...device.features].join(', ') || 'none'}`);
if (ours) log(`   handed over: ${bundle === 'webgpu' ? 'as the provider\'s "device" option, and ' : ''}ort.env.webgpu.device was set (${setter})`);
let worked = 0;
for (const [capture, slot] of Object.entries(result.captures)) {
    const right = slot.runs?.every(r => r === 'right');
    if (right) worked++;
    log(`   graph capture ${capture}: session ${slot.session}` + (slot.runs ? `; ort.env.webgpu.device ${slot.reportsOurDevice ? 'is' : 'is NOT'} this device; three runs: ${slot.runs.join(' | ')}` : ''));
}
result.worked = worked === 2;
if (kind === 'ort') check('S5 control: ONNX Runtime runs on its own device', result.worked, JSON.stringify(result.captures));
if (ours) {
    check('S5: the app\'s device is sound, whatever ONNX Runtime did with it: the WGSL kernel runs on it afterwards and equals mlpForward', alive && wgsl <= 1e-4,
        alive ? `worst ${wgsl.toExponential(1)}` : wgsl);
} else {
    log(`   after its last session was released, ONNX Runtime's own device is ${alive ? 'alive' : `gone: ${wgsl}`}`);
}
log(`   FINDING: ONNX Runtime ${result.worked ? 'RAN' : 'DID NOT RUN'} on ${kind === 'ort' ? 'its own device' : 'this app-made device'}.`);
log(`\nS5 JSON ${JSON.stringify(result)}`);
await finish(`S5: measured (${result.worked ? 'ran' : 'did not run'}).`);
log('S5 done');
