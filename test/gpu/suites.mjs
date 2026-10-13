// The GPU suite registry — one list for both hosts:
//   npm run test:gpu           Node + Dawn (test/gpu/run-gpu-tests.mjs)
//   npm run test:gpu:browser   headless Chrome (test/gpu/browser-run.mjs)
//
// A new GPU harness is one line here plus its folder: test/<name>-gpu/{run.js,index.html}, with
// run.js publishing its summary on `key` through test/gpu/harness-env.mjs.
// `timeout` (seconds) is for a suite that outlasts the browser runner's default of 120 s; an explicit
// --timeout still wins.

export const SUITES = [
    { name: 'shadows', file: 'test/shadows-gpu/run.js', key: '__SHADOW_RESULTS__' },
    { name: 'sky', file: 'test/environment-gpu/run.js', key: '__SKY_RESULTS__' },
    { name: 'visibility', file: 'test/visibility-gpu/run.js', key: '__VISIBILITY_RESULTS__' },
    { name: 'core', file: 'test/core-gpu/run.js', key: '__CORE_GPU_RESULTS__' },
    { name: 'onnx', file: 'test/onnx-gpu/run.js', key: '__ONNX_RESULTS__' },
    { name: 'picking', file: 'test/picking-gpu/run.js', key: '__PICKING_RESULTS__' },
    { name: 'text', file: 'test/text-gpu/run.js', key: '__TEXT_RESULTS__' },
    { name: 'probes', file: 'test/probes-gpu/run.js', key: '__PROBE_RESULTS__' },
    { name: 'agents', file: 'test/agents-gpu/run.js', key: '__AGENTS_RESULTS__' },
    // A8 runs ten 30-second episodes: about a minute under Node, and two in Chrome.
    { name: 'agent-tools', file: 'test/agent-tools-gpu/run.js', key: '__AGENT_TOOLS_RESULTS__', timeout: 360 },
];

/**
 * Render scenes smoke-tested through the CLI (test/gpu/render.mjs): exit 0, no WebGPU error, not a
 * uniform frame. The quick start is here because it is the first code every user and agent copies.
 * Node only — the CLI is the Node host.
 */
export const RENDER_SCENES = [
    { name: 'render:quick-start', file: 'test/gpu/scenes/quick-start.mjs' },
    { name: 'render:instancing', file: 'test/gpu/scenes/instancing.mjs' },
    { name: 'render:culling', file: 'test/gpu/scenes/culling.mjs' },
];
