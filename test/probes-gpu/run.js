// GI colour bleeding and feedback convergence, measured on rendered neutral floor pixels.
// Both hosts: npm run test:gpu / test:gpu:browser -- --only probes. No assets or goldens.
import { NullGraph, Camera } from '../../dist/index.js';
import { VisibilitySystem, InstanceBuffer } from '../../dist/visibility.js';
import { MegabufferBuilder, Primitives, StandardLayout } from '../../dist/geometry.js';
import { LambertMaterial } from '../../dist/materials.js';
import { LightSystem } from '../../dist/lights.js';
import { ShadowSystem } from '../../dist/shadows.js';
import { EnvironmentSystem, ProbeVolume } from '../../dist/environment.js';
import { mat4, vec4 } from 'gl-matrix';
import { harnessEnv } from '../gpu/harness-env.mjs';
import { readBuffer, readTextureRGBA8 } from '../gpu/readback.mjs';

const { log, check, fatal, finish, saveImage } = harnessEnv('__PROBE_RESULTS__');
const S = 256;
const engine = new NullGraph();
const init = await engine.init(null, { width: S, height: S });
if (!init.success) { fatal(init.error); throw new Error(init.error); }
const device = engine.device;
const env = new EnvironmentSystem(engine, {
    sourceSize: 16, prefilterMips: 3, prefilterSamples: 16, lutSize: 16,
});
env.setProceduralSky({
    zenith: [0.04, 0.04, 0.04], horizon: [0.04, 0.04, 0.04], ground: [0.04, 0.04, 0.04],
    sunDir: [-0.7, -1, 0], sunIntensity: 0,
});
engine.render([...env.getBakeStages()]);
env.copySHToParams();

const mega = new MegabufferBuilder(StandardLayout.strideFloats);
const floor = Primitives.createPlane(StandardLayout, 24, 24);
for (let i = 0; i < floor.vertices.length; i += StandardLayout.strideFloats) {
    const v = floor.vertices, y = v[i + 1], z = v[i + 2], ny = v[i + 4], nz = v[i + 5];
    v[i + 1] = z; v[i + 2] = -y; v[i + 4] = nz; v[i + 5] = -ny;
}
mega.addMesh('floor', floor.vertices, floor.indices);
const wall = Primitives.createCube(StandardLayout, 1, 1, 1);
mega.addMesh('wall', wall.vertices, wall.indices);
const vis = new VisibilitySystem(engine, { maxInstances: 2 });
vis.setGeometry(mega);
const instances = new InstanceBuffer(2);
instances.write(0, mat4.create(), [0, 0, 0, 17], 0);
const model = mat4.create();
mat4.fromScaling(model, [0.2, 4, 10]); model[12] = -2; model[13] = 2;
instances.write(1, model, [0, 0, 0, Math.sqrt(3)], 1);
vis.setInstances(instances.data, 2);

const lights = new LightSystem(engine, { technique: 'standard', capacity: { point: 0, directional: 1, spot: 0 } });
const direction = [-0.7, -1, 0];
const length = Math.hypot(...direction);
lights.setRawData(new Float32Array([0, 0, 0, 2, 1, 1, 1, 0.3, ...direction.map(v => v / length), 0, 0, 0, 0, 0]));
lights.upload();
const camera = new Camera(65, 1, 0.1, 80);
camera.updateView([4, 12, 10], [4, 0, 0]);
engine.updateCamera(camera);
const shadows = new ShadowSystem(engine, {
    cascades: 1, resolution: 256, shadowDistance: 40, casterExtrusion: 40, visibility: vis,
});
shadows.setLight(direction);
shadows.registerCaster('floor', 0); shadows.registerCaster('wall', 1);
const lighting = shadows.wrapLighting(lights);
// Neutral diffuse reflectance below one, so feedback has headroom and cannot saturate the target.
const floorMat = new LambertMaterial(engine, { baseColor: [0.6, 0.6, 0.6, 1], ambient: 0 });
const wallMat = new LambertMaterial(engine, { baseColor: [0.7, 0.02, 0.02, 1], ambient: 0 });
const color = device.createTexture({ size: [S, S], format: 'rgba16float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
const depth = device.createTexture({ size: [S, S], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
const pass = engine.createPass({ name: 'GI floor measurement', size: [S, S],
    colorAttachments: [{ view: color.createView(), clearValue: [0, 0, 0, 1], loadOp: 'clear', storeOp: 'store' }],
    depthStencilAttachment: { view: depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
});

// Average a 5x5 patch around each world-space floor point. Read linear float values, not
// tone-mapped/quantized bytes: saturation or an RGBA8 plateau must not masquerade as convergence.
const points = [[0, 0, 0], [7, 0, 0]];
const pixels = points.map(p => {
    const clip = vec4.transformMat4(vec4.create(), [...p, 1], camera.bufferData.subarray(0, 16));
    return [Math.floor((clip[0] / clip[3] + 1) * S / 2), Math.floor((1 - clip[1] / clip[3]) * S / 2)];
});
const out = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const sample = engine.createComputeStage({ name: 'GI linear floor readback', dispatchCount: [1, 1, 1], shaderCode: `
    @group(0) @binding(0) var image: texture_2d<f32>;
    @group(0) @binding(1) var<storage, read_write> result: array<vec4<f32>, 2>;
    @compute @workgroup_size(1) fn cs_main() {
        let centers = array<vec2<i32>, 2>(vec2<i32>(${pixels[0]}), vec2<i32>(${pixels[1]}));
        for (var p = 0u; p < 2u; p++) {
            var sum = vec4<f32>(0.0);
            for (var y = -2; y <= 2; y++) { for (var x = -2; x <= 2; x++) {
                sum += textureLoad(image, centers[p] + vec2<i32>(x, y), 0);
            } }
            result[p] = sum / 25.0;
        }
    }` });
engine.setComputePassBindGroup(sample, [{ binding: 0, resource: color.createView() }, { binding: 1, resource: { buffer: out } }]);
const read = async () => {
    engine.render([sample]);
    const values = new Float32Array(await readBuffer(device, out));
    return [Array.from(values.subarray(0, 3)), Array.from(values.subarray(4, 7))];
};
const red = rgb => rgb[0] - (rgb[1] + rgb[2]) / 2;
const fmt = pairs => pairs.map(rgb => '[' + rgb.map(v => v.toFixed(5)).join(', ') + ']').join(' / ');

async function runVolume(feedback) {
    log(`Building ${feedback ? 'feedback' : 'direct-only'} probe captures…`);
    const volume = new ProbeVolume(engine, {
        origin: [-1, 0.5, -1], spacing: 2, dims: [6, 2, 2], probesPerFrame: 1,
        captureFaceSize: 16, captureAtlasSize: 512, reflectionFaceSize: 16,
        normalBiasTexels: 0.5, borderFade: 0.1, distanceBias: 0,
        environment: env, visibility: vis, shadows,
    });
    for (const [name, index, instance] of [['floor', 0, floorMat], ['wall', 1, wallMat]]) {
        const indirect = volume.getShaderLayout();
        // Keep the volume's binding contract even for the no-feedback control: registration's
        // combined binder always supplies 7–13. A Phase-1-only layout would reject those entries.
        // The tiny nonzero touch keeps auto-layout bindings live, with negligible feedback.
        const oneBounce = { ...indirect, wgslLogic: indirect.wgslLogic.replace(
            'fn getIndirectDiffuse(', 'fn getIndirectDiffuseFeedback(') + `
            fn getIndirectDiffuse(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
                return (getIndirectDiffuseGlobal(p, n) + getIndirectDiffuseFeedback(p, n)) * 0.0000001;
            }` };
        volume.registerCaptureMesh(name, index, {
            kind: 'lambert', instance, lighting, ...(feedback ? {} : { indirect: oneBounce }),
        }, lights);
    }
    const batches = [['floor', 0, floorMat], ['wall', 1, wallMat]].map(([name, index, instance]) => {
        const batch = vis.createInstanceDrawBatch(pass, name, index, {
            targetFormat: 'rgba16float', material: { kind: 'lambert', instance, lighting, indirect: volume.getShaderLayout() },
        });
        volume.applyToBatch(batch, lights); shadows.applyToBatch(batch); return batch;
    });
    const frame = async (capture = true) => {
        if (capture) volume.update([4, 1, 0]);
        shadows.update(camera); vis.upload();
        engine.render([...vis.getComputeStages(), ...shadows.getPasses(),
            ...(capture ? [volume.getCapturePass(), ...volume.getProjectStages()] : []), pass]);
        return read();
    };
    const history = [];
    log('Capture pipelines ready; refreshing volume…');
    const refresh = async () => {
        volume.invalidate({ min: [-1, 0.5, -1], max: [9, 2.5, 1] });
        let result;
        while (volume.warmPending() > 0) result = await frame();
        return result;
    };
    for (let i = 0; i < 16; i++) history.push(await refresh());
    log(`${feedback ? 'feedback' : 'direct-only capture'} near red excess: ${history.map(v => red(v[0]).toFixed(5)).join(' → ')}; final near / far ${fmt(history.at(-1))}`);
    await saveImage(feedback ? 'gi-red-feedback' : 'gi-red-one-bounce',
        await readTextureRGBA8(device, color, S, S, 'rgba16float'), S, S);
    let disabled, neutral;
    if (feedback) {
        volume.setIntensity(0); disabled = await frame(false); volume.setIntensity(1);
        wallMat.updateUniforms({ baseColor: [0.7, 0.7, 0.7, 1] });
        for (let i = 0; i < 16; i++) neutral = await refresh();
        await saveImage('gi-neutral-wall', await readTextureRGBA8(device, color, S, S, 'rgba16float'), S, S);
        wallMat.updateUniforms({ baseColor: [0.7, 0.02, 0.02, 1] });
    }
    // ProbeVolume currently leaves capture batches registered at destroy. Unregister them in the
    // harness before destroying their resources, just as scene batches are explicitly removed.
    for (const batch of [...volume.getCapturePass().batches, ...batches]) vis.removeDrawBatch(batch);
    volume.destroy();
    return { history, disabled, neutral };
}

const single = await runVolume(false);
const multi = await runVolume(true);
const final = multi.history.at(-1), early = multi.history[0];
check('G1: both floor patches are drawn, finite and unsaturated in linear HDR',
    [...single.history, ...multi.history, multi.disabled, multi.neutral].every(pair => pair.flat().every(v => Number.isFinite(v) && v > 0.005 && v < 0.9)),
    `near / far pixels ${JSON.stringify(pixels)}; final ${fmt(final)}`);
check('G2: a red wall bleeds measurably more red onto the near floor than the far floor',
    red(final[0]) > 0.01 && red(final[0]) > red(final[1]) + 0.005,
    `red excess near=${red(final[0]).toFixed(5)} far=${red(final[1]).toFixed(5)}`);
check('G3: disabling volume lighting removes the red tint (same geometry, lights and shadows)',
    Math.abs(red(multi.disabled[0])) < 0.001 && red(final[0]) > red(multi.disabled[0]) + 0.01,
    fmt(multi.disabled));
check('G4: changing only the wall to neutral removes the red tint after refresh',
    Math.abs(red(multi.neutral[0])) < 0.001 && red(final[0]) > red(multi.neutral[0]) + 0.01,
    fmt(multi.neutral));
check('G5: feedback changes the initial estimate and adds red beyond direct-only captures',
    Math.abs(red(final[0]) - red(early[0])) > 0.001 && red(final[0]) > red(single.history.at(-1)[0]) + 0.002,
    `first=${red(early[0]).toFixed(5)} one-bounce=${red(single.history.at(-1)[0]).toFixed(5)} multi=${red(final[0]).toFixed(5)}`);
const delta = (a, b) => Math.max(...a.flat().map((v, i) => Math.abs(v - b.flat()[i])));
const initialDelta = delta(multi.history[0], multi.history[1]);
const lateDelta = Math.max(...multi.history.slice(12).map((v, i) => delta(v, multi.history[11 + i])));
check('G6: repeated multi-bounce refreshes converge, not brighten forever',
    initialDelta > 0.001 && lateDelta < 0.001 && lateDelta < initialDelta * 0.1,
    `initial max RGB step=${initialDelta.toFixed(6)} late max step=${lateDelta.toFixed(6)} (last four refreshes)`);
shadows.destroy(); vis.destroy(); env.destroy(); floorMat.destroy(); wallMat.destroy();
color.destroy(); depth.destroy(); out.destroy();
await finish('All probe GI GPU tests passed.');