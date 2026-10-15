// Cycle D, step 1, phase 4 review: what the Models field looks like under each post chain, at the
// zooms of Herd & Wolves. A look spike, not a suite. Rendered by test/gpu/render.mjs; run.mjs renders
// every chain at every zoom and tiles them into one sheet.
//
//   NG_LOOK_CHAIN  a key of CHAINS below          NG_LOOK_ZOOM  close | mid | default | far
//   NG_SITE_PUBLIC the test site's public/ folder (the models and the palette are read from there,
//                  never copied into this repository)
//
// The scene is the demo's (HerdWolvesGPU, phase 4): its 160 m field and grass fragments, 768 grass
// patches placed by its formula, 3,840 cows in herds and 256 foxes at their rest pose, welded and
// shaded by its animal fragment (faceted bands; the four palette swatches read here from the PNG).
// The camera is its orbit: azimuth 0.3, polar 0.4, near and far as it sets them.
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { rand01 } from '../../../dist/agents.js';
import { readPNG } from './png-read.mjs';

const SITE = process.env.NG_SITE_PUBLIC ?? 'D:/NULLGRAPH/Minimal_TypeScript_v6.0.1/starter-vite-ts/public';
const CHAIN = process.env.NG_LOOK_CHAIN ?? 'anime';
const ZOOM = process.env.NG_LOOK_ZOOM ?? 'default';
const DISTANCE = { close: 12, mid: 40, default: 190, far: 340 };
// NG_LOOK_FENCE=1 adds the site's fence (its own fence.ts, run by Node's type stripping) in stock toon
// wood as field.ts draws it; NG_LOOK_ZOOM=edge looks at it from inside the field.
const FENCE = process.env.NG_LOOK_FENCE === '1';
const HALF = 80, COWS = 3840, FOXES = 256;

// --- the demo's fragments (field.ts, models.ts), the palette swatches inlined ------------------
const FIELD_FRAGMENT = /* wgsl */ `
@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let p = in.worldPos.xz;
    let patches = sin(p.x * 0.13 + sin(p.y * 0.09)) * cos(p.y * 0.11);
    let bands = smoothstep(-0.15, 0.3, patches);
    let base = mix(vec3<f32>(0.14, 0.32, 0.09), vec3<f32>(0.32, 0.5, 0.16), bands);
    return vec4<f32>(base, 1.0);
}`;
const GRASS_FRAGMENT = /* wgsl */ `
@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let sun = abs(dot(normalize(in.worldNormal), normalize(vec3<f32>(0.4, 1.0, 0.3))));
    let ramp = 0.6 + 0.4 * smoothstep(0.4, 0.48, sun);
    return vec4<f32>(vec3<f32>(0.25, 0.46, 0.1) * ramp, 1.0);
}`;
const vec3 = c => `vec3<f32>(${c.map(x => x.toFixed(4)).join(', ')})`;
const animalFragment = swatch => /* wgsl */ `
@fragment fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    var base = ${vec3(swatch[0])};
    if (in.uv.x > 0.9) { base = ${vec3(swatch[1])}; }
    else if (in.uv.x > 0.4) { base = ${vec3(swatch[2])}; }
    else if (in.uv.x > 0.3) { base = ${vec3(swatch[3])}; }
    let n = normalize(cross(dpdx(in.worldPos), dpdy(in.worldPos)));
    let sun = abs(dot(n, normalize(vec3<f32>(0.4, 1.0, 0.3))));
    let light = 0.42 + 0.28 * smoothstep(0.25, 0.3, sun) + 0.3 * smoothstep(0.65, 0.7, sun);
    return vec4<f32>(base * light, 1.0);
}`;

// --- candidate ink: silhouettes from depth alone --------------------------------------------
// 1 / view depth is affine across any plane on screen, so its second difference is zero on the
// ground at any angle: no ink on flat ground. At a silhouette the pixel on the nearer surface sees a
// farther neighbour, and −(second difference) / w is the depth jump as a share of the farther depth:
// a 1.5 m cow on the ground is a large jump at 12 m and a small one at 190 m, so its line fades as
// the camera pulls back, with no distance constant. Only the nearer side is inked: one pixel wide.
const INK_WGSL = /* wgsl */ `
fn inkW(uv: vec2f) -> f32 {
    let d = sampleInput1(uv);
    return (camera.far - d * (camera.far - camera.near)) / (camera.near * camera.far);
}
fn effect(uv: vec2f) -> vec4f {
    let c = sampleInput0(uv);
    let t = passInfo.texelSize * params.width;
    let w0 = inkW(uv);
    let sx = inkW(uv - vec2f(t.x, 0.0)) + inkW(uv + vec2f(t.x, 0.0)) - 2.0 * w0;
    let sy = inkW(uv - vec2f(0.0, t.y)) + inkW(uv + vec2f(0.0, t.y)) - 2.0 * w0;
    let s1 = inkW(uv - t) + inkW(uv + t) - 2.0 * w0;
    let s2 = inkW(uv + vec2f(t.x, -t.y)) + inkW(uv + vec2f(-t.x, t.y)) - 2.0 * w0;
    let jump = -min(min(sx, sy), min(s1, s2)) / w0;
    let ink = smoothstep(params.threshold, params.threshold * 2.5, jump) * params.strength;
    return vec4f(mix(c.rgb, params.color, ink), c.a);
}`;

/** Candidate chains, by name. Each is a factory: PostChain rejects shared effect instances. */
function chains(post) {
    const { ChainPresets, CustomEffect, Tonemap, TonemapPresets, ColorGrade, FXAA, Outline, Bloom } = post;
    const ink = (o = {}) => new CustomEffect({
        name: 'Ink', code: INK_WGSL, inputs: ['chain', 'depth'],
        params: [
            { name: 'width', type: 'f32', default: o.width ?? 1 },
            { name: 'threshold', type: 'f32', default: o.threshold ?? 0.02 },
            { name: 'strength', type: 'f32', default: o.strength ?? 0.9 },
            { name: 'color', type: 'vec3f', default: o.color ?? [0.07, 0.09, 0.05] },
        ],
    });
    const soft = () => new ColorGrade({ contrast: 0.9, saturation: 0.95, temperature: 0.08, lift: [0.025, 0.025, 0.03] });
    return {
        faithful: { make: () => ChainPresets.faithful() },
        anime: { make: () => ChainPresets.anime() },
        // Only parameters of today's effects: depth-only Outline, thin, no posterize, AA before ink.
        outline1: { make: () => [new Tonemap(TonemapPresets.faithful), new FXAA(),
            new Outline({ thickness: 1, lumaSensitivity: 0, strength: 0.85, color: [0.07, 0.09, 0.05] })] },
        ink: { make: () => [soft(), new Tonemap(TonemapPresets.faithful), new FXAA(), ink()] },
        // The same at 1.5× resolution: the final pass's filtering anti-aliases the ink.
        'ink-ss': { make: () => [soft(), new Tonemap(TonemapPresets.faithful), ink({ width: 1.5 })], scale: 1.5 },
        // Round 2, on outline1: what makes it softer without washing it out.
        glow: { make: () => [new Bloom({ threshold: 0.8, knee: 0.4, intensity: 0.2, mips: 5 }), new Tonemap(TonemapPresets.faithful), new FXAA(),
            new Outline({ thickness: 1, lumaSensitivity: 0, strength: 0.85, color: [0.07, 0.09, 0.05] })] },
        agx: { make: () => [new Tonemap(TonemapPresets.natural), new FXAA(),
            new Outline({ thickness: 1, lumaSensitivity: 0, strength: 0.85, color: [0.07, 0.09, 0.05] })] },
        grade: { make: () => [new ColorGrade({ contrast: 0.95, saturation: 1.05, temperature: 0.05 }), new Tonemap(TonemapPresets.faithful), new FXAA(),
            new Outline({ thickness: 1, lumaSensitivity: 0, strength: 0.8, color: [0.1, 0.12, 0.06] })] },
        ss: { make: () => [new Tonemap(TonemapPresets.faithful), new FXAA(),
            new Outline({ thickness: 1.5, lumaSensitivity: 0, strength: 0.85, color: [0.07, 0.09, 0.05] })], scale: 1.5 },
    };
}

/** An sRGB byte as linear light: the demo samples the palette as an sRGB texture. */
const linear = b => { const c = b / 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };

export default async function scene({ engine, width, height, lib }) {
    const { Camera } = lib.core;
    const { StandardLayout, MegabufferBuilder } = lib.geometry;
    const { VisibilitySystem, InstanceBuffer } = lib.visibility;
    const { GLBParser, bakeRigidAnimation } = lib.loaders;
    const { OrbitalCameraState } = lib.cameras;
    const parse = name => {
        const b = readFileSync(`${SITE}/3dAssetsNotIncluded/${name}.glb`);
        return GLBParser.parse(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
    };
    // The animals as the demo bakes them: welded, scaled to their length (cow 1.3 m, fox 1.7 m).
    const animal = (name, length) => {
        const baked = bakeRigidAnimation(parse(`animal-${name}`), { clips: ['idle', 'walk', 'run', 'eat'] });
        let lo = Infinity, hi = -Infinity;
        for (let v = 2; v < baked.vertices.length; v += 8) { lo = Math.min(lo, baked.vertices[v]); hi = Math.max(hi, baked.vertices[v]); }
        const scale = length / (hi - lo), vertices = baked.vertices.slice();
        for (let v = 0; v < vertices.length; v += 8) for (let k = 0; k < 3; k++) vertices[v + k] *= scale;
        return { vertices, indices: baked.indices, radius: baked.bounds.radius * scale };
    };
    const cow = animal('cow', 1.3), fox = animal('fox', 1.7);
    const grass = parse('patch-grass-foliage').meshes[0];
    const palette = readPNG(readFileSync(`${SITE}/textures/colormap.png`));
    const swatch = [[0.03, 0.95], [0.97, 0.05], [0.7, 0.3], [0.22, 0.95]].map(([u, v]) => {
        const at = (Math.floor(v * palette.height) * palette.width + Math.floor(u * palette.width)) * 4;
        return [0, 1, 2].map(k => linear(palette.rgba[at + k]));
    });

    const mega = new MegabufferBuilder(StandardLayout.strideFloats);
    mega.addMesh('field', new Float32Array([-HALF, -0.02, -HALF, 0, 1, 0, 0, 0, HALF, -0.02, -HALF, 0, 1, 0, 1, 0,
        -HALF, -0.02, HALF, 0, 1, 0, 0, 1, HALF, -0.02, HALF, 0, 1, 0, 1, 1]), new Uint16Array([0, 2, 1, 1, 2, 3]));
    mega.addMesh('grass', grass.vertices, grass.indices);
    mega.addMesh('cow', cow.vertices, cow.indices);
    mega.addMesh('fox', fox.vertices, fox.indices);
    const fence = FENCE ? await import(pathToFileURL(`${SITE}/../src/data/engine-demos/demos/AgentsExperiments/HerdWolvesGPU/fence.ts`).href) : null;
    const fenceRows = fence ? fence.fenceRows(HALF) : [];
    if (fence) {
        const { post, rail } = fence.fenceMeshes();
        mega.addMesh('post', post.vertices, post.indices);
        mega.addMesh('rail', rail.vertices, rail.indices);
    }

    // Rows: the field, 768 grass patches (the demo's formula), cows in herds, foxes anywhere.
    const rows = new InstanceBuffer(1 + 768 + COWS + FOXES + fenceRows.length);
    let n = 0;
    rows.write(n++, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, HALF * Math.SQRT2], 0);
    let grassRadius = 0;
    for (let v = 0; v < grass.vertices.length; v += 8) grassRadius = Math.max(grassRadius, Math.hypot(...grass.vertices.subarray(v, v + 3)));
    for (let i = 1; i <= 768; i++) {
        const a = rand01(i, 83, 0) * Math.PI * 2, s = 0.65 + rand01(i, 83, 1) * 0.7;
        const x = (rand01(i, 83, 2) * 2 - 1) * (HALF - 2), z = (rand01(i, 83, 3) * 2 - 1) * (HALF - 2);
        rows.write(n++, [Math.cos(a) * s, 0, -Math.sin(a) * s, 0, 0, s, 0, 0, Math.sin(a) * s, 0, Math.cos(a) * s, 0, x, 0, z, 1], [0, 0, 0, grassRadius], 1);
    }
    let seed = 7;
    const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
    const gauss = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
    const place = (x, z, mesh, radius) => {
        const h = random() * Math.PI * 2, c = Math.cos(h), s = Math.sin(h);
        const cx = Math.max(-HALF + 1, Math.min(HALF - 1, x)), cz = Math.max(-HALF + 1, Math.min(HALF - 1, z));
        rows.write(n++, [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, cx, 0, cz, 1], [0, 0, 0, radius], mesh);
    };
    // The first herd and four foxes sit at the orbit's target, so the close views hold animals.
    for (let herd = 0; herd < COWS / 64; herd++) {
        const hx = herd ? (random() * 2 - 1) * (HALF - 8) : 3, hz = herd ? (random() * 2 - 1) * (HALF - 8) : -2;
        for (let k = 0; k < 64; k++) place(hx + gauss() * 5, hz + gauss() * 5, 2, cow.radius);
    }
    for (let k = 0; k < FOXES; k++) {
        const near = k < 4;
        place(near ? (random() * 2 - 1) * 4 : (random() * 2 - 1) * (HALF - 1), near ? (random() * 2 - 1) * 4 : (random() * 2 - 1) * (HALF - 1), 3, fox.radius);
    }

    for (const row of fenceRows) rows.write(n++, row.model, row.bounds, row.mesh === 'post' ? 4 : 5);

    // The post chain, the scene pass into it, the batches in that pass.
    const make = chains(lib.post)[CHAIN];
    if (!make) throw new Error(`NG_LOOK_CHAIN: no chain "${CHAIN}"; one of ${Object.keys(chains(lib.post)).join(', ')}`);
    const post = new lib.post.PostChain(engine, { gbuffer: { depth: 'depth24plus' }, scale: make.scale ?? 1 });
    post.setChain(make.make());
    const pass = engine.createPass({ name: 'Look scene', isMainScreenPass: false,
        ...post.sceneAttachments({ clearValue: { r: 0.12, g: 0.2, b: 0.09, a: 1 } }) });
    const vis = new VisibilitySystem(engine, { maxInstances: n });
    vis.setGeometry(mega);
    vis.setInstances(rows.data, n);
    const targetFormat = post.format;
    vis.createInstanceDrawBatch(pass, 'field', 0, { targetFormat, fragmentBody: FIELD_FRAGMENT });
    vis.createInstanceDrawBatch(pass, 'grass', 1, { targetFormat, fragmentBody: GRASS_FRAGMENT });
    vis.createInstanceDrawBatch(pass, 'cow', 2, { targetFormat, fragmentBody: animalFragment(swatch) });
    vis.createInstanceDrawBatch(pass, 'fox', 3, { targetFormat, fragmentBody: animalFragment(swatch) });
    if (fence) {
        const wood = rgb => ({ kind: 'toon', instance: new lib.materials.ToonMaterial(engine, {
            baseColor: [...rgb, 1], rampSteps: 3, rimLightColor: [0.1, 0.08, 0.06], rimLightPower: 3, specularThreshold: 0.98 }) });
        vis.createInstanceDrawBatch(pass, 'post', 4, { targetFormat, material: wood([0.2, 0.13, 0.08]) });
        vis.createInstanceDrawBatch(pass, 'rail', 5, { targetFormat, material: wood([0.25, 0.18, 0.11]) });
    }

    // The demo's orbit and clip planes.
    const distance = DISTANCE[ZOOM] ?? 12;
    const orbit = new OrbitalCameraState({ target: [0, 0, 0], distance, azimuthAngle: 0.3, polarAngle: 0.4, minDistance: 6, maxDistance: 340, maxPolarAngle: 1.45 });
    const camera = new Camera(60, width / height, Math.max(0.05, distance * 0.004), distance * 2 + HALF * 3);
    if (ZOOM === 'edge') camera.updateView([6, 3, -HALF + 10], [0, 0.6, -HALF - 1]);
    else camera.updateView(orbit.getPosition(), orbit.target);
    engine.updateCamera(camera);
    post.upload();
    vis.upload();
    return () => [...vis.getComputeStages(), pass, ...post.getPassSequence()];
}
