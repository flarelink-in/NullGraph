// src/extras/post/ShaderComposer.ts
//
// Assembles a complete WGSL module around an effect-authored fragment body.
// The author writes one function:
//
//     fn effect(uv: vec2f) -> vec4f { ... }
//
// Available to the body:
//   params.<field>            this effect's parameters
//   passInfo.resolution       vec2f, output target size in pixels
//   passInfo.texelSize        vec2f, 1.0 / resolution
//   camera.*                  full camera struct (invProj, near, far, time, ...)
//   sampleInput0(uv) ...      declared inputs, in declaration order
//   loadInput0(px) ...        integer-coordinate textureLoad, unfiltered
//
// NOTE: the design doc calls the header `pass`, but `pass` is a WGSL reserved
// word — the generated symbol is `passInfo`.

export const PASS_HEADER_FLOATS = 8;

/** Matches Camera.bufferData (108 floats / 432 bytes), see core/Camera.ts. */
const CAMERA_WGSL = `struct Camera {
    viewProj: mat4x4f,
    view: mat4x4f,
    proj: mat4x4f,
    invViewProj: mat4x4f,
    invView: mat4x4f,
    invProj: mat4x4f,
    eye: vec3f,
    time: f32,
    near: f32,
    far: f32,
    fovRads: f32,
    aspect: f32,
    _padEnd: vec4f,
};`;

/** Static per-pass header, written once at compile via updateBatchData. */
const PASS_HEADER_WGSL = `struct PassHeader {
    resolution: vec2f,
    texelSize: vec2f,
    passIndex: u32,
    _pad0: f32,
    _pad1: vec2f,
};`;

const VS_WGSL = `struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    var o: VSOut;
    o.pos = vec4f(p[vi], 0.0, 1.0);
    // Y FLIP: clip space is +Y up, texture space is +V down. Mandatory.
    o.uv = vec2f((p[vi].x + 1.0) * 0.5, 1.0 - (p[vi].y + 1.0) * 0.5);
    return o;
}`;

const SRGB_WGSL = `fn linearToSRGB(c: vec3f) -> vec3f {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3f(0.0031308));
}`;

export interface ComposerInput {
    /** Depth inputs bind as texture_depth_2d and sample via the nearest
     *  (non-filtering) sampler — depth24plus is not filterable. */
    depth: boolean;
}

export interface ComposedBinding {
    binding: number;
    kind: 'params' | 'linear-sampler' | 'nearest-sampler' | 'texture';
    /** For kind 'texture': index into the pass's input list. */
    inputIndex?: number;
}

export interface ComposeOptions {
    /** The effect-authored fragment body (must define fn effect(uv) -> vec4f). */
    body: string;
    /** Generated Params struct text from layoutFields(). */
    paramsWgsl: string;
    /** First Params member — referenced in fs_main to force static usage. */
    paramsTouchField: string;
    inputs: ComposerInput[];
    /** Append the linear->sRGB encode to the fragment output (final screen pass
     *  on a non-srgb swapchain). Effects never encode themselves. */
    encodeToSRGB: boolean;
    label?: string;
}

export interface ComposedShader {
    code: string;
    /** Group-1 bind group entries, in binding order. PostChain turns these into
     *  GPUBindGroupEntry objects. */
    bindings: ComposedBinding[];
}

export function composePassShader(opts: ComposeOptions): ComposedShader {
    const anyColor = opts.inputs.some(i => !i.depth);
    const anyDepth = opts.inputs.some(i => i.depth);

    const bindings: ComposedBinding[] = [{ binding: 0, kind: 'params' }];
    let nextBinding = 1;
    let linearBinding = -1;
    let nearestBinding = -1;
    if (anyColor) {
        linearBinding = nextBinding++;
        bindings.push({ binding: linearBinding, kind: 'linear-sampler' });
    }
    if (anyDepth) {
        nearestBinding = nextBinding++;
        bindings.push({ binding: nearestBinding, kind: 'nearest-sampler' });
    }

    const textureDecls: string[] = [];
    const helpers: string[] = [];
    const touches: string[] = [];

    if (linearBinding >= 0) {
        textureDecls.push(`@group(1) @binding(${linearBinding}) var sampLinear: sampler;`);
    }
    if (nearestBinding >= 0) {
        textureDecls.push(`@group(1) @binding(${nearestBinding}) var sampNearest: sampler;`);
    }

    opts.inputs.forEach((inp, i) => {
        const b = nextBinding++;
        bindings.push({ binding: b, kind: 'texture', inputIndex: i });
        if (inp.depth) {
            textureDecls.push(`@group(1) @binding(${b}) var tex${i}: texture_depth_2d;`);
            helpers.push(
                `fn sampleInput${i}(uv: vec2f) -> f32 { return textureSampleLevel(tex${i}, sampNearest, uv, 0); }`,
                `fn loadInput${i}(px: vec2i) -> f32 { return textureLoad(tex${i}, px, 0); }`,
            );
            touches.push(`    _ = textureSampleLevel(tex${i}, sampNearest, vec2f(0.5), 0);`);
        } else {
            textureDecls.push(`@group(1) @binding(${b}) var tex${i}: texture_2d<f32>;`);
            helpers.push(
                `fn sampleInput${i}(uv: vec2f) -> vec4f { return textureSample(tex${i}, sampLinear, uv); }`,
                `fn loadInput${i}(px: vec2i) -> vec4f { return textureLoad(tex${i}, px, 0); }`,
            );
            touches.push(`    _ = textureSampleLevel(tex${i}, sampLinear, vec2f(0.5), 0.0);`);
        }
    });

    // layout:'auto' prunes any binding the shader does not statically use, and
    // supplying an entry for a pruned binding throws at bind-group creation.
    // These phony assignments pin every declared binding — including the two
    // group-0 bindings createBatch supplies unconditionally.
    const touchBlock = [
        `    _ = camera.time;`,
        `    _ = passInfo.passIndex;`,
        `    _ = params.${opts.paramsTouchField};`,
        ...touches,
    ].join('\n');

    const encode = opts.encodeToSRGB
        ? `    color = vec4f(linearToSRGB(color.rgb), color.a);\n`
        : '';

    const code = `// Generated by NullGraph ShaderComposer${opts.label ? ` — ${opts.label}` : ''}
${CAMERA_WGSL}
@group(0) @binding(0) var<uniform> camera: Camera;

${PASS_HEADER_WGSL}
@group(0) @binding(1) var<storage, read> passInfo: PassHeader;

${opts.paramsWgsl}
@group(1) @binding(0) var<uniform> params: Params;
${textureDecls.join('\n')}

${VS_WGSL}

${opts.encodeToSRGB ? SRGB_WGSL + '\n' : ''}
${helpers.join('\n')}

// ---- effect body ----
${opts.body}
// ---------------------

@fragment fn fs_main(v: VSOut) -> @location(0) vec4f {
${touchBlock}
    var color = effect(v.uv);
${encode}    return color;
}
`;

    return { code, bindings };
}
