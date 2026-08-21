// Generates the complete WGSL module for every v1 post pass (Tonemap all
// curves, Bloom, SSAO, CustomEffect, debug blit) exactly as PostChain
// composes them, and parses each with wgsl_reflect to catch syntax errors,
// reserved words, and malformed declarations.
//   npm run build:lib && node test/post-shader-gen.test.mjs
import { WgslReflect } from '../node_modules/wgsl_reflect/wgsl_reflect.module.js';
import {
    Tonemap, Bloom, SSAO, CustomEffect,
    Vignette, ChromaticAberration, FilmGrain,
    Halation, Kuwahara, Bokeh,
    FXAA, ColorGrade, AutoExposure, LensFlare,
    CelShade, Outline, ChainPresets,
    EFFECT_ORDER, orderChain,
    layoutFields, composePassShader,
} from '../dist/post.js';

function collectPasses(effect, hasDepth, width = 1920, height = 1080) {
    const passes = [];
    const targets = new Map(); // handle -> spec (to know depth inputs)
    let id = 0;
    const ctx = {
        target: (spec, debugName) => {
            const h = { id: id++, debugName };
            targets.set(h, spec);
            return h;
        },
        pass: (req) => passes.push(req),
        has: () => hasDepth,
        width,
        height,
    };
    const decl = effect.declare(ctx);
    return { passes, targets, schema: decl.params };
}

function inputIsDepth(ref, targets) {
    if (ref === 'depth') return true;
    if (typeof ref === 'string') return false;
    return (targets.get(ref)?.format ?? '').startsWith('depth');
}

let failures = 0;
let count = 0;

function validate(label, effect, { encodeLast = false, hasDepth = true } = {}) {
    const { passes, targets, schema } = collectPasses(effect, hasDepth);
    const layout = layoutFields(schema);
    passes.forEach((req, i) => {
        count++;
        const composed = composePassShader({
            body: req.shaderCode,
            paramsWgsl: layout.wgsl,
            paramsTouchField: layout.touchField,
            inputs: req.inputs.map(ref => ({ depth: inputIsDepth(ref, targets) })),
            encodeToSRGB: encodeLast && i === passes.length - 1,
            label: req.name,
        });
        try {
            const reflect = new WgslReflect(composed.code);
            const entries = reflect.entry;
            if (!entries.vertex.length || !entries.fragment.length) {
                throw new Error('missing vs_main/fs_main entry point');
            }
            console.log(`  ok    ${label} / ${req.name}  (${composed.bindings.length} group-1 bindings)`);
        } catch (e) {
            failures++;
            console.log(`  FAIL  ${label} / ${req.name}\n        ${e.message}`);
        }
    });
}

validate('Tonemap aces (screen, sRGB encode)', new Tonemap({ curve: 'aces' }), { encodeLast: true });
validate('Tonemap reinhard', new Tonemap({ curve: 'reinhard' }));
validate('Tonemap agx', new Tonemap({ curve: 'agx' }));
validate('Tonemap none', new Tonemap({ curve: 'none' }));
validate('Bloom mips=6', new Bloom({ mips: 6 }));
validate('SSAO', new SSAO());
validate('Vignette', new Vignette());
validate('ChromaticAberration', new ChromaticAberration());
validate('FilmGrain', new FilmGrain());
validate('Halation', new Halation());
validate('Kuwahara', new Kuwahara());
validate('Bokeh (depth)', new Bokeh());
validate('Bokeh (no depth fallback)', new Bokeh(), { hasDepth: false });
validate('FXAA', new FXAA());
validate('ColorGrade', new ColorGrade());
validate('AutoExposure', new AutoExposure());
validate('LensFlare ghosts=4', new LensFlare());
validate('CelShade', new CelShade());
validate('Outline (depth)', new Outline());
validate('Outline (no depth fallback)', new Outline(), { hasDepth: false });

// Every chain preset must instantiate distinct effects whose passes all
// compose to valid WGSL with the preset's constructor options applied.
for (const [name, factory] of Object.entries(ChainPresets)) {
    const a = factory();
    const b = factory();
    count++;
    if (a.some((e, i) => e === b[i])) {
        failures++;
        console.log(`  FAIL  ChainPresets.${name} returns shared instances`);
    } else {
        console.log(`  ok    ChainPresets.${name} returns fresh instances (${a.length} effects)`);
    }
    for (const effect of a) validate(`ChainPresets.${name}`, effect);
}
validate('CustomEffect (param-less)', new CustomEffect({
    name: 'Invert',
    code: 'fn effect(uv: vec2f) -> vec4f { let c = sampleInput0(uv); return vec4f(1.0 - c.rgb, c.a); }',
}));
validate('CustomEffect (params + depth)', new CustomEffect({
    name: 'DepthFog',
    params: [
        { name: 'color', type: 'vec3f', default: [0.5, 0.6, 0.7] },
        { name: 'density', type: 'f32', default: 0.02 },
    ],
    inputs: ['chain', 'depth'],
    code: `fn effect(uv: vec2f) -> vec4f {
    let c = sampleInput0(uv);
    let d = sampleInput1(uv);
    let lin = camera.near * camera.far / (camera.far - d * (camera.far - camera.near));
    let f = 1.0 - exp(-params.density * lin);
    return vec4f(mix(c.rgb, params.color, f), c.a);
}`,
}));

// orderChain: canonical sort, stable, custom effects pinned to the end.
{
    const names = (arr) => arr.map(e => e.name);

    count++;
    const sorted = orderChain([
        new FilmGrain(), new Tonemap(), new Bloom(), new Vignette(), new SSAO(),
    ]);
    const expected = ['SSAO', 'Bloom', 'Tonemap', 'Vignette', 'FilmGrain'];
    if (JSON.stringify(names(sorted)) === JSON.stringify(expected)) {
        console.log('  ok    orderChain sorts built-ins into canonical order');
    } else {
        failures++;
        console.log(`  FAIL  orderChain order\n        got ${names(sorted).join(', ')}`);
    }

    count++;
    const custom = new CustomEffect({ name: 'MyFx', code: 'fn effect(uv: vec2f) -> vec4f { return sampleInput0(uv); }' });
    const withCustom = orderChain([new FilmGrain(), custom, new SSAO()]);
    // Unranked custom effect keeps original relative order at the end.
    if (JSON.stringify(names(withCustom)) === JSON.stringify(['SSAO', 'FilmGrain', 'MyFx'])) {
        console.log('  ok    orderChain pins unranked effects to the end, stably');
    } else {
        failures++;
        console.log(`  FAIL  orderChain custom placement\n        got ${names(withCustom).join(', ')}`);
    }

    count++;
    const input = [new Tonemap(), new SSAO()];
    const out = orderChain(input);
    if (out !== input && input[0].name === 'Tonemap') {
        console.log('  ok    orderChain is non-mutating');
    } else {
        failures++;
        console.log('  FAIL  orderChain mutated its input or returned the same array');
    }

    count++;
    if (EFFECT_ORDER.indexOf('Tonemap') > 0 && EFFECT_ORDER.indexOf('FilmGrain') === EFFECT_ORDER.length - 1) {
        console.log('  ok    EFFECT_ORDER has Tonemap mid-chain and FilmGrain last');
    } else {
        failures++;
        console.log('  FAIL  EFFECT_ORDER shape unexpected');
    }
}

// The debug blit PostChain compiles for debugView().
{
    count++;
    const minimal = layoutFields([]);
    const composed = composePassShader({
        body: 'fn effect(uv: vec2f) -> vec4f { return sampleInput0(uv); }',
        paramsWgsl: minimal.wgsl,
        paramsTouchField: minimal.touchField,
        inputs: [{ depth: false }],
        encodeToSRGB: true,
        label: 'DebugView blit',
    });
    try {
        new WgslReflect(composed.code);
        console.log('  ok    DebugView blit');
    } catch (e) {
        failures++;
        console.log(`  FAIL  DebugView blit\n        ${e.message}`);
    }
}

console.log(failures === 0
    ? `\nAll ${count} generated shader modules parsed cleanly.`
    : `\n${failures} of ${count} shader module(s) FAILED to parse.`);
process.exit(failures === 0 ? 0 : 1);
